"""Brain mask computation for fMRI volumes.

Two backends:

  - "synthstrip" (default): SynthStrip (Hoopes et al. 2022, NeuroImage). DL-based
    skull stripping designed for cross-modality. Works directly on EPI without
    needing T1 coregistration. Robust and accurate. Requires one of these
    executables on PATH:
        nipreps-synthstrip      (pip install nipreps-synthstrip; needs --model)
        mri_synthstrip          (full FreeSurfer; bakes in the model path)
        synthstrip-docker       (needs Docker)
        synthstrip-singularity  (needs Apptainer/Singularity)

  - "percentile": fallback intensity-thresholding + morphological cleanup.
    Pure numpy/scipy, no external tools. Empirically imperfect: tends to
    include some skull/scalp at low percentiles or carve into cerebellum at
    high percentiles. Use only when SynthStrip isn't available.

  - "auto" (default for the dispatch): try synthstrip; if unavailable, log a
    warning and fall back to percentile.

SynthStrip model file:
    nipreps-synthstrip ships the code but NOT the weights (they live in
    FreeSurfer's git-annex). To run it, the model file synthstrip.1.pt must
    be locatable. We search:
        1. $SYNTHSTRIP_MODEL (env var)
        2. /srv/synthstrip/synthstrip.1.pt          (admin-blessed shared)
        3. ~/shared/synthstrip/synthstrip.1.pt      (per-user shared dir)
    If none exists, we raise with a pointer to the README's setup instructions.
    The FreeSurfer wrapper (mri_synthstrip) bakes the model path in and does
    not need this lookup.

GPU:
    Both nipreps-synthstrip and mri_synthstrip accept `-g` to enable GPU.
    We auto-append `-g` whenever torch reports CUDA available.

Reference:
  https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
  https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip
  https://github.com/nipreps/synthstrip
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# Executables that all support the same -i / -m / -b CLI for SynthStrip.
# Order matters: prefer nipreps (pip-installable, lightest) over Docker/Singularity
# wrappers. mri_synthstrip is in the list for users who do have FreeSurfer.
_SYNTHSTRIP_CANDIDATES = (
    "nipreps-synthstrip",
    "mri_synthstrip",
    "synthstrip-docker",
    "synthstrip-singularity",
)

# Default search paths for the SynthStrip model weights file. nipreps-synthstrip
# requires --model; we resolve the path from this list (env var takes precedence).
_DEFAULT_SYNTHSTRIP_MODEL_PATHS = (
    "/srv/synthstrip/synthstrip.1.pt",
    str(Path.home() / "shared" / "synthstrip" / "synthstrip.1.pt"),
)


def find_synthstrip_executable() -> str | None:
    """Return the first available synthstrip executable on PATH, or None."""
    for name in _SYNTHSTRIP_CANDIDATES:
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def find_synthstrip_model() -> str | None:
    """Locate the SynthStrip model weights file (.pt).

    Search order:
      1. $SYNTHSTRIP_MODEL (env var). If set but the file doesn't exist, log
         a warning and continue searching.
      2. Hardcoded default paths (in order).

    Returns the first existing file path, or None if nothing found.
    """
    env_path = os.environ.get("SYNTHSTRIP_MODEL")
    if env_path:
        if Path(env_path).is_file():
            return env_path
        logger.warning(
            "SYNTHSTRIP_MODEL=%s is set but the file doesn't exist; "
            "falling back to default search paths.", env_path,
        )
    for p in _DEFAULT_SYNTHSTRIP_MODEL_PATHS:
        if Path(p).is_file():
            return p
    return None


def _is_nipreps_synthstrip(executable: str) -> bool:
    """nipreps-synthstrip is the pip package; needs --model. mri_synthstrip
    bakes the model path in. The Docker/Singularity wrappers also bake it in.
    """
    return Path(executable).name.startswith("nipreps-synthstrip")


def _gpu_available() -> bool:
    """Best-effort GPU detection for deciding whether to pass -g.

    Imports torch lazily so this module is usable in environments where torch
    is the wrong version or absent — masks fall back to CPU silently in that
    case. Returns False on any import or CUDA-init error.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _compute_synthstrip_mask(
    mean_volume: np.ndarray,
    affine: np.ndarray,
    executable: str,
    border: int = 1,
    no_csf: bool = False,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Run SynthStrip on a mean volume and return the brain mask.

    SynthStrip is an external program that reads/writes NIfTI files. We:
      1. Write mean_volume to a temp NIfTI.
      2. Build the CLI invocation, injecting --model and -g as needed.
      3. Read the produced mask, return as boolean array.
      4. Clean up temp files.

    Args:
        mean_volume: 3D float array, shape (X, Y, Z). The temporal mean of a run.
        affine: 4x4 voxel-to-world transform from the source NIfTI.
        executable: full path to a synthstrip-compatible executable.
        border: -b parameter. Distance in mm from brain boundary to include.
            Default 1 (SynthStrip's default). Use 0 for tighter masks.
        no_csf: if True, pass --no-csf so the mask excludes surrounding CSF.
            Only supported by mri_synthstrip; nipreps-synthstrip does not have
            this flag and passing no_csf=True with it is a hard error.
        use_gpu: tri-state. None (default) = auto-detect via torch.cuda.is_available().
            True = force GPU (-g). False = force CPU (no -g).

    Returns:
        3D boolean array, same shape as mean_volume.
    """
    if mean_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {mean_volume.shape}")

    # Resolve --model path now so we fail before tempdir/subprocess setup if missing.
    model_arg: list[str] = []
    if _is_nipreps_synthstrip(executable):
        model = find_synthstrip_model()
        if model is None:
            raise RuntimeError(
                "nipreps-synthstrip requires the SynthStrip model weights file, "
                "but it was not found.\n"
                f"Searched (in order): $SYNTHSTRIP_MODEL, then {list(_DEFAULT_SYNTHSTRIP_MODEL_PATHS)}.\n"
                "Either set SYNTHSTRIP_MODEL=/path/to/synthstrip.1.pt, or place the "
                "file at one of the default paths. See README.md for one-time download "
                "instructions (the weights live in FreeSurfer's git-annex)."
            )
        model_arg = ["--model", model]

    # --no-csf is mri_synthstrip-only.
    if no_csf and _is_nipreps_synthstrip(executable):
        raise ValueError(
            "no_csf=True is not supported by nipreps-synthstrip "
            "(no --no-csf flag in that fork). Set no_csf=False, or install "
            "FreeSurfer's mri_synthstrip if you need this option."
        )

    # GPU flag: both backends accept -g.
    gpu_arg: list[str] = []
    do_gpu = _gpu_available() if use_gpu is None else bool(use_gpu)
    if do_gpu:
        gpu_arg = ["-g"]

    with tempfile.TemporaryDirectory(prefix="synthstrip_") as tmpdir:
        tmpdir = Path(tmpdir)
        in_path = tmpdir / "input.nii.gz"
        mask_path = tmpdir / "mask.nii.gz"

        in_img = nib.Nifti1Image(mean_volume.astype(np.float32), affine=affine)
        nib.save(in_img, str(in_path))

        cmd = [
            executable,
            "-i", str(in_path),
            "-m", str(mask_path),
            "-b", str(border),
            *model_arg,
            *gpu_arg,
        ]
        if no_csf:
            cmd.append("--no-csf")

        logger.debug(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"SynthStrip failed (exit code {e.returncode}).\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"SynthStrip timed out after {e.timeout}s. "
                f"Container pull may be in progress; try running once manually first."
            ) from e

        if not mask_path.exists():
            raise RuntimeError(
                f"SynthStrip ran but didn't produce mask file at {mask_path}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Read result before tempdir is cleaned up.
        mask = np.asarray(nib.load(str(mask_path)).dataobj).astype(bool)

    if mask.shape != mean_volume.shape:
        raise RuntimeError(
            f"SynthStrip output shape {mask.shape} doesn't match input "
            f"{mean_volume.shape}. This shouldn't happen — file a bug."
        )

    return mask


def _compute_percentile_mask(
    mean_volume: np.ndarray,
    lower_percentile: float = 55.0,
    opening_iterations: int = 2,
    closing_iterations: int = 2,
) -> np.ndarray:
    """Percentile-threshold + morphology brain masking. Fallback method.

    Algorithm:
      1. Threshold at the lower_percentile-th percentile of non-zero voxels.
      2. Morphological opening — removes small bright specks (fat, skin, noise).
      3. Keep largest connected component (throws away eyeballs, sinus artifacts).
      4. Morphological closing — fills small holes (e.g., ventricles).
      5. Hole-filling for any topologically enclosed cavities that survived.

    Args:
        mean_volume: 3D float array. The temporal mean of a run.
        lower_percentile: percentile of non-zero voxels used as threshold.
            Lower = larger mask. 55 is empirically reasonable for IBC EPI.
        opening_iterations: opening strength (1 voxel per iteration).
        closing_iterations: closing strength (1 voxel per iteration).

    Returns:
        3D boolean array, same shape as input.
    """
    if mean_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {mean_volume.shape}")

    nonzero = mean_volume[mean_volume > 0]
    if nonzero.size == 0:
        raise ValueError("Input volume has no non-zero voxels — can't compute mask")

    threshold = np.percentile(nonzero, lower_percentile)
    mask = mean_volume > threshold
    mask = ndimage.binary_opening(mask, iterations=opening_iterations)

    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        raise ValueError(
            "Mask is empty after opening — try lower_percentile lower, "
            "or the input has no discernible brain"
        )

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    mask = labeled == largest_label

    mask = ndimage.binary_closing(mask, iterations=closing_iterations)
    mask = ndimage.binary_fill_holes(mask)

    return mask.astype(bool)


def compute_brain_mask(
    mean_volume: np.ndarray,
    affine: np.ndarray | None = None,
    method: Literal["auto", "synthstrip", "percentile"] = "auto",
    *,
    # synthstrip params
    border: int = 1,
    no_csf: bool = False,
    use_gpu: bool | None = None,
    # percentile params
    lower_percentile: float = 55.0,
    opening_iterations: int = 2,
    closing_iterations: int = 2,
) -> np.ndarray:
    """Compute a brain mask. Dispatches between SynthStrip and percentile.

    Args:
        mean_volume: 3D float array, the temporal mean of a BOLD run.
        affine: 4x4 voxel-to-world transform. REQUIRED for synthstrip; ignored
            for percentile. Get it from `VolumeReader.img.affine`.
        method:
            - "auto" (default): use synthstrip if available, else percentile.
            - "synthstrip": use synthstrip; raise if unavailable.
            - "percentile": use percentile masking unconditionally.
        border, no_csf, use_gpu: SynthStrip parameters (see _compute_synthstrip_mask).
        lower_percentile, opening_iterations, closing_iterations:
            percentile method parameters.

    Returns:
        3D boolean mask, same shape as mean_volume.
    """
    if method == "synthstrip":
        executable = find_synthstrip_executable()
        if executable is None:
            raise RuntimeError(
                "method='synthstrip' requested but no synthstrip executable found "
                f"on PATH. Looked for: {_SYNTHSTRIP_CANDIDATES}. "
                f"Install via `pip install nipreps-synthstrip` (then also place the "
                f"model weights — see README.md), or fall back with method='percentile'."
            )
        if affine is None:
            raise ValueError("synthstrip requires `affine`. Pass reader.img.affine.")
        return _compute_synthstrip_mask(
            mean_volume, affine, executable=executable,
            border=border, no_csf=no_csf, use_gpu=use_gpu,
        )

    if method == "percentile":
        return _compute_percentile_mask(
            mean_volume,
            lower_percentile=lower_percentile,
            opening_iterations=opening_iterations,
            closing_iterations=closing_iterations,
        )

    if method == "auto":
        executable = find_synthstrip_executable()
        if executable is not None:
            # Affine is optional in auto mode: SynthStrip needs SOME affine in
            # the temp NIfTI, but the mask values returned only depend on the
            # voxel grid. If the caller didn't supply one we fall back to
            # identity — the mask is still correct in voxel space, just lacks
            # a meaningful world-space orientation. (`method="synthstrip"`
            # still requires an explicit affine because at that point the
            # caller asked for it specifically.)
            if affine is None:
                logger.warning(
                    "synthstrip available but no affine given; using identity. "
                    "The mask will be correct in voxel space but won't have a "
                    "meaningful world-space orientation."
                )
                affine = np.eye(4)
            return _compute_synthstrip_mask(
                mean_volume, affine, executable=executable,
                border=border, no_csf=no_csf, use_gpu=use_gpu,
            )
        logger.warning(
            "synthstrip executable not found on PATH (looked for: %s). "
            "Falling back to percentile masking — known to be imperfect.",
            ", ".join(_SYNTHSTRIP_CANDIDATES),
        )
        return _compute_percentile_mask(
            mean_volume,
            lower_percentile=lower_percentile,
            opening_iterations=opening_iterations,
            closing_iterations=closing_iterations,
        )

    raise ValueError(f"Unknown method={method!r}; expected auto/synthstrip/percentile")


def mask_fraction(mask: np.ndarray) -> float:
    """Fraction of voxels that are inside the brain. Sanity check ≈ 0.2-0.5 for whole-brain BOLD."""
    return float(mask.mean())
