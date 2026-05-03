"""PyTorch Dataset classes for fMRI training.

Architecture:
  - BaseFMRIDataset handles the common stuff: manifest parsing, sample
    indexing, mask loading, normalization, z-axis cropping to target shape.
  - Subclasses (DenoisingDataset, SpatialSRDataset, TemporalSRDataset) override
    __getitem__ to return the right (input, target) pair for their model.
  - Degradation functions are STUBBED for the denoising path. They raise
    NotImplementedError until we implement them with each model's owner.
    (Spatial degradation IS implemented; pass make_spatial_degradation()
    as degrade_fn to SpatialSRDataset.)

Shape policy (option B, z-only crop):
  - All volumes are served at (X, Y, target_z) where X, Y match IBC's native
    in-plane (128×128). target_z is set in compute_metadata, stored in the
    manifest as `target_z`, and is the smallest native z across runs by
    default.
  - Per run, `z_start` (also in the manifest) tells us where to crop along z
    so the brain's z-bbox is centered in the window.
  - No padding anywhere. xy is unchanged.

For SpatialSRDataset (Option A), the input is at LR shape and the target at HR
shape — the model is responsible for upsampling.

Performance notes:
  - VolumeReaders are obtained via reader.get_reader (per-process cache), so
    multiple samples from the same run inside a worker share one open handle.
  - TemporalSR reads (t-gap, t, t+gap) as ONE range read, not three separate
    timepoint reads.
  - Mask cache stores ready-to-use float32 tensors; __getitem__ clones on
    access so in-place ops downstream cannot corrupt the cache.

Fork-safety:
  - nibabel image handles don't always fork cleanly. The reader cache is
    process-local (keyed by pid) and is naturally re-populated lazily in each
    DataLoader worker after fork. Do not call get_reader from the main
    process before spawning workers, or you risk a stale entry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .cropping import crop_z
from .degradation_spatial import (
    downsample_mask_to_lr,
    make_spatial_degradation,
    voxel_size_to_target_shape,
)
from .normalize import normalize
from .reader import get_reader

logger = logging.getLogger(__name__)


class BaseFMRIDataset(Dataset):
    """Shared behavior for all three task datasets.

    Subclasses must implement:
        - _build_sample_index() -> list of sample descriptors
        - __getitem__(idx) -> dict with at least "input", "target", "mask"
    """

    def __init__(
        self,
        manifest_path: str | Path,
        subject_filter: list[str] | None = None,
        check_files_exist: bool = True,
    ):
        """
        Args:
            manifest_path: path to JSON manifest produced by compute_metadata.py.
                Must be from the z_crop pipeline (has `target_z` and per-run
                `z_start` fields). Old padding-pipeline manifests are rejected.
            subject_filter: if given, keep only runs whose subject is in this list.
                Use this for train/val/test splits by subject. Subjects must match
                the manifest's exact string form (typically zero-padded, e.g. "01").
            check_files_exist: if True (default), validate at construction time
                that every kept run's BOLD file and mask file exist. Trades a
                small startup cost (one stat() per file) for fail-fast behavior
                instead of a mysterious crash deep in __getitem__. Set False
                when the cost matters and you trust your manifest.
        """
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open() as f:
            manifest = json.load(f)

        if manifest.get("pipeline") != "z_crop":
            raise RuntimeError(
                "Manifest is not from the z_crop pipeline. Re-run "
                "compute_metadata.py to regenerate (you'll need --overwrite "
                "since the on-disk masks have a different shape now)."
            )

        self.bids_root = Path(manifest["bids_root"])
        self.derivatives_dir = Path(manifest.get("derivatives_dir", ""))
        if not self.derivatives_dir.is_dir():
            raise RuntimeError(
                f"derivatives_dir missing or invalid in manifest: {self.derivatives_dir}. "
                "Did you run compute_metadata.py?"
            )

        if "target_shape" not in manifest or "target_z" not in manifest:
            raise RuntimeError(
                "manifest is missing 'target_shape' or 'target_z'. "
                "Re-run compute_metadata.py."
            )
        self.target_shape: tuple[int, int, int] = tuple(manifest["target_shape"])
        self.target_z: int = int(manifest["target_z"])

        # Keep only runs that have complete metadata.
        all_runs = manifest["runs"]
        runs = [
            r for r in all_runs
            if "norm_ref" in r and "mask_path" in r and "z_start" in r
        ]
        if len(runs) < len(all_runs):
            logger.warning(
                "Dropped %d/%d runs missing metadata. Re-run compute_metadata.py "
                "if you expected them.", len(all_runs) - len(runs), len(all_runs),
            )

        if subject_filter is not None:
            wanted = set(subject_filter)
            available = {r["subject"] for r in runs}
            unmatched = wanted - available
            if unmatched:
                raise ValueError(
                    f"subject_filter contains subjects not in manifest: {sorted(unmatched)}. "
                    f"Available: {sorted(available)}. Note: subjects are typically "
                    f"zero-padded ('01', not '1')."
                )
            runs = [r for r in runs if r["subject"] in wanted]

        if not runs:
            raise RuntimeError(
                f"No runs matched after filtering. subject_filter={subject_filter}"
            )

        if check_files_exist:
            self._validate_paths(runs)

        self.runs = runs
        # Cache HR (cropped) masks as ready-to-use float32 tensors keyed by
        # run_id. Negligible memory.
        self._mask_cache: dict[str, torch.Tensor] = {}

        self.samples = self._build_sample_index()
        if not self.samples:
            raise RuntimeError(
                f"{type(self).__name__} produced an empty sample index. "
                "Check filtering and (for TemporalSR) gap parameter."
            )

    def _validate_paths(self, runs: list[dict]) -> None:
        """Fail-fast check that referenced data and mask files exist."""
        missing = []
        for r in runs:
            if not (self.bids_root / r["path"]).is_file():
                missing.append(("bold", r["run_id"], self.bids_root / r["path"]))
            if not (self.derivatives_dir / r["mask_path"]).is_file():
                missing.append(("mask", r["run_id"], self.derivatives_dir / r["mask_path"]))
        if missing:
            n_show = 5
            preview = "\n".join(f"  [{kind}] {rid}: {p}" for kind, rid, p in missing[:n_show])
            extra = f"\n  ... and {len(missing) - n_show} more" if len(missing) > n_show else ""
            raise FileNotFoundError(
                f"{len(missing)} referenced file(s) missing on disk:\n{preview}{extra}"
            )

    def _build_sample_index(self) -> list:
        """Default: one sample per (run, timepoint). Subclasses can override."""
        samples = []
        for run_idx, run in enumerate(self.runs):
            for t in range(run["n_volumes"]):
                samples.append((run_idx, t))
        return samples

    def _get_hr_mask(self, run_idx: int) -> torch.Tensor:
        """Return a CLONED HR brain mask tensor (float32, target_shape).

        Mask on disk is already at the (X, Y, target_z) shape we serve.
        """
        run = self.runs[run_idx]
        run_id = run["run_id"]
        cached = self._mask_cache.get(run_id)
        if cached is None:
            mask_path = self.derivatives_dir / run["mask_path"]
            arr = np.asarray(nib.load(str(mask_path)).dataobj).astype(np.float32)
            if arr.shape != self.target_shape:
                raise RuntimeError(
                    f"Mask for {run_id} has shape {arr.shape}, expected "
                    f"target_shape {self.target_shape}. Re-run compute_metadata.py "
                    f"with the same target_z."
                )
            arr = np.ascontiguousarray(arr)
            cached = torch.from_numpy(arr)
            self._mask_cache[run_id] = cached
        return cached.clone()

    def _read_volume(self, run_idx: int, t: int) -> np.ndarray:
        """Read one volume (run_idx, t), normalize, crop z. Returns float32 (X,Y,target_z)."""
        run = self.runs[run_idx]
        path = self.bids_root / run["path"]
        z_start = int(run["z_start"])
        # get_reader is process-local, populated lazily inside the worker.
        reader = get_reader(path)
        vol = reader.read_volume(t).astype(np.float32)
        vol = normalize(vol, run["norm_ref"])
        vol = crop_z(vol, z_start, self.target_z)
        return vol

    def _read_range(self, run_idx: int, t_start: int, t_end: int) -> np.ndarray:
        """Read volumes [t_start, t_end), normalize, crop z. Returns (T, X, Y, target_z) float32.

        ONE underlying disk read for the contiguous span, not (t_end - t_start)
        independent ones. Crop is applied to the 4D block in one shot.
        """
        run = self.runs[run_idx]
        path = self.bids_root / run["path"]
        z_start = int(run["z_start"])
        reader = get_reader(path)
        block = reader.read_range(t_start, t_end).astype(np.float32)  # (X,Y,Z_native,T)
        block = normalize(block, run["norm_ref"])
        block = crop_z(block, z_start, self.target_z)                 # (X,Y,target_z,T)
        # Move T to the front: (T, X, Y, target_z)
        return np.moveaxis(block, -1, 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("Subclasses must implement __getitem__")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _not_implemented_degradation(*args, **kwargs):
    raise NotImplementedError(
        "Degradation function not yet implemented. "
        "Decide on the degradation model with your teammates first."
    )


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Wrap numpy → torch. Ensures C-contiguous to avoid from_numpy footguns."""
    return torch.from_numpy(np.ascontiguousarray(arr))


# ---------------------------------------------------------------------------
# Task-specific datasets
# ---------------------------------------------------------------------------


class DenoisingDataset(BaseFMRIDataset):
    """Samples for the denoising model.

    Returns dict with:
        input:  noisy volume,  shape (1, X, Y, target_z) (float32)
        target: clean volume,  shape (1, X, Y, target_z) (float32)
        mask:   HR brain mask, shape (1, X, Y, target_z) (float32)
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        degrade_fn=_not_implemented_degradation,
        check_files_exist: bool = True,
    ):
        super().__init__(
            manifest_path,
            subject_filter=subject_filter,
            check_files_exist=check_files_exist,
        )
        self.degrade_fn = degrade_fn

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        clean = self._read_volume(run_idx, t)             # (X, Y, target_z)
        noisy = self.degrade_fn(clean)                    # same shape
        mask = self._get_hr_mask(run_idx)                 # (X, Y, target_z) float32

        return {
            "input": _to_tensor(noisy).unsqueeze(0),
            "target": _to_tensor(clean).unsqueeze(0),
            "mask": mask.unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }


class SpatialSRDataset(BaseFMRIDataset):
    """Samples for the spatial SR model — Option A.

    Input is at LR shape (e.g. 64×64×42 for 3mm with target_z=84), target at
    HR shape (X, Y, target_z). The model is responsible for upsampling.

    Returns dict with:
        input:    LR volume,  shape (1, kx, ky, kz)
        target:   HR volume,  shape (1, X, Y, target_z)
        mask_hr:  HR mask,    shape (1, X, Y, target_z)
        mask_lr:  LR mask,    shape (1, kx, ky, kz)
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        degrade_fn=None,
        source_voxel_mm: float = 1.5,
        target_voxel_mm: float = 3.0,
        check_files_exist: bool = True,
    ):
        super().__init__(
            manifest_path,
            subject_filter=subject_filter,
            check_files_exist=check_files_exist,
        )
        self.source_voxel_mm = source_voxel_mm
        self.target_voxel_mm = target_voxel_mm
        # Default degradation: k-space truncation at the requested voxel ratio.
        # Spatial SR HAS a canonical degradation (unlike denoising), so making
        # callers pass it explicitly was wasted ceremony. Pass your own
        # degrade_fn if you want apodize=False or a different model.
        if degrade_fn is None:
            degrade_fn = make_spatial_degradation(
                source_voxel_mm=source_voxel_mm,
                target_voxel_mm=target_voxel_mm,
            )
        self.degrade_fn = degrade_fn

        # Precompute LR shape from target_shape (HR) and voxel ratio.
        self.lr_shape: tuple[int, int, int] = voxel_size_to_target_shape(
            self.target_shape, source_voxel_mm, target_voxel_mm,
        )

        # Cache LR masks separately (derived from HR masks).
        self._lr_mask_cache: dict[str, torch.Tensor] = {}

    def _get_lr_mask(self, run_idx: int) -> torch.Tensor:
        """Return a CLONED LR brain mask tensor (float32, lr_shape)."""
        run = self.runs[run_idx]
        run_id = run["run_id"]
        cached = self._lr_mask_cache.get(run_id)
        if cached is None:
            mask_path = self.derivatives_dir / run["mask_path"]
            hr_bool = np.asarray(nib.load(str(mask_path)).dataobj).astype(bool)
            lr_bool = downsample_mask_to_lr(hr_bool, self.lr_shape)
            arr = np.ascontiguousarray(lr_bool.astype(np.float32))
            cached = torch.from_numpy(arr)
            self._lr_mask_cache[run_id] = cached
        return cached.clone()

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        hr = self._read_volume(run_idx, t)          # (X, Y, target_z)
        lr = self.degrade_fn(hr)                    # (kx, ky, kz)

        if lr.shape != self.lr_shape:
            raise RuntimeError(
                f"degrade_fn returned shape {lr.shape}, expected {self.lr_shape}. "
                f"degrade_fn config inconsistent with Dataset (source/target voxel size)."
            )

        return {
            "input": _to_tensor(lr).unsqueeze(0),
            "target": _to_tensor(hr).unsqueeze(0),
            "mask_hr": self._get_hr_mask(run_idx).unsqueeze(0),
            "mask_lr": self._get_lr_mask(run_idx).unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }


class TemporalSRDataset(BaseFMRIDataset):
    """Samples for the temporal SR model.

    The "degradation" is implicit: by sampling t-gap and t+gap and predicting
    the volume at t, we simulate a half-rate acquisition where t was never
    measured. There is no separate degrade_fn.

    Returns dict with:
        input:  two neighbors stacked along channel, shape (2, X, Y, target_z)
        target: middle volume,                       shape (1, X, Y, target_z)
        mask:   HR brain mask,                       shape (1, X, Y, target_z)
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        gap: int = 1,
        check_files_exist: bool = True,
    ):
        if gap < 1:
            raise ValueError(f"gap must be >= 1, got {gap}")
        self.gap = gap
        super().__init__(
            manifest_path,
            subject_filter=subject_filter,
            check_files_exist=check_files_exist,
        )

    def _build_sample_index(self) -> list:
        """Only valid samples where both neighbors exist inside the run."""
        samples = []
        too_short: list[str] = []
        for run_idx, run in enumerate(self.runs):
            n = run["n_volumes"]
            if n < 2 * self.gap + 1:
                too_short.append(f"{run['run_id']} (n={n})")
                continue
            for t in range(self.gap, n - self.gap):
                samples.append((run_idx, t))
        if too_short:
            logger.warning(
                "TemporalSRDataset(gap=%d): dropped %d run(s) too short for any sample: %s",
                self.gap, len(too_short),
                ", ".join(too_short[:5]) + ("..." if len(too_short) > 5 else ""),
            )
        return samples

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        if self.gap == 1:
            block = self._read_range(run_idx, t - 1, t + 2)  # (3, X, Y, target_z)
            before, target, after = block[0], block[1], block[2]
        else:
            before = self._read_volume(run_idx, t - self.gap)
            target = self._read_volume(run_idx, t)
            after = self._read_volume(run_idx, t + self.gap)

        input_arr = np.stack([before, after], axis=0)  # (2, X, Y, target_z)

        return {
            "input": _to_tensor(input_arr),
            "target": _to_tensor(target).unsqueeze(0),
            "mask": self._get_hr_mask(run_idx).unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }
