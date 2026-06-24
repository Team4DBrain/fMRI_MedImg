"""Compute per-run metadata from a manifest: brain mask, norm reference, tSNR.

For each run:
  1. If a mask + complete metadata already exist on disk and --overwrite is not
     passed, skip the entire run (no I/O, no compute). Idempotent fast path.
  2. Otherwise read the full 4D once. Compute the temporal mean from the same
     buffer.
  3. Compute a brain mask at native shape (X, Y, Z). For this pipeline we
     require all runs to have the same shape — manifest stage drops outliers.
  4. Save the mask as-is (no cropping). Update the manifest entry with
     mask_path, norm_ref, tsnr, mask_fraction.

Pipeline shape:
  - target_shape on disk equals native shape: (128, 128, 93) for IBC.
  - Datasets read native data and serve it without cropping.
  - target_z is fixed at the manifest's require_z. Mismatch is a hard error.

Run from the command line:
    python -m data.compute_metadata --manifest manifest.json \
        --derivatives-dir /path/to/derivatives

Idempotent: if a mask already exists with the right shape AND norm_ref/tSNR are
already in the manifest entry AND --overwrite is not passed, the run is skipped
entirely (no 4D read). To force a recompute, pass --overwrite.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .manifest import load_manifest
from .masks import compute_brain_mask, mask_fraction
from .normalize import compute_norm_ref
from .reader import VolumeReader

logger = logging.getLogger(__name__)

# IBC has uniform xy. We don't crop or pad anything in this pipeline.
DEFAULT_TARGET_XY = 128

# Pipeline marker stored in the manifest so Datasets can detect old manifests
# (z_crop, padding) and refuse to load with a clear message.
PIPELINE_MARKER = "no_crop_v1"


@dataclass
class RunMetadata:
    """Extra fields we attach to each manifest entry after this stage."""

    mask_path: str  # relative to derivatives_dir
    norm_ref: float
    tsnr_mean_in_brain: float
    mask_fraction: float


def compute_tsnr_from_data(data: np.ndarray, mask: np.ndarray) -> float:
    """Compute mean tSNR over brain voxels.

    tSNR = mean(voxel_timecourse) / std(voxel_timecourse), per voxel.

    Pure: no I/O, no reader. Float64 accumulators on the temporal reduction —
    float32 sum over 300 timepoints accumulates enough drift to bias tSNR.
    """
    mean_tc = data.mean(axis=-1, dtype=np.float64)
    std_tc = data.std(axis=-1, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr_map = np.where(std_tc > 0, mean_tc / std_tc, 0.0)

    brain_tsnr = tsnr_map[mask]
    brain_tsnr = brain_tsnr[np.isfinite(brain_tsnr) & (brain_tsnr > 0)]
    if brain_tsnr.size == 0:
        return 0.0
    return float(brain_tsnr.mean())


def _entry_metadata_complete(
    entry: dict, derivatives_dir: Path, expected_shape: tuple,
) -> bool:
    """Return True if this entry already has all fields and a valid mask on disk.

    Resolves the mask path from the entry itself (`entry["mask_path"]`) rather
    than reconstructing it from the run_id. A hand-edited or relocated entry
    that points elsewhere is checked against where it actually points, not
    against a hardcoded path.
    """
    required_fields = ("mask_path", "norm_ref", "tsnr_mean_in_brain", "mask_fraction")
    if not all(f in entry for f in required_fields):
        return False
    mask_abs = derivatives_dir / entry["mask_path"]
    if not mask_abs.is_file():
        return False
    # Verify mask shape matches what the pipeline expects. Header read only.
    try:
        cached_shape = tuple(int(s) for s in nib.load(str(mask_abs)).shape)
    except Exception:
        return False
    return cached_shape == expected_shape


def process_run(
    run_entry: dict,
    bids_root: Path,
    derivatives_dir: Path,
    target_z: int,
    overwrite: bool = False,
    mask_method: str = "auto",
) -> RunMetadata:
    """Compute mask, norm_ref, tSNR for one run. Save mask at native shape.

    No cropping. The run's native shape MUST be (X, Y, target_z); this is
    enforced upstream by the manifest's require_z filter, and we double-check
    here for safety.

    Fast path: if mask exists on disk AND every metadata field is already in
    the entry AND --overwrite is not set, return cached values without
    touching the 4D file (saves ~5–10s per run on typical hardware).
    """
    run_path = bids_root / run_entry["path"]
    run_id = run_entry["run_id"]
    mask_rel = f"masks/{run_id}_mask.nii.gz"
    mask_abs = derivatives_dir / mask_rel

    # Validate manifest-recorded shape matches target_z. Manifest filtering
    # should already have enforced this, but defend in depth in case the user
    # crafted a manifest by hand or a bug slips through.
    manifest_shape = tuple(run_entry["shape"][:3])
    if manifest_shape[2] != target_z:
        raise ValueError(
            f"Run {run_id} has manifest z={manifest_shape[2]}, expected target_z={target_z}. "
            "Re-run the manifest stage with --require-z matching your target."
        )
    expected_shape = manifest_shape  # (X, Y, target_z)

    # Fast path — every cache check is a few stat()s, no 4D read.
    if not overwrite and _entry_metadata_complete(run_entry, derivatives_dir, expected_shape):
        logger.info(f"  Cached metadata complete for {run_id}; skipping recompute")
        return RunMetadata(
            mask_path=run_entry["mask_path"],
            norm_ref=float(run_entry["norm_ref"]),
            tsnr_mean_in_brain=float(run_entry["tsnr_mean_in_brain"]),
            mask_fraction=float(run_entry["mask_fraction"]),
        )

    # Slow path: read once, derive everything from one buffer.
    reader = VolumeReader(run_path)
    native_shape = reader.shape3d
    if native_shape != expected_shape:
        raise ValueError(
            f"Run {run_id}: NIfTI native shape {native_shape} disagrees with "
            f"manifest shape {expected_shape}. Manifest stale; rebuild it."
        )

    logger.info(f"  Reading {run_id} (shape {native_shape})...")
    full = reader.read_full(dtype=np.float32)              # (X, Y, Z, T)
    mean_vol = full.mean(axis=-1, dtype=np.float64).astype(np.float32)

    # Compute mask, or load existing one if shape-compatible.
    if mask_abs.exists() and not overwrite:
        logger.info(f"  Loading existing mask: {mask_rel}")
        mask = np.asarray(nib.load(str(mask_abs)).dataobj).astype(bool)
        if mask.shape != expected_shape:
            raise RuntimeError(
                f"Cached mask shape {mask.shape} != expected {expected_shape} "
                f"for {run_id}. The mask was built under a different pipeline. "
                "Re-run with --overwrite to regenerate."
            )
    else:
        logger.info(f"  Computing mask for {run_id} (method={mask_method})")
        mask = compute_brain_mask(
            mean_vol, affine=reader.img.affine, method=mask_method,
        )
        if mask.shape != expected_shape:
            raise RuntimeError(
                f"compute_brain_mask returned shape {mask.shape}, expected {expected_shape}. "
                "Bug in masks.compute_brain_mask."
            )
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine=reader.img.affine)
        mask_abs.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask_img, str(mask_abs))
        logger.info(f"  Wrote mask {mask_rel} (shape {expected_shape})")

    # norm_ref + tSNR over the in-brain voxels at native shape (no crop).
    norm_ref = compute_norm_ref(mean_vol, mask)
    tsnr = compute_tsnr_from_data(full, mask)
    frac = mask_fraction(mask)

    if frac > 0.55:
        logger.warning(
            f"  mask_fraction={frac:.3f} for {run_id} is suspiciously high "
            f"(expected ~0.2-0.4). Mask likely includes non-brain tissue. "
            f"Use mask_method=synthstrip for usable results."
        )

    return RunMetadata(
        mask_path=mask_rel,
        norm_ref=norm_ref,
        tsnr_mean_in_brain=tsnr,
        mask_fraction=frac,
    )


def compute_all(
    manifest_path: Path,
    derivatives_dir: Path,
    target_z: int | None = None,
    overwrite: bool = False,
    mask_method: str = "auto",
) -> None:
    """Process every run in the manifest, write updated manifest in place.

    Args:
        manifest_path: path to manifest JSON to read and update.
        derivatives_dir: where to write brain masks.
        target_z: required uniform z dimension. If None, read from manifest's
            `require_z` field. Mismatch with the manifest's value is an error.
        overwrite: recompute masks/metadata even if already complete on disk.
        mask_method: "auto", "synthstrip", or "percentile". See masks.py.
    """
    manifest = load_manifest(manifest_path)
    bids_root = Path(manifest["bids_root"])
    derivatives_dir = Path(derivatives_dir).resolve()
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    if not manifest["runs"]:
        raise RuntimeError(
            "Manifest contains zero runs. Stage 1 may have dropped everything "
            "due to require_z; check the bids root and the require-z value."
        )

    # Reconcile target_z with manifest's require_z (set at stage 1).
    manifest_require_z = manifest.get("require_z")
    if target_z is None:
        if manifest_require_z is None:
            # Old or hand-crafted manifest with no require_z. Infer from runs.
            zs = sorted({tuple(r["shape"][:3])[2] for r in manifest["runs"]})
            if len(zs) != 1:
                raise ValueError(
                    f"Manifest has runs with mixed z values {zs} and no require_z field. "
                    "Re-run the manifest stage with --require-z, or pass --target-z explicitly."
                )
            target_z = zs[0]
        else:
            target_z = manifest_require_z
    elif manifest_require_z is not None and manifest_require_z != target_z:
        raise ValueError(
            f"--target-z={target_z} disagrees with manifest's require_z={manifest_require_z}. "
            "These must match. Re-run the manifest stage if you want a different z."
        )

    # Validate every run conforms.
    native_shapes = [tuple(r["shape"][:3]) for r in manifest["runs"]]
    bad = [(r["run_id"], s) for r, s in zip(manifest["runs"], native_shapes) if s[2] != target_z]
    if bad:
        preview = ", ".join(f"{rid}(z={s[2]})" for rid, s in bad[:3])
        raise ValueError(
            f"{len(bad)} run(s) in the manifest don't match target_z={target_z}: {preview}"
            f"{'...' if len(bad) > 3 else ''}. Manifest stage failed to filter."
        )
    xs = {s[0] for s in native_shapes}
    ys = {s[1] for s in native_shapes}
    if len(xs) > 1 or len(ys) > 1:
        raise ValueError(
            f"Non-uniform xy across runs (x={xs}, y={ys}). "
            "This pipeline assumes uniform spatial shape."
        )
    native_x, native_y = next(iter(xs)), next(iter(ys))
    target_shape = (native_x, native_y, target_z)

    logger.info(f"Processing {manifest['n_runs']} runs from {bids_root}")
    logger.info(f"Writing derivatives to {derivatives_dir}")
    logger.info(f"Target shape (X, Y, Z): {target_shape}")
    logger.info(f"Mask method: {mask_method}")

    for i, entry in enumerate(manifest["runs"], start=1):
        logger.info(f"[{i}/{manifest['n_runs']}] {entry['run_id']}")
        try:
            metadata = process_run(
                entry, bids_root, derivatives_dir, target_z=target_z,
                overwrite=overwrite, mask_method=mask_method,
            )
        except Exception as e:
            logger.error(f"  FAILED on {entry['run_id']}: {e}")
            entry["metadata_error"] = str(e)
            continue

        entry["mask_path"] = metadata.mask_path
        entry["norm_ref"] = metadata.norm_ref
        entry["tsnr_mean_in_brain"] = metadata.tsnr_mean_in_brain
        entry["mask_fraction"] = metadata.mask_fraction
        # Clear any old error key from a prior failed run.
        entry.pop("metadata_error", None)
        # Drop any z_start left over from old z_crop manifests.
        entry.pop("z_start", None)
        logger.info(
            f"  norm_ref={metadata.norm_ref:.1f}  "
            f"tSNR={metadata.tsnr_mean_in_brain:.1f}  "
            f"mask_frac={metadata.mask_fraction:.3f}"
        )

    manifest["derivatives_dir"] = str(derivatives_dir)
    manifest["target_shape"] = list(target_shape)
    manifest["target_z"] = int(target_z)
    manifest["pipeline"] = PIPELINE_MARKER

    # Atomic write: a SIGKILL or disk-full mid-write must not corrupt the
    # existing manifest (which may have hours of mask computation behind it).
    manifest_path = Path(manifest_path)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)

    logger.info(f"Updated manifest written to {manifest_path}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSON")
    parser.add_argument(
        "--derivatives-dir", type=Path, required=True,
        help="Directory to write brain masks and other derivatives into",
    )
    parser.add_argument(
        "--target-z", type=int, default=None,
        help="Expected uniform z dimension. Default: read from manifest's "
             "`require_z` field. Must match the manifest if both are set.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute masks/metadata even if already complete on disk.",
    )
    parser.add_argument(
        "--mask-method", choices=["auto", "synthstrip", "percentile"], default="auto",
        help="Brain masking method. 'auto' (default) prefers synthstrip if "
             "installed and falls back to percentile with a warning. "
             "'synthstrip' raises if no synthstrip executable is found. "
             "'percentile' uses pure-Python intensity thresholding (imperfect).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    compute_all(
        args.manifest,
        args.derivatives_dir,
        target_z=args.target_z,
        overwrite=args.overwrite,
        mask_method=args.mask_method,
    )
    print(f"Done. Updated manifest: {args.manifest}")


if __name__ == "__main__":
    _cli()
