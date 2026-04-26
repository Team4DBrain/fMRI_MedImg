"""Compute per-run metadata from a manifest: brain mask, norm reference, tSNR.

Reads each run, computes its temporal mean, derives a brain mask, computes
a normalization reference, and computes tSNR. Pads the brain mask to the
configured target shape and writes it to disk. Updates the manifest.

Tracks the maximum z dimension seen across all runs and logs it. Raises if
any run exceeds the configured target z (silent truncation would be a bug).

Run from the command line:
    python -m src.data.compute_metadata --manifest manifest.json \\
        --derivatives-dir /path/to/derivatives \\
        --target-z 93

Idempotent: if a mask already exists and --overwrite is not passed, it is
reused (mask file loaded, metadata recomputed from it only if missing in manifest).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .manifest import load_manifest
from .masks import compute_brain_mask, mask_fraction
from .normalize import compute_norm_ref
from .padding import center_pad_mask
from .reader import VolumeReader

logger = logging.getLogger(__name__)

# Default target shape for the project. xy is fixed at 128 (IBC's in-plane
# is consistent), z defaults to 93 based on observed data so far. Configurable
# at runtime via --target-z.
DEFAULT_TARGET_XY = 128
DEFAULT_TARGET_Z = 93


@dataclass
class RunMetadata:
    """Extra fields we attach to each manifest entry after this stage."""

    mask_path: str  # relative to derivatives_dir
    norm_ref: float
    tsnr_mean_in_brain: float
    mask_fraction: float


def compute_tsnr(reader: VolumeReader, mask: np.ndarray) -> float:
    """Compute mean tSNR over brain voxels.

    tSNR = mean(voxel_timecourse) / std(voxel_timecourse), per voxel.
    We return the mean of that over brain voxels only.

    Reads the full 4D run. For a 262-volume run this is ~3.4 GB as float32 —
    manageable on any reasonable machine but not free. Called once per run offline.
    """
    data = np.asarray(reader.img.dataobj, dtype=np.float32)  # (X, Y, Z, T)
    mean_tc = data.mean(axis=-1)
    std_tc = data.std(axis=-1)

    # Avoid division by zero: where std is zero, voxel didn't vary over time
    # (almost certainly background). We just exclude those from the mean.
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr_map = np.where(std_tc > 0, mean_tc / std_tc, 0.0)

    brain_tsnr = tsnr_map[mask]
    brain_tsnr = brain_tsnr[np.isfinite(brain_tsnr) & (brain_tsnr > 0)]
    if brain_tsnr.size == 0:
        return 0.0
    return float(brain_tsnr.mean())


def process_run(
    run_entry: dict,
    bids_root: Path,
    derivatives_dir: Path,
    target_shape: tuple[int, int, int],
    overwrite: bool = False,
) -> RunMetadata:
    """Compute all per-run metadata. Writes the (padded) HR mask to disk.

    The mask saved to disk is at target_shape — already padded. This way the
    Dataset doesn't have to pad it on every load.

    Norm ref and tSNR are computed on the un-padded mean volume so they aren't
    biased by the zero-padding regions.
    """
    run_path = bids_root / run_entry["path"]
    run_id = run_entry["run_id"]
    mask_rel = f"masks/{run_id}_mask.nii.gz"
    mask_abs = derivatives_dir / mask_rel

    reader = VolumeReader(run_path)

    # Verify this run fits in target_shape — fail loud if not
    for axis, name in enumerate(["x", "y", "z"]):
        if reader.shape[axis] > target_shape[axis]:
            raise ValueError(
                f"Run {run_id} has {name}={reader.shape[axis]} which exceeds "
                f"target_{name}={target_shape[axis]}. "
                f"Re-run with a larger --target-{name}."
            )

    # Mean volume — computed once, used for mask + norm_ref. Not padded.
    logger.info(f"  Reading {run_id} (shape {reader.shape})...")
    mean_vol = reader.read_mean()  # float32, (X, Y, Z) at native shape

    # Brain mask: compute or load from cache. The on-disk mask is at
    # target_shape (padded). The unpadded mask is what we use locally for
    # norm_ref and tSNR (so they aren't biased by padding zeros).
    if mask_abs.exists() and not overwrite:
        logger.info(f"  Loading existing mask: {mask_rel}")
        mask_padded = np.asarray(nib.load(str(mask_abs)).dataobj).astype(bool)
        if mask_padded.shape != target_shape:
            raise RuntimeError(
                f"Cached mask shape {mask_padded.shape} != target_shape {target_shape} "
                f"for {run_id}. Try --overwrite."
            )
        # We need the unpadded version for norm_ref / tSNR. Crop back.
        # Using compute_pad_widths inverse: the brain is centered, so we can
        # crop the central region matching the original shape.
        pad_widths = []
        for axis in range(3):
            diff = target_shape[axis] - reader.shape[axis]
            before = diff // 2
            pad_widths.append((before, before + reader.shape[axis]))
        mask_unpadded = mask_padded[
            pad_widths[0][0]:pad_widths[0][1],
            pad_widths[1][0]:pad_widths[1][1],
            pad_widths[2][0]:pad_widths[2][1],
        ]
    else:
        logger.info(f"  Computing mask for {run_id}")
        mask_unpadded = compute_brain_mask(mean_vol)
        # Pad to target_shape before saving
        mask_padded = center_pad_mask(mask_unpadded, target_shape)
        # Save padded mask. Affine is approximate — strictly the padding shifts
        # the origin by `before` voxels, but this affine is just for viewer
        # alignment and is rarely critical here. If you visualize the mask
        # against the unpadded data, expect a small offset.
        mask_img = nib.Nifti1Image(mask_padded.astype(np.uint8), affine=reader.img.affine)
        mask_abs.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask_img, str(mask_abs))
        logger.info(f"  Wrote padded mask {mask_rel} (shape {target_shape})")

    # Normalization reference and tSNR: computed on the unpadded mask + data
    norm_ref = compute_norm_ref(mean_vol, mask_unpadded)
    tsnr = compute_tsnr(reader, mask_unpadded)
    frac = mask_fraction(mask_unpadded)

    return RunMetadata(
        mask_path=mask_rel,
        norm_ref=norm_ref,
        tsnr_mean_in_brain=tsnr,
        mask_fraction=frac,
    )


def compute_all(
    manifest_path: Path,
    derivatives_dir: Path,
    target_shape: tuple[int, int, int] = (DEFAULT_TARGET_XY, DEFAULT_TARGET_XY, DEFAULT_TARGET_Z),
    overwrite: bool = False,
) -> None:
    """Process every run in the manifest, write updated manifest in place.

    Args:
        manifest_path: path to manifest JSON to read and update.
        derivatives_dir: where to write masks and other derivatives.
        target_shape: (X, Y, Z) shape to which all masks (and later, data
            volumes during training) are padded. Default (128, 128, 93).
            Crashes if any run exceeds this — re-run with larger target.
        overwrite: recompute masks even if they already exist.
    """
    manifest = load_manifest(manifest_path)
    bids_root = Path(manifest["bids_root"])
    derivatives_dir = Path(derivatives_dir).resolve()
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {manifest['n_runs']} runs from {bids_root}")
    logger.info(f"Writing derivatives to {derivatives_dir}")
    logger.info(f"Target shape: {target_shape}")

    max_seen = [0, 0, 0]  # track max dimensions encountered

    for i, entry in enumerate(manifest["runs"], start=1):
        logger.info(f"[{i}/{manifest['n_runs']}] {entry['run_id']}")
        # Track max dims regardless of success
        for axis in range(3):
            if entry["shape"][axis] > max_seen[axis]:
                max_seen[axis] = entry["shape"][axis]

        try:
            metadata = process_run(
                entry, bids_root, derivatives_dir, target_shape, overwrite=overwrite,
            )
        except Exception as e:
            logger.error(f"  FAILED on {entry['run_id']}: {e}")
            entry["metadata_error"] = str(e)
            continue

        entry["mask_path"] = metadata.mask_path
        entry["norm_ref"] = metadata.norm_ref
        entry["tsnr_mean_in_brain"] = metadata.tsnr_mean_in_brain
        entry["mask_fraction"] = metadata.mask_fraction
        logger.info(
            f"  norm_ref={metadata.norm_ref:.1f}  "
            f"tSNR={metadata.tsnr_mean_in_brain:.1f}  "
            f"mask_frac={metadata.mask_fraction:.3f}"
        )

    manifest["derivatives_dir"] = str(derivatives_dir)
    manifest["target_shape"] = list(target_shape)
    manifest["max_observed_shape"] = max_seen

    with Path(manifest_path).open("w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Updated manifest written to {manifest_path}")
    logger.info(f"Max shape observed across all runs: {tuple(max_seen)}")
    if any(max_seen[a] > target_shape[a] for a in range(3)):
        # This shouldn't happen because process_run would have raised, but
        # double-check defensively.
        logger.warning(
            f"WARNING: max observed shape {tuple(max_seen)} exceeds target "
            f"{target_shape}. Re-run with a larger target."
        )
    elif any(max_seen[a] < target_shape[a] for a in range(3)):
        logger.info(
            f"Note: target shape {target_shape} is larger than observed max "
            f"{tuple(max_seen)}. You could use a tighter target to save padding."
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSON")
    parser.add_argument(
        "--derivatives-dir",
        type=Path,
        required=True,
        help="Directory to write brain masks and other derivatives into",
    )
    parser.add_argument(
        "--target-x",
        type=int,
        default=DEFAULT_TARGET_XY,
        help=f"Target X dimension after padding (default: {DEFAULT_TARGET_XY})",
    )
    parser.add_argument(
        "--target-y",
        type=int,
        default=DEFAULT_TARGET_XY,
        help=f"Target Y dimension after padding (default: {DEFAULT_TARGET_XY})",
    )
    parser.add_argument(
        "--target-z",
        type=int,
        default=DEFAULT_TARGET_Z,
        help=f"Target Z dimension after padding (default: {DEFAULT_TARGET_Z}). "
             f"Increase if any run exceeds this.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute masks even if they already exist on disk",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    target_shape = (args.target_x, args.target_y, args.target_z)
    compute_all(
        args.manifest,
        args.derivatives_dir,
        target_shape=target_shape,
        overwrite=args.overwrite,
    )
    print(f"Done. Updated manifest: {args.manifest}")


if __name__ == "__main__":
    _cli()
