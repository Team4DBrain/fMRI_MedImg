"""Compute per-run metadata from a manifest: brain mask, norm reference, tSNR,
and the per-run z-axis crop offset.

For each run:
  1. Read the full 4D once. Compute the temporal mean from the same buffer.
  2. Compute a brain mask at NATIVE shape (X, Y, Z_native).
  3. Compute a z-axis crop offset (`z_start`) that centers the brain's z-bbox
     in a target_z window. xy is unchanged (IBC has uniform xy=128×128).
  4. Crop the mask to (X, Y, target_z) using z_start. Update its affine.
     Save to disk.
  5. Compute norm_ref and tSNR on the in-brain voxels (cropping doesn't change
     these values since the bbox is fully contained in the crop window).
  6. Update the manifest entry with mask_path, z_start, norm_ref, tsnr,
     mask_fraction.

Pipeline shape:
  - target_shape on disk for masks is (X_native, Y_native, target_z) where
    X_native = Y_native = 128 for IBC.
  - Datasets read native data and crop z using the stored z_start.
  - target_z defaults to the smallest observed Z across runs (auto). Override
    with --target-z if needed.

Run from the command line:
    python -m src.data.compute_metadata --manifest manifest.json \
        --derivatives-dir /path/to/derivatives \
        --target-z 84

Idempotent: if a mask already exists and --overwrite is not passed, it is
reused. Existing-mask shape MUST match (X, Y, target_z) or we abort with a
clear error (re-run with --overwrite to regenerate stale masks).
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

from .cropping import compute_z_start, crop_z, update_affine_for_z_crop
from .manifest import load_manifest
from .masks import compute_brain_mask, mask_fraction
from .normalize import compute_norm_ref
from .reader import VolumeReader

logger = logging.getLogger(__name__)

# IBC has uniform xy. We don't crop or pad xy at all in option B.
DEFAULT_TARGET_XY = 128
# target_z is per-dataset: defaults to min observed native_z if not specified.


@dataclass
class RunMetadata:
    """Extra fields we attach to each manifest entry after this stage."""

    mask_path: str  # relative to derivatives_dir
    z_start: int    # offset into native z where the crop window begins
    norm_ref: float
    tsnr_mean_in_brain: float
    mask_fraction: float


def compute_tsnr_from_data(data_cropped: np.ndarray, mask: np.ndarray) -> float:
    """Compute mean tSNR over brain voxels from already-cropped 4D data.

    tSNR = mean(voxel_timecourse) / std(voxel_timecourse), per voxel.

    Pure: no I/O, no reader. Caller is responsible for passing the 4D run
    already cropped to (X, Y, target_z, T) so statistics match what the
    Dataset serves. Float64 accumulators on the temporal reduction —
    float32 sum over 300 timepoints accumulates enough drift to bias tSNR.
    """
    mean_tc = data_cropped.mean(axis=-1, dtype=np.float64)
    std_tc = data_cropped.std(axis=-1, dtype=np.float64)

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
    target_z: int,
    overwrite: bool = False,
    mask_method: str = "auto",
) -> RunMetadata:
    """Compute mask, z_start, norm_ref, tSNR for one run. Save cropped mask.

    The mask saved to disk is at (X, Y, target_z). xy is NOT modified —
    IBC's xy is uniform 128×128 across the dataset and there is nothing
    to gain by cropping or padding it.

    Args:
        target_z: the dataset-wide z target. The crop window is centered on
            the brain's z-bbox in this run.
        mask_method: "auto", "synthstrip", or "percentile". See masks.py.
    """
    run_path = bids_root / run_entry["path"]
    run_id = run_entry["run_id"]
    mask_rel = f"masks/{run_id}_mask.nii.gz"
    mask_abs = derivatives_dir / mask_rel

    reader = VolumeReader(run_path)
    native_shape = reader.shape3d
    expected_cropped_shape = (native_shape[0], native_shape[1], target_z)

    if native_shape[2] < target_z:
        raise ValueError(
            f"Run {run_id} has native z={native_shape[2]} which is smaller "
            f"than target_z={target_z}. The crop pipeline cannot grow volumes; "
            "lower --target-z so it fits the shortest run."
        )

    logger.info(f"  Reading {run_id} (native shape {native_shape})...")
    # Read the full 4D ONCE. Mean for masking + norm_ref, full for tSNR.
    # Previous version read the file twice — once for read_mean(), once
    # inside compute_tsnr(). For a 1.6 GB run that doubled the I/O cost
    # of the offline build. Holding the buffer for the duration of one
    # process_run is ~1.6 GB peak, released between runs.
    full = reader.read_full(dtype=np.float32)              # (X, Y, Z_native, T)
    mean_vol = full.mean(axis=-1, dtype=np.float64).astype(np.float32)

    # Mask + z_start: compute fresh or load from disk.
    if mask_abs.exists() and not overwrite and "z_start" in run_entry:
        logger.info(f"  Loading existing mask: {mask_rel}")
        mask_cropped = np.asarray(nib.load(str(mask_abs)).dataobj).astype(bool)
        if mask_cropped.shape != expected_cropped_shape:
            raise RuntimeError(
                f"Cached mask shape {mask_cropped.shape} != expected "
                f"{expected_cropped_shape} for {run_id}. The mask was likely "
                "built under a different target_z (or under the old padding "
                "pipeline). Re-run with --overwrite to regenerate."
            )
        z_start = int(run_entry["z_start"])
    else:
        logger.info(f"  Computing mask for {run_id} (method={mask_method})")
        mask_native = compute_brain_mask(
            mean_vol, affine=reader.img.affine, method=mask_method,
        )
        # Center the crop window on the brain's z-bbox.
        z_start = compute_z_start(mask_native, target_z)
        mask_cropped = crop_z(mask_native, z_start, target_z)

        # Save the cropped mask with an affine that reflects the z-shift,
        # so external viewers (FSLeyes, ITK-SNAP) place it in world space
        # consistently with the underlying anatomy.
        cropped_affine = update_affine_for_z_crop(reader.img.affine, z_start)
        mask_img = nib.Nifti1Image(mask_cropped.astype(np.uint8), affine=cropped_affine)
        mask_abs.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask_img, str(mask_abs))
        logger.info(
            f"  Wrote mask {mask_rel} (shape {expected_cropped_shape}, z_start={z_start})"
        )

    # norm_ref: same brain voxels whether we use native+native or cropped+cropped,
    # because the mask's brain bbox is fully inside the crop window. Use
    # cropped versions for consistency with what the Dataset will serve.
    mean_cropped = crop_z(mean_vol, z_start, target_z)
    norm_ref = compute_norm_ref(mean_cropped, mask_cropped)
    # tSNR from the same already-loaded 4D buffer; one disk read total.
    data_cropped = crop_z(full, z_start, target_z)
    tsnr = compute_tsnr_from_data(data_cropped, mask_cropped)
    frac = mask_fraction(mask_cropped)

    # Sanity warning. With z-only cropping the denominator changes slightly
    # (we removed non-brain z-slices), but ~0.2-0.4 is still the expected
    # range for whole-brain BOLD masks. >0.55 still indicates contamination.
    if frac > 0.55:
        logger.warning(
            f"  mask_fraction={frac:.3f} for {run_id} is suspiciously high "
            f"(expected ~0.2-0.4). Mask likely includes non-brain tissue. "
            f"Use mask_method=synthstrip for usable results."
        )

    return RunMetadata(
        mask_path=mask_rel,
        z_start=z_start,
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
        target_z: z-axis target. If None (default), auto-detect as the smallest
            observed native z across runs (so no run needs padding). xy is not
            cropped or padded.
        overwrite: recompute masks even if they already exist.
        mask_method: "auto", "synthstrip", or "percentile". See masks.py.
    """
    manifest = load_manifest(manifest_path)
    bids_root = Path(manifest["bids_root"])
    derivatives_dir = Path(derivatives_dir).resolve()
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    # Native shapes from the manifest. (build_manifest stores 4D shape; first 3
    # are spatial.)
    native_shapes = [tuple(r["shape"][:3]) for r in manifest["runs"]]
    xs = {s[0] for s in native_shapes}
    ys = {s[1] for s in native_shapes}
    zs = sorted({s[2] for s in native_shapes})
    if len(xs) > 1 or len(ys) > 1:
        raise ValueError(
            f"Non-uniform xy across runs (x={xs}, y={ys}). The current "
            "z-only crop pipeline assumes uniform xy. xy-cropping is not "
            "implemented; talk to the data pipeline owner before adding it."
        )
    native_x, native_y = next(iter(xs)), next(iter(ys))

    if target_z is None:
        target_z = min(zs)
        logger.info(
            f"target_z auto-detected as {target_z} (min z across {len(native_shapes)} runs)"
        )
    else:
        if target_z > min(zs):
            raise ValueError(
                f"--target-z={target_z} exceeds the smallest native z "
                f"({min(zs)} for at least one run). Cropping cannot grow "
                "volumes. Lower target_z or remove the offending run."
            )

    target_shape = (native_x, native_y, target_z)
    logger.info(f"Processing {manifest['n_runs']} runs from {bids_root}")
    logger.info(f"Writing derivatives to {derivatives_dir}")
    logger.info(f"Target shape (X, Y, Z_target): {target_shape}")
    logger.info(f"Native z values seen: {zs}")
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
        entry["z_start"] = metadata.z_start
        entry["norm_ref"] = metadata.norm_ref
        entry["tsnr_mean_in_brain"] = metadata.tsnr_mean_in_brain
        entry["mask_fraction"] = metadata.mask_fraction
        # Clear any old error key from a prior failed run.
        entry.pop("metadata_error", None)
        logger.info(
            f"  z_start={metadata.z_start}  norm_ref={metadata.norm_ref:.1f}  "
            f"tSNR={metadata.tsnr_mean_in_brain:.1f}  "
            f"mask_frac={metadata.mask_fraction:.3f}"
        )

    manifest["derivatives_dir"] = str(derivatives_dir)
    manifest["target_shape"] = list(target_shape)
    manifest["target_z"] = int(target_z)
    manifest["pipeline"] = "z_crop"  # marker so Datasets can detect old manifests

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
        "--derivatives-dir",
        type=Path,
        required=True,
        help="Directory to write brain masks and other derivatives into",
    )
    parser.add_argument(
        "--target-z",
        type=int,
        default=None,
        help="Target z dimension after cropping. Default: auto (smallest observed "
             "native z across runs). Must be <= the smallest native z; cannot grow.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute masks even if they already exist on disk",
    )
    parser.add_argument(
        "--mask-method",
        choices=["auto", "synthstrip", "percentile"],
        default="auto",
        help="Brain masking method. 'auto' (default) prefers synthstrip if "
             "installed and falls back to percentile with a warning. "
             "'synthstrip' requires mri_synthstrip / synthstrip-docker / "
             "synthstrip-singularity on PATH; raises if missing. "
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
