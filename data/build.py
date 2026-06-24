"""Run the full data preparation pipeline: manifest build + metadata compute.

This is a convenience wrapper that runs both stages in order. The two stages
also exist as standalone commands (`src.data.manifest` and
`src.data.compute_metadata`) if you want to run them separately — useful when
debugging or when you only need to redo one stage.

Usage:
    python -m src.data.build \\
        --bids-root /path/to/ibc_raw \\
        --out-dir /path/to/derivatives

    # Override expected z (default 93 — drops the IBC z=84 ses-00/01 anomaly)
    python -m src.data.build \\
        --bids-root /path/to/ibc_raw \\
        --out-dir /path/to/derivatives \\
        --target-z 93

The manifest goes to `<out-dir>/manifest.json` and the masks go to
`<out-dir>/masks/`.

Pipeline notes (no_crop_v1):
  - target_z is fixed across the dataset; runs with non-conforming z are
    dropped at the manifest stage (with a logged warning).
  - xy is left at native (IBC: 128×128). compute_metadata raises if any
    surviving run has different xy.
  - No cropping. The mask saved on disk is at native (= target) shape.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ._cli import DEFAULT_BIDS_ROOT, DEFAULT_DERIVATIVES_DIR, existing_dir
from .compute_metadata import compute_all
from .manifest import DEFAULT_REQUIRE_Z, build_manifest, write_manifest

logger = logging.getLogger(__name__)


def build_pipeline(
    bids_root: Path,
    out_dir: Path,
    target_z: int = DEFAULT_REQUIRE_Z,
    mask_method: str = "auto",
    overwrite: bool = False,
) -> None:
    """Run both stages: manifest build (filtering by target_z), then metadata compute.

    Args:
        bids_root: where the raw data lives (read-only).
        out_dir: where to write manifest.json and the masks/ subdir.
        target_z: required uniform z dimension. Runs that don't match are
            dropped at the manifest stage. Default 93 (IBC standard).
        mask_method: "auto", "synthstrip", or "percentile".
        overwrite: passed through to compute_metadata; recompute existing
            masks/metadata.
    """
    bids_root = Path(bids_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # Stage 1
    logger.info("=" * 60)
    logger.info("Stage 1: building manifest (require_z=%d)", target_z)
    logger.info("=" * 60)
    entries = build_manifest(bids_root, require_z=target_z)
    if not entries:
        logger.error(
            "No conforming BOLD files found under %s. Either nothing matched "
            "the BIDS naming pattern, or every file was filtered out by "
            "require_z=%d. Check the path and the z dimension of your data.",
            bids_root, target_z,
        )
        sys.exit(1)
    write_manifest(entries, bids_root, manifest_path, require_z=target_z)
    logger.info(f"Manifest written: {manifest_path} ({len(entries)} runs)")

    # Stage 2
    logger.info("")
    logger.info("=" * 60)
    logger.info("Stage 2: computing metadata (mask, norm_ref, tSNR)")
    logger.info("=" * 60)
    compute_all(
        manifest_path,
        derivatives_dir=out_dir,
        target_z=target_z,
        overwrite=overwrite,
        mask_method=mask_method,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Done. Manifest + derivatives at: {out_dir}")
    logger.info("=" * 60)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root", type=existing_dir, default=str(DEFAULT_BIDS_ROOT),
        help=f"Path to BIDS root (raw data). Default: {DEFAULT_BIDS_ROOT} "
             "(team VM convention). Must exist.",
    )
    parser.add_argument(
        "--out-dir", type=existing_dir, default=str(DEFAULT_DERIVATIVES_DIR),
        help=f"Where to write manifest.json and masks/ subdir. Default: "
             f"{DEFAULT_DERIVATIVES_DIR} (team VM convention). Must exist; "
             "create it explicitly first if it doesn't.",
    )
    parser.add_argument(
        "--target-z", type=int, default=DEFAULT_REQUIRE_Z,
        help=f"Required uniform z dimension. Default: {DEFAULT_REQUIRE_Z} "
             "(IBC standard; drops the z=84 ses-00/01 anomaly). "
             "Runs not matching this z are logged and dropped at stage 1.",
    )
    parser.add_argument(
        "--mask-method",
        choices=["auto", "synthstrip", "percentile"],
        default="auto",
        help="Brain masking method (see masks.compute_brain_mask). Default 'auto'.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute masks/metadata even if already complete on disk.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    build_pipeline(
        bids_root=args.bids_root,
        out_dir=args.out_dir,
        target_z=args.target_z,
        mask_method=args.mask_method,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    _cli()
