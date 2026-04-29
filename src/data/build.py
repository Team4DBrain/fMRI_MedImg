"""Run the full data preparation pipeline: manifest build + metadata compute.

This is a convenience wrapper that runs both stages in order. The two stages
also exist as standalone commands (`src.data.manifest` and
`src.data.compute_metadata`) if you want to run them separately — useful when
debugging or when you only need to redo one stage.

Usage:
    python -m src.data.build \\
        --bids-root /path/to/ibc_raw \\
        --out-dir /path/to/derivatives \\
        --target-z 93

The manifest goes to `<out-dir>/manifest.json` and the masks go to
`<out-dir>/masks/`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .compute_metadata import DEFAULT_TARGET_XY, DEFAULT_TARGET_Z, compute_all
from .manifest import build_manifest, write_manifest

logger = logging.getLogger(__name__)


def build_pipeline(
    bids_root: Path,
    out_dir: Path,
    target_shape: tuple[int, int, int],
    mask_method: str = "auto",
    overwrite: bool = False,
) -> None:
    """Run both stages: manifest build, then metadata compute.

    Args:
        bids_root: where the raw data lives (read-only).
        out_dir: where to write manifest.json and the masks/ subdir.
        target_shape: (X, Y, Z) padding target. Crashes per-run if exceeded.
        mask_method: "auto", "synthstrip", or "percentile".
        overwrite: passed through to compute_metadata; recompute existing masks.
    """
    bids_root = Path(bids_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # Stage 1
    logger.info("=" * 60)
    logger.info("Stage 1: building manifest")
    logger.info("=" * 60)
    entries = build_manifest(bids_root)
    if not entries:
        logger.error(
            f"No BOLD files found under {bids_root}. "
            f"Check the path and that files match the BIDS naming pattern."
        )
        sys.exit(1)
    write_manifest(entries, bids_root, manifest_path)
    logger.info(f"Manifest written: {manifest_path} ({len(entries)} runs)")

    # Stage 2
    logger.info("")
    logger.info("=" * 60)
    logger.info("Stage 2: computing metadata (mask, norm_ref, tSNR)")
    logger.info("=" * 60)
    compute_all(
        manifest_path,
        derivatives_dir=out_dir,
        target_shape=target_shape,
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
        "--bids-root", type=Path, required=True,
        help="Path to BIDS root (raw data).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Where to write manifest.json and masks/ subdir.",
    )
    parser.add_argument(
        "--target-x", type=int, default=DEFAULT_TARGET_XY,
        help=f"Target X dimension after padding (default: {DEFAULT_TARGET_XY})",
    )
    parser.add_argument(
        "--target-y", type=int, default=DEFAULT_TARGET_XY,
        help=f"Target Y dimension after padding (default: {DEFAULT_TARGET_XY})",
    )
    parser.add_argument(
        "--target-z", type=int, default=DEFAULT_TARGET_Z,
        help=f"Target Z dimension after padding (default: {DEFAULT_TARGET_Z}). "
             f"Increase if any run exceeds this.",
    )
    parser.add_argument(
        "--mask-method",
        choices=["auto", "synthstrip", "percentile"],
        default="auto",
        help="Brain masking method (see masks.compute_brain_mask). Default 'auto'.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute masks even if they already exist on disk.",
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
        target_shape=(args.target_x, args.target_y, args.target_z),
        mask_method=args.mask_method,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    _cli()
