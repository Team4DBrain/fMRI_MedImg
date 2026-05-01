"""Run the full data preparation pipeline: manifest build + metadata compute.

This is a convenience wrapper that runs both stages in order. The two stages
also exist as standalone commands (`src.data.manifest` and
`src.data.compute_metadata`) if you want to run them separately — useful when
debugging or when you only need to redo one stage.

Usage:
    python -m src.data.build \\
        --bids-root /path/to/ibc_raw \\
        --out-dir /path/to/derivatives

    # Override z target (default: auto = smallest observed native z)
    python -m src.data.build \\
        --bids-root /path/to/ibc_raw \\
        --out-dir /path/to/derivatives \\
        --target-z 84

The manifest goes to `<out-dir>/manifest.json` and the masks go to
`<out-dir>/masks/`.

Pipeline notes (option B, z-only crop):
  - xy is left at native (IBC: 128×128). Not configurable; the underlying
    compute_metadata raises if any run has a different xy.
  - z is cropped per-run to a fixed `target_z`, centered on the brain's
    z-bbox. target_z auto-detects to min(native_z) across the manifest's
    runs unless --target-z is passed explicitly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .compute_metadata import compute_all
from .manifest import build_manifest, write_manifest

logger = logging.getLogger(__name__)


def build_pipeline(
    bids_root: Path,
    out_dir: Path,
    target_z: int | None = None,
    mask_method: str = "auto",
    overwrite: bool = False,
) -> None:
    """Run both stages: manifest build, then metadata compute.

    Args:
        bids_root: where the raw data lives (read-only).
        out_dir: where to write manifest.json and the masks/ subdir.
        target_z: z-axis crop target. If None, auto-detect as the smallest
            observed native z across runs.
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
    logger.info("Stage 2: computing metadata (mask, z_start, norm_ref, tSNR)")
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
        "--bids-root", type=Path, required=True,
        help="Path to BIDS root (raw data).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Where to write manifest.json and masks/ subdir.",
    )
    parser.add_argument(
        "--target-z", type=int, default=None,
        help="Target z dimension after cropping. Default: auto (smallest "
             "observed native z across runs). Must be <= the smallest native z; "
             "cannot grow.",
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
        target_z=args.target_z,
        mask_method=args.mask_method,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    _cli()
