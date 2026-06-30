"""Inference CLI for fMRI temporal interpolation.

Loads a checkpoint, reads a 4D BOLD NIfTI, and writes a new one with synthetic
frames between the originals. Two modes:

    --mode insert (default)
        For each pair (V_i, V_{i+1}), predict the in-between frame. Output is
        2T-1 long: original frames interleaved with the synthetic ones. The
        model was trained on (V_t, V_{t+1}, V_{t+2}) triplets, so here we just
        feed it consecutive frames as the neighbours.

    --mode fill-gaps
        Same calls, but return only the T-1 synthetic frames.

The output keeps the input's affine and header (TR is halved when it's readable
from pixdim[4]).

Example:

    python main.py \\
        --weights checkpoints/pretrained/model_weights.pt \\
        --input data/sub-01_ses-00_task-X_bold.nii.gz \\
        --output results/sub-01_ses-00_task-X_bold_2x.nii.gz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference import FMRIInterpolator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="Path to model_weights.pt or a full checkpoint.")
    p.add_argument("--input", required=True, help="Input 4D BOLD NIfTI (*.nii or *.nii.gz).")
    p.add_argument("--output", required=True, help="Where to write the interpolated NIfTI.")
    p.add_argument("--mode", choices=["insert", "fill-gaps"], default="insert",
                   help="insert: output 2T-1 frames (orig + interpolated). "
                        "fill-gaps: output only the T-1 synthetic frames.")
    p.add_argument("--norm-mode", choices=["zscore", "percentile"], default="zscore",
                   help="Must match the normalization used at training time.")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--residual", action="store_true",
                   help="Set if the checkpoint was trained with residual=True.")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="Output NIfTI dtype.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    interpolator = FMRIInterpolator(
        args.weights,
        norm_mode=args.norm_mode,
        device=args.device,
        base_channels=args.base_channels,
        depth=args.depth,
        residual=args.residual,
    )
    interpolator.interpolate(
        args.input,
        args.output,
        mode=args.mode,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
