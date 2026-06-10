"""main.py — inference pipeline for fMRI temporal interpolation.

Loads a trained checkpoint, reads a 4D BOLD NIfTI, and generates a new 4D
NIfTI with one interpolated frame inserted between every pair of original
frames. The output time axis is therefore (2T - 1) long for a T-frame input.

Output frame ordering, for input frames V_0, V_1, ..., V_{T-1}:

    out_0  = V_0
    out_1  = f(V_0, V_2)          # interpolated between V_0 and V_2 ... but
                                   # better: insert between consecutive frames,
                                   # see --mode below.
    out_2  = V_1
    ...

Two modes are supported:

    --mode insert (default)
        For each consecutive pair (V_i, V_{i+1}), predict an intermediate frame
        f(V_i, V_{i+1}). Output length = 2T - 1. The model was trained on
        triplets (V_t, V_{t+1}, V_{t+2}) where the "neighbors" were two steps
        apart, so this mode treats consecutive frames as those neighbors.

    --mode fill-gaps
        Same model call, but only inserted frames are returned (length T - 1).
        Useful if you want just the synthetic frames.

The output NIfTI keeps the original affine and header (with TR halved when
nibabel can read it from pixdim[4]).

Example:

    python main.py \\
        --weights checkpoints/pretrained/model_weights.pt \\
        --input data/sub-01_ses-00_task-X_bold.nii.gz \\
        --output results/sub-01_ses-00_task-X_bold_2x.nii.gz
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import UNet3D
from src.utils import pick_device


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


def load_weights(path: str, model: torch.nn.Module, device: torch.device) -> None:
    """Load weights from either model_weights.pt or a full checkpoint."""
    obj = torch.load(path, map_location=device)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state)


def normalize_pair(v_a: np.ndarray, v_b: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normalize a frame pair with input-only stats. Returns normalized pair + (offset, scale) for denorm.

    Mirrors the dataset normalization so inference statistics match training.
    """
    mask_source = np.abs(v_a) + np.abs(v_b)
    threshold = np.percentile(mask_source, 20)
    mask = mask_source > threshold
    if int(mask.sum()) >= 100:
        pool = np.concatenate([v_a[mask], v_b[mask]]).astype(np.float32, copy=False)
    else:
        pool = np.concatenate([v_a.ravel(), v_b.ravel()]).astype(np.float32, copy=False)

    if mode == "zscore":
        mu = float(pool.mean())
        sigma = float(max(pool.std(), 1e-6))
        return (v_a - mu) / sigma, (v_b - mu) / sigma, mu, sigma
    # percentile
    lo = float(np.percentile(pool, 1))
    hi = float(np.percentile(pool, 99))
    sigma = max(hi - lo, 1e-6)
    a = np.clip((v_a - lo) / sigma, 0.0, 1.0)
    b = np.clip((v_b - lo) / sigma, 0.0, 1.0)
    return a, b, lo, sigma


def to_xyz(volume: np.ndarray) -> np.ndarray:
    """Convert (D, H, W) tensor-order back to (X, Y, Z) NIfTI order."""
    return np.ascontiguousarray(volume.transpose(2, 1, 0))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input NIfTI lazily.
    img = nib.load(args.input)
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D BOLD, got shape {img.shape}")
    T = int(img.shape[-1])
    if T < 2:
        raise ValueError(f"Need at least 2 frames, got T={T}")

    # Load model.
    model = UNet3D(in_channels=2, out_channels=1,
                   base_channels=args.base_channels, depth=args.depth).to(device)
    load_weights(args.weights, model, device)
    model.eval()

    print(f"input={args.input} T={T} device={device.type} mode={args.mode}")

    spatial_xyz = img.shape[:3]   # (X, Y, Z)
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    # We assemble the output along the time axis.
    if args.mode == "insert":
        out_T = 2 * T - 1
    else:
        out_T = T - 1
    out_arr = np.empty(spatial_xyz + (out_T,), dtype=out_dtype)

    # In insert mode, originals land at even indices (0, 2, 4, ...) and
    # synthetic frames at odd indices (1, 3, 5, ...). In fill-gaps mode we
    # only write the synthetic frames.
    if args.mode == "insert":
        # Pre-copy original frames into even slots.
        for i in range(T):
            out_arr[..., 2 * i] = np.asarray(img.dataobj[..., i], dtype=out_dtype)

    with torch.no_grad():
        for i in range(T - 1):
            v_a = np.asarray(img.dataobj[..., i], dtype=np.float32)
            v_b = np.asarray(img.dataobj[..., i + 1], dtype=np.float32)
            a_n, b_n, offset, scale = normalize_pair(v_a, v_b, args.norm_mode)

            a_t = torch.from_numpy(np.ascontiguousarray(a_n.transpose(2, 1, 0))).float()
            b_t = torch.from_numpy(np.ascontiguousarray(b_n.transpose(2, 1, 0))).float()
            x = torch.stack([a_t, b_t], dim=0).unsqueeze(0).to(device)

            raw = model(x)
            pred = (0.5 * (x[:, 0:1] + x[:, 1:2]) + raw) if args.residual else raw

            mid_norm = pred.squeeze(0).squeeze(0).cpu().numpy()
            # Both zscore and percentile denormalize as: mid = mid_norm * scale + offset.
            mid = mid_norm * scale + offset
            mid_xyz = to_xyz(mid).astype(out_dtype)

            slot = 2 * i + 1 if args.mode == "insert" else i
            out_arr[..., slot] = mid_xyz

            if (i + 1) % 25 == 0 or i == T - 2:
                print(f"  pair {i + 1}/{T - 1} done")

    # Build output NIfTI with adjusted TR (halved) for insert mode.
    new_header = img.header.copy()
    if args.mode == "insert":
        try:
            pixdim = new_header["pixdim"].copy()
            if pixdim[4] > 0:
                pixdim[4] = pixdim[4] / 2.0
                new_header["pixdim"] = pixdim
        except Exception:
            pass

    out_img = nib.Nifti1Image(out_arr, affine=img.affine, header=new_header)
    nib.save(out_img, str(out_path))
    print(f"wrote {out_path}  shape={out_arr.shape}  dtype={out_arr.dtype}")


if __name__ == "__main__":
    main()
