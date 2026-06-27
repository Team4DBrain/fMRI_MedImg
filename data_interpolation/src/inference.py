"""Reusable inference API for fMRI temporal interpolation.

Other project modules should import `interpolate_file` for a one-shot call, or
`FMRIInterpolator` when processing multiple files with the same checkpoint.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .model import UNet3D
from .utils import pick_device


def load_weights(path: str | Path, model: torch.nn.Module, device: torch.device) -> None:
    """Load weights from either `model_weights.pt` or a full checkpoint."""
    obj = torch.load(path, map_location=device)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state)


def normalize_pair(v_a: np.ndarray, v_b: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normalize a frame pair with input-only stats.

    Returns the normalized pair and `(offset, scale)` for denormalization.
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

    if mode != "percentile":
        raise ValueError("norm_mode must be 'zscore' or 'percentile'")

    lo = float(np.percentile(pool, 1))
    hi = float(np.percentile(pool, 99))
    sigma = max(hi - lo, 1e-6)
    a = np.clip((v_a - lo) / sigma, 0.0, 1.0)
    b = np.clip((v_b - lo) / sigma, 0.0, 1.0)
    return a, b, lo, sigma


def to_xyz(volume: np.ndarray) -> np.ndarray:
    """Convert tensor order `(D, H, W)` back to NIfTI order `(X, Y, Z)`."""
    return np.ascontiguousarray(volume.transpose(2, 1, 0))


class FMRIInterpolator:
    """Load one trained interpolation model and apply it to BOLD NIfTI files."""

    def __init__(
        self,
        weights: str | Path,
        *,
        norm_mode: str = "zscore",
        device: str | None = None,
        base_channels: int = 32,
        depth: int = 4,
        residual: bool = False,
    ):
        self.weights = Path(weights)
        self.norm_mode = norm_mode
        self.device = pick_device(device)
        self.residual = residual

        self.model = UNet3D(
            in_channels=2,
            out_channels=1,
            base_channels=base_channels,
            depth=depth,
        ).to(self.device)
        load_weights(self.weights, self.model, self.device)
        self.model.eval()

    def interpolate(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        mode: str = "insert",
        dtype: str = "float32",
        verbose: bool = True,
    ) -> Path:
        """Create an interpolated NIfTI from a 4D BOLD input file.

        Args:
            input_path: Source 4D BOLD NIfTI (`*.nii` or `*.nii.gz`).
            output_path: Destination NIfTI path.
            mode: `"insert"` writes original and synthetic frames (`2T - 1`);
                `"fill-gaps"` writes only synthetic frames (`T - 1`).
            dtype: Output data dtype, `"float32"` or `"float64"`.
            verbose: Print progress.

        Returns:
            The written output path.
        """
        if mode not in {"insert", "fill-gaps"}:
            raise ValueError("mode must be 'insert' or 'fill-gaps'")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = nib.load(str(input_path))
        if len(img.shape) != 4:
            raise ValueError(f"Expected 4D BOLD, got shape {img.shape}")
        t_len = int(img.shape[-1])
        if t_len < 2:
            raise ValueError(f"Need at least 2 frames, got T={t_len}")

        if verbose:
            print(f"input={input_path} T={t_len} device={self.device.type} mode={mode}")

        spatial_xyz = img.shape[:3]
        out_dtype = np.float32 if dtype == "float32" else np.float64
        out_t = 2 * t_len - 1 if mode == "insert" else t_len - 1
        out_arr = np.empty(spatial_xyz + (out_t,), dtype=out_dtype)

        if mode == "insert":
            for i in range(t_len):
                out_arr[..., 2 * i] = np.asarray(img.dataobj[..., i], dtype=out_dtype)

        with torch.no_grad():
            for i in range(t_len - 1):
                v_a = np.asarray(img.dataobj[..., i], dtype=np.float32)
                v_b = np.asarray(img.dataobj[..., i + 1], dtype=np.float32)
                a_n, b_n, offset, scale = normalize_pair(v_a, v_b, self.norm_mode)

                a_t = torch.from_numpy(np.ascontiguousarray(a_n.transpose(2, 1, 0))).float()
                b_t = torch.from_numpy(np.ascontiguousarray(b_n.transpose(2, 1, 0))).float()
                x = torch.stack([a_t, b_t], dim=0).unsqueeze(0).to(self.device)

                raw = self.model(x)
                pred = (0.5 * (x[:, 0:1] + x[:, 1:2]) + raw) if self.residual else raw

                mid_norm = pred.squeeze(0).squeeze(0).cpu().numpy()
                mid = mid_norm * scale + offset
                mid_xyz = to_xyz(mid).astype(out_dtype)

                slot = 2 * i + 1 if mode == "insert" else i
                out_arr[..., slot] = mid_xyz

                if verbose and ((i + 1) % 25 == 0 or i == t_len - 2):
                    print(f"  pair {i + 1}/{t_len - 1} done")

        new_header = img.header.copy()
        if mode == "insert":
            try:
                pixdim = new_header["pixdim"].copy()
                if pixdim[4] > 0:
                    pixdim[4] = pixdim[4] / 2.0
                    new_header["pixdim"] = pixdim
            except Exception:
                pass

        out_img = nib.Nifti1Image(out_arr, affine=img.affine, header=new_header)
        nib.save(out_img, str(output_path))

        if verbose:
            print(f"wrote {output_path}  shape={out_arr.shape}  dtype={out_arr.dtype}")
        return output_path


def interpolate_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    weights: str | Path = Path(__file__).resolve().parent.parent.parent / "weights" / "temporal" / "model_weights.pt",
    norm_mode: str = "zscore",
    device: str | None = None,
    base_channels: int = 32,
    depth: int = 4,
    residual: bool = False,
    mode: str = "insert",
    dtype: str = "float32",
    verbose: bool = True,
) -> Path:
    """One-shot helper for modules that just need an interpolated output file."""
    interpolator = FMRIInterpolator(
        weights,
        norm_mode=norm_mode,
        device=device,
        base_channels=base_channels,
        depth=depth,
        residual=residual,
    )
    return interpolator.interpolate(
        input_path,
        output_path,
        mode=mode,
        dtype=dtype,
        verbose=verbose,
    )
