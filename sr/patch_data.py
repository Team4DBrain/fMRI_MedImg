"""Patch-based training dataset for ``srcnn3d_patch``.

Purpose:
    Expand each training volume into multiple random HR/LR patch pairs per epoch,
    using aligned crops from the full-volume tensors produced by
    ``SpatialSRDataset`` (global k-space LR, then crop — not per-patch degrade).
Effects:
    Train loader length becomes ``n_train_volumes * patches_per_volume``.
    Validation still uses full volumes via ``build_loaders`` in ``data.py``.
Influences:
    ``patch_hr_shape``, ``patches_per_volume``, and ``seed`` from ``SRConfig``.
How to change safely:
  Keep the return dict keys identical to ``SpatialSRDataset`` so ``train.py``
  does not need patch-specific batch handling beyond shape alignment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def hr_to_lr_crop(
    hr_start: tuple[int, int, int],
    hr_size: tuple[int, int, int],
    hr_shape: tuple[int, int, int],
    lr_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Map an HR crop box to proportional LR indices and sizes."""
    lr_start: list[int] = []
    lr_size: list[int] = []
    for axis in range(3):
        scale = lr_shape[axis] / hr_shape[axis]
        lr_size.append(max(1, int(round(hr_size[axis] * scale))))
        lr_start.append(int(round(hr_start[axis] * scale)))
        lr_start[-1] = min(lr_start[-1], lr_shape[axis] - lr_size[-1])
        lr_start[-1] = max(0, lr_start[-1])
        if lr_start[-1] + lr_size[-1] > lr_shape[axis]:
            lr_size[-1] = lr_shape[axis] - lr_start[-1]
    return tuple(lr_start), tuple(lr_size)


def crop_spatial_5d(
    tensor: torch.Tensor,
    start: tuple[int, int, int],
    size: tuple[int, int, int],
) -> torch.Tensor:
    """Crop trailing (D, H, W) dims; supports (C, D, H, W) and (N, C, D, H, W)."""
    d0, h0, w0 = start
    pd, ph, pw = size
    if tensor.ndim == 4:
        return tensor[:, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]
    if tensor.ndim == 5:
        return tensor[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(tensor.shape)}")


def random_hr_crop_start(
    hr_shape: tuple[int, int, int],
    patch_hr_shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Sample a valid top-left corner for an HR patch inside ``hr_shape``."""
    starts: list[int] = []
    for axis in range(3):
        max_start = hr_shape[axis] - patch_hr_shape[axis]
        if max_start < 0:
            raise ValueError(
                f"patch_hr_shape {patch_hr_shape} exceeds volume shape {hr_shape} "
                f"on axis {axis}"
            )
        starts.append(int(rng.integers(0, max_start + 1)) if max_start > 0 else 0)
    return tuple(starts)


class PatchTrainingDataset(Dataset):
    """Random HR/LR patches from full-volume ``SpatialSRDataset`` samples."""

    def __init__(
        self,
        base: Dataset,
        *,
        patch_hr_shape: tuple[int, int, int],
        patches_per_volume: int,
        seed: int,
    ) -> None:
        if patches_per_volume < 1:
            raise ValueError("patches_per_volume must be >= 1")
        self.base = base
        self.patch_hr_shape = tuple(patch_hr_shape)
        self.patches_per_volume = int(patches_per_volume)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base) * self.patches_per_volume

    def __getitem__(self, idx: int) -> dict[str, Any]:
        volume_idx = idx // self.patches_per_volume
        patch_idx = idx % self.patches_per_volume
        full = self.base[volume_idx]

        target = full["target"]
        inputs = full["input"]
        mask_hr = full["mask_hr"]
        mask_lr = full["mask_lr"]

        hr_shape = tuple(target.shape[-3:])
        lr_shape = tuple(inputs.shape[-3:])
        hr_start = random_hr_crop_start(
            hr_shape, self.patch_hr_shape, self._patch_rng(volume_idx, patch_idx)
        )
        lr_start, lr_size = hr_to_lr_crop(
            hr_start, self.patch_hr_shape, hr_shape, lr_shape
        )

        out: dict[str, Any] = {
            "input": crop_spatial_5d(inputs, lr_start, lr_size),
            "target": crop_spatial_5d(target, hr_start, self.patch_hr_shape),
            "mask_hr": crop_spatial_5d(mask_hr, hr_start, self.patch_hr_shape),
            "mask_lr": crop_spatial_5d(mask_lr, lr_start, lr_size),
        }
        if "run_id" in full:
            out["run_id"] = full["run_id"]
        if "t" in full:
            out["t"] = full["t"]
        return out

    def _patch_rng(self, volume_idx: int, patch_idx: int) -> np.random.Generator:
        seed = self.seed + volume_idx * 1009 + patch_idx
        return np.random.default_rng(seed)
