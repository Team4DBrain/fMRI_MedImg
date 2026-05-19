"""Spatial alignment between valid-conv predictions and HR targets.

Purpose:
    ``SRCNN3DPatch`` uses valid convolutions (9-1-5), so predictions are
    smaller than the trilinearly upsampled input. Training and metrics must
    crop ``target`` and ``mask_hr`` to ``pred``'s shape before loss/metrics.
Effects:
    Keeps masked MSE/PSNR/SSIM defined on the same voxel set the network
    actually predicts; avoids silent broadcasting errors in PyTorch losses.
Influences:
    ``SRCNN3D_PATCH_RECEPTIVE_SHRINK`` in ``models.py`` must match the
    stacked valid receptive field of conv1+conv2+conv3.
How to change safely:
    If kernel sizes or padding change, update ``RECEPTIVE_SHRINK`` in both
    modules and the patch minimum-size check in ``config.validate``.
"""

from __future__ import annotations

import torch

from src.sr.models import SRCNN3D_PATCH_RECEPTIVE_SHRINK

RECEPTIVE_SHRINK = SRCNN3D_PATCH_RECEPTIVE_SHRINK


def center_crop_spatial(
    tensor: torch.Tensor,
    target_spatial: tuple[int, int, int],
) -> torch.Tensor:
    """Center-crop the trailing (D, H, W) dims of a (N, C, D, H, W) tensor."""
    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor (N,C,D,H,W), got shape {tuple(tensor.shape)}")
    _, _, d, h, w = tensor.shape
    td, th, tw = target_spatial
    if td > d or th > h or tw > w:
        raise ValueError(
            f"Cannot crop {tensor.shape[-3:]} to target {target_spatial}: target is larger."
        )
    sd = (d - td) // 2
    sh = (h - th) // 2
    sw = (w - tw) // 2
    return tensor[:, :, sd:sd + td, sh: sh + th, sw: sw + tw]


def center_crop_to_pred(
    target: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop ``target`` and ``mask`` to match ``pred``'s spatial size (centered)."""
    spatial = tuple(pred.shape[-3:])
    if tuple(target.shape[-3:]) == spatial:
        return target, mask
    return (
        center_crop_spatial(target, spatial),
        center_crop_spatial(mask, spatial),
    )


def align_pred_target_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pred, target, mask) with identical spatial shapes."""
    if tuple(pred.shape[-3:]) == tuple(target.shape[-3:]):
        return pred, target, mask
    cropped_target, cropped_mask = center_crop_to_pred(target, mask, pred)
    return pred, cropped_target, cropped_mask
