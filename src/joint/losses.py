"""Loss + metrics for the joint denoise + spatial-SR model.

Everything operates in HR space and is brain-masked with ``mask_hr`` (the HR
mask from the batch — NOT ``mask_lr``). All reductions cast to fp32 internally
so they are safe under autocast: a ``.sum()`` over ~1.5M HR voxels in fp16 would
lose precision and can under/overflow.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_charbonnier(
    pred: torch.Tensor,      # (B,1,128,128,93)
    target: torch.Tensor,    # (B,1,128,128,93)
    mask_hr: torch.Tensor,   # (B,1,128,128,93) — float weights, ~0/1
    eps: float = 1e-3,
) -> torch.Tensor:
    """Brain-masked Charbonnier loss: mean of sqrt((pred-target)^2 + eps^2)
    over in-brain voxels. Denominator clamped so an all-zero mask can't divide
    by zero (it then returns 0 with zero gradient — a no-op, never a NaN)."""
    diff = pred.float() - target.float()
    charb = torch.sqrt(diff * diff + eps * eps)
    m = mask_hr.float()
    num = (charb * m).sum()
    den = m.sum().clamp(min=1.0)
    return num / den


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sample: bool = False,
) -> torch.Tensor:
    """Mask-weighted MSE. If ``per_sample`` reduce over everything but the batch
    dim and return a (B,) tensor; otherwise a scalar."""
    se = (pred.float() - target.float()) ** 2
    m = mask.float()
    if per_sample:
        dims = tuple(range(1, se.ndim))
        return (se * m).sum(dims) / m.sum(dims).clamp(min=1.0)
    return (se * m).sum() / m.sum().clamp(min=1.0)


def masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    peak: float = 1.0,
    eps: float = 1e-12,
    per_sample: bool = False,
) -> torch.Tensor:
    """Brain-masked PSNR (dB). ``peak=1.0`` is the per-run normalisation anchor
    (in-brain ~1.0); a few saturation outliers exceed it, but a fixed peak keeps
    PSNR comparable across runs."""
    mse = masked_mse(pred, target, mask, per_sample=per_sample)
    return 10.0 * torch.log10((peak * peak) / (mse + eps))


def _gaussian_window_3d(ws: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return (w / w.sum()).view(1, 1, ws, ws, ws)


def masked_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_hr: torch.Tensor,
    ws: int = 7,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Windowed 3D SSIM (Gaussian window), averaged over the brain mask.

    A true 3D window (rather than per-slice 2D) is used because the through-plane
    axis (46->93) is the one most affected by the anisotropic upsampler.
    """
    pred = pred.float()
    target = target.float()
    m = mask_hr.float()
    w = _gaussian_window_3d(ws, sigma, pred.device, pred.dtype)
    pad = ws // 2
    mu1 = F.conv3d(pred, w, padding=pad)
    mu2 = F.conv3d(target, w, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv3d(pred * pred, w, padding=pad) - mu1_sq
    s2 = F.conv3d(target * target, w, padding=pad) - mu2_sq
    s12 = F.conv3d(pred * target, w, padding=pad) - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return (ssim_map * m).sum() / m.sum().clamp(min=1.0)
