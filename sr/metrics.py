"""Reporting metrics (no gradients).

Purpose:
    Provide a single function that computes every metric we want to track
    per validation/inference sample, so the training loop and the eval/infer
    CLIs report the same numbers in the same units.
Effects:
    Returned dicts feed ``metrics.json`` and the per-epoch checkpoints,
    which means the user can compare runs side by side without rerunning.
Influences:
    All inputs must be ``(N, 1, D, H, W)`` single-channel tensors. The
    masked variants weight by ``mask`` in [0, 1]; the unmasked variants
    cover the whole volume so non-brain reconstruction quality is visible.
How to change safely:
    Adding a metric: extend ``compute_full_metrics`` and document the new
    key. Do not rename existing keys -- they are persisted on disk.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from sr.losses import (
    compute_dual_domain_masked_mse,
    focal_frequency_loss,
    kspace_mse_loss,
    l1_loss,
    masked_l1_loss,
    masked_mse_loss,
    merge_dual_domain_kwargs,
    merge_ffl_kwargs,
    mse_loss,
)
from sr.shape_utils import align_pred_target_mask


def volume_intensity_stats(volume: np.ndarray) -> dict[str, float]:
    """min / max / mean over all voxels of one 3D volume (numpy, any shape).

    Purpose:
        Quick intensity sanity check when inspecting infer outputs; catches
        collapsed predictions or scale mismatches vs target before plotting.
    Effects:
        Returns plain floats suitable for logging and JSON sidecars.
    """
    flat = np.asarray(volume, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("volume_intensity_stats: empty array")
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    """PSNR in dB from a mean-squared-error value.

    Returns 99.0 when MSE is effectively zero (identical volumes), matching
    the same convention used in the old code so plots stay comparable.
    """
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def _ssim_window(spatial: tuple[int, int, int]) -> int:
    """Pick an odd window <= 7 that fits in the smallest spatial dim."""
    smallest = min(spatial)
    win = min(7, smallest)
    if win % 2 == 0:
        win -= 1
    return max(1, win)


def masked_local_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mask-weighted local 3D SSIM.

    Computes the SSIM map with an average-pooling window (cheaper than a
    Gaussian and good enough for reporting), then averages the map weighted
    by the pooled mask so only in-brain regions count.
    """
    _b, c, d, h, w = pred.shape
    if c != 1:
        raise ValueError("masked_local_ssim_3d expects single-channel (N, 1, D, H, W)")
    win = _ssim_window((d, h, w))
    pad = win // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def pool(tensor: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(tensor, kernel_size=win, stride=1, padding=pad)

    mu_x = pool(pred)
    mu_y = pool(target)
    var_x = (pool(pred * pred) - mu_x * mu_x).clamp(min=0.0)
    var_y = (pool(target * target) - mu_y * mu_y).clamp(min=0.0)
    cov = pool(pred * target) - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2).clamp(min=eps)
    ssim_map = num / den

    weights = pool(mask)
    denom = weights.sum().clamp(min=eps)
    return (ssim_map * weights).sum() / denom


def unmasked_local_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """SSIM over the whole volume; equivalent to ``masked_local_ssim_3d`` with mask=1."""
    return masked_local_ssim_3d(
        pred, target, torch.ones_like(pred), data_range=data_range
    )


@torch.no_grad()
def compute_full_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    *,
    training_loss_name: str | None = None,
    training_loss_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute every reporting metric for one batch (no gradients).

    Returned keys (all floats):
        mse, masked_mse, l1, masked_l1,
        psnr, masked_psnr, ssim, masked_ssim
    Optional extra keys when ``training_loss_name`` matches a parameterised
    training objective (``dual_domain_masked_mse``, ``kspace_mse``, or
    ``focal_frequency``) so validation ``val_<loss_name>`` matches the optimiser.
    """
    pred, target, mask = align_pred_target_mask(pred, target, mask)
    values: dict[str, float] = {}
    values["mse"] = float(mse_loss(pred, target, mask).item())
    values["masked_mse"] = float(masked_mse_loss(pred, target, mask).item())
    values["l1"] = float(l1_loss(pred, target, mask).item())
    values["masked_l1"] = float(masked_l1_loss(pred, target, mask).item())
    values["psnr"] = psnr_from_mse(values["mse"], data_range=data_range)
    values["masked_psnr"] = psnr_from_mse(values["masked_mse"], data_range=data_range)
    values["ssim"] = float(
        unmasked_local_ssim_3d(pred, target, data_range=data_range).item()
    )
    values["masked_ssim"] = float(
        masked_local_ssim_3d(pred, target, mask, data_range=data_range).item()
    )
    if training_loss_name == "dual_domain_masked_mse":
        m = merge_dual_domain_kwargs(training_loss_kwargs)
        values["dual_domain_masked_mse"] = float(
            compute_dual_domain_masked_mse(
                pred,
                target,
                mask,
                alpha=m["alpha"],
                beta=m["beta"],
                kspace_high_freq_weight=m["kspace_high_freq_weight"],
            ).item()
        )
    elif training_loss_name == "kspace_mse":
        boost = float(
            (training_loss_kwargs or {}).get("kspace_high_freq_weight", 0.0)
        )
        values["kspace_mse"] = float(
            kspace_mse_loss(pred, target, mask, high_freq_boost=boost).item()
        )
    elif training_loss_name == "focal_frequency":
        m = merge_ffl_kwargs(training_loss_kwargs)
        values["focal_frequency"] = float(
            focal_frequency_loss(
                pred,
                target,
                mask,
                alpha=m["alpha"],
                log_matrix=m["log_matrix"],
                batch_matrix=m["batch_matrix"],
            ).item()
        )
    return values


def average_metric_dicts(per_batch: list[dict[str, float]]) -> dict[str, float]:
    """Per-key arithmetic mean across batches.

    Used by the validation loop to reduce per-batch metric dicts to one
    per-epoch dict without baking the reduction policy into the metrics
    themselves.
    """
    if not per_batch:
        return {}
    keys = per_batch[0].keys()
    n = float(len(per_batch))
    return {key: sum(d[key] for d in per_batch) / n for key in keys}
