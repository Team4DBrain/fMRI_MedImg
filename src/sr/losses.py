"""Training loss functions and the loss registry.

Purpose:
    Centralise every loss the training loop can optimize, behind a single
    uniform signature ``fn(pred, target, mask) -> scalar Tensor``. The
    registry lets the CLI swap objectives by name without touching the loop.
Effects:
    The chosen loss drives gradient updates. Mask-aware variants restrict
    optimization to in-brain voxels via the dataset's ``mask_hr`` tensor.
Influences:
    Every loss expects all three tensors with broadcastable shapes
    ``(N, 1, D, H, W)``. ``mask`` is in [0, 1]; unmasked variants ignore it
    so callers can pass a dummy.
How to change safely:
    Register new losses in ``LOSS_REGISTRY``. Keep the signature uniform so
    ``resolve_loss`` and the training/eval code do not need special cases.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.nn import functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def mse_loss(
    pred: torch.Tensor, target: torch.Tensor, _mask: torch.Tensor
) -> torch.Tensor:
    """Plain MSE over the full predicted volume; ignores ``_mask``."""
    return F.mse_loss(pred, target)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE averaged over mask-weighted voxels.

    ``eps`` guards against division by zero on empty masks but should never
    fire in practice (validate_paths and the dataset reject empty masks).
    """
    sq_err = (pred - target) ** 2
    denom = torch.clamp(mask.sum(), min=eps)
    return (sq_err * mask).sum() / denom


def l1_loss(
    pred: torch.Tensor, target: torch.Tensor, _mask: torch.Tensor
) -> torch.Tensor:
    """Plain L1/MAE; ignores ``_mask``."""
    return F.l1_loss(pred, target)


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mask-weighted L1, mirrors ``masked_mse_loss`` for outlier robustness."""
    abs_err = torch.abs(pred - target)
    denom = torch.clamp(mask.sum(), min=eps)
    return (abs_err * mask).sum() / denom


LOSS_REGISTRY: dict[str, LossFn] = {
    "mse": mse_loss,
    "masked_mse": masked_mse_loss,
    "l1": l1_loss,
    "masked_l1": masked_l1_loss,
}


def resolve_loss(name: str) -> LossFn:
    """Look ``name`` up in ``LOSS_REGISTRY`` with a helpful error message."""
    key = name.strip().lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[key]
