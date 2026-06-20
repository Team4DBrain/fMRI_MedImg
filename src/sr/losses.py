"""Training loss functions and the loss registry.

Purpose:
    Centralise every loss the training loop can optimize, behind a single
    uniform signature ``fn(pred, target, mask) -> scalar Tensor``. The
    registry lets the CLI swap objectives by name without touching the loop.
Effects:
    The chosen loss drives gradient updates. Mask-aware variants restrict
    optimization to in-brain voxels via the dataset's ``mask_hr`` tensor.
    Dual-domain losses add a k-space term so optimisation is not driven only
    by image-domain MSE (which often favours blurry means).
Influences:
    Every loss expects all three tensors with broadcastable shapes
    ``(N, 1, D, H, W)``. ``mask`` is in [0, 1]; unmasked variants ignore it
    so callers can pass a dummy. Parameterised losses read ``loss_kwargs``
    from ``SRConfig`` via ``resolve_loss`` so runs stay reproducible from
    ``config.json``.
How to change safely:
    Register new losses in ``LOSS_REGISTRY``. Keep the signature uniform so
    ``resolve_loss`` and the training/eval code do not need special cases.
    For losses with hyperparameters, add defaults in ``resolve_loss`` and
    validate ranges in ``SRConfig.validate``.
"""

from __future__ import annotations

from typing import Any, Callable

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
    #mask_reduced = torch.clamp(mask, min=0.25)
    #denom = torch.clamp(mask_reduced.sum(), min=eps)
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


def _fftn_batch_volumes(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 3D FFT over spatial axes for real single-channel batches.

    Purpose:
        Map each (D,H,W) volume to complex k-space with ``norm='ortho'`` so
        spatial and spectral energy stay on comparable scales (Parseval).
    Effects:
        Returns shape ``(N, D, H, W)`` complex; callers compare spectra in MSE.
    """
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError(f"Expected pred/target (N, 1, D, H, W), got {tuple(x.shape)}")
    return torch.fft.fftn(x.squeeze(1), dim=(-3, -2, -1), norm="ortho")


def _kspace_radial_weight(
    spatial: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    high_freq_boost: float,
) -> torch.Tensor:
    """Non-negative weights over frequency grid; ``high_freq_boost`` up-weights
    large |k| (periphery of the FFT grid) relative to DC when > 0.

    Purpose:
        Let the k-space term penalise missing high frequencies more than the
        easy low-frequency bulk that image MSE already captures.
    Effects:
        Shape ``(1, D, H, W)`` for broadcasting against ``(N, D, H, W)`` complex
        magnitudes. When ``high_freq_boost == 0``, returns ones (uniform MSE).
    """
    if high_freq_boost < 0:
        raise ValueError("high_freq_boost must be >= 0")
    d, h, w = spatial
    fd = torch.fft.fftfreq(d, d=1.0, device=device, dtype=dtype)
    fh = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)
    fw = torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype)
    gd, gh, gw = torch.meshgrid(fd, fh, fw, indexing="ij")
    r2 = gd * gd + gh * gh + gw * gw
    r2_max = r2.max().clamp(min=1e-12)
    wgt = 1.0 + high_freq_boost * (r2 / r2_max)
    return wgt.unsqueeze(0)


def kspace_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    _mask: torch.Tensor,
    *,
    high_freq_boost: float = 0.0,
) -> torch.Tensor:
    """Mean squared magnitude of the complex spectrum difference (3D, ortho FFT).

    Purpose:
        Penalise frequency content mismatch; complements image MSE which is
        blind to phase/detail trade-offs that still look blurry.
    Effects:
        Uses full volume (mask ignored) so skull/air ringing still influences
        gradients unless combined with an image-domain masked term.
    """
    fp = _fftn_batch_volumes(pred)
    ft = _fftn_batch_volumes(target)
    diff = fp - ft
    mag_sq = diff.abs() ** 2
    if high_freq_boost <= 0.0:
        return mag_sq.mean()
    w = _kspace_radial_weight(
        (pred.shape[2], pred.shape[3], pred.shape[4]),
        device=pred.device,
        dtype=pred.dtype,
        high_freq_boost=high_freq_boost,
    )
    wsum = w.sum().clamp(min=1e-12)
    spatial_sum = (mag_sq * w).sum(dim=(-3, -2, -1))
    return (spatial_sum / wsum).mean()


def compute_dual_domain_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    kspace_high_freq_weight: float = 0.0,
) -> torch.Tensor:
    """``alpha`` * masked image MSE + ``beta`` * (weighted) k-space MSE.

    Purpose:
        Stabilise training with masked image-domain fidelity while pushing the
        spectrum toward the target so high-frequency content is not washed out
        as easily as with image MSE alone.
    Effects:
        ``kspace_high_freq_weight`` forwards into ``kspace_mse_loss`` as
        ``high_freq_boost``; larger values emphasise outer k-space more.
    """
    img = masked_mse_loss(pred, target, mask)
    spec = kspace_mse_loss(
        pred, target, mask, high_freq_boost=kspace_high_freq_weight
    )
    return alpha * img + beta * spec


def focal_frequency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    _mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    log_matrix: bool = False,
    batch_matrix: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Focal Frequency Loss (FFL) on 3D orthonormal FFT spectra.

    Purpose:
        Down-weight easy frequency bins (often low-freq / already matched) and
        up-weight bins where pred and target disagree — mitigates spectral bias
        toward blurry means in image-only MSE training.
    Effects:
        Per-frequency weight ``w`` is derived from ``|F(pred)-F(target)|^alpha``,
        normalised to [0, 1], then detached so gradients flow only through the
        squared spectral error (ICCV 2021 formulation, extended to 3D volumes).
    Influences:
        ``alpha`` controls focal strength (larger → harder focus on bad bins).
        ``log_matrix`` / ``batch_matrix`` mirror the reference implementation.
        Mask is ignored (full-volume spectrum, like ``kspace_mse_loss``).
    """
    if alpha < 0:
        raise ValueError("focal_frequency_loss: alpha must be >= 0")

    fp = _fftn_batch_volumes(pred)
    ft = _fftn_batch_volumes(target)
    diff = fp - ft
    freq_distance = diff.abs().pow(2)

    matrix_tmp = diff.abs().clamp(min=eps).pow(alpha)
    if log_matrix:
        matrix_tmp = torch.log(matrix_tmp + 1.0)

    if batch_matrix:
        denom = matrix_tmp.max().clamp(min=eps)
    else:
        denom = matrix_tmp.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp(min=eps)
    weight = (matrix_tmp / denom).clamp(0.0, 1.0)
    weight = weight.nan_to_num(0.0).detach()

    return (weight * freq_distance).mean()


def merge_ffl_kwargs(loss_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Defaults for ``focal_frequency``; shared by training and validation metrics."""
    defaults: dict[str, Any] = {
        "alpha": 1.0,
        "log_matrix": False,
        "batch_matrix": False,
    }
    if not loss_kwargs:
        return defaults
    out = dict(defaults)
    for key in defaults:
        if key in loss_kwargs:
            out[key] = loss_kwargs[key]
    out["alpha"] = float(out["alpha"])
    out["log_matrix"] = bool(out["log_matrix"])
    out["batch_matrix"] = bool(out["batch_matrix"])
    return out


def merge_dual_domain_kwargs(
    loss_kwargs: dict[str, Any] | None,
) -> dict[str, float]:
    """Defaults for ``dual_domain_masked_mse``; same merge used in metrics."""
    defaults: dict[str, float] = {
        "alpha": 0.5,
        "beta": 0.5,
        "kspace_high_freq_weight": 0.0,
    }
    if not loss_kwargs:
        return defaults
    out = dict(defaults)
    for key in defaults:
        if key in loss_kwargs:
            out[key] = float(loss_kwargs[key])
    return out


def _dual_domain_masked_mse_factory(kw: dict[str, Any]) -> LossFn:
    merged = merge_dual_domain_kwargs(kw)
    alpha = merged["alpha"]
    beta = merged["beta"]
    kspace_high_freq_weight = merged["kspace_high_freq_weight"]

    def fn(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return compute_dual_domain_masked_mse(
            p,
            t,
            m,
            alpha=alpha,
            beta=beta,
            kspace_high_freq_weight=kspace_high_freq_weight,
        )

    return fn


def _focal_frequency_factory(kw: dict[str, Any]) -> LossFn:
    merged = merge_ffl_kwargs(kw)

    def fn(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return focal_frequency_loss(
            p,
            t,
            m,
            alpha=merged["alpha"],
            log_matrix=merged["log_matrix"],
            batch_matrix=merged["batch_matrix"],
        )

    return fn


# Simple registry entries (no extra kwargs). Parameterised losses are built in
# ``resolve_loss`` so ``LOSS_REGISTRY`` stays picklable / JSON-serialisable by name.
LOSS_REGISTRY: dict[str, LossFn] = {
    "mse": mse_loss,
    "masked_mse": masked_mse_loss,
    "l1": l1_loss,
    "masked_l1": masked_l1_loss,
}


def loss_names_for_validation() -> frozenset[str]:
    """All loss names accepted by ``resolve_loss`` (registry + parameterised)."""
    return frozenset(LOSS_REGISTRY.keys()) | frozenset(
        {"dual_domain_masked_mse", "kspace_mse", "focal_frequency"}
    )


def resolve_loss(
    name: str, loss_kwargs: dict[str, Any] | None = None
) -> LossFn:
    """Return a loss callable; ``loss_kwargs`` tune parameterised objectives."""
    key = name.strip().lower()
    kw = dict(loss_kwargs or {})
    if key == "dual_domain_masked_mse":
        return _dual_domain_masked_mse_factory(kw)
    if key == "kspace_mse":
        boost = float(kw.get("kspace_high_freq_weight", 0.0))

        def _kfn(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            return kspace_mse_loss(p, t, m, high_freq_boost=boost)

        return _kfn
    if key == "focal_frequency":
        return _focal_frequency_factory(kw)
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {sorted(loss_names_for_validation())}"
        )
    return LOSS_REGISTRY[key]

