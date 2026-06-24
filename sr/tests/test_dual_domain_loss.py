"""Tests for dual-domain (image + k-space) SR losses."""

from __future__ import annotations

import torch

from sr.losses import (
    compute_dual_domain_masked_mse,
    focal_frequency_loss,
    kspace_mse_loss,
    merge_dual_domain_kwargs,
    merge_ffl_kwargs,
    resolve_loss,
)


def test_merge_dual_domain_kwargs_defaults() -> None:
    assert merge_dual_domain_kwargs(None) == merge_dual_domain_kwargs({})
    m = merge_dual_domain_kwargs({"alpha": 0.7})
    assert m["alpha"] == 0.7
    assert m["beta"] == 0.5


def test_identical_pred_target_small_loss() -> None:
    torch.manual_seed(0)
    n, d, h, w = 2, 16, 16, 12
    x = torch.randn(n, 1, d, h, w)
    mask = torch.ones_like(x)
    loss = compute_dual_domain_masked_mse(x, x, mask)
    assert loss.item() == 0.0


def test_kspace_mse_backward() -> None:
    n, d, h, w = 1, 8, 8, 8
    pred = torch.randn(n, 1, d, h, w, requires_grad=True)
    target = torch.randn(n, 1, d, h, w)
    loss = kspace_mse_loss(pred, target, torch.ones_like(pred), high_freq_boost=0.5)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_merge_ffl_kwargs_defaults() -> None:
    m = merge_ffl_kwargs({"alpha": 2.0})
    assert m["alpha"] == 2.0
    assert m["log_matrix"] is False


def test_focal_frequency_identical_zero() -> None:
    x = torch.randn(1, 1, 8, 8, 8)
    mask = torch.ones_like(x)
    loss = focal_frequency_loss(x, x, mask)
    assert loss.item() == 0.0


def test_focal_frequency_backward() -> None:
    pred = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    target = torch.randn(1, 1, 8, 8, 8)
    loss = focal_frequency_loss(pred, target, torch.ones_like(pred), alpha=1.0)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_resolve_dual_domain_runs() -> None:
    fn = resolve_loss(
        "dual_domain_masked_mse",
        {"alpha": 0.6, "beta": 0.4, "kspace_high_freq_weight": 0.1},
    )
    p = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    t = torch.randn(1, 1, 8, 8, 8)
    m = torch.ones(1, 1, 8, 8, 8)
    out = fn(p, t, m)
    out.backward()
    assert torch.isfinite(out)


def test_resolve_focal_frequency_runs() -> None:
    fn = resolve_loss("focal_frequency", {"alpha": 1.0, "log_matrix": True})
    p = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    t = torch.randn(1, 1, 8, 8, 8)
    out = fn(p, t, torch.ones(1, 1, 8, 8, 8))
    out.backward()
    assert torch.isfinite(out)
