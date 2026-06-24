"""Tests for SRCNN3DDeepXL shape contract and registry wiring."""

from __future__ import annotations

import torch

from sr.config import SRConfig, validate
from sr.models import MODEL_REGISTRY, SRCNN3DDeepXL, build_model, count_parameters


def test_forward_output_shape_matches_config() -> None:
    shape = (128, 128, 93)
    model = SRCNN3DDeepXL(shape)
    x = torch.randn(2, 1, *shape)
    out = model(x)
    assert out.shape == (2, 1, *shape)


def test_defaults_are_larger_than_srcnn3d_deep() -> None:
    shape = (128, 128, 93)
    xl = SRCNN3DDeepXL(shape)
    from sr.models import SRCNN3DDeep

    deep = SRCNN3DDeep(shape)
    assert count_parameters(xl) > count_parameters(deep)
    assert count_parameters(xl) == 785_201


def test_build_model_from_config() -> None:
    config = SRConfig(model_name="srcnn3d_deep_xl", model_kwargs={"n_feats": 72})
    validate(config)
    model = build_model(config)
    assert isinstance(model, SRCNN3DDeepXL)
    assert model.n_feats == 72


def test_registry_contains_key() -> None:
    assert "srcnn3d_deep_xl" in MODEL_REGISTRY
