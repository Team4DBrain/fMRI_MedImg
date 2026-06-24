"""Tests for standalone NIfTI inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sr.config import SRConfig
from sr.infer import (
    _prepare_lr_volume,
    default_sr_output_path,
    default_sr_preview_path,
    resolve_sr_output_path,
)


def test_default_sr_output_path_nii_gz() -> None:
    assert default_sr_output_path(Path("/data/sub-01_bold.nii.gz")) == Path(
        "/data/sub-01_bold_sr.nii.gz"
    )


def test_default_sr_output_path_nii() -> None:
    assert default_sr_output_path(Path("vol.nii")) == Path("vol_sr.nii.gz")


def test_resolve_sr_output_path_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "results"
    resolved = resolve_sr_output_path(
        Path("/data/sub-01_bold.nii.gz"),
        out_dir,
    )
    assert resolved == out_dir / "sub-01_bold_sr.nii.gz"


def test_resolve_sr_output_path_trailing_slash(tmp_path: Path) -> None:
    resolved = resolve_sr_output_path(
        Path("/data/vol.nii.gz"),
        Path(str(tmp_path / "out") + "/"),
    )
    assert resolved == tmp_path / "out" / "vol_sr.nii.gz"


def test_resolve_sr_output_path_explicit_file() -> None:
    assert resolve_sr_output_path(
        Path("/data/vol.nii.gz"),
        Path("/tmp/custom.nii.gz"),
    ) == Path("/tmp/custom.nii.gz")


def test_default_sr_preview_path() -> None:
    assert default_sr_preview_path(
        Path("/tmp/sub-01_bold_sr.nii.gz")
    ) == Path("/tmp/sub-01_bold_sr.png")

def test_prepare_lr_volume_skips_degradation_for_lr_shape() -> None:
    config = SRConfig()
    lr_shape = (64, 64, 46)
    vol = np.ones(lr_shape, dtype=np.float32)
    lr, mode, ground_truth = _prepare_lr_volume(vol, norm_ref=100.0, config=config)
    assert mode == "lr_native"
    assert ground_truth is None
    assert lr.shape == lr_shape
    assert np.isclose(lr.mean(), 0.01)


def test_prepare_lr_volume_degrades_hr_shape() -> None:
    config = SRConfig()
    hr_shape = tuple(config.output_patch_shape)
    vol = np.random.RandomState(0).randn(*hr_shape).astype(np.float32) + 100.0
    lr, mode, ground_truth = _prepare_lr_volume(vol, norm_ref=100.0, config=config)
    assert mode == "hr_degraded"
    assert ground_truth is not None
    assert ground_truth.shape == hr_shape
    assert lr.shape == (64, 64, 46)


def test_prepare_lr_volume_rejects_unknown_shape() -> None:
    config = SRConfig()
    vol = np.ones((32, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="neither the expected HR"):
        _prepare_lr_volume(vol, norm_ref=1.0, config=config)


def test_make_sr_output_preview_with_ground_truth(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from sr.infer import make_sr_output_preview

    lr = np.random.RandomState(0).rand(8, 8, 6).astype(np.float32)
    pred = np.random.RandomState(1).rand(16, 16, 12).astype(np.float32)
    gt = np.random.RandomState(2).rand(16, 16, 12).astype(np.float32)
    out = tmp_path / "preview.png"
    make_sr_output_preview(
        input_lr=lr,
        prediction_vol=pred,
        ground_truth_vol=gt,
        output_path=out,
    )
    assert out.is_file()
