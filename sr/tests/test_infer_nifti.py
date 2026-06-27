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


def test_infer_nifti_4d_full_run(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")

    from sr.checkpoint import EpochState, capture_rng_state, save_epoch
    from sr.config import SRConfig, to_json
    from sr.infer import infer_nifti
    from sr.models import build_model

    hr_shape = (16, 16, 12)
    n_time = 3
    config = SRConfig(output_patch_shape=hr_shape)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    to_json(config, run_dir / "config.json")

    model = build_model(config)
    state = EpochState(
        epoch_number=1,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        scheduler_state_dict=None,
        rng_state=capture_rng_state(),
        metrics_history=[{"epoch": 1, "train_loss": 0.0}],
        best_val_loss=0.0,
        best_epoch_number=1,
    )
    checkpoint = save_epoch(run_dir, state)

    rng = np.random.RandomState(0)
    data_4d = (rng.rand(*hr_shape, n_time).astype(np.float32) + 50.0) * 100.0
    input_path = tmp_path / "bold.nii.gz"
    affine = np.eye(4)
    zooms = (1.5, 1.5, 1.5, 2.0)
    img = nib.Nifti1Image(data_4d, affine)
    img.header.set_zooms(zooms)
    nib.save(img, str(input_path))

    output_path = tmp_path / "bold_sr.nii.gz"
    result = infer_nifti(checkpoint, input_path, output_path)

    out_img = nib.load(str(output_path))
    assert out_img.shape == (*hr_shape, n_time)
    assert result["n_volumes"] == n_time
    assert np.all(np.isfinite(out_img.get_fdata()))
    assert tuple(float(z) for z in out_img.header.get_zooms()) == zooms


def test_infer_nifti_4d_lr_native_run(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")

    from data.degradation_spatial import voxel_size_to_target_shape
    from sr.checkpoint import EpochState, capture_rng_state, save_epoch
    from sr.config import SRConfig, to_json
    from sr.infer import infer_nifti
    from sr.models import build_model

    hr_shape = (16, 16, 12)
    n_time = 4
    config = SRConfig(output_patch_shape=hr_shape)
    lr_shape = voxel_size_to_target_shape(
        hr_shape, config.source_voxel_mm, config.target_voxel_mm
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    to_json(config, run_dir / "config.json")

    model = build_model(config)
    state = EpochState(
        epoch_number=1,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        scheduler_state_dict=None,
        rng_state=capture_rng_state(),
        metrics_history=[{"epoch": 1, "train_loss": 0.0}],
        best_val_loss=0.0,
        best_epoch_number=1,
    )
    checkpoint = save_epoch(run_dir, state)

    rng = np.random.RandomState(1)
    data_4d = (rng.rand(*lr_shape, n_time).astype(np.float32) + 50.0) * 100.0
    input_path = tmp_path / "bold_lr.nii.gz"
    affine = np.eye(4)
    lr_zooms = (3.0, 3.0, 3.0, 2.0)
    img = nib.Nifti1Image(data_4d, affine)
    img.header.set_zooms(lr_zooms)
    nib.save(img, str(input_path))

    output_path = tmp_path / "bold_lr_sr.nii.gz"
    result = infer_nifti(checkpoint, input_path, output_path, norm_ref=100.0)

    out_img = nib.load(str(output_path))
    assert out_img.shape == (*hr_shape, n_time)
    assert result["n_volumes"] == n_time
    assert np.all(np.isfinite(out_img.get_fdata()))
    hr_zooms = (config.source_voxel_mm, config.source_voxel_mm, config.source_voxel_mm, 2.0)
    assert tuple(float(z) for z in out_img.header.get_zooms()) == hr_zooms
