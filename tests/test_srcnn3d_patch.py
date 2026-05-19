"""Tests for SRCNN3DPatch, patch dataset, and shape alignment."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from src.sr.config import SRConfig, validate
from src.sr.data import build_loaders, resolve_sample_split
from src.sr.forward import model_forward
from src.sr.metrics import compute_full_metrics
from src.sr.models import SRCNN3D_PATCH_RECEPTIVE_SHRINK, SRCNN3DPatch
from src.sr.patch_data import PatchTrainingDataset, hr_to_lr_crop
from src.sr.shape_utils import align_pred_target_mask, center_crop_spatial


class _FakeVolumeDataset(Dataset):
    """Minimal full-volume sample for patch cropping tests."""

    def __init__(self) -> None:
        self.hr_shape = (32, 32, 24)
        self.lr_shape = (16, 16, 12)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict:
        hr = torch.full((1, *self.hr_shape), float(idx + 1))
        lr = torch.full((1, *self.lr_shape), float(idx + 10))
        mask_hr = torch.ones((1, *self.hr_shape))
        mask_lr = torch.ones((1, *self.lr_shape))
        return {
            "input": lr,
            "target": hr,
            "mask_hr": mask_hr,
            "mask_lr": mask_lr,
            "run_id": f"run_{idx}",
            "t": 0,
        }


def test_srcnn3d_patch_forward_shape():
    patch_hr = (48, 48, 48)
    lr = (24, 24, 24)
    model = SRCNN3DPatch(output_patch_shape=(128, 128, 93))
    x = torch.randn(2, 1, *lr)
    pred = model(x, upsample_to=patch_hr)
    expected = tuple(p - SRCNN3D_PATCH_RECEPTIVE_SHRINK for p in patch_hr)
    assert tuple(pred.shape[-3:]) == expected


def test_center_crop_to_pred():
    target = torch.randn(1, 1, 48, 48, 48)
    mask = torch.ones(1, 1, 48, 48, 48)
    pred = torch.randn(1, 1, 36, 36, 36)
    t2, m2 = align_pred_target_mask(pred, target, mask)[1:]
    assert tuple(t2.shape[-3:]) == (36, 36, 36)
    assert tuple(m2.shape[-3:]) == (36, 36, 36)
    center = center_crop_spatial(target, (36, 36, 36))
    assert torch.allclose(t2, center)


def test_hr_to_lr_crop_alignment():
    hr_shape = (32, 32, 24)
    lr_shape = (16, 16, 12)
    hr_start = (4, 6, 2)
    hr_size = (16, 16, 12)
    lr_start, lr_size = hr_to_lr_crop(hr_start, hr_size, hr_shape, lr_shape)
    assert lr_size == (8, 8, 6)
    assert lr_start == (2, 3, 1)


def test_patch_training_dataset_length_and_crop():
    base = _FakeVolumeDataset()
    patch_shape = (16, 16, 13)
    ppv = 32
    ds = PatchTrainingDataset(
        base, patch_hr_shape=patch_shape, patches_per_volume=ppv, seed=42
    )
    assert len(ds) == len(base) * ppv
    sample = ds[0]
    assert tuple(sample["target"].shape[-3:]) == patch_shape
    assert tuple(sample["input"].shape[-3:]) == (8, 8, 6)


def test_patch_training_reproducible_crops():
    base = _FakeVolumeDataset()
    kwargs = dict(patch_hr_shape=(16, 16, 13), patches_per_volume=4, seed=7)
    a = PatchTrainingDataset(base, **kwargs)[3]
    b = PatchTrainingDataset(base, **kwargs)[3]
    assert torch.equal(a["target"], b["target"])


def test_compute_full_metrics_with_smaller_pred():
    target = torch.randn(1, 1, 20, 20, 20)
    mask = torch.ones(1, 1, 20, 20, 20)
    pred = torch.randn(1, 1, 8, 8, 8)
    metrics = compute_full_metrics(pred, target, mask)
    assert "masked_mse" in metrics
    assert metrics["masked_mse"] >= 0.0


def test_model_forward_patch_uses_target_shape():
    model = SRCNN3DPatch(output_patch_shape=(128, 128, 93))
    inputs = torch.randn(1, 1, 8, 8, 8)
    target = torch.randn(1, 1, 16, 16, 16)
    pred = model_forward(model, inputs, target, "srcnn3d_patch")
    assert tuple(pred.shape[-3:]) == (4, 4, 4)


def test_build_loaders_patch_train_full_val(synthetic_pipeline):
    """Train loader expands patches; val loader keeps one sample per volume."""
    from src.data.datasets import SpatialSRDataset
    from src.data.degradation_spatial import make_spatial_degradation

    manifest_path, _ = synthetic_pipeline
    patch_shape = (16, 16, 13)
    config = SRConfig(
        manifest_path=manifest_path,
        model_name="srcnn3d_patch",
        patch_hr_shape=patch_shape,
        patches_per_volume=32,
        train_split=0.8,
        batch_size=2,
        seed=42,
    )
    validate(config)
    train_loader, val_loader, split_info = build_loaders(config)
    degrade = make_spatial_degradation(1.5, 3.0)
    n_full = len(SpatialSRDataset(manifest_path=manifest_path, degrade_fn=degrade))
    train_indices, val_indices, _ = resolve_sample_split(n_full, config)
    assert len(train_loader.dataset) == len(train_indices) * 32
    assert val_loader is not None
    assert len(val_loader.dataset) == len(val_indices)
    assert split_info["patch_training"] is True
    batch = next(iter(train_loader))
    assert tuple(batch["target"].shape[-3:]) == patch_shape


def test_validate_rejects_patch_too_large():
    config = SRConfig(
        model_name="srcnn3d_patch",
        patch_hr_shape=(200, 200, 200),
        output_patch_shape=(128, 128, 93),
        manifest_path=__import__("pathlib").Path("/nonexistent"),
    )
    with pytest.raises(ValueError, match="exceeds"):
        validate(config)
