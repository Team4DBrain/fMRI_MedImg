"""Unit tests for spatial degradation. The most important one is the scale
sanity test for kspace_downsample_3d — this is the regression test for the
"LR is 64x too bright" bug.

Run:
    python -m pytest tests/test_degradation_spatial.py -v
or simply:
    python tests/test_degradation_spatial.py
"""

from __future__ import annotations

import numpy as np
import pytest

from data.degradation_spatial import (
    downsample_mask_to_lr,
    kspace_downsample_3d,
    make_spatial_degradation,
    voxel_size_to_target_shape,
)


# ---------------------------------------------------------------------------
# Scale sanity — the regression test for the inverted-scale bug.
# A constant-magnitude blob in must give a similar-magnitude blob out.
# ---------------------------------------------------------------------------

def test_kspace_downsample_preserves_intensity_scale_constant_volume():
    """Constant volume in -> constant volume out at the same magnitude."""
    hr = np.full((128, 128, 92), 500.0, dtype=np.float32)
    lr = kspace_downsample_3d(hr, (64, 64, 46), apodize=False)
    # The interior of the LR should match the HR constant. Boundaries can have
    # ringing from the discontinuity at the volume edge; check the center.
    center = lr[16:48, 16:48, 12:34]
    np.testing.assert_allclose(center.mean(), 500.0, rtol=1e-3)


def test_kspace_downsample_preserves_intensity_scale_blob():
    """Blob in -> blob out at the same scale (within reasonable tolerance)."""
    hr = np.zeros((128, 128, 92), dtype=np.float32)
    xx, yy, zz = np.ogrid[:128, :128, :92]
    blob = (xx - 64) ** 2 + (yy - 64) ** 2 + (zz - 46) ** 2 < 30 ** 2
    hr[blob] = 1000.0

    lr = kspace_downsample_3d(hr, (64, 64, 46))

    # Mean inside a same-relative-region in LR should be close to HR mean
    # in that region. Hamming apodization will smooth the blob slightly and
    # reduce the peak; we accept up to ~30% reduction in mean-of-bright-region.
    hr_in_blob = hr[blob].mean()
    # Take roughly the central core of LR which should be filled with blob.
    lr_core = lr[24:40, 24:40, 18:28]
    assert lr_core.mean() > 0.5 * hr_in_blob, (
        f"LR core mean {lr_core.mean():.1f} far below HR in-blob mean "
        f"{hr_in_blob:.1f} — scale is wrong"
    )
    assert lr_core.mean() < 1.5 * hr_in_blob, (
        f"LR core mean {lr_core.mean():.1f} far above HR in-blob mean "
        f"{hr_in_blob:.1f} — scale is wrong (was 64x in the broken version)"
    )


def test_kspace_downsample_overall_mean_matches_hr():
    """Global mean should be approximately preserved (Parseval-ish at DC)."""
    rng = np.random.default_rng(0)
    hr = rng.uniform(100, 1000, size=(128, 128, 92)).astype(np.float32)
    lr = kspace_downsample_3d(hr, (64, 64, 46), apodize=False)
    np.testing.assert_allclose(lr.mean(), hr.mean(), rtol=5e-2)


def test_kspace_downsample_output_shape():
    hr = np.zeros((128, 128, 92), dtype=np.float32)
    lr = kspace_downsample_3d(hr, (64, 64, 46))
    assert lr.shape == (64, 64, 46)
    assert lr.dtype == np.float32


def test_kspace_downsample_rejects_invalid_target_shape():
    hr = np.zeros((128, 128, 92), dtype=np.float32)
    # target larger than source on any axis
    with pytest.raises(ValueError):
        kspace_downsample_3d(hr, (130, 64, 46))
    # target zero
    with pytest.raises(ValueError):
        kspace_downsample_3d(hr, (0, 64, 46))


def test_kspace_downsample_rejects_non_3d():
    with pytest.raises(ValueError):
        kspace_downsample_3d(np.zeros((4, 4)), (2, 2, 2))


# ---------------------------------------------------------------------------
# voxel_size_to_target_shape
# ---------------------------------------------------------------------------

def test_voxel_size_to_target_shape_halving():
    assert voxel_size_to_target_shape((128, 128, 92), 1.5, 3.0) == (64, 64, 46)


def test_voxel_size_to_target_shape_target_must_be_larger():
    with pytest.raises(ValueError):
        voxel_size_to_target_shape((128, 128, 92), 3.0, 1.5)


# ---------------------------------------------------------------------------
# downsample_mask_to_lr
# ---------------------------------------------------------------------------

def test_downsample_mask_preserves_brain_fraction():
    shape = (128, 128, 93)
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    hr_mask = ((xx - 64) / 45) ** 2 + ((yy - 64) / 55) ** 2 + ((zz - 46) / 35) ** 2 < 1
    lr_mask = downsample_mask_to_lr(hr_mask, (64, 64, 46))
    # Same fraction within a few percent
    assert abs(lr_mask.mean() - hr_mask.mean()) < 0.05


def test_downsample_mask_correct_shape():
    hr_mask = np.zeros((128, 128, 93), dtype=bool)
    hr_mask[40:80, 40:80, 20:60] = True
    lr_mask = downsample_mask_to_lr(hr_mask, (64, 64, 46))
    assert lr_mask.shape == (64, 64, 46)
    assert lr_mask.dtype == bool


# ---------------------------------------------------------------------------
# make_spatial_degradation factory
# ---------------------------------------------------------------------------

def test_make_spatial_degradation_returns_correct_shape():
    degrade = make_spatial_degradation(source_voxel_mm=1.5, target_voxel_mm=3.0)
    hr = np.zeros((128, 128, 92), dtype=np.float32)
    lr = degrade(hr)
    assert lr.shape == (64, 64, 46)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
