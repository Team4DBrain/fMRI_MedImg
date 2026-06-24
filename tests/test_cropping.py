"""Unit tests for cropping.py — z-bbox-centered crop logic."""

from __future__ import annotations

import numpy as np
import pytest

from data.cropping import compute_z_start, crop_z, update_affine_for_z_crop


def _make_mask(z_native, z_lo, z_hi, shape_xy=(8, 8)):
    """Mask with brain occupying z slices [z_lo, z_hi)."""
    m = np.zeros(shape_xy + (z_native,), dtype=bool)
    m[..., z_lo:z_hi] = True
    return m


def test_z_start_centers_bbox():
    """Brain at z=20..40 (extent 20), native=60, target=30 -> z_start=15
    (window 15..45, bbox 20..40 fully inside, centered)."""
    m = _make_mask(z_native=60, z_lo=20, z_hi=40)
    z_start = compute_z_start(m, target_z=30)
    assert z_start == 15


def test_z_start_clamps_to_zero_when_brain_at_top():
    """Brain hugging slice 0; target window must clamp to z_start=0."""
    m = _make_mask(z_native=60, z_lo=0, z_hi=20)
    z_start = compute_z_start(m, target_z=30)
    assert z_start == 0


def test_z_start_clamps_to_max_when_brain_at_bottom():
    """Brain hugging the high z; clamp to z_native - target_z."""
    m = _make_mask(z_native=60, z_lo=40, z_hi=60)
    z_start = compute_z_start(m, target_z=30)
    assert z_start == 30


def test_z_start_zero_when_target_equals_native():
    m = _make_mask(z_native=24, z_lo=5, z_hi=20)
    assert compute_z_start(m, target_z=24) == 0


def test_z_start_rejects_target_larger_than_native():
    m = _make_mask(z_native=24, z_lo=5, z_hi=20)
    with pytest.raises(ValueError, match="exceeds native"):
        compute_z_start(m, target_z=30)


def test_z_start_rejects_bbox_too_tall():
    """Brain z-extent > target -> can't fit, must raise."""
    m = _make_mask(z_native=60, z_lo=0, z_hi=50)
    with pytest.raises(ValueError, match="exceeds target_z"):
        compute_z_start(m, target_z=30)


def test_z_start_rejects_empty_mask():
    m = np.zeros((8, 8, 30), dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        compute_z_start(m, target_z=20)


def test_crop_z_basic():
    arr = np.arange(60).reshape(1, 1, 60)
    cropped = crop_z(arr, z_start=10, target_z=20)
    assert cropped.shape == (1, 1, 20)
    np.testing.assert_array_equal(cropped[0, 0, :], np.arange(10, 30))


def test_crop_z_4d():
    arr = np.zeros((4, 4, 30, 5), dtype=np.float32)
    arr[..., 10:20, :] = 1.0
    cropped = crop_z(arr, z_start=10, target_z=10)
    assert cropped.shape == (4, 4, 10, 5)
    assert (cropped == 1.0).all()


def test_crop_z_rejects_invalid_window():
    arr = np.zeros((4, 4, 20))
    with pytest.raises(ValueError, match="outside z-range"):
        crop_z(arr, z_start=15, target_z=10)
    with pytest.raises(ValueError, match="outside z-range"):
        crop_z(arr, z_start=-1, target_z=5)


def test_crop_z_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="3D or 4D"):
        crop_z(np.zeros((4, 4)), z_start=0, target_z=2)


def test_update_affine_for_z_crop_diagonal():
    """For a diagonal affine, shift origin's z-component by z_start * voxel_size."""
    a = np.diag([1.5, 1.5, 1.5, 1.0])
    a[:3, 3] = [10.0, 20.0, 30.0]
    new_a = update_affine_for_z_crop(a, z_start=4)
    # z direction is (0, 0, 1.5); shift = 4 * (0, 0, 1.5) = (0, 0, 6)
    np.testing.assert_allclose(new_a[:3, 3], [10.0, 20.0, 36.0])
    # other entries unchanged
    np.testing.assert_array_equal(new_a[:3, :3], a[:3, :3])


def test_update_affine_for_z_crop_does_not_mutate():
    a = np.eye(4)
    update_affine_for_z_crop(a, z_start=5)
    np.testing.assert_array_equal(a, np.eye(4))


def test_update_affine_for_z_crop_oblique():
    """For an oblique affine, the z-direction vector isn't axis-aligned."""
    a = np.array([
        [1.5, 0.0, 0.1, 10.0],
        [0.0, 1.5, 0.0, 20.0],
        [0.0, 0.0, 1.5, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    new_a = update_affine_for_z_crop(a, z_start=2)
    # z direction = (0.1, 0, 1.5); shift = 2 * (0.1, 0, 1.5) = (0.2, 0, 3.0)
    np.testing.assert_allclose(new_a[:3, 3], [10.2, 20.0, 33.0])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
