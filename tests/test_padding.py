"""Padding round-trip tests. Covers the formerly-duplicated logic in
compute_metadata that's now shared via padding.crop_from_padded.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.padding import (
    center_pad_mask,
    center_pad_volume,
    compute_pad_widths,
    crop_from_padded,
)


@pytest.mark.parametrize(
    "native,target",
    [
        ((128, 128, 84), (128, 128, 93)),   # odd diff on z
        ((128, 128, 93), (128, 128, 93)),   # zero diff (no-op)
        ((127, 128, 92), (128, 128, 93)),   # odd diff on x and z
        ((100, 100, 100), (128, 128, 128)), # even diff
        ((1, 1, 1), (3, 5, 7)),             # extreme padding
    ],
)
def test_pad_then_crop_volume_is_identity(native, target):
    rng = np.random.default_rng(0)
    src = rng.standard_normal(native).astype(np.float32)
    padded = center_pad_volume(src, target)
    assert padded.shape == target
    back = crop_from_padded(padded, native)
    assert back.shape == native
    np.testing.assert_array_equal(back, src)


@pytest.mark.parametrize(
    "native,target",
    [
        ((128, 128, 84), (128, 128, 93)),
        ((127, 128, 92), (128, 128, 93)),
        ((100, 100, 100), (128, 128, 128)),
    ],
)
def test_pad_then_crop_mask_is_identity(native, target):
    rng = np.random.default_rng(0)
    src = rng.integers(0, 2, size=native, dtype=np.int8).astype(bool)
    padded = center_pad_mask(src, target)
    back = crop_from_padded(padded, native)
    np.testing.assert_array_equal(back, src)


def test_compute_pad_widths_rejects_oversized_source():
    with pytest.raises(ValueError):
        compute_pad_widths((130, 128, 84), (128, 128, 93))


def test_pad_widths_odd_diff_extra_on_high_side():
    pads = compute_pad_widths((128, 128, 84), (128, 128, 93))
    assert pads == ((0, 0), (0, 0), (4, 5))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
