"""Reader cache tests. Verifies get_reader returns the same instance for the
same path within a process, and that read_range matches independent reads.
"""

from __future__ import annotations

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from data.reader import VolumeReader, clear_reader_cache, get_reader


@pytest.fixture
def tmp_nii(tmp_path):
    """Tiny fake 4D nii.gz."""
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((8, 8, 6, 5)) * 100 + 500).astype(np.int16)
    path = tmp_path / "fake_bold.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path, data


def test_volume_reader_basic(tmp_nii):
    path, data = tmp_nii
    r = VolumeReader(path)
    assert r.shape == (8, 8, 6, 5)
    assert r.shape3d == (8, 8, 6)
    assert r.n_volumes == 5

    np.testing.assert_array_equal(r.read_volume(0), data[..., 0])
    np.testing.assert_array_equal(r.read_volume(4), data[..., 4])
    with pytest.raises(IndexError):
        r.read_volume(5)
    with pytest.raises(IndexError):
        r.read_volume(-1)


def test_read_range_matches_per_volume(tmp_nii):
    path, data = tmp_nii
    r = VolumeReader(path)
    block = r.read_range(1, 4)  # (X,Y,Z,3)
    assert block.shape == (8, 8, 6, 3)
    for i, t in enumerate(range(1, 4)):
        np.testing.assert_array_equal(block[..., i], r.read_volume(t))


def test_read_range_invalid(tmp_nii):
    path, _ = tmp_nii
    r = VolumeReader(path)
    with pytest.raises(IndexError):
        r.read_range(2, 2)  # empty
    with pytest.raises(IndexError):
        r.read_range(0, 6)  # past end


def test_read_full_dtype(tmp_nii):
    path, data = tmp_nii
    r = VolumeReader(path)
    full = r.read_full(dtype=np.float32)
    assert full.dtype == np.float32
    assert full.shape == (8, 8, 6, 5)
    np.testing.assert_allclose(full, data.astype(np.float32))


def test_read_mean_matches_manual(tmp_nii):
    path, data = tmp_nii
    r = VolumeReader(path)
    # Reference: float64 accumulator, then cast to float32 — matches the
    # implementation. Using float32 throughout would diverge by ~1 ULP * N
    # which slips past assert_allclose default rtol.
    expected = data.astype(np.float64).mean(axis=-1).astype(np.float32)
    np.testing.assert_allclose(r.read_mean(), expected)


def test_get_reader_returns_same_instance(tmp_nii):
    path, _ = tmp_nii
    clear_reader_cache()
    r1 = get_reader(path)
    r2 = get_reader(path)
    assert r1 is r2


def test_get_reader_resolves_path(tmp_nii):
    """Different string forms of the same path hit the same cache entry."""
    path, _ = tmp_nii
    clear_reader_cache()
    r1 = get_reader(str(path))
    r2 = get_reader(path)  # Path object
    assert r1 is r2


def test_clear_reader_cache(tmp_nii):
    path, _ = tmp_nii
    clear_reader_cache()
    r1 = get_reader(path)
    clear_reader_cache()
    r2 = get_reader(path)
    assert r1 is not r2


def test_volume_reader_rejects_non_4d(tmp_path):
    """Constructor should fail on a 3D NIfTI."""
    data = np.zeros((8, 8, 6), dtype=np.int16)
    path = tmp_path / "anat.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    with pytest.raises(ValueError):
        VolumeReader(path)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
