"""Tests for the noise degradation (RicianNoise, Compose) and the JointDataset.

The most important property is NON-NEGATIVITY: Rician magnitude noise must never
produce negative voxels (real MRI magnitude data can't be negative, and negatives
would give a denoiser a trivial "tell"). There are also checks for shape/dtype,
deterministic vs. independent RNG behaviour, picklability (needed for the 'spawn'
DataLoader start method), and an end-to-end JointDataset smoke test on synthetic
data.

Run:
    python -m pytest tests/test_degradation_noise.py -v
"""

from __future__ import annotations

import hashlib
import pickle
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from src.data.compute_metadata import compute_all
from src.data.datasets import JointDataset, SpatialSRDataset
from src.data.degradation_noise import Compose, RicianNoise, make_noise
from src.data.degradation_spatial import make_spatial_degradation
from src.data.manifest import build_manifest, write_manifest


# ---------------------------------------------------------------------------
# RicianNoise
# ---------------------------------------------------------------------------

def test_rician_preserves_shape_and_dtype():
    vol = np.ones((16, 16, 12), dtype=np.float32)
    out = RicianNoise(0.05, 0.05, seed=0)(vol)
    assert out.shape == vol.shape
    assert out.dtype == np.float32


def test_rician_is_non_negative_even_on_zero_background():
    """THE key property: magnitude noise is never negative, even where the
    signal is exactly zero (background). Plain additive Gaussian would go
    negative here."""
    vol = np.zeros((20, 20, 16), dtype=np.float32)
    out = RicianNoise(0.05, 0.10, seed=1)(vol)
    assert (out >= 0).all(), "Rician noise produced negative values"
    # Pure-noise background becomes a positive Rayleigh floor, not zeros.
    assert (out > 0).any()


def test_rician_is_finite():
    vol = np.full((16, 16, 12), 1.0, dtype=np.float32)
    out = RicianNoise(0.02, 0.10, seed=2)(vol)
    assert np.isfinite(out).all()


def test_rician_seed_is_deterministic():
    vol = np.full((16, 16, 12), 1.0, dtype=np.float32)
    a = RicianNoise(0.05, 0.05, seed=7)(vol)
    b = RicianNoise(0.05, 0.05, seed=7)(vol)
    np.testing.assert_array_equal(a, b)


def test_rician_without_seed_is_independent():
    """Fresh entropy per call → two draws differ. This is what gives forked
    DataLoader workers independent noise without a worker_init_fn."""
    vol = np.full((16, 16, 12), 1.0, dtype=np.float32)
    a = RicianNoise(0.05, 0.10)(vol)
    b = RicianNoise(0.05, 0.10)(vol)
    assert not np.array_equal(a, b)


def test_rician_zero_sigma_is_passthrough_copy():
    vol = np.full((8, 8, 6), 0.7, dtype=np.float32)
    out = RicianNoise(0.0, 0.0)(vol)
    np.testing.assert_array_equal(out, vol)
    assert out is not vol  # must not alias the input
    assert out.dtype == np.float32


def test_rician_high_snr_std_matches_sigma():
    """At high SNR (constant bright signal), Rician ≈ Gaussian, so the std of
    the noised output should be close to sigma."""
    sigma = 0.05
    vol = np.ones((40, 40, 40), dtype=np.float32)
    out = RicianNoise(sigma, sigma, seed=0)(vol)
    std = float(out.std())
    assert 0.9 * sigma < std < 1.1 * sigma, f"std {std:.4f} not ≈ sigma {sigma}"


def test_rician_rejects_bad_sigma():
    with pytest.raises(ValueError):
        RicianNoise(-0.01, 0.1)
    with pytest.raises(ValueError):
        RicianNoise(0.1, 0.05)  # min > max


def test_rician_picklable_and_identical_after_roundtrip():
    """Must pickle for multiprocessing 'spawn'. With a fixed seed the round-trip
    must behave identically."""
    vol = np.full((12, 12, 10), 1.0, dtype=np.float32)
    noise = RicianNoise(0.04, 0.04, seed=123)
    rt = pickle.loads(pickle.dumps(noise))
    assert isinstance(rt, RicianNoise)
    np.testing.assert_array_equal(noise(vol), rt(vol))


def test_make_noise_returns_rician():
    n = make_noise(0.03, 0.10)
    assert isinstance(n, RicianNoise)
    assert n.sigma_min == 0.03 and n.sigma_max == 0.10


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def test_compose_applies_in_order():
    add1 = lambda x: x + 1.0          # noqa: E731 (terse for test)
    times2 = lambda x: x * 2.0        # noqa: E731
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    # (0 + 1) * 2 = 2  vs  (0 * 2) + 1 = 1  → order matters
    np.testing.assert_allclose(Compose([add1, times2])(vol), 2.0)
    np.testing.assert_allclose(Compose([times2, add1])(vol), 1.0)


def test_compose_empty_is_passthrough():
    vol = np.full((4, 4, 4), 3.0, dtype=np.float32)
    np.testing.assert_array_equal(Compose([])(vol), vol)


def test_compose_rejects_non_callable_step():
    with pytest.raises(TypeError):
        Compose([make_spatial_degradation(), 42])


def test_compose_spatial_then_noise_shape_and_nonneg():
    """The joint degradation: HR → (spatial downsample) → (Rician) → noisy LR.
    Output must be at LR shape and non-negative."""
    hr = np.full((128, 128, 92), 500.0, dtype=np.float32)
    degrade = Compose([
        make_spatial_degradation(source_voxel_mm=1.5, target_voxel_mm=3.0),
        RicianNoise(0.05, 0.05, seed=0),
    ])
    out = degrade(hr)
    assert out.shape == (64, 64, 46)
    assert (out >= 0).all()
    assert np.isfinite(out).all()


def test_compose_picklable_and_identical_after_roundtrip():
    """A composed spatial+noise degradation must pickle (the JointDataset relies
    on this for spawn workers)."""
    hr = np.full((128, 128, 92), 500.0, dtype=np.float32)
    degrade = Compose([
        make_spatial_degradation(1.5, 3.0),
        RicianNoise(0.05, 0.05, seed=99),
    ])
    rt = pickle.loads(pickle.dumps(degrade))
    assert isinstance(rt, Compose)
    np.testing.assert_array_equal(degrade(hr), rt(hr))


# ---------------------------------------------------------------------------
# JointDataset (synthetic, end-to-end)
# ---------------------------------------------------------------------------

UNIFORM_Z = 24
TARGET_SHAPE = (32, 32, UNIFORM_Z)


def _stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "big")


def _make_synthetic_bids(root, subject="01", session="03", task="Synth",
                         direction="ap", n_vols=8, shape=TARGET_SHAPE):
    func_dir = root / f"sub-{subject}" / f"ses-{session}" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    name = f"sub-{subject}_ses-{session}_task-{task}_dir-{direction}_bold.nii.gz"
    path = func_dir / name
    seed = _stable_seed(name)
    rng = np.random.default_rng(seed)
    data = np.zeros(shape + (n_vols,), dtype=np.int16)
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    r2 = (((xx - shape[0] / 2) / (shape[0] / 4)) ** 2
          + ((yy - shape[1] / 2) / (shape[1] / 4)) ** 2
          + ((zz - shape[2] / 2) / (shape[2] / 4)) ** 2)
    base = np.exp(-r2 * 0.6) * 800.0
    for t in range(n_vols):
        vol = base + rng.standard_normal(shape).astype(np.float32) * 5
        data[..., t] = vol.clip(0).astype(np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@pytest.fixture
def synthetic_manifest(tmp_path):
    bids_root = tmp_path / "bids"
    out_dir = tmp_path / "out"
    bids_root.mkdir()
    out_dir.mkdir()
    _make_synthetic_bids(bids_root, subject="01")
    _make_synthetic_bids(bids_root, subject="02")
    entries = build_manifest(bids_root, require_z=UNIFORM_Z)
    manifest_path = out_dir / "manifest.json"
    write_manifest(entries, bids_root, manifest_path, require_z=UNIFORM_Z)
    compute_all(manifest_path, derivatives_dir=out_dir, target_z=UNIFORM_Z,
                mask_method="percentile")
    return manifest_path


def test_jointdataset_sample_shapes_and_nonneg(synthetic_manifest):
    ds = JointDataset(synthetic_manifest, noise_seed=0)
    s = ds[0]
    lr = (16, 16, 12)  # 32/2, 32/2, 24/2
    assert s["input"].shape == (1, *lr)
    assert s["target"].shape == (1, 32, 32, UNIFORM_Z)
    assert s["mask_hr"].shape == (1, 32, 32, UNIFORM_Z)
    assert s["mask_lr"].shape == (1, *lr)
    # Noisy LR input must be finite and non-negative (Rician).
    assert torch.isfinite(s["input"]).all()
    assert (s["input"] >= 0).all()


def test_jointdataset_default_degrade_is_picklable_compose(synthetic_manifest):
    """The default joint degradation must be a picklable Compose so spawn
    workers can transfer it."""
    ds = JointDataset(synthetic_manifest)
    assert isinstance(ds.degrade_fn, Compose)
    rt = pickle.loads(pickle.dumps(ds.degrade_fn))
    assert isinstance(rt, Compose)


def test_jointdataset_target_is_clean_input_is_noisier(synthetic_manifest):
    """The HR target must be the clean volume (identical to what SpatialSRDataset
    serves), and the joint input must be the spatial-SR LR plus noise — i.e.
    different from, and noisier than, the clean LR."""
    joint = JointDataset(synthetic_manifest, sigma_min=0.05, sigma_max=0.05,
                         noise_seed=0)
    spatial = SpatialSRDataset(synthetic_manifest)
    js, ss = joint[0], spatial[0]
    # Same clean HR target.
    torch.testing.assert_close(js["target"], ss["target"])
    # Joint LR input differs from the clean spatial LR input.
    assert not torch.equal(js["input"], ss["input"])


def test_spatial_sr_still_works_unchanged(synthetic_manifest):
    """Regression: adding JointDataset must not disturb SpatialSRDataset."""
    ds = SpatialSRDataset(synthetic_manifest)
    s = ds[0]
    assert s["input"].shape == (1, 16, 16, 12)
    assert s["target"].shape == (1, 32, 32, UNIFORM_Z)
    assert "mask_hr" in s and "mask_lr" in s


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
