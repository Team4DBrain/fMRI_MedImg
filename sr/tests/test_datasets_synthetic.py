"""End-to-end Dataset tests on synthetic data, no_crop_v1 pipeline.

Synthetic IBC-shaped runs with uniform z (require_z screens out non-conforming
runs at the manifest stage). We expect:
  - all served samples at the manifest's target_shape
  - manifest pipeline marker == 'no_crop_v1'
  - SR scale sanity (regression for the inverted-scale bug)
  - mask cache returns clones, missing files fail fast, etc.
  - non-conforming z runs are dropped at manifest build with a warning
"""

from __future__ import annotations

import hashlib
import json
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.compute_metadata import compute_all
from data.datasets import (
    DenoisingDataset,
    SpatialSRDataset,
    TemporalSRDataset,
)
from data.degradation_spatial import (
    SpatialDegradation,
    make_spatial_degradation,
)
from data.manifest import build_manifest, write_manifest


def _stable_seed(name: str) -> int:
    """Stable 32-bit seed from a string. Python's built-in hash() is salted
    per-interpreter (PYTHONHASHSEED), so seeding with it gives different
    test data across pytest invocations and breaks reproducibility.
    """
    return int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "big")


def _make_synthetic_bids(root, subject="01", session="00", task="Synth",
                         direction="ap", n_vols=12, shape=(32, 32, 24)):
    """Write a tiny synthetic 4D nii.gz with a BIDS-conformant filename."""
    func_dir = root / f"sub-{subject}" / f"ses-{session}" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    name = f"sub-{subject}_ses-{session}_task-{task}_dir-{direction}_bold.nii.gz"
    path = func_dir / name

    seed = _stable_seed(name)
    rng = np.random.default_rng(seed)
    data = np.zeros(shape + (n_vols,), dtype=np.int16)
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    z_center = shape[2] / 2 + (seed % 5) - 2  # ±2 voxels offset per run
    r2 = (((xx - shape[0] / 2) / (shape[0] / 4)) ** 2
          + ((yy - shape[1] / 2) / (shape[1] / 4)) ** 2
          + ((zz - z_center) / (shape[2] / 4)) ** 2)
    base = np.exp(-r2 * 0.6) * 800.0
    for t in range(n_vols):
        vol = base + rng.standard_normal(shape).astype(np.float32) * 5
        data[..., t] = vol.clip(0).astype(np.int16)

    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


# ---------------------------------------------------------------------------
# Fixture: uniform z=24, three subjects. The shared "happy-path" pipeline.
# ---------------------------------------------------------------------------

UNIFORM_Z = 24
TARGET_SHAPE = (32, 32, UNIFORM_Z)


@pytest.fixture
def synthetic_pipeline(tmp_path):
    """Synthetic dataset with uniform z=24. Returns (manifest_path, target_z)."""
    bids_root = tmp_path / "bids"
    out_dir = tmp_path / "out"
    bids_root.mkdir()
    out_dir.mkdir()

    _make_synthetic_bids(bids_root, subject="01", n_vols=10, shape=TARGET_SHAPE)
    _make_synthetic_bids(bids_root, subject="02", n_vols=10, shape=TARGET_SHAPE)
    _make_synthetic_bids(bids_root, subject="03", n_vols=10, shape=TARGET_SHAPE)

    entries = build_manifest(bids_root, require_z=UNIFORM_Z)
    manifest_path = out_dir / "manifest.json"
    write_manifest(entries, bids_root, manifest_path, require_z=UNIFORM_Z)

    compute_all(
        manifest_path,
        derivatives_dir=out_dir,
        target_z=UNIFORM_Z,
        mask_method="percentile",
    )

    return manifest_path, UNIFORM_Z


# ---------------------------------------------------------------------------
# Manifest / pipeline marker
# ---------------------------------------------------------------------------

def test_manifest_marker_is_no_crop_v1(synthetic_pipeline):
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    assert m["pipeline"] == "no_crop_v1"
    assert m["target_z"] == UNIFORM_Z
    assert m["target_shape"] == list(TARGET_SHAPE)
    assert m["require_z"] == UNIFORM_Z


def test_runs_have_no_z_start(synthetic_pipeline):
    """no_crop_v1 manifest entries must NOT carry z_start (legacy field)."""
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    for r in m["runs"]:
        assert "z_start" not in r, f"{r['run_id']} unexpectedly carries z_start"


# ---------------------------------------------------------------------------
# Manifest filtering: non-conforming z runs are dropped
# ---------------------------------------------------------------------------

def test_manifest_drops_nonconforming_z(tmp_path, caplog):
    """Mix of z=24 and z=28 runs. require_z=24 should drop the z=28 ones."""
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    _make_synthetic_bids(bids_root, subject="01", shape=(32, 32, 24))
    _make_synthetic_bids(bids_root, subject="02", shape=(32, 32, 24))
    _make_synthetic_bids(bids_root, subject="03", shape=(32, 32, 28))

    import logging
    with caplog.at_level(logging.WARNING):
        entries = build_manifest(bids_root, require_z=24)

    assert len(entries) == 2
    assert all(e.shape[2] == 24 for e in entries)
    assert any("Dropping" in r.message and "z=28" in r.message for r in caplog.records)


def test_manifest_no_filter_keeps_all_z(tmp_path):
    """require_z=None disables filtering — all runs survive."""
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    _make_synthetic_bids(bids_root, subject="01", shape=(32, 32, 24))
    _make_synthetic_bids(bids_root, subject="02", shape=(32, 32, 28))

    entries = build_manifest(bids_root, require_z=None)
    assert len(entries) == 2


def test_compute_all_rejects_target_z_mismatch(tmp_path):
    """If manifest's require_z disagrees with the explicit target_z, raise."""
    bids_root = tmp_path / "bids"
    out_dir = tmp_path / "out"
    bids_root.mkdir()
    out_dir.mkdir()
    _make_synthetic_bids(bids_root, subject="01", shape=(32, 32, 24))
    entries = build_manifest(bids_root, require_z=24)
    manifest_path = out_dir / "manifest.json"
    write_manifest(entries, bids_root, manifest_path, require_z=24)

    with pytest.raises(ValueError, match="disagrees"):
        compute_all(manifest_path, derivatives_dir=out_dir, target_z=28,
                    mask_method="percentile")


# ---------------------------------------------------------------------------
# Dataset behavior
# ---------------------------------------------------------------------------

def test_denoising_serves_target_shape(synthetic_pipeline):
    manifest_path, target_z = synthetic_pipeline

    def add_noise(clean):
        return (clean + np.random.normal(0, 0.05, clean.shape)).astype(np.float32)

    ds = DenoisingDataset(manifest_path, degrade_fn=add_noise)
    sample = ds[0]
    expected = (1, 32, 32, target_z)
    assert sample["input"].shape == expected
    assert sample["target"].shape == expected
    assert sample["mask"].shape == expected
    assert sample["mask"].dtype == torch.float32
    mvals = set(sample["mask"].unique().tolist())
    assert mvals.issubset({0.0, 1.0})


def test_denoising_validates_degrade_fn_output(synthetic_pipeline):
    """A buggy degrade_fn that returns the wrong shape must raise."""
    manifest_path, _ = synthetic_pipeline

    def bad_degrade(clean):
        # Drop a slice — wrong shape
        return clean[..., :-1]

    ds = DenoisingDataset(manifest_path, degrade_fn=bad_degrade)
    with pytest.raises(RuntimeError, match="DenoisingDataset.*shape"):
        ds[0]


def test_denoising_rejects_nonfinite_degrade_fn(synthetic_pipeline):
    """A degrade_fn that produces NaN/Inf must raise (not silently propagate)."""
    manifest_path, _ = synthetic_pipeline

    def nan_degrade(clean):
        out = clean.copy()
        out.flat[0] = np.nan
        return out

    ds = DenoisingDataset(manifest_path, degrade_fn=nan_degrade)
    with pytest.raises(RuntimeError, match="non-finite"):
        ds[0]


def test_spatial_sr_scale_sanity(synthetic_pipeline):
    """Regression for the inverted-scale bug: LR brain mean ≈ HR brain mean."""
    manifest_path, _ = synthetic_pipeline

    degrade = make_spatial_degradation(source_voxel_mm=1.5, target_voxel_mm=3.0)
    ds = SpatialSRDataset(manifest_path, degrade_fn=degrade)
    sample = ds[0]

    hr_brain_mean = sample["target"][sample["mask_hr"] > 0.5].mean().item()
    lr_brain_mean = sample["input"][sample["mask_lr"] > 0.5].mean().item()
    ratio = lr_brain_mean / max(hr_brain_mean, 1e-6)
    assert 0.5 < ratio < 2.0, (
        f"LR/HR brain mean ratio = {ratio:.2f} — scale bug regression "
        "(broken version was ~64x)"
    )


def test_spatial_sr_default_degradation_picklable(synthetic_pipeline):
    """SpatialDegradation must be picklable so multiprocessing spawn workers
    can transfer it. Closures from a factory cannot. This is a regression
    test for the closure-vs-class refactor."""
    import pickle
    manifest_path, _ = synthetic_pipeline
    ds = SpatialSRDataset(manifest_path)
    assert isinstance(ds.degrade_fn, SpatialDegradation)
    # Round-trip through pickle.
    roundtrip = pickle.loads(pickle.dumps(ds.degrade_fn))
    assert isinstance(roundtrip, SpatialDegradation)
    # Behaves identically.
    sample_arr = np.full(TARGET_SHAPE, 100.0, dtype=np.float32)
    out_a = ds.degrade_fn(sample_arr)
    out_b = roundtrip(sample_arr)
    np.testing.assert_array_equal(out_a, out_b)


def test_spatial_sr_no_voxel_mm_attrs(synthetic_pipeline):
    """SpatialSRDataset should NOT expose source_voxel_mm/target_voxel_mm as
    attrs — they would silently lie when a custom degrade_fn is passed.
    lr_shape is the canonical attr."""
    manifest_path, _ = synthetic_pipeline
    ds = SpatialSRDataset(manifest_path)
    assert hasattr(ds, "lr_shape")
    assert not hasattr(ds, "source_voxel_mm")
    assert not hasattr(ds, "target_voxel_mm")


def test_temporal_sr_basic(synthetic_pipeline):
    manifest_path, target_z = synthetic_pipeline
    ds = TemporalSRDataset(manifest_path, gap=1)
    sample = ds[0]
    assert sample["input"].shape == (2, 32, 32, target_z)
    assert sample["target"].shape == (1, 32, 32, target_z)
    assert sample["mask"].shape == (1, 32, 32, target_z)


def test_temporal_sr_gap2_uses_single_read(synthetic_pipeline, monkeypatch):
    """For gap>=2, TemporalSR should issue a single _read_range call, not
    three separate _read_volume calls."""
    manifest_path, _ = synthetic_pipeline
    ds = TemporalSRDataset(manifest_path, gap=2)

    range_calls = []
    volume_calls = []
    orig_range = ds._read_range
    orig_volume = ds._read_volume

    def spy_range(run_idx, t_start, t_end):
        range_calls.append((run_idx, t_start, t_end))
        return orig_range(run_idx, t_start, t_end)

    def spy_volume(run_idx, t):
        volume_calls.append((run_idx, t))
        return orig_volume(run_idx, t)

    monkeypatch.setattr(ds, "_read_range", spy_range)
    monkeypatch.setattr(ds, "_read_volume", spy_volume)

    _ = ds[0]
    assert len(range_calls) == 1, f"expected 1 range read, got {len(range_calls)}"
    assert len(volume_calls) == 0, (
        f"expected 0 per-volume reads with single-range optimization, "
        f"got {len(volume_calls)}"
    )


def test_temporal_sr_drops_runs_too_short(synthetic_pipeline):
    manifest_path, _ = synthetic_pipeline
    with pytest.raises(RuntimeError, match="empty sample index"):
        TemporalSRDataset(manifest_path, gap=10)


def test_temporal_sr_rejects_bad_gap(synthetic_pipeline):
    manifest_path, _ = synthetic_pipeline
    with pytest.raises(ValueError):
        TemporalSRDataset(manifest_path, gap=0)


def test_subject_filter_unmatched_raises(synthetic_pipeline):
    manifest_path, _ = synthetic_pipeline
    with pytest.raises(ValueError, match="not in manifest"):
        DenoisingDataset(manifest_path, subject_filter=["99"])


def test_dataloader_smoketest(synthetic_pipeline):
    manifest_path, target_z = synthetic_pipeline
    ds = DenoisingDataset(manifest_path, degrade_fn=lambda x: x)
    loader = DataLoader(ds, batch_size=4, num_workers=0, shuffle=True)
    batch = next(iter(loader))
    assert batch["input"].shape == (4, 1, 32, 32, target_z)


def test_mask_cache_returns_clones(synthetic_pipeline):
    """Mutating a returned mask must not corrupt the cache."""
    manifest_path, _ = synthetic_pipeline
    ds = DenoisingDataset(manifest_path, degrade_fn=lambda x: x)
    s1 = ds[0]
    original_sum = s1["mask"].sum().item()
    s1["mask"][:] = 0.0
    s2 = ds[0]
    assert s2["mask"].sum().item() == original_sum


def test_mask_cache_is_bounded(synthetic_pipeline):
    """The bounded cache caps memory: with cache_size=1 and >1 runs, the cache
    should evict between accesses but still serve consistent values."""
    manifest_path, _ = synthetic_pipeline
    ds = DenoisingDataset(manifest_path, degrade_fn=lambda x: x, mask_cache_size=1)

    # Touch every run to force eviction churn.
    seen_sums = {}
    for run_idx, run in enumerate(ds.runs):
        m = ds._get_hr_mask(run_idx)
        seen_sums[run["run_id"]] = m.sum().item()

    # Cache should hold AT MOST one entry, despite touching 3 runs.
    assert len(ds._mask_cache._d) == 1

    # Repeat access — values must still be correct (not stale clones).
    for run_idx, run in enumerate(ds.runs):
        m = ds._get_hr_mask(run_idx)
        assert m.sum().item() == seen_sums[run["run_id"]]


def test_check_files_exist_catches_missing(synthetic_pipeline):
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    m["runs"][0]["path"] = "does/not/exist_bold.nii.gz"
    with open(manifest_path, "w") as f:
        json.dump(m, f)
    with pytest.raises(FileNotFoundError):
        DenoisingDataset(manifest_path, degrade_fn=lambda x: x)


def test_old_z_crop_manifest_rejected(synthetic_pipeline):
    """A manifest from the old z_crop pipeline must be rejected with a clear
    message that points at re-running the build."""
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    m["pipeline"] = "z_crop"
    with open(manifest_path, "w") as f:
        json.dump(m, f)
    with pytest.raises(RuntimeError, match="no_crop_v1"):
        DenoisingDataset(manifest_path, degrade_fn=lambda x: x)


def test_inconsistent_run_shape_rejected(synthetic_pipeline):
    """A manifest where some run's shape disagrees with target_shape is
    rejected at Dataset construction (defends against hand-edited manifests)."""
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    # Tamper: claim run 0 has a different z than target_shape.
    bad_shape = list(m["runs"][0]["shape"])
    bad_shape[2] = 99
    m["runs"][0]["shape"] = bad_shape
    with open(manifest_path, "w") as f:
        json.dump(m, f)
    with pytest.raises(RuntimeError, match="target_shape"):
        DenoisingDataset(manifest_path, degrade_fn=lambda x: x)


# ---------------------------------------------------------------------------
# Idempotent fast path: re-running compute_all without --overwrite should not
# re-read the 4D files.
# ---------------------------------------------------------------------------

def test_compute_all_skips_when_metadata_complete(synthetic_pipeline, monkeypatch):
    """Second call to compute_all (without overwrite) must not call read_full."""
    from data import compute_metadata as cm
    from data import reader as reader_mod

    manifest_path, _ = synthetic_pipeline
    derivatives_dir = manifest_path.parent

    # Sanity: first pass already populated metadata.
    with open(manifest_path) as f:
        m = json.load(f)
    assert all("norm_ref" in r for r in m["runs"])

    read_full_calls = {"n": 0}
    orig = reader_mod.VolumeReader.read_full

    def counting_read_full(self, *a, **kw):
        read_full_calls["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(reader_mod.VolumeReader, "read_full", counting_read_full)

    cm.compute_all(manifest_path, derivatives_dir=derivatives_dir,
                   target_z=UNIFORM_Z, mask_method="percentile")

    assert read_full_calls["n"] == 0, (
        f"compute_all re-read {read_full_calls['n']} 4D files even though "
        "metadata was complete and --overwrite was False"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
