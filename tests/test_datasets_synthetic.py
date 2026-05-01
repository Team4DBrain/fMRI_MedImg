"""End-to-end Dataset tests on synthetic data, z_crop pipeline.

Synthetic IBC-shaped runs with NON-uniform z (some 24, some 28). We expect:
  - target_z auto-detected to min (24)
  - per-run z_start computed and stored in manifest
  - all served samples at (X, Y, 24)
  - SR scale sanity (regression for the inverted-scale bug)
  - mask cache returns clones, missing files fail fast, etc.
"""

from __future__ import annotations

import json
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.compute_metadata import compute_all
from src.data.datasets import (
    DenoisingDataset,
    SpatialSRDataset,
    TemporalSRDataset,
)
from src.data.degradation_spatial import make_spatial_degradation
from src.data.manifest import build_manifest, write_manifest


def _make_synthetic_bids(root, subject="01", session="00", task="Synth",
                         direction="ap", n_vols=12, shape=(32, 32, 24)):
    """Write a tiny synthetic 4D nii.gz with a BIDS-conformant filename."""
    func_dir = root / f"sub-{subject}" / f"ses-{session}" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    name = f"sub-{subject}_ses-{session}_task-{task}_dir-{direction}_bold.nii.gz"
    path = func_dir / name

    # Smooth blob at run-specific z-position so different runs need different z_start.
    rng = np.random.default_rng(hash(name) % (2**32))
    data = np.zeros(shape + (n_vols,), dtype=np.int16)
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    z_center = shape[2] / 2 + (hash(name) % 5) - 2  # ±2 voxels offset per run
    r2 = (((xx - shape[0] / 2) / (shape[0] / 4)) ** 2
          + ((yy - shape[1] / 2) / (shape[1] / 4)) ** 2
          + ((zz - z_center) / (shape[2] / 4)) ** 2)
    base = np.exp(-r2 * 0.6) * 800.0
    for t in range(n_vols):
        vol = base + rng.standard_normal(shape).astype(np.float32) * 5
        data[..., t] = vol.clip(0).astype(np.int16)

    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@pytest.fixture
def synthetic_pipeline(tmp_path):
    """Synthetic dataset with NON-uniform z. Returns (manifest_path, target_z)."""
    bids_root = tmp_path / "bids"
    out_dir = tmp_path / "out"
    bids_root.mkdir()
    out_dir.mkdir()

    # 3 subjects: two with z=28, one with z=24. target_z should auto = 24.
    _make_synthetic_bids(bids_root, subject="01", n_vols=10, shape=(32, 32, 28))
    _make_synthetic_bids(bids_root, subject="02", n_vols=10, shape=(32, 32, 28))
    _make_synthetic_bids(bids_root, subject="03", n_vols=10, shape=(32, 32, 24))

    entries = build_manifest(bids_root)
    manifest_path = out_dir / "manifest.json"
    write_manifest(entries, bids_root, manifest_path)

    compute_all(
        manifest_path,
        derivatives_dir=out_dir,
        target_z=None,            # auto -> 24
        mask_method="percentile",
    )

    return manifest_path, 24


def test_target_z_auto_detected_to_min(synthetic_pipeline):
    manifest_path, target_z = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    assert m["target_z"] == target_z
    assert m["target_shape"] == [32, 32, 24]
    assert m["pipeline"] == "z_crop"


def test_per_run_z_start_in_manifest(synthetic_pipeline):
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    for r in m["runs"]:
        assert "z_start" in r
        # Run with native z=24 must have z_start=0 (no cropping possible)
        if r["shape"][2] == 24:
            assert r["z_start"] == 0


def test_denoising_serves_target_z_shape(synthetic_pipeline):
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
    # Mask values should be 0 or 1
    mvals = set(sample["mask"].unique().tolist())
    assert mvals.issubset({0.0, 1.0})


def test_spatial_sr_scale_sanity(synthetic_pipeline):
    """Regression test for the inverted-scale bug. After normalize, brain
    voxel mean should be near 1.0 in both LR and HR."""
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


def test_temporal_sr_basic(synthetic_pipeline):
    manifest_path, target_z = synthetic_pipeline
    ds = TemporalSRDataset(manifest_path, gap=1)
    sample = ds[0]
    assert sample["input"].shape == (2, 32, 32, target_z)
    assert sample["target"].shape == (1, 32, 32, target_z)
    assert sample["mask"].shape == (1, 32, 32, target_z)


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


def test_check_files_exist_catches_missing(synthetic_pipeline):
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    m["runs"][0]["path"] = "does/not/exist_bold.nii.gz"
    with open(manifest_path, "w") as f:
        json.dump(m, f)
    with pytest.raises(FileNotFoundError):
        DenoisingDataset(manifest_path, degrade_fn=lambda x: x)


def test_old_padding_manifest_rejected(tmp_path, synthetic_pipeline):
    """A manifest without pipeline=z_crop should be rejected."""
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    del m["pipeline"]
    with open(manifest_path, "w") as f:
        json.dump(m, f)
    with pytest.raises(RuntimeError, match="z_crop"):
        DenoisingDataset(manifest_path, degrade_fn=lambda x: x)


def test_target_z_too_large_rejected(tmp_path):
    """Asking for target_z > min native z must fail loudly."""
    bids_root = tmp_path / "bids"
    out_dir = tmp_path / "out"
    bids_root.mkdir()
    out_dir.mkdir()
    _make_synthetic_bids(bids_root, subject="01", shape=(32, 32, 24))
    _make_synthetic_bids(bids_root, subject="02", shape=(32, 32, 28))
    entries = build_manifest(bids_root)
    manifest_path = out_dir / "manifest.json"
    write_manifest(entries, bids_root, manifest_path)
    with pytest.raises(ValueError, match="exceeds the smallest native z"):
        compute_all(manifest_path, derivatives_dir=out_dir, target_z=26,
                    mask_method="percentile")


def test_mask_z_bbox_centered_per_run(synthetic_pipeline):
    """The cropped mask should contain the brain. We synthesized blobs at
    slightly different z-centers per run, so z_start should differ."""
    manifest_path, _ = synthetic_pipeline
    with open(manifest_path) as f:
        m = json.load(f)
    z_starts = [r["z_start"] for r in m["runs"]]
    # All runs that need cropping (native z=28, target=24) should have z_start
    # somewhere reasonable. Run with native=24 should have z_start=0.
    # We don't assert exact values — synthetic seeded blobs vary.
    for r in m["runs"]:
        if r["shape"][2] == 24:
            assert r["z_start"] == 0
        else:
            assert 0 <= r["z_start"] <= r["shape"][2] - 24


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
