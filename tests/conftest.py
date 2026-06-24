"""Shared pytest fixtures for the CAI-MedImg test suite."""

from __future__ import annotations

import hashlib

import nibabel as nib
import numpy as np
import pytest

from data.compute_metadata import compute_all
from data.manifest import build_manifest, write_manifest

UNIFORM_Z = 24
TARGET_SHAPE = (32, 32, UNIFORM_Z)


def _stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "big")


def _make_synthetic_bids(
    root,
    subject="01",
    session="00",
    task="Synth",
    direction="ap",
    n_vols=12,
    shape=(32, 32, 24),
):
    func_dir = root / f"sub-{subject}" / f"ses-{session}" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    name = f"sub-{subject}_ses-{session}_task-{task}_dir-{direction}_bold.nii.gz"
    path = func_dir / name
    seed = _stable_seed(name)
    rng = np.random.default_rng(seed)
    data = np.zeros(shape + (n_vols,), dtype=np.int16)
    xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    z_center = shape[2] / 2 + (seed % 5) - 2
    r2 = (
        ((xx - shape[0] / 2) / (shape[0] / 4)) ** 2
        + ((yy - shape[1] / 2) / (shape[1] / 4)) ** 2
        + ((zz - z_center) / (shape[2] / 4)) ** 2
    )
    base = np.exp(-r2 * 0.6) * 800.0
    for t in range(n_vols):
        vol = base + rng.standard_normal(shape).astype(np.float32) * 5
        data[..., t] = vol.clip(0).astype(np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@pytest.fixture
def synthetic_pipeline(tmp_path):
    """Synthetic no_crop_v1 manifest + derivatives. Returns (manifest_path, target_z)."""
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
