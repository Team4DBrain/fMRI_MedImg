"""Subject-disjoint splits and dataset / dataloader construction for the joint model.

A split is a SUBJECT-level decision: no subject appears in two sets. Splitting by
timepoint would leak a subject's anatomy across train and eval and inflate
metrics, so we only ever split on ``subject``. The concrete subject->split
assignment is defined where training happens (the VM, on its own manifest); this
module is just the machinery — it validates the lists against the manifest and
wires them into ``JointDataset``.

Noise reproducibility note: val/test datasets are built with ``noise_seed=None``
(fresh Rician noise per sample). That keeps the noise *per-sample distinct* (the
degenerate alternative — a single fixed ``noise_seed`` — gives every sample the
identical noise field). The cost is that the exact noise is not reproducible
run-to-run; averaged over a whole split the metric variance from this is tiny.
Fully reproducible per-sample val noise would need a small hook in the data layer
(passing sample identity to the degradation) — a separate decision for the data
owner, not made here.
"""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from data.datasets import JointDataset

from .config import Config


def load_subjects(manifest_path: str | Path) -> list[str]:
    """All subject ids present in a manifest (zero-padded, e.g. '01')."""
    runs = json.loads(Path(manifest_path).read_text())["runs"]
    return sorted({r["subject"] for r in runs})


def make_splits(manifest_path, val_subjects, test_subjects=()) -> dict:
    """Build a subject-disjoint split. ``val_subjects``/``test_subjects`` are
    explicit subject-id lists; everything else becomes train. Validates against
    the manifest and refuses overlaps or unknown subjects."""
    val = set(val_subjects)
    test = set(test_subjects)
    overlap = val & test
    if overlap:
        raise ValueError(f"val/test subjects overlap: {sorted(overlap)}")
    available = set(load_subjects(manifest_path))
    unknown = (val | test) - available
    if unknown:
        raise ValueError(
            f"unknown subjects {sorted(unknown)}; available {sorted(available)} "
            "(subjects are zero-padded, e.g. '01')"
        )
    train = sorted(available - val - test)
    if not train:
        raise ValueError("no subjects left for training after removing val/test")
    return {"train": train, "val": sorted(val), "test": sorted(test)}


def build_dataset(cfg: Config, manifest_path, subjects, noise_seed=None) -> JointDataset:
    """A JointDataset over the given subjects, using the config's degradation
    parameters. ``noise_seed=None`` => fresh per-sample noise (see module note)."""
    return JointDataset(
        manifest_path,
        subject_filter=list(subjects),
        source_voxel_mm=cfg.train.source_voxel_mm,
        target_voxel_mm=cfg.train.target_voxel_mm,
        sigma_min=cfg.train.sigma_min,
        sigma_max=cfg.train.sigma_max,
        noise_seed=noise_seed,
    )


def build_loaders(cfg: Config, manifest_path, splits: dict):
    """Construct (train_loader, val_loader, train_ds, val_ds) from a split dict."""
    tcfg = cfg.train
    train_ds = build_dataset(cfg, manifest_path, splits["train"])
    val_ds = build_dataset(cfg, manifest_path, splits["val"])
    train_loader = DataLoader(
        train_ds, batch_size=tcfg.batch_size, shuffle=True,
        num_workers=tcfg.num_workers, pin_memory=True,
        persistent_workers=(tcfg.num_workers > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.val_batch_size, shuffle=False,
        num_workers=tcfg.num_workers, pin_memory=True,
        persistent_workers=(tcfg.num_workers > 0),
    )
    return train_loader, val_loader, train_ds, val_ds
