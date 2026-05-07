"""Construct reproducible train/validation loaders for spatial SR.

Purpose:
    Bridge validated config values to dataset and DataLoader objects used by
    training and evaluation.
Effects:
    Controls which subjects are seen by each split, how degradation is applied,
    and whether batch ordering/worker RNG are reproducible.
Influences:
    Behavior depends on manifest content, split config, voxel settings, and
    deterministic policy.
How to change safely:
    Keep split resolution, degradation creation, and loader seeding aligned with
    `src.sr.config.DEFAULT_CONFIG` and downstream expectations in
    `src.sr.training`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation


def _available_subjects(manifest_path: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    subjects = {str(run["subject"]) for run in manifest.get("runs", []) if "subject" in run}
    ordered = sorted(subjects)
    if not ordered:
        raise RuntimeError("Manifest contains no subjects.")
    return ordered


def _resolve_subject_split(config: dict, manifest_path: Path) -> tuple[list[str], list[str]]:
    """Resolve final subject lists for training and validation.

    Purpose:
        Convert high-level split config into explicit subject lists consumed by
        `SpatialSRDataset`.
    Effects:
        Determines which data contributes to gradient updates vs validation
        metrics, directly affecting generalization estimates.
    Influences:
        Priority order is explicit lists > derived random split > all-train
        default; derived splits depend on `seed` and `train_split`.
    How to change safely:
        Preserve deterministic split behavior and validation safety checks
        (minimum subject count) when adjusting split policy.
    """
    configured_train = config.get("train_subjects")
    configured_val = config.get("val_subjects")
    if configured_train is not None and configured_val is not None:
        train_subjects = [str(s) for s in configured_train]
        val_subjects = [str(s) for s in configured_val]
        if not train_subjects:
            raise ValueError("Explicit train_subjects must be non-empty.")
        return train_subjects, val_subjects

    subjects = _available_subjects(manifest_path)
    if not bool(config.get("enable_subject_split", False)):
        # No split: everything goes to train, no validation set.
        return subjects, []

    if len(subjects) < 2:
        raise ValueError(
            "At least two subjects are required for independent train/val split "
            "when enable_subject_split=True. Provide a manifest with >=2 subjects "
            "or disable subject splitting."
        )

    rng = np.random.default_rng(int(config["seed"]))
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * float(config["train_split"])))
    split_idx = min(split_idx, len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def _make_loader(dataset, config: dict, *, shuffle: bool, generator_seed: int):
    deterministic = bool(config.get("deterministic", False))
    pin_memory = torch.cuda.is_available()
    worker_init_fn = None
    if deterministic:

        def _seed_worker(worker_id: int) -> None:
            worker_seed = int(config["seed"]) + worker_id
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _seed_worker

    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(generator_seed),
    )


def create_dataloaders(config: dict):
    """Create train/validation loaders configured for spatial SR experiments.

    Purpose:
        Build the canonical data pipeline consumed by training and eval code.
    Effects:
        Instantiates degradation, datasets, and loaders; returns split metadata
        used for logging and run artifacts.
    Influences:
        Loader behavior is affected by voxel-size degradation settings, subject
        split configuration, batch size, worker count, and deterministic mode.
    How to change safely:
        Keep returned tuple structure stable because `run.py` and
        `training.py` depend on it; if adding fields, update all callers.
    """
    manifest_path = Path(config["manifest_path"])
    degrade_fn = make_spatial_degradation(
        source_voxel_mm=float(config["source_voxel_mm"]),
        target_voxel_mm=float(config["target_voxel_mm"]),
    )
    train_subjects, val_subjects = _resolve_subject_split(config, manifest_path)

    train_dataset = SpatialSRDataset(
        manifest_path=manifest_path,
        subject_filter=train_subjects,
        degrade_fn=degrade_fn,
    )

    if val_subjects:
        val_dataset = SpatialSRDataset(
            manifest_path=manifest_path,
            subject_filter=val_subjects,
            degrade_fn=degrade_fn,
        )
    else:
        val_dataset = None

    train_loader = _make_loader(train_dataset, config, shuffle=True, generator_seed=int(config["seed"]) + 101)
    if val_dataset is not None:
        val_loader = _make_loader(val_dataset, config, shuffle=False, generator_seed=int(config["seed"]) + 202)
        val_samples = len(val_dataset)
    else:
        val_loader = None
        val_samples = 0

    dataset_size = len(train_dataset) + val_samples
    split_info = {
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train_samples": len(train_dataset),
        "val_samples": val_samples,
    }
    return train_loader, val_loader, dataset_size, split_info
