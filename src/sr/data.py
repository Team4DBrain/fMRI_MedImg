"""Data loading adapter for SR training on src.data manifest datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation


class SRSpatialDatasetAdapter(Dataset):
    """Adapter that maps src.data sample dicts to SR trainer tuple format."""

    def __init__(self, spatial_dataset: SpatialSRDataset):
        self.spatial_dataset = spatial_dataset

    def __len__(self) -> int:
        return len(self.spatial_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.spatial_dataset[idx]
        return sample["input"].float(), sample["target"].float()


def _available_subjects(manifest_path: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    runs = manifest.get("runs", [])
    subjects = {
        str(run["subject"])
        for run in runs
        if "subject" in run and "norm_ref" in run and "mask_path" in run
    }
    ordered = sorted(subjects)
    if not ordered:
        raise RuntimeError("Manifest contains no usable runs with subject, norm_ref, and mask_path.")
    return ordered


def _subject_split(config: dict, manifest_path: Path) -> tuple[list[str], list[str]]:
    configured_train = config.get("train_subjects")
    configured_val = config.get("val_subjects")
    if configured_train is not None:
        train_subjects = [str(s) for s in configured_train]
        val_subjects = [str(s) for s in configured_val] if configured_val is not None else []
        return train_subjects, val_subjects

    subjects = _available_subjects(manifest_path)
    rng = np.random.default_rng(int(config["seed"]))
    shuffled = list(subjects)
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * float(config["train_split"])))
    if split_idx >= len(shuffled) and len(shuffled) > 1:
        split_idx = len(shuffled) - 1

    train_subjects = shuffled[:split_idx]
    val_subjects = shuffled[split_idx:]
    return train_subjects, val_subjects


def create_dataloaders(config: dict):
    """Build train/validation DataLoaders from manifest-backed spatial SR dataset."""
    manifest_path = Path(config["manifest_path"])
    degrade_fn = make_spatial_degradation(
        source_voxel_mm=float(config["source_voxel_mm"]),
        target_voxel_mm=float(config["target_voxel_mm"]),
    )
    train_subjects, val_subjects = _subject_split(config, manifest_path)

    train_base = SpatialSRDataset(
        manifest_path=manifest_path,
        subject_filter=train_subjects,
        degrade_fn=degrade_fn,
        source_voxel_mm=float(config["source_voxel_mm"]),
        target_voxel_mm=float(config["target_voxel_mm"]),
    )
    train_dataset = SRSpatialDatasetAdapter(train_base)

    val_loader = None
    val_dataset_size = 0
    if val_subjects:
        val_base = SpatialSRDataset(
            manifest_path=manifest_path,
            subject_filter=val_subjects,
            degrade_fn=degrade_fn,
            source_voxel_mm=float(config["source_voxel_mm"]),
            target_voxel_mm=float(config["target_voxel_mm"]),
        )
        val_dataset = SRSpatialDatasetAdapter(val_base)
        val_dataset_size = len(val_dataset)
    else:
        val_dataset = None

    deterministic = bool(config.get("deterministic", False))
    pin_memory = torch.cuda.is_available()
    worker_init_fn = None
    if deterministic:
        def _seed_worker(worker_id: int) -> None:
            worker_seed = int(config["seed"]) + worker_id
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _seed_worker

    train_generator = torch.Generator().manual_seed(int(config["seed"]) + 101)
    val_generator = torch.Generator().manual_seed(int(config["seed"]) + 202)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=val_generator,
        )

    dataset_size = len(train_dataset) + val_dataset_size
    return train_loader, val_loader, dataset_size
