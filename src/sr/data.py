"""Data loading for SR training on the current manifest format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.degradation_spatial import make_spatial_degradation
from src.data.normalize import normalize
from src.data.reader import get_reader


class SRSpatialManifestDataset(Dataset):
    """SR dataset for the current manifest/data pipeline."""

    def __init__(
        self,
        manifest_path: Path,
        subject_filter: list[str] | None,
        degrade_fn,
    ):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open(encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)

        self.bids_root = Path(manifest["bids_root"])
        self.target_shape = tuple(manifest.get("target_shape", []))
        if not self.target_shape:
            raise RuntimeError("manifest is missing target_shape")

        self.degrade_fn = degrade_fn

        runs = [
            run for run in manifest.get("runs", [])
            if "subject" in run and "path" in run and "n_volumes" in run and "norm_ref" in run
        ]
        if subject_filter is not None:
            wanted = set(subject_filter)
            runs = [run for run in runs if str(run["subject"]) in wanted]

        filtered_runs: list[dict] = []
        dropped = 0
        for run in runs:
            run_shape = tuple(run.get("shape", [])[:3])
            if run_shape and run_shape != self.target_shape:
                dropped += 1
                continue
            filtered_runs.append(run)

        if not filtered_runs:
            raise RuntimeError(
                "No usable runs in manifest after filtering for SR loader compatibility."
            )
        if dropped:
            print(f"[sr.data] Dropped {dropped} runs not compatible with target shape {self.target_shape}.")

        self.runs = filtered_runs
        self.samples = [
            (run_idx, t)
            for run_idx, run in enumerate(self.runs)
            for t in range(int(run["n_volumes"]))
        ]
        if not self.samples:
            raise RuntimeError("No samples available in SRSpatialManifestDataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        run_idx, t = self.samples[idx]
        run = self.runs[run_idx]
        reader = get_reader(self.bids_root / run["path"])
        hr = reader.read_volume(int(t)).astype(np.float32)
        hr = normalize(hr, run["norm_ref"])

        if tuple(hr.shape) != self.target_shape:
            raise RuntimeError(
                f"Run {run.get('run_id', run_idx)} has sample shape {hr.shape}, expected {self.target_shape}."
            )

        lr = self.degrade_fn(hr)
        return (
            torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0).float(),
            torch.from_numpy(np.ascontiguousarray(hr)).unsqueeze(0).float(),
        )


def _available_subjects(manifest_path: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    runs = manifest.get("runs", [])
    subjects = {
        str(run["subject"])
        for run in runs
        if "subject" in run and "norm_ref" in run and "path" in run
    }
    ordered = sorted(subjects)
    if not ordered:
        raise RuntimeError("Manifest contains no usable runs with subject, norm_ref, and path.")
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

    train_dataset = SRSpatialManifestDataset(
        manifest_path=manifest_path,
        subject_filter=train_subjects,
        degrade_fn=degrade_fn,
    )

    val_loader = None
    val_dataset_size = 0
    if val_subjects:
        val_dataset = SRSpatialManifestDataset(
            manifest_path=manifest_path,
            subject_filter=val_subjects,
            degrade_fn=degrade_fn,
        )
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
