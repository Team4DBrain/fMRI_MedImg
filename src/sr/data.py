"""Train/val DataLoader construction for spatial SR.

Purpose:
    Turn an ``SRConfig`` + a manifest on disk into reproducible train and
    validation DataLoaders. Subject-level splitting is the default so the
    val set is never the same patients as the train set.
Effects:
    Determines which volumes a model sees, in what order, with what
    degradation. Seeding policy here is what makes runs (and resumes)
    deterministic.
Influences:
    Behaviour depends on ``manifest_path``, ``train_split``,
    ``train_subjects``/``val_subjects`` (explicit override) and
    ``source_voxel_mm``/``target_voxel_mm`` (degradation).
How to change safely:
    Keep ``build_loaders`` returning ``(train_loader, val_loader|None,
    split_info)`` -- the training loop and ``split.json`` writer both rely
    on that shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation
from src.sr.config import SRConfig


def _available_subjects(manifest_path: Path) -> list[str]:
    """Return all unique subjects in the manifest in deterministic order."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    subjects = {str(run["subject"]) for run in manifest.get("runs", []) if "subject" in run}
    if not subjects:
        raise RuntimeError(f"Manifest at {manifest_path} contains no subjects.")
    return sorted(subjects)


def resolve_subject_split(
    config: SRConfig,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Decide which subjects go into train vs. val.

    Resolution order (first match wins):
        1. Explicit ``train_subjects`` AND ``val_subjects`` from config.
        2. ``train_split == 1.0`` -> everything trains, no validation.
        3. Seeded shuffle of all manifest subjects, split at
           ``int(len * train_split)`` (clamped to keep both sides non-empty).
    """
    if config.train_subjects is not None and config.val_subjects is not None:
        train = [str(s) for s in config.train_subjects]
        val = [str(s) for s in config.val_subjects]
        if not train:
            raise ValueError("Explicit train_subjects must be non-empty.")
        return train, val, {"source": "explicit"}

    subjects = _available_subjects(config.manifest_path)
    if config.train_split == 1.0:
        return subjects, [], {"source": "all_train"}

    if len(subjects) < 2:
        raise ValueError(
            f"Need at least 2 subjects for a train/val split (manifest has "
            f"{len(subjects)}). Either pass --train-split 1.0 to disable "
            "validation or extend the manifest."
        )

    rng = np.random.default_rng(int(config.seed))
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * float(config.train_split))
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    return shuffled[:split_idx], shuffled[split_idx:], {"source": "random_seeded"}


def _seed_worker(seed: int) -> Any:
    """Return a ``worker_init_fn`` closure with the given base seed.

    PyTorch DataLoader workers each fork once and stay alive for the run;
    seeding them by ``(seed + worker_id)`` keeps batches reproducible across
    process boundaries without depending on PyTorch's own automatic seeding.
    """

    def _init(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


def _make_loader(
    dataset: SpatialSRDataset,
    config: SRConfig,
    *,
    shuffle: bool,
    generator_seed: int,
) -> DataLoader:
    """Build one DataLoader with seeded shuffling and seeded workers."""
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(generator_seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker(config.seed),
        generator=generator,
    )


def build_loaders(
    config: SRConfig,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    """Build train + (optional) val loaders and a split-info dict.

    Returned ``split_info`` is what gets serialised to ``split.json`` once
    at the start of a run, so users can inspect the split without loading
    a checkpoint.
    """
    degrade_fn = make_spatial_degradation(
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
    )
    train_subjects, val_subjects, split_meta = resolve_subject_split(config)

    train_dataset = SpatialSRDataset(
        manifest_path=Path(config.manifest_path),
        subject_filter=train_subjects,
        degrade_fn=degrade_fn,
    )

    val_dataset: SpatialSRDataset | None = None
    if val_subjects:
        val_dataset = SpatialSRDataset(
            manifest_path=Path(config.manifest_path),
            subject_filter=val_subjects,
            degrade_fn=degrade_fn,
        )

    # Distinct generator seeds keep train and val shuffling independent.
    train_loader = _make_loader(
        train_dataset, config, shuffle=True, generator_seed=config.seed + 101
    )
    val_loader: DataLoader | None
    if val_dataset is not None:
        val_loader = _make_loader(
            val_dataset, config, shuffle=False, generator_seed=config.seed + 202
        )
    else:
        val_loader = None

    split_info = {
        "source": split_meta["source"],
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
    }
    return train_loader, val_loader, split_info


def write_split_json(run_dir: Path, split_info: dict[str, Any]) -> None:
    """Persist ``split_info`` to ``run_dir/split.json`` atomically."""
    path = Path(run_dir) / "split.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    tmp.replace(path)
