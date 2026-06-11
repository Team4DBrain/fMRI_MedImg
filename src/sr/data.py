"""Train/val DataLoader construction for spatial SR.

Purpose:
    Turn an ``SRConfig`` + a manifest on disk into reproducible train and
    validation DataLoaders using a seeded random split across the full
    dataset sample index.
Effects:
    Determines which volumes/timepoints a model sees, in what order, with
    what degradation. Seeding policy here is what makes runs (and resumes)
    deterministic.
Influences:
    Behaviour depends on ``manifest_path``, ``train_split`` and
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
from torch.utils.data import DataLoader, Subset

from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation
from src.sr.config import SRConfig


def resolve_sample_split(
    n_samples: int, config: SRConfig
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Decide which sample indices go into train vs. val."""
    if n_samples < 1:
        raise ValueError("Dataset is empty after manifest filtering.")

    if config.train_split == 1.0:
        return list(range(n_samples)), [], {"source": "all_train"}

    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 samples for a train/val split (dataset has "
            f"{n_samples}). Either pass --train-split 1.0 to disable "
            "validation or extend the manifest."
        )

    rng = np.random.default_rng(int(config.seed))
    shuffled = list(range(n_samples))
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
    full_dataset = SpatialSRDataset(
        manifest_path=Path(config.manifest_path),
        degrade_fn=degrade_fn,
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
    )
    train_indices, val_indices, split_meta = resolve_sample_split(
        len(full_dataset), config
    )
    train_dataset = Subset(full_dataset, train_indices)

    val_dataset: Subset[SpatialSRDataset] | None = None
    if val_indices:
        val_dataset = Subset(full_dataset, val_indices)

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
        "train_indices": train_indices,
        "val_indices": val_indices,
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
