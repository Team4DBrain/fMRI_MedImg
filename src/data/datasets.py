"""PyTorch Dataset classes for fMRI training.

Architecture:
  - BaseFMRIDataset handles the common stuff: manifest, reader, mask loading,
    normalization, sample indexing, and padding to target_shape.
  - Subclasses (DenoisingDataset, SpatialSRDataset, TemporalSRDataset) override
    __getitem__ to return the right (input, target) pair for their model.
  - Degradation functions are STUBBED. They raise NotImplementedError until we
    implement them with each model's owner. (Spatial degradation IS implemented
    in degradation_spatial.py — pass make_spatial_degradation() as degrade_fn.)

Padding: all volumes are center-padded to manifest["target_shape"] (default
128×128×93). This makes batching possible across runs of different shapes
and keeps the brain at a roughly consistent absolute position in the grid.

For SpatialSRDataset (Option A), the input is at LR shape and the target at HR
shape — the model is responsible for upsampling.

Important: nibabel image handles don't fork cleanly. We open VolumeReader
inside __getitem__, not __init__. This costs a few ms per sample but avoids
subtle bugs with num_workers>0.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation_spatial import (
    downsample_mask_to_lr,
    voxel_size_to_target_shape,
)
from .normalize import normalize
from .padding import center_pad_volume
from .reader import VolumeReader


class BaseFMRIDataset(Dataset):
    """Shared behavior for all three task datasets.

    Subclasses must implement:
        - _build_sample_index() -> list of sample descriptors
        - __getitem__(idx) -> dict with at least "input", "target", "mask"
    """

    def __init__(
        self,
        manifest_path: str | Path,
        subject_filter: list[str] | None = None,
    ):
        """
        Args:
            manifest_path: path to JSON manifest produced by compute_metadata.py.
            subject_filter: if given, keep only runs whose subject is in this list.
                Use this for train/val/test splits by subject.
        """
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open() as f:
            manifest = json.load(f)

        self.bids_root = Path(manifest["bids_root"])
        self.derivatives_dir = Path(manifest.get("derivatives_dir", ""))
        if not self.derivatives_dir.is_dir():
            raise RuntimeError(
                f"derivatives_dir missing or invalid in manifest: {self.derivatives_dir}. "
                "Did you run compute_metadata.py?"
            )

        if "target_shape" not in manifest:
            raise RuntimeError(
                "manifest is missing 'target_shape' field. Re-run compute_metadata.py."
            )
        self.target_shape: tuple[int, int, int] = tuple(manifest["target_shape"])

        # Keep only runs that have complete metadata.
        runs = manifest["runs"]
        runs = [r for r in runs if "norm_ref" in r and "mask_path" in r]

        if subject_filter is not None:
            wanted = set(subject_filter)
            runs = [r for r in runs if r["subject"] in wanted]

        if not runs:
            raise RuntimeError(
                f"No runs matched after filtering. subject_filter={subject_filter}"
            )

        self.runs = runs
        # Cache HR (padded) masks by run_id. ~170 KB each at 128×128×93 bool.
        self._mask_cache: dict[str, np.ndarray] = {}

        self.samples = self._build_sample_index()

    def _build_sample_index(self) -> list:
        """Default: one sample per (run, timepoint). Subclasses can override."""
        samples = []
        for run_idx, run in enumerate(self.runs):
            for t in range(run["n_volumes"]):
                samples.append((run_idx, t))
        return samples

    def _get_hr_mask(self, run_idx: int) -> np.ndarray:
        """Load (and cache) the HR brain mask. Already at target_shape."""
        run = self.runs[run_idx]
        run_id = run["run_id"]
        if run_id not in self._mask_cache:
            mask_path = self.derivatives_dir / run["mask_path"]
            mask = np.asarray(nib.load(str(mask_path)).dataobj).astype(bool)
            if mask.shape != self.target_shape:
                raise RuntimeError(
                    f"Mask for {run_id} has shape {mask.shape}, expected "
                    f"target_shape {self.target_shape}. Re-run compute_metadata.py "
                    f"with the same target_shape."
                )
            self._mask_cache[run_id] = mask
        return self._mask_cache[run_id]

    def _read_and_normalize(self, run_idx: int, t: int) -> np.ndarray:
        """Read volume at (run_idx, t), normalize, pad to target_shape, return float32.

        Returns shape == target_shape, regardless of source volume's native shape.
        """
        run = self.runs[run_idx]
        path = self.bids_root / run["path"]
        # Open reader here, not in __init__: fork-safety for DataLoader workers.
        reader = VolumeReader(path)
        vol = reader.read_volume(t).astype(np.float32)
        vol = normalize(vol, run["norm_ref"])
        vol = center_pad_volume(vol, self.target_shape, pad_value=0.0)
        return vol

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("Subclasses must implement __getitem__")


# ---------------------------------------------------------------------------
# Task-specific datasets
# ---------------------------------------------------------------------------


def _not_implemented_degradation(*args, **kwargs):
    raise NotImplementedError(
        "Degradation function not yet implemented. "
        "Decide on the degradation model with your teammates first."
    )


class DenoisingDataset(BaseFMRIDataset):
    """Samples for the denoising model.

    Returns dict with:
        input:  noisy volume,  shape (1, X, Y, Z) at target_shape
        target: clean volume,  shape (1, X, Y, Z) at target_shape
        mask:   HR brain mask, shape (1, X, Y, Z) at target_shape
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        degrade_fn=_not_implemented_degradation,
    ):
        super().__init__(manifest_path, subject_filter=subject_filter)
        self.degrade_fn = degrade_fn

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        clean = self._read_and_normalize(run_idx, t)  # (X, Y, Z) padded
        mask = self._get_hr_mask(run_idx)             # (X, Y, Z) padded
        noisy = self.degrade_fn(clean)                # same shape

        return {
            "input": torch.from_numpy(noisy).unsqueeze(0),
            "target": torch.from_numpy(clean).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }


class SpatialSRDataset(BaseFMRIDataset):
    """Samples for the spatial SR model — Option A.

    Input is at LR shape (e.g., 64×64×46 for 3mm), target is at HR shape
    (e.g., 128×128×93 for 1.5mm). The model is responsible for upsampling.

    Returns dict with:
        input:    LR volume,    shape (1, kx, ky, kz)
        target:   HR volume,    shape (1, X, Y, Z)
        mask_hr:  HR mask,      shape (1, X, Y, Z) — for HR-domain loss/eval
        mask_lr:  LR mask,      shape (1, kx, ky, kz) — for LR-domain logic if needed
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        degrade_fn=_not_implemented_degradation,
        source_voxel_mm: float = 1.5,
        target_voxel_mm: float = 3.0,
    ):
        super().__init__(manifest_path, subject_filter=subject_filter)
        self.degrade_fn = degrade_fn
        self.source_voxel_mm = source_voxel_mm
        self.target_voxel_mm = target_voxel_mm

        # Precompute LR shape from target_shape (HR) and voxel ratio.
        self.lr_shape: tuple[int, int, int] = voxel_size_to_target_shape(
            self.target_shape, source_voxel_mm, target_voxel_mm,
        )

        # Cache LR masks separately (derived from HR masks).
        self._lr_mask_cache: dict[str, np.ndarray] = {}

    def _get_lr_mask(self, run_idx: int) -> np.ndarray:
        """Derive (and cache) the LR brain mask from the HR mask."""
        run = self.runs[run_idx]
        run_id = run["run_id"]
        if run_id not in self._lr_mask_cache:
            hr_mask = self._get_hr_mask(run_idx)
            lr_mask = downsample_mask_to_lr(hr_mask, self.lr_shape)
            self._lr_mask_cache[run_id] = lr_mask
        return self._lr_mask_cache[run_id]

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        hr = self._read_and_normalize(run_idx, t)   # padded (X, Y, Z)
        hr_mask = self._get_hr_mask(run_idx)        # padded (X, Y, Z)
        lr = self.degrade_fn(hr)                    # (kx, ky, kz)
        lr_mask = self._get_lr_mask(run_idx)        # (kx, ky, kz)

        # Sanity check: LR shape from degrade_fn must match what the Dataset expects.
        # If this fails, the degrade_fn config is inconsistent with the Dataset config.
        assert lr.shape == self.lr_shape, (
            f"degrade_fn returned shape {lr.shape}, expected {self.lr_shape}"
        )

        return {
            "input": torch.from_numpy(lr).unsqueeze(0),
            "target": torch.from_numpy(hr).unsqueeze(0),
            "mask_hr": torch.from_numpy(hr_mask).unsqueeze(0),
            "mask_lr": torch.from_numpy(lr_mask).unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }


class TemporalSRDataset(BaseFMRIDataset):
    """Samples for the temporal SR model.

    The "degradation" for this task is implicit: by sampling neighbors at t-gap
    and t+gap and predicting the middle volume at t, we simulate a half-rate
    acquisition where t was never measured. There is no separate degrade_fn.

    Returns dict with:
        input:  two neighbors stacked along channel, shape (2, X, Y, Z)
        target: middle volume,                       shape (1, X, Y, Z)
        mask:   HR brain mask,                       shape (1, X, Y, Z)
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        gap: int = 1,
    ):
        """
        Args:
            gap: distance in timepoints between target and each neighbor input.
                gap=1 means use t-1 and t+1 to predict t (simulates 2x
                temporal downsampling).
        """
        self.gap = gap
        super().__init__(manifest_path, subject_filter=subject_filter)

    def _build_sample_index(self) -> list:
        """Only valid samples where both neighbors exist inside the run."""
        samples = []
        for run_idx, run in enumerate(self.runs):
            n = run["n_volumes"]
            for t in range(self.gap, n - self.gap):
                samples.append((run_idx, t))
        return samples

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        target = self._read_and_normalize(run_idx, t)
        before = self._read_and_normalize(run_idx, t - self.gap)
        after = self._read_and_normalize(run_idx, t + self.gap)
        mask = self._get_hr_mask(run_idx)

        input_tensor = np.stack([before, after], axis=0)  # (2, X, Y, Z)

        return {
            "input": torch.from_numpy(input_tensor),
            "target": torch.from_numpy(target).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }
