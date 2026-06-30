"""Lazy fMRI triplet dataset for temporal interpolation.

Reads 4D BOLD NIfTI files (X, Y, Z, T) and turns every valid time index t into
one training example:

    input  x = [V_t, V_{t+2}]   (2, D, H, W)
    target y = V_{t+1}           (1, D, H, W)

Volumes are sliced lazily through nibabel.dataobj — nothing is preprocessed to
disk, which matters because local disk is tight.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class FMRIInterpolationDataset(Dataset):
    """Stream (V_t, V_{t+1}, V_{t+2}) triplets from one or many BOLD files."""

    def __init__(
        self,
        root: str | None = None,
        norm_mode: str = "zscore",
        file_list: list[str] | None = None,
    ):
        """Index every valid triplet without loading volume data.

        Pass `file_list` for explicit files (local demos/smoke tests), or `root`
        to scan a BIDS tree. norm_mode is "zscore" or "percentile".
        """
        super().__init__()

        if norm_mode not in {"zscore", "percentile"}:
            raise ValueError("norm_mode must be 'zscore' or 'percentile'")
        if file_list is None and root is None:
            raise ValueError("Either root or file_list must be provided")

        self.root = root
        self.norm_mode = norm_mode

        if file_list is None:
            files = sorted(str(p) for p in Path(root).rglob("*_bold.nii.gz"))
        else:
            files = [str(p) for p in file_list]

        if not files:
            raise FileNotFoundError("No *_bold.nii.gz files found")

        self.files = files

        # Flat sample index: dataset idx -> (file idx, time index t).
        self._index: list[tuple[int, int]] = []
        # file idx -> its sample indices, used by split_by_file().
        self._file_to_indices: dict[int, list[int]] = {}
        # Per-process nibabel image cache, avoids reopening the .gz repeatedly.
        self._proxy_cache: dict[str, nib.spatialimages.SpatialImage] = {}

        # Headers only here — no array data is read.
        for file_idx, path in enumerate(self.files):
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            img = nib.load(path)
            if len(img.shape) != 4:
                raise ValueError(f"Expected 4D BOLD file, got shape {img.shape}: {path}")

            t_len = int(img.shape[-1])
            # Need V_t, V_{t+1}, V_{t+2}.
            if t_len < 3:
                continue

            self._file_to_indices[file_idx] = []
            for t in range(t_len - 2):
                sample_idx = len(self._index)
                self._index.append((file_idx, t))
                self._file_to_indices[file_idx].append(sample_idx)

        if not self._index:
            raise ValueError("No files have at least 3 time points")

    def __len__(self) -> int:
        return len(self._index)

    def _load_img(self, path: str):
        img = self._proxy_cache.get(path)
        if img is None:
            img = nib.load(path)
            self._proxy_cache[path] = img
        return img

    @staticmethod
    def _stats_pool(v_t: np.ndarray, v_t2: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Voxels used for the per-triplet normalization stats."""
        # Prefer masked voxels, but fall back to everything if the mask came
        # out empty/tiny on an unusual file.
        if int(mask.sum()) >= 100:
            return np.concatenate([v_t[mask], v_t2[mask]]).astype(np.float32, copy=False)
        return np.concatenate([v_t.ravel(), v_t2.ravel()]).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict:
        """Load and normalize one sample.

        Returns x (2,D,H,W), y (1,D,H,W), mask (1,D,H,W), stats (mu, scale) for
        denormalization, plus the source path and time index t.
        """
        file_idx, t = self._index[idx]
        path = self.files[file_idx]
        img = self._load_img(path)

        # Pull only the three adjacent volumes off disk.
        arr = np.asarray(img.dataobj[..., t:t + 3], dtype=np.float32)
        v_t = arr[..., 0]
        v_t1 = arr[..., 1]
        v_t2 = arr[..., 2]

        # Cheap signal mask from the inputs only — never touch the target, or
        # normalization stats would leak from it.
        mask_source = np.abs(v_t) + np.abs(v_t2)
        threshold = np.percentile(mask_source, 20)
        mask = mask_source > threshold

        pool = self._stats_pool(v_t, v_t2, mask)

        if self.norm_mode == "zscore":
            mu = float(pool.mean())
            sigma = float(max(pool.std(), 1e-6))
            v_t = (v_t - mu) / sigma
            v_t1 = (v_t1 - mu) / sigma
            v_t2 = (v_t2 - mu) / sigma
        else:
            # Percentile mode: robust clip to [0, 1].
            lo = float(np.percentile(pool, 1))
            hi = float(np.percentile(pool, 99))
            sigma = max(hi - lo, 1e-6)
            mu = lo
            v_t = np.clip((v_t - lo) / sigma, 0.0, 1.0)
            v_t1 = np.clip((v_t1 - lo) / sigma, 0.0, 1.0)
            v_t2 = np.clip((v_t2 - lo) / sigma, 0.0, 1.0)

        # NIfTI (X, Y, Z) -> torch (D, H, W).
        tensors = [
            torch.from_numpy(np.ascontiguousarray(v.transpose(2, 1, 0))).float()
            for v in (v_t, v_t1, v_t2)
        ]
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask.transpose(2, 1, 0).astype(np.float32))
        ).unsqueeze(0)

        x = torch.stack([tensors[0], tensors[2]], dim=0)
        y = tensors[1].unsqueeze(0)

        return {
            "x": x,
            "y": y,
            "mask": mask_tensor,
            "stats": (mu, sigma),
            "path": path,
            "t": t,
        }


def split_by_file(
    dataset: FMRIInterpolationDataset,
    val_fraction: float = 0.1,
    seed: int = 0,
):
    """Split at the file level so adjacent timepoints from one run never end up
    in both train and val.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    file_indices = np.arange(len(dataset.files))
    rng.shuffle(file_indices)

    n_val = max(1, int(round(len(file_indices) * val_fraction)))
    # Always leave at least one file for training.
    if len(file_indices) > 1:
        n_val = min(n_val, len(file_indices) - 1)
    val_files = set(int(i) for i in file_indices[:n_val])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for file_idx, sample_indices in dataset._file_to_indices.items():
        if file_idx in val_files:
            val_indices.extend(sample_indices)
        else:
            train_indices.extend(sample_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
