"""dataset.py — lazy fMRI triplet dataset for temporal interpolation.

The dataset reads 4D BOLD NIfTI files shaped like:

    (X, Y, Z, T) = (128, 128, 84, time)

For every valid time index t, it builds one supervised training example:

    input  x = [V_t, V_{t+2}]      shape (2, D, H, W)
    target y = V_{t+1}             shape (1, D, H, W)

Nothing is preprocessed to disk. NIfTI volumes are sliced lazily through
`nibabel.dataobj`, which is important because local disk is limited.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class FMRIInterpolationDataset(Dataset):
    """Stream fMRI interpolation triplets from one or many BOLD files."""

    def __init__(
        self,
        root: str | None = None,
        norm_mode: str = "zscore",
        file_list: list[str] | None = None,
    ):
        """Index all available `(V_t, V_{t+1}, V_{t+2})` triplets.

        Args:
            root: BIDS-style root directory. Used when `file_list` is absent.
            norm_mode: `"zscore"` or `"percentile"`.
            file_list: Explicit BOLD files. Useful for local demos/smoke tests.
        """
        super().__init__()

        # Normalization is intentionally limited to the two modes in plan.md.
        if norm_mode not in {"zscore", "percentile"}:
            raise ValueError("norm_mode must be 'zscore' or 'percentile'")

        # The dataset needs either a root to scan or an explicit file list.
        if file_list is None and root is None:
            raise ValueError("Either root or file_list must be provided")

        # Keep settings as attributes so checkpoints/notebooks can inspect them.
        self.root = root
        self.norm_mode = norm_mode

        # Phase B: recursively find all BOLD files under a BIDS root.
        if file_list is None:
            files = sorted(str(p) for p in Path(root).rglob("*_bold.nii.gz"))

        # Phase A: use exactly the file(s) the user passed.
        else:
            files = [str(p) for p in file_list]

        # Fail early if the path selection found nothing.
        if not files:
            raise FileNotFoundError("No *_bold.nii.gz files found")

        # Public list of indexed BOLD files.
        self.files = files

        # Flat sample index: dataset index -> (file index, time index t).
        self._index: list[tuple[int, int]] = []

        # Reverse map used by split_by_file() to avoid temporal leakage.
        self._file_to_indices: dict[int, list[int]] = {}

        # Per-process cache of nibabel image objects. This avoids reopening .gz.
        self._proxy_cache: dict[str, nib.spatialimages.SpatialImage] = {}

        # Build the sample index by reading only headers, not full arrays.
        for file_idx, path in enumerate(self.files):
            # A bad path should be reported before training begins.
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            # nib.load reads the header and creates a lazy data proxy.
            img = nib.load(path)

            # This project expects functional BOLD files with time as axis 4.
            if len(img.shape) != 4:
                raise ValueError(f"Expected 4D BOLD file, got shape {img.shape}: {path}")

            # Number of timepoints in the BOLD run.
            t_len = int(img.shape[-1])

            # Need at least V_t, V_{t+1}, and V_{t+2}.
            if t_len < 3:
                continue

            # Store all sample indices belonging to this file.
            self._file_to_indices[file_idx] = []

            # Each t creates one triplet: t, t+1, t+2.
            for t in range(t_len - 2):
                sample_idx = len(self._index)
                self._index.append((file_idx, t))
                self._file_to_indices[file_idx].append(sample_idx)

        # If every file was too short, training cannot proceed.
        if not self._index:
            raise ValueError("No files have at least 3 time points")

    def __len__(self) -> int:
        """Return the number of available triplets."""
        return len(self._index)

    def _load_img(self, path: str):
        """Return a cached nibabel image for a BOLD file."""
        # Check whether this process already opened the NIfTI file.
        img = self._proxy_cache.get(path)

        # First access opens the file and stores the lazy proxy.
        if img is None:
            img = nib.load(path)
            self._proxy_cache[path] = img

        return img

    @staticmethod
    def _stats_pool(v_t: np.ndarray, v_t2: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Choose voxels used to compute per-triplet normalization stats."""
        # Use masked voxels when the lightweight mask has enough signal voxels.
        if int(mask.sum()) >= 100:
            return np.concatenate([v_t[mask], v_t2[mask]]).astype(np.float32, copy=False)

        # Fallback protects against unusual files where the mask is empty/tiny.
        return np.concatenate([v_t.ravel(), v_t2.ravel()]).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict:
        """Load and normalize one interpolation training sample.

        Returns:
            A dictionary with:
                x:    input tensor,  shape (2, D, H, W)
                y:    target tensor, shape (1, D, H, W)
                mask: brain-ish mask, shape (1, D, H, W)
                stats: `(mu, sigma_or_scale)` for denormalization
                path: source file path
                t:    source time index
        """
        # Resolve the flat dataset index into a file and timepoint.
        file_idx, t = self._index[idx]
        path = self.files[file_idx]

        # Get the lazy NIfTI proxy for this file.
        img = self._load_img(path)

        # Slice only three adjacent volumes from disk/proxy: V_t, V_t+1, V_t+2.
        arr = np.asarray(img.dataobj[..., t:t + 3], dtype=np.float32)

        # Split the three volumes while they are still in NIfTI axis order.
        v_t = arr[..., 0]
        v_t1 = arr[..., 1]
        v_t2 = arr[..., 2]

        # Build a lightweight signal mask from inputs only. Target is not used.
        mask_source = np.abs(v_t) + np.abs(v_t2)

        # The 20th percentile removes mostly-background voxels cheaply.
        threshold = np.percentile(mask_source, 20)
        mask = mask_source > threshold

        # Use the same voxel pool for z-score or percentile stats.
        pool = self._stats_pool(v_t, v_t2, mask)

        if self.norm_mode == "zscore":
            # Compute input-only mean and std; no target leakage.
            mu = float(pool.mean())
            sigma = float(max(pool.std(), 1e-6))

            # Apply the same affine transform to inputs and target.
            v_t = (v_t - mu) / sigma
            v_t1 = (v_t1 - mu) / sigma
            v_t2 = (v_t2 - mu) / sigma

        else:
            # Percentile mode clips robustly and scales to [0, 1].
            lo = float(np.percentile(pool, 1))
            hi = float(np.percentile(pool, 99))
            sigma = max(hi - lo, 1e-6)
            mu = lo

            # Same lower/upper bounds are applied to all three volumes.
            v_t = np.clip((v_t - lo) / sigma, 0.0, 1.0)
            v_t1 = np.clip((v_t1 - lo) / sigma, 0.0, 1.0)
            v_t2 = np.clip((v_t2 - lo) / sigma, 0.0, 1.0)

        # Convert NIfTI axis order (X, Y, Z) to PyTorch order (D, H, W).
        tensors = [
            torch.from_numpy(np.ascontiguousarray(v.transpose(2, 1, 0))).float()
            for v in (v_t, v_t1, v_t2)
        ]

        # Mask gets the same axis permutation and an explicit channel dimension.
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask.transpose(2, 1, 0).astype(np.float32))
        ).unsqueeze(0)

        # Input channels are the two neighboring frames.
        x = torch.stack([tensors[0], tensors[2]], dim=0)

        # Target has one channel: the true middle frame.
        y = tensors[1].unsqueeze(0)

        # Keep path/time/stats for visualization and denormalization.
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
    """Split a dataset at file level, not triplet level.

    This avoids leakage where adjacent timepoints from the same BOLD run appear
    in both training and validation.
    """
    # Validate the requested validation size.
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    # Shuffle file indices reproducibly.
    rng = np.random.default_rng(seed)
    file_indices = np.arange(len(dataset.files))
    rng.shuffle(file_indices)

    # At least one file should go to validation when possible.
    n_val = max(1, int(round(len(file_indices) * val_fraction)))

    # If there are multiple files, keep at least one file for training.
    if len(file_indices) > 1:
        n_val = min(n_val, len(file_indices) - 1)

    # Validation file ids.
    val_files = set(int(i) for i in file_indices[:n_val])

    # Convert file ids back into flat dataset sample indices.
    train_indices: list[int] = []
    val_indices: list[int] = []
    for file_idx, sample_indices in dataset._file_to_indices.items():
        if file_idx in val_files:
            val_indices.extend(sample_indices)
        else:
            train_indices.extend(sample_indices)

    # Return standard PyTorch Subset objects.
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
