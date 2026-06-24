"""Lazy volume reader for 4D BOLD files.

Wraps nibabel's lazy .dataobj indexing. Reading one volume out of a 300-volume
run does NOT decompress the other 299 — provided indexed_gzip is installed so
the .nii.gz can be accessed at random offsets. Without it, nibabel falls back
to sequential gzip and every random read decompresses from byte 0. See
requirements.txt.

Design notes:
  - nibabel image handles don't always survive process fork cleanly. When used
    inside a PyTorch DataLoader with num_workers>0, open the image inside
    __getitem__, not in __init__. This reader class is cheap to construct
    (doesn't decompress anything), so that's fine.
  - There is a per-process reader cache (`get_reader`) so multiple samples from
    the same run within a worker don't re-parse the gzip header. The cache is
    process-local and is naturally re-populated in each DataLoader worker
    after fork.
  - We read the native dtype (usually int16). Conversion to float happens
    downstream at normalization time.
"""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import DTypeLike


class VolumeReader:
    """Read one or more 3D volumes lazily from a 4D NIfTI file.

    Example:
        reader = VolumeReader("/path/to/sub-01_..._bold.nii.gz")
        vol_0 = reader.read_volume(0)           # shape (X, Y, Z)
        vols = reader.read_range(0, 10)         # shape (X, Y, Z, 10)
        mean = reader.read_mean()               # shape (X, Y, Z), float32

    For repeated access to the same file inside a DataLoader worker, prefer
    `get_reader(path)` which caches by path within the current process.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.img = nib.load(str(self.path))  # header only, no decompression
        shape = tuple(int(s) for s in self.img.shape)
        if len(shape) != 4:
            raise ValueError(f"Expected 4D NIfTI, got shape {shape} for {self.path}")
        self.shape: tuple[int, int, int, int] = shape
        self.shape3d: tuple[int, int, int] = shape[:3]
        self.n_volumes: int = shape[-1]

    def read_volume(self, t: int) -> np.ndarray:
        """Read a single 3D volume at timepoint t.

        Returns the array as nibabel produces it. That is usually the file's
        on-disk dtype (e.g., int16 for IBC), but if the NIfTI header has
        scl_slope != 1 or scl_inter != 0 nibabel applies the rescaling and
        returns float64. The dataset path explicitly casts to float32 before
        normalization, so the dtype variability does not propagate downstream.
        """
        if not (0 <= t < self.n_volumes):
            raise IndexError(f"t={t} out of range [0, {self.n_volumes})")
        # .dataobj supports numpy-style slicing without decompressing the full file
        # IF indexed_gzip is installed. See module docstring.
        return np.asarray(self.img.dataobj[..., t])

    def read_range(self, t_start: int, t_end: int) -> np.ndarray:
        """Read volumes [t_start, t_end). Returns 4D array (X, Y, Z, t_end-t_start).

        Use this for any contiguous span instead of looping read_volume — even
        with indexed_gzip, one range read is faster than N independent reads.

        Dtype follows the same rule as `read_volume`: native on-disk dtype
        normally, float64 if NIfTI rescaling slopes are set.
        """
        if not (0 <= t_start < t_end <= self.n_volumes):
            raise IndexError(
                f"Invalid range [{t_start}, {t_end}) for n_volumes={self.n_volumes}"
            )
        return np.asarray(self.img.dataobj[..., t_start:t_end])

    def read_full(self, dtype: DTypeLike = np.float32) -> np.ndarray:
        """Read the entire 4D run in one shot, cast to the given dtype.

        For a 128×128×93×262 run as float32 this is ~1.6 GB. Used by
        compute_metadata for offline mean/tSNR computation; not appropriate
        in a per-sample training path.
        """
        return np.asarray(self.img.dataobj, dtype=dtype)

    def read_mean(self) -> np.ndarray:
        """Compute the temporal mean of the entire run. Returns float32 (X, Y, Z).

        Reads everything; not free. Called once per run offline.

        Accumulator is float64: a float32 sum over ~300 BOLD timepoints can
        accumulate ~3 ULP × N drift, which the 98th-percentile norm_ref is
        robust to but the per-voxel tSNR isn't. Result is cast back to float32.
        """
        return self.read_full(dtype=np.float32).mean(axis=-1, dtype=np.float64).astype(np.float32)

    def __repr__(self) -> str:
        return f"VolumeReader(path={self.path.name}, shape={self.shape})"


# ---------------------------------------------------------------------------
# Per-process reader cache for use inside DataLoader workers.
#
# Why: opening a .nii.gz parses the gzip header and (with indexed_gzip) builds
# a seek index. Doing this once per __getitem__ means every sample pays the
# header cost. With this cache, a worker pays it once per (process, run).
#
# Why module-level: PyTorch DataLoader workers are forked AFTER the Dataset
# is constructed. A module-level dict in the parent gets copy-on-written into
# each child, but we never populate it in the parent (Datasets do not call
# get_reader in __init__). Workers populate their own copy lazily on first
# __getitem__. Cache is naturally per-worker.
#
# We key by (pid, resolved_path) so a stale entry from a forked-but-unused
# parent process can never match.
# ---------------------------------------------------------------------------

_READER_CACHE: dict[tuple[int, str], VolumeReader] = {}


def get_reader(path: str | Path) -> VolumeReader:
    """Return a process-cached VolumeReader for `path`. Constructs on first call."""
    key = (os.getpid(), str(Path(path).resolve()))
    reader = _READER_CACHE.get(key)
    if reader is None:
        reader = VolumeReader(path)
        _READER_CACHE[key] = reader
    return reader


def clear_reader_cache() -> None:
    """Drop all cached readers. Use only if you know what you're doing."""
    _READER_CACHE.clear()
