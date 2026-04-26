"""Lazy volume reader for 4D BOLD files.

Wraps nibabel's lazy .dataobj indexing. Reading one volume out of a 300-volume
run does NOT decompress the other 299.

Design notes:
  - nibabel image handles don't always survive process fork cleanly. When used
    inside a PyTorch DataLoader with num_workers>0, open the image inside
    __getitem__, not in __init__. This reader class is cheap to construct
    (doesn't decompress anything), so that's fine.
  - We cast to int16 on read where possible (IBC data is natively int16).
    Conversion to float happens downstream at normalization time.
"""

from __future__ import annotations

from pathlib import Path
import nibabel as nib
import numpy as np


class VolumeReader:
    """Read one or more 3D volumes lazily from a 4D NIfTI file.

    Example:
        reader = VolumeReader("/path/to/sub-01_..._bold.nii.gz")
        vol_0 = reader.read_volume(0)           # shape (X, Y, Z)
        vols_slice = reader.read_range(0, 10)   # shape (X, Y, Z, 10)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.img = nib.load(str(self.path))  # header only, no decompression
        self.shape = tuple(int(s) for s in self.img.shape)
        if len(self.shape) != 4:
            raise ValueError(f"Expected 4D NIfTI, got shape {self.shape} for {self.path}")
        self.n_volumes = self.shape[-1]

    def read_volume(self, t: int) -> np.ndarray:
        """Read a single 3D volume at timepoint t. Returns array of native dtype (usually int16)."""
        if not (0 <= t < self.n_volumes):
            raise IndexError(f"t={t} out of range [0, {self.n_volumes})")
        # .dataobj supports numpy-style slicing without decompressing the full file.
        return np.asarray(self.img.dataobj[..., t])

    def read_range(self, t_start: int, t_end: int) -> np.ndarray:
        """Read volumes [t_start, t_end). Returns 4D array (X, Y, Z, t_end-t_start)."""
        if not (0 <= t_start < t_end <= self.n_volumes):
            raise IndexError(
                f"Invalid range [{t_start}, {t_end}) for n_volumes={self.n_volumes}"
            )
        return np.asarray(self.img.dataobj[..., t_start:t_end])

    def read_mean(self) -> np.ndarray:
        """Compute the temporal mean of the entire run. Reads everything — slow for big runs.

        Returned as float32
        """
        # Load full data once. For a 262-volume IBC run this is ~860 MB as int16,
        # ~3.4 GB if we upcast to float64. We explicitly ask for float32 to keep it reasonable.
        data = np.asarray(self.img.dataobj, dtype=np.float32)
        return data.mean(axis=-1)

    def __repr__(self) -> str:
        return f"VolumeReader(path={self.path.name}, shape={self.shape})"
