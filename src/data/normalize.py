"""Intensity normalization for fMRI volumes.

We use per-run scalar normalization:
    normalized = volume / norm_ref
where norm_ref is the 98th percentile of brain-masked voxels from the run's
temporal mean. This:
  - Keeps all runs on a consistent scale for the model.
  - Preserves spatial contrast (unlike per-voxel z-scoring).
  - Preserves temporal BOLD dynamics.
  - Is trivially reversible.

The `norm_ref` is computed once per run (offline, in compute_metadata.py) and
stored in the manifest. At training time, we just divide.
"""

from __future__ import annotations

import numpy as np


def compute_norm_ref(
    mean_volume: np.ndarray,
    mask: np.ndarray,
    percentile: float = 98.0,
) -> float:
    """Compute the scalar normalization reference from a mean volume and brain mask.

    Uses a high percentile (not max) for robustness against bright outlier voxels
    (vasculature, motion spikes). 98 gives a stable "typical bright brain voxel"
    reference across runs.

    Args:
        mean_volume: temporal mean of a BOLD run, shape (X, Y, Z).
        mask: boolean brain mask, same shape.
        percentile: which percentile of in-brain voxels to use. 98 is robust.

    Returns:
        A positive scalar. Raises if mask is empty or reference is non-positive.
    """
    if mean_volume.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: volume {mean_volume.shape} vs mask {mask.shape}"
        )
    brain_voxels = mean_volume[mask]
    if brain_voxels.size == 0:
        raise ValueError("Empty brain mask — can't compute norm_ref")

    ref = float(np.percentile(brain_voxels, percentile))
    if ref <= 0:
        raise ValueError(f"Computed norm_ref={ref} is non-positive; data looks wrong")
    return ref


def normalize(volume: np.ndarray, norm_ref: float) -> np.ndarray:
    """Scale a volume by its run's norm_ref. Non-destructive."""
    if norm_ref <= 0:
        raise ValueError(f"norm_ref must be positive, got {norm_ref}")
    return volume / norm_ref


def denormalize(normalized: np.ndarray, norm_ref: float) -> np.ndarray:
    """Invert normalize(). Useful for visualization and evaluation in original units."""
    return normalized * norm_ref
