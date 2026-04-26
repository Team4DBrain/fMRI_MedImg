"""Center-padding utilities for fMRI volumes.

We pad to a fixed target shape so volumes of varying sizes can be batched
together. Center-padding preserves the relative position of the brain
within the volume bounds.

Padding values:
  - For data volumes: pad with 0 (matches the natural background).
  - For boolean masks: pad with False (background).

Padding is symmetric where possible. For odd differences, the extra voxel
goes to the high-index side (consistent convention).
"""

from __future__ import annotations

import numpy as np


def compute_pad_widths(
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Compute (before, after) pad widths per axis to center source in target.

    Raises if any source axis exceeds the target — silent truncation would be
    a real bug, so we make it loud.

    Returns:
        ((px_before, px_after), (py_before, py_after), (pz_before, pz_after))
        suitable for passing to np.pad.
    """
    pad = []
    for axis in range(3):
        diff = target_shape[axis] - source_shape[axis]
        if diff < 0:
            raise ValueError(
                f"Source axis {axis} has size {source_shape[axis]} > "
                f"target {target_shape[axis]}. Increase target_shape or "
                f"crop the source."
            )
        before = diff // 2
        after = diff - before  # extra voxel goes to the high side if odd
        pad.append((before, after))
    return tuple(pad)


def center_pad_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-pad a 3D volume to target_shape with the given fill value.

    The brain stays roughly centered in the padded grid (assuming it was
    centered in the source, which IBC's acquisition geometry ensures).
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    pad_widths = compute_pad_widths(volume.shape, target_shape)
    return np.pad(volume, pad_widths, mode="constant", constant_values=pad_value)


def center_pad_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Center-pad a 3D boolean mask to target_shape with False outside."""
    if mask.dtype != bool:
        # Allow uint8 0/1 masks but coerce to bool for the result.
        mask = mask.astype(bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    pad_widths = compute_pad_widths(mask.shape, target_shape)
    return np.pad(mask, pad_widths, mode="constant", constant_values=False)
