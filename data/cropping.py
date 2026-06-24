"""Brain-bbox-centered cropping along the z-axis.

IBC has uniform x and y (128×128) but varying z (84..93). We crop z-only to
a fixed target_z. The crop window is centered on the brain mask's z-bounding
box so the crop never carves into anatomy.

Why z-only and not full bbox cropping in xy too:
  - x and y don't vary across IBC. Cropping them would discard FOV that
    might matter for noise/artifact context, with zero shape benefit.
  - Keeping x and y at native resolution preserves the affine in those
    dimensions and avoids a bigger refactor for marginal gains.

Per-run we store `z_start` (the offset into native z where the crop begins).
The cropped volume's affine origin shifts by `z_start * voxel_size_z`.
"""

from __future__ import annotations

import numpy as np


def compute_z_start(
    mask: np.ndarray,
    target_z: int,
) -> int:
    """Compute the z-axis crop offset that centers the mask's z-bbox in target_z.

    Args:
        mask: 3D boolean mask at NATIVE shape (X, Y, Z_native).
        target_z: desired output z-extent. Must be <= mask.shape[2].

    Returns:
        z_start: integer in [0, Z_native - target_z] such that
                 [z_start, z_start + target_z) contains the mask's z-bbox.

    Raises:
        ValueError: if the mask is empty, target_z exceeds native z, or the
                    z-bbox is taller than target_z.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    z_native = mask.shape[2]
    if target_z > z_native:
        raise ValueError(
            f"target_z={target_z} exceeds native z={z_native}. "
            "Cropping cannot grow the volume; this run is shorter than target."
        )
    if target_z == z_native:
        return 0

    # z-bbox of mask
    z_any = mask.any(axis=(0, 1))  # (Z,) — True where any in-brain voxel exists
    if not z_any.any():
        raise ValueError("Mask is empty — cannot compute bbox.")
    z_lo = int(np.argmax(z_any))                     # first True
    z_hi = int(z_native - np.argmax(z_any[::-1]))    # one past last True
    bbox_extent = z_hi - z_lo
    if bbox_extent > target_z:
        raise ValueError(
            f"Brain z-bbox extent {bbox_extent} (slices {z_lo}..{z_hi}) "
            f"exceeds target_z={target_z}. Either target_z is too small, "
            "or the mask is contaminated. Inspect the mask."
        )

    # Center the bbox in the crop window.
    bbox_center = (z_lo + z_hi) / 2.0
    z_start = int(round(bbox_center - target_z / 2.0))
    # Clamp to valid range
    z_start = max(0, min(z_start, z_native - target_z))
    # Sanity: bbox must be fully inside [z_start, z_start + target_z).
    # Clamping above handles edge cases where the brain hugs one side.
    if z_lo < z_start or z_hi > z_start + target_z:
        # Shouldn't happen given the extent check above, but belt and suspenders.
        raise RuntimeError(
            f"Internal: z_start={z_start} doesn't contain bbox [{z_lo},{z_hi})"
        )
    return z_start


def crop_z(
    array: np.ndarray,
    z_start: int,
    target_z: int,
) -> np.ndarray:
    """Crop the z-axis of a 3D or 4D array to [z_start, z_start + target_z).

    For 3D: array[..., z_start:z_start+target_z]
    For 4D (X, Y, Z, T): same — z is axis 2.
    """
    if array.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array, got shape {array.shape}")
    if z_start < 0 or z_start + target_z > array.shape[2]:
        raise ValueError(
            f"Crop window [{z_start}, {z_start + target_z}) outside z-range "
            f"[0, {array.shape[2]})"
        )
    return array[:, :, z_start : z_start + target_z]


def update_affine_for_z_crop(
    affine: np.ndarray,
    z_start: int,
) -> np.ndarray:
    """Shift the affine origin to reflect a z-axis crop.

    The voxel-to-world mapping for the cropped volume needs the origin moved
    by z_start voxels along the z direction. The z-direction vector is
    affine[:3, 2]; the origin is affine[:3, 3].

    Returns a new affine; does not mutate the input.
    """
    if affine.shape != (4, 4):
        raise ValueError(f"Expected (4,4) affine, got {affine.shape}")
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + z_start * affine[:3, 2]
    return new_affine
