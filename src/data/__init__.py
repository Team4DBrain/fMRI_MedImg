"""Data loading and preprocessing for the fMRI project."""

# cropping is currently UNUSED in the no_crop_v1 pipeline. Kept exported for
# tests and possible future re-introduction (e.g., if we want xy bbox cropping
# down the line). compute_metadata and datasets do not import from here.
from .cropping import compute_z_start, crop_z, update_affine_for_z_crop
from .degradation_spatial import (
    SpatialDegradation,
    downsample_mask_to_lr,
    kspace_downsample_3d,
    make_spatial_degradation,
    voxel_size_to_target_shape,
)
from .masks import (
    compute_brain_mask,
    find_synthstrip_executable,
    find_synthstrip_model,
    mask_fraction,
)
from .normalize import compute_norm_ref, denormalize, normalize
from .reader import VolumeReader, clear_reader_cache, get_reader

__all__ = [
    "VolumeReader",
    "get_reader",
    "clear_reader_cache",
    "compute_brain_mask",
    "find_synthstrip_executable",
    "find_synthstrip_model",
    "mask_fraction",
    "compute_norm_ref",
    "normalize",
    "denormalize",
    "compute_z_start",
    "crop_z",
    "update_affine_for_z_crop",
    "kspace_downsample_3d",
    "make_spatial_degradation",
    "SpatialDegradation",
    "voxel_size_to_target_shape",
    "downsample_mask_to_lr",
]
