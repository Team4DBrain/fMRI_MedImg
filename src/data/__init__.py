"""Data loading and preprocessing for the fMRI project."""

from .cropping import compute_z_start, crop_z, update_affine_for_z_crop
from .degradation_spatial import (
    downsample_mask_to_lr,
    kspace_downsample_3d,
    make_spatial_degradation,
    voxel_size_to_target_shape,
)
from .masks import compute_brain_mask, mask_fraction
from .normalize import compute_norm_ref, denormalize, normalize
from .reader import VolumeReader, clear_reader_cache, get_reader

__all__ = [
    "VolumeReader",
    "get_reader",
    "clear_reader_cache",
    "compute_brain_mask",
    "mask_fraction",
    "compute_norm_ref",
    "normalize",
    "denormalize",
    "compute_z_start",
    "crop_z",
    "update_affine_for_z_crop",
    "kspace_downsample_3d",
    "make_spatial_degradation",
    "voxel_size_to_target_shape",
    "downsample_mask_to_lr",
]
