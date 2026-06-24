"""Spatial resolution degradation for the SR model — Option A.

We simulate a target-voxel-size acquisition (e.g., 3mm) from 1.5mm IBC source
data by truncating 3D k-space and producing an output at the truncated resolution.

This is "Option A" from the design discussion:
  - Input volume:  (X, Y, Z) at source voxel size, e.g. (128, 128, 93) at 1.5mm
  - Output volume: (kx, ky, kz) at target voxel size, e.g. (64, 64, 46) at 3mm
  - Smaller output. The SR model receives the small LR volume and must
    upsample back to the HR target.

Why k-space truncation rather than image-space blur+downsample:
  - Physically correct: a real lower-resolution acquisition has lower k-space
    sampling extent. Truncation models exactly what the scanner does.
  - Image-space Gaussian blur is a different operation with different artifacts.

Why apodization (Hamming window):
  - A hard box cutoff at the kept-region boundary causes Gibbs ringing
    in image space (sinc PSF in image domain).
  - Hamming softens the cutoff. Real acquisitions don't have perfectly sharp
    k-space cutoffs either, so this is more realistic.

Mask derivation:
  Brain masks are stored at HR. The LR mask used during training is derived
  from the HR mask at runtime via spatial downsampling. We don't k-space-
  truncate the mask itself — masks are metadata, not signal, and we want a
  clean boolean mask without ringing artifacts.

Picklability note:
  `make_spatial_degradation` returns a SpatialDegradation INSTANCE, not a
  closure. Closures from factory functions cannot be pickled by stock
  `pickle`, which breaks `multiprocessing` `spawn` start method (used on
  macOS/Windows or whenever `mp.set_start_method('spawn')` is invoked).
  Top-level callable classes pickle cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def _hamming_window_3d(shape: tuple[int, int, int]) -> np.ndarray:
    """Separable 3D Hamming window of the given shape, centered."""
    wx = np.hamming(shape[0])
    wy = np.hamming(shape[1])
    wz = np.hamming(shape[2])
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def kspace_downsample_3d(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    apodize: bool = True,
) -> np.ndarray:
    """Truncate 3D k-space and return the volume at the truncated (LR) shape.

    Steps:
      1. FFT the input volume to k-space.
      2. fftshift to put DC at the center.
      3. Crop the central target_shape region of k-space.
      4. Apply a Hamming apodization within the cropped region (if apodize=True).
      5. ifftshift, IFFT, take REAL part. Scale to compensate for size change.

    The output has shape `target_shape`, NOT the input shape. The model is
    responsible for upsampling.

    Real part vs. magnitude:
      For an even target M on any input N, the natural central crop spans
      frequencies [-M/2, M/2 - 1] — asymmetric about DC, missing +M/2. On
      real-valued input this breaks Hermitian symmetry and the IFFT picks up
      nonzero imaginary parts (energy folded in from the missing Nyquist bin).
      `np.abs` would inflate the magnitude on non-DC content; `.real` cleanly
      drops that fold-back and matches the band-limited downsample of a
      Hermitian-symmetric crop. Apodization-induced sinc ringing can produce
      small negative values, which `.real` honours and `np.abs` would lie about.

    Args:
        volume: 3D real-valued array (X, Y, Z).
        target_shape: (kx, ky, kz) — the desired LR shape.
        apodize: if True, apply Hamming window in k-space.

    Returns:
        3D real-valued array of shape target_shape, dtype float32.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {volume.shape}")
    for axis in range(3):
        if not (1 <= target_shape[axis] <= volume.shape[axis]):
            raise ValueError(
                f"target_shape[{axis}]={target_shape[axis]} out of range "
                f"[1, {volume.shape[axis]}] for input shape {volume.shape}"
            )

    # FFT and shift DC to center
    kspace = np.fft.fftshift(np.fft.fftn(volume))

    # Crop the central region
    full_shape = volume.shape
    slices = []
    for axis in range(3):
        center = full_shape[axis] // 2
        half = target_shape[axis] // 2
        start = center - half
        end = start + target_shape[axis]
        slices.append(slice(start, end))
    kspace_cropped = kspace[slices[0], slices[1], slices[2]]

    # Apodization
    if apodize:
        window = _hamming_window_3d(target_shape)
        kspace_cropped = kspace_cropped * window

    # Inverse FFT, take REAL part. See docstring for why .real (not np.abs).
    image = np.fft.ifftn(np.fft.ifftshift(kspace_cropped))

    # Scale compensation. np.fft.ifftn divides by the output size M, but the
    # cropped DC equals N * mean(volume) (carried over from the unnormalized
    # forward FFT on the source of size N). So image[0] = (N/M) * mean(volume),
    # and we must MULTIPLY by M/N to recover the original intensity scale.
    # (A constant volume in must yield a near-constant volume out at the same
    # magnitude — see tests/test_degradation_spatial.py.)
    scale = np.prod(target_shape) / np.prod(full_shape)
    return (image.real * scale).astype(np.float32)


def voxel_size_to_target_shape(
    source_shape: tuple[int, int, int],
    source_voxel_mm: float = 1.5,
    target_voxel_mm: float = 3.0,
) -> tuple[int, int, int]:
    """Compute LR target shape for a given source/target voxel size."""
    if target_voxel_mm <= source_voxel_mm:
        raise ValueError(
            f"target voxel size ({target_voxel_mm}mm) must be larger than "
            f"source ({source_voxel_mm}mm) for a downsampling operation"
        )
    ratio = source_voxel_mm / target_voxel_mm
    return tuple(max(1, int(round(s * ratio))) for s in source_shape)


def downsample_mask_to_lr(
    hr_mask: np.ndarray,
    target_shape: tuple[int, int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Derive an LR brain mask from an HR mask via spatial downsampling.

    Uses linear interpolation followed by thresholding. We do NOT apply
    k-space truncation to the mask — masks are metadata, and we want a
    clean boolean output without ringing.

    Args:
        hr_mask: 3D boolean array at HR (e.g., 128×128×93).
        target_shape: desired LR shape (e.g., 64×64×46).
        threshold: voxel is "in brain" if interpolated density >= this.

    Returns:
        3D boolean array of shape target_shape.
    """
    if hr_mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {hr_mask.shape}")
    if hr_mask.dtype != bool:
        hr_mask = hr_mask.astype(bool)

    zoom_factors = tuple(t / s for t, s in zip(target_shape, hr_mask.shape))
    soft = ndimage.zoom(hr_mask.astype(np.float32), zoom_factors, order=1)

    # zoom can occasionally produce a slightly off shape due to floating-point
    # rounding. Crop or pad if needed.
    if soft.shape != target_shape:
        result = np.zeros(target_shape, dtype=np.float32)
        slices = tuple(slice(0, min(t, s)) for t, s in zip(target_shape, soft.shape))
        result[slices] = soft[slices]
        soft = result

    return soft >= threshold


@dataclass
class SpatialDegradation:
    """Picklable callable that applies k-space-truncation degradation.

    Stored as a top-level dataclass (not a closure) so that
    multiprocessing's `spawn` start method, which requires every transferred
    object to be picklable by name, can serialize it. Closures from a factory
    function fail this requirement.

    Use `make_spatial_degradation(...)` to construct one with the project's
    standard defaults.

    Attributes:
        source_voxel_mm: source voxel size (e.g., 1.5 for IBC).
        target_voxel_mm: simulated LR voxel size (must be > source).
        apodize: if True, apply Hamming apodization in k-space (standard).
    """

    source_voxel_mm: float = 1.5
    target_voxel_mm: float = 3.0
    apodize: bool = True

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        target_shape = voxel_size_to_target_shape(
            volume.shape, self.source_voxel_mm, self.target_voxel_mm,
        )
        return kspace_downsample_3d(volume, target_shape, apodize=self.apodize)


def make_spatial_degradation(
    source_voxel_mm: float = 1.5,
    target_voxel_mm: float = 3.0,
    apodize: bool = True,
) -> SpatialDegradation:
    """Build a degradation function compatible with SpatialSRDataset.

    Returns a SpatialDegradation instance (a picklable callable). Pass it as
    the `degrade_fn=` arg to SpatialSRDataset.
    """
    return SpatialDegradation(
        source_voxel_mm=source_voxel_mm,
        target_voxel_mm=target_voxel_mm,
        apodize=apodize,
    )
