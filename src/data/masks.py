"""Brain mask computation for fMRI volumes.

Custom implementation using intensity thresholding + morphological cleanup.
Deliberately simple — we own and understand every line.

KNOWN LIMITATIONS:
This is a percentile-based intensity threshold approach. Empirically it tends
to either include too much skull/scalp (low percentile) or carve into the
cerebellum (high percentile). It cannot cleanly separate brain from bright
non-brain tissue (fat, scalp) because they overlap in EPI intensity.

For local development this is acceptable but flawed. The plan is to replace
with SynthStrip (Hoopes et al. 2022, NeuroImage) when running on the VM:
  - State-of-the-art DL-based skull stripping designed for cross-modality
  - Works directly on EPI without needing T1 coregistration
  - Distributed via FreeSurfer or as a standalone Docker container
  - GitHub: https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip

TODO: When FSL or SynthStrip is available, swap compute_brain_mask to use it.
The function signature can stay the same (mean_volume in, bool array out) so
no downstream code changes are needed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_brain_mask(
    mean_volume: np.ndarray,
    lower_percentile: float = 55.0,
    opening_iterations: int = 2,
    closing_iterations: int = 2,
) -> np.ndarray:
    """Compute a brain mask from a temporally-averaged 3D volume.

    Algorithm:
      1. Threshold at the `lower_percentile`-th percentile of non-zero voxels.
         Rationale: exact zeros are padding/air, so percentile-on-nonzero gives
         a more informative threshold than a simple percentile of all voxels.
      2. Morphological opening: removes small bright specks outside the brain
         (fat, skin, noise).
      3. Keep largest connected component: throws away disconnected regions
         (eyeballs, sinus artifacts).
      4. Morphological closing: fills small holes inside the brain (e.g.,
         ventricles can have lower signal than threshold).

    Args:
        mean_volume: 3D array, shape (X, Y, Z). Should be the temporal mean of
            a BOLD run, not a single volume (too noisy).
        lower_percentile: percentile of non-zero voxels used as threshold.
            Lower = larger mask. 20 is a good default for EPI data.
        opening_iterations: morphological opening strength.
        closing_iterations: morphological closing strength.

    Returns:
        Boolean array, same shape as input, True inside brain.

    Raises:
        ValueError: if input is not 3D or is all zeros.
    """
    if mean_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {mean_volume.shape}")

    nonzero = mean_volume[mean_volume > 0]
    if nonzero.size == 0:
        raise ValueError("Input volume has no non-zero voxels — can't compute mask")

    threshold = np.percentile(nonzero, lower_percentile)
    mask = mean_volume > threshold

    # Opening = erosion then dilation. Removes small bright specks.
    mask = ndimage.binary_opening(mask, iterations=opening_iterations)

    # Keep only the largest connected component.
    # Labels every connected region with a unique integer; component 0 is background.
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        raise ValueError(
            "Mask is empty after opening — try lower_percentile lower, "
            "or the input has no discernible brain"
        )

    # Count voxels in each component (skipping label 0 = background).
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest_label = int(np.argmax(sizes))
    mask = labeled == largest_label

    # Closing = dilation then erosion. Fills small holes (e.g., ventricles).
    mask = ndimage.binary_closing(mask, iterations=closing_iterations)

    # Final hole-filling pass for any cavities that survived closing.
    mask = ndimage.binary_fill_holes(mask)

    return mask.astype(bool)


def mask_fraction(mask: np.ndarray) -> float:
    """Fraction of voxels that are inside the brain. Sanity check ≈ 0.2-0.5 for whole-brain BOLD."""
    return float(mask.mean())
