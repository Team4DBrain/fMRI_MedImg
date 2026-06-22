"""Noise degradation for the denoising / joint restoration models.

Rician noise model
------------------
We add i.i.d. zero-mean Gaussian noise to the REAL and IMAGINARY channels of
the signal, then take the magnitude:

    real  = signal + N(0, sigma)
    imag  =          N(0, sigma)
    noisy = sqrt(real**2 + imag**2)

This is how an MRI magnitude image physically acquires noise: the complex MR
signal is corrupted by zero-mean Gaussian thermal noise in each channel, and
the reconstructed magnitude image is sqrt(real**2 + imag**2). The result is
Rician-distributed.

Why Rician rather than plain additive Gaussian:
  - MRI magnitude data is NON-NEGATIVE. Plain additive Gaussian produces
    negative voxels that never occur in a real scan, handing the model a
    trivial "tell" to spot the corrupted input instead of learning to denoise.
    A Rician magnitude is sqrt(.) so it is strictly >= 0.
  - In high-signal regions (brain), Rician ~= signal + Gaussian, matching the
    intuitive additive-noise picture.
  - In near-zero regions (background), the magnitude of pure complex noise is
    Rayleigh-distributed -- a small POSITIVE "noise floor", which is exactly
    what real MRI background looks like (never zero, never negative).

Noise scale (sigma is in NORMALIZED units)
------------------------------------------
The pipeline normalizes every run by its norm_ref (98th percentile of in-brain
voxels in the temporal mean), so a typical bright brain voxel ~= 1.0. Hence
sigma = 0.05 means "noise std = 5% of a bright brain voxel", consistently
across runs regardless of the raw scanner scale.

This deliberately differs from the "fraction of max intensity" convention used
by the starter code (mri_utils.py add_noise: sigma = noise_level * max(image)),
which is unstable on this dataset: a single uint16-saturated voxel (~2.5 after
normalization) would inflate the noise for the whole volume. Anchoring to the
per-run normalization is outlier-robust.

The default range U(0.03, 0.10) is grounded in two independent sources:
  - the MRI-denoising literature trains on ~1-9% of the signal scale; and
  - this dataset's measured in-brain tSNR ~ 14-21 (median ~18) implies an
    intrinsic temporal noise of ~5-7% of the brain scale.
Both bracket the same band. They are constructor defaults, not hard constants;
override sigma_min/sigma_max freely.

A caveat worth knowing: the "clean" target carries the scanner's own intrinsic
noise (finite tSNR), so a model trained against it learns to denoise down to
that intrinsic floor, not to a perfectly noiseless image. This is the normal
situation for supervised denoising on real data.

Picklability
------------
RicianNoise and Compose are top-level dataclasses (NOT closures) so they
survive multiprocessing's 'spawn' start method (the default on Windows/macOS,
or wherever set_start_method('spawn') is used). Closures returned from a
factory cannot be pickled. This mirrors SpatialDegradation in
degradation_spatial.py.

RNG / reproducibility
---------------------
By default each __call__ draws from a fresh np.random.default_rng() seeded from
OS entropy, so forked DataLoader workers produce INDEPENDENT noise without a
worker_init_fn -- numpy's *global* RNG would otherwise hand every forked worker
the identical noise stream. The trade-off is that noise is not reproducible
run-to-run.

Pass an explicit `seed` for determinism. NOTE: a fixed seed makes __call__
produce the SAME noise draw on every invocation (it does not see a sample
index), so all samples receive identical noise. That is useful for tests and
debugging. For a fixed but per-sample-distinct validation set, derive a
per-sample seed in the training/eval code rather than relying on this field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RicianNoise:
    """Picklable callable that adds Rician noise to a (normalized) volume.

    Attributes:
        sigma_min, sigma_max: noise std drawn U(sigma_min, sigma_max) per call,
            in normalized units (fraction of norm_ref ~= a bright brain voxel).
            Set sigma_min == sigma_max for a fixed level.
        seed: if None (default), a fresh entropy-seeded RNG is used per call,
            giving independent noise across DataLoader workers. If an int, the
            RNG is seeded deterministically (same noise on every call -- see
            the module docstring for the per-sample caveat).
    """

    sigma_min: float = 0.03
    sigma_max: float = 0.10
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.sigma_min < 0 or self.sigma_max < 0:
            raise ValueError(
                f"sigma must be non-negative, got [{self.sigma_min}, {self.sigma_max}]"
            )
        if self.sigma_min > self.sigma_max:
            raise ValueError(
                f"sigma_min ({self.sigma_min}) must be <= sigma_max ({self.sigma_max})"
            )

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        sigma = (
            0.0 if self.sigma_max == 0.0 else float(rng.uniform(self.sigma_min, self.sigma_max))
        )
        if sigma == 0.0:
            # No noise requested: return a float32 copy (never alias the input).
            return volume.astype(np.float32, copy=True)
        # Float64 accumulators: squaring + sqrt on float32 loses precision near
        # the noise floor. Cast back to float32 at the end for the loader.
        sig = volume.astype(np.float64, copy=False)
        real = sig + rng.normal(0.0, sigma, size=volume.shape)
        imag = rng.normal(0.0, sigma, size=volume.shape)
        noisy = np.sqrt(real * real + imag * imag)
        return noisy.astype(np.float32)


@dataclass
class Compose:
    """Picklable callable that chains degradations left-to-right.

    Compose([SpatialDegradation(), RicianNoise()]) applies spatial degradation
    FIRST, then noise -- the physically correct order for the joint denoise+SR
    task: thermal noise lives in the acquired (low-resolution) k-space, so it is
    added AFTER spatial downsampling, not before.

    Every step must be a picklable callable (volume -> volume), e.g. a top-level
    dataclass, so the whole Compose survives 'spawn'. A list of plain closures
    would not pickle.

    Attributes:
        steps: ordered list of callables, each mapping an ndarray to an ndarray.
    """

    steps: list

    def __post_init__(self) -> None:
        self.steps = list(self.steps)
        for i, step in enumerate(self.steps):
            if not callable(step):
                raise TypeError(
                    f"Compose step {i} is not callable: {type(step).__name__}"
                )

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        out = volume
        for step in self.steps:
            out = step(out)
        return out


def make_noise(
    sigma_min: float = 0.03,
    sigma_max: float = 0.10,
    seed: int | None = None,
) -> RicianNoise:
    """Build a Rician-noise degradation with the project's standard defaults.

    Returns a RicianNoise instance (a picklable callable). Mirrors
    make_spatial_degradation in degradation_spatial.py.
    """
    return RicianNoise(sigma_min=sigma_min, sigma_max=sigma_max, seed=seed)
