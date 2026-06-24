"""PyTorch Dataset classes for fMRI training.

Architecture:
  - BaseFMRIDataset handles the common stuff: manifest parsing, sample
    indexing, mask loading, normalization.
  - Subclasses (DenoisingDataset, SpatialSRDataset, TemporalSRDataset) override
    __getitem__ to return the right (input, target) pair for their model.
  - Degradation functions are STUBBED for the denoising path. They raise
    NotImplementedError until we implement them with each model's owner.
    (Spatial degradation IS implemented; pass make_spatial_degradation()
    as degrade_fn to SpatialSRDataset.)

Shape policy (no_crop_v1 pipeline):
  - Every run in the manifest has the same native shape, e.g. (128, 128, 93).
  - That shape is the served shape. No cropping. No padding.
  - Manifest stage drops anything that doesn't match `require_z`.
  - cropping.py exists in the codebase but is unused by this pipeline.

For SpatialSRDataset (Option A), the input is at LR shape and the target at HR
shape — the model is responsible for upsampling.

Performance notes:
  - VolumeReaders are obtained via reader.get_reader (per-process cache), so
    multiple samples from the same run inside a worker share one open handle.
  - TemporalSR reads a (2*gap+1) span as ONE range read, not three separate
    timepoint reads.
  - Mask cache is bounded (LRU). Default cap of 32 entries × ~6 MB per HR mask
    is ~200 MB peak per worker, which is sustainable on training nodes.

Fork-safety:
  - nibabel image handles don't always fork cleanly. The reader cache is
    process-local (keyed by pid) and is naturally re-populated lazily in each
    DataLoader worker after fork. Do not call get_reader from the main
    process before spawning workers, or you risk a stale entry.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation_spatial import (
    SpatialDegradation,
    downsample_mask_to_lr,
    make_spatial_degradation,
    voxel_size_to_target_shape,
)
from .normalize import normalize
from .reader import get_reader

logger = logging.getLogger(__name__)

# Manifest pipeline marker that this Dataset accepts. Older manifests
# ('z_crop' or none) are rejected with a clear message.
ACCEPTED_PIPELINE = "no_crop_v1"

# Default cap on the per-worker mask cache. Each HR mask at IBC shape
# (128*128*93 float32) is ~6 MB. 32 caps memory at ~200 MB per worker.
DEFAULT_MASK_CACHE_SIZE = 32


class _BoundedTensorCache:
    """Tiny LRU cache for tensor blobs. OrderedDict-based; not thread-safe.

    Safe in standard DataLoader usage because each worker accesses the cache
    from a single thread. `pin_memory=True` and `prefetch_factor` spawn helper
    threads but those only move tensors, they don't touch the dataset's caches.
    Don't share an instance across threads that BOTH access the cache, or add
    a lock.
    """

    def __init__(self, max_size: int):
        self._max = max(1, int(max_size))
        self._d: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        t = self._d.get(key)
        if t is not None:
            self._d.move_to_end(key)
        return t

    def set(self, key: str, value: torch.Tensor) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        while len(self._d) > self._max:
            self._d.popitem(last=False)


class BaseFMRIDataset(Dataset):
    """Shared behavior for all three task datasets.

    Subclasses must implement:
        - _build_sample_index() -> list of sample descriptors
        - __getitem__(idx) -> dict with at least "input", "target", "mask"
    """

    def __init__(
        self,
        manifest_path: str | Path,
        subject_filter: list[str] | None = None,
        check_files_exist: bool = True,
        mask_cache_size: int = DEFAULT_MASK_CACHE_SIZE,
    ):
        """
        Args:
            manifest_path: path to JSON manifest produced by the data build.
                Must be from the no_crop_v1 pipeline. Older 'z_crop' manifests
                are rejected.
            subject_filter: if given, keep only runs whose subject is in this list.
                Use this for train/val/test splits by subject. Subjects must match
                the manifest's exact string form (typically zero-padded, e.g. "01").
            check_files_exist: if True (default), validate at construction time
                that every kept run's BOLD file and mask file exist. Trades a
                small startup cost (one stat() per file) for fail-fast behavior.
                Set False when the cost matters and you trust your manifest.
            mask_cache_size: max number of HR masks held in the per-worker cache.
                Each mask is ~6 MB at IBC shape. Default 32 → ~200 MB peak.
        """
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open() as f:
            manifest = json.load(f)

        if manifest.get("pipeline") != ACCEPTED_PIPELINE:
            raise RuntimeError(
                f"Manifest pipeline marker is {manifest.get('pipeline')!r}, "
                f"expected {ACCEPTED_PIPELINE!r}. Re-run the data build "
                f"(`python -m data.build ...`) to regenerate the manifest "
                f"and masks under the current pipeline."
            )

        self.bids_root = Path(manifest["bids_root"])
        # Validate derivatives_dir explicitly. `manifest.get(..., "")` followed
        # by Path(...) is a trap: Path("") resolves to Path(".") which is_dir()
        # always returns True from any working directory, so a missing field
        # silently passed validation and failed much later with a confusing
        # FileNotFoundError on a relative mask path.
        deriv = manifest.get("derivatives_dir")
        if not deriv:
            raise RuntimeError(
                "manifest is missing 'derivatives_dir'. Re-run compute_metadata.py."
            )
        self.derivatives_dir = Path(deriv)
        if not self.derivatives_dir.is_dir():
            raise RuntimeError(
                f"derivatives_dir from manifest does not exist: {self.derivatives_dir}. "
                "Did the derivatives directory get moved or deleted?"
            )

        if "target_shape" not in manifest or "target_z" not in manifest:
            raise RuntimeError(
                "manifest is missing 'target_shape' or 'target_z'. "
                "Re-run compute_metadata.py."
            )
        self.target_shape: tuple[int, int, int] = tuple(manifest["target_shape"])
        self.target_z: int = int(manifest["target_z"])

        # Keep only runs that have complete metadata.
        all_runs = manifest["runs"]
        runs = [
            r for r in all_runs
            if "norm_ref" in r and "mask_path" in r
        ]
        if len(runs) < len(all_runs):
            logger.warning(
                "Dropped %d/%d runs missing metadata. Re-run compute_metadata.py "
                "if you expected them.", len(all_runs) - len(runs), len(all_runs),
            )

        # Hard shape conformance check: every run's native (= served) shape
        # must equal target_shape. Manifest stage should have enforced this;
        # verify defensively to fail loudly on hand-edited manifests.
        bad_shape = [r for r in runs if tuple(r["shape"][:3]) != self.target_shape]
        if bad_shape:
            preview = ", ".join(
                f"{r['run_id']}({tuple(r['shape'][:3])})" for r in bad_shape[:3]
            )
            raise RuntimeError(
                f"{len(bad_shape)} run(s) have shape != target_shape={self.target_shape}: "
                f"{preview}{'...' if len(bad_shape) > 3 else ''}. "
                "Manifest is inconsistent — rebuild it."
            )

        if subject_filter is not None:
            wanted = set(subject_filter)
            available = {r["subject"] for r in runs}
            unmatched = wanted - available
            if unmatched:
                raise ValueError(
                    f"subject_filter contains subjects not in manifest: {sorted(unmatched)}. "
                    f"Available: {sorted(available)}. Note: subjects are typically "
                    f"zero-padded ('01', not '1')."
                )
            runs = [r for r in runs if r["subject"] in wanted]

        if not runs:
            raise RuntimeError(
                f"No runs matched after filtering. subject_filter={subject_filter}"
            )

        if check_files_exist:
            self._validate_paths(runs)

        self.runs = runs
        self._mask_cache = _BoundedTensorCache(mask_cache_size)

        self.samples = self._build_sample_index()
        if not self.samples:
            raise RuntimeError(
                f"{type(self).__name__} produced an empty sample index. "
                "Check filtering and (for TemporalSR) gap parameter."
            )

    def _validate_paths(self, runs: list[dict]) -> None:
        """Fail-fast check that referenced data and mask files exist.

        Runs once at construction in the main process. PyTorch's DataLoader
        forks workers AFTER __init__ returns, so this does not re-stat in
        every worker; the workers inherit the constructed Dataset.
        """
        missing = []
        for r in runs:
            if not (self.bids_root / r["path"]).is_file():
                missing.append(("bold", r["run_id"], self.bids_root / r["path"]))
            if not (self.derivatives_dir / r["mask_path"]).is_file():
                missing.append(("mask", r["run_id"], self.derivatives_dir / r["mask_path"]))
        if missing:
            n_show = 5
            preview = "\n".join(f"  [{kind}] {rid}: {p}" for kind, rid, p in missing[:n_show])
            extra = f"\n  ... and {len(missing) - n_show} more" if len(missing) > n_show else ""
            raise FileNotFoundError(
                f"{len(missing)} referenced file(s) missing on disk:\n{preview}{extra}"
            )

    def _build_sample_index(self) -> list:
        """Default: one sample per (run, timepoint). Subclasses can override."""
        samples = []
        for run_idx, run in enumerate(self.runs):
            for t in range(run["n_volumes"]):
                samples.append((run_idx, t))
        return samples

    def _get_hr_mask(self, run_idx: int) -> torch.Tensor:
        """Return a CLONED HR brain mask tensor (float32, target_shape).

        Mask on disk is at native shape, which equals target_shape.
        """
        run = self.runs[run_idx]
        run_id = run["run_id"]
        cached = self._mask_cache.get(run_id)
        if cached is None:
            mask_path = self.derivatives_dir / run["mask_path"]
            arr = np.asarray(nib.load(str(mask_path)).dataobj).astype(np.float32)
            if arr.shape != self.target_shape:
                raise RuntimeError(
                    f"Mask for {run_id} has shape {arr.shape}, expected "
                    f"target_shape {self.target_shape}. Re-run compute_metadata.py "
                    f"to regenerate."
                )
            arr = np.ascontiguousarray(arr)
            cached = torch.from_numpy(arr)
            self._mask_cache.set(run_id, cached)
        return cached.clone()

    def _read_volume(self, run_idx: int, t: int) -> np.ndarray:
        """Read one volume (run_idx, t) and normalize. Returns float32 (X, Y, Z)."""
        run = self.runs[run_idx]
        path = self.bids_root / run["path"]
        # get_reader is process-local, populated lazily inside the worker.
        reader = get_reader(path)
        vol = reader.read_volume(t).astype(np.float32)
        return normalize(vol, run["norm_ref"])

    def _read_range(self, run_idx: int, t_start: int, t_end: int) -> np.ndarray:
        """Read [t_start, t_end), normalize. Returns (T, X, Y, Z) float32.

        ONE underlying disk read for the contiguous span, not (t_end - t_start)
        independent ones.
        """
        run = self.runs[run_idx]
        path = self.bids_root / run["path"]
        reader = get_reader(path)
        block = reader.read_range(t_start, t_end).astype(np.float32)  # (X,Y,Z,T)
        block = normalize(block, run["norm_ref"])
        return np.moveaxis(block, -1, 0)                              # (T,X,Y,Z)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("Subclasses must implement __getitem__")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _not_implemented_degradation(*args, **kwargs):
    raise NotImplementedError(
        "Degradation function not yet implemented. "
        "Decide on the degradation model with your teammates first."
    )


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Wrap numpy → torch float32. Casts non-float32 inputs to float32 to keep
    a consistent tensor dtype across the loader. C-contiguous to avoid the
    from_numpy stride footguns."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _validate_degraded(
    out: np.ndarray,
    expected_shape: tuple,
    *,
    label: str,
) -> np.ndarray:
    """Check a degrade_fn output. Returns the array as float32, contiguous.

    Symmetric checking across DenoisingDataset and SpatialSRDataset — both
    used to differ on whether they validated their degrade_fn output, which
    let bugs in custom degradations slip into batches as silent shape/dtype
    mismatches.
    """
    if not isinstance(out, np.ndarray):
        raise TypeError(
            f"{label} degrade_fn returned {type(out).__name__}, expected np.ndarray"
        )
    if out.shape != expected_shape:
        raise RuntimeError(
            f"{label} degrade_fn returned shape {out.shape}, expected {expected_shape}."
        )
    if not np.isfinite(out).all():
        raise RuntimeError(
            f"{label} degrade_fn returned non-finite values (NaN or Inf)."
        )
    if out.dtype != np.float32:
        out = out.astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Task-specific datasets
# ---------------------------------------------------------------------------


class DenoisingDataset(BaseFMRIDataset):
    """Samples for the denoising model.

    Returns dict with:
        input:  noisy volume,  shape (1, X, Y, Z) (float32)
        target: clean volume,  shape (1, X, Y, Z) (float32)
        mask:   HR brain mask, shape (1, X, Y, Z) (float32)
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        degrade_fn=_not_implemented_degradation,
        check_files_exist: bool = True,
        mask_cache_size: int = DEFAULT_MASK_CACHE_SIZE,
    ):
        super().__init__(
            manifest_path,
            subject_filter=subject_filter,
            check_files_exist=check_files_exist,
            mask_cache_size=mask_cache_size,
        )
        self.degrade_fn = degrade_fn

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        clean = self._read_volume(run_idx, t)             # (X, Y, Z)
        noisy = self.degrade_fn(clean)                    # same shape expected
        # Validate output: shape must match clean, dtype coerced to float32.
        # Catches bugs in user-supplied degrade_fn that previously slipped
        # silently through DenoisingDataset (asymmetric with SpatialSRDataset).
        noisy = _validate_degraded(noisy, clean.shape, label="DenoisingDataset")
        mask = self._get_hr_mask(run_idx)                 # (X, Y, Z) float32

        return {
            "input": _to_tensor(noisy).unsqueeze(0),
            "target": _to_tensor(clean).unsqueeze(0),
            "mask": mask.unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }


class SpatialSRDataset(BaseFMRIDataset):
    """Samples for the spatial SR model — Option A.

    Input is at LR shape (e.g. 64×64×46 for 3mm with target_z=93), target at
    HR shape (X, Y, target_z). The model is responsible for upsampling.

    Returns dict with:
        input:    LR volume,  shape (1, kx, ky, kz)
        target:   HR volume,  shape (1, X, Y, Z)
        mask_hr:  HR mask,    shape (1, X, Y, Z)
        mask_lr:  LR mask,    shape (1, kx, ky, kz)
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        degrade_fn=None,
        lr_shape: tuple[int, int, int] | None = None,
        source_voxel_mm: float = 1.5,
        target_voxel_mm: float = 3.0,
        check_files_exist: bool = True,
        mask_cache_size: int = DEFAULT_MASK_CACHE_SIZE,
    ):
        super().__init__(
            manifest_path,
            subject_filter=subject_filter,
            check_files_exist=check_files_exist,
            mask_cache_size=mask_cache_size,
        )
        # lr_shape resolution: explicit > derived from voxel sizes.
        # If a custom degrade_fn is passed, the caller must either pass
        # lr_shape explicitly OR ensure the voxel sizes match what their
        # degrade_fn produces — we no longer keep voxel_mm as instance state
        # (it would silently disagree with a custom degrade_fn).
        if lr_shape is None:
            lr_shape = voxel_size_to_target_shape(
                self.target_shape, source_voxel_mm, target_voxel_mm,
            )
        self.lr_shape: tuple[int, int, int] = tuple(lr_shape)

        if degrade_fn is None:
            # Default degradation: k-space truncation at the requested voxel ratio.
            degrade_fn = make_spatial_degradation(
                source_voxel_mm=source_voxel_mm,
                target_voxel_mm=target_voxel_mm,
            )
        self.degrade_fn = degrade_fn

        # Construction-time consistency probe. lr_shape and degrade_fn can
        # disagree (e.g., user passed a custom lr_shape but left degrade_fn
        # defaulted to the voxel-ratio implementation, or vice versa). Probe
        # with a zero volume of target_shape and verify the output shape
        # matches lr_shape — fail here, not at the first __getitem__.
        probe_in = np.zeros(self.target_shape, dtype=np.float32)
        try:
            probe_out = self.degrade_fn(probe_in)
        except Exception as e:
            raise RuntimeError(
                f"degrade_fn raised on a probe input of shape {self.target_shape}: {e}"
            ) from e
        if probe_out.shape != self.lr_shape:
            raise ValueError(
                f"degrade_fn output shape {probe_out.shape} does not match "
                f"lr_shape {self.lr_shape}. Either pass a degrade_fn whose "
                f"output matches lr_shape, or omit lr_shape so it's derived "
                f"from the same voxel ratio as the default degrade_fn."
            )

        # LR masks derived from HR masks; cached separately, also bounded.
        self._lr_mask_cache = _BoundedTensorCache(mask_cache_size)

    def _get_lr_mask(self, run_idx: int) -> torch.Tensor:
        """Return a CLONED LR brain mask tensor (float32, lr_shape)."""
        run = self.runs[run_idx]
        run_id = run["run_id"]
        cached = self._lr_mask_cache.get(run_id)
        if cached is None:
            mask_path = self.derivatives_dir / run["mask_path"]
            hr_bool = np.asarray(nib.load(str(mask_path)).dataobj).astype(bool)
            lr_bool = downsample_mask_to_lr(hr_bool, self.lr_shape)
            arr = np.ascontiguousarray(lr_bool.astype(np.float32))
            cached = torch.from_numpy(arr)
            self._lr_mask_cache.set(run_id, cached)
        return cached.clone()

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        hr = self._read_volume(run_idx, t)                  # (X, Y, Z)
        lr = self.degrade_fn(hr)                            # (kx, ky, kz)
        lr = _validate_degraded(lr, self.lr_shape, label="SpatialSRDataset")

        return {
            "input": _to_tensor(lr).unsqueeze(0),
            "target": _to_tensor(hr).unsqueeze(0),
            "mask_hr": self._get_hr_mask(run_idx).unsqueeze(0),
            "mask_lr": self._get_lr_mask(run_idx).unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }


class TemporalSRDataset(BaseFMRIDataset):
    """Samples for the temporal SR model.

    The "degradation" is implicit: by sampling t-gap and t+gap and predicting
    the volume at t, we simulate a half-rate acquisition where t was never
    measured. There is no separate degrade_fn.

    Returns dict with:
        input:  two neighbors stacked along channel, shape (2, X, Y, Z)
        target: middle volume,                       shape (1, X, Y, Z)
        mask:   HR brain mask,                       shape (1, X, Y, Z)
    """

    def __init__(
        self,
        manifest_path,
        subject_filter=None,
        gap: int = 1,
        check_files_exist: bool = True,
        mask_cache_size: int = DEFAULT_MASK_CACHE_SIZE,
    ):
        if gap < 1:
            raise ValueError(f"gap must be >= 1, got {gap}")
        self.gap = gap
        super().__init__(
            manifest_path,
            subject_filter=subject_filter,
            check_files_exist=check_files_exist,
            mask_cache_size=mask_cache_size,
        )

    def _build_sample_index(self) -> list:
        """Only valid samples where both neighbors exist inside the run."""
        samples = []
        too_short: list[str] = []
        for run_idx, run in enumerate(self.runs):
            n = run["n_volumes"]
            if n < 2 * self.gap + 1:
                too_short.append(f"{run['run_id']} (n={n})")
                continue
            for t in range(self.gap, n - self.gap):
                samples.append((run_idx, t))
        if too_short:
            logger.warning(
                "TemporalSRDataset(gap=%d): dropped %d run(s) too short for any sample: %s",
                self.gap, len(too_short),
                ", ".join(too_short[:5]) + ("..." if len(too_short) > 5 else ""),
            )
        return samples

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples[idx]
        # Single contiguous read covers all three timepoints regardless of gap.
        # Block has 2*gap+1 frames; we need indices 0, gap, 2*gap (= last).
        block = self._read_range(run_idx, t - self.gap, t + self.gap + 1)  # (T, X, Y, Z)
        before = block[0]
        target = block[self.gap]
        after = block[-1]

        input_arr = np.stack([before, after], axis=0)  # (2, X, Y, Z)

        return {
            "input": _to_tensor(input_arr),
            "target": _to_tensor(target).unsqueeze(0),
            "mask": self._get_hr_mask(run_idx).unsqueeze(0),
            "run_id": self.runs[run_idx]["run_id"],
            "t": t,
        }
