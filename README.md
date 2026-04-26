# fMRI Restoration Project — Data Pipeline

CS/AI student project on the [IBC dataset](https://openneuro.org/datasets/ds002685/versions/2.0.0).
We're training three models that take "imperfect" fMRI scans and produce cleaner versions:

1. **Denoising model** — noisy → clean, same resolution
2. **Spatial SR model** — low-res (3mm) → high-res (1.5mm)
3. **Temporal SR model** — interpolate a missing volume from neighbors

This repo currently contains the **data pipeline** only. Models will follow.

## What's in this repo

```
src/data/
  manifest.py             # Walk BIDS tree, parse filenames, build a JSON manifest
  compute_metadata.py     # Compute brain mask + norm_ref + tSNR per run
  reader.py               # Lazy 3D-volume access to 4D NIfTI files
  masks.py                # Brain mask via percentile + morphology (TO BE REPLACED)
  normalize.py            # Per-run scalar normalization (volume / norm_ref)
  padding.py              # Center-pad volumes/masks to a fixed target shape
  datasets.py             # PyTorch Dataset classes (Denoising/SpatialSR/TemporalSR)
  degradation_spatial.py  # k-space truncation for spatial SR (Option A)

tests/
  test_data_local.py      # End-to-end smoke test on real data
```

## How it works

**Stage 1 — manifest** (one-time, fast):
```
python -m src.data.manifest --bids-root /path/to/ibc_raw --out manifest.json
```
Walks the BIDS tree, lists every BOLD run with shape and metadata. Outputs JSON.

**Stage 2 — per-run metadata** (one-time, slower; reads each file fully twice):
```
python -m src.data.compute_metadata --manifest manifest.json \
    --derivatives-dir /path/to/derivatives --target-z 93
```
Computes brain mask (saved as `.nii.gz`), normalization reference (98th percentile
of in-brain voxels), and tSNR. Pads everything to `target_shape`. Updates manifest
in place.

`--target-z 93` is the configured padding height. If any run is taller, the script
crashes with a clear error; bump `--target-z` and rerun.

**Stage 3 — training** (every epoch, on the fly):
```python
from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation
from torch.utils.data import DataLoader

degrade = make_spatial_degradation(source_voxel_mm=1.5, target_voxel_mm=3.0)
ds = SpatialSRDataset(
    "manifest.json",
    subject_filter=["01", "04"],     # train subjects
    degrade_fn=degrade,
)
loader = DataLoader(ds, batch_size=8, num_workers=2, shuffle=True)
```

Each sample comes back as a dict with input/target/mask tensors.

## Design decisions worth knowing

- **Raw data only**, no preprocessing pipeline yet (no motion correction, no
  distortion correction). May add later.
- **Center-padding** to a fixed `target_shape` for batching across runs of
  different native shapes.
- **Per-run scalar normalization** (`volume / norm_ref`). Brain voxels end up
  near 1.0, background near 0. Reversible. Not z-scored — preserves spatial
  contrast and BOLD temporal dynamics.
- **CPU degradation in the DataLoader workers**, not GPU. Cleaner training loop,
  CPU is otherwise idle during training.
- **Spatial SR is "Option A"**: model input is at LR shape, target at HR shape.
  Model is responsible for upsampling.
- **Temporal SR has no separate degradation**. Sampling at gap=1 (predict t from
  t-1, t+1) IS the degradation — equivalent to half-rate acquisition.

## Open issues / things to be aware of

### Mask quality is imperfect
The current masking uses percentile thresholding + morphology. Empirically it
includes some skull/scalp and may carve into the cerebellum on some runs.
**Plan**: replace with [SynthStrip](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip)
or FSL `bet` when running on the VM. Function signature stays the same so no
downstream code change is needed.

### Stub degradations
- `DenoisingDataset` requires you to pass a `degrade_fn`. None implemented yet —
  the noise model is the denoising owner's call. (Group decided on a noise2noise
  approach, so noise modeling may not be needed at all — TBD.)
- `SpatialSRDataset` has a working degradation in `degradation_spatial.py`.
- `TemporalSRDataset` has no degradation by design (see above).

### Sample API differs by Dataset
- `DenoisingDataset` / `TemporalSRDataset`: `mask` key
- `SpatialSRDataset`: `mask_hr` and `mask_lr` keys (LR mask is derived from HR
  via `downsample_mask_to_lr`)

### Dtype heterogeneity in IBC
Some IBC files are stored as int16, others as float32 (with ~20× larger raw
intensities). This is not preprocessing — it's a storage-format difference
across releases. Per-run normalization handles it cleanly.

### Local test artifacts
Running `tests/test_data_local.py` creates `_local_test_workdir/` with a
manifest and derivatives. That dir is gitignored.

## Local testing

Requires Python 3.12 and the deps in `requirements.txt`:
```
pip install -r requirements.txt
python tests/test_data_local.py --bids-root path/to/local/test/data --target-z 93
```

The test exercises manifest building, metadata computation, all three Datasets,
and DataLoader batching across runs of different native shapes. Takes 10-15 min
on a laptop with 9 IBC runs.
