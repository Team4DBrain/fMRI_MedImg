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
  build.py                # Wrapper: runs manifest + metadata in one go
  manifest.py             # Walk BIDS tree, parse filenames, build a JSON manifest
  compute_metadata.py     # Compute brain mask + norm_ref + tSNR per run
  reader.py               # Lazy 3D-volume access to 4D NIfTI files
  masks.py                # Brain masking: SynthStrip (preferred) or percentile fallback
  normalize.py            # Per-run scalar normalization (volume / norm_ref)
  padding.py              # Center-pad volumes/masks to a fixed target shape
  datasets.py             # PyTorch Dataset classes (Denoising/SpatialSR/TemporalSR)
  degradation_spatial.py  # k-space truncation for spatial SR (Option A)

notebooks/
  compare_masks.py        # Visualize percentile vs SynthStrip masks side by side

tests/
  test_data_local.py      # End-to-end smoke test on real data
```

## How it works

### Where things live

Three kinds of paths the pipeline cares about:

- **Raw IBC data** — `--bids-root`. On the VM: `/srv/fMRI-data/`.
- **Derivatives** (manifest, masks) — `--out-dir`. Pick a writable path.
- **Code** — this repo, wherever you clone it.

The manifest is not committed to git. It has absolute paths to data and is
regenerated whenever the dataset changes. Treat it as a derivative that lives
next to the masks.

### Build the manifest + derivatives (one-time, slower)

Single command that runs both data-prep stages in order:

```
python -m src.data.build \
    --bids-root /srv/fMRI-data \
    --out-dir <out> \
    --target-z 93 \
    --mask-method auto
```

This produces:
- `<out>/manifest.json` — one entry per BOLD run with shape + metadata
- `<out>/masks/*_mask.nii.gz` — per-run brain masks, padded to target_shape

Stage 1 (manifest build) runs in seconds. Stage 2 (metadata) reads each 4D file
fully twice and is the slow part — expect ~10-30s per run.

`--target-z 93` is the padding height. If any run is taller, the script logs
an error per-run and continues; bump `--target-z` and rerun (with `--overwrite`
to redo the previously-OK runs).

`--mask-method`:
- `auto` (default): uses SynthStrip if available on PATH; falls back to
  percentile with a warning.
- `synthstrip`: requires `mri_synthstrip`, `synthstrip-docker`, or
  `synthstrip-singularity` on PATH; raises if missing.
- `percentile`: pure-Python intensity threshold + morphology. Works without
  any external tools but produces imperfect masks (includes some skull/scalp).

### Or run the two stages separately

If you want to inspect the manifest before committing to the slow metadata
step, or only redo one stage:

```
python -m src.data.manifest \
    --bids-root /srv/fMRI-data \
    --out <out>/manifest.json

python -m src.data.compute_metadata \
    --manifest <out>/manifest.json \
    --derivatives-dir <out> \
    --target-z 93 \
    --mask-method auto
```

`build.py` just calls these two in order.

### Stage 3 — training (every epoch, on the fly)
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

### Mask quality depends on which method runs
Two backends, dispatched by `--mask-method`:
- **SynthStrip** (preferred, default `auto`): DL-based, designed for cross-modality
  EPI. Robust. Requires `mri_synthstrip`, `synthstrip-docker`, or
  `synthstrip-singularity` on PATH.
- **Percentile fallback**: pure Python, no external tools, but produces imperfect
  masks. Tends to include some skull/scalp; may carve cerebellum at high
  thresholds. Acceptable for code-correctness testing, not for final results.

When SynthStrip isn't installed, `auto` mode falls back to percentile and prints
a warning.

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

## Smoke test

Optional end-to-end test on a small data subset, useful before running the full
pipeline:

```
pip install -r requirements.txt
python tests/test_data_local.py --bids-root <path/to/test/data> --target-z 93
```

Exercises manifest, metadata, all three Datasets, and DataLoader batching
across runs of different native shapes.
