# CAI-MedImg — fMRI Restoration Project

CAI-MedImg is a modular medical-imaging project for testing different fMRI
processing pipelines and comparing which approach gives better results.

Each team member develops one pipeline component in a separate subfolder. The
components should expose a simple interface so they can be combined and tested
against each other in the full project. The main goal is to run different
pipeline choices, evaluate their outputs, and keep the best-performing methods
for the final system.

This work uses the [IBC dataset](https://openneuro.org/datasets/ds002685/versions/2.0.0).
We're training three models that take "imperfect" fMRI scans and produce cleaner versions:

1. **Denoising model** — noisy → clean, same resolution
2. **Spatial SR model** — low-res (3mm) → high-res (1.5mm)
3. **Temporal SR model** — interpolate a missing volume from neighbors

This repo contains the **data pipeline** (`data/`), a **spatial SR trainer**
(`sr/`), a **temporal interpolation module** (`data_interpolation/`), and a
**denoising stack** (`Denoising/`). LR brain volumes → HR super-resolution
uses `srcnn3d` or `rcan3d`. Denoising and temporal SR are dataset-ready in
`data/`; `data_interpolation/` provides a standalone temporal-interpolation
stack; `Denoising/` includes a trained U-Net and inference scripts.

## Project modules

| Module | Purpose |
|--------|---------|
| `data/` | BIDS manifest, brain masks, PyTorch datasets (denoising, spatial SR, temporal SR) |
| `sr/` | Spatial super-resolution training/eval/infer CLI (`srcnn3d`, `rcan3d`) |
| `data_interpolation/` | Temporal fMRI interpolation — takes a 4D BOLD NIfTI and generates a new file with interpolated time frames |
| `Denoising/` | 3D U-Net denoising — training and inference on fMRI volumes |

See [data_interpolation/README.md](data_interpolation/README.md) for setup and usage of the interpolation module.

## What's in this repo

```
data/
  build.py                # Wrapper: runs manifest + metadata in one go
  manifest.py             # Walk BIDS tree, parse filenames, build a JSON manifest
  compute_metadata.py     # Compute brain mask + norm_ref + tSNR per run
  reader.py               # Lazy 3D-volume access to 4D NIfTI files (with per-process cache)
  masks.py                # Brain masking: SynthStrip (preferred) or percentile fallback
  normalize.py            # Per-run scalar normalization (volume / norm_ref)
  cropping.py             # Z-axis bbox-centered crop — UNUSED in no_crop_v1, kept for tests
  datasets.py             # PyTorch Dataset classes (Denoising/SpatialSR/TemporalSR)
  degradation_spatial.py  # k-space truncation for spatial SR (Option A)

sr/
  cli.py          # CLI: train, eval, infer, debug (`python -m sr`)
  train.py        # training loop, checkpoints, TensorBoard
  data.py         # train/val DataLoaders (sample split, SpatialSRDataset)
  models.py       # SRCNN3D, RCAN3D registry
  config.py       # defaults and validation
  runs/           # checkpoints (not committed)

data_interpolation/
  main.py         # Entry point for temporal interpolation
  train.py        # Training script
  eval.py         # Evaluation script
  src/            # Model, dataset, loss, inference utilities
  configs/        # Default training config
  notebooks/      # Quick tests and full training notebooks

Denoising/
  train.py                # U-Net training
  apply_denoise_3d.py     # Inference on NIfTI volumes
  model.py                # 3D U-Net architecture
  mri_unet.pth            # Pretrained weights

tests/
  test_cropping.py              # Z-bbox crop, affine update (covers the unused cropping.py)
  test_degradation_spatial.py   # Unit tests incl. k-space scale regression
  test_reader.py                # VolumeReader and per-process cache
  test_datasets_synthetic.py    # End-to-end Datasets on a synthetic BIDS layout
  sr/                           # SR models, metrics, reproducibility helpers
```

Run the tests with `python -m pytest tests/ -v`.

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
python -m data.build \
    --bids-root /srv/fMRI-data \
    --out-dir <out> \
    --mask-method auto
```

This produces:
- `<out>/manifest.json` — one entry per BOLD run with shape + metadata
  (top-level `target_shape`, `target_z`, `pipeline: "no_crop_v1"`, `require_z`)
- `<out>/masks/*_mask.nii.gz` — per-run brain masks at `(X, Y, target_z)` =
  the run's native shape (no cropping in this pipeline)

Stage 1 (manifest build) runs in seconds. Stage 2 (metadata) reads each 4D
file fully once — typically ~5-15s per run, dominated by SynthStrip.

`--target-z` is optional. By default it's `93` (IBC's standard slab) and any
run whose native z differs is dropped at the manifest stage with a logged
warning — this screens out the documented `z=84` anomaly in some IBC ses-00
/ ses-01 sessions. Pass an explicit value if you want a different slab.
Pass `--require-z 0` to the manifest stage if you want to keep every run
regardless of z (and accept that compute_metadata will then refuse to run on
a mixed-z manifest).

`--mask-method`:
- `auto` (default): uses SynthStrip if any of `nipreps-synthstrip`,
  `mri_synthstrip`, `synthstrip-docker`, or `synthstrip-singularity` is on
  PATH; falls back to percentile with a warning.
- `synthstrip`: requires one of those executables on PATH; raises if none
  is found.
- `percentile`: pure-Python intensity threshold + morphology. Works without
  any external tools but produces imperfect masks (includes some skull/scalp).

### Or run the two stages separately

If you want to inspect the manifest before committing to the slow metadata
step, or only redo one stage:

```
python -m data.manifest \
    --bids-root /srv/fMRI-data \
    --out <out>/manifest.json

python -m data.compute_metadata \
    --manifest <out>/manifest.json \
    --derivatives-dir <out> \
    --mask-method auto
```

`build.py` just calls these two in order.

### Spatial SR training (CLI, `sr/`)

With an enriched `manifest.json` (from `compute_metadata`) and masks on disk, you can train and evaluate **3mm → 1.5mm** spatial super-resolution from the repo root. Training uses **mask-weighted MSE** on HR voxels; validation logs **masked MSE, PSNR, and local 3D SSIM**. Checkpoints and TensorBoard logs go under `sr/runs/<model_name>/<timestamp>/` (`config.json`, `split.json`, `metrics.json`, `epochs/best.pt`, etc.).

```bash
# Train (example)
python -m sr train --model-name srcnn3d
python -m sr train --model-name rcan3d --epochs 20 --batch-size 4

# Evaluate on the held-out val split
python -m sr eval \
  --checkpoint sr/runs/srcnn3d/<timestamp>/epochs/best.pt

# Single-sample inference + optional PNG export
python -m sr infer \
  --checkpoint sr/runs/srcnn3d/<timestamp>/epochs/best.pt \
  --sample-index 0
```

Useful flags: `--manifest-path`, `--train-split`, `--loss-name`, `--run-root`, `--resume-dir`. Full module reference: [sr/README.md](sr/README.md).

### Using the data in your training code

Once `build.py` has produced `<out>/manifest.json` and `<out>/masks/`, you do
not run any more data pipeline scripts. All loading and preprocessing happens
inside the Dataset, per-sample.

The three Datasets share the same construction pattern. Pick the one for your
model, instantiate it, hand it to a DataLoader. Each sample comes back as a
dict; the keys differ slightly per task — see below.

#### Spatial SR — `SpatialSRDataset`

```python
from data.datasets import SpatialSRDataset
from data.degradation_spatial import make_spatial_degradation
from torch.utils.data import DataLoader

degrade = make_spatial_degradation(source_voxel_mm=1.5, target_voxel_mm=3.0)

train_ds = SpatialSRDataset(
    "<out>/manifest.json",
    subject_filter=["01", "04", "07"],
    degrade_fn=degrade,
)
val_ds = SpatialSRDataset(
    "<out>/manifest.json",
    subject_filter=["11"],
    degrade_fn=degrade,
)
train_loader = DataLoader(train_ds, batch_size=4, num_workers=2, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=4, num_workers=2, shuffle=False)
```

Each batch is a dict with:

| key       | shape                  | meaning                                   |
|-----------|------------------------|-------------------------------------------|
| `input`   | `(B, 1, 64, 64, 46)`   | LR volume (3mm simulated acquisition)     |
| `target`  | `(B, 1, 128, 128, 93)` | HR volume (1.5mm ground truth)            |
| `mask_hr` | `(B, 1, 128, 128, 93)` | brain mask at HR — for HR-domain loss     |
| `mask_lr` | `(B, 1, 64, 64, 46)`   | brain mask at LR — derived from `mask_hr` |

(Z dimensions shown are for `target_z=93`, the IBC default. If you set a
different `--target-z`, all the z values above scale accordingly. xy is
always 128×128 / 64×64.)

Your model takes `input` (LR) and must produce something the shape of `target`
(HR) — it has to upsample internally. Compute loss in HR space, weighted by
`mask_hr`:

```python
for batch in train_loader:
    lr        = batch["input"].to("cuda")
    hr_target = batch["target"].to("cuda")
    mask      = batch["mask_hr"].to("cuda")

    hr_pred = model(lr)
    loss = ((hr_pred - hr_target) ** 2 * mask).sum() / mask.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### Denoising — `DenoisingDataset`

```python
from data.datasets import DenoisingDataset
from torch.utils.data import DataLoader

# You decide what counts as "noisy". Pass a function that takes a clean
# numpy volume and returns a noisy one. Stub example:
def add_gaussian_noise(clean, sigma=0.05):
    import numpy as np
    return clean + np.random.normal(0, sigma, clean.shape).astype(np.float32)

train_ds = DenoisingDataset(
    "<out>/manifest.json",
    subject_filter=["01", "04", "07"],
    degrade_fn=add_gaussian_noise,
)
train_loader = DataLoader(train_ds, batch_size=4, num_workers=2, shuffle=True)
```

Each batch:

| key      | shape                  | meaning                              |
|----------|------------------------|--------------------------------------|
| `input`  | `(B, 1, 128, 128, 93)` | noisy volume (output of degrade_fn)  |
| `target` | `(B, 1, 128, 128, 93)` | clean volume                         |
| `mask`   | `(B, 1, 128, 128, 93)` | brain mask                           |

Both input and target are full-resolution. If you go noise2noise (where both
sides are noisy and there's no "clean" target), the current API doesn't fit out
of the box — talk to the data pipeline owner before going down that path.

```python
for batch in train_loader:
    noisy = batch["input"].to("cuda")
    clean = batch["target"].to("cuda")
    mask  = batch["mask"].to("cuda")

    pred = model(noisy)
    loss = ((pred - clean) ** 2 * mask).sum() / mask.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### Temporal SR — `TemporalSRDataset`

No `degrade_fn` here. The sampler skips a timepoint and gives you its two
neighbors as input — that's the degradation.

```python
from data.datasets import TemporalSRDataset
from torch.utils.data import DataLoader

train_ds = TemporalSRDataset(
    "<out>/manifest.json",
    subject_filter=["01", "04", "07"],
    gap=1,    # use t-1 and t+1 to predict t. Bump to simulate larger TR multipliers.
)
train_loader = DataLoader(train_ds, batch_size=4, num_workers=2, shuffle=True)
```

Each batch:

| key      | shape                  | meaning                                     |
|----------|------------------------|---------------------------------------------|
| `input`  | `(B, 2, 128, 128, 93)` | two neighbors stacked: [t-gap, t+gap]       |
| `target` | `(B, 1, 128, 128, 93)` | the "missing" middle volume at time t       |
| `mask`   | `(B, 1, 128, 128, 93)` | brain mask                                  |

```python
for batch in train_loader:
    neighbors = batch["input"].to("cuda")   # 2 channels: before, after
    middle    = batch["target"].to("cuda")
    mask      = batch["mask"].to("cuda")

    pred = model(neighbors)
    loss = ((pred - middle) ** 2 * mask).sum() / mask.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The loss formulas above are illustrative — masked L1, SSIM, or other choices
are all viable. Train/val/test splits in `subject_filter` are placeholders too;
the group decides who's in which.

## Design decisions worth knowing

- **Raw data only**, no preprocessing pipeline yet (no motion correction, no
  distortion correction). May add later.
- **No cropping, no padding** (the `no_crop_v1` pipeline). Every run is served
  at its native shape (`128×128×93` for IBC). Runs whose native z differs from
  `--target-z` are dropped at the manifest stage rather than reshaped — this
  screens out the documented IBC `z=84` anomaly in some early sessions. The
  `cropping.py` module is kept in the repo for tests but is not imported by
  the live data path.
- **Per-run scalar normalization** (`volume / norm_ref`). Brain voxels end up
  near 1.0, background near 0. Reversible. Not z-scored — preserves spatial
  contrast and BOLD temporal dynamics.
- **CPU degradation in the DataLoader workers**, not GPU. Cleaner training loop,
  CPU is otherwise idle during training.
- **Spatial SR is "Option A"**: model input is at LR shape, target at HR shape.
  Model is responsible for upsampling.
- **Temporal SR has no separate degradation**. Sampling at gap=1 (predict t from
  t-1, t+1) IS the degradation — equivalent to half-rate acquisition.
- **k-space LR scale**: `kspace_downsample_3d` takes the real part of the IFFT
  (not magnitude — see the function's docstring for why) and scales by `M/N`
  (output size / input size) to preserve mean intensity. There's a regression
  test for this in `tests/test_degradation_spatial.py`; if you touch the scale
  logic or the .real/.abs choice, run the tests.
- **`indexed_gzip` is required** (pinned in `requirements.txt`). Without it,
  every random-access volume read decompresses the gzip stream from byte 0.
  nibabel picks it up automatically when present — no code change needed.
- **VolumeReader caching**. `reader.get_reader(path)` is a per-process cache.
  Each DataLoader worker keeps one open handle per run instead of reopening
  on every `__getitem__`. Cache is keyed by `(pid, resolved_path)` so it's
  fork-safe.

## Open issues / things to be aware of

### Mask quality depends on which method runs
Two backends, dispatched by `--mask-method`:
- **SynthStrip** (preferred, default `auto`): DL-based, designed for cross-modality
  EPI. Robust. Requires one of `nipreps-synthstrip` (pip-installable, lightest
  — also needs the model weights file; see below), `mri_synthstrip` (full
  FreeSurfer install), `synthstrip-docker`, or `synthstrip-singularity` on PATH.
- **Percentile fallback**: pure Python, no external tools, but produces imperfect
  masks. Tends to include some skull/scalp; may carve cerebellum at high
  thresholds. Acceptable for code-correctness testing, not for final results.

When SynthStrip isn't installed, `auto` mode falls back to percentile and prints
a warning. `compute_metadata` also logs a per-run warning when `mask_fraction`
exceeds 0.55 (a rough sanity floor — whole-brain BOLD masks should be ~0.2-0.4
of the native volume, so anything above 0.55 almost certainly contains
non-brain tissue).

### Stub degradations
- `DenoisingDataset` requires you to pass a `degrade_fn`. None implemented yet —
  the noise model is the denoising owner's call. (Group decided on a noise2noise
  approach, so noise modeling may not be needed at all — TBD.)
- `SpatialSRDataset` has a working degradation in `degradation_spatial.py`.
- `TemporalSRDataset` has no degradation by design (see above).

### Dtype heterogeneity in IBC
Some IBC files are stored as int16, others as float32 (with ~20× larger raw
intensities). This is not preprocessing — it's a storage-format difference
across releases. Per-run normalization handles it cleanly.
