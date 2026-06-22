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
  __init__.py             # Re-exports the public API (so `from src.data import …` works)
  _cli.py                 # Shared CLI helpers: team-VM default paths + arg validators
  build.py                # Wrapper: runs manifest + metadata in one go
  manifest.py             # Walk BIDS tree, parse filenames, build a JSON manifest
  compute_metadata.py     # Compute brain mask + norm_ref + tSNR per run
  reader.py               # Lazy 3D-volume access to 4D NIfTI files (with per-process cache)
  masks.py                # Brain masking: SynthStrip (preferred) or percentile fallback
  normalize.py            # Per-run scalar normalization (volume / norm_ref)
  cropping.py             # Z-axis bbox-centered crop — UNUSED in no_crop_v1, kept for tests
  datasets.py             # PyTorch Dataset classes (Denoising/SpatialSR/TemporalSR/Joint)
  degradation_spatial.py  # k-space truncation for spatial SR (Option A)
  degradation_noise.py    # Rician noise + Compose (for the denoising / joint models)

tests/
  test_cropping.py              # Z-bbox crop, affine update (covers the unused cropping.py)
  test_degradation_spatial.py   # Unit tests incl. k-space scale regression
  test_degradation_noise.py     # Rician noise, Compose, JointDataset end-to-end
  test_reader.py                # VolumeReader and per-process cache
  test_datasets_synthetic.py    # End-to-end Datasets on a synthetic BIDS layout
```

Run the tests with `python -m pytest tests/ -v`. `pytest` itself is a dev
dependency only — it's not pinned in `requirements.txt`; install it
separately (`pip install pytest`) if you want to run the suite.

## How it works

### Where things live

Three kinds of paths the pipeline cares about:

- **Raw IBC data** — `--bids-root`. Default: `/srv/fMRI-data` (team VM).
- **Derivatives** (manifest, masks) — `--out-dir`. Default: `/srv/venvs/team4dbrain/derivatives` (team VM).
- **Code** — this repo, wherever you clone it.

The CLI defaults are tied to the team VM. They must exist on disk or the
command fails immediately with a clear error — no silent fallback. Off-VM
(your laptop, CI, a different cluster), pass `--bids-root` and `--out-dir`
explicitly.

The manifest is not committed to git. It has absolute paths to data and is
regenerated whenever the dataset changes. Treat it as a derivative that lives
next to the masks.

### Build the manifest + derivatives (one-time, slower)

Single command that runs both data-prep stages in order. On the team VM
the defaults cover `--bids-root` and `--out-dir`, so this is the whole
invocation:

```
python -m src.data.build --mask-method auto
```

Pass `--bids-root <path>` and/or `--out-dir <path>` to override the
team-VM defaults (e.g. when running on your laptop).

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

Note that the build wrapper exposes `--target-z` (which it forwards as the
manifest stage's `require_z` filter and the metadata stage's expected uniform
shape). The standalone `manifest.py` instead exposes the same knob as
`--require-z`. Pass `--target-z 0` to `build.py` (or `--require-z 0` to
`manifest.py`) to keep every run regardless of z; downstream
`compute_metadata` will then refuse to run on a mixed-z manifest.

`--mask-method`:
- `auto` (default): uses SynthStrip if any of `nipreps-synthstrip`,
  `mri_synthstrip`, `synthstrip-docker`, or `synthstrip-singularity` is on
  PATH; falls back to percentile with a warning.
- `synthstrip`: requires one of those executables on PATH; raises if none
  is found.
- `percentile`: pure-Python intensity threshold + morphology. Works without
  any external tools but produces imperfect masks (includes some skull/scalp).

#### SynthStrip model weights

`nipreps-synthstrip` (the pip-installable variant) ships the code but **not**
the model weights. The other variants (`mri_synthstrip`, the docker/singularity
wrappers) bake the path in and don't need this lookup.

The pipeline searches for `synthstrip.1.pt` in this order:

1. `$SYNTHSTRIP_MODEL` — env var, if set. If set but the file is missing, a
   warning is logged and the search falls through to the next entries.
2. `/srv/synthstrip/synthstrip.1.pt` — admin-blessed shared location (this is
   where the model lives on the team VM).
3. `~/shared/synthstrip/synthstrip.1.pt` — per-user fallback.

If none exist, `nipreps-synthstrip` raises with a pointer back to here.
The weights live in FreeSurfer's git-annex; see the upstream
[SynthStrip docs](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) for
the official download. On the team VM the file is already in place at
`/srv/synthstrip/synthstrip.1.pt`, so nothing further is needed there.

### Or run the two stages separately

If you want to inspect the manifest before committing to the slow metadata
step, or only redo one stage. Defaults again cover the team VM paths:

```
python -m src.data.manifest

python -m src.data.compute_metadata --mask-method auto
```

Override with `--bids-root`, `--out`, `--manifest`, `--derivatives-dir`
when running off-VM.

`build.py` just calls these two in order.

### Using the data in your training code

Once `build.py` has produced `<out>/manifest.json` and `<out>/masks/`, you do
not run any more data pipeline scripts. All loading and preprocessing happens
inside the Dataset, per-sample.

The three Datasets share the same construction pattern. Pick the one for your
model, instantiate it, hand it to a DataLoader. Each sample comes back as a
dict; the keys differ slightly per task — see below.

#### Spatial SR — `SpatialSRDataset`

```python
from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation
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
from src.data.datasets import DenoisingDataset
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
from src.data.datasets import TemporalSRDataset
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

#### Joint denoise + spatial SR — `JointDataset`

For a single model that **denoises and upsamples at once**. Same structure as
Spatial SR (Option A: LR input, HR target), except the LR input is also
corrupted with noise. The default degradation composes, in order, k-space
spatial downsampling **then** Rician noise — that order is deliberate (thermal
noise lives in the acquired low-resolution k-space, so it is added *after*
downsampling).

```python
from src.data.datasets import JointDataset
from torch.utils.data import DataLoader

train_ds = JointDataset(
    "<out>/manifest.json",
    subject_filter=["01", "04", "07"],
    sigma_min=0.03, sigma_max=0.10,   # Rician noise std range (normalized units)
    # noise_seed=None,                # default: fresh noise per sample (see below)
)
train_loader = DataLoader(train_ds, batch_size=4, num_workers=2, shuffle=True)
```

Each batch:

| key       | shape                  | meaning                                   |
|-----------|------------------------|-------------------------------------------|
| `input`   | `(B, 1, 64, 64, 46)`   | **noisy** LR volume                       |
| `target`  | `(B, 1, 128, 128, 93)` | clean HR volume                           |
| `mask_hr` | `(B, 1, 128, 128, 93)` | brain mask at HR — for HR-domain loss     |
| `mask_lr` | `(B, 1, 64, 64, 46)`   | brain mask at LR                          |

The model takes `input` (noisy LR) and must produce the shape of `target`
(clean HR) — denoising and upsampling in one pass. Compute loss in HR space
weighted by `mask_hr`, same as Spatial SR.

Noise model (`src/data/degradation_noise.py`):
- **Rician**, not additive Gaussian. Gaussian noise on a magnitude image would
  produce negative voxels that never occur in a real scan and that the model
  could exploit as a trivial "tell". Rician noise is the magnitude of a
  complex-Gaussian-corrupted signal — strictly non-negative, with a positive
  background "floor" exactly like real MRI.
- **sigma is in normalized units** (fraction of `norm_ref` ≈ a bright brain
  voxel), so the default `U(0.03, 0.10)` means "3–10% of a bright brain voxel"
  consistently across runs. That range is grounded in the MRI-denoising
  literature (~1–9% of signal) and this dataset's measured in-brain tSNR
  (~14–21 → ~5–7% intrinsic noise).
- **RNG**: by default each sample gets fresh noise (so forked DataLoader workers
  produce independent noise without a `worker_init_fn`). Pass `noise_seed=<int>`
  for determinism — but note a fixed seed gives *every* sample the same noise
  draw; for a fixed-but-varied validation set, derive a per-sample seed in your
  training code.
- The "clean" HR target still carries the scanner's own intrinsic noise (finite
  tSNR), so the model learns to denoise down to that floor, not to a perfectly
  noiseless image — standard for supervised denoising on real data.

To customize, pass your own `degrade_fn` (a picklable callable, e.g. a
`Compose([SpatialDegradation(...), RicianNoise(...)])`); it fully replaces the
default composition.

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
- **Noise model is Rician** (`degradation_noise.py`), applied in normalized
  units. For the joint model, noise is composed *after* spatial downsampling
  (`Compose([SpatialDegradation, RicianNoise])`) because thermal noise lives in
  the acquired LR k-space. Rician (magnitude of complex-Gaussian-corrupted
  signal) is used over additive Gaussian so output stays non-negative like a
  real magnitude image. Default σ range `U(0.03, 0.10)` — see `JointDataset`
  above for the grounding and the RNG/seed caveat.
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
  — also needs the model weights file; see "SynthStrip model weights" above),
  `mri_synthstrip` (full FreeSurfer install), `synthstrip-docker`, or
  `synthstrip-singularity` on PATH.
- **Percentile fallback**: pure Python, no external tools, but produces imperfect
  masks. Tends to include some skull/scalp; may carve cerebellum at high
  thresholds. Acceptable for code-correctness testing, not for final results.

When SynthStrip isn't installed, `auto` mode falls back to percentile and prints
a warning. `compute_metadata` also logs a per-run warning when `mask_fraction`
exceeds 0.55 (a rough sanity floor — whole-brain BOLD masks should be ~0.2-0.4
of the native volume, so anything above 0.55 almost certainly contains
non-brain tissue).

### Degradations — status per dataset
- `SpatialSRDataset`: working k-space degradation in `degradation_spatial.py`.
- `JointDataset`: working default degradation — spatial downsample then Rician
  noise (`Compose([SpatialDegradation, RicianNoise])`), see `degradation_noise.py`.
- `TemporalSRDataset`: no degradation by design (the sampling IS the degradation).
- `DenoisingDataset`: still defaults to a `NotImplementedError` stub so a missing
  decision can't slip silently into a run. A Rician noise model now exists
  (`RicianNoise` / `make_noise` in `degradation_noise.py`) and can be passed as
  its `degrade_fn`. The standalone denoiser's final noise choice is still the
  denoising owner's call — the group floated a noise2noise approach, in which
  case the supervised `degrade_fn` here may not be the path taken.

### Dtype heterogeneity in IBC
IBC's BOLD files come in three on-disk dtypes across releases: **int16**,
**float32**, and **uint16** (yes — all three appear in the same dataset).
This is a storage-format difference, not preprocessing. The float32 / uint16
files have ~20-50× larger raw intensities than int16. Per-run normalization
(divide by the 98th-percentile brain voxel) collapses all three to roughly
the same range, so downstream training code doesn't have to care.

One thing worth knowing: a few uint16 runs we've inspected have voxels
saturated at the dtype max (65535) — real scanner clipping, not pipeline
artifacts. The percentile-based `norm_ref` keeps these from blowing up the
normalized scale (saturated voxels just end up at ~2-3× a typical brain
voxel's normalized value), but if your model is sensitive to bright
outliers, expect these to show up.