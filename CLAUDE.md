# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fMRI BOLD restoration pipeline combining three model types:
- **Denoising** — U-Net, removes Rician thermal noise
- **Spatial SR** — RCAN3D/SRCNN3D, upscales 3mm LR → 1.5mm HR
- **Temporal interpolation** — UNet3D, synthesizes missing middle frames (T → 2T-1)
- **Joint model** — single network doing both denoising + spatial SR at once

The central `orchestrator.py` runs these as **subprocesses** (conflicting torch/nibabel imports) and evaluates them fairly with one shared degraded input.

## VM Environment

- **Python kernel:** `/srv/venvs/team4dbrain/team4dbrain/bin/python`
- **Jupyter kernel name:** `team4dbrain`
- **Raw BOLD data (BIDS):** `/srv/fMRI-data/`
- **Joint model weights:** `/srv/venvs/team4dbrain/joint_model/best.pt`
- **Manifest (46 runs, norm_ref + masks):** `/srv/venvs/team4dbrain/derivatives/manifest_big.json`
- **Precomputed masks:** `/srv/venvs/team4dbrain/derivatives/masks/`
- **SynthStrip weights:** `/srv/synthstrip/synthstrip.1.pt`
- **SR checkpoints:** `weights/sr/sr_*.pt` (in repo)
- **Denoiser weights:** `weights/denoiser/mri_unet_robust.pth` (in repo)
- **Temporal interp weights:** `weights/temporal/model_weights.pt` (in repo)

All shared `/srv/venvs/team4dbrain/` files are owned by group `team4dbrain` with group-read permissions — all team members can read them.

## Common Commands

**Run the full orchestrator:**
```bash
# Joint model only
python orchestrator.py -i /srv/fMRI-data/<run>.nii.gz -o runs/joint --steps joint

# Cascade: denoise → spatial SR
python orchestrator.py -i /srv/fMRI-data/<run>.nii.gz -o runs/cascade --steps denoise sr

# Temporal interpolation only (on already-restored run)
python orchestrator.py -i <input>.nii.gz -o runs/interp --steps interp

# Quick smoke test (10 frames, reproducible)
python orchestrator.py -i /srv/fMRI-data/<run>.nii.gz -o runs/test --steps joint --truncate 10 --seed 0
```

**SR training & evaluation:**
```bash
python -m sr train --model-name rcan3d --epochs 50 --batch-size 4
python -m sr eval --run-dir sr/runs/<model>/<timestamp>
python -m sr infer --run-dir sr/runs/<model>/<timestamp> --output out.nii.gz
```

**Joint model training (VM only):**
```bash
python -m joint.run --manifest /srv/venvs/team4dbrain/derivatives/manifest_big.json \
  --profile vm --val <subjects> --test <subjects>
```

**Temporal interpolation inference:**
```bash
python data_interpolation/main.py \
  --weights weights/temporal/model_weights.pt \
  --input <run.nii.gz> --output <out.nii.gz> --mode insert
```

**Data pipeline (one-time build, needs SynthStrip):**
```bash
python -m data.build --mask-method auto
```

**SR tests:**
```bash
cd /home/özkan/fMRI_MedImg && /srv/venvs/team4dbrain/team4dbrain/bin/python -m pytest sr/tests/ -v
# Single test:
/srv/venvs/team4dbrain/team4dbrain/bin/python -m pytest sr/tests/test_degradation_spatial.py -v
```

## Architecture & Data Flow

### Orchestrator pipeline (the fair-comparison harness)

```
Input NIfTI (HR, 128×128×93×T)
  │
  ├─ Resolve: norm_ref + brain mask from manifest (or compute fresh)
  ├─ Degrade ONCE (based on --steps):
  │    spatial: HR → LR (k-space truncation)  ← only if sr/joint in steps
  │    noise:   LR + Rician                   ← only if denoise/joint in steps
  │
  ├─ Chain steps (each writes an intermediate NIfTI):
  │    denoise  → Denoising/pipeline_api.py   (LR noisy → LR clean)
  │    sr       → sr/infer.py                 (LR → HR)
  │    joint    → joint/puppetmaster.py        (LR noisy → HR clean, one model)
  │    interp   → data_interpolation/main.py  (T frames → 2T-1 frames)
  │
  └─ Output: final.nii.gz, metrics.json, slides/, run_config.json
```

Degradation parameters (noise sigma, k-space ratio) are **read from the joint checkpoint config** — they cannot drift from training conditions.

### Normalization convention

All models use **per-run scalar normalization**: `volume / norm_ref` where `norm_ref` is the 98th percentile of the brain temporal mean. Not z-score. This preserves spatial contrast and is reversible. The norm_ref is stored in the manifest.

### Dataset shapes

| Dataset | Input shape | Target shape |
|---|---|---|
| `SpatialSRDataset` | (1, 64, 64, 46) LR | (1, 128, 128, 93) HR |
| `DenoisingDataset` | (1, 128, 128, 93) noisy | (1, 128, 128, 93) clean |
| `TemporalSRDataset` | (2, 128, 128, 93) neighbors | (1, 128, 128, 93) middle |
| `JointDataset` | (1, 64, 64, 46) noisy LR | (1, 128, 128, 93) clean HR |

### Why subprocesses?

`joint/puppetmaster.py`, `sr/infer.py`, `Denoising/pipeline_api.py`, and `data_interpolation/main.py` are launched as **separate processes** by the orchestrator because their torch/nibabel version requirements conflict. Each step writes its output to `work/stepNN_<name>.nii.gz` before the next step reads it.

### Model registries (sr/)

`sr/models.py` has `MODEL_REGISTRY` and `sr/components.py` has `OPTIMIZER_REGISTRY` / `SCHEDULER_REGISTRY`. New models/optimizers/schedulers are registered by adding to these dicts — no other wiring needed.

### Rician noise

Applied **after** spatial downsampling (noise lives in acquired LR k-space). Implementation: `magnitude(complex_gaussian(LR))` in normalized units. See `data/degradation_noise.py`.

## Orchestrator Output Structure

```
<output>/
├── final.nii.gz           # or final_FAILED.nii.gz on failure
├── metrics.json           # PSNR, SSIM, tSNR, pipeline_failures
├── run_config.json        # full provenance (input, steps, seed, params)
├── slides/                # reference vs output montages (PNG)
└── work/                  # intermediates (deleted on success, kept on failure)
```
