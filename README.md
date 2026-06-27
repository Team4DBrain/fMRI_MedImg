# `orchestrator.py` — modular fMRI restoration pipeline harness

One script that runs the team's four model endpoints in **configurable combinations**
on the **same input**, with the **same degradation**, scored by the **same metrics** —
so you can fairly compare pipelines (e.g. the **joint** model vs. a **denoise→SR
cascade**) and get pictures + numbers out of every run.

It sits at the **repo root** (next to `joint/`, `data/`, `sr/`, `Denoising/`,
`data_interpolation/`). It runs the helper code (degradation, normalization,
masking, metrics) **in-process**, and calls each model endpoint as a **subprocess
in its own working directory** (the endpoints have conflicting import roots, so
they can't share one Python process).

---

## TL;DR

```bash
# from the repo root, env active:
cd ~/CAI-MedImg && source /srv/venvs/team4dbrain/setup_env.sh

# the joint model:
python orchestrator.py -i /srv/fMRI-data/sub-13_ses-16_task-PainMovie_dir-pa_bold.nii.gz \
                       -o runs/painmovie_joint   --steps joint

# the denoise→SR cascade on the SAME degraded input (the comparison):
python orchestrator.py -i /srv/fMRI-data/sub-13_ses-16_task-PainMovie_dir-pa_bold.nii.gz \
                       -o runs/painmovie_cascade --steps denoise sr

# quick reproducible test on 10 random timepoints:
python orchestrator.py -i <run> -o runs/quick --steps joint --truncate 10 --seed 0
```

Each run writes a directory:

```
<output>/final.nii.gz        final 4D run
<output>/metrics.json        PSNR / SSIM / tSNR
<output>/slides/*.png        reference-vs-output montages
<output>/run_config.json     provenance (args, seed, start, norm_ref, …)
<output>/work/               reference, degraded, per-step intermediates
```

---

## Arguments

| flag | default | meaning |
|---|---|---|
| `--input`, `-i` | — | input 4D BOLD run (`.nii.gz`), full-resolution (128×128×93×T) |
| `--output`, `-o` | — | output **directory** (created) |
| `--steps` | *(empty)* | ordered endpoint steps from `{denoise, sr, joint, interp}`. Repeat a name to run it twice. Empty = identity passthrough (no steps, **no degradation**; `final` == `reference`) — a harness sanity check, not a degraded baseline. |
| `--degrade-once` | `yes` | `yes` = Architecture A (degrade once, fair). `no` = Architecture B (black-box chain). |
| `--truncate` | `0` | take N consecutive frames from a **random** valid start (0 = whole run). |
| `--seed` | `0` | seeds the truncation start **and** the degradation noise (reproducible). |
| `--sr-model` | `rcan3d` | SR model key (e.g. `rcan3d`, `srcnn3d_deep`); resolves `models/sr_<key>_*_best.pt` (no enum validation — an unknown key fails at lookup). |
| `--interp-mode` | `fill-gaps` | interp output mode. `fill-gaps` = only synthetic frames (T−1); `insert` = originals + synthetic (2T−1). |
| `--keep-intermediates` | `yes` | keep `work/` (degraded + per-step runs). `no` deletes it at the end. |

---

## The two architectures (`--degrade-once`)

### A — degrade once, then compare (`yes`, default)
The orchestrator degrades the input **once** and feeds that identical run to the
chosen steps. Degradation is **conditional on the steps**:

- **spatial** degrade (HR→LR, k-space truncation) iff `sr` **or** `joint` is in `--steps`
- **noise** degrade (Rician) iff `denoise` **or** `joint` is in `--steps`
- applied **spatial-then-noise** when both.

So `denoise sr` and `joint` both see the *same* noisy low-res input → an
apples-to-apples comparison. Here `joint`/`sr` run **LR-native** (selected purely by
input shape, 64×64×46): `joint` detects the LR input and skips its internal degradation;
`sr` is a pure SR model that just upsamples — so neither degrades a second time.

| `--steps` | spatial? | noise? | what the first stage receives |
|---|---|---|---|
| `joint` | ✅ | ✅ | noisy LR (64×64×46) |
| `denoise sr` | ✅ | ✅ | noisy LR (64×64×46) → denoise → SR |
| `sr` | ✅ | ❌ | clean LR (64×64×46) |
| `denoise` | ❌ | ✅ | noisy HR (128×128×93) |
| `interp` | ❌ | ❌ | clean HR (no spatial/noise degrade) |

### B — black-box chain (`no`)
No orchestrator degradation. The raw input goes to the first step and each endpoint
does whatever it does natively (`joint`/`sr` self-degrade as a round-trip). **Cascades
double-degrade here** — this is a contrast baseline, not a fair comparison. (Noise is
each endpoint's own and not seed-controlled.)

---

## The steps

| step | endpoint | what it does | resolution |
|---|---|---|---|
| `joint` | `joint.puppetmaster` | denoise **and** super-resolve in one model | LR→HR |
| `sr` | `sr` (`infer_nifti`) | spatial super-resolution | LR→HR |
| `denoise` | `Denoising/pipeline_api` | denoise (Noise2Noise U-Net) | preserves resolution |
| `interp` | `data_interpolation` | temporal interpolation | preserves space, changes T |

Steps run in the order listed, chaining each output into the next input. The
orchestrator does **not** reshape between steps — each endpoint auto-detects its own
input resolution (joint/sr key off HR 128×128×93 vs LR 64×64×46). **Meaningful** pipelines:
`joint`, `sr`, `denoise sr` (the cascade), `denoise`, `interp`, and combinations
with `interp` for temporal work. Nonsensical chains (e.g. `sr sr`) will run but
double-process — read `metrics.json`/slides before trusting any exotic combo.

---

## Normalization (one scale per run)

`norm_ref` = the 98th percentile of the brain temporal-mean (the training
convention). The orchestrator resolves it once:

- **manifest run** → look up the stored `norm_ref` and load its precomputed brain mask.
- **new run** → compute it: temporal mean → brain mask (SynthStrip via `data.masks`,
  auto-falls back to percentile) → `compute_norm_ref`.

It is passed **explicitly** to `joint` (`--norm-ref`) and `sr` (`infer_nifti
norm_ref=`) so both use the **identical, training-faithful** scale. `denoise`
self-normalizes internally (per-slice percentile — colleague code) but denormalizes
back to physical units, so its output is still comparable. Eval normalizes by this
`norm_ref`, so PSNR/SSIM are in the same units as the training/eval numbers. It is
always a **whole-run** quantity — truncation does not change it. The degradation
parameters (Rician σ range, voxel sizes, HR shape) are read from the **joint
checkpoint config** at runtime, so the orchestrator's degradation can't drift from
what the models trained on.

---

## Evaluation → `metrics.json`

- **tSNR** (in-brain mean of temporal-mean ÷ temporal-std) for the **output** and the
  **reference**, plus their **ratio**. Always computed (it's scale- and T-agnostic).
- **masked PSNR + SSIM** vs the reference, **only when the time axis is unchanged**
  (i.e. no `interp`). Computed per timepoint in normalized units (peak 1.0) and
  averaged, reusing the joint model's metric code so the numbers match training.
- **`interp` in the pipeline** changes the number of frames, so per-frame PSNR/SSIM is
  undefined (synthetic frames have no aligned ground truth). Then:
  - PSNR/SSIM are reported as `null` with a note, and
  - a **leave-out** PSNR/L1 is produced by running `data_interpolation/eval.py` on the
    reference (predict each held-out interior frame from its neighbors, dropping the
    two unpredictable ends) under `interp_leaveout`. *(SSIM is not part of that
    leave-out in this version — `eval.py` reports PSNR/L1 only.)*

The reference is the (optionally truncated) **clean HR input** — every pipeline is
trying to reconstruct it.

---

## Slides

`<output>/slides/t###.png`: montages at a few timepoints × a few axial slices, with
columns **degraded input** (only when the orchestrator actually degraded — Architecture A with a
spatial/noise step) | **pipeline output** | **reference**.
Intensity is windowed to the reference's 99.5th percentile so the panels are
comparable. Needs `matplotlib` (skipped with a message if missing).

---

## Robustness

Every endpoint call is wrapped in a **soft check**: a non-zero exit, missing output,
or timeout is captured (which step + a stderr tail) into `metrics.json`
(`pipeline_failures`) and the run stops cleanly with exit code 1 — never a silent
crash or half-written output. Evaluation and slides are best-effort and won't bring
down a run that otherwise produced an output.

> **Known wiring risk (sr):** `sr`'s inference loads its config from a checkpoint's
> *run-directory* (`config.json`); the shipped weights are loose files in `models/`.
> If `sr` can't find a config it will fail — the soft check will report it clearly.
> If that happens, drop a `config.json` next to the checkpoint (or in its run dir)
> and re-run.

---

## Examples

```bash
# Joint vs cascade on a held-out test run (the thesis comparison), full run:
python orchestrator.py -i <sub-13 run> -o runs/joint    --steps joint
python orchestrator.py -i <sub-13 run> -o runs/cascade  --steps denoise sr
# then diff runs/joint/metrics.json vs runs/cascade/metrics.json

# SR-only baseline (no denoise) on clean LR:
python orchestrator.py -i <run> -o runs/sr_only --steps sr

# Black-box chain for contrast (each endpoint self-degrades):
python orchestrator.py -i <run> -o runs/bbox --steps denoise sr --degrade-once no

# Temporal interpolation, all-synthetic frames:
python orchestrator.py -i <run> -o runs/interp --steps interp --interp-mode fill-gaps
```

## Prerequisites
- Run from the repo root with the env active (`source /srv/venvs/team4dbrain/setup_env.sh`).
- The model endpoints + their weights must be in place: `joint` (hardcoded shared
  weights), `sr` (`models/sr_*_best.pt`), `denoise` (`Denoising/mri_unet_robust.pth`),
  `interp` (`data_interpolation/checkpoints/pretrained/model_weights.pt`).
- A GPU is used automatically when available (CPU otherwise — slower).
