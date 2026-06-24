# `sr` — Spatial Super-Resolution

Minimal, modular 3D SR pipeline for fMRI volumes. Four commands, one
config dataclass, one checkpoint format. No automatic post-training
analysis — the user composes plots, comparisons, and reports themselves.

## Layout

```
sr/
├── config.py       # SRConfig dataclass, defaults, JSON IO, validate
├── models.py       # SRCNN3D, RCAN3D, MODEL_REGISTRY, build_model
├── losses.py       # mse, masked_mse, kspace_mse, dual_domain_masked_mse, registries
├── metrics.py      # psnr, masked/unmasked SSIM, compute_full_metrics
├── components.py   # OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
├── data.py         # build_loaders, dataset split, seeded workers
├── checkpoint.py   # EpochState save/load, find_latest_epoch
├── train.py        # train(config, resume_dir=None)
├── infer.py        # evaluate, infer_one, infer_nifti, previews, list_samples
├── debug.py        # masks, prediction error, baseline comparison, loss curve
└── cli.py + __main__.py   # `python -m sr ...`
```

Each module owns one responsibility. To swap a loss, model, optimizer or
scheduler, add an entry to the corresponding registry and pass its name
via the CLI — no edits to `train.py` required.

## Defaults

| Field | Default |
|---|---|
| `manifest_path` | `/srv/venvs/team4dbrain/derivatives/manifest.json` |
| `run_root` | `sr/runs` |
| `model_name` | `srcnn3d` |
| `output_patch_shape` | `(128, 128, 93)` |
| `source_voxel_mm` -> `target_voxel_mm` | `1.5 -> 3.0` |
| `train_split` | `0.8` (train) / `0.2` (val), shuffled by `seed` |
| `loss_name`, `loss_kwargs` | `masked_mse`, `{}` (see dual-domain keys below) |
| `optimizer_name`, `learning_rate` | `adam`, `1e-3` |
| `scheduler_name`, `scheduler_kwargs` | `plateau`, `{"factor":0.5,"patience":3}` |
| `seed`, `deterministic`, `strict_finite_loss` | `42`, `true`, `true` |
| `batch_size`, `num_epochs`, `num_workers`, `log_interval` | `4`, `20`, `0`, `10` |
| `tensorboard` | `true` |

## Run artifacts

```
sr/runs/<model_name>/<timestamp>/
├── config.json         # written once at start of run, source of truth
├── split.json          # written once, resolved train/val sample split
├── metrics.json        # rewritten atomically every epoch (plain JSON)
├── tb/                 # TensorBoard scalars (if --tensorboard)
└── epochs/
    ├── epoch_001.pt    # full EpochState (model + opt + sched + RNG + history)
    ├── epoch_002.pt
    ├── best.pt         # rolling best val checkpoint (written when val improves)
    └── ...
```

`epoch_NNN.pt` is fully self-contained: kill the process at any point and
resume from the last successfully-written epoch with zero information
loss. `epochs/best.pt` is updated whenever validation improves; NIfTI
infer with `--model-name` resolves to `best.pt` when present, otherwise
the latest `epoch_NNN.pt`.

## CLI

```bash
# Train from scratch (uses every default)
python -m sr train

# Train with custom knobs
python -m sr train \
  --model-name rcan3d \
  --model-kwargs '{"n_feats": 48, "n_resgroups": 3}' \
  --loss-name masked_l1 \
  --optimizer-name adamw \
  --optimizer-kwargs '{"weight_decay": 1e-4}' \
  --scheduler-name cosine \
  --scheduler-kwargs '{"T_max": 20}' \
  --epochs 20 --batch-size 4 --lr 1e-3

# Dual-domain (masked image MSE + orthonormal 3D FFT k-space MSE)
python -m sr train \
  --loss-name dual_domain_masked_mse \
  --loss-kwargs '{"alpha": 0.5, "beta": 0.5, "kspace_high_freq_weight": 0.5}'

# Focal Frequency Loss (dynamic k-space weighting, ICCV 2021)
python -m sr train \
  --loss-name focal_frequency \
  --loss-kwargs '{"alpha": 1.0, "log_matrix": false, "batch_matrix": false}'

python -m sr train --resume-dir sr/runs/srcnn3d/20260511_120000

# Evaluate a checkpoint on its saved val split
python -m sr eval \
  --checkpoint sr/runs/srcnn3d/<run>/epochs/epoch_010.pt \
  --report ./report.json

# List samples available in the manifest used by a checkpoint
python -m sr infer \
  --checkpoint sr/runs/srcnn3d/<run>/epochs/best.pt \
  --list-samples

# Infer one manifest sample (metrics + optional figures)
python -m sr infer \
  --checkpoint sr/runs/srcnn3d/<run>/epochs/best.pt \
  --subject 01 --session 00 --task ArchiStandard --direction ap --t 12 \
  --preview \
  --axis coronal --slice-level 0.4 \
  --save-png ./infer_single_slice.png \
  --save-npy ./infer_pred.npy

# Infer a standalone NIfTI (3D or 4D; writes HR volume)
python -m sr infer \
  --input /path/to/vol.nii.gz \
  --model-name rcan3d \
  --run-root output \
  --output ~/results/ \
  --preview

# Same, but pin a specific checkpoint instead of auto-resolving best/latest
python -m sr infer \
  --input /path/to/vol.nii.gz \
  --checkpoint output/rcan3d/<run>/epochs/best.pt \
  --output ~/results/vol_sr.nii.gz \
  --preview --slice-level 0.5
```

### `infer` modes

| Mode | Required flags | Writes |
|------|----------------|--------|
| NIfTI file | `--input`, `--model-name` (or `--checkpoint`) | `<stem>_sr.nii.gz` (HR `128×128×93` by default) |
| Manifest sample | `--checkpoint`, at least one selector (`--subject`, …) | Metrics to stdout; optional PNG/NPY |
| List samples | `--checkpoint` or `--manifest-path`, `--list-samples` | Sample table only |

**NIfTI input behaviour**

- HR-native volumes (`128×128×93` at 1.5 mm) are k-space degraded to LR
  internally, then super-resolved. LR-native volumes (`64×64×46` at 3 mm)
  are fed to the model directly.
- `--output` may be a file path or a directory (e.g. `~/results/`); a
  directory receives `<input_stem>_sr.nii.gz`.
- 4D inputs use timepoint `t=0` unless `--t` is set.

**Preview (`--preview`)**

- Off by default. Pass `--preview` to write a multi-axis PNG montage
  (axial / coronal / sagittal).
- Rows: Input (LR), SR output, and Ground truth (HR) when GT is available
  (HR NIfTI inputs and manifest samples). LR-only NIfTI inputs omit the GT
  row.
- NIfTI infer writes `<output_stem>.png` beside the output NIfTI.
  Manifest infer writes `infer_<subject>_<session>_<task>_<direction>_t<t>.png`
  in the current directory.
- `--save-png` adds an extra single-axis figure (`--axis`, `--slice-level`).

```bash
# Debug: masks + degradation (no checkpoint)
python -m sr debug \
  --subject 01 --session 03 --task HcpEmotion --direction ap --t 0 \
  --save-png ./debug_masks.png

# Debug: prediction error vs ground truth + trilinear baseline
python -m sr debug \
  --checkpoint sr/runs/srcnn3d/<run>/epochs/epoch_010.pt \
  --subject 01 --session 03 --task HcpEmotion --direction ap --t 0 \
  --figure both --error-map abs --mask-errors \
  --save-dir ./debug_out --plot-loss-curve
```

## Debug command

`debug` inspects one manifest sample without training. Use it to verify masks,
degradation, and (with `--checkpoint`) whether the model improves over the
built-in trilinear upsampling baseline.

| Flag | Purpose |
|------|---------|
| `--figure masks` | 2×2 HR/LR images and masks (default without checkpoint) |
| `--figure infer` | 1×5 GT, prediction, error, baseline, baseline error |
| `--figure both` | Both figures (default with `--checkpoint`) |
| `--error-map abs\|signed\|squared` | How to colour pred vs target |
| `--mask-errors` | Zero error outside the HR brain mask |
| `--save-dir DIR` | Writes `masks.png`, `infer.png`, optional `loss_curve.png` |
| `--plot-loss-curve` | Plot `metrics.json` from the checkpoint run |

**Why loss can drop quickly (what to look for)**

- `SRCNN3D` already trilinearly upsamples LR to HR before conv layers — epoch 1
  loss is mostly “refine interpolation,” not learn SR from scratch.
- Masked losses (~34% brain voxels) look lower than full-volume MSE.
- Compare `masked_mse_baseline` vs `masked_mse_pred` in CLI output; small gap
  means the network adds little beyond upsampling.

**Suggested follow-up visualizations** (not all built in; compose as needed)

- K-space magnitude diff when training with `kspace_mse` / `dual_domain_masked_mse`
- Side-by-side error maps for `epoch_001.pt` vs `epoch_N.pt`
- Error histograms in-mask vs out-of-mask
- Multi-slice montage (`--preview`, or several `--slice-level` values with `--save-png`)
- TensorBoard batch scalars under `run_dir/tb/`

## Resume contract

`--resume-dir` reads `config.json` from the run directory and ignores
every other config flag. To change a value for the resumed run, edit
`config.json` in place first (typical case: extend `num_epochs` to keep
training longer).

The newest `epochs/epoch_NNN.pt` is the resume point. Training continues
in the same directory; the metrics history is preserved and appended.

## User-side analysis (no auto plots)

```python
import json, matplotlib.pyplot as plt
history = json.loads(open("sr/runs/srcnn3d/<run>/metrics.json").read())
epochs = [h["epoch"] for h in history]
plt.plot(epochs, [h["train_loss"] for h in history], label="train")
plt.plot(epochs, [h.get("val_masked_mse") for h in history], label="val")
plt.xlabel("epoch"); plt.legend(); plt.show()
```

To inspect any saved epoch:

```python
import torch
state = torch.load("sr/runs/srcnn3d/<run>/epochs/epoch_007.pt", map_location="cpu")
print(state["best_epoch_number"], state["best_val_loss"])
print(state["metrics_history"][-1])
```

## Why each design choice exists

- **Per-epoch full state**: every checkpoint is a resumable, self-contained
  snapshot. The "last successfully written epoch" is always the resume
  point, even after SIGKILL.
- **No auto post-training analysis**: keeps the trainer focused and lets
  the user compose comparisons across runs without baked-in assumptions.
- **Registries for losses/models/optimizers/schedulers**: swap behaviour
  by name + JSON kwargs. No need to read the training loop to extend it.
- **Maximum metric tracking**: validation logs every loss + every metric
  every epoch, so cross-run comparisons don't depend on which loss was
  optimized.
- **Explicit config**: every value driving a run lands in `config.json`.
  Resume refuses to mix CLI overrides with a saved config so the
  reproduction story stays simple.
