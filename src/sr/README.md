# `src/sr` — Spatial Super-Resolution

Minimal, modular 3D SR pipeline for fMRI volumes. Three commands, one
config dataclass, one checkpoint format. No automatic post-training
analysis — the user composes plots, comparisons, and reports themselves.

## Layout

```
src/sr/
├── config.py       # SRConfig dataclass, defaults, JSON IO, validate
├── models.py       # SRCNN3D, RCAN3D, MODEL_REGISTRY, build_model
├── losses.py       # mse, masked_mse, l1, masked_l1 + LOSS_REGISTRY
├── metrics.py      # psnr, masked/unmasked SSIM, compute_full_metrics
├── components.py   # OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
├── data.py         # build_loaders, subject split, seeded workers
├── checkpoint.py   # EpochState save/load, find_latest_epoch
├── train.py        # train(config, resume_dir=None)
├── infer.py        # evaluate, infer_one, list_samples, make_slice_figure
└── cli.py + __main__.py   # `python -m src.sr ...`
```

Each module owns one responsibility. To swap a loss, model, optimizer or
scheduler, add an entry to the corresponding registry and pass its name
via the CLI — no edits to `train.py` required.

## Defaults

| Field | Default |
|---|---|
| `manifest_path` | `/srv/venvs/team4dbrain/derivatives/manifest.json` |
| `run_root` | `src/sr/runs` |
| `model_name` | `srcnn3d` |
| `output_patch_shape` | `(128, 128, 93)` |
| `source_voxel_mm` -> `target_voxel_mm` | `1.5 -> 3.0` |
| `train_split` | `0.8` (train) / `0.2` (val), shuffled by `seed` |
| `loss_name` | `masked_mse` |
| `optimizer_name`, `learning_rate` | `adam`, `1e-3` |
| `scheduler_name`, `scheduler_kwargs` | `plateau`, `{"factor":0.5,"patience":3}` |
| `seed`, `deterministic`, `strict_finite_loss` | `42`, `true`, `true` |
| `batch_size`, `num_epochs`, `num_workers`, `log_interval` | `4`, `20`, `0`, `10` |
| `tensorboard` | `true` |

## Run artifacts

```
src/sr/runs/<model_name>/<timestamp>/
├── config.json         # written once at start of run, source of truth
├── split.json          # written once, resolved train/val subjects
├── metrics.json        # rewritten atomically every epoch (plain JSON)
├── tb/                 # TensorBoard scalars (if --tensorboard)
└── epochs/
    ├── epoch_001.pt    # full EpochState (model + opt + sched + RNG + history)
    ├── epoch_002.pt
    └── ...
```

`epoch_NNN.pt` is fully self-contained: kill the process at any point and
resume from the last successfully-written epoch with zero information
loss. There is no `final.pt`, no `best.pt`, no `metrics_summary.json` —
those are user-side compositions.

## CLI

```bash
# Train from scratch (uses every default)
python -m src.sr train

# Train with custom knobs
python -m src.sr train \
  --model-name rcan3d \
  --model-kwargs '{"n_feats": 48, "n_resgroups": 3}' \
  --loss-name masked_l1 \
  --optimizer-name adamw \
  --optimizer-kwargs '{"weight_decay": 1e-4}' \
  --scheduler-name cosine \
  --scheduler-kwargs '{"T_max": 20}' \
  --epochs 20 --batch-size 4 --lr 1e-3

# Resume an interrupted run (config flags are rejected; saved config wins)
python -m src.sr train --resume-dir src/sr/runs/srcnn3d/20260511_120000

# Evaluate a checkpoint on its saved val split
python -m src.sr eval \
  --checkpoint src/sr/runs/srcnn3d/<run>/epochs/epoch_010.pt \
  --report ./report.json

# List samples available in the manifest used by a checkpoint
python -m src.sr infer \
  --checkpoint src/sr/runs/srcnn3d/<run>/epochs/epoch_010.pt \
  --list-samples

# Infer one sample, save a slice figure
python -m src.sr infer \
  --checkpoint src/sr/runs/srcnn3d/<run>/epochs/epoch_010.pt \
  --subject 01 --session 00 --task ArchiStandard --direction ap --t 12 \
  --axis coronal --slice-level 0.4 \
  --save-png ./infer_preview.png \
  --save-npy ./infer_pred.npy
```

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
history = json.loads(open("src/sr/runs/srcnn3d/<run>/metrics.json").read())
epochs = [h["epoch"] for h in history]
plt.plot(epochs, [h["train_loss"] for h in history], label="train")
plt.plot(epochs, [h.get("val_masked_mse") for h in history], label="val")
plt.xlabel("epoch"); plt.legend(); plt.show()
```

To inspect any saved epoch:

```python
import torch
state = torch.load("src/sr/runs/srcnn3d/<run>/epochs/epoch_007.pt", map_location="cpu")
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
