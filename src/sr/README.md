## `src/sr` - Spatial Super-Resolution Pipeline

This package trains and runs 3D super-resolution models that reconstruct a
high-resolution (HR) fMRI volume from a simulated low-resolution (LR) volume.
Two model backbones (`srcnn3d`, `rcan3d`) share one common data and training
pipeline.

## Models

- **`srcnn3d`**: Trilinear upsample to HR, then a shallow 3D SRCNN stack. Default
  in [`config.py`](config.py) because it is cheaper in memory and time while the
  pipeline is validated.
- **`rcan3d`**: 3D adaptation of the RCAN topology from the reference 2D
  implementation (residual channel-attention blocks inside **residual groups**,
  then a final body convolution, then a **global skip** from head features—see
  [`model.py`](model.py)). LR→HR sizing uses **trilinear** interpolation to
  `output_patch_shape`, not learnable PixelShuffle, which matches fixed HR patch
  geometry from the dataset. It is **not** weight-compatible with ~/RCAN
  checkpoints. Constructor knobs include `n_feats`, `n_resgroups`,
  `n_resblocks`, and `reduction` (passed via `model_kwargs` in config).

## End-to-end program flow

The CLI entrypoint is `python -m src.sr.run <command>`.

1. Build config from defaults + CLI overrides (`config.py`, `run.py`).
2. Validate config to fail fast on bad values (`validate_config`).
3. Set seed and deterministic backend policy for reproducibility.
4. Build train/val loaders (`data.py` -> `SpatialSRDataset`).
5. Build model from the registry (`model.py`).
6. Execute one command path:
   - `train`: optimize model and save artifacts/checkpoints.
   - `eval`: load checkpoint and report masked metrics.
   - `infer`: run one sample and optionally save prediction.
   - `plot-loss`: render loss curve from run history.

## Why each training step exists

- **Manifest-driven loading**: keeps paths, shapes, and metadata centralized and
  consistent.
- **Spatial degradation (HR -> LR)**: creates realistic LR inputs via k-space
  truncation so SR learning matches the target acquisition scenario.
- **Mask-aware loss/metrics**: focuses optimization/evaluation on in-brain
  voxels instead of background.
- **Train/val subject split**: supports cleaner validation and reduces leakage
  risk when split mode is enabled.
- **Checkpointing + logs**: enables resume, model selection (`best.pt`), and
  reproducible experiment tracking.

## Scope

- single spatial SR task
- model registry with `srcnn3d` and `rcan3d`
- canonical dataset contract from `src.data.datasets.SpatialSRDataset`
- mask-aware MSE/PSNR/SSIM reporting
- reproducible splits and run artifacts

## Module map

- `config.py`: defaults, seed/device helpers, config validation
- `model.py`: `SRCNN3D`, `RCAN3D`, registry/factory
- `data.py`: subject splitting and DataLoader creation
- `training.py`: losses, metrics, training loop, checkpointing, loss plotting
- `run.py`: CLI parser and command execution

## Data contract

`SpatialSRDataset` returns dictionaries with:

- `input`: LR volume `(1, kx, ky, kz)`
- `target`: HR volume `(1, X, Y, Z)`
- `mask_hr`: HR brain mask `(1, X, Y, Z)`
- `mask_lr`: LR brain mask `(1, kx, ky, kz)`
- `run_id`, `t`: sample metadata

Training/eval use `mask_hr` for masked MSE, derived PSNR, and masked local
3D SSIM.

## Shape behavior (important)

- `output_patch_shape` is actively used by the model forward pass for
  interpolation target size.
- `input_patch_shape` exists in default config but is currently not used to
  construct the dataset pipeline.
- LR input shape is determined by manifest HR shape plus voxel ratio
  (`source_voxel_mm` -> `target_voxel_mm`) through spatial degradation.

## CLI commands and parameters

### Commands

- `train`
- `eval`
- `infer`
- `plot-loss`

### Common options

- `--seed INT`
- `--batch-size INT`
- `--epochs INT`
- `--lr FLOAT`
- `--model-name {srcnn3d,rcan3d}`
- `--train-split FLOAT`
- `--num-workers INT`
- `--log-interval INT`
- `--checkpoint-interval INT`
- `--manifest-path PATH`
- `--run-root PATH`
- `--device {cpu,cuda}`
- `--deterministic` / `--no-deterministic`
- `--strict-finite-loss` / `--no-strict-finite-loss`

### Train-related options

- `--resume-checkpoint PATH`

### Eval/Infer options

- `--checkpoint-path PATH` (required for `eval` and `infer`)
- `--output-shape D H W`
- `--inference-index INT`
- `--save-output-npy PATH` (infer only)
- `--visualize` (infer only; show center-slice figure for input/prediction/target)
- `--visualize-output PATH` (infer only; save the visualization as PNG)
- `--visualize-direction {axial,coronal,sagittal}` (infer only; default `axial`)
- `--visualize-level FLOAT` (infer only; relative slice level in `[0,1]`, default `0.5`)
- `--eval-report PATH` (eval only; default `./eval_report.json`)

### Plot-loss options

- `--run-dir PATH` (required)
- `--plot-output PATH` (optional)

## Run artifacts

Each training run writes to `src/sr/runs/<model_name>/<timestamp>/`:

- `config.json` (effective config)
- `split.json` (resolved train/val subjects and sample counts)
- `tb/` TensorBoard logs
- `epochs/epoch_XXX/checkpoint.pt`
- `best.pt` (best validation checkpoint when validation exists)
- `final.pt`
- `metrics_summary.json` (final summary metrics)
- `metrics_history.json` (epoch-wise train/val loss + LR)
- `loss_curve.png`

## Typical usage

```bash
# Train
python -m src.sr.run train \
  --manifest-path ./manifest.json \
  --model-name srcnn3d \
  --batch-size 4 \
  --epochs 20 \
  --lr 1e-3

# Plot loss for a finished run
python -m src.sr.run plot-loss \
  --run-dir ./src/sr/runs/srcnn3d/<run>

# Evaluate a checkpoint
python -m src.sr.run eval \
  --manifest-path ./manifest.json \
  --checkpoint-path ./src/sr/runs/srcnn3d/<run>/best.pt \
  --eval-report ./eval_report.json

# Inference on one sample
python -m src.sr.run infer \
  --manifest-path ./manifest.json \
  --checkpoint-path ./src/sr/runs/srcnn3d/<run>/best.pt \
  --inference-index 0 \
  --save-output-npy ./prediction.npy \
  --visualize-output ./prediction_preview.png \
  --visualize-direction coronal \
  --visualize-level 0.3
```
