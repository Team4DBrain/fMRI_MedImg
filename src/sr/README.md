## `src/sr` — Spatial SR

`src/sr` is now focused on one task: **spatial super-resolution of a brain scan**.
Input is a simulated LR volume, output is an HR volume. Both `srcnn3d` and `rcan3d`
run on the same training/eval pipeline.

## Scope

- single SR task (spatial only)
- two model backbones via registry
- canonical data contract from `src.data.datasets.SpatialSRDataset`
- mask-aware training/evaluation in brain voxels
- reproducible train/val splits and run artifacts

## Module map

- `config.py`: defaults, seed/device, config validation
- `model.py`: `SRCNN3D`, `RCAN3D`, registry/factory
- `data.py`: train/val split and dataloader creation
- `training.py`: masked loss, training loop, checkpoints
- `run.py`: CLI commands `train`, `eval`, `infer`

## Data contract

The loader consumes the enriched `manifest.json` and returns dictionaries with:

- `input`: LR volume `(1, kx, ky, kz)`
- `target`: HR volume `(1, X, Y, Z)`
- `mask_hr`: HR brain mask `(1, X, Y, Z)`
- `mask_lr`: LR brain mask `(1, kx, ky, kz)`

Training and validation use `mask_hr` for masked MSE, PSNR (from MSE), and masked local SSIM (3D sliding window).

## Run artifacts

Each run writes to `src/sr/runs/<model_name>/<timestamp>/`:

- `config.json` (effective config)
- `split.json` (resolved train/val subjects + sample counts)
- `tb/` TensorBoard logs
- `epochs/epoch_XXX/checkpoint.pt`
- `best.pt`
- `final.pt`
- `metrics_summary.json` (final train/val MSE, PSNR, SSIM, best val MSE)

## Usage

```bash
python -m src.sr.run train --manifest-path ./manifest.json --model-name srcnn3d
python -m src.sr.run eval --manifest-path ./manifest.json --checkpoint-path ./src/sr/runs/srcnn3d/<run>/best.pt
# optional: --eval-report ./eval_report.json (default writes eval_report.json in cwd)
python -m src.sr.run infer --manifest-path ./manifest.json --checkpoint-path ./src/sr/runs/srcnn3d/<run>/best.pt --inference-index 0
```
