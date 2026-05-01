## `src/sr` — Super-resolution Training Stack

## High-level overview

`src/sr` is the experiment and training layer for 3D super-resolution models.  
It consumes manifest-backed datasets from `src/data`, builds a selected model via registry/factory logic, runs safety checks, and executes reproducible training with checkpointing and TensorBoard logging.

This module is intentionally model-pluggable: training orchestration is shared, while architecture choice is configuration-driven.

---

## Scope and boundaries

`src/sr` is responsible for:
- model selection and instantiation
- run configuration and reproducibility policy
- dataloader construction for SR training
- sanity/overfit checks
- training/validation loop execution
- checkpoint and run artifact management

`src/sr` is not responsible for raw fMRI preprocessing; that is handled in `src/data`.

---

## Package structure

- `config.py`  
  Central defaults and validation (`DEFAULT_CONFIG`, deterministic setup, device selection).

- `model.py`  
  Model implementations and model registry/factory (`MODEL_REGISTRY`, `select_model`, `build_model_from_config`).

- `data.py`  
  Adapter layer from `src.data.datasets.SpatialSRDataset` to SR trainer tensor format.

- `checks.py`  
  Fast correctness checks (`run_sanity_checks`, `run_tiny_overfit_check`).

- `training.py`  
  End-to-end training orchestration: epoch loops, metrics, scheduler, resume, checkpointing.

- `__init__.py`  
  Public package exports.

---

## Data flow through `src/sr`

1. Configuration is built from defaults + CLI overrides.
2. Config is validated for ranges, model name, geometry, and manifest availability.
3. Data loaders are created from manifest-backed `SpatialSRDataset`.
4. Model is instantiated from the configured `model_name`.
5. Optional checks are run (sanity and tiny overfit).
6. Training loop runs with train/validation metrics and checkpoint persistence.

The canonical runtime entrypoint is `run.py` at repository root.

---

## Configuration contract

Important keys in `DEFAULT_CONFIG`:

- training core: `batch_size`, `num_epochs`, `learning_rate`, `train_split`
- reproducibility: `seed`, `deterministic`, `num_workers`
- data source: `manifest_path`
- spatial degradation: `source_voxel_mm`, `target_voxel_mm`
- model: `model_name`, `model_kwargs`
- run management: `run_root`, `checkpoint_interval`, `resume_checkpoint`
- safety: `strict_finite_loss`

Validation enforces:
- numeric ranges and positivity constraints
- known model name in registry
- SR geometry compatibility (`output_patch_shape` > `input_patch_shape`)
- manifest path existence

---

## Model registry and future multi-model support

`model.py` uses a registry pattern:

- `MODEL_REGISTRY` maps a name (`str`) to constructor (`Callable[..., nn.Module]`)
- `select_model(name, **kwargs)` resolves and builds a model
- `build_model_from_config(config)` is the standardized factory entrypoint

To add a new model:
1. implement `nn.Module`
2. register it in `MODEL_REGISTRY`
3. pass `model_name` (and optional `model_kwargs`) via config/CLI

No trainer rewrite should be required for additional models with compatible input/output contracts.

---

## Safety and reproducibility features

- Explicit deterministic backend policy support
- Seeded split and DataLoader generators
- Optional finite-loss fail-fast (`strict_finite_loss`)
- Atomic checkpoint writes (`.tmp` then replace)
- Run configuration persisted to disk for traceability

`checks.py` should be run before long experiments:
- `run_sanity_checks`: verifies one forward/backward update path
- `run_tiny_overfit_check`: verifies loss can decrease on a tiny sample

---

## Training outputs

For each run, `training.py` writes one directory under `src/sr/runs/<model_name>/<timestamp>`:
- serialized effective config (`config.json`)
- TensorBoard logs in run directory (`tb/`)
- per-epoch subdirectories under `epochs/` (for periodic checkpoints)
- `best.pt` and `final.pt` in the run directory

Run directory naming:
- `model_name` as first-level folder
- `<timestamp>` (`YYYYMMDD_HHMMSS`) as second-level folder
- Example: `src/sr/runs/srcnn3d/20260501_145500`

Only model name and timestamp are encoded in paths. All run details and hyperparameters are stored in `config.json`.

This supports reproducibility, restartability, and offline analysis.

---

## Available data for SR

`src/sr` does not read ad-hoc `.npy` lists. It consumes the manifest-driven data pipeline from `src/data`.

Required input artifact:
- `manifest.json` that already contains per-run metadata from `src.data.compute_metadata`

Required manifest fields (used by SR data adapter):
- global: `bids_root`, `derivatives_dir`, `target_shape`, `runs`
- per run: `run_id`, `subject`, `path`, `n_volumes`, `norm_ref`, `mask_path`

Splitting behavior:
- by default, subjects are shuffled deterministically by seed and split with `train_split`
- optional fixed split via `train_subjects` / `val_subjects` in config

---

## CLI parameter reference (`run.py`)

Common arguments:
- positional `command`: `sanity`, `overfit`, `checks`, `train`
- `--manifest-path`: path to enriched manifest
- `--model-name`: architecture key from model registry (`srcnn3d`, `rcan3d`)
- `--device`: force `cpu` or `cuda` (otherwise auto)

Training/control:
- `--epochs`: total epochs for `train`
- `--batch-size`: mini-batch size
- `--lr`: learning rate
- `--train-split`: ratio for train subjects (`0..1`)
- `--num-workers`: DataLoader workers
- `--log-interval`: print frequency (batches)
- `--checkpoint-interval`: epochs between periodic checkpoint saves
- `--resume-checkpoint`: checkpoint path to continue training

Reproducibility/safety:
- `--seed`: global seed
- `--deterministic` / `--no-deterministic`: backend determinism toggle
- `--strict-finite-loss` / `--no-strict-finite-loss`: fail on NaN/Inf losses

Shape/degradation:
- `--input-shape D H W`: expected LR patch shape
- `--output-shape D H W`: expected HR patch shape
- `source_voxel_mm` / `target_voxel_mm` are config-level values in `DEFAULT_CONFIG`

---

## Usage from repository root

```bash
python run.py sanity --manifest-path ./manifest.json
python run.py overfit --overfit-steps 20 --manifest-path ./manifest.json
python run.py checks --overfit-steps 20 --manifest-path ./manifest.json
python run.py train --epochs 20 --model-name srcnn3d --manifest-path ./manifest.json
```

`manifest.json` must already be enriched by `src.data.compute_metadata`.

---

## Professional development guidelines for this module

- Keep training code model-agnostic; avoid hardcoded architecture branches.
- Keep configuration explicit and validated early.
- Make all failure modes actionable (clear exceptions and context).
- Preserve deterministic behavior for comparable experiments.
- Add tests for each new model registration and config edge case.

This ensures `src/sr` remains maintainable and scientifically reliable as the number of supported models grows.
