# Super-Resolution Training Process (`sr.ipynb`)

This document explains the full training workflow used in `sr.ipynb`.

## 1) Goal of the training

The notebook trains a 3D super-resolution model that takes a **low-resolution patch** as input and predicts a **higher-resolution patch** as output.

- Input patch size: `33 x 33 x 33`
- Output patch size: `50 x 50 x 50`

The configuration enforces this super-resolution requirement:

- `output_patch_shape` must be larger than `input_patch_shape` in every dimension.

---

## 2) Setup and configuration

At the start, the notebook:

1. Imports all required libraries (`torch`, `numpy`, `tensorboard`, etc.).
2. Defines `set_seed(seed)` to make runs reproducible.
3. Builds a central `CONFIG` dictionary with hyperparameters and file paths.
4. Selects device (`cuda` if available, otherwise `cpu`).

Important config fields:

- `batch_size`, `num_epochs`, `learning_rate`
- `manifest_path` (produced by `src.data.manifest` and enriched by `src.data.compute_metadata`)
- `input_patch_shape`, `output_patch_shape`
- `train_split` and loader settings
- checkpoint/log paths

---

## 3) Model architecture (factory-selected)

The training/check flows build models through `build_model_from_config(config)`.
Supported model names today:

- `srcnn3d`
- `rcan3d`

The default config still uses `SRCNN3D`.

### Forward path

1. **Upsampling first**  
   Input is upsampled to `output_patch_shape` with trilinear interpolation.

2. **Convolutional refinement**  
   Three 3D conv layers are applied:
   - Conv1: `1 -> 64`, kernel `9`, padding `4`
   - Conv2: `64 -> 32`, kernel `1`
   - Conv3: `32 -> 1`, kernel `5`, padding `2`

3. **Activation**  
   ReLU after Conv1 and Conv2.

Because padding is used, the network keeps spatial size after upsampling, so prediction shape matches the HR target shape.

### Weight initialization

`_initialize_weights()` initializes conv weights with small normal values and zero biases for stable training start.

---

## 4) Data pipeline and sample generation

The SR loader path is an adapter around `src.data.datasets.SpatialSRDataset`.

### Input data format

It expects a manifest describing BIDS runs plus derived metadata (`norm_ref`, `mask_path`, `target_shape`).

### Sample creation (`__getitem__`)

For each sample:

1. Select a run and time index from manifest-backed samples.
2. Load the HR frame and normalize using per-run `norm_ref`.
3. Create LR input using spatial degradation (`make_spatial_degradation`).
4. Return tensors with channel dimension:
   - `x_tensor`: LR volume `[1, d, h, w]`
   - `y_tensor`: HR target `[1, D, H, W]`

This gives supervised LR->HR training pairs.

---

## 5) DataLoaders, split, and deterministic seeding

`create_dataloaders(config)`:

1. Reads subjects from manifest and builds deterministic subject split by `train_split`.
2. Instantiates train/validation `SpatialSRDataset` instances with subject filters.
3. Creates `DataLoader`s with seeded generators:
   - train loader: shuffled
   - val loader: not shuffled
4. Returns loaders and dataset size.

---

## 6) Loss, optimizer, scheduler

The notebook uses:

- **Loss**: Mean Squared Error (`MSELoss`)
- **Optimizer**: Adam
- **LR Scheduler**: `ReduceLROnPlateau`

It also computes PSNR from MSE for a more interpretable quality metric.

---

## 7) Validation and training steps

### `validate_one_epoch(...)`

- Puts model in eval mode.
- Disables gradients.
- Runs all validation batches.
- Returns average validation loss.

### `train_one_epoch(...)`

- Puts model in train mode.
- For each batch:
  - forward pass
  - MSE loss
  - backpropagation
  - optimizer step
- Logs batch loss to TensorBoard.
- Returns average training loss.

---

## 8) Checkpointing and resume

### `save_checkpoint(...)`

Saves (atomically via `*.tmp` write + replace):

- epoch
- model weights
- optimizer state
- best validation loss
- serialized config

### `maybe_resume_training(...)`

If `resume_checkpoint` is set, it loads model and optimizer states and returns:

- `start_epoch`
- `best_val_loss`

This allows continuing interrupted training runs.

---

## 9) Full training orchestration (`run_training`)

`run_training(config)` is the end-to-end training controller:

1. Creates run directory structure as `runs/<model_name>/<timestamp>`.
2. Saves config JSON for reproducibility.
3. Builds loaders and prints dataset stats.
4. Creates TensorBoard writer.
5. Optionally resumes from checkpoint.
6. For each epoch:
   - run train epoch
   - run validation epoch (if available)
   - step LR scheduler
   - fail fast on non-finite losses when `strict_finite_loss=True`
   - log train/val loss, LR, PSNR
   - save periodic checkpoint
   - update/save best checkpoint
7. Saves final checkpoint and closes writer.

Outputs:

- TensorBoard logs under `runs/<model_name>/<timestamp>/tb`
- Periodic checkpoints under `runs/<model_name>/<timestamp>/epochs/epoch_XXX/checkpoint.pt`
- Best/final checkpoints under `runs/<model_name>/<timestamp>/best.pt` and `final.pt`

---

## 10) Safety checks before long training

### `run_sanity_checks(config)`

Runs one real training step and checks:

- tensor dimensions
- channel count
- output-label shape consistency

Purpose: fail fast if pipeline shapes are wrong.

### `run_tiny_overfit_check(config, steps=20)`

Trains a tiny model on one sample for a few steps.

Purpose: verify the setup can overfit a trivial case.  
If loss does not decrease, this is a strong warning that data/model wiring may be incorrect.

---

## 11) Typical execution order

Recommended notebook order:

1. Run imports and config.
2. Run model definition and shape test.
3. Run dataset/dataloader cell.
4. Run training utilities cell.
5. Run:
   - `run_sanity_checks(CONFIG)`
   - `run_tiny_overfit_check(CONFIG, steps=20)`
6. Start full run with:
   - `run_training(CONFIG)`

---

## 12) What this process gives you

After successful training, you get:

- a trained SR model (`best.pt` and `final.pt`)
- reproducible run metadata (`config.json`)
- training curves in TensorBoard (loss, PSNR, LR)

This makes the workflow usable for both experimentation and repeatable model development.
