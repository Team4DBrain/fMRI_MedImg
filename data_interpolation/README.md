# data_interpolation

3D U-Net for **fMRI temporal interpolation**. Given two BOLD volumes `V_t` and
`V_{t+2}`, the model predicts the missing middle volume `V_{t+1}`. Applied
repeatedly, this doubles the effective temporal resolution of a 4D BOLD run.

```
input  x = [V_t, V_{t+2}]   shape (2, D, H, W)
target y = V_{t+1}          shape (1, D, H, W)
```

## Repo layout

```
data_interpolation/
├── train.py              # config-driven training entry point
├── eval.py               # held-out evaluation (model vs naive baseline)
├── main.py               # inference: take a BOLD file, generate interpolated NIfTI
├── configs/default.yaml  # all hyperparameters
├── src/
│   ├── dataset.py        # lazy fMRI triplet dataset
│   ├── model.py          # 3D U-Net
│   ├── loss.py           # hybrid L1 + 3D SSIM
│   └── utils.py          # device, seed, config helpers
├── notebooks/
│   └── quick_test.ipynb  # end-to-end smoke test: train → save → infer → plot
├── checkpoints/
│   └── pretrained/       # shipped weights (~90 MB) + history.json
├── data/                 # (gitignored) drop your BOLD files here
└── results/              # (gitignored) eval metrics, generated NIfTI, figures
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.1+ recommended. CUDA (H100/A100), Apple MPS, and
CPU are all supported.

## Quick smoke test

Verify the pipeline runs end-to-end before any real training:

```bash
jupyter notebook notebooks/quick_test.ipynb
```

The notebook trains for 1 epoch on ~8 triplets, saves a checkpoint, reloads
it, runs inference on a held-out frame, and shows the comparison figure. Takes
a few minutes on a laptop.

## Training

Drop a BIDS-style folder (or just a few `*_bold.nii.gz` files) into `data/`,
then:

```bash
python train.py --config configs/default.yaml
```

Override any config key from the CLI:

```bash
python train.py --config configs/default.yaml \
    train.epochs=50 train.lr=5e-5 train.batch_size=4 \
    checkpoint.dir=checkpoints/my_run
```

Outputs (under `checkpoint.dir`):

| File | Purpose |
| --- | --- |
| `last.pt` | full resume state (model + optimizer + scheduler + history) |
| `best.pt` | best-validation full state |
| `model_weights.pt` | last-epoch weights only (use this for inference) |
| `best_weights.pt` | best-validation weights only |
| `history.json` | per-epoch train/val loss |

To resume:

```bash
python train.py --config configs/default.yaml checkpoint.resume=checkpoints/my_run/last.pt
```

## Evaluation

```bash
python eval.py \
    --weights checkpoints/pretrained/model_weights.pt \
    --file    data/sub-01_ses-00_task-X_bold.nii.gz \
    --output-dir results/eval_pretrained \
    --history checkpoints/pretrained/history.json
```

Writes `metrics.csv`, `metrics.json`, and (when `--history` is given) a
`loss_curve.png`. The summary line prints whether the model beats the naive
midpoint baseline `0.5 * (V_t + V_{t+2})` across the held-out frames.

## Inference — generate new data

`main.py` runs the trained model over a BOLD file and writes a new NIfTI with
synthetic frames inserted:

```bash
python main.py \
    --weights checkpoints/pretrained/model_weights.pt \
    --input  data/sub-01_ses-00_task-X_bold.nii.gz \
    --output results/sub-01_ses-00_task-X_bold_2x.nii.gz
```

Modes:

- `--mode insert` (default): output `2T - 1` frames — every original frame
  plus one interpolated frame between each pair. The output header's TR
  (`pixdim[4]`) is halved.
- `--mode fill-gaps`: output `T - 1` frames — only the synthetic frames.

Match `--norm-mode` and `--residual` to whatever was used at training time.
The shipped pretrained checkpoint was trained with `norm-mode=zscore` and
**without** residual, so the defaults are correct.

## Pretrained weights

`checkpoints/pretrained/model_weights.pt` (~90 MB) is the best full-training
model trained on `ds002685_ses-00`. See `checkpoints/pretrained/README.md`
for architecture/training details.

## Notes

- Spatial shape is preserved by the U-Net's size-aware decoder, so the model
  works on any BOLD volume — the training data was `(D=84, H=128, W=128)`.
- The dataset reads NIfTI files lazily via `nibabel.dataobj`; nothing is
  preprocessed to disk.
- File-level train/val split prevents temporal leakage when multiple BOLD
  files are present.
