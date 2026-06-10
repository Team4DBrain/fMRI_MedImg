# Pretrained checkpoint

`model_weights.pt` — 3D U-Net trained for fMRI temporal interpolation.

## Architecture
- `UNet3D(in_channels=2, out_channels=1, base_channels=32, depth=4)`
- ~22M parameters

## Training data
- Source dataset: `ds002685`, session `ses-00`
- BIDS-style multi-file split, file-level train/val (val_fraction=0.1)
- Normalization: `zscore` (per-triplet, input-only stats)
- Loss: hybrid L1 + 3D-SSIM (alpha=0.5, data_range=2.0)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5), cosine LR schedule

## Training schedule
- 5 epochs, batch size 4, CUDA bf16 autocast
- Final epoch — train_loss=0.0665, val_loss=0.0641

See `history.json` for the full per-epoch curve.

## Loading
```python
import torch
from src.model import UNet3D

model = UNet3D(in_channels=2, out_channels=1, base_channels=32, depth=4)
model.load_state_dict(torch.load("checkpoints/pretrained/model_weights.pt", map_location="cpu"))
model.eval()
```

Or use `main.py`:
```bash
python main.py \
    --weights checkpoints/pretrained/model_weights.pt \
    --input  data/your_bold.nii.gz \
    --output results/your_bold_2x.nii.gz
```

## Notes
- Trained with `residual=False`. Inference must NOT pass `--residual`.
- Input expects `norm_mode=zscore` (the inference scripts default to this).
- Spatial shape was `(D=84, H=128, W=128)` during training; the U-Net adapts
  to other shapes via size-aware decoder interpolation.
