"""eval.py — held-out evaluation of a trained checkpoint vs the naive baseline.

For every valid t in a BOLD file, computes L1 and PSNR for:

    naive baseline:   0.5 * (V_t + V_{t+2})
    model prediction: f(V_t, V_{t+2})

Writes results/metrics.csv, results/metrics.json, and results/loss_curve.png
(when --history is given).

Example:

    python eval.py \\
        --weights checkpoints/pretrained/model_weights.pt \\
        --file data/sub-01_ses-00_task-X_bold.nii.gz \\
        --output-dir results/eval_run
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import FMRIInterpolationDataset
from src.model import UNet3D
from src.utils import pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="Path to model_weights.pt or a full checkpoint.")
    p.add_argument("--file", required=True, help="A single *_bold.nii.gz file to evaluate over.")
    p.add_argument("--t-start", type=int, default=0)
    p.add_argument("--t-end", type=int, default=None, help="Inclusive cap on t. Default: all valid t.")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--norm-mode", choices=["zscore", "percentile"], default="zscore")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--residual", action="store_true", help="Use residual inference (output = naive + model).")
    p.add_argument("--output-dir", default="results/eval")
    p.add_argument("--history", default=None, help="Optional history.json to plot the training loss curve.")
    return p.parse_args()


def load_weights(weights_path: str, model: torch.nn.Module, device: torch.device) -> None:
    """Load a state_dict from either model_weights.pt or a full checkpoint."""
    obj = torch.load(weights_path, map_location=device)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state)


def psnr(mse: float, data_range: float = 2.0) -> float:
    """Peak signal-to-noise ratio in dB, given MSE and assumed data range."""
    if mse <= 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FMRIInterpolationDataset(file_list=[args.file], norm_mode=args.norm_mode)
    model = UNet3D(in_channels=2, out_channels=1,
                   base_channels=args.base_channels, depth=args.depth).to(device)
    load_weights(args.weights, model, device)
    model.eval()

    rows = []
    t_end = args.t_end if args.t_end is not None else len(dataset)
    data_range = 1.0 if args.norm_mode == "percentile" else 2.0
    with torch.no_grad():
        for idx in range(args.t_start, min(t_end, len(dataset))):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            y = sample["y"].unsqueeze(0).to(device)
            raw = model(x)
            pred = (0.5 * (x[:, 0:1] + x[:, 1:2]) + raw) if args.residual else raw
            naive = 0.5 * (x[:, 0:1] + x[:, 1:2])
            row = {
                "t": sample["t"],
                "model_l1": float((pred - y).abs().mean().cpu()),
                "naive_l1": float((naive - y).abs().mean().cpu()),
                "model_mse": float(((pred - y) ** 2).mean().cpu()),
                "naive_mse": float(((naive - y) ** 2).mean().cpu()),
            }
            row["model_psnr"] = psnr(row["model_mse"], data_range)
            row["naive_psnr"] = psnr(row["naive_mse"], data_range)
            rows.append(row)

    # Aggregate.
    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    summary = {
        "n_samples": len(rows),
        "file": args.file,
        "weights": args.weights,
        "mean_model_l1": mean("model_l1"),
        "mean_naive_l1": mean("naive_l1"),
        "mean_model_psnr": mean("model_psnr"),
        "mean_naive_psnr": mean("naive_psnr"),
        "model_beats_naive_pct": float(
            np.mean([r["model_l1"] < r["naive_l1"] for r in rows]) * 100
        ) if rows else 0.0,
    }

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_t": rows}, f, indent=2)

    print("=== Evaluation summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Optional training-loss-curve plot.
    if args.history:
        try:
            import matplotlib.pyplot as plt
            with open(args.history, "r", encoding="utf-8") as f:
                history = json.load(f)
            epochs = [h["epoch"] for h in history]
            train = [h.get("train_loss") for h in history]
            val = [h.get("val_loss") for h in history]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, train, label="train")
            if any(v is not None for v in val):
                ax.plot(epochs, val, label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "loss_curve.png", dpi=120)
            plt.close(fig)
            print(f"  loss_curve: {out_dir / 'loss_curve.png'}")
        except Exception as exc:
            print(f"  (skipped loss_curve.png: {exc})")


if __name__ == "__main__":
    main()
