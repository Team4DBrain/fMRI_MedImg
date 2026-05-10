"""Expose a CLI for training, evaluating, inferring, and plotting SR runs.

Purpose:
    Provide one command interface that resolves config, validates inputs, and
    dispatches to the correct SR workflow.
Effects:
    Determines which execution path runs and what artifacts/reports are written.
Influences:
    Behavior depends on CLI flags, defaults in `src.sr.config`, and checkpoint
    metadata for eval/infer.
How to change safely:
    Keep CLI arguments, config override mapping, and command dispatch logic
    synchronized with README usage docs and checkpoint expectations.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

# Allow both invocation styles:
# - python -m src.sr.run
# - python ./src/sr/run.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.data.degradation_spatial import make_spatial_degradation
from src.sr import (
    DEFAULT_CONFIG,
    LOSS_NAMES,
    build_model_from_config,
    get_device,
    run_training,
    set_seed,
    validate_config,
)
from src.sr.data import create_dataloaders
from src.sr.training import masked_local_ssim_3d, masked_mse_loss, psnr_from_mse, write_loss_curve_png


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SR training/evaluation/inference.")
    parser.add_argument(
        "command",
        choices=["train", "eval", "infer", "plot-loss"],
        help="What to run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--loss-name",
        choices=LOSS_NAMES,
        default=None,
        help=(
            "Training objective. masked_mse keeps the original brain-masked MSE; "
            "mse/l1 are unmasked whole-volume losses; masked_l1 is brain-masked MAE."
        ),
    )
    parser.add_argument(
        "--model-name",
        choices=["srcnn3d", "rcan3d"],
        default=None,
        help="Model architecture to train/check/infer.",
    )
    parser.add_argument("--train-split", type=float, default=None, help="Train split in [0,1].")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--log-interval", type=int, default=None, help="Batch log frequency.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Epoch checkpoint frequency.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Path to manifest.json produced by src.data.manifest/compute_metadata.",
    )
    parser.add_argument("--run-root", type=Path, default=None, help="Directory for run logs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory for plot-loss (contains metrics_history.json).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional PNG output path for plot-loss (default: <run-dir>/loss_curve.png).",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Model checkpoint path to load for eval/infer.",
    )
    parser.add_argument(
        "--output-shape",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=None,
        help="Output patch shape, e.g. --output-shape 128 128 128",
    )
    parser.add_argument(
        "--inference-index",
        type=int,
        default=0,
        help="Sample index from dataset for inference.",
    )
    parser.add_argument(
        "--save-output-npy",
        type=Path,
        default=None,
        help="Optional path to save predicted output volume as .npy.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize center slice(s) for input/prediction/target during infer.",
    )
    parser.add_argument(
        "--visualize-output",
        type=Path,
        default=None,
        help="Optional PNG path for infer visualization. If omitted, plot is shown interactively.",
    )
    parser.add_argument(
        "--visualize-direction",
        choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Slice direction for infer visualization (default: axial).",
    )
    parser.add_argument(
        "--visualize-level",
        type=float,
        default=0.5,
        help="Relative slice level in [0,1] for infer visualization (default: 0.5 center).",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=None,
        help="Optional JSON path for eval metrics (default: eval_report.json in cwd).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force compute device. Default: auto-detect.",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Enable deterministic behavior for reproducibility.",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic behavior.",
    )
    parser.set_defaults(deterministic=None)
    parser.add_argument(
        "--strict-finite-loss",
        dest="strict_finite_loss",
        action="store_true",
        help="Fail fast on non-finite losses.",
    )
    parser.add_argument(
        "--no-strict-finite-loss",
        dest="strict_finite_loss",
        action="store_false",
        help="Allow non-finite losses (not recommended).",
    )
    parser.set_defaults(strict_finite_loss=None)
    return parser


def _apply_overrides(args: argparse.Namespace) -> dict:
    config = deepcopy(DEFAULT_CONFIG)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.loss_name is not None:
        config["loss_name"] = args.loss_name
    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.train_split is not None:
        config["train_split"] = args.train_split
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.log_interval is not None:
        config["log_interval"] = args.log_interval
    if args.checkpoint_interval is not None:
        config["checkpoint_interval"] = args.checkpoint_interval
    if args.manifest_path is not None:
        config["manifest_path"] = args.manifest_path
    if args.run_root is not None:
        config["run_root"] = args.run_root
    if args.resume_checkpoint is not None:
        config["resume_checkpoint"] = args.resume_checkpoint
    if args.output_shape is not None:
        config["output_patch_shape"] = tuple(args.output_shape)
    if args.deterministic is not None:
        config["deterministic"] = args.deterministic
    if args.strict_finite_loss is not None:
        config["strict_finite_loss"] = args.strict_finite_loss

    return config


def _print_effective_config(config: dict, command: str, device: str) -> None:
    print(f"[run] Command: {command}")
    print(f"[run] Device: {device}")
    print(
        "[run] Config: "
        f"seed={config['seed']} batch_size={config['batch_size']} epochs={config['num_epochs']} "
        f"lr={config['learning_rate']} train_split={config['train_split']} "
        f"loss={config['loss_name']} "
        f"model={config['model_name']} deterministic={config['deterministic']} "
        f"output_shape={config['output_patch_shape']}"
    )
    print(f"[run] Manifest: {config['manifest_path']}")


def _load_checkpoint_config(checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Invalid checkpoint config payload in: {checkpoint_path}")
    return checkpoint_config


def _merge_inference_config(config: dict, args: argparse.Namespace, device: str) -> dict:
    if args.checkpoint_path is None:
        raise ValueError("--checkpoint-path is required for commands 'eval' and 'infer'.")
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_config = _load_checkpoint_config(checkpoint_path, device=device)
    merged = deepcopy(config)

    for key in (
        "model_name",
        "model_kwargs",
        "output_patch_shape",
        "source_voxel_mm",
        "target_voxel_mm",
        "manifest_path",
    ):
        if key in checkpoint_config:
            merged[key] = checkpoint_config[key]

    # CLI values must win over checkpoint config.
    cli_overrides = _apply_overrides(args)
    for key, value in cli_overrides.items():
        if value != DEFAULT_CONFIG[key]:
            merged[key] = value

    merged["manifest_path"] = Path(merged["manifest_path"])
    merged["output_patch_shape"] = tuple(merged["output_patch_shape"])
    merged["checkpoint_path"] = checkpoint_path
    return merged


def _build_eval_loader(config: dict):
    _, val_loader, _, _ = create_dataloaders(config)
    return val_loader


def _run_eval(config: dict, device: str, eval_report: Path | None) -> None:
    model = build_model_from_config(config).to(device)
    ckpt_path = config["checkpoint_path"]
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loader = _build_eval_loader(config)
    running_mse = 0.0
    running_ssim = 0.0
    n_batches = max(1, len(val_loader))
    with torch.no_grad():
        for batch in val_loader:
            input_tensor = batch["input"].to(device)
            target_tensor = batch["target"].to(device)
            mask_hr = batch["mask_hr"].to(device)
            pred = model(input_tensor)
            running_mse += masked_mse_loss(pred, target_tensor, mask_hr).item()
            running_ssim += masked_local_ssim_3d(pred, target_tensor, mask_hr).item()
    mean_mse = running_mse / n_batches
    mean_ssim = running_ssim / n_batches
    print(f"[eval] checkpoint={ckpt_path}")
    print(f"[eval] masked_mse={mean_mse:.6f}")
    print(f"[eval] masked_psnr={psnr_from_mse(mean_mse):.2f}")
    print(f"[eval] masked_ssim={mean_ssim:.4f}")
    report_path = eval_report if eval_report is not None else Path("eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "manifest_path": str(config["manifest_path"]),
        "masked_mse": mean_mse,
        "masked_psnr": psnr_from_mse(mean_mse),
        "masked_ssim": mean_ssim,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval] wrote {report_path}")


def _run_infer(config: dict, args: argparse.Namespace, device: str) -> None:
    checkpoint_path = config["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    degrade_fn = make_spatial_degradation(
        source_voxel_mm=float(config["source_voxel_mm"]),
        target_voxel_mm=float(config["target_voxel_mm"]),
    )
    from src.data.datasets import SpatialSRDataset

    dataset = SpatialSRDataset(manifest_path=Path(config["manifest_path"]), subject_filter=None, degrade_fn=degrade_fn)
    if len(dataset) == 0:
        raise RuntimeError("No samples available for inference.")
    if args.inference_index < 0 or args.inference_index >= len(dataset):
        raise IndexError(f"--inference-index out of range: {args.inference_index} (dataset size: {len(dataset)})")

    sample = dataset[args.inference_index]
    input_tensor = sample["input"].unsqueeze(0).to(device)
    target_tensor = sample["target"]

    with torch.no_grad():
        prediction_tensor = model(input_tensor).squeeze(0).cpu()

    prediction = prediction_tensor.squeeze(0).numpy()
    input_volume = input_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    target_volume = target_tensor.squeeze(0).detach().cpu().numpy()

    print(f"[inference] Loaded checkpoint: {checkpoint_path}")
    print(f"[inference] Model: {config['model_name']}")
    print(f"[inference] Sample index: {args.inference_index}/{len(dataset) - 1}")
    print(f"[inference] Input shape: {input_volume.shape}")
    print(f"[inference] Prediction shape: {prediction.shape}")
    print(f"[inference] Target shape: {target_volume.shape}")

    if args.save_output_npy is not None:
        args.save_output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_output_npy, prediction)
        print(f"[inference] Saved output to: {args.save_output_npy}")

    if args.visualize or args.visualize_output is not None:
        _visualize_inference(
            input_volume=input_volume,
            prediction_volume=prediction,
            target_volume=target_volume,
            output_path=args.visualize_output,
            show_interactive=bool(args.visualize and args.visualize_output is None),
            direction=args.visualize_direction,
            level=args.visualize_level,
        )
    m = sample["mask_hr"].unsqueeze(0)
    pred_b = prediction_tensor.unsqueeze(0)
    tgt_b = target_tensor.unsqueeze(0)
    mse = float(masked_mse_loss(pred_b, tgt_b, m).item())
    ssim = float(masked_local_ssim_3d(pred_b, tgt_b, m).item())
    print(f"[infer] masked_mse={mse:.6f} masked_psnr={psnr_from_mse(mse):.2f} masked_ssim={ssim:.4f}")


def _visualize_inference(
    input_volume: np.ndarray,
    prediction_volume: np.ndarray,
    target_volume: np.ndarray,
    output_path: Path | None,
    show_interactive: bool,
    direction: str,
    level: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --visualize/--visualize-output but is not installed."
        ) from exc

    if not 0.0 <= level <= 1.0:
        raise ValueError(f"--visualize-level must be in [0,1], got {level}.")

    direction_to_axis = {"sagittal": 0, "coronal": 1, "axial": 2}
    if direction not in direction_to_axis:
        raise ValueError(
            f"Unknown --visualize-direction '{direction}'. "
            f"Expected one of: {', '.join(direction_to_axis)}"
        )

    def _extract_slice(volume: np.ndarray) -> tuple[np.ndarray, int]:
        axis = direction_to_axis[direction]
        dim = int(volume.shape[axis])
        idx = min(dim - 1, max(0, int(round(level * (dim - 1)))))
        if direction == "axial":
            img = volume[:, :, idx]
        elif direction == "coronal":
            img = volume[:, idx, :]
        else:  # sagittal
            img = volume[idx, :, :]
        return img, idx

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    input_slice, input_idx = _extract_slice(input_volume)
    pred_slice, pred_idx = _extract_slice(prediction_volume)
    target_slice, target_idx = _extract_slice(target_volume)
    panels = (
        ("Input (LR)", input_slice, input_idx),
        ("Prediction (HR)", pred_slice, pred_idx),
        ("Target (HR)", target_slice, target_idx),
    )
    for ax, (title, image, idx) in zip(axes, panels):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{title}\n{direction} slice={idx} (level={level:.2f})")
        ax.axis("off")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[infer] wrote visualization: {output_path}")

    if show_interactive:
        plt.show()

    plt.close(fig)


def _run_plot_loss(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        raise ValueError("--run-dir is required for command 'plot-loss'.")
    run_dir = Path(args.run_dir)
    if run_dir.is_file():
        run_dir = run_dir.parent
    history_path = run_dir / "metrics_history.json"
    if not history_path.exists():
        raise FileNotFoundError(
            f"Missing {history_path}. "
            "Use a run directory path (not a file), or re-run training with updated code to generate "
            "metrics_history.json."
        )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    output_path = args.plot_output if args.plot_output is not None else (run_dir / "loss_curve.png")
    ok = write_loss_curve_png(history, output_path)
    if not ok:
        raise RuntimeError("Could not create loss plot PNG. Check matplotlib installation.")
    print(f"[plot-loss] wrote {output_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _apply_overrides(args)
    device = args.device or get_device()

    if args.command in {"infer", "eval"}:
        config = _merge_inference_config(config, args, device=device)
        validate_config(config)
        set_seed(config["seed"], deterministic=config["deterministic"])
        _print_effective_config(config, args.command, device)
        if args.command == "eval":
            _run_eval(config, device=device, eval_report=args.eval_report)
        else:
            _run_infer(config, args, device=device)
        return

    if args.command == "plot-loss":
        _run_plot_loss(args)
        return

    validate_config(config)
    set_seed(config["seed"], deterministic=config["deterministic"])
    _print_effective_config(config, args.command, device)

    if args.command == "train":
        run_training(config, device=device)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
