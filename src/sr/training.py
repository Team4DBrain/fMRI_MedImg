"""Run optimization, validation, logging, and checkpoint lifecycle for SR.

Purpose:
    Encapsulate the full training loop and metric computations used by the SR
    CLI so experiments are reproducible and auditable.
Effects:
    Controls gradient updates, validation reporting, LR scheduling, checkpoint
    persistence, and run summaries consumed by downstream analysis.
Influences:
    Runtime behavior depends on config values, model architecture, dataset
    splits, and optional resume checkpoints.
How to change safely:
    Preserve metric names/file outputs expected by `run.py` and tests, and keep
    checkpoint payload fields stable for backward compatibility.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .config import get_device
from .data import create_dataloaders
from .model import build_model_from_config


def psnr_from_mse(mse_value: float, data_range: float = 1.0) -> float:
    """Convert reconstruction error to a PSNR value used for reporting quality.

    Purpose:
        Express MSE on a logarithmic scale that is easier to compare across
        runs.
    Effects:
        Used in training/eval summaries and logs; affects interpretation only,
        not optimization.
    Influences:
        Depends on chosen `data_range` and lower-bound handling for near-zero
        MSE values.
    How to change safely:
        Keep conversion formula aligned across training and eval paths so
        historical metric comparisons remain meaningful.
    """
    if mse_value <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range**2) / mse_value)


def masked_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute MSE over voxels selected by a mask.

    Purpose:
        Focus optimization on in-brain regions where SR quality matters.
    Effects:
        Directly defines training loss and one of the core validation metrics.
    Influences:
        Result depends on mask coverage and `eps`, which stabilizes division for
        sparse masks.
    How to change safely:
        Keep mask semantics consistent with dataset outputs (`mask_hr`), and
        update metric interpretation/tests if weighting strategy changes.
    """
    sq_err = (outputs - targets) ** 2
    weighted = sq_err * mask
    denom = torch.clamp(mask.sum(), min=eps)
    return weighted.sum() / denom


def _ssim_window_size(spatial: tuple[int, int, int]) -> int:
    m = min(spatial)
    w = min(7, m)
    if w % 2 == 0:
        w -= 1
    return max(1, w)


def masked_local_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate structural similarity over masked 3D local neighborhoods.

    Purpose:
        Provide a perceptual-style quality signal complementary to masked MSE.
    Effects:
        Used for validation/eval reporting and model comparison.
    Influences:
        Sensitivity is controlled by window size logic, `data_range`, and mask
        weighting; input must be single-channel.
    How to change safely:
        Keep channel/shape assumptions explicit and align any formula updates
        with eval code so reported SSIM remains comparable.
    """
    _b, c, d, h, w = pred.shape
    if c != 1:
        raise ValueError("masked_local_ssim_3d expects a single channel (B,1,D,H,W)")
    win = _ssim_window_size((d, h, w))
    pad = win // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def pool(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=win, stride=1, padding=pad)

    mu_x = pool(pred)
    mu_y = pool(target)
    var_x = (pool(pred * pred) - mu_x * mu_x).clamp(min=0.0)
    var_y = (pool(target * target) - mu_y * mu_y).clamp(min=0.0)
    cov = pool(pred * target) - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2).clamp(min=eps)
    ssim_map = num / den

    mask_w = pool(mask)
    weighted = (ssim_map * mask_w).sum()
    denom = mask_w.sum().clamp(min=eps)
    return weighted / denom


def validate_one_epoch(model, val_loader, device):
    """Run validation metrics for one epoch without gradient updates.

    Purpose:
        Quantify generalization after each training epoch.
    Effects:
        Produces masked MSE/SSIM used in logs, scheduler decisions, and best
        checkpoint selection.
    Influences:
        Behavior depends on whether a validation loader exists; missing loader
        returns NaNs by design.
    How to change safely:
        Maintain returned dictionary keys (`mse`, `ssim`) because callers and
        summaries depend on them.
    """
    if val_loader is None:
        return {"mse": float("nan"), "ssim": float("nan")}
    model.eval()
    running_mse = 0.0
    running_ssim = 0.0
    n_batches = max(1, len(val_loader))
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            labels = batch["target"].to(device)
            mask = batch["mask_hr"].to(device)
            outputs = model(inputs)
            running_mse += masked_mse_loss(outputs, labels, mask).item()
            running_ssim += masked_local_ssim_3d(outputs, labels, mask).item()
    return {"mse": running_mse / n_batches, "ssim": running_ssim / n_batches}


def train_one_epoch(
    epoch_index,
    model,
    train_loader,
    optimizer,
    device,
    tb_writer,
    log_interval=10,
    strict_finite_loss: bool = True,
):
    """Execute one full training epoch over the training loader.

    Purpose:
        Perform forward/backward optimization steps and collect train-loss
        statistics.
    Effects:
        Updates model weights, writes batch loss to TensorBoard, and returns the
        epoch-average training loss.
    Influences:
        Behavior depends on optimizer state, `strict_finite_loss`, and
        `log_interval` verbosity.
    How to change safely:
        Keep loss definition aligned with validation/eval metrics and preserve
        global-step semantics used by TensorBoard comparisons.
    """
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["input"].to(device)
        labels = batch["target"].to(device)
        mask = batch["mask_hr"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = masked_mse_loss(outputs, labels, mask)
        if strict_finite_loss:
            ensure_finite_loss(loss, epoch_index=epoch_index, batch_idx=batch_idx)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step = epoch_index * len(train_loader) + batch_idx
        tb_writer.add_scalar("batch/loss_train", loss.item(), global_step)

        if (batch_idx + 1) % log_interval == 0:
            print(f"Epoch {epoch_index + 1} Batch {batch_idx + 1}/{len(train_loader)} loss={loss.item():.6f}")

    return running_loss / max(1, len(train_loader))


def save_checkpoint(path: Path, epoch: int, model, optimizer, best_val_loss: float, config: dict):
    """Persist model/optimizer state plus metadata for resume and eval.

    Purpose:
        Create recoverable snapshots that support interruption recovery and
        model reuse.
    Effects:
        Writes checkpoint payload atomically via temporary file replacement.
    Influences:
        Payload content depends on current epoch, optimizer state, and config.
    How to change safely:
        Preserve core keys (`model_state_dict`, `optimizer_state_dict`,
        `config`) to avoid breaking older checkpoints.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def ensure_finite_loss(loss: torch.Tensor, epoch_index: int, batch_idx: int) -> None:
    """Guardrail that aborts training when loss becomes NaN/Inf.

    Purpose:
        Prevent silently continuing corrupted optimization runs.
    Effects:
        Raises `FloatingPointError` with epoch/batch context when non-finite
        loss appears.
    Influences:
        Triggering depends on numerical stability and whether strict checking is
        enabled by config.
    How to change safely:
        Keep failure message contextual so teammates can quickly debug unstable
        settings or data issues.
    """
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(
            f"Non-finite loss detected at epoch={epoch_index + 1}, batch={batch_idx + 1}: {loss_value}"
        )


def maybe_resume_training(model, optimizer, checkpoint_path, device):
    """Restore training state from a checkpoint when resume is requested.

    Purpose:
        Continue interrupted runs without restarting optimization history.
    Effects:
        Loads model/optimizer states and returns start epoch with best-val
        baseline for later checkpoint decisions.
    Influences:
        Behavior depends on checkpoint existence and payload integrity.
    How to change safely:
        Keep returned tuple contract stable because `run_training` relies on the
        exact `(start_epoch, best_val_loss)` semantics.
    """
    if checkpoint_path is None:
        return 0, float("inf")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, best_val_loss


def build_training_components(model, config: dict):
    """Build optimizer and LR scheduler used by the SR training loop.

    Purpose:
        Centralize default optimization policy for consistent experiments.
    Effects:
        Defines how parameters are updated and how learning rate is reduced in
        response to validation trends.
    Influences:
        Depends on `learning_rate` and scheduler hyperparameters encoded here.
    How to change safely:
        Coordinate any scheduler policy change with README/docs so users know
        how `--lr` may evolve during training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return optimizer, scheduler


def write_loss_curve_png(history: dict[str, list[float]], output_path: Path) -> bool:
    """Render epoch-wise loss history to a PNG for quick run inspection.

    Purpose:
        Provide a lightweight visual summary of convergence without requiring
        TensorBoard.
    Effects:
        Writes an image file when plotting dependencies and history are valid.
    Influences:
        Depends on matplotlib availability and presence of non-empty epoch/train
        history.
    How to change safely:
        Keep input history keys consistent with `run_training` outputs or adjust
        both producer and consumer together.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[train] Skipping loss_curve.png (matplotlib unavailable: {exc})")
        return False

    epochs = history.get("epoch", [])
    train_loss = history.get("train_mse", [])
    val_loss = history.get("val_mse", [])
    if not epochs or not train_loss:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, marker="o", linewidth=1.8, label="train_mse")
    finite_val = [
        (epoch, value)
        for epoch, value in zip(epochs, val_loss, strict=False)
        if math.isfinite(float(value))
    ]
    if finite_val:
        val_epochs, val_values = zip(*finite_val)
        ax.plot(val_epochs, val_values, marker="o", linewidth=1.8, label="val_mse")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss over epochs")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def run_training(config: dict, model=None, device: str | None = None):
    """Run the complete SR training lifecycle for one experiment configuration.

    Purpose:
        Execute data loading, optimization, validation, logging, checkpointing,
        and final metric artifact generation in one reproducible workflow.
    Effects:
        Produces trained weights and run artifacts under the configured run
        directory, including summaries used for model selection.
    Influences:
        Runtime behavior is shaped by config (model choice, LR schedule, split
        policy, checkpoint frequency, finite-loss policy) and resume state.
    How to change safely:
        Preserve artifact filenames/JSON keys expected by tooling and tests; if
        changing output schema, update downstream consumers together.
    """
    if device is None:
        device = get_device()
    if model is None:
        model = build_model_from_config(config).to(device)

    optimizer, scheduler = build_training_components(model, config)

    model_name = str(config["model_name"]).strip().lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config["run_root"] / model_name / timestamp
    epochs_dir = run_dir / "epochs"
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as file_obj:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, file_obj, indent=2)

    train_loader, val_loader, dataset_size, split_info = create_dataloaders(config)
    print(f"Dataset size: {dataset_size}, train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Validation batches: {len(val_loader)}")
    else:
        print("No validation set (subject split disabled and no explicit val_subjects).")

    with open(run_dir / "split.json", "w", encoding="utf-8") as file_obj:
        json.dump(split_info, file_obj, indent=2)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    writer.add_text("run/model_name", model_name)
    writer.add_text("run/timestamp", timestamp)
    writer.add_text("run/device", device)
    writer.add_text("run/manifest", str(config["manifest_path"]))

    start_epoch, best_val_loss = maybe_resume_training(
        model,
        optimizer,
        config["resume_checkpoint"],
        device,
    )

    last_val_metrics: dict[str, float] = {"mse": float("nan"), "ssim": float("nan")}
    history: dict[str, list[float]] = {"epoch": [], "train_mse": [], "val_mse": [], "lr": []}
    for epoch in range(start_epoch, config["num_epochs"]):
        train_loss = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            device,
            writer,
            log_interval=config["log_interval"],
            strict_finite_loss=bool(config.get("strict_finite_loss", True)),
        )

        val_metrics = validate_one_epoch(model, val_loader, device)
        val_loss = val_metrics["mse"]
        scheduler.step(val_loss)
        last_val_metrics = val_metrics

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("epoch/loss_train", train_loss, epoch)
        writer.add_scalar("epoch/lr", current_lr, epoch)
        train_psnr = psnr_from_mse(train_loss)
        writer.add_scalar("epoch/psnr_train", train_psnr, epoch)
        history["epoch"].append(float(epoch + 1))
        history["train_mse"].append(float(train_loss))
        history["lr"].append(float(current_lr))

        if val_loader is not None:
            writer.add_scalar("epoch/loss_val", val_loss, epoch)
            val_psnr = psnr_from_mse(val_loss)
            writer.add_scalar("epoch/psnr_val", val_psnr, epoch)
            writer.add_scalar("epoch/ssim_val", val_metrics["ssim"], epoch)
            history["val_mse"].append(float(val_loss))
            print(
                f"Epoch {epoch + 1}/{config['num_epochs']} "
                f"train={train_loss:.6f} val_mse={val_loss:.6f} "
                f"psnr_train={train_psnr:.2f} psnr_val={val_psnr:.2f} "
                f"ssim_val={val_metrics['ssim']:.4f} lr={current_lr:.2e}"
            )
        else:
            history["val_mse"].append(float("nan"))
            print(
                f"Epoch {epoch + 1}/{config['num_epochs']} "
                f"train={train_loss:.6f} psnr_train={train_psnr:.2f} lr={current_lr:.2e}"
            )

        if (epoch + 1) % config["checkpoint_interval"] == 0:
            epoch_dir = epochs_dir / f"epoch_{epoch + 1:03d}"
            save_checkpoint(epoch_dir / "checkpoint.pt", epoch, model, optimizer, best_val_loss, config)

        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(run_dir / "best.pt", epoch, model, optimizer, best_val_loss, config)

    save_checkpoint(run_dir / "final.pt", config["num_epochs"] - 1, model, optimizer, best_val_loss, config)
    writer.close()
    summary = {
        "best_val_mse": best_val_loss if val_loader is not None else float("nan"),
        "final_train_mse": train_loss,
        "final_val_mse": last_val_metrics["mse"],
        "final_val_psnr": psnr_from_mse(last_val_metrics["mse"]),
        "final_val_ssim": last_val_metrics["ssim"],
        "num_epochs": int(config["num_epochs"]),
        "manifest_path": str(config["manifest_path"]),
        "model_name": model_name,
    }
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    with open(run_dir / "metrics_history.json", "w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)
    curve_written = write_loss_curve_png(history, run_dir / "loss_curve.png")
    if curve_written:
        print(f"[train] wrote {run_dir / 'loss_curve.png'}")
    print(f"Training complete. Run dir: {run_dir}")
