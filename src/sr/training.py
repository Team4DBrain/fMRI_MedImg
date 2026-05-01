"""Training and checkpoint orchestration for 3D SR."""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import get_device
from .data import create_dataloaders
from .model import build_model_from_config


def psnr_from_mse(mse_value: float, data_range: float = 1.0) -> float:
    """Convert MSE to PSNR."""
    if mse_value <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range**2) / mse_value)


def validate_one_epoch(model, val_loader, loss_fn, device):
    """Run one validation pass without gradients."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
    return running_loss / max(1, len(val_loader))


def train_one_epoch(
    epoch_index,
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    tb_writer,
    log_interval=10,
    strict_finite_loss: bool = True,
):
    """Execute one optimization epoch."""
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
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
    """Save model, optimizer, and metadata checkpoint."""
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
    """Raise when training loss is non-finite."""
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(
            f"Non-finite loss detected at epoch={epoch_index + 1}, batch={batch_idx + 1}: {loss_value}"
        )


def maybe_resume_training(model, optimizer, checkpoint_path, device):
    """Restore model/optimizer from checkpoint when provided."""
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
    """Create default loss, optimizer, and scheduler."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return loss_fn, optimizer, scheduler


def run_training(config: dict, model=None, device: str | None = None):
    """Train model end-to-end using provided config."""
    if device is None:
        device = get_device()
    if model is None:
        model = build_model_from_config(config).to(device)

    loss_fn, optimizer, scheduler = build_training_components(model, config)

    model_name = str(config["model_name"]).strip().lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config["run_root"] / model_name / timestamp
    epochs_dir = run_dir / "epochs"
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as file_obj:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, file_obj, indent=2)

    train_loader, val_loader, dataset_size = create_dataloaders(config)
    print(f"Dataset size: {dataset_size}, train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Validation batches: {len(val_loader)}")

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

    for epoch in range(start_epoch, config["num_epochs"]):
        train_loss = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            writer,
            log_interval=config["log_interval"],
            strict_finite_loss=bool(config.get("strict_finite_loss", True)),
        )

        val_loss = None
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("epoch/loss_train", train_loss, epoch)
        writer.add_scalar("epoch/lr", current_lr, epoch)
        train_psnr = psnr_from_mse(train_loss)
        writer.add_scalar("epoch/psnr_train", train_psnr, epoch)

        if val_loss is not None:
            writer.add_scalar("epoch/loss_val", val_loss, epoch)
            val_psnr = psnr_from_mse(val_loss)
            writer.add_scalar("epoch/psnr_val", val_psnr, epoch)
            print(
                f"Epoch {epoch + 1}/{config['num_epochs']} "
                f"train={train_loss:.6f} val={val_loss:.6f} "
                f"psnr_train={train_psnr:.2f} psnr_val={val_psnr:.2f} lr={current_lr:.2e}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{config['num_epochs']} "
                f"train={train_loss:.6f} psnr_train={train_psnr:.2f} lr={current_lr:.2e}"
            )

        if (epoch + 1) % config["checkpoint_interval"] == 0:
            epoch_dir = epochs_dir / f"epoch_{epoch + 1:03d}"
            save_checkpoint(epoch_dir / "checkpoint.pt", epoch, model, optimizer, best_val_loss, config)

        metric_for_best = val_loss if val_loss is not None else train_loss
        if metric_for_best < best_val_loss:
            best_val_loss = metric_for_best
            save_checkpoint(run_dir / "best.pt", epoch, model, optimizer, best_val_loss, config)

    save_checkpoint(run_dir / "final.pt", config["num_epochs"] - 1, model, optimizer, best_val_loss, config)
    writer.close()
    print(f"Training complete. Run dir: {run_dir}")
