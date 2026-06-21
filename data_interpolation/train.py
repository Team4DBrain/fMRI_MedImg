"""train.py — config-driven training entry point for fMRI interpolation.

Usage:

    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml train.epochs=50 train.lr=5e-5
    python train.py --config configs/default.yaml checkpoint.resume=checkpoints/run/last.pt

Checkpoints written under `checkpoint.dir`:

    last.pt            full resume state (model + optimizer + scheduler + history)
    best.pt            best validation checkpoint (only when val split exists)
    model_weights.pt   lightweight model weights only (for inference / main.py)
    history.json       per-epoch loss history
"""

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

# Make src/ importable when running as `python train.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import FMRIInterpolationDataset, split_by_file
from src.loss import HybridL1SSIMLoss
from src.model import UNet3D
from src.utils import apply_overrides, deep_update, load_config, pick_device, seed_everything


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse `--config` plus dotted-key overrides like `train.epochs=10`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to a YAML config (default: configs/default.yaml)")
    args, overrides = parser.parse_known_args()
    return args, overrides


def build_config(args: argparse.Namespace, overrides: list[str]) -> dict:
    """Load YAML config and apply CLI dotted overrides."""
    config = load_config(args.config)
    if overrides:
        apply_overrides(config, overrides)
    return config


def make_loader(dataset, batch_size: int, num_workers: int, device: torch.device, shuffle: bool) -> DataLoader:
    """DataLoader with device-appropriate defaults."""
    is_cuda = device.type == "cuda"
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": is_cuda,
        "persistent_workers": bool(is_cuda and num_workers > 0),
    }
    if is_cuda and num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module when wrapped by torch.compile."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def zero_initialize_residual_head(model: torch.nn.Module) -> None:
    """Zero the regression head so a fresh residual model equals the naive midpoint."""
    inner = unwrap_model(model)
    torch.nn.init.zeros_(inner.head.weight)
    if inner.head.bias is not None:
        torch.nn.init.zeros_(inner.head.bias)


def save_checkpoint(path: Path, *, model, optimizer, scheduler, config, epoch,
                    train_loss, val_loss, best_val, history) -> None:
    """Write the full resume checkpoint to `path`."""
    inner = unwrap_model(model)
    torch.save({
        "epoch": epoch,
        "model": inner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "history": history,
    }, path)


def save_weights(path: Path, model: torch.nn.Module) -> None:
    """Write inference-only weights to `path`."""
    torch.save(unwrap_model(model).state_dict(), path)


def save_history(history: list[dict], path: Path) -> None:
    """Write loss history as JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def maybe_resume(resume_path: str | None, model, optimizer, scheduler, device, target_epochs, lr):
    """Optionally load a `last.pt` to continue a previous run."""
    if not resume_path:
        return 1, float("inf"), []
    ckpt = torch.load(resume_path, map_location=device)
    unwrap_model(model).load_state_dict(ckpt.get("model", ckpt))
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    saved_epoch = int(ckpt.get("epoch", 0))
    # Recompute cosine LR for the extended schedule.
    scheduler.T_max = target_epochs
    scheduler.last_epoch = saved_epoch
    scheduler.base_lrs = [lr for _ in optimizer.param_groups]
    progress = min(saved_epoch, target_epochs) / max(target_epochs, 1)
    resumed_lr = 0.5 * lr * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["initial_lr"] = lr
        group["lr"] = resumed_lr
    best_val = float(ckpt.get("best_val", float("inf")))
    history = list(ckpt.get("history", []))
    print(f"resumed_from={resume_path} start_epoch={saved_epoch + 1}")
    return saved_epoch + 1, best_val, history


def run_epoch(model, loader, criterion, device, *, optimizer=None,
              residual=False, use_mask_loss=False, epoch=0, log_every=10):
    """Run one train or eval epoch; return mean loss."""
    training = optimizer is not None
    model.train(training)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )
    total = 0.0
    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with autocast_ctx:
                raw = model(x)
                pred = (0.5 * (x[:, 0:1] + x[:, 1:2]) + raw) if residual else raw
                loss, comps = criterion(
                    pred, y,
                    mask=mask if use_mask_loss else None,
                    return_components=True,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            naive = 0.5 * (x[:, 0:1] + x[:, 1:2])
            naive_l1 = ((naive - y).abs() * (mask if use_mask_loss and mask is not None else 1)).sum() \
                       / ((mask.sum().clamp(min=1)) if use_mask_loss and mask is not None else y.numel())
        total += float(loss.detach().cpu())

        if training and (step == 1 or step % log_every == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch} step={step}/{len(loader)} lr={lr:.3e} "
                f"loss={float(loss.detach().cpu()):.6f} "
                f"l1={float(comps['l1'].cpu()):.6f} "
                f"ssim={float(comps['ssim'].cpu()):.6f} "
                f"naive_l1={float(naive_l1.cpu()):.6f}"
            )
    return total / max(len(loader), 1)


def main() -> None:
    args, overrides = parse_args()
    config = build_config(args, overrides)

    # Flatten frequently-used config sections for readability.
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    ckpt_cfg = config.get("checkpoint", {})

    seed_everything(train_cfg.get("seed", 0))
    torch.use_deterministic_algorithms(False)

    device = pick_device(train_cfg.get("device"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Build dataset and optionally split by file.
    dataset = FMRIInterpolationDataset(
        root=data_cfg.get("root"),
        norm_mode=data_cfg.get("norm_mode", "zscore"),
        file_list=data_cfg.get("file_list"),
    )

    max_samples = train_cfg.get("max_samples")
    if max_samples is not None:
        train_set = Subset(dataset, list(range(min(max_samples, len(dataset)))))
        val_set = None
    elif len(dataset.files) > 1:
        train_set, val_set = split_by_file(
            dataset,
            val_fraction=data_cfg.get("val_fraction", 0.1),
            seed=train_cfg.get("seed", 0),
        )
    else:
        train_set, val_set = dataset, None

    # Pick effective batch size / num_workers based on device when null.
    is_cuda = device.type == "cuda"
    batch_size = train_cfg.get("batch_size") or (4 if is_cuda else 1)
    num_workers = train_cfg.get("num_workers")
    if num_workers is None:
        num_workers = 8 if is_cuda else 0

    train_loader = make_loader(train_set, batch_size, num_workers, device, shuffle=True)
    val_loader = make_loader(val_set, batch_size, num_workers, device, shuffle=False) if val_set is not None else None

    # Model, loss, optimizer, scheduler.
    model = UNet3D(
        in_channels=model_cfg.get("in_channels", 2),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        depth=model_cfg.get("depth", 4),
    ).to(device)

    residual = bool(train_cfg.get("residual", False))
    if residual and not ckpt_cfg.get("resume"):
        zero_initialize_residual_head(model)
        print("residual=True: head zero-initialized")

    data_range = 1.0 if data_cfg.get("norm_mode", "zscore") == "percentile" else 2.0
    criterion = HybridL1SSIMLoss(alpha=train_cfg.get("alpha", 0.5), data_range=data_range).to(device)

    lr = float(train_cfg.get("lr", 1e-4))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
    epochs = int(train_cfg.get("epochs", 20))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/run"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val, history = maybe_resume(
        ckpt_cfg.get("resume"), model, optimizer, scheduler, device, epochs, lr
    )

    if device.type == "cuda" and train_cfg.get("compile"):
        model = torch.compile(model, mode="max-autotune")

    print(
        f"device={device.type} files={len(dataset.files)} samples={len(dataset)} "
        f"train_samples={len(train_set)} val_samples={len(val_set) if val_set is not None else 0} "
        f"batch_size={batch_size} epochs={epochs}"
    )

    if start_epoch > epochs:
        print(f"checkpoint already at epoch {start_epoch - 1}; nothing to do")
        return

    use_mask_loss = bool(train_cfg.get("use_mask_loss", False))
    log_every = int(train_cfg.get("log_every", 10))
    save_every = max(1, int(train_cfg.get("save_every", 1)))

    for epoch in range(start_epoch, epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, residual=residual,
            use_mask_loss=use_mask_loss, epoch=epoch, log_every=log_every,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(
                model, val_loader, criterion, device,
                residual=residual, use_mask_loss=use_mask_loss, epoch=epoch,
            )

        improved = val_loss is not None and val_loss < best_val
        if improved:
            best_val = val_loss

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": epoch_lr,
        })
        scheduler.step()

        if (epoch % save_every == 0) or (epoch == epochs):
            save_checkpoint(
                ckpt_dir / "last.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                config=config, epoch=epoch,
                train_loss=train_loss, val_loss=val_loss,
                best_val=best_val, history=history,
            )
            save_weights(ckpt_dir / "model_weights.pt", model)
            save_history(history, ckpt_dir / "history.json")

        if improved:
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                config=config, epoch=epoch,
                train_loss=train_loss, val_loss=val_loss,
                best_val=best_val, history=history,
            )
            save_weights(ckpt_dir / "best_weights.pt", model)

        print(
            f"epoch={epoch} train_loss={train_loss:.6f}"
            + (f" val_loss={val_loss:.6f}" if val_loss is not None else "")
            + (" (best)" if improved else "")
        )


if __name__ == "__main__":
    main()
