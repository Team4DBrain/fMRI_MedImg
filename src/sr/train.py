"""Training loop with maximum metric tracking and lossless resume.

Purpose:
    Drive the per-epoch lifecycle: data -> forward -> backward -> validation
    -> persist a self-contained ``EpochState`` -> mirror metrics to JSON.
    Each completed epoch is the new "last good" resume point.
Effects:
    Writes ``config.json``, ``split.json``, ``metrics.json``, TensorBoard
    logs and per-epoch ``epochs/epoch_NNN.pt`` files under the run dir.
    No automatic post-training analysis -- the user composes that.
Influences:
    All behaviour is driven by the validated ``SRConfig``. Resume reads
    the saved ``config.json`` and ignores the live config to enforce
    "resume only with the same configuration".
How to change safely:
    Keep the artifact filenames stable -- they are part of the public
    contract with eval/infer. Add new metric keys via ``compute_full_metrics``
    rather than inventing keys ad hoc here.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.sr.checkpoint import (
    EpochState,
    capture_rng_state,
    find_latest_epoch,
    load_epoch,
    restore_rng_state,
    save_epoch,
    write_metrics_json,
)
from src.sr.components import build_optimizer, build_scheduler, step_scheduler
from src.sr.config import (
    SRConfig,
    auto_device,
    from_json,
    seed_everything,
    summary,
    to_json,
    validate,
)
from src.sr.data import build_loaders, write_split_json
from src.sr.losses import resolve_loss
from src.sr.metrics import average_metric_dicts, compute_full_metrics
from src.sr.forward import model_forward
from src.sr.models import build_model, count_parameters
from src.sr.shape_utils import align_pred_target_mask


BANNER = "=" * 60


def _banner(title: str) -> None:
    """Print a clearly visible section header. Used between phases."""
    print(BANNER)
    print(f"[train] Phase: {title}")
    print(BANNER)


def _prepare_run_directory(
    config: SRConfig, resume_dir: Path | None
) -> tuple[Path, SRConfig]:
    """Create or resolve the run directory and pick the active config.

    Returns:
        ``(run_dir, effective_config)``. On a fresh run, ``effective_config``
        is the input config and a new ``run_dir`` is created. On a resume,
        ``run_dir`` is the given directory and ``effective_config`` comes
        from its saved ``config.json``.
    """
    if resume_dir is not None:
        run_dir = Path(resume_dir).resolve()
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: {config_path} does not exist. "
                "Pass a run directory that contains config.json."
            )
        saved_config: SRConfig = from_json(config_path)
        print(f"[train] Resume requested from {run_dir}")
        print(f"[train] Using saved config from {config_path}")
        return run_dir, saved_config

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.run_root) / config.model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    to_json(config, run_dir / "config.json")
    print(f"[train] Starting new run in {run_dir}")
    return run_dir, config


def _train_one_epoch(
    *,
    epoch_index: int,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    loss_name: str,
    model_name: str,
    device: str,
    log_interval: int,
    strict_finite_loss: bool,
    tb_writer: SummaryWriter | None,
) -> tuple[float, float]:
    """Single training pass over ``loader``. Returns (mean_loss, duration_s)."""
    model.train()
    running = 0.0
    n_batches = max(1, len(loader))
    start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask_hr"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model_forward(model, inputs, target, model_name)
        pred, target, mask = align_pred_target_mask(pred, target, mask)
        loss = loss_fn(pred, target, mask)

        if strict_finite_loss and not math.isfinite(float(loss.detach().item())):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch_index + 1}, batch={batch_idx + 1}: "
                f"{loss.item()}"
            )

        loss.backward()
        optimizer.step()

        running += float(loss.item())

        if tb_writer is not None:
            global_step = epoch_index * n_batches + batch_idx
            tb_writer.add_scalar("batch/train_loss", float(loss.item()), global_step)

        if (batch_idx + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch {epoch_index + 1}  step {batch_idx + 1}/{n_batches}  "
                f"{loss_name}={loss.item():.6f}  lr={current_lr:.2e}"
            )

    duration = time.perf_counter() - start
    return running / n_batches, duration


@torch.no_grad()
def _validate_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    device: str,
) -> tuple[dict[str, float], float]:
    """Validation pass that reports all losses + all metrics per batch.

    Returns ``(mean_metrics, duration_s)``. Keys mirror
    ``metrics.compute_full_metrics`` so the training log and analytic
    notebooks read the same names.
    """
    model.eval()
    per_batch: list[dict[str, float]] = []
    start = time.perf_counter()
    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask_hr"].to(device)
        pred = model_forward(model, inputs, target, model_name)
        per_batch.append(compute_full_metrics(pred, target, mask))
    duration = time.perf_counter() - start
    return average_metric_dicts(per_batch), duration


def _log_epoch_to_tb(
    tb_writer: SummaryWriter | None,
    epoch_index: int,
    record: dict[str, Any],
) -> None:
    if tb_writer is None:
        return
    for key, value in record.items():
        if key == "epoch" or value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            tb_writer.add_scalar(f"epoch/{key}", float(value), epoch_index)


def _format_epoch_summary(record: dict[str, Any]) -> str:
    parts = [
        f"epoch={int(record['epoch'])}",
        f"lr={record['lr']:.2e}",
        f"train_loss={record['train_loss']:.6f}",
        f"train_s={record['train_duration_s']:.1f}",
    ]
    if record.get("val_masked_mse") is not None:
        parts.extend(
            [
                f"val_masked_mse={record['val_masked_mse']:.6f}",
                f"val_masked_psnr={record['val_masked_psnr']:.2f}",
                f"val_masked_ssim={record['val_masked_ssim']:.4f}",
                f"val_s={record['val_duration_s']:.1f}",
            ]
        )
    return " ".join(parts)


def train(config: SRConfig, resume_dir: Path | None = None) -> Path:
    """Run the full training lifecycle and return the run directory.

    Whether starting fresh or resuming, the same loop body runs; only the
    starting epoch and the seed/RNG behaviour differ.
    """
    _banner("PREP")

    run_dir, config = _prepare_run_directory(config, resume_dir)
    validate(config)
    device = auto_device()
    print(f"[train] Device: {device}")
    print(summary(config))

    seed_everything(config.seed, config.deterministic)

    train_loader, val_loader, split_info = build_loaders(config)
    if resume_dir is None:
        write_split_json(run_dir, split_info)
    print(f"[train] split_source  = {split_info['source']}")
    print(
        f"[train] samples: train={split_info['train_samples']}  "
        f"val={split_info['val_samples']}  "
        f"train_batches={len(train_loader)}"
        + (f"  val_batches={len(val_loader)}" if val_loader is not None else "")
    )

    model = build_model(config).to(device)
    print(f"[train] model={config.model_name}  params={count_parameters(model):,}")

    optimizer = build_optimizer(config, model)
    scheduler, scheduler_needs_val = build_scheduler(config, optimizer)
    loss_fn = resolve_loss(config.loss_name)

    if val_loader is None and scheduler_needs_val:
        raise ValueError(
            f"scheduler '{config.scheduler_name}' needs validation loss but "
            "no validation set is configured. Use train_split<1.0 or pick "
            "another scheduler (e.g. 'none')."
        )

    metrics_history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch_number = 0
    start_epoch = 0

    if resume_dir is not None:
        latest = find_latest_epoch(run_dir)
        if latest is None:
            raise FileNotFoundError(
                f"Cannot resume {run_dir}: no epoch checkpoint found under epochs/"
            )
        state = load_epoch(latest, map_location=device)
        model.load_state_dict(state.model_state_dict)
        optimizer.load_state_dict(state.optimizer_state_dict)
        if scheduler is not None and state.scheduler_state_dict is not None:
            scheduler.load_state_dict(state.scheduler_state_dict)
        restore_rng_state(state.rng_state)
        metrics_history = list(state.metrics_history)
        best_val_loss = state.best_val_loss
        best_epoch_number = state.best_epoch_number
        start_epoch = state.epoch_number
        print(f"[train] Resumed from {latest}")
        print(
            f"[train] last completed epoch = {start_epoch}, "
            f"best_val_loss = {best_val_loss:.6f} (epoch {best_epoch_number})"
        )
        if start_epoch >= config.num_epochs:
            print(
                f"[train] Nothing to do: completed epochs ({start_epoch}) >= "
                f"num_epochs ({config.num_epochs}). Edit config.json to extend."
            )
            return run_dir

    tb_writer: SummaryWriter | None = None
    if config.tensorboard:
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        tb_writer.add_text("run/model_name", config.model_name)
        tb_writer.add_text("run/manifest", str(config.manifest_path))
        tb_writer.add_text("run/device", device)

    try:
        for epoch_index in range(start_epoch, config.num_epochs):
            epoch_number = epoch_index + 1
            _banner(f"EPOCH {epoch_number}/{config.num_epochs}")

            train_loss, train_duration = _train_one_epoch(
                epoch_index=epoch_index,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                loss_name=config.loss_name,
                model_name=config.model_name,
                device=device,
                log_interval=config.log_interval,
                strict_finite_loss=config.strict_finite_loss,
                tb_writer=tb_writer,
            )

            if val_loader is not None:
                val_metrics, val_duration = _validate_one_epoch(
                    model=model,
                    loader=val_loader,
                    model_name=config.model_name,
                    device=device,
                )
            else:
                val_metrics, val_duration = {}, 0.0

            current_val_loss = (
                float(val_metrics[config.loss_name])
                if val_metrics and config.loss_name in val_metrics
                else None
            )
            step_scheduler(scheduler, scheduler_needs_val, current_val_loss)

            record: dict[str, Any] = {
                "epoch": epoch_number,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_loss),
                "train_duration_s": float(train_duration),
                "val_duration_s": float(val_duration),
            }
            # Prefix all val_* keys so train/val metric names never collide.
            for key, value in val_metrics.items():
                record[f"val_{key}"] = float(value)

            metrics_history.append(record)

            if current_val_loss is not None and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch_number = epoch_number

            _log_epoch_to_tb(tb_writer, epoch_index, record)
            print(f"[train] summary  {_format_epoch_summary(record)}")
            if best_epoch_number == epoch_number and current_val_loss is not None:
                print(
                    f"[train] new best {config.loss_name}={best_val_loss:.6f} "
                    f"at epoch {best_epoch_number}"
                )

            state = EpochState(
                epoch_number=epoch_number,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=(
                    scheduler.state_dict() if scheduler is not None else None
                ),
                rng_state=capture_rng_state(),
                metrics_history=metrics_history,
                best_val_loss=best_val_loss,
                best_epoch_number=best_epoch_number,
                loss_name=config.loss_name,
            )
            written = save_epoch(run_dir, state)
            write_metrics_json(run_dir, metrics_history)
            print(f"[train] wrote {written.relative_to(run_dir.parent)}")
    finally:
        if tb_writer is not None:
            tb_writer.close()

    _banner("DONE")
    print(f"[train] run_dir = {run_dir}")
    print(
        f"[train] last completed epoch = {metrics_history[-1]['epoch']}"
        if metrics_history
        else "[train] no epochs completed"
    )
    if best_epoch_number:
        print(
            f"[train] best epoch = {best_epoch_number} "
            f"({config.loss_name}={best_val_loss:.6f})"
        )
    return run_dir
