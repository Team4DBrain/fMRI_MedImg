"""Sanity checks for SR data/model/training pipeline."""

import torch

from .config import get_device
from .data import create_dataloaders
from .model import build_model_from_config
from .training import build_training_components


def run_sanity_checks(config: dict, model=None, device: str | None = None) -> None:
    """Run a one-batch forward/backward pass with assertions."""
    if device is None:
        device = get_device()
    print("[sanity] Starting sanity check...")
    print(f"[sanity] Using device: {device}")
    if model is None:
        print(f"[sanity] Building model '{config['model_name']}' from config.")
        model = build_model_from_config(config).to(device)

    loss_fn, optimizer, _ = build_training_components(model, config)
    print("[sanity] Creating dataloader and fetching one batch...")
    train_loader, _, _ = create_dataloaders(config)
    inputs, labels = next(iter(train_loader))
    print(f"[sanity] Batch shapes - inputs: {tuple(inputs.shape)}, labels: {tuple(labels.shape)}")

    assert inputs.ndim == 5 and labels.ndim == 5, "Expected BCHWD tensors"
    assert inputs.shape[1] == 1 and labels.shape[1] == 1, "Expected single-channel volumes"

    model.train()
    print("[sanity] Running forward/backward/optimizer step...")
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    assert outputs.shape == labels.shape, f"Output {outputs.shape} and labels {labels.shape} mismatch"

    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"[sanity] Sanity check passed. One-batch loss: {loss.item():.6f}")


def run_tiny_overfit_check(config: dict, steps: int = 20, device: str | None = None) -> None:
    """Train on one sample for a few steps to verify overfit behavior."""
    if device is None:
        device = get_device()
    print("[overfit] Starting tiny overfit check...")
    print(f"[overfit] Using device: {device} | steps: {steps}")

    train_loader, _, _ = create_dataloaders(config)
    inputs, labels = next(iter(train_loader))
    inputs = inputs[:1].to(device)
    labels = labels[:1].to(device)
    print(f"[overfit] Single-sample shapes - inputs: {tuple(inputs.shape)}, labels: {tuple(labels.shape)}")

    tiny_model = build_model_from_config(config).to(device)
    loss_fn = torch.nn.MSELoss()
    tiny_optimizer = torch.optim.Adam(tiny_model.parameters(), lr=config["learning_rate"])

    losses = []
    tiny_model.train()
    log_every = max(1, steps // 5)
    for step_idx in range(steps):
        tiny_optimizer.zero_grad(set_to_none=True)
        outputs = tiny_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        tiny_optimizer.step()
        losses.append(loss.item())
        if step_idx == 0 or (step_idx + 1) % log_every == 0 or step_idx == steps - 1:
            print(f"[overfit] Step {step_idx + 1}/{steps} loss={loss.item():.6f}")

    improvement = losses[0] - losses[-1]
    print(
        "[overfit] Tiny overfit check done: "
        f"start={losses[0]:.6f}, end={losses[-1]:.6f}, delta={improvement:.6f}"
    )
    if losses[-1] >= losses[0]:
        print("[overfit] Warning: loss did not decrease. Inspect data normalization and target pairing.")
