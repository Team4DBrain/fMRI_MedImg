"""Optimizer and scheduler registries with explicit factory helpers.

Purpose:
    Make the training loop independent of which optimizer/scheduler is
    used. The CLI sets ``optimizer_name`` and ``scheduler_name`` (plus
    kwargs), this module turns those into ready-to-step objects.
Effects:
    Decides how parameters are updated and how the learning rate evolves
    over epochs. Default policy is Adam + ReduceLROnPlateau, which matches
    the original pipeline so old training behaviour is reproducible.
Influences:
    Behaviour depends entirely on ``config.optimizer_kwargs`` /
    ``config.scheduler_kwargs``. Both default to empty dicts, so omit a kwarg
    to get the upstream PyTorch default.
How to change safely:
    Register new entries in the registries. If a scheduler needs
    ``val_loss`` in its ``step`` (like plateau), add its key to
    ``SCHEDULERS_NEEDING_VAL_LOSS`` so the loop calls ``step`` correctly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

from src.sr.config import SRConfig

OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

# ``None`` means "no scheduler"; the training loop just never calls step.
SCHEDULER_REGISTRY: dict[str, type | None] = {
    "plateau": ReduceLROnPlateau,
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "none": None,
}

# Schedulers whose ``step`` requires the latest validation loss as input.
# Everything else gets ``step()`` called without arguments.
SCHEDULERS_NEEDING_VAL_LOSS: frozenset[str] = frozenset({"plateau"})


def build_optimizer(config: SRConfig, model: torch.nn.Module) -> Optimizer:
    """Construct the configured optimizer with ``learning_rate`` + ``optimizer_kwargs``.

    ``learning_rate`` is mandatory and passed positionally to PyTorch's
    ``Optimizer`` constructor. Anything else (betas, weight_decay, ...) lives
    in ``optimizer_kwargs`` and is forwarded as-is.
    """
    if config.optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{config.optimizer_name}'. "
            f"Available: {sorted(OPTIMIZER_REGISTRY)}"
        )
    cls = OPTIMIZER_REGISTRY[config.optimizer_name]
    extra: dict[str, Any] = dict(config.optimizer_kwargs)
    if "lr" in extra:
        raise ValueError(
            "Put the learning rate in config.learning_rate, not optimizer_kwargs['lr']."
        )
    return cls(model.parameters(), lr=config.learning_rate, **extra)


def build_scheduler(
    config: SRConfig, optimizer: Optimizer
) -> tuple[Any | None, bool]:
    """Construct the configured LR scheduler.

    Returns:
        ``(scheduler, needs_val_loss)``. ``scheduler`` is ``None`` for
        ``scheduler_name='none'``; the training loop skips the step in
        that case. ``needs_val_loss`` tells the loop whether to forward
        the validation loss into ``step``.
    """
    if config.scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{config.scheduler_name}'. "
            f"Available: {sorted(SCHEDULER_REGISTRY)}"
        )
    cls = SCHEDULER_REGISTRY[config.scheduler_name]
    if cls is None:
        return None, False
    scheduler = cls(optimizer, **config.scheduler_kwargs)
    needs_val = config.scheduler_name in SCHEDULERS_NEEDING_VAL_LOSS
    return scheduler, needs_val


def step_scheduler(
    scheduler: Any | None, needs_val_loss: bool, val_loss: float | None
) -> None:
    """Single, explicit place that knows how each scheduler is stepped."""
    if scheduler is None:
        return
    if needs_val_loss:
        if val_loss is None:
            raise ValueError(
                f"Scheduler '{type(scheduler).__name__}' needs a validation loss, "
                "but training ran without a validation set. Use scheduler_name='none' "
                "or 'cosine'/'step' for runs without validation."
            )
        scheduler.step(val_loss)
    else:
        scheduler.step()
