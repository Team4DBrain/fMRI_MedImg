"""Per-epoch checkpoints with everything needed for a lossless resume.

Purpose:
    After each completed epoch we want to be able to kill the process and
    pick up exactly where we left off -- same weights, same optimizer
    state, same RNG, same metric history. This module owns that contract.
Effects:
    Writes ``run_dir/epochs/epoch_NNN.pt`` atomically and mirrors the
    metric history to ``run_dir/metrics.json`` so a human can inspect
    progress without ``torch.load``.
Influences:
    Resume correctness depends on which fields are captured. RNG state
    covers Python/NumPy/torch CPU/torch CUDA; DataLoader worker order is
    still bound to ``seed + worker_id`` (see ``src.sr.data``).
How to change safely:
    Add new fields to ``EpochState`` and ``_state_to_payload``/
    ``_payload_to_state`` together. Never silently drop a field from the
    payload -- old runs are reloaded by this code.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class EpochState:
    """Everything needed to resume training right after epoch ``epoch_number``.

    ``epoch_number`` is 1-based: ``epoch_number=3`` means epochs 1, 2 and 3
    are done, the next thing to run is epoch 4. ``metrics_history[i]`` is
    the dict written for epoch ``i + 1``.
    """

    epoch_number: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    rng_state: dict[str, Any]
    metrics_history: list[dict[str, Any]]
    best_val_loss: float
    best_epoch_number: int
    loss_name: str = "masked_mse"
    extra: dict[str, Any] = field(default_factory=dict)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG the training loop touches."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by ``capture_rng_state``.

    Missing keys are tolerated for forward-compat; e.g. an older checkpoint
    without a CUDA RNG state simply leaves CUDA RNG untouched.

    ``torch.set_rng_state`` requires the CPU RNG blob on CPU (uint8 /
    ByteTensor). Checkpoints loaded with ``torch.load(..., map_location=cuda)``
    may have moved that tensor to GPU; normalize here so resume always works.
    """
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state and state["torch_cpu"] is not None:
        cpu_rng = state["torch_cpu"]
        if isinstance(cpu_rng, torch.Tensor):
            cpu_rng = cpu_rng.detach().cpu().contiguous()
        torch.set_rng_state(cpu_rng)
    cuda_states = state.get("torch_cuda_all")
    if cuda_states is not None and torch.cuda.is_available():
        # Each entry must be a uint8 / ByteTensor blob (typically CPU tensors
        # after ``torch.load(..., map_location='cpu')``); ``set_rng_state``
        # clones then installs on the matching device index.
        normalized: list[torch.Tensor] = []
        for s in cuda_states:
            if not isinstance(s, torch.Tensor):
                raise TypeError(
                    f"torch_cuda_all must contain only tensors, got {type(s).__name__}"
                )
            normalized.append(s.detach().cpu().contiguous())
        torch.cuda.set_rng_state_all(normalized)


def _normalize_rng_state_after_load(rng_state: dict[str, Any] | None) -> None:
    """In-place fix for RNG tensors after ``torch.load``.

    Purpose: keep RNG blobs on CPU as contiguous tensors so ``restore_rng_state``
    can pass them to ``torch.set_rng_state`` / ``torch.cuda.set_rng_state_all``
    without dtype/device surprises. Effect: normalizes ``torch_cpu`` and each
    entry of ``torch_cuda_all`` when present. Influences: only ``load_epoch``.
    Change guidance: if adding new RNG keys, mirror the same normalization rules.
    """
    if not rng_state:
        return
    t = rng_state.get("torch_cpu")
    if isinstance(t, torch.Tensor):
        rng_state["torch_cpu"] = t.detach().cpu().contiguous()

    cuda_all = rng_state.get("torch_cuda_all")
    if cuda_all is None:
        return
    if isinstance(cuda_all, (list, tuple)):
        rng_state["torch_cuda_all"] = [
            s.detach().cpu().contiguous() if isinstance(s, torch.Tensor) else s
            for s in cuda_all
        ]


def _epochs_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "epochs"


def epoch_path(run_dir: Path, epoch_number: int) -> Path:
    """Canonical path of the per-epoch checkpoint file."""
    return _epochs_dir(run_dir) / f"epoch_{int(epoch_number):03d}.pt"


def _state_to_payload(state: EpochState) -> dict[str, Any]:
    return {
        "epoch_number": state.epoch_number,
        "model_state_dict": state.model_state_dict,
        "optimizer_state_dict": state.optimizer_state_dict,
        "scheduler_state_dict": state.scheduler_state_dict,
        "rng_state": state.rng_state,
        "metrics_history": state.metrics_history,
        "best_val_loss": state.best_val_loss,
        "best_epoch_number": state.best_epoch_number,
        "loss_name": state.loss_name,
        "extra": state.extra,
    }


def _payload_to_state(payload: dict[str, Any]) -> EpochState:
    return EpochState(
        epoch_number=int(payload["epoch_number"]),
        model_state_dict=payload["model_state_dict"],
        optimizer_state_dict=payload["optimizer_state_dict"],
        scheduler_state_dict=payload.get("scheduler_state_dict"),
        rng_state=payload.get("rng_state", {}),
        metrics_history=list(payload.get("metrics_history", [])),
        best_val_loss=float(payload.get("best_val_loss", float("inf"))),
        best_epoch_number=int(payload.get("best_epoch_number", 0)),
        loss_name=str(payload.get("loss_name", "masked_mse")),
        extra=dict(payload.get("extra", {})),
    )


def save_epoch(run_dir: Path, state: EpochState) -> Path:
    """Write ``state`` atomically to ``run_dir/epochs/epoch_NNN.pt``.

    The temp + replace pattern guarantees that a kill mid-write leaves the
    previous epoch's checkpoint intact, so the "last complete epoch is the
    resume point" contract holds even under SIGKILL.
    """
    path = epoch_path(run_dir, state.epoch_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(_state_to_payload(state), tmp)
    tmp.replace(path)
    return path


def load_epoch(path: Path, map_location: str | None = None) -> EpochState:
    """Load an ``EpochState`` written by ``save_epoch``.

    ``weights_only=False`` is required because the payload intentionally
    contains non-tensor objects (numpy RNG state, metric history dicts).
    The file is produced by this code and lives next to the run's
    ``config.json``, so trust is on par with reading any other artifact
    from the run directory.

    Checkpoints are always loaded with ``map_location='cpu'``. Passing
    ``map_location=cuda`` to ``torch.load`` rewrites every tensor in the file,
    including RNG byte blobs, which breaks ``torch.set_rng_state`` and
    ``torch.cuda.set_rng_state_all``. Callers move weights to the training
    device by loading into modules already on that device (``load_state_dict``).
    The ``map_location`` argument is kept for call-site compatibility but is
    ignored.
    """
    _ = map_location
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"File at {path} is not a valid EpochState payload.")
    _normalize_rng_state_after_load(payload.get("rng_state"))
    return _payload_to_state(payload)


def list_epoch_files(run_dir: Path) -> list[Path]:
    """All ``epoch_NNN.pt`` files in numeric order. Empty list when there are none."""
    epochs = _epochs_dir(run_dir)
    if not epochs.is_dir():
        return []
    files = []
    for path in epochs.glob("epoch_*.pt"):
        try:
            int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        files.append(path)
    files.sort(key=lambda p: int(p.stem.split("_")[1]))
    return files


def find_latest_epoch(run_dir: Path) -> Path | None:
    """Return the highest-numbered ``epoch_NNN.pt`` in ``run_dir`` or ``None``."""
    files = list_epoch_files(run_dir)
    return files[-1] if files else None


def write_metrics_json(run_dir: Path, history: list[dict[str, Any]]) -> Path:
    """Write the metric history as plain JSON, atomically.

    Mirrors ``EpochState.metrics_history`` so users can plot/inspect
    progress without loading a ``.pt`` file. Rewritten after every epoch.
    """
    path = Path(run_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def run_dir_for_checkpoint(checkpoint_path: Path) -> Path:
    """Resolve the run directory that owns ``epochs/epoch_NNN.pt``.

    Convention: ``<run_dir>/epochs/epoch_NNN.pt``. Walking up two levels
    is enough; we double-check by confirming ``config.json`` is there.
    """
    checkpoint_path = Path(checkpoint_path)
    candidate = checkpoint_path.parent.parent
    if not (candidate / "config.json").is_file():
        raise FileNotFoundError(
            f"Could not locate run directory for {checkpoint_path}: "
            f"expected {candidate / 'config.json'} to exist."
        )
    return candidate
