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
    still bound to ``seed + worker_id`` (see ``sr.data``).
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


def best_epoch_path(run_dir: Path) -> Path:
    """Canonical path of the rolling best checkpoint.

    Purpose:
        Give eval/infer a stable, name-independent handle to the best epoch
        so users no longer have to read ``metrics.json`` and hand-pick an
        ``epoch_NNN.pt``. The srcnn3d runs peak early then degrade after a
        loss spike, so "the last epoch" is often *not* the best one.
    Effect:
        Lives at ``<run_dir>/epochs/best.pt`` -- inside ``epochs/`` so that
        ``run_dir_for_checkpoint`` still resolves the owning run via
        ``parent.parent``. The ``epoch_*`` glob in ``list_epoch_files`` does
        not match ``best.pt``, so resume's ``find_latest_epoch`` ignores it
        and resume order is unchanged.
    Change guidance:
        Keep this under ``epochs/`` and keep the ``best.pt`` name; eval/infer
        commands and any user scripts point at this path.
    """
    return _epochs_dir(run_dir) / "best.pt"


def save_best_epoch(run_dir: Path, state: EpochState) -> Path:
    """Atomically (over)write ``epochs/best.pt`` with ``state``.

    Mirrors ``save_epoch`` (tmp + replace) so a crash mid-write leaves the
    previous ``best.pt`` intact. Payload format is identical to a per-epoch
    checkpoint, so ``load_epoch`` reads it without any special case.
    """
    path = best_epoch_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(_state_to_payload(state), tmp)
    tmp.replace(path)
    return path


def load_epoch(path: Path) -> EpochState:
    """Load an ``EpochState`` written by ``save_epoch``.

    ``weights_only=False`` is required because the payload intentionally
    contains non-tensor objects (numpy RNG state, metric history dicts).
    The file is produced by this code and lives next to the run's
    ``config.json``, so trust is on par with reading any other artifact
    from the run directory.

    Checkpoints are always loaded with ``map_location='cpu'``. Loading onto
    CUDA here rewrites every tensor in the file, including RNG byte blobs,
    which breaks ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all``.
    Callers move weights to the training device via ``load_state_dict`` on
    modules already on that device.
    """
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


def list_run_dirs_for_model(model_name: str, run_root: Path) -> list[Path]:
    """Return timestamped run directories for ``model_name``, oldest first."""
    model_dir = Path(run_root) / model_name
    if not model_dir.is_dir():
        return []
    return sorted(
        candidate.resolve()
        for candidate in model_dir.iterdir()
        if candidate.is_dir() and (candidate / "config.json").is_file()
    )


def resolve_checkpoint_for_model(
    model_name: str,
    *,
    run_root: Path | None = None,
    run_dir: Path | None = None,
    checkpoint: Path | None = None,
) -> Path:
    """Pick a checkpoint for ``model_name`` when the user did not pass one explicitly.

    Resolution order for an explicit ``run_dir``: ``epochs/best.pt``, else the
    latest ``epoch_NNN.pt``. Without ``run_dir``, use the newest run directory
    under ``run_root/<model_name>/`` with the same checkpoint preference.
    """
    if checkpoint is not None:
        ckpt = Path(checkpoint).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        if model_name is not None:
            saved_name = load_config_for_inference(
                ckpt, model_name=model_name
            ).model_name
            if saved_name != model_name:
                raise ValueError(
                    f"--model-name {model_name!r} disagrees with checkpoint "
                    f"config model_name={saved_name!r}."
                )
        return ckpt

    candidate_runs: list[Path]
    if run_dir is not None:
        candidate_runs = [Path(run_dir).resolve()]
    else:
        from sr.config import SRConfig

        root = Path(run_root or SRConfig().run_root)
        candidate_runs = list_run_dirs_for_model(model_name, root)
        if not candidate_runs:
            raise FileNotFoundError(
                f"No training runs found for model {model_name!r} under {root / model_name}. "
                "Train first or pass --checkpoint / --run-dir."
            )
        candidate_runs = [candidate_runs[-1]]

    run_path = candidate_runs[0]
    from sr.config import from_json

    saved_name = from_json(run_path / "config.json").model_name
    if saved_name != model_name:
        raise ValueError(
            f"--model-name {model_name!r} disagrees with {run_path}/config.json "
            f"model_name={saved_name!r}."
        )

    best = best_epoch_path(run_path)
    if best.is_file():
        return best
    latest = find_latest_epoch(run_path)
    if latest is not None:
        return latest
    raise FileNotFoundError(
        f"No checkpoints found under {run_path / 'epochs'}. Train or resume first."
    )


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

    Raises ``FileNotFoundError`` when the checkpoint is standalone (e.g.
    ``models/foo_best.pt``). Use ``try_run_dir_for_checkpoint`` or
    ``load_config_for_inference`` for portable weights.
    """
    checkpoint_path = Path(checkpoint_path)
    candidate = try_run_dir_for_checkpoint(checkpoint_path)
    if candidate is None:
        raise FileNotFoundError(
            f"Could not locate run directory for {checkpoint_path}. "
            "Standalone checkpoints need a sidecar <name>.config.json, "
            "an embedded config in the .pt file, or --config on the CLI."
        )
    return candidate


def try_run_dir_for_checkpoint(checkpoint_path: Path) -> Path | None:
    """Return the training run dir for a checkpoint, or ``None`` if unknown."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name != "epochs":
        return None
    candidate = checkpoint_path.parent.parent
    if (candidate / "config.json").is_file():
        return candidate
    return None


def find_config_json_for_checkpoint(checkpoint_path: Path) -> Path | None:
    """Locate a ``config.json`` for a checkpoint through several conventions.

    Search order:
      1. Training layout: ``<run_dir>/epochs/*.pt`` -> ``<run_dir>/config.json``
      2. Sidecar: ``<stem>.config.json`` next to the ``.pt`` file
      3. Sidecar: ``<stem>_config.json``
      4. ``config.json`` in the same directory as the checkpoint
    """
    ckpt = Path(checkpoint_path).expanduser().resolve()
    candidates: list[Path] = []
    run_dir = try_run_dir_for_checkpoint(ckpt)
    if run_dir is not None:
        candidates.append(run_dir / "config.json")
    stem = ckpt.stem
    candidates.extend(
        [
            ckpt.with_name(f"{stem}.config.json"),
            ckpt.with_name(f"{stem}_config.json"),
            ckpt.parent / "config.json",
        ]
    )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _config_from_checkpoint_payload(checkpoint_path: Path) -> "SRConfig | None":
    """Read an embedded ``config`` dict from a checkpoint payload, if present."""
    from sr.config import SRConfig, _config_from_dict

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    raw = payload.get("config")
    if raw is None:
        extra = payload.get("extra")
        if isinstance(extra, dict):
            raw = extra.get("config")
    if raw is None:
        return None
    if isinstance(raw, SRConfig):
        return raw
    if isinstance(raw, dict):
        return _config_from_dict(raw)
    return None


def load_config_for_inference(
    checkpoint_path: Path,
    *,
    model_name: str | None = None,
    config_path: Path | None = None,
) -> "SRConfig":
    """Build an ``SRConfig`` for eval/infer without requiring a full run directory.

    Resolution order: explicit ``config_path`` -> sidecar/run ``config.json``
    -> config embedded in the ``.pt`` payload -> minimal defaults when
    ``model_name`` is supplied (may not match custom ``model_kwargs``).
    """
    from sr.config import SRConfig, from_json

    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if config_path is not None:
        config = from_json(Path(config_path))
    else:
        sidecar = find_config_json_for_checkpoint(ckpt)
        if sidecar is not None:
            config = from_json(sidecar)
        else:
            embedded = _config_from_checkpoint_payload(ckpt)
            if embedded is not None:
                config = embedded
            elif model_name is not None:
                config = SRConfig(model_name=model_name)
                print(
                    f"[infer] warning: no config found beside {ckpt.name}; "
                    f"using SRConfig defaults for model_name={model_name!r}. "
                    "If weights fail to load, add a sidecar "
                    f"{ckpt.stem}.config.json or pass --config."
                )
            else:
                raise FileNotFoundError(
                    f"No config found for checkpoint {ckpt}. "
                    f"Add {ckpt.stem}.config.json next to the weights, embed "
                    "config in the .pt file, or pass --config / --model-name."
                )

    if model_name is not None and config.model_name != model_name:
        raise ValueError(
            f"model_name={model_name!r} disagrees with checkpoint config "
            f"model_name={config.model_name!r}."
        )
    return config


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint dict and verify it looks like an ``EpochState`` payload."""
    payload = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"File at {checkpoint_path} is not a valid EpochState payload."
        )
    return payload


def _verify_weights_match_config(
    payload: dict[str, Any], config: "SRConfig"
) -> None:
    """Fail fast when ``model_state_dict`` does not match ``config`` architecture."""
    from sr.models import build_model

    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)


def embed_config_in_checkpoint(
    checkpoint_path: Path,
    *,
    config_path: Path | None = None,
    backup: bool = True,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    """Embed ``SRConfig`` into ``payload['extra']['config']`` without touching weights.

    Purpose:
        Ship a single ``.pt`` file that infer can load without a sidecar JSON.
    Safety:
        - Resolves config from ``config_path`` or an existing sidecar/run JSON.
        - Verifies ``strict=True`` weight load before and after the write.
        - Refuses to overwrite an embedded config unless ``force=True``.
        - Writes via ``.tmp`` + atomic replace; optional ``.pt.bak`` backup first.
    """
    import shutil

    from sr.config import SRConfig, _config_to_dict, from_json, validate

    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    payload = _load_checkpoint_payload(ckpt)

    existing_raw: dict[str, Any] | None = None
    extra = payload.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("config"), dict):
        existing_raw = extra["config"]
    elif isinstance(payload.get("config"), dict):
        existing_raw = payload["config"]

    if existing_raw is not None and not force:
        embedded = _config_from_checkpoint_payload(ckpt)
        if embedded is None:
            raise ValueError(
                f"{ckpt.name} has an embedded config field but it could not be parsed."
            )
        validate(embedded)
        _verify_weights_match_config(payload, embedded)
        if dry_run:
            print(
                f"[embed-config] dry-run OK: already embeds model_name="
                f"{embedded.model_name!r} in {ckpt}"
            )
            return ckpt
        raise ValueError(
            f"{ckpt.name} already embeds a config. Re-run with --force to replace."
        )

    if config_path is not None:
        config = from_json(Path(config_path))
    else:
        sidecar = find_config_json_for_checkpoint(ckpt)
        if sidecar is None:
            raise FileNotFoundError(
                f"No config source for {ckpt}. Pass --config or add "
                f"{ckpt.stem}.config.json beside the checkpoint."
            )
        config = from_json(sidecar)

    validate(config)

    _verify_weights_match_config(payload, config)

    new_extra = dict(extra) if isinstance(extra, dict) else {}
    new_extra["config"] = _config_to_dict(config)
    payload["extra"] = new_extra
    payload.pop("config", None)

    _verify_weights_match_config(payload, config)

    if dry_run:
        print(
            f"[embed-config] dry-run OK: would embed model_name={config.model_name!r} "
            f"into {ckpt}"
        )
        return ckpt

    backup_path: Path | None = None
    if backup:
        backup_path = ckpt.with_name(ckpt.name + ".bak")
        shutil.copy2(ckpt, backup_path)
        print(f"[embed-config] backup -> {backup_path}")

    tmp = ckpt.with_name(ckpt.name + ".tmp")
    torch.save(payload, tmp)
    try:
        written = _load_checkpoint_payload(tmp)
        _verify_weights_match_config(written, config)
        embedded = _config_from_checkpoint_payload(tmp)
        if embedded is None or embedded.model_name != config.model_name:
            raise RuntimeError("Post-write verification failed: embedded config missing.")
        tmp.replace(ckpt)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    print(f"[embed-config] embedded config into {ckpt}")
    return ckpt
