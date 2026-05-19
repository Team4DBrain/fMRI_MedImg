"""Runtime configuration for the spatial-SR pipeline.

Purpose:
    Provide one explicit, serialisable place that holds every value driving
    training, evaluation and inference. Every field has a default so a fresh
    user can run the pipeline without guessing parameters, while the saved
    ``config.json`` makes each completed run fully reproducible.
Effects:
    Drives model/data/optimizer construction and the train/val split. The
    JSON form lands in the run directory and is the single source of truth
    when a run is resumed.
Influences:
    CLI flags in ``src.sr.cli`` override defaults at construction time.
    ``validate(...)`` is called once before any expensive work to fail fast
    on inconsistent values.
How to change safely:
    Add a new field with a default. Update ``validate`` if the field needs
    range/registry checks. Keep ``to_json``/``from_json`` in sync if the
    field is not JSON-serialisable out of the box (Path/tuple are handled).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_MANIFEST_PATH = Path("/srv/venvs/team4dbrain/derivatives/manifest.json")
DEFAULT_RUN_ROOT = Path("src/sr/runs")
DEFAULT_OUTPUT_SHAPE: tuple[int, int, int] = (128, 128, 93)
DEFAULT_PATCH_HR_SHAPE: tuple[int, int, int] = (48, 48, 48)
# Must match valid 9-1-5 shrink in SRCNN3DPatch (+1 so output has >=1 voxel).
MIN_PATCH_HR_EDGE = 13


@dataclass
class SRConfig:
    """All knobs the SR pipeline reads at runtime.

    Defaults reflect the IBC dataset at 1.5 mm (HR) -> 3 mm (LR) with the
    no_crop_v1 data pipeline. Edit ``config.json`` directly inside a run
    directory if you want to resume with a different number of epochs.
    """

    # Reproducibility / safety
    seed: int = 42
    deterministic: bool = True
    strict_finite_loss: bool = True

    # Paths
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    run_root: Path = DEFAULT_RUN_ROOT

    # Model
    model_name: str = "srcnn3d"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    output_patch_shape: tuple[int, int, int] = DEFAULT_OUTPUT_SHAPE

    # Patch training (used when model_name == "srcnn3d_patch")
    patch_hr_shape: tuple[int, int, int] = DEFAULT_PATCH_HR_SHAPE
    patches_per_volume: int = 32

    # Spatial degradation
    source_voxel_mm: float = 1.5
    target_voxel_mm: float = 3.0

    # Dataset split (0.8/0.2 by default; seeded random split over all samples)
    train_split: float = 0.8

    # Training loop
    batch_size: int = 15
    num_epochs: int = 20
    num_workers: int = 0
    log_interval: int = 10

    # Loss + optimizer + scheduler (modular: swap names, kwargs stay JSON)
    loss_name: str = "masked_mse"
    optimizer_name: str = "adam"
    learning_rate: float = 1e-3
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_name: str = "plateau"
    scheduler_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"factor": 0.5, "patience": 3}
    )

    # Logging
    tensorboard: bool = True


def to_json(config: SRConfig, path: Path) -> None:
    """Write ``config`` to ``path`` as pretty-printed JSON.

    Atomic via tmp + replace so an interrupt cannot leave a half-written
    file that breaks a later resume.
    """
    payload = _config_to_dict(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def from_json(path: Path) -> SRConfig:
    """Load an ``SRConfig`` from a JSON file written by ``to_json``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _config_from_dict(raw)


def _config_to_dict(config: SRConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["manifest_path"] = str(config.manifest_path)
    payload["run_root"] = str(config.run_root)
    payload["output_patch_shape"] = list(config.output_patch_shape)
    payload["patch_hr_shape"] = list(config.patch_hr_shape)
    return payload


def _config_from_dict(raw: dict[str, Any]) -> SRConfig:
    kwargs: dict[str, Any] = dict(raw)
    if "manifest_path" in kwargs:
        kwargs["manifest_path"] = Path(kwargs["manifest_path"])
    if "run_root" in kwargs:
        kwargs["run_root"] = Path(kwargs["run_root"])
    if "output_patch_shape" in kwargs:
        kwargs["output_patch_shape"] = tuple(kwargs["output_patch_shape"])
    if "patch_hr_shape" in kwargs:
        kwargs["patch_hr_shape"] = tuple(kwargs["patch_hr_shape"])
    return SRConfig(**kwargs)


def validate(config: SRConfig) -> None:
    """Fail fast on inconsistent or impossible configurations.

    Performs registry lookups via late imports to avoid a circular import
    chain (``models``/``losses``/``components`` all read ``SRConfig`` types).
    """
    from src.sr.components import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
    from src.sr.losses import LOSS_REGISTRY
    from src.sr.models import MODEL_REGISTRY

    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if config.log_interval < 1:
        raise ValueError("log_interval must be >= 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if not 0.0 < config.train_split <= 1.0:
        raise ValueError("train_split must be in (0, 1]")
    if config.source_voxel_mm <= 0 or config.target_voxel_mm <= 0:
        raise ValueError("source_voxel_mm and target_voxel_mm must be > 0")
    if len(config.output_patch_shape) != 3:
        raise ValueError("output_patch_shape must have 3 entries (D, H, W)")
    if len(config.patch_hr_shape) != 3:
        raise ValueError("patch_hr_shape must have 3 entries (D, H, W)")
    if config.patches_per_volume < 1:
        raise ValueError("patches_per_volume must be >= 1")

    if config.model_name == "srcnn3d_patch":
        for axis, edge in enumerate(config.patch_hr_shape):
            if edge < MIN_PATCH_HR_EDGE:
                raise ValueError(
                    f"patch_hr_shape[{axis}]={edge} must be >= {MIN_PATCH_HR_EDGE} "
                    f"so valid conv output has at least one voxel per axis"
                )
            if edge > config.output_patch_shape[axis]:
                raise ValueError(
                    f"patch_hr_shape[{axis}]={edge} exceeds "
                    f"output_patch_shape[{axis}]={config.output_patch_shape[axis]}"
                )

    if config.model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{config.model_name}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    if config.loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_name '{config.loss_name}'. "
            f"Available: {sorted(LOSS_REGISTRY)}"
        )
    if config.optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer_name '{config.optimizer_name}'. "
            f"Available: {sorted(OPTIMIZER_REGISTRY)}"
        )
    if config.scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler_name '{config.scheduler_name}'. "
            f"Available: {sorted(SCHEDULER_REGISTRY)}"
        )

    manifest = Path(config.manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest_path does not exist: {manifest}")


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python/NumPy/torch and toggle deterministic backends.

    Calling this once in the main process is enough; DataLoader workers
    are seeded separately via ``worker_init_fn`` in ``src.sr.data``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def auto_device() -> str:
    """Pick ``cuda`` when available, else ``cpu``. Single source of truth."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def summary(config: SRConfig) -> str:
    """Human-readable, one-screen block of the active configuration.

    Used right after ``validate`` so users see exactly what the run will do
    before any data is read. The order mirrors the dataclass for greppability.
    """
    lines = [
        "Configuration",
        f"  seed              = {config.seed}",
        f"  deterministic     = {config.deterministic}",
        f"  strict_finite     = {config.strict_finite_loss}",
        f"  manifest_path     = {config.manifest_path}",
        f"  run_root          = {config.run_root}",
        f"  model_name        = {config.model_name}",
        f"  model_kwargs      = {config.model_kwargs}",
        f"  output_patch      = {tuple(config.output_patch_shape)}",
        f"  patch_hr_shape    = {tuple(config.patch_hr_shape)}",
        f"  patches_per_vol   = {config.patches_per_volume}",
        f"  source/target_mm  = {config.source_voxel_mm} -> {config.target_voxel_mm}",
        f"  train_split       = {config.train_split}",
        f"  batch_size        = {config.batch_size}",
        f"  num_epochs        = {config.num_epochs}",
        f"  num_workers       = {config.num_workers}",
        f"  log_interval      = {config.log_interval}",
        f"  loss_name         = {config.loss_name}",
        f"  optimizer_name    = {config.optimizer_name}",
        f"  learning_rate     = {config.learning_rate}",
        f"  optimizer_kwargs  = {config.optimizer_kwargs}",
        f"  scheduler_name    = {config.scheduler_name}",
        f"  scheduler_kwargs  = {config.scheduler_kwargs}",
        f"  tensorboard       = {config.tensorboard}",
    ]
    return "\n".join(lines)
