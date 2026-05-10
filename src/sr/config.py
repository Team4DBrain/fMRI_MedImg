"""Define and validate runtime configuration for the spatial SR pipeline.

Purpose:
    Centralize defaults and reproducibility policy so train/eval/infer resolve
    behavior from one source of truth.
Effects:
    Directly affects model selection, data loading, optimization settings,
    output geometry, and safety checks in `src.sr.run` and `src.sr.training`.
Influences:
    Values can be overridden by CLI flags in `src.sr.run._apply_overrides` and
    partially replaced by checkpoint config for eval/infer.
How to change safely:
    Keep keys aligned with `run.py` override handling and
    `training.py`/`data.py` consumers; update `validate_config` whenever adding
    new config fields that can break runtime assumptions.
"""

import random
from pathlib import Path

import numpy as np
import torch

from .model import MODEL_REGISTRY

INPUT_DIM_X = 64
INPUT_DIM_Y = 64
INPUT_DIM_Z = 46
INPUT_DIM = (INPUT_DIM_X, INPUT_DIM_Y, INPUT_DIM_Z)
OUTPUT_DIM_X = 128
OUTPUT_DIM_Y = 128
OUTPUT_DIM_Z = 93
OUTPUT_DIM = (OUTPUT_DIM_X, OUTPUT_DIM_Y, OUTPUT_DIM_Z)
LOSS_NAMES = ("masked_mse", "mse", "masked_l1", "l1")

DEFAULT_CONFIG = {
    "seed": 42,
    "deterministic": True,
    "batch_size": 4,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "train_split": 0.8,
    "num_workers": 0,
    "log_interval": 10,
    "checkpoint_interval": 1,
    "run_root": Path("./src/sr/runs"),
    "manifest_path": Path("./manifest.json"),
    "input_patch_shape": INPUT_DIM,
    "output_patch_shape": OUTPUT_DIM,
    "source_voxel_mm": 1.5,
    "target_voxel_mm": 3.0,
    "train_subjects": None,
    "val_subjects": None,
    "enable_subject_split": False,
    "model_name": "srcnn3d",
    "model_kwargs": {},
    "loss_name": "masked_mse",
    "resume_checkpoint": None,
    "strict_finite_loss": True,
}


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds and backend policy for reproducible experiments.

    Purpose:
        Make data order, initialization, and stochastic ops repeatable.
    Effects:
        Changes Python/NumPy/PyTorch RNG streams and then applies deterministic
        backend policy, which affects runtime speed vs reproducibility.
    Influences:
        Effective behavior depends on CUDA availability and the `deterministic`
        flag passed from config/CLI.
    How to change safely:
        Keep this as the single place where all RNGs are seeded before training
        or evaluation; if a new RNG source is introduced, seed it here too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    apply_deterministic_policy(deterministic)


def apply_deterministic_policy(deterministic: bool) -> None:
    """Toggle deterministic algorithm/backends used by PyTorch.

    Purpose:
        Control the trade-off between reproducibility and throughput.
    Effects:
        Enables deterministic algorithms and disables cuDNN benchmarking when
        requested; otherwise restores faster non-deterministic defaults.
    Influences:
        Behavior differs on CPU-only vs CUDA runs because cuDNN flags only apply
        when available.
    How to change safely:
        If backend flags are updated, keep both branches explicit so users can
        still intentionally choose reproducible or performance-oriented runs.
    """
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


def get_device() -> str:
    """Return the default compute device for SR commands.

    Purpose:
        Provide one consistent auto-selection policy for train/eval/infer.
    Effects:
        Determines where tensors/models are allocated when the user does not
        pass `--device`.
    Influences:
        Depends on `torch.cuda.is_available()` at runtime.
    How to change safely:
        Keep return values compatible with `argparse` `--device` choices and
        with `.to(device)` calls throughout the SR module.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_config(config: dict) -> None:
    """Fail fast when config values would produce invalid or unsafe runs.

    Purpose:
        Catch misconfiguration before expensive training/evaluation starts.
    Effects:
        Rejects impossible splits, invalid optimization settings, unknown model
        names, malformed kwargs, and missing manifest path.
    Influences:
        Validation criteria depend on model registry contents and whether
        subject splitting is enabled.
    How to change safely:
        Add checks whenever new config keys drive behavior in data/model/trainer
        code, so errors remain early and actionable.
    """
    if bool(config.get("enable_subject_split", False)):
        if not 0.0 < float(config["train_split"]) <= 1.0:
            raise ValueError("train_split must be in (0, 1]. when subject split is enabled.")
    if int(config["batch_size"]) < 1:
        raise ValueError("batch_size must be >= 1.")
    if int(config["num_epochs"]) < 1:
        raise ValueError("num_epochs must be >= 1.")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers must be >= 0.")
    if int(config["checkpoint_interval"]) < 1:
        raise ValueError("checkpoint_interval must be >= 1.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be > 0.")
    if float(config["source_voxel_mm"]) <= 0 or float(config["target_voxel_mm"]) <= 0:
        raise ValueError("source_voxel_mm and target_voxel_mm must be > 0.")
    loss_name = str(config.get("loss_name", "masked_mse")).strip().lower()
    if loss_name not in LOSS_NAMES:
        available = ", ".join(LOSS_NAMES)
        raise ValueError(f"Unknown loss_name '{config.get('loss_name')}'. Available: {available}")

    model_name = str(config["model_name"]).strip().lower()
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name '{config['model_name']}'. Available: {available}")
    if not isinstance(config.get("model_kwargs", {}), dict):
        raise ValueError("model_kwargs must be a dictionary.")

    manifest_path = Path(config["manifest_path"])
    if not manifest_path.exists():
        raise ValueError(f"manifest_path does not exist: {manifest_path}")
