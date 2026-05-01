"""Configuration and reproducibility utilities for SR training."""

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

DEFAULT_CONFIG = {
    "seed": 42,
    "deterministic": True,
    "batch_size": 8,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "train_split": 0.9,
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
    "model_name": "srcnn3d",
    "model_kwargs": {},
    "samples_per_timepoint": 2,
    "resume_checkpoint": None,
    "strict_finite_loss": True,
}


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    apply_deterministic_policy(deterministic)


def apply_deterministic_policy(deterministic: bool) -> None:
    """Apply deterministic backend policy for reproducible runs."""
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
    """Return preferred torch device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_config(config: dict) -> None:
    """Validate configuration constraints for SR setup."""
    if not 0.0 < float(config["train_split"]) <= 1.0:
        raise ValueError("train_split must be in (0, 1].")
    if int(config["batch_size"]) < 1:
        raise ValueError("batch_size must be >= 1.")
    if int(config["num_epochs"]) < 1:
        raise ValueError("num_epochs must be >= 1.")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers must be >= 0.")
    if int(config["samples_per_timepoint"]) < 1:
        raise ValueError("samples_per_timepoint must be >= 1.")
    if int(config["checkpoint_interval"]) < 1:
        raise ValueError("checkpoint_interval must be >= 1.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be > 0.")
    if float(config["source_voxel_mm"]) <= 0 or float(config["target_voxel_mm"]) <= 0:
        raise ValueError("source_voxel_mm and target_voxel_mm must be > 0.")

    model_name = str(config["model_name"]).strip().lower()
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name '{config['model_name']}'. Available: {available}")
    if not isinstance(config.get("model_kwargs", {}), dict):
        raise ValueError("model_kwargs must be a dictionary.")

    if any(
        o <= i
        for i, o in zip(
            config["input_patch_shape"],
            config["output_patch_shape"],
        )
    ):
        raise ValueError(
            "For super-resolution, output_patch_shape must be larger than input_patch_shape in every dimension"
        )
    manifest_path = Path(config["manifest_path"])
    if not manifest_path.exists():
        raise ValueError(f"manifest_path does not exist: {manifest_path}")
