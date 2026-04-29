"""Configuration and reproducibility utilities for SR training."""

import random
from pathlib import Path

import numpy as np
import torch

INPUT_DIM = 64
OUTPUT_DIM = 128

DEFAULT_CONFIG = {
    "seed": 42,
    "batch_size": 8,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "train_split": 0.9,
    "num_workers": 0,
    "log_interval": 10,
    "checkpoint_interval": 1,
    "run_root": Path("./runs"),
    "checkpoint_root": Path("./checkpoints"),
    "data_file": Path("./data/degraded_list.npy"),
    "gt_file": Path("./data/gt_list.npy"),
    "input_patch_shape": (INPUT_DIM, INPUT_DIM, INPUT_DIM),
    "output_patch_shape": (OUTPUT_DIM, OUTPUT_DIM, OUTPUT_DIM),
    "samples_per_timepoint": 2,
    "resume_checkpoint": None,
}


def set_seed(seed: int) -> None:
    """Set RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return preferred torch device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_config(config: dict) -> None:
    """Validate configuration constraints for SR setup."""
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
