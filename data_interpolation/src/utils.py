"""utils.py — shared helpers: device selection, seeding, config loading."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def pick_device(requested: str | None = None) -> torch.device:
    """Choose CUDA > MPS > CPU when no device is explicitly requested."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` and return `base`."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(config: dict, dotted_pairs: list[str]) -> dict:
    """Apply CLI overrides like `train.epochs=10` to a nested config dict."""
    for pair in dotted_pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got: {pair}")
        key, raw = pair.split("=", 1)
        # Try to parse YAML scalar so numbers/bools become typed values.
        value = yaml.safe_load(raw)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return config
