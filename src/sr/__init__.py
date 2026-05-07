"""Re-export stable SR entry points used by CLI and external callers.

Purpose:
    Provide a concise import surface (`src.sr`) without exposing internal file
    structure details to downstream code.
Effects:
    Controls which symbols are considered public and import-stable.
Influences:
    Public surface changes when module-level exports or `__all__` are modified.
How to change safely:
    Add/remove exports intentionally and keep references in `run.py` and tests
    aligned to avoid import regressions.
"""

from .config import DEFAULT_CONFIG, apply_deterministic_policy, get_device, set_seed, validate_config
from .data import create_dataloaders
from .model import MODEL_REGISTRY, RCAN3D, SRCNN3D, build_model_from_config, select_model
from .training import run_training

__all__ = [
    "DEFAULT_CONFIG",
    "set_seed",
    "apply_deterministic_policy",
    "get_device",
    "validate_config",
    "MODEL_REGISTRY",
    "SRCNN3D",
    "RCAN3D",
    "select_model",
    "build_model_from_config",
    "create_dataloaders",
    "run_training",
]
