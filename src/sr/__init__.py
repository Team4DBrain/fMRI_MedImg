"""SR training package split from legacy notebook."""

from .checks import run_sanity_checks, run_tiny_overfit_check
from .config import (
    DEFAULT_CONFIG,
    INPUT_DIM,
    OUTPUT_DIM,
    apply_deterministic_policy,
    get_device,
    set_seed,
    validate_config,
)
from .data import SRSpatialManifestDataset, create_dataloaders
from .model import RCAN3D, SRCNN3D, build_model_from_config, select_model
from .training import (
    build_training_components,
    ensure_finite_loss,
    maybe_resume_training,
    psnr_from_mse,
    run_training,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)

__all__ = [
    "INPUT_DIM",
    "OUTPUT_DIM",
    "DEFAULT_CONFIG",
    "set_seed",
    "apply_deterministic_policy",
    "get_device",
    "validate_config",
    "SRCNN3D",
    "RCAN3D",
    "select_model",
    "build_model_from_config",
    "SRSpatialManifestDataset",
    "create_dataloaders",
    "psnr_from_mse",
    "ensure_finite_loss",
    "validate_one_epoch",
    "train_one_epoch",
    "save_checkpoint",
    "maybe_resume_training",
    "build_training_components",
    "run_training",
    "run_sanity_checks",
    "run_tiny_overfit_check",
]
