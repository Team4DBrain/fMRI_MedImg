"""Joint denoise + spatial super-resolution model (consumes data.JointDataset).

Public API::

    from joint import build_config, build_model
    cfg = build_config("vm")          # or "smoke" for local code tests
    model = build_model(cfg.model)    # JointRCAN3D: (B,1,64,64,46) -> (B,1,128,128,93)
"""
from .config import Config, ModelConfig, TrainConfig, build_config
from .losses import masked_charbonnier, masked_psnr, masked_ssim_3d
from .model import JointRCAN3D, build_model, count_params

__all__ = [
    "Config",
    "ModelConfig",
    "TrainConfig",
    "build_config",
    "JointRCAN3D",
    "build_model",
    "count_params",
    "masked_charbonnier",
    "masked_psnr",
    "masked_ssim_3d",
]
