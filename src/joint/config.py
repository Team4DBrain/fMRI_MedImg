"""Configuration for the joint denoise + spatial-SR model (src/joint).

One architecture, two presets:

  - "vm"    : the real model, sized for the project VM (~half an H100, ~47 GB).
              This is what training actually uses. Optimised for quality/capacity,
              NOT for any local GPU.
  - "smoke" : a deliberately tiny instantiation of the SAME architecture, used
              only to exercise code paths (forward / shape / grad, overfit-a-few)
              on a weak local GPU before uploading to the VM. A test convenience,
              never a modelling choice.

Everything is plain dataclasses so a full config can be asdict()'d into a
checkpoint for reproducibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    in_ch: int = 1
    channels: int = 96           # C — LR-body feature width
    n_groups: int = 4            # residual groups (RIR)
    n_blocks: int = 6            # RCABs per group
    reduction: int = 16          # channel-attention squeeze ratio
    res_scale: float = 0.1       # residual-branch scaling (EDSR-style; 1.0 for shallow nets)
    hr_refine_blocks: int = 3    # RCABs at HR after upsampling
    out_size: tuple = (128, 128, 93)   # HR target (X, Y, Z)
    use_checkpoint: bool = False       # gradient checkpointing on the residual groups


@dataclass
class TrainConfig:
    # --- data / degradation (forwarded to JointDataset) ---
    source_voxel_mm: float = 1.5
    target_voxel_mm: float = 3.0
    sigma_min: float = 0.03
    sigma_max: float = 0.10
    # --- optimisation ---
    lr: float = 4e-4
    betas: tuple = (0.9, 0.99)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    charb_eps: float = 1e-3
    epochs: int = 100
    batch_size: int = 8
    val_batch_size: int = 4
    grad_accum: int = 1
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    # --- AMP ---
    use_amp: bool = True
    amp_dtype: str = "bf16"       # "bf16" | "fp16" | "auto"
    # --- data loading ---
    num_workers: int = 8
    persistent_workers: bool = True
    # --- eval / reproducibility ---
    val_every: int = 1
    seed: int = 1234
    deterministic: bool = False
    cudnn_benchmark: bool = True


@dataclass
class Config:
    profile: str = "vm"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # provenance (filled at runtime, saved into the checkpoint)
    manifest_path: str = ""
    manifest_sha256: str = ""
    git_commit: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Per-profile overrides. Anything not listed keeps the dataclass default above.
_PROFILES = {
    "vm": dict(
        model=dict(channels=96, n_groups=4, n_blocks=6, reduction=16,
                   res_scale=0.1, hr_refine_blocks=3, use_checkpoint=False),
        train=dict(batch_size=8, val_batch_size=4, lr=4e-4, grad_accum=1,
                   warmup_steps=500, use_amp=True, amp_dtype="bf16",
                   num_workers=8, persistent_workers=True, cudnn_benchmark=True),
    ),
    "smoke": dict(
        # ~1M params — measured to fit ~2.7 GB at batch 1 on an 8 GB card, and
        # large enough to overfit convincingly (probe: +7 dB over baseline on one
        # sample). It is NOT the modelling choice — just a local-test size.
        model=dict(channels=32, n_groups=3, n_blocks=4, reduction=16,
                   res_scale=1.0, hr_refine_blocks=1, use_checkpoint=False),
        train=dict(batch_size=2, val_batch_size=2, lr=3e-4, grad_accum=1,
                   warmup_steps=10, use_amp=False, amp_dtype="fp16",
                   num_workers=0, persistent_workers=False, cudnn_benchmark=False),
    ),
}


def build_config(profile: str = "vm") -> Config:
    """Build a Config from a named profile ('vm' or 'smoke')."""
    if profile not in _PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(_PROFILES)}")
    cfg = Config(profile=profile)
    overrides = _PROFILES[profile]
    for k, v in overrides.get("model", {}).items():
        setattr(cfg.model, k, v)
    for k, v in overrides.get("train", {}).items():
        setattr(cfg.train, k, v)
    return cfg
