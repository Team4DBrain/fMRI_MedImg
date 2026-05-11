"""3D super-resolution model architectures and the model registry.

Purpose:
    Keep all architectures and their construction in one module so train,
    eval and inference build models the same way from a config dict.
Effects:
    Determines how an LR volume is mapped to an HR prediction.
    ``output_patch_shape`` controls the trilinear pre-upsampling target;
    the rest of the network operates at HR resolution.
Influences:
    Choices come from the config (``model_name``, ``model_kwargs``,
    ``output_patch_shape``). Add new models by registering them in
    ``MODEL_REGISTRY`` with a constructor that accepts ``output_patch_shape``.
How to change safely:
    Keep input/output as single-channel ``(N, 1, D, H, W)`` tensors.
    Forward must return tensors shaped ``(N, 1, *output_patch_shape)``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.sr.config import SRConfig


class SRCNN3D(nn.Module):
    """3D SRCNN baseline: trilinear pre-upsampling + three conv layers.

    Purpose:
        Smallest viable SR model. Easy to read, cheap to train, useful as a
        sanity baseline before reaching for heavier architectures.
    Effects:
        Output shape always equals ``output_patch_shape``. Learns a
        per-voxel refinement of a trilinearly upsampled LR volume.
    Influences:
        Capacity is fixed; behaviour scales with optimizer/scheduler choice.
    How to change safely:
        Keep conv channels small and the in/out channel count at 1.
    """

    def __init__(self, output_patch_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=5, padding=2)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=self.output_patch_shape, mode="trilinear", align_corners=False
        )
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return self.conv3(x)


class _ChannelAttention3D(nn.Module):
    """Squeeze-excitation gate over feature channels (RCAN core block)."""

    def __init__(self, n_feats: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, n_feats // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.gate = nn.Sequential(
            nn.Conv3d(n_feats, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, n_feats, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class _RCAB3D(nn.Module):
    """Residual Channel-Attention Block: two convs, ReLU, attention gate."""

    def __init__(self, n_feats: int, reduction: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1),
        )
        self.attention = _ChannelAttention3D(n_feats, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.body(x))


class _ResidualGroup3D(nn.Module):
    """Stack of RCABs + grouping conv + group-level residual."""

    def __init__(self, n_feats: int, n_resblocks: int, reduction: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            _RCAB3D(n_feats, reduction) for _ in range(n_resblocks)
        ]
        blocks.append(nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCAN3D(nn.Module):
    """3D RCAN with trilinear LR->HR upsampling (no learnable upsampler).

    Purpose:
        A capacity step above ``SRCNN3D`` using residual groups with
        channel attention. Pre-upsamples LR to HR with trilinear
        interpolation (the HR grid is fixed by the manifest), then refines.
    Effects:
        Memory and compute scale with ``n_feats``, ``n_resgroups`` and
        ``n_resblocks``. Same single-channel I/O contract as ``SRCNN3D``.
    Influences:
        Defaults are chosen to fit on a small VRAM budget; bump ``n_feats``
        first when you have more headroom.
    How to change safely:
        Keep the final tail conv producing one channel and preserve the
        global head/body skip path so gradients flow.
    """

    def __init__(
        self,
        output_patch_shape: tuple[int, int, int],
        n_feats: int = 32,
        n_resgroups: int = 2,
        n_resblocks: int = 2,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)
        self.head = nn.Conv3d(1, n_feats, kernel_size=3, padding=1)
        groups: list[nn.Module] = [
            _ResidualGroup3D(n_feats, n_resblocks, reduction)
            for _ in range(n_resgroups)
        ]
        groups.append(nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*groups)
        self.tail = nn.Conv3d(n_feats, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=self.output_patch_shape, mode="trilinear", align_corners=False
        )
        features = self.head(x)
        features = features + self.body(features)
        return self.tail(features)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "srcnn3d": SRCNN3D,
    "rcan3d": RCAN3D,
}


def build_model(config: SRConfig) -> nn.Module:
    """Instantiate the model named in ``config`` with ``model_kwargs``.

    ``output_patch_shape`` is always passed; it cannot be overridden via
    ``model_kwargs`` because the data pipeline depends on it.
    """
    if config.model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{config.model_name}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    kwargs: dict[str, Any] = dict(config.model_kwargs)
    if "output_patch_shape" in kwargs and tuple(kwargs["output_patch_shape"]) != tuple(
        config.output_patch_shape
    ):
        raise ValueError(
            "output_patch_shape in model_kwargs disagrees with config.output_patch_shape; "
            "remove the duplicate or align them."
        )
    kwargs["output_patch_shape"] = tuple(config.output_patch_shape)
    return MODEL_REGISTRY[config.model_name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count. Used in startup banners."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
