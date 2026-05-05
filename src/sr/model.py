"""Model definitions and factory helpers for 3D super-resolution."""

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN3D(nn.Module):
    """3D SRCNN-style network for super-resolution patches."""

    def __init__(self, output_patch_shape=(50, 50, 50)):
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)

        self._initialize_weights()

    def forward(self, x):
        x = F.interpolate(x, size=self.output_patch_shape, mode="trilinear", align_corners=False)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class RCAN3D(nn.Module):
    """Lightweight 3D RCAN variant with channel attention."""

    def __init__(
        self,
        output_patch_shape=(50, 50, 50),
        n_feats: int = 32,
        n_blocks: int = 4,
        reduction: int = 8,
    ):
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)
        self.head = nn.Conv3d(1, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualChannelAttentionBlock3D(n_feats=n_feats, reduction=reduction) for _ in range(n_blocks)]
        )
        self.tail = nn.Conv3d(n_feats, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, size=self.output_patch_shape, mode="trilinear", align_corners=False)
        feat = self.head(x)
        feat = feat + self.body(feat)
        return self.tail(feat)


class ChannelAttention3D(nn.Module):
    """Squeeze-and-excitation style channel attention for 3D feature maps."""

    def __init__(self, n_feats: int, reduction: int):
        super().__init__()
        hidden = max(1, n_feats // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(n_feats, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, n_feats, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.mlp(self.pool(x))
        return x * weights


class ResidualChannelAttentionBlock3D(nn.Module):
    """Residual block with channel attention."""

    def __init__(self, n_feats: int, reduction: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1),
        )
        self.ca = ChannelAttention3D(n_feats=n_feats, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.block(x))


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "srcnn3d": SRCNN3D,
    "rcan3d": RCAN3D,
}


def select_model(model_name: str, **model_kwargs: Any) -> nn.Module:
    """Instantiate a model from the registry."""
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    return MODEL_REGISTRY[key](**model_kwargs)


def build_model_from_config(config: dict) -> nn.Module:
    """Build model using config keys only."""
    model_name = str(config["model_name"])
    model_kwargs = dict(config.get("model_kwargs", {}))
    model_kwargs.setdefault("output_patch_shape", tuple(config["output_patch_shape"]))
    return select_model(model_name, **model_kwargs)
