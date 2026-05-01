"""Model definitions and factory helpers for 3D super-resolution."""

import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable


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
    """3D RCAN-style network for super-resolution patches."""

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
