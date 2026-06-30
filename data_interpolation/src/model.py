"""3D U-Net for fMRI temporal interpolation.

Supervised mapping:

    input  x = [V_t, V_{t+2}]   (B, 2, D, H, W)
    output y = Vhat_{t+1}        (B, 1, D, H, W)

Spatial dims are preserved end to end (84 x 128 x 128 for the local data).
MaxPool3d floors odd dimensions on the way down, so each decoder step resizes
back to its skip tensor's shape instead of assuming the sizes line up. That's
why inputs never need padding or cropping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two (Conv3d -> norm -> activation) layers, spatial size unchanged."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # InstanceNorm instead of BatchNorm because we train at batch=1.
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net with a size-aware decoder.

    Default channel schedule: encoder 32 -> 64 -> 128 -> 256, bottleneck 512.

    Args:
        in_channels: 2 here (V_t and V_{t+2}).
        out_channels: 1 here (the predicted middle frame).
        base_channels: Width of the first level. Drop below 32 only for smoke tests.
        depth: Number of down/up levels before the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")
        if base_channels < 1:
            raise ValueError("base_channels must be >= 1")

        # e.g. [32, 64, 128, 256, 512] for the defaults.
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoder = nn.ModuleList()
        self.encoder.append(ConvBlock3D(in_channels, channels[0]))
        for i in range(1, depth + 1):
            self.encoder.append(ConvBlock3D(channels[i - 1], channels[i]))

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Each decoder block takes decoder features concatenated with the
        # matching encoder skip, hence channels[i] + channels[i - 1] inputs.
        self.decoder = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.decoder.append(ConvBlock3D(channels[i] + channels[i - 1], channels[i - 1]))

        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the missing middle frame. x: (B, 2, D, H, W) -> (B, 1, D, H, W)."""
        skips: list[torch.Tensor] = []

        h = x
        for enc in self.encoder[:-1]:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        # Bottleneck.
        h = self.encoder[-1](h)

        for dec, skip in zip(self.decoder, reversed(skips), strict=True):
            # Resize to the skip's exact D/H/W before concatenating, otherwise
            # odd-dimension flooring in the encoder causes a shape mismatch.
            h = F.interpolate(
                h,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        # No final activation: normalized fMRI can be negative.
        return self.head(h)


if __name__ == "__main__":
    model = UNet3D()
    x = torch.randn(2, 2, 84, 128, 128)
    with torch.no_grad():
        y = model(x)

    expected = (2, 1, 84, 128, 128)
    assert y.shape == expected, f"expected {expected}, got {tuple(y.shape)}"

    params = sum(p.numel() for p in model.parameters())
    print(f"out_shape={tuple(y.shape)} params={params:,}")
