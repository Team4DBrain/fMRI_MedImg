"""model.py — 3D U-Net for fMRI temporal interpolation.

The model learns this supervised mapping:

    input  x = [V_t, V_{t+2}]      shape (B, 2, D, H, W)
    output y = Vhat_{t+1}          shape (B, 1, D, H, W)

The important project constraint is that spatial dimensions are preserved.
For the local data that means:

    input  spatial shape: (D=84, H=128, W=128)
    output spatial shape: (D=84, H=128, W=128)

The encoder uses MaxPool3d, which floors odd dimensions during downsampling.
To avoid shape drift in the decoder, every upsampling step explicitly resizes
to the skip-connection tensor's spatial shape. This is why inputs do not need
padding, cropping, or resizing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Small reusable block: Conv3d -> norm -> activation, repeated twice."""

    def __init__(self, in_channels: int, out_channels: int):
        """Create one U-Net convolution block.

        Args:
            in_channels: Number of feature channels entering this block.
            out_channels: Number of feature channels leaving this block.
        """
        super().__init__()

        # Padding=1 keeps the spatial size unchanged for a 3x3x3 kernel.
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # InstanceNorm is stable for small batch sizes, including batch=1.
            nn.InstanceNorm3d(out_channels, affine=True),
            # LeakyReLU avoids dead activations while staying cheap.
            nn.LeakyReLU(0.01, inplace=True),
            # A second convolution gives the block more local 3D context.
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer 3D convolution block."""
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net with a size-aware decoder.

    Default channels are:

        encoder: 32 -> 64 -> 128 -> 256
        bottleneck: 512
        decoder: 256 -> 128 -> 64 -> 32

    Args:
        in_channels: Input channels. For this project, 2 = V_t and V_{t+2}.
        out_channels: Output channels. For this project, 1 = Vhat_{t+1}.
        base_channels: Width of the first level. Keep 32 for real training;
            reduce it only for quick smoke tests.
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

        # Fail early on invalid architecture settings.
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if base_channels < 1:
            raise ValueError("base_channels must be >= 1")

        # Channel schedule, e.g. [32, 64, 128, 256, 512] for defaults.
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Encoder stores one ConvBlock per resolution level.
        self.encoder = nn.ModuleList()

        # First encoder block consumes the raw two-channel fMRI input.
        self.encoder.append(ConvBlock3D(in_channels, channels[0]))

        # Later encoder blocks consume feature maps from the previous level.
        for i in range(1, depth + 1):
            self.encoder.append(ConvBlock3D(channels[i - 1], channels[i]))

        # Downsampling halves D/H/W at each level, flooring odd dimensions.
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Decoder blocks run from bottleneck back to the original resolution.
        self.decoder = nn.ModuleList()
        for i in range(depth, 0, -1):
            # Each decoder block receives concatenated channels:
            #   current decoder features + matching encoder skip features.
            self.decoder.append(ConvBlock3D(channels[i] + channels[i - 1], channels[i - 1]))

        # Final 1x1x1 convolution maps features to one regression channel.
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the missing middle fMRI volume.

        Args:
            x: Tensor shaped (B, 2, D, H, W).

        Returns:
            Tensor shaped (B, 1, D, H, W), with the same spatial size as x.
        """
        # Save encoder outputs so the decoder can recover fine spatial detail.
        skips: list[torch.Tensor] = []

        # h is the feature tensor that flows through the network.
        h = x

        # Run all encoder levels except the last one; the last is bottleneck.
        for enc in self.encoder[:-1]:
            # Extract local 3D features at the current resolution.
            h = enc(h)

            # Save this resolution for the matching decoder skip connection.
            skips.append(h)

            # Move to the next coarser resolution.
            h = self.pool(h)

        # Bottleneck: deepest, lowest-resolution representation.
        h = self.encoder[-1](h)

        # Walk back up through the decoder, pairing each block with a skip.
        for dec, skip in zip(self.decoder, reversed(skips), strict=True):
            # Critical shape-preservation step: resize to skip's exact D/H/W.
            h = F.interpolate(
                h,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

            # Concatenate encoder detail and decoder context on channel dim.
            h = torch.cat([skip, h], dim=1)

            # Fuse the concatenated features.
            h = dec(h)

        # Regression head. No activation because normalized fMRI can be negative.
        return self.head(h)


if __name__ == "__main__":
    # Self-check used during development and before real training.
    model = UNet3D()

    # Native Phase A tensor shape: batch=2, channels=2, D=84, H=128, W=128.
    x = torch.randn(2, 2, 84, 128, 128)

    # No gradients are needed for a shape sanity check.
    with torch.no_grad():
        y = model(x)

    # The whole architectural point: output spatial shape must match input.
    expected = (2, 1, 84, 128, 128)
    assert y.shape == expected, f"expected {expected}, got {tuple(y.shape)}"

    # Print parameter count so the user can verify the expected model size.
    params = sum(p.numel() for p in model.parameters())
    print(f"out_shape={tuple(y.shape)} params={params:,}")
