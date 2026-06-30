"""Hybrid L1 + 3D SSIM loss for fMRI interpolation.

    loss = alpha * L1 + (1 - alpha) * (1 - SSIM)

L1 handles voxel-wise intensity, SSIM3D handles local 3D structure. SSIM is a
similarity score (1.0 = identical), so it enters the loss as 1 - SSIM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridL1SSIMLoss(nn.Module):
    """L1 plus full-volume 3D SSIM.

    Args:
        alpha: Weight on L1; (1 - alpha) weights (1 - SSIM).
        data_range: Intensity range for the SSIM constants. 2.0 for z-scored
            data, 1.0 for percentile-scaled data.
        kernel_size: Gaussian window size (must be odd).
        sigma: Gaussian standard deviation.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        data_range: float = 2.0,
        kernel_size: int = 7,
        sigma: float = 1.5,
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.alpha = float(alpha)
        self.data_range = float(data_range)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Buffers so .to(device) carries the kernels along with the module.
        self.register_buffer("kernel_d", kernel_1d.view(1, 1, kernel_size, 1, 1))
        self.register_buffer("kernel_h", kernel_1d.view(1, 1, 1, kernel_size, 1))
        self.register_buffer("kernel_w", kernel_1d.view(1, 1, 1, 1, kernel_size))

    def _separable_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        """Separable 3D Gaussian blur (three 1D passes = 21 taps vs 343)."""
        channels = x.shape[1]
        padding = self.kernel_size // 2

        # Depthwise: one kernel copy per channel so channels blur independently.
        kernel_d = self.kernel_d.to(dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
        kernel_h = self.kernel_h.to(dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
        kernel_w = self.kernel_w.to(dtype=x.dtype).repeat(channels, 1, 1, 1, 1)

        x = F.conv3d(x, kernel_d, padding=(padding, 0, 0), groups=channels)
        x = F.conv3d(x, kernel_h, padding=(0, padding, 0), groups=channels)
        x = F.conv3d(x, kernel_w, padding=(0, 0, padding), groups=channels)
        return x

    def _ssim3d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean full-volume SSIM for 5D tensors (B, C, D, H, W)."""
        # The variance terms are E[X^2] - E[X]^2. Under bf16 autocast that
        # difference loses precision, goes negative, and the divide blows up to
        # ~1e8. Force float32 here — and disable autocast explicitly, since
        # conv3d under autocast would otherwise downcast back to bf16.
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()

            c1 = (0.01 * self.data_range) ** 2
            c2 = (0.03 * self.data_range) ** 2

            mu_x = self._separable_gaussian(pred)
            mu_y = self._separable_gaussian(target)
            mu_x_sq = mu_x.pow(2)
            mu_y_sq = mu_y.pow(2)
            mu_xy = mu_x * mu_y

            # clamp_min(0): even in float32 the subtraction can dip slightly
            # negative on flat patches.
            sigma_x_sq = (self._separable_gaussian(pred * pred) - mu_x_sq).clamp_min(0.0)
            sigma_y_sq = (self._separable_gaussian(target * target) - mu_y_sq).clamp_min(0.0)
            sigma_xy = self._separable_gaussian(pred * target) - mu_xy

            numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
            return (numerator / denominator.clamp_min(1e-12)).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_components: bool = False,
    ):
        """Return the loss, optionally with detached {"l1", "ssim"} components.

        mask, if given, is applied to the L1 term only.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
        if pred.ndim != 5:
            raise ValueError(f"expected 5D tensors (B,C,D,H,W), got pred.ndim={pred.ndim}")

        abs_err = torch.abs(pred - target)

        if mask is None:
            l1 = abs_err.mean()
        else:
            if mask.shape != pred.shape:
                mask = torch.broadcast_to(mask, pred.shape)
            mask = mask.to(dtype=pred.dtype, device=pred.device)
            # clamp guards against an all-zero mask.
            l1 = (abs_err * mask).sum() / mask.sum().clamp(min=1)

        ssim = self._ssim3d(pred, target)
        loss = self.alpha * l1 + (1.0 - self.alpha) * (1.0 - ssim)

        if return_components:
            return loss, {"l1": l1.detach(), "ssim": ssim.detach()}
        return loss
