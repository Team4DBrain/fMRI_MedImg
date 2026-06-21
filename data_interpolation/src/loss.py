"""loss.py — hybrid L1 + 3D SSIM loss for fMRI interpolation.

The model predicts a full 3D middle fMRI frame. We compare the prediction to
the real middle frame with two complementary terms:

    L1      -> voxel-wise intensity accuracy
    SSIM3D  -> local 3D structural similarity

Final loss:

    loss = alpha * L1 + (1 - alpha) * (1 - SSIM)

SSIM is a score where 1.0 is best, so the loss uses `1 - SSIM`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridL1SSIMLoss(nn.Module):
    """Compute L1 plus full-volume 3D SSIM.

    Args:
        alpha: Weight on L1. `1-alpha` is the weight on `1-SSIM`.
        data_range: Expected intensity range for SSIM constants.
            Use 2.0 for z-scored data, 1.0 for percentile-scaled data.
        kernel_size: Gaussian window size for local SSIM statistics.
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

        # The mixture coefficient must be a valid convex weight.
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        # Odd kernel sizes have a clean center voxel.
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        # Store simple scalar settings.
        self.alpha = float(alpha)
        self.data_range = float(data_range)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        # Build a 1D Gaussian kernel centered at zero.
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Register kernels as buffers so `.to(device)` moves them with the loss.
        self.register_buffer("kernel_d", kernel_1d.view(1, 1, kernel_size, 1, 1))
        self.register_buffer("kernel_h", kernel_1d.view(1, 1, 1, kernel_size, 1))
        self.register_buffer("kernel_w", kernel_1d.view(1, 1, 1, 1, kernel_size))

    def _separable_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a cheap separable 3D Gaussian blur.

        A full 7x7x7 kernel has 343 values. Three 1D passes need only 21
        values and are much cheaper for large 3D volumes.
        """
        # SSIM should blur each channel independently.
        channels = x.shape[1]

        # Same padding keeps D/H/W unchanged after convolution.
        padding = self.kernel_size // 2

        # Repeat the kernel once per channel for depthwise convolution.
        kernel_d = self.kernel_d.to(dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
        kernel_h = self.kernel_h.to(dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
        kernel_w = self.kernel_w.to(dtype=x.dtype).repeat(channels, 1, 1, 1, 1)

        # Blur along D, then H, then W.
        x = F.conv3d(x, kernel_d, padding=(padding, 0, 0), groups=channels)
        x = F.conv3d(x, kernel_h, padding=(0, padding, 0), groups=channels)
        x = F.conv3d(x, kernel_w, padding=(0, 0, padding), groups=channels)
        return x

    def _ssim3d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean full-volume SSIM for 5D tensors `(B,C,D,H,W)`."""
        # SSIM variance is `E[X^2] - E[X]^2`. Under bf16 autocast this cancels
        # to negative values and the divide explodes (~1e8). Disable autocast
        # AND cast to float32 — casting alone is not enough because conv3d
        # under autocast forces its inputs back to bf16 regardless.
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()

            # SSIM constants prevent division by zero in low-variance windows.
            c1 = (0.01 * self.data_range) ** 2
            c2 = (0.03 * self.data_range) ** 2

            # Local means.
            mu_x = self._separable_gaussian(pred)
            mu_y = self._separable_gaussian(target)

            # Mean products used in the SSIM equation.
            mu_x_sq = mu_x.pow(2)
            mu_y_sq = mu_y.pow(2)
            mu_xy = mu_x * mu_y

            # Local variances and covariance. Clamp to >=0 — even in float32
            # the subtraction can land slightly negative on homogeneous patches.
            sigma_x_sq = (self._separable_gaussian(pred * pred) - mu_x_sq).clamp_min(0.0)
            sigma_y_sq = (self._separable_gaussian(target * target) - mu_y_sq).clamp_min(0.0)
            sigma_xy = self._separable_gaussian(pred * target) - mu_xy

            # Standard SSIM formula.
            numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

            # Reduce to one scalar over batch, channels, and voxels.
            return (numerator / denominator.clamp_min(1e-12)).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_components: bool = False,
    ):
        """Return loss, optionally with detached L1/SSIM components.

        Args:
            pred: Predicted volume, shape `(B,1,D,H,W)`.
            target: Ground-truth middle volume, same shape as `pred`.
            mask: Optional brain-ish mask. It affects L1 only.
            return_components: If True, also return `{"l1": ..., "ssim": ...}`.
        """
        # Loss only makes sense when prediction and target align exactly.
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

        # PyTorch Conv3d expects 5D tensors.
        if pred.ndim != 5:
            raise ValueError(f"expected 5D tensors (B,C,D,H,W), got pred.ndim={pred.ndim}")

        # Per-voxel absolute error.
        abs_err = torch.abs(pred - target)

        # Default: average L1 over every voxel.
        if mask is None:
            l1 = abs_err.mean()

        else:
            # DataLoader gives mask shape `(B,1,D,H,W)`, but broadcast if needed.
            if mask.shape != pred.shape:
                mask = torch.broadcast_to(mask, pred.shape)

            # Match dtype/device before multiplying.
            mask = mask.to(dtype=pred.dtype, device=pred.device)

            # Masked average; clamp protects against an all-zero mask.
            l1 = (abs_err * mask).sum() / mask.sum().clamp(min=1)

        # SSIM remains full-volume by design in Phase A.
        ssim = self._ssim3d(pred, target)

        # Convert SSIM score into a loss term.
        loss = self.alpha * l1 + (1.0 - self.alpha) * (1.0 - ssim)

        # Training logs should not keep computation graphs alive.
        if return_components:
            return loss, {"l1": l1.detach(), "ssim": ssim.detach()}

        return loss
