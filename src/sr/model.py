"""Define SR model architectures and the model-selection factory.

Purpose:
    Keep architecture implementations and model instantiation policy in one
    module so training/eval/infer use the same construction path.
Effects:
    Determines how LR inputs are upsampled and transformed into HR predictions,
    and which architecture can be selected by config.
Influences:
    Effective model behavior depends on config-driven `model_name`,
    `output_patch_shape`, and optional `model_kwargs`.
How to change safely:
    Register new models in `MODEL_REGISTRY`, keep constructor signatures
    compatible with config-driven creation, and preserve expected tensor shapes.
"""

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN3D(nn.Module):
    """Baseline 3D SRCNN-style network for LR->HR volume reconstruction.

    Purpose:
        Provide a simple SR baseline with trilinear upsampling followed by
        shallow convolutional refinement.
    Effects:
        Produces HR predictions at `output_patch_shape`, which directly impacts
        loss/metric alignment with HR targets.
    Influences:
        Output geometry is controlled by `output_patch_shape`; learning behavior
        depends on optimizer/scheduler configuration in training.
    How to change safely:
        Keep single-channel input/output contract and ensure any architecture
        changes preserve output shape expected by training/eval code.
    """

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


class ResidualGroup3D(nn.Module):
    """Stack of RCAB-style blocks plus a grouping conv and a skip (RCAN RG).

    Purpose:
        Mirror the 2D `ResidualGroup` in ~/RCAN: several channel-attention
        residual blocks, then a 3×3×3 conv, with an additive skip around the
        whole group so gradients and low-frequency structure flow more easily.
    Effects:
        Each group refines features and fuses them before the next group; depth
        scales with `n_resblocks` per group and the number of groups in `RCAN3D`.
    Influences:
        VRAM and latency grow with `n_feats`, `n_resblocks`, and spatial size.
    How to change safely:
        Keep the final conv output channels equal to `n_feats` so the group skip
        `x + body(x)` stays shape-valid; match any topology change in checkpoints.
    """

    def __init__(self, n_feats: int, n_resblocks: int, reduction: int):
        super().__init__()
        layers: list[nn.Module] = [
            ResidualChannelAttentionBlock3D(n_feats=n_feats, reduction=reduction)
            for _ in range(n_resblocks)
        ]
        layers.append(nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCAN3D(nn.Module):
    """3D RCAN-style SR: residual groups (RCABs), global body skip, trilinear HR size.

    Purpose:
        Provide a capacity step above `SRCNN3D` while following the RCAN layout
        from the reference 2D code (~/RCAN): head → stacked residual groups →
        final body conv → global residual with head features → tail conv. LR is
        upsampled to `output_patch_shape` with trilinear interpolation (fixed HR
        grid from the manifest), not learnable 2D PixelShuffle, which fits this
        pipeline’s voxel geometry.
    Effects:
        Predicts single-channel HR at `output_patch_shape`; capacity and memory
        scale with `n_resgroups`, `n_resblocks`, and `n_feats`.
    Influences:
        Hyperparameters and optimizer settings in training config; cannot load
        2D RCAN weights without ad-hoc inflation (different dims and upsampler).
    How to change safely:
        Preserve NCDHW shapes; after changing group counts or widths, retrain or
        migrate checkpoints; tune `n_feats` first if GPU memory is tight.
    """

    def __init__(
        self,
        output_patch_shape=(50, 50, 50),
        n_feats: int = 32,
        n_resgroups: int = 2,
        n_resblocks: int = 2,
        reduction: int = 8,
    ):
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)
        self.head = nn.Conv3d(1, n_feats, kernel_size=3, padding=1)
        body_layers: list[nn.Module] = [
            ResidualGroup3D(n_feats=n_feats, n_resblocks=n_resblocks, reduction=reduction)
            for _ in range(n_resgroups)
        ]
        body_layers.append(nn.Conv3d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body_layers)
        self.tail = nn.Conv3d(n_feats, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, size=self.output_patch_shape, mode="trilinear", align_corners=False)
        feat = self.head(x)
        feat = feat + self.body(feat)
        return self.tail(feat)


class ChannelAttention3D(nn.Module):
    """Reweight feature channels based on global context.

    Purpose:
        Emphasize informative channels and suppress less useful responses inside
        RCAN blocks.
    Effects:
        Modifies feature amplitudes before residual addition, influencing what
        spatial/semantic details are carried forward.
    Influences:
        Strength and bottleneck size depend on `n_feats` and `reduction`.
    How to change safely:
        Keep output shape identical to input shape so residual paths remain
        valid; update RCAN defaults if changing reduction behavior.
    """

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
    """Local residual transform with channel-attention gating.

    Purpose:
        Learn refinements while preserving stable gradient flow via a residual
        skip connection.
    Effects:
        Adds attention-modulated residual features to the input tensor, shaping
        RCAN feature quality and convergence behavior.
    Influences:
        Behavior depends on convolution widths and attention reduction ratio.
    How to change safely:
        Preserve residual shape compatibility (`x + ...`) and keep activation
        ordering consistent unless retraining and benchmarks justify changes.
    """

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
    """Instantiate one registered SR architecture by name.

    Purpose:
        Decouple callers from concrete classes and centralize model-name
        validation.
    Effects:
        Determines which network class is created for train/eval/infer.
    Influences:
        Accepted names are governed by `MODEL_REGISTRY`; kwargs are forwarded
        directly to the target model constructor.
    How to change safely:
        Keep error messages explicit and ensure new registry entries have stable
        constructor arguments for config-driven usage.
    """
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    return MODEL_REGISTRY[key](**model_kwargs)


def build_model_from_config(config: dict) -> nn.Module:
    """Build a model from normalized runtime config fields.

    Purpose:
        Provide one consistent path from config dict to model instance.
    Effects:
        Applies defaults (`output_patch_shape`) and forwards custom kwargs,
        ensuring checkpoint/train/infer all construct compatible models.
    Influences:
        Model identity and constructor behavior depend on `model_name`,
        `model_kwargs`, and `output_patch_shape`.
    How to change safely:
        If config schema evolves, keep this function backward-compatible with
        checkpointed configs or add explicit migration logic.
    """
    model_name = str(config["model_name"])
    model_kwargs = dict(config.get("model_kwargs", {}))
    model_kwargs.setdefault("output_patch_shape", tuple(config["output_patch_shape"]))
    return select_model(model_name, **model_kwargs)
