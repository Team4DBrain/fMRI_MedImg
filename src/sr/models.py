"""3D super-resolution model architectures and the model registry.

Purpose:
    Keep all architectures and their construction in one module so train,
    eval and inference build models the same way from a config dict.
Effects:
    Determines how an LR volume is mapped to an HR prediction.
    ``output_patch_shape`` controls the trilinear pre-upsampling target;
    the rest of the network operates at HR resolution. ``model_name`` in
    the config picks which class is instantiated -- existing run dirs and
    checkpoints are bound to the name they were trained with, so renaming
    a registered key breaks resume/eval/infer for those runs.
Influences:
    Choices come from the config (``model_name``, ``model_kwargs``,
    ``output_patch_shape``). Add new architectures by registering them in
    ``MODEL_REGISTRY`` with a constructor that accepts ``output_patch_shape``.
    The CLI's ``--model-name`` flag (``cli.py``) and ``validate()`` in
    ``config.py`` both read this registry, so a single new entry shows up
    everywhere automatically -- no other files need editing.
Naming convention for SRCNN-family variants:
    ``srcnn3d``        - canonical baseline. Original SRCNN-style 9-1-5
                         layout with a global residual skip and small-init
                         reconstruction head (~53k params). Stable, easy
                         to read, the default architecture comparisons
                         start from.
    ``srcnn3d_deep``   - stacked 3x3x3 feature extractor + 3x3 mapping +
                         LeakyReLU(0.1), still with the global residual
                         skip and small-init head (~280k params). More
                         expressive than ``srcnn3d`` but also more
                         expensive per voxel.
    Future variants should follow ``srcnn3d_<distinctive-feature>`` (e.g.
    ``srcnn3d_attn``) so users can scan the registry without reading code.
How to change safely:
    Keep input/output as single-channel ``(N, 1, D, H, W)`` tensors.
    Forward must return tensors shaped ``(N, 1, *output_patch_shape)``.
    Do not silently change the architecture behind an existing
    ``model_name``: register a new key instead, otherwise old run dirs
    fail to resume because their saved ``state_dict`` no longer matches.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.sr.config import SRConfig


class SRCNN3D(nn.Module):
    """Classic 3D SRCNN (9-1-5 kernels) with a global residual skip.

    Purpose:
        Canonical 3D-SR baseline. Mirrors the original 2014 SRCNN layout
        adapted to 3D: a wide first conv (kernel 9), a ``1x1`` non-linear
        mapping conv, and a 5x5x5 reconstruction conv. Stable, easy to
        read, and the reference every other architecture in this module
        (``SRCNN3DDeep``, ``RCAN3D``) is benchmarked against.
    Effects:
        Output shape always equals ``output_patch_shape``. ``forward``
        returns ``trilinear(x) + conv3(features)``: the reconstruction
        conv is initialised with a tiny std, so at step 0 the model emits
        essentially the trilinear baseline. This stabilises early
        optimisation but also means absolute loss values look small from
        the first step -- the meaningful number is the gap between the
        model and the trilinear baseline (see ``baseline_*`` metrics in
        ``metrics.json``), not the raw loss.
    Influences:
        Capacity is fixed (~53k parameters), small enough to train fast
        on a tight VRAM budget. Only one ReLU non-linearity in the
        feature path and a ``1x1`` mapping conv that discards spatial
        context -- those are the trade-offs ``SRCNN3DDeep`` addresses
        when you want more expressiveness at higher cost.
    How to change safely:
        Treat the architecture as frozen so existing run dirs trained on
        this name keep resuming cleanly. If you want to tweak it,
        register a new ``MODEL_REGISTRY`` key (e.g. ``srcnn3d_<feature>``)
        instead of editing this class.
    """

    def __init__(self, output_patch_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=5, padding=2)

        # Kaiming for the ReLU stack, then overwrite the reconstruction
        # head with a tiny std so the residual is ~0 at init -- model
        # starts at the trilinear baseline. Same recipe as ``SRCNN3DDeep``.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(
            x, size=self.output_patch_shape, mode="trilinear", align_corners=False
        )
        h = F.relu(self.conv1(x_up))
        h = F.relu(self.conv2(h))
        return x_up + self.conv3(h)


class SRCNN3DDeep(nn.Module):
    """Deeper SRCNN-3D variant: stacked 3x3 convs + LeakyReLU + residual.

    Purpose:
        Higher-capacity SRCNN-family alternative for when ``SRCNN3D`` (the
        9-1-5 baseline) plateaus. Replaces the wide single ``9^3`` conv
        with a stack of three ``3^3`` convs and the ``1x1`` mapping conv
        with a ``3x3x3`` one, so spatial context is preserved on the
        whole path to the reconstruction head and there are more
        non-linearities to learn through.
    Effects:
        Output shape always equals ``output_patch_shape``. ``forward``
        returns ``trilinear(x) + recon(features)``. The reconstruction
        head is initialised with a very small std so at step 0 the model
        emits essentially the trilinear baseline. Same caveat as
        ``SRCNN3D``: the meaningful number is the gap to the trilinear
        baseline (``baseline_*`` metrics in ``metrics.json``), not the
        raw loss.
    Influences:
        Architecture vs. ``SRCNN3D`` (the 9-1-5 baseline):
          * Feature extractor: three stacked ``3x3x3`` convs replace the
            original single ``9x9x9`` conv. The total receptive field of
            the whole network (13 voxels) is unchanged versus the 9-1-5
            layout, but stacking adds two extra non-linearities, which
            is what actually lifts representational capacity. Note: in
            3D with 64 channels per layer the inner ``64->64`` convs
            dominate compute, so this is **more** expensive than the
            single ``9^3`` first layer -- the gain is expressiveness,
            not cheaper compute.
          * Non-linear mapping: ``3x3x3`` instead of ``1x1``, which
            preserves spatial context on the path to the head.
          * Activation: ``LeakyReLU(0.1)`` avoids dying units under the
            small-std reconstruction-head init.
        Parameter count ~280k (vs ~53k for ``SRCNN3D``). For even larger
        models add residual blocks (cf. ``RCAN3D``) rather than widening
        these convs further.
    How to change safely:
        Keep single-channel I/O and the global ``x_up + residual`` skip.
        If you change the activation, re-tune the Kaiming ``a`` parameter
        and the reconstruction-head init together: those are paired
        choices that keep training stable around the baseline.
    """

    def __init__(self, output_patch_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)
        self.feat1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.feat2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.feat3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.map = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.recon = nn.Conv3d(32, 1, kernel_size=5, padding=2)

        # Kaiming init tuned for LeakyReLU(slope=0.1) keeps per-layer
        # activation variance roughly constant through the depth so we do
        # not need any normalisation layer. The reconstruction head is
        # then *overwritten* with a tiny std so the residual is ~0 at
        # init: the model starts at the trilinear baseline and training
        # only has to learn the correction on top.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.recon.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.recon.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(
            x, size=self.output_patch_shape, mode="trilinear", align_corners=False
        )
        h = F.leaky_relu(self.feat1(x_up), negative_slope=0.1)
        h = F.leaky_relu(self.feat2(h), negative_slope=0.1)
        h = F.leaky_relu(self.feat3(h), negative_slope=0.1)
        h = F.leaky_relu(self.map(h), negative_slope=0.1)
        return x_up + self.recon(h)


# Valid 9-1-5 convolutions shrink each spatial dim by this many voxels.
SRCNN3D_PATCH_RECEPTIVE_SHRINK = 12


class SRCNN3DPatch(nn.Module):
    """3D SRCNN with valid convolutions and paper-style patch training.

    Purpose:
        Experiment variant of ``SRCNN3D`` that matches the original SRCNN
        training recipe: no padding (valid conv), loss on the central valid
        region, and patch-based training via ``PatchTrainingDataset``.
    Effects:
        Output spatial size is ``upsample_to`` minus ``SRCNN3D_PATCH_RECEPTIVE_SHRINK``
        per axis. Callers crop ``target``/``mask_hr`` with ``center_crop_to_pred``.
        Full-volume val/infer pass ``upsample_to=output_patch_shape`` (via target shape).
    Influences:
        ``patch_hr_shape`` and ``patches_per_volume`` in ``SRConfig`` drive training
        crops only; ``output_patch_shape`` still sets full-volume HR grid size.
    How to change safely:
        Keep kernel sizes 9/1/5 and padding=0 so ``SRCNN3D_PATCH_RECEPTIVE_SHRINK``
        stays in sync with ``src.sr.shape_utils``.
    """

    def __init__(self, output_patch_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.output_patch_shape = tuple(output_patch_shape)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding=0)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=5, padding=0)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        upsample_to: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        size = tuple(upsample_to) if upsample_to is not None else self.output_patch_shape
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
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
    "srcnn3d_patch": SRCNN3DPatch,
    "srcnn3d_deep": SRCNN3DDeep,
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
