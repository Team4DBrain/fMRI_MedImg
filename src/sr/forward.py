"""Model forward dispatch for architecture-specific upsampling.

Purpose:
    Centralise how ``srcnn3d_patch`` chooses its trilinear upsampling target
    so train, eval and infer stay consistent without importing the training loop.
Effects:
    ``SRCNN3DPatch`` upsamples to the batch HR spatial size (patch or full).
    Other models ignore ``target`` and use their built-in ``output_patch_shape``.
How to change safely:
    Add new branches only when a model needs non-default forward kwargs.
"""

from __future__ import annotations

import torch
from torch import nn


def model_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    model_name: str,
) -> torch.Tensor:
    """Run ``model`` on ``inputs``; patch variant upsamples to ``target`` spatial size."""
    if model_name == "srcnn3d_patch":
        return model(inputs, upsample_to=tuple(target.shape[-3:]))
    return model(inputs)
