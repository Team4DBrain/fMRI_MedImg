"""Joint denoise + spatial super-resolution model.

``JointRCAN3D``: one per-volume 3D network mapping a noisy low-res BOLD volume
``(1, 64, 64, 46)`` to a clean high-res volume ``(1, 128, 128, 93)``.

Topology (LR-heavy: most compute at 64x64x46, shallow refinement at HR):

    input (B,1,64,64,46)
      -> stem  Conv3d 1->C
      -> RIR body: n_groups x (n_blocks x RCAB) + global body skip        [LR]
      -> FactoredUpsampler3D:  PixelShuffle x2 in-plane (64->128), then
                               trilinear interpolate z 46->93, then fuse conv  [-> HR]
      -> hr_refine: hr_refine_blocks x RCAB                               [HR]
      -> head  Conv3d C->1 (linear, no clamp)
      -> + trilinear-upsampled input   => network predicts the HR residual
    output (B,1,128,128,93)

Design notes:
  - RCAB (RCAN-style residual channel-attention block); no BatchNorm — it would
    fight the per-run calibrated ~1.0 intensity and is degenerate at small batch.
  - The z axis is 46 -> 93 = x2.02, NOT a clean x2. PixelShuffle does the exact
    x2 in-plane only; the odd size 93 is reached by F.interpolate(size=...),
    which a fixed-stride x2 (giving 92) cannot do.
  - ``PixelShuffle3D`` channel ordering and ``icnr_init_`` are kept in this one
    file on purpose: ICNR yields a nearest-neighbour init for THIS shuffle's
    (C-outer, r-inner) channel layout. Swapping one without the other silently
    reintroduces checkerboard artifacts.
  - The output head is linear: the net predicts a residual over the trilinear
    baseline, so the head must be free to emit negative corrections. (Strict
    non-negativity, if a downstream consumer needs it, belongs at eval, never in
    the training forward.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class ChannelAttention3D(nn.Module):
    """SE-style channel attention (RCAN CA), lifted to 3D.

    NOTE: the global-avg-pool descriptor is low-frequency biased — if HR detail
    recovery underperforms, make it contrast-aware by adding a std-pool term.
    Isolated here so that is a one-place change.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class RCAB3D(nn.Module):
    """Residual channel-attention block: conv -> relu -> conv -> CA, + skip. No BN."""

    def __init__(self, channels: int, reduction: int = 16, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            ChannelAttention3D(channels, reduction),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class ResidualGroup3D(nn.Module):
    """A stack of RCABs + one conv, with a group-level long skip (RIR)."""

    def __init__(self, channels: int, n_blocks: int, reduction: int = 16,
                 res_scale: float = 1.0):
        super().__init__()
        self.blocks = nn.Sequential(
            *[RCAB3D(channels, reduction, res_scale) for _ in range(n_blocks)]
        )
        self.tail = nn.Conv3d(channels, channels, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.tail(self.blocks(x))


class PixelShuffle3D(nn.Module):
    """3D sub-pixel rearrange with independent per-axis upscale factors.

    Expects ``C * rx * ry * rz`` input channels in a (C-outer, (rx,ry,rz)-inner)
    layout and produces ``C`` output channels. This layout is exactly what
    ``icnr_init_`` assumes — keep the two in sync.
    """

    def __init__(self, rx: int, ry: int, rz: int):
        super().__init__()
        self.rx, self.ry, self.rz = rx, ry, rz

    def forward(self, x):
        b, crrr, X, Y, Z = x.shape
        r = self.rx * self.ry * self.rz
        c = crrr // r
        x = x.view(b, c, self.rx, self.ry, self.rz, X, Y, Z)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()   # b,c,X,rx,Y,ry,Z,rz
        return x.view(b, c, X * self.rx, Y * self.ry, Z * self.rz)


def icnr_init_(weight: torch.Tensor, upscale, init=nn.init.kaiming_normal_) -> None:
    """ICNR init (Aitken et al. 2017): initialise the sub-pixel conv so the
    PixelShuffle output equals a nearest-neighbour upsample at init, removing the
    checkerboard a sub-pixel layer otherwise starts with. Matches the
    (C-outer, r-inner) layout of ``PixelShuffle3D``.
    """
    r = upscale[0] * upscale[1] * upscale[2]
    out_c = weight.shape[0]
    sub = torch.zeros(out_c // r, *weight.shape[1:])
    init(sub)
    sub = sub.repeat_interleave(r, dim=0)
    with torch.no_grad():
        weight.copy_(sub)


class FactoredUpsampler3D(nn.Module):
    """``(B,C,64,64,46) -> (B,C,128,128,93)``.

    x2 in-plane via sub-pixel shuffle (exact), then z 46->93 via trilinear
    interpolate (the only way to hit the odd size 93), then a fuse conv that
    cleans interpolation softness / residual shuffle artifacts.
    """

    def __init__(self, channels: int, out_size=(128, 128, 93)):
        super().__init__()
        self.out_size = tuple(out_size)
        self.expand = nn.Conv3d(channels, channels * 4, 3, padding=1)   # rx*ry*rz = 2*2*1
        self.shuffle = PixelShuffle3D(2, 2, 1)
        self.act = nn.ReLU(inplace=True)
        self.fuse = nn.Conv3d(channels, channels, 3, padding=1)
        icnr_init_(self.expand.weight, upscale=(2, 2, 1))

    def forward(self, x):
        x = self.shuffle(self.expand(x))                              # (B,C,128,128,46)
        x = self.act(x)
        x = F.interpolate(x, size=self.out_size, mode="trilinear",
                          align_corners=False)                        # (B,C,128,128,93)
        return self.fuse(x)


class JointRCAN3D(nn.Module):
    def __init__(self, in_ch: int = 1, channels: int = 96, n_groups: int = 4,
                 n_blocks: int = 6, reduction: int = 16, res_scale: float = 0.1,
                 hr_refine_blocks: int = 3, out_size=(128, 128, 93),
                 use_checkpoint: bool = False):
        super().__init__()
        self.out_size = tuple(out_size)
        self.use_checkpoint = use_checkpoint

        self.head = nn.Conv3d(in_ch, channels, 3, padding=1)            # stem (LR)
        self.groups = nn.ModuleList(
            [ResidualGroup3D(channels, n_blocks, reduction, res_scale)
             for _ in range(n_groups)]
        )
        self.body_tail = nn.Conv3d(channels, channels, 3, padding=1)    # closes the global body skip
        self.upsampler = FactoredUpsampler3D(channels, out_size=self.out_size)
        self.hr_refine = nn.Sequential(
            *[RCAB3D(channels, reduction, res_scale) for _ in range(hr_refine_blocks)]
        )
        self.tail = nn.Conv3d(channels, in_ch, 3, padding=1)           # output head (linear)

    def forward(self, x):                                              # x: (B,1,64,64,46)
        base = F.interpolate(x, size=self.out_size, mode="trilinear",
                             align_corners=False)                      # (B,1,128,128,93)
        feat = self.head(x)                                            # (B,C,64,64,46)
        body = feat
        for g in self.groups:
            if self.use_checkpoint and self.training:
                body = checkpoint(g, body, use_reentrant=False)
            else:
                body = g(body)
        feat = feat + self.body_tail(body)                             # RIR global skip (LR)
        up = self.upsampler(feat)                                      # (B,C,128,128,93)
        up = self.hr_refine(up)
        out = self.tail(up)                                            # (B,1,128,128,93)
        return out + base                                             # predict the HR residual


def build_model(cfg: ModelConfig) -> JointRCAN3D:
    return JointRCAN3D(
        in_ch=cfg.in_ch, channels=cfg.channels, n_groups=cfg.n_groups,
        n_blocks=cfg.n_blocks, reduction=cfg.reduction, res_scale=cfg.res_scale,
        hr_refine_blocks=cfg.hr_refine_blocks, out_size=cfg.out_size,
        use_checkpoint=cfg.use_checkpoint,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Minimal architecture smoke test (no pytest). Uses the tiny "smoke" config
    # so it runs on a weak local GPU / CPU. Run: python -m src.joint.model
    from .config import build_config

    cfg = build_config("smoke")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    print(f"[model-smoke] device={device} params={count_params(model) / 1e6:.3f}M")

    x = torch.rand(2, 1, 64, 64, 46, device=device)
    out = model(x)
    assert out.shape == (2, 1, 128, 128, 93), out.shape
    assert torch.isfinite(out).all()
    print(f"[model-smoke] forward OK {tuple(out.shape)}")

    out.sum().backward()
    named = dict(model.named_parameters())
    for name in ("head.weight", "upsampler.expand.weight", "tail.weight"):
        g = named[name].grad
        assert g is not None and g.abs().sum() > 0, f"no grad at {name}"
    print("[model-smoke] grad flow OK (head, upsampler.expand, tail)")
    print("[model-smoke] PASS")
