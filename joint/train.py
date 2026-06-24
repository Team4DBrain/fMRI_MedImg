"""Training, overfit-sanity, and evaluation for the joint model.

Quick local code smoke (no data needed)::

    python -m src.joint.train

Add a real-data overfit sanity check (a few JointDataset volumes)::

    python -m src.joint.train --manifest ../_local_derivatives/manifest.json

Real training is driven by ``train(cfg, train_loader, val_loader)`` — on the VM,
build the subject-disjoint train/val loaders and call it. The overfit sanity
check is the gate to clear before any full run.

The overfit check is deliberately strict: because the model output is
``residual + trilinear(input)``, the trilinear skip alone already scores well,
so "loss went down" is NOT proof the SR body works. We additionally require the
trained model to BEAT the trilinear baseline, clear a per-sample PSNR floor, and
not collapse the in-brain variance (mean-predictor guard).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, build_config
from .losses import masked_charbonnier, masked_psnr, masked_ssim_3d
from .model import build_model, count_params


# --------------------------------------------------------------------------
# Reproducibility / provenance
# --------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # NOTE: torch.use_deterministic_algorithms(True) may raise on 3D-conv
        # backward (no deterministic kernel on some CUDA builds). If so, fall back
        # to seed-only reproducibility and document it in the run log.
    else:
        torch.backends.cudnn.benchmark = benchmark


def source_git_commit() -> str:
    """HEAD of the SOURCE repo (where this file lives), not the data/manifest dir."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def manifest_sha256(manifest_path: str) -> str:
    try:
        return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    except Exception:
        return "unknown"


def stamp_provenance(cfg: Config, manifest_path: str | None) -> Config:
    cfg.git_commit = source_git_commit()
    if manifest_path:
        cfg.manifest_path = str(manifest_path)
        cfg.manifest_sha256 = manifest_sha256(manifest_path)
    return cfg


# --------------------------------------------------------------------------
# AMP
# --------------------------------------------------------------------------

def resolve_amp(tcfg, device):
    """Return (enabled, autocast_dtype, GradScaler). fp16 needs the scaler; bf16
    (and disabled) do not. bf16 on Ampere works but is slow — the 'vm' profile
    uses bf16 for the H100; locally AMP is off in the 'smoke' profile."""
    enabled = bool(tcfg.use_amp) and device.type == "cuda"
    if not enabled:
        return False, torch.float32, torch.amp.GradScaler(device.type, enabled=False)
    if tcfg.amp_dtype == "bf16" or (tcfg.amp_dtype == "auto" and torch.cuda.is_bf16_supported()):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))
    return True, dtype, scaler


def make_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    """Linear warmup -> cosine decay, stepped once per optimizer step."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, tcfg, device,
                    amp_enabled, amp_dtype, step, epoch=0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    t_log, samples_since = time.time(), 0
    for i, batch in enumerate(loader):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        mh = batch["mask_hr"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            pred = model(x)
            loss = masked_charbonnier(pred, y, mh, eps=tcfg.charb_eps)
        scaler.scale(loss / tcfg.grad_accum).backward()
        samples_since += x.size(0)
        if (i + 1) % tcfg.grad_accum == 0:
            if tcfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if tcfg.log_interval and (step == 1 or step % tcfg.log_interval == 0):
                dt = time.time() - t_log
                sps = samples_since / dt if dt > 0 else 0.0
                print(f"[train] e{epoch + 1} step {step} batch {i + 1}/{n_batches} "
                      f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{sps:.1f} samp/s", flush=True)
                t_log, samples_since = time.time(), 0
    # Flush a trailing partial accumulation window so its grads aren't carried
    # into the next epoch, and so the optimizer-step count matches total_steps.
    if len(loader) % tcfg.grad_accum != 0:
        if tcfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
    return step


@torch.no_grad()
def evaluate(model, loader, device, charb_eps=1e-3):
    model.eval()
    tot = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "n": 0}
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        mh = batch["mask_hr"].to(device)
        pred = model(x)
        b = x.size(0)
        tot["loss"] += masked_charbonnier(pred, y, mh, charb_eps).item() * b
        tot["psnr"] += masked_psnr(pred, y, mh).item() * b
        tot["ssim"] += masked_ssim_3d(pred, y, mh).item() * b
        tot["n"] += b
    n = max(1, tot["n"])
    return {"loss": tot["loss"] / n, "psnr": tot["psnr"] / n, "ssim": tot["ssim"] / n}


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, epoch, cfg, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": cfg.to_dict(),       # full config incl. seeds, charb_eps, manifest hash, git commit
        "metrics": metrics,
        "torch_version": torch.__version__,
    }, path)


def train(cfg: Config, train_loader, val_loader, device=None, ckpt_dir="checkpoints"):
    """Full training driver (intended for the VM). Caller builds the loaders."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.train.seed, cfg.train.deterministic, cfg.train.cudnn_benchmark)
    model = build_model(cfg.model).to(device)
    print(f"[train] profile={cfg.profile} params={count_params(model) / 1e6:.2f}M device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            betas=tuple(cfg.train.betas), weight_decay=cfg.train.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.train.grad_accum))
    total_steps = cfg.train.epochs * steps_per_epoch
    sched = make_scheduler(opt, cfg.train.warmup_steps, total_steps, cfg.train.min_lr_ratio)
    amp_enabled, amp_dtype, scaler = resolve_amp(cfg.train, device)

    step, best = 0, float("inf")
    for epoch in range(cfg.train.epochs):
        step = train_one_epoch(model, train_loader, opt, sched, scaler, cfg.train,
                               device, amp_enabled, amp_dtype, step, epoch=epoch)
        if (epoch + 1) % cfg.train.val_every == 0:
            metrics = evaluate(model, val_loader, device, cfg.train.charb_eps)
            print(f"[train] epoch {epoch + 1} step {step} | "
                  f"val loss {metrics['loss']:.4f} psnr {metrics['psnr']:.2f} "
                  f"ssim {metrics['ssim']:.4f}", flush=True)
            if metrics["loss"] < best:
                best = metrics["loss"]
                save_checkpoint(Path(ckpt_dir) / "best.pt", model, opt, sched,
                                scaler, step, epoch, cfg, metrics)
        save_checkpoint(Path(ckpt_dir) / "last.pt", model, opt, sched,
                        scaler, step, epoch, cfg, metrics=None)
    return model


# --------------------------------------------------------------------------
# Overfit sanity check (the gate before any full run)
# --------------------------------------------------------------------------

def _cache_samples(dataset, k, device):
    """Read the first k samples once and pin them on the device. Reading once is
    load-bearing: JointDataset draws FRESH Rician noise per __getitem__, so the
    noisy input must be frozen for the loss to be able to collapse."""
    xs, ys, ms = [], [], []
    for i in range(k):
        s = dataset[i]
        xs.append(s["input"])
        ys.append(s["target"])
        ms.append(s["mask_hr"])
    return (torch.stack(xs).to(device),
            torch.stack(ys).to(device),
            torch.stack(ms).to(device))


def overfit_sanity(cfg: Config, dataset, device, k=1, steps=600, lr=5e-4,
                   beat_loss_frac=0.75, beat_db=2.0, per_sample_beat_db=1.0,
                   var_ratio_range=(0.4, 2.5)):
    """Overfit k fixed samples and assert the SR body actually contributes.

    Because the output is ``residual + trilinear(input)``, "loss went down" is not
    proof of anything — the trilinear skip alone already scores ~18 dB. So every
    bar is RELATIVE to that baseline: the trained model must cut the masked loss by
    a margin, beat the baseline PSNR (mean AND per-sample), and not collapse the
    in-brain variance (mean-predictor guard).

    Absolute high-PSNR / near-zero-loss bars are deliberately NOT used: the HR
    target carries scanner noise the model cannot memorise, so an absolute-fidelity
    bar would be unreachable by construction, not a sign of a bug. (Probe: a ~1M
    model reaches ~25 dB / +7 dB over baseline on one sample and is still rising.)
    """
    set_seed(cfg.train.seed)
    model = build_model(cfg.model).to(device)
    print(f"[overfit] profile={cfg.profile} params={count_params(model) / 1e6:.3f}M "
          f"k={k} steps={steps} device={device}")
    x, y, m = _cache_samples(dataset, k, device)

    # Trilinear-skip baseline — what the trained model must beat.
    base = F.interpolate(x, size=model.out_size, mode="trilinear", align_corners=False)
    base_loss = masked_charbonnier(base, y, m, cfg.train.charb_eps).item()
    base_psnr = masked_psnr(base, y, m).item()
    base_ps = masked_psnr(base, y, m, per_sample=True)             # (k,)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    model.train()
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = masked_charbonnier(model(x), y, m, cfg.train.charb_eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % max(1, steps // 6) == 0:
            print(f"[overfit] step {s:4d} loss {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(x)
        final_loss = masked_charbonnier(pred, y, m, cfg.train.charb_eps).item()
        ps_psnr = masked_psnr(pred, y, m, per_sample=True)         # (k,)
        mean_psnr = float(ps_psnr.mean())
        per_sample_gain = float((ps_psnr - base_ps).min())         # worst sample's gain over its own baseline
        mb = m > 0.5
        var_ratio = pred[mb].float().var().item() / (y[mb].float().var().item() + 1e-12)

    checks = {
        "beat_baseline_loss": final_loss < beat_loss_frac * base_loss,
        "beat_baseline_psnr": mean_psnr > base_psnr + beat_db,
        "per_sample_beats_baseline": per_sample_gain > per_sample_beat_db,
        "var_ratio_ok": var_ratio_range[0] < var_ratio < var_ratio_range[1],
    }
    print(f"[overfit] baseline : loss {base_loss:.5f}  psnr {base_psnr:.2f} dB")
    print(f"[overfit] trained  : loss {final_loss:.6f}  psnr {mean_psnr:.2f} dB  "
          f"(+{mean_psnr - base_psnr:.2f} mean, +{per_sample_gain:.2f} worst)  var_ratio {var_ratio:.2f}")
    for name, ok in checks.items():
        print(f"[overfit]   [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print(f"[overfit] {'PASS' if passed else 'FAIL'}")
    assert passed, f"overfit sanity FAILED: { {n: v for n, v in checks.items() if not v} }"
    return passed


# --------------------------------------------------------------------------
# __main__ code smoke (no pytest)
# --------------------------------------------------------------------------

def _smoke():
    ap = argparse.ArgumentParser(description="Joint model code smoke test")
    ap.add_argument("--manifest", default=None,
                    help="optional manifest.json path; if given, run a real-data overfit sanity")
    ap.add_argument("--profile", default="smoke", help="config profile (default: smoke)")
    args = ap.parse_args()

    cfg = build_config(args.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device} torch={torch.__version__} profile={cfg.profile}")

    # (a) build
    model = build_model(cfg.model).to(device)
    print(f"[smoke] (a) model built: {count_params(model) / 1e6:.3f}M params")

    # (b) forward synthetic -> assert HR shape + finite
    x = torch.rand(2, 1, 64, 64, 46, device=device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 128, 128, 93), out.shape
    assert torch.isfinite(out).all()
    print(f"[smoke] (b) forward OK {tuple(out.shape)}")

    # (c) one backward + step; assert grads reach stem, upsampler, AND output head
    #     (proves the main path is in autograd, not just the parameter-free skip)
    model.train()
    y = torch.rand(2, 1, 128, 128, 93, device=device)
    m = (torch.rand(2, 1, 128, 128, 93, device=device) > 0.3).float()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    opt.zero_grad(set_to_none=True)
    loss = masked_charbonnier(model(x), y, m, cfg.train.charb_eps)
    loss.backward()
    named = dict(model.named_parameters())
    for name in ("head.weight", "upsampler.expand.weight", "tail.weight"):
        g = named[name].grad
        assert g is not None and g.abs().sum() > 0, f"no gradient at {name}"
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"[smoke] (c) backward+step OK loss={loss.item():.4f} gradnorm={float(gn):.3f}")

    # (d) real-data overfit sanity (only if a manifest is given)
    if args.manifest:
        from src.data.datasets import JointDataset
        ds = JointDataset(
            args.manifest,
            source_voxel_mm=cfg.train.source_voxel_mm,
            target_voxel_mm=cfg.train.target_voxel_mm,
            sigma_min=cfg.train.sigma_min,
            sigma_max=cfg.train.sigma_max,
        )
        s0 = ds[0]
        assert s0["input"].shape == (1, 64, 64, 46), s0["input"].shape
        assert s0["target"].shape == (1, 128, 128, 93), s0["target"].shape
        assert s0["mask_hr"].shape == (1, 128, 128, 93), s0["mask_hr"].shape
        print("[smoke] (d) JointDataset contract OK; running overfit sanity...")
        overfit_sanity(cfg, ds, device)
    else:
        print("[smoke] (d) skipped (pass --manifest <path> for the real-data overfit)")

    print("[smoke] ALL PASS")


if __name__ == "__main__":
    _smoke()
