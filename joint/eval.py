"""Evaluate a trained joint model on a subject split (typically the held-out test).

    python -m joint.eval \\
        --manifest /srv/venvs/team4dbrain/derivatives/manifest.json \\
        --ckpt runs/joint01/best.pt --subjects 07 13

Loads the model from the checkpoint's *saved* config (so the architecture and
degradation parameters match training exactly), runs brain-masked PSNR / SSIM /
Charbonnier over the chosen subjects, and reports overall + per-subject numbers.
Voxel-fidelity metrics only — fMRI-specific evaluation (tSNR, task-GLM
preservation) is a separate, later step.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .config import Config, ModelConfig, TrainConfig
from .losses import masked_charbonnier, masked_psnr, masked_ssim_3d
from .model import build_model
from .splits import build_dataset


def rebuild_config(cfg_dict: dict) -> Config:
    """Reconstruct a Config from a checkpoint's asdict()'d config."""
    return Config(
        profile=cfg_dict.get("profile", "vm"),
        model=ModelConfig(**cfg_dict["model"]),
        train=TrainConfig(**cfg_dict["train"]),
        manifest_path=cfg_dict.get("manifest_path", ""),
        manifest_sha256=cfg_dict.get("manifest_sha256", ""),
        git_commit=cfg_dict.get("git_commit", ""),
    )


def load_checkpoint(ckpt_path, device):
    """Return (model_in_eval_mode, config, raw_checkpoint)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = rebuild_config(ck["config"])
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


@torch.no_grad()
def evaluate_subjects(model, cfg: Config, manifest_path, subjects, device,
                      per_subject: bool = True) -> dict:
    """Masked PSNR/SSIM/loss over the subjects, plus an 'all' aggregate."""
    groups: dict[str, list[str]] = {}
    if per_subject:
        for s in subjects:
            groups[s] = [s]
    groups["all"] = list(subjects)

    results = {}
    for name, subs in groups.items():
        ds = build_dataset(cfg, manifest_path, subs)
        loader = DataLoader(ds, batch_size=cfg.train.val_batch_size,
                            shuffle=False, num_workers=0)
        tot = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "n": 0}
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            mh = batch["mask_hr"].to(device)
            pred = model(x)
            b = x.size(0)
            tot["loss"] += masked_charbonnier(pred, y, mh, cfg.train.charb_eps).item() * b
            tot["psnr"] += masked_psnr(pred, y, mh).item() * b
            tot["ssim"] += masked_ssim_3d(pred, y, mh).item() * b
            tot["n"] += b
        n = max(1, tot["n"])
        results[name] = {"loss": tot["loss"] / n, "psnr": tot["psnr"] / n,
                         "ssim": tot["ssim"] / n, "n": tot["n"]}
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate a joint-model checkpoint")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", nargs="+", required=True,
                    help="subject ids to evaluate, e.g. --subjects 07 13")
    ap.add_argument("--no-per-subject", action="store_true",
                    help="report only the aggregate, not per-subject")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_checkpoint(args.ckpt, device)
    print(f"[eval] ckpt epoch={ck.get('epoch')} step={ck.get('step')} "
          f"git={cfg.git_commit[:8]} profile={cfg.profile} device={device}")
    results = evaluate_subjects(model, cfg, args.manifest, args.subjects, device,
                                per_subject=not args.no_per_subject)
    for name, m in results.items():
        print(f"[eval] {name:>8}: n={m['n']:5d}  loss {m['loss']:.4f}  "
              f"psnr {m['psnr']:.2f} dB  ssim {m['ssim']:.4f}")


if __name__ == "__main__":
    main()
