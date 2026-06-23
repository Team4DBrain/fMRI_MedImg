"""Launch a real training run (intended for the VM).

Example (subjects are placeholders — choose the held-out split on the VM)::

    python -m src.joint.run \\
        --manifest /srv/venvs/team4dbrain/derivatives/manifest.json \\
        --profile vm --val 06 14 --test 07 13 \\
        --ckpt-dir runs/joint01

Wires manifest + subject split -> JointDataset loaders -> train(). The test
subjects are held out entirely here (never loaded); evaluate them afterwards with
``python -m src.joint.eval``.
"""
from __future__ import annotations

import argparse

from .config import build_config
from .splits import build_loaders, make_splits
from .train import stamp_provenance, train


def main():
    ap = argparse.ArgumentParser(description="Train the joint denoise+SR model")
    ap.add_argument("--manifest", required=True, help="path to manifest.json")
    ap.add_argument("--profile", default="vm", help="config profile (default: vm)")
    ap.add_argument("--val", nargs="+", required=True,
                    help="val subject ids, e.g. --val 06 14")
    ap.add_argument("--test", nargs="*", default=[],
                    help="held-out test subject ids (not loaded during training)")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    # Per-run / per-machine overrides (None => keep the profile's value). Useful on a
    # shared box where free GPU memory and core count vary from run to run.
    ap.add_argument("--batch-size", type=int, default=None, help="override config batch_size")
    ap.add_argument("--num-workers", type=int, default=None, help="override config num_workers")
    ap.add_argument("--grad-accum", type=int, default=None, help="override config grad_accum")
    ap.add_argument("--lr", type=float, default=None, help="override config lr")
    args = ap.parse_args()

    cfg = build_config(args.profile)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
        cfg.train.persistent_workers = args.num_workers > 0   # DataLoader rejects True at 0 workers
    if args.grad_accum is not None:
        cfg.train.grad_accum = args.grad_accum
    if args.lr is not None:
        cfg.train.lr = args.lr
    stamp_provenance(cfg, args.manifest)

    splits = make_splits(args.manifest, val_subjects=args.val, test_subjects=args.test)
    print(f"[run] profile={cfg.profile} git={cfg.git_commit[:8]}")
    print(f"[run] batch={cfg.train.batch_size} workers={cfg.train.num_workers} "
          f"grad_accum={cfg.train.grad_accum} lr={cfg.train.lr} "
          f"amp={cfg.train.amp_dtype if cfg.train.use_amp else 'off'}")
    print(f"[run] split train={splits['train']}")
    print(f"[run]       val  ={splits['val']}  test(held out)={splits['test']}")

    train_loader, val_loader, train_ds, val_ds = build_loaders(cfg, args.manifest, splits)
    print(f"[run] samples: train={len(train_ds)} val={len(val_ds)} | "
          f"batches: train={len(train_loader)} val={len(val_loader)}")
    train(cfg, train_loader, val_loader, ckpt_dir=args.ckpt_dir)


if __name__ == "__main__":
    main()
