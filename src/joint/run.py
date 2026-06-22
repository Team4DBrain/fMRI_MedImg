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
    args = ap.parse_args()

    cfg = build_config(args.profile)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    stamp_provenance(cfg, args.manifest)

    splits = make_splits(args.manifest, val_subjects=args.val, test_subjects=args.test)
    print(f"[run] profile={cfg.profile} git={cfg.git_commit[:8]}")
    print(f"[run] split train={splits['train']}")
    print(f"[run]       val  ={splits['val']}  test(held out)={splits['test']}")

    train_loader, val_loader, train_ds, val_ds = build_loaders(cfg, args.manifest, splits)
    print(f"[run] samples: train={len(train_ds)} val={len(val_ds)} | "
          f"batches: train={len(train_loader)} val={len(val_loader)}")
    train(cfg, train_loader, val_loader, ckpt_dir=args.ckpt_dir)


if __name__ == "__main__":
    main()
