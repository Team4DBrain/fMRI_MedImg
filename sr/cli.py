"""Command-line entry point. ``python -m sr <command> ...``.

Purpose:
    Convert CLI flags into an ``SRConfig`` (or load one from disk) and
    dispatch to ``train``, ``evaluate`` or ``infer_one``. Every flag has
    a default sourced from ``SRConfig`` so a fresh user can start with
    ``python -m sr train`` alone.
Effects:
    For ``train``: creates a new run directory or resumes an existing one.
    For ``eval``: prints validation metrics, optionally writes a report.
    For ``infer``: prints sample-level metrics, optionally writes PNG/NPY.
Influences:
    On resume (``--resume-dir``) any non-default config flag triggers a
    hard error so the saved config remains the single source of truth.
How to change safely:
    Match every new ``SRConfig`` field with a CLI flag here (and update the
    ``_DEFAULT_VALUES`` map). Keep the help texts honest about defaults --
    users rely on them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from sr.config import SRConfig
from sr.debug import add_debug_arguments, run_debug
from sr.infer import (
    default_sr_output_path,
    evaluate,
    format_sample_table,
    infer_nifti,
    infer_one,
    make_sr_output_preview,
    print_volume_intensity_stats,
    list_samples,
    make_slice_figure,
    resolve_sr_output_path,
    select_sample,
)
from sr.checkpoint import resolve_checkpoint_for_model
from sr.losses import loss_names_for_validation
from sr.models import MODEL_REGISTRY
from sr.components import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from sr.train import train


# Build a "default config" sentinel once so resume-mode can detect when the
# user passed a non-default flag and reject it with a clear message.
_DEFAULT_CONFIG = SRConfig()


# ---------------------------------------------------------------------------
# argparse construction
# ---------------------------------------------------------------------------


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Existing run directory to resume. Ignores other config flags.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle deterministic algorithms / cuDNN policy.",
    )
    parser.add_argument(
        "--strict-finite-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail fast when training loss becomes NaN or Inf.",
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=None, help="Path to manifest.json"
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--model-name", choices=sorted(MODEL_REGISTRY), default=None
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="JSON dict of extra kwargs forwarded to the model constructor.",
    )
    parser.add_argument(
        "--output-shape",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=None,
        help="HR output shape (D H W).",
    )
    parser.add_argument(
        "--patch-hr-shape",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=None,
        help="HR training patch size for srcnn3d_patch (D H W).",
    )
    parser.add_argument(
        "--patches-per-volume",
        type=int,
        default=None,
        help="Random training patches drawn per volume (srcnn3d_patch).",
    )
    parser.add_argument("--source-voxel-mm", type=float, default=None)
    parser.add_argument("--target-voxel-mm", type=float, default=None)
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Fraction of dataset samples used for training (1.0 disables validation).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, dest="num_epochs")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--loss-name",
        choices=sorted(loss_names_for_validation()),
        default=None,
    )
    parser.add_argument(
        "--loss-kwargs",
        type=str,
        default=None,
        help="JSON dict of extra kwargs for parameterised losses (e.g. dual_domain_masked_mse).",
    )
    parser.add_argument(
        "--optimizer-name",
        choices=sorted(OPTIMIZER_REGISTRY),
        default=None,
    )
    parser.add_argument(
        "--lr", type=float, default=None, dest="learning_rate"
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help=(
            "Clip the global gradient L2 norm to this value each step "
            "(training stability). Omit to disable (default); 1.0 is a "
            "typical stabilising value. Prevents the mid-training loss "
            "spikes seen in earlier srcnn3d runs."
        ),
    )
    parser.add_argument(
        "--optimizer-kwargs",
        type=str,
        default=None,
        help="JSON dict of extra kwargs for the optimizer.",
    )
    parser.add_argument(
        "--scheduler-name",
        choices=sorted(SCHEDULER_REGISTRY),
        default=None,
    )
    parser.add_argument(
        "--scheduler-kwargs",
        type=str,
        default=None,
        help="JSON dict of extra kwargs for the scheduler.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write per-epoch and per-batch TensorBoard scalars.",
    )


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional override; default uses the manifest stored in the run config.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON output path for metrics.",
    )


def _add_infer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="3D/4D NIfTI to super-resolve. 4D without --t processes the full run.",
    )
    parser.add_argument(
        "--model-name",
        choices=sorted(MODEL_REGISTRY),
        default=None,
        help="Model to use with --input (resolves checkpoint from latest run).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NIfTI path (default: <input_stem>_sr.nii.gz beside the input).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory when resolving --model-name (default: latest run).",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run root scanned for --model-name when --run-dir is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt (required for manifest infer; optional with --input).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional override; default uses the manifest stored in the run config.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="Print the manifest's sample table and exit (no inference).",
    )
    parser.add_argument("--subject", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--direction", choices=["ap", "pa"], default=None)
    parser.add_argument(
        "--t",
        type=int,
        default=None,
        help="Timepoint index. For 4D NIfTI: omit to process all volumes; set to infer one slice only.",
    )
    parser.add_argument(
        "--axis",
        choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Slice direction for the figure (default: axial).",
    )
    parser.add_argument(
        "--slice-level",
        type=float,
        default=0.5,
        help="Relative slice level in [0, 1] for preview PNGs (default: 0.5 = center).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write an LR vs SR preview PNG (with ground truth when available). "
        "For full 4D runs, preview uses the middle timepoint only.",
    )
    parser.add_argument(
        "--save-png",
        type=Path,
        default=None,
        help="Optional extra single-slice figure path (in addition to --preview).",
    )
    parser.add_argument("--save-npy", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sr",
        description="Spatial super-resolution CLI (train / eval / infer / debug).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _add_train_arguments(sub.add_parser("train", help="Train (or resume) a model."))
    _add_eval_arguments(sub.add_parser("eval", help="Evaluate a checkpoint."))
    _add_infer_arguments(sub.add_parser("infer", help="Run one-sample inference."))
    add_debug_arguments(
        sub.add_parser(
            "debug",
            help="Inspect HR/LR images and masks (no checkpoint).",
        )
    )

    return parser


# ---------------------------------------------------------------------------
# Translating CLI args into an SRConfig
# ---------------------------------------------------------------------------


# Mapping CLI attribute -> SRConfig field. Most match by name, a few are
# renamed for ergonomic CLI flags (e.g. --epochs -> num_epochs).
_CLI_TO_CONFIG: dict[str, str] = {
    "seed": "seed",
    "deterministic": "deterministic",
    "strict_finite_loss": "strict_finite_loss",
    "manifest_path": "manifest_path",
    "run_root": "run_root",
    "model_name": "model_name",
    "model_kwargs": "model_kwargs",
    "output_shape": "output_patch_shape",
    "patch_hr_shape": "patch_hr_shape",
    "patches_per_volume": "patches_per_volume",
    "source_voxel_mm": "source_voxel_mm",
    "target_voxel_mm": "target_voxel_mm",
    "train_split": "train_split",
    "batch_size": "batch_size",
    "num_epochs": "num_epochs",
    "num_workers": "num_workers",
    "log_interval": "log_interval",
    "loss_name": "loss_name",
    "loss_kwargs": "loss_kwargs",
    "optimizer_name": "optimizer_name",
    "learning_rate": "learning_rate",
    "grad_clip_norm": "grad_clip_norm",
    "optimizer_kwargs": "optimizer_kwargs",
    "scheduler_name": "scheduler_name",
    "scheduler_kwargs": "scheduler_kwargs",
    "tensorboard": "tensorboard",
}

# CLI attributes whose value is parsed from a JSON string.
_JSON_FIELDS: frozenset[str] = frozenset(
    {"model_kwargs", "optimizer_kwargs", "scheduler_kwargs", "loss_kwargs"}
)


def _parse_value(cli_attr: str, value: Any) -> Any:
    """Lightweight CLI->config conversions (JSON decode, tuple coercion)."""
    if cli_attr in _JSON_FIELDS and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--{cli_attr.replace('_', '-')} is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            raise SystemExit(f"--{cli_attr.replace('_', '-')} must be a JSON object.")
        return parsed
    if cli_attr in ("output_shape", "patch_hr_shape"):
        return tuple(int(v) for v in value)
    return value


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for cli_attr, config_field in _CLI_TO_CONFIG.items():
        if not hasattr(args, cli_attr):
            continue
        value = getattr(args, cli_attr)
        if value is None:
            continue
        overrides[config_field] = _parse_value(cli_attr, value)
    return overrides


def _config_from_args(args: argparse.Namespace) -> SRConfig:
    """Build an SRConfig from defaults + non-None CLI overrides."""
    overrides = _collect_overrides(args)
    return dataclasses.replace(_DEFAULT_CONFIG, **overrides)


def _forbid_config_overrides_on_resume(args: argparse.Namespace) -> None:
    """Resume must use the saved config; refuse any non-None config flag."""
    overrides = _collect_overrides(args)
    if overrides:
        keys = ", ".join(sorted(overrides))
        raise SystemExit(
            "Resume rejects config overrides; the saved config.json is the "
            f"single source of truth. Offending flags: {keys}. Edit "
            "<run_dir>/config.json directly if you need different values."
        )


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


def _run_train(args: argparse.Namespace) -> None:
    if args.resume_dir is not None:
        _forbid_config_overrides_on_resume(args)
        # The saved config wins; pass a dummy default config -- train() will
        # replace it with the on-disk one before doing anything.
        train(_DEFAULT_CONFIG, resume_dir=args.resume_dir)
        return
    config: SRConfig = _config_from_args(args)
    train(config, resume_dir=None)


def _run_eval(args: argparse.Namespace) -> None:
    evaluate(
        checkpoint_path=args.checkpoint,
        override_manifest=args.manifest_path,
        report_path=args.report,
    )


def _run_infer(args: argparse.Namespace) -> None:
    if args.input is not None:
        if args.model_name is None:
            raise SystemExit("--model-name is required when using --input.")
        checkpoint = resolve_checkpoint_for_model(
            args.model_name,
            run_root=args.run_root,
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
        )
        output_path = resolve_sr_output_path(args.input, args.output)
        result = infer_nifti(
            checkpoint,
            args.input,
            output_path,
            t=args.t,
            write_preview=args.preview,
            slice_level=args.slice_level,
        )
        if args.save_png is not None:
            make_slice_figure(
                input_vol=result["input_lr"],
                prediction_vol=result["prediction"],
                target_vol=result.get("ground_truth"),
                axis=args.axis,
                slice_level=args.slice_level,
                output_path=args.save_png,
                show=False,
            )
        if args.save_npy is not None:
            import numpy as np

            path = Path(args.save_npy)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, result["prediction"])
            print(f"[infer] wrote prediction npy -> {path}")
        return

    if args.list_samples:
        if args.checkpoint is None and args.manifest_path is None:
            raise SystemExit(
                "--list-samples requires --checkpoint or --manifest-path."
            )
        manifest = args.manifest_path
        if manifest is None:
            from sr.checkpoint import run_dir_for_checkpoint
            from sr.config import from_json as _from_json

            run_dir = run_dir_for_checkpoint(args.checkpoint)
            manifest = _from_json(run_dir / "config.json").manifest_path
        print(format_sample_table(list_samples(manifest)))
        return

    if args.checkpoint is None:
        raise SystemExit(
            "Manifest-based infer requires --checkpoint (or use --input for NIfTI files)."
        )

    selection_filters = {
        "subject": args.subject,
        "session": args.session,
        "task": args.task,
        "direction": args.direction,
        "t": args.t,
    }
    if all(value is None for value in selection_filters.values()):
        raise SystemExit(
            "infer requires --input, --list-samples, or at least one selector "
            "(--subject/--session/--task/--direction/--t). See --help."
        )

    manifest = args.manifest_path
    if manifest is None:
        from sr.checkpoint import run_dir_for_checkpoint
        from sr.config import from_json as _from_json

        run_dir = run_dir_for_checkpoint(args.checkpoint)
        manifest = _from_json(run_dir / "config.json").manifest_path

    chosen = select_sample(manifest, **selection_filters)
    print(
        "[infer] selected sample: "
        f"subject={chosen['subject']} session={chosen['session']} "
        f"task={chosen['task']} direction={chosen['direction']} "
        f"run_id={chosen['run_id']} t={chosen['t']}"
    )
    result = infer_one(
        args.checkpoint, chosen, override_manifest=args.manifest_path
    )
    print(f"[infer] axis={args.axis} slice_level={args.slice_level:.2f}")
    print(f"[infer] input.shape      = {result['input'].shape}")
    print(f"[infer] prediction.shape = {result['prediction'].shape}")
    print(f"[infer] target.shape     = {result['target'].shape}")
    print_volume_intensity_stats(result["volume_stats"])
    for key in sorted(result["metrics"]):
        print(f"[infer] {key:>14} = {result['metrics'][key]:.6f}")

    if args.preview:
        preview_name = (
            f"infer_{chosen['subject']}_{chosen['session']}_{chosen['task']}_"
            f"{chosen['direction']}_t{chosen['t']}.png"
        )
        make_sr_output_preview(
            input_lr=result["input"],
            prediction_vol=result["prediction"],
            ground_truth_vol=result["target"],
            output_path=Path(preview_name),
            slice_level=args.slice_level,
        )

    if args.save_npy is not None:
        import numpy as np

        path = Path(args.save_npy)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, result["prediction"])
        print(f"[infer] wrote prediction npy -> {path}")

    if args.save_png is not None:
        make_slice_figure(
            input_vol=result["input"],
            prediction_vol=result["prediction"],
            target_vol=result["target"],
            axis=args.axis,
            slice_level=args.slice_level,
            output_path=args.save_png,
            show=False,
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "train":
        _run_train(args)
    elif args.command == "eval":
        _run_eval(args)
    elif args.command == "infer":
        _run_infer(args)
    elif args.command == "debug":
        run_debug(args)
    else:
        parser.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
