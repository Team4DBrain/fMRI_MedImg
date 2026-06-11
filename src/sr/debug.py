"""Debug helpers for masks, degradation, prediction error, and loss diagnostics.

Purpose:
    Inspect dataset outputs and (optionally) one checkpointed prediction on a
    single manifest sample — without running train/eval. Surfaces trilinear
    baseline error vs model error to explain fast training loss drops.
Effects:
    Prints shapes, mask coverage, per-sample metrics, and loss decomposition.
    Writes PNG figures (masks grid, infer/error grid, optional loss curve).
Influences:
    ``--manifest-path``, sample selectors, ``--checkpoint``, degradation
    voxel sizes (from checkpoint config when a checkpoint is given).
How to change safely:
    Add new panels or stats here; keep training loop imports out of this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation
from src.sr.checkpoint import run_dir_for_checkpoint
from src.sr.config import SRConfig, from_json
from src.sr.infer import format_sample_table, infer_one, list_samples, select_sample
from src.sr.losses import (
    kspace_mse_loss,
    masked_mse_loss,
    merge_dual_domain_kwargs,
)

AXIS_TO_INDEX: dict[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}
FigureMode = Literal["masks", "infer", "both"]
ErrorMapMode = Literal["abs", "signed", "squared"]


def _extract_slice(
    volume: np.ndarray,
    *,
    axis: str,
    slice_level: float,
) -> tuple[np.ndarray, int]:
    """Return one 2D slice and its index along ``axis``."""
    if axis not in AXIS_TO_INDEX:
        raise ValueError(f"Unknown axis '{axis}'. Choose one of: {sorted(AXIS_TO_INDEX)}")
    if not 0.0 <= slice_level <= 1.0:
        raise ValueError(f"slice_level must be in [0, 1], got {slice_level}")

    idx_axis = AXIS_TO_INDEX[axis]
    dim = int(volume.shape[idx_axis])
    idx = min(dim - 1, max(0, int(round(slice_level * (dim - 1)))))
    if axis == "axial":
        return volume[:, :, idx], idx
    if axis == "coronal":
        return volume[:, idx, :], idx
    return volume[idx, :, :], idx


def _tensor_to_volume(tensor: torch.Tensor) -> np.ndarray:
    """(1, D, H, W) or (D, H, W) -> float32 numpy."""
    arr = tensor.detach().cpu().numpy()
    while arr.ndim > 3:
        arr = arr.squeeze(0)
    return np.asarray(arr, dtype=np.float32)


def _trilinear_baseline_np(lr_volume: np.ndarray, hr_shape: tuple[int, ...]) -> np.ndarray:
    """Upsample LR to HR shape the same way SRCNN3D's first layer does."""
    lr = torch.from_numpy(np.ascontiguousarray(lr_volume, dtype=np.float32))
    lr = lr.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(lr, size=hr_shape, mode="trilinear", align_corners=False)
    return up.squeeze(0).squeeze(0).numpy()


def _error_volume(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    mode: ErrorMapMode,
) -> np.ndarray:
    diff = pred - target
    if mode == "abs":
        return np.abs(diff)
    if mode == "signed":
        return diff
    return diff * diff


def _apply_mask_slice(error_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    out = error_slice.copy()
    out[mask_slice <= 0.5] = 0.0
    return out


def _volume_to_batch(volume: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32)).view(
        1, 1, *volume.shape
    )


def load_debug_sample(
    manifest_path: Path,
    selection: dict[str, Any],
    *,
    source_voxel_mm: float,
    target_voxel_mm: float,
) -> dict[str, Any]:
    """Load HR/LR image and mask tensors for one (run, timepoint)."""
    degrade_fn = make_spatial_degradation(
        source_voxel_mm=float(source_voxel_mm),
        target_voxel_mm=float(target_voxel_mm),
    )
    dataset = SpatialSRDataset(
        manifest_path=Path(manifest_path),
        degrade_fn=degrade_fn,
        source_voxel_mm=float(source_voxel_mm),
        target_voxel_mm=float(target_voxel_mm),
    )

    sample_index: int | None = None
    for idx, (run_idx, t) in enumerate(dataset.samples):
        if dataset.runs[run_idx]["run_id"] == selection["run_id"] and t == selection["t"]:
            sample_index = idx
            break
    if sample_index is None:
        raise RuntimeError(
            f"Could not locate run_id={selection['run_id']} t={selection['t']} "
            "in the dataset. Check manifest filters and paths."
        )

    sample = dataset[sample_index]
    return {
        "run_id": selection["run_id"],
        "subject": selection.get("subject"),
        "session": selection.get("session"),
        "task": selection.get("task"),
        "direction": selection.get("direction"),
        "t": selection["t"],
        "hr_image": _tensor_to_volume(sample["target"]),
        "hr_mask": _tensor_to_volume(sample["mask_hr"]),
        "lr_image": _tensor_to_volume(sample["input"]),
        "lr_mask": _tensor_to_volume(sample["mask_lr"]),
        "lr_shape": dataset.lr_shape,
        "hr_shape": dataset.target_shape,
    }


def load_debug_infer(
    checkpoint_path: Path,
    manifest_path: Path,
    selection: dict[str, Any],
    *,
    override_manifest: Path | None = None,
) -> dict[str, Any]:
    """Run inference and attach mask + trilinear baseline for debug plots."""
    result = infer_one(
        checkpoint_path,
        selection,
        override_manifest=override_manifest,
    )
    run_dir = run_dir_for_checkpoint(checkpoint_path)
    config = from_json(run_dir / "config.json")
    effective_manifest = Path(override_manifest or config.manifest_path)

    mask_payload = load_debug_sample(
        effective_manifest,
        selection,
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
    )
    target = result["target"]
    prediction = result["prediction"]
    baseline = _trilinear_baseline_np(result["input"], target.shape)

    return {
        **result,
        "hr_mask": mask_payload["hr_mask"],
        "baseline": baseline,
        "loss_name": config.loss_name,
        "loss_kwargs": dict(config.loss_kwargs),
        "run_dir": run_dir,
    }


def print_debug_summary(payload: dict[str, Any]) -> None:
    """Print shapes and mask coverage for the masks figure."""
    hr = payload["hr_image"]
    lr = payload["lr_image"]
    hr_m = payload["hr_mask"]
    lr_m = payload["lr_mask"]

    def coverage(mask: np.ndarray) -> float:
        return float(np.mean(mask > 0.5))

    print(
        "[debug] sample: "
        f"subject={payload.get('subject')} session={payload.get('session')} "
        f"task={payload.get('task')} direction={payload.get('direction')} "
        f"run_id={payload['run_id']} t={payload['t']}"
    )
    print(f"[debug] hr_image.shape = {hr.shape}  (original / target)")
    print(f"[debug] hr_mask.shape  = {hr_m.shape}  coverage={coverage(hr_m):.4f}")
    print(f"[debug] lr_image.shape = {lr.shape}  (downgraded / input)")
    print(f"[debug] lr_mask.shape  = {lr_m.shape}  coverage={coverage(lr_m):.4f}")
    print(
        f"[debug] hr intensity: min={hr.min():.4f} max={hr.max():.4f} "
        f"mean={hr.mean():.4f}"
    )
    print(
        f"[debug] lr intensity: min={lr.min():.4f} max={lr.max():.4f} "
        f"mean={lr.mean():.4f}"
    )


def print_infer_summary(payload: dict[str, Any]) -> None:
    """Print per-sample metrics and baseline vs model error decomposition."""
    pred = payload["prediction"]
    target = payload["target"]
    baseline = payload["baseline"]
    mask = payload["hr_mask"]

    pred_t = _volume_to_batch(pred)
    target_t = _volume_to_batch(target)
    baseline_t = _volume_to_batch(baseline)
    mask_t = _volume_to_batch(mask)

    pred_mse = float(masked_mse_loss(pred_t, target_t, mask_t).item())
    baseline_mse = float(masked_mse_loss(baseline_t, target_t, mask_t).item())
    pred_full_mse = float(((pred - target) ** 2).mean())
    baseline_full_mse = float(((baseline - target) ** 2).mean())

    print("[debug] --- inference metrics (sample) ---")
    for key in sorted(payload["metrics"]):
        print(f"[debug] {key:>22} = {payload['metrics'][key]:.6f}")
    print(f"[debug] {'masked_mse_pred':>22} = {pred_mse:.6f}")
    print(f"[debug] {'masked_mse_baseline':>22} = {baseline_mse:.6f}")
    print(f"[debug] {'full_mse_pred':>22} = {pred_full_mse:.6f}")
    print(f"[debug] {'full_mse_baseline':>22} = {baseline_full_mse:.6f}")
    print(
        f"[debug] model improvement (baseline−pred masked MSE) = "
        f"{baseline_mse - pred_mse:.6f}"
    )

    loss_name = payload.get("loss_name", "")
    loss_kwargs = payload.get("loss_kwargs") or {}
    if loss_name == "kspace_mse":
        boost = float(loss_kwargs.get("kspace_high_freq_weight", 0.0))
        k_pred = float(
            kspace_mse_loss(pred_t, target_t, mask_t, high_freq_boost=boost).item()
        )
        k_base = float(
            kspace_mse_loss(baseline_t, target_t, mask_t, high_freq_boost=boost).item()
        )
        print(f"[debug] {'kspace_mse_pred':>22} = {k_pred:.6f}")
        print(f"[debug] {'kspace_mse_baseline':>22} = {k_base:.6f}")
    elif loss_name == "dual_domain_masked_mse":
        merged = merge_dual_domain_kwargs(loss_kwargs)
        boost = merged["kspace_high_freq_weight"]
        k_pred = float(
            kspace_mse_loss(pred_t, target_t, mask_t, high_freq_boost=boost).item()
        )
        k_base = float(
            kspace_mse_loss(baseline_t, target_t, mask_t, high_freq_boost=boost).item()
        )
        print(f"[debug] {'image_mse_pred':>22} = {pred_mse:.6f}")
        print(f"[debug] {'image_mse_baseline':>22} = {baseline_mse:.6f}")
        print(f"[debug] {'kspace_mse_pred':>22} = {k_pred:.6f}")
        print(f"[debug] {'kspace_mse_baseline':>22} = {k_base:.6f}")


def _import_matplotlib(*, agg: bool) -> Any:
    try:
        import matplotlib

        if agg:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for debug figures. Install it or use stats-only output."
        ) from exc


def save_debug_figure(
    payload: dict[str, Any],
    *,
    axis: str,
    slice_level: float,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Save a 2×2 figure: HR image, HR mask, LR image, LR mask."""
    plt = _import_matplotlib(agg=output_path is not None)

    hr_slice, hr_idx = _extract_slice(
        payload["hr_image"], axis=axis, slice_level=slice_level
    )
    hr_mask_slice, _ = _extract_slice(
        payload["hr_mask"], axis=axis, slice_level=slice_level
    )
    lr_slice, lr_idx = _extract_slice(
        payload["lr_image"], axis=axis, slice_level=slice_level
    )
    lr_mask_slice, _ = _extract_slice(
        payload["lr_mask"], axis=axis, slice_level=slice_level
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = [
        (axes[0, 0], hr_slice, "gray", f"Original (HR)\n{axis} slice={hr_idx}"),
        (axes[0, 1], hr_mask_slice, "viridis", f"HR mask\n{axis} slice={hr_idx}"),
        (axes[1, 0], lr_slice, "gray", f"Downgraded (LR)\n{axis} slice={lr_idx}"),
        (axes[1, 1], lr_mask_slice, "viridis", f"LR mask\n{axis} slice={lr_idx}"),
    ]
    for ax, image, cmap, title in panels:
        vmin = 0.0 if "mask" in title.lower() else None
        vmax = 1.0 if "mask" in title.lower() else None
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"debug masks: {payload['run_id']} t={payload['t']}",
        fontsize=11,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[debug] wrote figure -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_infer_figure(
    payload: dict[str, Any],
    *,
    axis: str,
    slice_level: float,
    error_map: ErrorMapMode,
    mask_errors: bool,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Save GT, prediction, error, trilinear baseline, and baseline error."""
    plt = _import_matplotlib(agg=output_path is not None)

    target = payload["target"]
    pred = payload["prediction"]
    baseline = payload["baseline"]

    target_slice, idx = _extract_slice(target, axis=axis, slice_level=slice_level)
    pred_slice, _ = _extract_slice(pred, axis=axis, slice_level=slice_level)
    baseline_slice, _ = _extract_slice(baseline, axis=axis, slice_level=slice_level)
    mask_slice, _ = _extract_slice(payload["hr_mask"], axis=axis, slice_level=slice_level)

    err_vol = _error_volume(pred, target, mode=error_map)
    base_err_vol = _error_volume(baseline, target, mode="abs")
    err_slice, _ = _extract_slice(err_vol, axis=axis, slice_level=slice_level)
    base_err_slice, _ = _extract_slice(base_err_vol, axis=axis, slice_level=slice_level)

    if mask_errors:
        err_slice = _apply_mask_slice(err_slice, mask_slice)
        base_err_slice = _apply_mask_slice(base_err_slice, mask_slice)

    gray_stack = np.stack([target_slice, pred_slice, baseline_slice])
    vmin = float(gray_stack.min())
    vmax = float(gray_stack.max())

    err_cmap = "RdBu_r" if error_map == "signed" else "hot"
    err_title = {
        "abs": "|pred − target|",
        "signed": "pred − target",
        "squared": "(pred − target)²",
    }[error_map]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    gray_panels = [
        (axes[0], target_slice, "Target (GT)"),
        (axes[1], pred_slice, "Prediction"),
        (axes[3], baseline_slice, "Trilinear baseline"),
    ]
    for ax, image, title in gray_panels:
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{axis} slice={idx}")
        ax.axis("off")

    im_err = axes[2].imshow(err_slice, cmap=err_cmap)
    axes[2].set_title(f"{err_title}\n{axis} slice={idx}")
    axes[2].axis("off")
    fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)

    im_base = axes[4].imshow(base_err_slice, cmap="hot")
    axes[4].set_title(f"|baseline − target|\n{axis} slice={idx}")
    axes[4].axis("off")
    fig.colorbar(im_base, ax=axes[4], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"debug infer: {payload['run_id']} t={payload['t']}  "
        f"loss={payload.get('loss_name', '')}",
        fontsize=11,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[debug] wrote figure -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_loss_curve(
    run_dir: Path,
    output_path: Path,
    *,
    show: bool = False,
) -> None:
    """Plot train_loss and validation metrics from ``run_dir/metrics.json``."""
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.json at {metrics_path}")

    history: list[dict[str, Any]] = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )
    if not history:
        raise RuntimeError(f"{metrics_path} is empty.")

    plt = _import_matplotlib(agg=True)
    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, marker="o", label="train_loss")

    val_keys = sorted(
        k
        for k in history[0]
        if k.startswith("val_") and k not in {"val_duration_s"}
    )
    for key in val_keys[:4]:
        series = [h.get(key) for h in history]
        if any(v is not None for v in series):
            ax.plot(
                epochs,
                [float(v) if v is not None else np.nan for v in series],
                marker="o",
                label=key,
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / metric")
    ax.set_title(f"Run metrics: {run_dir.name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[debug] wrote loss curve -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def add_debug_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags for the ``debug`` subcommand."""
    defaults = SRConfig()
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=defaults.manifest_path,
        help="Path to manifest.json (default: SRConfig manifest_path).",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="Print manifest sample table and exit.",
    )
    parser.add_argument("--subject", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--direction", choices=["ap", "pa"], default=None)
    parser.add_argument("--t", type=int, default=None, help="Timepoint index.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for prediction/error panels (uses run config).",
    )
    parser.add_argument(
        "--figure",
        choices=["masks", "infer", "both"],
        default=None,
        help="Figure set: masks (default), infer (needs checkpoint), or both.",
    )
    parser.add_argument(
        "--error-map",
        choices=["abs", "signed", "squared"],
        default="abs",
        help="How to render pred vs target error (default: abs).",
    )
    parser.add_argument(
        "--mask-errors",
        action="store_true",
        help="Zero error voxels outside the HR brain mask in error panels.",
    )
    parser.add_argument(
        "--source-voxel-mm",
        type=float,
        default=None,
        help="HR voxel size for degradation (default: checkpoint config or SRConfig).",
    )
    parser.add_argument(
        "--target-voxel-mm",
        type=float,
        default=None,
        help="Simulated LR voxel size (default: checkpoint config or SRConfig).",
    )
    parser.add_argument(
        "--axis",
        choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Slice direction for figures.",
    )
    parser.add_argument(
        "--slice-level",
        type=float,
        default=0.5,
        help="Relative slice position in [0, 1] (default: 0.5 = center).",
    )
    parser.add_argument(
        "--save-png",
        type=Path,
        default=None,
        help="Write primary figure (infer if checkpoint, else masks).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Write masks.png, infer.png, and/or loss_curve.png into this directory.",
    )
    parser.add_argument(
        "--plot-loss-curve",
        action="store_true",
        help="Plot metrics.json from the checkpoint run directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib windows (needs a display).",
    )


def _resolve_figure_mode(args: argparse.Namespace) -> FigureMode:
    if args.checkpoint is not None:
        return args.figure or "both"
    if args.figure in ("infer", "both"):
        raise SystemExit("--figure infer/both requires --checkpoint.")
    return args.figure or "masks"


def _resolve_voxel_sizes(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path | None,
) -> tuple[float, float]:
    defaults = SRConfig()
    if checkpoint_path is not None:
        config = from_json(run_dir_for_checkpoint(checkpoint_path) / "config.json")
        source = float(args.source_voxel_mm or config.source_voxel_mm)
        target = float(args.target_voxel_mm or config.target_voxel_mm)
        return source, target
    source = float(args.source_voxel_mm or defaults.source_voxel_mm)
    target = float(args.target_voxel_mm or defaults.target_voxel_mm)
    return source, target


def run_debug(args: argparse.Namespace) -> None:
    """Entry point for ``python -m src.sr debug ...``."""
    manifest = Path(args.manifest_path)
    if not manifest.is_file():
        raise SystemExit(f"manifest_path does not exist: {manifest}")

    if args.list_samples:
        print(format_sample_table(list_samples(manifest)))
        return

    selection_filters = {
        "subject": args.subject,
        "session": args.session,
        "task": args.task,
        "direction": args.direction,
        "t": args.t,
    }
    if all(value is None for value in selection_filters.values()):
        raise SystemExit(
            "debug requires either --list-samples or at least one selector "
            "(--subject/--session/--task/--direction/--t). See --help."
        )

    figure_mode = _resolve_figure_mode(args)
    chosen = select_sample(manifest, **selection_filters)
    source_mm, target_mm = _resolve_voxel_sizes(args, checkpoint_path=args.checkpoint)

    mask_payload: dict[str, Any] | None = None
    infer_payload: dict[str, Any] | None = None

    if figure_mode in ("masks", "both"):
        mask_payload = load_debug_sample(
            manifest,
            chosen,
            source_voxel_mm=source_mm,
            target_voxel_mm=target_mm,
        )
        print_debug_summary(mask_payload)

    if figure_mode in ("infer", "both"):
        if args.checkpoint is None:
            raise SystemExit("Internal error: infer figure without checkpoint.")
        infer_payload = load_debug_infer(
            Path(args.checkpoint),
            manifest,
            chosen,
            override_manifest=manifest,
        )
        print_infer_summary(infer_payload)

    wants_figure = (
        args.save_png is not None
        or args.save_dir is not None
        or args.show
    )
    if args.plot_loss_curve and args.checkpoint is None:
        raise SystemExit("--plot-loss-curve requires --checkpoint.")

    if not wants_figure and not args.plot_loss_curve:
        print(
            "[debug] No figures written. Pass --save-png, --save-dir, --show, "
            "and/or --plot-loss-curve."
        )
        return

    slice_kw = {
        "axis": args.axis,
        "slice_level": float(args.slice_level),
        "show": bool(args.show),
    }
    infer_kw = {
        **slice_kw,
        "error_map": args.error_map,
        "mask_errors": bool(args.mask_errors),
    }

    save_dir = Path(args.save_dir) if args.save_dir is not None else None

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if mask_payload is not None:
            save_debug_figure(mask_payload, output_path=save_dir / "masks.png", **slice_kw)
        if infer_payload is not None:
            save_infer_figure(
                infer_payload, output_path=save_dir / "infer.png", **infer_kw
            )
        if args.plot_loss_curve and infer_payload is not None:
            plot_loss_curve(
                infer_payload["run_dir"],
                save_dir / "loss_curve.png",
                show=bool(args.show),
            )
    elif args.save_png is not None:
        if infer_payload is not None:
            save_infer_figure(infer_payload, output_path=args.save_png, **infer_kw)
        elif mask_payload is not None:
            save_debug_figure(mask_payload, output_path=args.save_png, **slice_kw)

    if args.plot_loss_curve and save_dir is None and infer_payload is not None:
        curve_path = (
            args.save_png.parent / "loss_curve.png"
            if args.save_png is not None
            else Path("loss_curve.png")
        )
        plot_loss_curve(
            infer_payload["run_dir"],
            curve_path,
            show=bool(args.show),
        )

    if args.show and save_dir is None and args.save_png is None:
        if mask_payload is not None and figure_mode in ("masks", "both"):
            save_debug_figure(mask_payload, output_path=None, **slice_kw)
        if infer_payload is not None and figure_mode in ("infer", "both"):
            save_infer_figure(infer_payload, output_path=None, **infer_kw)
