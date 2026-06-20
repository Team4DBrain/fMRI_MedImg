"""Debug helpers for masks, degradation, prediction error, and loss diagnostics.

Purpose:
    Inspect dataset outputs and (optionally) one checkpointed prediction on a
    single manifest sample — without running train/eval. Surfaces trilinear
    baseline error vs model error to explain fast training loss drops.
Effects:
    Prints shapes, mask coverage, per-sample metrics, and loss decomposition.
    Writes PNG figures (masks grid, infer/error grid, optional loss curve).
    Training runs also maintain ``<run_dir>/debug/`` with fixed-sample evolution
    grids (every epoch shown) and loss curves.
Influences:
    ``--manifest-path``, sample selectors, ``--checkpoint``, degradation
    voxel sizes (from checkpoint config when a checkpoint is given).
How to change safely:
    Add new panels or stats here; keep training loop imports out of this file.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation
from src.sr.checkpoint import (
    best_epoch_path,
    find_latest_epoch,
    list_epoch_files,
    load_epoch,
    run_dir_for_checkpoint,
)
from src.sr.config import SRConfig, auto_device, from_json
from src.sr.infer import format_sample_table, infer_one, list_samples, select_sample
from src.sr.models import build_model
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
    effective_manifest = Path(override_manifest or manifest_path)
    return _build_infer_payload(checkpoint_path, effective_manifest, selection)


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


# ---------------------------------------------------------------------------
# Run debug bundle (fixed samples, evolution plots, training hooks)
# ---------------------------------------------------------------------------

RUN_DEBUG_DIRNAME = "debug"
EVOLUTION_COLS_PER_ROW = 8


@dataclass(frozen=True)
class FixedDebugSample:
    """One manifest sample used for every run's debug bundle.

    Purpose:
        Pin three manifest runs per slice axis (coronal, axial, sagittal) so
        training progress is comparable across experiments without re-picking.
    Effects:
        Drives mask figures and evolution PNGs under ``<run_dir>/debug/``.
    Influences:
        Edit ``FIXED_DEBUG_SAMPLES`` when the manifest changes; run
        ``python -m src.sr debug --populate-runs`` to refresh all runs.
    """

    key: str
    subject: str
    session: str
    task: str
    direction: Literal["ap", "pa"]
    t: int
    axis: str = "coronal"
    slice_level: float = 0.6
    note: str = ""


# Three runs per slice axis (coronal / axial / sagittal), chosen from the
# production manifest for high t-SNR and paradigm diversity.
FIXED_DEBUG_SAMPLES: tuple[FixedDebugSample, ...] = (
    # --- coronal ---
    FixedDebugSample(
        key="coronal_ap_sub07_ses34_GoodBadUgly_t070",
        subject="07",
        session="34",
        task="GoodBadUgly",
        direction="ap",
        t=70,
        axis="coronal",
        slice_level=0.6,
        note="Movie/clips, high AP t-SNR",
    ),
    FixedDebugSample(
        key="coronal_pa_sub02_ses04_HcpEmotion_t069",
        subject="02",
        session="04",
        task="HcpEmotion",
        direction="pa",
        t=69,
        axis="coronal",
        slice_level=0.6,
        note="HCP emotion, high PA t-SNR",
    ),
    FixedDebugSample(
        key="coronal_ap_sub06_ses08_ContRing_t012",
        subject="06",
        session="08",
        task="ContRing",
        direction="ap",
        t=12,
        axis="coronal",
        slice_level=0.6,
        note="Contrast ring visual localiser",
    ),
    # --- axial ---
    FixedDebugSample(
        key="axial_ap_sub01_ses03_HcpLanguage_t109",
        subject="01",
        session="03",
        task="HcpLanguage",
        direction="ap",
        t=109,
        axis="axial",
        slice_level=0.5,
        note="HCP language, mid-run",
    ),
    FixedDebugSample(
        key="axial_pa_sub07_ses25_Stroop_t052",
        subject="07",
        session="25",
        task="Stroop",
        direction="pa",
        t=52,
        axis="axial",
        slice_level=0.5,
        note="Stroop interference",
    ),
    FixedDebugSample(
        key="axial_pa_sub09_ses31_SpatialNavigation_t075",
        subject="09",
        session="31",
        task="SpatialNavigation",
        direction="pa",
        t=75,
        axis="axial",
        slice_level=0.5,
        note="Spatial navigation",
    ),
    # --- sagittal ---
    FixedDebugSample(
        key="sagittal_pa_sub07_ses35_EmoReco_t099",
        subject="07",
        session="35",
        task="EmoReco",
        direction="pa",
        t=99,
        axis="sagittal",
        slice_level=0.5,
        note="Emotion recognition, mid-run",
    ),
    FixedDebugSample(
        key="sagittal_pa_sub11_ses30_BiologicalMotion1_t102",
        subject="11",
        session="30",
        task="BiologicalMotion1",
        direction="pa",
        t=102,
        axis="sagittal",
        slice_level=0.5,
        note="Biological motion",
    ),
    FixedDebugSample(
        key="sagittal_pa_sub13_ses03_RSVPLanguage_t155",
        subject="13",
        session="03",
        task="RSVPLanguage",
        direction="pa",
        t=155,
        axis="sagittal",
        slice_level=0.5,
        note="RSVP language, mid-run",
    ),
)


def run_debug_dir(run_dir: Path) -> Path:
    """Return ``<run_dir>/debug/`` for artifact output."""
    path = Path(run_dir).resolve() / RUN_DEBUG_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_debug_output_dir(debug_dir: Path) -> None:
    """Remove a run's debug output tree so the next populate starts fresh."""
    debug_dir = Path(debug_dir)
    if debug_dir.is_dir():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)


def discover_run_dirs(run_root: Path) -> list[Path]:
    """List every training run directory that contains ``config.json``."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        return []

    run_dirs: list[Path] = []
    for model_dir in sorted(run_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name == RUN_DEBUG_DIRNAME:
            continue
        for candidate in sorted(model_dir.iterdir()):
            if candidate.is_dir() and (candidate / "config.json").is_file():
                run_dirs.append(candidate.resolve())
    return run_dirs


def _selection_for_fixed_sample(
    manifest_path: Path,
    sample: FixedDebugSample,
    *,
    warn: bool = False,
) -> dict[str, Any] | None:
    try:
        return select_sample(
            manifest_path,
            subject=sample.subject,
            session=sample.session,
            task=sample.task,
            direction=sample.direction,
            t=sample.t,
        )
    except ValueError as exc:
        if warn:
            print(f"[debug] skipping fixed sample {sample.key}: {exc}")
        return None


def _available_fixed_samples(manifest_path: Path) -> list[FixedDebugSample]:
    """Return fixed debug samples that resolve in the run manifest."""
    return [
        sample
        for sample in FIXED_DEBUG_SAMPLES
        if _selection_for_fixed_sample(manifest_path, sample) is not None
    ]


def _write_run_debug_manifest(
    debug_dir: Path,
    *,
    run_dir: Path,
    available_samples: list[FixedDebugSample],
) -> None:
    payload = {
        "version": 2,
        "run_dir": str(run_dir),
        "catalog": [asdict(sample) for sample in FIXED_DEBUG_SAMPLES],
        "available": [sample.key for sample in available_samples],
    }
    (debug_dir / "samples.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _load_target_slice(
    manifest_path: Path,
    sample: FixedDebugSample,
    *,
    source_voxel_mm: float,
    target_voxel_mm: float,
) -> np.ndarray:
    selection = _selection_for_fixed_sample(manifest_path, sample)
    if selection is None:
        raise RuntimeError(f"fixed sample unavailable in manifest: {sample.key}")
    payload = load_debug_sample(
        manifest_path,
        selection,
        source_voxel_mm=source_voxel_mm,
        target_voxel_mm=target_voxel_mm,
    )
    target_slice, _ = _extract_slice(
        payload["hr_image"],
        axis=sample.axis,
        slice_level=sample.slice_level,
    )
    return target_slice


@torch.no_grad()
def _forward_sample(
    model: torch.nn.Module,
    config: SRConfig,
    device: str,
    manifest_path: Path,
    selection: dict[str, Any],
) -> np.ndarray:
    """Run one manifest sample through ``model`` and return the prediction volume."""
    degrade_fn = make_spatial_degradation(
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
    )
    dataset = SpatialSRDataset(
        manifest_path=Path(manifest_path),
        degrade_fn=degrade_fn,
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
    )
    sample_index = None
    for idx, (run_idx, t) in enumerate(dataset.samples):
        if dataset.runs[run_idx]["run_id"] == selection["run_id"] and t == selection["t"]:
            sample_index = idx
            break
    if sample_index is None:
        raise RuntimeError(
            f"Could not locate run_id={selection['run_id']} t={selection['t']} "
            "in the dataset."
        )
    sample = dataset[sample_index]
    inputs = sample["input"].unsqueeze(0).to(device)
    pred = model(inputs)
    return _tensor_to_volume(pred)


def _prediction_slice_for_epoch(
    run_dir: Path,
    config: SRConfig,
    sample: FixedDebugSample,
    *,
    epoch_number: int,
    model: torch.nn.Module,
    device: str,
    manifest_path: Path,
) -> np.ndarray:
    selection = _selection_for_fixed_sample(manifest_path, sample)
    if selection is None:
        raise RuntimeError(f"fixed sample unavailable in manifest: {sample.key}")
    prediction = _forward_sample(model, config, device, manifest_path, selection)
    pred_slice, _ = _extract_slice(
        prediction,
        axis=sample.axis,
        slice_level=sample.slice_level,
    )
    return pred_slice


def _compute_all_epoch_prediction_slices(
    run_dir: Path,
    config: SRConfig,
    samples: list[FixedDebugSample],
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Run every saved epoch checkpoint once and collect slices for all samples."""
    manifest_path = Path(config.manifest_path)
    slices_by_key: dict[str, list[tuple[int, np.ndarray]]] = {
        sample.key: [] for sample in samples
    }

    for checkpoint in list_epoch_files(run_dir):
        epoch_number = int(checkpoint.stem.split("_")[1])
        device = auto_device()
        model = build_model(config).to(device)
        state = load_epoch(checkpoint, map_location=device)
        model.load_state_dict(state.model_state_dict)
        model.eval()
        for sample in samples:
            try:
                pred_slice = _prediction_slice_for_epoch(
                    run_dir,
                    config,
                    sample,
                    epoch_number=epoch_number,
                    model=model,
                    device=device,
                    manifest_path=manifest_path,
                )
            except Exception as exc:
                print(
                    f"[debug] warning: epoch {epoch_number} failed for "
                    f"{sample.key}: {exc}"
                )
                continue
            slices_by_key[sample.key].append((epoch_number, pred_slice))

    return slices_by_key


def plot_prediction_evolution(
    *,
    target_slice: np.ndarray,
    epoch_slices: list[tuple[int, np.ndarray]],
    sample: FixedDebugSample,
    output_path: Path,
) -> None:
    """Save GT + every epoch prediction in a multi-row grid (no subsampling)."""
    if not epoch_slices:
        return

    plt = _import_matplotlib(agg=True)
    panels: list[tuple[str, np.ndarray]] = [("Target (GT)", target_slice)]
    panels.extend((f"epoch {epoch_number}", pred_slice) for epoch_number, pred_slice in epoch_slices)

    ncol = EVOLUTION_COLS_PER_ROW
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow))
    axes_list = np.atleast_1d(axes).ravel().tolist()

    gray_stack = np.stack([arr for _, arr in panels])
    vmin = float(gray_stack.min())
    vmax = float(gray_stack.max())

    for ax, (title, image) in zip(axes_list, panels):
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes_list[len(panels) :]:
        ax.axis("off")

    fig.suptitle(
        f"Prediction evolution: {sample.direction} sub-{sample.subject} "
        f"ses-{sample.session} {sample.task} t={sample.t}\n"
        f"{sample.axis} slice level={sample.slice_level:.2f}  "
        f"({len(epoch_slices)} epochs, all shown)",
        fontsize=10,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[debug] wrote prediction evolution -> {output_path}")
    plt.close(fig)


def _write_mask_figures(
    debug_dir: Path,
    manifest_path: Path,
    *,
    source_voxel_mm: float,
    target_voxel_mm: float,
    samples: list[FixedDebugSample],
) -> None:
    masks_dir = debug_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        output_path = masks_dir / f"{sample.key}.png"
        try:
            selection = _selection_for_fixed_sample(manifest_path, sample)
            if selection is None:
                continue
            payload = load_debug_sample(
                manifest_path,
                selection,
                source_voxel_mm=source_voxel_mm,
                target_voxel_mm=target_voxel_mm,
            )
            save_debug_figure(
                payload,
                axis=sample.axis,
                slice_level=sample.slice_level,
                output_path=output_path,
                show=False,
            )
        except Exception as exc:
            print(f"[debug] warning: mask figure failed for {sample.key}: {exc}")


def _write_evolution_figures(
    run_dir: Path,
    config: SRConfig,
    *,
    debug_dir: Path,
) -> None:
    manifest_path = Path(config.manifest_path)
    evolution_dir = debug_dir / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)

    samples = _available_fixed_samples(manifest_path)
    if not samples or not list_epoch_files(run_dir):
        return

    slices_by_key = _compute_all_epoch_prediction_slices(run_dir, config, samples)
    for sample in samples:
        epoch_slices = slices_by_key.get(sample.key, [])
        if not epoch_slices:
            continue
        try:
            target_slice = _load_target_slice(
                manifest_path,
                sample,
                source_voxel_mm=float(config.source_voxel_mm),
                target_voxel_mm=float(config.target_voxel_mm),
            )
        except Exception as exc:
            print(f"[debug] warning: evolution failed for {sample.key}: {exc}")
            continue
        try:
            plot_prediction_evolution(
                target_slice=target_slice,
                epoch_slices=epoch_slices,
                sample=sample,
                output_path=evolution_dir / f"{sample.key}.png",
            )
        except Exception as exc:
            print(f"[debug] warning: evolution plot failed for {sample.key}: {exc}")


def _checkpoint_for_debug_infer(run_dir: Path) -> Path | None:
    best = best_epoch_path(run_dir)
    if best.is_file():
        return best
    return find_latest_epoch(run_dir)


def _build_infer_payload(
    checkpoint_path: Path,
    manifest_path: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Like ``load_debug_infer`` but reuses a caller-loaded ``result`` when provided."""
    result = infer_one(checkpoint_path, selection, override_manifest=manifest_path)
    run_dir = run_dir_for_checkpoint(checkpoint_path)
    config = from_json(run_dir / "config.json")
    mask_payload = load_debug_sample(
        manifest_path,
        selection,
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
    )
    target = result["target"]
    return {
        **result,
        "hr_mask": mask_payload["hr_mask"],
        "baseline": _trilinear_baseline_np(result["input"], target.shape),
        "loss_name": config.loss_name,
        "loss_kwargs": dict(config.loss_kwargs),
        "run_dir": run_dir,
    }


def _write_latest_infer_figures(
    run_dir: Path, config: SRConfig, *, debug_dir: Path
) -> None:
    checkpoint = _checkpoint_for_debug_infer(run_dir)
    if checkpoint is None:
        return

    latest_dir = debug_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.manifest_path)

    for sample in _available_fixed_samples(manifest_path):
        output_path = latest_dir / f"{sample.key}_infer.png"
        try:
            selection = _selection_for_fixed_sample(manifest_path, sample)
            if selection is None:
                continue
            infer_payload = _build_infer_payload(checkpoint, manifest_path, selection)
            save_infer_figure(
                infer_payload,
                axis=sample.axis,
                slice_level=sample.slice_level,
                error_map="abs",
                mask_errors=True,
                output_path=output_path,
                show=False,
            )
        except Exception as exc:
            print(f"[debug] warning: latest infer failed for {sample.key}: {exc}")


def _log_fixed_sample_coverage(manifest_path: Path, available: list[FixedDebugSample]) -> None:
    if not available:
        print(
            f"[debug] warning: none of the fixed debug samples are in {manifest_path}. "
            "Edit FIXED_DEBUG_SAMPLES in src/sr/debug.py if the manifest differs."
        )
    elif len(available) < len(FIXED_DEBUG_SAMPLES):
        missing = [s.key for s in FIXED_DEBUG_SAMPLES if s not in available]
        print(
            f"[debug] note: {len(missing)} fixed sample(s) not in manifest: "
            + ", ".join(missing)
        )


def ensure_run_debug_layout(run_dir: Path, config: SRConfig) -> Path:
    """Create ``<run_dir>/debug/`` when a run starts (no wipe on resume)."""
    debug_dir = run_debug_dir(run_dir)
    manifest_path = Path(config.manifest_path)
    available = _available_fixed_samples(manifest_path)
    _log_fixed_sample_coverage(manifest_path, available)
    _write_run_debug_manifest(
        debug_dir, run_dir=run_dir, available_samples=available
    )
    _write_mask_figures(
        debug_dir,
        manifest_path,
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
        samples=available,
    )
    print(
        f"[debug] run debug layout ready -> {debug_dir} "
        f"({len(available)}/{len(FIXED_DEBUG_SAMPLES)} samples)"
    )
    return debug_dir


def update_run_debug_after_epoch(run_dir: Path, *, epoch_number: int) -> None:
    """Recompute evolution and loss-curve figures from all epoch checkpoints."""
    _ = epoch_number
    run_dir = Path(run_dir).resolve()
    config = from_json(run_dir / "config.json")
    debug_dir = run_debug_dir(run_dir)

    _write_evolution_figures(run_dir, config, debug_dir=debug_dir)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        plot_loss_curve(run_dir, debug_dir / "loss_curve.png")


def finalize_run_debug(run_dir: Path) -> None:
    """Write heavier debug artifacts once training finishes."""
    run_dir = Path(run_dir).resolve()
    config = from_json(run_dir / "config.json")
    debug_dir = run_debug_dir(run_dir)
    _write_latest_infer_figures(run_dir, config, debug_dir=debug_dir)


def populate_run_debug(run_dir: Path, *, clear: bool = True) -> Path:
    """Refresh the central debug bundle for one training run."""
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in run directory: {run_dir}")

    config = from_json(config_path)
    debug_dir = Path(run_dir) / RUN_DEBUG_DIRNAME
    if clear:
        clear_debug_output_dir(debug_dir)
    else:
        run_debug_dir(run_dir)

    manifest_path = Path(config.manifest_path)
    available = _available_fixed_samples(manifest_path)
    _log_fixed_sample_coverage(manifest_path, available)
    _write_run_debug_manifest(
        debug_dir, run_dir=run_dir, available_samples=available
    )
    _write_mask_figures(
        debug_dir,
        manifest_path,
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
        samples=available,
    )
    _write_evolution_figures(run_dir, config, debug_dir=debug_dir)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            plot_loss_curve(run_dir, debug_dir / "loss_curve.png")
        except Exception as exc:
            print(f"[debug] warning: loss curve failed for {run_dir.name}: {exc}")

    _write_latest_infer_figures(run_dir, config, debug_dir=debug_dir)
    print(f"[debug] populated run debug bundle -> {debug_dir}")
    return debug_dir


def populate_all_runs_debug(
    run_root: Path | None = None,
    *,
    run_dir: Path | None = None,
) -> list[Path]:
    """Clear and regenerate debug output for all runs (or one run) under ``run_root``."""
    if run_dir is not None:
        return [populate_run_debug(run_dir, clear=True)]

    root = Path(run_root or SRConfig().run_root)
    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        print(f"[debug] no training runs found under {root}")
        return []

    print(f"[debug] populating debug/ for {len(run_dirs)} run(s) under {root}")
    written: list[Path] = []
    for candidate in run_dirs:
        debug_dir = candidate / RUN_DEBUG_DIRNAME
        try:
            written.append(populate_run_debug(candidate, clear=True))
        except Exception as exc:
            print(f"[debug] skipping {candidate}: {exc}")
            if debug_dir.is_dir():
                shutil.rmtree(debug_dir)
                print(f"[debug] removed {debug_dir}")
            traceback.print_exc()
    return written


def safe_update_run_debug_after_epoch(run_dir: Path, *, epoch_number: int) -> None:
    """Training hook: never abort the run when debug artifact generation fails."""
    try:
        update_run_debug_after_epoch(run_dir, epoch_number=epoch_number)
    except Exception as exc:
        print(f"[train] warning: run debug update failed after epoch {epoch_number}: {exc}")
        traceback.print_exc()


def add_debug_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags for the ``debug`` subcommand."""
    defaults = SRConfig()
    parser.add_argument(
        "--populate-runs",
        action="store_true",
        help=(
            "Clear and regenerate <run_dir>/debug/ for every training run "
            "(all models). Default when no sample selectors are given."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=defaults.run_root,
        help="Run root to scan with --populate-runs (default: SRConfig run_root).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Limit --populate-runs to a single training run directory.",
    )
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
    selection_filters = {
        "subject": args.subject,
        "session": args.session,
        "task": args.task,
        "direction": args.direction,
        "t": args.t,
    }
    wants_single_sample = any(value is not None for value in selection_filters.values())

    if args.populate_runs or (
        not wants_single_sample
        and not args.list_samples
        and args.checkpoint is None
        and args.save_png is None
        and args.save_dir is None
        and not args.show
        and not args.plot_loss_curve
    ):
        populate_all_runs_debug(args.run_root, run_dir=args.run_dir)
        return

    manifest = Path(args.manifest_path)
    if not manifest.is_file():
        raise SystemExit(f"manifest_path does not exist: {manifest}")

    if args.list_samples:
        print(format_sample_table(list_samples(manifest)))
        return

    if not wants_single_sample:
        raise SystemExit(
            "debug requires either --populate-runs, --list-samples, or sample "
            "selectors (--subject/--session/--task/--direction/--t). See --help."
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
