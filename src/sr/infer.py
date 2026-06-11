"""Evaluation and single-sample inference utilities.

Purpose:
    Load a trained checkpoint and either (a) score it on the saved
    validation split or (b) run one sample of the user's choosing and
    optionally produce a slice figure. Both code paths are explicit: the
    config that controls behaviour comes from the run directory of the
    checkpoint, with no hidden defaults.
Effects:
    Prints metrics, optionally writes a JSON report (eval) or PNG/NPY
    artifacts (infer). Never overwrites training artifacts.
Influences:
    Sample selection uses manifest filters; if filters match multiple
    samples without a timepoint, the user gets a clear list and error.
How to change safely:
    Keep ``select_sample`` strict: if filters are ambiguous, raise. Do not
    silently pick the first match -- that has burned us before.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.datasets import SpatialSRDataset
from src.data.degradation_spatial import make_spatial_degradation
from src.sr.checkpoint import load_epoch, run_dir_for_checkpoint
from src.sr.config import SRConfig, auto_device, from_json
from src.sr.data import build_loaders
from src.sr.metrics import compute_full_metrics, volume_intensity_stats
from src.sr.models import build_model


AXIS_TO_INDEX: dict[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}


# ---------------------------------------------------------------------------
# Manifest-driven sample listing/selection
# ---------------------------------------------------------------------------


def list_samples(manifest_path: Path) -> list[dict[str, Any]]:
    """Return one row per run in the manifest with selector-relevant fields."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        if "norm_ref" not in run or "mask_path" not in run:
            continue
        rows.append(
            {
                "run_id": run["run_id"],
                "subject": run.get("subject"),
                "session": run.get("session"),
                "task": run.get("task"),
                "direction": run.get("direction"),
                "n_volumes": int(run.get("n_volumes", 0)),
            }
        )
    rows.sort(key=lambda r: (r["subject"] or "", r["session"] or "", r["task"] or "", r["run_id"]))
    return rows


def format_sample_table(rows: list[dict[str, Any]]) -> str:
    """Human-readable table of ``list_samples`` output."""
    header = f"{'subject':<8} {'session':<8} {'task':<24} {'dir':<4} {'n_vol':>6}  run_id"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{(row['subject'] or ''):<8} {(row['session'] or ''):<8} "
            f"{(row['task'] or ''):<24} {(row['direction'] or ''):<4} "
            f"{row['n_volumes']:>6}  {row['run_id']}"
        )
    return "\n".join(lines)


def select_sample(
    manifest_path: Path,
    *,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    direction: str | None = None,
    t: int | None = None,
) -> dict[str, Any]:
    """Resolve filters to exactly one (run, timepoint).

    Raises:
        ValueError if filters match zero runs, multiple runs (with a
        candidate list in the message), or if ``t`` is missing/out of range.
    """
    rows = list_samples(manifest_path)
    candidates = [
        row
        for row in rows
        if (
            (subject is None or row["subject"] == subject)
            and (session is None or row["session"] == session)
            and (task is None or row["task"] == task)
            and (direction is None or row["direction"] == direction)
        )
    ]
    if not candidates:
        raise ValueError(
            "No manifest run matches the given filters "
            f"(subject={subject}, session={session}, task={task}, direction={direction})."
        )
    if len(candidates) > 1:
        preview = "\n".join(
            f"  - subject={c['subject']} session={c['session']} task={c['task']} "
            f"direction={c['direction']} run_id={c['run_id']}"
            for c in candidates[:10]
        )
        more = "" if len(candidates) <= 10 else f"\n  ... ({len(candidates) - 10} more)"
        raise ValueError(
            "Filters are ambiguous, multiple runs match:\n"
            f"{preview}{more}\nTighten the filters until exactly one run remains."
        )
    chosen = candidates[0]
    if t is None:
        raise ValueError(
            f"Run '{chosen['run_id']}' has {chosen['n_volumes']} timepoints. "
            "Pass --t <int> to pick one."
        )
    if not 0 <= int(t) < chosen["n_volumes"]:
        raise ValueError(
            f"t={t} out of range [0, {chosen['n_volumes']}) for run '{chosen['run_id']}'."
        )
    chosen["t"] = int(t)
    return chosen


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def _load_model_from_checkpoint(
    checkpoint_path: Path, override_manifest: Path | None = None
) -> tuple[torch.nn.Module, SRConfig, str]:
    """Load the model + config bound to a given epoch checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    run_dir = run_dir_for_checkpoint(checkpoint_path)
    config = from_json(run_dir / "config.json")
    if override_manifest is not None:
        config.manifest_path = Path(override_manifest)
    device = auto_device()
    model = build_model(config).to(device)
    state = load_epoch(checkpoint_path, map_location=device)
    model.load_state_dict(state.model_state_dict)
    model.eval()
    return model, config, device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    *,
    override_manifest: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the saved validation split through ``checkpoint_path``.

    Returns the averaged metric dict and (optionally) writes it to JSON.
    Prints every value so a user piping output to logs sees the numbers.
    """
    model, config, device = _load_model_from_checkpoint(checkpoint_path, override_manifest)

    _, val_loader, split_info = build_loaders(config)
    if val_loader is None:
        raise RuntimeError(
            "Cannot evaluate: the saved configuration has no validation split. "
            "Re-train with train_split < 1.0 or run inference instead."
        )

    per_batch: list[dict[str, float]] = []
    for batch in val_loader:
        pred = model(batch["input"].to(device))
        target = batch["target"].to(device)
        mask = batch["mask_hr"].to(device)
        per_batch.append(
            compute_full_metrics(
                pred,
                target,
                mask,
                training_loss_name=config.loss_name,
                training_loss_kwargs=config.loss_kwargs,
            )
        )

    from src.sr.metrics import average_metric_dicts

    averaged = average_metric_dicts(per_batch)
    print(f"[eval] checkpoint   = {checkpoint_path}")
    print(f"[eval] manifest     = {config.manifest_path}")
    print(f"[eval] split_source = {split_info['source']}")
    print(f"[eval] val_samples  = {split_info['val_samples']}")
    for key in sorted(averaged):
        print(f"[eval] {key:>14} = {averaged[key]:.6f}")

    payload = {
        "checkpoint": str(checkpoint_path),
        "manifest_path": str(config.manifest_path),
        "split_source": split_info["source"],
        "val_samples": split_info["val_samples"],
        "metrics": averaged,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[eval] wrote report -> {report_path}")
    return payload


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------


def print_volume_intensity_stats(volume_stats: dict[str, dict[str, float]]) -> None:
    """Print min/max/mean for input, prediction, and target (from ``infer_one``)."""
    for label in ("input", "prediction", "target"):
        s = volume_stats[label]
        print(
            f"[infer] {label:>12}  min={s['min']:.6f}  "
            f"max={s['max']:.6f}  mean={s['mean']:.6f}"
        )


@torch.no_grad()
def infer_one(
    checkpoint_path: Path,
    selection: dict[str, Any],
    *,
    override_manifest: Path | None = None,
) -> dict[str, Any]:
    """Run one (run, timepoint) through the model and return tensors + metrics.

    ``selection`` is the dict returned by ``select_sample`` (must contain
    ``subject``, ``run_id``, ``t``).
    """
    model, config, device = _load_model_from_checkpoint(checkpoint_path, override_manifest)

    degrade_fn = make_spatial_degradation(
        source_voxel_mm=float(config.source_voxel_mm),
        target_voxel_mm=float(config.target_voxel_mm),
    )
    dataset = SpatialSRDataset(
        manifest_path=Path(config.manifest_path),
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
            "in the filtered dataset. The manifest may have changed since training."
        )

    sample = dataset[sample_index]
    inputs = sample["input"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    mask = sample["mask_hr"].unsqueeze(0).to(device)
    pred = model(inputs)

    metrics = compute_full_metrics(
        pred,
        target,
        mask,
        training_loss_name=config.loss_name,
        training_loss_kwargs=config.loss_kwargs,
    )
    input_np = inputs.squeeze(0).squeeze(0).detach().cpu().numpy()
    prediction_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
    target_np = target.squeeze(0).squeeze(0).detach().cpu().numpy()
    volume_stats = {
        "input": volume_intensity_stats(input_np),
        "prediction": volume_intensity_stats(prediction_np),
        "target": volume_intensity_stats(target_np),
    }
    return {
        "run_id": selection["run_id"],
        "subject": selection["subject"],
        "session": selection.get("session"),
        "task": selection.get("task"),
        "direction": selection.get("direction"),
        "t": selection["t"],
        "metrics": metrics,
        "volume_stats": volume_stats,
        "input": input_np,
        "prediction": prediction_np,
        "target": target_np,
    }


def make_slice_figure(
    input_vol: np.ndarray,
    prediction_vol: np.ndarray,
    target_vol: np.ndarray,
    *,
    axis: str,
    slice_level: float,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Render input/prediction/target side-by-side at one slice.

    ``slice_level`` is a relative position in [0, 1]; 0.5 means center.
    Each panel's title states the axis, the resolved slice index and the
    level, so the figure is self-documenting in printouts.
    """
    try:
        import matplotlib
        if output_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for slice figures. Install it or pass "
            "--save-npy only."
        ) from exc

    if axis not in AXIS_TO_INDEX:
        raise ValueError(f"Unknown axis '{axis}'. Choose one of: {sorted(AXIS_TO_INDEX)}")
    if not 0.0 <= slice_level <= 1.0:
        raise ValueError(f"slice_level must be in [0, 1], got {slice_level}")

    def extract(volume: np.ndarray) -> tuple[np.ndarray, int]:
        idx_axis = AXIS_TO_INDEX[axis]
        dim = int(volume.shape[idx_axis])
        idx = min(dim - 1, max(0, int(round(slice_level * (dim - 1)))))
        if axis == "axial":
            return volume[:, :, idx], idx
        if axis == "coronal":
            return volume[:, idx, :], idx
        return volume[idx, :, :], idx

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        ("Input (LR)", extract(input_vol)),
        ("Prediction", extract(prediction_vol)),
        ("Target (HR)", extract(target_vol)),
    ]
    for ax, (title, (image, idx)) in zip(axes, panels):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{title}\n{axis} slice={idx} (level={slice_level:.2f})")
        ax.axis("off")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[infer] wrote figure -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)
