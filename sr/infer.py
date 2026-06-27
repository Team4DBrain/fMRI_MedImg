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

from sr.checkpoint import (
    load_config_for_inference,
    load_epoch,
    resolve_checkpoint_for_model,
    try_run_dir_for_checkpoint,
)
from sr.config import SRConfig, auto_device, from_json
from sr.data import build_loaders, build_spatial_sr_dataset, resolve_dataset_sample
from sr.metrics import average_metric_dicts, compute_full_metrics, volume_intensity_stats
from sr.models import build_model
from sr.forward import model_forward

from data.degradation_spatial import (
    make_spatial_degradation,
    voxel_size_to_target_shape,
)
from data.normalize import denormalize, normalize
from data.reader import get_reader


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


def extract_slice(
    volume: np.ndarray,
    *,
    axis: str,
    slice_level: float,
) -> tuple[np.ndarray, int]:
    """Return one 2D slice and its index along ``axis``.

    Purpose:
        Shared slice extraction for infer figures and debug evolution plots
        so axis naming and level-to-index mapping stay consistent.
    """
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


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    override_manifest: Path | None = None,
    *,
    model_name: str | None = None,
    config_path: Path | None = None,
) -> tuple[torch.nn.Module, SRConfig, str]:
    """Load the model + config bound to a given epoch checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    config = load_config_for_inference(
        checkpoint_path,
        model_name=model_name,
        config_path=config_path,
    )
    if override_manifest is not None:
        config.manifest_path = Path(override_manifest)
    device = auto_device()
    model = build_model(config).to(device)
    state = load_epoch(checkpoint_path)
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
    model_name: str | None = None,
    config_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the saved validation split through ``checkpoint_path``.

    Returns the averaged metric dict and (optionally) writes it to JSON.
    Prints every value so a user piping output to logs sees the numbers.
    """
    model, config, device = _load_model_from_checkpoint(
        checkpoint_path,
        override_manifest,
        model_name=model_name,
        config_path=config_path,
    )

    _, val_loader, split_info = build_loaders(config)
    if val_loader is None:
        raise RuntimeError(
            "Cannot evaluate: the saved configuration has no validation split. "
            "Re-train with train_split < 1.0 or run inference instead."
        )

    per_batch: list[dict[str, float]] = []
    for batch in val_loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask_hr"].to(device)
        pred = model_forward(model, inputs, target, config.model_name)
        per_batch.append(
            compute_full_metrics(
                pred,
                target,
                mask,
                training_loss_name=config.loss_name,
                training_loss_kwargs=config.loss_kwargs,
            )
        )

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
# NIfTI file inference (no manifest)
# ---------------------------------------------------------------------------


def default_sr_output_path(input_path: Path) -> Path:
    """Return ``<input_stem>_sr.nii.gz`` next to the source file."""
    input_path = Path(input_path)
    name = input_path.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        stem = name[: -len(".nii")]
    else:
        stem = input_path.stem
    return input_path.with_name(f"{stem}_sr.nii.gz")


def _looks_like_nifti_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".nii.gz") or name.endswith(".nii")


def resolve_sr_output_path(input_path: Path, output_path: Path | None) -> Path:
    """Resolve ``--output`` to a concrete ``.nii.gz`` file path.

    If ``output_path`` is omitted, writes beside the input. If it is an existing
    directory, a trailing slash path, or a path without a NIfTI extension, the
    default ``<stem>_sr.nii.gz`` name is placed inside that directory.
    """
    default_file = default_sr_output_path(input_path)
    if output_path is None:
        return default_file

    out = Path(output_path).expanduser()
    if out.exists() and out.is_dir():
        return out / default_file.name
    if str(output_path).rstrip().endswith(("/", "\\")):
        out.mkdir(parents=True, exist_ok=True)
        return out / default_file.name
    if not _looks_like_nifti_file(out):
        out.mkdir(parents=True, exist_ok=True)
        return out / default_file.name
    return out


def default_sr_preview_path(nifti_output_path: Path) -> Path:
    """Return a PNG preview path beside the SR NIfTI (``.nii.gz`` -> ``.png``)."""
    path = Path(nifti_output_path)
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[: -len(".nii.gz")] + ".png")
    if name.endswith(".nii"):
        return path.with_name(name[: -len(".nii")] + ".png")
    return path.with_suffix(".png")


def _lr_shape_for_config(config: SRConfig) -> tuple[int, int, int]:
    return voxel_size_to_target_shape(
        tuple(config.output_patch_shape),
        config.source_voxel_mm,
        config.target_voxel_mm,
    )


def _prepare_lr_volume(
    volume: np.ndarray,
    norm_ref: float,
    config: SRConfig,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Normalize and optionally degrade a NIfTI volume to model LR input.

    Skips k-space degradation when the file is already at the simulated LR
    grid (e.g. 64×64×46 for 3 mm). Applies degradation when it matches the
    configured HR grid (e.g. 128×128×93 at 1.5 mm).

    Returns ``(lr, mode, ground_truth)`` where ``ground_truth`` is the
    normalized HR volume when degradation was applied, else ``None``.
    """
    shape = tuple(int(s) for s in volume.shape)
    hr_shape = tuple(int(s) for s in config.output_patch_shape)
    lr_shape = _lr_shape_for_config(config)

    if shape == lr_shape:
        return normalize(volume, norm_ref), "lr_native", None

    if shape == hr_shape:
        degrade = make_spatial_degradation(
            source_voxel_mm=config.source_voxel_mm,
            target_voxel_mm=config.target_voxel_mm,
        )
        hr_clean = normalize(volume, norm_ref)
        return degrade(hr_clean), "hr_degraded", hr_clean

    raise ValueError(
        f"Input shape {shape} is neither the expected HR shape {hr_shape} "
        f"nor the LR shape {lr_shape} for "
        f"{config.source_voxel_mm}mm -> {config.target_voxel_mm}mm. "
        "Pass native-resolution HR (degraded internally) or pre-downsampled LR."
    )


def _nifti_is_4d(path: Path) -> bool:
    """Return True when ``path`` is a 4D NIfTI (header-only check)."""
    import nibabel as nib

    return len(nib.load(str(path)).shape) == 4


def _write_nifti_output(
    data: np.ndarray,
    in_img: "nibabel.Nifti1Image",
    output_path: Path,
) -> None:
    """Write a 3D/4D NIfTI drop-in with the input's affine, zooms, and TR."""
    import nibabel as nib

    out_img = nib.Nifti1Image(data.astype(np.float32), in_img.affine)
    out_img.header.set_zooms(in_img.header.get_zooms())
    try:
        out_img.header.set_xyzt_units(*in_img.header.get_xyzt_units())
    except Exception:
        pass
    out_img.header.set_data_dtype(np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))


def _load_nifti_volume(path: Path, *, t: int | None) -> tuple[np.ndarray, "nibabel.Nifti1Image"]:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 3:
        volume = data
    elif data.ndim == 4:
        if t is None:
            raise ValueError(
                f"4D input {path} requires --t for single-volume infer, or omit --t "
                "to process the full run."
            )
        time_idx = int(t)
        if not 0 <= time_idx < data.shape[3]:
            raise ValueError(
                f"t={time_idx} out of range [0, {data.shape[3]}) for 4D input {path}"
            )
        volume = data[..., time_idx]
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {data.shape} from {path}")
    return np.ascontiguousarray(volume), img


def _norm_ref_from_volume(volume: np.ndarray, percentile: float = 98.0) -> float:
    """Scalar normalization for standalone NIfTI (no brain mask on disk)."""
    ref = float(np.percentile(volume, percentile))
    if ref <= 0:
        raise ValueError(
            "Could not derive a positive norm_ref from the input volume; "
            "check that the NIfTI contains brain BOLD data."
        )
    return ref


@torch.no_grad()
def _infer_nifti_4d_run(
    model: torch.nn.Module,
    config: SRConfig,
    device: str,
    input_path: Path,
    output_path: Path,
    *,
    norm_ref: float | None = None,
    write_preview: bool = False,
    preview_path: Path | None = None,
    slice_level: float = 0.5,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Super-resolve every timepoint in a 4D HR BOLD run and stack the result."""
    import nibabel as nib

    reader = get_reader(input_path)
    hr_shape = tuple(int(s) for s in config.output_patch_shape)
    if reader.shape3d != hr_shape:
        raise ValueError(
            f"4D full-run infer expects HR spatial shape {hr_shape}, got "
            f"{reader.shape3d}. LR-native 4D runs are not supported; pass --t "
            "for single-volume infer on one timepoint."
        )

    T = reader.n_volumes
    in_img = nib.load(str(input_path))
    ref = (
        float(norm_ref)
        if norm_ref is not None
        else _norm_ref_from_volume(reader.read_mean())
    )

    target_shape = tuple(config.output_patch_shape)
    target = torch.zeros((1, 1, *target_shape), device=device, dtype=torch.float32)

    X, Y, Z = hr_shape
    out = np.zeros((X, Y, Z, T), dtype=np.float32)
    preview_lr: np.ndarray | None = None
    preview_pred: np.ndarray | None = None
    preview_gt: np.ndarray | None = None
    preview_t = T // 2 if write_preview else None

    print(
        f"[infer] 4D run: {T} volumes | norm_ref={ref:.6f} | "
        f"device={device} | checkpoint={checkpoint_path.name}",
        flush=True,
    )
    print("[infer] applied k-space degradation (HR -> LR) per timepoint")

    for t_idx in range(T):
        hr_vol = reader.read_volume(t_idx).astype(np.float32)
        lr, _, ground_truth = _prepare_lr_volume(hr_vol, ref, config)
        inputs = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
        pred = model_forward(model, inputs, target, config.model_name)
        pred_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
        out[..., t_idx] = denormalize(pred_np, ref)

        if preview_t is not None and t_idx == preview_t:
            preview_lr = lr
            preview_pred = pred_np
            preview_gt = ground_truth

        if (t_idx + 1) % 50 == 0 or t_idx + 1 == T:
            print(f"[infer]   {t_idx + 1}/{T}", flush=True)

    _write_nifti_output(out, in_img, output_path)
    print(f"[infer] wrote NIfTI -> {output_path}  shape={out.shape}")

    preview_out: Path | None = None
    if write_preview and preview_lr is not None and preview_pred is not None:
        preview_out = (
            Path(preview_path)
            if preview_path is not None
            else default_sr_preview_path(output_path)
        )
        make_sr_output_preview(
            input_lr=preview_lr,
            prediction_vol=preview_pred,
            ground_truth_vol=preview_gt,
            output_path=preview_out,
            slice_level=slice_level,
        )
        if preview_t is not None:
            print(f"[infer] preview from timepoint t={preview_t}")

    volume_stats = {
        "input": volume_intensity_stats(preview_lr if preview_lr is not None else lr),
        "prediction": volume_intensity_stats(
            preview_pred if preview_pred is not None else pred_np
        ),
    }
    if preview_gt is not None:
        volume_stats["target"] = volume_intensity_stats(preview_gt)

    print(f"[infer] checkpoint   = {checkpoint_path}")
    print(f"[infer] input        = {input_path}")
    print(f"[infer] output       = {output_path}")
    print(f"[infer] norm_ref     = {ref:.6f}")
    print(f"[infer] output.shape = {out.shape}")
    print_volume_intensity_stats(volume_stats)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "norm_ref": ref,
        "prediction_physical": out,
        "volume_stats": volume_stats,
        "preview_path": str(preview_out) if preview_out is not None else None,
        "n_volumes": T,
    }


@torch.no_grad()
def infer_nifti(
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path | None = None,
    *,
    t: int | None = None,
    norm_ref: float | None = None,
    write_preview: bool = False,
    preview_path: Path | None = None,
    slice_level: float = 0.5,
    model_name: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Super-resolve a 3D/4D NIfTI and write ``<stem>_sr.nii.gz`` by default.

    3D inputs and ``4D + --t`` produce one HR volume. A 4D input with no
    ``--t`` processes every timepoint and writes a stacked 4D drop-in.
    """
    input_path = Path(input_path)
    output_path = resolve_sr_output_path(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, config, device = _load_model_from_checkpoint(
        checkpoint_path,
        model_name=model_name,
        config_path=config_path,
    )

    if t is None and _nifti_is_4d(input_path):
        return _infer_nifti_4d_run(
            model,
            config,
            device,
            input_path,
            output_path,
            norm_ref=norm_ref,
            write_preview=write_preview,
            preview_path=preview_path,
            slice_level=slice_level,
            checkpoint_path=Path(checkpoint_path),
        )

    volume, img = _load_nifti_volume(input_path, t=t)

    ref = float(norm_ref) if norm_ref is not None else _norm_ref_from_volume(volume)
    lr, input_mode, ground_truth = _prepare_lr_volume(volume, ref, config)
    if input_mode == "lr_native":
        print("[infer] input already at LR resolution; skipping degradation")
    else:
        print("[infer] applied k-space degradation (HR -> LR)")

    inputs = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
    target_shape = tuple(config.output_patch_shape)
    target = torch.zeros(
        (1, 1, *target_shape), device=device, dtype=inputs.dtype
    )
    pred = model_forward(model, inputs, target, config.model_name)
    prediction_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
    prediction_phys = denormalize(prediction_np, ref)

    _write_nifti_output(prediction_phys, img, output_path)
    print(f"[infer] wrote NIfTI -> {output_path}")

    preview_path: Path | None = None
    if write_preview:
        preview_path = Path(preview_path) if preview_path is not None else default_sr_preview_path(
            output_path
        )
        make_sr_output_preview(
            input_lr=lr,
            prediction_vol=prediction_np,
            ground_truth_vol=ground_truth,
            output_path=preview_path,
            slice_level=slice_level,
        )

    volume_stats = {
        "input": volume_intensity_stats(lr),
        "prediction": volume_intensity_stats(prediction_np),
    }
    if ground_truth is not None:
        volume_stats["target"] = volume_intensity_stats(ground_truth)
    print(f"[infer] checkpoint   = {checkpoint_path}")
    print(f"[infer] input        = {input_path}")
    print(f"[infer] output       = {output_path}")
    print(f"[infer] norm_ref     = {ref:.6f}")
    print(f"[infer] input.shape  = {lr.shape}")
    print(f"[infer] output.shape = {prediction_np.shape}")
    print_volume_intensity_stats(volume_stats)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "norm_ref": ref,
        "input_lr": lr,
        "ground_truth": ground_truth,
        "prediction": prediction_np,
        "prediction_physical": prediction_phys,
        "volume_stats": volume_stats,
        "preview_path": str(preview_path) if preview_path is not None else None,
    }


def make_sr_output_preview(
    input_lr: np.ndarray,
    prediction_vol: np.ndarray,
    *,
    ground_truth_vol: np.ndarray | None = None,
    output_path: Path,
    slice_level: float = 0.5,
) -> None:
    """Save an LR / SR / (optional GT) montage across axial, coronal, sagittal."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to visualize SR output. Install matplotlib "
            "or omit --preview."
        ) from exc

    if not 0.0 <= slice_level <= 1.0:
        raise ValueError(f"slice_level must be in [0, 1], got {slice_level}")

    axes_names = ("axial", "coronal", "sagittal")
    rows: list[tuple[str, np.ndarray]] = [
        ("Input (LR)", input_lr),
        ("SR output", prediction_vol),
    ]
    if ground_truth_vol is not None:
        rows.append(("Ground truth (HR)", ground_truth_vol))

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.5 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for row_idx, (row_label, volume) in enumerate(rows):
        for col, axis in enumerate(axes_names):
            slice_img, slice_idx = extract_slice(
                volume, axis=axis, slice_level=slice_level
            )
            axes[row_idx, col].imshow(slice_img, cmap="gray", origin="lower")
            axes[row_idx, col].set_title(f"{row_label}\n{axis} z={slice_idx}")
            axes[row_idx, col].axis("off")
    fig.suptitle("Spatial super-resolution preview", fontsize=14)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[infer] wrote preview -> {output_path}")


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------


def print_volume_intensity_stats(volume_stats: dict[str, dict[str, float]]) -> None:
    """Print min/max/mean for volumes present in ``volume_stats``."""
    for label in ("input", "prediction", "target"):
        if label not in volume_stats:
            continue
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
    model_name: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run one (run, timepoint) through the model and return tensors + metrics.

    ``selection`` is the dict returned by ``select_sample`` (must contain
    ``subject``, ``run_id``, ``t``).
    """
    model, config, device = _load_model_from_checkpoint(
        checkpoint_path,
        override_manifest,
        model_name=model_name,
        config_path=config_path,
    )

    dataset = build_spatial_sr_dataset(
        config.manifest_path,
        source_voxel_mm=config.source_voxel_mm,
        target_voxel_mm=config.target_voxel_mm,
    )
    sample = resolve_dataset_sample(dataset, selection["run_id"], selection["t"])
    inputs = sample["input"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    mask = sample["mask_hr"].unsqueeze(0).to(device)
    pred = model_forward(model, inputs, target, config.model_name)

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
    mask_hr_np = mask.squeeze(0).squeeze(0).detach().cpu().numpy()
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
        "mask_hr": mask_hr_np,
    }


def make_slice_figure(
    input_vol: np.ndarray,
    prediction_vol: np.ndarray,
    target_vol: np.ndarray | None = None,
    *,
    axis: str,
    slice_level: float,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Render input/prediction/(optional target) side-by-side at one slice.

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

    fig, axes = plt.subplots(1, 3 if target_vol is not None else 2, figsize=(14, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    panels: list[tuple[str, tuple[np.ndarray, int]]] = [
        ("Input (LR)", extract_slice(input_vol, axis=axis, slice_level=slice_level)),
        ("Prediction", extract_slice(prediction_vol, axis=axis, slice_level=slice_level)),
    ]
    if target_vol is not None:
        panels.append(
            ("Target (HR)", extract_slice(target_vol, axis=axis, slice_level=slice_level))
        )
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
