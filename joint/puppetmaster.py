"""puppetmaster.py — one-call inference endpoint for the joint denoise+SR model.

Hand it a whole BOLD run and an output path; it runs the joint model over every
timepoint and writes the reconstructed run as a 4D NIfTI.

It auto-detects the input resolution and runs in one of two modes:

  - HR input (128x128x93) -> "round-trip" mode: degrade each volume the way the
    model was trained (k-space truncation + Rician noise), then forward-pass.
    This is the standalone demo on existing IBC data: model(degrade(HR)).

  - LR input (64x64x46) -> "lr-native" mode: the volume is assumed to be ALREADY
    degraded, so no degradation is applied — it is fed straight to the model.
    This is what an external orchestrator uses when it degrades the run ONCE and
    compares joint against a denoise+SR cascade on the identical noisy LR.

Two ways to call it — identical work either way:

    # Python
    from joint.puppetmaster import run
    run("/srv/fMRI-data/sub-13_ses-16_task-PainMovie_dir-pa_bold.nii.gz",
        "/tmp/sub-13_painmovie_pred.nii.gz")

    # CLI
    python -m joint.puppetmaster \
        --input  /srv/fMRI-data/sub-13_ses-16_task-PainMovie_dir-pa_bold.nii.gz \
        --output /tmp/sub-13_painmovie_pred.nii.gz

Normalization: the model trains on per-run normalized data, so each volume is
divided by ``norm_ref`` before the forward pass and multiplied back after.
``norm_ref`` is resolved in this order: the value passed in (``--norm-ref``);
else, for an HR input, looked up from manifest_big by filename; else, for an LR
input, computed from the input's own temporal mean (98th percentile).

Output is denormalized to physical units. In HR mode it carries the input's
affine + header (a voxel/time-aligned drop-in). In LR mode the HR affine is
derived from the LR affine (voxel size halved); an orchestrator that holds the
true HR run can override that afterward.

Noise (HR/round-trip mode only) is fresh and random per call, so reruns are not
bit-identical — intentional (mimics real thermal noise).

Shared locations:
  weights  : <repo>/weights/joint/best.pt                         (joint03, the 4M model)
  manifest : /srv/venvs/team4dbrain/derivatives/manifest_big.json (46 runs)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from data.degradation_noise import Compose, RicianNoise
from data.degradation_spatial import make_spatial_degradation, voxel_size_to_target_shape
from data.reader import get_reader

from .eval import load_checkpoint

# --- shared locations -------------------------------------------------------
# joint checkpoint is versioned in the repo (weights/joint/best.pt); the manifest
# still lives on the project VM.
WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent / "weights" / "joint" / "best.pt")
MANIFEST_PATH = "/srv/venvs/team4dbrain/derivatives/manifest_big.json"


def _find_run(manifest_path: str, input_path: str) -> dict:
    """Match an input bold file to its manifest run entry (for norm_ref + shape).

    The manifest stores each run's bold as '<run_id>_bold.nii.gz'. We match on the
    input's basename first, then fall back to the run_id derived from it. Raises a
    clear error unless exactly one run matches — this is how an HR input without an
    explicit ``norm_ref`` gets its per-run norm_ref.
    """
    runs = json.loads(Path(manifest_path).read_text())["runs"]
    in_base = os.path.basename(str(input_path))
    matches = [r for r in runs if r.get("path") == in_base]
    if not matches:
        stem = in_base
        for suffix in ("_bold.nii.gz", ".nii.gz", "_bold.nii", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        matches = [r for r in runs if r.get("run_id") == stem]
    if len(matches) != 1:
        raise ValueError(
            f"input {in_base!r} matched {len(matches)} runs in {manifest_path} "
            f"(need exactly 1). An HR input with no --norm-ref is looked up here for "
            f"its per-run norm_ref. Pass --norm-ref, or add the run to a manifest "
            f"(compute_metadata.py) first."
        )
    return matches[0]


def _norm_ref_from_reader(reader, percentile: float = 98.0) -> float:
    """Fallback norm_ref for a standalone LR run: 98th pct of its temporal mean."""
    acc = None
    for t in range(reader.n_volumes):
        v = reader.read_volume(t).astype(np.float64)
        acc = v if acc is None else acc + v
    mean = acc / max(reader.n_volumes, 1)
    ref = float(np.percentile(mean, percentile))
    return ref if ref > 0 else 1.0


@torch.no_grad()
def run(input_path: str, output_path: str, norm_ref: float | None = None) -> str:
    """Forward-pass a whole BOLD run through the joint model and save the result.

    Args:
        input_path:  a 4D BOLD run. HR (128,128,93) -> degraded internally then
                     reconstructed; LR (64,64,46) -> assumed already degraded,
                     fed straight to the model.
        output_path: where to write the reconstructed 4D NIfTI (.nii.gz).
        norm_ref:    per-run normalization scale. If None: HR looks it up in
                     manifest_big; LR computes it from the input's temporal mean.

    Returns:
        output_path (for convenience / chaining).
    """
    input_path = str(input_path)
    output_path = str(output_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input run not found: {input_path}")

    # 1. model (+ the exact training config it was saved with) from shared weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(WEIGHTS_PATH, device)

    hr_shape = tuple(int(s) for s in cfg.model.out_size)                 # (128, 128, 93)
    lr_shape = tuple(int(s) for s in voxel_size_to_target_shape(
        hr_shape, cfg.train.source_voxel_mm, cfg.train.target_voxel_mm))  # (64, 64, 46)

    reader = get_reader(input_path)
    in_img = nib.load(input_path)        # header only, no decompression
    T = reader.n_volumes
    in_shape = reader.shape3d

    # 2. pick the mode from the input resolution
    if in_shape == hr_shape:
        mode = "hr"                       # clean HR in -> degrade -> reconstruct
        if norm_ref is None:
            entry = _find_run(MANIFEST_PATH, input_path)
            norm_ref = float(entry["norm_ref"])
            tag = entry["run_id"]
        else:
            tag = os.path.basename(input_path)
        degrade = Compose([
            make_spatial_degradation(
                source_voxel_mm=cfg.train.source_voxel_mm,
                target_voxel_mm=cfg.train.target_voxel_mm,
            ),
            RicianNoise(
                sigma_min=cfg.train.sigma_min,
                sigma_max=cfg.train.sigma_max,
                seed=None,                # fresh entropy per call -> independent noise per t
            ),
        ])
        out_affine = in_img.affine
    elif in_shape == lr_shape:
        mode = "lr"                       # already-degraded LR in -> forward only
        if norm_ref is None:
            norm_ref = _norm_ref_from_reader(reader)
        tag = os.path.basename(input_path)
        degrade = None
        # derive an HR affine from the LR affine (voxel size halved, FOV preserved);
        # an orchestrator holding the true HR run can overwrite this afterward.
        out_affine = in_img.affine.copy()
        out_affine[:3, :3] = in_img.affine[:3, :3] * (
            cfg.train.source_voxel_mm / cfg.train.target_voxel_mm)
    else:
        raise ValueError(
            f"input spatial shape {in_shape} is neither the HR shape {hr_shape} "
            f"nor the LR shape {lr_shape}. Pass a full-res IBC run, or a "
            f"pre-degraded LR run."
        )

    norm_ref = float(norm_ref)
    print(f"[puppetmaster] {tag}: {T} volumes | mode={mode} | norm_ref={norm_ref:.1f} | "
          f"device={device.type} | weights={os.path.basename(WEIGHTS_PATH)}", flush=True)

    # 3. per-timepoint: normalize -> (degrade if HR) -> forward -> denormalize -> stack
    X, Y, Z = hr_shape
    out = np.zeros((X, Y, Z, T), dtype=np.float32)
    for t in range(T):
        vol = reader.read_volume(t).astype(np.float32) / norm_ref      # normalized
        lr = degrade(vol) if mode == "hr" else vol                     # LR (64,64,46)
        x = torch.from_numpy(np.ascontiguousarray(lr)).float()[None, None].to(device)
        out[..., t] = model(x)[0, 0].cpu().numpy() * norm_ref          # physical units
        if (t + 1) % 50 == 0 or t + 1 == T:
            print(f"[puppetmaster]   {t + 1}/{T}", flush=True)

    # 4. write the 4D NIfTI. HR mode: faithful drop-in (input affine + voxel sizes + TR).
    #    LR mode: derived HR affine; carry HR voxel size + the input's TR.
    out_img = nib.Nifti1Image(out, out_affine)
    if mode == "hr":
        out_img.header.set_zooms(in_img.header.get_zooms())
    else:
        sv = float(cfg.train.source_voxel_mm)
        in_zooms = in_img.header.get_zooms()
        tr = float(in_zooms[3]) if len(in_zooms) > 3 else 0.0
        out_img.header.set_zooms((sv, sv, sv, tr) if tr > 0 else (sv, sv, sv))
    try:
        out_img.header.set_xyzt_units(*in_img.header.get_xyzt_units())
    except Exception:
        pass
    out_img.header.set_data_dtype(np.float32)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    nib.save(out_img, output_path)
    print(f"[puppetmaster] saved {tag} -> {output_path}  shape={out.shape}", flush=True)
    return output_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Forward-pass a whole BOLD run through the joint denoise+SR model. "
                    "HR input (128^3) is degraded internally then reconstructed; LR "
                    "input (64^3) is assumed pre-degraded and fed straight to the model."
    )
    ap.add_argument("--input", "-i", required=True,
                    help="input 4D BOLD run (.nii.gz): HR (128x128x93) or LR (64x64x46)")
    ap.add_argument("--output", "-o", required=True,
                    help="output path for the reconstructed 4D run (.nii.gz)")
    ap.add_argument("--norm-ref", type=float, default=None,
                    help="per-run normalization scale; if omitted, HR looks it up in "
                         "manifest_big and LR computes it from the input")
    args = ap.parse_args(argv)
    run(args.input, args.output, norm_ref=args.norm_ref)


if __name__ == "__main__":
    main()
