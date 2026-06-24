"""puppetmaster.py — one-call inference endpoint for the joint denoise+SR model.

Hand it a whole BOLD run and an output path; it degrades each volume the way the
model was trained (k-space truncation + Rician noise), runs the forward pass,
denormalizes back to physical units, and writes the reconstructed run as a 4D
NIfTI that is a voxel- and time-aligned drop-in for the input.

Two ways to call it — identical work either way:

    # Python
    from joint.puppetmaster import run
    run("/srv/fMRI-data/sub-13_ses-16_task-PainMovie_dir-pa_bold.nii.gz",
        "/tmp/sub-13_painmovie_pred.nii.gz")

    # CLI
    python -m joint.puppetmaster \
        --input  /srv/fMRI-data/sub-13_ses-16_task-PainMovie_dir-pa_bold.nii.gz \
        --output /tmp/sub-13_painmovie_pred.nii.gz

What this is (and isn't): a ROUND-TRIP on existing IBC data — model(degrade(HR)) —
not an enhancer for arbitrary new low-res scans. The model only knows our synthetic
degradation, so the input must be a full-resolution IBC run. It also must be one of
the runs in manifest_big: that is how its per-run ``norm_ref`` (and expected shape)
is looked up. To serve a NEW run, first add it to a manifest with a mask + norm_ref
(compute_metadata.py), then point this endpoint at that manifest.

Noise is fresh and random on every call (independent per timepoint), so reruns are
NOT bit-identical — that is intentional (mimics real thermal noise).

Hardcoded shared locations (exist on the project VM):
  weights  : /srv/venvs/team4dbrain/joint_model/best.pt           (joint03, the 4M model)
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
from data.degradation_spatial import make_spatial_degradation
from data.reader import get_reader

from .eval import load_checkpoint

# --- hardcoded shared locations on the VM -----------------------------------
WEIGHTS_PATH = "/srv/venvs/team4dbrain/joint_model/best.pt"
MANIFEST_PATH = "/srv/venvs/team4dbrain/derivatives/manifest_big.json"


def _find_run(manifest_path: str, input_path: str) -> dict:
    """Match an input bold file to its manifest run entry (for norm_ref + shape).

    The manifest stores each run's bold as '<run_id>_bold.nii.gz'. We match on the
    input's basename first, then fall back to the run_id derived from it. Raises a
    clear error unless exactly one run matches — this endpoint only serves the runs
    that are in the manifest (that is where norm_ref comes from).
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
            f"(need exactly 1). This endpoint only serves the runs already in the "
            f"manifest, because it needs their per-run norm_ref. To run a new file, "
            f"add it to a manifest with a mask + norm_ref (compute_metadata.py) first."
        )
    return matches[0]


@torch.no_grad()
def run(input_path: str, output_path: str) -> str:
    """Forward-pass a whole BOLD run through the joint model and save the result.

    Args:
        input_path:  a 4D full-resolution BOLD run (X,Y,Z,T) that is one of the
                     runs in manifest_big.
        output_path: where to write the reconstructed 4D NIfTI (use a .nii.gz path).

    Returns:
        output_path (for convenience / chaining).

    The output is denormalized to physical units and carries the input's affine and
    header (voxel sizes + TR), so it overlays the original voxel- and time-aligned.
    Each timepoint is degraded with fresh random Rician noise.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input run not found: {input_path}")

    # 1. model (+ the exact training config it was saved with) from shared weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(WEIGHTS_PATH, device)

    # 2. locate this run in the manifest -> per-run norm_ref + expected HR shape
    entry = _find_run(MANIFEST_PATH, input_path)
    norm_ref = float(entry["norm_ref"])
    run_id = entry["run_id"]

    # 3. lazy reader over the input + a header-only load for the output geometry
    reader = get_reader(input_path)
    hr_shape = tuple(int(s) for s in cfg.model.out_size)   # (128, 128, 93)
    if reader.shape3d != hr_shape:
        raise ValueError(
            f"input spatial shape {reader.shape3d} != model HR shape {hr_shape}. "
            f"This endpoint expects full-resolution IBC runs."
        )
    T = reader.n_volumes
    in_img = nib.load(input_path)   # header only, no decompression

    # 4. rebuild the EXACT training degradation; fresh random noise each call
    degrade = Compose([
        make_spatial_degradation(
            source_voxel_mm=cfg.train.source_voxel_mm,
            target_voxel_mm=cfg.train.target_voxel_mm,
        ),
        RicianNoise(
            sigma_min=cfg.train.sigma_min,
            sigma_max=cfg.train.sigma_max,
            seed=None,   # None => fresh entropy per call => independent noise per timepoint
        ),
    ])

    # 5. per-timepoint: normalize -> degrade -> forward -> denormalize -> stack
    print(f"[puppetmaster] {run_id}: {T} volumes | norm_ref={norm_ref:.1f} | "
          f"device={device.type} | weights={os.path.basename(WEIGHTS_PATH)}", flush=True)
    X, Y, Z = hr_shape
    out = np.zeros((X, Y, Z, T), dtype=np.float32)
    for t in range(T):
        hr = reader.read_volume(t).astype(np.float32) / norm_ref   # normalized HR volume
        lr = degrade(hr)                                           # noisy LR (64,64,46)
        x = torch.from_numpy(np.ascontiguousarray(lr)).float()[None, None].to(device)
        pred = model(x)[0, 0].cpu().numpy()                        # normalized HR prediction
        out[..., t] = pred * norm_ref                              # back to physical units
        if (t + 1) % 50 == 0 or t + 1 == T:
            print(f"[puppetmaster]   {t + 1}/{T}", flush=True)

    # 6. write a faithful drop-in: input affine + voxel sizes + TR, float32 data
    out_img = nib.Nifti1Image(out, in_img.affine)
    out_img.header.set_zooms(in_img.header.get_zooms())            # voxel sizes + TR
    try:
        out_img.header.set_xyzt_units(*in_img.header.get_xyzt_units())
    except Exception:
        pass
    out_img.header.set_data_dtype(np.float32)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    nib.save(out_img, output_path)
    print(f"[puppetmaster] saved {run_id} -> {output_path}  shape={out.shape}", flush=True)
    return output_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Forward-pass a whole BOLD run through the joint denoise+SR model "
                    "(degrade -> model -> stacked 4D output). Input must be a run in "
                    "manifest_big."
    )
    ap.add_argument("--input", "-i", required=True,
                    help="input 4D BOLD run (.nii.gz) — one of the manifest_big runs")
    ap.add_argument("--output", "-o", required=True,
                    help="output path for the reconstructed 4D run (.nii.gz)")
    args = ap.parse_args(argv)
    run(args.input, args.output)


if __name__ == "__main__":
    main()
