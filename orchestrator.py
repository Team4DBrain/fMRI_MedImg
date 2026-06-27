#!/usr/bin/env python
"""orchestrator.py — modular fMRI restoration pipeline harness.

Sits at the repo ROOT (next to joint/, data/, sr/, Denoising/, data_interpolation/)
and drives the four model endpoints in configurable combinations so you can
compare pipelines (e.g. the joint model vs. a denoise+SR cascade) on the SAME
input, with the SAME degradation, and score them with the SAME metrics.

It runs the helper code (degradation, normalization, masking, metrics) IN-PROCESS
from the `data`/`joint` packages, and calls each MODEL endpoint as a SUBPROCESS in
its own working directory (because the endpoints have conflicting import roots).

--------------------------------------------------------------------------------
USAGE (from the repo root, with the env active):

    python orchestrator.py \
        --input  /srv/fMRI-data/sub-13_ses-16_task-PainMovie_dir-pa_bold.nii.gz \
        --output runs/painmovie_cascade \
        --steps denoise sr

    python orchestrator.py -i <run> -o runs/painmovie_joint --steps joint
    python orchestrator.py -i <run> -o runs/quick --steps joint --truncate 10 --seed 0

See ORCHESTRATOR_README.md for the full guide.

--------------------------------------------------------------------------------
TWO ARCHITECTURES (--degrade-once):

  yes (default) — Architecture A: degrade the input ONCE and feed that identical
      run to the chosen stages. Degradation is conditional on the steps:
          spatial (HR->LR, k-space) iff `sr` or `joint` is in --steps
          noise   (Rician)          iff `denoise` or `joint` is in --steps
      applied spatial-then-noise. The fair comparison. joint/sr run LR-native.

  no — Architecture B (black-box chain): no orchestrator degradation; the raw input
      is fed to the first step and each endpoint does whatever it does natively
      (joint/sr self-degrade as a round-trip). Cascades double-degrade — a baseline
      for contrast, not a fair comparison.

NOTE: empty --steps is an IDENTITY pass (no steps, no degradation; final == the
clean reference) — useful only to sanity-check the eval harness, NOT a degraded
baseline (degradation is gated on which steps are present).

--------------------------------------------------------------------------------
NORMALIZATION (one scale per run = 98th pct of the brain temporal-mean):
  resolved from manifest_big for known runs, else computed (mask -> mean -> p98).
  It is a WHOLE-run quantity (truncation does not change it). Passed explicitly to
  joint (--norm-ref) and sr (infer_nifti norm_ref=) so both use the identical,
  training-faithful scale. denoise self-normalizes internally (colleague code) but
  denormalizes to physical units, so its output is still comparable.

DEGRADATION PARAMS (sigmas, voxel sizes, HR shape) are read from the joint
checkpoint config at runtime so the orchestrator can never drift from training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

# This file lives at the repo root; data/, joint/, sr/, ... are siblings.
REPO_ROOT = Path(__file__).resolve().parent

from data.degradation_spatial import make_spatial_degradation, voxel_size_to_target_shape
from data.degradation_noise import RicianNoise
from data.normalize import compute_norm_ref
from data.masks import compute_brain_mask
from data.reader import get_reader

# --------------------------------------------------------------------------
# Shared VM locations + DEFAULT degradation constants (overridden at runtime
# from the joint checkpoint config — see resolve_degradation_params()).
# --------------------------------------------------------------------------
WEIGHTS_PATH = "/srv/venvs/team4dbrain/joint_model/best.pt"
MANIFEST_PATH = Path("/srv/venvs/team4dbrain/derivatives/manifest_big.json")
DERIV_DIR = Path("/srv/venvs/team4dbrain/derivatives")

DEFAULT_SIGMA_MIN = 0.03
DEFAULT_SIGMA_MAX = 0.10
DEFAULT_SOURCE_MM = 1.5
DEFAULT_TARGET_MM = 3.0
DEFAULT_HR_SHAPE = (128, 128, 93)

VALID_STEPS = ("denoise", "sr", "joint", "interp")
SPATIAL_STEPS = {"sr", "joint"}      # reconstruct spatial resolution -> need spatial degrade
NOISE_STEPS = {"denoise", "joint"}   # remove noise                   -> need noise degrade

STEP_TIMEOUT_S = 3600                 # generous per-endpoint subprocess cap


# ==========================================================================
# degradation params (from the joint checkpoint, so they match training)
# ==========================================================================
def resolve_degradation_params(weights_path: str) -> dict:
    """Read sigma/voxel/HR-shape from the joint checkpoint config so the
    orchestrator's one-shot degradation matches what the models trained on.
    Falls back to the module defaults (with a warning) if it can't be read."""
    p = dict(sigma_min=DEFAULT_SIGMA_MIN, sigma_max=DEFAULT_SIGMA_MAX,
             source_mm=DEFAULT_SOURCE_MM, target_mm=DEFAULT_TARGET_MM,
             hr_shape=DEFAULT_HR_SHAPE, source="defaults")
    try:
        import torch
        ck = torch.load(weights_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        tr, mo = cfg["train"], cfg["model"]
        p.update(sigma_min=float(tr["sigma_min"]), sigma_max=float(tr["sigma_max"]),
                 source_mm=float(tr["source_voxel_mm"]), target_mm=float(tr["target_voxel_mm"]),
                 hr_shape=tuple(int(s) for s in mo["out_size"]), source="joint_checkpoint")
    except Exception as exc:
        print(f"[orch] WARNING: could not read degradation params from {weights_path} "
              f"({exc}); using defaults {p}")
    p["lr_shape"] = tuple(int(s) for s in voxel_size_to_target_shape(
        p["hr_shape"], p["source_mm"], p["target_mm"]))
    return p


# ==========================================================================
# norm_ref + brain mask resolution
# ==========================================================================
def resolve_norm_ref_and_mask(input_path: Path):
    """Return (norm_ref: float, mask (X,Y,Z) bool, source: str).

    Manifest runs (matched by EXACT bold-basename or EXACT run_id stem — never a
    prefix) reuse the stored norm_ref + precomputed mask. Otherwise compute from
    the run: temporal mean -> brain mask (SynthStrip via data.masks, auto-falls
    back to percentile) -> 98th-pct brain norm_ref. norm_ref is a WHOLE-run value.
    """
    in_base = os.path.basename(str(input_path))
    stem = in_base
    for suf in ("_bold.nii.gz", ".nii.gz", "_bold.nii", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    if MANIFEST_PATH.is_file():
        runs = json.loads(MANIFEST_PATH.read_text()).get("runs", [])
        matches = [r for r in runs
                   if os.path.basename(r.get("path", "")) == in_base or r.get("run_id") == stem]
        if len(matches) == 1:
            r = matches[0]
            mp = r.get("mask_path")
            if "norm_ref" in r and mp:
                mask_abs = DERIV_DIR / mp
                if mask_abs.is_file():               # any missing key / absent mask -> compute
                    mask = np.asarray(nib.load(str(mask_abs)).dataobj).astype(bool)
                    return float(r["norm_ref"]), mask, f"manifest:{r.get('run_id', stem)}"
        elif len(matches) > 1:
            print(f"[orch] WARNING: {len(matches)} manifest runs matched {in_base!r}; "
                  f"computing norm_ref/mask instead of guessing")

    reader = get_reader(str(input_path))
    mean = reader.read_mean()                                  # (X,Y,Z) float32
    affine = nib.load(str(input_path)).affine
    mask = compute_brain_mask(mean, affine=affine, method="auto")
    return float(compute_norm_ref(mean, mask)), mask, "computed"


# ==========================================================================
# truncation
# ==========================================================================
def load_reference(input_path: Path, truncate: int, seed: int):
    """Load the (optionally truncated) input as the clean-HR reference.

    Returns (ref_data (X,Y,Z,T) float32, affine, header, start, T). Truncation
    takes `truncate` consecutive frames from a random valid start.
    """
    img = nib.load(str(input_path))
    if img.ndim != 4:
        raise ValueError(f"expected a 4D BOLD run, got shape {img.shape}")
    T_full = int(img.shape[3])

    if not truncate or truncate >= T_full:
        if truncate and truncate >= T_full:
            print(f"[orch] --truncate {truncate} >= run length {T_full}; using whole run")
        start, T = 0, T_full
    else:
        T = int(truncate)
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, T_full - T + 1))
        print(f"[orch] truncating to {T} frames from start={start} (seed={seed})")

    data = np.asarray(img.dataobj[..., start:start + T], dtype=np.float32)
    if data.shape[3] < 1:
        raise ValueError(f"truncation produced 0 frames (truncate={truncate}, T_full={T_full})")
    return data, img.affine, img.header, start, T


# ==========================================================================
# degradation (Architecture A)
# ==========================================================================
def degrade_once(ref_data: np.ndarray, norm_ref: float, spatial: bool,
                 noise: bool, seed: int, params: dict):
    """Degrade the reference ONCE (conditional spatial + noise), per-timepoint seeded.

    Operates in normalized units (the regime sigmas are defined in), then returns to
    physical units. Per-timepoint noise is independent across time yet reproducible
    from `seed`. Returns (degraded (X,Y,Z,T) float32, out_shape3d).
    """
    T = ref_data.shape[3]
    spatial_fn = make_spatial_degradation(
        source_voxel_mm=params["source_mm"], target_voxel_mm=params["target_mm"]
    ) if spatial else None
    out_shape = params["lr_shape"] if spatial else params["hr_shape"]
    out = np.zeros((*out_shape, T), dtype=np.float32)
    for t in range(T):
        vol = ref_data[..., t] / norm_ref                     # normalized HR
        if spatial_fn is not None:
            vol = spatial_fn(vol)                             # -> normalized LR
        if noise:
            vol = RicianNoise(sigma_min=params["sigma_min"], sigma_max=params["sigma_max"],
                              seed=seed + t)(vol)             # reproducible, independent per t
        out[..., t] = (vol * norm_ref).astype(np.float32)     # back to physical
        if (t + 1) % 50 == 0 or t + 1 == T:
            print(f"[orch] degrade {t + 1}/{T}", flush=True)
    return out, out_shape


def _affine_for_shape(ref_affine: np.ndarray, shape3d, hr_shape) -> np.ndarray:
    """Affine for a (possibly downsampled) grid derived from the HR reference affine.

    FOV-preserving per-axis voxel scaling (HR_N / N). This is spatial metadata only
    (metrics use the arrays + the HR mask); the origin's half-voxel central-crop
    shift is not modeled, so the LR affine is approximate for viewing/overlay.
    """
    aff = ref_affine.copy()
    if tuple(shape3d) != tuple(hr_shape):
        scale = np.array([hr_shape[i] / shape3d[i] for i in range(3)], dtype=float)
        aff[:3, :3] = ref_affine[:3, :3] * scale[np.newaxis, :]
    return aff


def save_run(data: np.ndarray, affine: np.ndarray, ref_header, path: Path):
    """Write a 4D run, carrying the reference TR (pixdim[4]) into the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    zooms = [float(z) for z in nib.affines.voxel_sizes(affine)]
    try:
        rz = ref_header.get_zooms()
        tr = float(rz[3]) if len(rz) > 3 else 0.0
    except Exception:
        tr = 0.0
    img.header.set_zooms((*zooms, tr) if tr > 0 else tuple(zooms))
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


# ==========================================================================
# endpoint invocation (subprocess, per-endpoint cwd) + soft check
# ==========================================================================
def resolve_sr_checkpoint(sr_model: str) -> str:
    cands = sorted((REPO_ROOT / "models").glob(f"sr_{sr_model}_*_best.pt"))
    if not cands:
        raise FileNotFoundError(
            f"no SR checkpoint at {REPO_ROOT}/models/sr_{sr_model}_*_best.pt")
    return str(cands[-1])     # latest by timestamped name


def build_step_command(name: str, in_path: Path, out_path: Path, *,
                       norm_ref: float, sr_model: str, interp_mode: str):
    """Return (cmd_list, cwd) for one endpoint. Paths are absolute."""
    ip, op = str(in_path.resolve()), str(out_path.resolve())
    if name == "joint":
        return ([sys.executable, "-m", "joint.puppetmaster",
                 "-i", ip, "-o", op, "--norm-ref", repr(float(norm_ref))],
                REPO_ROOT)
    if name == "sr":
        ckpt = resolve_sr_checkpoint(sr_model)
        # call infer_nifti directly so we can inject the exact norm_ref (the CLI
        # doesn't expose it); also where the sr-wiring soft-check lands.
        code = (f"from sr.infer import infer_nifti; "
                f"infer_nifti({ckpt!r}, {ip!r}, {op!r}, norm_ref={float(norm_ref)!r})")
        return ([sys.executable, "-c", code], REPO_ROOT)
    if name == "denoise":
        code = (f"from pipeline_api import denoise_run; "
                f"denoise_run({ip!r}, {op!r})")
        return ([sys.executable, "-c", code], REPO_ROOT / "Denoising")
    if name == "interp":
        return ([sys.executable, "main.py",
                 "--weights", "checkpoints/pretrained/model_weights.pt",
                 "--input", ip, "--output", op, "--mode", interp_mode],
                REPO_ROOT / "data_interpolation")
    raise ValueError(f"unknown step {name!r}")


def run_step(name: str, in_path: Path, out_path: Path, **kw):
    """Run one endpoint as a subprocess. Returns (ok, detail). Never raises.

    Pre-deletes the output so `out_path.exists()` means THIS run produced it (no
    stale-success). Always writes the full stdout/stderr to a per-step .log.
    """
    try:
        cmd, cwd = build_step_command(name, in_path, out_path, **kw)
    except Exception as exc:
        return False, f"command build failed: {exc}"

    try:
        out_path.unlink()                       # ensure existence == freshness
    except FileNotFoundError:
        pass
    log_path = out_path.parent / (out_path.name.split(".nii")[0] + ".log")

    print(f"[orch] step '{name}'  (cwd={cwd})", flush=True)
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=STEP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            log_path.write_text(f"$ {' '.join(cmd)}\n\nTIMEOUT after {STEP_TIMEOUT_S}s")
        except Exception:
            pass
        return False, f"timed out after {STEP_TIMEOUT_S}s (log: {log_path})"
    except Exception as exc:
        return False, f"subprocess error: {exc}"

    try:
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    except Exception:
        pass

    if r.returncode != 0 or not out_path.exists():
        tail = (r.stderr or r.stdout or "").strip()[-1500:]
        return False, (f"exit={r.returncode}, output_present={out_path.exists()} "
                       f"(full log: {log_path})\n--- stderr tail ---\n{tail}")
    return True, "ok"


# ==========================================================================
# evaluation
# ==========================================================================
def _masked_psnr_ssim(final_path: Path, ref_data: np.ndarray, mask: np.ndarray,
                      norm_ref: float):
    """Per-timepoint masked PSNR + SSIM in NORMALIZED units (peak 1.0), averaged.

    Reuses the joint model's metric implementations so numbers match training/eval.
    Raises ValueError on a shape mismatch (caller turns that into a skipped metric).
    """
    import torch
    from joint.losses import masked_psnr, masked_ssim_3d

    fr = get_reader(str(final_path))
    if tuple(fr.shape3d) != tuple(mask.shape):
        raise ValueError(f"output shape {fr.shape3d} != mask shape {mask.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    T = min(fr.n_volumes, ref_data.shape[3])
    psnrs, ssims = [], []
    for t in range(T):
        fin = fr.read_volume(t).astype(np.float32) / norm_ref
        ref = ref_data[..., t] / norm_ref
        p = torch.from_numpy(np.ascontiguousarray(fin))[None, None].to(device)
        g = torch.from_numpy(np.ascontiguousarray(ref))[None, None].to(device)
        psnrs.append(float(masked_psnr(p, g, m)))
        ssims.append(float(masked_ssim_3d(p, g, m)))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def _tsnr_in_brain(run_path: Path, mask: np.ndarray) -> float:
    """In-brain mean tSNR (temporal mean / temporal std), streamed over time.

    The std floor is scale-relative (a fraction of the in-brain median signal) and
    per-voxel tSNR is capped, so a temporally-flat voxel can't blow up the average.
    Raises ValueError on a shape mismatch (caller turns that into a skipped metric).
    """
    r = get_reader(str(run_path))
    if tuple(r.shape3d) != tuple(mask.shape):
        raise ValueError(f"run shape {r.shape3d} != mask shape {mask.shape}")
    T = r.n_volumes
    s = ss = None
    for t in range(T):
        v = r.read_volume(t).astype(np.float64)
        s = v.copy() if s is None else s + v
        ss = v * v if ss is None else ss + v * v
    mean = s / T
    std = np.sqrt(np.maximum(ss / T - mean * mean, 0.0))
    sig = float(np.median(mean[mask]))
    std_floor = max(1e-3 * abs(sig), 1e-6)
    tsnr = mean / np.maximum(std, std_floor)
    return float(np.mean(np.clip(tsnr[mask], 0.0, 1000.0)))


def _interp_leaveout(ref_path: Path, work_dir: Path) -> dict:
    """interp's own eval.py on the reference (predict held-out interior frames,
    dropping the unpredictable ends). PSNR/L1 only (no SSIM)."""
    out_dir = work_dir / "interp_eval"
    cmd = [sys.executable, "eval.py",
           "--weights", "checkpoints/pretrained/model_weights.pt",
           "--file", str(ref_path.resolve()),
           "--output-dir", str(out_dir.resolve())]
    try:
        r = subprocess.run(cmd, cwd=str(REPO_ROOT / "data_interpolation"),
                           capture_output=True, text=True, timeout=STEP_TIMEOUT_S)
        summary = out_dir / "metrics.json"
        if r.returncode == 0 and summary.is_file():
            return json.loads(summary.read_text()).get("summary", {})
        return {"error": (r.stderr or r.stdout or "").strip()[-800:]}
    except Exception as exc:
        return {"error": str(exc)}


def evaluate(final_path: Path, ref_path: Path, ref_data: np.ndarray, mask: np.ndarray,
             norm_ref: float, steps: list[str], work_dir: Path) -> dict:
    """Metrics dict. Each metric is guarded independently so one failure (e.g. a
    shape mismatch from a partially-failed pipeline) never wipes out the rest."""
    metrics: dict = {"notes": []}
    try:
        final_T = get_reader(str(final_path)).n_volumes
    except Exception as exc:
        return {"error": f"could not open final output {final_path}: {exc}"}
    ref_T = int(ref_data.shape[3])
    metrics["reference_timepoints"] = ref_T
    metrics["output_timepoints"] = final_T

    # tSNR — each side guarded so the reference value survives an output mismatch.
    for key, path in (("tsnr_reference", ref_path), ("tsnr_output", final_path)):
        try:
            metrics[key] = _tsnr_in_brain(path, mask)
        except Exception as exc:
            metrics[key] = None
            metrics["notes"].append(f"{key} skipped: {exc}")
    if metrics.get("tsnr_output") and metrics.get("tsnr_reference"):
        metrics["tsnr_ratio"] = metrics["tsnr_output"] / metrics["tsnr_reference"]
    else:
        metrics["tsnr_ratio"] = None

    # PSNR/SSIM only when the time axis is unchanged.
    if final_T == ref_T:
        try:
            metrics["psnr_db"], metrics["ssim"] = _masked_psnr_ssim(
                final_path, ref_data, mask, norm_ref)
        except Exception as exc:
            metrics["psnr_db"] = metrics["ssim"] = None
            metrics["notes"].append(f"PSNR/SSIM skipped: {exc}")
    else:
        metrics["psnr_db"] = metrics["ssim"] = None
        metrics["notes"].append(
            "time axis changed (interp); per-frame PSNR/SSIM is undefined "
            "(synthetic frames have no aligned ground truth). tSNR reported; "
            "interp leave-out below if applicable.")
        if "interp" in steps:
            metrics["interp_leaveout"] = _interp_leaveout(ref_path, work_dir)
    return metrics


# ==========================================================================
# slides
# ==========================================================================
def make_slides(ref_data: np.ndarray, final_path: Path, slides_dir: Path,
                degraded_path: Path | None = None, n_times: int = 3):
    """Reference-vs-output montages at a few timepoints x a few axial slices.

    A `degraded input` column is shown only when a degraded run was actually
    written (Architecture A with a spatial/noise step)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[orch] slides skipped (matplotlib unavailable: {exc})")
        return
    slides_dir.mkdir(parents=True, exist_ok=True)
    fr = get_reader(str(final_path))
    ref_T, fin_T = ref_data.shape[3], fr.n_volumes
    Z = ref_data.shape[2]
    z_levels = [int(Z * f) for f in (0.35, 0.5, 0.65)]
    t_idxs = np.linspace(0, min(ref_T, fin_T) - 1, n_times).astype(int)

    have_deg = degraded_path is not None and Path(degraded_path).is_file()
    dr = get_reader(str(degraded_path)) if have_deg else None
    cols = 3 if have_deg else 2
    col_titles = (["degraded input", "pipeline output", "reference (clean)"]
                  if have_deg else ["pipeline output", "reference (clean)"])

    for ti in t_idxs:
        ref_vol = ref_data[..., ti]
        fin_vol = fr.read_volume(int(ti)).astype(np.float32)
        deg_vol = dr.read_volume(int(ti)).astype(np.float32) if have_deg else None
        vmax = float(np.percentile(ref_vol, 99.5)) or 1.0
        fig, ax = plt.subplots(len(z_levels), cols, figsize=(3.2 * cols, 3.2 * len(z_levels)))
        ax = np.atleast_2d(ax)
        panels_for = (lambda: [deg_vol, fin_vol, ref_vol]) if have_deg else (lambda: [fin_vol, ref_vol])
        for r, z in enumerate(z_levels):
            for c, vol in enumerate(panels_for()):
                zz = min(z, vol.shape[2] - 1)
                ax[r, c].imshow(vol[:, :, zz].T, origin="lower", cmap="gray",
                                vmin=0, vmax=vmax, interpolation="nearest")
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                if r == 0:
                    ax[r, c].set_title(col_titles[c], fontsize=10)
                if c == 0:
                    ax[r, c].set_ylabel(f"z={z}", fontsize=9)
        fig.suptitle(f"t={int(ti)}", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(slides_dir / f"t{int(ti):03d}.png"), dpi=110)
        plt.close(fig)
    print(f"[orch] slides -> {slides_dir} ({len(t_idxs)} montages)")


# ==========================================================================
# main
# ==========================================================================
def _nonneg_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Modular fMRI restoration pipeline harness (joint vs cascade etc.).")
    ap.add_argument("--input", "-i", required=True, help="input 4D BOLD run (.nii.gz)")
    ap.add_argument("--output", "-o", required=True, help="output DIRECTORY")
    ap.add_argument("--steps", nargs="*", default=[], choices=VALID_STEPS,
                    help="ordered endpoint steps, e.g. --steps denoise sr (repeat to run "
                         "twice; empty = identity passthrough, no degradation)")
    ap.add_argument("--degrade-once", choices=["yes", "no"], default="yes",
                    help="yes = Architecture A (degrade once, fair); no = B (black-box chain)")
    ap.add_argument("--truncate", type=_nonneg_int, default=0,
                    help="take N consecutive frames from a random start (0 = whole run)")
    ap.add_argument("--seed", type=_nonneg_int, default=0, help="seed (>=0) for truncation + degradation noise")
    ap.add_argument("--sr-model", default="rcan3d", help="SR model key (default rcan3d)")
    ap.add_argument("--interp-mode", choices=["insert", "fill-gaps"], default="fill-gaps",
                    help="interp output mode (default fill-gaps)")
    ap.add_argument("--keep-intermediates", choices=["yes", "no"], default="yes")
    args = ap.parse_args(argv)

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    steps = list(args.steps)
    degrade_a = (args.degrade_once == "yes")

    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")

    print(f"[orch] input={input_path}\n[orch] steps={steps or '(identity passthrough)'} "
          f"degrade_once={args.degrade_once} truncate={args.truncate} seed={args.seed}")

    # 0) degradation params (from the joint checkpoint -> can't drift from training)
    params = resolve_degradation_params(WEIGHTS_PATH)
    hr_shape, lr_shape = params["hr_shape"], params["lr_shape"]
    print(f"[orch] degradation params ({params['source']}): "
          f"sigma=({params['sigma_min']},{params['sigma_max']}) "
          f"{params['source_mm']}->{params['target_mm']}mm  HR={hr_shape} LR={lr_shape}")

    # 1) norm_ref + brain mask
    norm_ref, mask, nr_src = resolve_norm_ref_and_mask(input_path)
    print(f"[orch] norm_ref={norm_ref:.2f} ({nr_src}) | mask_fraction={mask.mean():.3f}")
    if tuple(mask.shape) != tuple(hr_shape):
        print(f"[orch] WARNING: mask shape {mask.shape} != HR {hr_shape}; "
              f"masked metrics may be skipped")

    # 2) reference (optionally truncated) = the clean-HR ground truth
    ref_data, ref_affine, ref_header, start, T = load_reference(input_path, args.truncate, args.seed)
    if tuple(ref_data.shape[:3]) != tuple(hr_shape):
        raise ValueError(f"input spatial shape {ref_data.shape[:3]} != expected HR {hr_shape}")
    ref_path = work / "reference.nii.gz"
    save_run(ref_data, ref_affine, ref_header, ref_path)

    # 3) build the starting run for the pipeline
    degraded_path = None
    if degrade_a:
        spatial = bool(SPATIAL_STEPS & set(steps))
        noise = bool(NOISE_STEPS & set(steps))
        print(f"[orch] degrade-once: spatial={spatial} noise={noise}")
        if spatial or noise:
            deg, deg_shape = degrade_once(ref_data, norm_ref, spatial, noise, args.seed, params)
            degraded_path = work / "degraded.nii.gz"
            save_run(deg, _affine_for_shape(ref_affine, deg_shape, hr_shape), ref_header, degraded_path)
            cur = degraded_path
        else:
            cur = ref_path           # no spatial/noise steps -> nothing to degrade
    else:
        cur = ref_path               # Architecture B: feed the raw reference

    # 4) run the steps in order, chaining output -> input
    failures = []
    for i, name in enumerate(steps):
        out_path = work / f"step{i:02d}_{name}.nii.gz"
        ok, detail = run_step(name, cur, out_path, norm_ref=norm_ref,
                              sr_model=args.sr_model, interp_mode=args.interp_mode)
        if not ok:
            failures.append({"step_index": i, "step": name, "detail": detail})
            print(f"[orch] STEP FAILED: {name}\n{detail}")
            break
        cur = out_path

    # 5) finalize. On failure the artifact is named *_INCOMPLETE so it can't be
    #    mistaken for a successful result.
    incomplete = bool(failures)
    for stale in ("final.nii.gz", "final_INCOMPLETE.nii.gz"):
        (out_dir / stale).unlink(missing_ok=True)
    final_name = "final_INCOMPLETE.nii.gz" if incomplete else "final.nii.gz"
    final_path = out_dir / final_name
    shutil.copyfile(cur, final_path)

    # 6) evaluate + slides (best-effort; never crash the run over a metric/figure)
    try:
        metrics = evaluate(final_path, ref_path, ref_data, mask, norm_ref, steps, work)
    except Exception as exc:
        metrics = {"error": f"evaluation failed: {exc}"}
    metrics["final_output"] = final_name
    metrics["final_produced_by"] = (f"step{len(steps) - 1}" if steps and not incomplete
                                    else ("degrade" if (incomplete and degraded_path and cur == degraded_path)
                                          else "identity/reference" if cur == ref_path else str(cur.name)))
    if failures:
        metrics["pipeline_failures"] = failures
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        make_slides(ref_data, final_path, out_dir / "slides", degraded_path)
    except Exception as exc:
        print(f"[orch] slides failed: {exc}")

    # 7) provenance
    (out_dir / "run_config.json").write_text(json.dumps({
        "input": str(input_path), "steps": steps, "degrade_once": args.degrade_once,
        "truncate": args.truncate, "truncate_start": start, "timepoints": T,
        "seed": args.seed, "sr_model": args.sr_model, "interp_mode": args.interp_mode,
        "norm_ref": norm_ref, "norm_ref_source": nr_src, "mask_fraction": float(mask.mean()),
        "degradation_params": params, "incomplete": incomplete,
    }, indent=2))

    if args.keep_intermediates == "no":
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n[orch] DONE -> {out_dir}")
    print(f"[orch] metrics: { {k: v for k, v in metrics.items() if k != 'notes'} }")
    if failures:
        print("[orch] NOTE: pipeline had failures (see metrics.json); "
              f"artifact written as {final_name}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
