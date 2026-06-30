"""nb_utils.py — shared helpers for the presentation notebooks.

Keeps the notebooks themselves thin: all the model wiring, the midterm-style
figures (tri-planar Ground-truth / Prediction / |Error|, error-localization MIPs)
and the orchestrator pipeline runner live here.

Everything is import-light at module load; heavy deps (torch) are imported inside
the functions that need them, so a notebook can `import nb_utils` even on a box
that hasn't activated the env yet.

Designed for the temporal-interpolation model (data_interpolation/), but the
pipeline section drives the repo-root orchestrator.py for any combination of
the four endpoints (joint / sr / denoise / interp).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# paths — notebooks/ lives at the repo root, next to orchestrator.py
# --------------------------------------------------------------------------
NB_DIR = Path(__file__).resolve().parent
REPO_ROOT = NB_DIR.parent
INTERP_DIR = REPO_ROOT / "data_interpolation"


@dataclass
class Config:
    """Edit these once per machine (in the first notebook cell) and pass around.

    On the VM the defaults below should mostly be right; only `data_dir` /
    `bold_file` usually need pointing at a real run.
    """
    # where the 4D BOLD runs live (VM mount)
    data_dir: Path = Path("/srv/fMRI-data")
    # a single run to demo on; if None, the first file found in data_dir is used
    bold_file: Path | None = None

    # interpolation checkpoint (relative to data_interpolation/)
    interp_weights: Path = REPO_ROOT / "weights" / "temporal" / "model_weights.pt"
    interp_history: Path = REPO_ROOT / "weights" / "temporal" / "history.json"
    norm_mode: str = "zscore"          # must match training (pretrained = zscore)
    residual: bool = False             # pretrained = non-residual
    base_channels: int = 32
    depth: int = 4
    device: str | None = None          # None -> auto (cuda > mps > cpu)

    # where notebook runs/figures are written
    out_dir: Path = NB_DIR / "outputs"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def resolve_bold(self) -> Path:
        if self.bold_file is not None:
            return Path(self.bold_file)
        files = list_bold_files(self.data_dir)
        if not files:
            raise FileNotFoundError(
                f"no *_bold.nii.gz under {self.data_dir} — set cfg.bold_file explicitly")
        return files[0]


def list_bold_files(data_dir: str | Path) -> list[Path]:
    """All 4D BOLD runs under a directory (sorted)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*_bold.nii.gz"))


# --------------------------------------------------------------------------
# environment / data inspection
# --------------------------------------------------------------------------
def env_report() -> dict:
    """A quick dict of versions + device, for a sanity cell at the top."""
    info = {"python": sys.version.split()[0], "repo_root": str(REPO_ROOT)}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(getattr(torch.backends, "mps", None)
                                     and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:                       # noqa: BLE001
        info["torch"] = f"unavailable ({exc})"
    try:
        import nibabel
        info["nibabel"] = nibabel.__version__
    except Exception as exc:                       # noqa: BLE001
        info["nibabel"] = f"unavailable ({exc})"
    return info


def bold_info(path: str | Path) -> dict:
    """Shape / voxel size / TR of a BOLD run, header-only (no full load)."""
    import nibabel as nib
    img = nib.load(str(path))
    zooms = img.header.get_zooms()
    return {
        "file": os.path.basename(str(path)),
        "shape": tuple(int(s) for s in img.shape),
        "voxel_mm": tuple(round(float(z), 3) for z in zooms[:3]),
        "tr_s": float(zooms[3]) if len(zooms) > 3 else None,
        "dtype": str(img.get_data_dtype()),
    }


# --------------------------------------------------------------------------
# interpolation model
# --------------------------------------------------------------------------
def load_interpolator(cfg: Config):
    """Construct the FMRIInterpolator from the data_interpolation package."""
    if str(INTERP_DIR) not in sys.path:
        sys.path.insert(0, str(INTERP_DIR))
    from src.inference import FMRIInterpolator
    return FMRIInterpolator(
        str(cfg.interp_weights),
        norm_mode=cfg.norm_mode,
        device=cfg.device,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        residual=cfg.residual,
    )


@dataclass
class TripletResult:
    """One interpolation example, all volumes in NIfTI (X, Y, Z) physical units."""
    t: int
    v_prev: np.ndarray      # V_t
    v_next: np.ndarray      # V_{t+2}
    gt: np.ndarray          # V_{t+1}  (ground truth middle frame)
    pred: np.ndarray        # model prediction of V_{t+1}
    naive: np.ndarray       # 0.5 * (V_t + V_{t+2})  baseline
    meta: dict = field(default_factory=dict)

    @property
    def error(self) -> np.ndarray:
        return np.abs(self.gt - self.pred)

    @property
    def naive_error(self) -> np.ndarray:
        return np.abs(self.gt - self.naive)


def predict_triplet(cfg: Config, interp, path: str | Path, t: int) -> TripletResult:
    """Run the model on triplet (V_t, V_{t+1}, V_{t+2}) at time `t`.

    Reuses the package's dataset so normalization matches training/eval exactly,
    then denormalizes everything back to physical BOLD units for display/metrics.
    """
    import torch
    if str(INTERP_DIR) not in sys.path:
        sys.path.insert(0, str(INTERP_DIR))
    from src.dataset import FMRIInterpolationDataset

    ds = FMRIInterpolationDataset(file_list=[str(path)], norm_mode=cfg.norm_mode)
    # map a raw time index t to the flat sample index for this (single) file
    sample_idx = next((i for i, (_, tt) in enumerate(ds._index) if tt == t), None)
    if sample_idx is None:
        raise IndexError(f"t={t} out of range for {os.path.basename(str(path))} "
                         f"(valid t: 0..{len(ds) - 1})")
    s = ds[sample_idx]
    mu, sigma = s["stats"]

    x = s["x"].unsqueeze(0).to(interp.device)
    with torch.no_grad():
        raw = interp.model(x)
        pred_n = (0.5 * (x[:, 0:1] + x[:, 1:2]) + raw) if cfg.residual else raw
        naive_n = 0.5 * (x[:, 0:1] + x[:, 1:2])

    def denorm_xyz(t_dhw) -> np.ndarray:
        # (1,1,D,H,W) normalized -> (X,Y,Z) physical
        a = t_dhw.squeeze().cpu().numpy() * sigma + mu
        return np.ascontiguousarray(a.transpose(2, 1, 0))

    return TripletResult(
        t=t,
        v_prev=denorm_xyz(x[:, 0:1]),
        v_next=denorm_xyz(x[:, 1:2]),
        gt=denorm_xyz(s["y"].unsqueeze(0)),
        pred=denorm_xyz(pred_n),
        naive=denorm_xyz(naive_n),
        meta={"file": os.path.basename(str(path)), "mu": float(mu),
              "sigma": float(sigma), "n_triplets": len(ds)},
    )


# --------------------------------------------------------------------------
# metrics  (normalized units, matching data_interpolation/eval.py)
# --------------------------------------------------------------------------
def _psnr(mse: float, data_range: float) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def triplet_metrics(res: TripletResult, norm_mode: str = "zscore") -> dict:
    """L1 + PSNR for model and naive, computed in normalized units like eval.py."""
    sigma = res.meta.get("sigma", 1.0) or 1.0
    data_range = 1.0 if norm_mode == "percentile" else 2.0
    # back to normalized units so PSNR is comparable to the training/eval numbers
    g = (res.gt - res.meta.get("mu", 0.0)) / sigma
    p = (res.pred - res.meta.get("mu", 0.0)) / sigma
    n = (res.naive - res.meta.get("mu", 0.0)) / sigma
    m_mse, n_mse = float(((p - g) ** 2).mean()), float(((n - g) ** 2).mean())
    return {
        "t": res.t,
        "model_l1": float(np.abs(p - g).mean()),
        "naive_l1": float(np.abs(n - g).mean()),
        "model_psnr": _psnr(m_mse, data_range),
        "naive_psnr": _psnr(n_mse, data_range),
    }


def sweep_metrics(cfg: Config, interp, path: str | Path,
                  t_list: list[int] | None = None, stride: int = 1) -> "list[dict]":
    """Metrics across many t (for a curve / aggregate). Returns list of dicts."""
    import torch
    if str(INTERP_DIR) not in sys.path:
        sys.path.insert(0, str(INTERP_DIR))
    from src.dataset import FMRIInterpolationDataset
    ds = FMRIInterpolationDataset(file_list=[str(path)], norm_mode=cfg.norm_mode)
    data_range = 1.0 if cfg.norm_mode == "percentile" else 2.0
    if t_list is None:
        t_list = list(range(0, len(ds), max(1, stride)))
    rows = []
    with torch.no_grad():
        for idx in t_list:
            if idx >= len(ds):
                continue
            s = ds[idx]
            x = s["x"].unsqueeze(0).to(interp.device)
            y = s["y"].unsqueeze(0).to(interp.device)
            raw = interp.model(x)
            pred = (0.5 * (x[:, 0:1] + x[:, 1:2]) + raw) if cfg.residual else raw
            naive = 0.5 * (x[:, 0:1] + x[:, 1:2])
            m_mse = float(((pred - y) ** 2).mean())
            n_mse = float(((naive - y) ** 2).mean())
            rows.append({
                "t": int(s["t"]),
                "model_l1": float((pred - y).abs().mean()),
                "naive_l1": float((naive - y).abs().mean()),
                "model_psnr": _psnr(m_mse, data_range),
                "naive_psnr": _psnr(n_mse, data_range),
            })
    return rows


# --------------------------------------------------------------------------
# figures  (the midterm-style outputs)
# --------------------------------------------------------------------------
def _slices(vol: np.ndarray, frac=(0.5, 0.5, 0.5)):
    """Return (axial, coronal, sagittal) 2D slices, display-oriented."""
    X, Y, Z = vol.shape
    x, y, z = (int(X * frac[0]), int(Y * frac[1]), int(Z * frac[2]))
    axial = vol[:, :, z].T
    coronal = vol[:, y, :].T
    sagittal = vol[x, :, :].T
    return axial, coronal, sagittal


def triplanar_compare(res: TripletResult, *, frac=(0.5, 0.5, 0.5),
                      error_cmap="jet", save_path: str | Path | None = None,
                      title: str | None = None):
    """The midterm triptych: rows = axial/coronal/sagittal, cols = GT / Pred / |Error|."""
    import matplotlib.pyplot as plt

    gt_s = _slices(res.gt, frac)
    pr_s = _slices(res.pred, frac)
    er_s = _slices(res.error, frac)
    row_names = ["axial", "coronal", "sagittal"]

    vmax = float(np.percentile(res.gt, 99.5)) or 1.0
    emax = float(np.percentile(res.error, 99.0)) or 1.0

    fig, ax = plt.subplots(3, 3, figsize=(11, 11))
    col_titles = ["Ground truth  V_{t+1}", "Prediction  V_{t+1}", "|Error|"]
    for r in range(3):
        for c, (data, cmap, vlims) in enumerate((
            (gt_s[r], "gray", (0, vmax)),
            (pr_s[r], "gray", (0, vmax)),
            (er_s[r], error_cmap, (0, emax)),
        )):
            im = ax[r, c].imshow(data, origin="lower", cmap=cmap,
                                 vmin=vlims[0], vmax=vlims[1], interpolation="nearest")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(col_titles[c], fontsize=12, fontweight="bold")
            if c == 0:
                ax[r, c].set_ylabel(row_names[r], fontsize=11)
            if c == 2:
                fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.04)

    sup = title or (f"{res.meta.get('file', '')}  ·  t={res.t} -> t+1={res.t + 1}")
    m = triplet_metrics(res)
    sup += (f"\nmodel L1={m['model_l1']:.4f}  PSNR={m['model_psnr']:.2f} dB"
            f"   |   naive L1={m['naive_l1']:.4f}  PSNR={m['naive_psnr']:.2f} dB")
    fig.suptitle("Ground truth   vs   Prediction   vs   |Error|   (tri-planar)\n" + sup,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


def error_localization(res: TripletResult, *, anat_cmap="gray",
                       mip_cmap="hot", overlay_cmap="hot",
                       save_path: str | Path | None = None):
    """Where does the error live? Three MIP projections + best axial overlay.

    Mirrors the midterm 'Error localization' panel: max-intensity projections of
    the |error| volume over Z (top-down), Y (front) and X (side), plus the single
    worst axial slice with the error overlaid on the anatomy.
    """
    import matplotlib.pyplot as plt

    err = res.error
    mip_z = err.max(axis=2).T      # over Z -> top-down (X,Y)
    mip_y = err.max(axis=1).T      # over Y -> front  (X,Z)
    mip_x = err.max(axis=0).T      # over X -> side   (Y,Z)

    # worst axial slice by summed error
    z_best = int(err.sum(axis=(0, 1)).argmax())
    anat = res.gt[:, :, z_best].T
    err_slice = err[:, :, z_best].T

    emax = float(np.percentile(err, 99.5)) or 1.0
    vmax = float(np.percentile(res.gt, 99.5)) or 1.0

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    titles = ["MIP — max over Z\n(top-down)", "MIP — max over Y\n(front)",
              "MIP — max over X\n(side)", f"Best axial slice z={z_best}\nanat + error overlay"]
    for a, data, t in zip(ax[:3], (mip_z, mip_y, mip_x), titles[:3]):
        a.imshow(data, origin="lower", cmap=mip_cmap, vmin=0, vmax=emax,
                 interpolation="nearest")
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])

    ax[3].imshow(anat, origin="lower", cmap=anat_cmap, vmin=0, vmax=vmax,
                 interpolation="nearest")
    masked = np.ma.masked_less(err_slice, 0.15 * emax)
    ax[3].imshow(masked, origin="lower", cmap=overlay_cmap, vmin=0, vmax=emax,
                 alpha=0.6, interpolation="nearest")
    ax[3].set_title(titles[3], fontsize=10); ax[3].set_xticks([]); ax[3].set_yticks([])

    fig.suptitle(f"Error localization — {res.meta.get('file', '')}  ·  t={res.t}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


def show_loss_curve(history_path: str | Path, save_path: str | Path | None = None):
    """Training/val loss curve from a history.json."""
    import matplotlib.pyplot as plt
    history = json.loads(Path(history_path).read_text())
    epochs = [h["epoch"] for h in history]
    train = [h.get("train_loss") for h in history]
    val = [h.get("val_loss") for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, label="train", lw=2)
    if any(v is not None for v in val):
        ax.plot(epochs, val, label="val", lw=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=120)
    return fig


# --------------------------------------------------------------------------
# pipelines  (drive the repo-root orchestrator.py)
# --------------------------------------------------------------------------
def run_pipeline(steps: list[str], input_path: str | Path, out_dir: str | Path,
                 *, truncate: int = 0, seed: int = 0, noise: str = "auto",
                 degrade_once: str = "yes", sr_model: str = "rcan3d",
                 interp_mode: str = "fill-gaps", extra: list[str] | None = None,
                 echo: bool = True) -> dict:
    """Run one orchestrator pipeline and return its parsed metrics.json.

    `steps` is e.g. ["joint"] or ["denoise", "sr"] or ["interp"] or [] (identity).
    Returns a dict: {"steps", "out_dir", "returncode", "metrics", "run_config"}.
    Never raises on a pipeline failure — inspect ["returncode"]/["metrics"].
    """
    out_dir = Path(out_dir)
    cmd = [sys.executable, str(REPO_ROOT / "orchestrator.py"),
           "-i", str(Path(input_path).resolve()),
           "-o", str(out_dir.resolve()),
           "--degrade-once", degrade_once, "--noise", noise,
           "--seed", str(seed), "--sr-model", sr_model,
           "--interp-mode", interp_mode]
    if truncate:
        cmd += ["--truncate", str(truncate)]
    if steps:
        cmd += ["--steps", *steps]
    if extra:
        cmd += extra

    if echo:
        print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if echo:
        tail = (r.stdout or "").strip().splitlines()[-8:]
        print("\n".join(tail))
        if r.returncode != 0:
            print("--- stderr tail ---")
            print((r.stderr or "").strip()[-1200:])

    def _read(name):
        p = out_dir / name
        return json.loads(p.read_text()) if p.is_file() else None

    return {
        "steps": steps or ["identity"],
        "label": "+".join(steps) if steps else "identity",
        "out_dir": str(out_dir),
        "returncode": r.returncode,
        "metrics": _read("metrics.json"),
        "run_config": _read("run_config.json"),
    }


def metrics_table(results: list[dict]):
    """Tidy comparison table across pipeline results. Returns a pandas DataFrame."""
    import pandas as pd
    keys = ["psnr_db", "ssim", "tsnr_output", "tsnr_reference", "tsnr_ratio",
            "output_timepoints", "reference_timepoints"]
    rows = []
    for res in results:
        m = res.get("metrics") or {}
        row = {"pipeline": res["label"],
               "ok": res["returncode"] == 0 and not m.get("pipeline_failures")}
        for k in keys:
            row[k] = m.get(k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("pipeline")


def show_slides(out_dir: str | Path, max_n: int = 3):
    """Display the orchestrator's reference-vs-output montage PNGs inline."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    slides = sorted(Path(out_dir).glob("slides/*.png"))[:max_n]
    if not slides:
        print(f"no slides under {out_dir}/slides")
        return
    for s in slides:
        img = mpimg.imread(str(s))
        fig, ax = plt.subplots(figsize=(10, 10 * img.shape[0] / max(img.shape[1], 1)))
        ax.imshow(img); ax.axis("off"); ax.set_title(s.name, fontsize=10)
        plt.show()


def bar_compare(df, metric: str = "psnr_db", save_path: str | Path | None = None):
    """Bar chart of one metric across pipelines (NaNs skipped)."""
    import matplotlib.pyplot as plt
    sub = df[metric].dropna()
    if sub.empty:
        print(f"no values for {metric}")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sub.index.astype(str), sub.values, color="#4C72B0")
    ax.set_ylabel(metric); ax.set_title(f"{metric} by pipeline")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(sub.values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=120)
    return fig


# ===========================================================================
# Pipeline trial helpers — leave-out evaluation with pre/post spatial steps
# ===========================================================================

_DEGRADE_SPATIAL = None
_DEGRADE_NOISE   = None


def _get_degradation():
    global _DEGRADE_SPATIAL, _DEGRADE_NOISE
    if _DEGRADE_SPATIAL is None:
        for p in (str(REPO_ROOT),):
            if p not in sys.path:
                sys.path.insert(0, p)
        from data.degradation_spatial import SpatialDegradation
        from data.degradation_noise import RicianNoise
        _DEGRADE_SPATIAL = SpatialDegradation(source_voxel_mm=1.5, target_voxel_mm=3.0)
        _DEGRADE_NOISE   = RicianNoise(sigma_min=0.05, sigma_max=0.05, seed=0)
    return _DEGRADE_SPATIAL, _DEGRADE_NOISE


@dataclass
class TrialModels:
    """All pre-loaded models for pipeline trials. Build with load_trial_models()."""
    interp:       object
    sr_model:     object
    sr_config:    object
    sr_device:    object
    denoiser:     object
    den_device:   object
    joint_model:  object
    joint_cfg:    object
    joint_device: object


def load_trial_models(cfg) -> TrialModels:
    """Load every model once. Call at the start of the trials notebook."""
    import torch
    for p in (str(REPO_ROOT), str(INTERP_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 1. Temporal interpolation
    interp = load_interpolator(cfg)

    # 2. Spatial SR (RCAN3D checkpoint)
    sr_ckpts = sorted((REPO_ROOT / "weights" / "sr").glob("sr_rcan3d_*_best.pt"))
    if not sr_ckpts:
        raise FileNotFoundError("No sr_rcan3d_*_best.pt found in weights/sr/")
    from sr.infer import _load_model_from_checkpoint
    sr_model, sr_config, sr_device = _load_model_from_checkpoint(sr_ckpts[-1])

    # 3. Denoiser (SimpleUNet, slice-by-slice)
    den_dir = str(REPO_ROOT / "Denoising")
    if den_dir not in sys.path:
        sys.path.insert(0, den_dir)
    from model import SimpleUNet
    den_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = SimpleUNet().to(den_device)
    denoiser.load_state_dict(torch.load(
        str(REPO_ROOT / "weights" / "denoiser" / "mri_unet_robust.pth"),
        map_location=den_device, weights_only=True))
    denoiser.eval()

    # 4. Joint denoise+SR
    from joint.eval import load_checkpoint as _joint_load
    jw = REPO_ROOT / "weights" / "joint" / "best.pt"
    if not jw.exists():
        jw = Path("/srv/venvs/team4dbrain/joint_model/best.pt")
    joint_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_model, joint_cfg, _ = _joint_load(str(jw), joint_device)

    print(f"[load_trial_models] interp={interp.device}  sr={sr_device}  "
          f"denoise={den_device}  joint={joint_device}")
    return TrialModels(
        interp=interp,
        sr_model=sr_model, sr_config=sr_config, sr_device=sr_device,
        denoiser=denoiser, den_device=den_device,
        joint_model=joint_model, joint_cfg=joint_cfg, joint_device=joint_device,
    )


# ---------------------------------------------------------------------------
# Per-volume operations (numpy → numpy, no file I/O)
# ---------------------------------------------------------------------------

def _predict_pair(
    models: TrialModels,
    v_a: np.ndarray,
    v_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the interp model on two neighbor volumes in XYZ order.
    Returns (pred, naive) in XYZ physical units.
    """
    import torch
    from src.inference import normalize_pair

    interp = models.interp
    a_dhw = np.ascontiguousarray(v_a.T)   # XYZ → DHW
    b_dhw = np.ascontiguousarray(v_b.T)

    a_n, b_n, mu, sigma = normalize_pair(a_dhw, b_dhw, interp.norm_mode)
    inp = torch.from_numpy(np.stack([a_n, b_n])[None].astype(np.float32)).to(interp.device)

    with torch.no_grad():
        raw = interp.model(inp)
        pred_n  = (0.5 * (inp[:, 0:1] + inp[:, 1:2]) + raw) if interp.residual else raw
        naive_n =  0.5 * (inp[:, 0:1] + inp[:, 1:2])

    def _back(t):
        return np.ascontiguousarray((t.squeeze().cpu().numpy() * sigma + mu).T)  # DHW→XYZ

    return _back(pred_n), _back(naive_n)


def _sr_vol(vol_lr: np.ndarray, models: TrialModels, norm_ref: float | None = None) -> np.ndarray:
    """Super-resolve a 3D LR numpy volume → HR numpy via the SR checkpoint."""
    import torch
    from sr.forward import model_forward

    ref = norm_ref or (float(np.percentile(vol_lr, 98)) or 1.0)
    inputs = torch.from_numpy(vol_lr.astype(np.float32) / ref).unsqueeze(0).unsqueeze(0).to(models.sr_device)
    dummy  = torch.zeros((1, 1, 1, 1, 1), device=models.sr_device, dtype=inputs.dtype)
    pred   = model_forward(models.sr_model, inputs, dummy, models.sr_config.model_name)
    return pred.squeeze(0).squeeze(0).detach().cpu().numpy() * ref


def _joint_vol(vol_lr: np.ndarray, models: TrialModels, norm_ref: float | None = None) -> np.ndarray:
    """Denoise+super-resolve a 3D LR numpy volume → HR numpy via the joint model."""
    import torch

    ref = norm_ref or (float(np.percentile(vol_lr, 98)) or 1.0)
    x   = torch.from_numpy(np.ascontiguousarray(vol_lr.astype(np.float32) / ref))[None, None].to(models.joint_device)
    with torch.no_grad():
        out = models.joint_model(x)[0, 0].cpu().numpy()
    return out * ref


def _denoise_vol(vol: np.ndarray, models: TrialModels) -> np.ndarray:
    """Denoise a 3D numpy volume slice-by-slice using the U-Net denoiser."""
    import torch

    X, Y, Z = vol.shape
    out = np.zeros_like(vol, dtype=np.float32)
    p   = float(np.percentile(vol, 99)) or 1.0
    with torch.no_grad():
        for z in range(Z):
            s = np.clip(vol[:, :, z], 0, p) / p
            inp = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(models.den_device)
            out[:, :, z] = models.denoiser(inp).cpu().squeeze().numpy() * p
    return out


def _apply_steps(
    vol: np.ndarray,
    steps: list[str],
    models: TrialModels,
    norm_ref: float,
) -> np.ndarray:
    """Apply a sequence of named processing steps to a 3D volume."""
    deg_spatial, deg_noise = _get_degradation()
    for step in steps:
        if   step == "spatial": vol = deg_spatial(vol)
        elif step == "noise":   vol = deg_noise(vol)
        elif step == "sr":      vol = _sr_vol(vol, models, norm_ref)
        elif step == "joint":   vol = _joint_vol(vol, models, norm_ref)
        elif step == "denoise": vol = _denoise_vol(vol, models)
        else: raise ValueError(f"Unknown step {step!r}. Valid: spatial, noise, sr, joint, denoise")
    return vol


# ---------------------------------------------------------------------------
# Leave-out trial
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """One leave-out interpolation trial — all volumes in XYZ physical units."""
    name:       str
    label:      str
    t:          int
    gt:         np.ndarray   # clean HR ground truth
    pred:       np.ndarray   # model prediction (after post-steps)
    naive:      np.ndarray   # naive average  (after post-steps)
    pre_steps:  list
    post_steps: list
    metrics:    dict

    @property
    def error(self):       return np.abs(self.gt - self.pred)

    @property
    def naive_error(self): return np.abs(self.gt - self.naive)


def run_trial(
    bold_path:  str | Path,
    t:          int,
    pre_steps:  list[str],
    post_steps: list[str],
    models:     TrialModels,
    name:       str = "trial",
    label:      str = "",
    norm_ref:   float | None = None,
) -> TrialResult:
    """Hold out frame t, process its two neighbours, interpolate, post-process.

    pre_steps  — applied to each neighbour BEFORE interpolation (e.g. ["spatial"])
    post_steps — applied to the interpolated frame AFTER  (e.g. ["sr"])
    GT is always the clean HR frame at position t.
    """
    import nibabel as nib

    data = nib.load(str(bold_path)).get_fdata(dtype=np.float32)
    T    = data.shape[-1]
    if not (1 <= t < T - 1):
        raise IndexError(f"t={t} must be an interior frame (1 .. {T-2})")

    gt     = data[..., t]
    v_prev = data[..., t - 1].copy()
    v_next = data[..., t + 1].copy()
    ref    = norm_ref or (float(np.percentile(data.mean(-1), 98)) or 1.0)

    p_prev = _apply_steps(v_prev, pre_steps,  models, ref)
    p_next = _apply_steps(v_next, pre_steps,  models, ref)

    pred_raw, naive_raw = _predict_pair(models, p_prev, p_next)

    pred  = _apply_steps(pred_raw,  post_steps, models, ref)
    naive = _apply_steps(naive_raw, post_steps, models, ref)

    dr    = float(np.percentile(gt, 99.5)) or 1.0
    mse_m = float(np.mean((pred  - gt) ** 2))
    mse_n = float(np.mean((naive - gt) ** 2))
    l1_m  = float(np.mean(np.abs(pred  - gt)))
    l1_n  = float(np.mean(np.abs(naive - gt)))

    def _psnr(mse): return float(10 * np.log10(dr ** 2 / max(mse, 1e-12)))

    return TrialResult(
        name=name, label=label or name, t=t,
        gt=gt, pred=pred, naive=naive,
        pre_steps=pre_steps, post_steps=post_steps,
        metrics=dict(model_l1=l1_m, naive_l1=l1_n,
                     model_psnr=_psnr(mse_m), naive_psnr=_psnr(mse_n),
                     model_beats=l1_m < l1_n),
    )


def sweep_trial(
    bold_path:  str | Path,
    pre_steps:  list[str],
    post_steps: list[str],
    models:     TrialModels,
    name:       str = "trial",
    label:      str = "",
    stride:     int = 10,
    norm_ref:   float | None = None,
) -> list[dict]:
    """Sweep run_trial over interior timepoints; return list of metric dicts."""
    import nibabel as nib
    T  = nib.load(str(bold_path)).shape[-1]
    ts = list(range(1, T - 1, stride))
    rows = []
    for i, t in enumerate(ts):
        print(f"  {name}: t={t} ({i+1}/{len(ts)})", end="\r", flush=True)
        r = run_trial(bold_path, t, pre_steps, post_steps, models,
                      name=name, label=label, norm_ref=norm_ref)
        rows.append({"t": t, "label": label or name, **r.metrics})
    print(f"  {name}: done ({len(ts)} frames)          ")
    return rows


# ---------------------------------------------------------------------------
# Visualization for trials
# ---------------------------------------------------------------------------

def show_trial_simple(result: TrialResult, *, title: str = "", save_path=None):
    """Classic tri-planar figure — raw error maps, hot colormap (original style)."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    gt, pred, naive = result.gt, result.pred, result.naive
    err_m, err_n    = result.error, result.naive_error
    m = result.metrics

    cx, cy, cz = [s // 2 for s in gt.shape]
    vmax = float(np.percentile(gt, 99.5)) or 1.0
    emax = float(np.percentile(np.maximum(err_m, err_n), 99)) or 1.0

    planes = [
        (gt[cx,:,:].T, pred[cx,:,:].T, err_m[cx,:,:].T, err_n[cx,:,:].T, "Sagittal"),
        (gt[:,cy,:].T, pred[:,cy,:].T, err_m[:,cy,:].T, err_n[:,cy,:].T, "Coronal"),
        (gt[:,:,cz].T, pred[:,:,cz].T, err_m[:,:,cz].T, err_n[:,:,cz].T, "Axial"),
    ]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.04, hspace=0.12)

    col_titles = ["Ground Truth", "Predicted", "|Error| model", "|Error| naive"]
    for col, ct in enumerate(col_titles):
        fig.add_subplot(gs[0, col]).set_title(ct, fontsize=11, fontweight="bold", pad=6)

    for row, (g, p, em, en, plane) in enumerate(planes):
        specs = [(g,"gray",0,vmax),(p,"gray",0,vmax),(em,"hot",0,emax),(en,"hot",0,emax)]
        for col, (d, cmap, vmin_, vmax_) in enumerate(specs):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(d, cmap=cmap, vmin=vmin_, vmax=vmax_, origin="lower", aspect="auto")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(plane, fontsize=9, labelpad=4)
            if col >= 2 and row == 2:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    beat   = "✓ beats naive" if m["model_beats"] else "✗ loses to naive"
    suptit = (
        f"{title}  (t={result.t})\n"
        f"PSNR  model={m['model_psnr']:.2f} dB  naive={m['naive_psnr']:.2f} dB   |   "
        f"L1  model={m['model_l1']:.4f}  naive={m['naive_l1']:.4f}   |   {beat}"
    )
    fig.suptitle(suptit, fontsize=10, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.show()
    return fig


def show_trial(result: TrialResult, *, title: str = "", save_path=None, error_vmax=None):
    """Tri-planar GT / Predicted / |Error model| / |Error naive| figure."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    gt, pred, naive = result.gt, result.pred, result.naive
    err_m, err_n    = result.error, result.naive_error
    m = result.metrics

    cx, cy, cz = [s // 2 for s in gt.shape]
    vmax = float(np.percentile(gt, 99.5)) or 1.0

    # normalise errors to [0,1] and mask out background
    mask   = gt > 0.05 * vmax
    err_mn = err_m / vmax
    err_nn = err_n / vmax
    if error_vmax is not None:
        emax = float(error_vmax)
    else:
        emax = float(np.percentile(np.maximum(err_mn[mask], err_nn[mask]), 99)) or 0.1

    def _masked(arr2d, mask2d):
        out = np.ma.array(arr2d, mask=~mask2d)
        return out

    planes = [
        (gt[cx,:,:].T,   pred[cx,:,:].T,   err_mn[cx,:,:].T,  err_nn[cx,:,:].T,  mask[cx,:,:].T,  "Sagittal"),
        (gt[:,cy,:].T,   pred[:,cy,:].T,    err_mn[:,cy,:].T,  err_nn[:,cy,:].T,  mask[:,cy,:].T,  "Coronal"),
        (gt[:,:,cz].T,   pred[:,:,cz].T,   err_mn[:,:,cz].T,  err_nn[:,:,cz].T,  mask[:,:,cz].T,  "Axial"),
    ]

    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.05, hspace=0.15)

    col_titles = ["Ground Truth", "Prediction", "|Error| model", "|Error| naive"]
    for col, ct in enumerate(col_titles):
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_title(ct, fontsize=12, fontweight="bold", pad=8)
        ax0.axis("off")

    for row, (g, p, em, en, mk, plane) in enumerate(planes):
        anat_kw  = dict(cmap="gray",    vmin=0, vmax=vmax, origin="lower", aspect="auto")
        err_kw   = dict(cmap="inferno", vmin=0, vmax=emax, origin="lower", aspect="auto")

        ax_g  = fig.add_subplot(gs[row, 0])
        ax_p  = fig.add_subplot(gs[row, 1])
        ax_em = fig.add_subplot(gs[row, 2])
        ax_en = fig.add_subplot(gs[row, 3])

        ax_g.imshow(g,  **anat_kw)
        ax_p.imshow(p,  **anat_kw)
        # show anatomy in background, overlay masked error
        ax_em.imshow(g, **anat_kw)
        im = ax_em.imshow(_masked(em, mk), **err_kw, alpha=0.85)
        ax_en.imshow(g, **anat_kw)
        ax_en.imshow(_masked(en, mk), **err_kw, alpha=0.85)

        for ax, lbl in zip([ax_g, ax_p, ax_em, ax_en], [""] * 4):
            ax.axis("off")
        ax_g.set_ylabel(plane, fontsize=10, labelpad=4)

        if row == 2:
            fig.colorbar(im, ax=ax_en, fraction=0.046, pad=0.04,
                         label="Normalised |error|")

    beat   = "✓ beats naive" if m["model_beats"] else "✗ loses to naive"
    norm_l1_m = m["model_l1"] / vmax
    norm_l1_n = m["naive_l1"] / vmax
    suptit = (
        f"{title}  (t={result.t})\n"
        f"PSNR  model={m['model_psnr']:.2f} dB   naive={m['naive_psnr']:.2f} dB   |   "
        f"L1 (norm)  model={norm_l1_m:.4f}   naive={norm_l1_n:.4f}   |   {beat}"
    )
    fig.suptitle(suptit, fontsize=11, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.show()
    return fig


def sweep_plot(sweeps: dict, *, title: str = "", save_path=None):
    """Line plots of PSNR and normalised L1 over time for multiple pipeline variants."""
    import matplotlib.pyplot as plt
    import pandas as pd

    PALETTE = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for i, (name, rows) in enumerate(sweeps.items()):
        df    = pd.DataFrame(rows)
        color = PALETTE[i % len(PALETTE)]
        label = rows[0].get("label", name) if rows else name
        axes[0].plot(df["t"], df["model_psnr"],             color=color, lw=2,   label=label)
        axes[0].plot(df["t"], df["naive_psnr"], "--",       color=color, lw=1,   alpha=0.4)
        axes[1].plot(df["t"], df["model_l1"] / df["model_l1"].max(), color=color, lw=2, label=label)
    axes[0].set_xlabel("Frame t"); axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR over time  (dashed = naive baseline)")
    axes[0].legend(fontsize=8, loc="lower right"); axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Frame t"); axes[1].set_ylabel("Normalised L1")
    axes[1].set_title("L1 error over time  (per-pipeline normalised for shape)")
    axes[1].legend(fontsize=8, loc="upper right"); axes[1].grid(alpha=0.25)
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.show()
    return fig


def pipelines_summary(sweeps: dict, *, save_path=None):
    """Bar charts + summary DataFrame for all pipeline variants."""
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for name, mlist in sweeps.items():
        df    = pd.DataFrame(mlist)
        label = mlist[0].get("label", name) if mlist else name
        rows.append(dict(
            pipeline        = label,
            mean_psnr       = round(df["model_psnr"].mean(), 3),
            mean_l1         = round(df["model_l1"].mean(), 4),
            pct_beats_naive = round(df["model_beats"].mean() * 100, 1),
        ))
    summary = pd.DataFrame(rows).set_index("pipeline")

    PALETTE = plt.cm.tab10.colors[:len(summary)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    metrics = [
        ("mean_psnr",       "Mean PSNR (dB)",       "Higher is better ↑"),
        ("mean_l1",         "Mean L1 (raw)",         "Lower is better ↓"),
        ("pct_beats_naive", "% Frames beating naive","Higher is better ↑"),
    ]
    for ax, (col, yl, sub) in zip(axes, metrics):
        vals = summary[col]
        bars = ax.bar(range(len(vals)), vals.values, color=PALETTE, alpha=0.88, width=0.6,
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=38, ha="right", fontsize=8.5)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(f"{yl}\n{sub}", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + vals.max() * 0.012,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Pipeline Comparison — Leave-out Evaluation  (GT = real held-out frame)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.show()
    return fig, summary
