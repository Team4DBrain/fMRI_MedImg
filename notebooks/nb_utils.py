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
