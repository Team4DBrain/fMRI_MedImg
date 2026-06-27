# notebooks/ — presentation & experiments

Thin, presentable notebooks for showing the **temporal-interpolation** work and
comparing **full restoration pipelines**. All the model wiring and the
midterm-style figures live in [`nb_utils.py`](nb_utils.py); the notebooks just
narrate and call it.

## Run order

| notebook | what it shows |
|---|---|
| [`00_setup_and_data.ipynb`](00_setup_and_data.ipynb) | env check, set paths once (`Config`), list runs, first look at a BOLD volume |
| [`01_interpolation_results.ipynb`](01_interpolation_results.ipynb) | **my part** — tri-planar GT/Pred/\|Error\|, error-localization MIPs, naive-baseline comparison, L1/PSNR across the run, loss curve, optional 2× NIfTI |
| [`02_pipeline_comparison.ipynb`](02_pipeline_comparison.ipynb) | `joint` vs `denoise→sr` cascade vs `sr`-only vs `interp`, via `orchestrator.py`; metrics table + bar charts + montage slides |

## Setup (on the VM)

```bash
cd ~/CAI-MedImg                       # repo root (where orchestrator.py lives)
source /srv/venvs/team4dbrain/setup_env.sh
jupyter lab notebooks/               # or jupyter notebook
```

Then in `00_setup_and_data` edit one cell:

```python
cfg = nb.Config(
    data_dir="/srv/fMRI-data",   # folder of *_bold.nii.gz runs
    bold_file=None,              # or pin one run explicitly
)
```

That `Config` is the only thing you normally touch — every notebook starts by
constructing it.

## What you need in place

- **Env active** (torch + nibabel + matplotlib + pandas).
- **Interp weights:** `data_interpolation/checkpoints/pretrained/model_weights.pt`
  (shipped; zscore norm, non-residual — the `Config` defaults match).
- **Pipeline weights** (only for `02`): `joint` (shared VM weights), `sr`
  (`models/sr_*_best.pt`), `denoise` (`Denoising/mri_unet_robust.pth`).
- **Data:** at least one 4D BOLD run under `data_dir`.

## Notes

- Figures and pipeline runs are written under `notebooks/outputs/` (gitignored
  by the repo's existing rules for generated artifacts — verify before committing).
- `02` shells out to `orchestrator.py`; each run can take minutes on GPU. Use
  `TRUNC` (the `truncate=` arg) to keep demo runs short, set it to `0` for a
  full run.
- `interp` changes the number of frames, so PSNR/SSIM vs the original is
  undefined — the orchestrator reports a leave-out PSNR/L1 + tSNR instead.
- The notebooks use `%autoreload`, so editing `nb_utils.py` takes effect without
  restarting the kernel.
