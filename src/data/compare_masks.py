"""Compare percentile vs SynthStrip brain masks.

Drop this into a Jupyter cell (or run as a script) to visualize how the two
methods differ on your data. Useful for verifying SynthStrip is working
correctly after you install it.

Setup:
  1. Install SynthStrip via Docker (the easiest path on Windows):
       Install Docker Desktop: https://www.docker.com/products/docker-desktop/
       Then download the wrapper script:
         curl -O https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/synthstrip-docker
       On Windows, save as `synthstrip-docker` (no extension), make sure it's
       on your PATH (e.g., put it in your conda env's Scripts directory).
       Or rename to `synthstrip-docker.bat` with a one-liner that calls
       `docker run --rm -v %cd%:/data freesurfer/synthstrip:latest %*`.

  2. First run will pull the Docker image (~2 GB, one-time).

  3. Run this cell. It compares masks and saves PNGs.

If SynthStrip isn't available, the cell will note that and only show
percentile results.
"""

# ---- BEGIN COPYABLE NOTEBOOK CELL ----

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Adjust this path to your actual code directory.
PROJECT_ROOT = r"C:\Users\Andriy\Studies\Sem¿\project\test code"
sys.path.insert(0, PROJECT_ROOT)

from src.data.masks import (  # noqa: E402
    _compute_percentile_mask,
    _compute_synthstrip_mask,
    find_synthstrip_executable,
)
from src.data.reader import VolumeReader  # noqa: E402

# Files to compare. Mix of int16 (sub-01) and float32 (sub-04+) for variety.
DATA_ROOT = Path(r"C:\Users\Andriy\Studies\Sem¿\project\data\test")
files_to_check = [
    DATA_ROOT / "sub-01" / "ses-00" / "func" / "sub-01_ses-00_task-ArchiSocial_dir-ap_bold.nii.gz",
    DATA_ROOT / "sub-04" / "ses-04" / "func" / "sub-04_ses-04_task-ArchiSpatial_dir-ap_bold.nii.gz",
    DATA_ROOT / "sub-07" / "ses-05" / "func" / "sub-07_ses-05_task-ClipsVal_dir-pa_run-01_bold.nii.gz",
]

# Check if SynthStrip is available
synthstrip_exe = find_synthstrip_executable()
if synthstrip_exe:
    print(f"SynthStrip found: {synthstrip_exe}")
    print("Will compare percentile vs SynthStrip masks.")
else:
    print("WARNING: SynthStrip not on PATH. Looked for: mri_synthstrip, "
          "synthstrip-docker, synthstrip-singularity.")
    print("Will only show percentile masks. Install SynthStrip to compare.")
    print()
    print("Install instructions:")
    print("  1. Install Docker Desktop")
    print("  2. Download wrapper: https://raw.githubusercontent.com/freesurfer/"
          "freesurfer/dev/mri_synthstrip/synthstrip-docker")
    print("  3. Make it executable on PATH (or use a .bat wrapper on Windows)")

# Process each file
for f in files_to_check:
    if not f.exists():
        print(f"\nSkipping (not found): {f}")
        continue

    print(f"\n--- {f.name} ---")
    reader = VolumeReader(f)
    print(f"  Reading mean (this takes ~30s for full IBC runs)...")
    mean_vol = reader.read_mean()

    # Mid-slice indices on each axis for the three views
    x_mid = mean_vol.shape[0] // 2
    y_mid = mean_vol.shape[1] // 2
    z_mid = mean_vol.shape[2] // 2

    # Percentile mask (always works)
    print("  Computing percentile mask...")
    mask_pct = _compute_percentile_mask(mean_vol, lower_percentile=55.0)
    pct_frac = mask_pct.mean()

    # SynthStrip mask if available
    if synthstrip_exe:
        print("  Computing SynthStrip mask (first run pulls Docker image, ~2GB)...")
        try:
            mask_ss = _compute_synthstrip_mask(
                mean_vol, reader.img.affine, executable=synthstrip_exe,
                border=1, no_csf=False,
            )
            ss_frac = mask_ss.mean()
            n_cols = 2  # show both
        except Exception as e:
            print(f"  SynthStrip failed: {e}")
            mask_ss = None
            n_cols = 1
    else:
        mask_ss = None
        n_cols = 1

    # Plot. 3 rows (axial, coronal, sagittal) × n_cols
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 15), squeeze=False)
    fig.suptitle(f.name, fontsize=12)

    views = [
        ("axial", lambda v: v[:, :, z_mid].T),
        ("coronal", lambda v: v[:, y_mid, :].T),
        ("sagittal", lambda v: v[x_mid, :, :].T),
    ]

    for row, (view_name, slicer) in enumerate(views):
        # Percentile column
        axes[row, 0].imshow(slicer(mean_vol), cmap="gray", origin="lower")
        axes[row, 0].imshow(slicer(mask_pct), alpha=0.4, cmap="Reds", origin="lower")
        axes[row, 0].set_title(f"{view_name} — percentile (frac={pct_frac:.3f})")

        # SynthStrip column if we have it
        if mask_ss is not None:
            axes[row, 1].imshow(slicer(mean_vol), cmap="gray", origin="lower")
            axes[row, 1].imshow(slicer(mask_ss), alpha=0.4, cmap="Greens", origin="lower")
            axes[row, 1].set_title(f"{view_name} — SynthStrip (frac={ss_frac:.3f})")

    plt.tight_layout()
    out_name = f"mask_compare_{f.stem.replace('.nii', '')}.png"
    plt.savefig(out_name, dpi=80)
    plt.show()
    print(f"  Saved {out_name}")

print("\nDone.")

# ---- END COPYABLE NOTEBOOK CELL ----
