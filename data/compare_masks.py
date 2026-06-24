"""Compare percentile vs SynthStrip brain masks visually.

Drop the function below into a Jupyter cell and call it with your paths.
Saves PNG comparisons for each file. Falls back to percentile-only if
SynthStrip isn't installed.

Example call:
    compare_masks(
        project_root=r"C:\\Users\\Andriy\\Studies\\Sem¿\\project\\test code",
        data_root=r"C:\\Users\\Andriy\\Studies\\Sem¿\\project\\data\\test",
        max_files=3,
    )
"""

# ---- BEGIN COPYABLE NOTEBOOK CELL ----

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compare_masks(
    project_root: str | Path,
    data_root: str | Path,
    max_files: int | None = None,
    output_dir: str | Path = ".",
    percentile: float = 55.0,
    synthstrip_border: int = 1,
    synthstrip_no_csf: bool = False,
):
    """Compare percentile vs SynthStrip masks across all *_bold.nii.gz under data_root.

    Args:
        project_root: path to the repo root. Added to sys.path.
        data_root: directory to search for *_bold.nii.gz files (recursive).
        max_files: if set, only process the first N files. None = all.
        output_dir: where to save the PNG comparisons.
        percentile: percentile param for the percentile method.
        synthstrip_border: -b flag for synthstrip (mm border).
        synthstrip_no_csf: --no-csf flag for synthstrip.
    """
    project_root = Path(project_root)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.masks import (
        _compute_percentile_mask,
        _compute_synthstrip_mask,
        find_synthstrip_executable,
    )
    from data.reader import VolumeReader

    files_to_check = sorted(data_root.rglob("*_bold.nii.gz"))
    if max_files is not None:
        files_to_check = files_to_check[:max_files]

    if not files_to_check:
        print(f"No *_bold.nii.gz files found under {data_root}")
        return

    print(f"Will process {len(files_to_check)} file(s)")

    synthstrip_exe = find_synthstrip_executable()
    if synthstrip_exe:
        print(f"SynthStrip found: {synthstrip_exe}")
    else:
        print("WARNING: SynthStrip not on PATH. Will only show percentile masks.")
        print("Install instructions in the docstring of masks.py.")

    for f in files_to_check:
        print(f"\n--- {f.name} ---")
        reader = VolumeReader(f)
        print("  Reading mean (~30s for full IBC runs)...")
        mean_vol = reader.read_mean()

        x_mid = mean_vol.shape[0] // 2
        y_mid = mean_vol.shape[1] // 2
        z_mid = mean_vol.shape[2] // 2

        print(f"  Computing percentile mask (p={percentile})...")
        mask_pct = _compute_percentile_mask(mean_vol, lower_percentile=percentile)
        pct_frac = mask_pct.mean()

        if synthstrip_exe:
            print("  Computing SynthStrip mask (first run pulls Docker image, ~2GB)...")
            try:
                mask_ss = _compute_synthstrip_mask(
                    mean_vol, reader.img.affine,
                    executable=synthstrip_exe,
                    border=synthstrip_border, no_csf=synthstrip_no_csf,
                )
                ss_frac = mask_ss.mean()
                n_cols = 2
            except Exception as e:
                print(f"  SynthStrip failed: {e}")
                mask_ss = None
                n_cols = 1
        else:
            mask_ss = None
            n_cols = 1

        fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 15), squeeze=False)
        fig.suptitle(f.name, fontsize=12)

        views = [
            ("axial", lambda v: v[:, :, z_mid].T),
            ("coronal", lambda v: v[:, y_mid, :].T),
            ("sagittal", lambda v: v[x_mid, :, :].T),
        ]

        for row, (view_name, slicer) in enumerate(views):
            axes[row, 0].imshow(slicer(mean_vol), cmap="gray", origin="lower")
            axes[row, 0].imshow(slicer(mask_pct), alpha=0.4, cmap="Reds", origin="lower")
            axes[row, 0].set_title(f"{view_name} — percentile (frac={pct_frac:.3f})")

            if mask_ss is not None:
                axes[row, 1].imshow(slicer(mean_vol), cmap="gray", origin="lower")
                axes[row, 1].imshow(slicer(mask_ss), alpha=0.4, cmap="Greens", origin="lower")
                axes[row, 1].set_title(f"{view_name} — SynthStrip (frac={ss_frac:.3f})")

        plt.tight_layout()
        out_path = output_dir / f"mask_compare_{f.stem.replace('.nii', '')}.png"
        plt.savefig(out_path, dpi=80)
        plt.show()
        print(f"  Saved {out_path}")

    print("\nDone.")


# Example usage:
# compare_masks(
#     project_root=r"C:\Users\Andriy\Studies\Sem¿\project\test code",
#     data_root=r"C:\Users\Andriy\Studies\Sem¿\project\data\test",
#     max_files=3,
# )

# ---- END COPYABLE NOTEBOOK CELL ----
