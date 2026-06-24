"""Build a manifest of fMRI runs from a BIDS-formatted dataset.

This module does a fast pass: walks the directory, parses filenames, reads
NIfTI headers (not data). Produces a JSON manifest with one entry per run.

Does NOT compute brain masks or metadata — that's compute_metadata.py's job.

Run from the command line:
    python -m src.data.manifest --bids-root /path/to/ibc_raw --out manifest.json

By default, only runs whose native z dimension equals --require-z (default 93)
are kept; non-conforming runs are logged and dropped. This screens out the IBC
ses-00/ses-01 z=84 anomaly (documented FOV irregularity in early sessions).
Pass --require-z 0 to disable filtering and keep every run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib

from ._cli import (
    DEFAULT_BIDS_ROOT,
    DEFAULT_MANIFEST_PATH,
    existing_dir,
    writable_file,
)

logger = logging.getLogger(__name__)

# Default expected z dimension for IBC. Drops the documented 84-slice anomaly
# in early sessions. Override with --require-z if your dataset differs.
DEFAULT_REQUIRE_Z = 93

# Regex for BIDS BOLD filenames. Two valid patterns IBC uses:
#   sub-<id>_ses-<id>_task-<n>_dir-<ap|pa>_bold.nii.gz                  (most tasks)
#   sub-<id>_ses-<id>_task-<n>_dir-<ap|pa>_run-<NN>_bold.nii.gz         (multi-run tasks like Clips)
# `_run-NN` is the BIDS "run" entity, used when the same subject/session/task is
# acquired multiple times in one session (e.g. ClipsTrn run-01, run-02, run-03).
# Optional non-capturing group (?:..)? makes the whole _run-NN piece optional;
# the inner (?P<run>...) captures just the number when present.
BOLD_FILENAME_RE = re.compile(
    r"^sub-(?P<subject>[^_]+)"
    r"_ses-(?P<session>[^_]+)"
    r"_task-(?P<task>[^_]+)"
    r"_dir-(?P<direction>ap|pa)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_bold\.nii\.gz$"
)


@dataclass
class RunEntry:
    """One BOLD run's metadata. Keep this flat — makes JSON cleaner."""

    run_id: str  # unique identifier, derived from filename
    subject: str
    session: str
    task: str
    direction: str  # 'ap' or 'pa'
    path: str  # relative to bids_root
    shape: tuple[int, int, int, int]  # (X, Y, Z, T)
    n_volumes: int
    dtype: str  # scanner's native dtype, usually 'int16'
    run: str | None = None  # optional BIDS "run" entity (e.g., "01" for run-01)


def parse_bold_filename(filename: str) -> dict | None:
    """Extract BIDS entities from a BOLD filename. Returns None if not a BOLD file."""
    match = BOLD_FILENAME_RE.match(filename)
    if match is None:
        return None
    return match.groupdict()


def build_manifest(
    bids_root: Path,
    require_z: int | None = DEFAULT_REQUIRE_Z,
) -> list[RunEntry]:
    """Walk the BIDS root and build a list of RunEntry for every BOLD file found.

    Args:
        bids_root: BIDS-formatted dataset root.
        require_z: if set, drop runs whose native z != require_z. The IBC dataset
            has a documented FOV anomaly in early sessions producing z=84 instead
            of the expected z=93; require_z=93 (default) screens those out. Pass
            None or 0 to keep all runs regardless of z.

    Raises FileNotFoundError if bids_root doesn't exist.
    Warns (logs) on files that look like BOLD but don't parse — doesn't raise.
    Warns and drops runs whose z doesn't match require_z; logs a summary.
    """
    bids_root = Path(bids_root).resolve()
    if not bids_root.is_dir():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    # Find BOLD files anywhere under bids_root, supporting both layouts:
    #   nested BIDS: sub-XX/ses-YY/func/*_bold.nii.gz
    #   flat:        *_bold.nii.gz directly in bids_root
    # rglob walks recursively, so it finds both. Dedupe by absolute path
    # (just in case of symlinks/hard links pointing to the same file).
    seen: set[Path] = set()
    bold_files: list[Path] = []
    for path in sorted(bids_root.rglob("*_bold.nii.gz")):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        bold_files.append(path)
    logger.info(f"Found {len(bold_files)} BOLD files under {bids_root}")

    entries: list[RunEntry] = []
    dropped_z: list[tuple[str, int]] = []  # (run_id, native_z) for the summary
    for path in bold_files:
        parsed = parse_bold_filename(path.name)
        if parsed is None:
            logger.warning(f"Filename doesn't match expected BIDS pattern, skipping: {path.name}")
            continue

        # Read NIfTI header only — doesn't decompress the data block.
        # Cheap: a few milliseconds per file.
        try:
            img = nib.load(str(path))
            shape = tuple(int(s) for s in img.shape)
            dtype = str(img.get_data_dtype())
        except Exception as e:
            logger.warning(f"Couldn't read NIfTI header for {path.name}: {e}")
            continue

        if len(shape) != 4:
            logger.warning(f"{path.name} is not 4D (shape={shape}), skipping")
            continue

        # z-conformance filter. Documented IBC anomaly: ses-00/ses-01 sometimes
        # have z=84 instead of the standard z=93 (Pinho et al. 2020 attribute
        # this to a transient FOV irregularity affecting cerebellar coverage).
        # Drop these so the model sees consistent FOV across the dataset.
        if require_z is not None and require_z > 0 and shape[2] != require_z:
            run_id = path.name.replace("_bold.nii.gz", "")
            logger.warning(
                "Dropping %s: native z=%d != require_z=%d "
                "(known IBC FOV anomaly affects ses-00/ses-01 of some subjects).",
                run_id, shape[2], require_z,
            )
            dropped_z.append((run_id, shape[2]))
            continue

        run_id = path.name.replace("_bold.nii.gz", "")
        entry = RunEntry(
            run_id=run_id,
            subject=parsed["subject"],
            session=parsed["session"],
            task=parsed["task"],
            direction=parsed["direction"],
            path=str(path.relative_to(bids_root)),
            shape=shape,
            n_volumes=shape[-1],
            dtype=dtype,
            run=parsed.get("run"),  # None when no _run-NN in filename
        )
        entries.append(entry)

    if dropped_z:
        zs = sorted({z for _, z in dropped_z})
        logger.warning(
            "Dropped %d run(s) with non-conforming z (saw %s; expected %d)",
            len(dropped_z), zs, require_z,
        )

    return entries


def write_manifest(
    entries: list[RunEntry],
    bids_root: Path,
    out_path: Path,
    require_z: int | None = None,
) -> None:
    """Serialize the manifest to JSON. Atomic: write to .tmp then os.replace.

    A SIGKILL or disk-full mid-write leaves the previous manifest intact
    rather than producing a half-written JSON that breaks every downstream
    loader. os.replace is atomic on POSIX and Windows.

    `require_z` is recorded in the manifest for traceability so compute_metadata
    can detect a stage1/stage2 mismatch and refuse to run with inconsistent
    target_z.
    """
    payload = {
        "version": 1,
        "bids_root": str(Path(bids_root).resolve()),
        "n_runs": len(entries),
        "require_z": require_z,
        "runs": [asdict(e) for e in entries],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)
    logger.info(f"Wrote manifest with {len(entries)} runs to {out_path}")


def load_manifest(path: Path) -> dict:
    """Load a manifest JSON file. Returns the raw dict (not dataclasses)."""
    with Path(path).open() as f:
        return json.load(f)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root", type=existing_dir, default=str(DEFAULT_BIDS_ROOT),
        help=f"Path to BIDS root directory. Default: {DEFAULT_BIDS_ROOT} "
             "(team VM convention). Must exist.",
    )
    parser.add_argument(
        "--out", type=writable_file, default=str(DEFAULT_MANIFEST_PATH),
        help=f"Output manifest JSON path. Default: {DEFAULT_MANIFEST_PATH}. "
             "Parent directory must exist.",
    )
    parser.add_argument(
        "--require-z", type=int, default=DEFAULT_REQUIRE_Z,
        help=f"Drop runs whose native z != this value. Default: {DEFAULT_REQUIRE_Z} "
             "(matches IBC's standard slab; screens out the z=84 ses-00/01 anomaly). "
             "Pass 0 to disable filtering and keep every run.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    require_z = args.require_z if args.require_z > 0 else None
    entries = build_manifest(args.bids_root, require_z=require_z)
    write_manifest(entries, args.bids_root, args.out, require_z=require_z)
    print(f"OK: {len(entries)} runs written to {args.out}")


if __name__ == "__main__":
    _cli()
