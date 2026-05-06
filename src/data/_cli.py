"""Shared CLI helpers — default paths and argparse type validators.

The defaults here are tied to the team's VM layout (raw IBC data lives at
`/srv/fMRI-data`, team-shared derivatives live at `/srv/T4Dbrains/derivatives`).
They make `python -m src.data.build` work without arguments on the VM.

Anywhere else (CI, a contributor's laptop, a different VM), the defaults
won't exist on disk and the existence validators will fail fast with a clear
error — pass the explicit `--bids-root` / `--out-dir` flags to override.

Library entry points (`build_pipeline`, `compute_all`, the Datasets) are NOT
affected by these defaults — only the CLI parsers are. The synthetic test
suite calls those library entry points directly with `tmp_path` and is
unaffected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_BIDS_ROOT = Path("/srv/fMRI-data")
DEFAULT_DERIVATIVES_DIR = Path("/srv/T4Dbrains/derivatives")
DEFAULT_MANIFEST_PATH = DEFAULT_DERIVATIVES_DIR / "manifest.json"

# argparse gotcha: pass these as `default=str(DEFAULT_...)` (not the Path
# itself). argparse only applies `type=` to defaults when the default value
# is a string; non-string defaults are stored verbatim, bypassing
# `existing_dir` / `existing_file` validation. Using a Path default would
# silently accept a missing directory.


def existing_dir(s: str) -> Path:
    """argparse type validator: the path must already exist as a directory."""
    p = Path(s)
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {p}")
    return p


def existing_file(s: str) -> Path:
    """argparse type validator: the path must already exist as a regular file."""
    p = Path(s)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {p}")
    return p


def writable_file(s: str) -> Path:
    """argparse type validator: the file may not yet exist, but its parent must.

    Use for output file paths where we want to fail early if the destination
    directory is wrong (typo, dir wasn't created), without requiring the file
    itself to pre-exist.
    """
    p = Path(s)
    if not p.parent.is_dir():
        raise argparse.ArgumentTypeError(
            f"parent directory does not exist: {p.parent}"
        )
    return p
