"""Centralised filesystem paths for the project.

Importing this module has **no side effects** — it only computes path
constants. Call :func:`ensure_dirs` (or rely on the writers, which create
their own parent directory) before writing artifacts. The project root can
be relocated via the ``SLIPPAGE_ROOT`` environment variable, which is
convenient for Docker/CI or read-only checkouts.
"""

from __future__ import annotations

import os
from pathlib import Path

# __file__ is src/slippage/paths.py — three .parent hops reach the repo root,
# unless SLIPPAGE_ROOT overrides it.
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = Path(os.environ.get("SLIPPAGE_ROOT", _DEFAULT_ROOT)).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"

ALL_DIRS = (
    RAW_DIR, PROCESSED_DIR,
    RESULTS_DIR, FIGURES_DIR,
    CHECKPOINTS_DIR, SCALERS_DIR,
)


def ensure_dirs() -> None:
    """Create all project output directories if they don't exist.

    Called explicitly by the CLIs/scripts that write artifacts, so merely
    importing the package never touches the filesystem.
    """
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
