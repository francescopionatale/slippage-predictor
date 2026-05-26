"""Centralised filesystem paths for the project."""

from __future__ import annotations

from pathlib import Path

# __file__ is src/slippage/paths.py — three .parent hops reach the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"

for _d in (
    RAW_DIR, PROCESSED_DIR,
    RESULTS_DIR, FIGURES_DIR,
    CHECKPOINTS_DIR, SCALERS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
