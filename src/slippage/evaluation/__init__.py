"""Evaluation: global metrics, segment breakdown, per-ticker."""

from slippage.evaluation.baselines import build_baselines
from slippage.evaluation.metrics import (
    global_metrics,
    mae,
    med_ae,
    per_ticker_metrics,
    rmse,
)
from slippage.evaluation.segments import evaluate_all, segment_breakdown
from slippage.evaluation.significance import (
    DMResult,
    block_bootstrap_mae_ci,
    compare_against_reference,
    diebold_mariano,
)

__all__ = [
    "mae",
    "rmse",
    "med_ae",
    "global_metrics",
    "per_ticker_metrics",
    "segment_breakdown",
    "evaluate_all",
    "build_baselines",
    "diebold_mariano",
    "block_bootstrap_mae_ci",
    "compare_against_reference",
    "DMResult",
]
