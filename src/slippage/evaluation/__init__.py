"""Evaluation: global metrics, segment breakdown, per-ticker."""

from slippage.evaluation.metrics import (
    global_metrics,
    mae,
    med_ae,
    per_ticker_metrics,
    rmse,
)
from slippage.evaluation.segments import evaluate_all, segment_breakdown

__all__ = [
    "mae",
    "rmse",
    "med_ae",
    "global_metrics",
    "per_ticker_metrics",
    "segment_breakdown",
    "evaluate_all",
]
