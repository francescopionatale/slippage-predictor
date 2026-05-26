"""Plotting functions for training history, diagnostics, and features."""

from slippage.viz._legacy import *  # noqa: F401,F403
from slippage.viz._legacy import (
    FIGURES_DIR,
    plot_alpha_sensitivity,
    plot_error_by_segment,
    plot_feature_importance,
    plot_mae_by_ticker,
    plot_pred_vs_actual,
    plot_qq_residuals,
    plot_residual_distribution,
    plot_residuals_vs_predicted,
    plot_spread_cs_distribution,
    plot_training_history,
    plot_walk_forward_timeline,
)

__all__ = [
    "FIGURES_DIR",
    "plot_pred_vs_actual",
    "plot_residuals_vs_predicted",
    "plot_residual_distribution",
    "plot_qq_residuals",
    "plot_mae_by_ticker",
    "plot_walk_forward_timeline",
    "plot_training_history",
    "plot_alpha_sensitivity",
    "plot_feature_importance",
    "plot_spread_cs_distribution",
    "plot_error_by_segment",
]
