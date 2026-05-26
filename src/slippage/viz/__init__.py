"""Plotting functions: training history, diagnostics, and features."""

from slippage.paths import FIGURES_DIR
from slippage.viz._style import _apply_style
from slippage.viz.diagnostics import (
    plot_mae_by_ticker,
    plot_pred_vs_actual,
    plot_qq_residuals,
    plot_residual_distribution,
    plot_residuals_vs_predicted,
    plot_walk_forward_timeline,
)
from slippage.viz.features import (
    plot_alpha_sensitivity,
    plot_error_by_segment,
    plot_feature_importance,
    plot_spread_cs_distribution,
)
from slippage.viz.training import plot_training_history

# Apply the shared matplotlib style on first import — preserves the
# import-time side effect of the pre-refactor flat viz module.
_apply_style()

__all__ = [
    "FIGURES_DIR",
    # diagnostics
    "plot_pred_vs_actual",
    "plot_residuals_vs_predicted",
    "plot_residual_distribution",
    "plot_qq_residuals",
    "plot_mae_by_ticker",
    "plot_walk_forward_timeline",
    # features
    "plot_alpha_sensitivity",
    "plot_feature_importance",
    "plot_spread_cs_distribution",
    "plot_error_by_segment",
    # training
    "plot_training_history",
]
