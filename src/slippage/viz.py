"""Visualisation functions shared across notebooks."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slippage.paths import FIGURES_DIR

_PALETTE = "viridis"


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime: np.ndarray | None = None,
    regime_label: str = "vol_quartile",
    title: str = "Predicted vs Actual Slippage",
    save_as: str | None = "pred_vs_actual.png",
) -> plt.Figure:
    """Scatter of predicted vs actual slippage coloured by market regime."""
    fig, ax = plt.subplots(figsize=(7, 6))

    if regime is not None:
        unique = np.unique(regime)
        cmap = mpl.colormaps[_PALETTE].resampled(len(unique))
        for i, lbl in enumerate(unique):
            mask = regime == lbl
            ax.scatter(
                y_true[mask], y_pred[mask],
                s=8, alpha=0.4, color=cmap(i), label=str(lbl),
            )
        ax.legend(title=regime_label, fontsize=8, markerscale=2)
    else:
        ax.scatter(y_true, y_pred, s=8, alpha=0.3, color="steelblue")

    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="perfect prediction")
    ax.set_xlabel("Actual slippage (bps)")
    ax.set_ylabel("Predicted slippage (bps)")
    ax.set_title(title)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()

    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig


def plot_error_by_segment(
    breakdown_df: pd.DataFrame,
    metric: str = "mae_bps",
    save_as: str | None = "error_by_segment.png",
) -> plt.Figure:
    """Faceted bar chart of MAE by segment bucket."""
    segments = breakdown_df["segment"].unique()
    n_seg = len(segments)
    fig, axes = plt.subplots(1, n_seg, figsize=(4 * n_seg, 5), sharey=False)
    if n_seg == 1:
        axes = [axes]

    for ax, seg in zip(axes, segments):
        sub = breakdown_df[breakdown_df["segment"] == seg].sort_values(metric)
        ax.barh(sub["bucket"], sub[metric], color="steelblue")
        ax.set_title(seg.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel(metric.replace("_", " "))
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Error Breakdown by Market Regime", fontsize=12, y=1.02)
    fig.tight_layout()

    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return fig


def plot_alpha_sensitivity(
    alphas: list[float],
    val_maes: list[float],
    best_alpha: float | None = None,
    save_as: str | None = "alpha_sensitivity.png",
) -> plt.Figure:
    """Line plot showing val MAE vs alpha (impact scale factor)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, val_maes, "o-", color="steelblue", lw=2)
    if best_alpha is not None:
        idx = alphas.index(best_alpha) if best_alpha in alphas else None
        if idx is not None:
            ax.axvline(best_alpha, color="red", ls="--", lw=1.5, label=f"best α={best_alpha}")
            ax.legend()
    ax.set_xlabel("α (impact scale)")
    ax.set_ylabel("Validation MAE (bps)")
    ax.set_title("Alpha Sensitivity — Proxy Calibration")
    ax.set_xscale("log")
    fig.tight_layout()

    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig


def plot_feature_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    seed: int = 42,
    save_as: str | None = "feature_importance.png",
) -> plt.Figure:
    """Permutation importance for a PyTorch model.

    The model is wrapped in a sklearn-compatible predict function so that
    sklearn's permutation_importance can be used directly.
    """
    import torch
    from sklearn.inspection import permutation_importance
    from sklearn.base import BaseEstimator

    class _TorchWrapper(BaseEstimator):
        def __init__(self, net) -> None:
            self.net = net

        def fit(self, X, y):
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            self.net.eval()
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32)
                return self.net(t).squeeze(1).numpy()

        def score(self, X: np.ndarray, y: np.ndarray) -> float:
            preds = self.predict(X)
            return -float(np.abs(preds - y).mean())

    wrapper = _TorchWrapper(model)
    result = permutation_importance(
        wrapper, X_val, y_val, n_repeats=n_repeats,
        random_state=seed, scoring="neg_mean_absolute_error",
    )
    importance = result.importances_mean
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Mean decrease in MAE (bps) — permutation importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()

    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig


def plot_training_history(
    history: dict,
    save_as: str | None = "training_history.png",
) -> plt.Figure:
    """Plot train loss and val MAE over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(history["train_loss"], label="Train Huber loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(history["val_mae"], color="orange", label="Val MAE (bps)")
    if "best_epoch" in history:
        ax2.axvline(history["best_epoch"] - 1, color="red", ls="--", lw=1, label=f"best epoch {history['best_epoch']}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (bps)")
    ax2.set_title("Validation MAE")
    ax2.legend()

    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig
