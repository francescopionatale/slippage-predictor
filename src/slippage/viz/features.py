"""Feature-engineering and segment-analysis plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slippage.paths import FIGURES_DIR


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
    from sklearn.base import BaseEstimator
    from sklearn.inspection import permutation_importance

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


def plot_spread_cs_distribution(
    proxy_df: pd.DataFrame,
    bins: int = 60,
    save_as: str | None = "spread_cs_distribution.png",
) -> plt.Figure:
    """Faceted histogram of the Corwin-Schultz spread estimate per ticker.

    The CS spread estimate clips negative values to zero, so a non-trivial
    fraction of bars typically read exactly 0. We annotate that fraction
    on each subplot so the reader knows how much of the distribution is
    effectively a point mass.
    """
    if "ticker" not in proxy_df.columns:
        raise ValueError("plot_spread_cs_distribution expects a 'ticker' column")

    tickers = sorted(proxy_df["ticker"].unique())
    n = len(tickers)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows),
                             sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, ticker in zip(axes_flat, tickers):
        sub = proxy_df.loc[proxy_df["ticker"] == ticker, "spread_cs"].dropna()
        zero_frac = float((sub == 0).mean())
        # Clip for the bin range so a few large outliers don't squash the histogram
        clipped = sub.clip(upper=float(sub.quantile(0.99)))
        ax.hist(clipped, bins=bins, color="steelblue", alpha=0.85)
        ax.set_title(ticker, fontsize=10)
        ax.set_xlabel("spread_cs")
        ax.set_ylabel("count")
        ax.text(
            0.97, 0.95, f"{zero_frac:.1%} zeros\nn={len(sub):,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ccc"),
        )
    # Hide any extra axes if tickers don't fill the grid
    for ax in axes_flat[len(tickers):]:
        ax.set_visible(False)

    fig.suptitle("Corwin–Schultz spread distribution per ticker", fontsize=12, y=1.0)
    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return fig
