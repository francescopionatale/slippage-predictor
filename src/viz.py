"""Visualisation functions shared across notebooks."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import FIGURES_DIR


def _apply_style() -> None:
    """Apply the shared matplotlib style.

    Idempotent — safe to call from multiple modules. Falls back gracefully
    if the seaborn style isn't bundled (older matplotlib).
    """
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["savefig.dpi"] = 150


_apply_style()

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

    mae = float(np.abs(y_true - y_pred).mean())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax.text(
        0.03, 0.97,
        f"MAE = {mae:.3f} bps\nR² = {r2:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                  boxstyle="round,pad=0.4"),
    )

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


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_as: str | None = "residuals_vs_predicted.png",
) -> plt.Figure:
    """Scatter of residuals vs predicted values with a 20-bin binned-mean smoother."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=6, alpha=0.15, color="steelblue")
    ax.axhline(0.0, color="red", lw=1.0, linestyle="--")

    # 20-bin binned mean (no statsmodels dependency)
    bins = np.linspace(y_pred.min(), y_pred.max(), 21)
    bin_idx = np.digitize(y_pred, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 19)
    bin_means_x, bin_means_y = [], []
    for b in range(20):
        mask = bin_idx == b
        if mask.sum() >= 2:
            bin_means_x.append(float(y_pred[mask].mean()))
            bin_means_y.append(float(residuals[mask].mean()))
    if len(bin_means_x) > 1:
        ax.plot(bin_means_x, bin_means_y, color="orange", lw=1.8, label="binned mean")
        ax.legend(fontsize=8)

    mean_res = float(residuals.mean())
    std_res = float(residuals.std())
    ax.text(
        0.03, 0.97,
        f"mean = {mean_res:.3f}\nstd = {std_res:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                  boxstyle="round,pad=0.4"),
    )
    ax.set_xlabel("Predicted slippage (bps)")
    ax.set_ylabel("Residual (actual − predicted, bps)")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_as: str | None = "residual_distribution.png",
) -> plt.Figure:
    """Histogram of residuals overlaid with a fitted normal PDF."""
    from scipy.stats import kurtosis, norm, skew  # scipy is already a dependency

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=60, density=True, color="steelblue", alpha=0.75)
    x = np.linspace(residuals.min(), residuals.max(), 300)
    mu, sigma = float(residuals.mean()), float(residuals.std())
    ax.plot(x, norm.pdf(x, mu, sigma), color="red", lw=1.8, label="normal fit")
    ax.legend(fontsize=8)

    skw = float(skew(residuals))
    kurt = float(kurtosis(residuals))
    ax.text(
        0.97, 0.97,
        f"skew = {skw:.3f}\nkurtosis = {kurt:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                  boxstyle="round,pad=0.4"),
    )
    ax.set_xlabel("Residual (bps)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig


def plot_qq_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_as: str | None = "qq_residuals.png",
) -> plt.Figure:
    """Normal Q-Q plot of residuals."""
    from scipy.stats import probplot

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Residuals")
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
        ax2.axvline(
            history["best_epoch"] - 1, color="red", ls="--", lw=1,
            label=f"best epoch {history['best_epoch']}",
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (bps)")
    ax2.set_title("Validation MAE")
    ax2.legend()

    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig
