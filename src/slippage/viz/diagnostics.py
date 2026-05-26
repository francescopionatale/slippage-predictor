"""Model-diagnostic plots: pred-vs-actual, residuals, per-ticker, walk-forward."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slippage.paths import FIGURES_DIR
from slippage.viz._style import _PALETTE


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


def plot_mae_by_ticker(
    per_ticker: dict[str, dict[str, float]],
    save_as: str | None = "mae_by_ticker.png",
) -> plt.Figure:
    """Grouped bar chart: per-ticker MLP MAE vs Heuristic MAE, sorted by MLP MAE."""
    if not per_ticker:
        raise ValueError("plot_mae_by_ticker: per_ticker dict is empty")

    tickers = sorted(per_ticker.keys(),
                     key=lambda t: per_ticker[t].get("mlp_mae_bps", float("inf")))
    mlp_vals = [per_ticker[t].get("mlp_mae_bps", float("nan")) for t in tickers]
    heur_vals = [per_ticker[t].get("heuristic_mae_bps", float("nan")) for t in tickers]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tickers))
    width = 0.4
    ax.bar(x - width / 2, mlp_vals, width, label="MLP", color="steelblue")
    ax.bar(x + width / 2, heur_vals, width, label="Heuristic", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=30, ha="right")
    ax.set_ylabel("MAE (bps)")
    ax.set_title("Per-ticker MAE: MLP vs Heuristic (sorted by MLP MAE)")
    ax.legend()
    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig


def plot_walk_forward_timeline(
    folds_df: pd.DataFrame,
    save_as: str | None = "walk_forward_timeline.png",
) -> plt.Figure:
    """Gantt-style timeline of expanding-window walk-forward folds.

    Each fold draws a blue training-window bar and an orange test-window bar.
    Test bars are annotated with the MLP MAE of that fold.
    """
    required = {"fold", "model", "train_start", "train_end",
                "test_start", "test_end", "mae_bps"}
    missing = required - set(folds_df.columns)
    if missing:
        raise ValueError(f"plot_walk_forward_timeline: missing columns {sorted(missing)}")

    mlp = folds_df[folds_df["model"] == "mlp"].sort_values("fold").reset_index(drop=True)
    if mlp.empty:
        raise ValueError("plot_walk_forward_timeline: no rows for model='mlp'")

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(mlp) + 2.0))
    for _, row in mlp.iterrows():
        ts = pd.Timestamp(row["train_start"])
        te = pd.Timestamp(row["train_end"])
        vs = pd.Timestamp(row["test_start"])
        ve = pd.Timestamp(row["test_end"])
        y = int(row["fold"])
        ax.barh(y, (te - ts).total_seconds() / 86_400, left=ts,
                height=0.55, color="steelblue", alpha=0.5, label="train" if y == 0 else None)
        ax.barh(y, (ve - vs).total_seconds() / 86_400, left=vs,
                height=0.55, color="darkorange", alpha=0.85, label="test" if y == 0 else None)
        mid = vs + (ve - vs) / 2
        ax.text(mid, y, f"MAE={row['mae_bps']:.2f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(sorted(mlp["fold"].unique().tolist()))
    ax.set_yticklabels([f"Fold {int(f)}" for f in sorted(mlp["fold"].unique().tolist())])
    ax.invert_yaxis()
    ax.set_xlabel("Date")
    ax.set_title("Walk-forward expanding-window folds (train + test windows, MLP MAE bps)")
    ax.legend(loc="lower right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig
