"""Generate per-run pred-vs-actual scatter plots + a combined grid.

For each run under ``results/runs/<run_id>/`` this loads the saved
checkpoint, re-builds the canonical test split, and writes a
``pred_vs_actual.png`` scatter inside that run directory. A single
6-panel comparison grid is also written to
``results/comparisons/figures/pred_vs_actual_grid.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import SlippageMLP
from paths import RESULTS_DIR
from pipeline import build_full_dataset

if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["savefig.dpi"] = 150

RUNS_DIR = RESULTS_DIR / "runs"
COMP_FIG_DIR = RESULTS_DIR / "comparisons" / "figures"
COMP_FIG_DIR.mkdir(parents=True, exist_ok=True)


def predict(model: SlippageMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


def load_model(run_dir: Path, cfg: dict) -> SlippageMLP:
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    model = SlippageMLP(n_features=ckpt["n_features"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model


def scatter_panel(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vol_q: np.ndarray,
    title: str,
) -> None:
    cmap = plt.get_cmap("viridis").resampled(4)
    for i, lbl in enumerate(["Q1_low_vol", "Q2", "Q3", "Q4_high_vol"]):
        mask = vol_q == lbl
        ax.scatter(
            y_true[mask], y_pred[mask],
            s=6, alpha=0.35, color=cmap(i), label=lbl,
        )
    # Slippage is non-negative; clip the lower bound to 0 so the diagonal
    # reference and the cloud are easier to read.
    lim = max(float(y_true.max()), float(y_pred.max())) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=0.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Actual (bps)", fontsize=8)
    ax.set_ylabel("Predicted (bps)", fontsize=8)
    ax.tick_params(labelsize=7)

    mae = float(np.abs(y_true - y_pred).mean())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax.text(
        0.03, 0.97,
        f"MAE = {mae:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                  boxstyle="round,pad=0.3"),
    )


def main() -> None:
    run_dirs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"No runs under {RUNS_DIR}. Run scripts/run_experiments.py first.")
        return

    print("Building dataset (using cached parquet)...")
    _, _, split = build_full_dataset()
    y_true = split.y_test
    vol_q = pd.qcut(
        split.test_df["vol_rolling"], q=4,
        labels=["Q1_low_vol", "Q2", "Q3", "Q4_high_vol"],
    ).values

    # Per-run pred arrays kept for the grid
    panels: list[tuple[str, np.ndarray, float]] = []

    for run_dir in run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        model = load_model(run_dir, cfg)
        y_pred = predict(model, split.X_test)
        mae = metrics["mlp"]["mae_bps"]
        panels.append((run_dir.name, y_pred, mae))

        fig, ax = plt.subplots(figsize=(6, 5.2))
        scatter_panel(ax, y_true, y_pred, vol_q,
                      f"{run_dir.name}\nTest MAE = {mae:.3f} bps")
        ax.legend(title="vol_quartile", fontsize=7, markerscale=2, loc="upper left")
        fig.tight_layout()
        fig.savefig(run_dir / "pred_vs_actual.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {run_dir / 'pred_vs_actual.png'}  (MAE={mae:.3f})")

    # Combined 2x3 grid
    n = len(panels)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    for ax, (name, y_pred, mae) in zip(axes, panels):
        scatter_panel(ax, y_true, y_pred, vol_q, f"{name}\nMAE = {mae:.3f}")
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="vol_quartile", loc="upper center",
               ncol=4, fontsize=8, markerscale=2,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Predicted vs actual slippage — all runs", fontsize=12, y=1.06)
    fig.tight_layout()
    grid_path = COMP_FIG_DIR / "pred_vs_actual_grid.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrid written to: {grid_path}")


if __name__ == "__main__":
    main()
