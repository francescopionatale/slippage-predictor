"""Aggregate per-run diagnostics into a single comparison view.

Reads every ``results/runs/<run_id>/`` directory produced by
``run_experiments.py`` and writes:

  - results/comparisons/summary.csv      : one row per run (global metrics + cfg)
  - results/comparisons/segments.csv     : long-format segment breakdown
  - results/comparisons/summary.json     : same as summary.csv but JSON
  - results/comparisons/seed_stats.json  : mean ± std across seed-only runs
  - results/comparisons/figures/         : comparison plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import RESULTS_DIR

if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["savefig.dpi"] = 150

RUNS_DIR = RESULTS_DIR / "runs"
COMP_DIR = RESULTS_DIR / "comparisons"
FIG_DIR = COMP_DIR / "figures"
COMP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_runs() -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "history.json"
        if not (cfg_path.exists() and metrics_path.exists() and history_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        history = json.loads(history_path.read_text())
        rows.append({
            "run_id": run_dir.name,
            "config": cfg,
            "metrics": metrics,
            "history": history,
        })
    return rows


def build_summary(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        cfg = r["config"]
        m = r["metrics"]
        h = r["history"]
        rec = {
            "run_id": r["run_id"],
            "seed": cfg["seed"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "alpha": cfg["alpha"],
            "epochs_run": h["n_epochs_run"],
            "best_epoch": h["best_epoch"],
            "best_val_mae_bps": h["best_val_mae"],
            "elapsed_seconds": h.get("elapsed_seconds"),
            "mlp_mae_bps": m["mlp"]["mae_bps"],
            "mlp_rmse_bps": m["mlp"]["rmse_bps"],
            "mlp_med_ae_bps": m["mlp"]["med_ae_bps"],
            "mean_mae_bps": m["mean"]["mae_bps"],
            "linear_mae_bps": m["linear"]["mae_bps"],
            "heuristic_mae_bps": m["heuristic"]["mae_bps"],
        }
        records.append(rec)
    return pd.DataFrame(records).sort_values("run_id").reset_index(drop=True)


def build_segments(rows: list[dict]) -> pd.DataFrame:
    parts = []
    for r in rows:
        seg = pd.DataFrame(r["metrics"]["mlp"]["segment_breakdown"])
        seg.insert(0, "run_id", r["run_id"])
        parts.append(seg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def seed_stats(summary: pd.DataFrame) -> dict:
    """Mean ± std across runs that vary only the seed (default config)."""
    default_mask = (
        (summary["dropout"] == 0.1)
        & (summary["lr"].apply(lambda v: abs(v - 1e-3) < 1e-12))
        & (summary["batch_size"] == 256)
        & (summary["alpha"] == 2.0)
    )
    sub = summary[default_mask]
    if len(sub) < 2:
        return {"note": "fewer than 2 seed-only runs — skipped"}
    out = {"n_runs": int(len(sub)), "seeds": sub["seed"].tolist()}
    for col in ("mlp_mae_bps", "mlp_rmse_bps", "mlp_med_ae_bps",
                "best_val_mae_bps", "best_epoch"):
        out[col] = {
            "mean": float(sub[col].mean()),
            "std": float(sub[col].std(ddof=1)),
            "min": float(sub[col].min()),
            "max": float(sub[col].max()),
        }
    return out


def plot_test_mae_bar(summary: pd.DataFrame, path: Path) -> None:
    sorted_summary = summary.sort_values("mlp_mae_bps").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(sorted_summary))
    width = 0.2
    mlp_bars = ax.bar(x - 1.5 * width, sorted_summary["mlp_mae_bps"], width, label="MLP")
    ax.bar(x - 0.5 * width, sorted_summary["heuristic_mae_bps"], width, label="Heuristic")
    ax.bar(x + 0.5 * width, sorted_summary["linear_mae_bps"],    width, label="Linear")
    ax.bar(x + 1.5 * width, sorted_summary["mean_mae_bps"],      width, label="Mean")
    # Highlight the best MLP run
    if len(mlp_bars):
        mlp_bars[0].set_color("#1f77b4")
        mlp_bars[0].set_edgecolor("#d62728")
        mlp_bars[0].set_linewidth(1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_summary["run_id"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test MAE (bps)")
    ax.set_title("Test MAE across runs — sorted by MLP MAE (best first)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_val_curves(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in rows:
        hist = r["history"]
        ax.plot(
            range(1, hist["n_epochs_run"] + 1),
            hist["val_mae"],
            label=r["run_id"],
            alpha=0.85,
            linewidth=1.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MAE (bps)")
    ax.set_title("Validation MAE over training (per run)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_seed_box(summary: pd.DataFrame, path: Path) -> None:
    default_mask = (
        (summary["dropout"] == 0.1)
        & (summary["lr"].apply(lambda v: abs(v - 1e-3) < 1e-12))
        & (summary["batch_size"] == 256)
        & (summary["alpha"] == 2.0)
    )
    seed_runs = summary[default_mask]
    other = summary[~default_mask]
    if len(seed_runs) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(np.zeros(len(seed_runs)), seed_runs["mlp_mae_bps"],
               s=60, label="default-config seeds", color="tab:blue")
    for i, row in enumerate(other.itertuples()):
        ax.scatter([i + 1], [row.mlp_mae_bps],
                   s=70, marker="D", label=row.run_id)
    seed_mean = seed_runs["mlp_mae_bps"].mean()
    seed_std = seed_runs["mlp_mae_bps"].std(ddof=1)
    ax.axhline(seed_mean, color="tab:blue", linestyle="--", alpha=0.5,
               label=f"seed mean ± std = {seed_mean:.3f} ± {seed_std:.3f}")
    ax.fill_between(
        [-0.5, len(other) + 0.5],
        seed_mean - seed_std,
        seed_mean + seed_std,
        color="tab:blue",
        alpha=0.1,
    )
    ax.set_xticks([0] + list(range(1, len(other) + 1)))
    labels = ["seeds"] + list(other["run_id"])
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test MAE (bps)")
    ax.set_title("Seed sensitivity vs hyperparameter perturbations")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    rows = load_runs()
    if not rows:
        print(f"No runs found under {RUNS_DIR}. Run scripts/run_experiments.py first.")
        return

    summary = build_summary(rows)
    segments = build_segments(rows)
    stats = seed_stats(summary)

    summary.to_csv(COMP_DIR / "summary.csv", index=False)
    segments.to_csv(COMP_DIR / "segments.csv", index=False)
    summary.to_json(COMP_DIR / "summary.json", orient="records", indent=2)
    (COMP_DIR / "seed_stats.json").write_text(json.dumps(stats, indent=2))

    plot_test_mae_bar(summary, FIG_DIR / "test_mae_by_run.png")
    plot_val_curves(rows, FIG_DIR / "val_curves.png")
    plot_seed_box(summary, FIG_DIR / "seed_vs_hparam.png")

    print("\n=== Per-run summary ===")
    cols = ["run_id", "epochs_run", "best_epoch", "best_val_mae_bps",
            "mlp_mae_bps", "mlp_rmse_bps", "heuristic_mae_bps"]
    with pd.option_context("display.width", 140, "display.max_colwidth", 40):
        print(summary[cols].to_string(index=False))

    print("\n=== Seed-only stats (default config) ===")
    print(json.dumps(stats, indent=2))

    print(f"\nWrote summary + figures under: {COMP_DIR}")


if __name__ == "__main__":
    main()
