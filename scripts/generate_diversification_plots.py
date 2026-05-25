"""Generate the cross-ticker diversification plots.

Reads ``results/metrics.json`` for the per-ticker MAE breakdown and
``results/walk_forward/folds.csv`` for the timeline, then writes:

  - results/figures/mae_by_ticker.png
  - results/figures/walk_forward_timeline.png
"""

from __future__ import annotations

import json

import pandas as pd

from paths import RESULTS_DIR
from viz import plot_mae_by_ticker, plot_walk_forward_timeline

METRICS = RESULTS_DIR / "metrics.json"
FOLDS = RESULTS_DIR / "walk_forward" / "folds.csv"


def main() -> None:
    if METRICS.exists():
        metrics = json.loads(METRICS.read_text())
        per_ticker = metrics.get("per_ticker", {})
        if per_ticker:
            print("Writing mae_by_ticker.png ...")
            plot_mae_by_ticker(per_ticker)
        else:
            print(f"[skip] no per_ticker section in {METRICS} — run python -m evaluate first")
    else:
        print(f"[skip] {METRICS} not found — run python -m evaluate first")

    if FOLDS.exists():
        folds_df = pd.read_csv(FOLDS, parse_dates=["train_start", "train_end",
                                                   "test_start", "test_end"])
        required = {"train_start", "test_start", "test_end"}
        if required.issubset(folds_df.columns):
            print("Writing walk_forward_timeline.png ...")
            plot_walk_forward_timeline(folds_df)
        else:
            missing = sorted(required - set(folds_df.columns))
            print(f"[skip] {FOLDS} missing columns {missing} — rerun run_walk_forward.py")
    else:
        print(f"[skip] {FOLDS} not found — run scripts/run_walk_forward.py first")

    print("Done.")


if __name__ == "__main__":
    main()
