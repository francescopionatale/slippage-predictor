"""Execute expanding-window walk-forward CV on the real dataset.

Writes:
  - results/walk_forward/folds.csv     : one row per (fold, model)
  - results/walk_forward/summary.csv   : mean ± std per model
  - results/walk_forward/summary.json  : same as JSON for the report builder
"""

from __future__ import annotations

import time

from paths import RESULTS_DIR
from pipeline import build_full_dataset
from walk_forward import summarise, walk_forward_cv

OUT_DIR = RESULTS_DIR / "walk_forward"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Building dataset (cached parquet)...")
    data, proxy_all, _ = build_full_dataset()
    del data

    print(f"Proxy rows: {len(proxy_all):,}")
    print("Running walk-forward CV (5 folds, 2-month test windows)...")
    t0 = time.time()
    folds = walk_forward_cv(
        proxy_all, n_folds=5, test_months=2.0,
        seed=42, dropout=0.1, verbose=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    folds.to_csv(OUT_DIR / "folds.csv", index=False)
    summary = summarise(folds)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    summary.to_json(OUT_DIR / "summary.json", orient="records", indent=2)

    print("\n=== Walk-forward summary (mean ± std across folds) ===")
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:10s}  "
            f"MAE = {row['mae_bps_mean']:.3f} ± {row['mae_bps_std']:.3f}  "
            f"RMSE = {row['rmse_bps_mean']:.3f} ± {row['rmse_bps_std']:.3f}"
        )
    print(f"\nArtifacts under {OUT_DIR}")


if __name__ == "__main__":
    main()
