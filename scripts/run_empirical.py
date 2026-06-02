"""Real-data track: train against an empirical illiquidity target.

Unlike ``train.py`` (which fits the synthetic, circular proxy), this script
builds a forward-looking Amihud-illiquidity target (``proxy.empirical``)
and trains the MLP + baselines to *forecast* it from current market
features. Because the label is a noisy estimate of a real quantity the
model does not directly observe, the resulting error is genuine predictive
error — not formula re-derivation.

Writes ``results/empirical_metrics.json`` with MAE + bootstrap CIs and a
Diebold-Mariano test of the MLP against each baseline.
"""

from __future__ import annotations

import json

import pandas as pd

from slippage.config import Config
from slippage.data import download_ohlcv, temporal_split
from slippage.evaluation import build_baselines, compare_against_reference, global_metrics
from slippage.features import (
    FEATURE_NAMES_MARKET,
    FEATURE_NAMES_MICRO,
    compute_market_features,
    compute_microstructure_features,
)
from slippage.paths import RESULTS_DIR, ensure_dirs
from slippage.proxy import build_empirical_target
from slippage.training import predict, train

# Market features + the microstructure bundle + Kyle's lambda — the genuine
# liquidity signals for forecasting illiquidity (no synthetic order size).
EMPIRICAL_FEATURES = FEATURE_NAMES_MARKET + FEATURE_NAMES_MICRO + ["kyle_lambda"]


def build_empirical_dataset(config: Config) -> pd.DataFrame:
    data = download_ohlcv(config.data.tickers)
    parts = []
    for ticker, df in data.items():
        mf = compute_market_features(df, vol_window=config.features.vol_window)
        micro = compute_microstructure_features(df, window=config.features.vol_window)
        mf = mf.join(micro, how="inner")
        target = build_empirical_target(df, mf, window=config.features.vol_window)
        target["ticker"] = ticker
        parts.append(target)
    return pd.concat(parts).sort_index()


def main() -> None:
    ensure_dirs()
    config = Config.load()
    print("Building empirical (forward-illiquidity) dataset...")
    proxy_all = build_empirical_dataset(config)
    split = temporal_split(proxy_all, feature_cols=EMPIRICAL_FEATURES)
    print(f"Train {len(split.X_train):,}  Val {len(split.X_val):,}  Test {len(split.X_test):,}")

    model, _ = train(
        split, n_features=len(EMPIRICAL_FEATURES),
        epochs=config.training.epochs, verbose=True, checkpoint_path=None,
    )
    y_pred = predict(model, split.X_test)

    # No synthetic order size on this track → skip the √-impact heuristic.
    baselines = build_baselines(split, EMPIRICAL_FEATURES, include_heuristic=False)
    predictions = {"mlp": y_pred}
    predictions.update({name: fn(split.X_test) for name, fn in baselines.items()})

    print("\n=== Empirical-target test metrics (bps) ===")
    for name, pred in predictions.items():
        m = global_metrics(split.y_test, pred)
        print(f"  {name:8s}  MAE={m['mae_bps']:.3f}  RMSE={m['rmse_bps']:.3f}")

    comparison = compare_against_reference(split.y_test, predictions, reference="mlp")
    print("\n=== MLP vs baselines (Diebold-Mariano) ===")
    for name, entry in comparison.items():
        if "dm_vs_reference" in entry:
            dm = entry["dm_vs_reference"]
            verdict = "MLP better" if dm["favored_reference"] else "not significant / worse"
            print(f"  vs {name:8s}  p={dm['p_value']:.4f}  -> {verdict}")

    out = {
        "metrics": {n: global_metrics(split.y_test, p) for n, p in predictions.items()},
        "comparison": comparison,
        "n_test": int(len(split.y_test)),
        "features": EMPIRICAL_FEATURES,
    }
    path = RESULTS_DIR / "empirical_metrics.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
