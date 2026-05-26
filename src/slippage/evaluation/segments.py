"""Segment breakdown (size / vol / tod / side / vol-regime) and the
full evaluation orchestrator that writes results/metrics.json."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from slippage.evaluation.metrics import global_metrics, per_ticker_metrics
from slippage.paths import RESULTS_DIR


def segment_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute error breakdown across five market-regime dimensions.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of true and predicted slippage_bps for the test set, aligned
        row-by-row with ``test_df``.
    test_df:
        DataFrame with the test-set rows from the proxy (same length and
        ordering as ``y_true``). Must contain columns ``order_size_fraction``,
        ``vol_rolling``, ``side``, ``vol_ratio``. Index must be a
        DatetimeIndex for time-of-day bucketing.

    Returns
    -------
    DataFrame with columns: segment, bucket, n_samples, mae_bps, rmse_bps, med_ae_bps.
    Buckets are emitted in the natural order of the underlying categorical
    (not alphabetical).
    """
    if len(test_df) != len(y_true):
        raise ValueError(
            f"test_df length ({len(test_df)}) must match y_true length ({len(y_true)}). "
            "Pass split.test_df directly from temporal_split."
        )

    errors = np.abs(y_true - y_pred)
    records = []

    def _add_records(segment: str, labels) -> None:
        # If labels is a pandas Categorical, iterate categories to keep order.
        if isinstance(labels, pd.Categorical):
            unique_labels = list(labels.categories)
            label_arr = np.asarray(labels)
        else:
            unique_labels = list(pd.unique(labels))
            label_arr = np.asarray(labels)
        for lbl in unique_labels:
            mask = label_arr == lbl
            if mask.sum() == 0:
                continue
            records.append({
                "segment": segment,
                "bucket": str(lbl),
                "n_samples": int(mask.sum()),
                "mae_bps": float(errors[mask].mean()),
                "rmse_bps": float(np.sqrt((errors[mask] ** 2).mean())),
                "med_ae_bps": float(np.median(errors[mask])),
            })

    # 1. Order size quartiles
    size_q = pd.qcut(test_df["order_size_fraction"], q=4,
                     labels=["Q1_small", "Q2", "Q3", "Q4_large"])
    _add_records("size_quartile", size_q.values)

    # 2. Volatility quartiles
    vol_q = pd.qcut(test_df["vol_rolling"], q=4,
                    labels=["Q1_low_vol", "Q2", "Q3", "Q4_high_vol"])
    _add_records("vol_quartile", vol_q.values)

    # 3. Time of day (Eastern time buckets, ordered open → mid → close)
    idx = test_df.index
    if idx.tz is not None:
        eastern = idx.tz_convert("America/New_York")
    else:
        eastern = idx
    hour = eastern.hour + eastern.minute / 60.0
    tod_raw = np.where(hour < 10.5, "open",
                       np.where(hour >= 15.0, "close", "mid"))
    tod_cat = pd.Categorical(tod_raw, categories=["open", "mid", "close"], ordered=True)
    _add_records("time_of_day", tod_cat)

    # 4. Buy vs sell
    side_raw = np.where(test_df["side"].values == 1.0, "buy", "sell")
    side_cat = pd.Categorical(side_raw, categories=["buy", "sell"], ordered=False)
    _add_records("side", side_cat)

    # 5. Volume regime (normal vs anomalous)
    vol_regime_raw = np.where(test_df["vol_ratio"].values >= 2.0,
                              "high_vol_ratio", "normal")
    vol_regime_cat = pd.Categorical(vol_regime_raw,
                                    categories=["normal", "high_vol_ratio"],
                                    ordered=True)
    _add_records("volume_regime", vol_regime_cat)

    return pd.DataFrame(records)


def evaluate_all(
    split,
    model_predict_fn,
    baselines: dict,
) -> dict:
    """Run full evaluation and save results to results/metrics.json.

    Parameters
    ----------
    split:
        SplitData object (carries test_df internally).
    model_predict_fn:
        Callable that takes X (numpy) and returns y_pred (numpy).
    baselines:
        Dict mapping name -> predict callable (takes X, returns y_pred).

    Returns
    -------
    Nested dict of metrics (also saved to disk).
    """
    results: dict = {}

    y_test = split.y_test

    # MLP metrics
    y_pred_mlp = model_predict_fn(split.X_test)
    results["mlp"] = global_metrics(y_test, y_pred_mlp)
    results["mlp"]["segment_breakdown"] = segment_breakdown(
        y_test, y_pred_mlp, split.test_df
    ).to_dict(orient="records")

    # Baseline metrics
    baseline_preds: dict[str, np.ndarray] = {}
    for name, pred_fn in baselines.items():
        y_pred_b = pred_fn(split.X_test)
        results[name] = global_metrics(y_test, y_pred_b)
        baseline_preds[name] = y_pred_b

    # Per-ticker breakdown (MLP + heuristic only — the two we compare in the report)
    preds_for_ticker = {"mlp": y_pred_mlp}
    if "heuristic" in baseline_preds:
        preds_for_ticker["heuristic"] = baseline_preds["heuristic"]
    results["per_ticker"] = per_ticker_metrics(split.test_df, y_test, preds_for_ticker)

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
