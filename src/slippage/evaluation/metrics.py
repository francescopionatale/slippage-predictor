"""Pure metric helpers: MAE / RMSE / MedAE + per-ticker aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def med_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae_bps": mae(y_true, y_pred),
        "rmse_bps": rmse(y_true, y_pred),
        "med_ae_bps": med_ae(y_true, y_pred),
    }


def per_ticker_metrics(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Per-ticker MAE/RMSE/MedAE for each model in ``preds``.

    Returns ``{ticker: {f"{model}_mae_bps": ..., f"{model}_rmse_bps": ...,
    f"{model}_med_ae_bps": ..., "n_samples": int}}``.
    """
    if "ticker" not in test_df.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    tickers = sorted(test_df["ticker"].unique())
    for t in tickers:
        mask = (test_df["ticker"] == t).values
        if mask.sum() == 0:
            continue
        row: dict[str, float] = {"n_samples": int(mask.sum())}
        for name, y_pred in preds.items():
            row[f"{name}_mae_bps"] = mae(y_true[mask], y_pred[mask])
            row[f"{name}_rmse_bps"] = rmse(y_true[mask], y_pred[mask])
            row[f"{name}_med_ae_bps"] = med_ae(y_true[mask], y_pred[mask])
        out[t] = row
    return out
