"""Synthetic slippage proxy construction and alpha calibration.

Since real execution data is unavailable, slippage is constructed from
OHLCV bars using the next bar's average price as the execution benchmark
plus a market-impact penalty scaled by volatility and order size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_proxy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    alpha: float = 2.0,
) -> pd.DataFrame:
    """Build the synthetic slippage label for every row in ``features``.

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame with a DatetimeIndex.
    features:
        Output of ``add_synthetic_orders``; must contain columns
        ``side``, ``order_size_fraction``, and ``vol_rolling``.
    alpha:
        Market-impact scale factor. Higher alpha → larger impact penalty.

    Returns
    -------
    DataFrame equal to ``features`` with added ``slippage_bps`` and
    ``arrival_price`` columns. Rows where the next bar is unavailable
    (last bar of the series) are dropped.
    """
    # Align features to the OHLCV index (features may be a subset after dropna)
    shared = features.index.intersection(df.index)
    feats = features.loc[shared].copy()

    arrival = df["close"].loc[shared]
    next_o = df["open"].shift(-1).loc[shared]
    next_h = df["high"].shift(-1).loc[shared]
    next_l = df["low"].shift(-1).loc[shared]
    next_c = df["close"].shift(-1).loc[shared]
    base_exec = (next_o + next_h + next_l + next_c) / 4.0

    valid = base_exec.notna()
    feats = feats.loc[valid]
    arrival = arrival.loc[valid]
    base_exec = base_exec.loc[valid]

    impact = alpha * feats["order_size_fraction"] * feats["vol_rolling"] * arrival
    exec_price = base_exec + feats["side"] * impact
    slippage = feats["side"] * (exec_price - arrival)

    feats = feats.copy()
    feats["slippage_bps"] = 10_000.0 * slippage / arrival
    feats["arrival_price"] = arrival
    return feats


# ---------------------------------------------------------------------------
# Alpha calibration
# ---------------------------------------------------------------------------

def calibrate_alpha(
    df: pd.DataFrame,
    features_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    alphas: list[float] | None = None,
) -> tuple[float, dict[float, float]]:
    """Find the alpha that minimises MAE on the validation set.

    The MAE is computed between the proxy ``slippage_bps`` built with each
    candidate alpha and the reference labels ``y_val``. The alpha with the
    lowest MAE is selected.

    To avoid issues with duplicate timestamps in ``features_val``, both
    inputs are matched by position (not by label). The caller must ensure
    ``features_val`` and ``y_val`` are aligned row-by-row.

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame (same one used to build the reference proxy).
    features_val:
        Validation-fold features (output of ``add_synthetic_orders``
        restricted to the validation period). Must NOT contain duplicate
        rows from label-based indexing; pass an iloc slice.
    y_val:
        Reference slippage labels, length must equal ``len(features_val)``
        after dropping bars without t+1.
    alphas:
        Candidate alpha values to sweep.

    Returns
    -------
    best_alpha, dict mapping alpha -> MAE
    """
    if alphas is None:
        alphas = [0.5, 1.0, 2.0, 5.0, 10.0]

    y_val_arr = np.asarray(y_val)

    sensitivity: dict[float, float] = {}
    for a in alphas:
        proxy = build_proxy(df, features_val, alpha=a)
        # Match by position (proxy may have dropped tail bars)
        n = min(len(proxy), len(y_val_arr))
        if n == 0:
            sensitivity[a] = float("nan")
            continue
        mae = float(np.abs(proxy["slippage_bps"].values[:n] - y_val_arr[:n]).mean())
        sensitivity[a] = mae

    best_alpha = min(sensitivity, key=lambda a: sensitivity[a])
    return best_alpha, sensitivity
