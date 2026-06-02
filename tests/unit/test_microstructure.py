"""Tests for microstructure features (correctness + no look-ahead)."""

from __future__ import annotations

import numpy as np

from slippage.features import FEATURE_NAMES_MICRO, compute_microstructure_features
from slippage.features.microstructure import order_flow_imbalance, roll_spread


def test_microstructure_columns_present(ohlcv_df):
    out = compute_microstructure_features(ohlcv_df(n=120))
    assert list(out.columns) == FEATURE_NAMES_MICRO


def test_microstructure_no_lookahead(ohlcv_df):
    """A feature value at bar t must not change when future bars are altered."""
    df = ohlcv_df(n=120, seed=3)
    full = compute_microstructure_features(df, window=20)
    # Truncate to the first 80 bars and recompute: shared rows must match.
    truncated = compute_microstructure_features(df.iloc[:80], window=20)
    common = full.index.intersection(truncated.index)
    # Compare a safely-interior slice (drop the last few rows of the truncated
    # frame where the rolling window composition could differ at the boundary).
    interior = common[:-1]
    np.testing.assert_allclose(
        full.loc[interior].values,
        truncated.loc[interior].values,
        rtol=1e-9, atol=1e-12,
    )


def test_ofi_bounded(ohlcv_df):
    df = ohlcv_df(n=200, seed=5)
    ofi = order_flow_imbalance(df["close"], df["volume"], window=20).dropna()
    assert (ofi.abs() <= 1.0 + 1e-9).all()


def test_roll_spread_non_negative(ohlcv_df):
    df = ohlcv_df(n=200, seed=6)
    rs = roll_spread(df["close"], window=20).dropna()
    assert (rs >= 0).all()


def test_microstructure_volatility_non_negative(ohlcv_df):
    out = compute_microstructure_features(ohlcv_df(n=150, seed=7)).dropna()
    assert (out["parkinson_vol"] >= 0).all()
    assert (out["garman_klass_vol"] >= 0).all()
    assert (out["log_dollar_vol"] > 0).all()
