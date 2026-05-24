"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from features import (
    FEATURE_NAMES,
    add_synthetic_orders,
    compute_market_features,
    _corwin_schultz_spread,
    _time_of_day_encoding,
)


def test_market_features_no_nan(ohlcv_df):
    feats = compute_market_features(ohlcv_df())
    assert feats.isna().sum().sum() == 0


def test_market_features_no_lookahead(ohlcv_df):
    """Features at index i must not use data from index > i."""
    df = ohlcv_df()
    feats_full = compute_market_features(df)
    feats_trunc = compute_market_features(df.iloc[:61])
    shared = feats_full.index.intersection(feats_trunc.index)
    assert len(shared) > 0
    pd.testing.assert_frame_equal(
        feats_full.loc[shared].reset_index(drop=True),
        feats_trunc.loc[shared].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )


def test_corwin_schultz_non_negative(ohlcv_df):
    df = ohlcv_df(n=200)
    spread = _corwin_schultz_spread(df["high"], df["low"])
    assert (spread.dropna() >= 0).all()


def test_corwin_schultz_wider_bar_higher_spread():
    """Wider H-L range should generally produce a higher spread estimate."""
    n = 50
    index = pd.date_range("2024-01-02", periods=n, freq="1h", tz="UTC")
    close = np.ones(n) * 100.0
    high_narrow = pd.Series(close * 1.001, index=index)
    low_narrow = pd.Series(close * 0.999, index=index)
    high_wide = pd.Series(close * 1.02, index=index)
    low_wide = pd.Series(close * 0.98, index=index)
    narrow = _corwin_schultz_spread(high_narrow, low_narrow).dropna().mean()
    wide = _corwin_schultz_spread(high_wide, low_wide).dropna().mean()
    assert wide > narrow


def test_add_synthetic_orders_balanced(ohlcv_df):
    feats = compute_market_features(ohlcv_df(n=50))
    expanded = add_synthetic_orders(feats, rng=np.random.default_rng(0))
    counts = expanded["side"].value_counts()
    assert counts[1.0] == counts[-1.0]


def test_add_synthetic_orders_columns(ohlcv_df):
    feats = compute_market_features(ohlcv_df(n=50))
    expanded = add_synthetic_orders(feats, rng=np.random.default_rng(0))
    for col in FEATURE_NAMES:
        assert col in expanded.columns


def test_add_synthetic_orders_requires_rng(ohlcv_df):
    feats = compute_market_features(ohlcv_df(n=50))
    with pytest.raises(ValueError, match="requires an explicit"):
        add_synthetic_orders(feats)


def test_tod_encoding_range():
    index = pd.date_range("2024-01-02 14:30", periods=8, freq="1h", tz="UTC")
    sin_vals, cos_vals = _time_of_day_encoding(index)
    assert ((sin_vals >= -1) & (sin_vals <= 1)).all()
    assert ((cos_vals >= -1) & (cos_vals <= 1)).all()
