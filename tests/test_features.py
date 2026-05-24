"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from slippage.features import (
    FEATURE_NAMES,
    add_synthetic_orders,
    compute_market_features,
    _corwin_schultz_spread,
    _time_of_day_encoding,
)


def _make_ohlcv(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.date_range("2024-01-02 09:30", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def test_market_features_no_nan():
    df = _make_ohlcv(100)
    feats = compute_market_features(df)
    assert feats.isna().sum().sum() == 0


def test_market_features_no_lookahead():
    """Features at index i must not use data from index > i."""
    df = _make_ohlcv(100)
    feats_full = compute_market_features(df)
    # Truncate after bar 60 and recompute: feature values at shared indices must match
    feats_trunc = compute_market_features(df.iloc[:61])
    shared = feats_full.index.intersection(feats_trunc.index)
    assert len(shared) > 0
    pd.testing.assert_frame_equal(
        feats_full.loc[shared].reset_index(drop=True),
        feats_trunc.loc[shared].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )


def test_corwin_schultz_non_negative():
    df = _make_ohlcv(200)
    spread = _corwin_schultz_spread(df["high"], df["low"])
    assert (spread.dropna() >= 0).all()


def test_corwin_schultz_wider_bar_higher_spread():
    """Wider H-L range should generally produce a higher spread estimate."""
    n = 50
    index = pd.date_range("2024-01-02", periods=n, freq="1h", tz="UTC")
    close = np.ones(n) * 100.0
    # Narrow bars
    high_narrow = pd.Series(close * 1.001, index=index)
    low_narrow = pd.Series(close * 0.999, index=index)
    # Wide bars
    high_wide = pd.Series(close * 1.02, index=index)
    low_wide = pd.Series(close * 0.98, index=index)
    narrow = _corwin_schultz_spread(high_narrow, low_narrow).dropna().mean()
    wide = _corwin_schultz_spread(high_wide, low_wide).dropna().mean()
    assert wide > narrow


def test_add_synthetic_orders_balanced():
    df = _make_ohlcv(50)
    feats = compute_market_features(df)
    expanded = add_synthetic_orders(feats, rng=np.random.default_rng(0))
    counts = expanded["side"].value_counts()
    assert counts[1.0] == counts[-1.0]


def test_add_synthetic_orders_columns():
    df = _make_ohlcv(50)
    feats = compute_market_features(df)
    expanded = add_synthetic_orders(feats, rng=np.random.default_rng(0))
    for col in FEATURE_NAMES:
        assert col in expanded.columns


def test_add_synthetic_orders_requires_rng():
    df = _make_ohlcv(50)
    feats = compute_market_features(df)
    with pytest.raises(ValueError, match="requires an explicit"):
        add_synthetic_orders(feats)


def test_tod_encoding_range():
    index = pd.date_range("2024-01-02 14:30", periods=8, freq="1h", tz="UTC")
    sin_vals, cos_vals = _time_of_day_encoding(index)
    assert ((sin_vals >= -1) & (sin_vals <= 1)).all()
    assert ((cos_vals >= -1) & (cos_vals <= 1)).all()
