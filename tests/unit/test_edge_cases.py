"""Edge-case tests: degenerate inputs to feature engineering and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slippage.data import temporal_split
from slippage.features import FEATURE_NAMES, compute_market_features


def test_market_features_single_bar_is_empty(ohlcv_df):
    out = compute_market_features(ohlcv_df(n=1))
    assert out.empty


def test_market_features_insufficient_history_is_empty(ohlcv_df):
    # Fewer rows than the volatility window -> everything dropped by dropna.
    out = compute_market_features(ohlcv_df(n=10), vol_window=20)
    assert out.empty


def test_market_features_enough_history_non_empty(ohlcv_df):
    out = compute_market_features(ohlcv_df(n=60), vol_window=20)
    assert len(out) > 0
    assert not out.isna().any().any()


def _tiny_proxy(n):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    data = {c: rng.standard_normal(n) for c in FEATURE_NAMES}
    data["slippage_bps"] = np.abs(rng.standard_normal(n)) + 1
    return pd.DataFrame(data, index=idx)


@pytest.mark.parametrize("n", [1, 2])
def test_temporal_split_too_few_rows_raises(n):
    with pytest.raises(ValueError):
        temporal_split(_tiny_proxy(n))


@pytest.mark.parametrize("train_frac,val_frac", [(1.0, 0.0), (0.9, 0.2), (0.0, 0.5)])
def test_temporal_split_bad_fractions_raise(train_frac, val_frac):
    with pytest.raises(ValueError):
        temporal_split(_tiny_proxy(500), train_frac=train_frac, val_frac=val_frac)


def test_temporal_split_valid_produces_three_nonempty_folds():
    split = temporal_split(_tiny_proxy(500))
    assert len(split.X_train) > 0
    assert len(split.X_val) > 0
    assert len(split.X_test) > 0
