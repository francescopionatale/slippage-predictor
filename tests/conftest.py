"""Shared pytest fixtures for synthetic test data."""

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_NAMES


@pytest.fixture
def ohlcv_df():
    """Factory fixture: synthetic OHLCV DataFrame.

    Usage::

        def test_x(ohlcv_df):
            df = ohlcv_df()                           # n=100, seed=0
            df = ohlcv_df(n=80, seed=1, min_hl=0.001) # wider bars
    """
    def _make(n=100, seed=0, min_hl=0.0, max_hl=0.005):
        rng = np.random.default_rng(seed)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
        high = close * (1 + rng.uniform(min_hl, max_hl, n))
        low = close * (1 - rng.uniform(min_hl, max_hl, n))
        open_ = close * (1 + rng.normal(0, 0.002, n))
        volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
        index = pd.date_range("2024-01-02 09:30", periods=n, freq="1h", tz="UTC")
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
    return _make


@pytest.fixture
def proxy_df():
    """Factory fixture: synthetic proxy DataFrame with random features and slippage_bps.

    Usage::

        def test_x(proxy_df):
            df = proxy_df()         # n=300, seed=0
            df = proxy_df(n=200)
    """
    def _make(n=300, seed=0):
        rng = np.random.default_rng(seed)
        index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
        data = {col: rng.standard_normal(n) for col in FEATURE_NAMES}
        data["slippage_bps"] = rng.standard_normal(n) * 10
        return pd.DataFrame(data, index=index)
    return _make
