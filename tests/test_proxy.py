"""Unit tests for slippage proxy construction."""

import numpy as np
import pandas as pd
import pytest

from features import add_synthetic_orders, compute_market_features
from proxy import build_proxy, calibrate_alpha


def _make_ohlcv(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.date_range("2024-01-02 09:30", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _make_dataset(seed: int = 1):
    df = _make_ohlcv(seed=seed)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(seed))
    proxy = build_proxy(df, feats, alpha=2.0)
    return df, feats, proxy


def test_proxy_no_nan():
    _, _, proxy = _make_dataset()
    assert proxy["slippage_bps"].isna().sum() == 0


def test_proxy_impact_shifts_buy_up():
    """A buy order with alpha>0 must have higher slippage than alpha=0 (no impact)."""
    df = _make_ohlcv()
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(1))
    proxy_no_impact = build_proxy(df, feats, alpha=0.0)
    proxy_impact = build_proxy(df, feats, alpha=2.0)
    buys_no = proxy_no_impact[proxy_no_impact["side"] == 1.0]
    buys_wi = proxy_impact[proxy_impact["side"] == 1.0]
    assert buys_wi["slippage_bps"].mean() > buys_no["slippage_bps"].mean()


def test_proxy_sign_sell():
    """Sell order (side=-1) slippage: arrival - exec = arrival - (base - impact) > 0."""
    df, feats, proxy = _make_dataset()
    sells = proxy[proxy["side"] == -1.0]
    assert (sells["slippage_bps"] >= 0).mean() > 0.5


def test_proxy_monotone_in_alpha():
    """Higher alpha → larger average slippage_bps (more market impact)."""
    df = _make_ohlcv()
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(0))
    prev_mean = None
    for alpha in [0.5, 2.0, 5.0]:
        proxy = build_proxy(df, feats, alpha=alpha)
        mean_slip = proxy["slippage_bps"].mean()
        if prev_mean is not None:
            assert mean_slip > prev_mean, f"alpha={alpha} should give higher slippage than previous"
        prev_mean = mean_slip


def test_proxy_last_bar_dropped():
    """The last bar has no t+1, so it must be absent from the proxy."""
    df = _make_ohlcv()
    last_ts = df.index[-1]
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(0))
    proxy = build_proxy(df, feats, alpha=2.0)
    assert last_ts not in proxy.index or proxy.loc[last_ts, "slippage_bps"].isna().all()


def test_calibrate_alpha_returns_best():
    df = _make_ohlcv(n=80)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(0))
    proxy_ref = build_proxy(df, feats, alpha=2.0)
    y_ref = proxy_ref["slippage_bps"]
    best, sensitivity = calibrate_alpha(df, feats, y_ref, alphas=[0.5, 2.0, 5.0])
    # alpha=2.0 compared to itself should yield MAE=0
    assert sensitivity[2.0] == pytest.approx(0.0, abs=1e-6)
    assert best == 2.0
