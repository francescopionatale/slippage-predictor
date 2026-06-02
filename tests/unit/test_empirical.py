"""Tests for the empirical (non-circular) illiquidity target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from slippage.features import compute_market_features
from slippage.proxy import amihud_illiquidity, build_empirical_target, rolling_kyle_lambda


def _ohlcv(n=400, seed=0, volume=None):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.003, n))
    high = close * (1 + rng.uniform(0, 0.004, n))
    low = close * (1 - rng.uniform(0, 0.004, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    if volume is None:
        volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.date_range("2024-01-02 09:30", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def test_amihud_higher_when_volume_lower():
    df = _ohlcv(seed=1)
    low_vol = df.copy()
    low_vol["volume"] = df["volume"] / 100.0
    illiq_normal = amihud_illiquidity(df["close"], df["volume"]).mean()
    illiq_low = amihud_illiquidity(low_vol["close"], low_vol["volume"]).mean()
    assert illiq_low > illiq_normal


def test_amihud_non_negative():
    df = _ohlcv(seed=2)
    illiq = amihud_illiquidity(df["close"], df["volume"]).dropna()
    assert (illiq >= 0).all()
    assert len(illiq) > 0


def test_kyle_lambda_finite_and_shaped():
    df = _ohlcv(seed=3)
    kl = rolling_kyle_lambda(df["close"], df["volume"], window=20)
    assert len(kl) == len(df)
    assert np.isfinite(kl.dropna()).all()


def test_build_empirical_target_is_forward_looking_and_clean():
    df = _ohlcv(seed=4)
    feats = compute_market_features(df)
    out = build_empirical_target(df, feats, window=20, horizon=1)
    assert "slippage_bps" in out.columns
    assert "kyle_lambda" in out.columns
    assert out["slippage_bps"].notna().all()
    assert (out["slippage_bps"] >= 0).all()
    # Forward target drops the final row(s); result is shorter than features.
    assert len(out) < len(feats)


def test_empirical_target_tracks_real_illiquidity():
    """A regime with a liquidity drop in the second half should show a higher
    median target there than in the (liquid) first half."""
    n = 600
    rng = np.random.default_rng(7)
    volume = np.concatenate([
        rng.integers(4_000_000, 5_000_000, n // 2).astype(float),  # liquid
        rng.integers(50_000, 100_000, n - n // 2).astype(float),   # illiquid
    ])
    df = _ohlcv(n=n, seed=7, volume=volume)
    feats = compute_market_features(df)
    out = build_empirical_target(df, feats, window=20, horizon=1)
    first_half = out.iloc[: len(out) // 2]["slippage_bps"].median()
    second_half = out.iloc[len(out) // 2:]["slippage_bps"].median()
    assert second_half > first_half
