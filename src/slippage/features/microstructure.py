"""Microstructure features that genuinely drive execution cost.

These are the liquidity/impact signals a real slippage model relies on,
and — unlike the synthetic proxy's hand-built modulators — they are
computable from OHLCV and carry information independent of the synthetic
label. They are used on the empirical (real-target) track and are
available as an opt-in extension to the synthetic feature set.

All features use only data up to and including the current bar (no
look-ahead): rolling windows end at ``t`` and the Roll estimator pairs
``(t-1, t)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def _amihud(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Rolling Amihud illiquidity (kept local so ``features`` stays a leaf
    module with no dependency on ``proxy``; mirrors ``proxy.empirical``)."""
    ret = np.log(close / close.shift(1))
    dollar_volume = (close * volume).clip(lower=_EPS)
    return (ret.abs() / dollar_volume).rolling(window, min_periods=window).mean()


FEATURE_NAMES_MICRO = [
    "log_dollar_vol",
    "turnover",
    "amihud",
    "roll_spread",
    "ofi",
    "parkinson_vol",
    "garman_klass_vol",
]


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson (1980) range-based volatility — more efficient than close-to-close."""
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(window, min_periods=window).mean() / (4.0 * np.log(2.0)))


def garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Garman-Klass (1980) OHLC range volatility estimator."""
    hl = np.log(high / low) ** 2
    co = np.log(close / open_) ** 2
    term = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return np.sqrt(term.rolling(window, min_periods=window).mean().clip(lower=0))


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll's (1984) effective-spread estimator from serial price-change covariance.

    ``spread = 2·√(-cov(Δp_t, Δp_{t-1}))`` when the covariance is negative
    (the bid-ask bounce), else 0. An independent cross-check on the
    Corwin-Schultz estimator.
    """
    dp = close.diff()
    cov = dp.rolling(window, min_periods=window).cov(dp.shift(1))
    return 2.0 * np.sqrt((-cov).clip(lower=0))


def order_flow_imbalance(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Tick-rule order-flow imbalance: net signed volume over total volume.

    Sign of the bar return classifies the bar as buy (+) or sell (−)
    initiated (Lee-Ready approximation). Values in ``[-1, 1]``; persistent
    imbalance is one of the strongest real slippage predictors.
    """
    ret = np.log(close / close.shift(1))
    signed = np.sign(ret) * volume
    num = signed.rolling(window, min_periods=window).sum()
    den = volume.rolling(window, min_periods=window).sum()
    return num / den.replace(0, np.nan)


def compute_microstructure_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute the microstructure feature bundle (no look-ahead).

    Returns one row per input bar; rows with insufficient history carry NaN
    and should be dropped by the caller (or joined and dropna'd).
    """
    out = pd.DataFrame(index=df.index)
    dollar_vol = df["close"] * df["volume"]
    out["log_dollar_vol"] = np.log1p(dollar_vol)
    out["turnover"] = dollar_vol / dollar_vol.rolling(window, min_periods=window).mean()
    out["amihud"] = _amihud(df["close"], df["volume"], window=window)
    out["roll_spread"] = roll_spread(df["close"], window=window)
    out["ofi"] = order_flow_imbalance(df["close"], df["volume"], window=window)
    out["parkinson_vol"] = parkinson_vol(df["high"], df["low"], window=window)
    out["garman_klass_vol"] = garman_klass_vol(
        df["open"], df["high"], df["low"], df["close"], window=window
    )
    return out
