"""Market-side features computed from OHLCV data with no look-ahead bias."""

from __future__ import annotations

import numpy as np
import pandas as pd

from slippage.features.spread import _corwin_schultz_spread

FEATURE_NAMES_MARKET = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "vol_rolling", "range_rel", "vol_ratio",
    "tod_sin", "tod_cos", "spread_cs", "is_rth",
]


def compute_market_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Compute market-side features from OHLCV data.

    All rolling operations use only past data (ending at the current bar),
    so there is no look-ahead bias.

    Parameters
    ----------
    df:
        DataFrame with columns [open, high, low, close, volume] and a
        DatetimeIndex (timezone-aware recommended).
    vol_window:
        Rolling window length used for volatility and volume ratio.

    Returns
    -------
    DataFrame with one row per input bar containing the feature columns.
    Rows with insufficient history (NaN in any feature) are dropped.
    """
    out = pd.DataFrame(index=df.index)

    log_ret = np.log(df["close"] / df["close"].shift(1))

    out["ret_1"] = log_ret
    out["ret_3"] = np.log(df["close"] / df["close"].shift(3))
    out["ret_6"] = np.log(df["close"] / df["close"].shift(6))
    out["ret_12"] = np.log(df["close"] / df["close"].shift(12))

    out["vol_rolling"] = log_ret.rolling(vol_window, min_periods=vol_window).std()

    out["range_rel"] = (df["high"] - df["low"]) / df["close"]

    vol_ma = df["volume"].rolling(vol_window, min_periods=vol_window).mean()
    out["vol_ratio"] = df["volume"] / vol_ma

    out["tod_sin"], out["tod_cos"] = _time_of_day_encoding(df.index)

    out["spread_cs"] = _corwin_schultz_spread(df["high"], df["low"])

    # Regular trading hours flag (NYSE 09:30–16:00 Eastern)
    hours = eastern_hours(df.index)
    out["is_rth"] = ((hours >= 9.5) & (hours < 16.0)).astype(float)

    out = out.dropna()
    return out


def eastern_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Return the hour-of-day (0–24, fractional) in America/New_York.

    Naive indices are assumed to already be in Eastern wall-clock time;
    tz-aware indices are converted. Shared by ``_time_of_day_encoding``
    here and by ``proxy.build_proxy`` for its ``tod_mult`` factor.
    """
    if index.tz is None:
        hours = index.hour + index.minute / 60.0
    else:
        eastern = index.tz_convert("America/New_York")
        hours = eastern.hour + eastern.minute / 60.0
    return np.asarray(hours, dtype=float)


def _time_of_day_encoding(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Encode time of day as sin/cos on a 24-hour cycle (Eastern time).

    Using the full 24h cycle (rather than mapping NYSE 9:30–16:00 to one
    period) means extended-hours bars get unique encodings — important
    because yfinance 1h data includes pre- and post-market bars.
    """
    hours = eastern_hours(index)
    frac = hours / 24.0
    tod_sin = pd.Series(np.sin(2 * np.pi * frac), index=index)
    tod_cos = pd.Series(np.cos(2 * np.pi * frac), index=index)
    return tod_sin, tod_cos
