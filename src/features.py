"""Feature engineering for slippage prediction.

Market features are computed using only data up to bar t (no look-ahead).
Synthetic order features are sampled per bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Market features
# ---------------------------------------------------------------------------

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


def _corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Estimate bid-ask spread via the Corwin-Schultz (2012) formula.

    Uses pairs of consecutive bars (t-1, t). Returns a Series aligned with
    the input index. Negative estimates (noise artefacts) are clipped to 0.

    Reference: Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask
    Spreads from Daily High and Low Prices."
    """
    ln_h = np.log(high)
    ln_l = np.log(low)

    beta = (ln_h - ln_l) ** 2 + (ln_h.shift(1) - ln_l.shift(1)) ** 2
    gamma = (
        np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1)))
    ) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)  # = 3 - 2√2 ≈ 0.172
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)
    return spread


# ---------------------------------------------------------------------------
# Synthetic order features
# ---------------------------------------------------------------------------

FEATURE_NAMES_MARKET = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "vol_rolling", "range_rel", "vol_ratio",
    "tod_sin", "tod_cos", "spread_cs", "is_rth",
]

FEATURE_NAMES_ORDER = ["side", "order_size_fraction", "urgency"]
FEATURE_NAMES = FEATURE_NAMES_MARKET + FEATURE_NAMES_ORDER

# Features actually fed to the model — `side` is excluded because it
# cancels algebraically in the proxy formula (see proxy.build_proxy) and
# the model should not be allowed to learn anything from it.
FEATURE_NAMES_TRAINING = [f for f in FEATURE_NAMES if f != "side"]


def add_synthetic_orders(
    market_feats: pd.DataFrame,
    rng: np.random.Generator | None = None,
    orders_per_bar: int = 4,
    size_low: float = 0.001,
    size_high: float = 0.05,
) -> pd.DataFrame:
    """Expand market features by generating synthetic order features.

    For each input bar, ``orders_per_bar`` synthetic orders are created with
    alternating buy/sell sides and uniformly sampled size and urgency. The
    default 4 = 2 buy + 2 sell per bar gives wider per-bar coverage of the
    (size, urgency) plane and ~2× the training rows of the previous default.

    Parameters
    ----------
    market_feats:
        Output of ``compute_market_features``.
    rng:
        NumPy random Generator for reproducibility.
    orders_per_bar:
        Number of synthetic orders per bar. Must be even for balanced sides.
    size_low, size_high:
        Uniform bounds for ``order_size_fraction`` (fraction of avg volume).

    Returns
    -------
    DataFrame with ``len(market_feats) * orders_per_bar`` rows containing
    all market + order features.
    """
    if rng is None:
        raise ValueError(
            "add_synthetic_orders requires an explicit `rng` to keep "
            "synthetic-order generation reproducible and independent per ticker."
        )

    n = len(market_feats)
    rows = []
    for _ in range(orders_per_bar // 2):
        for side in [+1.0, -1.0]:
            chunk = market_feats.copy()
            chunk["side"] = side
            chunk["order_size_fraction"] = rng.uniform(size_low, size_high, size=n)
            chunk["urgency"] = rng.uniform(0.0, 1.0, size=n)
            rows.append(chunk)

    return pd.concat(rows).sort_index()
