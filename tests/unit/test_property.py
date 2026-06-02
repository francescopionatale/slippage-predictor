"""Property-based tests (hypothesis) for the proxy's economic invariants."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slippage.features.market import _time_of_day_encoding
from slippage.proxy import build_proxy

_TS = pd.Timestamp("2024-01-03 14:00", tz="America/New_York")


def _one_row(size, vol, urgency, spread_cs, close=100.0):
    idx = pd.DatetimeIndex([_TS])
    df = pd.DataFrame({"close": [close]}, index=idx)
    feats = pd.DataFrame(
        {"side": [1.0], "order_size_fraction": [size], "vol_rolling": [vol],
         "urgency": [urgency], "spread_cs": [spread_cs]},
        index=idx,
    )
    return build_proxy(df, feats, alpha=2.0, impact_noise=0.0)["slippage_bps"].iloc[0]


@given(
    size=st.floats(1e-4, 0.1),
    vol=st.floats(1e-4, 0.05),
    urgency=st.floats(0.0, 1.0),
    spread_cs=st.floats(0.0, 0.01),
)
@settings(max_examples=150, deadline=None)
def test_proxy_non_negative(size, vol, urgency, spread_cs):
    assert _one_row(size, vol, urgency, spread_cs) >= 0.0


@given(
    size_lo=st.floats(1e-4, 0.05),
    bump=st.floats(1e-3, 0.05),
    vol=st.floats(1e-3, 0.05),
)
@settings(max_examples=100, deadline=None)
def test_proxy_monotone_in_size(size_lo, bump, vol):
    lo = _one_row(size_lo, vol, 0.5, 0.002)
    hi = _one_row(size_lo + bump, vol, 0.5, 0.002)
    assert hi >= lo


@given(
    vol_lo=st.floats(1e-4, 0.04),
    bump=st.floats(1e-3, 0.02),
    size=st.floats(1e-3, 0.05),
)
@settings(max_examples=100, deadline=None)
def test_proxy_monotone_in_vol(vol_lo, bump, size):
    lo = _one_row(size, vol_lo, 0.5, 0.002)
    hi = _one_row(size, vol_lo + bump, 0.5, 0.002)
    assert hi >= lo


@given(
    ts=st.datetimes(
        min_value=pd.Timestamp("2020-01-01").to_pydatetime(),
        max_value=pd.Timestamp("2025-12-31").to_pydatetime(),
    )
)
@settings(max_examples=100, deadline=None)
def test_tod_encoding_on_unit_circle(ts):
    idx = pd.DatetimeIndex([pd.Timestamp(ts)])
    sin, cos = _time_of_day_encoding(idx)
    assert (sin.iloc[0] ** 2 + cos.iloc[0] ** 2) == pytest.approx(1.0, abs=1e-9)


@given(size=st.floats(1e-4, 0.1), vol=st.floats(1e-4, 0.05))
@settings(max_examples=50, deadline=None)
def test_proxy_zero_alpha_gives_zero(size, vol):
    idx = pd.DatetimeIndex([_TS])
    df = pd.DataFrame({"close": [100.0]}, index=idx)
    feats = pd.DataFrame(
        {"side": [1.0], "order_size_fraction": [size], "vol_rolling": [vol],
         "urgency": [0.5], "spread_cs": [0.002]},
        index=idx,
    )
    # alpha must be > 0 per config, but the formula itself scales linearly in
    # alpha; a near-zero alpha yields a near-zero cost.
    out = build_proxy(df, feats, alpha=1e-9, impact_noise=0.0)["slippage_bps"].iloc[0]
    assert out >= 0.0
    assert out < 1e-3
