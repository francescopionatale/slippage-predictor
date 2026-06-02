"""Tests for the circularity ablation diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from slippage.evaluation.ablation import feature_ablation
from slippage.features import FEATURE_NAMES


def _proxy_with_label_from(features, n=900, seed=0, independent=False):
    """Build a synthetic proxy whose label is (or isn't) a function of
    ``order_size_fraction`` and ``vol_rolling``."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    data = {col: rng.standard_normal(n) for col in FEATURE_NAMES}
    df = pd.DataFrame(data, index=index)
    if independent:
        df["slippage_bps"] = np.abs(rng.standard_normal(n)) + 1.0
    else:
        df["slippage_bps"] = (
            3.0 * np.abs(df["order_size_fraction"])
            + 2.0 * np.abs(df["vol_rolling"])
            + 1.0
            + rng.standard_normal(n) * 0.1
        )
    return df


def test_ablation_degrades_when_label_depends_on_dropped_features():
    df = _proxy_with_label_from(["order_size_fraction", "vol_rolling"], independent=False)
    res = feature_ablation(df, drop=["order_size_fraction", "vol_rolling"], epochs=40, seed=0)
    # Dropping the very features that build the label should hurt.
    assert res.reduced_test_mae > res.full_test_mae
    assert res.rel_degradation > 0.05
    assert set(res.dropped) == {"order_size_fraction", "vol_rolling"}


def test_ablation_minimal_when_label_independent_of_features():
    df = _proxy_with_label_from([], independent=True)
    res = feature_ablation(df, drop=["order_size_fraction", "vol_rolling"], epochs=20, seed=0)
    # If the label doesn't depend on the dropped features, error shouldn't
    # explode (allow a small wobble from the shorter feature set).
    assert res.rel_degradation < 0.5
