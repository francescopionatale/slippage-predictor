"""Unit tests for temporal splitting and dataset utilities."""

import numpy as np
import pandas as pd
import pytest

from slippage.dataset import SlippageDataset, temporal_split
from slippage.features import FEATURE_NAMES


def _make_proxy_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    data = {col: rng.standard_normal(n) for col in FEATURE_NAMES}
    data["slippage_bps"] = rng.standard_normal(n) * 10
    return pd.DataFrame(data, index=index)


def test_no_temporal_overlap():
    df = _make_proxy_df(300)
    split = temporal_split(df)
    assert len(set(split.train_df.index) & set(split.val_df.index)) == 0
    assert len(set(split.train_df.index) & set(split.test_df.index)) == 0
    assert len(set(split.val_df.index) & set(split.test_df.index)) == 0


def test_chronological_order():
    df = _make_proxy_df(300)
    split = temporal_split(df)
    assert split.train_df.index.max() <= split.val_df.index.min()
    assert split.val_df.index.max() <= split.test_df.index.min()


def test_split_df_matches_arrays():
    """split.test_df must have same length and aligned target as split.y_test."""
    df = _make_proxy_df(300)
    split = temporal_split(df)
    assert len(split.test_df) == len(split.y_test)
    assert len(split.val_df) == len(split.y_val)
    assert len(split.train_df) == len(split.y_train)
    # Target alignment
    np.testing.assert_allclose(split.test_df["slippage_bps"].values, split.y_test)


def test_scaler_fitted_on_train_only():
    """Scaler mean/std must be derived from train only; val/test should differ."""
    df = _make_proxy_df(300)
    split = temporal_split(df)
    # If scaler were fit on all data, the mean of X_train would not be ~0
    # After fit_transform on train, X_train columns have mean ≈ 0 and std ≈ 1
    assert abs(split.X_train.mean()) < 0.1
    assert abs(split.X_train.std() - 1.0) < 0.1
    # Val/test are only transformed, so their mean may differ from 0
    # (can't assert exact values, but check that val is not identical to train distribution)
    # Instead verify the scaler's fitted mean matches the train column means
    train_means = df.iloc[:int(0.65 * 300)][FEATURE_NAMES].values.mean(axis=0)
    np.testing.assert_allclose(split.scaler.mean_, train_means, rtol=1e-5)


def test_fractions_sum():
    n = 200
    df = _make_proxy_df(n)
    split = temporal_split(df, train_frac=0.65, val_frac=0.15)
    total = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert total == n


def test_dataset_shapes():
    df = _make_proxy_df(100)
    split = temporal_split(df)
    ds = SlippageDataset(split.X_train, split.y_train)
    X, y = ds[0]
    assert X.shape == (len(FEATURE_NAMES),)
    assert y.shape == (1,)
