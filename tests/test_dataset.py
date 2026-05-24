"""Unit tests for temporal splitting and dataset utilities."""

import numpy as np
import pytest

from dataset import SlippageDataset, temporal_split
from features import FEATURE_NAMES


def test_no_temporal_overlap(proxy_df):
    split = temporal_split(proxy_df())
    assert len(set(split.train_df.index) & set(split.val_df.index)) == 0
    assert len(set(split.train_df.index) & set(split.test_df.index)) == 0
    assert len(set(split.val_df.index) & set(split.test_df.index)) == 0


def test_chronological_order(proxy_df):
    split = temporal_split(proxy_df())
    assert split.train_df.index.max() <= split.val_df.index.min()
    assert split.val_df.index.max() <= split.test_df.index.min()


def test_split_df_matches_arrays(proxy_df):
    """split.test_df must have same length and aligned target as split.y_test."""
    split = temporal_split(proxy_df())
    assert len(split.test_df) == len(split.y_test)
    assert len(split.val_df) == len(split.y_val)
    assert len(split.train_df) == len(split.y_train)
    np.testing.assert_allclose(split.test_df["slippage_bps"].values, split.y_test)


def test_scaler_fitted_on_train_only(proxy_df):
    """Scaler mean/std must be derived from train only."""
    df = proxy_df()
    split = temporal_split(df)
    assert abs(split.X_train.mean()) < 0.1
    assert abs(split.X_train.std() - 1.0) < 0.1
    train_means = df.iloc[:int(0.65 * 300)][FEATURE_NAMES].values.mean(axis=0)
    np.testing.assert_allclose(split.scaler.mean_, train_means, rtol=1e-5)


def test_fractions_sum(proxy_df):
    n = 200
    split = temporal_split(proxy_df(n=n), train_frac=0.65, val_frac=0.15)
    total = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert total == n


def test_dataset_shapes(proxy_df):
    split = temporal_split(proxy_df(n=100))
    ds = SlippageDataset(split.X_train, split.y_train)
    X, y = ds[0]
    assert X.shape == (len(FEATURE_NAMES),)
    assert y.shape == (1,)
