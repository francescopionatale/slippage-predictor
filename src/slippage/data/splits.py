"""Temporal train/val/test split and PyTorch Dataset.

The split is purely chronological (no shuffling) to prevent look-ahead bias:
  - Train: first 65% of bars
  - Val:   next 15%
  - Test:  final 20%

StandardScaler is fit only on the training fold. The SplitData container
carries both the scaled numpy arrays and the underlying filtered DataFrames,
so callers never have to re-slice the original proxy (which is unsafe under
duplicate indices or NaN drops).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from slippage.features import FEATURE_NAMES_TRAINING


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def temporal_split(
    proxy_df: pd.DataFrame,
    train_frac: float = 0.65,
    val_frac: float = 0.15,
    feature_cols: list[str] | None = None,
    target_col: str = "slippage_bps",
) -> SplitData:
    """Chronological train/val/test split with safe scaling.

    Parameters
    ----------
    proxy_df:
        Output of ``build_proxy`` containing feature columns and the
        ``slippage_bps`` target. Must be sorted by time.
    train_frac, val_frac:
        Fraction of rows for train and validation. Remaining go to test.
    feature_cols:
        Columns to use as model input. Defaults to ``FEATURE_NAMES_TRAINING``
        (everything except ``side``, which cancels algebraically in the
        proxy and should not enter the model).
    target_col:
        Name of the label column.

    Returns
    -------
    SplitData with numpy arrays, the train-only scaler, and the
    train/val/test DataFrame slices.
    """
    if feature_cols is None:
        feature_cols = FEATURE_NAMES_TRAINING

    df = proxy_df.sort_index().dropna(subset=feature_cols + [target_col])
    n = len(df)
    i_val = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))

    train_df = df.iloc[:i_val]
    val_df = df.iloc[i_val:i_test]
    test_df = df.iloc[i_test:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


class SlippageDataset(Dataset):
    """PyTorch Dataset wrapping feature matrix and labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
