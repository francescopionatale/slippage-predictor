"""Baseline predictors for slippage comparison.

Three baselines are compared against the MLP:
  1. MeanPredictor  — always predicts the training-set mean.
  2. LinearBaseline — Ridge regression on the same scaled features.
  3. HeuristicBaseline — β * order_size_fraction * vol_rolling, β calibrated on val.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class MeanPredictor:
    """Always predicts the mean of the training target."""

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanPredictor":
        self._mean = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)


class LinearBaseline:
    """Ridge regression on all scaled feature columns."""

    def __init__(self, alpha: float = 1.0) -> None:
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBaseline":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class HeuristicBaseline:
    """Slippage estimated as β * √order_size_fraction * vol_rolling * 10 000.

    Mirrors the square-root impact law in the proxy (Almgren et al. 2005).
    β is chosen by grid-search on the validation set. The size and vol columns
    are un-scaled internally using the train-only scaler statistics that must
    be passed by the caller.
    """

    def __init__(
        self,
        feature_names: list[str],
        betas: list[float] | None = None,
    ) -> None:
        if betas is None:
            betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        self._betas = betas
        self._beta: float = 1.0

        try:
            self._size_idx = feature_names.index("order_size_fraction")
            self._vol_idx = feature_names.index("vol_rolling")
        except ValueError as exc:
            raise ValueError(
                "HeuristicBaseline requires 'order_size_fraction' and 'vol_rolling' in features."
            ) from exc

    def _unscale(
        self,
        X: np.ndarray,
        mean_train: np.ndarray,
        std_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        size_raw = X[:, self._size_idx] * std_train[self._size_idx] + mean_train[self._size_idx]
        vol_raw = X[:, self._vol_idx] * std_train[self._vol_idx] + mean_train[self._vol_idx]
        return size_raw, vol_raw

    def fit(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mean_train: np.ndarray,
        std_train: np.ndarray,
    ) -> "HeuristicBaseline":
        """Choose β that minimises MAE on the validation set.

        Parameters
        ----------
        X_val, y_val:
            Validation-fold features (already scaled) and slippage labels.
        mean_train, std_train:
            Per-feature mean and std from the train-only StandardScaler. Used
            to un-scale the two relevant columns to their raw values.
        """
        size_raw, vol_raw = self._unscale(X_val, mean_train, std_train)
        product = np.sqrt(np.maximum(size_raw, 0.0)) * vol_raw * 10_000.0

        best_beta, best_mae = 1.0, float("inf")
        for beta in self._betas:
            pred = beta * product
            mae = float(np.abs(pred - y_val).mean())
            if mae < best_mae:
                best_mae = mae
                best_beta = beta
        self._beta = best_beta
        return self

    def predict_raw(self, size: np.ndarray, vol: np.ndarray) -> np.ndarray:
        """Predict from raw (un-scaled) size and vol arrays."""
        return self._beta * np.sqrt(np.maximum(size, 0.0)) * vol * 10_000.0

    def predict(
        self,
        X: np.ndarray,
        mean_train: np.ndarray,
        std_train: np.ndarray,
    ) -> np.ndarray:
        size_raw, vol_raw = self._unscale(X, mean_train, std_train)
        return self.predict_raw(size_raw, vol_raw)

    @property
    def beta(self) -> float:
        return self._beta
