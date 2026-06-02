"""Unit tests for baseline predictors."""

import numpy as np

from slippage.features import FEATURE_NAMES
from slippage.models import GBMBaseline, HeuristicBaseline, LinearBaseline, MeanPredictor


def _toy_data(n_train: int = 200, n_val: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, len(FEATURE_NAMES)))
    y_train = rng.standard_normal(n_train) * 5.0 + 2.0
    X_val = rng.standard_normal((n_val, len(FEATURE_NAMES)))
    y_val = rng.standard_normal(n_val) * 5.0 + 2.0
    return X_train, y_train, X_val, y_val


def test_mean_predictor_constant():
    X_train, y_train, X_val, _ = _toy_data()
    pred = MeanPredictor().fit(X_train, y_train).predict(X_val)
    assert np.all(pred == pred[0]), "MeanPredictor must return a constant."
    assert abs(pred[0] - y_train.mean()) < 1e-9


def test_mean_predictor_output_shape():
    X_train, y_train, X_val, _ = _toy_data()
    pred = MeanPredictor().fit(X_train, y_train).predict(X_val)
    assert pred.shape == (len(X_val),)


def test_linear_baseline_output_shape():
    X_train, y_train, X_val, _ = _toy_data()
    pred = LinearBaseline().fit(X_train, y_train).predict(X_val)
    assert pred.shape == (len(X_val),)


def test_linear_baseline_beats_mean_on_correlated_data():
    """Linear model should outperform mean on data with a true linear signal."""
    rng = np.random.default_rng(7)
    n = 300
    X = rng.standard_normal((n, 3))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.standard_normal(n) * 0.5
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:], y[200:]

    mean_mae = float(np.abs(MeanPredictor().fit(X_train, y_train).predict(X_val) - y_val).mean())
    lin_mae = float(np.abs(LinearBaseline().fit(X_train, y_train).predict(X_val) - y_val).mean())
    assert lin_mae < mean_mae


def test_heuristic_baseline_beta_calibrated():
    _, _, X_val, y_val = _toy_data()
    scaler_mean = np.zeros(len(FEATURE_NAMES))
    scaler_std = np.ones(len(FEATURE_NAMES))

    heuristic = HeuristicBaseline(FEATURE_NAMES)
    heuristic.fit(X_val, y_val, scaler_mean, scaler_std)
    assert heuristic.beta > 0


def test_heuristic_baseline_output_shape():
    X_train, y_train, X_val, y_val = _toy_data()
    scaler_mean = np.zeros(len(FEATURE_NAMES))
    scaler_std = np.ones(len(FEATURE_NAMES))
    h = HeuristicBaseline(FEATURE_NAMES)
    h.fit(X_val, y_val, scaler_mean, scaler_std)
    pred = h.predict(X_val, scaler_mean, scaler_std)
    assert pred.shape == (len(X_val),)


def test_gbm_baseline_output_shape_and_non_negative():
    X_train, y_train, X_val, _ = _toy_data()
    gbm = GBMBaseline(max_iter=20).fit(X_train, y_train)
    pred = gbm.predict(X_val)
    assert pred.shape == (len(X_val),)
    assert (pred >= 0).all(), "slippage predictions must be non-negative"


def test_gbm_baseline_beats_mean_on_nonlinear_signal():
    """GBM should capture a nonlinear interaction the mean cannot."""
    rng = np.random.default_rng(11)
    n = 600
    X = rng.uniform(0, 1, (n, 3))
    y = np.abs(np.sqrt(X[:, 0]) * X[:, 1] * 10.0 + rng.standard_normal(n) * 0.1)
    X_train, y_train = X[:450], y[:450]
    X_val, y_val = X[450:], y[450:]

    mean_pred = MeanPredictor().fit(X_train, y_train).predict(X_val)
    gbm_pred = GBMBaseline(max_iter=100).fit(X_train, y_train).predict(X_val)
    mean_mae = float(np.abs(mean_pred - y_val).mean())
    gbm_mae = float(np.abs(gbm_pred - y_val).mean())
    assert gbm_mae < mean_mae
