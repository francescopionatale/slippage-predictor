"""Shared baseline construction.

The Mean / Linear / Heuristic baselines are fit identically in the
evaluation CLI, the experiment sweep, and walk-forward CV. Centralising
that here keeps the three call sites from drifting and gives a single
insertion point for new baselines (e.g. the gradient-boosting model).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from slippage.data import SplitData
from slippage.features import FEATURE_NAMES_TRAINING
from slippage.models import GBMBaseline, HeuristicBaseline, LinearBaseline, MeanPredictor

Predictor = Callable[[np.ndarray], np.ndarray]


def build_baselines(
    split: SplitData,
    feature_names: list[str] | None = None,
    include_gbm: bool = True,
    include_heuristic: bool = True,
) -> dict[str, Predictor]:
    """Fit every baseline on ``split`` and return name → predict-callable.

    Each value is a function mapping a scaled feature matrix ``X`` to a
    slippage prediction array, so callers can apply them uniformly to
    ``split.X_test`` (walk-forward) or to arbitrary segments (evaluation).

    Parameters
    ----------
    split:
        Fitted :class:`~slippage.data.SplitData` (carries the train-only
        scaler used to un-scale the heuristic's size/vol columns).
    feature_names:
        Feature column order matching ``split.X_*``. Defaults to
        ``FEATURE_NAMES_TRAINING``.
    include_gbm:
        Whether to fit the gradient-boosting baseline (skip for speed in
        per-fold walk-forward runs if desired).
    include_heuristic:
        Whether to fit the √-impact heuristic. Requires
        ``order_size_fraction`` and ``vol_rolling`` in ``feature_names``;
        set False for the empirical-target track which has no order size.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES_TRAINING

    scaler = split.scaler

    mean_pred = MeanPredictor().fit(split.X_train, split.y_train)
    linear_pred = LinearBaseline().fit(split.X_train, split.y_train)

    baselines: dict[str, Predictor] = {
        "mean": mean_pred.predict,
        "linear": linear_pred.predict,
    }

    if include_heuristic:
        heuristic = HeuristicBaseline(feature_names)
        heuristic.fit(split.X_val, split.y_val, scaler.mean_, scaler.scale_)
        baselines["heuristic"] = lambda X: heuristic.predict(X, scaler.mean_, scaler.scale_)

    if include_gbm:
        gbm = GBMBaseline().fit(split.X_train, split.y_train)
        baselines["gbm"] = gbm.predict

    return baselines
