"""Training loop and prediction helpers."""

from slippage.training.quantile import (
    DEFAULT_QUANTILES,
    interval_coverage,
    pinball_loss,
    predict_quantiles,
    train_quantile,
)
from slippage.training.trainer import (
    load_model,
    load_scaler,
    predict,
    train,
    train_from_config,
)

__all__ = [
    "train",
    "train_from_config",
    "predict",
    "load_model",
    "load_scaler",
    "train_quantile",
    "predict_quantiles",
    "pinball_loss",
    "interval_coverage",
    "DEFAULT_QUANTILES",
]
