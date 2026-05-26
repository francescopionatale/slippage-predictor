"""Market, spread, and synthetic-order features."""

from slippage.features._legacy import (
    FEATURE_NAMES,
    FEATURE_NAMES_MARKET,
    FEATURE_NAMES_ORDER,
    FEATURE_NAMES_TRAINING,
    add_synthetic_orders,
    compute_market_features,
    eastern_hours,
)
from slippage.features._legacy import _corwin_schultz_spread as _corwin_schultz_spread
from slippage.features._legacy import _time_of_day_encoding as _time_of_day_encoding

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_NAMES_MARKET",
    "FEATURE_NAMES_ORDER",
    "FEATURE_NAMES_TRAINING",
    "compute_market_features",
    "add_synthetic_orders",
    "eastern_hours",
]
