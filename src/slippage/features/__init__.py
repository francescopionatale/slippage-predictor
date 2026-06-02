"""Market, spread, and synthetic-order features."""

from slippage.features.market import (
    FEATURE_NAMES_MARKET,
    compute_market_features,
    eastern_hours,
)
from slippage.features.market import _time_of_day_encoding as _time_of_day_encoding
from slippage.features.microstructure import (
    FEATURE_NAMES_MICRO,
    compute_microstructure_features,
)
from slippage.features.order import (
    FEATURE_NAMES,
    FEATURE_NAMES_ORDER,
    FEATURE_NAMES_TRAINING,
    add_synthetic_orders,
)
from slippage.features.spread import _corwin_schultz_spread as _corwin_schultz_spread

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_NAMES_MARKET",
    "FEATURE_NAMES_MICRO",
    "FEATURE_NAMES_ORDER",
    "FEATURE_NAMES_TRAINING",
    "compute_market_features",
    "compute_microstructure_features",
    "add_synthetic_orders",
    "eastern_hours",
]
