"""Slippage label construction: synthetic proxy + empirical (non-circular) target."""

from slippage.proxy.calibration import calibrate_alpha
from slippage.proxy.empirical import (
    amihud_illiquidity,
    build_empirical_target,
    rolling_kyle_lambda,
)
from slippage.proxy.label import build_proxy

__all__ = [
    "build_proxy",
    "calibrate_alpha",
    "amihud_illiquidity",
    "rolling_kyle_lambda",
    "build_empirical_target",
]
