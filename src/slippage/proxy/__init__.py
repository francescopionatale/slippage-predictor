"""Synthetic slippage label construction + alpha calibration."""

from slippage.proxy.calibration import calibrate_alpha
from slippage.proxy.label import build_proxy

__all__ = ["build_proxy", "calibrate_alpha"]
