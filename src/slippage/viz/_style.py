"""Shared matplotlib style for all slippage.viz plots."""

from __future__ import annotations

import matplotlib.pyplot as plt

_PALETTE = "viridis"


def _apply_style() -> None:
    """Apply the shared matplotlib style.

    Idempotent — safe to call from multiple modules. Falls back gracefully
    if the seaborn style isn't bundled (older matplotlib).
    """
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["savefig.dpi"] = 150
