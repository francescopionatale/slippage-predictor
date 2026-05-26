#!/usr/bin/env python
"""Thin CLI wrapper — see `slippage.data.loader.main` for the real entry point.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --interval 5m --start 2024-11-01
"""

from slippage.data.loader import main

if __name__ == "__main__":
    main()
