#!/usr/bin/env python
"""Thin CLI wrapper — evaluate the canonical checkpoint, write metrics.json.

Equivalent to `python -m slippage.evaluation`.
"""

from slippage.evaluation.__main__ import main

if __name__ == "__main__":
    main()
