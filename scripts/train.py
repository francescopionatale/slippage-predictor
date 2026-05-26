#!/usr/bin/env python
"""Thin CLI wrapper — train the canonical MLP and save the checkpoint.

Equivalent to `python -m slippage.training.trainer`.
"""

from slippage.training.trainer import main

if __name__ == "__main__":
    main()
