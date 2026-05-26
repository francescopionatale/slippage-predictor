#!/usr/bin/env python
"""End-to-end pipeline: download data, train the canonical MLP, evaluate.

Use this for a smoke test of the full chain. For the full diagnostic
report (sweep, walk-forward, residuals, per-ticker breakdown), use the
`make report` target which orchestrates the more expensive scripts.
"""

from __future__ import annotations

from slippage.data.loader import main as download_main
from slippage.evaluation.__main__ import main as evaluate_main
from slippage.training.trainer import main as train_main


def main() -> None:
    print("=== [1/3] Download data ===")
    download_main()
    print("\n=== [2/3] Train canonical MLP ===")
    train_main()
    print("\n=== [3/3] Evaluate on test set ===")
    evaluate_main()


if __name__ == "__main__":
    main()
