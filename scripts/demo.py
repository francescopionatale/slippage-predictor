"""Offline end-to-end demo — `make demo`.

Runs the full pipeline (features → proxy → split → train → evaluate) on the
small committed sample in ``data/sample/`` with no network access, in well
under a minute. Prints MAE for the MLP vs every baseline, a Diebold-Mariano
significance test, and writes a pred-vs-actual figure to
``docs/assets/demo_pred_vs_actual.png`` — the README hero image.

This is the fastest way to confirm a clean checkout works and to see a
result without downloading two years of market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slippage.evaluation import build_baselines, compare_against_reference, global_metrics
from slippage.features import FEATURE_NAMES_TRAINING, add_synthetic_orders, compute_market_features
from slippage.logging import configure_logging, get_logger
from slippage.paths import PROJECT_ROOT
from slippage.proxy import build_proxy
from slippage.training import predict, train

logger = get_logger(__name__)

SAMPLE = PROJECT_ROOT / "data" / "sample" / "sample_ohlcv.parquet"
FIGURE = PROJECT_ROOT / "docs" / "assets" / "demo_pred_vs_actual.png"


def build_sample_proxy() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE)
    order_rng = np.random.default_rng(42)
    proxy_rng = np.random.default_rng(43)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=order_rng)
    proxy = build_proxy(df, feats, alpha=2.0, impact_noise=0.20, rng=proxy_rng)
    proxy["ticker"] = "DEMO"
    return proxy.sort_index()


def save_figure(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    lim = float(max(y_true.max(), y_pred.max())) * 1.05
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, edgecolors="none")
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="perfect")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual slippage (bps)")
    ax.set_ylabel("Predicted slippage (bps)")
    ax.set_title("Demo: MLP predicted vs actual slippage")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=110)
    plt.close(fig)


def main() -> None:
    configure_logging()
    if not SAMPLE.exists():
        raise SystemExit(f"sample data not found at {SAMPLE}")

    from slippage.data import temporal_split

    logger.info("Building proxy from committed sample (%s)...", SAMPLE.name)
    proxy = build_sample_proxy()
    split = temporal_split(proxy)
    logger.info(
        "rows  train=%d val=%d test=%d", len(split.X_train), len(split.X_val), len(split.X_test)
    )

    model, _ = train(
        split, n_features=len(FEATURE_NAMES_TRAINING),
        epochs=40, verbose=False, checkpoint_path=None,
    )
    y_pred = predict(model, split.X_test)

    predictions = {"mlp": y_pred}
    predictions.update(
        {n: fn(split.X_test) for n, fn in build_baselines(split).items()}
    )

    print("\n=== Demo test metrics (bps) ===")
    for name, pred in predictions.items():
        m = global_metrics(split.y_test, pred)
        print(f"  {name:9s}  MAE={m['mae_bps']:6.3f}  RMSE={m['rmse_bps']:6.3f}")

    print("\n=== MLP vs baselines (Diebold-Mariano, block-bootstrap MAE CI) ===")
    table = compare_against_reference(split.y_test, predictions, reference="mlp", n_boot=400)
    for name, entry in table.items():
        lo, hi = entry["mae_ci"]
        tag = ""
        if "dm_vs_reference" in entry:
            dm = entry["dm_vs_reference"]
            verdict = "MLP better" if dm["favored_reference"] else "n.s."
            tag = f"  DM p={dm['p_value']:.3f} ({verdict})"
        print(f"  {name:9s}  MAE={entry['mae']:6.3f}  CI=[{lo:.3f}, {hi:.3f}]{tag}")

    save_figure(split.y_test, y_pred)
    print(f"\nWrote figure: {FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
