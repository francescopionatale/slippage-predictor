"""CLI entry point for ``python -m slippage.evaluation``."""

from __future__ import annotations

import torch

from slippage.evaluation.segments import evaluate_all
from slippage.features import FEATURE_NAMES_TRAINING
from slippage.models import HeuristicBaseline, LinearBaseline, MeanPredictor, SlippageMLP
from slippage.paths import CHECKPOINTS_DIR
from slippage.pipeline import build_full_dataset
from slippage.training import predict


def main() -> None:
    print("Loading data and building split...")
    _, _, split = build_full_dataset()

    checkpoint = torch.load(CHECKPOINTS_DIR / "model_checkpoint.pt", weights_only=True)
    model = SlippageMLP(n_features=checkpoint["n_features"])
    model.load_state_dict(checkpoint["state_dict"])

    scaler = split.scaler
    heuristic = HeuristicBaseline(FEATURE_NAMES_TRAINING)
    heuristic.fit(split.X_val, split.y_val, scaler.mean_, scaler.scale_)

    baselines = {
        "mean": lambda X: MeanPredictor().fit(split.X_train, split.y_train).predict(X),
        "linear": LinearBaseline().fit(split.X_train, split.y_train).predict,
        "heuristic": lambda X: heuristic.predict(X, scaler.mean_, scaler.scale_),
    }

    results = evaluate_all(split, lambda X: predict(model, X), baselines)

    print("\n=== Global Metrics (test set, bps) ===")
    for name, m in results.items():
        if not isinstance(m, dict) or "mae_bps" not in m:
            continue  # skip nested breakdowns (per_ticker, segment_breakdown)
        print(
            f"  {name:12s}  MAE={m['mae_bps']:.2f}  "
            f"RMSE={m['rmse_bps']:.2f}  MedAE={m['med_ae_bps']:.2f}"
        )

    per_ticker = results.get("per_ticker", {})
    if per_ticker:
        print("\n=== Per-ticker MAE (MLP vs Heuristic, bps) ===")
        for t in sorted(per_ticker.keys(), key=lambda k: per_ticker[k].get("mlp_mae_bps", 0)):
            r = per_ticker[t]
            mlp_v = r.get("mlp_mae_bps", float("nan"))
            heur_v = r.get("heuristic_mae_bps", float("nan"))
            print(f"  {t:6s}  n={int(r['n_samples']):>6,}  MLP={mlp_v:6.3f}  Heur={heur_v:6.3f}")


if __name__ == "__main__":
    main()
