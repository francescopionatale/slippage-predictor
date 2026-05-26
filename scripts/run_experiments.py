"""Run a sweep of training experiments and persist per-run diagnostics.

Each run writes to ``results/runs/<run_id>/``:
    - config.json    : the hyperparameters used
    - history.json   : per-epoch train_loss and val_mae
    - metrics.json   : test-set global metrics + segment breakdown for the MLP
                       and all three baselines
    - checkpoint.pt  : best (lowest val MAE) model weights

Runs are skipped if ``metrics.json`` already exists, so the script is safe
to re-invoke and resume an interrupted sweep.
"""

from __future__ import annotations

import json
import time

from baselines import HeuristicBaseline, LinearBaseline, MeanPredictor
from evaluate import evaluate_all
from features import FEATURE_NAMES_TRAINING
from paths import RESULTS_DIR
from pipeline import build_full_dataset
from train import predict, train

RUNS_DIR = RESULTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_run_id(cfg: dict) -> str:
    return (
        f"seed{cfg['seed']:04d}"
        f"_drop{int(cfg['dropout'] * 100):02d}"
        f"_lr{cfg['lr']:.0e}"
        f"_bs{cfg['batch_size']}"
        f"_a{cfg['alpha']:.1f}"
    )


def build_run_configs() -> list[dict]:
    """A focused sweep: seed sensitivity + light hyperparameter perturbation.

    Keeps total runtime bounded — we want comparable diagnostics, not a full
    grid search. The first three runs vary only the seed (separates noise
    from signal); the rest probe one hyperparameter at a time around the
    default configuration.
    """
    default = dict(seed=42, dropout=0.1, lr=1e-3, batch_size=256, alpha=2.0)

    configs: list[dict] = []
    # Seed sensitivity (default config, three seeds)
    for seed in (42, 7, 2024):
        configs.append({**default, "seed": seed})
    # Dropout sweep
    configs.append({**default, "dropout": 0.0})
    configs.append({**default, "dropout": 0.3})
    # Learning-rate sweep
    configs.append({**default, "lr": 3e-3})
    return configs


def run_one(cfg: dict, split, baselines: dict) -> dict:
    run_id = make_run_id(cfg)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"  [skip] {run_id} already complete")
        return json.loads(metrics_path.read_text())

    print(f"  [train] {run_id}")
    t0 = time.time()

    model, history = train(
        split,
        n_features=len(FEATURE_NAMES_TRAINING),
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        dropout=cfg["dropout"],
        verbose=False,
        checkpoint_path=run_dir / "checkpoint.pt",
    )

    elapsed = time.time() - t0

    # Persist history + config
    history_serializable = {
        "train_loss": [float(x) for x in history["train_loss"]],
        "val_mae": [float(x) for x in history["val_mae"]],
        "best_epoch": int(history["best_epoch"]),
        "best_val_mae": float(history["best_val_mae"]),
        "n_epochs_run": len(history["train_loss"]),
        "elapsed_seconds": elapsed,
    }
    (run_dir / "history.json").write_text(json.dumps(history_serializable, indent=2))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Full evaluation on test set (writes results/metrics.json globally too,
    # so we relocate the per-run copy after).
    results = evaluate_all(split, lambda X: predict(model, X), baselines)
    metrics_path.write_text(json.dumps(results, indent=2))
    print(
        f"     done in {elapsed:5.1f}s  "
        f"epochs={history_serializable['n_epochs_run']:3d}  "
        f"best_val_mae={history['best_val_mae']:.4f}  "
        f"test_mae={results['mlp']['mae_bps']:.4f}"
    )
    return results


def main() -> None:
    print("Building dataset (downloads cached parquet on first run)...")
    t0 = time.time()
    _, _, split = build_full_dataset()
    print(
        f"  Train: {len(split.X_train):,}  "
        f"Val: {len(split.X_val):,}  "
        f"Test: {len(split.X_test):,}  "
        f"({time.time() - t0:.1f}s)"
    )

    # Baselines are fit once on the canonical split (they are deterministic).
    scaler = split.scaler
    heuristic = HeuristicBaseline(FEATURE_NAMES_TRAINING)
    heuristic.fit(split.X_val, split.y_val, scaler.mean_, scaler.scale_)
    baselines = {
        "mean": lambda X: MeanPredictor().fit(split.X_train, split.y_train).predict(X),
        "linear": LinearBaseline().fit(split.X_train, split.y_train).predict,
        "heuristic": lambda X: heuristic.predict(X, scaler.mean_, scaler.scale_),
    }

    configs = build_run_configs()
    print(f"\nRunning {len(configs)} configurations:")
    for cfg in configs:
        run_one(cfg, split, baselines)

    print(f"\nAll runs persisted under: {RUNS_DIR}")


if __name__ == "__main__":
    main()
