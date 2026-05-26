"""Expanding-window walk-forward cross-validation for the slippage MLP.

A single 65/15/20 chronological hold-out (``dataset.temporal_split``) is
one data point. Walk-forward gives ``n_folds`` data points: at each
fold the model trains on ``[t0, train_end_i)`` and tests on the next
``test_months`` window. ``train_end_i`` advances by a fixed step so the
training history *grows* fold over fold (expanding window), which is
the cleanest fit for the ~24-month dataset where a 12-month rolling
window would leave too little test data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from slippage.data import SplitData
from slippage.evaluation import global_metrics
from slippage.features import FEATURE_NAMES_TRAINING
from slippage.models import HeuristicBaseline, LinearBaseline, MeanPredictor
from slippage.training import predict, train

_DAYS_PER_MONTH = 30.44


def _months_between(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    return (t1 - t0).total_seconds() / (_DAYS_PER_MONTH * 86_400)


def _scaled_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    val_frac: float = 0.15,
) -> SplitData:
    """Build a SplitData for one fold: last ``val_frac`` of train → val."""
    n_train = len(train_df)
    i_val = int(n_train * (1 - val_frac))
    real_train = train_df.iloc[:i_val]
    val_df = train_df.iloc[i_val:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(real_train[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    return SplitData(
        X_train=X_train,
        y_train=real_train[target_col].values,
        X_val=X_val,
        y_val=val_df[target_col].values,
        X_test=X_test,
        y_test=test_df[target_col].values,
        scaler=scaler,
        train_df=real_train,
        val_df=val_df,
        test_df=test_df,
    )


def walk_forward_cv(
    proxy_df: pd.DataFrame,
    n_folds: int = 5,
    test_months: float = 2.0,
    seed: int = 42,
    dropout: float = 0.1,
    epochs: int = 100,
    feature_cols: list[str] | None = None,
    target_col: str = "slippage_bps",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run expanding-window walk-forward CV.

    Parameters
    ----------
    proxy_df:
        Output of ``pipeline.build_full_proxy`` (must be time-sorted and
        carry ``slippage_bps`` plus all feature columns).
    n_folds:
        Number of folds. Step length is computed so the last fold's test
        window ends at the dataset's last timestamp.
    test_months:
        Test-window length in months for every fold.
    seed, dropout, epochs:
        Passed through to ``train.train`` for the MLP.

    Returns
    -------
    DataFrame with one row per (fold, model) pair containing
    ``mae_bps``, ``rmse_bps``, ``med_ae_bps``, ``n_train``, ``n_test``,
    and the timestamps ``train_start``, ``train_end``, ``test_start``,
    ``test_end`` for the fold's training and test windows.
    """
    if feature_cols is None:
        feature_cols = FEATURE_NAMES_TRAINING

    df = proxy_df.sort_index().dropna(subset=feature_cols + [target_col])
    if df.empty:
        raise ValueError("walk_forward_cv: proxy_df is empty after dropna")

    t0, t_last = df.index[0], df.index[-1]
    total_months = _months_between(t0, t_last)
    if total_months <= test_months:
        raise ValueError(
            f"walk_forward_cv: dataset spans {total_months:.1f} months, "
            f"shorter than test_months={test_months}"
        )

    step_months = (total_months - test_months) / n_folds
    records: list[dict] = []

    for i in range(n_folds):
        train_end_offset = pd.Timedelta(days=(i + 1) * step_months * _DAYS_PER_MONTH)
        test_end_offset = train_end_offset + pd.Timedelta(days=test_months * _DAYS_PER_MONTH)
        train_end = t0 + train_end_offset
        test_end = t0 + test_end_offset

        train_df = df.loc[(df.index >= t0) & (df.index < train_end)]
        test_df = df.loc[(df.index >= train_end) & (df.index < test_end)]

        if len(train_df) < 200 or len(test_df) < 50:
            if verbose:
                print(
                    f"  fold {i}: skipped (n_train={len(train_df)}, "
                    f"n_test={len(test_df)})"
                )
            continue

        split = _scaled_split(train_df, test_df, feature_cols, target_col)

        model, _ = train(
            split,
            n_features=len(feature_cols),
            epochs=epochs,
            seed=seed,
            dropout=dropout,
            verbose=False,
            checkpoint_path=None,  # don't litter results/ with per-fold weights
        )
        y_pred_mlp = predict(model, split.X_test)

        scaler = split.scaler
        heuristic = HeuristicBaseline(feature_cols)
        heuristic.fit(split.X_val, split.y_val, scaler.mean_, scaler.scale_)
        mean_pred = MeanPredictor().fit(split.X_train, split.y_train)
        linear_pred = LinearBaseline().fit(split.X_train, split.y_train)

        predictions = {
            "mlp": y_pred_mlp,
            "mean": mean_pred.predict(split.X_test),
            "linear": linear_pred.predict(split.X_test),
            "heuristic": heuristic.predict(split.X_test, scaler.mean_, scaler.scale_),
        }

        for name, y_pred in predictions.items():
            metrics = global_metrics(split.y_test, y_pred)
            records.append({
                "fold": i,
                "model": name,
                "n_train": len(split.y_train),
                "n_test": len(split.y_test),
                "train_start": train_df.index.min(),
                "train_end": train_end,
                "test_start": test_df.index.min(),
                "test_end": test_df.index.max(),
                **metrics,
            })

        if verbose:
            mae = global_metrics(split.y_test, y_pred_mlp)["mae_bps"]
            print(
                f"  fold {i}: train_end={train_end.date()} "
                f"n_train={len(split.y_train):,} n_test={len(split.y_test):,} "
                f"mlp_mae={mae:.4f}"
            )

    return pd.DataFrame(records)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of every metric across folds, one row per model."""
    if results.empty:
        return results
    metric_cols = ["mae_bps", "rmse_bps", "med_ae_bps"]
    grouped = results.groupby("model")[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()
