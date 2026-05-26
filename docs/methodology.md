# Methodology

This document covers the modelling and evaluation methodology used by
slippage-predictor. For the feature catalogue and the synthetic-proxy
construction see [`feature_engineering.md`](feature_engineering.md) and
[`proxy_construction.md`](proxy_construction.md) respectively.

## Why a *methodologically* correct project

Real broker execution data (IS/VWAP benchmarks, Level-2 fills) is not
freely available. This project therefore demonstrates a careful
end-to-end methodology — chronological hold-outs, train-only scaling,
explicit baseline comparisons, walk-forward CV — rather than claiming
production-grade slippage prediction. The synthetic label is a
deliberate choice so the rest of the pipeline can be exercised honestly.

## Train / val / test split

A purely chronological split, no shuffling, to avoid look-ahead bias:

- Train: first 65% of bars (per ticker, then merged)
- Val:   next 15%
- Test:  final 20%

The `StandardScaler` is fit on the training fold only and then applied
to the validation and test folds. Any deviation from this convention
(e.g. fitting on the full dataset before splitting) silently leaks
distributional information from the test set into the model.

## Baselines

The MLP is benchmarked against three baselines on the same scaled
features and the same metrics:

1. **Mean predictor** — always predicts the training-set mean. Bottom
   floor: any model that doesn't beat this is broken.
2. **Ridge regression** — linear model on the full feature matrix.
   Quantifies how much of the signal a linear function can capture.
3. **Heuristic** — `β × √(order_size_fraction) × vol_rolling × 10 000`,
   with `β` calibrated on the validation set. Mirrors the leading term
   of the proxy formula (Almgren-Chriss square-root impact); the gap
   MLP − Heuristic measures how much the proxy's interaction terms
   (urgency × impact, spread × size, time-of-day) contribute beyond a
   simple β·√size·vol product.

## Model architecture

```
Linear(13 → 64) → ReLU → Dropout(0.1)
Linear(64 → 32) → ReLU
Linear(32 → 1)  → Softplus
```

The Softplus output activation enforces non-negative predictions
(slippage is always a positive cost). The two hidden widths (64, 32)
are deliberate — the proxy formula has roughly 5 interacting modulators,
which fit comfortably in a small MLP. A larger network does not
improve fit measurably (Block 2's seed sensitivity ≈ 0.02 bps σ).

## Training

- **Loss**: Huber with `δ` adaptive to the training-set scale
  (`δ = max(1, median(|y_train|))`, ≈ 2–3 bps).
- **Optimiser**: Adam, `lr=1e-3`.
- **Scheduler**: `ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)`.
- **Early stopping**: patience 10 on validation MAE.
- **Reproducibility**: explicit `seed` to `train()`, independent per-
  ticker RNGs for synthetic-order generation.

## Walk-forward cross-validation

A single 65/15/20 split is one data point. The walk-forward setup
(`slippage.walk_forward.walk_forward_cv`) generates `n_folds` data
points: at each fold the model trains on `[t₀, train_end_i)` and tests
on the next `test_months` window. `train_end_i` advances by a fixed
step, growing the training history fold over fold (expanding window).

This is the cleanest fit for ~24 months of hourly bars — a rolling-window
scheme would leave too little training data in early folds.

## Per-ticker breakdown

`evaluate_all` slices the test set by ticker and reports per-ticker MAE
for the MLP and the heuristic baseline (see `metrics["per_ticker"]`).
On the diversified 11-ticker universe, illiquid names (PRCT/MGNI, IWM)
show structurally higher MAE — expected from the proxy's
spread-penalty term, which amplifies impact where Corwin–Schultz
spreads are wider. The MLP still beats the heuristic ~2:1 on every
ticker individually.

## Circularity disclosure

The slippage label is a closed-form function of the same features the
model observes. The MLP is therefore performing **noisy function
approximation** of the proxy formula, not learning from real execution
data. All reported MAE numbers measure fit to the proxy, not
real-world execution cost accuracy. The MLP's advantage over the
heuristic reflects the proxy's interaction terms that the heuristic
cannot express, not any market-learning ability.
