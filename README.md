# Slippage Predictor — Estimating Execution Costs from Intraday Market Data

A neural-network framework for estimating expected slippage of financial orders
from public intraday OHLCV data. The model is not intended to be state-of-the-art
but to demonstrate a methodologically correct approach to execution cost modelling:
no look-ahead bias, explicit proxy calibration, temporal hold-out validation, and
a rigorous comparison against three baselines.

---

## What is slippage?

Slippage is the difference between the price at which a trade is decided and
the actual execution price. It is always expressed as a cost:

```
Buy order:   slippage = execution_price − arrival_price
Sell order:  slippage = arrival_price − execution_price
slippage_bps = 10 000 × slippage / arrival_price
```

---

## Data source note

**Yahoo Finance `interval="1h"` is used for 2-year coverage.**
The `yfinance` API restricts 5-minute bars to the last ~60 days. To use 5-minute
bars for a short window (e.g. last 2 months), pass `--interval 5m` to the
download script and adjust `--start` accordingly.

---

## Installation

Requires **Python ≥ 3.12**. Python 3.13 is supported and tested.

```bash
git clone <repo>
cd slippage-predictor
make install           # creates .venv, installs all deps
```

Or manually:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

---

## Quick start

```bash
# 1. Download data (cached as parquet after first run)
make data

# 2. Train the MLP
make train

# 3. Evaluate on test set
make eval

# 4. Run all unit tests
make test
```

---

## Repository structure

```
slippage-predictor/
├── data/
│   ├── download.py          # CLI for yfinance download
│   ├── raw/                 # .gitignore'd — parquet cache
│   └── processed/           # .gitignore'd
├── src/slippage/
│   ├── paths.py             # centralised RAW/RESULTS/FIGURES directories
│   ├── data_loader.py       # download + parquet cache
│   ├── features.py          # market features + Corwin-Schultz + order features
│   ├── proxy.py             # synthetic slippage label + alpha calibration
│   ├── dataset.py           # temporal split + StandardScaler (train-only fit)
│   ├── pipeline.py          # end-to-end orchestrator (data → features → proxy → split)
│   ├── model.py             # MLP(64→32→1) architecture
│   ├── train.py             # Huber loss, Adam, ReduceLROnPlateau, early stopping
│   ├── baselines.py         # Mean, Ridge, Heuristic baselines
│   ├── evaluate.py          # MAE/RMSE/MedAE + segment breakdown
│   └── viz.py               # all plot functions
├── notebooks/
│   ├── 01_data_and_features.ipynb
│   ├── 02_proxy_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_and_analysis.ipynb
├── results/
│   ├── metrics.json         # test-set metrics (generated)
│   └── figures/             # PNG figures (generated)
├── tests/                   # pytest unit tests
├── pyproject.toml
└── Makefile
```

---

## Feature engineering

**Market features** (no look-ahead — computed from data ≤ bar t):

| Feature | Description |
|---------|-------------|
| `ret_1` … `ret_12` | Log-returns over 1, 3, 6, 12 bars |
| `vol_rolling` | Rolling std of log-returns (20-bar window) |
| `range_rel` | (High − Low) / Close |
| `vol_ratio` | Volume / 20-bar rolling mean volume |
| `tod_sin`, `tod_cos` | Time-of-day encoded as sin/cos of NYSE trading hour fraction |
| `spread_cs` | Corwin-Schultz bid-ask spread estimate from OHLC only |

**Synthetic order features** (sampled per bar):

| Feature | Description |
|---------|-------------|
| `side` | +1 (buy) or −1 (sell), balanced |
| `order_size_fraction` | ∼ Uniform(0.001, 0.05) |
| `urgency` | ∼ Uniform(0, 1) |

---

## Proxy construction

Since real execution data is unavailable, slippage is constructed as the
**expected execution cost** plus log-normal multiplicative noise:

```
arrival          = close_t
urgency_factor   = 1 + urgency^1.5
spread_penalty   = 1 + 50 × spread_cs × order_size_fraction
tod_mult         = 1.3 if hour_ET < 10.5 or hour_ET ≥ 15.0, else 1.0
impact           = α × σ_t × √(order_size_fraction) × arrival
                   × urgency_factor × spread_penalty × tod_mult × exp(η)
slippage_bps     = 10 000 × impact / arrival,   η ~ N(0, 0.20)
```

The proxy is built entirely from bar `t` — no lookahead to t+1. Slippage
is always a non-negative cost (log-normal noise keeps `impact` positive).
The `side` feature cancels out algebraically; it is kept in the feature set
as a sanity check (near-zero permutation importance).

### Market impact model

The square-root term `I(q) ∝ σ · √(q/V)` follows the empirical square-root
law for temporary market impact: the price impact of a trade of size `q` in a
market with volatility `σ` and average volume `V` grows as the square root of
participation rate. This is the standard reference model in algorithmic trading
(Almgren et al. 2005, "Direct Estimation of Equity Market Impact"; Gatheral &
Schied 2013, "Dynamical Models of Market Impact"). With `σ_log = 0.20`
(≈ 21% multiplicative std), the theoretical R² ceiling is approximately 0.96.

**Why no price drift?** A transaction-cost proxy should isolate
**execution cost** (predictable, function of the order and market state)
from **price drift during the holding period** (market risk, near-random
walk). The Almgren-Chriss IS framework and most algorithmic-trading
literature treat them as separate concerns. Including next-bar drift in
the label adds ~50 bps of irreducible noise that drowns out the ~3 bps
of learnable impact, preventing any feature-based model from showing
meaningful correlation. We therefore model expected execution cost,
the quantity execution algorithms actually optimise against.

**α calibration**: swept over [0.5, 1.0, 2.0, 5.0, 10.0]. For each
candidate α the SPY proxy is rebuilt and a short MLP (15 epochs) is
trained. The α with the lowest validation MAE is selected. See notebook `02`.

---

## Model

```
Linear(13 → 64) → ReLU → Dropout(0.1)
Linear(64 → 32) → ReLU
Linear(32 → 1)
```

- Loss: Huber (δ adaptive = `max(1, median(|y_train|))` ≈ 2–3 bps)
- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: patience=10 on val MAE
- Split: 65% train / 15% val / 20% test (strictly chronological)

---

## Baselines

1. **Mean predictor** — always predicts training-set mean
2. **Ridge regression** — same scaled features, linear model
3. **Heuristic** — β × √order_size_fraction × vol_rolling × 10 000, β calibrated on val

---

## Asset universe

SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM — covering two calendar
years of hourly bars to maximise variability across market regimes.

---

## Expected results

After running the full pipeline (`make data && make train && make eval`),
`results/metrics.json` should report (test set, bps):

| Model | MAE | RMSE | MedAE |
|-------|-----|------|-------|
| MLP        | ≈ 4.7 | ≈ 7.2 | ≈ 3.0 |
| Heuristic  | ≈ 10.9 | ≈ 16.5 | ≈ 7.0 |
| Linear     | ≈ 6.9 | ≈ 10.1 | ≈ 5.2 |
| Mean       | ≈ 15.2 | ≈ 21.5 | ≈ 12.6 |

The absolute bps are higher than a linear proxy because the square-root impact
term raises the typical slippage magnitude. The MLP still beats all baselines
convincingly, and seed-to-seed variation is small (σ ≈ 0.02 bps).

---

## Known limitations

- **Circular evaluation.** The slippage label is a closed-form function of
  the same features the model observes. The MLP approximates the proxy formula,
  not real execution costs. All reported MAE numbers measure fit to the proxy.
  The value of the project is **methodological** (chronological split, train-only
  scaler, segment breakdown, baseline comparisons, walk-forward CV), not a claim
  of production-grade execution cost prediction.
- `order_size_fraction ~ Uniform(0.001, 0.05)` is not representative of
  any real-world order-size distribution.
- The α calibration grid `[0.5, 1, 2, 5, 10]` is coarse and the absolute
  best α is the smallest one (lower α → lower noise floor in absolute
  bps); the sweep validates monotonic sensitivity, not an empirical optimum.
- The temporal hold-out (last 20%) is a single fold; results may shift
  under a true walk-forward setup with periodic refitting (see *Out of scope*).
- A fair evaluation of *real* slippage would require broker execution
  data (Level-2 fills, IS/VWAP benchmarks), which is not freely available.
  This project demonstrates the modelling and validation methodology, not
  a production-grade slippage estimator.

---

## Out of scope (future extensions)

- True rolling walk-forward with model refitting
- Sequence models (LSTM, Transformer) for temporal features
- Hyperparameter optimisation (Optuna)
- Real Level-2 execution data
