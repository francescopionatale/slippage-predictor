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

Since real execution data is unavailable, slippage is constructed synthetically:

```
arrival          = close_t
base_exec        = mean(open, high, low, close) of bar t+1
impact_penalty   = α × order_size_fraction × vol_rolling × arrival
exec_price       = base_exec + side × impact_penalty
slippage_bps     = 10 000 × side × (exec_price − arrival) / arrival
```

**α calibration**: swept over [0.5, 1.0, 2.0, 5.0, 10.0]; the value
minimising MAE on the validation set is chosen. See notebook `02`.

---

## Model

```
Linear(13 → 64) → ReLU → Dropout(0.1)
Linear(64 → 32) → ReLU
Linear(32 → 1)
```

- Loss: Huber (δ=1.0)
- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: patience=10 on val MAE
- Split: 65% train / 15% val / 20% test (strictly chronological)

---

## Baselines

1. **Mean predictor** — always predicts training-set mean
2. **Ridge regression** — same scaled features, linear model
3. **Heuristic** — β × order_size_fraction × vol_rolling × 10 000, β calibrated on val

---

## Asset universe

SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM — covering two calendar
years of hourly bars to maximise variability across market regimes.

---

## Known limitations

The most important caveat is the **circularity between the synthetic proxy
and the model features**. The proxy formula

```
slippage_bps = 10 000 × side × (base_exec_{t+1} − close_t) / close_t
             + 10 000 × α × order_size_fraction × vol_rolling
```

uses `order_size_fraction` and `vol_rolling` as inputs — and those same two
variables are also features fed to the MLP. As a consequence:

- A meaningful fraction of the test-set MAE reduction over the mean baseline
  comes from the model **re-learning the deterministic impact term**, not from
  genuinely predicting execution cost.
- The residual `base_exec_{t+1} − close_t` is the only true predictive
  challenge, and it is essentially next-bar price drift — notoriously close
  to a random walk.
- **Test metrics therefore overstate real-world generalisation capability.**
  A fair evaluation would require actual broker execution data (Level 2
  fills, IS/VWAP benchmarks), which is not freely available.

Additional caveats:

- The α calibration grid `[0.5, 1, 2, 5, 10]` is coarse, and α=2.0 serves as
  the reference proxy, so calibration mainly validates robustness to α
  perturbation rather than discovering an empirical optimum.
- `order_size_fraction ~ Uniform(0.001, 0.05)` is not representative of any
  real-world order-size distribution.
- The temporal hold-out (last 20%) is a single fold; results may shift under
  a true walk-forward setup with periodic refitting (see *Out of scope*).
- `results/metrics.json` contains **illustrative values**, not numbers
  produced by running the training pipeline.

---

## Out of scope (future extensions)

- True rolling walk-forward with model refitting
- Sequence models (LSTM, Transformer) for temporal features
- Hyperparameter optimisation (Optuna)
- Real Level-2 execution data
