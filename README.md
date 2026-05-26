# Slippage Predictor — Estimating Execution Costs from Intraday Market Data

A neural-network framework for estimating expected slippage of financial
orders from public intraday OHLCV data. The project demonstrates a
methodologically correct approach to execution-cost modelling — no
look-ahead bias, explicit proxy calibration, temporal hold-out
validation, walk-forward CV, and rigorous baseline comparison — rather
than claiming production-grade prediction.

📖 **Detailed methodology lives in [`docs/`](docs/):**
- [`docs/methodology.md`](docs/methodology.md) — split design, baselines, model, walk-forward CV.
- [`docs/feature_engineering.md`](docs/feature_engineering.md) — market + spread + synthetic-order features.
- [`docs/proxy_construction.md`](docs/proxy_construction.md) — square-root impact formula and rationale.

---

## What is slippage?

Slippage is the difference between the price at which a trade is
decided and the actual execution price. It is always expressed as a
cost:

```
Buy order:   slippage = execution_price − arrival_price
Sell order:  slippage = arrival_price − execution_price
slippage_bps = 10 000 × slippage / arrival_price
```

---

## Data source

Hourly OHLCV bars from Yahoo Finance (`yfinance`, `interval="1h"`),
covering ~2 years (yfinance restricts 1h history to ~730 days). To
use 5-minute bars on a shorter window, pass `--interval 5m` to
`scripts/download_data.py`.

---

## Installation

Requires **Python ≥ 3.12** (3.13 is tested in CI).

```bash
git clone https://github.com/francescopionatale/slippage-predictor.git
cd slippage-predictor
make install           # creates .venv, installs editable + dev deps
```

After install, three console scripts are on PATH:
`slippage-download`, `slippage-train`, `slippage-eval`.

---

## Quick start

```bash
# 1. Download data (cached as parquet after first run)
make data

# 2. Train the canonical MLP + evaluate on the test set
make train
make eval

# 3. Run the full diagnostic pipeline (sweep, walk-forward, residuals,
#    per-ticker breakdown, self-contained HTML report)
make report

# Shortcut: train + eval + full report
make all

# Tests (unit + integration)
make test
```

The diagnostic HTML lands at `results/comparisons/report.html` with
PNGs base64-embedded — share it as a single file.

---

## Repository structure

```
slippage-predictor/
├── configs/                       # YAML defaults — canonical reference
│   ├── data.yaml
│   ├── features.yaml
│   ├── model.yaml
│   ├── proxy.yaml
│   └── training.yaml
├── data/
│   ├── raw/                       # .gitignore'd parquet cache
│   └── processed/
├── src/slippage/                  # the package
│   ├── config.py                  # pydantic config loader
│   ├── paths.py                   # centralised filesystem paths
│   ├── pipeline.py                # data → features → proxy → split
│   ├── walk_forward.py            # expanding-window CV
│   ├── data/                      # loader + temporal split
│   ├── features/                  # market + spread + synthetic-order
│   ├── proxy/                     # square-root impact label + α calibration
│   ├── models/                    # MLP + baselines + PyTorch Dataset
│   ├── training/                  # Huber + Adam + early stop
│   ├── evaluation/                # metrics, segments, per-ticker
│   └── viz/                       # training, diagnostics, features plots
├── scripts/                       # thin CLIs + diagnostic orchestrators
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_pipeline.py
│   ├── run_experiments.py
│   ├── run_walk_forward.py
│   ├── compare_runs.py
│   ├── plot_pred_vs_actual.py
│   ├── generate_residual_plots.py
│   ├── generate_diversification_plots.py
│   └── build_report.py
├── notebooks/                     # 01-04 narrative notebooks
├── tests/{unit,integration}/      # pytest
├── artifacts/{checkpoints,scalers}/  # .gitignore'd model weights
├── results/                       # .gitignore'd evaluation outputs
├── docs/                          # methodology + feature + proxy docs
├── .github/workflows/ci.yml       # pytest + ruff + mypy on push / PR
├── LICENSE                        # MIT
├── pyproject.toml
└── Makefile
```

---

## Asset universe

11 tickers spanning the liquidity spectrum, two calendar years of
hourly bars:

| Ticker | Sector       | Liquidity profile               |
|--------|--------------|---------------------------------|
| SPY    | ETF          | Ultra-liquid, S&P 500           |
| QQQ    | ETF          | Ultra-liquid, Nasdaq-100        |
| IWM    | ETF          | Liquid, wider spread than SPY   |
| AAPL   | Tech         | Ultra-liquid                    |
| MSFT   | Tech         | Ultra-liquid                    |
| GOOGL  | Tech         | Ultra-liquid                    |
| JPM    | Financials   | Ultra-liquid                    |
| GS     | Financials   | Liquid, rate-sensitive          |
| XOM    | Energy       | Liquid, commodity-adjacent      |
| SLB    | Energy       | Liquid, commodity-adjacent      |
| PRCT†  | Healthcare   | Illiquid small-cap (stress test)|

† Auto-fallback to `MGNI` (Magnite) if PRCT returns fewer than 500
cleaned bars.

Diversification is **cross-ticker**, not cross-regime: yfinance
restricts 1h bars to ~730 days, so all tickers share the same time
window. The illiquid name is included to expose the model to wider
Corwin–Schultz spreads.

---

## Expected results

After running the full pipeline (`make all`), `results/metrics.json`
reports (test set, bps):

| Model      |   MAE  |  RMSE  | MedAE |
|------------|-------:|-------:|------:|
| MLP        |  ≈ 5.3 |  ≈ 9.3 | ≈ 3.2 |
| Heuristic  | ≈ 12.1 | ≈ 20.3 | ≈ 7.3 |
| Linear     |  ≈ 7.9 | ≈ 13.7 | ≈ 5.4 |
| Mean       | ≈ 17.7 | ≈ 30.2 | ≈ 12.9 |

The aggregate MAE is structurally higher than a mega-cap-only
universe: the illiquid name (PRCT/MGNI) has per-ticker MAE ≈ 12.6 bps
vs ≈ 2.5 bps for SPY, and this enters the aggregate. The MLP beats
the heuristic ~2:1 on every ticker individually — see the per-ticker
chart in the diagnostic report.

---

## Known limitations

- **Ticker universe is US equity / ETF only.** Cross-asset
  generalisation (futures, FX, crypto) is not tested.
- **Circular evaluation.** The slippage label is a closed-form
  function of the same features the model observes — the MLP
  approximates the proxy formula, not real execution costs. See
  [`docs/methodology.md`](docs/methodology.md) for the full
  disclosure.
- **Synthetic order distribution.** `order_size_fraction ~ Uniform
  (0.001, 0.05)` and `urgency ~ Uniform(0, 1)` bear no relation to
  any real order-flow distribution.
- **No real execution data or Level-2 features.** A production
  estimator requires broker fill records, bid-ask quotes, and
  order-book depth.

---

## Out of scope (future extensions)

- True rolling walk-forward with model refitting
- Sequence models (LSTM, Transformer) for temporal features
- Hyperparameter optimisation (Optuna)
- Real Level-2 execution data
