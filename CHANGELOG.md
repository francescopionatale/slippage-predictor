# Changelog

All notable changes to this project are documented here. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — Scientific content & structure overhaul

### Added — scientific content
- **Empirical, non-circular target** (`proxy/empirical.py`): forward-looking
  Amihud illiquidity and Kyle's λ, plus a `scripts/run_empirical.py`
  real-data track that *forecasts* illiquidity rather than re-deriving a
  formula.
- **Circularity diagnostics** (`evaluation/ablation.py`,
  `scripts/ablation_circularity.py`): feature ablation and label
  perturbation that quantify how much apparent skill is formula memorisation.
- **Distributional quantile head** (`training/quantile.py`): pinball-loss
  multi-quantile output with prediction intervals and coverage evaluation.
- **GBM baseline** (`models/baselines.py`): a fair strong nonlinear
  competitor to the MLP.
- **Statistical significance** (`evaluation/significance.py`):
  Diebold-Mariano test + block-bootstrap MAE confidence intervals.
- **Microstructure features** (`features/microstructure.py`): dollar
  volume, Amihud, Roll spread, order-flow imbalance, Garman-Klass /
  Parkinson volatility.

### Added — structure & tooling
- `make demo` + committed `data/sample/` for a <1 min offline end-to-end run.
- `Dockerfile` for a reproducible CPU environment.
- Coverage gate (`--cov-fail-under=80`), golden-file, edge-case, and
  `hypothesis` property tests; checkpoint + scaler round-trip; determinism.
- `CITATION.cff`, `CHANGELOG.md`, GitHub Pages workflow.

### Changed
- **Config is now live**: every `configs/*.yaml` value drives the run via
  `Config.load()` → `train_from_config` (previously decorative). Pydantic
  models gained range/cross-field validation.
- `SlippageMLP` accepts `hidden` / `activation` / `n_outputs` (the
  canonical default reproduces the original architecture and checkpoint).
- Training is deterministic (seeded DataLoader) and aborts on non-finite
  loss instead of silently returning random weights.
- `print` calls replaced with structured `logging`; `paths.py` no longer
  performs filesystem I/O at import time (`ensure_dirs()`).
- Loader gained retry/backoff, a `DataDownloadError`, column validation,
  and a clear error when no tickers survive.
- Baseline construction de-duplicated into `evaluation.build_baselines`.

## [0.1.0] — Initial release
- Synthetic √-impact slippage proxy, MLP + mean/linear/heuristic baselines,
  temporal split, walk-forward CV, HTML diagnostic report.
