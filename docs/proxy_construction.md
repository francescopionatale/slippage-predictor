# Proxy construction

Since real execution data is unavailable, slippage is constructed as
the **expected execution cost** plus log-normal multiplicative noise.
The implementation lives in
`src/slippage/proxy/{label,calibration}.py`.

## Formula

```
arrival          = close_t
urgency_factor   = 1 + urgency^1.5
spread_penalty   = 1 + 50 × spread_cs × order_size_fraction
tod_mult         = 1.3 if hour_ET < 10.5 or hour_ET ≥ 15.0
                   else 1.0
impact           = α × σ_t × √(order_size_fraction) × arrival
                   × urgency_factor × spread_penalty × tod_mult × exp(η)
slippage_bps     = 10 000 × impact / arrival,    η ~ N(0, 0.20)
```

`σ_t` is `vol_rolling` (the 20-bar rolling std of log-returns), and
the entire formula uses only data observable at bar `t` — no t+1
lookahead. The `side` column cancels algebraically because
`side² = 1` everywhere.

## Square-root impact law

The `√(order_size_fraction)` term follows the empirical square-root
law for *temporary* market impact:

```
I(q) ∝ σ · √(q / V)
```

i.e. the price impact of a trade of size `q` in a market with
volatility `σ` and average volume `V` grows as the square root of the
participation rate. This is the standard reference model in
algorithmic trading:

- Almgren et al. (2005), *Direct Estimation of Equity Market Impact.*
- Gatheral & Schied (2013), *Dynamical Models of Market Impact.*

The `impact_noise = σ_log = 0.20` (≈ 21% multiplicative std) puts the
theoretical R² ceiling at approximately 0.96 — the bound the MLP
should approach on the test set under ideal calibration.

## Why no price drift?

A transaction-cost proxy must isolate **execution cost** (predictable,
function of the order and market state) from **price drift during the
holding period** (market risk, near-random walk). The Almgren-Chriss
IS framework and most algorithmic-trading literature treat them as
separate concerns.

Including next-bar drift in the label would add ~50 bps of
irreducible noise that drowns out the ~3 bps of learnable impact,
preventing any feature-based model from showing meaningful correlation.
This project therefore models the expected execution cost — the
quantity execution algorithms actually optimise against.

## Multiplicative modulators

Three multipliers convert the base sqrt-impact term into a non-trivial
function that the heuristic baseline cannot replicate:

- **`urgency_factor = 1 + urgency^1.5`** ∈ [1, 2]. High-urgency orders
  pay more impact (they consume more of the displayed depth).
- **`spread_penalty = 1 + 50 × spread_cs × order_size_fraction`**.
  Trading wide-spread instruments costs more, with the penalty growing
  with order size (a small order pays the half-spread, a large order
  walks the book).
- **`tod_mult ∈ {1.0, 1.3}`**. Opening (< 10:30 ET) and closing (≥ 15:00 ET)
  carry a 30% impact premium reflecting the well-documented intraday
  liquidity smile.

The MLP's advantage over the heuristic baseline (`β·√size·vol·10⁴`) is
the amount of interaction structure these three terms add to the
label. It is *not* evidence of market-learning ability.

## Alpha calibration

`slippage.proxy.calibrate_alpha` performs a grid search over
`[0.5, 1.0, 2.0, 5.0, 10.0]` of the global impact scale α. For each
candidate it rebuilds the proxy, trains a 15-epoch MLP, and records
the best validation MAE. The α with the lowest val MAE wins.

`α = 2.0` is used as the canonical scale throughout (see notebook
`02_proxy_construction.ipynb` for the sensitivity sweep). The sweep
validates **monotonic sensitivity** of the MAE to α, not an empirical
optimum — lower α shrinks the noise floor in absolute bps so the
smallest α always looks best on absolute MAE.

## Log-normal noise

`η ~ N(0, 0.20)` adds multiplicative noise via `exp(η)`. This:

- Keeps `impact` strictly positive (real execution costs cannot be
  negative).
- Makes the noise multiplicatively symmetric on a log scale, matching
  the standard volatility model for multiplicative shocks in finance.
- Sets a hard R² ceiling for any model: the explained variance cannot
  exceed `1 − var(exp(η)) / var(impact·exp(η))` ≈ 0.96 at σ_log = 0.20.

Reproducibility requires an explicit `numpy.random.Generator` passed
into `build_proxy` whenever `impact_noise > 0`.
