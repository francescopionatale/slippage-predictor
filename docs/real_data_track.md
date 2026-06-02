# Real-data track: an empirical, non-circular target

## Why

The synthetic proxy (see [`proxy_construction.md`](proxy_construction.md))
is a closed-form function of the model's own features, so a model trained
and scored on it can only re-derive arithmetic — its "skill" is not
transferable. The real-data track replaces that label with an
**empirically estimated** market-impact quantity the model must genuinely
*forecast*.

## The target

For each bar we estimate liquidity from data the model does not directly
read at prediction time:

- **Amihud illiquidity** — `mean(|return| / dollar_volume)` over a rolling
  window (Amihud, 2002): how much a unit of trading moves the price.
- **Kyle's λ** — the slope of returns on signed dollar order flow, with
  order-flow sign from the tick rule (Kyle, 1985).

The **label** is the Amihud illiquidity realised `horizon` bars *in the
future*, median-normalised to a bps-like scale:

```
target_bps(t) = base_bps · Amihud(t + horizon) / median(Amihud)
```

Because the target looks forward, predicting it from current features is a
real, falsifiable task — there is no closed-form shortcut, so the model can
actually be wrong. The scale is *relative* (not a broker-calibrated cost);
the point is that the circularity is broken.

## Features

The track uses the standard market features plus the microstructure bundle
(`features/microstructure.py`): dollar volume, Amihud, Roll spread,
order-flow imbalance, Garman-Klass / Parkinson volatility, and Kyle's λ.
There is no synthetic order size, so the √-impact heuristic baseline is not
applicable; the model is compared against mean, linear, and GBM baselines
with a Diebold-Mariano significance test.

## Run it

```bash
python scripts/run_empirical.py   # writes results/empirical_metrics.json
```

## Caveats

Amihud/λ are *estimates* from OHLCV, not realised execution cost; they are a
defensible, freely-available proxy for genuine price impact but still fall
short of broker fills or Level-2 depth. This track demonstrates that the
pipeline can learn a non-circular signal, not that it reproduces a
production transaction-cost model.
