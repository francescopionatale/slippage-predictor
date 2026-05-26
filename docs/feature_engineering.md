# Feature engineering

All features are computed with strict no-look-ahead semantics: the row
for bar `t` uses only data ≤ `t`. The implementation lives in
`src/slippage/features/{market,spread,order}.py`.

## Market features

Computed by `slippage.features.compute_market_features(df, vol_window=20)`.

| Feature | Description |
|---------|-------------|
| `ret_1` … `ret_12` | Log-returns over 1, 3, 6, 12 bars (`log(close_t / close_{t-k})`) |
| `vol_rolling`      | Rolling std of 1-bar log-returns (20-bar window by default) |
| `range_rel`        | `(High − Low) / Close` — intra-bar range as a fraction of close |
| `vol_ratio`        | `volume_t / rolling_mean(volume, 20)` |
| `tod_sin`, `tod_cos` | Time-of-day on a 24-hour cycle (Eastern time), sin/cos encoded |
| `spread_cs`        | Corwin–Schultz bid-ask spread estimate from OHLC only |
| `is_rth`           | 1.0 if the bar falls within 09:30–16:00 Eastern, else 0.0 |

The 24-hour `tod` encoding (rather than mapping just NYSE 9:30–16:00 to
one period) gives unique encodings to pre- and post-market bars, which
1h yfinance data sometimes includes.

## Corwin–Schultz spread estimator

The Corwin & Schultz (2012) formula estimates the bid-ask spread from
the High/Low pair of two consecutive bars. The implementation lives in
`slippage.features.spread._corwin_schultz_spread`.

Negative estimates (a known noise artifact of the closed-form
derivation) are clipped to zero. On the v3 ticker universe ~5-15% of
hourly bars yield exactly zero; the remainder land roughly between
1–15 bps for ultra-liquid names and 20–60 bps for the illiquid name.
The `plot_spread_cs_distribution` plot in the diagnostic report
visualises this per ticker.

Reference: Corwin & Schultz (2012), *A Simple Way to Estimate Bid-Ask
Spreads from Daily High and Low Prices*.

## Synthetic order features

Computed by `slippage.features.add_synthetic_orders(market_feats, rng,
orders_per_bar=4, size_low=0.001, size_high=0.05)`.

| Feature | Description |
|---------|-------------|
| `side`                  | +1 (buy) or −1 (sell), balanced per bar  |
| `order_size_fraction`   | ∼ Uniform(0.001, 0.05) — fraction of avg volume |
| `urgency`               | ∼ Uniform(0, 1) — proxy for execution aggressiveness |

Per bar, `orders_per_bar` synthetic orders are generated (default 4 =
2 buy + 2 sell). Reproducibility requires an explicit `numpy.random.
Generator` — `pipeline.build_full_proxy` derives an independent
deterministic seed per ticker so order generation never depends on dict
iteration order.

## Feature catalogue

Three constants in `slippage.features` enumerate the feature columns:

- `FEATURE_NAMES_MARKET` (11 names): the market features above.
- `FEATURE_NAMES_ORDER` (3 names): `side`, `order_size_fraction`, `urgency`.
- `FEATURE_NAMES = FEATURE_NAMES_MARKET + FEATURE_NAMES_ORDER` (14 names).
- `FEATURE_NAMES_TRAINING` (13 names): `FEATURE_NAMES` minus `side`.

`side` is dropped from the training matrix because it cancels
algebraically in the proxy formula (see [proxy_construction.md](proxy_construction.md)).
It is kept in the feature dataframe so segment breakdown and visual
diagnostics can still bucket by buy/sell.
