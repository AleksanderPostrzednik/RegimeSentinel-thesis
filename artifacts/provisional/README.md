# PROVISIONAL — thesis-v1 tables and charts

PROVISIONAL — exploratory artifact summary; no canonical model selected and no superiority claim is made.

This package is generated only from the existing, hash-verified `thesis-v1` artifacts.
It reports all three recorded variants separately: GARCH(1,1) Student-t, GARCH(1,1) normal,
and `fallback_not_ms_garch`. It does not select a canonical model and does not state superiority.

## Scope

- BTC-USD and ETH-USD are shown separately.
- VaR/ES are positive-loss quantities at 95% and 99%.
- Valid forecast counts and fit failures are shown explicitly; missing fits are not silently imputed.
- Observed exceedances are compared with the expectation for valid forecasts; the full-OOS expectation is retained to expose missing-fit coverage.
- Kupiec, Christoffersen independence, Christoffersen conditional coverage, quantile loss and FZ0 are reported as diagnostics.
- Differences are descriptive `BTC-USD minus ETH-USD`; they are not a model ranking.
- `fallback_not_ms_garch` is a protocol fallback label, not an MS-GARCH result.

## Tables

- `tables/model_metrics.csv` and `tables/model_metrics.md` — one row per variant, instrument and confidence level.
- `tables/btc_eth_differences.csv` and `tables/btc_eth_differences.md` — descriptive BTC-minus-ETH differences.

## Charts

All SVG charts carry the `PROVISIONAL` notice in their title and accessibility description.

- `charts/backtest_pvalues_95.svg`
- `charts/backtest_pvalues_99.svg`
- `charts/exceedances_95.svg`
- `charts/exceedances_99.svg`
- `charts/fit_success_rate.svg`
- `charts/mean_var_es_95.svg`
- `charts/mean_var_es_99.svg`
- `charts/scores_95.svg`
- `charts/scores_99.svg`

## Provenance

- Protocol: `thesis-v1` (`33a4ff32c11743811a94d6e1eababc72c2d9711d7cb9d32bd3ad47db171cc9f5`).
- Snapshot: `yahoo-btc-eth-daily-close-2021-07-20_2026-07-19` (`eecdf04ef85451ba1eae8ed8ad776ea5079b43156863d431d4ff38572493240d`).
- Baseline source run: `2026-08-14T15:36:31Z`.
- Regime source run: `2026-08-14T15:44:14Z`.
- Full source and output hashes are in `manifest.json`.

## Reproduction

From the repository root run:

```bash
PYTHONPATH=worker/src python3 -m regime_sentinel_worker.artifacts.provisional_report
PYTHONPATH=worker/src python3 -m unittest worker.tests.test_provisional_report -v
```

Do not copy this provisional package into the thesis as a final result without Alek's methodological review.
