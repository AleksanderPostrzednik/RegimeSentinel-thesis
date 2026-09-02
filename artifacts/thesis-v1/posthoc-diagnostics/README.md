# thesis-v1 post-hoc diagnostics

Status: POST_HOC_DESCRIPTIVE

This separate descriptive package reads existing thesis-v1 artifacts only. It
is outside the frozen experiment and does not change protocol, data, window,
models, gates, or existing baseline/regime/provisional artifacts.

## Contents

- charts/log_returns.svg: log-returns for BTC-USD and ETH-USD.
- charts/var_es_exceedances_95.svg: existing baseline Student-t VaR/ES at 95 percent with exceedances.
- charts/var_es_exceedances_99.svg: existing baseline Student-t VaR/ES at 99 percent with exceedances.
- charts/msgarch_preflight_loglikelihood.svg: preflight starts and recorded objective values.
- charts/msgarch_preflight_occupancy.svg: preflight starts and recorded occupancy.
- tables/baseline_diagnostics.csv and tables/baseline_diagnostics.md: available baseline diagnostics, including separate Christoffersen independence and conditional coverage p-values.
- tables/msgarch_preflight_diagnostics.csv and tables/msgarch_preflight_diagnostics.md: all preflight attempts.
- metadata.json: scope and residual availability.
- manifest.json: provenance and source/output hashes.

VaR/ES charts use existing GARCH(1,1) Student-t baseline forecasts. They plot
the stored loss, VaR, ES, and exceedance flag. No forecast was recomputed.

The regime source is labelled only fallback_not_ms_garch. It is not treated as
a MS-GARCH result. This package selects no winner and does not revise any OOS
conclusion.

## Missing diagnostics

The final preserved preflight contains 10 rows, with
10 successful fits,
10 finite log-likelihood values and 10 rows containing
occupancy. The repeatability gate remains a source-run limitation; no values
are imputed.

Standardized residual series were not stored in the existing baseline forecast
JSON or regime fallback forecast artifacts. No silent refit was performed.
Adding residual diagnostics requires a separate methodological choice and a
pipeline output contract that persists per-origin standardized residuals,
followed by a separately identified run and provenance review.

## Reproduction

From the repository root:

    PYTHONPATH=worker/src python3 -m regime_sentinel_worker.artifacts.posthoc_diagnostic_report
    PYTHONPATH=worker/src python3 -m unittest worker.tests.test_posthoc_diagnostic_report -v

The generator is deterministic for the same source artifacts. It does not
download data, invoke Rscript, or fit any MS-GARCH model.
