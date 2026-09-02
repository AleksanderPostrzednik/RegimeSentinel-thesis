# Controlled R/MSGARCH attempt 1

Run ID: `true-msgarch-attempt-1-20260818T153348Z`.

This package records the first real thesis-v1 preflight through
Python -> Rscript -> `MSGARCH::FitML`. It does not contain a rolling
MS-GARCH run and it does not relabel or execute the fallback.

## Outcome

- R 4.4.3 and MSGARCH 2.51 were available through the pinned user-local runtime.
- All 10 FitML processes returned parsable two-state fits.
- Nine fits passed the response contract.
- BTC-USD failed the repeatability gate: maximum canonical parameter delta
  `12.862472166204588 > 1e-6`.
- ETH-USD failed the repeatability gate:
  `0.01577760859298616 > 1e-6`; start index 2 also failed the 5% occupancy
  floor with state-2 occupancy `0.0007827370526442916`.
- The joint gate therefore failed. Rolling was not started,
  `n_valid_forecasts = 0`, and fallback was not triggered.

## Package map

- `manifest.json`: versions, hashes, counts and gate status.
- `environment.json`: platform, exact runtime pins, input hashes and commands.
- `preflight.json`: all ten fit payloads and validation checks.
- `preflight_integrity_review.json`: deterministic post-run revalidation of
  the preserved fit payloads; no FitML call was repeated.
- `baseline_gate.json`: immutable baseline verification input.
- `logs/preflight/`: raw stdout/stderr for all ten real FitML calls.
- `logs/smoke/minimal-btc-before-par0-fix.log`: minimal implementation failure.
- `logs/smoke/minimal-btc-after-par0-fix.log`: first successful real fit.

Integrity note: the preserved raw `preflight.json` incorrectly reports the
aggregate ETH occupancy check as `true`; the per-attempt error and failed gate
were correct. The integrity review recomputes occupancy as `false`, and commit
`5b18876` fixes aggregation for future runs.

## Exact reproduction

From the repository root, create the pinned project-local runtime:

```bash
worker/r/bootstrap_msgarch_env.sh
```

Verify it:

```bash
command -v .runtime/msgarch-r4.4.3-msgarch2.51/env/bin/R
command -v .runtime/msgarch-r4.4.3-msgarch2.51/env/bin/Rscript
.runtime/msgarch-r4.4.3-msgarch2.51/env/bin/R --version
.runtime/msgarch-r4.4.3-msgarch2.51/env/bin/Rscript --version
.runtime/msgarch-r4.4.3-msgarch2.51/env/bin/Rscript -e 'stopifnot(requireNamespace("MSGARCH", quietly=TRUE)); cat(as.character(packageVersion("MSGARCH")))'
```

Run the complete preflight stage into a new empty directory:

```bash
cd worker
PYTHONPATH=src python3 -m regime_sentinel_worker.main msgarch-preflight \
  --rscript ../.runtime/msgarch-r4.4.3-msgarch2.51/env/bin/Rscript \
  --artifacts ../artifacts/msgarch-attempt-1-rerun/<new-run-id> \
  --baseline-artifacts ../artifacts/baseline
```

The command deliberately refuses a non-empty output directory.
The frozen protocol and existing baseline/fallback/provisional artifacts are
read-only inputs and are never overwritten by this stage.
