# Controlled R/MSGARCH attempt 2

Run ID: `true-msgarch-attempt-2-20260818T174828Z`.

This package records the second and final targeted thesis-v1 stabilization
attempt for the real Python -> Rscript -> `MSGARCH::FitML` preflight. It
continues commit `53caba53e1612fd797dd2addffe3c84f51158c47` from PR #31 and
uses code commit `0d99fe083425c60a4138fde944528e98291b7400`.

## Outcome

- The shared-`nu` initialization bug was fixed without changing the model,
  likelihood, BFGS optimizer, protocol, data, window, state count, thresholds,
  or gate.
- The existing R 4.4.3 / MSGARCH 2.51 runtime was reused; bootstrap was not
  repeated.
- All 10 FitML calls returned valid fits and all optimizers reported
  `convergence = 0`.
- BTC-USD failed repeatability with canonical delta
  `0.7106245999964016 > 1e-6`.
- ETH-USD failed repeatability with canonical delta
  `8.361508087301651 > 1e-6`.
- Every occupancy check passed. The old degenerate ETH fit was not reproduced;
  the smallest new ETH occupancy was `0.06781293052010788 > 0.05`.
- The joint gate failed. Normal robustness and rolling were not started, no
  rolling/scored OOS forecast artifact was produced, no OOS observation was
  used, and no third stabilization attempt is authorized.
- Decision recommendation: close the Student-t MS-GARCH path for thesis-v1 and
  proceed with protocol variant B, explicitly labelled
  `fallback_not_ms_garch`. This package prepares that decision but does not
  execute or relabel the fallback.

## Package map

- `manifest.json`: runtime, input, code, gate status, and artifact hashes.
- `environment.json`: reused runtime, exact command, input hashes, and scope.
- `preflight.json`: preserved payloads for all ten real FitML calls.
- `numerical_diagnostics.json`: optimizer, Hessian, clustering, label,
  occupancy, root-cause, and variant-B decision evidence.
- `baseline_gate.json`: read-only verification of the existing baseline.
- `logs/preflight/`: raw stdout/stderr for all ten calls, including intended
  and effective starts, seeds, optimizer status, counts, objective, and Hessian.

## Evidence boundary

The Hessian is the raw finite-difference Hessian returned by
`stats::optim` in the reduced, transformed nine-dimensional optimizer
coordinate system. It is not a Hessian in natural or canonical parameter
coordinates.

The input consists only of the first 500 centered returns per instrument,
from 2021-07-21 through 2022-12-02. Their hashes are unchanged from attempt 1:

- BTC-USD: `e57dca967a82dc0cf4c9f37521e3059ddeefd57f5d44f73f1e3f242e67d383fa`;
- ETH-USD: `026f98ed9f6701f9cf984e5cf78bd7926fed920d602fb2626fcb7107e63d435a`.

No later observation was used to choose a start, fit, model, or decision. The
raw `preflight.json` and all logs are preserved; the post-run files only
summarize them.

## Exact command used

```bash
cd worker
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m regime_sentinel_worker.main msgarch-preflight \
  --rscript /tmp/regimesentinel-msgarch-20260818-attempt1/env/bin/Rscript \
  --artifacts /home/alek/projects/regimeSentinel-true-msgarch-attempt-2/artifacts/thesis-v1/runs/true-msgarch-attempt-2-20260818T174828Z \
  --baseline-artifacts /home/alek/projects/regimeSentinel-true-msgarch-attempt-2/artifacts/thesis-v1/baseline
```

The `/tmp` runtime is ephemeral. A future authorized reproduction must first
verify it or use the repository's pinned project-local runtime, and must always
write to another new empty directory.
