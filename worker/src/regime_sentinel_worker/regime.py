"""Regime stage: MSGARCH preflight, rolling forecast, and protocol fallback."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from regime_sentinel_worker.artifacts.io import artifact_hashes, base_manifest, started_at_utc, write_json
from regime_sentinel_worker.experiment_protocol import validate_protocol_file
from regime_sentinel_worker.pipeline.backtest.christoffersen import christoffersen_conditional_coverage, christoffersen_independence
from regime_sentinel_worker.pipeline.backtest.kupiec import kupiec_uc
from regime_sentinel_worker.pipeline.ingest import load_frozen_snapshot
from regime_sentinel_worker.pipeline.models.fallback import fit_markov_variance, normal_mixture_risk
from regime_sentinel_worker.pipeline.models.msgarch import (
    MsgarchFitError,
    MsgarchRunner,
    PreflightResult,
    run_preflight,
    validate_msgarch_fit,
)
from regime_sentinel_worker.pipeline.preprocess import build_log_returns
from regime_sentinel_worker.pipeline.risk.var_es import fz0_score, is_exceedance, quantile_loss
from regime_sentinel_worker.pipeline.baseline import verify_baseline_artifacts


def _date_string(value: Any) -> str:
    return value if isinstance(value, str) else value.isoformat()


def _protocol_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _center_training_window(values: Sequence[float]) -> tuple[tuple[float, ...], float]:
    if not values:
        raise ValueError("training window must not be empty")
    mean_percent = statistics.fmean(values)
    return tuple(float(value) - mean_percent for value in values), mean_percent


def _values_sha256(values: Sequence[float]) -> str:
    payload = "".join(f"{float(value):.17g}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summary(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    key = f"{confidence:.2f}"
    valid = [row for row in rows if row.get("fit_success") and key in row.get("risk", {})]
    flags = [bool(row["risk"][key]["exceedance"]) for row in valid]
    tail = 1.0 - confidence
    result: dict[str, Any] = {
        "confidence": confidence,
        "n_valid_forecasts": len(valid),
        "n_exceedances": sum(flags),
        "fit_success_rate": len(valid) / len(rows) if rows else 0.0,
        "kupiec": kupiec_uc(flags, tail_probability=tail).as_dict(),
        "christoffersen_independence": christoffersen_independence(flags, tail_probability=tail).as_dict(),
        "christoffersen_conditional_coverage": christoffersen_conditional_coverage(flags, tail_probability=tail).as_dict(),
    }
    if valid:
        result["mean_var_quantile_loss"] = statistics.fmean(
            quantile_loss(row["actual_loss_percent"], row["risk"][key]["var_percent"], confidence)
            for row in valid
        )
        result["mean_fz0_score"] = statistics.fmean(
            fz0_score(row["actual_loss_percent"], row["risk"][key]["var_percent"], row["risk"][key]["es_percent"], confidence)
            for row in valid
        )
    else:
        result["mean_var_quantile_loss"] = None
        result["mean_fz0_score"] = None
    return result


def _parameter_stability(forecast_path: Path) -> dict[str, Any]:
    rows = json.loads(forecast_path.read_text(encoding="utf-8"))
    values_by_parameter: dict[str, list[float]] = {}
    for row in rows:
        if not row.get("fit_success"):
            continue
        for name, value in (row.get("fit", {}).get("parameters", {}) or {}).items():
            numeric = float(value)
            if math.isfinite(numeric):
                values_by_parameter.setdefault(name, []).append(numeric)

    parameters: dict[str, Any] = {}
    for name, values in sorted(values_by_parameter.items()):
        mean = statistics.fmean(values)
        minimum, maximum = min(values), max(values)
        parameters[name] = {
            "n": len(values),
            "mean": mean,
            "min": minimum,
            "max": maximum,
            "range": maximum - minimum,
            "relative_range": (maximum - minimum) / max(abs(mean), 1e-12),
        }
    return {
        "forecast_path": forecast_path.name,
        "parameter_count": len(parameters),
        "fit_rows_with_parameters": sum(bool(row.get("fit_success")) for row in rows),
        "parameters": parameters,
        "max_relative_range": max(
            (value["relative_range"] for value in parameters.values()),
            default=0.0,
        ),
    }


def _preflight_artifact_dict(result: PreflightResult, artifact_root: Path) -> dict[str, Any]:
    payload = result.as_dict()
    artifact_root = artifact_root.resolve()
    for attempt in payload["attempts"]:
        log_path = Path(attempt["log_path"]).resolve()
        try:
            attempt["log_path"] = log_path.relative_to(artifact_root).as_posix()
        except ValueError:
            attempt["log_path"] = log_path.as_posix()
    return payload


def _write_failures(path: Path, failures: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures), encoding="utf-8")


def _row(
    *,
    date: str,
    values: list[float],
    mean_percent: float,
    risk: dict[str, dict[str, float]],
    fit_payload: dict[str, Any],
) -> dict[str, Any]:
    row_risk: dict[str, Any] = {}
    for key, value in risk.items():
        var_percent = float(value["var_percent"])
        es_percent = float(value["es_percent"])
        row_risk[key] = {
            "confidence": float(key),
            "var_percent": var_percent,
            "es_percent": es_percent,
            "exceedance": is_exceedance(-values[-1], var_percent),
        }
    return {
        "forecast_date": date,
        "actual_return_percent": values[-1],
        "actual_loss_percent": -values[-1],
        "training_mean_percent": mean_percent,
        "fit_success": True,
        "fit": fit_payload,
        "risk": row_risk,
    }


def _rolling_msgarch(
    *,
    instrument: str,
    values: list[float],
    dates: tuple[Any, ...],
    window: int,
    confidence_levels: tuple[float, ...],
    runner: MsgarchRunner,
    preflight: PreflightResult,
    root: Path,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best = next(
        attempt.fit for attempt in preflight.attempts
        if attempt.success and attempt.fit is not None and attempt.start_index == preflight.best_attempt_index
    )
    par0 = list(best.parameters.values())
    log_root = root / "logs" / "rolling_msgarch"
    for origin in range(window, len(values)):
        training = values[origin - window : origin]
        mean_percent = statistics.fmean(training)
        log_path = log_root / f"{instrument.replace('-', '_')}_{origin:04d}.log"
        try:
            fit = runner.fit(
                [value - mean_percent for value in training],
                start_index=0,
                mode="rolling",
                par0=par0,
                log_path=log_path,
            )
            validation_errors = validate_msgarch_fit(fit, occupancy_minimum=0.0)
            if validation_errors:
                raise MsgarchFitError("invalid rolling fit: " + "; ".join(validation_errors))
            par0 = list(fit.parameters.values())
            risk_with_forecast_mean = {
                key: {
                    "var_percent": float(value["var_percent"]) - mean_percent,
                    "es_percent": float(value["es_percent"]) - mean_percent,
                }
                for key, value in fit.risk.items()
            }
            rows.append(_row(
                date=_date_string(dates[origin]),
                values=[*training, values[origin]],
                mean_percent=mean_percent,
                risk=risk_with_forecast_mean,
                fit_payload={
                    "engine": "R-MSGARCH-FitML",
                    "model_label": "MS-GARCH",
                    "log_likelihood": fit.log_likelihood,
                    "parameters": fit.parameters,
                    "transition_matrix": [list(row) for row in fit.transition_matrix],
                    "filtered_last": list(fit.filtered_last),
                    "state_order": list(fit.state_order),
                    "unconditional_volatility": list(fit.unconditional_volatility),
                    "forecast_state_probability": "filtered",
                    "smoothed_probability_use": "post_hoc_only",
                    "viterbi_use": "post_hoc_path_only",
                },
            ))
        except Exception as exc:
            failure = {
                "instrument": instrument,
                "model": "MS-GARCH",
                "origin_index": origin,
                "forecast_date": _date_string(dates[origin]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "log_path": log_path.relative_to(root).as_posix(),
            }
            failures.append(failure)
            rows.append({
                "forecast_date": _date_string(dates[origin]),
                "actual_return_percent": values[origin],
                "actual_loss_percent": -values[origin],
                "training_mean_percent": mean_percent,
                "fit_success": False,
                "risk": {},
                "fit_error": failure,
            })
    summary = {
        "instrument": instrument,
        "model_label": "MS-GARCH",
        "engine": "R-MSGARCH-FitML",
        "forecast_count": len(rows),
        "fit_success_count": sum(bool(row["fit_success"]) for row in rows),
        "fit_failure_count": sum(not row["fit_success"] for row in rows),
        "fit_success_rate": sum(bool(row["fit_success"]) for row in rows) / len(rows),
        "risk": {f"{confidence:.2f}": _summary(rows, confidence) for confidence in confidence_levels},
    }
    write_json(root / "MS-GARCH" / instrument / "forecasts.json", rows)
    write_json(root / "MS-GARCH" / instrument / "summary.json", summary)
    return summary


def _rolling_fallback(
    *,
    instrument: str,
    values: list[float],
    dates: tuple[Any, ...],
    window: int,
    confidence_levels: tuple[float, ...],
    root: Path,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for origin in range(window, len(values)):
        training = values[origin - window : origin]
        mean_percent = statistics.fmean(training)
        try:
            fit = fit_markov_variance([value - mean_percent for value in training])
            risk = {
                f"{confidence:.2f}": {
                    "var_percent": normal_mixture_risk(
                        fit=fit, mean_percent=mean_percent, confidence=confidence
                    ).var,
                    "es_percent": normal_mixture_risk(
                        fit=fit, mean_percent=mean_percent, confidence=confidence
                    ).es,
                }
                for confidence in confidence_levels
            }
            rows.append(_row(
                date=_date_string(dates[origin]),
                values=[*training, values[origin]],
                mean_percent=mean_percent,
                risk=risk,
                fit_payload={
                    "engine": "python-statsmodels-markov-regression",
                    "model_label": "fallback_not_ms_garch",
                    "log_likelihood": fit.log_likelihood,
                    "parameters": fit.parameters,
                    "variances": list(fit.variances),
                    "transition_matrix": [list(row) for row in fit.transition_matrix],
                    "filtered_last": list(fit.filtered_last),
                    "occupancy": list(fit.occupancy),
                    "forecast_state_probability": "filtered",
                },
            ))
        except Exception as exc:
            failure = {
                "instrument": instrument,
                "model": "fallback_not_ms_garch",
                "origin_index": origin,
                "forecast_date": _date_string(dates[origin]),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            rows.append({
                "forecast_date": _date_string(dates[origin]),
                "actual_return_percent": values[origin],
                "actual_loss_percent": -values[origin],
                "training_mean_percent": mean_percent,
                "fit_success": False,
                "risk": {},
                "fit_error": failure,
            })
    summary = {
        "instrument": instrument,
        "model_label": "fallback_not_ms_garch",
        "engine": "python-statsmodels-markov-regression",
        "forecast_count": len(rows),
        "fit_success_count": sum(bool(row["fit_success"]) for row in rows),
        "fit_failure_count": sum(not row["fit_success"] for row in rows),
        "fit_success_rate": sum(bool(row["fit_success"]) for row in rows) / len(rows),
        "risk": {f"{confidence:.2f}": _summary(rows, confidence) for confidence in confidence_levels},
    }
    write_json(root / "fallback_not_ms_garch" / instrument / "forecasts.json", rows)
    write_json(root / "fallback_not_ms_garch" / instrument / "summary.json", summary)
    return summary


def _comparison(
    *,
    baseline_root: Path,
    regime_root: Path,
    instrument: str,
    variant: str,
) -> dict[str, Any]:
    baseline = json.loads((baseline_root / "student_t" / instrument / "summary.json").read_text(encoding="utf-8"))
    regime = json.loads((regime_root / variant / instrument / "summary.json").read_text(encoding="utf-8"))
    baseline_forecasts = baseline_root / "student_t" / instrument / "forecasts.json"
    regime_forecasts = regime_root / variant / instrument / "forecasts.json"
    return {
        "instrument": instrument,
        "selection_rule": "feasibility_and_stability_only; never OOS performance",
        "oos_model_selection_used": False,
        "models": {
            "garch11-studentt": baseline,
            variant: regime,
        },
        "dimensions": {
            "fit_and_forecast_success_rate": {
                "baseline": baseline["fit_success_rate"],
                "regime": regime["fit_success_rate"],
            },
            "var_quantile_loss": {
                confidence: {
                    "baseline": baseline["risk"][confidence]["mean_var_quantile_loss"],
                    "regime": regime["risk"][confidence]["mean_var_quantile_loss"],
                }
                for confidence in baseline["risk"]
            },
            "joint_var_es_fz0": {
                confidence: {
                    "baseline": baseline["risk"][confidence]["mean_fz0_score"],
                    "regime": regime["risk"][confidence]["mean_fz0_score"],
                }
                for confidence in baseline["risk"]
            },
            "stability": {
                "baseline": _parameter_stability(baseline_forecasts),
                variant: _parameter_stability(regime_forecasts),
            },
            "observed_vs_expected_exceedances": {
                confidence: {
                    "baseline": {
                        "observed": baseline["risk"][confidence]["n_exceedances"],
                        "expected": baseline["forecast_count"] * (1.0 - float(confidence)),
                    },
                    "regime": {
                        "observed": regime["risk"][confidence]["n_exceedances"],
                        "expected": regime["forecast_count"] * (1.0 - float(confidence)),
                    },
                }
                for confidence in baseline["risk"]
            },
            "coverage_and_independence": {
                confidence: {
                    "baseline": {
                        "kupiec": baseline["risk"][confidence]["kupiec"],
                        "independence": baseline["risk"][confidence]["christoffersen_independence"],
                    },
                    "regime": {
                        "kupiec": regime["risk"][confidence]["kupiec"],
                        "independence": regime["risk"][confidence]["christoffersen_independence"],
                    },
                }
                for confidence in baseline["risk"]
            },
        },
    }


def _execute_msgarch_preflight(
    *,
    processed: dict[str, Any],
    window: int,
    protocol: dict[str, Any],
    runner: MsgarchRunner,
    root: Path,
) -> tuple[dict[str, PreflightResult], dict[str, Any]]:
    results: dict[str, PreflightResult] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for instrument in ("BTC-USD", "ETH-USD"):
        initial_window = processed[instrument].model_returns_percent[:window]
        centered_window, training_mean = _center_training_window(initial_window)
        inputs[instrument] = {
            "observation_count": len(centered_window),
            "training_mean_percent": training_mean,
            "centered_mean_percent": statistics.fmean(centered_window),
            "centered_values_sha256": _values_sha256(centered_window),
        }
        results[instrument] = run_preflight(
            instrument=instrument,
            values=centered_window,
            runner=runner,
            log_root=root / "logs" / "preflight",
            starts=5,
            occupancy_minimum=float(protocol["models"]["regimeCandidate"]["preflight"]["minimumStateOccupancy"]),
            transition_tolerance=float(protocol["models"]["regimeCandidate"]["preflight"]["transitionRowSumTolerance"]),
            repeat_tolerance=float(protocol["models"]["regimeCandidate"]["preflight"]["repeatParameterTolerance"]),
        )

    payload = {
        instrument: _preflight_artifact_dict(result, root)
        for instrument, result in results.items()
    }
    for instrument, result_payload in payload.items():
        result_payload["input"] = inputs[instrument]
    return results, payload


def run_msgarch_preflight_stage(
    *,
    protocol_path: str | Path,
    repo_root: str | Path,
    baseline_root: str | Path,
    artifact_root: str | Path,
    rscript: str = "Rscript",
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = validate_protocol_file(protocol_file, repo)
    snapshot = load_frozen_snapshot(protocol_file, repo_root=repo)
    processed = build_log_returns(snapshot, scale_factor=protocol["transformation"]["scaleFactor"])
    info = protocol["informationSet"]
    window = int(info["estimationWindowReturns"])
    expected_forecasts = int(info["expectedForecastsPerInstrument"])
    root = Path(artifact_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    started = started_at_utc()
    gate = verify_baseline_artifacts(baseline_root, expected_forecasts=expected_forecasts)
    write_json(
        root / "baseline_gate.json",
        {"complete": gate.complete, "errors": list(gate.errors), "manifest": gate.manifest},
    )
    manifest = base_manifest(
        protocol_id=protocol["protocolId"],
        protocol_sha256=_protocol_sha256(protocol_file),
        snapshot_id=snapshot.snapshot_id,
        snapshot_file_sha256=snapshot.file_sha256,
        repo_root=repo,
        seed=int(protocol["reproducibility"]["masterSeed"]),
        started=started,
        rscript=rscript,
    )
    if not gate.complete:
        manifest.update({
            "stage": "msgarch_preflight",
            "status": "blocked_baseline_incomplete",
            "baseline_gate_errors": list(gate.errors),
            "fallback_triggered": False,
            "completed_at_utc": started_at_utc(),
        })
        manifest["artifact_sha256"] = artifact_hashes(root)
        write_json(root / "manifest.json", manifest)
        return manifest

    runner = MsgarchRunner(
        repo_root=repo,
        rscript=rscript,
        seed=int(protocol["reproducibility"]["masterSeed"]),
    )
    results, payload = _execute_msgarch_preflight(
        processed=processed,
        window=window,
        protocol=protocol,
        runner=runner,
        root=root,
    )
    write_json(root / "preflight.json", payload)

    attempts = [attempt for result in results.values() for attempt in result.attempts]
    passed = all(result.passed for result in results.values())
    manifest.update({
        "stage": "msgarch_preflight",
        "status": "passed" if passed else "failed",
        "model_variant": "MS-GARCH",
        "engine": "R-MSGARCH-FitML",
        "states": int(protocol["models"]["regimeCandidate"]["states"]),
        "forecast_state_probability": protocol["informationSet"]["stateProbabilityForForecast"],
        "attempt_count": len(attempts),
        "fit_return_count": sum(attempt.fit is not None for attempt in attempts),
        "valid_fit_count": sum(attempt.success for attempt in attempts),
        "preflight_failure_count": sum(not attempt.success for attempt in attempts),
        "preflight_passed_for_both_instruments": passed,
        "fallback_triggered": False,
        "completed_at_utc": started_at_utc(),
    })
    manifest["artifact_sha256"] = artifact_hashes(root)
    write_json(root / "manifest.json", manifest)
    return manifest


def run_regime_stage(
    *,
    protocol_path: str | Path,
    repo_root: str | Path,
    baseline_root: str | Path,
    artifact_root: str | Path,
    rscript: str = "Rscript",
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = validate_protocol_file(protocol_file, repo)
    snapshot = load_frozen_snapshot(protocol_file, repo_root=repo)
    processed = build_log_returns(snapshot, scale_factor=protocol["transformation"]["scaleFactor"])
    info = protocol["informationSet"]
    window = int(info["estimationWindowReturns"])
    expected_forecasts = int(info["expectedForecastsPerInstrument"])
    confidence_levels = tuple(float(value) for value in protocol["risk"]["confidenceLevels"])
    root = Path(artifact_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = started_at_utc()
    gate = verify_baseline_artifacts(baseline_root, expected_forecasts=expected_forecasts)
    write_json(root / "baseline_gate.json", {"complete": gate.complete, "errors": list(gate.errors), "manifest": gate.manifest})

    manifest = base_manifest(
        protocol_id=protocol["protocolId"],
        protocol_sha256=_protocol_sha256(protocol_file),
        snapshot_id=snapshot.snapshot_id,
        snapshot_file_sha256=snapshot.file_sha256,
        repo_root=repo,
        seed=int(protocol["reproducibility"]["masterSeed"]),
        started=started,
        rscript=rscript,
    )
    if not gate.complete:
        manifest.update({
            "stage": "regime",
            "status": "blocked_baseline_incomplete",
            "model_variant": None,
            "fallback_triggered": False,
            "fallback_reason": "baseline_artifacts_incomplete",
            "baseline_gate_errors": list(gate.errors),
            "completed_at_utc": started_at_utc(),
        })
        manifest["artifact_sha256"] = artifact_hashes(root)
        write_json(root / "manifest.json", manifest)
        return manifest

    runner = MsgarchRunner(repo_root=repo, rscript=rscript, seed=int(protocol["reproducibility"]["masterSeed"]))
    preflight_results, preflight_payload = _execute_msgarch_preflight(
        processed=processed,
        window=window,
        protocol=protocol,
        runner=runner,
        root=root,
    )
    write_json(
        root / "preflight.json",
        preflight_payload,
    )
    preflight_passed = all(result.passed for result in preflight_results.values())
    failures: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    if preflight_passed:
        for instrument in ("BTC-USD", "ETH-USD"):
            series = processed[instrument]
            summaries[instrument] = _rolling_msgarch(
                instrument=instrument,
                values=list(series.model_returns_percent),
                dates=series.dates,
                window=window,
                confidence_levels=confidence_levels,
                runner=runner,
                preflight=preflight_results[instrument],
                root=root,
                failures=failures,
            )
        rolling_success = {
            instrument: summaries[instrument]["fit_success_rate"]
            for instrument in summaries
        }
        rolling_passed = all(rate >= 0.99 for rate in rolling_success.values())
    else:
        rolling_success = {}
        rolling_passed = False

    fallback_triggered = not preflight_passed or not rolling_passed
    variant = "MS-GARCH"
    if fallback_triggered:
        variant = "fallback_not_ms_garch"
        for instrument in ("BTC-USD", "ETH-USD"):
            series = processed[instrument]
            summaries[instrument] = _rolling_fallback(
                instrument=instrument,
                values=list(series.model_returns_percent),
                dates=series.dates,
                window=window,
                confidence_levels=confidence_levels,
                root=root,
                failures=failures,
            )
    for instrument in ("BTC-USD", "ETH-USD"):
        write_json(root / "comparison" / f"{instrument}.json", _comparison(
            baseline_root=Path(baseline_root),
            regime_root=root,
            instrument=instrument,
            variant=variant,
        ))
    _write_failures(root / "fit_failures.jsonl", failures)
    manifest.update({
        "stage": "regime",
        "status": "complete" if not failures else "complete_with_fit_failures",
        "model_variant": variant,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": (
            "preflight_failed_for_at_least_one_instrument"
            if not preflight_passed
            else "rolling_success_below_0.99_for_at_least_one_instrument"
            if not rolling_passed
            else None
        ),
        "preflight_passed_for_both_instruments": preflight_passed,
        "rolling_success_rate": rolling_success,
        "summary": summaries,
        "fit_failure_count": len(failures),
        "preflight_failure_count": sum(
            not attempt.success
            for result in preflight_results.values()
            for attempt in result.attempts
        ),
        "completed_at_utc": started_at_utc(),
    })
    manifest["artifact_sha256"] = artifact_hashes(root)
    write_json(root / "manifest.json", manifest)
    return manifest
