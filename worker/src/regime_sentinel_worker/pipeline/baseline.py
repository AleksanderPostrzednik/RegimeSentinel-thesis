"""Baseline GARCH(1,1) rolling experiment and artifact gate."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from regime_sentinel_worker.artifacts.io import (
    artifact_hashes,
    base_manifest,
    started_at_utc,
    write_json,
)
from regime_sentinel_worker.artifacts.report import write_csv
from regime_sentinel_worker.experiment_protocol import validate_protocol_file
from regime_sentinel_worker.pipeline.backtest.christoffersen import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
)
from regime_sentinel_worker.pipeline.backtest.kupiec import kupiec_uc
from regime_sentinel_worker.pipeline.ingest import (
    build_input_manifest,
    load_frozen_snapshot,
    write_input_manifest,
)
from regime_sentinel_worker.pipeline.models.garch import GarchFit, fit_garch11
from regime_sentinel_worker.pipeline.preprocess import ReturnSeries, build_log_returns
from regime_sentinel_worker.pipeline.risk.var_es import (
    fz0_score,
    is_exceedance,
    parametric_var_es,
    quantile_loss,
)


MODEL_INNOVATIONS = ("student_t", "normal")
SUMMARY_FIELDS = (
    "instrument",
    "innovation",
    "confidence",
    "forecast_count",
    "valid_forecast_count",
    "fit_success_count",
    "fit_failure_count",
    "fit_success_rate",
    "expected_exceedances_valid",
    "expected_exceedances_full_oos",
    "observed_exceedances",
    "observed_exceedance_rate",
    "mean_var_quantile_loss",
    "mean_fz0_score",
    "kupiec_statistic",
    "kupiec_p_value",
    "kupiec_reject_at_5pct",
    "christoffersen_independence_statistic",
    "christoffersen_independence_p_value",
    "christoffersen_independence_reject_at_5pct",
    "christoffersen_conditional_coverage_statistic",
    "christoffersen_conditional_coverage_p_value",
    "christoffersen_conditional_coverage_reject_at_5pct",
)


@dataclass(frozen=True)
class BaselineVerification:
    complete: bool
    errors: tuple[str, ...]
    manifest: dict[str, Any] | None


def _protocol_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_dict(fit: GarchFit) -> dict[str, Any]:
    return {
        "innovation": fit.innovation,
        "parameters": fit.parameters,
        "sigma_percent": fit.sigma_percent,
        "log_likelihood": fit.log_likelihood,
        "aic": fit.aic,
        "bic": fit.bic,
        "convergence_flag": fit.convergence_flag,
    }


def _summary(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    key = f"{confidence:.2f}"
    valid = [row for row in rows if row.get("fit_success") and key in row["risk"]]
    flags = [bool(row["risk"][key]["exceedance"]) for row in valid]
    tail = 1.0 - confidence
    uc = kupiec_uc(flags, tail_probability=tail)
    independence = christoffersen_independence(flags, tail_probability=tail)
    conditional = christoffersen_conditional_coverage(flags, tail_probability=tail)
    quantile_scores = [row["risk"][key]["quantile_loss"] for row in valid]
    fz0_scores = [row["risk"][key]["fz0"] for row in valid]
    observed = sum(flags)
    return {
        "confidence": confidence,
        "tail_probability": tail,
        "forecast_count": len(rows),
        "n_valid_forecasts": len(valid),
        "n_exceedances": observed,
        "expected_exceedances_valid": len(valid) * tail,
        "expected_exceedances_full_oos": len(rows) * tail,
        "observed_exceedance_rate": observed / len(valid) if valid else None,
        "fit_success_rate": sum(bool(row["fit_success"]) for row in rows) / len(rows)
        if rows
        else 0.0,
        "mean_var_quantile_loss": statistics.fmean(quantile_scores)
        if quantile_scores
        else None,
        "mean_fz0_score": statistics.fmean(fz0_scores) if fz0_scores else None,
        "kupiec": uc.as_dict(),
        "christoffersen_independence": independence.as_dict(),
        "christoffersen_conditional_coverage": conditional.as_dict(),
    }


def _risk_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in row.items()
        if key not in {"fit", "risk", "fit_error", "risk_error"}
    }
    fit = row.get("fit") or {}
    output["fit_parameters_json"] = json.dumps(fit.get("parameters"), sort_keys=True)
    output["fit_log_likelihood"] = fit.get("log_likelihood")
    output["fit_aic"] = fit.get("aic")
    output["fit_bic"] = fit.get("bic")
    output["fit_sigma_percent"] = fit.get("sigma_percent")
    output["fit_convergence_flag"] = fit.get("convergence_flag")
    output["fit_error_json"] = json.dumps(row.get("fit_error"), sort_keys=True)
    output["risk_error"] = row.get("risk_error")
    for key, risk in sorted(row.get("risk", {}).items()):
        output[f"var_{key}_percent"] = risk["var_percent"]
        output[f"es_{key}_percent"] = risk["es_percent"]
        output[f"exceedance_{key}"] = risk["exceedance"]
        output[f"quantile_loss_{key}"] = risk["quantile_loss"]
        output[f"fz0_{key}"] = risk["fz0"]
    return output


def _summary_csv_row(
    instrument: str,
    innovation: str,
    confidence: float,
    model_summary: dict[str, Any],
    risk_summary: dict[str, Any],
) -> dict[str, Any]:
    uc = risk_summary["kupiec"]
    independence = risk_summary["christoffersen_independence"]
    conditional = risk_summary["christoffersen_conditional_coverage"]
    return {
        "instrument": instrument,
        "innovation": innovation,
        "confidence": confidence,
        "forecast_count": model_summary["forecast_count"],
        "valid_forecast_count": risk_summary["n_valid_forecasts"],
        "fit_success_count": model_summary["fit_success_count"],
        "fit_failure_count": model_summary["fit_failure_count"],
        "fit_success_rate": model_summary["fit_success_rate"],
        "expected_exceedances_valid": risk_summary["expected_exceedances_valid"],
        "expected_exceedances_full_oos": risk_summary["expected_exceedances_full_oos"],
        "observed_exceedances": risk_summary["n_exceedances"],
        "observed_exceedance_rate": risk_summary["observed_exceedance_rate"],
        "mean_var_quantile_loss": risk_summary["mean_var_quantile_loss"],
        "mean_fz0_score": risk_summary["mean_fz0_score"],
        "kupiec_statistic": uc["statistic"],
        "kupiec_p_value": uc["p_value"],
        "kupiec_reject_at_5pct": uc["reject_at_5pct"],
        "christoffersen_independence_statistic": independence["statistic"],
        "christoffersen_independence_p_value": independence["p_value"],
        "christoffersen_independence_reject_at_5pct": independence["reject_at_5pct"],
        "christoffersen_conditional_coverage_statistic": conditional["statistic"],
        "christoffersen_conditional_coverage_p_value": conditional["p_value"],
        "christoffersen_conditional_coverage_reject_at_5pct": conditional["reject_at_5pct"],
    }


def _returns_dict(processed: dict[str, ReturnSeries]) -> dict[str, Any]:
    return {
        "model_scale_factor": 100,
        "by_instrument": {
            instrument: {
                "dates": list(series.dates),
                "log_returns": list(series.log_returns),
                "model_returns": list(series.model_returns_percent),
            }
            for instrument, series in processed.items()
        },
    }


def verify_baseline_artifacts(
    artifact_root: str | Path,
    *,
    expected_forecasts: int,
    instruments: tuple[str, str] = ("BTC-USD", "ETH-USD"),
) -> BaselineVerification:
    """Verify counts and hashes without recomputing the experiment."""

    root = Path(artifact_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return BaselineVerification(False, ("baseline manifest.json is missing",), None)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BaselineVerification(False, (f"cannot parse baseline manifest: {exc}",), None)

    errors: list[str] = []
    if manifest.get("protocol_id") != "thesis-v1":
        errors.append("baseline manifest protocol_id is not thesis-v1")
    if manifest.get("status") not in {"complete", "complete_with_fit_failures"}:
        errors.append("baseline manifest status is not complete")
    if not manifest.get("git_commit") or manifest.get("git_commit") == "unavailable":
        errors.append("baseline manifest does not identify a git commit")
    for innovation in MODEL_INNOVATIONS:
        for instrument in instruments:
            forecast_path = root / innovation / instrument / "forecasts.json"
            summary_path = root / innovation / instrument / "summary.json"
            if not forecast_path.is_file():
                errors.append(f"missing forecasts artifact: {forecast_path.relative_to(root)}")
                continue
            if not summary_path.is_file():
                errors.append(f"missing summary artifact: {summary_path.relative_to(root)}")
            try:
                rows = json.loads(forecast_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"cannot parse {forecast_path.relative_to(root)}: {exc}")
                continue
            if not isinstance(rows, list) or len(rows) != expected_forecasts:
                errors.append(
                    f"{forecast_path.relative_to(root)} must contain {expected_forecasts} rows"
                )
    expected_hashes = manifest.get("artifact_sha256", {})
    actual_hashes = artifact_hashes(root)
    if set(expected_hashes) != set(actual_hashes):
        missing = sorted(set(actual_hashes) - set(expected_hashes))
        stale = sorted(set(expected_hashes) - set(actual_hashes))
        if missing:
            errors.append(f"manifest omits artifacts: {', '.join(missing)}")
        if stale:
            errors.append(f"manifest references missing artifacts: {', '.join(stale)}")
    for relative_path, expected_hash in expected_hashes.items():
        path = root / relative_path
        if not path.is_file() or _protocol_sha256(path) != expected_hash:
            errors.append(f"artifact hash mismatch: {relative_path}")
    return BaselineVerification(not errors, tuple(errors), manifest)


def run_baseline(
    *,
    protocol_path: str | Path,
    repo_root: str | Path,
    artifact_root: str | Path,
    fit_function: Callable[..., GarchFit] = fit_garch11,
    seed: int | None = None,
    forecast_limit: int | None = None,
) -> dict[str, Any]:
    """Run the complete fixed-window, refit-every-origin thesis-v1 baseline.

    forecast_limit is intentionally available only to the short integration test.
    The CLI does not expose it and therefore always runs all 1325 origins.
    """

    repo = Path(repo_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = validate_protocol_file(protocol_file, repo)
    snapshot = load_frozen_snapshot(protocol_file, repo_root=repo)
    processed = build_log_returns(
        snapshot,
        scale_factor=protocol["transformation"]["scaleFactor"],
    )
    info = protocol["informationSet"]
    window = int(info["estimationWindowReturns"])
    expected_forecasts = int(info["expectedForecastsPerInstrument"])
    confidence_levels = tuple(float(value) for value in protocol["risk"]["confidenceLevels"])
    protocol_seed = int(protocol["reproducibility"]["masterSeed"])
    if seed is not None and seed != protocol_seed:
        raise ValueError(f"seed must remain frozen at {protocol_seed}")
    if forecast_limit is not None and forecast_limit < 1:
        raise ValueError("forecast_limit must be positive")

    run_root = Path(artifact_root).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    started_at = started_at_utc()
    started_clock = time.perf_counter()

    input_manifest = build_input_manifest(
        snapshot,
        snapshot_path=str(snapshot.path.relative_to(repo)),
    )
    write_input_manifest(input_manifest, run_root / "input-manifest.json")
    write_json(run_root / "returns.json", _returns_dict(processed))
    failures: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    summary_csv_rows: list[dict[str, Any]] = []

    for innovation in MODEL_INNOVATIONS:
        summaries[innovation] = {}
        for instrument in ("BTC-USD", "ETH-USD"):
            series = processed[instrument]
            values = list(series.model_returns_percent)
            dates = series.dates
            available_forecasts = len(values) - window
            count = (
                available_forecasts
                if forecast_limit is None
                else min(forecast_limit, available_forecasts)
            )
            if forecast_limit is None and count != expected_forecasts:
                raise ValueError(f"OOS forecast count mismatch for {instrument}")
            rows: list[dict[str, Any]] = []
            instrument_root = run_root / innovation / instrument
            for origin in range(window, window + count):
                training = values[origin - window : origin]
                mean_percent = statistics.fmean(training)
                row: dict[str, Any] = {
                    "instrument": instrument,
                    "innovation": innovation,
                    "origin_index": origin,
                    "forecast_date": dates[origin],
                    "training_window_start": dates[origin - window],
                    "training_window_end": dates[origin - 1],
                    "actual_return_percent": values[origin],
                    "actual_loss_percent": -values[origin],
                    "training_mean_percent": mean_percent,
                    "fit_success": False,
                    "risk": {},
                }
                try:
                    fit = fit_function(
                        [value - mean_percent for value in training],
                        innovation=innovation,
                    )
                except Exception as exc:
                    failure = {
                        "instrument": instrument,
                        "innovation": innovation,
                        "forecast_date": dates[origin],
                        "origin_index": origin,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    row["fit_error"] = failure
                    failures.append(failure)
                else:
                    row["fit_success"] = True
                    row["fit"] = _fit_dict(fit)
                    for confidence in confidence_levels:
                        key = f"{confidence:.2f}"
                        try:
                            risk = parametric_var_es(
                                forecast_mean_percent=mean_percent,
                                sigma_percent=fit.sigma_percent,
                                confidence=confidence,
                                innovation=innovation,
                                student_t_df=fit.parameters.get("nu")
                                if innovation == "student_t"
                                else None,
                            )
                            actual_loss = row["actual_loss_percent"]
                            row["risk"][key] = {
                                "confidence": confidence,
                                "var_percent": risk.var,
                                "es_percent": risk.es,
                                "exceedance": is_exceedance(actual_loss, risk.var),
                                "quantile_loss": quantile_loss(
                                    actual_loss,
                                    risk.var,
                                    confidence,
                                ),
                                "fz0": fz0_score(
                                    actual_loss,
                                    risk.var,
                                    risk.es,
                                    confidence,
                                ),
                            }
                        except Exception as exc:
                            row.setdefault("risk_error", []).append(
                                {
                                    "confidence": confidence,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )
                rows.append(row)

            write_json(instrument_root / "forecasts.json", rows)
            csv_rows = [_risk_csv_row(row) for row in rows]
            csv_fields = sorted({key for row in csv_rows for key in row})
            write_csv(
                instrument_root / "forecasts.csv",
                csv_rows,
                fieldnames=csv_fields,
            )
            summary = {
                "instrument": instrument,
                "innovation": innovation,
                "forecast_count": len(rows),
                "fit_success_count": sum(bool(row["fit_success"]) for row in rows),
                "fit_failure_count": sum(not row["fit_success"] for row in rows),
                "fit_success_rate": sum(bool(row["fit_success"]) for row in rows) / len(rows),
                "risk": {
                    f"{confidence:.2f}": _summary(rows, confidence)
                    for confidence in confidence_levels
                },
            }
            write_json(instrument_root / "summary.json", summary)
            summaries[innovation][instrument] = summary
            for confidence in confidence_levels:
                summary_csv_rows.append(
                    _summary_csv_row(
                        instrument,
                        innovation,
                        confidence,
                        summary,
                        summary["risk"][f"{confidence:.2f}"],
                    )
                )

    write_json(run_root / "fit_failures.json", failures)
    write_csv(
        run_root / "fit_failures.csv",
        failures,
        fieldnames=(
            "instrument",
            "innovation",
            "forecast_date",
            "origin_index",
            "error_type",
            "error",
        ),
    )
    write_json(
        run_root / "summary.json",
        {
            "protocol_id": protocol["protocolId"],
            "expected_forecasts_per_instrument": expected_forecasts,
            "models": summaries,
        },
    )
    write_csv(run_root / "summary.csv", summary_csv_rows, fieldnames=SUMMARY_FIELDS)

    completed_at = started_at_utc()
    all_counts_complete = all(
        summaries[innovation][instrument]["forecast_count"]
        == (expected_forecasts if forecast_limit is None else forecast_limit)
        for innovation in MODEL_INNOVATIONS
        for instrument in ("BTC-USD", "ETH-USD")
    )
    manifest = base_manifest(
        protocol_id=protocol["protocolId"],
        protocol_sha256=_protocol_sha256(protocol_file),
        snapshot_id=snapshot.snapshot_id,
        snapshot_file_sha256=snapshot.file_sha256,
        repo_root=repo,
        seed=protocol_seed,
        started=started_at,
    )
    manifest.update(
        {
            "stage": "baseline",
            "status": (
                "complete"
                if all_counts_complete and not failures
                else "complete_with_fit_failures"
                if all_counts_complete
                else "failed"
            ),
            "expected_forecasts_per_instrument": expected_forecasts,
            "forecast_count_by_instrument": {
                instrument: summaries["student_t"][instrument]["forecast_count"]
                for instrument in ("BTC-USD", "ETH-USD")
            },
            "fit_failure_count": len(failures),
            "full_protocol_run": forecast_limit is None,
            "completed_at_utc": completed_at,
            "duration_seconds": time.perf_counter() - started_clock,
            "summary": summaries,
        }
    )
    manifest["artifact_sha256"] = artifact_hashes(run_root)
    write_json(run_root / "manifest.json", manifest)
    return manifest
