from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
from datetime import date, datetime, time, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

from regime_sentinel_worker.artifacts.io import (
    artifact_hashes,
    git_commit,
    sha256_file,
    write_json,
)
from regime_sentinel_worker.experiment_protocol import _validate_json_schema_instance
from regime_sentinel_worker.pipeline.ingest import INSTRUMENTS, PriceSnapshot, load_price_snapshot
from regime_sentinel_worker.pipeline.models.fallback import (
    MarkovVarianceFit,
    fit_markov_variance,
    normal_mixture_risk,
)
from regime_sentinel_worker.pipeline.models.garch import GarchFit, fit_garch11
from regime_sentinel_worker.pipeline.preprocess import build_log_returns
from regime_sentinel_worker.pipeline.risk.var_es import VaREsForecast, parametric_var_es

UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "protocol" / "monitoring-v1.json"
PROTOCOL_SCHEMA_PATH = REPO_ROOT / "contracts" / "monitoring-protocol.v1.schema.json"
RESULT_SCHEMA_PATH = REPO_ROOT / "contracts" / "monitoring-result.v1.schema.json"
HISTORY_SCHEMA_PATH = REPO_ROOT / "contracts" / "monitoring-history.v1.schema.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "monitoring-v1"
SOURCE_URLS = [
    "https://finance.yahoo.com/quote/BTC-USD/history/",
    "https://finance.yahoo.com/quote/ETH-USD/history/",
]

PriceRows = dict[str, Sequence[tuple[str, float]]]
PriceFetcher = Callable[[Sequence[str], date, date], PriceRows]
GarchFitter = Callable[..., GarchFit]
RegimeFitter = Callable[[Sequence[float]], MarkovVarianceFit]


class MonitoringValidationError(ValueError):
    pass


class MonitoringRunError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitoringValidationError(f"Cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MonitoringValidationError(f"Expected a JSON object in {path}")
    return value


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise MonitoringValidationError("UTC timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _validate_now(now_utc: datetime) -> datetime:
    if now_utc.tzinfo is None:
        raise MonitoringValidationError("now_utc must be timezone-aware")
    return now_utc.astimezone(UTC)


def validate_monitoring_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(path)
    schema = _read_json(PROTOCOL_SCHEMA_PATH)
    errors: list[str] = []
    _validate_json_schema_instance(protocol, schema, errors)

    variants = protocol.get("estimation", {}).get("baseline", {}).get("variants", [])
    expected_variants = [
        {"modelId": "garch11-studentt", "innovation": "student_t"},
        {"modelId": "garch11-normal", "innovation": "normal"},
    ]
    if variants != expected_variants:
        errors.append("$.estimation.baseline.variants must contain the approved variants in order")
    window = protocol.get("estimation", {}).get("windowReturns")
    minimum_rows = protocol.get("data", {}).get("minimumPriceRows")
    if isinstance(window, int) and isinstance(minimum_rows, int) and minimum_rows < window + 1:
        errors.append("$.data.minimumPriceRows must allow the configured return window")

    if errors:
        raise MonitoringValidationError("Invalid monitoring protocol:\n- " + "\n- ".join(errors))
    return protocol


def validate_monitoring_result(value: dict[str, Any]) -> None:
    schema = _read_json(RESULT_SCHEMA_PATH)
    errors: list[str] = []
    _validate_json_schema_instance(value, schema, errors)

    instruments = value.get("instruments", [])
    names = [item.get("instrument") for item in instruments if isinstance(item, dict)]
    if names != list(INSTRUMENTS):
        errors.append("$.instruments must contain BTC-USD and ETH-USD in canonical order")
    if errors:
        raise MonitoringValidationError("Invalid monitoring result:\n- " + "\n- ".join(errors))


def validate_monitoring_history(value: dict[str, Any]) -> None:
    schema = _read_json(HISTORY_SCHEMA_PATH)
    errors: list[str] = []
    _validate_json_schema_instance(value, schema, errors)

    instruments = value.get("instruments", [])
    names = [item.get("instrument") for item in instruments if isinstance(item, dict)]
    if names != list(INSTRUMENTS):
        errors.append("$.instruments must contain BTC-USD and ETH-USD in canonical order")

    window = value.get("historyWindow", {})
    point_count = window.get("pointCount") if isinstance(window, dict) else None
    canonical_dates: list[str] | None = None
    if isinstance(point_count, int):
        for instrument_index, instrument in enumerate(instruments):
            if not isinstance(instrument, dict):
                continue
            points = instrument.get("points")
            if not isinstance(points, list):
                continue
            if len(points) != point_count:
                errors.append(
                    f"$.instruments[{instrument_index}].points must match historyWindow.pointCount"
                )
                continue
            dates: list[str] = []
            for point_index, point in enumerate(points):
                if not isinstance(point, dict):
                    continue
                raw_day = point.get("dateUtc")
                if not isinstance(raw_day, str):
                    continue
                try:
                    parsed_day = date.fromisoformat(raw_day)
                except ValueError:
                    errors.append(
                        f"$.instruments[{instrument_index}].points[{point_index}].dateUtc is invalid"
                    )
                    continue
                if parsed_day.isoformat() != raw_day:
                    errors.append(
                        f"$.instruments[{instrument_index}].points[{point_index}].dateUtc is not canonical"
                    )
                dates.append(raw_day)
            if len(dates) == len(points):
                if len(set(dates)) != len(dates):
                    errors.append(
                        f"$.instruments[{instrument_index}].points contains duplicate dates"
                    )
                if dates != sorted(dates):
                    errors.append(
                        f"$.instruments[{instrument_index}].points must be ordered by date"
                    )
                if canonical_dates is None:
                    canonical_dates = dates
                elif dates != canonical_dates:
                    errors.append("$.instruments must use the same daily UTC dates")

            latest_indices = [
                index
                for index, point in enumerate(points)
                if isinstance(point, dict) and point.get("isLatest") is True
            ]
            expected_latest = [] if point_count == 0 else [point_count - 1]
            if latest_indices != expected_latest:
                errors.append(
                    f"$.instruments[{instrument_index}].points must mark only the final point as latest"
                )

    if isinstance(window, dict) and isinstance(point_count, int):
        start = window.get("startDateUtc")
        end = window.get("endDateUtc")
        if point_count == 0:
            if start is not None or end is not None:
                errors.append("$.historyWindow empty history must use null startDateUtc and endDateUtc")
        elif canonical_dates:
            if start != canonical_dates[0] or end != canonical_dates[-1]:
                errors.append("$.historyWindow dates must match the first and last history points")

    if errors:
        raise MonitoringValidationError("Invalid monitoring history:\n- " + "\n- ".join(errors))


def validate_monitoring_pair(
    result: dict[str, Any], history: dict[str, Any]
) -> None:
    errors: list[str] = []
    if history.get("runId") != result.get("runId"):
        errors.append("history runId must match the latest result")
    if history.get("generatedAtUtc") != result.get("generatedAtUtc"):
        errors.append("history generatedAtUtc must match the latest result")
    result_provenance = result.get("provenance", {})
    history_provenance = history.get("provenance", {})
    for field in (
        "snapshotId",
        "snapshotFileSha256",
        "snapshotContentSha256",
        "protocolSha256",
    ):
        if history_provenance.get(field) != result_provenance.get(field):
            errors.append(f"history {field} must match the latest result")

    history_window = history.get("historyWindow", {})
    if history_window.get("pointCount") != 0 and history_window.get(
        "endDateUtc"
    ) != result.get("dataWindow", {}).get("observedEndUtc"):
        errors.append("history endDateUtc must match the latest observation")

    history_by_instrument = {
        item.get("instrument"): item
        for item in history.get("instruments", [])
        if isinstance(item, dict)
    }
    for instrument in result.get("instruments", []):
        if not isinstance(instrument, dict):
            continue
        history_points = history_by_instrument.get(instrument.get("instrument"), {}).get(
            "points", []
        )
        if history_points:
            latest_point = history_points[-1]
            if latest_point.get("dateUtc") != instrument.get("observationDateUtc"):
                errors.append(f"{instrument.get('instrument')} history date must match latest")
            if latest_point.get("closeUsd") != instrument.get("currentPrice"):
                errors.append(f"{instrument.get('instrument')} history close must match latest")
    if errors:
        raise MonitoringValidationError("Monitoring result/history mismatch:\n- " + "\n- ".join(errors))


def _canonical_content_hash(dates: Sequence[str], prices: dict[str, Sequence[float]]) -> str:
    canonical = "".join(
        f"{day},{prices['BTC-USD'][index]:.6f},{prices['ETH-USD'][index]:.6f}\n"
        for index, day in enumerate(dates)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalise_price_rows(rows_by_instrument: PriceRows) -> tuple[list[str], dict[str, list[float]]]:
    if set(rows_by_instrument) != set(INSTRUMENTS):
        raise MonitoringValidationError("Price input must contain exactly BTC-USD and ETH-USD")

    dates_by_instrument: dict[str, list[str]] = {}
    prices: dict[str, list[float]] = {}
    for instrument in INSTRUMENTS:
        dates: list[str] = []
        values: list[float] = []
        seen: set[str] = set()
        previous: date | None = None
        for raw_day, raw_price in rows_by_instrument[instrument]:
            try:
                parsed_day = date.fromisoformat(raw_day)
            except ValueError as exc:
                raise MonitoringValidationError(f"{instrument}: invalid UTC date {raw_day!r}") from exc
            if parsed_day.isoformat() != raw_day:
                raise MonitoringValidationError(f"{instrument}: date is not canonical ISO format: {raw_day!r}")
            if raw_day in seen:
                raise MonitoringValidationError(f"{instrument}: duplicate date {raw_day}")
            if previous is not None and parsed_day != previous + timedelta(days=1):
                raise MonitoringValidationError(
                    f"{instrument}: missing or unordered date after {previous.isoformat()}"
                )
            price = float(raw_price)
            if not math.isfinite(price) or price <= 0:
                raise MonitoringValidationError(f"{instrument}: price must be finite and positive on {raw_day}")
            seen.add(raw_day)
            dates.append(raw_day)
            values.append(price)
            previous = parsed_day
        if not dates:
            raise MonitoringValidationError(f"{instrument}: no price rows")
        dates_by_instrument[instrument] = dates
        prices[instrument] = values

    shared_date_set = set(dates_by_instrument[INSTRUMENTS[1]])
    shared_dates = [
        day for day in dates_by_instrument[INSTRUMENTS[0]] if day in shared_date_set
    ]
    if not shared_dates:
        raise MonitoringValidationError("BTC-USD and ETH-USD have no shared daily UTC dates")

    previous_shared: date | None = None
    for raw_day in shared_dates:
        parsed_day = date.fromisoformat(raw_day)
        if previous_shared is not None and parsed_day != previous_shared + timedelta(days=1):
            raise MonitoringValidationError(
                f"BTC-USD and ETH-USD shared dates have a gap after {previous_shared.isoformat()}"
            )
        previous_shared = parsed_day

    shared_prices: dict[str, list[float]] = {}
    for instrument in INSTRUMENTS:
        price_by_date = dict(
            zip(dates_by_instrument[instrument], prices[instrument], strict=True)
        )
        shared_prices[instrument] = [price_by_date[day] for day in shared_dates]
    return shared_dates, shared_prices


def build_snapshot_payload(
    *,
    rows_by_instrument: PriceRows,
    retrieved_at_utc: datetime,
    requested_start: date,
    requested_end: date,
) -> dict[str, Any]:
    dates, prices = _normalise_price_rows(rows_by_instrument)
    if date.fromisoformat(dates[-1]) > requested_end:
        raise MonitoringValidationError(
            f"Snapshot includes an incomplete UTC day after {requested_end.isoformat()}: {dates[-1]}"
        )
    observed_start = dates[0]
    observed_end = dates[-1]
    snapshot_id = f"monitoring-v1-yahoo-daily-{observed_start}_{observed_end}"
    rows = [
        [day, prices["BTC-USD"][index], prices["ETH-USD"][index]]
        for index, day in enumerate(dates)
    ]
    return {
        "schemaVersion": 1,
        "snapshotId": snapshot_id,
        "provenance": {
            "dataKind": "real",
            "provider": "Yahoo Finance",
            "accessMethod": "yfinance 0.2.65",
            "sourceUrls": SOURCE_URLS,
            "retrievedAtUtc": _format_utc(retrieved_at_utc),
            "requestedStartUtc": requested_start.isoformat(),
            "requestedEndUtc": requested_end.isoformat(),
            "observedStart": observed_start,
            "observedEnd": observed_end,
            "frequency": "daily",
            "timezone": "UTC",
            "field": "unadjusted close",
            "transformation": "none"
        },
        "quality": {
            "rowCount": len(rows),
            "missingSharedCloseCount": 0,
            "contentSha256": _canonical_content_hash(dates, prices)
        },
        "columns": ["date_utc", "BTC-USD", "ETH-USD"],
        "rows": rows
    }


def download_yahoo_price_rows(
    instruments: Sequence[str], requested_start: date, requested_end: date
) -> PriceRows:
    try:
        installed_version = metadata.version("yfinance")
    except metadata.PackageNotFoundError as exc:
        raise MonitoringRunError("yfinance 0.2.65 is required for network retrieval") from exc
    if installed_version != "0.2.65":
        raise MonitoringRunError(f"Expected yfinance 0.2.65, found {installed_version}")

    import yfinance as yf

    end_exclusive = requested_end + timedelta(days=1)
    result: PriceRows = {}
    for instrument in instruments:
        frame = yf.Ticker(instrument).history(
            start=requested_start.isoformat(),
            end=end_exclusive.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        if "Close" not in frame:
            raise MonitoringRunError(f"Yahoo Finance returned no Close series for {instrument}")
        result[instrument] = [
            (timestamp.date().isoformat(), float(close))
            for timestamp, close in frame["Close"].items()
            if not math.isnan(float(close))
        ]
    return result


def _risk_payload(source_model_id: str, forecasts: Sequence[VaREsForecast]) -> dict[str, Any]:
    return {
        "sourceModelId": source_model_id,
        "unit": "percent_return",
        "levels": {
            f"{forecast.confidence:.2f}": {
                "confidence": forecast.confidence,
                "var": forecast.var,
                "es": forecast.es,
            }
            for forecast in forecasts
        },
    }


def _baseline_payload(
    *,
    model_id: str,
    fit: GarchFit,
    mean_percent: float,
    confidence_levels: Sequence[float],
) -> dict[str, Any]:
    if fit.convergence_flag != 0:
        raise MonitoringRunError(f"{model_id} did not converge (flag {fit.convergence_flag})")
    student_t_df = fit.parameters.get("nu") if fit.innovation == "student_t" else None
    forecasts = [
        parametric_var_es(
            forecast_mean_percent=mean_percent,
            sigma_percent=fit.sigma_percent,
            confidence=confidence,
            innovation=fit.innovation,
            student_t_df=student_t_df,
        )
        for confidence in confidence_levels
    ]
    return {
        "modelId": model_id,
        "modelLabel": "GARCH(1,1)",
        "status": "fit_success",
        "innovation": fit.innovation,
        "convergenceFlag": fit.convergence_flag,
        "logLikelihood": fit.log_likelihood,
        "aic": fit.aic,
        "bic": fit.bic,
        "parameters": fit.parameters,
        "forecastVolatilityPercent": fit.sigma_percent,
        "risk": _risk_payload(model_id, forecasts),
    }


def _regime_payload(
    *, fit: MarkovVarianceFit, mean_percent: float, confidence_levels: Sequence[float]
) -> dict[str, Any]:
    filtered = [float(value) for value in fit.filtered_last]
    one_step = [
        sum(filtered[source] * fit.transition_matrix[source][target] for source in range(2))
        for target in range(2)
    ]
    total = sum(one_step)
    if not math.isfinite(total) or total <= 0:
        raise MonitoringRunError("Regime forecast probabilities are invalid")
    one_step = [value / total for value in one_step]
    selected_index = max(range(2), key=lambda index: filtered[index])
    lower_variance_index = min(range(2), key=lambda index: fit.variances[index])
    forecast_variance = sum(
        one_step[index] * fit.variances[index] for index in range(2)
    )
    if not math.isfinite(forecast_variance) or forecast_variance <= 0:
        raise MonitoringRunError("Regime variance forecast is invalid")
    forecasts = [
        normal_mixture_risk(fit=fit, mean_percent=mean_percent, confidence=confidence)
        for confidence in confidence_levels
    ]
    return {
        "modelId": "markov-variance-2state",
        "modelLabel": "fallback_not_ms_garch",
        "status": "fit_success",
        "fallbackReason": "operational_protocol_does_not_attempt_true_ms_garch",
        "probabilityType": "filtered",
        "selectedRegime": selected_index + 1,
        "state": "lower_variance" if selected_index == lower_variance_index else "higher_variance",
        "probability": filtered[selected_index],
        "filteredProbabilities": filtered,
        "oneStepProbabilities": one_step,
        "forecastVolatilityPercent": math.sqrt(forecast_variance),
        "risk": _risk_payload("markov-variance-2state", forecasts),
        "fit": {
            "logLikelihood": fit.log_likelihood,
            "variances": list(fit.variances),
            "transitionMatrix": [list(row) for row in fit.transition_matrix],
            "occupancy": list(fit.occupancy),
            "parameters": fit.parameters,
        },
    }

def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "scipy", "pandas", "arch", "statsmodels", "yfinance"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _snapshot_retrieved_at(snapshot: PriceSnapshot) -> str:
    value = snapshot.provenance.get("retrievedAtUtc")
    if not isinstance(value, str):
        raise MonitoringValidationError("Snapshot provenance is missing retrievedAtUtc")
    _parse_utc(value)
    return value


def _freshness_payload(
    *,
    observed_end_utc: str,
    last_complete_day_utc: date,
    evaluated_at_utc: datetime,
    maximum_age_hours_after_observation_close: int,
) -> dict[str, Any]:
    observed_day = date.fromisoformat(observed_end_utc)
    if observed_day > last_complete_day_utc:
        raise MonitoringValidationError(
            "Snapshot includes a day that is not complete yet: "
            f"lastCompleteDayUtc={last_complete_day_utc.isoformat()}, "
            f"observedEndUtc={observed_end_utc}"
        )

    observation_completed_at = datetime.combine(
        observed_day + timedelta(days=1), time.min, tzinfo=UTC
    )
    stale_after = observation_completed_at + timedelta(
        hours=maximum_age_hours_after_observation_close
    )
    age_days = (last_complete_day_utc - observed_day).days
    age_hours = round(
        max(0.0, (evaluated_at_utc - observation_completed_at).total_seconds() / 3600),
        6,
    )
    if evaluated_at_utc >= stale_after:
        raise MonitoringValidationError(
            "Snapshot exceeds freshness limit: "
            f"observedEndUtc={observed_end_utc}, "
            f"lastCompleteDayUtc={last_complete_day_utc.isoformat()}, "
            f"ageDays={age_days}, ageHours={age_hours}, "
            f"staleAfterUtc={_format_utc(stale_after)}, "
            f"evaluatedAtUtc={_format_utc(evaluated_at_utc)}, "
            "maximumAgeHoursAfterObservationClose="
            f"{maximum_age_hours_after_observation_close}"
        )

    return {
        "status": "fresh",
        "evaluatedAtUtc": _format_utc(evaluated_at_utc),
        "lastCompleteDayUtc": last_complete_day_utc.isoformat(),
        "observationDateUtc": observed_end_utc,
        "observationCompletedAtUtc": _format_utc(observation_completed_at),
        "ageDays": age_days,
        "ageHours": age_hours,
        "maximumAgeHoursAfterObservationClose": maximum_age_hours_after_observation_close,
        "staleAfterUtc": _format_utc(stale_after),
    }


def _population_standard_deviation(values: Sequence[float]) -> float:
    if not values:
        raise MonitoringValidationError("Realized volatility requires at least one return")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    result = math.sqrt(variance)
    if not math.isfinite(result):
        raise MonitoringValidationError("Realized volatility is not finite")
    return result


def build_monitoring_history(
    *,
    snapshot: PriceSnapshot,
    protocol: dict[str, Any],
    run_id: str,
    generated_at_utc: datetime,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    history_settings = protocol["history"]
    maximum_points = history_settings["maximumPoints"]
    windows = history_settings["realizedVolatilityWindowsDays"]
    start_index = max(0, len(snapshot.dates) - maximum_points)
    point_count = len(snapshot.dates) - start_index

    histories: list[dict[str, Any]] = []
    for instrument in INSTRUMENTS:
        prices = [float(value) for value in snapshot.prices[instrument]]
        returns: list[float | None] = [None]
        returns.extend(
            100.0 * math.log(prices[index] / prices[index - 1])
            for index in range(1, len(prices))
        )

        realized_by_window: dict[int, list[float | None]] = {}
        for window in windows:
            rolling: list[float | None] = []
            for index in range(len(returns)):
                first_return_index = index - window + 1
                if first_return_index < 1:
                    rolling.append(None)
                    continue
                sample = returns[first_return_index : index + 1]
                if len(sample) != window or any(value is None for value in sample):
                    rolling.append(None)
                    continue
                rolling.append(
                    _population_standard_deviation(
                        [float(value) for value in sample if value is not None]
                    )
                )
            realized_by_window[window] = rolling

        points = [
            {
                "dateUtc": snapshot.dates[index],
                "closeUsd": prices[index],
                "dailyReturnPercent": returns[index],
                "realizedVolatility7dPercent": realized_by_window[7][index],
                "realizedVolatility30dPercent": realized_by_window[30][index],
                "isLatest": index == len(snapshot.dates) - 1,
            }
            for index in range(start_index, len(snapshot.dates))
        ]
        histories.append({"instrument": instrument, "points": points})

    history = {
        "schemaVersion": "monitoring-history.v1",
        "protocolId": "monitoring-v1",
        "runId": run_id,
        "generatedAtUtc": _format_utc(generated_at_utc),
        "state": "complete",
        "historyWindow": {
            "startDateUtc": snapshot.dates[start_index] if point_count else None,
            "endDateUtc": snapshot.dates[-1] if point_count else None,
            "pointCount": point_count,
            "timezone": "UTC",
            "closedDaysOnly": True,
            "availableRangesDays": history_settings["rangesDays"],
        },
        "definitions": {
            "dailyReturn": {
                "method": history_settings["returnDefinition"],
                "formula": "100 * ln(close_t / close_t-1)",
                "unit": history_settings["returnUnit"],
            },
            "realizedVolatility": {
                "method": history_settings["realizedVolatilityEstimator"],
                "windowsDays": windows,
                "unit": history_settings["returnUnit"],
                "annualized": history_settings["annualized"],
            },
        },
        "instruments": histories,
        "provenance": {
            "dataKind": provenance["dataKind"],
            "provider": provenance["provider"],
            "accessMethod": provenance["accessMethod"],
            "sourceUrls": provenance["sourceUrls"],
            "priceField": provenance["priceField"],
            "timezone": "UTC",
            "retrievedAtUtc": provenance["retrievedAtUtc"],
            "snapshotId": provenance["snapshotId"],
            "snapshotFileSha256": provenance["snapshotFileSha256"],
            "snapshotContentSha256": provenance["snapshotContentSha256"],
            "protocolSha256": provenance["protocolSha256"],
            "codeCommit": provenance["codeCommit"],
        },
    }
    validate_monitoring_history(history)
    return history


def _fit_instrument(
    *,
    instrument: str,
    snapshot: PriceSnapshot,
    model_returns: Sequence[float],
    protocol: dict[str, Any],
    garch_fitter: GarchFitter,
    regime_fitter: RegimeFitter,
) -> dict[str, Any]:
    window = protocol["estimation"]["windowReturns"]
    training = tuple(float(value) for value in model_returns[-window:])
    if len(training) != window:
        raise MonitoringValidationError(f"{instrument}: expected {window} returns, found {len(training)}")
    mean_percent = sum(training) / len(training)
    centered = tuple(value - mean_percent for value in training)
    confidence_levels = protocol["risk"]["confidenceLevels"]

    baseline_fits: list[dict[str, Any]] = []
    for variant in protocol["estimation"]["baseline"]["variants"]:
        try:
            fit = garch_fitter(centered, innovation=variant["innovation"])
            baseline_fits.append(
                _baseline_payload(
                    model_id=variant["modelId"],
                    fit=fit,
                    mean_percent=mean_percent,
                    confidence_levels=confidence_levels,
                )
            )
        except Exception as exc:
            if isinstance(exc, MonitoringRunError):
                raise
            raise MonitoringRunError(f"{instrument} {variant['modelId']} fit failed: {exc}") from exc

    try:
        regime_fit = regime_fitter(centered)
        regime = _regime_payload(
            fit=regime_fit,
            mean_percent=mean_percent,
            confidence_levels=confidence_levels,
        )
    except Exception as exc:
        if isinstance(exc, MonitoringRunError):
            raise
        raise MonitoringRunError(f"{instrument} markov-variance-2state fit failed: {exc}") from exc

    primary_id = protocol["estimation"]["baseline"]["primaryModelId"]
    primary = next(item for item in baseline_fits if item["modelId"] == primary_id)
    return {
        "instrument": instrument,
        "currentPrice": snapshot.prices[instrument][-1],
        "observationDateUtc": snapshot.dates[-1],
        "fitStatus": "complete",
        "modelLabel": "fallback_not_ms_garch",
        "volatility": {
            "sourceModelId": primary_id,
            "unit": "percent_return",
            "forecastPercent": primary["forecastVolatilityPercent"],
        },
        "risk": primary["risk"],
        "baselineFits": baseline_fits,
        "regime": regime,
    }


def run_monitoring(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    now_utc: datetime | None = None,
    snapshot_path: Path | None = None,
    price_fetcher: PriceFetcher = download_yahoo_price_rows,
    garch_fitter: GarchFitter = fit_garch11,
    regime_fitter: RegimeFitter = fit_markov_variance,
) -> dict[str, Any]:
    now = _validate_now(now_utc or datetime.now(UTC)).replace(microsecond=0)
    protocol = validate_monitoring_protocol(protocol_path)
    last_complete_day = now.date() - timedelta(
        days=protocol["schedule"]["completeDayLag"]
    )
    requested_start = last_complete_day - timedelta(
        days=protocol["data"]["historyCalendarDays"]
    )
    run_id = f"monitoring-v1-{now.strftime('%Y%m%dT%H%M%SZ')}"
    temporary_dir = artifact_root / f".{run_id}.tmp"
    final_dir = artifact_root / "runs" / run_id
    latest_path = artifact_root / "latest.json"
    latest_temp = artifact_root / f".latest-{run_id}.tmp"

    if temporary_dir.exists() or final_dir.exists():
        raise MonitoringRunError(f"Run directory already exists for {run_id}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True)

    try:
        local_snapshot_path = temporary_dir / "snapshot.json"
        if snapshot_path is None:
            rows = price_fetcher(INSTRUMENTS, requested_start, last_complete_day)
            snapshot_payload = build_snapshot_payload(
                rows_by_instrument=rows,
                retrieved_at_utc=now,
                requested_start=requested_start,
                requested_end=last_complete_day,
            )
            write_json(local_snapshot_path, snapshot_payload)
        else:
            if not snapshot_path.is_file():
                raise MonitoringValidationError(f"Snapshot does not exist: {snapshot_path}")
            shutil.copyfile(snapshot_path, local_snapshot_path)

        snapshot = load_price_snapshot(local_snapshot_path)
        minimum_rows = protocol["data"]["minimumPriceRows"]
        if len(snapshot.dates) < minimum_rows:
            raise MonitoringValidationError(
                f"Snapshot has {len(snapshot.dates)} rows; at least {minimum_rows} are required"
            )
        freshness = _freshness_payload(
            observed_end_utc=snapshot.dates[-1],
            last_complete_day_utc=last_complete_day,
            evaluated_at_utc=now,
            maximum_age_hours_after_observation_close=protocol["data"]["freshness"][
                "maximumAgeHoursAfterObservationClose"
            ],
        )

        return_series = build_log_returns(
            snapshot, scale_factor=protocol["estimation"]["returnScale"]
        )
        instruments = [
            _fit_instrument(
                instrument=instrument,
                snapshot=snapshot,
                model_returns=return_series[instrument].model_returns_percent,
                protocol=protocol,
                garch_fitter=garch_fitter,
                regime_fitter=regime_fitter,
            )
            for instrument in INSTRUMENTS
        ]

        result = {
            "schemaVersion": "monitoring-result.v1",
            "protocolId": "monitoring-v1",
            "runId": run_id,
            "generatedAtUtc": _format_utc(now),
            "state": "complete",
            "dataWindow": {
                "observedStartUtc": snapshot.dates[0],
                "observedEndUtc": snapshot.dates[-1],
                "rowCount": len(snapshot.dates),
                "snapshotSha256": snapshot.file_sha256,
            },
            "freshness": freshness,
            "modelPolicy": {
                "trueMsgarchAttempted": False,
                "regimeModelLabel": "fallback_not_ms_garch",
                "winningModelSelected": False,
            },
            "instruments": instruments,
            "provenance": {
                "dataKind": "real",
                "provider": "Yahoo Finance",
                "accessMethod": "yfinance 0.2.65",
                "sourceUrls": SOURCE_URLS,
                "priceField": "unadjusted_close",
                "retrievedAtUtc": _snapshot_retrieved_at(snapshot),
                "snapshotId": snapshot.snapshot_id,
                "snapshotFileSha256": snapshot.file_sha256,
                "snapshotContentSha256": snapshot.content_sha256,
                "protocolSha256": sha256_file(protocol_path),
                "codeCommit": git_commit(REPO_ROOT),
                "runtimeVersions": _runtime_versions(),
            },
        }
        history = build_monitoring_history(
            snapshot=snapshot,
            protocol=protocol,
            run_id=run_id,
            generated_at_utc=now,
            provenance=result["provenance"],
        )
        validate_monitoring_result(result)
        validate_monitoring_pair(result, history)
        write_json(temporary_dir / "result.json", result)
        write_json(temporary_dir / "history.json", history)
        write_json(
            temporary_dir / "manifest.json",
            {
                "schemaVersion": "monitoring-manifest.v1",
                "protocolId": "monitoring-v1",
                "runId": run_id,
                "state": "complete",
                "generatedAtUtc": _format_utc(now),
                "artifacts": artifact_hashes(temporary_dir),
            },
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_dir, final_dir)
        write_json(latest_temp, result)
        os.replace(latest_temp, latest_path)
        return {
            "runId": run_id,
            "runDirectory": str(final_dir),
            "latestPath": str(latest_path),
            "historyPath": str(final_dir / "history.json"),
            "observedStartUtc": snapshot.dates[0],
            "observedEndUtc": snapshot.dates[-1],
            "rowCount": len(snapshot.dates),
        }
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        latest_temp.unlink(missing_ok=True)
        raise


def _command_validate_protocol(args: argparse.Namespace) -> None:
    protocol = validate_monitoring_protocol(args.protocol)
    print(json.dumps({"status": "valid", "protocolId": protocol["protocolId"]}, indent=2))


def _command_run(args: argparse.Namespace) -> None:
    now = _parse_utc(args.now_utc) if args.now_utc else datetime.now(UTC)
    summary = run_monitoring(
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        now_utc=now,
    )
    print(json.dumps(summary, indent=2))


def _command_reproduce(args: argparse.Namespace) -> None:
    if args.now_utc:
        now = _parse_utc(args.now_utc)
    else:
        snapshot_payload = _read_json(args.snapshot)
        retrieved_at = snapshot_payload.get("provenance", {}).get("retrievedAtUtc")
        if not isinstance(retrieved_at, str):
            raise MonitoringValidationError("Snapshot provenance is missing retrievedAtUtc")
        now = _parse_utc(retrieved_at)
    summary = run_monitoring(
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        now_utc=now,
        snapshot_path=args.snapshot,
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RegimeSentinel operational daily monitoring")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-protocol")
    validate_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    validate_parser.set_defaults(handler=_command_validate_protocol)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    run_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    run_parser.add_argument("--now-utc")
    run_parser.set_defaults(handler=_command_run)

    reproduce_parser = subparsers.add_parser("reproduce")
    reproduce_parser.add_argument("--snapshot", type=Path, required=True)
    reproduce_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    reproduce_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    reproduce_parser.add_argument("--now-utc")
    reproduce_parser.set_defaults(handler=_command_reproduce)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
