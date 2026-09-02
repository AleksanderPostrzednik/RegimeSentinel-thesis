"""Validate the frozen RegimeSentinel thesis experiment protocol.

The JSON Schema describes the interchange contract. This module adds domain invariants
that JSON Schema cannot express, including snapshot checksums, rolling-window dates and
the information boundary between filtered and smoothed regime probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = REPO_ROOT / "experiments" / "thesis-v1" / "protocol.json"


class ProtocolValidationError(ValueError):
    """Raised when a protocol violates its schema-level or domain contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(f"- {error}" for error in errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _content_sha256(rows: list[list[Any]]) -> str:
    canonical_rows = "".join(
        f"{row[0]},{float(row[1]):.6f},{float(row[2]):.6f}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(canonical_rows).hexdigest()


def _parse_iso_date(value: Any, label: str, errors: list[str]) -> date | None:
    if not isinstance(value, str):
        errors.append(f"{label} must be an ISO date string")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{label} is not a real ISO date: {value!r}")
        return None


def _parse_utc_timestamp(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        errors.append(f"{label} must be an ISO timestamp ending in Z")
        return
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        errors.append(f"{label} is not a real ISO timestamp: {value!r}")


def _expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _resolve_local_schema_ref(reference: str, schema_root: dict[str, Any]) -> Any:
    """Resolve a local JSON Pointer such as ``#/$defs/data``."""

    if not reference.startswith("#/"):
        raise ValueError(f"only local JSON Schema references are supported: {reference}")

    current: Any = schema_root
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise ValueError(f"cannot resolve JSON Schema reference: {reference}")
        current = current[token]
    return current


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    raise ValueError(f"unsupported JSON Schema type: {expected_type}")


def _has_valid_format(value: str, format_name: str) -> bool:
    try:
        if format_name == "date":
            date.fromisoformat(value)
            return True
        if format_name == "date-time":
            datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
            return True
        if format_name == "uri":
            parsed = urlparse(value)
            return bool(parsed.scheme and (parsed.netloc or parsed.scheme == "urn"))
    except ValueError:
        return False
    raise ValueError(f"unsupported JSON Schema format: {format_name}")


def _validate_json_schema_node(
    value: Any,
    schema: Any,
    schema_root: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Validate the keywords used by the checked-in draft 2020-12 contract.

    The project keeps this validator dependency-free so the frozen protocol can be
    checked in a clean checkout. Unsupported schema types or formats fail closed.
    """

    if schema is True:
        return
    if schema is False:
        errors.append(f"schema validation at {path}: value is forbidden")
        return
    if not isinstance(schema, dict):
        raise ValueError(f"invalid JSON Schema node at {path}")

    if "$ref" in schema:
        referenced = _resolve_local_schema_ref(schema["$ref"], schema_root)
        _validate_json_schema_node(value, referenced, schema_root, path, errors)
        if len(schema) == 1:
            return

    expected_type = schema.get("type")
    if expected_type is not None:
        if not isinstance(expected_type, str):
            raise ValueError(f"unsupported JSON Schema type declaration at {path}")
        if not _matches_json_type(value, expected_type):
            errors.append(f"schema validation at {path}: expected {expected_type}")
            return

    if "const" in schema and value != schema["const"]:
        errors.append(f"schema validation at {path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"schema validation at {path}: value is not in the allowed enum")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"schema validation at {path}: string is shorter than minLength")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            errors.append(f"schema validation at {path}: string does not match pattern {pattern!r}")
        format_name = schema.get("format")
        if format_name is not None and not _has_valid_format(value, format_name):
            errors.append(f"schema validation at {path}: invalid {format_name} format")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"schema validation at {path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"schema validation at {path}: value is above maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"schema validation at {path}: value is not above exclusiveMinimum")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"schema validation at {path}: value is not below exclusiveMaximum")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"schema validation at {path}: array is shorter than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"schema validation at {path}: array is longer than maxItems")
        if schema.get("uniqueItems"):
            canonical_items = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
            if len(canonical_items) != len(set(canonical_items)):
                errors.append(f"schema validation at {path}: array items are not unique")

        prefix_items = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefix_items):
            if index < len(value):
                _validate_json_schema_node(value[index], item_schema, schema_root, f"{path}[{index}]", errors)

        remaining_items_schema = schema.get("items", True)
        for index in range(len(prefix_items), len(value)):
            _validate_json_schema_node(
                value[index],
                remaining_items_schema,
                schema_root,
                f"{path}[{index}]",
                errors,
            )

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = sorted(field for field in required if field not in value)
        if missing:
            errors.append(f"schema validation at {path}: missing required properties: {missing}")

        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in value:
                _validate_json_schema_node(
                    value[field],
                    field_schema,
                    schema_root,
                    f"{path}.{field}",
                    errors,
                )

        additional = schema.get("additionalProperties", True)
        extras = sorted(set(value) - set(properties))
        if additional is False and extras:
            errors.append(f"schema validation at {path}: unexpected properties: {extras}")
        elif isinstance(additional, dict):
            for field in extras:
                _validate_json_schema_node(
                    value[field],
                    additional,
                    schema_root,
                    f"{path}.{field}",
                    errors,
                )


def _validate_json_schema_instance(
    value: Any,
    schema: dict[str, Any],
    errors: list[str],
) -> None:
    _validate_json_schema_node(value, schema, schema, "$", errors)


def validate_protocol(protocol: dict[str, Any], repo_root: Path = REPO_ROOT) -> None:
    """Validate one parsed protocol and its referenced snapshot.

    Raises:
        ProtocolValidationError: when one or more deterministic checks fail.
    """

    errors: list[str] = []
    required_top_level = {
        "$schema",
        "schemaVersion",
        "protocolId",
        "status",
        "frozenAtUtc",
        "objective",
        "data",
        "transformation",
        "informationSet",
        "models",
        "risk",
        "evaluation",
        "reproducibility",
        "changeControl",
        "literatureAnchors",
    }
    _expect(isinstance(protocol, dict), "protocol must be a JSON object", errors)
    if not isinstance(protocol, dict):
        raise ProtocolValidationError(errors)

    missing = sorted(required_top_level - set(protocol))
    unexpected = sorted(set(protocol) - required_top_level)
    _expect(not missing, f"missing top-level fields: {missing}", errors)
    _expect(not unexpected, f"unexpected top-level fields: {unexpected}", errors)
    if missing:
        raise ProtocolValidationError(errors)

    _expect(protocol["schemaVersion"] == 1, "schemaVersion must equal 1", errors)
    _expect(protocol["status"] == "frozen", "status must equal frozen", errors)
    _expect(protocol["protocolId"] == "thesis-v1", "protocolId must equal thesis-v1", errors)
    _parse_utc_timestamp(protocol["frozenAtUtc"], "frozenAtUtc", errors)

    schema_path = (repo_root / protocol["$schema"].replace("../../", "")).resolve()
    expected_schema_path = (repo_root / "contracts" / "experiment-protocol.v1.schema.json").resolve()
    _expect(schema_path == expected_schema_path, "protocol must reference the v1 schema", errors)
    if schema_path.exists():
        try:
            schema = _load_json(schema_path)
            _expect(
                schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
                "contract must use JSON Schema draft 2020-12",
                errors,
            )
            _validate_json_schema_instance(protocol, schema, errors)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot parse protocol schema: {exc}")
    else:
        errors.append(f"protocol schema does not exist: {schema_path}")

    if errors:
        raise ProtocolValidationError(errors)

    data = protocol["data"]
    snapshot_contract = data["snapshot"]
    _expect(data["instruments"] == ["BTC-USD", "ETH-USD"], "instrument order must be BTC-USD, ETH-USD", errors)
    _expect(data["frequency"] == "daily", "frequency must be daily", errors)
    _expect(data["timezone"] == "UTC", "data timezone must be UTC", errors)
    _expect(data["priceField"] == "unadjusted_close", "price field must be unadjusted_close", errors)
    _expect(data["joinPolicy"] == "inner_shared_utc_dates", "BTC/ETH must use shared UTC dates", errors)
    _expect(data["missingDataPolicy"] == "fail_no_imputation", "missing prices must fail without imputation", errors)
    _expect(data["duplicateDatePolicy"] == "fail", "duplicate dates must fail", errors)
    _expect(data["nonPositivePricePolicy"] == "fail", "non-positive prices must fail", errors)

    snapshot_path = (repo_root / snapshot_contract["path"]).resolve()
    try:
        snapshot_path.relative_to(repo_root.resolve())
    except ValueError:
        errors.append("snapshot path escapes the repository")
    if not snapshot_path.exists():
        errors.append(f"snapshot does not exist: {snapshot_path}")
        raise ProtocolValidationError(errors)

    actual_file_sha = _sha256(snapshot_path)
    _expect(
        actual_file_sha == snapshot_contract["fileSha256"],
        f"snapshot fileSha256 mismatch: expected {snapshot_contract['fileSha256']}, got {actual_file_sha}",
        errors,
    )

    try:
        snapshot = _load_json(snapshot_path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot parse snapshot: {exc}")
        raise ProtocolValidationError(errors)

    rows = snapshot.get("rows")
    _expect(isinstance(rows, list) and bool(rows), "snapshot rows must be a non-empty array", errors)
    if not isinstance(rows, list) or not rows:
        raise ProtocolValidationError(errors)

    _expect(snapshot.get("columns") == ["date_utc", "BTC-USD", "ETH-USD"], "snapshot columns are not canonical", errors)
    _expect(snapshot.get("snapshotId") == snapshot_contract["snapshotId"], "snapshotId mismatch", errors)
    _expect(len(rows) == snapshot_contract["rowCount"], "snapshot rowCount does not match rows", errors)
    _expect(snapshot.get("quality", {}).get("rowCount") == len(rows), "snapshot quality.rowCount mismatch", errors)
    _expect(snapshot.get("quality", {}).get("missingSharedCloseCount") == 0, "snapshot contains missing shared closes", errors)

    provenance = snapshot.get("provenance", {})
    for field, expected in (
        ("provider", snapshot_contract["provider"]),
        ("accessMethod", snapshot_contract["accessMethod"]),
        ("retrievedAtUtc", snapshot_contract["retrievedAtUtc"]),
        ("observedStart", snapshot_contract["observedStart"]),
        ("observedEnd", snapshot_contract["observedEnd"]),
        ("frequency", "daily"),
        ("timezone", "UTC"),
        ("field", "unadjusted close"),
    ):
        _expect(provenance.get(field) == expected, f"snapshot provenance.{field} mismatch", errors)

    dates: list[date] = []
    seen_dates: set[date] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 3:
            errors.append(f"snapshot row {index} must contain date, BTC close and ETH close")
            continue
        parsed_date = _parse_iso_date(row[0], f"snapshot row {index} date", errors)
        if parsed_date is not None:
            if parsed_date in seen_dates:
                errors.append(f"snapshot contains duplicate date {parsed_date.isoformat()}")
            seen_dates.add(parsed_date)
            dates.append(parsed_date)
        for instrument_index, instrument in ((1, "BTC-USD"), (2, "ETH-USD")):
            value = row[instrument_index]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"snapshot row {index} has invalid {instrument} close: {value!r}")

    if len(dates) == len(rows):
        for previous, current in zip(dates, dates[1:]):
            if (current - previous).days != 1:
                errors.append(f"snapshot UTC calendar is not daily between {previous} and {current}")
                break

    actual_content_sha = _content_sha256(rows)
    _expect(
        actual_content_sha == snapshot_contract["contentSha256"],
        f"snapshot contentSha256 mismatch: expected {snapshot_contract['contentSha256']}, got {actual_content_sha}",
        errors,
    )
    _expect(snapshot.get("quality", {}).get("contentSha256") == actual_content_sha, "snapshot embedded contentSha256 mismatch", errors)

    information = protocol["informationSet"]
    window = information["estimationWindowReturns"]
    return_count = len(rows) - 1
    expected_forecasts = return_count - window
    _expect(window == 500, "primary estimation window must remain frozen at 500 returns", errors)
    _expect(information["windowType"] == "fixed", "rolling window must be fixed, not expanding", errors)
    _expect(information["forecastHorizonDays"] == 1, "forecast horizon must be one day", errors)
    _expect(information["refitEveryForecasts"] == 1, "models must refit for every forecast", errors)
    _expect(information["expectedForecastsPerInstrument"] == expected_forecasts, "derived OOS forecast count mismatch", errors)
    _expect(information["firstForecastDate"] == rows[window + 1][0], "derived first OOS forecast date mismatch", errors)
    _expect(information["lastForecastDate"] == rows[-1][0], "last OOS forecast date mismatch", errors)
    _expect(information["stateProbabilityForForecast"] == "filtered", "OOS forecasts must use filtered probabilities", errors)
    _expect(information["smoothedProbabilityUse"] == "post_hoc_only", "smoothed probabilities must be post-hoc only", errors)
    _expect(information["hyperparametersSelectedBeforeOos"] is True, "hyperparameters must be selected before OOS", errors)

    transformation = protocol["transformation"]
    _expect(transformation["returnDefinition"] == "log", "returns must be logarithmic", errors)
    _expect(transformation["scaleFactor"] == 100, "model input must be scaled to percent", errors)
    _expect(transformation["lossDefinition"] == "loss_t = -return_t", "loss sign convention changed", errors)
    centering = transformation["centering"]
    _expect(centering["method"] == "rolling_training_sample_mean", "mean must be estimated from each training window", errors)
    _expect(centering["reestimatedAtEachRefit"] is True, "rolling mean must be re-estimated", errors)
    _expect(centering["forecastMeanAddedBack"] is True, "forecast mean must be added back", errors)

    models = protocol["models"]
    baseline = models["baseline"]
    candidate = models["regimeCandidate"]
    fallback = models["fallback"]
    _expect(baseline["order"] == [1, 1], "baseline must remain GARCH(1,1)", errors)
    _expect(candidate["states"] == 2, "regime candidate must have two states", errors)
    _expect(baseline["estimationMethod"] == "maximum_likelihood", "baseline estimation must use maximum likelihood", errors)
    _expect(candidate["estimationMethod"] == "maximum_likelihood", "MSGARCH FitML is maximum likelihood, not EM", errors)
    _expect(baseline["primaryInnovation"] == candidate["primaryInnovation"] == "student_t", "primary innovation must be Student-t for both models", errors)
    _expect(baseline["robustnessInnovation"] == candidate["robustnessInnovation"] == "normal", "normal innovation must be the shared robustness variant", errors)
    state_inference = candidate["stateInference"]
    _expect(state_inference["forecast"] == "filtered", "candidate forecast cannot use smoothed state probabilities", errors)
    _expect(state_inference["smoothed"] == "post_hoc_only", "smoothed probabilities are post-hoc only", errors)
    _expect(state_inference["viterbi"] == "post_hoc_path_only", "Viterbi is decoding, not estimation or forecasting", errors)
    _expect(candidate["preflight"]["usesInitialTrainingWindowOnly"] is True, "MSGARCH preflight must not inspect OOS results", errors)
    _expect(candidate["preflight"]["deterministicStarts"] >= 3, "MSGARCH preflight requires at least three deterministic starts", errors)
    _expect(fallback["label"] == "fallback_not_ms_garch", "fallback must never be labeled MS-GARCH", errors)
    _expect(fallback["forecastStateProbability"] == "filtered", "fallback forecast must use filtered probabilities", errors)
    _expect(fallback["trigger"]["basedOnPredictivePerformance"] is False, "fallback cannot be selected after comparing OOS performance", errors)
    _expect(fallback["trigger"]["initialPreflightMustPassForBothInstruments"] is True, "preflight must pass for both assets", errors)
    _expect(fallback["trigger"]["minimumRollingForecastSuccessRate"] == 0.99, "rolling success gate must remain 99%", errors)

    risk = protocol["risk"]
    _expect(risk["horizonDays"] == information["forecastHorizonDays"], "risk and forecast horizons differ", errors)
    _expect(risk["confidenceLevels"] == [0.95, 0.99], "confidence levels must remain 95% and 99%", errors)
    _expect(risk["tailProbabilities"] == [0.05, 0.01], "tail probabilities must remain 5% and 1%", errors)
    for confidence, tail in zip(risk["confidenceLevels"], risk["tailProbabilities"]):
        _expect(math.isclose(1 - confidence, tail, abs_tol=1e-12), f"tail probability {tail} does not match confidence {confidence}", errors)
    _expect(risk["varConvention"] == "positive_loss_quantile", "VaR sign convention changed", errors)
    _expect(risk["exceedanceRule"] == "loss_t > VaR_t", "exceedance rule changed", errors)

    evaluation = protocol["evaluation"]
    _expect(evaluation["testSignificance"] == 0.05, "test significance must remain 5%", errors)
    for confidence in risk["confidenceLevels"]:
        key = f"{confidence:.2f}"
        expected = expected_forecasts * (1 - confidence)
        actual = evaluation["expectedExceedancesPerInstrument"].get(key)
        _expect(isinstance(actual, (int, float)) and math.isclose(actual, expected, abs_tol=1e-10), f"expected exceedances for {key} must equal {expected}", errors)
    _expect(set(evaluation["varBacktests"]) == {
        "kupiec_unconditional_coverage",
        "christoffersen_independence",
        "christoffersen_conditional_coverage",
    }, "VaR backtest set is incomplete", errors)
    _expect(set(evaluation["forecastScores"]) == {"var_quantile_loss", "joint_var_es_fz0"}, "forecast score set is incomplete", errors)
    _expect(evaluation["esStandalonePassFailTest"] == "none", "ES must not receive an invented standalone pass/fail test", errors)
    comparison = evaluation["modelComparisonRule"]
    _expect(comparison["aicBicOnlyIsSufficient"] is False, "AIC/BIC alone cannot select the thesis model", errors)
    _expect(comparison["nonRejectionMeansModelIsProvenGood"] is False, "non-rejection cannot prove model quality", errors)
    _expect(len(comparison["requiredDimensions"]) >= 6, "model comparison needs all frozen dimensions", errors)

    reproducibility = protocol["reproducibility"]
    _expect(reproducibility["masterSeed"] == 20260722, "master seed changed", errors)
    _expect(reproducibility["timezone"] == "UTC", "run timezone must be UTC", errors)
    _expect(reproducibility["recordAllFitFailures"] is True, "all fit failures must be recorded", errors)
    _expect(reproducibility["silentModelSubstitution"] is False, "silent model substitution is forbidden", errors)
    required_manifest_fields = set(reproducibility["requiredRunManifestFields"])
    mandatory_manifest_fields = {
        "protocol_id",
        "protocol_sha256",
        "snapshot_id",
        "snapshot_file_sha256",
        "git_commit",
        "runtime_versions",
        "seed",
        "artifact_sha256",
    }
    _expect(mandatory_manifest_fields <= required_manifest_fields, "run manifest is missing reproducibility fields", errors)

    change_control = protocol["changeControl"]
    _expect("new protocolId" in change_control["resultsBlindRule"], "result-blind rule must require a new protocolId", errors)
    _expect("snapshot_or_cutoff_change" in change_control["requiresNewProtocol"], "snapshot changes must require a new protocol", errors)
    _expect("evaluation_metric_or_threshold_change" in change_control["requiresNewProtocol"], "evaluation changes must require a new protocol", errors)

    anchors = protocol["literatureAnchors"]
    anchor_ids = [anchor.get("id") for anchor in anchors]
    _expect(len(anchors) >= 8, "at least eight primary literature anchors are required", errors)
    _expect(len(anchor_ids) == len(set(anchor_ids)), "literature anchor IDs must be unique", errors)
    for anchor in anchors:
        _expect(str(anchor.get("url", "")).startswith("https://"), f"literature anchor {anchor.get('id')} must use HTTPS", errors)

    if errors:
        raise ProtocolValidationError(errors)


def validate_protocol_file(path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load and validate a protocol file, returning the parsed object."""

    checksum_path = path.with_suffix(".sha256")
    if not checksum_path.exists():
        raise ProtocolValidationError([f"protocol checksum file does not exist: {checksum_path}"])
    checksum_parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(checksum_parts) != 2 or checksum_parts[1] != path.name:
        raise ProtocolValidationError([f"invalid protocol checksum record: {checksum_path}"])
    actual_protocol_sha = _sha256(path)
    if checksum_parts[0] != actual_protocol_sha:
        raise ProtocolValidationError([
            f"protocol sha256 mismatch: expected {checksum_parts[0]}, got {actual_protocol_sha}"
        ])

    try:
        protocol = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError([f"cannot parse protocol {path}: {exc}"]) from exc
    validate_protocol(protocol, repo_root)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", nargs="?", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    try:
        protocol = validate_protocol_file(args.protocol.resolve())
    except ProtocolValidationError as exc:
        print(f"Experiment protocol validation failed:\n{exc}")
        raise SystemExit(1) from exc

    information = protocol["informationSet"]
    print("Experiment protocol validation passed:")
    print(f"- protocol: {protocol['protocolId']} ({protocol['status']})")
    print(f"- instruments: {', '.join(protocol['data']['instruments'])}")
    print(f"- snapshot: {protocol['data']['snapshot']['snapshotId']}")
    print(f"- rolling window: {information['estimationWindowReturns']} returns")
    print(
        f"- OOS: {information['firstForecastDate']}..{information['lastForecastDate']} "
        f"({information['expectedForecastsPerInstrument']} forecasts per instrument)"
    )
    print("- information boundary: filtered for forecasts; smoothed/Viterbi post-hoc only")


if __name__ == "__main__":
    main()
