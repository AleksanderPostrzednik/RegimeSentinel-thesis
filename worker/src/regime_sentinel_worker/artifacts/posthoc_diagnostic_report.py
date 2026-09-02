"""Build deterministic POST_HOC_DESCRIPTIVE diagnostics from existing artifacts."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "artifacts/thesis-v1/posthoc-diagnostics"
BASELINE = ROOT / "artifacts/thesis-v1/baseline"
REGIME = ROOT / "artifacts/thesis-v1/regime"
FINAL_RUN = ROOT / "artifacts/thesis-v1/runs/true-msgarch-attempt-2-20260818T174828Z"
FINAL_PREFLIGHT = FINAL_RUN / "preflight.json"
PROTOCOL = ROOT / "experiments/thesis-v1/protocol.json"
INPUT_MANIFEST = ROOT / "experiments/thesis-v1/input-manifest.json"
GENERATOR = Path(__file__).resolve()
INSTRUMENTS = ("BTC-USD", "ETH-USD")
CONFIDENCES = (0.95, 0.99)
COLOUR = {
    "BTC-USD": "#f7931a",
    "ETH-USD": "#627eea",
    "actual": "#111827",
    "var": "#d97706",
    "es": "#2563eb",
    "missing": "#dc2626",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def first_value(value: Any, keys: set[str]) -> Any:
    for record in walk_dicts(value):
        for key in keys:
            if key in record:
                return record[key]
    return None


def rows_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("forecasts", "rows", "data", "results"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    return []


def instrument(value: Any) -> str | None:
    text = str(value).upper()
    for name in INSTRUMENTS:
        if name in text or name.replace("-", "_") in text:
            return name
    return None


def model(value: Any) -> str:
    text = str(value).lower()
    if "student" in text or "student-t" in text or "student_t" in text:
        return "garch_student_t"
    if "normal" in text:
        return "garch_normal"
    return "unknown"


def confidence(value: Any) -> float | None:
    result = number(value)
    if result is None:
        return None
    if result > 1:
        result /= 100
    return result if result in CONFIDENCES else None


def confidence_key(value: Any) -> str | None:
    result = confidence(value)
    return None if result is None else f"{result:.2f}"


def forecast_sets() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(BASELINE.rglob("*.json")):
        if path.name in {"manifest.json", "returns.json", "fit_failures.json"}:
            continue
        if "summary" in path.name.lower():
            continue
        try:
            rows = rows_from_json(load_json(path))
        except (OSError, json.JSONDecodeError):
            continue
        if not rows or not any(isinstance(row.get("risk"), dict) for row in rows):
            continue
        name = instrument(rows[0].get("instrument")) or instrument(path)
        variant = model(path)
        if variant == "unknown":
            variant = model(rows[0].get("model", rows[0].get("model_variant")))
        if name is None or variant == "unknown" or (variant, name) in seen:
            continue
        seen.add((variant, name))
        result.append({"path": path, "rows": rows, "instrument": name, "model": variant})
    return sorted(result, key=lambda item: (item["model"], item["instrument"]))


def risk(row: dict[str, Any], level: float) -> dict[str, Any] | None:
    values = row.get("risk")
    if not isinstance(values, dict):
        return None
    for key, value in values.items():
        if confidence(key) == level and isinstance(value, dict):
            return value
    return None


def average(values: Iterable[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def nested_number(value: Any, aliases: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = "".join(c for c in str(key).lower() if c.isalnum())
            if normalized in aliases:
                result = number(child)
                if result is not None:
                    return result
            result = nested_number(child, aliases)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = nested_number(child, aliases)
            if result is not None:
                return result
    return None


def test_metric_number(value: Any, test_name: str) -> float | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = "".join(c for c in str(key).lower() if c.isalnum())
            if normalized == test_name:
                if isinstance(child, dict):
                    return nested_number(child, {"pvalue", "p"})
                return number(child)
            result = test_metric_number(child, test_name)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = test_metric_number(child, test_name)
            if result is not None:
                return result
    return None


def walk_summary(
    value: Any,
    context: dict[str, Any] | None = None,
    key_hint: str | None = None,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    current = dict(context or {})
    hinted_instrument = instrument(key_hint) if key_hint else None
    hinted_confidence = confidence(key_hint) if key_hint else None
    hinted_model = model(key_hint) if key_hint else "unknown"
    if hinted_instrument is not None:
        current.setdefault("instrument", hinted_instrument)
    if hinted_confidence is not None:
        current.setdefault("confidence", hinted_confidence)
    if hinted_model != "unknown":
        current.setdefault("model", hinted_model)

    if isinstance(value, dict):
        direct_instrument = instrument(value.get("instrument"))
        direct_confidence = confidence(value.get("confidence"))
        direct_model = model(value.get("model", value.get("model_variant", "")))
        if direct_instrument is not None:
            current["instrument"] = direct_instrument
        if direct_confidence is not None:
            current["confidence"] = direct_confidence
        if direct_model != "unknown":
            current["model"] = direct_model
        yield value, current
        for key, child in value.items():
            yield from walk_summary(child, current, str(key))
    elif isinstance(value, list):
        for child in value:
            yield from walk_summary(child, current, key_hint)


def summary_metrics() -> dict[tuple[str, str, str], dict[str, float]]:
    tests = {
        "kupiec_p_value": "kupiec",
        "christoffersen_independence_p_value": "christoffersenindependence",
        "christoffersen_conditional_coverage_p_value": "christoffersenconditionalcoverage",
    }
    result: dict[tuple[str, str, str], dict[str, float]] = {}
    for path in sorted(BASELINE.rglob("*summary*.json")):
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        initial_context = {
            "instrument": instrument(path),
            "model": model(path),
        }
        for record, context in walk_summary(data, initial_context):
            name = context.get("instrument") or instrument(record.get("instrument"))
            level = context.get("confidence")
            if level is None:
                level = confidence(record.get("confidence", record.get("risk")))
            variant = context.get("model", "unknown")
            if name is None or level not in CONFIDENCES or variant == "unknown":
                continue
            key = (variant, name, f"{level:.2f}")
            for metric_name, test_name in tests.items():
                value = test_metric_number(record, test_name)
                if value is not None:
                    result.setdefault(key, {})[metric_name] = value
    return result

def baseline_diagnostics(sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = summary_metrics()
    result: list[dict[str, Any]] = []
    for item in sets:
        forecasts = item["rows"]
        fit_success = sum(boolean(row.get("fit_success")) is True for row in forecasts)
        for level in CONFIDENCES:
            pairs = [
                (row, risk(row, level))
                for row in forecasts
                if boolean(row.get("fit_success")) is True and risk(row, level) is not None
            ]
            valid = len(pairs)
            flags = [boolean(value.get("exceedance")) for _, value in pairs]
            observed = sum(flag is True for flag in flags) if flags else None
            key = (item["model"], item["instrument"], f"{level:.2f}")
            metrics = summaries.get(key, {})
            result.append(
                {
                    "model": item["model"],
                    "instrument": item["instrument"],
                    "confidence": f"{level:.2f}",
                    "total_origins": len(forecasts),
                    "fit_success": fit_success,
                    "fit_success_rate": fit_success / len(forecasts) if forecasts else None,
                    "valid_forecasts": valid,
                    "observed_exceedances": observed,
                    "expected_exceedances": valid * (1 - level),
                    "kupiec_p_value": metrics.get("kupiec_p_value"),
                    "christoffersen_independence_p_value": metrics.get(
                        "christoffersen_independence_p_value"
                    ),
                    "christoffersen_conditional_coverage_p_value": metrics.get(
                        "christoffersen_conditional_coverage_p_value"
                    ),
                    "kupiec_reject_5pct": (
                        metrics["kupiec_p_value"] < 0.05 if "kupiec_p_value" in metrics else None
                    ),
                    "christoffersen_independence_reject_5pct": (
                        metrics["christoffersen_independence_p_value"] < 0.05
                        if "christoffersen_independence_p_value" in metrics
                        else None
                    ),
                    "christoffersen_conditional_coverage_reject_5pct": (
                        metrics["christoffersen_conditional_coverage_p_value"] < 0.05
                        if "christoffersen_conditional_coverage_p_value" in metrics
                        else None
                    ),
                    "mean_quantile_loss": average(
                        [number(value.get("quantile_loss")) for _, value in pairs]
                    ),
                    "mean_fz0": average([number(value.get("fz0")) for _, value in pairs]),
                }
            )
    return result


def returns_data() -> dict[str, dict[str, Any]]:
    data = load_json(BASELINE / "returns.json")
    values = data.get("by_instrument", {}) if isinstance(data, dict) else {}
    return {
        name: payload
        for name, payload in values.items()
        if name in INSTRUMENTS and isinstance(payload, dict)
    }


def preflight_diagnostics() -> list[dict[str, Any]]:
    data = load_json(FINAL_PREFLIGHT)
    result: list[dict[str, Any]] = []

    def visit(value: Any, hint: str | None = None) -> None:
        if isinstance(value, dict):
            attempts = value.get("attempts", value.get("starts"))
            if isinstance(attempts, list):
                name = instrument(value.get("instrument")) or hint
                if name in INSTRUMENTS:
                    for attempt in attempts:
                        if not isinstance(attempt, dict):
                            continue
                        raw_start = attempt.get("start_index", attempt.get("start", 0))
                        start = int(number(raw_start) or 0)
                        if "start_index" in attempt or start == 0:
                            start += 1
                        fit = attempt.get("fit") if isinstance(attempt.get("fit"), dict) else {}
                        objective = number(
                            fit.get(
                                "log_likelihood",
                                fit.get(
                                    "logLikelihood",
                                    fit.get("objective", attempt.get("log_likelihood")),
                                ),
                            )
                        )
                        diagnostics = fit.get("diagnostics")
                        optimizer = diagnostics.get("optimizer") if isinstance(diagnostics, dict) else {}
                        optimizer_objective = number(
                            optimizer.get("objective") if isinstance(optimizer, dict) else None
                        )
                        occupancy = fit.get("occupancy", attempt.get("occupancy"))
                        if isinstance(occupancy, list):
                            occ = [number(item) for item in occupancy[:2]]
                        elif isinstance(occupancy, dict):
                            occ = [
                                number(occupancy.get("state_1", occupancy.get("state1"))),
                                number(occupancy.get("state_2", occupancy.get("state2"))),
                            ]
                        else:
                            occ = []
                        occ += [None] * (2 - len(occ))
                        errors = attempt.get("errors", [])
                        if isinstance(errors, str):
                            errors = [errors]
                        if not isinstance(errors, list):
                            errors = []
                        result.append(
                            {
                                "instrument": name,
                                "start": start,
                                "success": boolean(attempt.get("success")),
                                "log_likelihood": objective,
                                "objective": optimizer_objective,
                                "occupancy_state_1": occ[0],
                                "occupancy_state_2": occ[1],
                                "error_count": len(errors),
                                "log_path": (
                                    f"{rel(FINAL_RUN)}/{str(attempt.get('log_path', attempt.get('log_file', attempt.get('log')))).lstrip('/')}"
                                    if attempt.get("log_path", attempt.get("log_file", attempt.get("log")))
                                    else None
                                ),
                                "first_error": str(errors[0]) if errors else None,
                            }
                        )
                return
            for key, child in value.items():
                visit(child, instrument(key) or hint)
        elif isinstance(value, list):
            for child in value:
                visit(child, hint)

    visit(data)
    return sorted(result, key=lambda row: (row["instrument"], row["start"]))


def forecast_series(
    item: dict[str, Any], level: float
) -> tuple[
    list[str],
    list[float | None],
    list[float | None],
    list[float | None],
    list[bool | None],
]:
    values = []
    for row in item["rows"]:
        if boolean(row.get("fit_success")) is not True:
            continue
        value = risk(row, level)
        if value is None:
            continue
        values.append(
            (
                str(row.get("forecast_date", row.get("date", ""))),
                number(row.get("actual_loss_percent")),
                number(value.get("var_percent")),
                number(value.get("es_percent")),
                boolean(value.get("exceedance")),
            )
        )
    values.sort(key=lambda item: item[0])
    if not values:
        return ([], [], [], [], [])
    columns = list(zip(*values))
    return tuple([list(column) for column in columns])  # type: ignore[return-value]


def svg_text(x: float, y: float, value: str, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" fill="#111827">{html.escape(value)}</text>'
    )


def path_for(
    values: list[float | None],
    x: float,
    y: float,
    width: float,
    height: float,
    low: float,
    high: float,
) -> str:
    commands: list[str] = []
    active = False
    for index, value in enumerate(values):
        if value is None:
            active = False
            continue
        x_pos = x if len(values) <= 1 else x + width * index / (len(values) - 1)
        ratio = (value - low) / (high - low) if high > low else 0.5
        y_pos = y + height - ratio * height
        commands.append(f'{"L" if active else "M"}{x_pos:.2f},{y_pos:.2f}')
        active = True
    return " ".join(commands)


def numeric_values(
    series: Iterable[tuple[str, list[float | None], str, str | None]]
) -> list[float]:
    return [
        value
        for _, points, _, _ in series
        for value in points
        if value is not None and math.isfinite(value)
    ]


def render_chart(
    title: str,
    dates: list[str],
    series: list[tuple[str, list[float | None], str, str | None]],
    y_label: str,
    markers: list[bool | None] | None = None,
) -> str:
    width, height = 1500, 760
    left, top, right, bottom = 100, 95, 35, 105
    chart_width, chart_height = width - left - right, height - top - bottom
    values = numeric_values(series)
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    if high <= low:
        low, high = low - 1.0, high + 1.0
    padding = (high - low) * 0.06
    low, high = low - padding, high + padding
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        "<desc>POST_HOC_DESCRIPTIVE chart from existing thesis-v1 artifacts.</desc>",
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        svg_text(width / 2, 42, title, 22, "middle"),
        svg_text(left - 65, top + chart_height / 2, y_label, 14, "middle"),
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" '
        f'y2="{top + chart_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" '
        'stroke="#374151"/>',
    ]
    for tick in range(5):
        ratio = tick / 4
        y_pos = top + chart_height - ratio * chart_height
        out.append(
            f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + chart_width}" '
            f'y2="{y_pos:.2f}" stroke="#e5e7eb"/>'
        )
        out.append(
            svg_text(left - 10, y_pos + 5, f"{low + ratio * (high - low):.4g}", 12, "end")
        )
    if dates:
        for index in range(min(7, len(dates))):
            date_index = round(index * (len(dates) - 1) / max(1, min(6, len(dates) - 1)))
            x_pos = left if len(dates) <= 1 else left + chart_width * date_index / (len(dates) - 1)
            out.append(svg_text(x_pos, top + chart_height + 28, dates[date_index][:10], 11, "middle"))
    for index, (name, points, colour, dash) in enumerate(series):
        path = path_for(points, left, top, chart_width, chart_height, low, high)
        if path:
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(
                f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"{dash_attr}/>'
            )
        legend_x = left + index * 205
        out.append(
            f'<line x1="{legend_x}" y1="70" x2="{legend_x + 28}" y2="70" '
            f'stroke="{colour}" stroke-width="3"/>'
        )
        out.append(svg_text(legend_x + 36, 75, name, 13))
    if markers:
        actual = series[0][1] if series else []
        for index, flag in enumerate(markers):
            if flag is not True or index >= len(actual) or actual[index] is None:
                continue
            x_pos = left if len(dates) <= 1 else left + chart_width * index / (len(dates) - 1)
            y_pos = top + chart_height - (actual[index] - low) / (high - low) * chart_height
            out.append(
                f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="4" '
                f'fill="{COLOUR["missing"]}"/>'
            )
    if not values:
        out.append(svg_text(width / 2, top + chart_height / 2, "No finite values recorded", 18, "middle"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_panels(
    title: str,
    panels: list[
        tuple[
            str,
            list[str],
            list[tuple[str, list[float | None], str, str | None]],
            list[bool | None] | None,
        ]
    ],
    y_label: str,
) -> str:
    width, panel_height = 1500, 430
    height = 115 + panel_height * len(panels)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        "<desc>POST_HOC_DESCRIPTIVE chart from existing thesis-v1 forecasts.</desc>",
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        svg_text(width / 2, 38, title, 22, "middle"),
    ]
    for panel_index, (name, dates, series, markers) in enumerate(panels):
        top, left, right = 75 + panel_index * panel_height, 100, 35
        bottom = 72
        chart_width, chart_height = width - left - right, panel_height - bottom - 20
        values = numeric_values(series)
        low, high = (min(values), max(values)) if values else (0.0, 1.0)
        if high <= low:
            low, high = low - 1.0, high + 1.0
        padding = (high - low) * 0.06
        low, high = low - padding, high + padding
        out.append(svg_text(left, top - 18, name, 18))
        out.append(svg_text(left - 65, top + chart_height / 2, y_label, 13, "middle"))
        out.append(
            f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" '
            f'y2="{top + chart_height}" stroke="#374151"/>'
        )
        out.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" '
            'stroke="#374151"/>'
        )
        for tick in range(5):
            ratio = tick / 4
            y_pos = top + chart_height - ratio * chart_height
            out.append(
                f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + chart_width}" '
                f'y2="{y_pos:.2f}" stroke="#e5e7eb"/>'
            )
            out.append(
                svg_text(left - 10, y_pos + 5, f"{low + ratio * (high - low):.4g}", 11, "end")
            )
        if dates:
            for index in range(min(6, len(dates))):
                date_index = round(index * (len(dates) - 1) / max(1, min(5, len(dates) - 1)))
                x_pos = left if len(dates) <= 1 else left + chart_width * date_index / (len(dates) - 1)
                out.append(svg_text(x_pos, top + chart_height + 25, dates[date_index][:12], 10, "middle"))
        for index, (series_name, points, colour, dash) in enumerate(series):
            path = path_for(points, left, top, chart_width, chart_height, low, high)
            if path:
                dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
                out.append(
                    f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"{dash_attr}/>'
                )
            legend_x = left + index * 205
            out.append(
                f'<line x1="{legend_x}" y1="{top - 42}" x2="{legend_x + 28}" '
                f'y2="{top - 42}" stroke="{colour}" stroke-width="3"/>'
            )
            out.append(svg_text(legend_x + 36, top - 37, series_name, 12))
        if markers:
            actual = series[0][1] if series else []
            for index, flag in enumerate(markers):
                if flag is not True or index >= len(actual) or actual[index] is None:
                    continue
                x_pos = left if len(dates) <= 1 else left + chart_width * index / (len(dates) - 1)
                y_pos = top + chart_height - (actual[index] - low) / (high - low) * chart_height
                out.append(
                    f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="4" '
                    f'fill="{COLOUR["missing"]}"/>'
                )
        if not values:
            out.append(svg_text(width / 2, top + chart_height / 2, "No finite values recorded", 17, "middle"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_preflight(
    title: str,
    rows: list[dict[str, Any]],
    fields: list[tuple[str, str, str]],
) -> str:
    panels = []
    for name in INSTRUMENTS:
        values = sorted(
            (row for row in rows if row["instrument"] == name),
            key=lambda row: row["start"],
        )
        dates = [f"start {row['start']}" for row in values]
        series = [
            (label, [number(row.get(key)) for row in values], colour, None)
            for label, key, colour in fields
        ]
        panels.append((name, dates, series, None))
    svg = render_panels(title, panels, "value")
    has_missing = any(
        number(row.get(key)) is None
        for row in rows
        for _, key, _ in fields
    )
    if not has_missing:
        return svg
    return svg.replace(
        "</svg>",
        '<text x="1450" y="52" text-anchor="end" font-family="sans-serif" '
        'font-size="12" fill="#dc2626">NA = not recorded</text>\n</svg>',
    )


def csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.17g}"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    result = "| " + " | ".join(fields) + " |\n"
    result += "| " + " | ".join("---" for _ in fields) + " |\n"
    for row in rows:
        result += "| " + " | ".join(str(csv_value(row.get(field))) for field in fields) + " |\n"
    return result


def source_hashes() -> dict[str, str]:
    paths: set[Path] = {PROTOCOL, INPUT_MANIFEST, GENERATOR}
    for root in (ROOT / "experiments/thesis-v1", BASELINE, REGIME, FINAL_RUN):
        if root.exists():
            paths.update(path for path in root.rglob("*") if path.is_file())
    return {rel(path): sha256(path) for path in sorted(paths) if path.is_file()}


def readme(preflight_rows: list[dict[str, Any]]) -> str:
    finite_ll = sum(row["log_likelihood"] is not None for row in preflight_rows)
    finite_occ = sum(
        row["occupancy_state_1"] is not None or row["occupancy_state_2"] is not None
        for row in preflight_rows
    )
    return f"""# thesis-v1 post-hoc diagnostics

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

The final preserved preflight contains {len(preflight_rows)} rows, with
{sum(row["success"] is True for row in preflight_rows)} successful fits,
{finite_ll} finite log-likelihood values and {finite_occ} rows containing
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
"""


def generate() -> dict[str, Any]:
    protocol_data = load_json(PROTOCOL)
    input_data = load_json(INPUT_MANIFEST)
    baseline_manifest = load_json(BASELINE / "manifest.json")
    regime_manifest = load_json(REGIME / "manifest.json")
    final_manifest = load_json(FINAL_RUN / "manifest.json")
    sets = forecast_sets()
    if len(sets) < 4:
        raise RuntimeError("Expected both baseline variants for both instruments")
    baseline_rows = baseline_diagnostics(sets)
    preflight_rows = preflight_diagnostics()
    returns = returns_data()
    PACKAGE.mkdir(parents=True, exist_ok=True)

    dates = sorted(
        {
            str(date)
            for payload in returns.values()
            for date in payload.get("dates", [])
        }
    )
    return_series = []
    for name in INSTRUMENTS:
        payload = returns.get(name, {})
        by_date = {
            str(date): number(value)
            for date, value in zip(payload.get("dates", []), payload.get("log_returns", []))
        }
        return_series.append(
            (name, [by_date.get(date) for date in dates], COLOUR[name], None)
        )
    write_text(
        PACKAGE / "charts/log_returns.svg",
        render_chart("BTC-USD i ETH-USD: log-returns", dates, return_series, "log-return"),
    )

    for level, suffix in ((0.95, "95"), (0.99, "99")):
        panels = []
        for name in INSTRUMENTS:
            item = next(
                item
                for item in sets
                if item["instrument"] == name and item["model"] == "garch_student_t"
            )
            dates, actual, var, es, flags = forecast_series(item, level)
            panels.append(
                (
                    name,
                    dates,
                    [
                        ("actual loss", actual, COLOUR["actual"], None),
                        ("VaR", var, COLOUR["var"], "8 5"),
                        ("ES", es, COLOUR["es"], "3 3"),
                    ],
                    flags,
                )
            )
        write_text(
            PACKAGE / f"charts/var_es_exceedances_{suffix}.svg",
            render_panels(
                f"Baseline Student-t: VaR/ES and exceedances at {int(level * 100)} percent",
                panels,
                "loss percent",
            ),
        )

    write_text(
        PACKAGE / "charts/msgarch_preflight_loglikelihood.svg",
        render_preflight(
            "MSGARCH preflight: start and log-likelihood/objective",
            preflight_rows,
            [("log-likelihood / objective", "log_likelihood", "#7c3aed")],
        ),
    )
    write_text(
        PACKAGE / "charts/msgarch_preflight_occupancy.svg",
        render_preflight(
            "MSGARCH preflight: start and occupancy",
            preflight_rows,
            [
                ("state 1 occupancy", "occupancy_state_1", "#059669"),
                ("state 2 occupancy", "occupancy_state_2", "#0f766e"),
            ],
        ),
    )

    baseline_fields = [
        "model",
        "instrument",
        "confidence",
        "total_origins",
        "fit_success",
        "fit_success_rate",
        "valid_forecasts",
        "observed_exceedances",
        "expected_exceedances",
        "kupiec_p_value",
        "christoffersen_independence_p_value",
        "christoffersen_conditional_coverage_p_value",
        "kupiec_reject_5pct",
        "christoffersen_independence_reject_5pct",
        "christoffersen_conditional_coverage_reject_5pct",
        "mean_quantile_loss",
        "mean_fz0",
    ]
    write_csv(PACKAGE / "tables/baseline_diagnostics.csv", baseline_rows, baseline_fields)
    write_text(
        PACKAGE / "tables/baseline_diagnostics.md",
        markdown_table(baseline_rows, baseline_fields),
    )

    preflight_fields = [
        "instrument",
        "start",
        "success",
        "log_likelihood",
        "objective",
        "occupancy_state_1",
        "occupancy_state_2",
        "error_count",
        "log_path",
        "first_error",
    ]
    write_csv(
        PACKAGE / "tables/msgarch_preflight_diagnostics.csv",
        preflight_rows,
        preflight_fields,
    )
    write_text(
        PACKAGE / "tables/msgarch_preflight_diagnostics.md",
        markdown_table(preflight_rows, preflight_fields),
    )

    candidates = {
        "standardized_residual",
        "standardized_residuals",
        "std_residual",
        "z_score",
        "z",
    }
    hits = sorted(
        {
            key
            for item in sets
            for row in item["rows"]
            for key in candidates
            if key in row
        }
    )
    residuals = {
        "available": bool(hits),
        "detected_fields": hits,
        "refit_performed": False,
        "missing_choice": (
            "Persist per-origin standardized residuals in a separately identified "
            "pipeline output, then review the methodological implications before "
            "generating residual diagnostics."
        ),
    }
    protocol_id = first_value(protocol_data, {"protocolId", "protocol_id"}) or "thesis-v1"
    metadata = {
        "analysis_boundary": "POST_HOC_DESCRIPTIVE",
        "protocol_id": protocol_id,
        "snapshot_id": first_value(input_data, {"snapshot_id", "snapshotId"}),
        "instruments": list(INSTRUMENTS),
        "baseline_diagnostic_rows": len(baseline_rows),
        "preflight_source": rel(FINAL_PREFLIGHT),
        "preflight_rows": len(preflight_rows),
        "preflight_success_rows": sum(row["success"] is True for row in preflight_rows),
        "preflight_finite_objective_rows": sum(row["objective"] is not None for row in preflight_rows),
        "standardized_residuals": residuals,
        "models_included": ["garch_student_t", "garch_normal"],
        "regime_context_label": "fallback_not_ms_garch",
        "winner_selected": False,
        "oos_conclusion_changed": False,
    }
    write_json(PACKAGE / "metadata.json", metadata)
    write_text(PACKAGE / "README.md", readme(preflight_rows))

    sources = source_hashes()
    artifact_hashes = {
        rel(path): sha256(path)
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "report_status": "posthoc_diagnostics",
        "analysis_boundary": "POST_HOC_DESCRIPTIVE",
        "post_hoc_descriptive": True,
        "canonical_model_selected": False,
        "model_selection_performed": False,
        "superiority_claim": False,
        "protocol_id": protocol_id,
        "protocol_sha256": sha256(PROTOCOL),
        "snapshot_id": first_value(input_data, {"snapshot_id", "snapshotId"}),
        "snapshot_file_sha256": first_value(
            input_data,
            {"snapshot_file_sha256", "snapshotFileSha256", "file_sha256"},
        ),
        "source_runs": {
            "baseline": {
                "manifest": rel(BASELINE / "manifest.json"),
                "git_commit": baseline_manifest.get("git_commit"),
                "completed_at_utc": baseline_manifest.get("completed_at_utc"),
                "status": baseline_manifest.get("status"),
            },
            "regime": {
                "manifest": rel(REGIME / "manifest.json"),
                "git_commit": regime_manifest.get("git_commit"),
                "completed_at_utc": regime_manifest.get("completed_at_utc"),
                "status": regime_manifest.get("status"),
                "model_variant": "fallback_not_ms_garch",
                "preflight_failure_count": regime_manifest.get("preflight_failure_count"),
            },
            "final_msgarch_attempt_2": {
                "manifest": rel(FINAL_RUN / "manifest.json"),
                "git_commit": final_manifest.get("git_commit"),
                "completed_at_utc": final_manifest.get("completed_at_utc"),
                "status": final_manifest.get("status"),
                "model_variant": final_manifest.get("model_variant"),
                "preflight_failure_count": final_manifest.get("preflight_failure_count"),
                "preflight_gate_failure_count": final_manifest.get("preflight_gate_failure_count"),
                "preflight_passed_for_both_instruments": final_manifest.get(
                    "preflight_passed_for_both_instruments"
                ),
            },
        },
        "source_sha256": sources,
        "artifact_sha256": artifact_hashes,
        "source_count": len(sources),
        "artifact_count": len(artifact_hashes),
        "report_files": sorted(artifact_hashes),
        "standardized_residuals": residuals,
        "constraints": {
            "new_data_downloaded": False,
            "new_msgarch_fit": False,
            "protocol_changed": False,
            "existing_artifacts_modified": False,
            "winner_selected": False,
            "oos_conclusion_changed": False,
        },
    }
    write_json(PACKAGE / "manifest.json", manifest)
    return manifest


def main() -> None:
    manifest = generate()
    print(
        json.dumps(
            {
                "package": rel(PACKAGE),
                "report_status": manifest["report_status"],
                "artifact_count": manifest["artifact_count"],
                "source_count": manifest["source_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
