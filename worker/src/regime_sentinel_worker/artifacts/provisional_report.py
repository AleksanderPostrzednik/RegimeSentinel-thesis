"""Generate reproducible, explicitly provisional thesis-v1 report artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


# This module is nested below the package root (`artifacts/`).
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "provisional"
PROTOCOL_PATH = Path("protocol/thesis-v1.json")
GENERATOR_PATH = Path("worker/src/regime_sentinel_worker/artifacts/provisional_report.py")
SNAPSHOT_ID = "yahoo-btc-eth-daily-close-2021-07-20_2026-07-19"
INSTRUMENTS = ("BTC-USD", "ETH-USD")
CONFIDENCES = (0.95, 0.99)
MODEL_SPECS = (
    ("garch11_student_t", "GARCH(1,1) Student-t", "baseline/student_t"),
    ("garch11_normal", "GARCH(1,1) normal", "baseline/normal"),
    ("fallback_not_ms_garch", "fallback_not_ms_garch", "fallback/fallback_not_ms_garch"),
)
MODEL_LABELS = {key: label for key, label, _ in MODEL_SPECS}
PROVISIONAL_NOTICE = (
    "PROVISIONAL — exploratory artifact summary; no canonical model selected and "
    "no superiority claim is made."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source_manifest(stage_root: Path) -> dict[str, Any]:
    manifest_path = stage_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"source manifest is not an object: {manifest_path}")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"source manifest has no artifact hashes: {manifest_path}")
    for relative, expected in hashes.items():
        path = stage_root / str(relative)
        if not path.is_file():
            raise ValueError(f"source artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"source artifact hash mismatch for {path}: expected {expected}, got {actual}"
            )
    return manifest


def _source_context(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    artifact_root = repo_root / "artifacts"
    baseline_manifest = _verify_source_manifest(artifact_root / "baseline")
    regime_manifest = _verify_source_manifest(artifact_root / "fallback")
    protocol_path = repo_root / PROTOCOL_PATH
    input_manifest_path = repo_root / "protocol" / "input-manifest.json"
    source_hashes: dict[str, str] = {
        PROTOCOL_PATH.as_posix(): sha256_file(protocol_path),
        input_manifest_path.relative_to(repo_root).as_posix(): sha256_file(input_manifest_path),
        GENERATOR_PATH.as_posix(): sha256_file(repo_root / GENERATOR_PATH),
    }
    for stage, manifest in (("baseline", baseline_manifest), ("fallback", regime_manifest)):
        stage_root = artifact_root / stage
        source_hashes[f"artifacts/{stage}/manifest.json"] = sha256_file(
            stage_root / "manifest.json"
        )
        for relative in manifest["artifact_sha256"]:
            source_hashes[f"artifacts/{stage}/{relative}"] = sha256_file(
                stage_root / relative
            )
    return baseline_manifest, regime_manifest, source_hashes


def _safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return fmean(values) if values else None


def _metric_record(
    repo_root: Path,
    model_key: str,
    instrument: str,
    confidence: float,
) -> dict[str, Any]:
    _, _, relative_root = next(spec for spec in MODEL_SPECS if spec[0] == model_key)
    root = repo_root / "artifacts" / relative_root
    summary = _read_json(root / instrument / "summary.json")
    forecasts = _read_json(root / instrument / "forecasts.json")
    if not isinstance(summary, dict) or not isinstance(forecasts, list):
        raise ValueError(f"invalid source payload for {model_key}/{instrument}")
    level = f"{confidence:.2f}"
    risk_summary = summary["risk"][level]
    valid_rows = [
        row for row in forecasts if row.get("fit_success") and level in row.get("risk", {})
    ]
    forecast_count = len(forecasts)
    valid_count = len(valid_rows)
    observed = sum(bool(row["risk"][level]["exceedance"]) for row in valid_rows)
    tail = 1.0 - confidence
    kupiec = risk_summary["kupiec"]
    independence = risk_summary["christoffersen_independence"]
    conditional = risk_summary["christoffersen_conditional_coverage"]
    return {
        "provisional": True,
        "model_key": model_key,
        "model_label": MODEL_LABELS[model_key],
        "instrument": instrument,
        "confidence": confidence,
        "confidence_label": f"{int(confidence * 100)}%",
        "forecast_count": forecast_count,
        "valid_forecasts": valid_count,
        "fit_success_count": int(summary["fit_success_count"]),
        "fit_failure_count": int(summary["fit_failure_count"]),
        "fit_success_rate": float(summary["fit_success_rate"]),
        "mean_var_percent": _safe_mean(
            float(row["risk"][level]["var_percent"]) for row in valid_rows
        ),
        "mean_es_percent": _safe_mean(
            float(row["risk"][level]["es_percent"]) for row in valid_rows
        ),
        "expected_exceedances_full_oos": forecast_count * tail,
        "expected_exceedances_valid": valid_count * tail,
        "observed_exceedances": observed,
        "observed_exceedance_rate": observed / valid_count if valid_count else None,
        "kupiec_p_value": kupiec.get("p_value"),
        "kupiec_reject_at_5pct": kupiec.get("reject_at_5pct"),
        "christoffersen_independence_p_value": independence.get("p_value"),
        "christoffersen_independence_reject_at_5pct": independence.get("reject_at_5pct"),
        "christoffersen_conditional_coverage_p_value": conditional.get("p_value"),
        "christoffersen_conditional_coverage_reject_at_5pct": conditional.get(
            "reject_at_5pct"
        ),
        "quantile_loss": risk_summary.get("mean_var_quantile_loss"),
        "fz0": risk_summary.get("mean_fz0_score"),
    }


def _difference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["model_key"], row["confidence"], row["instrument"]): row for row in rows}
    metrics = (
        "fit_success_rate",
        "mean_var_percent",
        "mean_es_percent",
        "valid_forecasts",
        "observed_exceedances",
        "observed_exceedance_rate",
        "kupiec_p_value",
        "christoffersen_independence_p_value",
        "christoffersen_conditional_coverage_p_value",
        "quantile_loss",
        "fz0",
    )
    differences: list[dict[str, Any]] = []
    for model_key, model_label, _ in MODEL_SPECS:
        for confidence in CONFIDENCES:
            btc = index[(model_key, confidence, "BTC-USD")]
            eth = index[(model_key, confidence, "ETH-USD")]
            row: dict[str, Any] = {
                "provisional": True,
                "model_key": model_key,
                "model_label": model_label,
                "confidence": confidence,
                "confidence_label": f"{int(confidence * 100)}%",
                "difference_definition": "BTC-USD minus ETH-USD",
            }
            for metric in metrics:
                btc_value = btc[metric]
                eth_value = eth[metric]
                row[f"btc_minus_eth_{metric}"] = (
                    float(btc_value) - float(eth_value)
                    if btc_value is not None and eth_value is not None
                    else None
                )
            differences.append(row)
    return differences


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [header for header, _ in columns]
    keys = [key for _, key in columns]
    lines = [
        f"# {PROVISIONAL_NOTICE}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(key)) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_text(value: str) -> str:
    return html.escape(value, quote=True)


def _svg_bar_chart(
    path: Path,
    *,
    title: str,
    y_label: str,
    categories: list[str],
    series: list[tuple[str, str, list[float | None]]],
    threshold: float | None = None,
) -> None:
    width, height = 1200, 620
    left, right, top, bottom = 100, 40, 96, 110
    chart_width, chart_height = width - left - right, height - top - bottom
    values = [float(value) for _, _, points in series for value in points if value is not None]
    maximum = max(values + ([threshold] if threshold is not None else [0.0]))
    maximum = max(maximum * 1.15, 1.0 if maximum <= 1 else maximum * 1.05)
    group_width = chart_width / max(len(categories), 1)
    bar_width = min(42.0, group_width / max(len(series) + 1, 1))

    def y(value: float) -> float:
        return top + chart_height - value / maximum * chart_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" '
        f'viewBox="0 0 {width} {height}">',
        f"<title id=\"title\">{_svg_text('PROVISIONAL — ' + title)}</title>",
        f'<desc id="desc">{_svg_text(PROVISIONAL_NOTICE)} { _svg_text(y_label) }</desc>',
        '<rect width="1200" height="620" fill="#ffffff"/>',
        '<text x="100" y="34" font-family="sans-serif" font-size="22" font-weight="600" fill="#17202a">'
        f"{_svg_text(title)}</text>",
        '<text x="100" y="60" font-family="sans-serif" font-size="13" fill="#5b6670">'
        f"{_svg_text(PROVISIONAL_NOTICE)}</text>",
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#34495e"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}" stroke="#34495e"/>',
        f'<text transform="translate(24 {top + chart_height / 2}) rotate(-90)" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14" fill="#34495e">{_svg_text(y_label)}</text>',
    ]
    tick_count = 5
    for index in range(tick_count + 1):
        value = maximum * index / tick_count
        tick_y = y(value)
        lines.append(
            f'<line x1="{left}" y1="{tick_y:.2f}" x2="{width - right}" y2="{tick_y:.2f}" '
            'stroke="#d7dde2" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{tick_y + 5:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12" fill="#34495e">{value:.2f}</text>'
        )
    if threshold is not None:
        threshold_y = y(threshold)
        lines.append(
            f'<line x1="{left}" y1="{threshold_y:.2f}" x2="{width - right}" y2="{threshold_y:.2f}" '
            'stroke="#a33b3b" stroke-width="2" stroke-dasharray="7 5"/>'
        )
        lines.append(
            f'<text x="{width - right - 6}" y="{threshold_y - 8:.2f}" text-anchor="end" '
            'font-family="sans-serif" font-size="12" fill="#a33b3b">5% threshold</text>'
        )
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        for series_index, (label, color, points) in enumerate(series):
            value = points[category_index]
            if value is None:
                continue
            x = center + (series_index - (len(series) - 1) / 2) * bar_width
            bar_y = y(float(value))
            bar_height = top + chart_height - bar_y
            lines.append(
                f'<rect x="{x - bar_width * 0.42:.2f}" y="{bar_y:.2f}" width="{bar_width * 0.84:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" data-series="{_svg_text(label)}">'
                f'<title>{_svg_text(label)}: {_format_cell(value)}</title></rect>'
            )
        for line_index, part in enumerate(category.split("\n")):
            lines.append(
                f'<text x="{center:.2f}" y="{top + chart_height + 24 + line_index * 16}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12" fill="#34495e">{_svg_text(part)}</text>'
            )
    legend_x = left
    legend_step = chart_width / max(len(series), 1)
    for index, (label, color, _) in enumerate(series):
        x = legend_x + index * legend_step
        lines.append(f'<rect x="{x}" y="76" width="12" height="12" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 18}" y="87" font-family="sans-serif" font-size="12" fill="#34495e">'
            f"{_svg_text(label)}</text>"
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _categories() -> list[str]:
    return [
        f"{instrument}\n{MODEL_LABELS[model_key]}"
        for instrument in INSTRUMENTS
        for model_key, _, _ in MODEL_SPECS
    ]


def _values(rows: list[dict[str, Any]], confidence: float, metric: str) -> list[float | None]:
    index = {(row["instrument"], row["model_key"], row["confidence"]): row for row in rows}
    return [
        index[(instrument, model_key, confidence)].get(metric)
        for instrument in INSTRUMENTS
        for model_key, _, _ in MODEL_SPECS
    ]


def _write_charts(output_root: Path, rows: list[dict[str, Any]]) -> list[str]:
    charts = output_root / "charts"
    categories = _categories()
    generated: list[str] = []
    fit_path = charts / "fit_success_rate.svg"
    _svg_bar_chart(
        fit_path,
        title="Fit success rate by instrument and provisional variant",
        y_label="Fit success rate",
        categories=categories,
        series=[("fit success", "#2f6f9f", _values(rows, 0.95, "fit_success_rate"))],
    )
    generated.append(fit_path.relative_to(output_root).as_posix())
    for confidence in CONFIDENCES:
        suffix = int(confidence * 100)
        for filename, title, y_label, metric_series, threshold in (
            (
                f"mean_var_es_{suffix}.svg",
                f"Mean VaR and ES at {suffix}% by instrument and provisional variant",
                "Mean positive loss (%)",
                [("mean VaR", "#2f6f9f", "mean_var_percent"), ("mean ES", "#b36b2c", "mean_es_percent")],
                None,
            ),
            (
                f"exceedances_{suffix}.svg",
                f"Observed versus expected exceedances at {suffix}%",
                "Number of exceedances",
                [
                    ("observed", "#7256a6", "observed_exceedances"),
                    ("expected valid", "#2f6f9f", "expected_exceedances_valid"),
                    ("expected full OOS", "#7a8793", "expected_exceedances_full_oos"),
                ],
                None,
            ),
            (
                f"backtest_pvalues_{suffix}.svg",
                f"Kupiec and Christoffersen p-values at {suffix}%",
                "p-value",
                [
                    ("Kupiec", "#2f6f9f", "kupiec_p_value"),
                    ("Christoffersen independence", "#b36b2c", "christoffersen_independence_p_value"),
                    ("Christoffersen conditional coverage", "#7256a6", "christoffersen_conditional_coverage_p_value"),
                ],
                0.05,
            ),
            (
                f"scores_{suffix}.svg",
                f"Quantile loss and FZ0 at {suffix}%",
                "Score (lower is not automatically a selection rule here)",
                [("quantile loss", "#2f6f9f", "quantile_loss"), ("FZ0", "#b36b2c", "fz0")],
                None,
            ),
        ):
            path = charts / filename
            _svg_bar_chart(
                path,
                title=title,
                y_label=y_label,
                categories=categories,
                series=[(label, color, _values(rows, confidence, metric)) for label, color, metric in metric_series],
                threshold=threshold,
            )
            generated.append(path.relative_to(output_root).as_posix())
    return generated


def _write_readme(output_root: Path, rows: list[dict[str, Any]], source_manifests: dict[str, Any]) -> str:
    lines = [
        "# PROVISIONAL — thesis-v1 tables and charts",
        "",
        PROVISIONAL_NOTICE,
        "",
        "This package is generated only from the existing, hash-verified `thesis-v1` artifacts.",
        "It reports all three recorded variants separately: GARCH(1,1) Student-t, GARCH(1,1) normal,",
        "and `fallback_not_ms_garch`. It does not select a canonical model and does not state superiority.",
        "",
        "## Scope",
        "",
        "- BTC-USD and ETH-USD are shown separately.",
        "- VaR/ES are positive-loss quantities at 95% and 99%.",
        "- Valid forecast counts and fit failures are shown explicitly; missing fits are not silently imputed.",
        "- Observed exceedances are compared with the expectation for valid forecasts; the full-OOS expectation is retained to expose missing-fit coverage.",
        "- Kupiec, Christoffersen independence, Christoffersen conditional coverage, quantile loss and FZ0 are reported as diagnostics.",
        "- Differences are descriptive `BTC-USD minus ETH-USD`; they are not a model ranking.",
        "- `fallback_not_ms_garch` is a protocol fallback label, not an MS-GARCH result.",
        "",
        "## Tables",
        "",
        "- `tables/model_metrics.csv` and `tables/model_metrics.md` — one row per variant, instrument and confidence level.",
        "- `tables/btc_eth_differences.csv` and `tables/btc_eth_differences.md` — descriptive BTC-minus-ETH differences.",
        "",
        "## Charts",
        "",
        "All SVG charts carry the `PROVISIONAL` notice in their title and accessibility description.",
        "",
        *[f"- `{path}`" for path in sorted(_list_chart_paths(output_root))],
        "",
        "## Provenance",
        "",
        f"- Protocol: `thesis-v1` (`{source_manifests['protocol_sha256']}`).",
        f"- Snapshot: `{SNAPSHOT_ID}` (`{source_manifests['snapshot_file_sha256']}`).",
        f"- Baseline source run: `{source_manifests['baseline_completed_at_utc']}`.",
        f"- Regime source run: `{source_manifests['regime_completed_at_utc']}`.",
        "- Full source and output hashes are in `manifest.json`.",
        "",
        "## Reproduction",
        "",
        "From the repository root run:",
        "",
        "```bash",
        "PYTHONPATH=worker/src python3 -m regime_sentinel_worker.artifacts.provisional_report",
        "PYTHONPATH=worker/src python3 -m unittest worker.tests.test_provisional_report -v",
        "```",
        "",
        "Do not copy this provisional package into the thesis as a final result without Alek's methodological review.",
        "",
    ]
    path = output_root / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.relative_to(output_root).as_posix()


def _list_chart_paths(output_root: Path) -> list[str]:
    return [path.relative_to(output_root).as_posix() for path in (output_root / "charts").glob("*.svg")]


def generate_provisional_report(repo_root: Path, output_root: Path) -> dict[str, Any]:
    """Generate the complete provisional package and return its manifest."""
    baseline_manifest, regime_manifest, source_hashes = _source_context(repo_root)
    protocol = _read_json(repo_root / PROTOCOL_PATH)
    if not isinstance(protocol, dict) or protocol.get("protocolId") != "thesis-v1":
        raise ValueError("the report requires the frozen thesis-v1 protocol")
    rows = [
        _metric_record(repo_root, model_key, instrument, confidence)
        for model_key, _, _ in MODEL_SPECS
        for instrument in INSTRUMENTS
        for confidence in CONFIDENCES
    ]
    differences = _difference_rows(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    tables = output_root / "tables"
    model_fields = list(rows[0].keys())
    difference_fields = list(differences[0].keys())
    _write_csv(tables / "model_metrics.csv", rows, model_fields)
    _write_csv(tables / "btc_eth_differences.csv", differences, difference_fields)
    _write_markdown_table(
        tables / "model_metrics.md",
        rows,
        [
            ("Instrument", "instrument"),
            ("Variant", "model_label"),
            ("Level", "confidence_label"),
            ("Fit success", "fit_success_rate"),
            ("Fit failures", "fit_failure_count"),
            ("Valid forecasts", "valid_forecasts"),
            ("Mean VaR %", "mean_var_percent"),
            ("Mean ES %", "mean_es_percent"),
            ("Observed", "observed_exceedances"),
            ("Expected valid", "expected_exceedances_valid"),
            ("Expected full OOS", "expected_exceedances_full_oos"),
            ("Kupiec p", "kupiec_p_value"),
            ("Christoffersen ind. p", "christoffersen_independence_p_value"),
            ("Christoffersen CC p", "christoffersen_conditional_coverage_p_value"),
            ("QL", "quantile_loss"),
            ("FZ0", "fz0"),
        ],
    )
    _write_markdown_table(
        tables / "btc_eth_differences.md",
        differences,
        [
            ("Variant", "model_label"),
            ("Level", "confidence_label"),
            ("Definition", "difference_definition"),
            ("Δ fit success", "btc_minus_eth_fit_success_rate"),
            ("Δ mean VaR %", "btc_minus_eth_mean_var_percent"),
            ("Δ mean ES %", "btc_minus_eth_mean_es_percent"),
            ("Δ exceedances", "btc_minus_eth_observed_exceedances"),
            ("Δ Kupiec p", "btc_minus_eth_kupiec_p_value"),
            ("Δ Christoffersen ind. p", "btc_minus_eth_christoffersen_independence_p_value"),
            ("Δ Christoffersen CC p", "btc_minus_eth_christoffersen_conditional_coverage_p_value"),
            ("Δ QL", "btc_minus_eth_quantile_loss"),
            ("Δ FZ0", "btc_minus_eth_fz0"),
        ],
    )
    (output_root / "metrics.json").write_text(
        json.dumps(
            {
                "provisional": True,
                "canonical_model_selected": False,
                "model_selection_performed": False,
                "superiority_claim": False,
                "rows": rows,
                "btc_eth_differences": differences,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    chart_paths = _write_charts(output_root, rows)
    source_context = {
        "protocol_sha256": source_hashes[PROTOCOL_PATH.as_posix()],
        "snapshot_file_sha256": baseline_manifest["snapshot_file_sha256"],
        "baseline_completed_at_utc": baseline_manifest["completed_at_utc"],
        "regime_completed_at_utc": regime_manifest["completed_at_utc"],
    }
    readme_path = _write_readme(output_root, rows, source_context)
    report_files = [
        readme_path,
        "metrics.json",
        "tables/model_metrics.csv",
        "tables/model_metrics.md",
        "tables/btc_eth_differences.csv",
        "tables/btc_eth_differences.md",
        *chart_paths,
    ]
    output_hashes = {
        relative: sha256_file(output_root / relative)
        for relative in sorted(report_files)
    }
    manifest = {
        "report_status": "provisional",
        "provisional": True,
        "canonical_model_selected": False,
        "model_selection_performed": False,
        "superiority_claim": False,
        "generator_version": "provisional-report-v1",
        "protocol_id": protocol["protocolId"],
        "protocol_sha256": source_context["protocol_sha256"],
        "snapshot_id": SNAPSHOT_ID,
        "snapshot_file_sha256": source_context["snapshot_file_sha256"],
        "source_runs": {
            "baseline_completed_at_utc": source_context["baseline_completed_at_utc"],
            "regime_completed_at_utc": source_context["regime_completed_at_utc"],
            "baseline_git_commit": baseline_manifest.get("git_commit"),
            "regime_git_commit": regime_manifest.get("git_commit"),
        },
        "models_included": [MODEL_LABELS[key] for key, _, _ in MODEL_SPECS],
        "instruments": list(INSTRUMENTS),
        "confidence_levels": list(CONFIDENCES),
        "source_sha256": source_hashes,
        "artifact_sha256": output_hashes,
        "report_files": report_files,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = generate_provisional_report(args.repo_root.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "report_status": manifest["report_status"],
                "output": str(args.output.resolve()),
                "files": len(manifest["artifact_sha256"]),
                "models": manifest["models_included"],
                "instruments": manifest["instruments"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
