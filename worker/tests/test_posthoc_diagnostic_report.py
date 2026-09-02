"""Tests for the thesis-v1 post-hoc diagnostic package."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from regime_sentinel_worker.artifacts import posthoc_diagnostic_report as report


PACKAGE = report.PACKAGE


def strict_load(path: Path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PosthocDiagnosticReportTests(unittest.TestCase):
    def assert_manifest_hashes(self, manifest):
        for relative, expected in manifest["artifact_sha256"].items():
            self.assertEqual(expected, file_sha256(report.ROOT / relative), relative)
        for relative, expected in manifest["source_sha256"].items():
            self.assertEqual(expected, file_sha256(report.ROOT / relative), relative)

    def test_clean_checkout_manifest_hashes_are_checked_before_regeneration(self):
        committed = strict_load(PACKAGE / "manifest.json")
        self.assertEqual(committed["analysis_boundary"], "POST_HOC_DESCRIPTIVE")
        self.assertTrue(committed["post_hoc_descriptive"])
        self.assert_manifest_hashes(committed)

        first = report.generate()
        first_artifacts = dict(first["artifact_sha256"])
        second = report.generate()

        self.assertEqual(second["analysis_boundary"], "POST_HOC_DESCRIPTIVE")
        self.assertFalse(second["canonical_model_selected"])
        self.assertFalse(second["model_selection_performed"])
        self.assertFalse(second["superiority_claim"])
        self.assertEqual(first_artifacts, second["artifact_sha256"])
        self.assertGreaterEqual(second["artifact_count"], 8)
        self.assertGreaterEqual(second["source_count"], 1)

        expected_charts = {
            "artifacts/posthoc-diagnostics/charts/log_returns.svg",
            "artifacts/posthoc-diagnostics/charts/var_es_exceedances_95.svg",
            "artifacts/posthoc-diagnostics/charts/var_es_exceedances_99.svg",
            "artifacts/posthoc-diagnostics/charts/msgarch_preflight_loglikelihood.svg",
            "artifacts/posthoc-diagnostics/charts/msgarch_preflight_occupancy.svg",
        }
        self.assertTrue(expected_charts.issubset(set(second["artifact_sha256"])))

    def test_tables_match_final_preflight_and_summary_json(self):
        report.generate()

        source_preflight = strict_load(report.FINAL_PREFLIGHT)
        source_rows = []
        for name, block in source_preflight.items():
            for attempt in block["attempts"]:
                fit = attempt["fit"]
                source_rows.append(
                    (
                        name,
                        attempt["start_index"] + 1,
                        fit["log_likelihood"],
                        fit["diagnostics"]["optimizer"]["objective"],
                        fit["occupancy"],
                    )
                )
        generated_preflight = read_csv(
            PACKAGE / "tables/msgarch_preflight_diagnostics.csv"
        )
        self.assertEqual(len(source_rows), 10)
        self.assertEqual(len(generated_preflight), 10)
        for generated, expected in zip(
            sorted(generated_preflight, key=lambda row: (row["instrument"], int(row["start"]))),
            sorted(source_rows, key=lambda row: (row[0], row[1])),
        ):
            self.assertEqual(generated["instrument"], expected[0])
            self.assertEqual(int(generated["start"]), expected[1])
            self.assertEqual(float(generated["log_likelihood"]), expected[2])
            self.assertEqual(float(generated["objective"]), expected[3])
            self.assertEqual(float(generated["occupancy_state_1"]), expected[4][0])
            self.assertEqual(float(generated["occupancy_state_2"]), expected[4][1])
            self.assertTrue(generated["success"] == "true")
            self.assertTrue(generated["log_path"].startswith(
                "artifacts/msgarch-attempt-2/"
            ))

        generated_baseline = read_csv(PACKAGE / "tables/baseline_diagnostics.csv")
        self.assertEqual(len(generated_baseline), 8)
        expected_baseline = report.baseline_diagnostics(report.forecast_sets())
        for generated, expected in zip(
            sorted(generated_baseline, key=lambda row: (row["model"], row["instrument"], row["confidence"])),
            sorted(expected_baseline, key=lambda row: (row["model"], row["instrument"], row["confidence"])),
        ):
            self.assertEqual(generated["model"], expected["model"])
            self.assertEqual(generated["instrument"], expected["instrument"])
            self.assertEqual(generated["confidence"], expected["confidence"])
            for field in (
                "kupiec_p_value",
                "christoffersen_independence_p_value",
                "christoffersen_conditional_coverage_p_value",
            ):
                self.assertNotEqual(generated[field], "NA")
                self.assertAlmostEqual(float(generated[field]), expected[field], places=12)

        lookup = {
            (row["model"], row["instrument"], row["confidence"]): row
            for row in generated_baseline
        }
        btc_normal_99 = lookup[("garch_normal", "BTC-USD", "0.99")]
        self.assertAlmostEqual(float(btc_normal_99["kupiec_p_value"]), 0.0018654043832517315)
        self.assertAlmostEqual(
            float(btc_normal_99["christoffersen_independence_p_value"]),
            0.3074328971851498,
        )
        self.assertAlmostEqual(
            float(btc_normal_99["christoffersen_conditional_coverage_p_value"]),
            0.004702902481832223,
        )

    def test_visualizations_match_source_series_and_exceedance_flags(self):
        report.generate()
        returns = report.returns_data()
        self.assertEqual(
            {name: len(payload["log_returns"]) for name, payload in returns.items()},
            {"BTC-USD": 1825, "ETH-USD": 1825},
        )
        log_chart = (PACKAGE / "charts/log_returns.svg").read_text(encoding="utf-8")
        self.assertIn("#f7931a", log_chart)
        self.assertIn("#627eea", log_chart)
        self.assertIn("2021-07-21", log_chart)
        self.assertIn("2026-07-19", log_chart)

        for level, suffix in ((0.95, "95"), (0.99, "99")):
            expected_markers = 0
            expected_paths = 0
            for instrument in report.INSTRUMENTS:
                item = next(
                    item
                    for item in report.forecast_sets()
                    if item["instrument"] == instrument
                    and item["model"] == "garch_student_t"
                )
                _, actual, var, es, flags = report.forecast_series(item, level)
                expected_markers += sum(flag is True for flag in flags)
                expected_paths += sum(
                    value is not None
                    for value in (actual, var, es)
                )
            chart = (
                PACKAGE / "charts" / f"var_es_exceedances_{suffix}.svg"
            ).read_text(encoding="utf-8")
            self.assertEqual(chart.count("<circle"), expected_markers)
            self.assertGreaterEqual(chart.count("<path"), 6)
            self.assertGreater(expected_paths, 0)

        for name in (
            "msgarch_preflight_loglikelihood.svg",
            "msgarch_preflight_occupancy.svg",
        ):
            chart = (PACKAGE / "charts" / name).read_text(encoding="utf-8")
            self.assertNotIn("No finite values recorded", chart)
            self.assertNotIn("NA = not recorded", chart)

    def test_json_and_svg_outputs_are_strict_and_parseable(self):
        report.generate()
        json_files = sorted(PACKAGE.rglob("*.json"))
        svg_files = sorted(PACKAGE.rglob("*.svg"))
        self.assertGreaterEqual(len(json_files), 2)
        self.assertGreaterEqual(len(svg_files), 5)

        for path in json_files:
            strict_load(path)
        for path in svg_files:
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertIn("POST_HOC_DESCRIPTIVE", path.read_text(encoding="utf-8"))

        for name in ("var_es_exceedances_95.svg", "var_es_exceedances_99.svg"):
            text = (PACKAGE / "charts" / name).read_text(encoding="utf-8")
            self.assertIn("<circle", text)

    def test_residual_boundary_is_explicit_and_no_refit_is_recorded(self):
        manifest = report.generate()
        residuals = manifest["standardized_residuals"]
        self.assertFalse(residuals["available"])
        self.assertFalse(residuals["refit_performed"])
        self.assertEqual(residuals["detected_fields"], [])
        self.assertIn("separately identified", residuals["missing_choice"])
        self.assertIn("fallback_not_ms_garch", (PACKAGE / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
