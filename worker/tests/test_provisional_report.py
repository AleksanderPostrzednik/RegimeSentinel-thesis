from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from regime_sentinel_worker.artifacts.provisional_report import generate_provisional_report


REPO_ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProvisionalReportTests(unittest.TestCase):
    def test_generates_all_provisional_metrics_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "provisional"
            manifest = generate_provisional_report(REPO_ROOT, output)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            chart_paths = [
                path for path in manifest["report_files"] if path.startswith("charts/")
            ]
            self.assertEqual(len(chart_paths), 9)
            self.assertTrue(all((output / path).is_file() for path in chart_paths))

        self.assertTrue(manifest["provisional"])
        self.assertFalse(manifest["canonical_model_selected"])
        self.assertFalse(manifest["model_selection_performed"])
        self.assertFalse(manifest["superiority_claim"])
        self.assertIn(
            "worker/src/regime_sentinel_worker/artifacts/provisional_report.py",
            manifest["source_sha256"],
        )
        self.assertEqual(len(metrics["rows"]), 12)
        self.assertEqual(len(metrics["btc_eth_differences"]), 6)
        self.assertEqual(
            {row["model_key"] for row in metrics["rows"]},
            {"garch11_student_t", "garch11_normal", "fallback_not_ms_garch"},
        )
        self.assertEqual(
            {row["instrument"] for row in metrics["rows"]}, {"BTC-USD", "ETH-USD"}
        )
        self.assertEqual({row["confidence"] for row in metrics["rows"]}, {0.95, 0.99})
        self.assertIn("expected_exceedances_valid", metrics["rows"][0])
        self.assertIn("expected_exceedances_full_oos", metrics["rows"][0])

    def test_generation_is_byte_deterministic_and_manifest_hashes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first_manifest = generate_provisional_report(REPO_ROOT, first)
            second_manifest = generate_provisional_report(REPO_ROOT, second)
            first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
            second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())

            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                    relative.as_posix(),
                )
            for relative, expected in first_manifest["artifact_sha256"].items():
                self.assertEqual(file_hash(first / relative), expected, relative)
            self.assertEqual(first_manifest, second_manifest)

    def test_report_text_explicitly_preserves_provisional_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "provisional"
            generate_provisional_report(REPO_ROOT, output)
            readme = (output / "README.md").read_text(encoding="utf-8")
            metrics = (output / "tables" / "model_metrics.md").read_text(encoding="utf-8")

            self.assertIn("PROVISIONAL", readme)
            self.assertIn("does not select a canonical model", readme)
            self.assertIn("PROVISIONAL", metrics)
            self.assertIn("fallback_not_ms_garch", metrics)
            self.assertIn("Expected valid", metrics)
            self.assertIn("Expected full OOS", metrics)
            chart = (output / "charts" / "backtest_pvalues_95.svg").read_text(encoding="utf-8")
            self.assertIn("<title id=\"title\">PROVISIONAL", chart)


if __name__ == "__main__":
    unittest.main()
