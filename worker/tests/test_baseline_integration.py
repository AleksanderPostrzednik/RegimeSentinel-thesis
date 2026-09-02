from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from regime_sentinel_worker.pipeline.baseline import run_baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "protocol" / "thesis-v1.json"


class BaselineIntegrationTests(unittest.TestCase):
    def test_short_real_snapshot_run_writes_both_instruments_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "integration"
            manifest = run_baseline(
                repo_root=REPO_ROOT,
                protocol_path=PROTOCOL_PATH,
                artifact_root=run_dir,
                forecast_limit=2,
            )
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["full_protocol_run"])
            self.assertEqual(manifest["forecast_count_by_instrument"], {"BTC-USD": 2, "ETH-USD": 2})
            self.assertEqual(manifest["fit_failure_count"], 0)
            for instrument in ("BTC-USD", "ETH-USD"):
                for innovation in ("student_t", "normal"):
                    with (run_dir / innovation / instrument / "forecasts.csv").open(
                        encoding="utf-8",
                        newline="",
                    ) as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(rows[0]["training_window_end"], "2022-12-02")
                    self.assertEqual(rows[0]["forecast_date"], "2022-12-03")


if __name__ == "__main__":
    unittest.main()
