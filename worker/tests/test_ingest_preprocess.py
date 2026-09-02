from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from regime_sentinel_worker.pipeline.ingest import (
    PriceSnapshot,
    SnapshotValidationError,
    load_price_snapshot,
)
from regime_sentinel_worker.pipeline.preprocess import build_log_returns


class IngestPreprocessTests(unittest.TestCase):
    def test_log_returns_are_scaled_and_keep_return_dates(self) -> None:
        snapshot = PriceSnapshot(
            path=Path("synthetic.json"),
            snapshot_id="synthetic",
            dates=("2024-01-01", "2024-01-02", "2024-01-03"),
            prices={
                "BTC-USD": (100.0, 110.0, 99.0),
                "ETH-USD": (50.0, 50.0, 55.0),
            },
            file_sha256="file",
            content_sha256="content",
            provenance={},
        )
        series = build_log_returns(snapshot)["BTC-USD"]
        self.assertEqual(series.dates, ("2024-01-02", "2024-01-03"))
        self.assertAlmostEqual(series.log_returns[0], 0.1 * 0 + 0.0953101798, places=8)
        self.assertAlmostEqual(series.model_returns_percent[0], 9.53101798, places=6)

    def test_loader_rejects_duplicate_dates(self) -> None:
        payload = {
            "snapshotId": "synthetic",
            "columns": ["date_utc", "BTC-USD", "ETH-USD"],
            "quality": {
                "rowCount": 2,
                "missingSharedCloseCount": 0,
                "contentSha256": "not-used",
            },
            "rows": [
                ["2024-01-01", 100.0, 50.0],
                ["2024-01-01", 101.0, 51.0],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SnapshotValidationError):
                load_price_snapshot(path)


if __name__ == "__main__":
    unittest.main()
