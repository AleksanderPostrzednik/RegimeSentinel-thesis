from __future__ import annotations

import math
import unittest
from pathlib import Path

from regime_sentinel_worker.pipeline.ingest import load_frozen_snapshot
from regime_sentinel_worker.pipeline.preprocess import MODEL_SCALE_FACTOR, build_log_returns


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/thesis-v1/protocol.json"


class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = load_frozen_snapshot(PROTOCOL_PATH, repo_root=REPO_ROOT)
        cls.returns = build_log_returns(cls.snapshot)

    def test_returns_have_expected_count_and_shared_dates(self) -> None:
        self.assertEqual(self.returns["BTC-USD"].count, 1825)
        self.assertEqual(self.returns["ETH-USD"].count, 1825)
        self.assertEqual(self.returns["BTC-USD"].dates[0], "2021-07-21")
        self.assertEqual(self.returns["BTC-USD"].dates[-1], "2026-07-19")
        self.assertEqual(self.returns["BTC-USD"].dates, self.returns["ETH-USD"].dates)

    def test_uses_log_definition_and_model_scale(self) -> None:
        btc_prices = self.snapshot.prices["BTC-USD"]
        btc = self.returns["BTC-USD"]

        expected_first = math.log(btc_prices[1] / btc_prices[0])
        self.assertAlmostEqual(btc.log_returns[0], expected_first, places=15)
        self.assertAlmostEqual(
            btc.model_returns_percent[0], MODEL_SCALE_FACTOR * expected_first, places=15
        )
        self.assertEqual(btc.scale_factor, 100)

    def test_does_not_center_or_change_return_values(self) -> None:
        for series in self.returns.values():
            for unscaled, scaled in zip(series.log_returns, series.model_returns_percent):
                self.assertEqual(scaled, 100 * unscaled)

    def test_rejects_non_protocol_scale(self) -> None:
        with self.assertRaises(ValueError):
            build_log_returns(self.snapshot, scale_factor=10)


if __name__ == "__main__":
    unittest.main()
