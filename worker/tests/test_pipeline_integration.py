from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from regime_sentinel_worker.pipeline.ingest import (
    build_input_manifest,
    load_frozen_snapshot,
    write_input_manifest,
)
from regime_sentinel_worker.pipeline.preprocess import build_log_returns


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "protocol/thesis-v1.json"


class PipelineIntegrationTests(unittest.TestCase):
    def test_snapshot_to_returns_and_manifest_is_deterministic(self) -> None:
        snapshot = load_frozen_snapshot(PROTOCOL_PATH, repo_root=REPO_ROOT)
        returns = build_log_returns(snapshot)
        manifest = build_input_manifest(
            snapshot,
            snapshot_path="data/snapshots/btc-eth-daily-close-2021-07-20_2026-07-19.json",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "input-manifest.json"
            write_input_manifest(manifest, manifest_path)
            written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(written_manifest, manifest.to_dict())
        self.assertEqual(written_manifest["price_observations_per_instrument"], 1826)
        self.assertEqual(written_manifest["log_return_observations_per_instrument"], 1825)
        self.assertEqual(len(returns["BTC-USD"].model_returns_percent), 1825)
        self.assertEqual(len(returns["ETH-USD"].model_returns_percent), 1825)
        self.assertEqual(returns["BTC-USD"].dates, returns["ETH-USD"].dates)


if __name__ == "__main__":
    unittest.main()
