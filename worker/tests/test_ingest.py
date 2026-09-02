from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from regime_sentinel_worker.pipeline.ingest import (
    SNAPSHOT_COLUMNS,
    SnapshotValidationError,
    build_input_manifest,
    load_frozen_snapshot,
    load_price_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "data/snapshots/btc-eth-daily-close-2021-07-20_2026-07-19.json"
PROTOCOL_PATH = REPO_ROOT / "protocol/thesis-v1.json"
MANIFEST_PATH = REPO_ROOT / "protocol/input-manifest.json"


class IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot_payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    def write_snapshot(self, payload: dict) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "snapshot.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def assert_snapshot_rejected(self, payload: dict, message: str) -> None:
        with self.assertRaises(SnapshotValidationError) as raised:
            load_price_snapshot(self.write_snapshot(payload))
        self.assertIn(message, str(raised.exception))

    def test_loads_frozen_snapshot_without_network(self) -> None:
        snapshot = load_frozen_snapshot(PROTOCOL_PATH, repo_root=REPO_ROOT)

        self.assertEqual(snapshot.snapshot_id, "yahoo-btc-eth-daily-close-2021-07-20_2026-07-19")
        self.assertEqual(snapshot.row_count, 1826)
        self.assertEqual(snapshot.observed_start, "2021-07-20")
        self.assertEqual(snapshot.observed_end, "2026-07-19")
        self.assertEqual(set(snapshot.prices), {"BTC-USD", "ETH-USD"})
        self.assertEqual(len(snapshot.prices["BTC-USD"]), 1826)
        self.assertEqual(len(snapshot.prices["ETH-USD"]), 1826)

    def test_manifest_contains_hashes_and_both_observation_counts(self) -> None:
        snapshot = load_frozen_snapshot(PROTOCOL_PATH, repo_root=REPO_ROOT)
        manifest = build_input_manifest(
            snapshot,
            snapshot_path="data/snapshots/btc-eth-daily-close-2021-07-20_2026-07-19.json",
        )

        manifest_dict = manifest.to_dict()
        self.assertEqual(manifest_dict["snapshot_file_sha256"], snapshot.file_sha256)
        self.assertEqual(manifest_dict["snapshot_content_sha256"], snapshot.content_sha256)
        self.assertEqual(manifest_dict["price_observations_per_instrument"], 1826)
        self.assertEqual(manifest_dict["log_return_observations_per_instrument"], 1825)
        self.assertEqual(
            json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
            manifest_dict,
        )

    def test_rejects_duplicate_date(self) -> None:
        payload = copy.deepcopy(self.snapshot_payload)
        payload["rows"][1][0] = payload["rows"][0][0]
        self.assert_snapshot_rejected(payload, "snapshot contains duplicate date 2021-07-20")

    def test_rejects_missing_daily_date(self) -> None:
        payload = copy.deepcopy(self.snapshot_payload)
        payload["rows"][1][0] = "2021-07-22"
        self.assert_snapshot_rejected(payload, "snapshot UTC calendar is not daily")

    def test_rejects_invalid_date(self) -> None:
        payload = copy.deepcopy(self.snapshot_payload)
        payload["rows"][0][0] = "2021-02-29"
        self.assert_snapshot_rejected(payload, "not a real ISO date")

    def test_rejects_missing_shared_price(self) -> None:
        payload = copy.deepcopy(self.snapshot_payload)
        payload["rows"][0][2] = None
        self.assert_snapshot_rejected(payload, "invalid ETH-USD close")

    def test_rejects_non_positive_price(self) -> None:
        payload = copy.deepcopy(self.snapshot_payload)
        payload["rows"][0][1] = 0
        self.assert_snapshot_rejected(payload, "invalid BTC-USD close")

    def test_rejects_non_canonical_columns(self) -> None:
        payload = copy.deepcopy(self.snapshot_payload)
        payload["columns"] = list(reversed(SNAPSHOT_COLUMNS))
        self.assert_snapshot_rejected(payload, "snapshot columns are not canonical")


if __name__ == "__main__":
    unittest.main()
