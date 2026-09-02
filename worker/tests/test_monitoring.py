from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from regime_sentinel_worker.artifacts.io import write_json
from regime_sentinel_worker.monitoring import (
    DEFAULT_PROTOCOL_PATH,
    MonitoringRunError,
    MonitoringValidationError,
    build_monitoring_history,
    build_snapshot_payload,
    run_monitoring,
    validate_monitoring_history,
    validate_monitoring_protocol,
    validate_monitoring_result,
)
from regime_sentinel_worker.pipeline.ingest import load_price_snapshot
from regime_sentinel_worker.pipeline.models.fallback import (
    FallbackFitError,
    MarkovVarianceFit,
    _require_optimizer_convergence,
)
from regime_sentinel_worker.pipeline.models.garch import GarchFit

UTC = timezone.utc
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "monitoring-v1-prices.json"
NOW = datetime(2026, 8, 25, 1, 35, tzinfo=UTC)


def fixture_rows(*, observed_end: date | None = None) -> dict[str, list[tuple[str, float]]]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    end = observed_end or date.fromisoformat(fixture["observedEndUtc"])
    row_count = fixture["rowCount"]
    start = end - timedelta(days=row_count - 1)
    result: dict[str, list[tuple[str, float]]] = {}
    for instrument, settings in fixture["series"].items():
        rows: list[tuple[str, float]] = []
        for index in range(row_count):
            log_price = (
                math.log(settings["startPrice"])
                + settings["dailyLogDrift"] * index
                + settings["cycleAmplitude"]
                * math.sin(2 * math.pi * index / settings["cycleDays"])
            )
            rows.append(((start + timedelta(days=index)).isoformat(), math.exp(log_price)))
        result[instrument] = rows
    return result


def fake_garch(values: object, *, innovation: str) -> GarchFit:
    parameters = {"omega": 0.05, "alpha[1]": 0.08, "beta[1]": 0.88}
    if innovation == "student_t":
        parameters["nu"] = 8.0
    return GarchFit(
        innovation=innovation,
        sigma_percent=2.4 if innovation == "student_t" else 2.2,
        parameters=parameters,
        log_likelihood=-700.0,
        aic=1408.0,
        bic=1425.0,
        convergence_flag=0,
    )


def fake_regime(values: object) -> MarkovVarianceFit:
    return MarkovVarianceFit(
        variances=(1.2, 8.5),
        transition_matrix=((0.96, 0.04), (0.12, 0.88)),
        filtered_last=(0.28, 0.72),
        occupancy=(0.61, 0.39),
        log_likelihood=-680.0,
        parameters={"variance[1]": 1.2, "variance[2]": 8.5},
    )


def write_fixture_snapshot(path: Path, *, observed_end: date = date(2026, 8, 24)) -> None:
    rows = fixture_rows(observed_end=observed_end)
    payload = build_snapshot_payload(
        rows_by_instrument=rows,
        retrieved_at_utc=NOW,
        requested_start=date.fromisoformat(rows["BTC-USD"][0][0]),
        requested_end=observed_end,
    )
    write_json(path, payload)


class MonitoringProtocolTests(unittest.TestCase):
    def test_protocol_is_operational_and_never_attempts_true_msgarch(self) -> None:
        protocol = validate_monitoring_protocol(DEFAULT_PROTOCOL_PATH)

        self.assertEqual(protocol["protocolId"], "monitoring-v1")
        self.assertEqual(protocol["mode"], "operational")
        self.assertFalse(protocol["estimation"]["regime"]["trueMsgarchAttempted"])
        self.assertEqual(
            protocol["estimation"]["regime"]["modelLabel"], "fallback_not_ms_garch"
        )

    def test_price_validation_rejects_duplicates_gaps_and_nonpositive_values(self) -> None:
        for mutation in ("duplicate", "gap", "nonpositive"):
            with self.subTest(mutation=mutation):
                rows = fixture_rows()
                if mutation == "duplicate":
                    rows["BTC-USD"].insert(10, rows["BTC-USD"][10])
                elif mutation == "gap":
                    rows["ETH-USD"].pop(10)
                else:
                    day, _ = rows["BTC-USD"][10]
                    rows["BTC-USD"][10] = (day, 0.0)
                with self.assertRaises(MonitoringValidationError):
                    build_snapshot_payload(
                        rows_by_instrument=rows,
                        retrieved_at_utc=NOW,
                        requested_start=date(2025, 4, 11),
                        requested_end=date(2026, 8, 24),
                    )


    def test_markov_adapter_rejects_nonconverged_optimizer(self) -> None:
        class FailedResult:
            mle_retvals = {"converged": False}

        class ConvergedResult:
            mle_retvals = {"converged": True}

        with self.assertRaisesRegex(FallbackFitError, "optimizer_nonconvergence"):
            _require_optimizer_convergence(FailedResult())
        _require_optimizer_convergence(ConvergedResult())

class MonitoringRunTests(unittest.TestCase):
    def test_offline_fixture_run_writes_only_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)

            summary = run_monitoring(
                artifact_root=artifact_root,
                now_utc=NOW,
                price_fetcher=lambda instruments, start, end: fixture_rows(),
                garch_fitter=fake_garch,
                regime_fitter=fake_regime,
            )

            latest = json.loads(
                (artifact_root / "latest.json").read_text(encoding="utf-8")
            )
            run_directory = Path(summary["runDirectory"])
            history = json.loads(
                (run_directory / "history.json").read_text(encoding="utf-8")
            )
            snapshot_payload = json.loads(
                (run_directory / "snapshot.json").read_text(encoding="utf-8")
            )
            validate_monitoring_result(latest)
            validate_monitoring_history(history)
            self.assertEqual(summary["historyPath"], str(run_directory / "history.json"))
            self.assertEqual(history["runId"], latest["runId"])
            self.assertEqual(history["historyWindow"]["pointCount"], 180)
            self.assertEqual(history["historyWindow"]["endDateUtc"], "2026-08-24")
            btc_points = history["instruments"][0]["points"]
            source_dates = [row[0] for row in snapshot_payload["rows"]]
            self.assertEqual(
                [point["dateUtc"] for point in btc_points],
                source_dates[-180:],
            )
            self.assertEqual(
                [index for index, point in enumerate(btc_points) if point["isLatest"]],
                [len(btc_points) - 1],
            )
            btc_prices = [row[1] for row in snapshot_payload["rows"]]
            returns_30 = [
                100 * math.log(btc_prices[index] / btc_prices[index - 1])
                for index in range(len(btc_prices) - 30, len(btc_prices))
            ]
            mean_30 = sum(returns_30) / len(returns_30)
            expected_30 = math.sqrt(
                sum((value - mean_30) ** 2 for value in returns_30)
                / len(returns_30)
            )
            self.assertAlmostEqual(
                btc_points[-1]["realizedVolatility30dPercent"], expected_30
            )
            self.assertEqual(summary["observedEndUtc"], "2026-08-24")
            self.assertEqual(latest["freshness"]["status"], "fresh")
            self.assertEqual(latest["freshness"]["ageDays"], 0)
            self.assertAlmostEqual(latest["freshness"]["ageHours"], 1.583333, places=6)
            self.assertEqual(
                latest["freshness"]["observationCompletedAtUtc"],
                "2026-08-25T00:00:00Z",
            )
            self.assertEqual(
                latest["freshness"]["staleAfterUtc"], "2026-08-27T00:00:00Z"
            )
            self.assertFalse(latest["modelPolicy"]["trueMsgarchAttempted"])
            self.assertEqual(
                [item["instrument"] for item in latest["instruments"]],
                ["BTC-USD", "ETH-USD"],
            )
            for instrument in latest["instruments"]:
                self.assertEqual(instrument["modelLabel"], "fallback_not_ms_garch")
                self.assertEqual(instrument["fitStatus"], "complete")
                self.assertEqual(set(instrument["risk"]["levels"]), {"0.95", "0.99"})
                self.assertEqual(instrument["regime"]["probabilityType"], "filtered")
            self.assertTrue((run_directory / "snapshot.json").is_file())
            self.assertTrue((run_directory / "result.json").is_file())
            self.assertTrue((run_directory / "history.json").is_file())
            self.assertTrue((run_directory / "manifest.json").is_file())

            invalid_history = json.loads(json.dumps(history))
            invalid_history["instruments"][0]["points"][-1]["dateUtc"] = (
                invalid_history["instruments"][0]["points"][-2]["dateUtc"]
            )
            with self.assertRaisesRegex(MonitoringValidationError, "duplicate dates"):
                validate_monitoring_history(invalid_history)

    def test_single_price_history_uses_null_instead_of_fabricated_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot_path = Path(directory) / "single-price.json"
            payload = build_snapshot_payload(
                rows_by_instrument={
                    "BTC-USD": [("2026-08-24", 100_000.0)],
                    "ETH-USD": [("2026-08-24", 4_000.0)],
                },
                retrieved_at_utc=NOW,
                requested_start=date(2026, 8, 24),
                requested_end=date(2026, 8, 24),
            )
            write_json(snapshot_path, payload)
            snapshot = load_price_snapshot(snapshot_path)
            history = build_monitoring_history(
                snapshot=snapshot,
                protocol=validate_monitoring_protocol(DEFAULT_PROTOCOL_PATH),
                run_id="monitoring-v1-20260825T013500Z",
                generated_at_utc=NOW,
                provenance={
                    "dataKind": "real",
                    "provider": "Yahoo Finance",
                    "accessMethod": "yfinance 0.2.65",
                    "sourceUrls": ["https://example.test/btc", "https://example.test/eth"],
                    "priceField": "unadjusted_close",
                    "retrievedAtUtc": "2026-08-25T01:35:00Z",
                    "snapshotId": snapshot.snapshot_id,
                    "snapshotFileSha256": snapshot.file_sha256,
                    "snapshotContentSha256": snapshot.content_sha256,
                    "protocolSha256": "a" * 64,
                    "codeCommit": "test",
                },
            )

            validate_monitoring_history(history)
            self.assertEqual(history["historyWindow"]["pointCount"], 1)
            for instrument in history["instruments"]:
                point = instrument["points"][0]
                self.assertIsNone(point["dailyReturnPercent"])
                self.assertIsNone(point["realizedVolatility7dPercent"])
                self.assertIsNone(point["realizedVolatility30dPercent"])
                self.assertTrue(point["isLatest"])


    def test_one_day_provider_delay_is_accepted_with_real_age(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            delayed_now = datetime(2026, 8, 26, 1, 35, tzinfo=UTC)

            summary = run_monitoring(
                artifact_root=artifact_root,
                now_utc=delayed_now,
                price_fetcher=lambda instruments, start, end: fixture_rows(),
                garch_fitter=fake_garch,
                regime_fitter=fake_regime,
            )

            latest = json.loads(
                (artifact_root / "latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["observedEndUtc"], "2026-08-24")
            self.assertEqual(latest["freshness"]["lastCompleteDayUtc"], "2026-08-25")
            self.assertEqual(latest["freshness"]["observationDateUtc"], "2026-08-24")
            self.assertEqual(latest["freshness"]["ageDays"], 1)
            self.assertAlmostEqual(latest["freshness"]["ageHours"], 25.583333, places=6)
            self.assertEqual(
                latest["freshness"]["staleAfterUtc"], "2026-08-27T00:00:00Z"
            )

    def test_snapshot_over_freshness_limit_keeps_previous_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "stale-snapshot.json"
            artifact_root = root / "artifacts"
            write_fixture_snapshot(snapshot_path, observed_end=date(2026, 8, 24))
            artifact_root.mkdir()
            previous = {"sentinel": "previous-complete-result"}
            write_json(artifact_root / "latest.json", previous)

            with self.assertRaisesRegex(
                MonitoringValidationError,
                "Snapshot exceeds freshness limit.*observedEndUtc=2026-08-24.*ageDays=2",
            ):
                run_monitoring(
                    artifact_root=artifact_root,
                    now_utc=datetime(2026, 8, 27, 0, 0, 1, tzinfo=UTC),
                    snapshot_path=snapshot_path,
                    garch_fitter=fake_garch,
                    regime_fitter=fake_regime,
                )

            self.assertEqual(
                json.loads((artifact_root / "latest.json").read_text(encoding="utf-8")),
                previous,
            )

    def test_different_provider_end_dates_use_latest_shared_complete_day(self) -> None:
        rows = fixture_rows()
        last_price = rows["BTC-USD"][-1][1]
        rows["BTC-USD"].append(("2026-08-25", last_price * 1.001))

        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            summary = run_monitoring(
                artifact_root=artifact_root,
                now_utc=datetime(2026, 8, 26, 1, 35, tzinfo=UTC),
                price_fetcher=lambda instruments, start, end: rows,
                garch_fitter=fake_garch,
                regime_fitter=fake_regime,
            )

            latest = json.loads(
                (artifact_root / "latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["observedEndUtc"], "2026-08-24")
            self.assertEqual(latest["dataWindow"]["rowCount"], 501)
            self.assertEqual(
                {item["observationDateUtc"] for item in latest["instruments"]},
                {"2026-08-24"},
            )

    def test_fit_failure_keeps_previous_latest(self) -> None:
        def failed_garch(values: object, *, innovation: str) -> GarchFit:
            raise RuntimeError("fixture fit failure")

        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            previous = {"sentinel": "previous-complete-result"}
            write_json(artifact_root / "latest.json", previous)

            with self.assertRaisesRegex(MonitoringRunError, "fit failed"):
                run_monitoring(
                    artifact_root=artifact_root,
                    now_utc=NOW,
                    price_fetcher=lambda instruments, start, end: fixture_rows(),
                    garch_fitter=failed_garch,
                    regime_fitter=fake_regime,
                )

            self.assertEqual(
                json.loads((artifact_root / "latest.json").read_text(encoding="utf-8")),
                previous,
            )


    def test_regime_fit_failure_keeps_previous_latest(self) -> None:
        def failed_regime(values: object) -> MarkovVarianceFit:
            raise FallbackFitError("optimizer_nonconvergence")

        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            previous = {"sentinel": "previous-complete-result"}
            write_json(artifact_root / "latest.json", previous)

            with self.assertRaisesRegex(
                MonitoringRunError,
                "markov-variance-2state fit failed: optimizer_nonconvergence",
            ):
                run_monitoring(
                    artifact_root=artifact_root,
                    now_utc=NOW,
                    price_fetcher=lambda instruments, start, end: fixture_rows(),
                    garch_fitter=fake_garch,
                    regime_fitter=failed_regime,
                )

            self.assertEqual(
                json.loads((artifact_root / "latest.json").read_text(encoding="utf-8")),
                previous,
            )
    def test_saved_snapshot_can_be_reproduced_without_network(self) -> None:
        def forbidden_network(
            instruments: object, requested_start: object, requested_end: object
        ) -> dict[str, list[tuple[str, float]]]:
            raise AssertionError("network must not be used during reproduction")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "snapshot.json"
            write_fixture_snapshot(snapshot_path)

            summary = run_monitoring(
                artifact_root=root / "reproduced",
                now_utc=NOW,
                snapshot_path=snapshot_path,
                price_fetcher=forbidden_network,
                garch_fitter=fake_garch,
                regime_fitter=fake_regime,
            )

            self.assertEqual(summary["observedStartUtc"], "2025-04-11")
            self.assertEqual(summary["observedEndUtc"], "2026-08-24")


if __name__ == "__main__":
    unittest.main()
