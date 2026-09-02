from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from regime_sentinel_worker.experiment_protocol import (
    ProtocolValidationError,
    validate_protocol,
    validate_protocol_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "protocol" / "thesis-v1.json"


class ExperimentProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def assert_invalid(self, protocol: dict, expected_message: str) -> None:
        with self.assertRaises(ProtocolValidationError) as raised:
            validate_protocol(protocol, REPO_ROOT)
        self.assertIn(expected_message, str(raised.exception))

    def test_frozen_protocol_is_valid(self) -> None:
        validate_protocol_file(PROTOCOL_PATH, REPO_ROOT)

    def test_rejects_snapshot_checksum_drift(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["data"]["snapshot"]["fileSha256"] = "0" * 64
        self.assert_invalid(protocol, "snapshot fileSha256 mismatch")

    def test_rejects_unknown_nested_protocol_field(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["informationSet"]["futureLeakage"] = False
        self.assert_invalid(
            protocol,
            "schema validation at $.informationSet: unexpected properties: ['futureLeakage']",
        )

    def test_rejects_missing_nested_protocol_field(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        del protocol["risk"]["esConvention"]
        self.assert_invalid(
            protocol,
            "schema validation at $.risk: missing required properties: ['esConvention']",
        )

    def test_rejects_smoothed_probability_in_oos_forecast(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["informationSet"]["stateProbabilityForForecast"] = "smoothed"
        protocol["models"]["regimeCandidate"]["stateInference"]["forecast"] = "smoothed"
        self.assert_invalid(
            protocol,
            "schema validation at $.informationSet.stateProbabilityForForecast: "
            "expected constant 'filtered'",
        )

    def test_rejects_window_change_without_new_protocol(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["informationSet"]["estimationWindowReturns"] = 501
        self.assert_invalid(protocol, "primary estimation window must remain frozen at 500 returns")

    def test_rejects_price_imputation(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["data"]["missingDataPolicy"] = "forward_fill"
        self.assert_invalid(
            protocol,
            "schema validation at $.data.missingDataPolicy: "
            "expected constant 'fail_no_imputation'",
        )

    def test_rejects_calling_em_the_msgarch_fitml_estimator(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["models"]["regimeCandidate"]["estimationMethod"] = "EM"
        self.assert_invalid(
            protocol,
            "schema validation at $.models.regimeCandidate.estimationMethod: "
            "expected constant 'maximum_likelihood'",
        )

    def test_rejects_mislabeling_fallback_as_ms_garch(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["models"]["fallback"]["label"] = "MS-GARCH"
        self.assert_invalid(
            protocol,
            "schema validation at $.models.fallback.label: "
            "expected constant 'fallback_not_ms_garch'",
        )

    def test_rejects_invented_es_pass_fail_test(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["evaluation"]["esStandalonePassFailTest"] = "custom_green_red_test"
        self.assert_invalid(
            protocol,
            "schema validation at $.evaluation.esStandalonePassFailTest: "
            "expected constant 'none'",
        )

    def test_rejects_model_selection_by_aic_bic_alone(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["evaluation"]["modelComparisonRule"]["aicBicOnlyIsSufficient"] = True
        self.assert_invalid(
            protocol,
            "schema validation at "
            "$.evaluation.modelComparisonRule.aicBicOnlyIsSufficient: "
            "expected constant False",
        )


if __name__ == "__main__":
    unittest.main()
