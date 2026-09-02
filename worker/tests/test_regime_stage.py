from __future__ import annotations

from dataclasses import replace

import sys
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from regime_sentinel_worker import main as worker_main
from regime_sentinel_worker.pipeline.models.fallback import MarkovVarianceFit
from regime_sentinel_worker.pipeline.models.msgarch import (
    MsgarchFit,
    _parse_stdout,
    repeat_parameter_delta,
    run_preflight,
    validate_msgarch_fit,
)
from regime_sentinel_worker.pipeline.baseline import verify_baseline_artifacts
from regime_sentinel_worker.regime import (
    _center_training_window,
    _parameter_stability,
    _rolling_fallback,
    _rolling_msgarch,
)


def valid_fit(start_index: int, *, delta: float = 0.0, occupancy: tuple[float, float] = (0.6, 0.4)) -> MsgarchFit:
    return MsgarchFit(
        start_index=start_index,
        log_likelihood=-100.0,
        parameters={
            "alpha0_1": 0.1 + delta,
            "alpha1_1": 0.05,
            "beta_1": 0.8,
            "nu_1": 8.0,
            "alpha0_2": 0.5 + delta,
            "alpha1_2": 0.1,
            "beta_2": 0.7,
        },
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        occupancy=occupancy,
        filtered_last=(0.7, 0.3),
        unconditional_volatility=(0.45, 0.95),
        state_order=(1, 2),
        risk={
            "0.95": {"var_percent": 2.0, "es_percent": 2.5},
            "0.99": {"var_percent": 3.0, "es_percent": 3.7},
        },
        stdout="",
        stderr="",
    )


class FakeRunner:
    def __init__(self, factory):
        self.factory = factory
        self.calls = []

    def fit(self, values, *, start_index, mode, par0=None, log_path=None):
        self.calls.append((len(values), start_index, mode, log_path))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("fake fit log\n", encoding="utf-8")
        return self.factory(start_index)


class RegimePreflightTests(unittest.TestCase):
    def test_requires_five_deterministic_starts_and_accepts_stable_fit(self):
        runner = FakeRunner(lambda index: valid_fit(index))
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(
                instrument="BTC-USD",
                values=[0.1] * 500,
                runner=runner,
                log_root=directory,
            )
        self.assertTrue(result.passed)
        self.assertEqual(len(runner.calls), 5)
        self.assertEqual([call[2] for call in runner.calls], ["preflight"] * 5)
        self.assertEqual(result.checks["repeat_parameter_delta"], 0.0)

    def test_rejects_unstable_best_fit_even_when_every_start_succeeds(self):
        runner = FakeRunner(lambda index: valid_fit(index, delta=index * 2e-6))
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(
                instrument="ETH-USD",
                values=[0.1] * 500,
                runner=runner,
                log_root=directory,
            )
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["repeatable_within_tolerance"])
        self.assertGreater(result.checks["repeat_parameter_delta"], 1e-6)

    def test_rejects_state_occupancy_below_protocol_floor(self):
        runner = FakeRunner(lambda index: valid_fit(index, occupancy=(0.96, 0.04)))
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(
                instrument="BTC-USD",
                values=[0.1] * 500,
                runner=runner,
                log_root=directory,
            )
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["occupancy"])

    def test_mixed_preflight_preserves_failed_occupancy_check(self):
        runner = FakeRunner(
            lambda index: (
                valid_fit(index, occupancy=(0.999, 0.001))
                if index == 2
                else valid_fit(index)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(
                instrument="ETH-USD",
                values=[0.1] * 500,
                runner=runner,
                log_root=directory,
            )
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["all_starts_successful_and_valid"])
        self.assertFalse(result.checks["occupancy"])
        self.assertEqual(sum(attempt.success for attempt in result.attempts), 4)

    def test_baseline_gate_fails_closed_when_artifacts_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = verify_baseline_artifacts(directory, expected_forecasts=1325)
        self.assertFalse(result.complete)
        self.assertIn("manifest.json is missing", result.errors[0])


    def test_fallback_writes_protocol_label_and_stability_metrics(self):
        fit = MarkovVarianceFit(
            variances=(1.0, 4.0),
            transition_matrix=((0.8, 0.2), (0.3, 0.7)),
            filtered_last=(0.6, 0.4),
            occupancy=(0.5, 0.5),
            log_likelihood=-10.0,
            parameters={"sigma2[0]": 1.0, "sigma2[1]": 4.0},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failures = []
            with patch(
                "regime_sentinel_worker.regime.fit_markov_variance",
                return_value=fit,
            ):
                summary = _rolling_fallback(
                    instrument="BTC-USD",
                    values=[0.1] * 503,
                    dates=tuple(f"2026-01-{index + 1:02d}" for index in range(503)),
                    window=500,
                    confidence_levels=(0.95, 0.99),
                    root=root,
                    failures=failures,
                )
            forecast_path = root / "fallback_not_ms_garch" / "BTC-USD" / "forecasts.json"
            self.assertEqual(summary["model_label"], "fallback_not_ms_garch")
            self.assertEqual(summary["fit_success_count"], 3)
            self.assertEqual(failures, [])
            self.assertEqual(_parameter_stability(forecast_path)["max_relative_range"], 0.0)

    def test_msgarch_risk_adds_back_the_rolling_forecast_mean(self):
        runner = FakeRunner(lambda index: valid_fit(index))
        with tempfile.TemporaryDirectory() as log_directory:
            preflight = run_preflight(
                instrument="BTC-USD",
                values=[0.1] * 500,
                runner=runner,
                log_root=log_directory,
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                failures = []
                _rolling_msgarch(
                    instrument="BTC-USD",
                    values=[1.0] * 501,
                    dates=tuple(str(index) for index in range(501)),
                    window=500,
                    confidence_levels=(0.95, 0.99),
                    runner=runner,
                    preflight=preflight,
                    root=root,
                    failures=failures,
                )
                rows = json.loads(
                    (
                        root / "MS-GARCH" / "BTC-USD" / "forecasts.json"
                    ).read_text(encoding="utf-8")
                )
        self.assertEqual(failures, [])
        self.assertAlmostEqual(rows[0]["risk"]["0.95"]["var_percent"], 1.0)
        self.assertAlmostEqual(rows[0]["risk"]["0.95"]["es_percent"], 1.5)

    def test_rolling_does_not_apply_preflight_occupancy_floor(self):
        preflight_runner = FakeRunner(lambda index: valid_fit(index))
        rolling_runner = FakeRunner(
            lambda index: valid_fit(index, occupancy=(0.99, 0.01))
        )
        with tempfile.TemporaryDirectory() as log_directory:
            preflight = run_preflight(
                instrument="BTC-USD",
                values=[0.1] * 500,
                runner=preflight_runner,
                log_root=log_directory,
            )
            with tempfile.TemporaryDirectory() as directory:
                failures = []
                summary = _rolling_msgarch(
                    instrument="BTC-USD",
                    values=[1.0] * 501,
                    dates=tuple(str(index) for index in range(501)),
                    window=500,
                    confidence_levels=(0.95, 0.99),
                    runner=rolling_runner,
                    preflight=preflight,
                    root=Path(directory),
                    failures=failures,
                )
        self.assertEqual(summary["fit_success_count"], 1)
        self.assertEqual(failures, [])

    def test_main_resolves_relative_rscript_before_runner_changes_cwd(self):
        relative_rscript = Path("..") / ".runtime" / "msgarch" / "Rscript"
        expected = str(relative_rscript.resolve())
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["worker", "msgarch-preflight", "--rscript", str(relative_rscript), "--artifacts", directory]
        ), patch.object(worker_main, "run_msgarch_preflight_stage", return_value={"ok": True}) as stage:
            worker_main.main()
        self.assertEqual(stage.call_args.kwargs["rscript"], expected)

    def test_center_training_window_matches_protocol(self):
        centered, mean_percent = _center_training_window([1.0, 2.0, 3.0])
        self.assertEqual(mean_percent, 2.0)
        self.assertAlmostEqual(sum(centered), 0.0)

    def test_failed_preflight_serializes_without_infinity(self):
        def fail(_start_index):
            raise RuntimeError("fit failed")

        runner = FakeRunner(fail)
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(
                instrument="BTC-USD",
                values=[0.1] * 500,
                runner=runner,
                log_root=directory,
            )
        self.assertIsNone(result.checks["repeat_parameter_delta"])
        json.dumps(result.as_dict(), allow_nan=False)

    def test_rejects_incomplete_risk_payload(self):
        fit = replace(
            valid_fit(0),
            risk={"0.95": {"var_percent": 2.0, "es_percent": 2.5}},
        )
        self.assertIn(
            "risk output must contain exactly 0.95 and 0.99",
            validate_msgarch_fit(fit),
        )

    def test_rolling_rejects_fit_that_fails_contract_validation(self):
        valid_runner = FakeRunner(lambda index: valid_fit(index))
        invalid_runner = FakeRunner(
            lambda index: replace(
                valid_fit(index),
                risk={"0.95": {"var_percent": 2.0, "es_percent": 2.5}},
            )
        )
        with tempfile.TemporaryDirectory() as log_directory:
            preflight = run_preflight(
                instrument="BTC-USD",
                values=[0.1] * 500,
                runner=valid_runner,
                log_root=log_directory,
            )
            with tempfile.TemporaryDirectory() as directory:
                failures = []
                summary = _rolling_msgarch(
                    instrument="BTC-USD",
                    values=[1.0] * 501,
                    dates=tuple(str(index) for index in range(501)),
                    window=500,
                    confidence_levels=(0.95, 0.99),
                    runner=invalid_runner,
                    preflight=preflight,
                    root=Path(directory),
                    failures=failures,
                )
        self.assertEqual(summary["fit_success_count"], 0)
        self.assertEqual(len(failures), 1)
        self.assertIn("risk output must contain exactly 0.95 and 0.99", failures[0]["error"])

    def test_canonical_parameter_delta_ignores_label_switching(self):
        base = valid_fit(0)
        switched = replace(
            base,
            parameters={
                "alpha0_1": 0.5,
                "alpha1_1": 0.1,
                "beta_1": 0.7,
                "nu_1": 8.0,
                "alpha0_2": 0.1,
                "alpha1_2": 0.05,
                "beta_2": 0.8,
            },
            transition_matrix=((0.8, 0.2), (0.1, 0.9)),
            occupancy=(0.4, 0.6),
            filtered_last=(0.3, 0.7),
            unconditional_volatility=(0.95, 0.45),
            state_order=(2, 1),
        )
        self.assertEqual(repeat_parameter_delta(base, switched), 0.0)

    def test_parser_preserves_optimizer_and_hessian_diagnostics(self):
        stdout = "\n".join(
            (
                "MASTER_SEED\t20260722",
                "EFFECTIVE_SEED\t20260724",
                "START_SOURCE\tdeterministic_grid",
                "START_PARAM\tnu_1\t10",
                "OPTIM_METHOD\tBFGS",
                "OPTIM_DO_PLM\ttrue",
                "OPTIM_DO_SE\tfalse",
                "OPTIM_CONVERGENCE\t0",
                "OPTIM_MESSAGE\t",
                "OPTIM_OBJECTIVE\t10",
                "OPTIM_COUNT\tfunction\t42",
                "OPTIM_START_RECEIVED\t1\tnu_1\t2.5",
                "OPTIM_START_USED\t1\tnu_1\t-2.5",
                "OPTIM_END\t1\tnu_1\t-1.5",
                "HESSIAN_AVAILABLE\ttrue",
                "HESSIAN_DIM\t2",
                "HESSIAN\t1\t1\t2",
                "HESSIAN\t1\t2\t0",
                "HESSIAN\t2\t1\t0",
                "HESSIAN\t2\t2\t1",
                "HESSIAN_EIGEN\t1\t2",
                "HESSIAN_EIGEN\t2\t1",
                "STATUS\tsuccess",
                "LOG_LIK\t-10",
                "PARAM\talpha0_1\t0.1",
                "PARAM\talpha1_1\t0.05",
                "PARAM\tbeta_1\t0.8",
                "PARAM\tnu_1\t8",
                "PARAM\talpha0_2\t0.5",
                "PARAM\talpha1_2\t0.1",
                "PARAM\tbeta_2\t0.7",
                "TRANS\t1\t1\t0.9",
                "TRANS\t1\t2\t0.1",
                "TRANS\t2\t1\t0.2",
                "TRANS\t2\t2\t0.8",
                "OCCUPANCY\t1\t0.6",
                "OCCUPANCY\t2\t0.4",
                "FILTERED_LAST\t1\t0.7",
                "FILTERED_LAST\t2\t0.3",
                "UNC_VOL\t1\t0.45",
                "UNC_VOL\t2\t0.95",
                "STATE_ORDER\t1\t2",
                "RISK\t0.95\t2\t2.5",
                "RISK\t0.99\t3\t3.7",
            )
        )
        fit = _parse_stdout(stdout, start_index=2)
        self.assertEqual(fit.diagnostics["master_seed"], 20260722)
        self.assertEqual(fit.diagnostics["effective_seed"], 20260724)
        optimizer = fit.diagnostics["optimizer"]
        self.assertEqual(optimizer["convergence"], 0)
        self.assertEqual(optimizer["objective"], 10.0)
        self.assertEqual(optimizer["start_used_transformed"][0]["value"], -2.5)
        self.assertEqual(optimizer["hessian"], [[2.0, 0.0], [0.0, 1.0]])
        self.assertEqual(optimizer["hessian_eigenvalues"], [2.0, 1.0])

    def test_r_script_preserves_named_start_parameters(self):
        script = (Path(__file__).parents[1] / "r" / "msgarch_fit.R").read_text(encoding="utf-8")
        self.assertIn("names(start) <- labels", script)
        self.assertIn('start[["P_2_1"]] <- 1 - persistence', script)
        self.assertIn('for (nu_name in c("nu_1", "nu_2"))', script)
        self.assertIn("for.se = TRUE", script)
        self.assertIn('if (identical(mode, "preflight"))', script)
        self.assertIn("fit_control$OptimFUN <- diagnostic_optim", script)
        self.assertIn("ctr = fit_control", script)
        self.assertIn('emit("OPTIM_CONVERGENCE"', script)
        self.assertIn("fit_control <- list(par0 = start", script)




if __name__ == "__main__":
    unittest.main()
