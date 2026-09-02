from __future__ import annotations

import math
import unittest

from regime_sentinel_worker.pipeline.risk.var_es import (
    fz0_score,
    is_exceedance,
    parametric_var_es,
    quantile_loss,
    standardized_innovation_tail,
)


class RiskMetricTests(unittest.TestCase):
    def test_normal_tail_and_positive_loss_var_es(self) -> None:
        quantile, conditional_mean = standardized_innovation_tail(
            innovation="normal",
            confidence=0.95,
        )
        self.assertAlmostEqual(quantile, -1.644853626951, places=10)
        self.assertAlmostEqual(conditional_mean, -2.0627128075, places=9)

        risk = parametric_var_es(
            forecast_mean_percent=0.0,
            sigma_percent=1.0,
            confidence=0.95,
            innovation="normal",
        )
        self.assertAlmostEqual(risk.var, 1.644853626951, places=10)
        self.assertAlmostEqual(risk.es, 2.0627128075, places=9)
        self.assertGreater(risk.es, risk.var)

    def test_student_t_tail_is_heavier_than_normal_at_99_percent(self) -> None:
        normal = parametric_var_es(
            forecast_mean_percent=0.0,
            sigma_percent=1.0,
            confidence=0.99,
            innovation="normal",
        )
        student_t = parametric_var_es(
            forecast_mean_percent=0.0,
            sigma_percent=1.0,
            confidence=0.99,
            innovation="student_t",
            student_t_df=5.0,
        )
        self.assertGreater(student_t.var, normal.var)
        self.assertGreater(student_t.es, student_t.var)

    def test_exceedance_is_strict(self) -> None:
        self.assertFalse(is_exceedance(2.0, 2.0))
        self.assertTrue(is_exceedance(2.000001, 2.0))

    def test_quantile_loss_and_fz0_use_loss_convention(self) -> None:
        self.assertAlmostEqual(quantile_loss(3.0, 2.0, 0.95), 0.95)
        self.assertAlmostEqual(quantile_loss(1.0, 2.0, 0.95), 0.05)
        expected = 1.0 / (0.05 * 4.0) + 2.0 / 4.0 + math.log(4.0) - 1.0
        self.assertAlmostEqual(fz0_score(3.0, 2.0, 4.0, 0.95), expected)

    def test_invalid_student_t_df_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            standardized_innovation_tail(
                innovation="student_t",
                confidence=0.95,
                student_t_df=2.0,
            )


if __name__ == "__main__":
    unittest.main()
