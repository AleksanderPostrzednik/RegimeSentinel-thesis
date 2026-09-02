from __future__ import annotations

import math
import unittest

from regime_sentinel_worker.pipeline.backtest.christoffersen import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
)
from regime_sentinel_worker.pipeline.backtest.kupiec import kupiec_uc


class BacktestTests(unittest.TestCase):
    def test_kupiec_handles_zero_exceedances(self) -> None:
        result = kupiec_uc([False] * 100, tail_probability=0.05)
        self.assertTrue(result.valid)
        self.assertEqual(result.exceedances, 0)
        self.assertTrue(math.isfinite(result.statistic or 0.0))
        self.assertGreater(result.statistic or 0.0, 0.0)

    def test_christoffersen_no_exceedances_has_zero_independence_statistic(self) -> None:
        result = christoffersen_independence([False] * 10, tail_probability=0.05)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.statistic or 0.0, 0.0)
        self.assertAlmostEqual(result.p_value or 0.0, 1.0)

    def test_christoffersen_handles_probability_one_transition_terms(self) -> None:
        sequence = [False, True, False, False, True, False]
        result = christoffersen_independence(sequence, tail_probability=0.05)
        self.assertTrue(result.valid)
        self.assertTrue(math.isfinite(result.statistic or 0.0))
        self.assertGreater(result.statistic or 0.0, 0.0)

    def test_conditional_coverage_combines_two_tests(self) -> None:
        sequence = [False, False, True, False, False, True, False, False, False, False]
        result = christoffersen_conditional_coverage(sequence, tail_probability=0.2)
        independence = christoffersen_independence(sequence, tail_probability=0.2)
        self.assertTrue(result.valid)
        self.assertGreaterEqual(result.statistic or 0.0, independence.statistic or 0.0)

    def test_empty_sequences_are_reported_as_invalid(self) -> None:
        result = kupiec_uc([], tail_probability=0.01)
        self.assertFalse(result.valid)
        conditional = christoffersen_conditional_coverage([], tail_probability=0.01)
        self.assertFalse(conditional.valid)


if __name__ == "__main__":
    unittest.main()
