"""Kupiec unconditional-coverage test."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import chi2


@dataclass(frozen=True)
class BacktestResult:
    test: str
    n_observations: int
    exceedances: int
    tail_probability: float
    statistic: float | None
    p_value: float | None
    reject_at_5pct: bool | None
    valid: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "test": self.test,
            "n_observations": self.n_observations,
            "exceedances": self.exceedances,
            "tail_probability": self.tail_probability,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "reject_at_5pct": self.reject_at_5pct,
            "valid": self.valid,
            "reason": self.reason,
        }


def _log_bernoulli(count: int, total: int, probability: float) -> float:
    if count < 0 or count > total:
        raise ValueError("count must be between zero and total")
    if not 0 < probability < 1:
        raise ValueError("probability must be strictly between zero and one")
    failures = total - count
    value = 0.0
    if count:
        value += count * math.log(probability)
    if failures:
        value += failures * math.log1p(-probability)
    return value


def kupiec_uc(
    exceedances: list[bool] | tuple[bool, ...],
    *,
    tail_probability: float,
) -> BacktestResult:
    """Run the likelihood-ratio UC test with stable boundary handling."""

    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must be strictly between zero and one")
    n_observations = len(exceedances)
    count = sum(bool(value) for value in exceedances)
    if n_observations == 0:
        return BacktestResult(
            test="kupiec_unconditional_coverage",
            n_observations=0,
            exceedances=0,
            tail_probability=tail_probability,
            statistic=None,
            p_value=None,
            reject_at_5pct=None,
            valid=False,
            reason="no_valid_forecasts",
        )
    observed_probability = count / n_observations
    log_likelihood_null = _log_bernoulli(count, n_observations, tail_probability)
    log_likelihood_alt = (
        _log_bernoulli(count, n_observations, observed_probability)
        if 0 < observed_probability < 1
        else 0.0
    )
    statistic = max(0.0, 2.0 * (log_likelihood_alt - log_likelihood_null))
    p_value = float(chi2.sf(statistic, 1))
    return BacktestResult(
        test="kupiec_unconditional_coverage",
        n_observations=n_observations,
        exceedances=count,
        tail_probability=tail_probability,
        statistic=statistic,
        p_value=p_value,
        reject_at_5pct=p_value < 0.05,
        valid=True,
    )
