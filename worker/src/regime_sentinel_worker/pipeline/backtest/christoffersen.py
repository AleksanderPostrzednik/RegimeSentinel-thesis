"""Christoffersen independence and conditional-coverage tests."""

from __future__ import annotations

import math

from scipy.stats import chi2

from .kupiec import BacktestResult


def _log_term(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability == 1:
        return 0.0
    if probability <= 0 or probability > 1:
        return -math.inf
    return count * math.log(probability)


def _invalid(test: str, tail_probability: float) -> BacktestResult:
    return BacktestResult(
        test=test,
        n_observations=0,
        exceedances=0,
        tail_probability=tail_probability,
        statistic=None,
        p_value=None,
        reject_at_5pct=None,
        valid=False,
        reason="at_least_two_valid_forecasts_required",
    )


def christoffersen_independence(
    exceedances: list[bool] | tuple[bool, ...],
    *,
    tail_probability: float,
) -> BacktestResult:
    """Test whether exceedances are independent first-order Bernoulli events."""

    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must be strictly between zero and one")
    n = len(exceedances)
    if n < 2:
        return _invalid("christoffersen_independence", tail_probability)

    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(exceedances, exceedances[1:]):
        if not previous and not current:
            n00 += 1
        elif not previous and current:
            n01 += 1
        elif previous and not current:
            n10 += 1
        else:
            n11 += 1

    total_transitions = n00 + n01 + n10 + n11
    total_exceedances = n01 + n11
    pi = total_exceedances / total_transitions
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    log_likelihood_independent = _log_term(total_exceedances, pi) + _log_term(
        total_transitions - total_exceedances,
        1.0 - pi,
    )
    log_likelihood_dependent = (
        _log_term(n01, pi0)
        + _log_term(n00, 1.0 - pi0)
        + _log_term(n11, pi1)
        + _log_term(n10, 1.0 - pi1)
    )
    if math.isinf(log_likelihood_independent) and math.isinf(log_likelihood_dependent):
        statistic = 0.0
    else:
        statistic = max(0.0, 2.0 * (log_likelihood_dependent - log_likelihood_independent))
    p_value = float(chi2.sf(statistic, 1))
    return BacktestResult(
        test="christoffersen_independence",
        n_observations=n,
        exceedances=sum(bool(value) for value in exceedances),
        tail_probability=tail_probability,
        statistic=statistic,
        p_value=p_value,
        reject_at_5pct=p_value < 0.05,
        valid=True,
    )


def christoffersen_conditional_coverage(
    exceedances: list[bool] | tuple[bool, ...],
    *,
    tail_probability: float,
) -> BacktestResult:
    """Combine Kupiec UC and Christoffersen independence (two degrees of freedom)."""

    from .kupiec import kupiec_uc

    uc = kupiec_uc(exceedances, tail_probability=tail_probability)
    independence = christoffersen_independence(
        exceedances,
        tail_probability=tail_probability,
    )
    if not uc.valid or not independence.valid:
        return BacktestResult(
            test="christoffersen_conditional_coverage",
            n_observations=len(exceedances),
            exceedances=sum(bool(value) for value in exceedances),
            tail_probability=tail_probability,
            statistic=None,
            p_value=None,
            reject_at_5pct=None,
            valid=False,
            reason="insufficient_valid_forecasts",
        )
    statistic = (uc.statistic or 0.0) + (independence.statistic or 0.0)
    p_value = float(chi2.sf(statistic, 2))
    return BacktestResult(
        test="christoffersen_conditional_coverage",
        n_observations=len(exceedances),
        exceedances=sum(bool(value) for value in exceedances),
        tail_probability=tail_probability,
        statistic=statistic,
        p_value=p_value,
        reject_at_5pct=p_value < 0.05,
        valid=True,
    )
