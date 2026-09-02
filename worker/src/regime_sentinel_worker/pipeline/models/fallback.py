"""Protocol fallback: two-state Markov-switching variance via statsmodels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

from regime_sentinel_worker.pipeline.risk.var_es import VaREsForecast


class FallbackFitError(RuntimeError):
    """Raised when the explicit Markov-variance fallback cannot fit."""


@dataclass(frozen=True)
class MarkovVarianceFit:
    variances: tuple[float, float]
    transition_matrix: tuple[tuple[float, float], tuple[float, float]]
    filtered_last: tuple[float, float]
    occupancy: tuple[float, float]
    log_likelihood: float
    parameters: dict[str, float]


def _require_optimizer_convergence(result: object) -> None:
    retvals = getattr(result, "mle_retvals", None)
    converged = retvals.get("converged") if hasattr(retvals, "get") else None
    if converged is not True:
        raise FallbackFitError(
            f"optimizer_nonconvergence: mle_retvals={retvals!r}"
        )


def fit_markov_variance(values: Sequence[float]) -> MarkovVarianceFit:
    if len(values) != 500:
        raise ValueError("fallback windows must contain exactly 500 returns")
    if any(not math.isfinite(float(value)) for value in values):
        raise FallbackFitError("fallback window contains a non-finite return")
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError as exc:
        raise FallbackFitError("dependency_unavailable: statsmodels is required by fallback") from exc
    try:
        model = MarkovRegression(
            endog=list(float(value) for value in values),
            k_regimes=2,
            trend="n",
            switching_variance=True,
        )
        result = model.fit(disp=False)
    except Exception as exc:
        raise FallbackFitError(f"fit_exception: {type(exc).__name__}: {exc}") from exc

    _require_optimizer_convergence(result)

    names = list(getattr(result.model, "param_names", []))
    params = [float(value) for value in result.params]
    if len(names) != len(params):
        raise FallbackFitError("statsmodels parameter names and values have different lengths")
    parameter_dict = dict(zip(names, params))
    variance_items = [
        (name, value)
        for name, value in parameter_dict.items()
        if "sigma2" in name.lower() or "variance" in name.lower()
    ]
    if len(variance_items) != 2:
        raise FallbackFitError(f"cannot identify two state variances from parameters: {names}")
    variances = tuple(float(value) for _, value in variance_items)
    if any(value <= 0 or not math.isfinite(value) for value in variances):
        raise FallbackFitError(f"invalid state variances: {variances}")

    transition = getattr(result, "regime_transition", None)
    if transition is None:
        raise FallbackFitError("statsmodels result omitted regime transition matrix")
    if getattr(transition, "ndim", 0) == 3:
        transition = transition[:, :, -1]
    raw_transition = tuple(
        tuple(float(transition[row, column]) for column in range(2))
        for row in range(2)
    )
    if all(abs(sum(row) - 1.0) > 1e-8 for row in raw_transition) and all(
        abs(sum(raw_transition[row][column] for row in range(2)) - 1.0) <= 1e-8
        for column in range(2)
    ):
        transition_matrix = tuple(tuple(raw_transition[row][column] for row in range(2)) for column in range(2))
    else:
        transition_matrix = raw_transition
    for row in transition_matrix:
        if any(value < 0 or value > 1 or not math.isfinite(value) for value in row):
            raise FallbackFitError(f"invalid transition matrix: {transition_matrix}")
        if abs(sum(row) - 1.0) > 1e-8:
            raise FallbackFitError(f"transition row does not sum to one: {transition_matrix}")

    filtered = getattr(result, "filtered_marginal_probabilities", None)
    if filtered is None:
        raise FallbackFitError("statsmodels result omitted filtered probabilities")
    if hasattr(filtered, "to_numpy"):
        filtered_values = filtered.to_numpy(dtype=float)
    else:
        filtered_values = filtered
    filtered_last = tuple(float(value) for value in filtered_values[-1])
    occupancy = tuple(float(sum(float(row[state]) for row in filtered_values) / len(filtered_values)) for state in range(2))
    if any(value < 0 or value > 1 or not math.isfinite(value) for value in filtered_last + occupancy):
        raise FallbackFitError("invalid filtered state probabilities")

    return MarkovVarianceFit(
        variances=(variances[0], variances[1]),
        transition_matrix=transition_matrix,
        filtered_last=filtered_last,
        occupancy=occupancy,
        log_likelihood=float(result.llf),
        parameters=parameter_dict,
    )


def normal_mixture_risk(
    *,
    fit: MarkovVarianceFit,
    mean_percent: float,
    confidence: float,
) -> VaREsForecast:
    """One-step filtered predictive normal-variance mixture risk."""

    probabilities = [
        fit.filtered_last[0] * fit.transition_matrix[0][0]
        + fit.filtered_last[1] * fit.transition_matrix[1][0],
        fit.filtered_last[0] * fit.transition_matrix[0][1]
        + fit.filtered_last[1] * fit.transition_matrix[1][1],
    ]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    sigmas = [math.sqrt(value) for value in fit.variances]
    normal = NormalDist()
    low, high = 0.0, max(sigmas) * 12.0
    for _ in range(160):
        middle = (low + high) / 2.0
        cdf = sum(probability * normal.cdf(middle / sigma) for probability, sigma in zip(probabilities, sigmas))
        if cdf < confidence:
            low = middle
        else:
            high = middle
    quantile = (low + high) / 2.0
    tail = 1.0 - confidence
    expected_tail = sum(
        probability * sigma * math.exp(-0.5 * (quantile / sigma) ** 2) / math.sqrt(2.0 * math.pi)
        for probability, sigma in zip(probabilities, sigmas)
    )
    return VaREsForecast(confidence, -mean_percent + quantile, -mean_percent + expected_tail / tail)
