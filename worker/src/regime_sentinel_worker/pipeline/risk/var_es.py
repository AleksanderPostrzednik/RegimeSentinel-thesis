"""Parametric positive-loss VaR, ES and forecast scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from scipy.stats import norm, t
except ImportError:
    norm = None
    t = None


@dataclass(frozen=True)
class VaREsForecast:
    confidence: float
    var: float
    es: float


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")


def standardized_innovation_tail(
    *,
    innovation: str,
    confidence: float,
    student_t_df: float | None = None,
) -> tuple[float, float]:
    """Return lower-tail quantile and conditional mean of a unit-variance innovation."""

    _validate_confidence(confidence)
    tail_probability = 1.0 - confidence
    if innovation == "normal":
        if norm is None:
            raise RuntimeError("dependency_unavailable: scipy is required for parametric risk")
        quantile = float(norm.ppf(tail_probability))
        conditional_mean = float(-norm.pdf(quantile) / tail_probability)
    elif innovation == "student_t":
        if t is None:
            raise RuntimeError("dependency_unavailable: scipy is required for Student-t risk")
        if student_t_df is None or student_t_df <= 2:
            raise ValueError("Student-t df must be greater than 2")
        raw_quantile = float(t.ppf(tail_probability, student_t_df))
        standardization = math.sqrt((student_t_df - 2.0) / student_t_df)
        quantile = standardization * raw_quantile
        conditional_raw_mean = -(
            (student_t_df + raw_quantile**2)
            / (student_t_df - 1.0)
            * t.pdf(raw_quantile, student_t_df)
            / tail_probability
        )
        conditional_mean = standardization * conditional_raw_mean
    else:
        raise ValueError(f"unsupported innovation: {innovation}")

    if not math.isfinite(quantile) or not math.isfinite(conditional_mean):
        raise ValueError("innovation tail returned a non-finite value")
    return quantile, conditional_mean


def parametric_var_es(
    *,
    forecast_mean_percent: float,
    sigma_percent: float,
    confidence: float,
    innovation: str,
    student_t_df: float | None = None,
) -> VaREsForecast:
    """Convert a one-step return forecast into positive-loss VaR and ES."""

    if sigma_percent <= 0 or not math.isfinite(sigma_percent):
        raise ValueError("sigma_percent must be finite and positive")
    quantile, conditional_mean = standardized_innovation_tail(
        innovation=innovation,
        confidence=confidence,
        student_t_df=student_t_df,
    )
    var = -(forecast_mean_percent + sigma_percent * quantile)
    es = -(forecast_mean_percent + sigma_percent * conditional_mean)
    return VaREsForecast(confidence=confidence, var=var, es=es)


def is_exceedance(loss_percent: float, var_percent: float) -> bool:
    """Protocol rule: equality is not an exceedance."""

    return loss_percent > var_percent


def quantile_loss(loss_percent: float, var_percent: float, confidence: float) -> float:
    """Pinball/quantile loss at the positive-loss confidence quantile."""

    _validate_confidence(confidence)
    indicator = 1.0 if loss_percent < var_percent else 0.0
    return (confidence - indicator) * (loss_percent - var_percent)


def fz0_score(
    loss_percent: float,
    var_percent: float,
    es_percent: float,
    confidence: float,
) -> float:
    """FZ0 joint VaR/ES score in the positive-loss upper-tail convention.

    This is the sign-transformed FZ0 score for lower-tail returns, with
    tail_probability = 1 - confidence and strict exceedance matching the
    experiment's loss > VaR rule.
    """

    _validate_confidence(confidence)
    if not math.isfinite(var_percent) or not math.isfinite(es_percent):
        raise ValueError("VaR and ES must be finite")
    if es_percent <= 0 or var_percent <= 0 or es_percent < var_percent:
        raise ValueError("FZ0 requires 0 < VaR <= ES in the loss convention")
    tail_probability = 1.0 - confidence
    exceedance = 1.0 if loss_percent > var_percent else 0.0
    return (
        exceedance * (loss_percent - var_percent) / (tail_probability * es_percent)
        + var_percent / es_percent
        + math.log(es_percent)
        - 1.0
    )
