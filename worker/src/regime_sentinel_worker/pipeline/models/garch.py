"""Single-regime GARCH(1,1) fitting through the Python arch package."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

try:
    import numpy as np
except ImportError:
    np = None


class GarchFitError(RuntimeError):
    """Raised when a GARCH fit or one-step forecast is unusable."""


@dataclass(frozen=True)
class GarchFit:
    innovation: str
    sigma_percent: float
    parameters: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    convergence_flag: int


def fit_garch11(
    centered_training_returns: Sequence[float],
    *,
    innovation: str,
) -> GarchFit:
    """Fit zero-mean GARCH(1,1) to externally centered percentage returns."""

    if innovation not in {"student_t", "normal"}:
        raise ValueError(f"unsupported innovation: {innovation}")
    if np is None:
        raise GarchFitError("dependency_unavailable: numpy is required by python-arch")
    values = np.asarray(tuple(centered_training_returns), dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise GarchFitError("training window must be a finite one-dimensional series")

    try:
        from arch import arch_model

        distribution = "StudentsT" if innovation == "student_t" else "normal"
        model = arch_model(
            values,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist=distribution,
            rescale=False,
        )
        result = model.fit(update_freq=0, disp="off")
    except Exception as exc:
        raise GarchFitError(f"fit_exception: {type(exc).__name__}: {exc}") from exc

    convergence_flag = int(getattr(result, "convergence_flag", 0))
    if convergence_flag != 0:
        message = getattr(result, "optimization_result", None)
        raise GarchFitError(
            f"optimizer_nonconvergence: flag={convergence_flag}; result={message!r}"
        )

    try:
        variance_forecast = result.forecast(horizon=1, reindex=False).variance
        variance = float(variance_forecast.iloc[-1, 0])
        sigma_percent = math.sqrt(variance)
        parameters = {str(name): float(value) for name, value in result.params.items()}
        log_likelihood = float(result.loglikelihood)
        aic = float(result.aic)
        bic = float(result.bic)
    except Exception as exc:
        raise GarchFitError(
            f"forecast_extraction_exception: {type(exc).__name__}: {exc}"
        ) from exc

    if not math.isfinite(sigma_percent) or sigma_percent <= 0:
        raise GarchFitError(f"invalid_forecast_sigma: {sigma_percent!r}")
    if not all(math.isfinite(value) for value in (*parameters.values(), log_likelihood, aic, bic)):
        raise GarchFitError("fit returned non-finite parameters or information criteria")

    omega = parameters.get("omega")
    alpha = parameters.get("alpha[1]")
    beta = parameters.get("beta[1]")
    if omega is None or alpha is None or beta is None:
        raise GarchFitError(f"fit omitted required GARCH parameters: {parameters!r}")
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise GarchFitError(
            f"fit violated GARCH constraints: omega={omega}, alpha={alpha}, beta={beta}"
        )
    if innovation == "student_t" and parameters.get("nu", 0) <= 2:
        raise GarchFitError(f"fit violated Student-t df constraint: {parameters!r}")

    return GarchFit(
        innovation=innovation,
        sigma_percent=sigma_percent,
        parameters=parameters,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        convergence_flag=convergence_flag,
    )
