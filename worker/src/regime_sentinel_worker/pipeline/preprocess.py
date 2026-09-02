"""Deterministic log-return transformation required by thesis-v1."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .ingest import INSTRUMENTS, PriceSnapshot


MODEL_SCALE_FACTOR = 100


@dataclass(frozen=True)
class ReturnSeries:
    """One instrument's log returns and model-scaled values."""

    instrument: str
    dates: tuple[str, ...]
    log_returns: tuple[float, ...]
    model_returns_percent: tuple[float, ...]
    scale_factor: int

    @property
    def count(self) -> int:
        return len(self.dates)


def build_log_returns(
    snapshot: PriceSnapshot,
    *,
    scale_factor: int = MODEL_SCALE_FACTOR,
) -> dict[str, ReturnSeries]:
    """Build ln(P_t / P_t-1) without imputation or row removal."""

    if scale_factor != MODEL_SCALE_FACTOR:
        raise ValueError("thesis-v1 model scale is fixed at 100")
    if snapshot.row_count < 2:
        raise ValueError("at least two prices are required to build returns")

    return_dates = snapshot.dates[1:]
    result: dict[str, ReturnSeries] = {}
    for instrument in INSTRUMENTS:
        prices = snapshot.prices[instrument]
        log_returns = tuple(
            math.log(current / previous)
            for previous, current in zip(prices, prices[1:])
        )
        model_returns = tuple(value * scale_factor for value in log_returns)
        if not all(math.isfinite(value) for value in (*log_returns, *model_returns)):
            raise ValueError(f"non-finite log return generated for {instrument}")
        result[instrument] = ReturnSeries(
            instrument=instrument,
            dates=return_dates,
            log_returns=log_returns,
            model_returns_percent=model_returns,
            scale_factor=scale_factor,
        )
    return result
