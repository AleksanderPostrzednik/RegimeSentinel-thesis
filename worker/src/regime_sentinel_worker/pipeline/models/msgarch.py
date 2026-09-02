"""R MSGARCH::FitML adapter and deterministic preflight checks."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


class MsgarchFitError(RuntimeError):
    """Raised when the R MSGARCH process returns no usable fit."""


@dataclass(frozen=True)
class MsgarchFit:
    start_index: int
    log_likelihood: float
    parameters: dict[str, float]
    transition_matrix: tuple[tuple[float, ...], ...]
    occupancy: tuple[float, ...]
    filtered_last: tuple[float, ...]
    unconditional_volatility: tuple[float, ...]
    state_order: tuple[int, ...]
    risk: dict[str, dict[str, float]]
    stdout: str
    stderr: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MsgarchAttempt:
    start_index: int
    success: bool
    fit: MsgarchFit | None
    errors: tuple[str, ...]
    log_path: str


@dataclass(frozen=True)
class PreflightResult:
    instrument: str
    attempts: tuple[MsgarchAttempt, ...]
    passed: bool
    checks: dict[str, Any]
    best_attempt_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "passed": self.passed,
            "checks": self.checks,
            "best_attempt_index": self.best_attempt_index,
            "attempts": [
                {
                    "start_index": attempt.start_index,
                    "success": attempt.success,
                    "errors": list(attempt.errors),
                    "log_path": attempt.log_path,
                    "fit": _fit_dict(attempt.fit) if attempt.fit else None,
                }
                for attempt in self.attempts
            ],
        }


def _fit_dict(fit: MsgarchFit | None) -> dict[str, Any] | None:
    if fit is None:
        return None
    return {
        "start_index": fit.start_index,
        "log_likelihood": fit.log_likelihood,
        "parameters": fit.parameters,
        "transition_matrix": [list(row) for row in fit.transition_matrix],
        "occupancy": list(fit.occupancy),
        "filtered_last": list(fit.filtered_last),
        "unconditional_volatility": list(fit.unconditional_volatility),
        "state_order": list(fit.state_order),
        "risk": fit.risk,
        "diagnostics": fit.diagnostics,
    }


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite R value: {value!r}")
    return parsed


def _parse_stdout(stdout: str, start_index: int) -> MsgarchFit:
    parameters: dict[str, float] = {}
    transition: dict[tuple[int, int], float] = {}
    occupancy: dict[int, float] = {}
    filtered: dict[int, float] = {}
    unc_vol: dict[int, float] = {}
    state_order: tuple[int, ...] = ()
    risk: dict[str, dict[str, float]] = {}
    log_likelihood: float | None = None
    status: str | None = None
    error: str | None = None
    master_seed: int | None = None
    effective_seed: int | None = None
    start_source: str | None = None
    start_parameters: dict[str, float] = {}
    optimizer_method: str | None = None
    optimizer_do_plm: bool | None = None
    optimizer_do_se: bool | None = None
    optimizer_convergence: int | None = None
    optimizer_message: str | None = None
    optimizer_objective: float | None = None
    optimizer_counts: dict[str, int] = {}
    optimizer_vectors: dict[str, dict[int, dict[str, Any]]] = {
        "received": {},
        "used": {},
        "end": {},
    }
    hessian_available: bool | None = None
    hessian_dimension: int | None = None
    hessian_entries: dict[tuple[int, int], float] = {}
    hessian_eigenvalues: dict[int, float] = {}

    for line in stdout.splitlines():
        fields = line.split("\t")
        if not fields or not fields[0]:
            continue
        key = fields[0]
        try:
            if key == "STATUS":
                status = fields[1]
            elif key == "ERROR":
                error = "\t".join(fields[1:])
            elif key == "LOG_LIK":
                log_likelihood = _float(fields[1])
            elif key == "PARAM":
                parameters[fields[1]] = _float(fields[2])
            elif key == "TRANS":
                transition[(int(fields[1]), int(fields[2]))] = _float(fields[3])
            elif key == "OCCUPANCY":
                occupancy[int(fields[1])] = _float(fields[2])
            elif key == "FILTERED_LAST":
                filtered[int(fields[1])] = _float(fields[2])
            elif key == "UNC_VOL":
                unc_vol[int(fields[1])] = _float(fields[2])
            elif key == "STATE_ORDER":
                state_order = tuple(int(value) for value in fields[1:])
            elif key == "RISK":
                risk[fields[1]] = {"var_percent": _float(fields[2]), "es_percent": _float(fields[3])}
            elif key == "MASTER_SEED":
                master_seed = int(fields[1])
            elif key == "EFFECTIVE_SEED":
                effective_seed = int(fields[1])
            elif key == "START_SOURCE":
                start_source = fields[1]
            elif key == "START_PARAM":
                start_parameters[fields[1]] = _float(fields[2])
            elif key == "OPTIM_METHOD":
                optimizer_method = fields[1]
            elif key == "OPTIM_DO_PLM":
                optimizer_do_plm = fields[1].lower() == "true"
            elif key == "OPTIM_DO_SE":
                optimizer_do_se = fields[1].lower() == "true"
            elif key == "OPTIM_CONVERGENCE":
                optimizer_convergence = int(fields[1])
            elif key == "OPTIM_MESSAGE":
                optimizer_message = "\t".join(fields[1:])
            elif key == "OPTIM_OBJECTIVE":
                optimizer_objective = _float(fields[1])
            elif key == "OPTIM_COUNT":
                optimizer_counts[fields[1]] = int(fields[2])
            elif key in {"OPTIM_START_RECEIVED", "OPTIM_START_USED", "OPTIM_END"}:
                vector_key = {
                    "OPTIM_START_RECEIVED": "received",
                    "OPTIM_START_USED": "used",
                    "OPTIM_END": "end",
                }[key]
                optimizer_vectors[vector_key][int(fields[1])] = {
                    "name": fields[2],
                    "value": _float(fields[3]),
                }
            elif key == "HESSIAN_AVAILABLE":
                hessian_available = fields[1].lower() == "true"
            elif key == "HESSIAN_DIM":
                hessian_dimension = int(fields[1])
            elif key == "HESSIAN":
                hessian_entries[(int(fields[1]), int(fields[2]))] = _float(fields[3])
            elif key == "HESSIAN_EIGEN":
                hessian_eigenvalues[int(fields[1])] = _float(fields[2])
        except (IndexError, TypeError, ValueError) as exc:
            raise MsgarchFitError(f"invalid R output line: {line!r}: {exc}") from exc

    if status != "success":
        raise MsgarchFitError(error or "R MSGARCH did not report success")
    if log_likelihood is None:
        raise MsgarchFitError("R MSGARCH output omitted log-likelihood")
    if not state_order:
        state_order = tuple(sorted(unc_vol, key=unc_vol.get))
    states = sorted(set(occupancy) | set(filtered) | set(unc_vol))
    if states != [1, 2]:
        raise MsgarchFitError(f"R MSGARCH output must contain states 1 and 2, got {states}")
    try:
        matrix = tuple(tuple(transition[(row, column)] for column in (1, 2)) for row in (1, 2))
    except KeyError as exc:
        raise MsgarchFitError(f"R MSGARCH output omitted transition entry {exc.args[0]}") from exc
    hessian = None
    if hessian_available and hessian_dimension is not None:
        expected_entries = hessian_dimension * hessian_dimension
        if len(hessian_entries) == expected_entries:
            hessian = [
                [
                    hessian_entries[(row, column)]
                    for column in range(1, hessian_dimension + 1)
                ]
                for row in range(1, hessian_dimension + 1)
            ]
    diagnostics = {
        "master_seed": master_seed,
        "effective_seed": effective_seed,
        "start_source": start_source,
        "start_parameters": start_parameters,
        "optimizer": {
            "method": optimizer_method,
            "do_plm": optimizer_do_plm,
            "do_se": optimizer_do_se,
            "convergence": optimizer_convergence,
            "message": optimizer_message,
            "objective": optimizer_objective,
            "counts": optimizer_counts,
            "start_received_transformed": [
                optimizer_vectors["received"][index]
                for index in sorted(optimizer_vectors["received"])
            ],
            "start_used_transformed": [
                optimizer_vectors["used"][index]
                for index in sorted(optimizer_vectors["used"])
            ],
            "end_transformed": [
                optimizer_vectors["end"][index]
                for index in sorted(optimizer_vectors["end"])
            ],
            "hessian_available": hessian_available,
            "hessian": hessian,
            "hessian_eigenvalues": [
                hessian_eigenvalues[index] for index in sorted(hessian_eigenvalues)
            ],
        },
    }
    return MsgarchFit(
        start_index=start_index,
        log_likelihood=log_likelihood,
        parameters=parameters,
        transition_matrix=matrix,
        occupancy=tuple(occupancy[state] for state in (1, 2)),
        filtered_last=tuple(filtered[state] for state in (1, 2)),
        unconditional_volatility=tuple(unc_vol[state] for state in (1, 2)),
        state_order=state_order,
        risk=risk,
        stdout=stdout,
        stderr="",
        diagnostics=diagnostics,
    )


class MsgarchRunner:
    """Invoke the pinned R implementation; never substitute another engine."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        rscript: str = "Rscript",
        seed: int = 20260722,
        timeout_seconds: float = 300.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("MSGARCH timeout must be positive")
        self.repo_root = Path(repo_root).resolve()
        self.rscript = rscript
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.script = self.repo_root / "worker" / "r" / "msgarch_fit.R"

    def fit(
        self,
        values: Sequence[float],
        *,
        start_index: int,
        mode: str,
        par0: Sequence[float] | None = None,
        log_path: str | Path | None = None,
    ) -> MsgarchFit:
        if mode not in {"preflight", "rolling"}:
            raise ValueError(f"unsupported MSGARCH mode: {mode}")
        if len(values) != 500:
            raise ValueError("MSGARCH windows must contain exactly 500 returns")
        with tempfile.TemporaryDirectory(prefix="regimesentinel-msgarch-") as directory:
            directory_path = Path(directory)
            data_path = directory_path / "data.txt"
            data_path.write_text("\n".join(f"{float(value):.17g}" for value in values) + "\n", encoding="utf-8")
            command = [
                self.rscript,
                str(self.script),
                "--data",
                str(data_path),
                "--mode",
                mode,
                "--start-index",
                str(start_index),
                "--seed",
                str(self.seed),
            ]
            if par0 is not None:
                par_path = directory_path / "par0.txt"
                par_path.write_text("\n".join(f"{float(value):.17g}" for value in par0) + "\n", encoding="utf-8")
                command.extend(["--par0", str(par_path)])
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                message = f"process_timeout: exceeded {self.timeout_seconds:g}s"
                if log_path is not None:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(log_path).write_text(message + "\n", encoding="utf-8")
                raise MsgarchFitError(message) from exc
            except OSError as exc:
                if log_path is not None:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(log_path).write_text(f"process_error: {type(exc).__name__}: {exc}\n", encoding="utf-8")
                raise MsgarchFitError(f"process_error: {type(exc).__name__}: {exc}") from exc
            output = completed.stdout
            if log_path is not None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(log_path).write_text(
                    output + "\n--- STDERR ---\n" + completed.stderr,
                    encoding="utf-8",
                )
            if completed.returncode != 0:
                raise MsgarchFitError(
                    f"R process exit={completed.returncode}: {completed.stderr.strip() or output.strip()}"
                )
            fit = _parse_stdout(output, start_index)
            return MsgarchFit(
                start_index=fit.start_index,
                log_likelihood=fit.log_likelihood,
                parameters=fit.parameters,
                transition_matrix=fit.transition_matrix,
                occupancy=fit.occupancy,
                filtered_last=fit.filtered_last,
                unconditional_volatility=fit.unconditional_volatility,
                state_order=fit.state_order,
                risk=fit.risk,
                stdout=output,
                stderr=completed.stderr,
                diagnostics=fit.diagnostics,
            )


def _state_parameter_errors(parameters: dict[str, float]) -> list[str]:
    errors: list[str] = []
    for state in (1, 2):
        prefix = f"_{state}"
        try:
            omega = parameters[f"alpha0{prefix}"]
            alpha = parameters[f"alpha1{prefix}"]
            beta = parameters[f"beta{prefix}"]
        except KeyError as exc:
            errors.append(f"missing parameter {exc.args[0]}")
            continue
        if omega <= 0:
            errors.append(f"alpha0_{state} must be > 0")
        if alpha < 0:
            errors.append(f"alpha1_{state} must be >= 0")
        if beta < 0:
            errors.append(f"beta_{state} must be >= 0")
        if alpha + beta >= 1:
            errors.append(f"alpha1_{state}+beta_{state} must be < 1")
    nus = [value for name, value in parameters.items() if name.startswith("nu")]
    if not nus or any(not math.isfinite(value) or value <= 2 for value in nus):
        errors.append("Student-t nu must be > 2")
    elif max(nus) - min(nus) > 1e-8:
        errors.append("Student-t nu must be shared across states")
    return errors


def validate_msgarch_fit(
    fit: MsgarchFit,
    *,
    occupancy_minimum: float = 0.05,
    transition_tolerance: float = 1e-8,
) -> list[str]:
    errors: list[str] = []
    if not math.isfinite(fit.log_likelihood):
        errors.append("log-likelihood is not finite")
    errors.extend(_state_parameter_errors(fit.parameters))
    if len(fit.transition_matrix) != 2 or any(len(row) != 2 for row in fit.transition_matrix):
        errors.append("transition matrix is not 2x2")
    for index, row in enumerate(fit.transition_matrix, start=1):
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in row):
            errors.append(f"transition row {index} contains an invalid probability")
        if abs(sum(row) - 1.0) > transition_tolerance:
            errors.append(f"transition row {index} does not sum to one within {transition_tolerance}")
    if (
        len(fit.occupancy) != 2
        or any(not math.isfinite(value) or value < occupancy_minimum or value > 1 for value in fit.occupancy)
    ):
        errors.append(f"state occupancy must be >= {occupancy_minimum}")
    elif abs(sum(fit.occupancy) - 1.0) > transition_tolerance:
        errors.append("state occupancy does not sum to one")
    if (
        len(fit.filtered_last) != 2
        or any(not math.isfinite(value) or value < 0 or value > 1 for value in fit.filtered_last)
        or abs(sum(fit.filtered_last) - 1.0) > transition_tolerance
    ):
        errors.append("filtered probabilities must contain two probabilities summing to one")
    if (
        len(fit.unconditional_volatility) != 2
        or any(not math.isfinite(value) or value <= 0 for value in fit.unconditional_volatility)
    ):
        errors.append("unconditional volatility must be finite and positive for both states")
    if len(fit.state_order) != 2 or set(fit.state_order) != {1, 2}:
        errors.append("state order must contain both original states")
    if set(fit.risk) != {"0.95", "0.99"}:
        errors.append("risk output must contain exactly 0.95 and 0.99")
    for confidence, risk in fit.risk.items():
        if set(risk) != {"var_percent", "es_percent"}:
            errors.append(f"risk {confidence} has an invalid payload")
        elif risk["var_percent"] <= 0 or risk["es_percent"] < risk["var_percent"]:
            errors.append(f"risk {confidence} must satisfy 0 < VaR <= ES")
    return errors


def _canonical_parameters(fit: MsgarchFit) -> tuple[float, ...]:
    order = fit.state_order
    values: list[float] = []
    for state in order:
        for name in ("alpha0", "alpha1", "beta"):
            values.append(fit.parameters[f"{name}_{state}"])
    nu = next(value for name, value in fit.parameters.items() if name.startswith("nu"))
    values.append(nu)
    for row in order:
        for column in order:
            values.append(fit.transition_matrix[row - 1][column - 1])
    return tuple(values)


def repeat_parameter_delta(best: MsgarchFit, other: MsgarchFit) -> float:
    first, second = _canonical_parameters(best), _canonical_parameters(other)
    if len(first) != len(second):
        return float("inf")
    return max(abs(left - right) for left, right in zip(first, second))


def run_preflight(
    *,
    instrument: str,
    values: Sequence[float],
    runner: MsgarchRunner,
    log_root: str | Path,
    starts: int = 5,
    occupancy_minimum: float = 0.05,
    transition_tolerance: float = 1e-8,
    repeat_tolerance: float = 1e-6,
) -> PreflightResult:
    if starts != 5:
        raise ValueError("thesis-v1 preflight requires exactly five starts")
    attempts: list[MsgarchAttempt] = []
    for start_index in range(starts):
        log_path = Path(log_root) / f"{instrument.replace('-', '_')}_start_{start_index + 1}.log"
        try:
            fit = runner.fit(values, start_index=start_index, mode="preflight", log_path=log_path)
        except Exception as exc:
            attempts.append(MsgarchAttempt(start_index, False, None, (str(exc),), log_path.as_posix()))
            continue
        errors = tuple(validate_msgarch_fit(fit, occupancy_minimum=occupancy_minimum, transition_tolerance=transition_tolerance))
        attempts.append(MsgarchAttempt(start_index, not errors, fit if not errors else fit, errors, log_path.as_posix()))
    fitted = [attempt for attempt in attempts if attempt.fit is not None]

    valid = [attempt for attempt in attempts if attempt.success and attempt.fit is not None]
    best = max(valid, key=lambda attempt: attempt.fit.log_likelihood) if valid else None
    repeat_delta = (
        max(repeat_parameter_delta(best.fit, attempt.fit) for attempt in valid if attempt is not best)
        if best is not None and len(valid) > 1
        else None
    )
    repeatable = (
        len(valid) == 5
        and repeat_delta is not None
        and repeat_delta <= repeat_tolerance
    )
    checks = {
        "exactly_five_starts": len(attempts) == 5,
        "all_starts_successful_and_valid": len(valid) == 5,
        "finite_log_likelihood": len(fitted) == 5
        and all(math.isfinite(attempt.fit.log_likelihood) for attempt in fitted),
        "parameter_constraints": len(fitted) == 5
        and all(not _state_parameter_errors(attempt.fit.parameters) for attempt in fitted),
        "transition_rows": len(fitted) == 5
        and all(
            not any("transition row" in error for error in attempt.errors)
            for attempt in attempts
        ),
        "occupancy": len(fitted) == 5
        and all(not any("occupancy" in error for error in attempt.errors) for attempt in attempts),
        "repeat_parameter_delta": repeat_delta,
        "repeatable_within_tolerance": repeatable,
    }
    passed = all(value is True for key, value in checks.items() if key != "repeat_parameter_delta") and checks["repeatable_within_tolerance"]
    return PreflightResult(instrument, tuple(attempts), passed, checks, best.start_index if best else None)
