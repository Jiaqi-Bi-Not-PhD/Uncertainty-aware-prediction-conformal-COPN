"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Protocol

import numpy as np

from numpy.typing import ArrayLike, NDArray

from scipy.optimize import minimize

from .probability import multiclass_log_loss, normalize_probabilities, softmax

class Calibrator(Protocol):
    name: str

    def transform(self, probabilities: ArrayLike) -> NDArray[np.float64]: ...

class NoCalibration:
    name = "none"

    def fit_oof(self, probabilities: ArrayLike, y_true: ArrayLike) -> "NoCalibration":
        multiclass_log_loss(y_true, probabilities)
        return self

    def transform(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        return normalize_probabilities(probabilities)

class TemperatureCalibrator:
    name = "temperature"

    def __init__(self) -> None:
        self.temperature_: float | None = None

    def fit_oof(
        self, probabilities: ArrayLike, y_true: ArrayLike
    ) -> "TemperatureCalibrator":
        p, y = _validate_fit_data(probabilities, y_true)

        def objective(x: NDArray[np.float64]) -> float:
            calibrated = softmax(np.log(p) / np.exp(x[0]))
            return multiclass_log_loss(y, calibrated)

        result = minimize(objective, np.zeros(1), method="L-BFGS-B", bounds=[(-5, 5)])
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"temperature calibration failed: {result.message}")
        self.temperature_ = float(np.exp(result.x[0]))
        return self

    def transform(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        if self.temperature_ is None:
            raise RuntimeError("calibrator has not been fitted")
        p = normalize_probabilities(probabilities)
        return softmax(np.log(p) / self.temperature_)

class VectorCalibrator:
    name = "vector"

    def __init__(self, ridge: float = 1e-2) -> None:
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.ridge = float(ridge)
        self.slopes_: NDArray[np.float64] | None = None
        self.intercepts_: NDArray[np.float64] | None = None

    def fit_oof(self, probabilities: ArrayLike, y_true: ArrayLike) -> "VectorCalibrator":
        p, y = _validate_fit_data(probabilities, y_true)
        k = p.shape[1]

        def unpack(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return x[:k], x[k:] - np.mean(x[k:])

        def objective(x: NDArray[np.float64]) -> float:
            slopes, intercepts = unpack(x)
            loss = multiclass_log_loss(y, softmax(np.log(p) * slopes + intercepts))
            return loss + self.ridge * float(np.sum((slopes - 1.0) ** 2))

        initial = np.r_[np.ones(k), np.zeros(k)]
        result = minimize(objective, initial, method="L-BFGS-B")
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"vector calibration failed: {result.message}")
        self.slopes_, self.intercepts_ = unpack(result.x)
        return self

    def transform(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        if self.slopes_ is None or self.intercepts_ is None:
            raise RuntimeError("calibrator has not been fitted")
        p = normalize_probabilities(probabilities)
        return softmax(np.log(p) * self.slopes_ + self.intercepts_)

class DirichletCalibrator:
    name = "dirichlet"

    def __init__(self, ridge: float = 1e-2) -> None:
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.ridge = float(ridge)
        self.matrix_: NDArray[np.float64] | None = None
        self.intercepts_: NDArray[np.float64] | None = None

    def fit_oof(
        self, probabilities: ArrayLike, y_true: ArrayLike
    ) -> "DirichletCalibrator":
        p, y = _validate_fit_data(probabilities, y_true)
        k = p.shape[1]
        identity = np.eye(k)

        def unpack(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            matrix = x[: k * k].reshape(k, k)
            intercepts = x[k * k :] - np.mean(x[k * k :])
            return matrix, intercepts

        def objective(x: NDArray[np.float64]) -> float:
            matrix, intercepts = unpack(x)
            calibrated = softmax(np.log(p) @ matrix.T + intercepts)
            penalty = self.ridge * float(np.sum((matrix - identity) ** 2))
            return multiclass_log_loss(y, calibrated) + penalty

        initial = np.r_[identity.ravel(), np.zeros(k)]
        result = minimize(objective, initial, method="L-BFGS-B")
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Dirichlet calibration failed: {result.message}")
        self.matrix_, self.intercepts_ = unpack(result.x)
        return self

    def transform(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        if self.matrix_ is None or self.intercepts_ is None:
            raise RuntimeError("calibrator has not been fitted")
        p = normalize_probabilities(probabilities)
        return softmax(np.log(p) @ self.matrix_.T + self.intercepts_)

@dataclass(frozen=True)
class CalibrationSelection:
    name: str
    calibrator: Calibrator
    losses: dict[str, float]

def fit_calibrator(
    name: str, probabilities_oof: ArrayLike, y_oof: ArrayLike, ridge: float = 1e-2
) -> Calibrator:
    """Fit a named calibrator; inputs are explicitly proper-fit OOF records."""

    factories = {
        "none": lambda: NoCalibration(),
        "temperature": lambda: TemperatureCalibrator(),
        "vector": lambda: VectorCalibrator(ridge),
        "dirichlet": lambda: DirichletCalibrator(ridge),
    }
    try:
        calibrator = factories[name]()
    except KeyError as exc:
        raise ValueError(f"unknown calibration family: {name}") from exc
    return calibrator.fit_oof(probabilities_oof, y_oof)

def select_calibrator(
    probabilities_oof: ArrayLike,
    y_oof: ArrayLike,
    candidates: Iterable[str] = ("none", "temperature", "vector", "dirichlet"),
    *,
    ridge: float = 1e-2,
    tie_tolerance: float = 1e-8,
) -> CalibrationSelection:
    """Select by OOF unweighted log loss, breaking ties toward lower complexity."""

    ordered = list(candidates)
    if not ordered:
        raise ValueError("at least one candidate is required")
    complexity = {"none": 0, "temperature": 1, "vector": 2, "dirichlet": 3}
    fitted: dict[str, Calibrator] = {}
    losses: dict[str, float] = {}
    for name in ordered:
        model = fit_calibrator(name, probabilities_oof, y_oof, ridge)
        fitted[name] = model
        losses[name] = multiclass_log_loss(
            y_oof, model.transform(probabilities_oof)
        )
    best_loss = min(losses.values())
    tied = [name for name, loss in losses.items() if loss <= best_loss + tie_tolerance]
    best = min(tied, key=lambda name: (complexity[name], ordered.index(name)))
    return CalibrationSelection(best, fitted[best], losses)

def _validate_fit_data(
    probabilities: ArrayLike, y_true: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    p = normalize_probabilities(probabilities)
    if p.ndim == 1:
        p = p[None, :]
    y = np.asarray(y_true, dtype=int)
    multiclass_log_loss(y, p)
    return p, y
