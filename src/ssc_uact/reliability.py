"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike, NDArray

from scipy.optimize import minimize

from scipy.special import expit

class MonotoneReliabilityModel:
    """Ridge logistic error model with nonnegative uncertainty coefficients."""

    def __init__(self, ridge: float = 0.1) -> None:
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.ridge = float(ridge)
        self.minimum_: NDArray[np.float64] | None = None
        self.range_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.coef_: NDArray[np.float64] | None = None

    def fit_oof(
        self, uncertainty_components_oof: ArrayLike, hard_errors_oof: ArrayLike
    ) -> "MonotoneReliabilityModel":
        x, errors = _validate_data(uncertainty_components_oof, hard_errors_oof)
        self.minimum_ = x.min(axis=0)
        maximum = x.max(axis=0)
        self.range_ = maximum - self.minimum_
        scaled = self._scale(x)
        initial_rate = np.clip(errors.mean(), 1e-6, 1 - 1e-6)
        initial = np.r_[np.log(initial_rate / (1 - initial_rate)), np.zeros(x.shape[1])]

        def objective(beta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            risk = expit(beta[0] + scaled @ beta[1:])
            eps = 1e-12
            loss = -np.mean(
                errors * np.log(np.clip(risk, eps, 1))
                + (1 - errors) * np.log(np.clip(1 - risk, eps, 1))
            )
            loss += self.ridge * float(beta[1:] @ beta[1:])
            residual = risk - errors
            gradient = np.r_[
                residual.mean(),
                scaled.T @ residual / len(errors) + 2 * self.ridge * beta[1:],
            ]
            return float(loss), gradient

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None)] + [(0.0, None)] * x.shape[1],
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"reliability fit failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.coef_ = np.maximum(result.x[1:], 0.0)
        return self

    def predict_error_risk(self, uncertainty_components: ArrayLike) -> NDArray[np.float64]:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("reliability model has not been fitted")
        x = np.asarray(uncertainty_components, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(self.coef_) or not np.isfinite(x).all():
            raise ValueError("uncertainty components are invalid or dimensionally incompatible")
        return expit(self.intercept_ + self._scale(x) @ self.coef_)

    def predict_srs(self, uncertainty_components: ArrayLike) -> NDArray[np.float64]:
        return 100.0 * (1.0 - self.predict_error_risk(uncertainty_components))

    def _scale(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.minimum_ is None or self.range_ is None:
            raise RuntimeError("reliability scaler has not been fitted")
        denominator = np.where(self.range_ > 0, self.range_, 1.0)
        return np.clip((x - self.minimum_) / denominator, 0.0, 1.0)

def hard_error_targets(probabilities: ArrayLike, y_true: ArrayLike) -> NDArray[np.int64]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.ndim != 2 or y.shape != (len(p),):
        raise ValueError("probabilities and y_true are misaligned")
    return (np.argmax(p, axis=1) != y).astype(int)

def _validate_data(
    components: ArrayLike, errors: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(components, dtype=float)
    y = np.asarray(errors, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),):
        raise ValueError("components and errors must align")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("components and errors must be finite")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("hard errors must be binary")
    return x, y
