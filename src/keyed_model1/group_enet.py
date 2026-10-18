"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

from .ap_tuning import ap_weighted_row_weights

from .common import CLASS_NAMES, metric_row, row_weights

def _softmax(eta: np.ndarray) -> np.ndarray:
    eta = eta - np.max(eta, axis=1, keepdims=True)
    exp_eta = np.exp(eta)
    return exp_eta / exp_eta.sum(axis=1, keepdims=True)

@dataclass
class FitState:
    objective: float
    iterations: int
    converged: bool
    step_size: float

class GroupedMultinomialElasticNet:
    """Reference-class multinomial logistic regression with source-block elastic-net penalty."""

    def __init__(
        self,
        alpha: float,
        lambda_: float,
        groups: dict[str, list[int]],
        max_iter: int = 400,
        tol: float = 1e-5,
    ) -> None:
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.groups = {name: np.array(cols, dtype=int) for name, cols in groups.items() if cols}
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.intercept_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.fit_state_: FitState | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "GroupedMultinomialElasticNet":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be a 2D design matrix.")
        if X.shape[1] == 0:
            raise ValueError("Cannot fit Model 1 with zero design columns.")
        n, p = X.shape
        k = len(CLASS_NAMES)
        if sample_weight is None:
            sample_weight = row_weights(y, n_classes=k)
        sample_weight = np.asarray(sample_weight, dtype=float)
        weight_sum = float(sample_weight.sum())

        B = np.zeros((p, k - 1), dtype=float)
        b0 = self._intercept_only(y, sample_weight, k)
        previous = self._objective(X, y, sample_weight, b0, B)
        step = self._initial_step(X, sample_weight)
        converged = False

        for iteration in range(1, self.max_iter + 1):
            grad_b0, grad_B = self._gradient(X, y, sample_weight, b0, B, weight_sum)
            accepted = False
            local_step = step
            for _ in range(40):
                candidate_b0 = b0 - local_step * grad_b0
                candidate_B = B - local_step * grad_B
                candidate_B = self._prox_group(candidate_B, local_step)
                current = self._objective(X, y, sample_weight, candidate_b0, candidate_B)
                if current <= previous + 1e-10:
                    accepted = True
                    break
                local_step *= 0.5
            if not accepted:
                candidate_b0 = b0 - local_step * grad_b0
                candidate_B = self._prox_group(B - local_step * grad_B, local_step)
                current = self._objective(X, y, sample_weight, candidate_b0, candidate_B)

            delta = abs(previous - current) / max(1.0, abs(previous))
            b0, B, previous, step = candidate_b0, candidate_B, current, min(local_step * 1.05, 1.0)
            if delta < self.tol:
                converged = True
                break

        self.intercept_ = b0
        self.coef_ = B
        self.fit_state_ = FitState(objective=float(previous), iterations=iteration, converged=converged, step_size=float(step))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted.")
        non_ref = X @ self.coef_ + self.intercept_
        eta = np.column_stack([non_ref, np.zeros(X.shape[0])])
        return _softmax(eta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def source_norms(self) -> dict[str, float]:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        return {
            source: float(np.linalg.norm(self.coef_[cols, :], ord="fro"))
            for source, cols in self.groups.items()
        }

    def coefficient_rows(self, feature_names: list[str]) -> list[dict[str, Any]]:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        rows: list[dict[str, Any]] = []
        for j, feature in enumerate(feature_names):
            row = {"feature_name": feature}
            for class_idx, class_name in enumerate(CLASS_NAMES[:-1]):
                row[f"coef_vs_healthy_{class_name}"] = float(self.coef_[j, class_idx])
            rows.append(row)
        return rows

    def _intercept_only(self, y: np.ndarray, sample_weight: np.ndarray, k: int) -> np.ndarray:
        weighted_counts = np.array([sample_weight[y == code].sum() for code in range(k)], dtype=float)
        probs = np.clip(weighted_counts / weighted_counts.sum(), 1e-12, 1.0)
        return np.log(probs[:-1] / probs[-1])

    def _objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        b0: np.ndarray,
        B: np.ndarray,
    ) -> float:
        proba = _softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
        nll = -np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0))
        weighted_nll = float(np.dot(sample_weight, nll) / sample_weight.sum())
        ridge = 0.5 * (1.0 - self.alpha) * float(np.sum(B * B))
        group = 0.0
        if self.alpha > 0:
            for cols in self.groups.values():
                group += np.sqrt(len(cols)) * float(np.linalg.norm(B[cols, :], ord="fro"))
        return weighted_nll + self.lambda_ * (ridge + self.alpha * group)

    def _gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        b0: np.ndarray,
        B: np.ndarray,
        weight_sum: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        k_minus_ref = B.shape[1]
        proba = _softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
        y_non_ref = np.zeros((len(y), k_minus_ref), dtype=float)
        for klass in range(k_minus_ref):
            y_non_ref[:, klass] = y == klass
        residual = (proba[:, :k_minus_ref] - y_non_ref) * sample_weight[:, None]
        grad_b0 = residual.sum(axis=0) / weight_sum
        grad_B = X.T @ residual / weight_sum
        grad_B += self.lambda_ * (1.0 - self.alpha) * B
        return grad_b0, grad_B

    def _prox_group(self, B: np.ndarray, step: float) -> np.ndarray:
        if self.alpha <= 0:
            return B
        out = B.copy()
        for cols in self.groups.values():
            block = out[cols, :]
            norm = float(np.linalg.norm(block, ord="fro"))
            threshold = step * self.lambda_ * self.alpha * np.sqrt(len(cols))
            if norm <= threshold:
                out[cols, :] = 0.0
            else:
                out[cols, :] = (1.0 - threshold / norm) * block
        return out

    def _initial_step(self, X: np.ndarray, sample_weight: np.ndarray) -> float:
        weighted_norm = float((sample_weight[:, None] * X * X).sum() / max(sample_weight.sum(), 1e-12))
        lipschitz_bound = 0.25 * max(weighted_norm, 1e-6) + self.lambda_ * max(0.0, 1.0 - self.alpha)
        return min(1.0, 1.0 / lipschitz_bound)

def lambda_max_for_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: dict[str, list[int]],
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    if alpha <= 0:
        raise ValueError("lambda_max is defined for alpha > 0.")
    y = np.asarray(y, dtype=int)
    weights = row_weights(y, n_classes=len(CLASS_NAMES)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    temp = GroupedMultinomialElasticNet(alpha=alpha, lambda_=1.0, groups=groups)
    b0 = temp._intercept_only(y, weights, len(CLASS_NAMES))
    B0 = np.zeros((X.shape[1], len(CLASS_NAMES) - 1), dtype=float)
    _, grad = temp._gradient(X, y, weights, b0, B0, float(weights.sum()))
    values = []
    for cols in groups.values():
        if not cols:
            continue
        block = grad[np.array(cols, dtype=int), :]
        values.append(float(np.linalg.norm(block, ord="fro") / (alpha * np.sqrt(len(cols)))))
    return max(values) if values else 1.0

def fit_one_group_enet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    groups: dict[str, list[int]],
    alpha: float,
    lambda_fraction: float,
    max_iter: int,
    tol: float,
    atypical_weight_multiplier: float = 1.0,
) -> dict[str, Any]:
    sample_weight = ap_weighted_row_weights(y_train, atypical_multiplier=atypical_weight_multiplier, n_classes=len(CLASS_NAMES))
    lambda_max = lambda_max_for_groups(X_train, y_train, groups, alpha, sample_weight=sample_weight)
    lambda_ = max(lambda_max * float(lambda_fraction), 1e-8)
    model = GroupedMultinomialElasticNet(alpha=alpha, lambda_=lambda_, groups=groups, max_iter=max_iter, tol=tol)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    proba = model.predict_proba(X_valid)
    metrics = metric_row(y_valid, proba)
    norms = model.source_norms()
    return {
        "alpha": float(alpha),
        "lambda_fraction": float(lambda_fraction),
        "atypical_weight_multiplier": float(atypical_weight_multiplier),
        "lambda": float(lambda_),
        "lambda_max": float(lambda_max),
        "proba": proba,
        "metrics": metrics,
        "nonzero_sources": int(sum(v > 1e-8 for v in norms.values())),
        "fit_state": model.fit_state_.__dict__ if model.fit_state_ is not None else {},
    }
