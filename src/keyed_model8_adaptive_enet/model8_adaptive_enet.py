"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

from keyed_model1.common import CLASS_NAMES, row_weights

def softmax(eta: np.ndarray) -> np.ndarray:
    eta = eta - np.max(eta, axis=1, keepdims=True)
    exp_eta = np.exp(eta)
    return exp_eta / exp_eta.sum(axis=1, keepdims=True)

@dataclass
class FitDiagnostics:
    n_iter: int
    converged: bool
    objective: float
    max_coef_change: float
    selected_block_count: int
    step_size: float
    warning: str = ""

class AdaptiveMIGroupedMultinomialElasticNet:
    """Multinomial adaptive source-grouped elastic net averaged over imputations."""

    def __init__(
        self,
        alpha: float,
        lambda_: float,
        gamma: float,
        groups: dict[str, list[int]],
        adaptive_weights: dict[str, float] | None = None,
        max_iter: int = 500,
        tol: float = 1e-5,
    ) -> None:
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.groups = {name: np.array(cols, dtype=int) for name, cols in groups.items() if cols}
        self.adaptive_weights = {name: float((adaptive_weights or {}).get(name, 1.0)) for name in self.groups}
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.intercept_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.fit_diagnostics_: FitDiagnostics | None = None

    def fit(
        self,
        X_list: list[np.ndarray],
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        init_intercept: np.ndarray | None = None,
        init_coef: np.ndarray | None = None,
    ) -> "AdaptiveMIGroupedMultinomialElasticNet":
        if not X_list:
            raise ValueError("X_list must contain at least one imputed design matrix.")
        y = np.asarray(y, dtype=int)
        X_list = [np.asarray(X, dtype=float) for X in X_list]
        p = X_list[0].shape[1]
        if p == 0:
            raise ValueError("Cannot fit adaptive Model 8 with zero design columns.")
        if any(X.shape != X_list[0].shape for X in X_list):
            raise ValueError("All imputed design matrices must have the same shape.")
        if X_list[0].shape[0] != len(y):
            raise ValueError("Design rows do not match y length.")
        k = len(CLASS_NAMES)
        weights = row_weights(y, n_classes=k) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Sample weights must be finite and positive.")

        B = np.zeros((p, k - 1), dtype=float) if init_coef is None else np.asarray(init_coef, dtype=float).copy()
        b0 = self._intercept_only(y, weights, k) if init_intercept is None else np.asarray(init_intercept, dtype=float).copy()
        previous = self._objective(X_list, y, weights, b0, B)
        step = self._initial_step(X_list, weights)
        converged = False
        max_change = np.inf
        warning = ""
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            grad_b0, grad_B = self._gradient(X_list, y, weights, b0, B)
            local_step = step
            accepted = False
            candidate_b0 = b0
            candidate_B = B
            current = previous
            for _ in range(40):
                candidate_b0 = b0 - local_step * grad_b0
                candidate_B = self._prox(B - local_step * grad_B, local_step)
                current = self._objective(X_list, y, weights, candidate_b0, candidate_B)
                if np.isfinite(current) and current <= previous + 1e-10:
                    accepted = True
                    break
                local_step *= 0.5
            if not accepted:
                warning = "line_search_not_accepted"
                break
            max_change = float(max(np.max(np.abs(candidate_B - B)), np.max(np.abs(candidate_b0 - b0))))
            rel_obj = abs(previous - current) / max(1.0, abs(previous))
            b0, B, previous, step = candidate_b0, candidate_B, current, min(local_step * 1.05, 1.0)
            if rel_obj < self.tol and max_change < max(1e-5, self.tol * 10):
                converged = True
                break

        self.intercept_ = b0
        self.coef_ = B
        self.fit_diagnostics_ = FitDiagnostics(
            n_iter=int(iteration),
            converged=bool(converged),
            objective=float(previous),
            max_coef_change=float(max_change if np.isfinite(max_change) else np.nan),
            selected_block_count=int(self.selected_block_count()),
            step_size=float(step),
            warning=warning,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("Model 8 adaptive elastic net is not fitted.")
        eta = np.column_stack([np.asarray(X, dtype=float) @ self.coef_ + self.intercept_, np.zeros(X.shape[0])])
        proba = softmax(eta)
        sums = proba.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-8):
            raise RuntimeError("Predicted class probabilities do not sum to one.")
        return proba

    def predict_proba_mi(self, X_list: list[np.ndarray]) -> np.ndarray:
        if not X_list:
            raise ValueError("X_list must contain at least one imputed design matrix.")
        probs = [self.predict_proba(X) for X in X_list]
        out = np.mean(probs, axis=0)
        out = np.clip(out, 1e-15, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def source_norms(self) -> dict[str, float]:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        return {source: float(np.linalg.norm(self.coef_[cols, :], ord="fro")) for source, cols in self.groups.items()}

    def selected_block_count(self) -> int:
        if self.coef_ is None:
            return 0
        return int(sum(norm > 1e-8 for norm in self.source_norms().values()))

    def coefficient_rows(self, feature_names: list[str]) -> list[dict[str, Any]]:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        rows: list[dict[str, Any]] = []
        for idx, feature in enumerate(feature_names):
            row = {"feature_name": feature}
            for class_idx, class_name in enumerate(CLASS_NAMES[:-1]):
                row[f"coef_vs_healthy_{class_name}"] = float(self.coef_[idx, class_idx])
            rows.append(row)
        return rows

    def _intercept_only(self, y: np.ndarray, sample_weight: np.ndarray, k: int) -> np.ndarray:
        counts = np.array([sample_weight[y == code].sum() for code in range(k)], dtype=float)
        probs = np.clip(counts / max(counts.sum(), 1e-12), 1e-12, 1.0)
        return np.log(probs[:-1] / probs[-1])

    def _objective(self, X_list: list[np.ndarray], y: np.ndarray, sample_weight: np.ndarray, b0: np.ndarray, B: np.ndarray) -> float:
        loss = 0.0
        for X in X_list:
            proba = softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
            nll = -np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0))
            loss += float(np.dot(sample_weight, nll) / sample_weight.sum())
        loss /= len(X_list)
        ridge = 0.5 * (1.0 - self.alpha) * float(np.sum(B * B))
        group = 0.0
        if self.alpha > 0:
            for source, cols in self.groups.items():
                group += self.adaptive_weights[source] * np.sqrt(len(cols)) * float(np.linalg.norm(B[cols, :], ord="fro"))
        return loss + self.lambda_ * (ridge + self.alpha * group)

    def _gradient(self, X_list: list[np.ndarray], y: np.ndarray, sample_weight: np.ndarray, b0: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k_minus_ref = B.shape[1]
        grad_b0 = np.zeros(k_minus_ref, dtype=float)
        grad_B = np.zeros_like(B)
        weight_sum = float(sample_weight.sum())
        y_non_ref = np.zeros((len(y), k_minus_ref), dtype=float)
        for klass in range(k_minus_ref):
            y_non_ref[:, klass] = y == klass
        for X in X_list:
            proba = softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
            residual = (proba[:, :k_minus_ref] - y_non_ref) * sample_weight[:, None]
            grad_b0 += residual.sum(axis=0) / weight_sum
            grad_B += X.T @ residual / weight_sum
        grad_b0 /= len(X_list)
        grad_B /= len(X_list)
        grad_B += self.lambda_ * (1.0 - self.alpha) * B
        return grad_b0, grad_B

    def _prox(self, B: np.ndarray, step: float) -> np.ndarray:
        if self.alpha <= 0:
            return B
        out = B.copy()
        for source, cols in self.groups.items():
            block = out[cols, :]
            norm = float(np.linalg.norm(block, ord="fro"))
            threshold = step * self.lambda_ * self.alpha * self.adaptive_weights[source] * np.sqrt(len(cols))
            if norm <= threshold:
                out[cols, :] = 0.0
            else:
                out[cols, :] = (1.0 - threshold / norm) * block
        return out

    def _initial_step(self, X_list: list[np.ndarray], sample_weight: np.ndarray) -> float:
        weighted_norm = 0.0
        for X in X_list:
            weighted_norm += float((sample_weight[:, None] * X * X).sum() / max(sample_weight.sum(), 1e-12))
        weighted_norm /= max(len(X_list), 1)
        lipschitz = 0.25 * max(weighted_norm, 1e-6) + self.lambda_ * max(0.0, 1.0 - self.alpha)
        return min(1.0, 1.0 / max(lipschitz, 1e-8))

def lambda_max_for_adaptive_groups(
    X_list: list[np.ndarray],
    y: np.ndarray,
    groups: dict[str, list[int]],
    alpha: float,
    adaptive_weights: dict[str, float],
    sample_weight: np.ndarray,
) -> float:
    if alpha <= 0:
        return 1.0
    temp = AdaptiveMIGroupedMultinomialElasticNet(alpha=alpha, lambda_=1.0, gamma=1.0, groups=groups, adaptive_weights=adaptive_weights)
    b0 = temp._intercept_only(np.asarray(y, dtype=int), sample_weight, len(CLASS_NAMES))
    B0 = np.zeros((X_list[0].shape[1], len(CLASS_NAMES) - 1), dtype=float)
    _, grad = temp._gradient(X_list, np.asarray(y, dtype=int), sample_weight, b0, B0)
    values = []
    for source, cols in groups.items():
        cols_array = np.array(cols, dtype=int)
        denom = float(alpha) * max(float(adaptive_weights.get(source, 1.0)), 1e-12) * np.sqrt(len(cols_array))
        values.append(float(np.linalg.norm(grad[cols_array, :], ord="fro") / denom))
    return max(values) if values else 1.0

def compute_adaptive_weights(
    X_list: list[np.ndarray],
    y: np.ndarray,
    groups: dict[str, list[int]],
    sample_weight: np.ndarray,
    gamma: float,
    max_iter: int,
    tol: float,
) -> tuple[dict[str, float], AdaptiveMIGroupedMultinomialElasticNet]:
    ridge_lambda = 1e-3
    initial = AdaptiveMIGroupedMultinomialElasticNet(
        alpha=0.0,
        lambda_=ridge_lambda,
        gamma=float(gamma),
        groups=groups,
        adaptive_weights={source: 1.0 for source in groups},
        max_iter=max_iter,
        tol=tol,
    ).fit(X_list, y, sample_weight=sample_weight)
    norms = initial.source_norms()
    epsilon = max(len(y), 1) ** -0.5
    weights = {source: float((norms.get(source, 0.0) + epsilon) ** (-float(gamma))) for source in groups}
    return weights, initial

def constant_proba(y_train: np.ndarray, n_rows: int, sample_weight: np.ndarray | None = None) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=int)
    weights = row_weights(y_train, n_classes=len(CLASS_NAMES)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    counts = np.array([weights[y_train == code].sum() for code in range(len(CLASS_NAMES))], dtype=float)
    probs = counts / max(counts.sum(), 1e-12)
    return np.tile(probs, (int(n_rows), 1))
