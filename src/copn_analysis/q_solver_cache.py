"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

class InvariantCachedCandidateContextV1:
    """One block-local cache with the original solver's exact operation order."""

    def __init__(self, *, adaptive_module: Any, X_list: Sequence[np.ndarray], y: np.ndarray, groups: Mapping[str, Sequence[int]], sample_weight: np.ndarray, adaptive_weights: Mapping[str, float], gamma: float, max_iter: int, tol: float) -> None:
        if not X_list:
            raise ValueError('X_list must contain at least one imputed design matrix.')
        self.adaptive_module = adaptive_module
        self.y = np.asarray(y, dtype=int)
        self.X_list = tuple((np.asarray(X, dtype=float) for X in X_list))
        p = self.X_list[0].shape[1]
        if p == 0:
            raise ValueError('Cannot fit adaptive Model 8 with zero design columns.')
        if any((X.shape != self.X_list[0].shape for X in self.X_list)):
            raise ValueError('All imputed design matrices must have the same shape.')
        if self.X_list[0].shape[0] != len(self.y):
            raise ValueError('Design rows do not match y length.')
        self.sample_weight = np.asarray(sample_weight, dtype=float)
        if self.sample_weight.shape != self.y.shape or np.any(~np.isfinite(self.sample_weight)) or np.any(self.sample_weight <= 0):
            raise ValueError('Sample weights must be finite and positive.')
        self._template = adaptive_module.AdaptiveMIGroupedMultinomialElasticNet(alpha=1.0, lambda_=1.0, gamma=float(gamma), groups=groups, adaptive_weights=adaptive_weights, max_iter=int(max_iter), tol=float(tol))
        self.groups = self._template.groups
        self.group_sqrt_lengths = {source: np.sqrt(len(cols)) for source, cols in self.groups.items()}
        self.row_index = np.arange(len(self.y))
        self.weight_sum = float(self.sample_weight.sum())
        k_minus_ref = len(adaptive_module.CLASS_NAMES) - 1
        self.y_non_ref = np.zeros((len(self.y), k_minus_ref), dtype=float)
        for klass in range(k_minus_ref):
            self.y_non_ref[:, klass] = self.y == klass
        weighted_norm = 0.0
        for X in self.X_list:
            weighted_norm += float((self.sample_weight[:, None] * X * X).sum() / max(self.sample_weight.sum(), 1e-12))
        weighted_norm /= max(len(self.X_list), 1)
        self.weighted_norm = float(weighted_norm)
        self._zero_gradient_data: np.ndarray | None = None
        self._weighted_norm_matrix_scans = len(self.X_list)
        self._zero_gradient_matrix_scans = 0
        self._lambda_max_calls = 0
        self._candidate_fit_calls = 0
        self._objective_calls = 0
        self._gradient_calls = 0
        self._prox_calls = 0

    def _new_model(self, *, alpha: float, lambda_: float, gamma: float) -> Any:
        model = copy.copy(self._template)
        model.alpha = float(alpha)
        model.lambda_ = float(lambda_)
        model.gamma = float(gamma)
        model.intercept_ = None
        model.coef_ = None
        model.fit_diagnostics_ = None
        return model

    def new_candidate_model(self, *, alpha: float, lambda_: float, gamma: float) -> Any:
        """Return one original model sharing only block-invariant structures."""
        return self._new_model(alpha=alpha, lambda_=lambda_, gamma=gamma)

    def _objective(self, model: Any, b0: np.ndarray, B: np.ndarray) -> float:
        self._objective_calls += 1
        loss = 0.0
        for X in self.X_list:
            proba = self.adaptive_module.softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
            nll = -np.log(np.clip(proba[self.row_index, self.y], 1e-15, 1.0))
            loss += float(np.dot(self.sample_weight, nll) / self.weight_sum)
        loss /= len(self.X_list)
        ridge = 0.5 * (1.0 - model.alpha) * float(np.sum(B * B))
        group = 0.0
        if model.alpha > 0:
            for source, cols in self.groups.items():
                group += model.adaptive_weights[source] * self.group_sqrt_lengths[source] * float(np.linalg.norm(B[cols, :], ord='fro'))
        return loss + model.lambda_ * (ridge + model.alpha * group)

    def _gradient_data(self, b0: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k_minus_ref = B.shape[1]
        grad_b0 = np.zeros(k_minus_ref, dtype=float)
        grad_B = np.zeros_like(B)
        for X in self.X_list:
            proba = self.adaptive_module.softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
            residual = (proba[:, :k_minus_ref] - self.y_non_ref) * self.sample_weight[:, None]
            grad_b0 += residual.sum(axis=0) / self.weight_sum
            grad_B += X.T @ residual / self.weight_sum
        grad_b0 /= len(self.X_list)
        grad_B /= len(self.X_list)
        return (grad_b0, grad_B)

    def _initial_step(self, model: Any) -> float:
        lipschitz = 0.25 * max(self.weighted_norm, 1e-06) + model.lambda_ * max(0.0, 1.0 - model.alpha)
        return min(1.0, 1.0 / max(lipschitz, 1e-08))

    def initial_step_for(self, model: Any) -> float:
        """Return the original candidate step from the cached weighted norm."""
        return self._initial_step(model)

    def lambda_max(self, *, alpha: float) -> float:
        self._lambda_max_calls += 1
        if alpha <= 0:
            return 1.0
        b0 = self._template._intercept_only(self.y, self.sample_weight, len(self.adaptive_module.CLASS_NAMES))
        B0 = np.zeros((self.X_list[0].shape[1], len(self.adaptive_module.CLASS_NAMES) - 1), dtype=float)
        if self._zero_gradient_data is None:
            _, gradient = self._gradient_data(b0, B0)
            self._zero_gradient_data = np.array(gradient, dtype=float, order='C', copy=True)
            self._zero_gradient_matrix_scans += len(self.X_list)
        gradient = self._zero_gradient_data.copy()
        gradient += 1.0 * (1.0 - float(alpha)) * B0
        values = []
        for source, cols in self.groups.items():
            denom = float(alpha) * max(float(self._template.adaptive_weights.get(source, 1.0)), 1e-12) * self.group_sqrt_lengths[source]
            values.append(float(np.linalg.norm(gradient[cols, :], ord='fro') / denom))
        return max(values) if values else 1.0
