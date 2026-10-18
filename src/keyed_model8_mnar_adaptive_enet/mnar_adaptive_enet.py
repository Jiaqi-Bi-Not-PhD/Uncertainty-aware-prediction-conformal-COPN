"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from keyed_model1.common import CLASS_NAMES

from .model8_adaptive_enet import AdaptiveMIGroupedMultinomialElasticNet, compute_adaptive_weights, constant_proba, lambda_max_for_adaptive_groups, softmax

def _validate_imputation_weights(imputation_weights: np.ndarray | None, m: int, n: int) -> np.ndarray | None:
    if imputation_weights is None:
        return None
    weights = np.asarray(imputation_weights, dtype=float)
    if weights.shape != (m, n):
        raise ValueError(f"imputation_weights must have shape {(m, n)}, got {weights.shape}.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("imputation_weights must be finite and nonnegative.")
    column_sums = weights.sum(axis=0)
    if not np.allclose(column_sums, 1.0, atol=1e-8):
        raise ValueError("imputation_weights must sum to one across imputations for each row.")
    return weights

class AdaptiveMNARMIGroupedMultinomialElasticNet(AdaptiveMIGroupedMultinomialElasticNet):
    """Model 8 grouped elastic net with MNAR imputation-weighted loss."""

    def fit(
        self,
        X_list: list[np.ndarray],
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        init_intercept: np.ndarray | None = None,
        init_coef: np.ndarray | None = None,
        imputation_weights: np.ndarray | None = None,
    ) -> "AdaptiveMNARMIGroupedMultinomialElasticNet":
        self.imputation_weights_ = _validate_imputation_weights(imputation_weights, len(X_list), len(y))
        super().fit(
            X_list,
            y,
            sample_weight=sample_weight,
            init_intercept=init_intercept,
            init_coef=init_coef,
        )
        return self

    def predict_proba_mi_weighted(self, X_list: list[np.ndarray], imputation_weights: np.ndarray | None = None) -> np.ndarray:
        if not X_list:
            raise ValueError("X_list must contain at least one imputed design matrix.")
        probs = [self.predict_proba(X) for X in X_list]
        if imputation_weights is None:
            out = np.mean(probs, axis=0)
        else:
            weights = _validate_imputation_weights(imputation_weights, len(X_list), X_list[0].shape[0])
            out = np.zeros_like(probs[0])
            for m, proba in enumerate(probs):
                out += proba * weights[m, :, None]
        out = np.clip(out, 1e-15, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        if not np.allclose(out.sum(axis=1), 1.0, atol=1e-8):
            raise RuntimeError("Weighted MI probabilities do not sum to one.")
        return out

    def _objective(self, X_list: list[np.ndarray], y: np.ndarray, sample_weight: np.ndarray, b0: np.ndarray, B: np.ndarray) -> float:
        loss = 0.0
        mi_weights = getattr(self, "imputation_weights_", None)
        if mi_weights is None:
            mi_weights = np.full((len(X_list), len(y)), 1.0 / len(X_list), dtype=float)
        for m, X in enumerate(X_list):
            proba = softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
            nll = -np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0))
            loss += float(np.dot(sample_weight * mi_weights[m], nll) / sample_weight.sum())
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
        mi_weights = getattr(self, "imputation_weights_", None)
        if mi_weights is None:
            mi_weights = np.full((len(X_list), len(y)), 1.0 / len(X_list), dtype=float)
        for m, X in enumerate(X_list):
            proba = softmax(np.column_stack([X @ B + b0, np.zeros(X.shape[0])]))
            combined_weight = sample_weight * mi_weights[m]
            residual = (proba[:, :k_minus_ref] - y_non_ref) * combined_weight[:, None]
            grad_b0 += residual.sum(axis=0) / weight_sum
            grad_B += X.T @ residual / weight_sum
        grad_B += self.lambda_ * (1.0 - self.alpha) * B
        return grad_b0, grad_B

def lambda_max_for_mnar_adaptive_groups(
    X_list: list[np.ndarray],
    y: np.ndarray,
    groups: dict[str, list[int]],
    alpha: float,
    adaptive_weights: dict[str, float],
    sample_weight: np.ndarray,
    imputation_weights: np.ndarray | None,
) -> float:
    if imputation_weights is None:
        return lambda_max_for_adaptive_groups(X_list, y, groups, alpha, adaptive_weights, sample_weight)
    if alpha <= 0:
        return 1.0
    temp = AdaptiveMNARMIGroupedMultinomialElasticNet(alpha=alpha, lambda_=1.0, gamma=1.0, groups=groups, adaptive_weights=adaptive_weights)
    temp.imputation_weights_ = _validate_imputation_weights(imputation_weights, len(X_list), len(y))
    b0 = temp._intercept_only(np.asarray(y, dtype=int), sample_weight, len(CLASS_NAMES))
    B0 = np.zeros((X_list[0].shape[1], len(CLASS_NAMES) - 1), dtype=float)
    _, grad = temp._gradient(X_list, np.asarray(y, dtype=int), sample_weight, b0, B0)
    values = []
    for source, cols in groups.items():
        cols_array = np.array(cols, dtype=int)
        denom = float(alpha) * max(float(adaptive_weights.get(source, 1.0)), 1e-12) * np.sqrt(len(cols_array))
        values.append(float(np.linalg.norm(grad[cols_array, :], ord="fro") / denom))
    return max(values) if values else 1.0

def compute_adaptive_weights_mnar(
    X_list: list[np.ndarray],
    y: np.ndarray,
    groups: dict[str, list[int]],
    sample_weight: np.ndarray,
    gamma: float,
    max_iter: int,
    tol: float,
    imputation_weights: np.ndarray | None,
) -> tuple[dict[str, float], AdaptiveMNARMIGroupedMultinomialElasticNet]:
    if imputation_weights is None:
        weights, initial = compute_adaptive_weights(X_list, y, groups, sample_weight, gamma, max_iter, tol)
        wrapped = AdaptiveMNARMIGroupedMultinomialElasticNet(0.0, 1e-3, gamma, groups, {source: 1.0 for source in groups}, max_iter=max_iter, tol=tol)
        wrapped.intercept_ = initial.intercept_
        wrapped.coef_ = initial.coef_
        wrapped.fit_diagnostics_ = initial.fit_diagnostics_
        return weights, wrapped
    ridge_lambda = 1e-3
    initial = AdaptiveMNARMIGroupedMultinomialElasticNet(
        alpha=0.0,
        lambda_=ridge_lambda,
        gamma=float(gamma),
        groups=groups,
        adaptive_weights={source: 1.0 for source in groups},
        max_iter=max_iter,
        tol=tol,
    ).fit(X_list, y, sample_weight=sample_weight, imputation_weights=imputation_weights)
    norms = initial.source_norms()
    epsilon = max(len(y), 1) ** -0.5
    weights = {source: float((norms.get(source, 0.0) + epsilon) ** (-float(gamma))) for source in groups}
    return weights, initial
