"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

def _gradient_from_probabilities_v1(model: Any, X_list: Sequence[np.ndarray], probabilities: Sequence[np.ndarray], sample_weight: np.ndarray, y_non_ref: np.ndarray, weight_sum: float, coefficient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the frozen gradient without repeating its forward passes."""
    k_minus_ref = coefficient.shape[1]
    grad_b0 = np.zeros(k_minus_ref, dtype=float)
    grad_coefficient = np.zeros_like(coefficient)
    for X, probability in zip(X_list, probabilities, strict=True):
        residual = (probability[:, :k_minus_ref] - y_non_ref) * sample_weight[:, None]
        grad_b0 += residual.sum(axis=0) / weight_sum
        grad_coefficient += X.T @ residual / weight_sum
    grad_b0 /= len(X_list)
    grad_coefficient /= len(X_list)
    grad_coefficient += model.lambda_ * (1.0 - model.alpha) * coefficient
    return (grad_b0, grad_coefficient)
