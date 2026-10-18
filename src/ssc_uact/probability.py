"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike, NDArray

DEFAULT_EPSILON = 1e-6

def normalize_probabilities(
    probabilities: ArrayLike, epsilon: float = DEFAULT_EPSILON
) -> NDArray[np.float64]:
    """Clip and renormalize one or more class-probability vectors.

    The returned array has the same dimensionality as the input.  Non-finite
    values, negative epsilon, and rows with fewer than two classes are rejected.
    """

    p = np.asarray(probabilities, dtype=float)
    one_dimensional = p.ndim == 1
    if one_dimensional:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (n, k), k >= 2")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    p = np.clip(p, epsilon, 1.0 - epsilon)
    p /= p.sum(axis=1, keepdims=True)
    return p[0] if one_dimensional else p

def softmax(scores: ArrayLike) -> NDArray[np.float64]:
    """Stable row-wise softmax."""

    z = np.asarray(scores, dtype=float)
    one_dimensional = z.ndim == 1
    if one_dimensional:
        z = z[None, :]
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("scores must be a finite one- or two-dimensional array")
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    result = exp_z / exp_z.sum(axis=1, keepdims=True)
    return result[0] if one_dimensional else result

def multiclass_log_loss(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Unweighted mean multiclass log loss."""

    p = normalize_probabilities(probabilities, epsilon)
    if p.ndim == 1:
        p = p[None, :]
    y = np.asarray(y_true, dtype=int)
    if y.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true must be one-dimensional and align with probabilities")
    if len(y) == 0 or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError("y_true contains an invalid class code")
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))
