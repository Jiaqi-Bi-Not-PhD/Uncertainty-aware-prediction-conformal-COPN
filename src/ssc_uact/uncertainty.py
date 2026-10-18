"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike, NDArray

from .probability import normalize_probabilities

def unit_interval_roundoff_tolerance(n_terms: int) -> float:
    """Return a conservative float64 bound for an ``n_terms`` reduction.

    ``gamma_n = n * eps / (1 - n * eps)`` is a conservative gamma-style
    bound using NumPy's machine epsilon (twice the usual unit roundoff).
    The additional factor of eight covers weight normalization and the
    subsequent dot-product reduction, including implementation-dependent
    BLAS reduction order.
    """

    if isinstance(n_terms, (bool, np.bool_)) or not isinstance(
        n_terms, (int, np.integer)
    ):
        raise ValueError("n_terms must be a positive integer")
    count = int(n_terms)
    if count <= 0:
        raise ValueError("n_terms must be a positive integer")
    scaled_epsilon = count * np.finfo(np.float64).eps
    if scaled_epsilon >= 1.0:
        raise ValueError("n_terms is too large for a finite float64 error bound")
    gamma_n = scaled_epsilon / (1.0 - scaled_epsilon)
    return float(8.0 * gamma_n)

def sanitize_unit_interval(
    values: ArrayLike,
    *,
    n_terms: int,
    name: str = "values",
) -> NDArray[np.float64]:
    """Validate a computed unit-interval quantity and clip only roundoff.

    Values outside the interval by no more than the conservative float64
    reduction bound are canonicalized to exactly zero or one.  Non-finite
    values and substantive interval violations remain fail-closed.
    """

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    tolerance = unit_interval_roundoff_tolerance(n_terms)
    if np.any(array < -tolerance) or np.any(array > 1.0 + tolerance):
        raise ValueError(
            f"{name} is outside [0, 1] beyond float64 roundoff tolerance"
        )
    return np.clip(array, 0.0, 1.0)

def predictive_entropy(probabilities: ArrayLike) -> NDArray[np.float64]:
    p = _as_rows(probabilities)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])

def top_two_ambiguity(probabilities: ArrayLike) -> NDArray[np.float64]:
    p = _as_rows(probabilities)
    ordered = np.sort(p, axis=1)
    return 1.0 - (ordered[:, -1] - ordered[:, -2])

def total_variation(first: ArrayLike, second: ArrayLike) -> NDArray[np.float64]:
    p = _as_rows(first)
    q = _as_rows(second)
    if p.shape != q.shape:
        raise ValueError("probability arrays must have identical shapes")
    return 0.5 * np.sum(np.abs(p - q), axis=1)

def js_disagreement(
    probability_stack: ArrayLike, weights: ArrayLike | None = None
) -> NDArray[np.float64]:
    stack = np.asarray(probability_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("probability_stack must have shape (rows, models, classes)")
    for model in range(stack.shape[1]):
        stack[:, model, :] = normalize_probabilities(stack[:, model, :])
    if weights is None:
        w = np.full(stack.shape[1], 1.0 / stack.shape[1])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (stack.shape[1],) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be nonnegative and match the model count")
        w = w / w.sum()
    mean = np.sum(stack * w[None, :, None], axis=1)
    h_mean = -np.sum(mean * np.log(mean), axis=1)
    h_models = -np.sum(stack * np.log(stack), axis=2)
    result = (h_mean - np.sum(h_models * w[None, :], axis=1)) / np.log(
        stack.shape[2]
    )
    return np.maximum(result, 0.0)

def information_deficit(
    deficit_scores: ArrayLike, source_weights: ArrayLike
) -> NDArray[np.float64]:
    scores = np.asarray(deficit_scores, dtype=float)
    weights = np.asarray(source_weights, dtype=float)
    if scores.ndim != 2 or weights.shape != (scores.shape[1],):
        raise ValueError("scores must be rows by sources and weights must match sources")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("deficit scores must be finite and lie in [0, 1]")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("source weights must be finite, nonnegative, and nonzero")
    values = scores @ (weights / weights.sum())
    return sanitize_unit_interval(
        values,
        n_terms=scores.shape[1],
        name="information deficit",
    )

def shortcut_similarity(
    clinical_probabilities: ArrayLike, missingness_control_probabilities: ArrayLike
) -> NDArray[np.float64]:
    return 1.0 - total_variation(
        clinical_probabilities, missingness_control_probabilities
    )

def mechanism_sensitivity(
    scenario_probabilities: ArrayLike, anchor_index: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return maximum TV from anchor plus classwise scenario minima/maxima.

    ``scenario_probabilities`` has shape ``(rows, scenarios, classes)``.
    """

    scenarios = np.asarray(scenario_probabilities, dtype=float)
    if scenarios.ndim != 3 or not 0 <= anchor_index < scenarios.shape[1]:
        raise ValueError("scenario probabilities or anchor_index are invalid")
    for scenario in range(scenarios.shape[1]):
        scenarios[:, scenario, :] = normalize_probabilities(
            scenarios[:, scenario, :]
        )
    anchor = scenarios[:, anchor_index, :]
    tv = 0.5 * np.sum(np.abs(scenarios - anchor[:, None, :]), axis=2)
    return tv.max(axis=1), scenarios.min(axis=1), scenarios.max(axis=1)

def _as_rows(probabilities: ArrayLike) -> NDArray[np.float64]:
    p = normalize_probabilities(probabilities)
    return p[None, :] if p.ndim == 1 else p
