"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Mapping

import numpy as np

from numpy.typing import ArrayLike, NDArray

@dataclass(frozen=True)
class ScenarioAggregationSpec:
    scenario_ids: tuple[str, ...]
    scenario_weights: tuple[float, ...]
    rho: float
    quantile_method: str = "weighted_ecdf_higher"
    registry_sha256: str = ""

@dataclass(frozen=True)
class ScoreMondrianLAC:
    thresholds: NDArray[np.float64]
    counts: NDArray[np.int64]
    target_ranks: NDArray[np.int64]
    miscoverage: NDArray[np.float64]
    automatic_inclusion: NDArray[np.bool_]
    automatic_inclusion_reason: tuple[str, ...]

@dataclass(frozen=True)
class AggregateSetPrediction:
    aggregate_scores: NDArray[np.float64]
    pre_fallback_mask: NDArray[np.bool_]
    conformal_core_mask: NDArray[np.bool_]
    fallback_flag: NDArray[np.bool_]
    fallback_class: NDArray[np.int64]
    normalized_fallback_scores: NDArray[np.float64]

def weighted_ecdf_higher(
    values: ArrayLike, weights: ArrayLike, rho: float
) -> NDArray[np.float64]:
    """Weighted empirical inverse CDF using a deterministic higher rule.

    ``values`` has shape ``(rows, scenarios, classes)``. Stable sorting keeps
    the implementation deterministic; equal values make the tie order
    immaterial to the selected score.
    """

    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if array.ndim != 3 or array.shape[1] == 0 or array.shape[2] < 2:
        raise ValueError("values must have shape (rows, scenarios, classes)")
    if weight.shape != (array.shape[1],):
        raise ValueError("weights must match the scenario dimension")
    if not np.isfinite(array).all():
        raise ValueError("scenario scores must all be finite")
    if not np.isfinite(weight).all() or np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError("weights must be finite, nonnegative, and not all zero")
    if not np.isfinite(rho) or not 0 < float(rho) <= 1:
        raise ValueError("rho must lie in (0, 1]")
    normalized = weight / weight.sum()
    order = np.argsort(array, axis=1, kind="stable")
    ordered_values = np.take_along_axis(array, order, axis=1)
    expanded_weights = np.broadcast_to(
        normalized[None, :, None], array.shape
    )
    ordered_weights = np.take_along_axis(expanded_weights, order, axis=1)
    cumulative = np.cumsum(ordered_weights, axis=1)
    selected_index = np.argmax(cumulative >= float(rho) - 1e-15, axis=1)
    return np.take_along_axis(
        ordered_values, selected_index[:, None, :], axis=1
    ).squeeze(axis=1)

def aggregate_lac_nonconformity(
    scenario_probabilities: ArrayLike,
    spec: ScenarioAggregationSpec,
    *,
    simplex_atol: float = 1e-7,
) -> NDArray[np.float64]:
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[2] < 2:
        raise ValueError(
            "scenario probabilities must have shape (rows, scenarios, classes)"
        )
    if probabilities.shape[1] != len(spec.scenario_ids):
        raise ValueError("scenario IDs do not match the probability scenario dimension")
    if len(set(spec.scenario_ids)) != len(spec.scenario_ids):
        raise ValueError("scenario IDs must be unique")
    if len(spec.scenario_weights) != len(spec.scenario_ids):
        raise ValueError("scenario weights and IDs are misaligned")
    if spec.quantile_method != "weighted_ecdf_higher":
        raise ValueError("only weighted_ecdf_higher is frozen for v3 Stage 2")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < -simplex_atol) or np.any(probabilities > 1 + simplex_atol):
        raise ValueError("probabilities fall outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=2), 1.0, atol=simplex_atol, rtol=0):
        raise ValueError("probability rows do not sum to one")
    return weighted_ecdf_higher(
        1.0 - probabilities, spec.scenario_weights, spec.rho
    )

def fit_score_mondrian_lac(
    calibration_scores: ArrayLike,
    calibration_labels: ArrayLike,
    miscoverage: ArrayLike,
    *,
    minimum_counts: Mapping[int, int] | None = None,
) -> ScoreMondrianLAC:
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("calibration scores must have shape (rows, classes)")
    if not np.isfinite(scores).all():
        raise ValueError("calibration scores must be finite")
    raw_labels = np.asarray(calibration_labels)
    if raw_labels.shape != (len(scores),):
        raise ValueError("calibration labels and scores are misaligned")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        if not np.issubdtype(raw_labels.dtype, np.number):
            raise ValueError("calibration labels must be integer class codes")
        numeric = raw_labels.astype(float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("calibration labels must be integer class codes")
    labels = raw_labels.astype(np.int64)
    if np.any((labels < 0) | (labels >= scores.shape[1])):
        raise ValueError("calibration labels fall outside the class registry")
    alpha = np.asarray(miscoverage, dtype=float)
    if alpha.shape != (scores.shape[1],) or np.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("miscoverage must have one value in (0, 1) per class")
    minimum = dict(minimum_counts or {})
    if any(int(key) < 0 or int(key) >= scores.shape[1] for key in minimum):
        raise ValueError("minimum-count class code is invalid")
    if any(int(value) < 0 for value in minimum.values()):
        raise ValueError("minimum calibration counts must be nonnegative")

    counts = np.bincount(labels, minlength=scores.shape[1]).astype(np.int64)
    thresholds = np.full(scores.shape[1], np.inf, dtype=float)
    target_ranks = np.empty(scores.shape[1], dtype=np.int64)
    automatic = np.ones(scores.shape[1], dtype=bool)
    reasons: list[str] = []
    for class_code in range(scores.shape[1]):
        class_scores = np.sort(scores[labels == class_code, class_code])
        count = len(class_scores)
        target_rank = int(np.ceil((count + 1) * (1.0 - alpha[class_code])))
        target_ranks[class_code] = target_rank
        if count == 0:
            reasons.append("NO_CALIBRATION_CASES")
        elif count < int(minimum.get(class_code, 0)):
            reasons.append("BELOW_MINIMUM_COUNT")
        elif target_rank > count:
            reasons.append("TARGET_RANK_N_PLUS_ONE")
        else:
            thresholds[class_code] = class_scores[target_rank - 1]
            automatic[class_code] = False
            reasons.append("FINITE_ORDER_STATISTIC")
    return ScoreMondrianLAC(
        thresholds=thresholds,
        counts=counts,
        target_ranks=target_ranks,
        miscoverage=alpha,
        automatic_inclusion=automatic,
        automatic_inclusion_reason=tuple(reasons),
    )

def apply_score_mondrian_lac(
    scores: ArrayLike,
    model: ScoreMondrianLAC,
    *,
    epsilon: float = 1e-12,
) -> AggregateSetPrediction:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(model.thresholds):
        raise ValueError("scores and threshold class dimensions do not match")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    thresholds = np.asarray(model.thresholds, dtype=float)
    if thresholds.shape != (values.shape[1],) or np.isnan(thresholds).any():
        raise ValueError("thresholds are invalid")
    pre_fallback = values <= thresholds[None, :]
    core = pre_fallback.copy()
    fallback = ~core.any(axis=1)
    fallback_class = np.full(len(values), -1, dtype=np.int64)
    normalized = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(thresholds)
    normalized[:, finite] = values[:, finite] / (thresholds[finite] + epsilon)
    if np.any(fallback):
        if not finite.all():
            raise AssertionError("an empty set cannot coexist with an infinite threshold")
        rows = np.flatnonzero(fallback)
        chosen = np.argmin(normalized[rows], axis=1)
        core[rows, chosen] = True
        fallback_class[rows] = chosen
    return AggregateSetPrediction(
        aggregate_scores=values,
        pre_fallback_mask=pre_fallback,
        conformal_core_mask=core,
        fallback_flag=fallback,
        fallback_class=fallback_class,
        normalized_fallback_scores=normalized,
    )
