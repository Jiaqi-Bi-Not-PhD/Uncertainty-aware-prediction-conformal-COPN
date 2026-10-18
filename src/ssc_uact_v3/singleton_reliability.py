"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

from typing import Sequence

import numpy as np

from numpy.typing import ArrayLike, NDArray

from scipy.optimize import minimize

from scipy.special import expit

from scipy.stats import rankdata

@dataclass(frozen=True)
class ReliabilityFit:
    minimum: NDArray[np.float64]
    feature_range: NDArray[np.float64]
    intercept: float
    coefficients: NDArray[np.float64]
    ridge: float
    converged: bool

@dataclass(frozen=True)
class PolicyThreshold:
    risk_threshold: float
    valid: bool
    accepted_count: int
    eligible_count: int
    accepted_rate: float
    singleton_error: float
    ap_precision: float

def fit_monotone_error_model(
    features: ArrayLike,
    errors: ArrayLike,
    *,
    ridge: float = 0.1,
) -> ReliabilityFit:
    """Fit ridge-logistic error risk with nonnegative feature coefficients."""

    x, y = _validate_binary_training(features, errors)
    if np.unique(y).size != 2:
        raise ValueError("error model requires both error outcomes")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    minimum = x.min(axis=0)
    feature_range = x.max(axis=0) - minimum
    scaled = _scale(x, minimum, feature_range)
    initial_rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    initial = np.r_[np.log(initial_rate / (1 - initial_rate)), np.zeros(x.shape[1])]

    def objective(beta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        risk = expit(beta[0] + scaled @ beta[1:])
        clipped = np.clip(risk, 1e-12, 1 - 1e-12)
        loss = -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
        loss += float(ridge) * float(beta[1:] @ beta[1:])
        residual = risk - y
        gradient = np.r_[
            residual.mean(),
            scaled.T @ residual / len(y) + 2 * float(ridge) * beta[1:],
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
    return ReliabilityFit(
        minimum=minimum.astype(float),
        feature_range=feature_range.astype(float),
        intercept=float(result.x[0]),
        coefficients=np.maximum(result.x[1:], 0.0).astype(float),
        ridge=float(ridge),
        converged=True,
    )

def predict_error_risk(
    model: ReliabilityFit, features: ArrayLike
) -> NDArray[np.float64]:
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None] if len(model.coefficients) == 1 else x[None, :]
    if (
        x.ndim != 2
        or x.shape[1] != len(model.coefficients)
        or not np.isfinite(x).all()
    ):
        raise ValueError("prediction features are invalid or dimensionally incompatible")
    return expit(
        model.intercept
        + _scale(x, model.minimum, model.feature_range) @ model.coefficients
    )

def a3_singleton_target(
    core_masks: ArrayLike, true_codes: ArrayLike
) -> tuple[NDArray[np.bool_], NDArray[np.int8], NDArray[np.int8]]:
    masks = np.asarray(core_masks)
    truth = np.asarray(true_codes)
    if masks.ndim != 1 or truth.shape != masks.shape:
        raise ValueError("core masks and truth are misaligned")
    if not np.issubdtype(masks.dtype, np.integer) or np.any((masks < 1) | (masks > 7)):
        raise ValueError("core masks must be nonempty three-class bit masks")
    if not np.issubdtype(truth.dtype, np.integer) or np.any((truth < 0) | (truth > 2)):
        raise ValueError("truth must use class codes 0, 1, 2")
    eligible = np.isin(masks.astype(np.uint8), np.array([1, 2, 4], dtype=np.uint8))
    label = np.full(len(masks), -1, dtype=np.int8)
    if np.any(eligible):
        label[eligible] = np.log2(masks[eligible].astype(np.uint8)).astype(np.int8)
    error = np.full(len(masks), -1, dtype=np.int8)
    error[eligible] = (label[eligible] != truth[eligible]).astype(np.int8)
    return eligible, label, error

def primary_ranking_scores(
    probabilities: ArrayLike, singleton_labels: ArrayLike
) -> dict[str, NDArray[np.float64]]:
    p = np.asarray(probabilities, dtype=float)
    label = np.asarray(singleton_labels)
    if p.ndim != 2 or p.shape[1] != 3 or label.shape != (len(p),):
        raise ValueError("probabilities and singleton labels are misaligned")
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=1), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError("primary probabilities are invalid")
    if not np.issubdtype(label.dtype, np.integer) or np.any((label < 0) | (label > 2)):
        raise ValueError("singleton labels must use class codes 0, 1, 2")
    ordered = np.sort(p, axis=1)
    entropy = -np.sum(p * np.log(np.clip(p, 1e-15, 1.0)), axis=1) / np.log(3.0)
    return {
        "one_minus_max_primary_probability": 1.0 - p.max(axis=1),
        "one_minus_primary_top_two_margin": 1.0
        - (ordered[:, -1] - ordered[:, -2]),
        "primary_normalized_entropy": entropy,
        "one_minus_primary_probability_of_A3_singleton_label": 1.0
        - p[np.arange(len(p)), label.astype(int)],
    }

def select_general_policy_threshold(
    true_codes: ArrayLike,
    singleton_labels: ArrayLike,
    estimated_error_risk: ArrayLike,
    *,
    error_ceiling: float,
    ap_precision_floor: float,
) -> PolicyThreshold:
    truth = np.asarray(true_codes, dtype=int)
    label = np.asarray(singleton_labels, dtype=int)
    risk = np.asarray(estimated_error_risk, dtype=float)
    if truth.shape != label.shape or truth.shape != risk.shape or truth.ndim != 1:
        raise ValueError("threshold inputs are misaligned")
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError("estimated risks must lie in [0, 1]")
    if np.any((truth < 0) | (truth > 2)) or np.any((label < 0) | (label > 2)):
        raise ValueError("class codes are invalid")
    if not 0 <= error_ceiling <= 1 or not 0 <= ap_precision_floor <= 1:
        raise ValueError("policy limits must lie in [0, 1]")
    errors = label != truth
    candidates = np.unique(np.r_[0.0, risk, 1.0])
    feasible: list[tuple[int, float, float, float]] = []
    for threshold in candidates:
        accepted = risk <= threshold
        if not np.any(accepted):
            continue
        singleton_error = float(errors[accepted].mean())
        ap = accepted & (label == 1)
        ap_precision = float((truth[ap] == 1).mean()) if np.any(ap) else np.nan
        if (
            singleton_error <= float(error_ceiling)
            and np.isfinite(ap_precision)
            and ap_precision >= float(ap_precision_floor)
        ):
            feasible.append(
                (int(accepted.sum()), float(threshold), singleton_error, ap_precision)
            )
    if not feasible:
        return PolicyThreshold(0.0, False, 0, len(risk), 0.0, np.nan, np.nan)
    count, threshold, singleton_error, ap_precision = max(
        feasible, key=lambda row: (row[0], -row[1])
    )
    return PolicyThreshold(
        risk_threshold=threshold,
        valid=True,
        accepted_count=count,
        eligible_count=len(risk),
        accepted_rate=count / len(risk),
        singleton_error=singleton_error,
        ap_precision=ap_precision,
    )

def select_hc_policy_threshold(
    true_codes: ArrayLike,
    singleton_labels: ArrayLike,
    estimated_error_risk: ArrayLike,
    *,
    error_cap: float,
) -> PolicyThreshold:
    """Select the largest risk cutoff satisfying the HC-singleton error cap."""

    truth = np.asarray(true_codes, dtype=int)
    label = np.asarray(singleton_labels, dtype=int)
    risk = np.asarray(estimated_error_risk, dtype=float)
    if truth.shape != label.shape or truth.shape != risk.shape or truth.ndim != 1:
        raise ValueError("HC threshold inputs are misaligned")
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError("estimated risks must lie in [0, 1]")
    if not 0 <= error_cap <= 1:
        raise ValueError("HC error cap must lie in [0, 1]")
    hc = label == 2
    if not np.any(hc):
        return PolicyThreshold(0.0, False, 0, 0, 0.0, np.nan, np.nan)
    feasible: list[tuple[int, float, float]] = []
    for threshold in np.unique(np.r_[0.0, risk[hc], 1.0]):
        accepted = hc & (risk <= threshold)
        if not np.any(accepted):
            continue
        singleton_error = float((truth[accepted] != 2).mean())
        if singleton_error <= float(error_cap):
            feasible.append((int(accepted.sum()), float(threshold), singleton_error))
    if not feasible:
        return PolicyThreshold(0.0, False, 0, int(hc.sum()), 0.0, np.nan, np.nan)
    count, threshold, singleton_error = max(
        feasible, key=lambda row: (row[0], -row[1])
    )
    return PolicyThreshold(
        risk_threshold=threshold,
        valid=True,
        accepted_count=count,
        eligible_count=int(hc.sum()),
        accepted_rate=count / int(hc.sum()),
        singleton_error=singleton_error,
        ap_precision=np.nan,
    )

def score_metrics(
    errors: ArrayLike,
    ranking_risk: ArrayLike,
    estimated_error_risk: ArrayLike,
    *,
    project_keys: Sequence[str],
    outer_split_ids: Sequence[str],
    seed: int,
) -> dict[str, float | int | bool]:
    error = np.asarray(errors, dtype=int)
    ranking = np.asarray(ranking_risk, dtype=float)
    estimated = np.asarray(estimated_error_risk, dtype=float)
    if (
        error.ndim != 1
        or ranking.shape != error.shape
        or estimated.shape != error.shape
        or len(project_keys) != len(error)
        or len(outer_split_ids) != len(error)
    ):
        raise ValueError("score metric inputs are misaligned")
    if len(error) == 0 or np.any((error != 0) & (error != 1)):
        raise ValueError("errors must be a nonempty binary vector")
    if not np.isfinite(ranking).all() or not np.isfinite(estimated).all():
        raise ValueError("risks must be finite")
    clipped = np.clip(estimated, 1e-12, 1 - 1e-12)
    aurc = selective_aurc(
        error,
        ranking,
        project_keys=project_keys,
        outer_split_ids=outer_split_ids,
        seed=seed,
    )
    oracle = selective_aurc(
        error,
        error.astype(float),
        project_keys=project_keys,
        outer_split_ids=outer_split_ids,
        seed=seed,
    )
    intercept, slope, calibration_valid = calibration_intercept_slope(error, clipped)
    return {
        "n": int(len(error)),
        "errors": int(error.sum()),
        "error_rate": float(error.mean()),
        "aurc": aurc,
        "oracle_aurc": oracle,
        "excess_aurc": aurc - oracle,
        "error_brier": float(np.mean((clipped - error) ** 2)),
        "error_log_loss": float(
            -np.mean(error * np.log(clipped) + (1 - error) * np.log(1 - clipped))
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_fit_valid": calibration_valid,
        "error_detection_auroc": binary_auroc(error, ranking),
        "error_detection_auprc": binary_average_precision(
            error,
            ranking,
            project_keys=project_keys,
            outer_split_ids=outer_split_ids,
            seed=seed,
        ),
        "ranking_tie_fraction": tie_fraction(ranking),
    }

def selective_aurc(
    errors: ArrayLike,
    risk: ArrayLike,
    *,
    project_keys: Sequence[str],
    outer_split_ids: Sequence[str],
    seed: int,
) -> float:
    error = np.asarray(errors, dtype=float)
    values = np.asarray(risk, dtype=float)
    if error.shape != values.shape or error.ndim != 1 or len(error) == 0:
        raise ValueError("AURC inputs are invalid")
    order = deterministic_risk_order(
        values,
        project_keys=project_keys,
        outer_split_ids=outer_split_ids,
        seed=seed,
        ascending=True,
    )
    cumulative = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    return float(cumulative.mean())

def deterministic_risk_order(
    risk: ArrayLike,
    *,
    project_keys: Sequence[str],
    outer_split_ids: Sequence[str],
    seed: int,
    ascending: bool,
) -> NDArray[np.int64]:
    values = np.asarray(risk, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("risk must be a finite vector")
    if len(project_keys) != len(values) or len(outer_split_ids) != len(values):
        raise ValueError("tie keys are misaligned")
    hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.sha256(
                    f"{int(seed)}|{key}|{split}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            for key, split in zip(project_keys, outer_split_ids, strict=True)
        ),
        dtype=np.uint64,
        count=len(values),
    )
    primary = values if ascending else -values
    return np.lexsort((hashes, primary)).astype(np.int64)

def calibration_intercept_slope(
    errors: ArrayLike, estimated_risk: ArrayLike
) -> tuple[float, float, bool]:
    y = np.asarray(errors, dtype=float)
    p = np.asarray(estimated_risk, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("calibration inputs are invalid")
    if np.unique(y).size != 2:
        return np.nan, np.nan, False
    clipped = np.clip(p, 1e-10, 1 - 1e-10)
    logit = np.log(clipped / (1 - clipped))

    def objective(beta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        fitted = expit(beta[0] + beta[1] * logit)
        q = np.clip(fitted, 1e-12, 1 - 1e-12)
        loss = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        residual = fitted - y
        gradient = np.array([residual.mean(), np.mean(residual * logit)])
        return float(loss), gradient

    result = minimize(
        objective,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-30.0, 30.0), (-30.0, 30.0)],
    )
    if not result.success or not np.isfinite(result.fun):
        return np.nan, np.nan, False
    return float(result.x[0]), float(result.x[1]), True

def binary_auroc(errors: ArrayLike, risk: ArrayLike) -> float:
    y = np.asarray(errors, dtype=int)
    score = np.asarray(risk, dtype=float)
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return np.nan
    ranks = rankdata(score, method="average")
    return float(
        (ranks[y == 1].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )

def binary_average_precision(
    errors: ArrayLike,
    risk: ArrayLike,
    *,
    project_keys: Sequence[str],
    outer_split_ids: Sequence[str],
    seed: int,
) -> float:
    y = np.asarray(errors, dtype=int)
    score = np.asarray(risk, dtype=float)
    positive = int(y.sum())
    if positive == 0:
        return np.nan
    order = deterministic_risk_order(
        score,
        project_keys=project_keys,
        outer_split_ids=outer_split_ids,
        seed=seed,
        ascending=False,
    )
    ordered = y[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].mean())

def tie_fraction(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("tie fraction needs a nonempty vector")
    return float(1.0 - len(np.unique(array)) / len(array))

def _validate_binary_training(
    features: ArrayLike, errors: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(errors, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),):
        raise ValueError("features and errors must align")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training data must be finite")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("errors must be binary")
    return x, y

def _scale(
    x: NDArray[np.float64],
    minimum: NDArray[np.float64],
    feature_range: NDArray[np.float64],
) -> NDArray[np.float64]:
    denominator = np.where(feature_range > 0, feature_range, 1.0)
    return np.clip((x - minimum) / denominator, 0.0, 1.0)
