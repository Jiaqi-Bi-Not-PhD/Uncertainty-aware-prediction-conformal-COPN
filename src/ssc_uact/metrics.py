"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from sklearn.metrics import average_precision_score, roc_auc_score

from .probability import multiclass_log_loss, normalize_probabilities

def probability_metrics(y_true: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    p = normalize_probabilities(probabilities)
    if p.ndim == 1:
        p = p[None, :]
    y = np.asarray(y_true, dtype=int)
    if y.shape != (len(p),):
        raise ValueError("y_true and probabilities are misaligned")
    one_hot = np.eye(p.shape[1])[y]
    result = {
        "log_loss": multiclass_log_loss(y, p),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
    }
    if p.shape[1] > 1 and np.unique(y == 1).size == 2:
        result["ap_pr_auc"] = float(average_precision_score(y == 1, p[:, 1]))
    else:
        result["ap_pr_auc"] = float("nan")
    return result

def error_metrics(hard_errors: ArrayLike, error_risk: ArrayLike) -> dict[str, float]:
    errors = np.asarray(hard_errors, dtype=int)
    risk = np.asarray(error_risk, dtype=float)
    if errors.shape != risk.shape or errors.ndim != 1:
        raise ValueError("hard_errors and error_risk must be aligned vectors")
    if np.any((errors < 0) | (errors > 1)) or np.any((risk < 0) | (risk > 1)):
        raise ValueError("errors must be binary and risks must lie in [0, 1]")
    clipped = np.clip(risk, 1e-12, 1 - 1e-12)
    result = {
        "error_log_loss": float(
            -np.mean(errors * np.log(clipped) + (1 - errors) * np.log(1 - clipped))
        ),
        "error_brier": float(np.mean((risk - errors) ** 2)),
        "aurc": aurc(errors, risk),
    }
    if np.unique(errors).size == 2:
        result["error_auroc"] = float(roc_auc_score(errors, risk))
        result["error_auprc"] = float(average_precision_score(errors, risk))
    else:
        result["error_auroc"] = result["error_auprc"] = float("nan")
    return result

def aurc(hard_errors: ArrayLike, error_risk: ArrayLike) -> float:
    errors = np.asarray(hard_errors, dtype=float)
    risk = np.asarray(error_risk, dtype=float)
    if errors.shape != risk.shape or errors.ndim != 1 or len(errors) == 0:
        raise ValueError("hard_errors and error_risk must be nonempty aligned vectors")
    order = np.argsort(risk, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return float(np.mean(cumulative_risk))

def conformal_metrics(
    y_true: ArrayLike, set_masks: ArrayLike
) -> dict[str, float]:
    masks = np.asarray(set_masks, dtype=bool)
    y = np.asarray(y_true, dtype=int)
    if masks.ndim != 2 or y.shape != (len(masks),):
        raise ValueError("labels and set masks are misaligned")
    result = {
        "average_set_size": float(masks.sum(axis=1).mean()),
        "singleton_rate": float(np.mean(masks.sum(axis=1) == 1)),
        "pair_rate": float(np.mean(masks.sum(axis=1) == 2)),
        "full_set_rate": float(np.mean(masks.sum(axis=1) == masks.shape[1])),
    }
    for class_code in range(masks.shape[1]):
        rows = y == class_code
        result[f"coverage_class_{class_code}"] = (
            float(np.mean(masks[rows, class_code])) if np.any(rows) else float("nan")
        )
    return result
