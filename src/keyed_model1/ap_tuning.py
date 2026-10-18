"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import math

from typing import Any

import numpy as np

from .common import CLASS_NAMES, balanced_accuracy_from_confusion, confusion_matrix_counts, macro_f1_from_confusion, multiclass_log_loss, normalize_text, row_weights

AP_CLASS_CODE = 1

DEFAULT_AP_THRESHOLDS = tuple(float(x) for x in np.round(np.arange(0.20, 0.81, 0.05), 2))

def _is_raw_argmax(ap_threshold: float | None) -> bool:

    if ap_threshold is None:
        return True
    try:
        return math.isnan(float(ap_threshold))
    except (TypeError, ValueError):
        return False

def ap_threshold_value(ap_threshold: float | None) -> float:
    return float("nan") if _is_raw_argmax(ap_threshold) else float(ap_threshold)

def ap_threshold_label(ap_threshold: float | None) -> str:
    return "argmax" if _is_raw_argmax(ap_threshold) else f"ap_ge_{float(ap_threshold):.2f}"

def ap_weighted_row_weights(y: np.ndarray, atypical_multiplier: float = 1.0, n_classes: int = 3) -> np.ndarray:
    weights = row_weights(y, n_classes=n_classes)
    if float(atypical_multiplier) != 1.0:
        weights[np.asarray(y, dtype=int) == AP_CLASS_CODE] *= float(atypical_multiplier)
    return weights

def predict_with_ap_threshold(proba: np.ndarray, ap_threshold: float | None = None) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if _is_raw_argmax(ap_threshold):
        return np.argmax(proba, axis=1).astype(int)
    threshold = float(ap_threshold)
    non_ap_pred = np.where(proba[:, 0] >= proba[:, 2], 0, 2)
    return np.where(proba[:, AP_CLASS_CODE] >= threshold, AP_CLASS_CODE, non_ap_pred).astype(int)

def metric_row_with_ap_threshold(y_true: np.ndarray, proba: np.ndarray, ap_threshold: float | None = None) -> dict[str, Any]:
    pred = predict_with_ap_threshold(proba, ap_threshold)
    matrix = confusion_matrix_counts(y_true, pred, n_classes=proba.shape[1])
    row: dict[str, Any] = {
        "n": int(len(y_true)),
        "balanced_accuracy": balanced_accuracy_from_confusion(matrix),
        "macro_f1": macro_f1_from_confusion(matrix),
        "log_loss": multiclass_log_loss(y_true, proba),
        "ap_threshold": ap_threshold_value(ap_threshold),
        "pred_rule": ap_threshold_label(ap_threshold),
    }
    for code, label in enumerate(CLASS_NAMES):
        denom = matrix[code, :].sum()
        row[f"recall_{normalize_text(label)}"] = float(matrix[code, code] / denom) if denom else np.nan
    tp = matrix[AP_CLASS_CODE, AP_CLASS_CODE]
    pred_ap = matrix[:, AP_CLASS_CODE].sum()
    true_ap = matrix[AP_CLASS_CODE, :].sum()
    ap_precision = float(tp / pred_ap) if pred_ap else 0.0
    ap_recall = float(tp / true_ap) if true_ap else np.nan
    row["ap_precision"] = ap_precision
    row["ap_f1"] = float(2 * ap_precision * ap_recall / (ap_precision + ap_recall)) if ap_precision + ap_recall > 0 else 0.0
    return row

def select_ap_threshold_for_folds(
    fold_predictions: list[dict[str, Any]],
    thresholds: tuple[float, ...] = DEFAULT_AP_THRESHOLDS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[float | None] = [None, *thresholds]
    summaries: list[dict[str, Any]] = []
    fold_metrics_by_candidate: dict[float, list[dict[str, Any]]] = {}
    for threshold in candidates:
        candidate_fold_rows = []
        for fold in fold_predictions:
            metrics = metric_row_with_ap_threshold(fold["y_true"], fold["proba"], threshold)
            candidate_fold_rows.append({"inner_split_id": fold["inner_split_id"], **metrics})
        threshold_key = ap_threshold_value(threshold)
        fold_metrics_by_candidate[threshold_key] = candidate_fold_rows
        summaries.append(
            {
                "selected_ap_threshold": threshold_key,
                "selected_pred_rule": ap_threshold_label(threshold),
                "mean_threshold_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in candidate_fold_rows])),
                "mean_threshold_macro_f1": float(np.mean([row["macro_f1"] for row in candidate_fold_rows])),
                "mean_threshold_log_loss": float(np.mean([row["log_loss"] for row in candidate_fold_rows])),
                "mean_threshold_ap_precision": float(np.mean([row["ap_precision"] for row in candidate_fold_rows])),
                "mean_threshold_ap_f1": float(np.mean([row["ap_f1"] for row in candidate_fold_rows])),
                "mean_threshold_recall_atypical": float(np.mean([row["recall_atypical"] for row in candidate_fold_rows])),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["mean_threshold_balanced_accuracy"],
            row["mean_threshold_macro_f1"],
            row["mean_threshold_ap_f1"],
            1 if row["selected_pred_rule"] == "argmax" else 0,
        ),
        reverse=True,
    )
    selected = summaries[0]
    return selected, fold_metrics_by_candidate[selected["selected_ap_threshold"]]
