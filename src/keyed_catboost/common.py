"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import re

import unicodedata

from dataclasses import dataclass, field

from typing import Any

import numpy as np

import pandas as pd

KEY_COL = "Project key"

OUTCOME_SHEET = "Outcomes"

OUTCOME_COL = "FINAL DISEASE DIAGNOSIS"

OUTCOME_MAP = {"Parkinson": 0, "Atypical": 1, "Healthy": 2}

CLASS_NAMES = ("Parkinson", "Atypical", "Healthy")

NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "missing", "."}

DEFAULT_GROUP_LEVELS = {
    "enrollement": 1,
    "enrollment": 1,
    "demographic": 1,
    "epidemiological": 1,
    "bdii": 1,
    "bai": 1,
    "ehi": 1,
    "fatigue_severity_scale": 1,
    "parkinson_severity_scale": 1,
    "pdq_8": 1,
    "pdq_39": 1,
    "schwab_england": 2,
    "schwab_and_england": 2,
    "scopa": 1,
    "apathy_scale": 1,
    "apathy_evaluation_self": 1,
    "apathy_evaluation_informant": 1,
    "frsbe": 2,
    "clinical": 2,
    "medication": 2,
    "mbic_joined": 2,
    "mbic": 2,
    "moca_joined": 3,
    "moca": 3,
    "mds_updrs_joined": 3,
    "mds_updrs": 3,
    "updrs_1_2_part_3": 3,
    "timed_up_go": 3,
    "neuropsychological_joined": 4,
    "neuropsychological": 4,
}

DEFAULT_EXCLUDE_PATTERNS = (
    r"\bproject_key\b",
    r"\bevent_name\b",
    r"\bcomplete\b",
    r"\bunnamed\b",
    r"\bfinal_disease_diagnosis\b",
    r"\bspecific_disease\b",
    r"\bdiagnosis\b",
    r"\bdiagnostic\b",
    r"\bpd_plus\b",
    r"\batypical_parkinsonism\b",
    r"\bhealthy_control\b",
    r"\bsite\b",
    r"\brecord_id\b",
    r"\bparticipant_id\b",
    r"\bsubject_id\b",
    r"\bredcap\b",
    r"\bcoordinator\b",
    r"\bpersonnel\b",
    r"\bdate\b",
    r"\btimestamp\b",
)

@dataclass(frozen=True)
class KeyedConfig:
    workbook_path: str = ""
    data_dictionary_path: str = ""
    output_dir: str = ""
    key_col: str = KEY_COL
    outcome_sheet: str = OUTCOME_SHEET
    outcome_col: str = OUTCOME_COL
    outcome_map: dict[str, int] = field(default_factory=lambda: dict(OUTCOME_MAP))
    required_sheets: list[str] | None = None
    predictor_sheets: list[str] | None = None
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    group_level_defaults: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_GROUP_LEVELS))
    budgets: tuple[int, ...] = (1, 2, 3, 4)
    min_observed_fraction: float = 0.02
    min_observed_per_class: dict[str, int] = field(
        default_factory=lambda: {"Parkinson": 10, "Atypical": 1, "Healthy": 5}
    )
    rare_category_min_count: int = 5
    max_categories: int = 40
    numeric_parse_threshold: float = 0.90
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    quasi_constant_threshold: float = 0.995
    outer_folds: int = 5
    outer_repeats: int = 3
    inner_folds: int = 4
    random_state: int = 20260421
    min_ap_in_outer_test: int = 5
    min_ap_in_inner_valid: int = 1
    parallel_jobs: int = 1
    catboost_iterations: int = 3000
    catboost_early_stopping_rounds: int = 100
    catboost_threads_per_fit: int = 4

def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")

def is_observed_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return False
    return True

def clean_category(value: Any, missing_token: str = "__MISSING__") -> str:
    if not is_observed_value(value):
        return missing_token
    token = str(value).strip()
    return token if token else missing_token

def parse_numeric(value: Any) -> float:
    if not is_observed_value(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    token = str(value).strip()
    lowered = token.lower()
    ordinal_map = {
        "checked": 1.0,
        "unchecked": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "true": 1.0,
        "false": 0.0,
        "present": 1.0,
        "absent": 0.0,
    }
    if lowered in ordinal_map:
        return ordinal_map[lowered]
    leading = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:-|:|=|$)", token)
    if leading:
        return float(leading.group(1))
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token):
        return float(token)
    return np.nan

def numeric_series(series: pd.Series) -> pd.Series:
    return series.map(parse_numeric).astype(float)

def class_weights(y: np.ndarray, n_classes: int = 3) -> dict[int, float]:
    weights = {}
    n = len(y)
    for k in range(n_classes):
        count = int(np.sum(y == k))
        if count <= 0:
            raise ValueError(f"Training split has no samples for class code {k}.")
        weights[k] = n / (n_classes * count)
    return weights

def make_repeated_stratified_splits(
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    min_ap_in_test: int = 1,
) -> list[dict[str, Any]]:
    y = np.asarray(y)
    splits = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(random_state + repeat * 1009)
        fold_bins = [[] for _ in range(n_splits)]
        for klass in sorted(np.unique(y)):
            idx = np.flatnonzero(y == klass)
            rng.shuffle(idx)
            for pos, sample_idx in enumerate(idx):
                fold_bins[pos % n_splits].append(int(sample_idx))
        all_idx = np.arange(len(y))
        for fold, test_list in enumerate(fold_bins):
            test_idx = np.array(sorted(test_list), dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_idx[train_mask]
            ap_count = int(np.sum(y[test_idx] == OUTCOME_MAP["Atypical"]))
            if ap_count < min_ap_in_test:
                raise ValueError(
                    f"Split repeat={repeat} fold={fold} has only {ap_count} Atypical samples."
                )
            splits.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": f"r{repeat:02d}_f{fold:02d}",
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    return splits

def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix

def balanced_accuracy_from_confusion(matrix: np.ndarray) -> float:
    recalls = []
    for k in range(matrix.shape[0]):
        denom = matrix[k, :].sum()
        recalls.append(matrix[k, k] / denom if denom else np.nan)
    return float(np.nanmean(recalls))

def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    scores = []
    for k in range(matrix.shape[0]):
        tp = matrix[k, k]
        fp = matrix[:, k].sum() - tp
        fn = matrix[k, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))

def multiclass_log_loss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    clipped = np.clip(proba, eps, 1.0 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true.astype(int)])))

def metric_row(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    matrix = confusion_matrix_counts(y_true, pred, n_classes=proba.shape[1])
    row: dict[str, Any] = {
        "n": int(len(y_true)),
        "balanced_accuracy": balanced_accuracy_from_confusion(matrix),
        "macro_f1": macro_f1_from_confusion(matrix),
        "log_loss": multiclass_log_loss(y_true, proba),
    }
    for code, label in enumerate(CLASS_NAMES):
        denom = matrix[code, :].sum()
        row[f"recall_{normalize_text(label)}"] = float(matrix[code, code] / denom) if denom else np.nan
    return row
