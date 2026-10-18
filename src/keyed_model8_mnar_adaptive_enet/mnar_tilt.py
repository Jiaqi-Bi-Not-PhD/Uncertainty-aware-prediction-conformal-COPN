"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from keyed_model1.common import numeric_series

from .model8_imputation import MixedChainedImputer

@dataclass
class MNARTiltResult:
    """Tilted imputed designs and row-normalized imputation weights."""

    matrices: list[np.ndarray]
    imputation_weights: np.ndarray
    diagnostics: pd.DataFrame

def _row_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=0, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -40, 40))
    denom = exp_scores.sum(axis=0, keepdims=True)
    denom[denom <= 0] = 1.0
    return exp_scores / denom

def _numeric_missing_mask(X_raw: pd.DataFrame, source_id: str) -> np.ndarray:
    if source_id not in X_raw.columns:
        return np.ones(len(X_raw), dtype=bool)
    return numeric_series(X_raw[source_id]).isna().to_numpy(dtype=bool)

def apply_mnar_covariate_tilt(
    imputer: MixedChainedImputer,
    X_raw: pd.DataFrame,
    matrices: list[np.ndarray],
    delta_global: float,
    delta_scenario_id: str,
    branch: str,
    outer_split_id: str,
    inner_split_id: str,
    budget: int,
) -> MNARTiltResult:
    """Apply practical MNAR tilts to MAR completed designs.

    Numeric/ordinal missing values are shifted on the training-fold standardized
    scale. Imputation weights are computed by a row-wise softmax over the
    numeric missing-value scores; at delta=0 this returns exactly equal weights.
    """

    if not matrices:
        raise ValueError("MNAR tilt requires at least one completed design matrix.")
    delta = float(delta_global)
    tilted = [np.asarray(matrix, dtype=float).copy() for matrix in matrices]
    scores = np.zeros((len(tilted), tilted[0].shape[0]), dtype=float)
    rows: list[dict[str, Any]] = []
    for spec in imputer.specs_:
        cols = imputer.groups_.get(spec.source_id, [])
        if not cols:
            continue
        missing = _numeric_missing_mask(X_raw, spec.source_id)
        if spec.kind != "numeric":
            rows.append(
                {
                    "outer_split_id": outer_split_id,
                    "inner_split_id": inner_split_id,
                    "budget": int(budget),
                    "branch": branch,
                    "delta_scenario_id": delta_scenario_id,
                    "delta_global": delta,
                    "source_id": spec.source_id,
                    "kind": spec.kind,
                    "mnar_tilt_mode": "categorical_tilt_disabled",
                    "missing_count": int(missing.sum()),
                    "before_missing_mean": np.nan,
                    "after_missing_mean": np.nan,
                    "before_missing_variance": np.nan,
                    "after_missing_variance": np.nan,
                }
            )
            continue
        value_col = int(cols[0])
        before_values = []
        after_values = []
        for m, matrix in enumerate(tilted):
            before = matrix[missing, value_col].copy()
            matrix[missing, value_col] = matrix[missing, value_col] + delta
            after = matrix[missing, value_col].copy()
            scores[m, missing] += delta * before
            before_values.extend(before.tolist())
            after_values.extend(after.tolist())
        rows.append(
            {
                "outer_split_id": outer_split_id,
                "inner_split_id": inner_split_id,
                "budget": int(budget),
                "branch": branch,
                "delta_scenario_id": delta_scenario_id,
                "delta_global": delta,
                "source_id": spec.source_id,
                "kind": spec.kind,
                "mnar_tilt_mode": "numeric_shift+importance_weighted_draws",
                "missing_count": int(missing.sum()),
                "before_missing_mean": float(np.mean(before_values)) if before_values else np.nan,
                "after_missing_mean": float(np.mean(after_values)) if after_values else np.nan,
                "before_missing_variance": float(np.var(before_values)) if before_values else np.nan,
                "after_missing_variance": float(np.var(after_values)) if after_values else np.nan,
            }
        )
    if abs(delta) <= 1e-15:
        imputation_weights = np.full((len(tilted), tilted[0].shape[0]), 1.0 / len(tilted), dtype=float)
    else:
        imputation_weights = _row_softmax(scores)
    if not np.allclose(imputation_weights.sum(axis=0), 1.0, atol=1e-10):
        raise RuntimeError("MNAR imputation weights are not row-normalized across imputations.")
    diagnostics = pd.DataFrame(rows)
    diagnostics["imputation_weight_min"] = float(imputation_weights.min()) if imputation_weights.size else np.nan
    diagnostics["imputation_weight_max"] = float(imputation_weights.max()) if imputation_weights.size else np.nan
    diagnostics["imputation_weight_entropy_mean"] = float(
        np.mean(-np.sum(imputation_weights * np.log(np.clip(imputation_weights, 1e-15, 1.0)), axis=0))
    ) if imputation_weights.size else np.nan
    diagnostics["delta_zero_equal_weights"] = bool(abs(delta) <= 1e-15 and np.allclose(imputation_weights, 1.0 / len(tilted)))
    return MNARTiltResult(tilted, imputation_weights, diagnostics)
