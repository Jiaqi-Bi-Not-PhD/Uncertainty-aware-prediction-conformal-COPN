"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from keyed_model1.common import is_observed_value

@dataclass
class OutcomeObservationModel:
    feature_names: list[str]
    center_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float
    l2: float

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = observation_feature_matrix(X_raw, self.feature_names)
        X = (X - self.center_) / self.scale_
        eta = X @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))

    def coefficient_rows(self) -> list[dict[str, Any]]:
        return [
            {"feature_name": name, "coef": float(coef), "intercept": float(self.intercept_) if i == 0 else np.nan}
            for i, (name, coef) in enumerate(zip(self.feature_names, self.coef_))
        ]

def observation_feature_names(source_ids: list[str]) -> list[str]:
    return [f"{source_id}__observed" for source_id in source_ids]

def observation_feature_matrix(X_raw: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    blocks = []
    for feature in feature_names:
        source_id = feature.rsplit("__observed", 1)[0]
        if source_id not in X_raw.columns:
            blocks.append(np.zeros(len(X_raw), dtype=float))
        else:
            blocks.append(X_raw[source_id].map(is_observed_value).to_numpy(dtype=float))
    return np.column_stack(blocks) if blocks else np.zeros((len(X_raw), 0), dtype=float)

def fit_outcome_observation_model(
    X_train_all: pd.DataFrame,
    c_train: np.ndarray,
    source_ids: list[str],
    l2: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> OutcomeObservationModel:
    c = np.asarray(c_train, dtype=float)
    if c.ndim != 1 or len(c) != len(X_train_all):
        raise ValueError("Outcome observation indicator length does not match training rows.")
    feature_names = observation_feature_names(source_ids)
    X = observation_feature_matrix(X_train_all, feature_names)
    if X.shape[1] == 0 or len(np.unique(c)) < 2:
        prior = np.clip(float(c.mean()) if len(c) else 1.0, 1e-4, 1.0 - 1e-4)
        return OutcomeObservationModel(feature_names, np.zeros(X.shape[1]), np.ones(X.shape[1]), np.zeros(X.shape[1]), float(np.log(prior / (1.0 - prior))), l2)
    center = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    Z = (X - center) / scale
    coef = np.zeros(Z.shape[1], dtype=float)
    prior = np.clip(float(c.mean()), 1e-4, 1.0 - 1e-4)
    intercept = float(np.log(prior / (1.0 - prior)))
    step = 1.0 / (0.25 * max(float((Z * Z).sum() / max(len(c), 1)), 1e-6) + float(l2))
    previous = np.inf
    for _ in range(int(max_iter)):
        eta = Z @ coef + intercept
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        residual = p - c
        grad_intercept = float(residual.mean())
        grad_coef = Z.T @ residual / max(len(c), 1) + float(l2) * coef / max(len(c), 1)
        new_intercept = intercept - step * grad_intercept
        new_coef = coef - step * grad_coef
        eta_new = Z @ new_coef + new_intercept
        p_new = np.clip(1.0 / (1.0 + np.exp(-np.clip(eta_new, -35, 35))), 1e-12, 1.0 - 1e-12)
        obj = float(-(c * np.log(p_new) + (1 - c) * np.log(1 - p_new)).mean() + 0.5 * float(l2) * np.sum(new_coef * new_coef) / max(len(c), 1))
        if abs(previous - obj) / max(1.0, abs(previous)) < float(tol):
            coef, intercept = new_coef, new_intercept
            break
        coef, intercept, previous = new_coef, new_intercept, obj
    return OutcomeObservationModel(feature_names, center, scale, coef, float(intercept), float(l2))
