"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

from .ap_tuning import select_ap_threshold_for_folds

from .common import KeyedConfig, make_repeated_stratified_splits, metric_row

from .preprocess_source_additive import SourceAdditivePreprocessor

@dataclass(frozen=True)
class StageScenario:
    budget: int
    stage1_lambda_fraction: float
    stage2_lambda_fraction: float
    gamma_a: float
    alpha: float = 0.5
    gamma_m: float = 2.5

    @property
    def scenario_id(self) -> str:
        return (
            f"J{self.budget}_lf1{self.stage1_lambda_fraction:g}_"
            f"lf2{self.stage2_lambda_fraction:g}_gA{self.gamma_a:g}"
        ).replace(".", "p")

def default_stage_scenarios(budgets: tuple[int, ...]) -> tuple[StageScenario, ...]:
    scenarios: list[StageScenario] = []
    for budget in budgets:
        for stage1_frac in (0.1, 0.01):
            for stage2_frac in (0.1, 0.01):
                for gamma_a in (1.0, 1.5, 2.0):
                    scenarios.append(StageScenario(int(budget), float(stage1_frac), float(stage2_frac), float(gamma_a)))
    return tuple(scenarios)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))

class BinarySourceAdditive:
    """Binary version of the Model 4 source-additive missingness-penalized learner."""

    def __init__(
        self,
        alpha: float,
        lambda_: float,
        value_groups: dict[str, list[int]],
        missing_groups: dict[str, list[int]],
        gamma_m: float,
        max_iter: int = 400,
        tol: float = 1e-5,
    ) -> None:
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.value_groups = {name: np.array(cols, dtype=int) for name, cols in value_groups.items() if cols}
        self.missing_groups = {name: np.array(cols, dtype=int) for name, cols in missing_groups.items() if cols}
        self.gamma_m = float(gamma_m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.value_dim_: int = 0

    def fit(self, X_value: np.ndarray, X_missing: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "BinarySourceAdditive":
        X_value = np.asarray(X_value, dtype=float)
        X_missing = np.asarray(X_missing, dtype=float)
        y = np.asarray(y, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)
        X = np.column_stack([X_value, X_missing])
        if X.shape[1] == 0:
            raise ValueError("BinarySourceAdditive cannot fit zero design columns.")
        self.value_dim_ = X_value.shape[1]
        weight_sum = float(sample_weight.sum())
        beta = np.zeros(X.shape[1], dtype=float)
        p0 = np.clip(np.dot(sample_weight, y) / max(weight_sum, 1e-12), 1e-8, 1.0 - 1e-8)
        b0 = float(np.log(p0 / (1.0 - p0)))
        previous = self._objective(X, y, sample_weight, b0, beta)
        step = self._initial_step(X, sample_weight)
        for _ in range(self.max_iter):
            grad_b0, grad_beta = self._gradient(X, y, sample_weight, b0, beta, weight_sum)
            local = step
            accepted = False
            for _ in range(30):
                candidate_b0 = b0 - local * grad_b0
                candidate_beta = self._prox(beta - local * grad_beta, local)
                current = self._objective(X, y, sample_weight, candidate_b0, candidate_beta)
                if current <= previous + 1e-10:
                    accepted = True
                    break
                local *= 0.5
            if not accepted:
                break
            delta = abs(previous - current) / max(1.0, abs(previous))
            b0, beta, previous, step = candidate_b0, candidate_beta, current, min(local * 1.05, 1.0)
            if delta < self.tol:
                break
        self.intercept_ = float(b0)
        self.coef_ = beta
        return self

    def predict_proba(self, X_value: np.ndarray, X_missing: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.column_stack([np.asarray(X_value, dtype=float), np.asarray(X_missing, dtype=float)])
        p1 = _sigmoid(self.intercept_ + X @ self.coef_)
        return np.column_stack([1.0 - p1, p1])

    def block_norms(self) -> tuple[dict[str, float], dict[str, float]]:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        value_norms = {source: float(np.linalg.norm(self.coef_[cols])) for source, cols in self.value_groups.items()}
        missing_norms = {
            source: float(np.linalg.norm(self.coef_[self.value_dim_ + np.array(cols, dtype=int)]))
            for source, cols in self.missing_groups.items()
        }
        return value_norms, missing_norms

    def _objective(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, b0: float, beta: np.ndarray) -> float:
        p = _sigmoid(b0 + X @ beta)
        nll = -(y * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))
        weighted_nll = float(np.dot(sample_weight, nll) / max(sample_weight.sum(), 1e-12))
        ridge = 0.5 * (1.0 - self.alpha) * float(np.sum(beta[: self.value_dim_] ** 2))
        ridge += 0.5 * (1.0 - self.alpha) * self.gamma_m * float(np.sum(beta[self.value_dim_ :] ** 2))
        group = 0.0
        if self.alpha > 0:
            for cols in self.value_groups.values():
                group += np.sqrt(len(cols)) * float(np.linalg.norm(beta[cols]))
            for cols in self.missing_groups.values():
                shifted = self.value_dim_ + np.array(cols, dtype=int)
                group += self.gamma_m * np.sqrt(len(cols)) * float(np.linalg.norm(beta[shifted]))
        return weighted_nll + self.lambda_ * (ridge + self.alpha * group)

    def _gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        b0: float,
        beta: np.ndarray,
        weight_sum: float,
    ) -> tuple[float, np.ndarray]:
        p = _sigmoid(b0 + X @ beta)
        residual = (p - y) * sample_weight
        grad_b0 = float(residual.sum() / max(weight_sum, 1e-12))
        grad_beta = X.T @ residual / max(weight_sum, 1e-12)
        grad_beta[: self.value_dim_] += self.lambda_ * (1.0 - self.alpha) * beta[: self.value_dim_]
        grad_beta[self.value_dim_ :] += self.lambda_ * (1.0 - self.alpha) * self.gamma_m * beta[self.value_dim_ :]
        return grad_b0, grad_beta

    def _prox(self, beta: np.ndarray, step: float) -> np.ndarray:
        out = beta.copy()
        if self.alpha <= 0:
            return out
        for cols in self.value_groups.values():
            block = out[cols]
            norm = float(np.linalg.norm(block))
            threshold = step * self.lambda_ * self.alpha * np.sqrt(len(cols))
            if norm <= threshold:
                out[cols] = 0.0
            else:
                out[cols] = (1.0 - threshold / norm) * block
        for cols in self.missing_groups.values():
            shifted = self.value_dim_ + np.array(cols, dtype=int)
            block = out[shifted]
            norm = float(np.linalg.norm(block))
            threshold = step * self.lambda_ * self.alpha * self.gamma_m * np.sqrt(len(cols))
            if norm <= threshold:
                out[shifted] = 0.0
            else:
                out[shifted] = (1.0 - threshold / norm) * block
        return out

    def _initial_step(self, X: np.ndarray, sample_weight: np.ndarray) -> float:
        weighted_norm = float((sample_weight[:, None] * X * X).sum() / max(sample_weight.sum(), 1e-12))
        lipschitz_bound = 0.25 * max(weighted_norm, 1e-6) + self.lambda_ * max(0.0, 1.0 - self.alpha) * max(1.0, self.gamma_m)
        return min(1.0, 1.0 / lipschitz_bound)

def lambda_max_binary(
    X_value: np.ndarray,
    X_missing: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    value_groups: dict[str, list[int]],
    missing_groups: dict[str, list[int]],
    alpha: float,
    gamma_m: float,
) -> float:
    if alpha <= 0:
        raise ValueError("lambda_max_binary requires alpha > 0.")
    X = np.column_stack([X_value, X_missing])
    p0 = np.clip(np.dot(sample_weight, y) / max(sample_weight.sum(), 1e-12), 1e-8, 1.0 - 1e-8)
    residual = (np.full(len(y), p0) - y) * sample_weight / max(sample_weight.sum(), 1e-12)
    grad = X.T @ residual
    values = []
    for cols in value_groups.values():
        block = grad[np.array(cols, dtype=int)]
        values.append(float(np.linalg.norm(block) / (alpha * np.sqrt(len(cols)))))
    offset = X_value.shape[1]
    for cols in missing_groups.values():
        block = grad[offset + np.array(cols, dtype=int)]
        values.append(float(np.linalg.norm(block) / (alpha * gamma_m * np.sqrt(len(cols)))))
    return max(values) if values else 1.0

def _binary_weights(y: np.ndarray, positive_multiplier: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    counts[counts <= 0] = 1.0
    total = float(len(y))
    w0 = total / (2.0 * counts[0])
    w1 = total / (2.0 * counts[1]) * float(positive_multiplier)
    return np.where(y == 1, w1, w0).astype(float)

def _fit_binary_stage_with_metadata(
    X_train: pd.DataFrame,
    binary_y_train: np.ndarray,
    X_valid: pd.DataFrame,
    metadata: pd.DataFrame,
    config: KeyedConfig,
    budget: int,
    lambda_fraction: float,
    alpha: float,
    gamma_m: float,
    class_names: tuple[str, str],
    class_codes: tuple[int, int],
    min_thresholds: dict[str, int],
    max_iter: int,
    positive_weight_multiplier: float = 1.0,
) -> dict[str, Any]:
    prep = SourceAdditivePreprocessor(
        config,
        budget=budget,
        class_names=class_names,
        class_codes=class_codes,
        min_observed_per_class=min_thresholds,
    )
    Xv_train, Xm_train = prep.fit_transform(X_train, binary_y_train, metadata)
    Xv_valid = prep.transform_value(X_valid)
    Xm_valid = prep.transform_missingness(X_valid)
    if Xv_train.shape[1] + Xm_train.shape[1] == 0 or len(np.unique(binary_y_train)) < 2:
        p1 = float(np.mean(binary_y_train))
        return {
            "proba": np.tile(np.array([1.0 - p1, p1]), (len(X_valid), 1)),
            "preprocessor": prep,
            "model": None,
            "value_norms": {},
            "missing_norms": {},
            "constant_p1": p1,
        }
    sample_weight = _binary_weights(binary_y_train, positive_multiplier=positive_weight_multiplier)
    lambda_max = lambda_max_binary(
        Xv_train,
        Xm_train,
        binary_y_train,
        sample_weight,
        prep.value_groups_,
        prep.missing_groups_,
        alpha=alpha,
        gamma_m=gamma_m,
    )
    model = BinarySourceAdditive(
        alpha=alpha,
        lambda_=max(lambda_max * lambda_fraction, 1e-8),
        value_groups=prep.value_groups_,
        missing_groups=prep.missing_groups_,
        gamma_m=gamma_m,
        max_iter=max_iter,
    )
    model.fit(Xv_train, Xm_train, binary_y_train, sample_weight)
    proba = model.predict_proba(Xv_valid, Xm_valid)
    value_norms, missing_norms = model.block_norms()
    return {
        "proba": proba,
        "preprocessor": prep,
        "model": model,
        "value_norms": value_norms,
        "missing_norms": missing_norms,
        "constant_p1": None,
    }

def _predict_hierarchical(stage1: dict[str, Any], stage2: dict[str, Any], X_full: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prep1: SourceAdditivePreprocessor = stage1["preprocessor"]
    prep2: SourceAdditivePreprocessor = stage2["preprocessor"]
    model1: BinarySourceAdditive | None = stage1["model"]
    model2: BinarySourceAdditive | None = stage2["model"]
    X1v = prep1.transform_value(X_full)
    X1m = prep1.transform_missingness(X_full)
    X2v = prep2.transform_value(X_full)
    X2m = prep2.transform_missingness(X_full)
    if model1 is None:
        q = np.full(len(X_full), float(stage1["constant_p1"]), dtype=float)
    else:
        q = model1.predict_proba(X1v, X1m)[:, 1]
    if model2 is None:
        r = np.full(len(X_full), float(stage2["constant_p1"]), dtype=float)
    else:
        r = model2.predict_proba(X2v, X2m)[:, 1]
    proba = np.column_stack([q * (1.0 - r), q * r, 1.0 - q])
    return proba, q, r

def _hierarchical_metrics(y_true: np.ndarray, proba: np.ndarray, q: np.ndarray, r: np.ndarray) -> dict[str, Any]:
    base = metric_row(y_true, proba)
    disease_true = (y_true != 2).astype(int)
    disease_auc = float(roc_auc_score(disease_true, q)) if len(np.unique(disease_true)) > 1 else np.nan
    disease_mask = y_true != 2
    ap_true = (y_true[disease_mask] == 1).astype(int)
    if disease_mask.any() and len(np.unique(ap_true)) > 1:
        stage2_auc = float(roc_auc_score(ap_true, r[disease_mask]))
        stage2_ap = float(average_precision_score(ap_true, r[disease_mask]))
    else:
        stage2_auc = np.nan
        stage2_ap = np.nan
    low_q_disease = float(np.mean(q[disease_mask] < 0.5)) if disease_mask.any() else np.nan
    base.update(
        {
            "stage1_disease_auc": disease_auc,
            "stage2_ap_vs_pd_auc": stage2_auc,
            "stage2_ap_vs_pd_pr_auc": stage2_ap,
            "true_disease_low_q_rate": low_q_disease,
        }
    )
    return base

def _build_inner_splits(y_outer_train: np.ndarray, config: KeyedConfig, split_id: str) -> list[dict[str, Any]]:
    seed_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(split_id))
    return make_repeated_stratified_splits(
        y_outer_train,
        n_splits=config.inner_folds,
        n_repeats=1,
        random_state=config.random_state + seed_offset,
        min_ap_in_test=config.min_ap_in_inner_valid,
    )

def _fit_two_stage(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metadata: pd.DataFrame,
    config: KeyedConfig,
    scenario: StageScenario,
    max_iter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    disease_y = (y_train != 2).astype(int)
    stage1 = _fit_binary_stage_with_metadata(
        X_train,
        disease_y,
        X_train,
        metadata,
        config,
        budget=int(scenario.budget),
        lambda_fraction=float(scenario.stage1_lambda_fraction),
        alpha=float(scenario.alpha),
        gamma_m=float(scenario.gamma_m),
        class_names=("Healthy", "Disease"),
        class_codes=(0, 1),
        min_thresholds={"Healthy": 5, "Disease": 10},
        max_iter=max_iter,
    )
    disease_mask = y_train != 2
    stage2_y = (y_train[disease_mask] == 1).astype(int)
    stage2 = _fit_binary_stage_with_metadata(
        X_train.loc[disease_mask].reset_index(drop=True),
        stage2_y,
        X_train,
        metadata,
        config,
        budget=int(scenario.budget),
        lambda_fraction=float(scenario.stage2_lambda_fraction),
        alpha=float(scenario.alpha),
        gamma_m=float(scenario.gamma_m),
        class_names=("Parkinson", "Atypical"),
        class_codes=(0, 1),
        min_thresholds={"Parkinson": 10, "Atypical": 1},
        max_iter=max_iter,
        positive_weight_multiplier=float(scenario.gamma_a),
    )
    return stage1, stage2

def _evaluate_scenario(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    metadata: pd.DataFrame,
    split_id: str,
    scenario: StageScenario,
    config: KeyedConfig,
    max_iter: int,
) -> dict[str, Any]:
    rows = []
    threshold_inputs: list[dict[str, Any]] = []
    for inner in _build_inner_splits(y_outer_train, config, split_id):
        tr = inner["train_idx"]
        va = inner["test_idx"]
        X_train = X_outer_train.iloc[tr].reset_index(drop=True)
        X_valid = X_outer_train.iloc[va].reset_index(drop=True)
        y_train = y_outer_train[tr]
        y_valid = y_outer_train[va]
        stage1, stage2 = _fit_two_stage(X_train, y_train, metadata, config, scenario, max_iter=max_iter)
        proba, q, r = _predict_hierarchical(stage1, stage2, X_valid)
        metrics = _hierarchical_metrics(y_valid, proba, q, r)
        threshold_inputs.append({"inner_split_id": inner["split_id"], "y_true": y_valid, "proba": proba})
        rows.append({"inner_split_id": inner["split_id"], **metrics})
    threshold_summary, threshold_fold_rows = select_ap_threshold_for_folds(threshold_inputs)
    threshold_by_fold = {
        row["inner_split_id"]: {f"thresholded_{key}": value for key, value in row.items() if key != "inner_split_id"}
        for row in threshold_fold_rows
    }
    for row in rows:
        row.update(threshold_by_fold.get(row["inner_split_id"], {}))
    fold_df = pd.DataFrame(rows)
    return {
        "scenario_id": scenario.scenario_id,
        "budget": int(scenario.budget),
        "stage1_lambda_fraction": float(scenario.stage1_lambda_fraction),
        "stage2_lambda_fraction": float(scenario.stage2_lambda_fraction),
        "gamma_a": float(scenario.gamma_a),
        "gamma_m": float(scenario.gamma_m),
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()),
        "mean_log_loss": float(fold_df["log_loss"].mean()),
        "mean_macro_f1": float(fold_df["macro_f1"].mean()),
        "mean_stage2_pr_auc": float(fold_df["stage2_ap_vs_pd_pr_auc"].mean()),
        **threshold_summary,
        "fold_rows": rows,
    }

def _select_scenario(summary: pd.DataFrame) -> pd.Series:
    ranked = summary.sort_values(
        [
            "mean_threshold_balanced_accuracy",
            "mean_threshold_macro_f1",
            "mean_threshold_ap_f1",
            "mean_balanced_accuracy",
            "mean_stage2_pr_auc",
            "mean_log_loss",
            "gamma_a",
            "stage2_lambda_fraction",
            "stage1_lambda_fraction",
        ],
        ascending=[False, False, False, False, False, True, False, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]
