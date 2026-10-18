"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from typing import Any

import numpy as np

import pandas as pd

def _fit_model1(
    workflow: Any,
    config: Any,
    dataset: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    selected: dict[str, Any],
    budget: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    prep = workflow.Model1Preprocessor(config, budget=budget)
    Z_train = prep.fit_transform(X_train, y_train, dataset.metadata)
    Z_score = prep.transform(X_score)
    ap_weight = float(selected.get("atypical_weight_multiplier", 1.0))
    if Z_train.shape[1] == 0:
        proba = workflow._constant_proba(y_train, len(X_score), ap_weight)
        model = None
        lambda_value = np.nan
    else:
        weights = workflow.ap_weighted_row_weights(
            y_train,
            atypical_multiplier=ap_weight,
            n_classes=len(workflow.CLASS_NAMES),
        )
        lambda_max = workflow.lambda_max_for_groups(
            Z_train,
            y_train,
            prep.groups_,
            float(selected["alpha"]),
            sample_weight=weights,
        )
        lambda_value = max(
            float(lambda_max) * float(selected["lambda_fraction"]), 1e-8
        )
        model = workflow.GroupedMultinomialElasticNet(
            alpha=float(selected["alpha"]),
            lambda_=lambda_value,
            groups=prep.groups_,
            max_iter=1200,
            tol=1e-5,
        ).fit(Z_train, y_train, sample_weight=weights)
        proba = model.predict_proba(Z_score)
    fitted = {
        "preprocessor": prep,
        "model": model,
        "config": config,
        "selected": selected,
    }
    diagnostics = {
        "feature_count": int(Z_train.shape[1]),
        "eligible_source_count": int(len(prep.groups_)),
        "lambda": lambda_value,
        "source_norms": model.source_norms() if model is not None else {},
    }
    return proba, fitted, diagnostics

def _fit_model2(
    workflow: Any,
    config: Any,
    dataset: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    selected: dict[str, Any],
    budget: int,
    inner: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    scenario = next(
        s
        for s in workflow.focused_scenarios()
        if str(s.scenario_id) == str(selected["scenario_id"])
    )
    prep = workflow.CatBoostPreprocessor(config, budget=budget)
    train_cb = prep.fit_transform(X_train, y_train, dataset.metadata)
    score_cb = prep.transform(X_score)
    if train_cb.shape[1] == 0:
        proba = np.full((len(X_score), 3), 1.0 / 3.0)
        model = None
        best_iteration = 0
    elif inner:
        result = workflow._train_predict_catboost(
            train_cb,
            y_train,
            score_cb,
            np.zeros(len(X_score), dtype=int),
            prep.cat_feature_indices_,
            scenario,
            config,
            seed=int(config.random_state),
        )
        # _train_predict_catboost uses the validation labels only for CatBoost's
        # early-stopping metric.  The caller replaces this path below with the
        # true validation labels.
        proba = result["proba"]
        model = None
        best_iteration = result["best_iteration"]
    else:
        model, proba, best_iteration = workflow._fit_final_outer_model(
            train_cb,
            y_train,
            score_cb,
            prep.cat_feature_indices_,
            scenario,
            config,
            int(selected["median_best_iteration"]),
        )
    fitted = {
        "preprocessor": prep,
        "model": model,
        "config": config,
        "scenario": scenario,
        "selected": selected,
    }
    return proba, fitted, {
        "feature_count": int(train_cb.shape[1]),
        "best_iteration": int(best_iteration),
    }

def _fit_model3(
    workflow: Any,
    config: Any,
    dataset: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    selected: dict[str, Any],
    budget: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    scenario = next(
        s
        for s in workflow.default_scenarios()
        if str(s.scenario_id) == str(selected["scenario_id"])
    )
    result = workflow.fit_predict_ebm(
        X_train,
        y_train,
        X_score,
        dataset.metadata,
        config,
        budget,
        scenario,
        max_iter=1000,
        random_state=int(config.random_state),
    )
    fitted = {
        "preprocessor": result["preprocessor"],
        "model": result["model"],
        "config": config,
        "scenario": scenario,
        "selected": selected,
    }
    diagnostics = {
        "feature_count": int(result["feature_count"]),
        "selected_interactions": result["selected_interactions"],
    }
    return result["proba"], fitted, diagnostics

def _fit_model5(
    workflow: Any,
    config: Any,
    dataset: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    selected: dict[str, Any],
    budget: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    scenario = next(
        s
        for s in workflow.default_stage_scenarios((budget,))
        if str(s.scenario_id) == str(selected["scenario_id"])
    )
    stage1, stage2 = workflow._fit_two_stage(
        X_train,
        y_train,
        dataset.metadata,
        config,
        scenario,
        max_iter=400,
    )
    proba, q, r = workflow._predict_hierarchical(stage1, stage2, X_score)
    fitted = {
        "stage1": stage1,
        "stage2": stage2,
        "config": config,
        "scenario": scenario,
        "selected": selected,
    }
    diagnostics = {
        "mean_q_disease": float(np.mean(q)),
        "mean_r_ap_given_disease": float(np.mean(r)),
    }
    return proba, fitted, diagnostics

def _fit_model2_inner(
    workflow: Any,
    config: Any,
    dataset: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    y_score: np.ndarray,
    selected: dict[str, Any],
    budget: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scenario = next(
        s
        for s in workflow.focused_scenarios()
        if str(s.scenario_id) == str(selected["scenario_id"])
    )
    prep = workflow.CatBoostPreprocessor(config, budget=budget)
    train_cb = prep.fit_transform(X_train, y_train, dataset.metadata)
    score_cb = prep.transform(X_score)
    if train_cb.shape[1] == 0:
        return np.full((len(X_score), 3), 1.0 / 3.0), {"feature_count": 0}
    result = workflow._train_predict_catboost(
        train_cb,
        y_train,
        score_cb,
        y_score,
        prep.cat_feature_indices_,
        scenario,
        config,
        seed=int(config.random_state),
    )
    return result["proba"], {
        "feature_count": int(train_cb.shape[1]),
        "best_iteration": int(result["best_iteration"]),
    }
