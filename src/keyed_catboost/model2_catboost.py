"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from .ap_tuning import ap_weighted_class_weights, select_ap_threshold_for_folds

from .common import CLASS_NAMES, KeyedConfig, make_repeated_stratified_splits, metric_row

from .preprocess_catboost import CatBoostPreprocessor

@dataclass(frozen=True)
class CatBoostScenario:
    scenario_id: str
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    bagging_temperature: float
    random_strength: float
    border_count: int
    min_data_in_leaf: int = 1
    bootstrap_type: str = "Bayesian"
    subsample: float | None = None
    atypical_weight_multiplier: float = 1.0

    def params(self, config: KeyedConfig, seed: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": int(config.catboost_iterations),
            "depth": int(self.depth),
            "learning_rate": float(self.learning_rate),
            "l2_leaf_reg": float(self.l2_leaf_reg),
            "random_strength": float(self.random_strength),
            "border_count": int(self.border_count),
            "min_data_in_leaf": int(self.min_data_in_leaf),
            "bootstrap_type": self.bootstrap_type,
            "random_seed": int(seed),
            "thread_count": int(config.catboost_threads_per_fit),
            "verbose": False,
            "allow_writing_files": False,
        }
        if self.bootstrap_type == "Bayesian":
            params["bagging_temperature"] = float(self.bagging_temperature)
        elif self.subsample is not None:
            params["subsample"] = float(self.subsample)
        return params

    @property
    def complexity_rank(self) -> tuple[Any, ...]:
        return (self.depth, self.l2_leaf_reg * -1, self.border_count, self.learning_rate, self.atypical_weight_multiplier)

def focused_scenarios() -> tuple[CatBoostScenario, ...]:
    base = (
        ("d4_lr0.02_l2_10_bayesian_b1_rs1_bc128_mdl1", 4, 0.02, 10, 1, 1, 128, 1),
        ("d4_lr0.03_l2_30_bayesian_b1_rs1_bc128_mdl5", 4, 0.03, 30, 1, 1, 128, 5),
        ("d5_lr0.02_l2_30_bayesian_b3_rs2_bc128_mdl5", 5, 0.02, 30, 3, 2, 128, 5),
        ("d6_lr0.02_l2_60_bayesian_b3_rs2_bc254_mdl10", 6, 0.02, 60, 3, 2, 254, 10),
        ("d3_lr0.06_l2_10_bayesian_b1_rs05_bc64_mdl5", 3, 0.06, 10, 1, 0.5, 64, 5),
        ("d4_lr0.06_l2_30_bayesian_b3_rs1_bc128_mdl10", 4, 0.06, 30, 3, 1, 128, 10),
        ("d5_lr0.03_l2_60_bayesian_b1_rs2_bc254_mdl10", 5, 0.03, 60, 1, 2, 254, 10),
        ("d6_lr0.01_l2_60_bayesian_b5_rs4_bc254_mdl25", 6, 0.01, 60, 5, 4, 254, 25),
        ("d3_lr0.03_l2_3_bayesian_b1_rs1_bc64_mdl1", 3, 0.03, 3, 1, 1, 64, 1),
        ("d4_lr0.01_l2_30_bayesian_b3_rs05_bc128_mdl5", 4, 0.01, 30, 3, 0.5, 128, 5),
        ("d5_lr0.06_l2_60_bayesian_b1_rs2_bc128_mdl25", 5, 0.06, 60, 1, 2, 128, 25),
        ("d6_lr0.03_l2_10_bayesian_b3_rs4_bc254_mdl5", 6, 0.03, 10, 3, 4, 254, 5),
    )
    scenarios: list[CatBoostScenario] = []
    for ap_weight in (1.0, 1.5, 2.0):
        for scenario in base:
            scenario_id = f"{scenario[0]}_apw{ap_weight:.3g}".replace(".", "p")
            scenarios.append(CatBoostScenario(scenario_id, *scenario[1:], atypical_weight_multiplier=float(ap_weight)))
    return tuple(scenarios)

def _require_catboost() -> tuple[Any, Any]:
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is required for Model 2 training. Install catboost in the Python environment "
            "used to run run_catboost.py. Audit mode does not require CatBoost."
        ) from exc
    return CatBoostClassifier, Pool

def _train_predict_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cat_features: list[int],
    scenario: CatBoostScenario,
    config: KeyedConfig,
    seed: int,
) -> dict[str, Any]:
    CatBoostClassifier, Pool = _require_catboost()
    weights = ap_weighted_class_weights(
        y_train,
        atypical_multiplier=scenario.atypical_weight_multiplier,
        n_classes=len(CLASS_NAMES),
    )
    params = scenario.params(config, seed)
    params["class_weights"] = [weights[code] for code in range(len(CLASS_NAMES))]
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_features)
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=int(config.catboost_early_stopping_rounds),
        verbose=False,
    )
    proba = model.predict_proba(valid_pool)
    return {
        "scenario_id": scenario.scenario_id,
        "proba": proba,
        "metrics": metric_row(y_valid, proba),
        "best_iteration": int(model.get_best_iteration() or config.catboost_iterations),
    }

def _evaluate_scenario(
    scenario: CatBoostScenario,
    fold_cache: list[dict[str, Any]],
    config: KeyedConfig,
    seed: int,
) -> dict[str, Any]:
    fold_rows = []
    threshold_inputs: list[dict[str, Any]] = []
    for fold in fold_cache:
        if fold["feature_count"] == 0:
            proba = np.full((len(fold["y_valid"]), len(CLASS_NAMES)), 1.0 / len(CLASS_NAMES))
            metrics = metric_row(fold["y_valid"], proba)
            best_iteration = 0
        else:
            result = _train_predict_catboost(
                fold["X_train"],
                fold["y_train"],
                fold["X_valid"],
                fold["y_valid"],
                fold["cat_features"],
                scenario,
                config,
                seed=seed,
            )
            metrics = result["metrics"]
            best_iteration = result["best_iteration"]
            proba = result["proba"]
        threshold_inputs.append({"inner_split_id": fold["inner_split_id"], "y_true": fold["y_valid"], "proba": proba})
        fold_rows.append(
            {
                "inner_split_id": fold["inner_split_id"],
                "scenario_id": scenario.scenario_id,
                "atypical_weight_multiplier": float(scenario.atypical_weight_multiplier),
                "best_iteration": best_iteration,
                **metrics,
            }
        )
    threshold_summary, threshold_fold_rows = select_ap_threshold_for_folds(threshold_inputs)
    threshold_by_fold = {
        row["inner_split_id"]: {f"thresholded_{key}": value for key, value in row.items() if key != "inner_split_id"}
        for row in threshold_fold_rows
    }
    for row in fold_rows:
        row.update(threshold_by_fold.get(row["inner_split_id"], {}))
    fold_df = pd.DataFrame(fold_rows)
    return {
        "scenario_id": scenario.scenario_id,
        "atypical_weight_multiplier": float(scenario.atypical_weight_multiplier),
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()),
        "mean_log_loss": float(fold_df["log_loss"].mean()),
        "mean_macro_f1": float(fold_df["macro_f1"].mean()),
        "median_best_iteration": int(np.median(fold_df["best_iteration"])),
        "complexity_rank": scenario.complexity_rank,
        **threshold_summary,
        "fold_rows": fold_rows,
    }

def _build_inner_cache(
    X_source: pd.DataFrame,
    y: np.ndarray,
    metadata: pd.DataFrame,
    config: KeyedConfig,
    budget: int,
    outer_split_id: str,
) -> list[dict[str, Any]]:
    seed_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(outer_split_id))
    splits = make_repeated_stratified_splits(
        y,
        n_splits=config.inner_folds,
        n_repeats=1,
        random_state=config.random_state + seed_offset,
        min_ap_in_test=config.min_ap_in_inner_valid,
    )
    cache = []
    for split in splits:
        train_idx = split["train_idx"]
        valid_idx = split["test_idx"]
        prep = CatBoostPreprocessor(config, budget=budget)
        X_train = prep.fit_transform(X_source.iloc[train_idx].reset_index(drop=True), y[train_idx], metadata)
        X_valid = prep.transform(X_source.iloc[valid_idx].reset_index(drop=True))
        cache.append(
            {
                "inner_split_id": split["split_id"],
                "X_train": X_train,
                "X_valid": X_valid,
                "y_train": y[train_idx],
                "y_valid": y[valid_idx],
                "cat_features": prep.cat_feature_indices_,
                "feature_count": int(X_train.shape[1]),
                "screening": prep.screening_,
            }
        )
    return cache

def _select_scenario(summary: pd.DataFrame, scenarios: tuple[CatBoostScenario, ...]) -> pd.Series:
    complexity = {scenario.scenario_id: scenario.complexity_rank for scenario in scenarios}
    ranked = summary.copy()
    ranked["complexity_sort"] = ranked["scenario_id"].map(complexity)
    ranked = ranked.sort_values(
        [
            "mean_threshold_balanced_accuracy",
            "mean_threshold_macro_f1",
            "mean_threshold_ap_f1",
            "mean_balanced_accuracy",
            "mean_log_loss",
            "complexity_sort",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]

def _fit_final_outer_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_features: list[int],
    scenario: CatBoostScenario,
    config: KeyedConfig,
    selected_iterations: int,
) -> tuple[Any, np.ndarray, int]:
    CatBoostClassifier, Pool = _require_catboost()
    seed = config.random_state
    weights = ap_weighted_class_weights(
        y_train,
        atypical_multiplier=scenario.atypical_weight_multiplier,
        n_classes=len(CLASS_NAMES),
    )
    params = scenario.params(config, seed)
    if selected_iterations > 0:
        params["iterations"] = max(25, min(int(config.catboost_iterations), int(selected_iterations)))
    params["class_weights"] = [weights[code] for code in range(len(CLASS_NAMES))]
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=False)
    proba = model.predict_proba(test_pool)
    return model, proba, int(model.get_best_iteration() or params["iterations"])
