"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import itertools

from concurrent.futures import ProcessPoolExecutor

from dataclasses import dataclass

from multiprocessing import get_context

from typing import Any

import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from .ap_tuning import ap_weighted_row_weights, select_ap_threshold_for_folds

from .common import CLASS_NAMES, KeyedConfig, make_repeated_stratified_splits, metric_row

from .preprocess_ebm import EBMPreprocessor

@dataclass(frozen=True)
class EBMScenario:
    scenario_id: str
    max_bins: int
    c_value: float
    interaction_count: int
    interaction_candidate_sources: int = 12
    min_pair_count: int = 10
    atypical_weight_multiplier: float = 1.0

def default_scenarios() -> tuple[EBMScenario, ...]:
    scenarios: list[EBMScenario] = []
    for ap_weight in (1.0, 1.5, 2.0):
        for max_bins in (6, 10):
            for c_value in (0.1, 1.0):
                for interaction_count in (0, 4, 8):
                    scenarios.append(
                        EBMScenario(
                            scenario_id=f"bins{max_bins}_C{c_value:g}_H{interaction_count}_apw{ap_weight:.3g}".replace(".", "p"),
                            max_bins=max_bins,
                            c_value=c_value,
                            interaction_count=interaction_count,
                            atypical_weight_multiplier=float(ap_weight),
                        )
                    )
    return tuple(scenarios)

def _constant_proba(y_train: np.ndarray, n_rows: int, atypical_weight_multiplier: float = 1.0) -> np.ndarray:
    weights = ap_weighted_row_weights(
        y_train,
        atypical_multiplier=atypical_weight_multiplier,
        n_classes=len(CLASS_NAMES),
    )
    counts = np.array([weights[y_train == code].sum() for code in range(len(CLASS_NAMES))], dtype=float)
    probs = counts / counts.sum()
    return np.tile(probs, (n_rows, 1))

def _make_lr(scenario: EBMScenario, max_iter: int, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(scenario.c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=int(max_iter),
        tol=1e-5,
        random_state=int(random_state),
    )

def _predict_proba_full(model: LogisticRegression, X: Any) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((raw.shape[0], len(CLASS_NAMES)), dtype=float)
    for idx, klass in enumerate(model.classes_):
        out[:, int(klass)] = raw[:, idx]
    denom = out.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return out / denom

def _rank_interaction_pairs(
    prep: EBMPreprocessor,
    model: LogisticRegression,
    interaction_count: int,
    candidate_sources: int,
) -> list[tuple[str, str]]:
    if interaction_count <= 0:
        return []
    coef = np.asarray(model.coef_, dtype=float)
    scores: list[tuple[str, float]] = []
    for source_id, cols in prep.source_feature_groups().items():
        if not cols:
            continue
        block = coef[:, cols]
        scores.append((source_id, float(np.linalg.norm(block, ord="fro"))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    top = [source for source, score in scores[: int(candidate_sources)] if score > 0]
    pair_scores = []
    score_map = dict(scores)
    for left, right in itertools.combinations(top, 2):
        pair_scores.append((left, right, score_map.get(left, 0.0) + score_map.get(right, 0.0)))
    pair_scores.sort(key=lambda item: (-item[2], item[0], item[1]))
    return [(left, right) for left, right, _ in pair_scores[: int(interaction_count)]]

def fit_predict_ebm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    metadata: pd.DataFrame,
    config: KeyedConfig,
    budget: int,
    scenario: EBMScenario,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    sample_weight = ap_weighted_row_weights(
        y_train,
        atypical_multiplier=scenario.atypical_weight_multiplier,
        n_classes=len(CLASS_NAMES),
    )
    prep_main = EBMPreprocessor(config, budget=budget, max_bins=scenario.max_bins)
    Z_main = prep_main.fit_transform(X_train, y_train, metadata)
    if Z_main.shape[1] == 0:
        return {
            "proba": _constant_proba(
                y_train,
                len(X_valid),
                atypical_weight_multiplier=scenario.atypical_weight_multiplier,
            ),
            "preprocessor": prep_main,
            "model": None,
            "selected_interactions": [],
            "feature_count": 0,
        }
    main_model = _make_lr(scenario, max_iter=max_iter, random_state=random_state)
    main_model.fit(Z_main, y_train, sample_weight=sample_weight)
    pairs = _rank_interaction_pairs(
        prep_main,
        main_model,
        interaction_count=scenario.interaction_count,
        candidate_sources=scenario.interaction_candidate_sources,
    )
    if pairs:
        prep = EBMPreprocessor(
            config,
            budget=budget,
            max_bins=scenario.max_bins,
            interaction_pairs=pairs,
            min_pair_count=scenario.min_pair_count,
        )
        Z_train = prep.fit_transform(X_train, y_train, metadata)
    else:
        prep = prep_main
        Z_train = Z_main
    model = _make_lr(scenario, max_iter=max_iter, random_state=random_state)
    model.fit(Z_train, y_train, sample_weight=sample_weight)
    Z_valid = prep.transform(X_valid)
    proba = _predict_proba_full(model, Z_valid)
    return {
        "proba": proba,
        "preprocessor": prep,
        "model": model,
        "selected_interactions": pairs,
        "feature_count": int(Z_train.shape[1]),
    }

def _build_inner_folds(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    config: KeyedConfig,
    outer_split_id: str,
) -> list[dict[str, Any]]:
    seed_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(outer_split_id))
    splits = make_repeated_stratified_splits(
        y_outer_train,
        n_splits=config.inner_folds,
        n_repeats=1,
        random_state=config.random_state + seed_offset,
        min_ap_in_test=config.min_ap_in_inner_valid,
    )
    folds = []
    for split in splits:
        train_idx = split["train_idx"]
        valid_idx = split["test_idx"]
        folds.append(
            {
                "inner_split_id": split["split_id"],
                "X_train": X_outer_train.iloc[train_idx].reset_index(drop=True),
                "y_train": y_outer_train[train_idx],
                "X_valid": X_outer_train.iloc[valid_idx].reset_index(drop=True),
                "y_valid": y_outer_train[valid_idx],
            }
        )
    return folds

def _evaluate_scenario(
    scenario: EBMScenario,
    folds: list[dict[str, Any]],
    metadata: pd.DataFrame,
    config: KeyedConfig,
    budget: int,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    rows = []
    threshold_inputs: list[dict[str, Any]] = []
    for fold in folds:
        result = fit_predict_ebm(
            fold["X_train"],
            fold["y_train"],
            fold["X_valid"],
            metadata,
            config,
            budget=budget,
            scenario=scenario,
            max_iter=max_iter,
            random_state=seed,
        )
        metrics = metric_row(fold["y_valid"], result["proba"])
        threshold_inputs.append({"inner_split_id": fold["inner_split_id"], "y_true": fold["y_valid"], "proba": result["proba"]})
        rows.append(
            {
                "inner_split_id": fold["inner_split_id"],
                "scenario_id": scenario.scenario_id,
                "max_bins": int(scenario.max_bins),
                "C": float(scenario.c_value),
                "interaction_count": int(scenario.interaction_count),
                "atypical_weight_multiplier": float(scenario.atypical_weight_multiplier),
                "selected_interaction_count": int(len(result["selected_interactions"])),
                "feature_count": int(result["feature_count"]),
                **metrics,
            }
        )
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
        "max_bins": int(scenario.max_bins),
        "C": float(scenario.c_value),
        "interaction_count": int(scenario.interaction_count),
        "atypical_weight_multiplier": float(scenario.atypical_weight_multiplier),
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()),
        "mean_log_loss": float(fold_df["log_loss"].mean()),
        "mean_macro_f1": float(fold_df["macro_f1"].mean()),
        "mean_feature_count": float(fold_df["feature_count"].mean()),
        "mean_selected_interaction_count": float(fold_df["selected_interaction_count"].mean()),
        **threshold_summary,
        "fold_rows": rows,
    }

_PROCESS_FOLDS: list[dict[str, Any]] | None = None

_PROCESS_METADATA: pd.DataFrame | None = None

_PROCESS_CONFIG: KeyedConfig | None = None

_PROCESS_BUDGET: int | None = None

_PROCESS_MAX_ITER: int | None = None

_PROCESS_SEED: int | None = None

def _init_worker(
    folds: list[dict[str, Any]],
    metadata: pd.DataFrame,
    config: KeyedConfig,
    budget: int,
    max_iter: int,
    seed: int,
) -> None:
    global _PROCESS_FOLDS, _PROCESS_METADATA, _PROCESS_CONFIG, _PROCESS_BUDGET, _PROCESS_MAX_ITER, _PROCESS_SEED
    _PROCESS_FOLDS = folds
    _PROCESS_METADATA = metadata
    _PROCESS_CONFIG = config
    _PROCESS_BUDGET = int(budget)
    _PROCESS_MAX_ITER = int(max_iter)
    _PROCESS_SEED = int(seed)

def _evaluate_worker(scenario: EBMScenario) -> dict[str, Any]:
    if _PROCESS_FOLDS is None or _PROCESS_METADATA is None or _PROCESS_CONFIG is None:
        raise RuntimeError("Model 3 worker is not initialized.")
    return _evaluate_scenario(
        scenario,
        _PROCESS_FOLDS,
        _PROCESS_METADATA,
        _PROCESS_CONFIG,
        int(_PROCESS_BUDGET),
        int(_PROCESS_MAX_ITER),
        int(_PROCESS_SEED),
    )

def evaluate_ebm_scenarios(
    scenarios: tuple[EBMScenario, ...] | list[EBMScenario],
    folds: list[dict[str, Any]],
    metadata: pd.DataFrame,
    config: KeyedConfig,
    budget: int,
    parallel_jobs: int,
    max_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    scenario_list = list(scenarios)
    if int(parallel_jobs) <= 1 or len(scenario_list) <= 1:
        return [
            _evaluate_scenario(s, folds, metadata, config, budget, max_iter=max_iter, seed=seed)
            for s in scenario_list
        ]
    max_workers = max(1, min(int(parallel_jobs), len(scenario_list)))
    try:
        mp_context = get_context("fork")
    except ValueError:
        mp_context = None
    pool_kwargs: dict[str, Any] = {
        "max_workers": max_workers,
        "initializer": _init_worker,
        "initargs": (folds, metadata, config, int(budget), int(max_iter), int(seed)),
    }
    if mp_context is not None:
        pool_kwargs["mp_context"] = mp_context
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        return list(pool.map(_evaluate_worker, scenario_list))

def _select_scenario(summary: pd.DataFrame) -> pd.Series:
    ranked = summary.sort_values(
        [
            "mean_threshold_balanced_accuracy",
            "mean_threshold_macro_f1",
            "mean_threshold_ap_f1",
            "mean_balanced_accuracy",
            "mean_log_loss",
            "interaction_count",
            "max_bins",
            "C",
        ],
        ascending=[False, False, False, False, True, True, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]
