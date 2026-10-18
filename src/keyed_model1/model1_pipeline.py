"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import pickle

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import get_context

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

from .ap_tuning import ap_threshold_label, ap_weighted_row_weights, select_ap_threshold_for_folds

from .common import CLASS_NAMES, KeyedConfig, make_repeated_stratified_splits, metric_row

from .group_enet import GroupedMultinomialElasticNet, fit_one_group_enet, lambda_max_for_groups

from .preprocess_model1 import Model1Preprocessor

DEFAULT_ALPHA_GRID = (0.05, 0.25, 0.50, 0.75, 0.95, 1.00)

DEFAULT_LAMBDA_FRACTIONS = tuple(float(x) for x in np.geomspace(1.0, 1e-3, 18))

DEFAULT_AP_WEIGHT_MULTIPLIERS = (1.0, 1.5, 2.0)

def _constant_proba(y_train: np.ndarray, n_rows: int, atypical_weight_multiplier: float = 1.0) -> np.ndarray:
    weights = ap_weighted_row_weights(
        y_train,
        atypical_multiplier=atypical_weight_multiplier,
        n_classes=len(CLASS_NAMES),
    )
    counts = np.array([weights[y_train == code].sum() for code in range(len(CLASS_NAMES))], dtype=float)
    probs = counts / counts.sum()
    return np.tile(probs, (n_rows, 1))

def _build_inner_fold_cache(
    X_train_source: pd.DataFrame,
    y_train: np.ndarray,
    metadata: pd.DataFrame,
    config: KeyedConfig,
    budget: int,
    outer_split_id: str,
) -> list[dict[str, Any]]:
    seed_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(outer_split_id))
    inner_splits = make_repeated_stratified_splits(
        y_train,
        n_splits=config.inner_folds,
        n_repeats=1,
        random_state=config.random_state + seed_offset,
        min_ap_in_test=config.min_ap_in_inner_valid,
    )
    cache = []
    for inner in inner_splits:
        inner_train = inner["train_idx"]
        inner_valid = inner["test_idx"]
        prep = Model1Preprocessor(config, budget=budget)
        Z_train = prep.fit_transform(X_train_source.iloc[inner_train].reset_index(drop=True), y_train[inner_train], metadata)
        Z_valid = prep.transform(X_train_source.iloc[inner_valid].reset_index(drop=True))
        cache.append(
            {
                "inner_split_id": inner["split_id"],
                "train_idx": inner_train,
                "valid_idx": inner_valid,
                "Z_train": Z_train,
                "Z_valid": Z_valid,
                "y_train": y_train[inner_train],
                "y_valid": y_train[inner_valid],
                "groups": prep.groups_,
                "screening": prep.screening_,
                "feature_count": int(Z_train.shape[1]),
                "eligible_source_count": len(prep.groups_),
            }
        )
    return cache

def _evaluate_scenario(
    alpha: float,
    lambda_fraction: float,
    atypical_weight_multiplier: float,
    fold_cache: list[dict[str, Any]],
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    fold_rows = []
    threshold_inputs: list[dict[str, Any]] = []
    for fold in fold_cache:
        if fold["feature_count"] == 0:
            proba = _constant_proba(fold["y_train"], len(fold["y_valid"]), atypical_weight_multiplier=atypical_weight_multiplier)
            metrics = metric_row(fold["y_valid"], proba)
            threshold_inputs.append({"inner_split_id": fold["inner_split_id"], "y_true": fold["y_valid"], "proba": proba})
            fold_rows.append(
                {
                    "inner_split_id": fold["inner_split_id"],
                    "alpha": alpha,
                    "lambda_fraction": lambda_fraction,
                    "atypical_weight_multiplier": float(atypical_weight_multiplier),
                    "lambda": np.nan,
                    "lambda_max": np.nan,
                    "nonzero_sources": 0,
                    **metrics,
                }
            )
            continue
        result = fit_one_group_enet(
            fold["Z_train"],
            fold["y_train"],
            fold["Z_valid"],
            fold["y_valid"],
            fold["groups"],
            alpha=alpha,
            lambda_fraction=lambda_fraction,
            max_iter=max_iter,
            tol=tol,
            atypical_weight_multiplier=atypical_weight_multiplier,
        )
        proba = result.pop("proba")
        threshold_inputs.append({"inner_split_id": fold["inner_split_id"], "y_true": fold["y_valid"], "proba": proba})
        fold_rows.append(
            {
                "inner_split_id": fold["inner_split_id"],
                "alpha": alpha,
                "lambda_fraction": lambda_fraction,
                "atypical_weight_multiplier": float(atypical_weight_multiplier),
                "lambda": result["lambda"],
                "lambda_max": result["lambda_max"],
                "nonzero_sources": result["nonzero_sources"],
                "fit_iterations": result["fit_state"].get("iterations"),
                "fit_converged": result["fit_state"].get("converged"),
                **result["metrics"],
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
    summary = {
        "alpha": float(alpha),
        "lambda_fraction": float(lambda_fraction),
        "atypical_weight_multiplier": float(atypical_weight_multiplier),
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()),
        "mean_log_loss": float(fold_df["log_loss"].mean()),
        "mean_macro_f1": float(fold_df["macro_f1"].mean()),
        "mean_nonzero_sources": float(fold_df["nonzero_sources"].mean()),
        **threshold_summary,
        "fold_rows": fold_rows,
    }
    return summary

_PROCESS_FOLD_CACHE: list[dict[str, Any]] | None = None

_PROCESS_MAX_ITER: int | None = None

_PROCESS_TOL: float | None = None

def _init_scenario_process_worker(fold_cache: list[dict[str, Any]], max_iter: int, tol: float) -> None:
    global _PROCESS_FOLD_CACHE, _PROCESS_MAX_ITER, _PROCESS_TOL
    _PROCESS_FOLD_CACHE = fold_cache
    _PROCESS_MAX_ITER = int(max_iter)
    _PROCESS_TOL = float(tol)

def _evaluate_scenario_chunk(scenarios: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    if _PROCESS_FOLD_CACHE is None or _PROCESS_MAX_ITER is None or _PROCESS_TOL is None:
        raise RuntimeError("Model 1 process worker was not initialized.")
    return [
        _evaluate_scenario(
            float(alpha),
            float(lambda_fraction),
            float(atypical_weight_multiplier),
            _PROCESS_FOLD_CACHE,
            _PROCESS_MAX_ITER,
            _PROCESS_TOL,
        )
        for alpha, lambda_fraction, atypical_weight_multiplier in scenarios
    ]

def _scenario_chunks(scenarios: list[tuple[float, float, float]], max_workers: int) -> list[list[tuple[float, float, float]]]:
    worker_count = max(1, min(int(max_workers), len(scenarios)))
    chunks: list[list[tuple[float, float, float]]] = [[] for _ in range(worker_count)]
    for index, scenario in enumerate(scenarios):
        chunks[index % worker_count].append(scenario)
    return [chunk for chunk in chunks if chunk]

def evaluate_model1_scenarios(
    scenarios: list[tuple[float, float] | tuple[float, float, float]] | tuple[tuple[float, float] | tuple[float, float, float], ...],
    fold_cache: list[dict[str, Any]],
    parallel_jobs: int,
    max_iter: int,
    tol: float,
) -> list[dict[str, Any]]:
    scenario_list = [
        (float(scenario[0]), float(scenario[1]), float(scenario[2]) if len(scenario) > 2 else 1.0)
        for scenario in scenarios
    ]
    if int(parallel_jobs) <= 1 or len(scenario_list) <= 1:
        return [
            _evaluate_scenario(alpha, lambda_fraction, atypical_weight_multiplier, fold_cache, max_iter, tol)
            for alpha, lambda_fraction, atypical_weight_multiplier in scenario_list
        ]

    chunks = _scenario_chunks(scenario_list, int(parallel_jobs))
    try:
        mp_context = get_context("fork")
    except ValueError:
        mp_context = None
    pool_kwargs: dict[str, Any] = {
        "max_workers": len(chunks),
        "initializer": _init_scenario_process_worker,
        "initargs": (fold_cache, max_iter, tol),
    }
    if mp_context is not None:
        pool_kwargs["mp_context"] = mp_context

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        for chunk_results in pool.map(_evaluate_scenario_chunk, chunks):
            results.extend(chunk_results)
    return results

def _select_scenario(summary: pd.DataFrame) -> pd.Series:
    ranked = summary.sort_values(
        [
            "mean_threshold_balanced_accuracy",
            "mean_threshold_macro_f1",
            "mean_threshold_ap_f1",
            "mean_balanced_accuracy",
            "mean_log_loss",
            "lambda_fraction",
            "alpha",
        ],
        ascending=[False, False, False, False, True, False, False],
        kind="mergesort",
    )
    return ranked.iloc[0]

def _fit_final_model(
    dataset: Any,
    config: KeyedConfig,
    output_dir: Path,
    metrics_df: pd.DataFrame,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    if metrics_df.empty:
        return {}
    candidate = metrics_df.sort_values(
        ["balanced_accuracy", "nonzero_source_count", "feature_count"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    budget = int(candidate["budget"])
    alpha = float(candidate["selected_alpha"])
    lambda_fraction = float(candidate["selected_lambda_fraction"])
    atypical_weight_multiplier = float(candidate.get("selected_atypical_weight_multiplier", 1.0))
    ap_threshold = float(candidate.get("selected_ap_threshold", np.nan))
    prep = Model1Preprocessor(config, budget=budget)
    Z = prep.fit_transform(dataset.X, dataset.y, dataset.metadata)
    if Z.shape[1] == 0:
        return {"budget": budget, "alpha": alpha, "lambda_fraction": lambda_fraction, "status": "no_eligible_features"}
    sample_weight = ap_weighted_row_weights(dataset.y, atypical_multiplier=atypical_weight_multiplier, n_classes=len(CLASS_NAMES))
    lambda_max = lambda_max_for_groups(Z, dataset.y, prep.groups_, alpha, sample_weight=sample_weight)
    lambda_value = max(lambda_max * lambda_fraction, 1e-8)
    model = GroupedMultinomialElasticNet(alpha=alpha, lambda_=lambda_value, groups=prep.groups_, max_iter=max_iter, tol=tol)
    model.fit(Z, dataset.y, sample_weight=sample_weight)
    norms = model.source_norms()
    selected_sources = pd.DataFrame(
        [
            {"source_id": source, "coefficient_frobenius_norm": norm, "selected": norm > 1e-8}
            for source, norm in sorted(norms.items())
        ]
    )
    selected_sources = selected_sources.merge(
        dataset.metadata[["source_id", "source_sheet", "source_group", "implementation_level", "raw_header"]],
        on="source_id",
        how="left",
    )
    selected_sources.to_csv(output_dir / "final_selected_sources.csv", index=False)
    pd.DataFrame(model.coefficient_rows(prep.feature_names_)).to_csv(output_dir / "final_design_coefficients.csv", index=False)
    prep.source_mapping().to_csv(output_dir / "final_source_to_design_mapping.csv", index=False)
    with (output_dir / "final_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "preprocessor": prep,
                "model": model,
                "config": config,
                "atypical_weight_multiplier": atypical_weight_multiplier,
                "ap_threshold": ap_threshold,
            },
            handle,
        )
    return {
        "budget": budget,
        "alpha": alpha,
        "lambda_fraction": lambda_fraction,
        "atypical_weight_multiplier": atypical_weight_multiplier,
        "ap_threshold": ap_threshold,
        "pred_rule": ap_threshold_label(ap_threshold),
        "lambda": float(lambda_value),
        "feature_count": int(Z.shape[1]),
        "eligible_source_count": int(len(prep.groups_)),
        "selected_source_count": int((selected_sources["selected"]).sum()),
        "fit_state": model.fit_state_.__dict__ if model.fit_state_ is not None else {},
    }
