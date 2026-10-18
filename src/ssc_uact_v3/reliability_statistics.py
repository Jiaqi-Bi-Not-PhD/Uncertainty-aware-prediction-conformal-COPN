"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import numpy as np

from numpy.typing import NDArray

import pandas as pd

from .singleton_reliability import deterministic_risk_order

@dataclass(frozen=True)
class PreparedMethod:
    budget: int
    method_id: str
    frame: pd.DataFrame
    cluster_indices: NDArray[np.int64]
    folds: tuple[NDArray[np.int64], ...]
    aurc_orders: tuple[NDArray[np.int64], ...]
    oracle_orders: tuple[NDArray[np.int64], ...]
    auprc_orders: tuple[NDArray[np.int64], ...]

def _prepare_method(
    frame: pd.DataFrame,
    *,
    budget: int,
    method_id: str,
    ranking_column: str,
    estimated_column: str,
    seed: int,
) -> PreparedMethod:
    method = frame[
        [
            "outer_split_id",
            "repeat_id",
            "outer_fold",
            "budget",
            "project_key",
            "true_code",
            "A3_singleton_label",
            "A3_singleton_error",
            "cluster_index",
            ranking_column,
            estimated_column,
        ]
    ].copy()
    method = method.rename(
        columns={ranking_column: "ranking_risk", estimated_column: "estimated_risk"}
    )
    split_ids = sorted(method["outer_split_id"].astype(str).unique())
    if len(split_ids) != 15:
        raise ValueError(f"J{budget} {method_id} does not contain 15 outer folds")
    folds: list[NDArray[np.int64]] = []
    aurc_orders: list[NDArray[np.int64]] = []
    oracle_orders: list[NDArray[np.int64]] = []
    auprc_orders: list[NDArray[np.int64]] = []
    for split_id in split_ids:
        indices = np.flatnonzero(method["outer_split_id"].astype(str).eq(split_id))
        fold = method.iloc[indices]
        projects = fold["project_key"].astype(str).tolist()
        splits = fold["outer_split_id"].astype(str).tolist()
        folds.append(indices.astype(np.int64))
        aurc_orders.append(
            deterministic_risk_order(
                fold["ranking_risk"].to_numpy(float),
                project_keys=projects,
                outer_split_ids=splits,
                seed=seed,
                ascending=True,
            )
        )
        oracle_orders.append(
            deterministic_risk_order(
                fold["A3_singleton_error"].to_numpy(float),
                project_keys=projects,
                outer_split_ids=splits,
                seed=seed,
                ascending=True,
            )
        )
        auprc_orders.append(
            deterministic_risk_order(
                fold["ranking_risk"].to_numpy(float),
                project_keys=projects,
                outer_split_ids=splits,
                seed=seed,
                ascending=False,
            )
        )
    return PreparedMethod(
        budget=budget,
        method_id=method_id,
        frame=method,
        cluster_indices=method["cluster_index"].to_numpy(np.int64),
        folds=tuple(folds),
        aurc_orders=tuple(aurc_orders),
        oracle_orders=tuple(oracle_orders),
        auprc_orders=tuple(auprc_orders),
    )

def _weighted_aurc_components(
    ordered_errors: Sequence[int] | NDArray[np.integer],
    ordered_weights: Sequence[int] | NDArray[np.integer],
    harmonic: NDArray[np.float64],
) -> tuple[float, int]:
    errors = np.asarray(ordered_errors, dtype=np.int8)
    weights = np.asarray(ordered_weights, dtype=np.int64)
    if errors.shape != weights.shape or errors.ndim != 1:
        raise ValueError("weighted AURC inputs are misaligned")
    if np.any((errors != 0) & (errors != 1)) or np.any(weights < 0):
        raise ValueError("weighted AURC requires binary errors and nonnegative weights")
    cumulative_count = 0
    cumulative_error = 0
    numerator = 0.0
    for error, weight in zip(errors, weights, strict=True):
        weight = int(weight)
        if weight == 0:
            continue
        end = cumulative_count + weight
        if end >= len(harmonic):
            raise ValueError("harmonic table is too short for weighted AURC")
        delta_h = harmonic[end] - harmonic[cumulative_count]
        numerator += (
            int(error) * weight
            + (cumulative_error - int(error) * cumulative_count) * delta_h
        )
        cumulative_error += int(error) * weight
        cumulative_count = end
    return float(numerator), cumulative_count

def _weighted_average_precision_components(
    ordered_errors_descending: Sequence[int] | NDArray[np.integer],
    ordered_weights: Sequence[int] | NDArray[np.integer],
    harmonic: NDArray[np.float64],
) -> tuple[float, int]:
    errors = np.asarray(ordered_errors_descending, dtype=np.int8)
    weights = np.asarray(ordered_weights, dtype=np.int64)
    if errors.shape != weights.shape or errors.ndim != 1:
        raise ValueError("weighted AP inputs are misaligned")
    if np.any((errors != 0) & (errors != 1)) or np.any(weights < 0):
        raise ValueError("weighted AP requires binary errors and nonnegative weights")
    cumulative_count = 0
    cumulative_positive = 0
    numerator = 0.0
    for error, weight in zip(errors, weights, strict=True):
        weight = int(weight)
        if weight == 0:
            continue
        end = cumulative_count + weight
        if end >= len(harmonic):
            raise ValueError("harmonic table is too short for weighted AP")
        if int(error) == 1:
            delta_h = harmonic[end] - harmonic[cumulative_count]
            numerator += weight + (cumulative_positive - cumulative_count) * delta_h
            cumulative_positive += weight
        cumulative_count = end
    return float(numerator), cumulative_positive

def _weighted_auroc_components(
    errors: Sequence[int] | NDArray[np.integer],
    risk: Sequence[float] | NDArray[np.floating],
    weights: Sequence[int] | NDArray[np.integer],
) -> tuple[float, float]:
    y = np.asarray(errors, dtype=np.int8)
    score = np.asarray(risk, dtype=float)
    weight = np.asarray(weights, dtype=np.int64)
    if y.shape != score.shape or y.shape != weight.shape or y.ndim != 1:
        raise ValueError("weighted AUROC inputs are misaligned")
    if np.any((y != 0) & (y != 1)) or np.any(weight < 0):
        raise ValueError("weighted AUROC inputs are invalid")
    order = np.argsort(score, kind="mergesort")
    score = score[order]
    y = y[order]
    weight = weight[order]
    if len(score) == 0:
        return 0.0, 0.0
    starts = np.r_[0, np.flatnonzero(np.diff(score) != 0) + 1]
    positive_group = np.add.reduceat(weight * y, starts).astype(float)
    negative_group = np.add.reduceat(weight * (1 - y), starts).astype(float)
    positives = float(positive_group.sum())
    negatives = float(negative_group.sum())
    pairs = positives * negatives
    if pairs == 0:
        return 0.0, 0.0
    negative_before = np.cumsum(negative_group) - negative_group
    favorable = float(
        np.sum(positive_group * (negative_before + 0.5 * negative_group))
    )
    return favorable, pairs

def _harmonic_numbers(maximum: int) -> NDArray[np.float64]:
    if maximum < 1:
        raise ValueError("harmonic table maximum must be positive")
    harmonic = np.zeros(maximum + 1, dtype=float)
    harmonic[1:] = np.cumsum(1.0 / np.arange(1, maximum + 1, dtype=float))
    return harmonic
