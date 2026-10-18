"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import math

from typing import Any

import numpy as np

import pandas as pd

from scipy.stats import beta

STRATA = ("Q1", "Q2", "Q3", "Q4")

CLASS_NAMES = {0: "PD", 1: "AP", 2: "HC"}

COVERAGE_TARGETS = {"PD": 0.90, "AP": 0.95, "HC": 0.90}

MINIMUM_VALID_REPLICATES = 1900

def metric_arrays(frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    true = frame["true_code"].to_numpy(dtype=np.int64)
    size = frame["core_size"].to_numpy(dtype=np.float64)
    names: list[str] = []
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []

    def add(name: str, values: Any, eligible: Any = None) -> None:
        value_array = np.asarray(values, dtype=np.float64)
        eligible_array = (
            np.ones(len(frame), dtype=np.float64)
            if eligible is None
            else np.asarray(eligible, dtype=np.float64)
        )
        names.append(name)
        numerators.append(value_array * eligible_array)
        denominators.append(eligible_array)

    add("core_coverage", frame["core_contains_true"])
    add("mention_coverage", frame["mention_contains_true"])
    add("mean_core_size", size)
    add("mean_SIG", frame["core_sig"])
    add("singleton_rate", size == 1)
    add("pair_rate", size == 2)
    add("full_set_rate", size == 3)
    add("AP_exclusion_rate", frame["ap_excluded"], frame["ap_eligible"])
    add(
        "dangerous_HC_singleton_rate",
        frame["dangerous_hc_singleton"],
        frame["dangerous_hc_eligible"],
    )
    add("mean_robust_expected_cost", frame["robust_expected_cost"])
    add("mean_realized_set_harm", frame["realized_set_harm"])
    add("forced_argmax_error_rate", frame["forced_argmax_error"])
    add("multiclass_log_loss", frame["multiclass_log_loss"])
    add("multiclass_brier", frame["multiclass_brier"])
    for code, label in CLASS_NAMES.items():
        eligible = true == code
        add(f"core_coverage_{label}", frame["core_contains_true"], eligible)
        add(f"mention_coverage_{label}", frame["mention_contains_true"], eligible)
    for mask in range(1, 8):
        add(f"set_mask_{mask}_rate", frame["core_mask"].eq(mask))
    return names, np.column_stack(numerators), np.column_stack(denominators)

def cluster_matrix(
    frame: pd.DataFrame,
    numerator: np.ndarray,
    denominator: np.ndarray,
    participant_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    codes = np.fromiter(
        (participant_index[str(key)] for key in frame["project_key"]),
        dtype=np.int64,
        count=len(frame),
    )
    n_participants = len(participant_index)
    cluster_num = np.zeros((n_participants, numerator.shape[1]), dtype=np.float64)
    cluster_den = np.zeros((n_participants, denominator.shape[1]), dtype=np.float64)
    np.add.at(cluster_num, codes, numerator)
    np.add.at(cluster_den, codes, denominator)
    return cluster_num, cluster_den

def summarize_group(
    frame: pd.DataFrame,
    method: str,
    stratum: str,
    participant_index: dict[str, int],
    bootstrap_weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    names, numerator, denominator = metric_arrays(frame)
    point_num = numerator.sum(axis=0)
    point_den = denominator.sum(axis=0)
    cluster_num, cluster_den = cluster_matrix(frame, numerator, denominator, participant_index)
    boot_num = bootstrap_weights @ cluster_num
    boot_den = bootstrap_weights @ cluster_den
    with np.errstate(divide="ignore", invalid="ignore"):
        point = point_num / point_den
        boot = boot_num / boot_den
    rows: list[dict[str, Any]] = []
    cache: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        values = boot[:, index]
        valid = values[np.isfinite(values)]
        cache[name] = values
        inferentially_usable = len(valid) >= MINIMUM_VALID_REPLICATES
        rows.append(
            {
                "method_id": method,
                "primary_stratum": stratum,
                "metric": name,
                "estimate": float(point[index]),
                "numerator": float(point_num[index]),
                "denominator": float(point_den[index]),
                "lower_95": (
                    float(np.quantile(valid, 0.025, method="linear")) if len(valid) else math.nan
                ),
                "upper_95": (
                    float(np.quantile(valid, 0.975, method="linear")) if len(valid) else math.nan
                ),
                "lower_one_sided_95": (
                    float(np.quantile(valid, 0.05, method="linear")) if len(valid) else math.nan
                ),
                "upper_one_sided_95": (
                    float(np.quantile(valid, 0.95, method="linear")) if len(valid) else math.nan
                ),
                "valid_replicates": int(len(valid)),
                "inferentially_usable": bool(inferentially_usable),
                "rows": int(len(frame)),
                "unique_participants": int(frame["project_key"].nunique()),
                "contributing_splits": int(frame["outer_split_id"].nunique()),
            }
        )
    return rows, cache

def cp_upper(events: int, denominator: int, confidence: float = 0.95) -> float:
    if denominator <= 0 or events < 0 or events > denominator:
        return math.nan
    if events == denominator:
        return 1.0
    return float(beta.ppf(confidence, events + 1, denominator - events))

def cp_lower(events: int, denominator: int, confidence: float = 0.95) -> float:
    """Exact one-sided lower confidence bound for a binomial proportion."""
    if denominator <= 0 or events < 0 or events > denominator:
        return math.nan
    if events == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, events, denominator - events + 1))

def exact_coverage_rows(frame: pd.DataFrame, method: str) -> list[dict[str, Any]]:
    """Retain exact pooled, repeat-specific, and participant-level coverage bounds.

    The participant-level event is deliberately conservative: every row contributed by
    that participant to the given frozen Q stratum must cover the true class.
    """
    rows: list[dict[str, Any]] = []

    def add(
        part: pd.DataFrame,
        *,
        stratum: str,
        label: str,
        aggregation_level: str,
        repeat_id: str | int | None,
        participant_all_rows: bool,
        estimand: str,
        inferential_role: str,
        target_applicable: bool,
    ) -> None:
        if participant_all_rows:
            coverage = part.groupby("project_key", observed=True)["core_contains_true"].all()
            events = int(coverage.sum())
            denominator = int(len(coverage))
            unique_participants = denominator
        else:
            events = int(part["core_contains_true"].sum())
            denominator = int(len(part))
            unique_participants = int(part["project_key"].nunique())
        rows.append(
            {
                "method_id": method,
                "primary_stratum": stratum,
                "class_name": label,
                "aggregation_level": aggregation_level,
                "repeat_id": repeat_id,
                "estimand": estimand,
                "inferential_role": inferential_role,
                "covered": events,
                "denominator": denominator,
                "coverage": events / denominator if denominator else math.nan,
                "clopper_pearson_lower_one_sided_95": cp_lower(events, denominator),
                "unique_participants": unique_participants,
                "target": COVERAGE_TARGETS[label],
                "target_applicable": target_applicable,
                "point_meets_target": (
                    bool(events / denominator >= COVERAGE_TARGETS[label])
                    if target_applicable and denominator
                    else None
                ),
                "exact_lower_bound_meets_target": (
                    bool(cp_lower(events, denominator) >= COVERAGE_TARGETS[label])
                    if target_applicable and denominator
                    else None
                ),
            }
        )

    repeat_ids = sorted(frame["repeat_id"].unique().tolist())
    for stratum in STRATA:
        stratum_frame = frame.loc[frame["primary_stratum"].eq(stratum)]
        for code, label in CLASS_NAMES.items():
            class_frame = stratum_frame.loc[stratum_frame["true_code"].eq(code)]
            add(
                class_frame,
                stratum=stratum,
                label=label,
                aggregation_level="pooled_rows",
                repeat_id=None,
                participant_all_rows=False,
                estimand="repeated_outer_test_row_coverage",
                inferential_role="descriptive_dependent_rows",
                target_applicable=False,
            )
            for repeat_id in repeat_ids:
                repeat_frame = class_frame.loc[class_frame["repeat_id"].eq(repeat_id)]
                add(
                    repeat_frame,
                    stratum=stratum,
                    label=label,
                    aggregation_level="repeat",
                    repeat_id=repeat_id,
                    participant_all_rows=False,
                    estimand="within_repeat_participant_coverage",
                    inferential_role="exact_primary_robustness",
                    target_applicable=True,
                )
            add(
                class_frame,
                stratum=stratum,
                label=label,
                aggregation_level="unique_participants_all_rows_covered",
                repeat_id=None,
                participant_all_rows=True,
                estimand="participant_all_contributed_rows_covered",
                inferential_role="conservative_sensitivity",
                target_applicable=False,
            )
    return rows

def unique_harm_rows(frame: pd.DataFrame, method: str, stratum: str) -> dict[str, Any]:
    grouped = frame.groupby("project_key", observed=True).agg(
        eligible=("dangerous_hc_eligible", "max"),
        event=("dangerous_hc_singleton", "max"),
    )
    grouped = grouped.loc[grouped["eligible"]]
    events = int(grouped["event"].sum())
    denominator = int(len(grouped))
    return {
        "method_id": method,
        "primary_stratum": stratum,
        "event": "unique_participant_any_repeat_dangerous_PD_or_AP_to_HC_singleton",
        "events": events,
        "denominator": denominator,
        "rate": events / denominator if denominator else math.nan,
        "clopper_pearson_upper_one_sided_95": cp_upper(events, denominator),
    }

def contrast_row(
    method: str,
    metric: str,
    estimates: dict[tuple[str, str, str], float],
    cache: dict[tuple[str, str, str], np.ndarray],
) -> dict[str, Any]:
    q4 = cache[(method, "Q4", metric)]
    q1 = cache[(method, "Q1", metric)]
    values = q4 - q1
    valid = values[np.isfinite(values)]
    inferentially_usable = len(valid) >= MINIMUM_VALID_REPLICATES
    estimate = estimates[(method, "Q4", metric)] - estimates[(method, "Q1", metric)]
    return {
        "method_id": method,
        "contrast": "Q4_minus_Q1",
        "metric": metric,
        "estimate": float(estimate),
        "lower_95": (
            float(np.quantile(valid, 0.025, method="linear")) if len(valid) else math.nan
        ),
        "upper_95": (
            float(np.quantile(valid, 0.975, method="linear")) if len(valid) else math.nan
        ),
        "lower_one_sided_95": (
            float(np.quantile(valid, 0.05, method="linear")) if len(valid) else math.nan
        ),
        "upper_one_sided_95": (
            float(np.quantile(valid, 0.95, method="linear")) if len(valid) else math.nan
        ),
        "valid_replicates": int(len(valid)),
        "inferentially_usable": bool(inferentially_usable),
    }

def repeat_and_split_metrics(frame: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    split_rows = []
    for columns, destination in [
        (["repeat_id", "primary_stratum"], rows),
        (["outer_split_id", "primary_stratum"], split_rows),
    ]:
        for keys, part in frame.groupby(columns, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {column: value for column, value in zip(columns, keys)}
            row.update(
                {
                    "method_id": method,
                    "rows": int(len(part)),
                    "unique_participants": int(part["project_key"].nunique()),
                    "core_coverage": float(part["core_contains_true"].mean()),
                    "mean_core_size": float(part["core_size"].mean()),
                    "full_set_rate": float(part["core_size"].eq(3).mean()),
                    "mean_SIG": float(part["core_sig"].mean()),
                    "forced_argmax_error_rate": float(part["forced_argmax_error"].mean()),
                }
            )
            destination.append(row)
    return pd.DataFrame(rows), pd.DataFrame(split_rows)

def global_efficiency(
    a2: pd.DataFrame,
    a5: pd.DataFrame,
    participant_index: dict[str, int],
    bootstrap_weights: np.ndarray,
) -> list[dict[str, Any]]:
    join_keys = ["outer_split_id", "project_key"]
    merged = a2.loc[:, [*join_keys, "core_size", "core_sig"]].merge(
        a5.loc[:, [*join_keys, "core_size", "core_sig"]],
        on=join_keys,
        validate="one_to_one",
        suffixes=("_A2", "_A5"),
    )
    merged["full_A2"] = merged["core_size_A2"].eq(3).astype(float)
    merged["full_A5"] = merged["core_size_A5"].eq(3).astype(float)
    codes = np.fromiter(
        (participant_index[str(key)] for key in merged["project_key"]),
        dtype=np.int64,
        count=len(merged),
    )
    n = len(participant_index)

    def cluster_sum(values: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        np.add.at(out, codes, values)
        return out

    count = cluster_sum(np.ones(len(merged), dtype=np.float64))
    size_a2 = cluster_sum(merged["core_size_A2"].to_numpy(float))
    size_a5 = cluster_sum(merged["core_size_A5"].to_numpy(float))
    sig_a2 = cluster_sum(merged["core_sig_A2"].to_numpy(float))
    sig_a5 = cluster_sum(merged["core_sig_A5"].to_numpy(float))
    full_a2 = cluster_sum(merged["full_A2"].to_numpy(float))
    full_a5 = cluster_sum(merged["full_A5"].to_numpy(float))

    point = {
        "absolute_mean_size_reduction": float(merged["core_size_A2"].mean() - merged["core_size_A5"].mean()),
        "relative_mean_size_reduction": float(
            (merged["core_size_A2"].mean() - merged["core_size_A5"].mean())
            / merged["core_size_A2"].mean()
        ),
        "mean_SIG_increase": float(merged["core_sig_A5"].mean() - merged["core_sig_A2"].mean()),
        "relative_full_set_reduction": float(
            (merged["full_A2"].mean() - merged["full_A5"].mean()) / merged["full_A2"].mean()
        ),
    }
    den = bootstrap_weights @ count
    mean_size_a2 = (bootstrap_weights @ size_a2) / den
    mean_size_a5 = (bootstrap_weights @ size_a5) / den
    mean_sig_a2 = (bootstrap_weights @ sig_a2) / den
    mean_sig_a5 = (bootstrap_weights @ sig_a5) / den
    mean_full_a2 = (bootstrap_weights @ full_a2) / den
    mean_full_a5 = (bootstrap_weights @ full_a5) / den
    replicates = {
        "absolute_mean_size_reduction": mean_size_a2 - mean_size_a5,
        "relative_mean_size_reduction": (mean_size_a2 - mean_size_a5) / mean_size_a2,
        "mean_SIG_increase": mean_sig_a5 - mean_sig_a2,
        "relative_full_set_reduction": (mean_full_a2 - mean_full_a5) / mean_full_a2,
    }
    thresholds = {
        "absolute_mean_size_reduction": 0.20,
        "relative_mean_size_reduction": 0.10,
        "mean_SIG_increase": 0.15,
        "relative_full_set_reduction": 0.20,
    }
    rows = []
    for metric, estimate in point.items():
        values = replicates[metric]
        valid = values[np.isfinite(values)]
        lower = float(np.quantile(valid, 0.05, method="linear"))
        rows.append(
            {
                "contrast": "A5_minus_A2_efficiency",
                "metric": metric,
                "estimate": estimate,
                "lower_95": float(np.quantile(valid, 0.025, method="linear")),
                "upper_95": float(np.quantile(valid, 0.975, method="linear")),
                "lower_one_sided_95": lower,
                "valid_replicates": int(len(valid)),
                "development_threshold": thresholds[metric],
                "point_meets_threshold": bool(estimate >= thresholds[metric]),
                "lower_bound_meets_threshold": bool(lower >= thresholds[metric]),
            }
        )
    return rows
