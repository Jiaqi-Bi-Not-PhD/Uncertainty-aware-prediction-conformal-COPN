"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from collections.abc import Iterable

import json

from typing import Any

import numpy as np

import pandas as pd

from .bootstrap import _binomial_upper

from .metrics import BIT_COUNT, set_contains_true

CLASS_NAMES = ("PD", "AP", "HC")

METHODS = ("A0", "A1", "A2", "A3", "A4", "A5")

INTERVAL_METHODS = ("A2", "A3", "A5")

INTERVAL_METRICS = (
    "core_coverage_PD",
    "core_coverage_AP",
    "core_coverage_HC",
    "mention_coverage_PD",
    "mention_coverage_AP",
    "mention_coverage_HC",
    "mean_set_size",
    "mean_SIG",
    "AP_exclusion_rate",
    "dangerous_HC_singleton_rate",
    "mean_robust_expected_cost",
    "class_balanced_robust_expected_cost",
    "class_balanced_core_coverage",
)

PAIRED_CONTRASTS = (
    ("A3_minus_A2", "A3", "A2"),
    ("A5_minus_A2", "A5", "A2"),
    ("A5_minus_A3", "A5", "A3"),
)

PANEL_LEVELS: dict[str, tuple[str, ...]] = {
    "overall": ("all",),
    "SSI_quintile": ("Q1", "Q2", "Q3", "Q4", "Q5"),
    "SSI_zero_phenotype": (
        "stable_singleton",
        "stable_pair",
        "stable_full",
        "scenario_sensitive",
    ),
    "information_deficit_quartile": ("Q1", "Q2", "Q3", "Q4"),
    "fallback_panel": ("all_rows", "exclude_A3_fallback_rows", "A3_fallback_rows_only"),
}

def panel_mask(frame: pd.DataFrame, panel: str, level: str) -> np.ndarray:
    if panel == "overall":
        return np.ones(len(frame), dtype=bool)
    if panel == "SSI_quintile":
        return frame["ssi_quintile"].eq(level).to_numpy(bool)
    if panel == "SSI_zero_phenotype":
        return frame["ssi_zero_phenotype"].eq(level).to_numpy(bool)
    if panel == "information_deficit_quartile":
        return frame["information_deficit_quartile"].eq(level).to_numpy(bool)
    if panel == "fallback_panel":
        fallback = frame["a3_fallback_flag"].to_numpy(bool)
        if level == "all_rows":
            return np.ones(len(frame), dtype=bool)
        if level == "exclude_A3_fallback_rows":
            return ~fallback
        if level == "A3_fallback_rows_only":
            return fallback
    raise ValueError(f"unknown Stage 5 panel/level: {panel}/{level}")

def summarize_whole_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in (3, 4):
        budget_frame = frame[frame["budget"].eq(budget)]
        for method in METHODS:
            method_frame = budget_frame[budget_frame["method_id"].eq(method)]
            for panel, levels in PANEL_LEVELS.items():
                for level in levels:
                    selected = method_frame[panel_mask(method_frame, panel, level)]
                    rows.append(_summarize_group(selected, budget, method, panel, level))
    return pd.DataFrame(rows)

def _summarize_group(
    group: pd.DataFrame,
    budget: int,
    method: str,
    panel: str,
    level: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "budget": budget,
        "method_id": method,
        "stratum": panel,
        "level": level,
        "repeated_rows": int(len(group)),
        "unique_participants": int(group["project_key"].nunique()),
        "contributing_outer_splits": int(group["outer_split_id"].nunique()),
        "outer_split_ids": "|".join(sorted(group["outer_split_id"].astype(str).unique())),
    }
    if group.empty:
        return _empty_summary(result)
    truth = group["true_code"].to_numpy(np.int8)
    core = group["core_mask"].to_numpy(np.uint8)
    mention = group["mention_mask"].to_numpy(np.uint8)
    core_contains = set_contains_true(core, truth)
    mention_contains = set_contains_true(mention, truth)
    size = BIT_COUNT[core]
    disease = np.isin(truth, [0, 1])
    is_ap = truth == 1
    is_hc = truth == 2
    result.update(
        {
            "core_covered": int(core_contains.sum()),
            "core_coverage": float(core_contains.mean()),
            "mention_covered": int(mention_contains.sum()),
            "mention_coverage": float(mention_contains.mean()),
            "set_size_sum": int(size.sum()),
            "mean_set_size": float(size.mean()),
            "SIG_sum": float(group["core_sig"].sum()),
            "mean_SIG": float(group["core_sig"].mean()),
            "singleton_count": int((size == 1).sum()),
            "singleton_rate": float((size == 1).mean()),
            "pair_count": int((size == 2).sum()),
            "pair_rate": float((size == 2).mean()),
            "full_set_count": int((size == 3).sum()),
            "full_set_rate": float((size == 3).mean()),
            "dangerous_HC_singleton_count": int((disease & (core == 4)).sum()),
            "dangerous_HC_singleton_denominator": int(disease.sum()),
            "dangerous_HC_singleton_rate": _ratio(
                int((disease & (core == 4)).sum()), int(disease.sum())
            ),
            "PD_to_HC_singleton_count": int(((truth == 0) & (core == 4)).sum()),
            "AP_to_HC_singleton_count": int(((truth == 1) & (core == 4)).sum()),
            "AP_exclusion_count": int((is_ap & ~core_contains).sum()),
            "AP_exclusion_rate": _ratio(int((is_ap & ~core_contains).sum()), int(is_ap.sum())),
            "AP_mention_exclusion_count": int((is_ap & ~mention_contains).sum()),
            "AP_mention_exclusion_rate": _ratio(
                int((is_ap & ~mention_contains).sum()), int(is_ap.sum())
            ),
            "HC_exclusion_count": int((is_hc & ~core_contains).sum()),
            "HC_exclusion_rate": _ratio(int((is_hc & ~core_contains).sum()), int(is_hc.sum())),
            "robust_expected_cost_sum": float(group["robust_expected_cost"].sum()),
            "mean_robust_expected_cost": float(group["robust_expected_cost"].mean()),
            "realized_set_harm_sum": float(group["realized_set_harm"].sum()),
            "mean_realized_set_harm": float(group["realized_set_harm"].mean()),
            "mean_SSI": float(group["ssi_raw"].mean()),
            "mean_information_deficit": float(group["information_deficit"].mean()),
            "A3_fallback_identity_count": int(group["a3_fallback_flag"].sum()),
        }
    )
    native = group["fallback_flag"].dropna()
    result["method_native_fallback_denominator"] = int(len(native))
    result["method_native_fallback_count"] = int(native.astype(bool).sum()) if len(native) else np.nan
    result["method_native_fallback_rate"] = (
        float(native.astype(bool).mean()) if len(native) else np.nan
    )
    class_coverage: list[float] = []
    class_cost: list[float] = []
    for code, name in enumerate(CLASS_NAMES):
        selected = truth == code
        denominator = int(selected.sum())
        core_numerator = int(core_contains[selected].sum())
        mention_numerator = int(mention_contains[selected].sum())
        cost_mean = (
            float(group.loc[selected, "robust_expected_cost"].mean())
            if denominator
            else np.nan
        )
        result[f"n_{name}"] = denominator
        result[f"core_covered_{name}"] = core_numerator
        result[f"core_coverage_{name}"] = _ratio(core_numerator, denominator)
        result[f"mention_covered_{name}"] = mention_numerator
        result[f"mention_coverage_{name}"] = _ratio(mention_numerator, denominator)
        result[f"mean_robust_expected_cost_{name}"] = cost_mean
        class_coverage.append(result[f"core_coverage_{name}"])
        class_cost.append(cost_mean)
    result["class_balanced_core_coverage"] = _complete_mean(class_coverage)
    result["class_balanced_robust_expected_cost"] = _complete_mean(class_cost)
    for mask in range(1, 8):
        result[f"core_mask_{mask}_count"] = int((core == mask).sum())
        result[f"core_mask_{mask}_rate"] = float((core == mask).mean())
    return _attach_unestimable_reasons(result)

def _empty_summary(result: dict[str, Any]) -> dict[str, Any]:
    count_names = [
        "core_covered",
        "mention_covered",
        "set_size_sum",
        "singleton_count",
        "pair_count",
        "full_set_count",
        "dangerous_HC_singleton_count",
        "dangerous_HC_singleton_denominator",
        "PD_to_HC_singleton_count",
        "AP_to_HC_singleton_count",
        "AP_exclusion_count",
        "AP_mention_exclusion_count",
        "HC_exclusion_count",
        "A3_fallback_identity_count",
        "method_native_fallback_denominator",
    ]
    value_names = [
        "core_coverage",
        "mention_coverage",
        "mean_set_size",
        "SIG_sum",
        "mean_SIG",
        "singleton_rate",
        "pair_rate",
        "full_set_rate",
        "dangerous_HC_singleton_rate",
        "AP_exclusion_rate",
        "AP_mention_exclusion_rate",
        "HC_exclusion_rate",
        "robust_expected_cost_sum",
        "mean_robust_expected_cost",
        "realized_set_harm_sum",
        "mean_realized_set_harm",
        "mean_SSI",
        "mean_information_deficit",
        "method_native_fallback_count",
        "method_native_fallback_rate",
        "class_balanced_core_coverage",
        "class_balanced_robust_expected_cost",
    ]
    for name in count_names:
        result[name] = 0
    for name in value_names:
        result[name] = np.nan
    for class_name in CLASS_NAMES:
        result[f"n_{class_name}"] = 0
        result[f"core_covered_{class_name}"] = 0
        result[f"core_coverage_{class_name}"] = np.nan
        result[f"mention_covered_{class_name}"] = 0
        result[f"mention_coverage_{class_name}"] = np.nan
        result[f"mean_robust_expected_cost_{class_name}"] = np.nan
    for mask in range(1, 8):
        result[f"core_mask_{mask}_count"] = 0
        result[f"core_mask_{mask}_rate"] = np.nan
    return _attach_unestimable_reasons(result)

def _attach_unestimable_reasons(result: dict[str, Any]) -> dict[str, Any]:
    """Record why every missing reported estimand is missing.

    The frozen Stage 5 protocol prohibits treating an empty denominator as a
    zero estimate.  A compact JSON mapping keeps the wide descriptive table
    schema stable while making every ``NaN`` auditable in the retained CSV.
    """

    reasons: dict[str, str] = {}
    for metric, value in result.items():
        if not isinstance(value, (float, np.floating)) or np.isfinite(value):
            continue
        if int(result.get("repeated_rows", 0)) == 0:
            reason = "empty_stratum:repeated_rows=0"
        elif metric.endswith("_PD"):
            reason = f"zero_true_class_denominator:n_PD={int(result.get('n_PD', 0))}"
        elif metric.endswith("_AP"):
            reason = f"zero_true_class_denominator:n_AP={int(result.get('n_AP', 0))}"
        elif metric.endswith("_HC"):
            reason = f"zero_true_class_denominator:n_HC={int(result.get('n_HC', 0))}"
        elif metric in {"AP_exclusion_rate", "AP_mention_exclusion_rate"}:
            reason = f"zero_true_class_denominator:n_AP={int(result.get('n_AP', 0))}"
        elif metric == "HC_exclusion_rate":
            reason = f"zero_true_class_denominator:n_HC={int(result.get('n_HC', 0))}"
        elif metric == "dangerous_HC_singleton_rate":
            reason = (
                "zero_disease_denominator:dangerous_HC_singleton_denominator="
                f"{int(result.get('dangerous_HC_singleton_denominator', 0))}"
            )
        elif metric.startswith("class_balanced_"):
            counts = [int(result.get(f"n_{name}", 0)) for name in CLASS_NAMES]
            reason = "one_or_more_true_classes_absent:n_PD,n_AP,n_HC=" + ",".join(
                str(count) for count in counts
            )
        elif metric.startswith("method_native_fallback_"):
            reason = (
                "native_fallback_not_applicable_or_unavailable:"
                f"denominator={int(result.get('method_native_fallback_denominator', 0))}"
            )
        else:
            reason = "metric_specific_denominator_unavailable"
        reasons[metric] = reason
    result["unestimable_metric_count"] = len(reasons)
    result["unestimable_reasons"] = json.dumps(
        reasons, sort_keys=True, separators=(",", ":")
    )
    return result

def bootstrap_whole_cohort(
    frame: pd.DataFrame,
    project_keys: np.ndarray,
    multiplicities: np.ndarray,
    *,
    minimum_valid_replicates: int = 1900,
) -> pd.DataFrame:
    keys = np.asarray(project_keys).astype(str)
    if keys.ndim != 1 or not len(keys) or len(set(keys)) != len(keys) or multiplicities.shape[1] != len(keys):
        raise ValueError("whole-cohort bootstrap cluster universe is invalid")
    replicate_count = multiplicities.shape[0]
    if replicate_count < minimum_valid_replicates:
        raise ValueError("insufficient whole-cohort bootstrap replicates")
    absolute: dict[tuple[int, str, str, str, str], tuple[float, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for budget in (3, 4):
        budget_frame = frame[frame["budget"].eq(budget)]
        for method in INTERVAL_METHODS:
            method_frame = budget_frame[budget_frame["method_id"].eq(method)]
            for panel, levels in PANEL_LEVELS.items():
                for level in levels:
                    group = method_frame[panel_mask(method_frame, panel, level)]
                    values = _bootstrap_group(group, keys, multiplicities)
                    for metric in INTERVAL_METRICS:
                        estimate, samples = values[metric]
                        absolute[(budget, method, panel, level, metric)] = (
                            estimate,
                            samples,
                        )
                        rows.append(
                            _interval_row(
                                budget=budget,
                                analysis="absolute",
                                method=method,
                                comparator="",
                                contrast_id="",
                                stratum=panel,
                                level=level,
                                metric=metric,
                                estimate=estimate,
                                values=samples,
                                minimum_valid=minimum_valid_replicates,
                            )
                        )
    for budget in (3, 4):
        for contrast_id, challenger, baseline in PAIRED_CONTRASTS:
            for panel, levels in PANEL_LEVELS.items():
                for level in levels:
                    for metric in INTERVAL_METRICS:
                        challenger_estimate, challenger_values = absolute[
                            (budget, challenger, panel, level, metric)
                        ]
                        baseline_estimate, baseline_values = absolute[
                            (budget, baseline, panel, level, metric)
                        ]
                        rows.append(
                            _interval_row(
                                budget=budget,
                                analysis="paired_difference_challenger_minus_baseline",
                                method=challenger,
                                comparator=baseline,
                                contrast_id=contrast_id,
                                stratum=panel,
                                level=level,
                                metric=metric,
                                estimate=challenger_estimate - baseline_estimate,
                                values=challenger_values - baseline_values,
                                minimum_valid=minimum_valid_replicates,
                            )
                        )
    return pd.DataFrame(rows)

def _bootstrap_group(
    group: pd.DataFrame,
    keys: np.ndarray,
    multiplicities: np.ndarray,
) -> dict[str, tuple[float, np.ndarray]]:
    replicate_count = multiplicities.shape[0]
    work = group.copy()
    truth = work["true_code"].to_numpy(np.int8)
    core = work["core_mask"].to_numpy(np.uint8)
    mention = work["mention_mask"].to_numpy(np.uint8)
    core_contains = set_contains_true(core, truth)
    mention_contains = set_contains_true(mention, truth)
    disease = np.isin(truth, [0, 1])
    ap = truth == 1
    contributions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, name in enumerate(CLASS_NAMES):
        selected = truth == code
        contributions[f"core_coverage_{name}"] = (
            (selected & core_contains).astype(float),
            selected.astype(float),
        )
        contributions[f"mention_coverage_{name}"] = (
            (selected & mention_contains).astype(float),
            selected.astype(float),
        )
        contributions[f"class_cost_{name}"] = (
            np.where(selected, work["robust_expected_cost"].to_numpy(float), 0.0),
            selected.astype(float),
        )
    contributions.update(
        {
            "mean_set_size": (BIT_COUNT[core].astype(float), np.ones(len(work))),
            "mean_SIG": (work["core_sig"].to_numpy(float), np.ones(len(work))),
            "AP_exclusion_rate": ((ap & ~core_contains).astype(float), ap.astype(float)),
            "dangerous_HC_singleton_rate": (
                (disease & (core == 4)).astype(float),
                disease.astype(float),
            ),
            "mean_robust_expected_cost": (
                work["robust_expected_cost"].to_numpy(float),
                np.ones(len(work)),
            ),
        }
    )
    cluster_columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric, (numerator, denominator) in contributions.items():
        work[f"__{metric}_n"] = numerator
        work[f"__{metric}_d"] = denominator
    columns = [
        column
        for metric in contributions
        for column in (f"__{metric}_n", f"__{metric}_d")
    ]
    if work.empty:
        aggregate = pd.DataFrame(0.0, index=keys, columns=columns)
    else:
        aggregate = (
            work.groupby("project_key", sort=True)[columns]
            .sum()
            .reindex(keys, fill_value=0.0)
        )
    output: dict[str, tuple[float, np.ndarray]] = {}
    class_coverage_samples: list[np.ndarray] = []
    class_coverage_estimates: list[float] = []
    class_cost_samples: list[np.ndarray] = []
    class_cost_estimates: list[float] = []
    for metric in contributions:
        numerator = aggregate[f"__{metric}_n"].to_numpy(float)
        denominator = aggregate[f"__{metric}_d"].to_numpy(float)
        estimate = _ratio(float(numerator.sum()), float(denominator.sum()))
        boot_n = multiplicities @ numerator
        boot_d = multiplicities @ denominator
        samples = np.divide(
            boot_n,
            boot_d,
            out=np.full(replicate_count, np.nan),
            where=boot_d > 0,
        )
        output[metric] = (estimate, samples)
        if metric.startswith("core_coverage_"):
            class_coverage_estimates.append(estimate)
            class_coverage_samples.append(samples)
        elif metric.startswith("class_cost_"):
            class_cost_estimates.append(estimate)
            class_cost_samples.append(samples)
    output["class_balanced_core_coverage"] = _balanced_samples(
        class_coverage_estimates, class_coverage_samples
    )
    output["class_balanced_robust_expected_cost"] = _balanced_samples(
        class_cost_estimates, class_cost_samples
    )
    return {metric: output[metric] for metric in INTERVAL_METRICS}

def _balanced_samples(
    estimates: Iterable[float], samples: Iterable[np.ndarray]
) -> tuple[float, np.ndarray]:
    estimate_array = np.asarray(list(estimates), dtype=float)
    sample_matrix = np.stack(list(samples), axis=1)
    estimate = _complete_mean(estimate_array.tolist())
    values = np.where(
        np.isfinite(sample_matrix).all(axis=1), sample_matrix.mean(axis=1), np.nan
    )
    return estimate, values

def _interval_row(
    *,
    budget: int,
    analysis: str,
    method: str,
    comparator: str,
    contrast_id: str,
    stratum: str,
    level: str,
    metric: str,
    estimate: float,
    values: np.ndarray,
    minimum_valid: int,
) -> dict[str, Any]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid):
        quantile = lambda probability: float(
            np.quantile(valid, probability, method="higher")
        )
        bootstrap_mean = float(valid.mean())
        lower_95 = quantile(0.025)
        upper_95 = quantile(0.975)
        lower_one = quantile(0.05)
        upper_one = quantile(0.95)
    else:
        bootstrap_mean = lower_95 = upper_95 = lower_one = upper_one = np.nan
    return {
        "budget": budget,
        "analysis": analysis,
        "method": method,
        "comparator": comparator,
        "contrast_id": contrast_id,
        "stratum": stratum,
        "level": level,
        "metric": metric,
        "estimate": estimate,
        "bootstrap_mean": bootstrap_mean,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "lower_one_sided_95": lower_one,
        "upper_one_sided_95": upper_one,
        "valid_replicates": int(len(valid)),
        "minimum_valid_replicates": minimum_valid,
        "inferentially_usable": bool(len(valid) >= minimum_valid),
        "interval_multiplicity_adjusted": False,
    }

def exact_safety_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in (3, 4):
        budget_frame = frame[frame["budget"].eq(budget)]
        for method in METHODS:
            method_frame = budget_frame[budget_frame["method_id"].eq(method)]
            for panel, levels in PANEL_LEVELS.items():
                for level in levels:
                    group = method_frame[panel_mask(method_frame, panel, level)].copy()
                    if group.empty:
                        continue
                    truth_by_key = group.groupby("project_key")["true_code"].first()
                    truth = group["true_code"].to_numpy(np.int8)
                    core = group["core_mask"].to_numpy(np.uint8)
                    mention = group["mention_mask"].to_numpy(np.uint8)
                    core_contains = set_contains_true(core, truth)
                    mention_contains = set_contains_true(mention, truth)
                    group["dangerous"] = np.isin(truth, [0, 1]) & (core == 4)
                    group["pd_to_hc"] = (truth == 0) & (core == 4)
                    group["ap_to_hc"] = (truth == 1) & (core == 4)
                    group["ap_excluded"] = (truth == 1) & ~core_contains
                    group["ap_mention_excluded"] = (truth == 1) & ~mention_contains
                    for event, eligible, column in (
                        (
                            "any_repeat_dangerous_HC_singleton",
                            truth_by_key.isin([0, 1]),
                            "dangerous",
                        ),
                        ("any_repeat_PD_to_HC_singleton", truth_by_key.eq(0), "pd_to_hc"),
                        ("any_repeat_AP_to_HC_singleton", truth_by_key.eq(1), "ap_to_hc"),
                        ("any_repeat_AP_exclusion", truth_by_key.eq(1), "ap_excluded"),
                        (
                            "any_repeat_AP_mention_exclusion",
                            truth_by_key.eq(1),
                            "ap_mention_excluded",
                        ),
                    ):
                        any_event = group.groupby("project_key")[column].any().reindex(
                            truth_by_key.index
                        )
                        selected = any_event[eligible]
                        events = int(selected.sum())
                        denominator = int(len(selected))
                        if denominator == 0:
                            continue
                        rows.append(
                            {
                                "budget": budget,
                                "method_id": method,
                                "stratum": panel,
                                "level": level,
                                "event": event,
                                "participant_events": events,
                                "participant_denominator": denominator,
                                "participant_rate": events / denominator,
                                "clopper_pearson_one_sided_upper_95": _binomial_upper(
                                    events, denominator, 0.05
                                ),
                            }
                        )
    return pd.DataFrame(rows)

def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else np.nan

def _complete_mean(values: list[float]) -> float:
    numeric = np.asarray(values, dtype=float)
    return float(numeric.mean()) if np.isfinite(numeric).all() else np.nan
