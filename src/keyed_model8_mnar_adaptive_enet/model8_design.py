"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from keyed_model1.common import CLASS_NAMES, is_observed_value, normalize_text, numeric_series

from .config_model8 import Model8Config

from .model8_imputation import MixedChainedImputer

@dataclass
class Model8FoldDesign:
    imputer: MixedChainedImputer
    train_mi: list[np.ndarray]
    valid_mi: list[np.ndarray]
    groups: dict[str, list[int]]
    feature_names: list[str]
    eligibility: pd.DataFrame
    threshold_audit: pd.DataFrame
    imputation_diagnostics: pd.DataFrame
    mapping: pd.DataFrame

def _base_exclusion_reason(meta: dict[str, Any], budget: int) -> str | None:
    if bool(meta.get("excluded", False)):
        return str(meta.get("exclude_reason", "metadata_excluded"))
    if int(meta.get("implementation_level", 4)) > int(budget):
        return "outside_budget"
    if int(meta.get("conflict_rows", 0) or 0) > 0:
        return "harmonization_conflict"
    text = normalize_text(f"{meta.get('source_id', '')} {meta.get('source_sheet', '')} {meta.get('raw_header', '')}")
    if any(token in text for token in ("final_disease_diagnosis", "specific_disease", "diagnosis")):
        return "diagnosis_leakage"
    return None

def _infer_kind(series: pd.Series, field_type: str, config: Model8Config) -> str:
    observed = series.loc[series.map(is_observed_value)]
    if observed.empty:
        return "unusable"
    numeric_fraction = float(numeric_series(observed).notna().mean())
    if numeric_fraction >= float(config.numeric_parse_threshold):
        return "numeric"
    if normalize_text(field_type) in {"notes", "descriptive", "file"}:
        return "unusable_text"
    unique_count = int(observed.astype(str).nunique(dropna=True))
    if unique_count > max(int(config.max_categories) * 5, 100):
        return "unusable_text"
    return "categorical"

def source_eligibility_audit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metadata: pd.DataFrame,
    config: Model8Config,
    budget: int,
    branch: str,
    outer_split_id: str,
    inner_split_id: str = "",
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    meta_by_source = metadata.set_index("source_id", drop=False)
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    eligible: list[str] = []
    for source_id in X_train.columns:
        meta = meta_by_source.loc[source_id].to_dict()
        reason = _base_exclusion_reason(meta, budget)
        observed_mask = X_train[source_id].map(is_observed_value).to_numpy(dtype=bool)
        observed_fraction = float(observed_mask.mean()) if len(observed_mask) else 0.0
        missing_fraction = float(1.0 - observed_fraction)
        class_counts = {label: int(np.sum(observed_mask & (y_train == code))) for code, label in enumerate(CLASS_NAMES)}
        kind = "excluded"

        if reason is None and missing_fraction > float(config.tau_miss):
            reason = "missing_fraction_gt_50pct"
        if reason is None and observed_fraction < float(config.min_observed_fraction):
            reason = "below_min_observed_fraction"
        for label, minimum in config.min_observed_per_class.items():
            if reason is None and class_counts.get(label, 0) < int(minimum):
                reason = f"below_min_observed_count:{label}"
        if reason is None:
            kind = _infer_kind(X_train[source_id], str(meta.get("field_type", "")), config)
            if kind == "numeric":
                parsed = numeric_series(X_train[source_id]).dropna()
                if len(parsed) < 2:
                    reason = "numeric_too_few_observed"
                else:
                    scale = float(parsed.std(ddof=1))
                    if not np.isfinite(scale) or scale <= 0:
                        reason = "numeric_quasi_constant"
            elif kind == "categorical":
                observed = X_train[source_id].loc[X_train[source_id].map(is_observed_value)].astype(str)
                freq = observed.value_counts(normalize=True)
                if len(freq) < 2:
                    reason = "categorical_quasi_constant"
                elif float(freq.iloc[0]) >= float(config.quasi_constant_threshold):
                    reason = "categorical_quasi_constant"
            else:
                reason = kind

        is_eligible = reason is None
        if is_eligible:
            eligible.append(source_id)
        rows.append(
            {
                "outer_split_id": outer_split_id,
                "inner_split_id": inner_split_id,
                "budget": int(budget),
                "branch": branch,
                "source_id": source_id,
                "source_sheet": meta.get("source_sheet", ""),
                "source_group": meta.get("source_group", ""),
                "raw_header": meta.get("raw_header", ""),
                "implementation_level": int(meta.get("implementation_level", 4)),
                "kind": kind,
                "eligible": bool(is_eligible),
                "exclude_reason": reason or "",
                "missing_fraction_train": missing_fraction,
                "observed_fraction_train": observed_fraction,
                "observed_parkinson_train": class_counts["Parkinson"],
                "observed_atypical_train": class_counts["Atypical"],
                "observed_healthy_train": class_counts["Healthy"],
                "tau_miss": float(config.tau_miss),
                "excluded_by_50pct": bool(missing_fraction > float(config.tau_miss)),
            }
        )
        threshold_rows.append(
            {
                "outer_split_id": outer_split_id,
                "inner_split_id": inner_split_id,
                "budget": int(budget),
                "branch": branch,
                "source_id": source_id,
                "missing_fraction_train": missing_fraction,
                "tau_miss": float(config.tau_miss),
                "passes_50pct_threshold": bool(missing_fraction <= float(config.tau_miss)),
                "eligible": bool(is_eligible),
            }
        )
    return eligible, pd.DataFrame(rows), pd.DataFrame(threshold_rows)

def build_fold_design(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Model8Config,
    budget: int,
    branch: str,
    outer_split_id: str,
    inner_split_id: str,
    mi: int,
    seed_context: str,
) -> Model8FoldDesign:
    source_ids, eligibility, threshold_audit = source_eligibility_audit(
        X_train,
        y_train,
        metadata,
        config,
        budget=budget,
        branch=branch,
        outer_split_id=outer_split_id,
        inner_split_id=inner_split_id,
    )
    imputer = MixedChainedImputer(
        rare_category_min_count=config.rare_category_min_count,
        max_categories=config.max_categories,
        numeric_parse_threshold=config.numeric_parse_threshold,
        winsor_lower_q=config.winsor_lower_q,
        winsor_upper_q=config.winsor_upper_q,
        imputer_type=config.imputer_type,
    ).fit(X_train, source_ids, metadata)
    train_mi = imputer.transform_many(X_train, mi=mi, base_seed=config.random_state, seed_context=f"{seed_context}:train")
    valid_mi = imputer.transform_many(X_valid, mi=mi, base_seed=config.random_state, seed_context=f"{seed_context}:valid")
    diagnostics = imputer.diagnostics_.copy()
    if not diagnostics.empty:
        diagnostics.insert(0, "inner_split_id", inner_split_id)
        diagnostics.insert(0, "outer_split_id", outer_split_id)
        diagnostics.insert(2, "budget", int(budget))
        diagnostics.insert(3, "branch", branch)
        diagnostics["mi"] = int(mi)
    mapping = imputer.source_mapping()
    if not mapping.empty:
        mapping.insert(0, "inner_split_id", inner_split_id)
        mapping.insert(0, "outer_split_id", outer_split_id)
        mapping.insert(2, "budget", int(budget))
        mapping.insert(3, "branch", branch)
    return Model8FoldDesign(
        imputer=imputer,
        train_mi=train_mi,
        valid_mi=valid_mi,
        groups=imputer.groups_,
        feature_names=imputer.feature_names_,
        eligibility=eligibility,
        threshold_audit=threshold_audit,
        imputation_diagnostics=diagnostics,
        mapping=mapping,
    )
