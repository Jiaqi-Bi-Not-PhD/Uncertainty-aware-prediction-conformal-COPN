"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from .common import CLASS_NAMES, KeyedConfig, clean_category, is_observed_value, numeric_series

@dataclass
class CatBoostSourceSpec:
    source_id: str
    source_group: str
    implementation_level: int
    kind: str
    feature_name: str
    exclude_reason: str = ""
    winsor_low: float | None = None
    winsor_high: float | None = None
    categories: list[str] | None = None

class CatBoostPreprocessor:
    """Fold-scoped CatBoost feature construction for the keyed value-first branch."""

    def __init__(self, config: KeyedConfig, budget: int) -> None:
        self.config = config
        self.budget = int(budget)
        self.specs_: list[CatBoostSourceSpec] = []
        self.feature_names_: list[str] = []
        self.cat_feature_indices_: list[int] = []
        self.screening_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> "CatBoostPreprocessor":
        rows: list[dict[str, Any]] = []
        specs: list[CatBoostSourceSpec] = []
        meta_by_source = metadata.set_index("source_id", drop=False)

        for source_id in X.columns:
            meta = meta_by_source.loc[source_id].to_dict()
            reason = self._base_exclusion_reason(meta)
            observed_mask = X[source_id].map(is_observed_value).to_numpy(dtype=bool)
            observed_fraction = float(observed_mask.mean())
            class_counts = {
                label: int(np.sum(observed_mask & (y == code)))
                for code, label in enumerate(CLASS_NAMES)
            }
            if reason is None and observed_fraction < self.config.min_observed_fraction:
                reason = "below_min_observed_fraction"
            for label, minimum in self.config.min_observed_per_class.items():
                if reason is None and class_counts.get(label, 0) < int(minimum):
                    reason = f"below_min_observed_count:{label}"

            kind = "excluded"
            if reason is None:
                kind = self._infer_kind(X[source_id], str(meta.get("field_type", "")))
                if kind == "numeric":
                    spec = self._fit_numeric(source_id, X[source_id], meta)
                elif kind == "categorical":
                    spec = self._fit_categorical(source_id, X[source_id], meta)
                else:
                    spec = CatBoostSourceSpec(
                        source_id,
                        str(meta.get("source_group", "")),
                        int(meta.get("implementation_level", 4)),
                        kind,
                        source_id,
                        exclude_reason="unusable_type",
                    )
                if spec.exclude_reason:
                    reason = spec.exclude_reason
                else:
                    specs.append(spec)

            rows.append(
                {
                    "source_id": source_id,
                    "source_sheet": meta.get("source_sheet", ""),
                    "source_group": meta.get("source_group", ""),
                    "implementation_level": int(meta.get("implementation_level", 4)),
                    "budget": self.budget,
                    "observed_fraction_train": observed_fraction,
                    "observed_parkinson_train": class_counts["Parkinson"],
                    "observed_atypical_train": class_counts["Atypical"],
                    "observed_healthy_train": class_counts["Healthy"],
                    "kind": kind,
                    "eligible": reason is None,
                    "exclude_reason": reason or "",
                    "feature_count": 1 if reason is None else 0,
                }
            )

        self.specs_ = specs
        self.feature_names_ = [spec.feature_name for spec in specs]
        self.cat_feature_indices_ = [i for i, spec in enumerate(specs) if spec.kind == "categorical"]
        self.screening_ = pd.DataFrame(rows)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        columns: dict[str, Any] = {}
        for spec in self.specs_:
            if spec.kind == "numeric":
                columns[spec.feature_name] = self._transform_numeric(X[spec.source_id], spec)
            elif spec.kind == "categorical":
                columns[spec.feature_name] = self._transform_categorical(X[spec.source_id], spec)
            else:
                raise ValueError(f"Unsupported CatBoost source kind: {spec.kind}")
        return pd.DataFrame(columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
        self.fit(X, y, metadata)
        return self.transform(X)

    def feature_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_id": spec.source_id,
                    "feature_name": spec.feature_name,
                    "kind": spec.kind,
                    "source_group": spec.source_group,
                    "implementation_level": spec.implementation_level,
                }
                for spec in self.specs_
            ]
        )

    def _base_exclusion_reason(self, meta: dict[str, Any]) -> str | None:
        if bool(meta.get("excluded", False)):
            return str(meta.get("exclude_reason", "metadata_excluded"))
        level = int(meta.get("implementation_level", 4))
        if level > self.budget:
            return "outside_budget"
        if int(meta.get("conflict_rows", 0) or 0) > 0:
            return "harmonization_conflict"
        return None

    def _infer_kind(self, series: pd.Series, field_type: str) -> str:
        observed = series.loc[series.map(is_observed_value)]
        if observed.empty:
            return "unusable"
        parsed = numeric_series(observed)
        numeric_fraction = float(parsed.notna().mean()) if len(observed) else 0.0
        if numeric_fraction >= self.config.numeric_parse_threshold:
            return "numeric"
        unique_count = int(observed.map(clean_category).nunique(dropna=True))
        if unique_count > max(self.config.max_categories * 5, 100):
            return "unusable_text"
        return "categorical"

    def _fit_numeric(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> CatBoostSourceSpec:
        parsed = numeric_series(series)
        observed = parsed.dropna()
        if len(observed) < 2 or float(observed.std(ddof=1)) <= 0:
            return CatBoostSourceSpec(
                source_id,
                str(meta.get("source_group", "")),
                int(meta.get("implementation_level", 4)),
                "numeric",
                source_id,
                exclude_reason="numeric_quasi_constant",
            )
        return CatBoostSourceSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="numeric",
            feature_name=source_id,
            winsor_low=float(observed.quantile(self.config.winsor_lower_q)),
            winsor_high=float(observed.quantile(self.config.winsor_upper_q)),
        )

    def _fit_categorical(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> CatBoostSourceSpec:
        observed = series.loc[series.map(is_observed_value)].map(clean_category)
        if observed.empty:
            return CatBoostSourceSpec(
                source_id,
                str(meta.get("source_group", "")),
                int(meta.get("implementation_level", 4)),
                "categorical",
                source_id,
                exclude_reason="categorical_too_few_observed",
            )
        counts = observed.value_counts()
        kept = counts.loc[counts >= self.config.rare_category_min_count].index.tolist()
        kept = kept[: self.config.max_categories]
        categories = sorted(set(kept + ["__OTHER__", "__MISSING__"]))
        if len(categories) <= 1:
            return CatBoostSourceSpec(
                source_id,
                str(meta.get("source_group", "")),
                int(meta.get("implementation_level", 4)),
                "categorical",
                source_id,
                exclude_reason="categorical_quasi_constant",
            )
        return CatBoostSourceSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="categorical",
            feature_name=source_id,
            categories=categories,
        )

    def _transform_numeric(self, series: pd.Series, spec: CatBoostSourceSpec) -> pd.Series:
        parsed = numeric_series(series)
        return parsed.clip(lower=spec.winsor_low, upper=spec.winsor_high)

    def _transform_categorical(self, series: pd.Series, spec: CatBoostSourceSpec) -> pd.Series:
        assert spec.categories is not None
        keep = set(spec.categories)
        tokens = series.map(lambda value: clean_category(value))
        return tokens.map(lambda token: token if token in keep else "__OTHER__").astype(str)
