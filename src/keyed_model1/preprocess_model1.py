"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from .common import CLASS_NAMES, KeyedConfig, clean_category, is_observed_value, normalize_text, numeric_series

@dataclass
class SourceSpec:
    source_id: str
    source_group: str
    implementation_level: int
    kind: str
    feature_names: list[str]
    exclude_reason: str = ""
    center: float | None = None
    scale: float | None = None
    winsor_low: float | None = None
    winsor_high: float | None = None
    categories: list[str] | None = None

class Model1Preprocessor:
    """Fold-scoped source screening and design-matrix construction for Model 1."""

    def __init__(self, config: KeyedConfig, budget: int) -> None:
        self.config = config
        self.budget = int(budget)
        self.specs_: list[SourceSpec] = []
        self.screening_: pd.DataFrame | None = None
        self.feature_names_: list[str] = []
        self.groups_: dict[str, list[int]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> "Model1Preprocessor":
        rows: list[dict[str, Any]] = []
        specs: list[SourceSpec] = []

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
            feature_names: list[str] = []
            if reason is None:
                kind = self._infer_kind(X[source_id], str(meta.get("field_type", "")))
                if kind == "numeric":
                    spec = self._fit_numeric(source_id, X[source_id], meta)
                elif kind == "categorical":
                    spec = self._fit_categorical(source_id, X[source_id], meta)
                else:
                    spec = SourceSpec(
                        source_id=source_id,
                        source_group=str(meta.get("source_group", "")),
                        implementation_level=int(meta.get("implementation_level", 4)),
                        kind=kind,
                        feature_names=[],
                        exclude_reason="unusable_type",
                    )
                if spec.exclude_reason:
                    reason = spec.exclude_reason
                    kind = spec.kind
                else:
                    feature_names = spec.feature_names
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
                    "feature_count": len(feature_names),
                }
            )

        self.specs_ = specs
        self.screening_ = pd.DataFrame(rows)
        self._refresh_feature_index()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for spec in self.specs_:
            if spec.kind == "numeric":
                blocks.append(self._transform_numeric(X[spec.source_id], spec))
            elif spec.kind == "categorical":
                blocks.append(self._transform_categorical(X[spec.source_id], spec))
            else:
                raise ValueError(f"Unsupported fitted source kind: {spec.kind}")
        if not blocks:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(blocks).astype(float)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        self.fit(X, y, metadata)
        return self.transform(X)

    def source_mapping(self) -> pd.DataFrame:
        rows = []
        for spec in self.specs_:
            for pos in self.groups_.get(spec.source_id, []):
                rows.append(
                    {
                        "source_id": spec.source_id,
                        "implementation_level": spec.implementation_level,
                        "kind": spec.kind,
                        "feature_index": int(pos),
                        "feature_name": self.feature_names_[pos],
                    }
                )
        return pd.DataFrame(rows)

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
        field_type_norm = normalize_text(field_type)
        parsed = numeric_series(observed)
        numeric_fraction = float(parsed.notna().mean()) if len(observed) else 0.0
        if numeric_fraction >= self.config.numeric_parse_threshold:
            return "numeric"
        if field_type_norm in {"notes", "descriptive", "file"}:
            return "unusable"
        unique_count = int(observed.map(clean_category).nunique(dropna=True))
        if unique_count > max(self.config.max_categories * 5, 100):
            return "unusable_text"
        return "categorical"

    def _fit_numeric(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> SourceSpec:
        parsed = numeric_series(series)
        observed = parsed.dropna()
        if len(observed) < 2:
            return SourceSpec(
                source_id,
                str(meta.get("source_group", "")),
                int(meta.get("implementation_level", 4)),
                "numeric",
                [],
                exclude_reason="numeric_too_few_observed",
            )
        low = float(observed.quantile(self.config.winsor_lower_q))
        high = float(observed.quantile(self.config.winsor_upper_q))
        clipped = observed.clip(lower=low, upper=high)
        scale = float(clipped.std(ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            return SourceSpec(
                source_id,
                str(meta.get("source_group", "")),
                int(meta.get("implementation_level", 4)),
                "numeric",
                [],
                exclude_reason="numeric_quasi_constant",
            )
        center = float(clipped.mean())
        feature_names = [f"{source_id}__value", f"{source_id}__missing_unknown"]
        return SourceSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="numeric",
            feature_names=feature_names,
            center=center,
            scale=scale,
            winsor_low=low,
            winsor_high=high,
        )

    def _fit_categorical(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> SourceSpec:
        observed = series.loc[series.map(is_observed_value)].map(clean_category)
        if observed.empty:
            return SourceSpec(
                source_id,
                str(meta.get("source_group", "")),
                int(meta.get("implementation_level", 4)),
                "categorical",
                [],
                exclude_reason="categorical_too_few_observed",
            )
        counts = observed.value_counts()
        kept = counts.loc[counts >= self.config.rare_category_min_count].index.tolist()
        kept = kept[: self.config.max_categories]
        categories = sorted(set(kept + ["__OTHER__", "__MISSING__"]))
        if len(categories) <= 1:
            return SourceSpec(
                source_id,
                str(meta.get("source_group", "")),
                int(meta.get("implementation_level", 4)),
                "categorical",
                [],
                exclude_reason="categorical_quasi_constant",
            )
        return SourceSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="categorical",
            feature_names=[f"{source_id}__cat__{normalize_text(cat)}" for cat in categories],
            categories=categories,
        )

    def _transform_numeric(self, series: pd.Series, spec: SourceSpec) -> np.ndarray:
        parsed = numeric_series(series)
        observed = parsed.notna().to_numpy(dtype=bool)
        clipped = parsed.clip(lower=spec.winsor_low, upper=spec.winsor_high)
        values = ((clipped - spec.center) / spec.scale).to_numpy(dtype=float).copy()
        values[~observed] = 0.0
        missing = (~observed).astype(float)
        return np.column_stack([values, missing])

    def _transform_categorical(self, series: pd.Series, spec: SourceSpec) -> np.ndarray:
        assert spec.categories is not None
        keep = set(spec.categories)
        tokens = series.map(lambda v: clean_category(v))
        tokens = tokens.map(lambda v: v if v in keep else "__OTHER__")
        out = np.zeros((len(series), len(spec.categories)), dtype=float)
        index = {category: i for i, category in enumerate(spec.categories)}
        for row, token in enumerate(tokens):
            out[row, index.get(token, index["__OTHER__"])] = 1.0
        return out

    def _refresh_feature_index(self) -> None:
        self.feature_names_ = []
        self.groups_ = {}
        for spec in self.specs_:
            start = len(self.feature_names_)
            self.feature_names_.extend(spec.feature_names)
            self.groups_[spec.source_id] = list(range(start, len(self.feature_names_)))
