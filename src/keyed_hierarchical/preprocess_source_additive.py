"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from .common import CLASS_NAMES, KeyedConfig, clean_category, is_observed_value, normalize_text, numeric_series

@dataclass
class ValueSpec:
    source_id: str
    source_group: str
    implementation_level: int
    kind: str
    value_feature_names: list[str]
    missing_feature_names: list[str]
    exclude_reason: str = ""
    center: float | None = None
    scale: float | None = None
    winsor_low: float | None = None
    winsor_high: float | None = None
    categories: list[str] | None = None
    missing_states: list[str] | None = None

class SourceAdditivePreprocessor:
    """Build value and missingness blocks separately for Model 4."""

    def __init__(
        self,
        config: KeyedConfig,
        budget: int,
        class_names: tuple[str, ...] | None = None,
        class_codes: tuple[int, ...] | None = None,
        min_observed_per_class: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self.budget = int(budget)
        self.class_names = tuple(class_names or CLASS_NAMES)
        if class_codes is None:
            self.class_codes = tuple(range(len(self.class_names)))
        else:
            self.class_codes = tuple(int(code) for code in class_codes)
        if len(self.class_names) != len(self.class_codes):
            raise ValueError("class_names and class_codes must have the same length.")
        self.min_observed_per_class = (
            dict(min_observed_per_class)
            if min_observed_per_class is not None
            else {name: int(config.min_observed_per_class.get(name, 0)) for name in self.class_names}
        )
        self.specs_: list[ValueSpec] = []
        self.screening_: pd.DataFrame | None = None
        self.value_feature_names_: list[str] = []
        self.missing_feature_names_: list[str] = []
        self.value_groups_: dict[str, list[int]] = {}
        self.missing_groups_: dict[str, list[int]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> "SourceAdditivePreprocessor":
        rows: list[dict[str, Any]] = []
        specs: list[ValueSpec] = []
        meta_by_source = metadata.set_index("source_id", drop=False)
        for source_id in X.columns:
            meta = meta_by_source.loc[source_id].to_dict()
            reason = self._base_exclusion_reason(meta)
            observed_mask = X[source_id].map(is_observed_value).to_numpy(dtype=bool)
            observed_fraction = float(observed_mask.mean())
            class_counts = {
                label: int(np.sum(observed_mask & (y == code)))
                for label, code in zip(self.class_names, self.class_codes)
            }
            if reason is None and observed_fraction < self.config.min_observed_fraction:
                reason = "below_min_observed_fraction"
            for label, minimum in self.min_observed_per_class.items():
                if reason is None and class_counts.get(label, 0) < int(minimum):
                    reason = f"below_min_observed_count:{label}"

            kind = "excluded"
            value_count = 0
            missing_count = 0
            if reason is None:
                kind = self._infer_kind(X[source_id], str(meta.get("field_type", "")))
                if kind == "numeric":
                    spec = self._fit_numeric(source_id, X[source_id], meta)
                elif kind == "categorical":
                    spec = self._fit_categorical(source_id, X[source_id], meta)
                else:
                    spec = ValueSpec(
                        source_id=source_id,
                        source_group=str(meta.get("source_group", "")),
                        implementation_level=int(meta.get("implementation_level", 4)),
                        kind=kind,
                        value_feature_names=[],
                        missing_feature_names=[],
                        exclude_reason="unusable_type",
                    )
                if spec.exclude_reason:
                    reason = spec.exclude_reason
                    kind = spec.kind
                else:
                    specs.append(spec)
                    value_count = len(spec.value_feature_names)
                    missing_count = len(spec.missing_feature_names)

            rows.append(
                {
                    "source_id": source_id,
                    "source_sheet": meta.get("source_sheet", ""),
                    "source_group": meta.get("source_group", ""),
                    "implementation_level": int(meta.get("implementation_level", 4)),
                    "budget": self.budget,
                    "observed_fraction_train": observed_fraction,
                    **{f"observed_{normalize_text(label)}_train": int(class_counts.get(label, 0)) for label in self.class_names},
                    "kind": kind,
                    "eligible": reason is None,
                    "exclude_reason": reason or "",
                    "value_feature_count": value_count,
                    "missing_feature_count": missing_count,
                }
            )

        self.specs_ = specs
        self.screening_ = pd.DataFrame(rows)
        self._refresh_feature_index()
        return self

    def transform_value(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for spec in self.specs_:
            if spec.kind == "numeric":
                blocks.append(self._transform_numeric_value(X[spec.source_id], spec))
            elif spec.kind == "categorical":
                blocks.append(self._transform_categorical_value(X[spec.source_id], spec))
            else:
                raise ValueError(f"Unsupported source kind for value branch: {spec.kind}")
        if not blocks:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(blocks).astype(float)

    def transform_missingness(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for spec in self.specs_:
            blocks.append(self._transform_missingness(X[spec.source_id], spec))
        if not blocks:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(blocks).astype(float)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X, y, metadata)
        return self.transform_value(X), self.transform_missingness(X)

    def source_mapping(self) -> pd.DataFrame:
        rows = []
        for spec in self.specs_:
            for pos in self.value_groups_.get(spec.source_id, []):
                rows.append(
                    {
                        "source_id": spec.source_id,
                        "block_kind": "value",
                        "feature_index": int(pos),
                        "feature_name": self.value_feature_names_[pos],
                    }
                )
            for pos in self.missing_groups_.get(spec.source_id, []):
                rows.append(
                    {
                        "source_id": spec.source_id,
                        "block_kind": "missingness",
                        "feature_index": int(pos),
                        "feature_name": self.missing_feature_names_[pos],
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

    def _fit_numeric(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> ValueSpec:
        parsed = numeric_series(series)
        observed = parsed.dropna()
        if len(observed) < 2:
            return ValueSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "numeric", [], [], "numeric_too_few_observed")
        low = float(observed.quantile(self.config.winsor_lower_q))
        high = float(observed.quantile(self.config.winsor_upper_q))
        clipped = observed.clip(lower=low, upper=high)
        scale = float(clipped.std(ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            return ValueSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "numeric", [], [], "numeric_quasi_constant")
        center = float(clipped.mean())
        missing_states = ["__MISSING__"]
        return ValueSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="numeric",
            value_feature_names=[f"{source_id}__value"],
            missing_feature_names=[f"{source_id}__missing"],
            center=center,
            scale=scale,
            winsor_low=low,
            winsor_high=high,
            missing_states=missing_states,
        )

    def _fit_categorical(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> ValueSpec:
        observed = series.loc[series.map(is_observed_value)].map(clean_category)
        if observed.empty:
            return ValueSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "categorical", [], [], "categorical_too_few_observed")
        counts = observed.value_counts()
        kept = counts.loc[counts >= self.config.rare_category_min_count].index.tolist()
        kept = kept[: self.config.max_categories]
        categories = sorted(set(kept))
        if len(categories) < 2:
            return ValueSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "categorical", [], [], "categorical_quasi_constant")
        return ValueSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="categorical",
            value_feature_names=[f"{source_id}__cat__{normalize_text(cat)}" for cat in categories],
            missing_feature_names=[f"{source_id}__missing"],
            categories=categories,
            missing_states=["__MISSING__"],
        )

    def _transform_numeric_value(self, series: pd.Series, spec: ValueSpec) -> np.ndarray:
        parsed = numeric_series(series)
        observed = parsed.notna().to_numpy(dtype=bool)
        clipped = parsed.clip(lower=spec.winsor_low, upper=spec.winsor_high)
        values = ((clipped - spec.center) / spec.scale).to_numpy(dtype=float, copy=True).reshape(-1, 1)
        values[~observed, 0] = 0.0
        return values

    def _transform_categorical_value(self, series: pd.Series, spec: ValueSpec) -> np.ndarray:
        assert spec.categories is not None
        observed = series.map(is_observed_value)
        tokens = series.map(clean_category)
        out = np.zeros((len(series), len(spec.categories)), dtype=float)
        ref = spec.categories[-1]
        for j, cat in enumerate(spec.categories):
            if cat == ref:
                continue
            out[:, j] = ((observed) & (tokens == cat)).astype(float).to_numpy()
        return out[:, : len(spec.value_feature_names)]

    def _transform_missingness(self, series: pd.Series, spec: ValueSpec) -> np.ndarray:
        observed = series.map(is_observed_value).to_numpy(dtype=bool)
        missing = (~observed).astype(float).reshape(-1, 1)
        return missing

    def _refresh_feature_index(self) -> None:
        self.value_feature_names_ = []
        self.missing_feature_names_ = []
        self.value_groups_ = {}
        self.missing_groups_ = {}
        for spec in self.specs_:
            v_start = len(self.value_feature_names_)
            self.value_feature_names_.extend(spec.value_feature_names)
            self.value_groups_[spec.source_id] = list(range(v_start, len(self.value_feature_names_)))
            m_start = len(self.missing_feature_names_)
            self.missing_feature_names_.extend(spec.missing_feature_names)
            self.missing_groups_[spec.source_id] = list(range(m_start, len(self.missing_feature_names_)))
