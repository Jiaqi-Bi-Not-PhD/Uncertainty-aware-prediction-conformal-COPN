"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import hashlib

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from keyed_model1.common import clean_category, is_observed_value, normalize_text, numeric_series

def deterministic_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return int(digest, 16) % (2**32 - 1)

@dataclass
class ImputationSpec:
    source_id: str
    kind: str
    feature_names: list[str]
    categories: list[str] | None = None
    center: float | None = None
    scale: float | None = None
    winsor_low: float | None = None
    winsor_high: float | None = None
    observed_values: np.ndarray | None = None
    category_values: list[str] | None = None
    category_probabilities: np.ndarray | None = None
    mode: str | None = None

class MixedChainedImputer:
    """Fold-scoped mixed-type imputer used by the practical Model 8 implementation.

    Numeric sources are parsed, winsorized, standardized, and stochastically
    imputed from the empirical training distribution. Nominal/binary sources use
    stochastic hot-deck draws from observed training levels with a mode fallback.
    A missingness indicator is retained in every source block.
    """

    def __init__(
        self,
        rare_category_min_count: int = 5,
        max_categories: int = 40,
        numeric_parse_threshold: float = 0.90,
        winsor_lower_q: float = 0.01,
        winsor_upper_q: float = 0.99,
        imputer_type: str = "mixed_chained",
    ) -> None:
        self.rare_category_min_count = int(rare_category_min_count)
        self.max_categories = int(max_categories)
        self.numeric_parse_threshold = float(numeric_parse_threshold)
        self.winsor_lower_q = float(winsor_lower_q)
        self.winsor_upper_q = float(winsor_upper_q)
        self.imputer_type = imputer_type
        self.specs_: list[ImputationSpec] = []
        self.groups_: dict[str, list[int]] = {}
        self.feature_names_: list[str] = []
        self.diagnostics_: pd.DataFrame = pd.DataFrame()

    def fit(self, X: pd.DataFrame, source_ids: list[str], metadata: pd.DataFrame) -> "MixedChainedImputer":
        specs: list[ImputationSpec] = []
        rows: list[dict[str, Any]] = []
        meta_by_source = metadata.set_index("source_id", drop=False) if not metadata.empty else pd.DataFrame()
        for source_id in source_ids:
            series = X[source_id]
            meta = meta_by_source.loc[source_id].to_dict() if source_id in meta_by_source.index else {}
            kind = self._infer_kind(series, str(meta.get("field_type", "")))
            missing_fraction = float(1.0 - series.map(is_observed_value).mean())
            if kind == "numeric":
                spec = self._fit_numeric(source_id, series)
            else:
                spec = self._fit_categorical(source_id, series)
            specs.append(spec)
            rows.append(
                {
                    "source_id": source_id,
                    "imputer_type": self.imputer_type,
                    "kind": spec.kind,
                    "missing_fraction_train": missing_fraction,
                    "feature_count": len(spec.feature_names),
                    "category_count": len(spec.categories or []),
                    "numeric_center": spec.center,
                    "numeric_scale": spec.scale,
                }
            )
        self.specs_ = specs
        self.diagnostics_ = pd.DataFrame(rows)
        self._refresh_feature_index()
        return self

    def transform_many(self, X: pd.DataFrame, mi: int, base_seed: int, seed_context: str) -> list[np.ndarray]:
        return [self.transform(X, deterministic_seed(base_seed, seed_context, m)) for m in range(int(mi))]

    def transform(self, X: pd.DataFrame, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        blocks = []
        for spec in self.specs_:
            series = X[spec.source_id]
            if spec.kind == "numeric":
                blocks.append(self._transform_numeric(series, spec, rng))
            elif spec.kind == "categorical":
                blocks.append(self._transform_categorical(series, spec, rng))
            else:
                raise ValueError(f"Unsupported imputation spec kind: {spec.kind}")
        if not blocks:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(blocks).astype(float)

    def source_mapping(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for spec in self.specs_:
            for pos in self.groups_.get(spec.source_id, []):
                rows.append(
                    {
                        "source_id": spec.source_id,
                        "feature_index": int(pos),
                        "feature_name": self.feature_names_[pos],
                        "kind": spec.kind,
                    }
                )
        return pd.DataFrame(rows)

    def _infer_kind(self, series: pd.Series, field_type: str) -> str:
        observed = series.loc[series.map(is_observed_value)]
        if observed.empty:
            return "categorical"
        parsed = numeric_series(observed)
        if float(parsed.notna().mean()) >= self.numeric_parse_threshold:
            return "numeric"
        field_type_norm = normalize_text(field_type)
        if field_type_norm in {"calc", "slider"}:
            return "numeric"
        return "categorical"

    def _fit_numeric(self, source_id: str, series: pd.Series) -> ImputationSpec:
        parsed = numeric_series(series)
        observed = parsed.dropna().astype(float)
        if observed.empty:
            observed = pd.Series([0.0])
        low = float(observed.quantile(self.winsor_lower_q))
        high = float(observed.quantile(self.winsor_upper_q))
        clipped = observed.clip(lower=low, upper=high)
        center = float(clipped.mean())
        scale = float(clipped.std(ddof=1))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return ImputationSpec(
            source_id=source_id,
            kind="numeric",
            feature_names=[f"{source_id}__value_imputed", f"{source_id}__missing_indicator"],
            center=center,
            scale=scale,
            winsor_low=low,
            winsor_high=high,
            observed_values=clipped.to_numpy(dtype=float),
        )

    def _fit_categorical(self, source_id: str, series: pd.Series) -> ImputationSpec:
        observed = series.loc[series.map(is_observed_value)].map(clean_category)
        if observed.empty:
            observed = pd.Series(["__OTHER__"])
        counts = observed.value_counts()
        kept = counts.loc[counts >= self.rare_category_min_count].index.tolist()
        if not kept:
            kept = counts.index[: min(len(counts), self.max_categories)].tolist()
        kept = kept[: self.max_categories]
        mapped = observed.map(lambda value: value if value in set(kept) else "__OTHER__")
        probs = mapped.value_counts(normalize=True).sort_index()
        categories = sorted(set(probs.index.tolist() + ["__OTHER__"]))
        prob = np.array([float(probs.get(cat, 0.0)) for cat in categories], dtype=float)
        if prob.sum() <= 0:
            prob[:] = 1.0 / len(prob)
        else:
            prob /= prob.sum()
        mode = str(mapped.mode().iloc[0]) if not mapped.mode().empty else categories[0]
        return ImputationSpec(
            source_id=source_id,
            kind="categorical",
            feature_names=[f"{source_id}__cat__{normalize_text(cat)}" for cat in categories] + [f"{source_id}__missing_indicator"],
            categories=categories,
            category_values=categories,
            category_probabilities=prob,
            mode=mode,
        )

    def _transform_numeric(self, series: pd.Series, spec: ImputationSpec, rng: np.random.Generator) -> np.ndarray:
        parsed = numeric_series(series)
        missing = parsed.isna().to_numpy(dtype=bool)
        values = parsed.astype(float).to_numpy(copy=True)
        if missing.any():
            observed = spec.observed_values if spec.observed_values is not None and len(spec.observed_values) else np.array([spec.center or 0.0])
            draw_idx = rng.integers(0, len(observed), size=int(missing.sum()))
            values[missing] = observed[draw_idx]
        values = np.clip(values, spec.winsor_low, spec.winsor_high)
        values = (values - float(spec.center or 0.0)) / float(spec.scale or 1.0)
        return np.column_stack([values, missing.astype(float)])

    def _transform_categorical(self, series: pd.Series, spec: ImputationSpec, rng: np.random.Generator) -> np.ndarray:
        assert spec.categories is not None and spec.category_probabilities is not None
        observed = series.map(is_observed_value).to_numpy(dtype=bool)
        tokens = series.map(clean_category).astype(str).to_numpy()
        missing = ~observed
        if missing.any():
            draws = rng.choice(spec.categories, size=int(missing.sum()), p=spec.category_probabilities)
            tokens[missing] = draws
        keep = set(spec.categories)
        tokens = np.array([token if token in keep else "__OTHER__" for token in tokens], dtype=object)
        index = {cat: i for i, cat in enumerate(spec.categories)}
        out = np.zeros((len(series), len(spec.categories) + 1), dtype=float)
        for row, token in enumerate(tokens):
            out[row, index.get(str(token), index.get("__OTHER__", 0))] = 1.0
        out[:, -1] = missing.astype(float)
        return out

    def _refresh_feature_index(self) -> None:
        self.feature_names_ = []
        self.groups_ = {}
        for spec in self.specs_:
            start = len(self.feature_names_)
            self.feature_names_.extend(spec.feature_names)
            self.groups_[spec.source_id] = list(range(start, len(self.feature_names_)))
