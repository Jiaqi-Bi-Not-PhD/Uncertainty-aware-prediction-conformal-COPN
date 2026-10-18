"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np

import pandas as pd

from scipy import sparse

from .common import CLASS_NAMES, KeyedConfig, clean_category, is_observed_value, normalize_text, numeric_series

@dataclass
class EBMSourceSpec:
    source_id: str
    source_group: str
    implementation_level: int
    kind: str
    tokens: list[str]
    feature_names: list[str]
    exclude_reason: str = ""
    cutpoints: list[float] | None = None
    winsor_low: float | None = None
    winsor_high: float | None = None

@dataclass
class EBMPairSpec:
    left_source: str
    right_source: str
    tokens: list[str]
    feature_names: list[str]

    @property
    def pair_id(self) -> str:
        return f"{self.left_source}__X__{self.right_source}"

class EBMPreprocessor:
    """Fold-scoped finite-table feature map for keyed EBM / GA2M.

    The downstream estimator is a regularized multinomial logistic model over
    one-hot main-effect and selected pairwise interaction cells. This keeps the
    fitted object interpretable as a finite-table GA2M while avoiding any
    leakage from validation or outer-test folds.
    """

    def __init__(
        self,
        config: KeyedConfig,
        budget: int,
        max_bins: int = 8,
        interaction_pairs: list[tuple[str, str]] | None = None,
        min_pair_count: int = 10,
        max_pair_cells: int = 80,
    ) -> None:
        self.config = config
        self.budget = int(budget)
        self.max_bins = int(max_bins)
        self.interaction_pairs = interaction_pairs or []
        self.min_pair_count = int(min_pair_count)
        self.max_pair_cells = int(max_pair_cells)
        self.specs_: list[EBMSourceSpec] = []
        self.pair_specs_: list[EBMPairSpec] = []
        self.screening_: pd.DataFrame | None = None
        self.feature_names_: list[str] = []
        self.source_slices_: dict[str, list[int]] = {}
        self.term_rows_: list[dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> "EBMPreprocessor":
        rows: list[dict[str, Any]] = []
        specs: list[EBMSourceSpec] = []
        token_cache: dict[str, pd.Series] = {}
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
            feature_count = 0
            if reason is None:
                kind = self._infer_kind(X[source_id], str(meta.get("field_type", "")))
                if kind == "numeric":
                    spec = self._fit_numeric(source_id, X[source_id], meta)
                elif kind == "categorical":
                    spec = self._fit_categorical(source_id, X[source_id], meta)
                else:
                    spec = EBMSourceSpec(
                        source_id=source_id,
                        source_group=str(meta.get("source_group", "")),
                        implementation_level=int(meta.get("implementation_level", 4)),
                        kind=kind,
                        tokens=[],
                        feature_names=[],
                        exclude_reason="unusable_type",
                    )
                if spec.exclude_reason:
                    reason = spec.exclude_reason
                    kind = spec.kind
                else:
                    specs.append(spec)
                    token_cache[source_id] = self._tokens_for_source(X[source_id], spec)
                    feature_count = len(spec.feature_names)

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
                    "feature_count": feature_count,
                }
            )

        self.specs_ = specs
        self.pair_specs_ = self._fit_pair_specs(token_cache)
        self.screening_ = pd.DataFrame(rows)
        self._refresh_feature_index()
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        n = len(X)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        source_tokens: dict[str, pd.Series] = {}
        token_to_col: dict[str, dict[str, int]] = {}

        for spec in self.specs_:
            tokens = self._tokens_for_source(X[spec.source_id], spec)
            source_tokens[spec.source_id] = tokens
            token_map = {token: idx for idx, token in enumerate(spec.tokens)}
            token_to_col[spec.source_id] = token_map
            col0 = self.source_slices_[spec.source_id][0]
            other_idx = token_map.get("__OTHER__")
            for row_idx, token in enumerate(tokens):
                local = token_map.get(str(token), other_idx)
                if local is None:
                    continue
                rows.append(row_idx)
                cols.append(col0 + local)
                data.append(1.0)

        for pair in self.pair_specs_:
            left = source_tokens[pair.left_source]
            right = source_tokens[pair.right_source]
            pair_tokens = left.astype(str) + "||" + right.astype(str)
            token_map = {token: idx for idx, token in enumerate(pair.tokens)}
            col0 = self.source_slices_[pair.pair_id][0]
            other_idx = token_map.get("__OTHER__")
            for row_idx, token in enumerate(pair_tokens):
                local = token_map.get(str(token), other_idx)
                if local is None:
                    continue
                rows.append(row_idx)
                cols.append(col0 + local)
                data.append(1.0)

        if not self.feature_names_:
            return sparse.csr_matrix((n, 0), dtype=float)
        mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, len(self.feature_names_)), dtype=float)
        return mat

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame) -> sparse.csr_matrix:
        self.fit(X, y, metadata)
        return self.transform(X)

    def term_manifest(self) -> pd.DataFrame:
        return pd.DataFrame(self.term_rows_)

    def source_feature_groups(self) -> dict[str, list[int]]:
        return {name: cols for name, cols in self.source_slices_.items() if "__X__" not in name}

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

    def _fit_numeric(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> EBMSourceSpec:
        parsed = numeric_series(series)
        observed = parsed.dropna()
        if len(observed) < 2:
            return EBMSourceSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "numeric", [], [], "numeric_too_few_observed")
        low = float(observed.quantile(self.config.winsor_lower_q))
        high = float(observed.quantile(self.config.winsor_upper_q))
        clipped = observed.clip(lower=low, upper=high)
        if float(clipped.std(ddof=1)) <= 0:
            return EBMSourceSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "numeric", [], [], "numeric_quasi_constant")
        target_bins = max(2, self.max_bins)
        quantiles = np.linspace(0.0, 1.0, target_bins + 1)[1:-1]
        cutpoints = sorted(set(float(v) for v in clipped.quantile(quantiles).to_numpy() if np.isfinite(v)))
        tokens = [f"bin_{i:02d}" for i in range(len(cutpoints) + 1)] + ["__MISSING__"]
        return EBMSourceSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="numeric",
            tokens=tokens,
            feature_names=[f"{source_id}__{token}" for token in tokens],
            cutpoints=cutpoints,
            winsor_low=low,
            winsor_high=high,
        )

    def _fit_categorical(self, source_id: str, series: pd.Series, meta: dict[str, Any]) -> EBMSourceSpec:
        observed = series.loc[series.map(is_observed_value)].map(clean_category)
        if observed.empty:
            return EBMSourceSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "categorical", [], [], "categorical_too_few_observed")
        counts = observed.value_counts()
        kept = counts.loc[counts >= self.config.rare_category_min_count].index.tolist()
        kept = kept[: self.config.max_categories]
        tokens = sorted(set(kept + ["__OTHER__", "__MISSING__"]))
        if len(tokens) <= 1:
            return EBMSourceSpec(source_id, str(meta.get("source_group", "")), int(meta.get("implementation_level", 4)), "categorical", [], [], "categorical_quasi_constant")
        return EBMSourceSpec(
            source_id=source_id,
            source_group=str(meta.get("source_group", "")),
            implementation_level=int(meta.get("implementation_level", 4)),
            kind="categorical",
            tokens=tokens,
            feature_names=[f"{source_id}__cat__{normalize_text(token)}" for token in tokens],
        )

    def _tokens_for_source(self, series: pd.Series, spec: EBMSourceSpec) -> pd.Series:
        if spec.kind == "numeric":
            parsed = numeric_series(series)
            observed = parsed.notna()
            clipped = parsed.clip(lower=spec.winsor_low, upper=spec.winsor_high)
            cutpoints = np.asarray(spec.cutpoints or [], dtype=float)
            bins = np.searchsorted(cutpoints, clipped.to_numpy(dtype=float), side="right")
            tokens = pd.Series([f"bin_{int(idx):02d}" for idx in bins], index=series.index, dtype=object)
            tokens.loc[~observed] = "__MISSING__"
            return tokens
        keep = set(spec.tokens)
        tokens = series.map(lambda value: clean_category(value))
        tokens = tokens.map(lambda token: token if token in keep else "__OTHER__")
        return tokens.astype(str)

    def _fit_pair_specs(self, token_cache: dict[str, pd.Series]) -> list[EBMPairSpec]:
        specs: list[EBMPairSpec] = []
        seen: set[tuple[str, str]] = set()
        eligible = {spec.source_id for spec in self.specs_}
        for left, right in self.interaction_pairs:
            if left == right:
                continue
            pair = tuple(sorted((left, right)))
            if pair in seen or pair[0] not in eligible or pair[1] not in eligible:
                continue
            seen.add(pair)
            tokens = token_cache[pair[0]].astype(str) + "||" + token_cache[pair[1]].astype(str)
            counts = tokens.value_counts()
            kept = counts.loc[counts >= self.min_pair_count].index.tolist()[: self.max_pair_cells]
            if not kept:
                continue
            pair_tokens = sorted(set(kept + ["__OTHER__"]))
            feature_names = [
                f"{pair[0]}__X__{pair[1]}__cell__{normalize_text(token)}" for token in pair_tokens
            ]
            specs.append(EBMPairSpec(pair[0], pair[1], pair_tokens, feature_names))
        return specs

    def _refresh_feature_index(self) -> None:
        self.feature_names_ = []
        self.source_slices_ = {}
        self.term_rows_ = []
        for spec in self.specs_:
            start = len(self.feature_names_)
            self.feature_names_.extend(spec.feature_names)
            cols = list(range(start, len(self.feature_names_)))
            self.source_slices_[spec.source_id] = cols
            for token, feature_name, col in zip(spec.tokens, spec.feature_names, cols):
                self.term_rows_.append(
                    {
                        "term_type": "main",
                        "source_id": spec.source_id,
                        "left_source": spec.source_id,
                        "right_source": "",
                        "kind": spec.kind,
                        "token": token,
                        "feature_index": int(col),
                        "feature_name": feature_name,
                    }
                )
        for pair in self.pair_specs_:
            start = len(self.feature_names_)
            self.feature_names_.extend(pair.feature_names)
            cols = list(range(start, len(self.feature_names_)))
            self.source_slices_[pair.pair_id] = cols
            for token, feature_name, col in zip(pair.tokens, pair.feature_names, cols):
                self.term_rows_.append(
                    {
                        "term_type": "interaction",
                        "source_id": pair.pair_id,
                        "left_source": pair.left_source,
                        "right_source": pair.right_source,
                        "kind": "pair",
                        "token": token,
                        "feature_index": int(col),
                        "feature_name": feature_name,
                    }
                )
