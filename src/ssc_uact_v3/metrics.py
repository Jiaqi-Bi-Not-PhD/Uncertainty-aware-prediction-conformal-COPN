"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import pandas as pd

from numpy.typing import ArrayLike, NDArray

CLASS_NAMES = ("PD", "AP", "HC")

SET_CODE_LABELS = {
    0: "",
    1: "PD",
    2: "AP",
    3: "PD|AP",
    4: "HC",
    5: "PD|HC",
    6: "AP|HC",
    7: "PD|AP|HC",
}

BIT_COUNT = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int8)

def bits_to_masks(bits: ArrayLike, *, allow_empty: bool = False) -> NDArray[np.bool_]:
    values = np.asarray(bits)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("set bits must be a one-dimensional integer array")
    numeric = values.astype(np.int64)
    lower = 0 if allow_empty else 1
    if np.any((numeric < lower) | (numeric > 7)):
        raise ValueError("set bits fall outside the three-class registry")
    return (numeric[:, None] & (1 << np.arange(3))[None, :]) != 0

def set_contains_true(bits: ArrayLike, true_code: ArrayLike) -> NDArray[np.bool_]:
    sets = np.asarray(bits, dtype=np.uint8)
    truth = np.asarray(true_code)
    if sets.shape != truth.shape or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("set bits and true codes are misaligned")
    if np.any((truth < 0) | (truth > 2)):
        raise ValueError("true codes violate PD=0, AP=1, HC=2")
    return (sets & (1 << truth.astype(np.uint8))) != 0

def scenario_sensitivity_index(scenario_bits: ArrayLike) -> NDArray[np.float64]:
    bits = np.asarray(scenario_bits)
    if bits.ndim != 2 or bits.shape[1] < 2 or not np.issubdtype(bits.dtype, np.integer):
        raise ValueError("scenario bits must have shape (rows, at least two scenarios)")
    if np.any((bits < 1) | (bits > 7)):
        raise ValueError("all scenario sets must be nonempty three-class masks")
    distance = np.zeros((8, 8), dtype=float)
    for left in range(1, 8):
        for right in range(1, 8):
            intersection = int(left & right).bit_count()
            union = int(left | right).bit_count()
            distance[left, right] = 1.0 - intersection / union
    counts = np.stack([(bits == mask).sum(axis=1) for mask in range(8)], axis=1)
    ordered_pair_distance = np.einsum(
        "ni,ij,nj->n", counts, distance, counts, optimize=True
    )
    scenarios = bits.shape[1]
    return 100.0 * ordered_pair_distance / (scenarios * (scenarios - 1))

def set_cost_matrix(
    harm_matrix: ArrayLike,
    *,
    pair_cost: float = 0.5,
    full_set_cost: float = 1.0,
) -> NDArray[np.float64]:
    harm = np.asarray(harm_matrix, dtype=float)
    if harm.shape != (3, 3) or not np.isfinite(harm).all() or np.any(harm < 0):
        raise ValueError("harm matrix must be finite, nonnegative, and 3 by 3")
    ambiguity = {1: 0.0, 2: float(pair_cost), 3: float(full_set_cost)}
    cost = np.full((8, 3), np.inf, dtype=float)
    for mask in range(1, 8):
        labels = np.flatnonzero(bits_to_masks(np.array([mask]))[0])
        size = len(labels)
        cost[mask] = ambiguity[size] + harm[:, labels].min(axis=1)
    return cost

def robust_expected_set_cost(
    selected_bits: ArrayLike,
    scenario_probabilities: ArrayLike,
    cost_by_mask_and_truth: ArrayLike,
) -> NDArray[np.float64]:
    bits = np.asarray(selected_bits)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    cost = np.asarray(cost_by_mask_and_truth, dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[0] != len(bits) or probabilities.shape[2] != 3:
        raise ValueError("selected sets and scenario probabilities are misaligned")
    if cost.shape != (8, 3) or np.any((bits < 1) | (bits > 7)):
        raise ValueError("set-cost lookup or selected set is invalid")
    selected_cost = cost[bits.astype(int)]
    expected = np.einsum(
        "nsk,nk->ns", probabilities, selected_cost, optimize=True
    )
    return expected.max(axis=1)

def summarize_set_rows(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    required = {
        *group_columns,
        "true_code",
        "core_mask",
        "mention_mask",
        "core_sig",
        "robust_expected_cost",
        "realized_set_harm",
        "ssi_raw",
        "fallback_flag",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"participant set rows lack {sorted(missing)}")
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for group_key, group in frame.groupby(grouper, sort=True, observed=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        truth = group["true_code"].to_numpy(np.int8)
        core = group["core_mask"].to_numpy(np.uint8)
        mention = group["mention_mask"].to_numpy(np.uint8)
        core_contains = set_contains_true(core, truth)
        mention_contains = set_contains_true(mention, truth)
        record: dict[str, object] = dict(zip(group_columns, keys, strict=True))
        record.update(
            {
                "n": len(group),
                "core_covered": int(core_contains.sum()),
                "core_coverage": float(core_contains.mean()),
                "mention_covered": int(mention_contains.sum()),
                "mention_coverage": float(mention_contains.mean()),
                "set_size_sum": int(BIT_COUNT[core].sum()),
                "mean_set_size": float(BIT_COUNT[core].mean()),
                "sig_sum": float(group["core_sig"].sum()),
                "mean_sig": float(group["core_sig"].mean()),
                "singleton_count": int((BIT_COUNT[core] == 1).sum()),
                "singleton_rate": float((BIT_COUNT[core] == 1).mean()),
                "pair_count": int((BIT_COUNT[core] == 2).sum()),
                "pair_rate": float((BIT_COUNT[core] == 2).mean()),
                "full_set_count": int((core == 7).sum()),
                "full_set_rate": float((core == 7).mean()),
                "dangerous_hc_singleton_count": int(
                    (((truth == 0) | (truth == 1)) & (core == 4)).sum()
                ),
                "dangerous_hc_singleton_denominator": int(
                    ((truth == 0) | (truth == 1)).sum()
                ),
                "pd_to_hc_singleton_count": int(((truth == 0) & (core == 4)).sum()),
                "ap_to_hc_singleton_count": int(((truth == 1) & (core == 4)).sum()),
                "mean_robust_expected_cost": float(
                    group["robust_expected_cost"].mean()
                ),
                "robust_expected_cost_sum": float(
                    group["robust_expected_cost"].sum()
                ),
                "mean_realized_set_harm": float(group["realized_set_harm"].mean()),
                "realized_set_harm_sum": float(group["realized_set_harm"].sum()),
                "mean_ssi": float(group["ssi_raw"].mean()),
                "ssi_sum": float(group["ssi_raw"].sum()),
                "median_ssi": float(group["ssi_raw"].median()),
            }
        )
        dangerous_denominator = record["dangerous_hc_singleton_denominator"]
        record["dangerous_hc_singleton_rate"] = (
            record["dangerous_hc_singleton_count"] / dangerous_denominator
            if dangerous_denominator
            else np.nan
        )
        fallback = group["fallback_flag"]
        if fallback.notna().any():
            valid = fallback.dropna().astype(bool)
            record["fallback_count"] = int(valid.sum())
            record["fallback_rate"] = float(valid.mean())
        else:
            record["fallback_count"] = np.nan
            record["fallback_rate"] = np.nan
        for class_code, class_name in enumerate(CLASS_NAMES):
            selected = truth == class_code
            denominator = int(selected.sum())
            covered = int(core_contains[selected].sum())
            mention_covered = int(mention_contains[selected].sum())
            record[f"n_{class_name}"] = denominator
            record[f"core_covered_{class_name}"] = covered
            record[f"core_coverage_{class_name}"] = (
                covered / denominator if denominator else np.nan
            )
            record[f"mention_covered_{class_name}"] = mention_covered
            record[f"mention_coverage_{class_name}"] = (
                mention_covered / denominator if denominator else np.nan
            )
        record["ap_exclusion_count"] = record["n_AP"] - record["core_covered_AP"]
        record["ap_exclusion_rate"] = (
            record["ap_exclusion_count"] / record["n_AP"]
            if record["n_AP"]
            else np.nan
        )
        for mask in range(1, 8):
            record[f"core_mask_{mask}_count"] = int((core == mask).sum())
        sensitivity = group.get("sensitivity_only_mask")
        if sensitivity is not None:
            values = sensitivity.to_numpy(np.uint8)
            for mask in range(8):
                record[f"sensitivity_mask_{mask}_count"] = int((values == mask).sum())
        rows.append(record)
    return pd.DataFrame(rows)

def set_composition_table(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    rows = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for group_key, group in frame.groupby(grouper, sort=True, observed=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(group_columns, keys, strict=True))
        for tier, column, masks in (
            ("core", "core_mask", range(1, 8)),
            ("sensitivity_only", "sensitivity_only_mask", range(8)),
        ):
            counts = group[column].value_counts().to_dict()
            for mask in masks:
                rows.append(
                    {
                        **identity,
                        "tier": tier,
                        "set_mask": mask,
                        "set_codes": SET_CODE_LABELS[mask],
                        "count": int(counts.get(mask, 0)),
                        "rate": float(counts.get(mask, 0) / len(group)),
                    }
                )
    return pd.DataFrame(rows)
