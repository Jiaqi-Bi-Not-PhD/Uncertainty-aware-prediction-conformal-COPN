"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import math

from typing import Mapping, Sequence

import numpy as np

from numpy.typing import ArrayLike, NDArray

SIMPLE_SCORE_ORDER = (
    "one_minus_max_primary_probability",
    "one_minus_primary_top_two_margin",
    "primary_normalized_entropy",
)

@dataclass(frozen=True)
class ScoreSelection:
    """Proper-fit-only selection result for the simple uncertainty family."""

    score_id: str
    aurc: float
    brier: float
    candidate_metrics: Mapping[str, Mapping[str, float]]

@dataclass(frozen=True)
class ProjectClusterBootstrap:
    project_keys: tuple[str, ...]
    multiplicities: NDArray[np.int16]
    seed: int

@dataclass(frozen=True)
class BootstrapEstimates:
    observed: float | NDArray[np.float64]
    replicates: NDArray[np.float64]

@dataclass(frozen=True)
class PercentileInterval:
    lower: float
    upper: float
    valid_replicates: int

def validate_probabilities(
    probabilities: ArrayLike,
    *,
    n_classes: int = 3,
    simplex_tolerance: float = 1e-8,
    allow_empty: bool = False,
    name: str = "probabilities",
) -> NDArray[np.float64]:
    """Validate a probability matrix without clipping or renormalizing it."""

    raw = np.asarray(probabilities)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a real numeric matrix")
    values = raw.astype(float, copy=False)
    if (
        values.ndim != 2
        or values.shape[1] != n_classes
        or (not allow_empty and values.shape[0] == 0)
    ):
        raise ValueError(f"{name} must have shape (rows, {n_classes})")
    if (
        isinstance(simplex_tolerance, bool)
        or not np.isfinite(simplex_tolerance)
        or simplex_tolerance < 0
    ):
        raise ValueError("simplex tolerance must be finite and nonnegative")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    if not np.all(np.abs(values.sum(axis=1) - 1.0) <= simplex_tolerance):
        raise ValueError(f"{name} rows must sum to one within tolerance")
    return values

def forced_argmax(probabilities: ArrayLike) -> NDArray[np.int64]:
    """Return calibrated-probability argmax, breaking ties by lowest code."""

    values = validate_probabilities(probabilities)
    return np.argmax(values, axis=1).astype(np.int64)

def classification_errors(
    probabilities: ArrayLike, true_codes: ArrayLike
) -> NDArray[np.int8]:
    values = validate_probabilities(probabilities)
    truth = _true_codes(true_codes, len(values))
    return (forced_argmax(values) != truth).astype(np.int8)

def normalized_entropy(probabilities: ArrayLike) -> NDArray[np.float64]:
    values = validate_probabilities(probabilities)
    log_values = np.zeros_like(values)
    np.log(values, out=log_values, where=values > 0.0)
    entropy = -np.sum(values * log_values, axis=1) / np.log(values.shape[1])
    return np.clip(entropy, 0.0, 1.0)

def one_minus_max_probability(probabilities: ArrayLike) -> NDArray[np.float64]:
    values = validate_probabilities(probabilities)
    return 1.0 - values.max(axis=1)

def one_minus_top_two_margin(probabilities: ArrayLike) -> NDArray[np.float64]:
    values = validate_probabilities(probabilities)
    two_largest = np.partition(values, values.shape[1] - 2, axis=1)[:, -2:]
    margin = two_largest.max(axis=1) - two_largest.min(axis=1)
    return 1.0 - margin

def uncertainty_scores(
    probabilities: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Return the three frozen Stage 7 simple uncertainty scores."""

    values = validate_probabilities(probabilities)
    return {
        "one_minus_max_primary_probability": one_minus_max_probability(values),
        "one_minus_primary_top_two_margin": one_minus_top_two_margin(values),
        "primary_normalized_entropy": normalized_entropy(values),
    }

def deterministic_project_hashes(
    project_keys: Sequence[object],
    *,
    seed: int = 20260802,
    namespace: str = "stage7_aurc",
) -> NDArray[np.uint64]:
    """Create fixed Project-key hash tie breakers independent of row order."""

    keys = _project_keys(project_keys, require_unique=True)
    seed_value = _integer(seed, "seed", minimum=0)
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("hash namespace must be a nonempty string")
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.sha256(
                    f"{seed_value}\x1f{namespace}\x1f{key}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            for key in keys
        ),
        dtype=np.uint64,
        count=len(keys),
    )

def deterministic_aurc(
    errors: ArrayLike,
    uncertainty: ArrayLike,
    *,
    project_keys: Sequence[object],
    seed: int = 20260802,
    outer_split_id: str | None = None,
) -> float:
    """Area under risk--coverage, retaining low-uncertainty rows first.

    Arrays must already be restricted to the intended role and one outer split.
    Equal uncertainty values are ordered by a fixed Project-key hash, never by
    incoming row position.
    """

    error = _binary_vector(errors, "errors")
    risk = _finite_vector(uncertainty, "uncertainty", length=len(error))
    if len(error) == 0:
        raise ValueError("AURC needs at least one row")
    keys = _project_keys(project_keys, length=len(error), require_unique=True)
    namespace = "stage7_aurc" if outer_split_id is None else f"stage7_aurc:{outer_split_id}"
    hashes = deterministic_project_hashes(keys, seed=seed, namespace=namespace)
    # String keys are a deterministic collision guard for the 64-bit hashes.
    order = np.lexsort((np.asarray(keys), hashes, risk))
    cumulative_risk = np.cumsum(error[order], dtype=float) / np.arange(
        1, len(error) + 1, dtype=float
    )
    return float(cumulative_risk.mean())

def fold_aurcs(
    errors: ArrayLike,
    uncertainty: ArrayLike,
    *,
    project_keys: Sequence[object],
    outer_split_ids: Sequence[object],
    seed: int = 20260802,
    expected_outer_splits: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """Calculate separate AURCs for already-scoped outer-test fold rows."""

    error = _binary_vector(errors, "errors")
    risk = _finite_vector(uncertainty, "uncertainty", length=len(error))
    keys = _project_keys(project_keys, length=len(error))
    splits = _fold_ids(outer_split_ids, len(error))
    unique_splits = tuple(sorted(set(splits)))
    if expected_outer_splits is not None and len(unique_splits) != _integer(
        expected_outer_splits, "expected outer splits", minimum=1
    ):
        raise ValueError("outer-split inventory is incomplete or unexpected")
    result: dict[str, dict[str, float | int]] = {}
    for split in unique_splits:
        selected = splits == split
        result[split] = {
            "rows": int(np.count_nonzero(selected)),
            "aurc": deterministic_aurc(
                error[selected],
                risk[selected],
                project_keys=np.asarray(keys)[selected],
                seed=seed,
                outer_split_id=split,
            ),
        }
    return result

def eligible_row_weighted_fold_aurc(
    errors: ArrayLike,
    uncertainty: ArrayLike,
    *,
    project_keys: Sequence[object],
    outer_split_ids: Sequence[object],
    seed: int = 20260802,
    expected_outer_splits: int | None = None,
) -> float:
    """Frozen primary aggregation: eligible-row-weighted mean fold AURC."""

    metrics = fold_aurcs(
        errors,
        uncertainty,
        project_keys=project_keys,
        outer_split_ids=outer_split_ids,
        seed=seed,
        expected_outer_splits=expected_outer_splits,
    )
    rows = np.asarray([item["rows"] for item in metrics.values()], dtype=float)
    values = np.asarray([item["aurc"] for item in metrics.values()], dtype=float)
    return float(np.average(values, weights=rows))

def cluster_bootstrap_fold_aurc(
    errors: ArrayLike,
    uncertainty: ArrayLike,
    *,
    project_keys: Sequence[object],
    outer_split_ids: Sequence[object],
    bootstrap: ProjectClusterBootstrap,
    seed: int = 20260802,
    expected_outer_splits: int = 15,
) -> BootstrapEstimates:
    """Bootstrap the row-weighted mean of fold-specific discrete AURCs.

    The shared Project-key multiplicity matrix carries every repeat/fold row
    belonging to a sampled participant.  Selection must already be frozen.
    The implementation is vectorized over replicates and never materializes an
    expanded participant table.
    """

    error = _binary_vector(errors, "errors")
    risk = _finite_vector(uncertainty, "uncertainty", length=len(error))
    keys = _project_keys(project_keys, length=len(error))
    splits = _fold_ids(outer_split_ids, len(error))
    split_values = tuple(sorted(set(splits)))
    if len(split_values) != _integer(
        expected_outer_splits, "expected outer splits", minimum=1
    ):
        raise ValueError("outer-split inventory is incomplete or unexpected")
    if not isinstance(bootstrap, ProjectClusterBootstrap):
        raise ValueError("bootstrap must be a ProjectClusterBootstrap")
    multiplicities = _multiplicity_matrix(bootstrap.multiplicities)
    bootstrap_keys = tuple(bootstrap.project_keys)
    if int(bootstrap.seed) != _integer(seed, "seed", minimum=0):
        raise ValueError("bootstrap and AURC seeds differ")
    if (
        len(bootstrap_keys) != multiplicities.shape[1]
        or np.any(multiplicities.sum(axis=1) != len(bootstrap_keys))
    ):
        raise ValueError("bootstrap key inventory is misaligned")
    if set(keys) != set(bootstrap_keys):
        raise ValueError("outer-test and bootstrap Project-key inventories differ")
    key_position = {key: index for index, key in enumerate(bootstrap_keys)}
    row_key_positions = np.asarray([key_position[key] for key in keys], dtype=np.int64)
    replicate_count = multiplicities.shape[0]
    maximum_weight = int(multiplicities.sum(axis=1).max())
    harmonic = _harmonic_numbers(maximum_weight)
    total_risk_sum = np.zeros(replicate_count, dtype=float)
    total_rows = np.zeros(replicate_count, dtype=np.int64)

    for split_id in split_values:
        selected = np.flatnonzero(splits == split_id)
        split_keys = tuple(keys[index] for index in selected)
        if len(set(split_keys)) != len(split_keys):
            raise ValueError("Project keys must be unique within each outer split")
        hashes = deterministic_project_hashes(
            split_keys, seed=seed, namespace=f"stage7_aurc:{split_id}"
        )
        local_order = np.lexsort(
            (np.asarray(split_keys), hashes, risk[selected])
        )
        ordered = selected[local_order]
        cumulative_weight = np.zeros(replicate_count, dtype=np.int64)
        cumulative_errors = np.zeros(replicate_count, dtype=np.int64)
        split_risk_sum = np.zeros(replicate_count, dtype=float)
        for row_index in ordered:
            row_weight = multiplicities[:, row_key_positions[row_index]].astype(
                np.int64, copy=False
            )
            next_weight = cumulative_weight + row_weight
            harmonic_delta = harmonic[next_weight] - harmonic[cumulative_weight]
            if error[row_index] == 0:
                split_risk_sum += cumulative_errors * harmonic_delta
            else:
                split_risk_sum += row_weight + (
                    cumulative_errors - cumulative_weight
                ) * harmonic_delta
                cumulative_errors += row_weight
            cumulative_weight = next_weight
        total_risk_sum += split_risk_sum
        total_rows += cumulative_weight

    replicates = np.divide(
        total_risk_sum,
        total_rows,
        out=np.full(replicate_count, np.nan, dtype=float),
        where=total_rows > 0,
    )
    observed = eligible_row_weighted_fold_aurc(
        error,
        risk,
        project_keys=keys,
        outer_split_ids=splits,
        seed=seed,
        expected_outer_splits=expected_outer_splits,
    )
    return BootstrapEstimates(observed=observed, replicates=replicates)

def select_simple_score(
    errors: ArrayLike,
    score_candidates: Mapping[str, ArrayLike],
    *,
    project_keys: Sequence[object],
    fixed_order: Sequence[str] = SIMPLE_SCORE_ORDER,
    tie_tolerance: float = 1e-12,
    seed: int = 20260802,
    outer_split_id: str | None = None,
) -> ScoreSelection:
    """Select on proper-fit rows by AURC, then Brier, then frozen order."""

    error = _binary_vector(errors, "errors")
    if len(error) == 0:
        raise ValueError("simple-score selection needs at least one row")
    order = tuple(fixed_order)
    if not order or len(set(order)) != len(order) or set(score_candidates) != set(order):
        raise ValueError("score candidates must exactly match the fixed score order")
    tolerance = _finite_scalar(tie_tolerance, "tie tolerance")
    if tolerance < 0:
        raise ValueError("tie tolerance must be nonnegative")
    keys = _project_keys(project_keys, length=len(error), require_unique=True)
    metrics: dict[str, dict[str, float]] = {}
    for score_id in order:
        score = _finite_vector(
            score_candidates[score_id], score_id, length=len(error)
        )
        if np.any((score < 0.0) | (score > 1.0)):
            raise ValueError(f"{score_id} must lie in [0, 1]")
        metrics[score_id] = {
            "aurc": deterministic_aurc(
                error,
                score,
                project_keys=keys,
                seed=seed,
                outer_split_id=outer_split_id,
            ),
            "brier": float(np.mean((score - error) ** 2)),
        }

    selected = order[0]
    for score_id in order[1:]:
        candidate = metrics[score_id]
        incumbent = metrics[selected]
        if candidate["aurc"] < incumbent["aurc"] - tolerance:
            selected = score_id
        elif (
            abs(candidate["aurc"] - incumbent["aurc"]) <= tolerance
            and candidate["brier"] < incumbent["brier"] - tolerance
        ):
            selected = score_id
    return ScoreSelection(
        score_id=selected,
        aurc=metrics[selected]["aurc"],
        brier=metrics[selected]["brier"],
        candidate_metrics=metrics,
    )

def select_simple_score_from_probabilities(
    probabilities: ArrayLike,
    true_codes: ArrayLike,
    *,
    project_keys: Sequence[object],
    tie_tolerance: float = 1e-12,
    seed: int = 20260802,
    outer_split_id: str | None = None,
) -> ScoreSelection:
    values = validate_probabilities(probabilities)
    truth = _true_codes(true_codes, len(values))
    return select_simple_score(
        (forced_argmax(values) != truth).astype(np.int8),
        uncertainty_scores(values),
        project_keys=project_keys,
        tie_tolerance=tie_tolerance,
        seed=seed,
        outer_split_id=outer_split_id,
    )

def percentile_interval(
    bootstrap_values: ArrayLike,
    *,
    lower_probability: float = 0.025,
    upper_probability: float = 0.975,
    minimum_valid_replicates: int = 1900,
) -> PercentileInterval:
    """Linear percentile interval with an explicit valid-replicate gate."""

    values = np.asarray(bootstrap_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("bootstrap values must be one-dimensional")
    lower = _finite_scalar(lower_probability, "lower probability")
    upper = _finite_scalar(upper_probability, "upper probability")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("percentile probabilities are invalid")
    minimum = _integer(
        minimum_valid_replicates, "minimum valid replicates", minimum=1
    )
    valid = values[np.isfinite(values)]
    if len(valid) < minimum:
        raise ValueError("too few valid bootstrap replicates")
    quantiles = np.quantile(valid, [lower, upper], method="linear")
    return PercentileInterval(float(quantiles[0]), float(quantiles[1]), len(valid))

def one_sided_exact_binomial_upper(
    events: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Clopper--Pearson one-sided upper bound, including zero events.

    The result for zero events is positive whenever ``total`` is finite; it
    must therefore never be reported as proof of zero risk.
    """

    event_count = _integer(events, "events", minimum=0)
    total_count = _integer(total, "total", minimum=1)
    if event_count > total_count:
        raise ValueError("events cannot exceed total")
    level = _finite_scalar(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if event_count == total_count:
        return 1.0
    # Imported lazily so the pure ranking helpers do not pay SciPy import cost.
    from scipy.stats import beta

    return float(beta.ppf(level, event_count + 1, total_count - event_count))

def _true_codes(values: ArrayLike, length: int) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != length or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("true codes must be an aligned integer vector")
    truth = raw.astype(np.int64, copy=False)
    if np.any((truth < 0) | (truth > 2)):
        raise ValueError("true codes must use PD=0, AP=1, HC=2")
    return truth

def _project_keys(
    values: Sequence[object],
    length: int | None = None,
    *,
    require_unique: bool = False,
) -> tuple[str, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError("Project keys must be a sequence") from exc
    keys: list[str] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (str, int, np.integer)
        ):
            raise ValueError("Project keys must be nonempty strings or integers")
        key = str(value)
        if not key:
            raise ValueError("Project keys must be nonempty strings or integers")
        keys.append(key)
    if length is not None and len(keys) != length:
        raise ValueError("Project keys are misaligned")
    if require_unique and len(set(keys)) != len(keys):
        raise ValueError("Project keys must be unique within the scoped rows")
    return tuple(keys)

def _fold_ids(values: Sequence[object], length: int) -> NDArray[np.str_]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError("inner fold IDs must be a sequence") from exc
    if len(raw_values) != length:
        raise ValueError("inner fold IDs are misaligned")
    folds = np.asarray([str(value) for value in raw_values])
    if np.any(folds == ""):
        raise ValueError("inner fold IDs must be nonempty")
    return folds

def _binary_vector(values: ArrayLike, name: str) -> NDArray[np.int8]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a binary numeric vector")
    numeric = raw.astype(float, copy=False)
    if not np.isfinite(numeric).all() or np.any((numeric != 0.0) & (numeric != 1.0)):
        raise ValueError(f"{name} must contain only zero and one")
    return numeric.astype(np.int8)

def _harmonic_numbers(maximum: int) -> NDArray[np.float64]:
    upper = _integer(maximum, "maximum harmonic index", minimum=1)
    result = np.zeros(upper + 1, dtype=float)
    result[1:] = np.cumsum(1.0 / np.arange(1, upper + 1, dtype=float))
    return result

def _finite_vector(
    values: ArrayLike,
    name: str,
    *,
    length: int | None = None,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a real numeric vector")
    numeric = raw.astype(float, copy=False)
    if length is not None and len(numeric) != length:
        raise ValueError(f"{name} is misaligned")
    if not allow_empty and len(numeric) == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must be finite")
    return numeric

def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite numeric")
    return numeric

def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be integer")
    numeric = int(value)
    if numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return numeric

def _multiplicity_matrix(values: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("bootstrap multiplicities must be an integer matrix")
    counts = raw.astype(np.int64, copy=False)
    if len(counts) == 0 or counts.shape[1] == 0 or np.any(counts < 0):
        raise ValueError("bootstrap multiplicities must be nonnegative and nonempty")
    totals = counts.sum(axis=1)
    if np.any(totals <= 0) or np.any(totals != totals[0]):
        raise ValueError("bootstrap multiplicity rows must have one fixed positive total")
    return counts
