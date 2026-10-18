"""Array interfaces to the Q arithmetic in Supplement S5.

Fitted coefficients and states must be supplied privately. Target records must
be excluded from both model selection and fitting before these functions run.
"""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
PRIMARY_STATE_MAP = np.asarray((0.0, 0.5, 0.5, 1.0, 1.0, 1.0), dtype='<f8')
LOW_DEFICIT_STATE_MAP = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype='<f8')
HIGH_DEFICIT_STATE_MAP = np.asarray((0.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype='<f8')
OUTPUT_IDS = ('frontier_q_dimnorm_primary_v1', 'frontier_q_rawnorm_primarymap_v1', 'frontier_q_dimnorm_lowdeficit_v1', 'frontier_q_dimnorm_highdeficit_v1')

def _normalized_feature_groups(source_ids: tuple[str, ...], feature_names: tuple[str, ...], feature_groups: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    if not feature_names or len(set(feature_names)) != len(feature_names) or any((not isinstance(name, str) or not name for name in feature_names)) or (not isinstance(feature_groups, Mapping)) or (not feature_groups) or any((source not in set(source_ids) for source in feature_groups)):
        raise ValueError('feature/source partition differs')
    normalized: dict[str, list[int]] = {}
    occupied: list[int] = []
    for source in source_ids:
        if source not in feature_groups:
            continue
        raw = feature_groups[source]
        if not isinstance(raw, (tuple, list)) or not raw:
            raise ValueError('eligible feature group is empty')
        if any((type(index) is not int for index in raw)):
            raise ValueError('feature-group index differs')
        indices = list(raw)
        if len(set(indices)) != len(indices) or any((index < 0 or index >= len(feature_names) for index in indices)):
            raise ValueError('feature-group index differs')
        normalized[source] = indices
        occupied.extend(indices)
    if sorted(occupied) != list(range(len(feature_names))):
        raise ValueError('source groups are not an exact disjoint exhaustive non-intercept partition')
    return normalized
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class SourceWeights:
    source_ids: tuple[str, ...]
    dimension_normalized: np.ndarray
    raw_normalized: np.ndarray

def source_weights(coefficients, *, feature_names, feature_groups, source_ids):
    """Normalize coefficient-block norms; intercepts are not included."""
    coef = np.asarray(coefficients, dtype=float)
    sources, names = (tuple(source_ids), tuple(feature_names))
    if coef.ndim != 2 or coef.shape != (len(names), 2) or (not np.isfinite(coef).all()):
        raise ValueError('Expected finite non-reference-class coefficients with shape (features, 2).')
    normalized_groups = _normalized_feature_groups(sources, names, feature_groups)
    norm_by_source: dict[str, float] = {}
    dimension_by_source: dict[str, int] = {}
    for source in sources:
        if source not in normalized_groups:
            norm_by_source[source] = 0.0
            dimension_by_source[source] = 0
            continue
        indices = normalized_groups[source]
        dimension_by_source[source] = len(indices)
        norm_by_source[source] = float(np.linalg.norm(coef[indices, :], ord='fro'))
    raw_norm = np.asarray([norm_by_source[source] for source in sources], dtype='<f8')
    dimensions = np.asarray([dimension_by_source[source] for source in sources], dtype='<i8')
    dimension_norm = np.zeros_like(raw_norm)
    positive_dimension = dimensions > 0
    dimension_norm[positive_dimension] = raw_norm[positive_dimension] / np.sqrt(dimensions[positive_dimension])
    raw_denominator = float(np.sum(raw_norm, dtype=np.float64))
    dimension_denominator = float(np.sum(dimension_norm, dtype=np.float64))
    if not np.isfinite(raw_denominator) or raw_denominator <= 0.0 or (not np.isfinite(dimension_denominator)) or (dimension_denominator <= 0.0):
        raise ValueError('source-weight denominator is degenerate')
    raw_weights = np.ascontiguousarray(raw_norm / raw_denominator, dtype='<f8')
    dimension_weights = np.ascontiguousarray(dimension_norm / dimension_denominator, dtype='<f8')
    raw_weights.setflags(write=False)
    dimension_weights.setflags(write=False)
    return SourceWeights(sources, dimension_weights, raw_weights)

def q_values(state_codes, *, source_ids, weights):
    """Return primary, raw-norm, low-state and high-state Q in frozen order.

    State codes: observed=0, not_applicable=1, structural=2, unknown=3,
    not_administered=4, artifact=5. Source order must match the coefficients.
    """
    if tuple(source_ids) != weights.source_ids:
        raise ValueError('State and coefficient source orders differ.')
    states = np.asarray(state_codes)
    if states.ndim != 2 or states.shape[1] != len(source_ids) or (not np.issubdtype(states.dtype, np.integer)) or np.any((states < 0) | (states > 5)):
        raise ValueError('Expected a record-by-source matrix of integer state codes 0 through 5.')
    states = np.asarray(state_codes)
    primary = PRIMARY_STATE_MAP[states]
    low = LOW_DEFICIT_STATE_MAP[states]
    high = HIGH_DEFICIT_STATE_MAP[states]
    values = np.column_stack((primary @ weights.dimension_normalized, primary @ weights.raw_normalized, low @ weights.dimension_normalized, high @ weights.dimension_normalized)).astype('<f8', copy=False)
    values = np.ascontiguousarray(values, dtype='<f8')
    if not np.isfinite(values).all() or np.any(values < -1e-15) or np.any(values > 1.0 + 1e-15):
        raise ValueError('computed Q values differ')
    values.setflags(write=False)
    return values

def freeze_q_groups(proper_primary_q, apply_primary_q):
    """Freeze higher-method development quartiles and keep equal scores together."""
    proper = np.asarray(proper_primary_q, dtype=float)
    apply = np.asarray(apply_primary_q, dtype=float)
    if proper.ndim != 1 or apply.ndim != 1 or (not proper.size) or (not apply.size):
        raise ValueError('Supply nonempty one-dimensional development and target scores.')
    if not np.isfinite(proper).all() or not np.isfinite(apply).all() or np.any((proper < 0) | (proper > 1)) or np.any((apply < 0) | (apply > 1)):
        raise ValueError('Q scores must be finite and in [0,1].')
    cuts = np.ascontiguousarray(np.quantile(proper, [0.25, 0.5, 0.75], method='higher'), dtype='<f8')
    codes = np.searchsorted(cuts, apply, side='left').astype('uint8') + 1
    labels = np.asarray([f'Q{int(code)}' for code in codes], dtype='<U2')
    return (cuts, labels)
