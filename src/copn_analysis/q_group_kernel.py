"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class ExactBatchedDotGeometryV2:
    group_count: int
    distinct_group_lengths: tuple[int, ...]
    feature_reference_count: int
    group_indices_disjoint: bool
    batched_dot_enabled: bool

class ExactBatchedDotGroupKernelV2:
    """Cache ordered group geometry and batch exact flattened self-dots."""

    def __init__(self, groups: Mapping[str, Sequence[int]], adaptive_weights: Mapping[str, float]) -> None:
        self.bound_groups = groups
        self.bound_adaptive_weights = adaptive_weights
        self.names = tuple((str(name) for name in groups))
        if not self.names:
            raise ValueError('At least one source group is required.')
        self.columns = tuple((np.asarray(groups[name], dtype=int) for name in self.names))
        if any((cols.ndim != 1 or len(cols) == 0 or np.any(cols < 0) or (len(np.unique(cols)) != len(cols)) for cols in self.columns)):
            raise ValueError('Group column geometry is malformed.')
        if set(adaptive_weights) != set(self.names):
            raise ValueError('Adaptive-weight group keys differ.')
        self.adaptive_weights = np.asarray([float(adaptive_weights[name]) for name in self.names], dtype=np.float64)
        if np.any(~np.isfinite(self.adaptive_weights)) or np.any(self.adaptive_weights <= 0):
            raise ValueError('Adaptive group weights must be finite and positive.')
        self.lengths = np.asarray([len(cols) for cols in self.columns], dtype=np.int64)
        self.sqrt_lengths = np.asarray([np.sqrt(len(cols)) for cols in self.columns], dtype=np.float64)
        self.weighted_sqrt_lengths = self.adaptive_weights * self.sqrt_lengths
        concatenated = np.concatenate(self.columns)
        disjoint = len(np.unique(concatenated)) == len(concatenated)
        self._batches: tuple[tuple[np.ndarray, np.ndarray], ...] = tuple(((positions, np.stack([self.columns[int(index)] for index in positions])) for length in np.unique(self.lengths) for positions in (np.flatnonzero(self.lengths == length),))) if disjoint else ()
        self.geometry = ExactBatchedDotGeometryV2(group_count=len(self.names), distinct_group_lengths=tuple((int(value) for value in np.unique(self.lengths))), feature_reference_count=int(len(concatenated)), group_indices_disjoint=bool(disjoint), batched_dot_enabled=bool(disjoint))
        self.batched_group_norm_calls = 0
        self.selected_block_count_calls = 0

    def _ordered_exact_norms(self, coefficient: np.ndarray) -> np.ndarray:
        self.batched_group_norm_calls += 1
        if not self.geometry.batched_dot_enabled:
            return np.fromiter((float(np.linalg.norm(coefficient[cols, :], ord='fro')) for cols in self.columns), dtype=np.float64, count=len(self.columns))
        norms = np.empty(len(self.columns), dtype=np.float64)
        for positions, index_matrix in self._batches:
            flat = coefficient[index_matrix, :].reshape(len(positions), -1)
            squared = np.vecdot(flat, flat, axis=1)
            np.sqrt(squared, out=squared)
            norms[positions] = squared
        return norms

    def group_penalty(self, coefficient: np.ndarray) -> float:
        norms = self._ordered_exact_norms(coefficient)
        products = self.weighted_sqrt_lengths * norms
        return float(np.add.accumulate(products, dtype=np.float64)[-1])

    def prox(self, model: Any, coefficient: np.ndarray, step: float) -> np.ndarray:
        if model.alpha <= 0:
            return coefficient
        if not self.geometry.batched_dot_enabled:
            return model._prox(coefficient, step)
        out = coefficient.copy()
        norms = self._ordered_exact_norms(out)
        common = step * model.lambda_ * model.alpha
        thresholds = common * self.adaptive_weights * self.sqrt_lengths
        retained = norms > thresholds
        scales = np.empty_like(norms)
        scales[retained] = 1.0 - thresholds[retained] / norms[retained]
        for positions, index_matrix in self._batches:
            blocks = out[index_matrix, :]
            keep = retained[positions]
            blocks[~keep, :, :] = 0.0
            blocks[keep, :, :] = scales[positions][keep, None, None] * blocks[keep, :, :]
            out[index_matrix, :] = blocks
        return out

    def selected_block_count(self, coefficient: np.ndarray) -> int:
        """Return the frozen source-norm threshold count from exact norms."""
        self.selected_block_count_calls += 1
        norms = self._ordered_exact_norms(coefficient)
        return int(np.count_nonzero(norms > 1e-08))

    def validate_bound_model(self, model: Any) -> None:
        """O(1) gate for shallow candidate copies of the validated template."""
        if model.groups is not self.bound_groups or model.adaptive_weights is not self.bound_adaptive_weights:
            raise ValueError('Model does not share the validated group-kernel mappings.')
