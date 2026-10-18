"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Mapping

import numpy as np

from numpy.typing import ArrayLike, NDArray

from .probability import normalize_probabilities

@dataclass(frozen=True)
class MondrianLAC:
    thresholds: NDArray[np.float64]
    counts: NDArray[np.int64]
    miscoverage: NDArray[np.float64]

    def predict_mask(self, probabilities: ArrayLike) -> NDArray[np.bool_]:
        return apply_mondrian_lac(probabilities, self.thresholds)

def fit_mondrian_lac(
    calibration_probabilities: ArrayLike,
    calibration_labels: ArrayLike,
    miscoverage: ArrayLike = (0.10, 0.05, 0.10),
    *,
    minimum_counts: Mapping[int, int] | None = None,
) -> MondrianLAC:
    """Fit classwise LAC thresholds from untouched conformal-calibration rows."""

    p = normalize_probabilities(calibration_probabilities)
    if p.ndim == 1:
        p = p[None, :]
    y = np.asarray(calibration_labels, dtype=int)
    alpha = np.asarray(miscoverage, dtype=float)
    if y.shape != (len(p),) or alpha.shape != (p.shape[1],):
        raise ValueError("calibration inputs are misaligned")
    if np.any((y < 0) | (y >= p.shape[1])) or np.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("class labels or miscoverage targets are invalid")
    minimum = {1: 19} if minimum_counts is None and p.shape[1] == 3 else {}
    if minimum_counts is not None:
        minimum = dict(minimum_counts)
    thresholds = np.empty(p.shape[1], dtype=float)
    counts = np.bincount(y, minlength=p.shape[1]).astype(np.int64)
    for class_code in range(p.shape[1]):
        scores = np.sort(1.0 - p[y == class_code, class_code])
        n_class = len(scores)
        if n_class < minimum.get(class_code, 0):
            thresholds[class_code] = np.inf
            continue
        rank = int(np.ceil((n_class + 1) * (1.0 - alpha[class_code])))
        thresholds[class_code] = scores[rank - 1] if rank <= n_class else np.inf
    return MondrianLAC(thresholds, counts, alpha)

def apply_mondrian_lac(
    probabilities: ArrayLike, thresholds: ArrayLike
) -> NDArray[np.bool_]:
    """Apply classwise thresholds and enlarge any empty set with the argmax."""

    p = normalize_probabilities(probabilities)
    if p.ndim == 1:
        p = p[None, :]
    q = np.asarray(thresholds, dtype=float)
    if q.shape != (p.shape[1],) or np.isnan(q).any():
        raise ValueError("thresholds must match class count and not contain NaN")
    mask = 1.0 - p <= q[None, :]
    empty = ~mask.any(axis=1)
    if np.any(empty):
        rows = np.flatnonzero(empty)
        mask[rows, np.argmax(p[rows], axis=1)] = True
    return mask
