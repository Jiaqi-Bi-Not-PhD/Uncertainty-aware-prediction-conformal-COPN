"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

def empirical_threshold_labels(
    values: np.ndarray,
    probabilities: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.asarray(values, dtype=float)
    if numeric.ndim != 1 or not np.isfinite(numeric).all() or len(numeric) == 0:
        raise ValueError("stratification values must be a finite nonempty vector")
    thresholds = np.quantile(numeric, probabilities, method="higher")
    labels = np.searchsorted(thresholds, numeric, side="left") + 1
    return labels.astype(np.int8), np.asarray(thresholds, dtype=float)
