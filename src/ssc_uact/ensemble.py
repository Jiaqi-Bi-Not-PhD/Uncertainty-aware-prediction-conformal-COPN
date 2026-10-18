"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike, NDArray

from .probability import normalize_probabilities

def fixed_blend(
    model2_probabilities: ArrayLike,
    model5_probabilities: ArrayLike,
    weights: tuple[float, float] = (0.6, 0.4),
) -> NDArray[np.float64]:
    """Return the preregistered calibrated Model 2/Model 5 convex blend."""

    if len(weights) != 2 or min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("weights must be a two-element probability simplex")
    p2 = normalize_probabilities(model2_probabilities)
    p5 = normalize_probabilities(model5_probabilities)
    if p2.shape != p5.shape:
        raise ValueError("model probability arrays must have identical shapes")
    return normalize_probabilities(weights[0] * p2 + weights[1] * p5)
