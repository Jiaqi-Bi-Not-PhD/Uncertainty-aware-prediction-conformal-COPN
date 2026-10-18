"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike, NDArray

def robust_union(scenario_set_masks: ArrayLike) -> NDArray[np.bool_]:
    """Union set masks with shape ``(rows, scenarios, classes)``."""

    masks = np.asarray(scenario_set_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1] == 0 or masks.shape[2] < 2:
        raise ValueError("scenario_set_masks must have shape (rows, scenarios, classes)")
    if np.any(~masks.any(axis=2)):
        raise ValueError("all scenario-specific sets must be nonempty")
    return masks.any(axis=1)
