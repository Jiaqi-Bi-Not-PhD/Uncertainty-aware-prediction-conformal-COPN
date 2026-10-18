"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

def _qstats(values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or (not np.isfinite(array).all()):
        raise RuntimeError('report values must be a nonempty finite vector')
    quantiles = np.quantile(array, QUANTILES)
    return {'n': int(array.size), 'mean': float(np.mean(array)), 'sd': float(np.std(array, ddof=1)) if array.size > 1 else 0.0, 'quantiles': {f'q{int(q * 100):02d}': float(v) for q, v in zip(QUANTILES, quantiles)}}
