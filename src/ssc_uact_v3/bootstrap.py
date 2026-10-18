"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import numpy as np

from scipy.stats import beta

def _bootstrap_multiplicities(
    clusters: int, replicates: int, seed: int
) -> np.ndarray:
    counts = np.empty((replicates, clusters), dtype=np.int16)
    probability = np.full(clusters, 1.0 / clusters)
    for replicate in range(replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        counts[replicate] = rng.multinomial(clusters, probability).astype(np.int16)
    if not np.all(counts.sum(axis=1) == clusters):
        raise AssertionError("bootstrap cluster multiplicities have the wrong total")
    return counts

def _binomial_upper(events: int, total: int, alpha: float) -> float:
    if total <= 0 or events < 0 or events > total:
        raise ValueError("invalid binomial count")
    if events == total:
        return 1.0
    return float(beta.ppf(1 - alpha, events + 1, total - events))
