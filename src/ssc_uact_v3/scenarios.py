"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from collections import Counter

import numpy as np

import pandas as pd

PRIMARY_SCIENTIFIC_ID = "primary"

_ZETA_CONTEXTS = (
    "zPD0_AP0_HC0",
    "zPD0_APm0p5_HC0",
    "zPD0_APp0p5_HC0",
    "zPDm0p5_AP0_HC0",
    "zPDp0p5_AP0_HC0",
)

_DELTAS = ("d0", "dm0p5", "dp0p5")

EXPECTED_SCIENTIFIC_IDS = (
    PRIMARY_SCIENTIFIC_ID,
    "mar_missc",
    "mar_missco_ipw",
    *(f"mnar_missc:{delta}:zPD0_AP0_HC0" for delta in _DELTAS),
    *(
        f"mnar_missco_ipw:{delta}:{zeta}"
        for delta in _DELTAS
        for zeta in _ZETA_CONTEXTS
    ),
)

def build_scenario_registry() -> pd.DataFrame:
    families = tuple(scenario_family(value) for value in EXPECTED_SCIENTIFIC_IDS)
    counts = Counter(families)
    rows = []
    for order, (scientific_id, family_id) in enumerate(
        zip(EXPECTED_SCIENTIFIC_IDS, families, strict=True)
    ):
        rows.append(
            {
                "registry_order": order,
                "scientific_scenario_id": scientific_id,
                "family_id": family_id,
                "contexts_in_family": counts[family_id],
                "family_balanced_weight": 0.2 / counts[family_id],
                "raw_equal_weight": 1.0 / len(EXPECTED_SCIENTIFIC_IDS),
                "aggregation_eligible": True,
                "sensitivity_union_eligible": True,
            }
        )
    registry = pd.DataFrame(rows)
    for column in ("family_balanced_weight", "raw_equal_weight"):
        if not np.isclose(registry[column].sum(), 1.0, atol=1e-12):
            raise AssertionError(f"{column} does not sum to one")
    family_mass = registry.groupby("family_id")["family_balanced_weight"].sum()
    if not np.allclose(family_mass.to_numpy(), 0.2, atol=1e-12):
        raise AssertionError("family-balanced registry does not assign equal family mass")
    return registry

def scenario_family(scientific_scenario_id: str) -> str:
    value = str(scientific_scenario_id)
    if value in ("primary", "mar_missc", "mar_missco_ipw"):
        return value
    if value.startswith("mnar_missco_ipw:"):
        return "mnar_missco_ipw"
    if value.startswith("mnar_missc:"):
        return "mnar_missc"
    raise ValueError(f"unknown scientific scenario ID: {value!r}")
