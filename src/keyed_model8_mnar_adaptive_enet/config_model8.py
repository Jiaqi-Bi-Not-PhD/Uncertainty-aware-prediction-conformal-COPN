"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass, field

from keyed_model1.common import DEFAULT_EXCLUDE_PATTERNS, DEFAULT_GROUP_LEVELS, KEY_COL, OUTCOME_COL, OUTCOME_MAP, OUTCOME_SHEET

@dataclass(frozen=True)
class Model8Config:
    """Resolved configuration for SSC Model 8.

    The first block mirrors the active keyed Model 1 config so that assembly,
    class coding, and split creation stay compatible with existing workflows.
    """

    workbook_path: str = ""
    data_dictionary_path: str = ""
    output_dir: str = ""
    key_col: str = KEY_COL
    outcome_sheet: str = OUTCOME_SHEET
    outcome_col: str = OUTCOME_COL
    outcome_map: dict[str, int] = field(default_factory=lambda: dict(OUTCOME_MAP))
    required_sheets: list[str] | None = None
    predictor_sheets: list[str] | None = None
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    group_level_defaults: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_GROUP_LEVELS))
    budgets: tuple[int, ...] = (1, 2, 3, 4)
    min_observed_fraction: float = 0.02
    min_observed_per_class: dict[str, int] = field(default_factory=lambda: {"Parkinson": 10, "Atypical": 1, "Healthy": 5})
    rare_category_min_count: int = 5
    max_categories: int = 40
    numeric_parse_threshold: float = 0.90
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    quasi_constant_threshold: float = 0.995
    outer_folds: int = 5
    outer_repeats: int = 3
    inner_folds: int = 4
    random_state: int = 20260523
    min_ap_in_outer_test: int = 5
    min_ap_in_inner_valid: int = 1
    parallel_jobs: int = 1

    tau_miss: float = 0.50
    branches: tuple[str, ...] = ("missc", "missco_ipw")
    alpha_grid: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95, 1.00)
    gamma_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    lambda_fractions: tuple[float, ...] = (
        1.0,
        0.666085,
        0.443669,
        0.295521,
        0.196842,
        0.131114,
        0.087333,
        0.058179,
        0.038747,
        0.025805,
        0.01719,
        0.011455,
        0.007627,
        0.00508,
        0.003384,
        0.002254,
        0.001501,
        0.001,
    )
    ap_weight_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    mi_screening: int = 10
    mi_final: int = 20
    imputer_type: str = "mixed_chained"
    missco_weight_floors: tuple[float, ...] = (0.05,)
    outcome_model_l2: float = 1.0
    outcome_model_max_iter: int = 300
    outcome_model_tol: float = 1e-6
    max_iter: int = 500
    tol: float = 1e-5
