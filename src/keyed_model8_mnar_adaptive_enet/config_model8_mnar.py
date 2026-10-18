"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

from .config_model8 import Model8Config

@dataclass(frozen=True)
class Model8MNARConfig(Model8Config):
    """Resolved configuration for the Model 8 MNAR sensitivity workflow."""

    output_dir: str = ""
    random_state: int = 20260524
    branches: tuple[str, ...] = ("mnar_missc", "mnar_missco_ipw")
    imputer_type: str = "mixed_chained_or_em"
    tau_miss: float = 0.50

    delta_grid: tuple[float, ...] = (-0.5, 0.0, 0.5)
    zeta_grid: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, -0.5, 0.0),
        (0.5, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
    )
    plausible_delta_abs_max: float = 0.5
    plausible_zeta_abs_max: float = 0.5
    delta_mode: str = "global_delta"
    covariate_tilt_modes: tuple[str, ...] = ("numeric_shift", "importance_weighted_draws")
    categorical_tilt_enabled: bool = False
    ppca_numeric_sensitivity_enabled: bool = False

    lambda_min_ratio: float = 0.001
    n_lambdas: int = 18
    rho_floor_grid: tuple[float, ...] = (0.05,)
    missco_weight_floors: tuple[float, ...] = (0.05,)
    primary_selection_metric: str = "mean_threshold_balanced_accuracy"
