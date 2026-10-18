"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pandas as pd

from .config_model8_mnar import Model8MNARConfig

from .mnar_sensitivity_grid import Model8MNARScenario

from .mnar_tilt import apply_mnar_covariate_tilt

from .model8_design import Model8FoldDesign, build_fold_design

@dataclass
class MNARFoldDesign:
    """Fold-scoped Model 8 design after MAR imputation and MNAR sensitivity tilt."""

    base_design: Model8FoldDesign
    train_mi: list[np.ndarray]
    valid_mi: list[np.ndarray]
    train_imputation_weights: np.ndarray
    valid_imputation_weights: np.ndarray
    groups: dict[str, list[int]]
    feature_names: list[str]
    imputer: object
    eligibility: pd.DataFrame
    threshold_audit: pd.DataFrame
    imputation_diagnostics: pd.DataFrame
    weight_diagnostics: pd.DataFrame
    mapping: pd.DataFrame

def _add_scenario_columns(frame: pd.DataFrame, scenario: Model8MNARScenario) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["scenario_id"] = scenario.scenario_id
    out["delta_scenario_id"] = scenario.delta_scenario_id
    out["zeta_scenario_id"] = scenario.zeta_scenario_id
    out["delta_global"] = float(scenario.delta_global)
    out["zeta_PD"] = float(scenario.zeta_PD)
    out["zeta_AP"] = float(scenario.zeta_AP)
    out["zeta_HC"] = float(scenario.zeta_HC)
    out["in_plausible_set"] = bool(scenario.in_plausible_set)
    out["is_mar_anchor"] = bool(scenario.is_mar_anchor)
    return out

def _check_tau_miss(eligibility: pd.DataFrame, config: Model8MNARConfig) -> None:
    if eligibility.empty or "eligible" not in eligibility.columns:
        return
    retained = eligibility.loc[eligibility["eligible"].astype(bool)]
    if retained.empty:
        return
    bad = retained.loc[retained["missing_fraction_train"].astype(float) > float(config.tau_miss) + 1e-12]
    if not bad.empty:
        examples = ", ".join(bad["source_id"].astype(str).head(5).tolist())
        raise RuntimeError(f"Primary Model 8 MNAR retained sources exceed tau_miss={config.tau_miss}: {examples}")

def build_mnar_fold_design(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Model8MNARConfig,
    budget: int,
    branch: str,
    outer_split_id: str,
    inner_split_id: str,
    mi: int,
    seed_context: str,
    scenario: Model8MNARScenario,
) -> MNARFoldDesign:
    base = build_fold_design(
        X_train,
        y_train,
        X_valid,
        metadata,
        config,
        budget=budget,
        branch=branch,
        outer_split_id=outer_split_id,
        inner_split_id=inner_split_id,
        mi=mi,
        seed_context=seed_context,
    )
    _check_tau_miss(base.eligibility, config)
    train_tilt = apply_mnar_covariate_tilt(
        base.imputer,
        X_train,
        base.train_mi,
        scenario.delta_global,
        scenario.delta_scenario_id,
        branch,
        outer_split_id,
        inner_split_id,
        budget,
    )
    valid_tilt = apply_mnar_covariate_tilt(
        base.imputer,
        X_valid,
        base.valid_mi,
        scenario.delta_global,
        scenario.delta_scenario_id,
        branch,
        outer_split_id,
        inner_split_id,
        budget,
    )
    imputation_diagnostics = pd.concat(
        [
            _add_scenario_columns(base.imputation_diagnostics, scenario),
            _add_scenario_columns(train_tilt.diagnostics.assign(diagnostic_sample="train"), scenario),
            _add_scenario_columns(valid_tilt.diagnostics.assign(diagnostic_sample="valid"), scenario),
        ],
        ignore_index=True,
        sort=False,
    )
    return MNARFoldDesign(
        base_design=base,
        train_mi=train_tilt.matrices,
        valid_mi=valid_tilt.matrices,
        train_imputation_weights=train_tilt.imputation_weights,
        valid_imputation_weights=valid_tilt.imputation_weights,
        groups=base.groups,
        feature_names=base.feature_names,
        imputer=base.imputer,
        eligibility=_add_scenario_columns(base.eligibility, scenario),
        threshold_audit=_add_scenario_columns(base.threshold_audit, scenario),
        imputation_diagnostics=imputation_diagnostics,
        weight_diagnostics=_add_scenario_columns(train_tilt.diagnostics.assign(weight_sample="train"), scenario),
        mapping=_add_scenario_columns(base.mapping, scenario),
    )
