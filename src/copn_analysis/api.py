"""Portable entry points around the manuscript's retained scientific routines.

The selected-fit functions preserve the original fitting code. Their inputs
must already respect the study's training, tuning, calibration and test roles.
"""
from __future__ import annotations

import importlib
from typing import Any

from . import model_fitting

_MODEL_MODULES = {
    "M1": ("keyed_model1.model1_pipeline", model_fitting._fit_model1),
    "M2": ("keyed_catboost.model2_catboost", model_fitting._fit_model2),
    "M3": ("keyed_ebm.model3_ebm", model_fitting._fit_model3),
    "M4": ("keyed_hierarchical.model5_hierarchical", model_fitting._fit_model5),
}


def fit_selected_probability_model(
    manuscript_model: str, *, config: Any, dataset: Any,
    X_train: Any, y_train: Any, X_score: Any, selected: dict, budget: int,
):
    """Fit a selected M1-M4 model using training records and score other records.

    ``dataset.metadata`` supplies the dictionary-derived source metadata.
    ``selected`` is a training-only selected tuning row. CatBoost additionally
    needs its selected ``median_best_iteration``. No tuning uses X_score labels.
    Returns probabilities, fitted objects and numerical diagnostics.
    """
    if manuscript_model not in _MODEL_MODULES:
        raise ValueError("Use M1, M2, M3 or M4; M5 uses fit_selected_m5.")
    module, function = _MODEL_MODULES[manuscript_model]
    workflow = importlib.import_module(module)
    kwargs = dict(workflow=workflow, config=config, dataset=dataset,
                  X_train=X_train, y_train=y_train, X_score=X_score,
                  selected=selected, budget=int(budget))
    if manuscript_model == "M2":
        kwargs["inner"] = False
    return function(**kwargs)


def fit_selected_m5(*, design, y_train, scenario, sample_weight, config, mnar=False):
    """Fit the original MAR or MNAR adaptive elastic-net sensitivity model.

    ``design`` must be built on training records with the corresponding
    model8_design.build_fold_design or mnar_imputation.build_mnar_fold_design.
    The scenario and optional outcome-observation weights are training-derived.
    """
    module = ("keyed_model8_mnar_adaptive_enet.mnar_workflow" if mnar else
              "keyed_model8_adaptive_enet.model8_workflow")
    return importlib.import_module(module)._fit_model8(
        design, y_train, scenario, sample_weight, config)
