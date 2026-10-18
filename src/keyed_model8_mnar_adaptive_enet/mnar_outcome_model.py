"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from typing import Any

import numpy as np

from keyed_model1.common import CLASS_NAMES, row_weights

from .model8_missco_weights import fit_outcome_observation_model

def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or np.any(~np.isfinite(w)) or np.any(w < 0):
        return float("nan")
    denom = float(np.sum(w * w))
    return float((w.sum() ** 2) / denom) if denom > 0 else float("nan")

def class_zeta_vector(zeta_pd: float, zeta_ap: float, zeta_hc: float) -> np.ndarray:
    return np.array([float(zeta_pd), float(zeta_ap), float(zeta_hc)], dtype=float)

def apply_zeta_to_rho(rho_base: np.ndarray, y_observed: np.ndarray, zeta_pd: float, zeta_ap: float, zeta_hc: float) -> np.ndarray:
    rho = np.clip(np.asarray(rho_base, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y_observed, dtype=int)
    zeta = class_zeta_vector(zeta_pd, zeta_ap, zeta_hc)
    logits = np.log(rho / (1.0 - rho)) + zeta[y]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))

def mnar_missco_ipw_weights(
    y_observed: np.ndarray,
    rho_base_observed: np.ndarray,
    floor: float,
    atypical_weight_multiplier: float,
    zeta_pd: float,
    zeta_ap: float,
    zeta_hc: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(y_observed, dtype=int)
    rho_tilted = apply_zeta_to_rho(rho_base_observed, y, zeta_pd, zeta_ap, zeta_hc)
    clipped = np.clip(rho_tilted, float(floor), 1.0)
    if np.any(~np.isfinite(clipped)) or np.any(clipped <= 0):
        raise ValueError("Invalid zeta-adjusted outcome-observation probabilities after truncation.")
    ipw = 1.0 / clipped
    class_balanced = row_weights(y, n_classes=len(CLASS_NAMES))
    weights = ipw * class_balanced
    if float(atypical_weight_multiplier) != 1.0:
        weights[y == 1] *= float(atypical_weight_multiplier)
    diagnostics: dict[str, Any] = {
        "missco_weight_floor": float(floor),
        "rho_floor": float(floor),
        "zeta_PD": float(zeta_pd),
        "zeta_AP": float(zeta_ap),
        "zeta_HC": float(zeta_hc),
        "rho_min_raw": float(np.min(rho_base_observed)) if len(rho_base_observed) else np.nan,
        "rho_min_zeta": float(np.min(rho_tilted)) if len(rho_tilted) else np.nan,
        "rho_min_truncated": float(np.min(clipped)) if len(clipped) else np.nan,
        "rho_mean_zeta": float(np.mean(rho_tilted)) if len(rho_tilted) else np.nan,
        "truncation_rate": float(np.mean(rho_tilted < float(floor))) if len(rho_tilted) else np.nan,
        "ipw_max": float(np.max(ipw)) if len(ipw) else np.nan,
        "ipw_mean": float(np.mean(ipw)) if len(ipw) else np.nan,
        "training_weight_max": float(np.max(weights)) if len(weights) else np.nan,
        "training_weight_mean": float(np.mean(weights)) if len(weights) else np.nan,
        "max_row_weight": float(np.max(weights)) if len(weights) else np.nan,
        "n_eff_overall": effective_sample_size(weights),
    }
    for code, label in enumerate(CLASS_NAMES):
        mask = y == code
        diagnostics[f"n_eff_{label}"] = effective_sample_size(weights[mask]) if np.any(mask) else np.nan
    diagnostics["n_eff_PD"] = diagnostics.get("n_eff_Parkinson", np.nan)
    diagnostics["n_eff_AP"] = diagnostics.get("n_eff_Atypical", np.nan)
    diagnostics["n_eff_HC"] = diagnostics.get("n_eff_Healthy", np.nan)
    return weights, diagnostics
