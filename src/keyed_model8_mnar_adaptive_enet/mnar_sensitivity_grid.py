"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import json

from dataclasses import dataclass

from typing import Any

from .config_model8_mnar import Model8MNARConfig

def _num_token(value: float) -> str:
    if value == 0:
        return "0"
    prefix = "p" if value > 0 else "m"
    return prefix + str(abs(float(value))).replace(".", "p")

@dataclass(frozen=True)
class Model8MNARScenario:
    """One fixed MNAR sensitivity scenario plus one inner tuning configuration."""

    branch: str
    alpha: float
    gamma: float
    lambda_fraction: float
    atypical_weight_multiplier: float
    delta_scenario_id: str
    delta_global: float
    zeta_scenario_id: str
    zeta_PD: float
    zeta_AP: float
    zeta_HC: float
    imputer_type: str
    mi: int
    rho_floor: float | None = None
    delta_mode: str = "global_delta"
    in_plausible_set: bool = True

    @property
    def scenario_id(self) -> str:
        base = (
            f"{self.branch}_{self.delta_scenario_id}_{self.zeta_scenario_id}_"
            f"a{self.alpha:.3g}_g{self.gamma:.3g}_lf{self.lambda_fraction:.4g}_"
            f"apw{self.atypical_weight_multiplier:.3g}"
        )
        if self.rho_floor is not None:
            base += f"_rf{self.rho_floor:.3g}"
        return base.replace(".", "p")

    @property
    def is_mar_anchor(self) -> bool:
        return self.delta_global == 0.0 and self.zeta_PD == 0.0 and self.zeta_AP == 0.0 and self.zeta_HC == 0.0

    def registry_row(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "branch": self.branch,
            "delta_scenario_id": self.delta_scenario_id,
            "zeta_scenario_id": self.zeta_scenario_id,
            "delta_mode": self.delta_mode,
            "delta_values": json.dumps({"global_delta": self.delta_global}, sort_keys=True),
            "delta_global": float(self.delta_global),
            "zeta_PD": float(self.zeta_PD),
            "zeta_AP": float(self.zeta_AP),
            "zeta_HC": float(self.zeta_HC),
            "is_mar_anchor": bool(self.is_mar_anchor),
            "in_plausible_set": bool(self.in_plausible_set),
            "imputer_type": self.imputer_type,
            "mi": int(self.mi),
            "rho_floor": self.rho_floor,
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "lambda_fraction": float(self.lambda_fraction),
            "atypical_weight_multiplier": float(self.atypical_weight_multiplier),
            "notes": "Fixed MNAR sensitivity scenario; delta/zeta are not selected by outer-test performance.",
        }

def delta_scenarios(config: Model8MNARConfig, max_delta_scenarios: int | None = None) -> list[dict[str, Any]]:
    ordered = sorted((float(delta) for delta in config.delta_grid), key=lambda value: (abs(value) > 1e-15, abs(value), value))
    rows = [
        {
            "delta_scenario_id": f"d{_num_token(float(delta))}",
            "delta_global": float(delta),
            "delta_mode": config.delta_mode,
        }
        for delta in ordered
    ]
    return rows[: int(max_delta_scenarios)] if max_delta_scenarios is not None else rows

def zeta_scenarios(config: Model8MNARConfig, max_zeta_scenarios: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for zeta_pd, zeta_ap, zeta_hc in config.zeta_grid:
        rows.append(
            {
                "zeta_scenario_id": f"zPD{_num_token(float(zeta_pd))}_AP{_num_token(float(zeta_ap))}_HC{_num_token(float(zeta_hc))}",
                "zeta_PD": float(zeta_pd),
                "zeta_AP": float(zeta_ap),
                "zeta_HC": float(zeta_hc),
            }
        )
    return rows[: int(max_zeta_scenarios)] if max_zeta_scenarios is not None else rows

def _is_plausible(config: Model8MNARConfig, delta: float, zeta_pd: float, zeta_ap: float, zeta_hc: float) -> bool:
    return (
        abs(float(delta)) <= float(config.plausible_delta_abs_max)
        and abs(float(zeta_pd)) <= float(config.plausible_zeta_abs_max)
        and abs(float(zeta_ap)) <= float(config.plausible_zeta_abs_max)
        and abs(float(zeta_hc)) <= 1e-12
    )

def scenario_registry(
    config: Model8MNARConfig,
    max_scenarios: int | None = None,
    branches: tuple[str, ...] | None = None,
    max_delta_scenarios: int | None = None,
    max_zeta_scenarios: int | None = None,
    mi: int | None = None,
    n_lambdas: int | None = None,
) -> list[Model8MNARScenario]:
    selected_branches = branches or tuple(config.branches)
    lambdas = tuple(config.lambda_fractions[: int(n_lambdas)]) if n_lambdas is not None else tuple(config.lambda_fractions)
    base_rows: list[dict[str, float]] = []
    for alpha in config.alpha_grid:
        for gamma in config.gamma_grid:
            for frac in lambdas:
                for apw in config.ap_weight_multipliers:
                    base_rows.append(
                        {
                            "alpha": float(alpha),
                            "gamma": float(gamma),
                            "lambda_fraction": float(frac),
                            "atypical_weight_multiplier": float(apw),
                        }
                    )
    if max_scenarios is not None:
        base_rows = base_rows[: int(max_scenarios)]

    out: list[Model8MNARScenario] = []
    zero_zeta = [{"zeta_scenario_id": "zPD0_AP0_HC0", "zeta_PD": 0.0, "zeta_AP": 0.0, "zeta_HC": 0.0}]
    for branch in selected_branches:
        zetas = zero_zeta if branch == "mnar_missc" else zeta_scenarios(config, max_zeta_scenarios)
        floors: tuple[float | None, ...] = (None,) if branch == "mnar_missc" else tuple(float(x) for x in config.missco_weight_floors)
        for delta in delta_scenarios(config, max_delta_scenarios):
            for zeta in zetas:
                plausible = _is_plausible(config, delta["delta_global"], zeta["zeta_PD"], zeta["zeta_AP"], zeta["zeta_HC"])
                for base in base_rows:
                    for floor in floors:
                        out.append(
                            Model8MNARScenario(
                                branch=branch,
                                delta_scenario_id=str(delta["delta_scenario_id"]),
                                delta_global=float(delta["delta_global"]),
                                delta_mode=str(delta["delta_mode"]),
                                zeta_scenario_id=str(zeta["zeta_scenario_id"]),
                                zeta_PD=float(zeta["zeta_PD"]),
                                zeta_AP=float(zeta["zeta_AP"]),
                                zeta_HC=float(zeta["zeta_HC"]),
                                imputer_type=config.imputer_type,
                                mi=int(mi or config.mi_screening),
                                rho_floor=floor,
                                in_plausible_set=plausible,
                                **base,
                            )
                        )
    return out
