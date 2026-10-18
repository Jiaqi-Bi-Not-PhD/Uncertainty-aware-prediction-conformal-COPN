"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numpy.typing import ArrayLike, NDArray

from .metrics import BIT_COUNT, robust_expected_set_cost

@dataclass(frozen=True)
class FrozenPolicyThresholds:
    reliability_threshold: float
    threshold_valid: bool
    hc_reliability_threshold: float
    hc_disease_threshold: float
    hc_threshold_valid: bool

@dataclass(frozen=True)
class SafetyLayerResult:
    displayed_bits: NDArray[np.uint8]
    robust_expected_cost: NDArray[np.float64]
    enlarged: NDArray[np.bool_]
    labels_added: NDArray[np.int8]
    reason_codes: tuple[str, ...]
    saved_hc_gate_passed: NDArray[np.bool_]

def apply_a4_srs_gate(
    conformal_bits: ArrayLike,
    scenario_probabilities: ArrayLike,
    srs: ArrayLike,
    thresholds: FrozenPolicyThresholds,
    cost_lookup: ArrayLike,
) -> SafetyLayerResult:
    """Enlarge only A3 singletons that fail the frozen general SRS gate."""

    base, scenarios, srs_values, costs = _validate_inputs(
        conformal_bits, scenario_probabilities, srs, cost_lookup
    )
    selected = base.copy()
    reasons: list[str] = []
    for index, mask in enumerate(base):
        if BIT_COUNT[mask] != 1:
            reasons.append("A4_NON_SINGLETON_RETAINED")
            continue
        if not np.isfinite(srs_values[index]):
            selected[index] = 7
            reasons.append("NONFINITE_SRS_FULL_SET")
            continue
        if not thresholds.threshold_valid:
            selected[index] = minimum_cost_superset(
                int(mask), scenarios[index], costs, strict=True
            )
            reasons.append("INVALID_GENERAL_THRESHOLD_ENLARGEMENT")
            continue
        if srs_values[index] < thresholds.reliability_threshold:
            selected[index] = minimum_cost_superset(
                int(mask), scenarios[index], costs, strict=True
            )
            reasons.append("LOW_RELIABILITY_ENLARGEMENT")
        else:
            reasons.append("A4_SRS_GATE_PASSED")
    risk = robust_expected_set_cost(selected, scenarios, costs)
    saved_hc = saved_hc_gate_pass(
        base, scenarios, srs_values, thresholds
    )
    return _result(base, selected, risk, reasons, saved_hc)

def apply_a5_harm_policy(
    a4_bits: ArrayLike,
    scenario_probabilities: ArrayLike,
    srs: ArrayLike,
    thresholds: FrozenPolicyThresholds,
    cost_lookup: ArrayLike,
    *,
    validated_non_ood: ArrayLike | None = None,
    fail_closed_when_ood_unavailable: bool = True,
) -> SafetyLayerResult:
    """Apply HC safety and minimum-risk superset selection to A4.

    The primary August implementation passes ``validated_non_ood=None`` and
    fails closed. A future validated OOD vector may be supplied only under a
    new protocol freeze.
    """

    base, scenarios, srs_values, costs = _validate_inputs(
        a4_bits, scenario_probabilities, srs, cost_lookup
    )
    if validated_non_ood is None:
        if not fail_closed_when_ood_unavailable:
            raise ValueError("unavailable OOD status must fail closed")
        non_ood = np.zeros(len(base), dtype=bool)
        ood_available = False
    else:
        non_ood = np.asarray(validated_non_ood)
        if non_ood.shape != (len(base),) or non_ood.dtype != bool:
            raise ValueError("validated_non_ood must be a boolean row vector")
        ood_available = True

    hc_gate = saved_hc_gate_pass(base, scenarios, srs_values, thresholds)
    selected = base.copy()
    reasons: list[str] = []
    for index, mask in enumerate(base):
        if not np.isfinite(srs_values[index]):
            selected[index] = 7
            reasons.append("NONFINITE_SRS_FULL_SET")
            continue
        mandatory_strict = False
        reason = "A5_COST_POLICY_RETAINED"
        if int(mask) == 4:
            if not ood_available:
                mandatory_strict = True
                reason = "HC_OOD_UNAVAILABLE_FAIL_CLOSED"
            elif not non_ood[index]:
                mandatory_strict = True
                reason = "HC_OOD_GATE_FAILED"
            elif not hc_gate[index]:
                mandatory_strict = True
                reason = "HC_SINGLETON_GATE_FAILED"
        chosen = minimum_cost_superset(
            int(mask), scenarios[index], costs, strict=mandatory_strict
        )
        selected[index] = chosen
        if chosen != mask and not mandatory_strict:
            reason = "COST_AWARE_ENLARGEMENT"
        reasons.append(reason)
    risk = robust_expected_set_cost(selected, scenarios, costs)
    return _result(base, selected, risk, reasons, hc_gate)

def saved_hc_gate_pass(
    displayed_bits: ArrayLike,
    scenario_probabilities: ArrayLike,
    srs: ArrayLike,
    thresholds: FrozenPolicyThresholds,
) -> NDArray[np.bool_]:
    bits = np.asarray(displayed_bits, dtype=np.uint8)
    scenarios = np.asarray(scenario_probabilities, dtype=float)
    srs_values = np.asarray(srs, dtype=float)
    if scenarios.ndim != 3 or scenarios.shape != (len(bits), scenarios.shape[1], 3):
        raise ValueError("scenario probabilities have an invalid shape")
    if srs_values.shape != (len(bits),):
        raise ValueError("SRS is misaligned")
    worst_disease = np.max(scenarios[:, :, 0] + scenarios[:, :, 1], axis=1)
    return (
        (bits == 4)
        & bool(thresholds.hc_threshold_valid)
        & np.isfinite(srs_values)
        & (srs_values >= thresholds.hc_reliability_threshold)
        & (worst_disease <= thresholds.hc_disease_threshold)
    )

def minimum_cost_superset(
    base_mask: int,
    scenario_probabilities: ArrayLike,
    cost_lookup: ArrayLike,
    *,
    strict: bool,
) -> np.uint8:
    if base_mask < 1 or base_mask > 7:
        raise ValueError("base mask must be a nonempty three-class set")
    scenarios = np.asarray(scenario_probabilities, dtype=float)
    costs = np.asarray(cost_lookup, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != 3:
        raise ValueError("one participant must have scenario-by-class probabilities")
    if costs.shape != (8, 3):
        raise ValueError("cost lookup must have shape (8, 3)")
    candidates = [
        mask
        for mask in range(1, 8)
        if (mask & base_mask) == base_mask and (not strict or mask != base_mask)
    ]
    if not candidates:
        return np.uint8(7)
    scored = []
    for mask in candidates:
        expected = scenarios @ costs[mask]
        scored.append((float(np.max(expected)), int(BIT_COUNT[mask]), mask))
    return np.uint8(min(scored)[2])

def _validate_inputs(
    bits: ArrayLike,
    scenario_probabilities: ArrayLike,
    srs: ArrayLike,
    cost_lookup: ArrayLike,
) -> tuple[
    NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    base = np.asarray(bits)
    scenarios = np.asarray(scenario_probabilities, dtype=float)
    srs_values = np.asarray(srs, dtype=float)
    costs = np.asarray(cost_lookup, dtype=float)
    if base.ndim != 1 or not np.issubdtype(base.dtype, np.integer):
        raise ValueError("set masks must be a one-dimensional integer vector")
    base = base.astype(np.uint8)
    if np.any((base < 1) | (base > 7)):
        raise ValueError("set masks must be nonempty three-class masks")
    if scenarios.ndim != 3 or scenarios.shape != (len(base), scenarios.shape[1], 3):
        raise ValueError("scenario probabilities and sets are misaligned")
    if scenarios.shape[1] == 0 or not np.isfinite(scenarios).all():
        raise ValueError("scenario probabilities must be finite and nonempty")
    if np.any((scenarios < -1e-10) | (scenarios > 1 + 1e-10)) or not np.allclose(
        scenarios.sum(axis=2), 1.0, atol=1e-8, rtol=0
    ):
        raise ValueError("scenario probabilities fail simplex validation")
    if srs_values.shape != (len(base),):
        raise ValueError("SRS and set rows are misaligned")
    if costs.shape != (8, 3) or np.isnan(costs[1:]).any():
        raise ValueError("cost lookup is invalid")
    return base, scenarios, srs_values, costs

def _result(
    base: NDArray[np.uint8],
    selected: NDArray[np.uint8],
    risk: NDArray[np.float64],
    reasons: list[str],
    saved_hc: NDArray[np.bool_],
) -> SafetyLayerResult:
    if np.any((base & ~selected) != 0):
        raise AssertionError("a safety layer removed a label")
    enlarged = selected != base
    labels_added = (BIT_COUNT[selected] - BIT_COUNT[base]).astype(np.int8)
    return SafetyLayerResult(
        displayed_bits=selected,
        robust_expected_cost=risk,
        enlarged=enlarged,
        labels_added=labels_added,
        reason_codes=tuple(reasons),
        saved_hc_gate_passed=saved_hc,
    )
