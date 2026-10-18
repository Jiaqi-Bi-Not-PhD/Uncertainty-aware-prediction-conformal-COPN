from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
RIDGE_ALPHA=0.0
RIDGE_LAMBDA=1e-3
SOLVER_MAX_ITER=1000
SOLVER_MAX_FUN=4000
SOLVER_MAX_COR=20
SOLVER_MAX_LS=50
SOLVER_FTOL=1e-12
SOLVER_GTOL=1e-6

"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from typing import Any, Mapping, Sequence
import numpy as np
FROZEN_TOLERANCE = 1e-05
FROZEN_MAX_ITER = 500

class RidgeInputError(RuntimeError):
    """Raised when the exact smooth ridge initializer fails closed."""

def _validated_inputs(X_list: Sequence[np.ndarray], y: np.ndarray, groups: Mapping[str, Sequence[int]], sample_weight: np.ndarray, *, gamma: float, max_iter: int, tol: float) -> tuple[list[np.ndarray], np.ndarray, dict[str, list[int]], np.ndarray]:
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or int(max_iter) != FROZEN_MAX_ITER:
        raise RidgeInputError('initializer iteration cap differs')
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, np.integer, np.floating)) or (not np.isfinite(float(tol))) or (float(tol) != FROZEN_TOLERANCE):
        raise RidgeInputError('initializer tolerance differs')
    if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, (int, float, np.integer, np.floating)) or (not np.isfinite(float(gamma))) or (float(gamma) <= 0):
        raise RidgeInputError('adaptive exponent differs')
    if not X_list:
        raise RidgeInputError('imputed training designs are absent')
    raw_arrays = [np.asarray(value) for value in X_list]
    if any((not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.complexfloating) or np.issubdtype(value.dtype, np.bool_) for value in raw_arrays)):
        raise RidgeInputError('training design dtype differs')
    arrays = [np.ascontiguousarray(value, dtype=np.float64) for value in raw_arrays]
    if arrays[0].ndim != 2 or min(arrays[0].shape) <= 0:
        raise RidgeInputError('training design geometry differs')
    shape = arrays[0].shape
    if any((value.shape != shape or np.any(~np.isfinite(value)) for value in arrays)):
        raise RidgeInputError('imputed training designs differ')
    raw_labels = np.asarray(y)
    raw_weights = np.asarray(sample_weight)
    if not np.issubdtype(raw_labels.dtype, np.integer) or np.issubdtype(raw_labels.dtype, np.bool_) or (not np.issubdtype(raw_weights.dtype, np.number)) or np.issubdtype(raw_weights.dtype, np.complexfloating) or np.issubdtype(raw_weights.dtype, np.bool_):
        raise RidgeInputError('training label or weight dtype differs')
    labels = np.ascontiguousarray(raw_labels, dtype=np.int64)
    weights = np.ascontiguousarray(raw_weights, dtype=np.float64)
    if labels.shape != (shape[0],) or set(labels.tolist()) != {0, 1, 2} or weights.shape != labels.shape or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise RidgeInputError('training labels or weights differ')
    if any((type(name) is not str or not name for name in groups)):
        raise RidgeInputError('source block name differs')
    normalized_groups: dict[str, list[int]] = {}
    for name, indices in groups.items():
        values = list(indices)
        if any((isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in values)):
            raise RidgeInputError('source block index dtype differs')
        normalized_groups[name] = [int(index) for index in values]
    if not normalized_groups or any((not indices for indices in normalized_groups.values())):
        raise RidgeInputError('source blocks are absent')
    flattened = [index for indices in normalized_groups.values() for index in indices]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != list(range(shape[1])):
        raise RidgeInputError('source blocks are not an exact design-column partition')
    return (arrays, labels, normalized_groups, weights)

def _original_step_verification(model: Any, X_list: list[np.ndarray], y: np.ndarray, weights: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if model.intercept_ is None or model.coef_ is None:
        raise RidgeInputError('ridge solution state is absent')
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    coefficient = np.asarray(model.coef_, dtype=np.float64)
    previous = float(model._objective(X_list, y, weights, intercept, coefficient))
    grad_intercept, grad_coefficient = model._gradient(X_list, y, weights, intercept, coefficient)
    initial_step = float(model._initial_step(X_list, weights))
    local_step = initial_step
    accepted = False
    candidate_intercept = intercept
    candidate_coefficient = coefficient
    current = previous
    halvings = 0
    for halvings in range(40):
        candidate_intercept = intercept - local_step * grad_intercept
        candidate_coefficient = model._prox(coefficient - local_step * grad_coefficient, local_step)
        current = float(model._objective(X_list, y, weights, candidate_intercept, candidate_coefficient))
        if np.isfinite(current) and current <= previous + 1e-10:
            accepted = True
            break
        local_step *= 0.5
    if not accepted:
        raise RidgeInputError('original-step convergence verification line search failed')
    max_change = float(max(np.max(np.abs(candidate_intercept - intercept)), np.max(np.abs(candidate_coefficient - coefficient))))
    relative_objective_change = abs(previous - current) / max(1.0, abs(previous))
    receipt = {'accepted': True, 'line_search_halvings': int(halvings), 'initial_step_size': initial_step, 'accepted_step_size': float(local_step), 'next_proposed_step_size': float(min(local_step * 1.05, 1.0)), 'objective': previous, 'next_objective': current, 'relative_objective_change': float(relative_objective_change), 'max_coef_or_intercept_change': max_change, 'passes_original_convergence_rule': bool(relative_objective_change < model.tol and max_change < max(1e-05, model.tol * 10.0))}
    return (receipt, np.ascontiguousarray(candidate_intercept, dtype=np.float64), np.ascontiguousarray(candidate_coefficient, dtype=np.float64))
class RidgeInitializationError(RuntimeError):
    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = dict(evidence or {})

def compute_q_adaptive_weights(X_list: Sequence[np.ndarray], y: np.ndarray, groups: Mapping[str, Sequence[int]], *, sample_weight: np.ndarray, gamma: float, max_iter: int, tol: float, adaptive_module: Any) -> tuple[dict[str, float], Any, Mapping[str, Any]]:
    """Fit the reported Q ridge initializer, returning weights, fitted model and numerical diagnostics."""
    try:
        arrays, labels, normalized_groups, weights = _validated_inputs(X_list, y, groups, sample_weight, gamma=gamma, max_iter=max_iter, tol=tol)
    except RidgeInputError as exc:
        raise RidgeInitializationError(str(exc)) from exc
    model = adaptive_module.AdaptiveMIGroupedMultinomialElasticNet(alpha=RIDGE_ALPHA, lambda_=RIDGE_LAMBDA, gamma=float(gamma), groups=normalized_groups, adaptive_weights={name: 1.0 for name in normalized_groups}, max_iter=int(max_iter), tol=float(tol))
    p = arrays[0].shape[1]
    intercept0 = np.asarray(model._intercept_only(labels, weights, 3), dtype=np.float64)
    coefficient0 = np.zeros((p, 2), dtype=np.float64)
    initial_vector = np.concatenate([intercept0, coefficient0.ravel(order='C')])
    initial_objective = float(model._objective(arrays, labels, weights, intercept0, coefficient0))

    def objective_and_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
        parameter = np.asarray(vector, dtype=np.float64)
        intercept = parameter[:2]
        coefficient = parameter[2:].reshape((p, 2), order='C')
        objective = float(model._objective(arrays, labels, weights, intercept, coefficient))
        grad_intercept, grad_coefficient = model._gradient(arrays, labels, weights, intercept, coefficient)
        gradient = np.concatenate([np.asarray(grad_intercept, dtype=np.float64), np.asarray(grad_coefficient, dtype=np.float64).ravel(order='C')])
        if not np.isfinite(objective) or np.any(~np.isfinite(gradient)):
            raise RidgeInitializationError('ridge objective or gradient became non-finite', evidence={'phase': 'objective_or_gradient_evaluation'})
        return (objective, gradient)
    with threadpool_limits(limits=1):
        try:
            result = minimize(objective_and_gradient, initial_vector, method='L-BFGS-B', jac=True, options={'maxiter': SOLVER_MAX_ITER, 'maxfun': SOLVER_MAX_FUN, 'maxcor': SOLVER_MAX_COR, 'maxls': SOLVER_MAX_LS, 'ftol': SOLVER_FTOL, 'gtol': SOLVER_GTOL})
        except BaseException as exc:
            raise RidgeInitializationError('SciPy L-BFGS-B invocation raised before returning a result', evidence={'optimizer_invocation_count': 1, 'optimizer_result_returned': False, 'exception_type': type(exc).__name__, 'exception_message': str(exc)}) from exc
    final_vector = np.asarray(result.x, dtype=np.float64)
    model.intercept_ = np.ascontiguousarray(final_vector[:2], dtype=np.float64)
    model.coef_ = np.ascontiguousarray(final_vector[2:].reshape((p, 2), order='C'), dtype=np.float64)
    lbfgs_objective, lbfgs_gradient = objective_and_gradient(final_vector)
    lbfgs_gradient_infinity_norm = float(np.max(np.abs(lbfgs_gradient)))
    try:
        verification, verified_intercept, verified_coefficient = _original_step_verification(model, arrays, labels, weights)
    except RidgeInputError as exc:
        raise RidgeInitializationError(str(exc), evidence={'solver': {'status': int(result.status), 'success': bool(result.success), 'message': str(result.message), 'iterations': int(result.nit), 'function_evaluations': int(result.nfev), 'gradient_evaluations': int(result.njev)}, 'lbfgs_objective': lbfgs_objective, 'lbfgs_gradient_infinity_norm': lbfgs_gradient_infinity_norm}) from exc
    model.intercept_ = verified_intercept
    model.coef_ = verified_coefficient
    verified_vector = np.concatenate([verified_intercept, verified_coefficient.ravel(order='C')])
    final_objective, final_gradient = objective_and_gradient(verified_vector)
    final_gradient_infinity_norm = float(np.max(np.abs(final_gradient)))
    epsilon = max(len(labels), 1) ** (-0.5)
    norms = model.source_norms()
    adaptive_weights = {name: float((norms.get(name, 0.0) + epsilon) ** (-float(gamma))) for name in normalized_groups}
    if set(adaptive_weights) != set(normalized_groups) or any((not np.isfinite(value) or value <= 0 for value in adaptive_weights.values())):
        raise RidgeInitializationError('adaptive weights are invalid', evidence={'solver': {'status': int(result.status), 'success': bool(result.success), 'message': str(result.message), 'iterations': int(result.nit), 'function_evaluations': int(result.nfev), 'gradient_evaluations': int(result.njev), 'optimizer_invocation_count': 1, 'optimizer_result_returned': True}})
    termination = {'success': bool(result.success), 'status': int(result.status), 'iterations': int(result.nit), 'initial_objective': initial_objective, 'final_objective': final_objective, 'gradient_infinity_norm': final_gradient_infinity_norm, 'original_step_verification': verification}
    exact_gates = {'native_status_zero': bool(result.success) and int(result.status) == 0, 'finite_objectives': bool(np.isfinite(initial_objective) and np.isfinite(lbfgs_objective) and np.isfinite(final_objective)), 'objective_nonincrease': bool(final_objective <= lbfgs_objective + 1e-12 and lbfgs_objective <= initial_objective), 'lbfgs_gradient_within_scientific_tolerance': bool(lbfgs_gradient_infinity_norm <= float(tol)), 'final_gradient_within_scientific_tolerance': bool(final_gradient_infinity_norm <= float(tol)), 'original_step_accepted': verification['accepted'] is True, 'original_convergence_rule': verification['passes_original_convergence_rule'] is True}
    termination['exact_gates'] = exact_gates
    if not all(exact_gates.values()):
        raise RidgeInitializationError('contractfix2 L-BFGS ridge initializer failed exact convergence gates', evidence=termination)
    selected_block_count = int(sum((value > 1e-08 for value in model.source_norms().values())))
    model.fit_diagnostics_ = adaptive_module.FitDiagnostics(n_iter=int(result.nit) + 1, converged=True, objective=float(final_objective), max_coef_change=float(verification['max_coef_or_intercept_change']), selected_block_count=selected_block_count, step_size=float(verification['next_proposed_step_size']), warning='')
    return (adaptive_weights, model, termination)
