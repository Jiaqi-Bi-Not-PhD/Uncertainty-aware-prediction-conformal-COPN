"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from . import q_gradient as _probability

@dataclass(frozen=True)
class ExactSoftmaxAccountingV1:
    candidate_fit_calls: int
    candidate_iterations: int
    objective_evaluations: int
    accepted_first_try_evaluations: int
    backtracking_evaluations: int
    line_search_failures: int

class ExactSoftmaxWorkspaceV1:
    """Reusable arrays bound to one block's immutable MI design and weights."""

    def __init__(self, X_list: Sequence[np.ndarray], y: np.ndarray, sample_weight: np.ndarray, *, class_count: int) -> None:
        if not X_list:
            raise ValueError('X_list must contain at least one imputed design matrix.')
        self._bound_X_list = X_list
        self._bound_y = y
        self._bound_sample_weight = sample_weight
        self.arrays = tuple((np.asarray(X, dtype=float) for X in X_list))
        self.labels = np.asarray(y, dtype=int)
        rows, columns = self.arrays[0].shape
        if columns == 0:
            raise ValueError('Cannot fit adaptive Model 8 with zero design columns.')
        if any((X.shape != (rows, columns) for X in self.arrays)):
            raise ValueError('All imputed design matrices must have the same shape.')
        if rows != len(self.labels):
            raise ValueError('Design rows do not match y length.')
        self.weights = np.asarray(sample_weight, dtype=float)
        if self.weights.shape != self.labels.shape or np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0):
            raise ValueError('Sample weights must be finite and positive.')
        self.class_count = int(class_count)
        if self.class_count < 2:
            raise ValueError('At least two outcome classes are required.')
        self.nonreference_count = self.class_count - 1
        self.row_index = np.arange(rows)
        self.weight_sum = float(self.weights.sum())
        self.y_non_ref = np.zeros((rows, self.nonreference_count), dtype=np.float64)
        for klass in range(self.nonreference_count):
            self.y_non_ref[:, klass] = self.labels == klass
        self.probabilities = tuple((np.empty((rows, self.class_count), dtype=np.float64) for _ in self.arrays))
        self.linear = np.empty((rows, self.nonreference_count), dtype=np.float64)
        self.row_maximum = np.empty(rows, dtype=np.float64)
        self.row_sum = np.empty(rows, dtype=np.float64)
        self.nll = np.empty(rows, dtype=np.float64)
        self._candidate_fit_calls = 0
        self._candidate_iterations = 0
        self._objective_evaluations = 0
        self._accepted_first_try_evaluations = 0
        self._backtracking_evaluations = 0
        self._line_search_failures = 0

    def matches_identity(self, X_list: Sequence[np.ndarray], y: np.ndarray, sample_weight: np.ndarray | None, *, class_count: int) -> bool:
        return int(class_count) == self.class_count and X_list is self._bound_X_list and (y is self._bound_y) and (sample_weight is self._bound_sample_weight)

    def _probability_into(self, X: np.ndarray, intercept: np.ndarray, coefficient: np.ndarray, out: np.ndarray) -> None:
        np.matmul(X, coefficient, out=self.linear)
        np.add(self.linear, intercept, out=self.linear)
        out[:, :self.nonreference_count] = self.linear
        out[:, self.nonreference_count] = 0.0
        np.max(out, axis=1, out=self.row_maximum)
        np.subtract(out, self.row_maximum[:, None], out=out)
        np.exp(out, out=out)
        np.sum(out, axis=1, out=self.row_sum)
        np.divide(out, self.row_sum[:, None], out=out)

    def objective_and_probabilities(self, model: Any, intercept: np.ndarray, coefficient: np.ndarray, group_kernel: Any) -> tuple[float, tuple[np.ndarray, ...]]:
        self._objective_evaluations += 1
        loss = 0.0
        for X, probability in zip(self.arrays, self.probabilities, strict=True):
            self._probability_into(X, intercept, coefficient, probability)
            self.nll[:] = probability[self.row_index, self.labels]
            np.clip(self.nll, 1e-15, 1.0, out=self.nll)
            np.log(self.nll, out=self.nll)
            np.negative(self.nll, out=self.nll)
            loss += float(np.dot(self.weights, self.nll) / self.weight_sum)
        loss /= len(self.arrays)
        ridge = 0.5 * (1.0 - model.alpha) * float(np.sum(coefficient * coefficient))
        group = 0.0
        if model.alpha > 0:
            group = float(group_kernel.group_penalty(coefficient))
        return (loss + model.lambda_ * (ridge + model.alpha * group), self.probabilities)

    def accounting(self) -> ExactSoftmaxAccountingV1:
        return ExactSoftmaxAccountingV1(candidate_fit_calls=self._candidate_fit_calls, candidate_iterations=self._candidate_iterations, objective_evaluations=self._objective_evaluations, accepted_first_try_evaluations=self._accepted_first_try_evaluations, backtracking_evaluations=self._backtracking_evaluations, line_search_failures=self._line_search_failures)

def fit_candidate_exact_softmax_workspace_v1(model: Any, X_list: Sequence[np.ndarray], y: np.ndarray, *, group_kernel: Any, sample_weight: np.ndarray | None=None, init_intercept: np.ndarray | None=None, init_coef: np.ndarray | None=None, initial_step: float | None=None, workspace: ExactSoftmaxWorkspaceV1 | None=None) -> Any:
    """Run the frozen candidate loop with only softmax buffers reused."""
    group_kernel.validate_bound_model(model)
    original_globals = model.fit.__func__.__globals__
    class_count = len(original_globals['CLASS_NAMES'])
    if workspace is not None:
        if not workspace.matches_identity(X_list, y, sample_weight, class_count=class_count):
            raise ValueError('Softmax workspace input identity differs.')
        exact_workspace = workspace
        arrays = exact_workspace.arrays
        labels = exact_workspace.labels
        weights = exact_workspace.weights
        p = arrays[0].shape[1]
    else:
        if not X_list:
            raise ValueError('X_list must contain at least one imputed design matrix.')
        labels = np.asarray(y, dtype=int)
        arrays = tuple((np.asarray(X, dtype=float) for X in X_list))
        p = arrays[0].shape[1]
        if p == 0:
            raise ValueError('Cannot fit adaptive Model 8 with zero design columns.')
        if any((X.shape != arrays[0].shape for X in arrays)):
            raise ValueError('All imputed design matrices must have the same shape.')
        if arrays[0].shape[0] != len(labels):
            raise ValueError('Design rows do not match y length.')
        weights = original_globals['row_weights'](labels, n_classes=class_count) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError('Sample weights must be finite and positive.')
        exact_workspace = ExactSoftmaxWorkspaceV1(arrays, labels, weights, class_count=class_count)
    exact_workspace._candidate_fit_calls += 1
    coefficient = np.zeros((p, class_count - 1), dtype=float) if init_coef is None else np.asarray(init_coef, dtype=float).copy()
    intercept = model._intercept_only(labels, weights, class_count) if init_intercept is None else np.asarray(init_intercept, dtype=float).copy()
    previous, probabilities = exact_workspace.objective_and_probabilities(model, intercept, coefficient, group_kernel)
    step = model._initial_step(list(arrays), weights) if initial_step is None else float(initial_step)
    if not np.isfinite(step) or step <= 0 or step > 1:
        raise ValueError('Initial candidate step must be finite and in (0, 1].')
    converged = False
    max_change = np.inf
    warning = ''
    iteration = 0
    for iteration in range(1, model.max_iter + 1):
        grad_intercept, grad_coefficient = _probability._gradient_from_probabilities_v1(model, arrays, probabilities, weights, exact_workspace.y_non_ref, exact_workspace.weight_sum, coefficient)
        local_step = step
        accepted = False
        candidate_intercept = intercept
        candidate_coefficient = coefficient
        candidate_probabilities = probabilities
        current = previous
        line_search_evaluations = 0
        for _ in range(40):
            line_search_evaluations += 1
            candidate_intercept = intercept - local_step * grad_intercept
            candidate_coefficient = group_kernel.prox(model, coefficient - local_step * grad_coefficient, local_step)
            current, candidate_probabilities = exact_workspace.objective_and_probabilities(model, candidate_intercept, candidate_coefficient, group_kernel)
            if np.isfinite(current) and current <= previous + 1e-10:
                accepted = True
                break
            local_step *= 0.5
        exact_workspace._backtracking_evaluations += max(0, line_search_evaluations - 1)
        if not accepted:
            exact_workspace._line_search_failures += 1
            warning = 'line_search_not_accepted'
            break
        if line_search_evaluations == 1:
            exact_workspace._accepted_first_try_evaluations += 1
        max_change = float(max(np.max(np.abs(candidate_coefficient - coefficient)), np.max(np.abs(candidate_intercept - intercept))))
        rel_obj = abs(previous - current) / max(1.0, abs(previous))
        intercept, coefficient, previous, step = (candidate_intercept, candidate_coefficient, current, min(local_step * 1.05, 1.0))
        probabilities = candidate_probabilities
        if rel_obj < model.tol and max_change < max(1e-05, model.tol * 10):
            converged = True
            break
    exact_workspace._candidate_iterations += int(iteration)
    model.intercept_ = intercept
    model.coef_ = coefficient
    diagnostics_class = original_globals['FitDiagnostics']
    model.fit_diagnostics_ = diagnostics_class(n_iter=int(iteration), converged=bool(converged), objective=float(previous), max_coef_change=float(max_change if np.isfinite(max_change) else np.nan), selected_block_count=group_kernel.selected_block_count(coefficient), step_size=float(step), warning=warning)
    return model
