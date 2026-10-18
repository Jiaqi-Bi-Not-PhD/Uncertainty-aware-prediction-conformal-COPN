"""Array interface to the numerical Q fit used in Supplement S5.

The caller builds each imputed design on its permitted training scope and
supplies weights without using target records. This module stores no fitted
state and does not select parameters using target outcomes.
"""
from .q_ridge import compute_q_adaptive_weights
from .q_solver_cache import InvariantCachedCandidateContextV1
from .q_group_kernel import ExactBatchedDotGroupKernelV2
from .q_softmax_solver import ExactSoftmaxWorkspaceV1, fit_candidate_exact_softmax_workspace_v1


def fit_selected_q_model(X_list, y, groups, *, sample_weight, gamma, alpha,
                         lambda_fraction, max_iter=500, tol=1e-5):
    """Refit a training-selected MAR MissC candidate from its ridge initializer.

    Returns the fitted model and numerical ridge diagnostics. Candidate
    selection must already exclude the records that will receive Q values.
    """
    from keyed_model8_adaptive_enet import model8_adaptive_enet as adaptive

    weights, ridge, diagnostics = compute_q_adaptive_weights(
        X_list, y, groups, sample_weight=sample_weight, gamma=gamma,
        max_iter=max_iter, tol=tol, adaptive_module=adaptive,
    )
    context = InvariantCachedCandidateContextV1(
        adaptive_module=adaptive, X_list=X_list, y=y, groups=groups,
        sample_weight=sample_weight, adaptive_weights=weights,
        gamma=gamma, max_iter=max_iter, tol=tol,
    )
    kernel = ExactBatchedDotGroupKernelV2(
        context.groups, context._template.adaptive_weights,
    )
    workspace = ExactSoftmaxWorkspaceV1(
        context.X_list, context.y, context.sample_weight,
        class_count=len(adaptive.CLASS_NAMES),
    )
    model = context.new_candidate_model(
        alpha=float(alpha),
        lambda_=context.lambda_max(alpha=float(alpha)) * float(lambda_fraction),
        gamma=float(gamma),
    )
    fit_candidate_exact_softmax_workspace_v1(
        model, context.X_list, context.y, group_kernel=kernel,
        sample_weight=context.sample_weight, init_intercept=ridge.intercept_,
        init_coef=ridge.coef_, initial_step=context.initial_step_for(model),
        workspace=workspace,
    )
    return model, diagnostics
