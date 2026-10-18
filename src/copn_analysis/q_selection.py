"""Training-only Q candidate ranking from Supplement S5.

Each supplied row must pool only validation segments allowed for the target
scope. Exclude that target from selection, fitting, imputation and preprocessing.
"""
import numpy as np

def select_q_candidate(normalized):
    eligible_indices = [index for index, row in enumerate(normalized) if row["pair_eligible"]]
    if not eligible_indices:
        raise ValueError("No Q candidate converged in every required pair fit.")
    fields = ("unweighted_balanced_accuracy", "AP_tie_exact_average_precision", "unweighted_multiclass_log_loss")
    if any(not np.isfinite([normalized[index][field] for field in fields]).all() for index in eligible_indices):
        raise ValueError("Q candidate metrics must be finite.")
    selected_index = min(
        eligible_indices,
        key=lambda index: (
            -normalized[index]["unweighted_balanced_accuracy"],
            -normalized[index]["AP_tie_exact_average_precision"],
            normalized[index]["unweighted_multiclass_log_loss"],
            normalized[index]["candidate_registry_position"],
        ),
    )
    return dict(normalized[selected_index])
