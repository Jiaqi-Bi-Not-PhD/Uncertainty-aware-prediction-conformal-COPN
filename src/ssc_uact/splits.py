"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import hashlib

from typing import Any

import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .data_audit import OUTCOME_CODES, canonicalize_outcome

class SplitManifestError(ValueError):
    """Raised when a split manifest leaks or violates the frozen contract."""

ROLE_PROPER = "proper_fit"

ROLE_CONFORMAL = "conformal_calibration"

ROLE_TEST = "outer_test"

ROLES = {ROLE_PROPER, ROLE_CONFORMAL, ROLE_TEST}

def split_manifest_hash(frame: pd.DataFrame) -> str:
    columns = [
        "project_key",
        "true_label",
        "true_code",
        "repeat_id",
        "outer_fold",
        "split_id",
        "role",
        "inner_fold",
    ]
    payload = frame[columns].sort_values(columns).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def make_split_manifest(
    participants: pd.DataFrame,
    *,
    key_column: str = "Project key",
    outcome_column: str = "Outcome",
    seed: int = 20260726,
    outer_repeats: int = 3,
    outer_folds: int = 5,
    proper_fit_fraction: float = 0.75,
    inner_folds: int = 4,
    minimum_ap_conformal_count: int = 19,
) -> pd.DataFrame:
    """Create the deterministic common manifest used by every model/scenario."""
    required = {key_column, outcome_column}
    if not required.issubset(participants.columns):
        raise SplitManifestError(f"participants lack columns: {sorted(required-participants.columns)}")
    frame = participants[[key_column, outcome_column]].copy()
    if frame[key_column].isna().any():
        raise SplitManifestError("participant keys may not be blank")
    frame["project_key"] = frame[key_column].astype(str).str.strip()
    if (frame["project_key"] == "").any() or frame["project_key"].duplicated().any():
        raise SplitManifestError("participant keys must be nonblank and unique")
    frame["true_label"] = frame[outcome_column].map(canonicalize_outcome)
    frame = frame.loc[frame["true_label"].notna(), ["project_key", "true_label"]]
    frame["true_code"] = frame["true_label"].map(OUTCOME_CODES).astype(int)
    frame = frame.sort_values("project_key", kind="stable").reset_index(drop=True)
    keys = frame["project_key"].to_numpy()
    labels = frame["true_label"].to_numpy()
    rows: list[dict[str, Any]] = []
    all_indices = np.arange(len(frame))

    for repeat_id in range(outer_repeats):
        outer = StratifiedKFold(
            n_splits=outer_folds, shuffle=True, random_state=seed + repeat_id
        )
        for outer_fold, (outer_train, outer_test) in enumerate(outer.split(keys, labels)):
            split_id = f"r{repeat_id:02d}_f{outer_fold:02d}"
            train_labels = labels[outer_train]
            selected: tuple[np.ndarray, np.ndarray] | None = None
            for attempt in range(100):
                chooser = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=proper_fit_fraction,
                    random_state=seed + repeat_id * 10000 + outer_fold * 100 + attempt,
                )
                proper_local, conformal_local = next(
                    chooser.split(np.zeros(len(outer_train)), train_labels)
                )
                if int(np.sum(train_labels[conformal_local] == "AP")) >= minimum_ap_conformal_count:
                    selected = (proper_local, conformal_local)
                    break
            if selected is None:
                raise SplitManifestError(
                    f"{split_id} cannot achieve AP conformal count >= "
                    f"{minimum_ap_conformal_count}"
                )
            proper = outer_train[selected[0]]
            conformal = outer_train[selected[1]]
            inner_assignment = np.full(len(frame), -1, dtype=int)
            inner = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=seed + 1_000_000 + repeat_id * 100 + outer_fold,
            )
            for inner_fold, (_, validation_local) in enumerate(
                inner.split(keys[proper], labels[proper])
            ):
                inner_assignment[proper[validation_local]] = inner_fold
            roles = np.full(len(frame), "", dtype=object)
            roles[proper] = ROLE_PROPER
            roles[conformal] = ROLE_CONFORMAL
            roles[outer_test] = ROLE_TEST
            if np.any(roles == ""):
                raise SplitManifestError(f"{split_id} did not assign every participant")
            for index in all_indices:
                rows.append(
                    {
                        "project_key": keys[index],
                        "true_label": labels[index],
                        "true_code": int(frame.iloc[index]["true_code"]),
                        "repeat_id": repeat_id,
                        "outer_fold": outer_fold,
                        "split_id": split_id,
                        "role": roles[index],
                        "inner_fold": (
                            int(inner_assignment[index]) if roles[index] == ROLE_PROPER else pd.NA
                        ),
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest["inner_fold"] = manifest["inner_fold"].astype("Int64")
    validate_split_manifest(
        manifest,
        outer_repeats=outer_repeats,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        minimum_ap_conformal_count=minimum_ap_conformal_count,
    )
    manifest["split_hash"] = split_manifest_hash(manifest)
    return manifest

def validate_split_manifest(
    manifest: pd.DataFrame,
    *,
    outer_repeats: int = 3,
    outer_folds: int = 5,
    inner_folds: int = 4,
    minimum_ap_conformal_count: int = 19,
) -> None:
    required = {
        "project_key", "true_label", "true_code", "repeat_id", "outer_fold",
        "split_id", "role", "inner_fold",
    }
    if not required.issubset(manifest.columns):
        raise SplitManifestError(f"manifest lacks columns: {sorted(required-manifest.columns)}")
    if not set(manifest["role"].unique()).issubset(ROLES):
        raise SplitManifestError("manifest contains an unknown role")
    if manifest.duplicated(["split_id", "project_key"]).any():
        raise SplitManifestError("a participant occurs more than once in an outer split")
    expected_splits = outer_repeats * outer_folds
    if manifest["split_id"].nunique() != expected_splits:
        raise SplitManifestError(f"expected {expected_splits} outer splits")
    counts = manifest.groupby("split_id")["project_key"].nunique()
    if counts.nunique() != 1:
        raise SplitManifestError("outer splits do not cover identical participant counts")
    test = manifest.loc[manifest["role"] == ROLE_TEST]
    per_repeat = test.groupby(["repeat_id", "project_key"]).size()
    if not (per_repeat == 1).all():
        raise SplitManifestError("each participant must be outer-test exactly once per repeat")
    for split_id, group in manifest.groupby("split_id", sort=False):
        if set(group["role"]) != ROLES:
            raise SplitManifestError(f"{split_id} does not contain all three roles")
        ap_count = int(
            ((group["role"] == ROLE_CONFORMAL) & (group["true_label"] == "AP")).sum()
        )
        if ap_count < minimum_ap_conformal_count:
            raise SplitManifestError(f"{split_id} has only {ap_count} conformal AP cases")
        proper = group.loc[group["role"] == ROLE_PROPER]
        if proper["inner_fold"].isna().any():
            raise SplitManifestError(f"{split_id} has missing inner assignments")
        if set(proper["inner_fold"].astype(int)) != set(range(inner_folds)):
            raise SplitManifestError(f"{split_id} does not use all inner folds")
        if group.loc[group["role"] != ROLE_PROPER, "inner_fold"].notna().any():
            raise SplitManifestError(f"{split_id} leaks inner assignments outside proper-fit")
