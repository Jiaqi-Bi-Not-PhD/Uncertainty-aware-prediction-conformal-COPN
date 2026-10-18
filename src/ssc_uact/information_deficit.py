"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

import io

import json

import math

import re

import zipfile

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Callable, Mapping, Sequence

import numpy as np

import pandas as pd

from .uncertainty import sanitize_unit_interval

class InformationDeficitError(ValueError):
    """Raised when deficit inputs violate the frozen keyed contract."""

PREDICTION_ROLES = ("proper_fit_oof", "conformal_cal", "outer_test")

CANONICAL_STATES = (
    "observed",
    "unknown",
    "not_applicable",
    "not_administered",
    "structural",
    "artifact",
)

FROZEN_STATE_DEFICITS: Mapping[str, float] = {
    "observed": 0.0,
    "unknown": 1.0,
    "not_applicable": 0.5,
    "not_administered": 1.0,
    "structural": 0.5,
    "artifact": 1.0,
}

COMMON_NULL_TOKENS = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "missing", "."}
)

FROZEN_EXPLICIT_STATE_TOKENS: Mapping[str, str] = {
    "not applicable": "not_applicable",
    "not applicable sans objet": "not_applicable",
    "not applicable sans object": "not_applicable",
    "sans objet": "not_applicable",
    "__not_applicable__": "not_applicable",
    "__not_administered__": "not_administered",
    "__structural__": "structural",
    "__artifact__": "artifact",
    "__unknown__": "unknown",
}

@dataclass(frozen=True)
class M1ReplayNorms:
    """Validated Model 1 norm maps and participant roles from one replay ZIP."""

    archive: Path
    outer_split_id: str
    budget: int
    inner_source_norms: Mapping[str, Mapping[str, float]]
    final_source_norms: Mapping[str, float]
    roles: pd.DataFrame
    metadata: Mapping[str, Any]

def _normalized_token(value: str) -> str:
    token = value.strip().lower().replace("_", " ")
    token = re.sub(r"[/|]+", " ", token)
    token = re.sub(r"\s+", " ", token)
    return token

def common_is_observed_value(value: Any) -> bool:
    """Frozen equivalent of the keyed-model ``common.is_observed_value``."""
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return not (
        isinstance(value, str) and value.strip().lower() in COMMON_NULL_TOKENS
    )

def missing_state_from_value(
    value: Any,
    *,
    is_observed_value: Callable[[Any], bool] = common_is_observed_value,
) -> str:
    """Classify one raw keyed-dataset value without inventing structure.

    Explicit frozen non-applicability/sentinel values are recognized first.
    Otherwise the common observed-value parser is authoritative; every
    unobserved value defaults to ``unknown``.
    """
    if isinstance(value, str):
        explicit = FROZEN_EXPLICIT_STATE_TOKENS.get(value.strip().lower())
        if explicit is None:
            explicit = FROZEN_EXPLICIT_STATE_TOKENS.get(_normalized_token(value))
        if explicit is not None:
            return explicit
    return "observed" if bool(is_observed_value(value)) else "unknown"

def _coerce_bool(series: pd.Series, *, inverse: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        result = series.fillna(False).astype(bool)
    else:
        tokens = series.fillna(False).astype(str).str.strip().str.lower()
        valid = {"true", "1", "yes", "y", "false", "0", "no", "n"}
        bad = ~tokens.isin(valid)
        if bad.any():
            raise InformationDeficitError(
                f"invalid approval/exclusion value {series.loc[bad].iloc[0]!r}"
            )
        result = tokens.isin({"true", "1", "yes", "y"})
    return ~result if inverse else result

def eligible_sources(source_metadata: pd.DataFrame, budget: int) -> list[str]:
    """Select globally approved sources whose frozen level is within J3/J4."""
    if int(budget) not in {3, 4}:
        raise InformationDeficitError("information deficit is defined only for J3/J4")
    required = {"source_id", "implementation_level"}
    missing = required.difference(source_metadata.columns)
    if missing:
        raise InformationDeficitError(
            f"source metadata lacks columns {sorted(missing)}"
        )
    frame = source_metadata.copy()
    frame["source_id"] = frame["source_id"].astype(str).str.strip()
    if frame["source_id"].eq("").any() or frame["source_id"].duplicated().any():
        raise InformationDeficitError("source metadata IDs must be nonblank and unique")
    if "approved" in frame:
        approved = _coerce_bool(frame["approved"])
    elif "excluded" in frame:
        approved = _coerce_bool(frame["excluded"], inverse=True)
    else:
        raise InformationDeficitError(
            "source metadata must explicitly contain approved or excluded"
        )
    levels = pd.to_numeric(frame["implementation_level"], errors="coerce")
    selected = frame.loc[approved & levels.notna() & levels.le(int(budget)), "source_id"]
    if selected.empty:
        raise InformationDeficitError(f"no approved sources are eligible for J{budget}")
    return selected.tolist()

def _validated_norm_map(value: Any, identity: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise InformationDeficitError(f"{identity} source_norms is missing or not a map")
    result: dict[str, float] = {}
    for source, raw_norm in value.items():
        source_id = str(source).strip()
        if not source_id:
            raise InformationDeficitError(f"{identity} contains a blank source ID")
        try:
            norm = float(raw_norm)
        except (TypeError, ValueError) as exc:
            raise InformationDeficitError(
                f"{identity} has a nonnumeric norm for {source_id}"
            ) from exc
        if not math.isfinite(norm) or norm < 0:
            raise InformationDeficitError(
                f"{identity} has invalid norm {norm!r} for {source_id}"
            )
        result[source_id] = norm
    return result

def _read_csv_member(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise InformationDeficitError(f"replay archive lacks {member}") from exc
    return pd.read_csv(io.BytesIO(payload))

def read_m1_replay_archive(path: str | Path) -> M1ReplayNorms:
    """Extract and validate ``metadata.json`` and keyed roles from a replay ZIP."""
    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise InformationDeficitError(
                    f"{archive_path} failed ZIP integrity at {corrupt}"
                )
            try:
                metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            except KeyError as exc:
                raise InformationDeficitError("replay archive lacks metadata.json") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InformationDeficitError("metadata.json is invalid") from exc
            oof = _read_csv_member(archive, "properfit_oof_predictions.csv")
            heldout = _read_csv_member(archive, "heldout_predictions.csv")
    except zipfile.BadZipFile as exc:
        raise InformationDeficitError(f"{archive_path} is not a valid ZIP") from exc

    if metadata.get("artifact_type") != "base_model_selected_replay":
        raise InformationDeficitError("archive is not a selected base-model replay")
    if str(metadata.get("model_id", "")).lower() != "model1":
        raise InformationDeficitError("information deficit requires a Model 1 replay")
    try:
        budget = int(metadata["budget"])
        outer_split_id = str(metadata["outer_split_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InformationDeficitError("metadata lacks a valid budget/outer_split_id") from exc
    if budget not in {3, 4} or not outer_split_id:
        raise InformationDeficitError("metadata has an invalid J3/J4 replay identity")

    role_required = {
        "project_key", "outer_split_id", "budget", "prediction_role", "inner_split_id"
    }
    for identity, frame in (("proper-fit OOF", oof), ("heldout", heldout)):
        missing = role_required.difference(frame.columns)
        if missing:
            raise InformationDeficitError(f"{identity} roles lack {sorted(missing)}")
        if set(frame["outer_split_id"].astype(str)) != {outer_split_id}:
            raise InformationDeficitError(f"{identity} outer split disagrees with metadata")
        if set(pd.to_numeric(frame["budget"], errors="coerce").dropna().astype(int)) != {
            budget
        }:
            raise InformationDeficitError(f"{identity} budget disagrees with metadata")
    if set(oof["prediction_role"].astype(str)) != {"proper_fit_oof"}:
        raise InformationDeficitError("OOF prediction roles do not match proper_fit_oof")
    if oof["inner_split_id"].isna().any() or oof["inner_split_id"].astype(str).eq("").any():
        raise InformationDeficitError("proper-fit OOF rows require inner_split_id")
    heldout_roles = set(heldout["prediction_role"].astype(str))
    if not heldout_roles or not heldout_roles.issubset({"conformal_cal", "outer_test"}):
        raise InformationDeficitError("heldout prediction roles are invalid")
    if heldout["inner_split_id"].notna().any():
        raise InformationDeficitError("heldout rows must not carry inner_split_id")

    roles = pd.concat(
        [
            oof.loc[:, ["project_key", "prediction_role", "inner_split_id"]],
            heldout.loc[:, ["project_key", "prediction_role", "inner_split_id"]],
        ],
        ignore_index=True,
    )
    roles["project_key"] = roles["project_key"].astype(str).str.strip()
    if roles["project_key"].eq("").any() or roles["project_key"].duplicated().any():
        raise InformationDeficitError("replay keys must be nonblank and unique")

    folds = metadata.get("fold_diagnostics")
    if not isinstance(folds, list) or not folds:
        raise InformationDeficitError("metadata lacks fold_diagnostics norm maps")
    inner_maps: dict[str, dict[str, float]] = {}
    for position, diagnostic in enumerate(folds):
        if not isinstance(diagnostic, Mapping) or "inner_split_id" not in diagnostic:
            raise InformationDeficitError(
                f"fold_diagnostics[{position}] lacks inner_split_id"
            )
        inner_id = str(diagnostic["inner_split_id"])
        if inner_id in inner_maps:
            raise InformationDeficitError(f"duplicate norm map for {inner_id}")
        inner_maps[inner_id] = _validated_norm_map(
            diagnostic.get("source_norms"), f"fold {inner_id}"
        )
    required_inner = set(oof["inner_split_id"].astype(str))
    if set(inner_maps) != required_inner:
        raise InformationDeficitError(
            "inner_split_id norm maps do not exactly match proper-fit OOF rows"
        )
    final = metadata.get("final_diagnostics")
    if not isinstance(final, Mapping):
        raise InformationDeficitError("metadata lacks final_diagnostics")
    final_map = _validated_norm_map(final.get("source_norms"), "final refit")
    return M1ReplayNorms(
        archive=archive_path.resolve(),
        outer_split_id=outer_split_id,
        budget=budget,
        inner_source_norms=inner_maps,
        final_source_norms=final_map,
        roles=roles,
        metadata=metadata,
    )

def _normalized_weights(
    norm_map: Mapping[str, float], approved_sources: Sequence[str], identity: str
) -> tuple[np.ndarray, bool, int]:
    approved = list(approved_sources)
    disallowed = sorted(set(norm_map).difference(approved))
    if disallowed:
        raise InformationDeficitError(
            f"{identity} norm map contains ineligible/unapproved sources {disallowed[:5]}"
        )
    norms = np.asarray([float(norm_map.get(source, 0.0)) for source in approved])
    if not np.isfinite(norms).all() or np.any(norms < 0):
        raise InformationDeficitError(f"{identity} norms must be finite and nonnegative")
    positive = int(np.count_nonzero(norms > 0))
    total = float(norms.sum())
    if total == 0.0:
        return np.full(len(approved), 1.0 / len(approved)), True, 0
    return norms / total, False, positive

def _state_frame(
    X: pd.DataFrame,
    sources: Sequence[str],
    *,
    missing_states: pd.DataFrame | None,
    is_observed_value: Callable[[Any], bool],
) -> pd.DataFrame:
    missing_columns = sorted(set(sources).difference(X.columns))
    if missing_columns:
        raise InformationDeficitError(
            f"keyed dataset lacks eligible source columns {missing_columns[:5]}"
        )
    if missing_states is None:
        return X.loc[:, sources].map(
            lambda value: missing_state_from_value(
                value, is_observed_value=is_observed_value
            )
        )
    state_missing = sorted(set(sources).difference(missing_states.columns))
    if state_missing or len(missing_states) != len(X):
        raise InformationDeficitError(
            "explicit missing-state matrix does not align with keyed dataset"
        )
    states = missing_states.loc[:, sources].copy()
    for source in sources:
        states[source] = states[source].astype(str).str.strip().str.lower()
    invalid = ~states.isin(CANONICAL_STATES)
    if invalid.to_numpy().any():
        row, column = np.argwhere(invalid.to_numpy())[0]
        raise InformationDeficitError(
            f"unknown missingness state {states.iat[row, column]!r} "
            f"for source {states.columns[column]}"
        )
    return states

def _validate_expected_roles(
    actual: pd.DataFrame, expected_roles: pd.DataFrame | None
) -> None:
    if expected_roles is None:
        return
    required = {"project_key", "prediction_role"}
    missing = required.difference(expected_roles.columns)
    if missing:
        raise InformationDeficitError(f"expected roles lack {sorted(missing)}")
    expected = expected_roles.copy()
    expected["project_key"] = expected["project_key"].astype(str).str.strip()
    if expected["project_key"].duplicated().any():
        raise InformationDeficitError("expected roles contain duplicate keys")
    merged = actual.merge(
        expected.loc[:, ["project_key", "prediction_role"]].rename(
            columns={"prediction_role": "expected_role"}
        ),
        on="project_key",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or not (
        merged["prediction_role"].astype(str) == merged["expected_role"].astype(str)
    ).all():
        raise InformationDeficitError("replay roles do not match expected keyed roles")

def compute_information_deficit(
    X: pd.DataFrame,
    project_keys: Sequence[Any] | pd.Series,
    source_metadata: pd.DataFrame,
    m1_replay_zip: str | Path,
    budget: int,
    *,
    missing_states: pd.DataFrame | None = None,
    is_observed_value: Callable[[Any], bool] = common_is_observed_value,
    expected_roles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute participant-level ``Q_i`` for one outer split and J3/J4.

    ``X`` and ``missing_states`` must use the actual keyed-dataset row order.
    Explicit missing states are preferred when available.  If they are absent,
    the frozen common-compatible parser is used and uncertain missingness is
    conservatively assigned ``unknown``.
    """
    requested_budget = int(budget)
    replay = read_m1_replay_archive(m1_replay_zip)
    if replay.budget != requested_budget:
        raise InformationDeficitError(
            f"requested J{requested_budget} but replay is J{replay.budget}"
        )
    keys = pd.Series(project_keys, dtype="object").astype(str).str.strip()
    if len(keys) != len(X) or keys.eq("").any() or keys.duplicated().any():
        raise InformationDeficitError(
            "project_keys must align one-to-one with keyed dataset rows"
        )
    if set(keys) != set(replay.roles["project_key"]):
        raise InformationDeficitError("replay keys do not match keyed dataset keys")
    _validate_expected_roles(replay.roles, expected_roles)

    sources = eligible_sources(source_metadata, requested_budget)
    states = _state_frame(
        X.reset_index(drop=True),
        sources,
        missing_states=(
            None if missing_states is None else missing_states.reset_index(drop=True)
        ),
        is_observed_value=is_observed_value,
    )
    deficits = states.replace(FROZEN_STATE_DEFICITS).astype(float)
    state_counts = {
        state: (states == state).sum(axis=1).to_numpy(int)
        for state in CANONICAL_STATES
    }
    key_to_row = {key: row for row, key in enumerate(keys)}
    output_rows: list[dict[str, Any]] = []
    for record in replay.roles.to_dict("records"):
        key = str(record["project_key"])
        role = str(record["prediction_role"])
        if role not in PREDICTION_ROLES:
            raise InformationDeficitError(f"unrecognized prediction role {role!r}")
        if role == "proper_fit_oof":
            inner_id = str(record["inner_split_id"])
            norm_map = replay.inner_source_norms.get(inner_id)
            if norm_map is None:
                raise InformationDeficitError(f"missing norm map for {inner_id}")
            norm_identity = inner_id
        else:
            inner_id = None
            norm_map = replay.final_source_norms
            norm_identity = "final_proper_fit_refit"
        weights, fallback, positive = _normalized_weights(
            norm_map, sources, norm_identity
        )
        row = key_to_row[key]
        try:
            score = float(
                sanitize_unit_interval(
                    np.dot(weights, deficits.iloc[row].to_numpy(float)),
                    n_terms=len(sources),
                    name="computed information deficit",
                )
            )
        except ValueError as exc:
            raise InformationDeficitError(
                "computed information deficit is outside [0, 1]"
            ) from exc
        diagnostic: dict[str, Any] = {
            "project_key": key,
            "prediction_role": role,
            "information_deficit": score,
            "inner_split_id": inner_id,
            "norm_map_id": norm_identity,
            "eligible_source_count": len(sources),
            "positive_norm_source_count": positive,
            "uniform_norm_fallback": fallback,
        }
        diagnostic.update(
            {
                f"{state}_source_count": int(counts[row])
                for state, counts in state_counts.items()
            }
        )
        output_rows.append(diagnostic)
    result = pd.DataFrame(output_rows)
    if result["information_deficit"].isna().any() or not result[
        "information_deficit"
    ].between(0.0, 1.0).all():
        raise InformationDeficitError("computed information deficit is outside [0, 1]")
    return result
