"""Scientific routines used by the manuscript; unused workflow/report code omitted."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from keyed_model1.ap_tuning import ap_weighted_row_weights
from keyed_model1.common import CLASS_NAMES, clean_key, exclusion_reason, file_hash, is_observed_value, normalize_text, parse_level, read_dictionary, resolve_dictionary_record, source_id_for, _prepare_sheet
from .config_model8_mnar import Model8MNARConfig as Model8Config
from .mnar_adaptive_enet import AdaptiveMNARMIGroupedMultinomialElasticNet, compute_adaptive_weights_mnar, constant_proba, lambda_max_for_mnar_adaptive_groups
from .mnar_imputation import MNARFoldDesign as Model8FoldDesign
from .mnar_outcome_model import fit_outcome_observation_model, mnar_missco_ipw_weights
from .mnar_sensitivity_grid import Model8MNARScenario as Model8Scenario, scenario_registry
from .shared import model8_root

@dataclass
class OutcomeKeyedDataset:
    X: pd.DataFrame
    keys: pd.Series
    y_raw: pd.Series
    y_code: np.ndarray
    c_observed: np.ndarray
    metadata: pd.DataFrame
    audit: pd.DataFrame
    workbook_hash: str
    excluded_outcomes: pd.DataFrame

def assemble_outcome_keyed_dataset(config: Model8Config, output_dir: Path | None=None) -> OutcomeKeyedDataset:
    workbook = Path(resolve_model8_path(config.workbook_path))
    dictionary_path = Path(resolve_model8_path(config.data_dictionary_path))
    dictionary = read_dictionary(dictionary_path if dictionary_path.exists() else None)
    lookup: dict[str, list[dict[str, Any]]] = {}
    if not dictionary.empty:
        from keyed_model1.common import dictionary_lookup
        lookup = dictionary_lookup(dictionary)
    xl = pd.ExcelFile(workbook)
    required = config.required_sheets or xl.sheet_names
    if config.outcome_sheet not in xl.sheet_names:
        raise ValueError(f"Workbook is missing outcome sheet '{config.outcome_sheet}'.")
    outcome_raw = pd.read_excel(workbook, sheet_name=config.outcome_sheet, dtype=object)
    outcome, outcome_audit = _prepare_sheet(outcome_raw, config.key_col, config.outcome_sheet)
    if config.outcome_col not in outcome.columns:
        raise ValueError(f"Outcomes sheet is missing '{config.outcome_col}'.")
    keys = outcome[config.key_col].map(clean_key).reset_index(drop=True)
    y_raw = outcome[config.outcome_col].reset_index(drop=True)
    y_mapped = y_raw.map(config.outcome_map)
    c_observed = y_mapped.notna().to_numpy(dtype=int)
    y_code = y_mapped.fillna(-1).astype(int).to_numpy()
    audit_rows = []
    outcome_audit['matched_modeled_keys'] = int(c_observed.sum())
    outcome_audit['outcome_keys_not_present_in_sheet'] = 0
    outcome_audit['sheet_keys_not_present_in_outcomes'] = 0
    audit_rows.append(outcome_audit)
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    existing_sources: set[str] = set()
    predictor_sheets = config.predictor_sheets or [sheet for sheet in required if sheet != config.outcome_sheet]
    outcome_keys = set(keys)
    modeled_keys = set(keys[c_observed.astype(bool)])
    for sheet in predictor_sheets:
        raw = pd.read_excel(workbook, sheet_name=sheet, dtype=object)
        prepared, sheet_audit = _prepare_sheet(raw, config.key_col, sheet)
        sheet_keys = set(prepared[config.key_col])
        sheet_audit['matched_modeled_keys'] = int(len(modeled_keys.intersection(sheet_keys)))
        sheet_audit['outcome_keys_not_present_in_sheet'] = int(len(outcome_keys.difference(sheet_keys)))
        sheet_audit['sheet_keys_not_present_in_outcomes'] = int(len(sheet_keys.difference(outcome_keys)))
        audit_rows.append(sheet_audit)
        prepared = prepared.set_index(config.key_col)
        source_columns = [col for col in prepared.columns if col != config.key_col]
        renamed: dict[Any, str] = {}
        for raw_col in source_columns:
            dict_record, match_method, match_score, candidates = resolve_dictionary_record(dictionary, lookup, str(raw_col), sheet)
            source_id = source_id_for(str(raw_col), existing_sources, dict_record)
            renamed[raw_col] = source_id
            dictionary_form_name = dict_record.get('form_name') if dict_record else None
            group = normalize_text(dictionary_form_name) if dict_record and is_observed_value(dictionary_form_name) else normalize_text(sheet)
            level = parse_level(dict_record.get('implementation_level')) if dict_record else None
            level_source = 'source_book'
            if level is None:
                level = int(config.group_level_defaults.get(group, 4))
                level_source = 'sheet_default_fallback'
            exclude = exclusion_reason(source_id, sheet, str(raw_col), dict_record, config.exclude_patterns)
            top_candidate = candidates[0] if candidates else {}
            metadata_rows.append({'source_id': source_id, 'source_sheet': sheet, 'source_group': group, 'dictionary_form_name': dictionary_form_name, 'raw_column': str(raw_col), 'raw_header': str(raw_col), 'field_name': None if not dict_record else dict_record.get('field_name'), 'field_type': None if not dict_record else dict_record.get('field_type'), 'role': None if not dict_record else dict_record.get('role'), 'implementation_type': None if not dict_record else dict_record.get('implementation_type'), 'implementation_level': int(level), 'implementation_level_source': level_source, 'dictionary_match_method': match_method, 'dictionary_match_score': float(match_score), 'closest_source_field_name': top_candidate.get('candidate_field_name', ''), 'closest_source_form_name': top_candidate.get('candidate_form_name', ''), 'closest_source_field_label': top_candidate.get('candidate_field_label', ''), 'closest_source_implementation_level': top_candidate.get('candidate_implementation_level', ''), 'closest_source_match_score': top_candidate.get('candidate_match_score', ''), 'joined_tab_provenance': sheet if 'joined' in normalize_text(sheet) else '', 'excluded': exclude is not None, 'exclude_reason': exclude or '', 'harmonization_rule': 'identity', 'conflict_rows': 0})
        frames.append(prepared[source_columns].rename(columns=renamed).reindex(keys))
    X = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=range(len(keys)))
    X.index = pd.RangeIndex(len(X))
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        observed = X.apply(lambda s: s.map(is_observed_value).mean())
        metadata['observed_fraction'] = metadata['source_id'].map(observed).astype(float)
    excluded = pd.DataFrame({config.key_col: keys[c_observed == 0], config.outcome_col: y_raw[c_observed == 0]})
    excluded['exclude_reason'] = 'missing_or_unrecognized_outcome'
    audit = pd.DataFrame(audit_rows)
    return OutcomeKeyedDataset(X=X, keys=keys, y_raw=y_raw, y_code=y_code, c_observed=c_observed, metadata=metadata, audit=audit, workbook_hash=file_hash(workbook), excluded_outcomes=excluded)

def resolve_model8_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    candidates = [model8_root() / path, model8_root().parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()

def _branch_training_weights(scenario: Model8Scenario, y_train: np.ndarray, design: Model8FoldDesign, X_train_observed: pd.DataFrame, missco_missing_X: pd.DataFrame, config: Model8Config) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    if scenario.branch == 'mnar_missc':
        return (ap_weighted_row_weights(y_train, atypical_multiplier=scenario.atypical_weight_multiplier, n_classes=len(CLASS_NAMES)), {'branch': 'mnar_missc', 'n_eff_overall': float(len(y_train)), 'n_eff_PD': float(np.sum(y_train == 0)), 'n_eff_AP': float(np.sum(y_train == 1)), 'n_eff_HC': float(np.sum(y_train == 2)), 'max_row_weight': float(np.nan), 'truncation_rate': float(np.nan)}, pd.DataFrame())
    if scenario.branch != 'mnar_missco_ipw':
        raise ValueError(f'Unsupported Model 8 branch: {scenario.branch}')
    source_ids = list(design.groups.keys())
    X_all = pd.concat([X_train_observed[source_ids], missco_missing_X.reindex(columns=source_ids)], ignore_index=True, sort=False)
    c_all = np.concatenate([np.ones(len(X_train_observed), dtype=int), np.zeros(len(missco_missing_X), dtype=int)])
    obs_model = fit_outcome_observation_model(X_all, c_all, source_ids, l2=config.outcome_model_l2, max_iter=config.outcome_model_max_iter, tol=config.outcome_model_tol)
    rho_train = obs_model.predict_proba(X_train_observed[source_ids])
    weights, diagnostics = mnar_missco_ipw_weights(y_train, rho_train, floor=float(scenario.rho_floor or 0.05), atypical_weight_multiplier=scenario.atypical_weight_multiplier, zeta_pd=scenario.zeta_PD, zeta_ap=scenario.zeta_AP, zeta_hc=scenario.zeta_HC)
    coef = pd.DataFrame(obs_model.coefficient_rows())
    return (weights, diagnostics, coef)

def _fit_model8(design: Model8FoldDesign, y_train: np.ndarray, scenario: Model8Scenario, sample_weight: np.ndarray, config: Model8Config) -> dict[str, Any]:
    if not design.groups or not design.train_mi or design.train_mi[0].shape[1] == 0:
        return {'model': None, 'proba': constant_proba(y_train, design.valid_mi[0].shape[0] if design.valid_mi else 0, sample_weight), 'lambda': np.nan, 'lambda_max': np.nan, 'adaptive_weights': {}, 'source_norms': {}, 'fit_diagnostics': {}}
    adaptive_weights, initial = compute_adaptive_weights_mnar(design.train_mi, y_train, design.groups, sample_weight=sample_weight, gamma=scenario.gamma, max_iter=max(50, min(config.max_iter, 250)), tol=config.tol, imputation_weights=design.train_imputation_weights)
    lambda_max = lambda_max_for_mnar_adaptive_groups(design.train_mi, y_train, design.groups, alpha=scenario.alpha, adaptive_weights=adaptive_weights, sample_weight=sample_weight, imputation_weights=design.train_imputation_weights)
    lambda_value = max(float(lambda_max) * float(scenario.lambda_fraction), 1e-08)
    model = AdaptiveMNARMIGroupedMultinomialElasticNet(alpha=scenario.alpha, lambda_=lambda_value, gamma=scenario.gamma, groups=design.groups, adaptive_weights=adaptive_weights, max_iter=config.max_iter, tol=config.tol).fit(design.train_mi, y_train, sample_weight=sample_weight, init_intercept=initial.intercept_, init_coef=initial.coef_, imputation_weights=design.train_imputation_weights)
    proba = model.predict_proba_mi_weighted(design.valid_mi, design.valid_imputation_weights)
    return {'model': model, 'proba': proba, 'lambda': float(lambda_value), 'lambda_max': float(lambda_max), 'adaptive_weights': adaptive_weights, 'source_norms': model.source_norms(), 'fit_diagnostics': model.fit_diagnostics_.__dict__ if model.fit_diagnostics_ is not None else {}}
