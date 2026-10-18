"""Scientific routines used by the manuscript; unused workflow/report code omitted."""
from __future__ import annotations
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
KEY_COL = 'Project key'
OUTCOME_SHEET = 'Outcomes'
OUTCOME_COL = 'FINAL DISEASE DIAGNOSIS'
OUTCOME_MAP = {'Parkinson': 0, 'Atypical': 1, 'Healthy': 2}
CLASS_NAMES = ('Parkinson', 'Atypical', 'Healthy')
NULL_TOKENS = {'', 'na', 'n/a', 'nan', 'none', 'null', 'missing', '.'}
DEFAULT_GROUP_LEVELS = {'enrollement': 1, 'enrollment': 1, 'demographic': 1, 'epidemiological': 1, 'bdii': 1, 'bai': 1, 'ehi': 1, 'fatigue_severity_scale': 1, 'parkinson_severity_scale': 1, 'pdq_8': 1, 'pdq_39': 1, 'schwab_england': 2, 'schwab_and_england': 2, 'scopa': 1, 'apathy_scale': 1, 'apathy_evaluation_self': 1, 'apathy_evaluation_informant': 1, 'frsbe': 2, 'clinical': 2, 'medication': 2, 'mbic_joined': 2, 'mbic': 2, 'moca_joined': 3, 'moca': 3, 'mds_updrs_joined': 3, 'mds_updrs': 3, 'updrs_1_2_part_3': 3, 'timed_up_go': 3, 'neuropsychological_joined': 4, 'neuropsychological': 4}
DEFAULT_EXCLUDE_PATTERNS = ('\\bproject_key\\b', '\\bevent_name\\b', '\\bcomplete\\b', '\\bunnamed\\b', '\\bfinal_disease_diagnosis\\b', '\\bspecific_disease\\b', '\\bdiagnosis\\b', '\\bdiagnostic\\b', '\\bpd_plus\\b', '\\batypical_parkinsonism\\b', '\\bhealthy_control\\b', '\\bsite\\b', '\\brecord_id\\b', '\\bparticipant_id\\b', '\\bsubject_id\\b', '\\bredcap\\b', '\\bcoordinator\\b', '\\bpersonnel\\b', '\\bdate\\b', '\\btimestamp\\b')

@dataclass(frozen=True)
class KeyedConfig:
    workbook_path: str = ''
    data_dictionary_path: str = ''
    output_dir: str = ''
    key_col: str = KEY_COL
    outcome_sheet: str = OUTCOME_SHEET
    outcome_col: str = OUTCOME_COL
    outcome_map: dict[str, int] = field(default_factory=lambda: dict(OUTCOME_MAP))
    required_sheets: list[str] | None = None
    predictor_sheets: list[str] | None = None
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    group_level_defaults: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_GROUP_LEVELS))
    budgets: tuple[int, ...] = (1, 2, 3, 4)
    min_observed_fraction: float = 0.02
    min_observed_per_class: dict[str, int] = field(default_factory=lambda: {'Parkinson': 10, 'Atypical': 1, 'Healthy': 5})
    rare_category_min_count: int = 5
    max_categories: int = 40
    numeric_parse_threshold: float = 0.9
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    quasi_constant_threshold: float = 0.995
    outer_folds: int = 5
    outer_repeats: int = 3
    inner_folds: int = 4
    random_state: int = 20260421
    min_ap_in_outer_test: int = 5
    min_ap_in_inner_valid: int = 1
    parallel_jobs: int = 1

def package_root() -> Path:
    return Path(__file__).resolve().parents[1]

def resolve_path(path_like: str | Path, base_dir: Path | None=None) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.extend([Path.cwd() / path, package_root() / path, package_root().parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()

def file_hash(path: Path, block_size: int=1024 * 1024) -> str:
    hasher = hashlib.blake2b(digest_size=20)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            hasher.update(block)
    return hasher.hexdigest()

def normalize_text(value: Any) -> str:
    text = '' if value is None else str(value)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()
    text = re.sub('\\s+', ' ', text)
    text = re.sub('[^a-z0-9]+', '_', text)
    return re.sub('_+', '_', text).strip('_')

def clean_key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    token = str(value).strip()
    if token.lower() in NULL_TOKENS:
        return None
    return token

def is_observed_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return False
    return True

def clean_category(value: Any, missing_token: str='__MISSING__') -> str:
    if not is_observed_value(value):
        return missing_token
    token = str(value).strip()
    return token if token else missing_token

def parse_level(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) and int(value) in {1, 2, 3, 4}:
        return int(value)
    match = re.search('([1-4])', str(value))
    return int(match.group(1)) if match else None

def parse_numeric(value: Any) -> float:
    if not is_observed_value(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and (not isinstance(value, bool)):
        return float(value)
    token = str(value).strip()
    lowered = token.lower()
    ordinal_map = {'checked': 1.0, 'unchecked': 0.0, 'yes': 1.0, 'no': 0.0, 'true': 1.0, 'false': 0.0, 'present': 1.0, 'absent': 0.0}
    if lowered in ordinal_map:
        return ordinal_map[lowered]
    leading = re.match('^\\s*([-+]?\\d+(?:\\.\\d+)?)\\s*(?:-|:|=|$)', token)
    if leading:
        return float(leading.group(1))
    if re.fullmatch('[-+]?\\d+(?:\\.\\d+)?', token):
        return float(token)
    return np.nan

def numeric_series(series: pd.Series) -> pd.Series:
    return series.map(parse_numeric).astype(float)

def key_case_conflicts(keys: Iterable[str | None]) -> list[dict[str, Any]]:
    buckets: dict[str, set[str]] = {}
    for key in keys:
        if key is None:
            continue
        buckets.setdefault(key.lower(), set()).add(key)
    return [{'casefold_key': folded, 'variants': sorted(variants)} for folded, variants in buckets.items() if len(variants) > 1]

def read_dictionary(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, dtype=object)
    keep = {}
    for column in df.columns:
        keep[normalize_text(column)] = column
    rename = {keep.get('variable_field_name'): 'field_name', keep.get('form_name'): 'form_name', keep.get('field_label'): 'field_label', keep.get('field_type'): 'field_type', keep.get('role'): 'role', keep.get('implementation_type'): 'implementation_type'}
    rename = {src: dst for src, dst in rename.items() if src is not None}
    out = df.rename(columns=rename)
    for col in ['field_name', 'form_name', 'field_label', 'field_type', 'role', 'implementation_type']:
        if col not in out.columns:
            out[col] = np.nan
    out['field_label_norm'] = out['field_label'].map(normalize_text)
    out['field_name_norm'] = out['field_name'].map(normalize_text)
    out['form_name_norm'] = out['form_name'].map(normalize_text)
    out['implementation_level'] = out['implementation_type'].map(parse_level)
    return out[['field_name', 'form_name', 'field_label', 'field_type', 'role', 'implementation_type', 'implementation_level', 'field_label_norm', 'field_name_norm', 'form_name_norm']].copy()

def dictionary_lookup(dictionary: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    if dictionary.empty:
        return lookup
    for _, row in dictionary.iterrows():
        record = row.to_dict()
        row_keys = set()
        for key_col in ('field_label_norm', 'field_name_norm'):
            key = record.get(key_col)
            if isinstance(key, str) and key and (key not in row_keys):
                lookup.setdefault(key, []).append(record)
                row_keys.add(key)
    return lookup

def _form_match_bonus(sheet_norm: str, form_norm: str) -> float:
    if not sheet_norm or not form_norm:
        return 0.0
    if sheet_norm == form_norm:
        return 0.06
    if sheet_norm in form_norm or form_norm in sheet_norm:
        return 0.04
    return 0.0

def _token_similarity(left: str, right: str) -> float:
    left_tokens = {token for token in left.split('_') if token}
    right_tokens = {token for token in right.split('_') if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

def dictionary_match_score(raw_column_norm: str, sheet_norm: str, record: dict[str, Any]) -> float:
    candidates = [str(record.get('field_label_norm', '') or ''), str(record.get('field_name_norm', '') or '')]
    scores = []
    for candidate in candidates:
        if not raw_column_norm or not candidate:
            continue
        seq = SequenceMatcher(None, raw_column_norm, candidate).ratio()
        tok = _token_similarity(raw_column_norm, candidate)
        containment = 0.0
        if raw_column_norm in candidate or candidate in raw_column_norm:
            containment = min(len(raw_column_norm), len(candidate)) / max(len(raw_column_norm), len(candidate))
        scores.append(max(seq, tok, containment))
    score = max(scores) if scores else 0.0
    score = min(1.0, score + _form_match_bonus(sheet_norm, str(record.get('form_name_norm', '') or '')))
    return float(score)

def closest_dictionary_matches(dictionary: pd.DataFrame, raw_column: str, sheet_name: str, limit: int=5) -> list[dict[str, Any]]:
    if dictionary.empty:
        return []
    raw_norm = normalize_text(raw_column)
    sheet_norm = normalize_text(sheet_name)
    rows = []
    for idx, row in dictionary.iterrows():
        record = row.to_dict()
        level = parse_level(record.get('implementation_level'))
        if level is None:
            continue
        score = dictionary_match_score(raw_norm, sheet_norm, record)
        rows.append({'candidate_dictionary_index': idx, 'candidate_field_name': record.get('field_name'), 'candidate_form_name': record.get('form_name'), 'candidate_field_label': record.get('field_label'), 'candidate_implementation_type': record.get('implementation_type'), 'candidate_implementation_level': level, 'candidate_match_score': score})
    rows.sort(key=lambda item: float(item['candidate_match_score']), reverse=True)
    return rows[:limit]

def dictionary_record_by_index(dictionary: pd.DataFrame, index: Any) -> dict[str, Any] | None:
    if index is None or dictionary.empty:
        return None
    if index in dictionary.index:
        return dictionary.loc[index].to_dict()
    return None

def resolve_dictionary_record(dictionary: pd.DataFrame, lookup: dict[str, list[dict[str, Any]]], raw_column: str, sheet_name: str, fuzzy_threshold: float=0.82) -> tuple[dict[str, Any] | None, str, float, list[dict[str, Any]]]:
    raw_norm = normalize_text(raw_column)
    sheet_norm = normalize_text(sheet_name)
    exact_matches = lookup.get(raw_norm, [])
    if exact_matches:
        exact_matches = sorted(exact_matches, key=lambda record: (parse_level(record.get('implementation_level')) is not None, _form_match_bonus(sheet_norm, str(record.get('form_name_norm', '') or ''))), reverse=True)
        return (exact_matches[0], 'exact', 1.0, [])
    candidates = closest_dictionary_matches(dictionary, raw_column, sheet_name)
    if candidates and float(candidates[0]['candidate_match_score']) >= fuzzy_threshold:
        record = dictionary_record_by_index(dictionary, candidates[0].get('candidate_dictionary_index'))
        if record:
            return (record, 'fuzzy', float(candidates[0]['candidate_match_score']), candidates)
    score = float(candidates[0]['candidate_match_score']) if candidates else 0.0
    return (None, 'fallback_group_default', score, candidates)

def source_id_for(raw_column: str, existing: set[str], dict_record: dict[str, Any] | None=None) -> str:
    if dict_record and is_observed_value(dict_record.get('field_name')):
        base = normalize_text(dict_record.get('field_name'))
    else:
        base = normalize_text(raw_column)
    if not base:
        base = 'source'
    source_id = base
    i = 2
    while source_id in existing:
        source_id = f'{base}_{i}'
        i += 1
    existing.add(source_id)
    return source_id

def exclusion_reason(source_id: str, sheet_name: str, raw_column: str, dict_record: dict[str, Any] | None, exclude_patterns: Iterable[str]) -> str | None:
    if dict_record:
        role = str(dict_record.get('role', '')).strip().lower()
        if role and role not in {'predictor', 'nan'}:
            return f'dictionary_role:{role}'
    text = normalize_text(f'{sheet_name} {raw_column} {source_id}')
    for pattern in exclude_patterns:
        if re.search(pattern, text):
            return f'pattern:{pattern}'
    return None

def collapse_duplicate_keys(df: pd.DataFrame, key_col: str, sheet_name: str) -> tuple[pd.DataFrame, int]:
    if not df.duplicated(key_col, keep=False).any():
        return (df, 0)
    rows = []
    conflicts = 0
    for key, sub in df.groupby(key_col, sort=False, dropna=False):
        row = {key_col: key}
        for col in df.columns:
            if col == key_col:
                continue
            observed = sub[col].loc[sub[col].map(is_observed_value)]
            if observed.empty:
                row[col] = np.nan
                continue
            values = [str(v) for v in observed]
            if len(set(values)) > 1:
                conflicts += 1
            row[col] = observed.iloc[0]
        rows.append(row)
    if conflicts:
        raise ValueError(f"Sheet '{sheet_name}' has duplicate Project key rows with {conflicts} conflicting fields.")
    return (pd.DataFrame(rows, columns=df.columns), 0)

@dataclass
class KeyedDataset:
    X: pd.DataFrame
    y: np.ndarray
    y_raw: pd.Series
    keys: pd.Series
    metadata: pd.DataFrame
    audit: pd.DataFrame
    outcome_counts: dict[str, int]
    excluded_outcomes: pd.DataFrame
    workbook_hash: str

def _prepare_sheet(df: pd.DataFrame, key_col: str, sheet_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if key_col not in df.columns:
        raise ValueError(f"Required key column '{key_col}' is absent from sheet '{sheet_name}'.")
    raw_rows = int(len(df))
    fully_blank = df.isna().all(axis=1)
    df = df.loc[~fully_blank].copy()
    df[key_col] = df[key_col].map(clean_key)
    case_conflicts = key_case_conflicts(df[key_col])
    non_key_cols = [c for c in df.columns if c != key_col]
    blank_key = df[key_col].isna()
    keyless_nonblank = int((blank_key & df[non_key_cols].map(is_observed_value).any(axis=1)).sum())
    df = df.loc[df[key_col].notna()].copy()
    duplicate_key_rows = int(df[key_col].duplicated(keep=False).sum())
    audit = {'sheet': sheet_name, 'raw_rows': raw_rows, 'fully_blank_rows_dropped': int(fully_blank.sum()), 'blank_key_nonblank_rows': keyless_nonblank, 'nonblank_keys': int(len(df)), 'duplicate_key_rows': duplicate_key_rows, 'case_only_key_conflicts': len(case_conflicts)}
    if case_conflicts:
        raise ValueError(f"Sheet '{sheet_name}' has case-only Project key conflicts: {case_conflicts[:5]}")
    if keyless_nonblank:
        raise ValueError(f"Sheet '{sheet_name}' has {keyless_nonblank} nonblank rows without Project key.")
    df, duplicate_conflicts = collapse_duplicate_keys(df, key_col, sheet_name)
    audit['duplicate_conflicting_fields'] = int(duplicate_conflicts)
    return (df, audit)

def assemble_keyed_dataset(config: KeyedConfig, output_dir: Path | None=None) -> KeyedDataset:
    workbook = resolve_path(config.workbook_path, package_root())
    dictionary_path = resolve_path(config.data_dictionary_path, package_root())
    dictionary = read_dictionary(dictionary_path if dictionary_path.exists() else None)
    lookup = dictionary_lookup(dictionary)
    xl = pd.ExcelFile(workbook)
    required = config.required_sheets or xl.sheet_names
    missing = sorted(set(required).difference(xl.sheet_names))
    if missing:
        raise ValueError(f'Workbook is missing required sheets: {missing}')
    if config.outcome_sheet not in xl.sheet_names:
        raise ValueError(f"Workbook is missing outcome sheet '{config.outcome_sheet}'.")
    outcome_raw = pd.read_excel(workbook, sheet_name=config.outcome_sheet, dtype=object)
    outcome, outcome_audit = _prepare_sheet(outcome_raw, config.key_col, config.outcome_sheet)
    if config.outcome_col not in outcome.columns:
        raise ValueError(f"Outcomes sheet is missing '{config.outcome_col}'.")
    if outcome[config.key_col].duplicated().any():
        raise ValueError('Outcomes.Project key is nonunique.')
    valid_mask = outcome[config.outcome_col].isin(config.outcome_map)
    excluded = outcome.loc[~valid_mask, [config.key_col, config.outcome_col]].copy()
    excluded['exclude_reason'] = 'missing_or_unrecognized_outcome'
    modeled = outcome.loc[valid_mask, [config.key_col, config.outcome_col]].copy()
    keys = modeled[config.key_col].reset_index(drop=True)
    y_raw = modeled[config.outcome_col].reset_index(drop=True)
    y = y_raw.map(config.outcome_map).astype(int).to_numpy()
    outcome_keys = set(outcome[config.key_col])
    modeled_keys = set(keys)
    audit_rows = []
    outcome_audit['matched_modeled_keys'] = int(len(modeled_keys))
    outcome_audit['outcome_keys_not_present_in_sheet'] = 0
    outcome_audit['sheet_keys_not_present_in_outcomes'] = 0
    audit_rows.append(outcome_audit)
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    existing_sources: set[str] = set()
    predictor_sheets = config.predictor_sheets or [s for s in required if s != config.outcome_sheet]
    for sheet in predictor_sheets:
        raw = pd.read_excel(workbook, sheet_name=sheet, dtype=object)
        prepared, sheet_audit = _prepare_sheet(raw, config.key_col, sheet)
        sheet_keys = set(prepared[config.key_col])
        sheet_audit['matched_modeled_keys'] = int(len(modeled_keys.intersection(sheet_keys)))
        sheet_audit['outcome_keys_not_present_in_sheet'] = int(len(outcome_keys.difference(sheet_keys)))
        sheet_audit['sheet_keys_not_present_in_outcomes'] = int(len(sheet_keys.difference(outcome_keys)))
        audit_rows.append(sheet_audit)
        if prepared[config.key_col].duplicated().any():
            raise ValueError(f"Sheet '{sheet}' has duplicate keys after validation.")
        prepared = prepared.set_index(config.key_col)
        source_columns = [c for c in prepared.columns if c != config.key_col]
        renamed: dict[Any, str] = {}
        for raw_col in source_columns:
            dict_record, match_method, match_score, candidates = resolve_dictionary_record(dictionary, lookup, str(raw_col), sheet)
            source_id = source_id_for(str(raw_col), existing_sources, dict_record)
            renamed[raw_col] = source_id
            dictionary_form_name = dict_record.get('form_name') if dict_record else None
            group = normalize_text(dictionary_form_name) if dict_record and is_observed_value(dictionary_form_name) else normalize_text(sheet)
            level = parse_level(dict_record.get('implementation_level')) if dict_record else None
            implementation_level_source = 'source_book'
            if level is None:
                level = int(config.group_level_defaults.get(group, 4))
                implementation_level_source = 'sheet_default_fallback'
            exclude = exclusion_reason(source_id, sheet, str(raw_col), dict_record, config.exclude_patterns)
            top_candidate = candidates[0] if candidates else {}
            metadata_rows.append({'source_id': source_id, 'source_sheet': sheet, 'source_group': group, 'dictionary_form_name': dictionary_form_name, 'raw_column': str(raw_col), 'raw_header': str(raw_col), 'field_name': None if not dict_record else dict_record.get('field_name'), 'field_type': None if not dict_record else dict_record.get('field_type'), 'role': None if not dict_record else dict_record.get('role'), 'implementation_type': None if not dict_record else dict_record.get('implementation_type'), 'implementation_level': int(level), 'implementation_level_source': implementation_level_source, 'dictionary_match_method': match_method, 'dictionary_match_score': float(match_score), 'closest_source_field_name': top_candidate.get('candidate_field_name', ''), 'closest_source_form_name': top_candidate.get('candidate_form_name', ''), 'closest_source_field_label': top_candidate.get('candidate_field_label', ''), 'closest_source_implementation_level': top_candidate.get('candidate_implementation_level', ''), 'closest_source_match_score': top_candidate.get('candidate_match_score', ''), 'joined_tab_provenance': sheet if 'joined' in normalize_text(sheet) else '', 'excluded': exclude is not None, 'exclude_reason': exclude or '', 'harmonization_rule': 'identity', 'conflict_rows': 0})
        frames.append(prepared[source_columns].rename(columns=renamed).reindex(keys))
    X = pd.concat(frames, axis=1)
    X.index = pd.RangeIndex(len(X))
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        observed = X.apply(lambda s: s.map(is_observed_value).mean())
        metadata['observed_fraction'] = metadata['source_id'].map(observed).astype(float)
    outcome_counts = y_raw.value_counts().reindex(config.outcome_map).fillna(0).astype(int).to_dict()
    audit = pd.DataFrame(audit_rows)
    return KeyedDataset(X=X, y=y, y_raw=y_raw, keys=keys, metadata=metadata, audit=audit, outcome_counts=outcome_counts, excluded_outcomes=excluded, workbook_hash=file_hash(workbook))

def class_weights(y: np.ndarray, n_classes: int=3) -> dict[int, float]:
    weights = {}
    n = len(y)
    for k in range(n_classes):
        count = int(np.sum(y == k))
        if count <= 0:
            raise ValueError(f'Training split has no samples for class code {k}.')
        weights[k] = n / (n_classes * count)
    return weights

def row_weights(y: np.ndarray, n_classes: int=3) -> np.ndarray:
    weights = class_weights(y, n_classes=n_classes)
    return np.array([weights[int(label)] for label in y], dtype=float)

def make_repeated_stratified_splits(y: np.ndarray, n_splits: int, n_repeats: int, random_state: int, min_ap_in_test: int=1) -> list[dict[str, Any]]:
    y = np.asarray(y)
    splits = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(random_state + repeat * 1009)
        fold_bins = [[] for _ in range(n_splits)]
        for klass in sorted(np.unique(y)):
            idx = np.flatnonzero(y == klass)
            rng.shuffle(idx)
            for pos, sample_idx in enumerate(idx):
                fold_bins[pos % n_splits].append(int(sample_idx))
        all_idx = np.arange(len(y))
        for fold, test_list in enumerate(fold_bins):
            test_idx = np.array(sorted(test_list), dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_idx[train_mask]
            ap_count = int(np.sum(y[test_idx] == OUTCOME_MAP['Atypical']))
            if ap_count < min_ap_in_test:
                raise ValueError(f'Split repeat={repeat} fold={fold} has only {ap_count} Atypical samples.')
            splits.append({'repeat': repeat, 'fold': fold, 'split_id': f'r{repeat:02d}_f{fold:02d}', 'train_idx': train_idx, 'test_idx': test_idx})
    return splits

def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int=3) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix

def balanced_accuracy_from_confusion(matrix: np.ndarray) -> float:
    recalls = []
    for k in range(matrix.shape[0]):
        denom = matrix[k, :].sum()
        recalls.append(matrix[k, k] / denom if denom else np.nan)
    return float(np.nanmean(recalls))

def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    scores = []
    for k in range(matrix.shape[0]):
        tp = matrix[k, k]
        fp = matrix[:, k].sum() - tp
        fn = matrix[k, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))

def multiclass_log_loss(y_true: np.ndarray, proba: np.ndarray, eps: float=1e-15) -> float:
    clipped = np.clip(proba, eps, 1.0 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true.astype(int)])))

def metric_row(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    matrix = confusion_matrix_counts(y_true, pred, n_classes=proba.shape[1])
    row: dict[str, Any] = {'n': int(len(y_true)), 'balanced_accuracy': balanced_accuracy_from_confusion(matrix), 'macro_f1': macro_f1_from_confusion(matrix), 'log_loss': multiclass_log_loss(y_true, proba)}
    for code, label in enumerate(CLASS_NAMES):
        denom = matrix[code, :].sum()
        row[f'recall_{normalize_text(label)}'] = float(matrix[code, code] / denom) if denom else np.nan
    return row
