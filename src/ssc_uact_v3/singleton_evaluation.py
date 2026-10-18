"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from __future__ import annotations
from typing import Mapping
import numpy as np
from numpy.typing import NDArray
import pandas as pd
SIMPLE_SCORE_IDS = ('one_minus_max_primary_probability', 'one_minus_primary_top_two_margin', 'primary_normalized_entropy')
MULTICOMPONENT_SCORE_ID = 'multicomponent_six_feature_monotone'
from .reliability_statistics import PreparedMethod, _weighted_aurc_components, _weighted_auroc_components, _weighted_average_precision_components
MINIMUM_VALID_REPLICATES = 1900
ROW_SELECTED_SCORE_ID = 'rowwise_training_selected_simple_score'
SCORE_IDS = (MULTICOMPONENT_SCORE_ID, *SIMPLE_SCORE_IDS, ROW_SELECTED_SCORE_ID)
COMPARATOR_SCORE_IDS = (*SIMPLE_SCORE_IDS, ROW_SELECTED_SCORE_ID)
REPORTED_METRICS = ('aurc', 'error_brier', 'error_log_loss', 'error_detection_auroc', 'error_detection_auprc')
LOWER_IS_BETTER = {'aurc': True, 'error_brier': True, 'error_log_loss': True, 'error_detection_auroc': False, 'error_detection_auprc': False}
IDENTITY_COLUMNS = ('outer_split_id', 'repeat_id', 'outer_fold', 'budget', 'project_key')

def _build_method_frames(base: pd.DataFrame, risks: pd.DataFrame, keys: NDArray[np.str_]) -> dict[tuple[int, str], pd.DataFrame]:
    identity = list(IDENTITY_COLUMNS)
    base_required = set(identity) | {'true_code', 'A3_core_mask', 'A3_singleton_label', 'A3_singleton_error', 'ssi_raw'}
    risk_required = set(identity) | {'true_code', 'A3_singleton_label', 'A3_singleton_error', 'score_id', 'training_selected_simple', 'raw_ranking_risk', 'estimated_error_risk'}
    if (missing := base_required.difference(base.columns)):
        raise ValueError(f'Stage 4 outer base lacks {sorted(missing)}')
    if (missing := risk_required.difference(risks.columns)):
        raise ValueError(f'Stage 4 outer risks lack {sorted(missing)}')
    if base.empty or base.duplicated(identity).any():
        raise ValueError('Stage 4 outer-base identity grid is empty or duplicated')
    if set(base['budget'].astype(int)) != {3, 4}:
        raise ValueError('Stage 4 reliability addendum requires J3 and J4')
    for budget in (3, 4):
        if base.loc[base['budget'].eq(budget), 'outer_split_id'].nunique() != 15:
            raise ValueError(f'J{budget} does not contain all 15 outer folds')
    if not base['A3_core_mask'].isin([1, 2, 4]).all():
        raise ValueError('Stage 4 outer base contains a non-singleton A3 row')
    expected_label = np.log2(base['A3_core_mask'].to_numpy(np.uint8)).astype(np.int8)
    if not np.array_equal(expected_label, base['A3_singleton_label'].to_numpy(np.int8)):
        raise ValueError('A3 singleton labels disagree with the saved masks')
    expected_error = (base['A3_singleton_label'].to_numpy(np.int8) != base['true_code'].to_numpy(np.int8)).astype(np.int8)
    if not np.array_equal(expected_error, base['A3_singleton_error'].to_numpy(np.int8)):
        raise ValueError('A3 singleton error target is inconsistent')
    if not np.isfinite(base['ssi_raw'].to_numpy(float)).all():
        raise ValueError('saved SSI must be finite')
    source_score_ids = set(risks['score_id'].astype(str).unique())
    expected_source_ids = {MULTICOMPONENT_SCORE_ID, *SIMPLE_SCORE_IDS}
    expected_source_ids.add('one_minus_primary_probability_of_A3_singleton_label')
    if source_score_ids != expected_source_ids:
        raise ValueError('Stage 4 source score families differ from the frozen five')
    if risks.duplicated(identity + ['score_id']).any():
        raise ValueError('Stage 4 risk identity/score grid is duplicated')
    if len(risks) != 5 * len(base) or not (risks.groupby(identity).size() == 5).all():
        raise ValueError('every A3 singleton must have exactly five saved score rows')
    selected = risks[risks['training_selected_simple'].astype(bool)]
    if len(selected) != len(base):
        raise ValueError('every A3 singleton needs one training-selected simple score')
    if not set(selected['score_id'].astype(str)).issubset(SIMPLE_SCORE_IDS):
        raise ValueError('training selection marked a non-prespecified simple score')
    if selected.groupby(['outer_split_id', 'budget'])['score_id'].nunique().max() != 1:
        raise ValueError('training-selected simple family changes within an outer cell')
    key_index = {key: index for index, key in enumerate(keys)}
    if not set(base['project_key'].astype(str)).issubset(key_index):
        raise ValueError('Stage 4 outer base contains a key outside the frozen universe')
    common = base[identity + ['true_code', 'A3_singleton_label', 'A3_singleton_error', 'ssi_raw']].copy()
    common['cluster_index'] = common['project_key'].astype(str).map(key_index).astype(np.int64)
    frames: dict[tuple[int, str], pd.DataFrame] = {}
    selections: dict[str, pd.DataFrame] = {score_id: risks[risks['score_id'].eq(score_id)].copy() for score_id in (MULTICOMPONENT_SCORE_ID, *SIMPLE_SCORE_IDS)}
    selections[ROW_SELECTED_SCORE_ID] = selected.copy()
    risk_columns = identity + ['true_code', 'A3_singleton_label', 'A3_singleton_error', 'raw_ranking_risk', 'estimated_error_risk']
    for score_id, score_rows in selections.items():
        joined = common.merge(score_rows[risk_columns], on=identity, how='left', validate='one_to_one', suffixes=('', '_risk_check'))
        for column in ('true_code', 'A3_singleton_label', 'A3_singleton_error'):
            check = f'{column}_risk_check'
            if joined[check].isna().any() or not np.array_equal(joined[column].to_numpy(), joined[check].to_numpy()):
                raise ValueError(f'saved {score_id} metadata is misaligned')
            joined = joined.drop(columns=check)
        numeric = joined[['raw_ranking_risk', 'estimated_error_risk']].to_numpy(float)
        if not np.isfinite(numeric).all():
            raise ValueError(f'saved {score_id} risks are missing or non-finite')
        if np.any((numeric[:, 1] <= 0) | (numeric[:, 1] >= 1)):
            raise ValueError(f'saved {score_id} estimated risks must lie inside (0, 1)')
        for budget in (3, 4):
            frame = joined[joined['budget'].eq(budget)].sort_values(identity).reset_index(drop=True)
            frames[budget, score_id] = frame
    for budget in (3, 4):
        reference = frames[budget, MULTICOMPONENT_SCORE_ID][identity]
        for score_id in SCORE_IDS[1:]:
            if not reference.equals(frames[budget, score_id][identity]):
                raise AssertionError('reliability score rows are not exactly paired')
    return frames

def _weighted_score_metrics(prepared: PreparedMethod, cluster_weights: NDArray[np.int64], harmonic: NDArray[np.float64]) -> dict[str, float | int]:
    """Stage 4 metric algebra without its unused calibration optimization."""
    frame = prepared.frame
    errors = frame['A3_singleton_error'].to_numpy(np.int8)
    ranking = frame['ranking_risk'].to_numpy(float)
    estimated = np.clip(frame['estimated_risk'].to_numpy(float), 1e-12, 1 - 1e-12)
    weights = cluster_weights[prepared.cluster_indices].astype(np.int64, copy=False)
    total = int(weights.sum())
    if total <= 0:
        return _invalid_metrics()
    aurc_numerator = oracle_numerator = 0.0
    auroc_numerator = auroc_denominator = 0.0
    auprc_numerator = auprc_denominator = 0.0
    for fold, aurc_order, oracle_order, auprc_order in zip(prepared.folds, prepared.aurc_orders, prepared.oracle_orders, prepared.auprc_orders, strict=True):
        fold_errors = errors[fold]
        fold_weights = weights[fold]
        aurc_part, _ = _weighted_aurc_components(fold_errors[aurc_order], fold_weights[aurc_order], harmonic)
        oracle_part, _ = _weighted_aurc_components(fold_errors[oracle_order], fold_weights[oracle_order], harmonic)
        auroc_part, auroc_pairs = _weighted_auroc_components(fold_errors, ranking[fold], fold_weights)
        auprc_part, positives = _weighted_average_precision_components(fold_errors[auprc_order], fold_weights[auprc_order], harmonic)
        aurc_numerator += aurc_part
        oracle_numerator += oracle_part
        auroc_numerator += auroc_part
        auroc_denominator += auroc_pairs
        auprc_numerator += auprc_part
        auprc_denominator += positives
    aurc = aurc_numerator / total
    oracle = oracle_numerator / total
    loss = -(errors * np.log(estimated) + (1 - errors) * np.log(1 - estimated))
    return {'n': total, 'errors': int(weights @ errors.astype(np.int64)), 'error_rate': float(weights @ errors / total), 'aurc': float(aurc), 'oracle_aurc': float(oracle), 'excess_aurc': float(aurc - oracle), 'error_brier': float(weights @ (estimated - errors) ** 2 / total), 'error_log_loss': float(weights @ loss / total), 'error_detection_auroc': float(auroc_numerator / auroc_denominator) if auroc_denominator > 0 else np.nan, 'error_detection_auprc': float(auprc_numerator / auprc_denominator) if auprc_denominator > 0 else np.nan}

def _invalid_metrics() -> dict[str, float | int]:
    return {'n': 0, 'errors': 0, 'error_rate': np.nan, 'aurc': np.nan, 'oracle_aurc': np.nan, 'excess_aurc': np.nan, 'error_brier': np.nan, 'error_log_loss': np.nan, 'error_detection_auroc': np.nan, 'error_detection_auprc': np.nan}

def _observed_frame(values: Mapping[tuple[int, str], Mapping[str, float | int]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for budget in (3, 4):
        oracle = float(values[budget, MULTICOMPONENT_SCORE_ID]['oracle_aurc'])
        for score_id in SCORE_IDS:
            metric = values[budget, score_id]
            if not np.isclose(oracle, float(metric['oracle_aurc']), atol=1e-14, rtol=0):
                raise AssertionError('paired scores do not share the same oracle AURC')
            rows.append({'budget': budget, 'score_id': score_id, 'n': int(metric['n']), 'errors': int(metric['errors']), 'error_rate': float(metric['error_rate']), 'aurc': float(metric['aurc']), 'oracle_aurc': float(metric['oracle_aurc']), 'excess_aurc': float(metric['excess_aurc']), 'error_brier': float(metric['error_brier']), 'error_log_loss': float(metric['error_log_loss']), 'error_detection_auroc': float(metric['error_detection_auroc']), 'error_detection_auprc': float(metric['error_detection_auprc']), 'primary_metric': 'aurc', 'outer_score_reselection_performed': False})
    return pd.DataFrame(rows)

def _paired_bootstrap(prepared: Mapping[tuple[int, str], PreparedMethod], observed: Mapping[tuple[int, str], Mapping[str, float | int]], multiplicities: NDArray[np.int16], harmonic: NDArray[np.float64]) -> pd.DataFrame:
    replicate_count = len(multiplicities)
    delta: dict[tuple[int, str, str], NDArray[np.float64]] = {(budget, comparator, metric): np.full(replicate_count, np.nan) for budget in (3, 4) for comparator in COMPARATOR_SCORE_IDS for metric in REPORTED_METRICS}
    for replicate, cluster_weights16 in enumerate(multiplicities):
        cluster_weights = cluster_weights16.astype(np.int64, copy=False)
        for budget in (3, 4):
            multi = _weighted_score_metrics(prepared[budget, MULTICOMPONENT_SCORE_ID], cluster_weights, harmonic)
            for comparator in COMPARATOR_SCORE_IDS:
                simple = _weighted_score_metrics(prepared[budget, comparator], cluster_weights, harmonic)
                if not np.isclose(float(multi['oracle_aurc']), float(simple['oracle_aurc']), equal_nan=True, atol=1e-14, rtol=0):
                    raise AssertionError('paired bootstrap scores have different oracle AURC')
                for metric in REPORTED_METRICS:
                    delta[budget, comparator, metric][replicate] = float(multi[metric]) - float(simple[metric])
    rows: list[dict[str, object]] = []
    for budget in (3, 4):
        for comparator in COMPARATOR_SCORE_IDS:
            for metric in REPORTED_METRICS:
                values = delta[budget, comparator, metric]
                valid = values[np.isfinite(values)]
                quantiles = np.full(5, np.nan) if len(valid) == 0 else np.quantile(valid, [0.025, 0.05, 0.5, 0.95, 0.975], method='higher')
                estimate = float(observed[budget, MULTICOMPONENT_SCORE_ID][metric]) - float(observed[budget, comparator][metric])
                rows.append({'budget': budget, 'challenger': MULTICOMPONENT_SCORE_ID, 'comparator': comparator, 'contrast': 'multicomponent_minus_simple', 'metric': metric, 'estimate': estimate, 'bootstrap_mean': float(valid.mean()) if len(valid) else np.nan, 'lower_95': float(quantiles[0]), 'lower_one_sided_95': float(quantiles[1]), 'bootstrap_median': float(quantiles[2]), 'upper_one_sided_95': float(quantiles[3]), 'upper_95': float(quantiles[4]), 'valid_replicates': int(len(valid)), 'minimum_valid_replicates': MINIMUM_VALID_REPLICATES, 'inferentially_usable': bool(len(valid) >= MINIMUM_VALID_REPLICATES), 'lower_is_better': LOWER_IS_BETTER[metric], 'metric_role': 'primary_descriptive_addendum' if metric == 'aurc' else 'supporting_non_overriding', 'may_reverse_completed_stage4_decision': False})
    return pd.DataFrame(rows)
