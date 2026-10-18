"""Scientific routines for the reported analysis; private inputs are supplied by the caller."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time
import math
import re
from typing import Any, Mapping
import unicodedata
STATE_ORDER = ('observed', 'not_applicable', 'structural', 'unknown', 'not_administered', 'artifact')
DICTIONARY_LABEL_STATES: Mapping[str, frozenset[str]] = {'not_applicable': frozenset({'not applicable', 'not applicable/sans object', 'not applicable/sans objet', '6: not applicable.', 'sans objet'}), 'not_administered': frozenset({'not administered', 'not administered/pas administré', '997 - test (or subtest) not administered, not in the study protocol at the time', '998- test (or subtest) not administered, not enough time during the day of assessment'}), 'unknown': frozenset({'5 - unable to rate', '5: unable to rate (ur).', 'did not answer (data missing)', 'incertain/uncertain', 'missing data', 'uncertain/incertain', 'unknown', 'unknown/inconnu', 'unsure/incertain'}), 'artifact': frozenset({'artifact', '__artifact__'}), 'structural': frozenset({'structural', '__structural__'})}
RAW_EXPLICIT_STATE_SENTINELS = {'not_applicable': 'not_applicable', '__not_applicable__': 'not_applicable', 'structural': 'structural', '__structural__': 'structural', 'not_administered': 'not_administered', '__not_administered__': 'not_administered', 'artifact': 'artifact', '__artifact__': 'artifact', 'unknown': 'unknown', '__unknown__': 'unknown'}
RAW_UNKNOWN_TOKENS = frozenset({'', '.', 'na', 'n/a', 'nan', 'none', 'null', 'missing'})
BRANCH_FIELD_RE = re.compile('\\[([A-Za-z0-9_]+)(?:\\([^\\]]+\\))?\\]')
CHOICE_RE = re.compile('^\\s*([^,]+?)\\s*,\\s*(.*?)\\s*$')

class QStateError(RuntimeError):
    """Raised when the state registry cannot be built without ambiguity."""

@dataclass(frozen=True)
class QStateAssessment:
    """One deterministic canonical-state classification."""
    state: str
    normalized_value: str
    canonical_choice_code: str | None
    canonical_choice_label: str | None
    value_rule: str
    branch_status: str

def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    text = unicodedata.normalize('NFC', str(value)).replace('\xa0', ' ')
    return ' '.join(text.strip().split()).casefold()

def _display_text(value: Any) -> str:
    if value is None:
        return ''
    text = unicodedata.normalize('NFC', str(value)).replace('\xa0', ' ')
    return ' '.join(text.strip().split())

def parse_choice_map(value: Any) -> dict[str, str]:
    """Parse an exact REDCap ``code, label | ...`` choice specification."""
    text = _display_text(value)
    if not text:
        return {}
    result: dict[str, str] = {}
    normalized_labels: dict[str, str] = {}
    for part in text.split('|'):
        match = CHOICE_RE.fullmatch(part)
        if match is None:
            raise QStateError(f'malformed dictionary choice: {part!r}')
        code = _display_text(match.group(1))
        label = _display_text(match.group(2))
        label_key = normalize_text(label)
        if not code or not label or code in result:
            raise QStateError('dictionary choice codes are blank or duplicated')
        if label_key in normalized_labels and normalized_labels[label_key] != code:
            raise QStateError('dictionary choice label is ambiguous across codes')
        result[code] = label
        normalized_labels[label_key] = code
    return result

def branch_fields(expression: Any) -> tuple[str, ...]:
    text = _display_text(expression)
    if not text:
        return ()
    fields = tuple(dict.fromkeys(BRANCH_FIELD_RE.findall(text)))
    if not fields:
        raise QStateError('branch expression has no recognized controller')
    return fields

def canonical_state_for_dictionary_label(label: Any) -> str | None:
    """Return a state only for an exact, prospectively enumerated whole label."""
    normalized = normalize_text(label)
    matches = [state for state, labels in DICTIONARY_LABEL_STATES.items() if normalized in labels]
    if len(matches) > 1:
        raise QStateError('dictionary state label has conflicting semantics')
    return matches[0] if matches else None

def _choice_code(value: Any) -> str:
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return _display_text(value)

def decode_tagged_scalar(value: Any) -> Any:
    """Decode the lossless scalar tags used by the bound predictor matrix."""
    if value is None:
        return None
    text = str(value)
    tag, separator, payload = text.partition(':')
    if not separator:
        raise QStateError('tagged predictor scalar is invalid')
    try:
        if tag == 's':
            return payload
        if tag == 'b' and payload in {'0', '1'}:
            return payload == '1'
        if tag == 'i':
            return int(payload)
        if tag == 'f':
            parsed = float.fromhex(payload)
            if not math.isfinite(parsed):
                raise QStateError('decoded predictor is non-finite')
            return parsed
        if tag == 'dt':
            return datetime.fromisoformat(payload)
        if tag == 'd':
            return date.fromisoformat(payload)
        if tag == 't':
            return time.fromisoformat(payload)
    except (TypeError, ValueError) as exc:
        raise QStateError('tagged predictor scalar payload is invalid') from exc
    raise QStateError('tagged predictor scalar type is invalid')

def _branch_status(value: bool | str | None) -> str:
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if value is None:
        return 'not_present'
    normalized = normalize_text(value)
    if normalized in {'true', 'false', 'unresolved', 'not_present'}:
        return normalized
    raise QStateError(f'invalid branch status: {value!r}')

def classify_q_state(value: Any, *, choice_code_to_label: Mapping[Any, Any] | None=None, branch_status: bool | str | None=None, explicit_state: str | None=None, value_is_tagged: bool=False) -> QStateAssessment:
    """Classify one value under the approved deterministic precedence.

    Source-specific choice resolution occurs before generic observed-value
    handling.  A false branch turns an otherwise unknown value into
    ``structural``; an unresolved branch never does.  Any observed-value versus
    explicit-missing or false-branch contradiction fails closed.
    """
    raw = decode_tagged_scalar(value) if value_is_tagged else value
    normalized = normalize_text(raw)
    canonical_code: str | None = None
    canonical_label: str | None = None
    state: str
    rule: str
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        state, rule = ('unknown', 'null_scalar_unknown')
    else:
        choices: dict[str, str] = {}
        labels: dict[str, str] = {}
        for raw_code, raw_label in (choice_code_to_label or {}).items():
            code = _choice_code(raw_code)
            label = _display_text(raw_label)
            label_key = normalize_text(label)
            if not code or not label:
                raise QStateError('choice code and label must be nonempty')
            if code in choices and choices[code] != label:
                raise QStateError('choice code is duplicated')
            if label_key in labels and labels[label_key] != code:
                raise QStateError('choice label is ambiguous')
            choices[code] = label
            labels[label_key] = code
        code = _choice_code(raw)
        resolved_code = code if code in choices else labels.get(normalized)
        if resolved_code is not None:
            canonical_code = resolved_code
            canonical_label = choices[resolved_code]
            choice_state = canonical_state_for_dictionary_label(canonical_label)
            if choice_state is None:
                state, rule = ('observed', 'exact_dictionary_choice_observed')
            else:
                state, rule = (choice_state, f'exact_dictionary_choice_{choice_state}')
        elif normalized in RAW_EXPLICIT_STATE_SENTINELS:
            state = RAW_EXPLICIT_STATE_SENTINELS[normalized]
            rule = f'exact_canonical_state_sentinel_{state}'
        elif normalized in RAW_UNKNOWN_TOKENS:
            state, rule = ('unknown', 'exact_null_or_bare_na_unknown')
        else:
            state, rule = ('observed', 'valid_value_observed')
    if explicit_state is not None:
        explicit = normalize_text(explicit_state).replace(' ', '_')
        if explicit not in STATE_ORDER or explicit == 'observed':
            raise QStateError('explicit metadata state is unmapped')
        if state == 'observed':
            raise QStateError('valid observed value contradicts an explicit missing marker')
        if state != 'unknown' and state != explicit:
            raise QStateError('explicit missing-state markers conflict')
        state, rule = (explicit, f'explicit_metadata_{explicit}')
    branch = _branch_status(branch_status)
    if state == 'observed' and branch == 'false':
        raise QStateError('valid observed value contradicts a false branch marker')
    if state == 'unknown' and branch == 'false':
        state, rule = ('structural', 'branch_proven_false_otherwise_missing')
    if state not in STATE_ORDER:
        raise QStateError('canonical Q state is unmapped')
    return QStateAssessment(state=state, normalized_value=normalized, canonical_choice_code=canonical_code, canonical_choice_label=canonical_label, value_rule=rule, branch_status=branch)
