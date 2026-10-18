"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from typing import Any

import pandas as pd

class DataAuditError(ValueError):
    """Raised when frozen-data invariants do not hold."""

OUTCOME_LABEL_MAP = {
    "pd": "PD",
    "parkinson": "PD",
    "parkinson's": "PD",
    "parkinsons": "PD",
    "ap": "AP",
    "atypical": "AP",
    "atypical parkinsonism": "AP",
    "hc": "HC",
    "healthy": "HC",
    "healthy control": "HC",
}

OUTCOME_CODES = {"PD": 0, "AP": 1, "HC": 2}

def canonicalize_outcome(value: Any) -> str | None:
    """Map workbook labels/codes to PD, AP, HC; preserve a truly blank outcome."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        reverse = {code: label for label, code in OUTCOME_CODES.items()}
        if int(value) in reverse:
            return reverse[int(value)]
    text = " ".join(str(value).strip().lower().split())
    if text in OUTCOME_LABEL_MAP:
        return OUTCOME_LABEL_MAP[text]
    if text.upper() in OUTCOME_CODES:
        return text.upper()
    raise DataAuditError(f"unknown outcome label: {value!r}")
