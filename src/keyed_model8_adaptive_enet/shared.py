"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from pathlib import Path

def model8_root() -> Path:
    return Path(__file__).resolve().parents[1]
