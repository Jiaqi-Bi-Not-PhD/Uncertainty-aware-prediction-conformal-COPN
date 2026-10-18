"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import re

def norm_header(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xad", "")
    return re.sub(r"\s+", " ", text).strip().lower()

def is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False

def distinct_nonblank(values: list[object]) -> list[object]:
    seen: list[object] = []
    seen_keys: set[str] = set()
    for value in values:
        if is_blank(value):
            continue
        key = repr(value)
        if key not in seen_keys:
            seen_keys.add(key)
            seen.append(value)
    return seen

def decode_checkbox(row: dict[str, object], options: list[tuple[str, str]]) -> object:
    chosen: list[str] = []
    for header, label in options:
        value = row.get(header)
        if isinstance(value, str) and value.strip().lower() == "checked":
            chosen.append(label)
    if not chosen:
        return None
    if len(chosen) == 1:
        return chosen[0]
    return " | ".join(chosen)

def row_dicts_by_key(ws) -> OrderedDict[str, dict[str, object]]:
    headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    rows: OrderedDict[str, dict[str, object]] = OrderedDict()
    for row in ws.iter_rows(min_row=2, values_only=True):
        record = {headers[idx]: row[idx] for idx in range(len(headers))}
        project_key = record.get("Project key") or record.get("project_key")
        if is_blank(project_key):
            continue
        rows[str(project_key)] = record
    return rows

def default_transform(record: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in record.items() if k is not None}

NEURO_CHECKBOX_HEADERS = {
    "Native language / Langue maternelle (choice=English / Anglais)",
    "Native language / Langue maternelle (choice=French / Français)",
    "Native language / Langue maternelle (choice=Other / Autre)",
    "Language of evaluation / Langue d'évaluation:   (choice=English / Anglais)",
    "Language of evaluation / Langue d'évaluation:   (choice=French / Français)",
    "Language of evaluation / Langue d'évaluation:   (choice=Other / Autre)",
}

def transform_neuropsychological(record: dict[str, object]) -> dict[str, object]:
    out = {
        k: v
        for k, v in record.items()
        if k is not None and k not in NEURO_CHECKBOX_HEADERS
    }

    native = decode_checkbox(
        record,
        [
            ("Native language / Langue maternelle (choice=English / Anglais)", "English / Anglais"),
            ("Native language / Langue maternelle (choice=French / Français)", "French / Français"),
            ("Native language / Langue maternelle (choice=Other / Autre)", "Other / Autre"),
        ],
    )
    if native is not None:
        out["Native language / Langue maternelle"] = native

    evaluation_language = decode_checkbox(
        record,
        [
            ("Language of evaluation / Langue d'évaluation:   (choice=English / Anglais)", "English / Anglais"),
            ("Language of evaluation / Langue d'évaluation:   (choice=French / Français)", "French / Français"),
            ("Language of evaluation / Langue d'évaluation:   (choice=Other / Autre)", "Other / Autre"),
        ],
    )
    if evaluation_language is not None:
        out["Language of evaluation / Langue d'évaluation:  "] = evaluation_language

    return out

CAPRI_NEURO_MAP = {
    "Immediate Recall raw": "Immediate recall raw",
    "Delayed Recall raw": "Delayed recall raw",
    "Trail A raw score": "Trail A raw score (time in sec.)",
    "Trail B raw score": "Trail B raw score (time in sec.)",
    "Digit Span Forward raw": "Digit Span Forward - total correct (Raw score) ",
    "Digit Span Backward raw": "Digit Span Backward - total correct (Raw score)",
    "Total Recall raw": "Trial total 1,2,3 (Raw score)",
    "FAS total raw": "Letter Fluency Total (Raw score)",
    "Animals total raw": "Semantic Fluency Animals (Raw score)",
    "Actions total raw ": "Semantic Fluency Actions (Raw score)",
    "Total (SR+SC) raw": "BNT sans indice + IS",
    "SR+SC+PC raw": "BNT sans indice + IS + IP",
}

def transform_neuropsychological_capri(record: dict[str, object]) -> dict[str, object]:
    out = {
        k: v
        for k, v in record.items()
        if k is not None and k not in CAPRI_NEURO_MAP
    }
    for source_header, canonical_header in CAPRI_NEURO_MAP.items():
        if source_header in record and not is_blank(record[source_header]):
            out[canonical_header] = record[source_header]
    return out

FAMILIES = [
    {
        "output_sheet": "MoCA (joined)",
        "ordered_sheets": ["MoCA", "MoCA-1", "MoCA-2"],
        "transformers": {
            "MoCA": default_transform,
            "MoCA-1": default_transform,
            "MoCA-2": default_transform,
        },
        "base_sheet": "MoCA-2",
        "notes": "Exact match family. Later versions override earlier versions column-by-column.",
    },
    {
        "output_sheet": "MDS-UPDRS (joined)",
        "ordered_sheets": ["MDS-UPDRS", "MDS-UPDRS-1"],
        "transformers": {
            "MDS-UPDRS": default_transform,
            "MDS-UPDRS-1": default_transform,
        },
        "base_sheet": "MDS-UPDRS-1",
        "notes": "Exact match family. MDS-UPDRS-1 overrides MDS-UPDRS for overlapping values.",
    },
    {
        "output_sheet": "MBIC (joined)",
        "ordered_sheets": ["MBIC", "MBIC (CaPRI)"],
        "transformers": {
            "MBIC": default_transform,
            "MBIC (CaPRI)": default_transform,
        },
        "base_sheet": "MBIC (CaPRI)",
        "notes": "Exact match family. MBIC (CaPRI) is treated as the later sheet only because the workbook has no explicit version number.",
    },
    {
        "output_sheet": "Neuropsychological (joined)",
        "ordered_sheets": ["Neuropsychological (CaPRI)", "Neuropsychological", "Neuropsychological V02"],
        "transformers": {
            "Neuropsychological (CaPRI)": transform_neuropsychological_capri,
            "Neuropsychological": transform_neuropsychological,
            "Neuropsychological V02": default_transform,
        },
        "base_sheet": "Neuropsychological V02",
        "notes": (
            "Neuropsychological V02 is treated as the latest explicit version. "
            "The original Neuropsychological sheet is used as the fallback full version, "
            "and Neuropsychological (CaPRI) is treated as a site-specific subset fallback."
        ),
    },
]

def canonical_headers(
    workbook,
    ordered_sheets: list[str],
    transformers: dict[str, callable],
    base_sheet: str,
) -> list[str]:
    base_headers = [workbook[base_sheet].cell(row=1, column=c).value for c in range(1, workbook[base_sheet].max_column + 1)]
    canonical = [h for h in base_headers if h is not None]
    seen = {norm_header(h) for h in canonical}
    for sheet in ordered_sheets:
        row_map = row_dicts_by_key(workbook[sheet])
        sample_record = next(iter(row_map.values()))
        transformed = transformers[sheet](sample_record)
        for header in transformed:
            if header is None:
                continue
            key = norm_header(header)
            if key not in seen:
                seen.add(key)
                canonical.append(header)
    return canonical

def master_project_keys(workbook) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sheet_name in ["Enrollement"] + [ws.title for ws in workbook.worksheets if ws.title != "Enrollement"]:
        ws = workbook[sheet_name]
        key_col = 1
        for value, in ws.iter_rows(min_row=2, min_col=key_col, max_col=key_col, values_only=True):
            if is_blank(value):
                continue
            text = str(value)
            if text not in seen:
                seen.add(text)
                ordered.append(text)
    return ordered

def family_join_data(workbook, family: dict, ordered_keys: list[str]) -> dict:
    ordered_sheets = family["ordered_sheets"]
    transformers = family["transformers"]
    headers = canonical_headers(workbook, ordered_sheets, transformers, family["base_sheet"])

    source_rows = {sheet: row_dicts_by_key(workbook[sheet]) for sheet in ordered_sheets}
    transformed_rows = {
        sheet: {key: transformers[sheet](record) for key, record in records.items()}
        for sheet, records in source_rows.items()
    }

    rows_out: list[list[object]] = []
    conflict_counts: defaultdict[str, int] = defaultdict(int)
    coverage: dict[str, set[str]] = {sheet: set() for sheet in ordered_sheets}

    for project_key in ordered_keys:
        merged: dict[str, object] = {"Project key": project_key}
        for header in headers:
            if header == "Project key":
                continue
            candidates: list[tuple[str, object]] = []
            for sheet in ordered_sheets:
                row = transformed_rows[sheet].get(project_key)
                if row is None:
                    continue
                value = row.get(header)
                if not is_blank(value):
                    candidates.append((sheet, value))
            if candidates:
                chosen_sheet, chosen_value = candidates[-1]
                merged[header] = chosen_value
                coverage[chosen_sheet].add(project_key)
                values_only = [value for _, value in candidates]
                if len(distinct_nonblank(values_only)) > 1:
                    conflict_counts[header] += 1
            else:
                merged[header] = None
        rows_out.append([merged.get(header) for header in headers])

    return {
        "headers": headers,
        "rows": rows_out,
        "conflict_counts": dict(sorted(conflict_counts.items(), key=lambda item: (-item[1], item[0]))),
        "coverage": {sheet: len(keys) for sheet, keys in coverage.items()},
    }

def write_joined_sheet(workbook, sheet_name: str, headers: list[str], rows: list[list[object]]) -> None:
    if sheet_name in workbook.sheetnames:
        del workbook[sheet_name]
    ws = workbook.create_sheet(title=sheet_name)
    ws.append(headers)
    for row in rows:
        ws.append(row)
    ws.freeze_panes = "A2"
