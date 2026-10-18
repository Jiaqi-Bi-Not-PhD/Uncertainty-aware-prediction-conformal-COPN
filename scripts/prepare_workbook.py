#!/usr/bin/env python3
"""Prepare an authorized local workbook without creating narrative/audit files."""
from pathlib import Path
import argparse
from copn_analysis.private_io import private_input, private_output
from openpyxl import load_workbook
from copn_preparation.diagnosis import clean, resolve_final_class, resolve_specific_disease
from copn_preparation.versioned_forms import FAMILIES, family_join_data, master_project_keys, write_joined_sheet


def prepare(input_path: Path, output_path: Path):
    input_path = private_input(input_path)
    output_path = private_output(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Choose a separate output workbook.")
    workbook = load_workbook(input_path)
    sheet = workbook["Outcomes"]
    headers = [clean(cell.value) for cell in sheet[1]]
    index = {name: col for col, name in enumerate(headers) if name}
    required = {"project_key", "enrolment_group_v2", "enrolment_group_v3",
                "diagnosis_determined", "diagnosis_pd", "diagnosis_pdplus", "diagnosis_pdplus_other"}
    if not required.issubset(index):
        raise ValueError("Input Outcomes sheet lacks fields required to resolve the final diagnosis.")
    rows = []
    for values in sheet.iter_rows(min_row=2, values_only=True):
        record = {name: values[col] for name, col in index.items()}
        rows.append((record["project_key"], resolve_final_class(record), resolve_specific_disease(record)))
    sheet.delete_cols(1, sheet.max_column)
    # delete_cols leaves worksheet row metadata; write by coordinate deliberately.
    for col, name in enumerate(["project_key", "FINAL DISEASE DIAGNOSIS", "SPECIFIC DISEASE"], 1):
        sheet.cell(1, col, name)
    for row, values in enumerate(rows, 2):
        for col, value in enumerate(values, 1):
            sheet.cell(row, col, value)
    keys = master_project_keys(workbook)
    for family in FAMILIES:
        result = family_join_data(workbook, family, keys)
        write_joined_sheet(workbook, family["output_sheet"], result["headers"], result["rows"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.input, args.output)
