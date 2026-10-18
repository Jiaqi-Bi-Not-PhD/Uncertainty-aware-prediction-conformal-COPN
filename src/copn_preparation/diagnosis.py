"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

DETERMINED_MAP = {
    "0": "Parkinson's Disease (PD)",
    "1": "Progressive Supranuclear Palsy (PSP)",
    "2": "Multiple System Atrophy (MSA)",
    "3": "Corticobasal Syndrome (CBS)",
    "4": "Dementia with Lewy Bodies (DLB)",
    "5": "Frontotemporal Dementia (FTD)",
    "6": "Essential Tremor (ET)",
    "7": "REM Sleep Behaviour Disorder (RBD)",
}

PDPLUS_MAP = {
    "...Progressive Supranuclear Palsy (PSP)/paralysie supranucléaire progressive (PSP)": "Progressive Supranuclear Palsy (PSP)",
    "...Multiple System Atrophy (MSA)/Atrophie multisystématisée (AMS)": "Multiple System Atrophy (MSA)",
    "...Corticobasal Syndrome (CBS)/Dégénérescence cortico-basale (DCB)": "Corticobasal Syndrome (CBS)",
    "...Dementia with Lewy Bodies (DLB)/Démence à corps de Lewy (DCL)": "Dementia with Lewy Bodies (DLB)",
    "...Frontotemporal Dementia (FTD)/Dégénérescence fronto-temporale (DFT)": "Frontotemporal Dementia (FTD)",
    "...Essential Tremor (ET)/Tremblements essentiel (TE)": "Essential Tremor (ET)",
    "...REM Sleep Behaviour Disorder (RBD)/trouble du comportement en sommeil paradoxal (TCSP)": "REM Sleep Behaviour Disorder (RBD)",
}

def clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())

def resolve_original_class(row: dict[str, object]) -> str:
    group_v2 = clean(row.get("enrolment_group_v2"))
    group_v3 = clean(row.get("enrolment_group_v3"))
    final_group = group_v3 or group_v2

    if final_group.startswith("Healthy control"):
        return "Healthy"
    if final_group.startswith("PD (") or final_group.startswith("PD   ("):
        return "Parkinson"
    if final_group.startswith("AP (") or final_group.startswith("PPS ("):
        return "Atypical"
    return ""

def resolve_final_class(row: dict[str, object]) -> str:
    group_v2 = clean(row.get("enrolment_group_v2"))
    group_v3 = clean(row.get("enrolment_group_v3"))
    final_group = group_v3 or group_v2
    diagnosis_determined = clean(row.get("diagnosis_determined"))

    if diagnosis_determined == "0":
        return "Parkinson"
    if diagnosis_determined in {"1", "2", "3", "4", "5", "6", "7"}:
        return "Atypical"
    return resolve_original_class({"enrolment_group_v2": group_v2, "enrolment_group_v3": group_v3})

def resolve_specific_disease(row: dict[str, object]) -> str:
    group_v2 = clean(row.get("enrolment_group_v2"))
    group_v3 = clean(row.get("enrolment_group_v3"))
    final_group = group_v3 or group_v2
    diagnosis_determined = clean(row.get("diagnosis_determined"))
    diagnosis_pd = clean(row.get("diagnosis_pd"))
    diagnosis_pdplus = clean(row.get("diagnosis_pdplus"))
    diagnosis_pdplus_other = clean(row.get("diagnosis_pdplus_other"))

    if diagnosis_determined in DETERMINED_MAP:
        return DETERMINED_MAP[diagnosis_determined]
    if final_group.startswith("Healthy control"):
        return "Healthy control"
    if diagnosis_pd == "Yes/Oui":
        return "Parkinson's Disease (PD)"
    if diagnosis_pdplus in PDPLUS_MAP:
        return PDPLUS_MAP[diagnosis_pdplus]
    if diagnosis_pdplus.startswith("...Not Determined"):
        if final_group.startswith("AP (") or final_group.startswith("PPS ("):
            return "Atypical Parkinsonism (not determined)"
        return "Diagnosis not determined"
    if diagnosis_pdplus.startswith("...Other"):
        return diagnosis_pdplus_other or "Other diagnosis"
    if final_group.startswith("PD (") or final_group.startswith("PD   ("):
        return "Parkinson's Disease (PD)"
    if final_group.startswith("AP (") or final_group.startswith("PPS ("):
        return "Atypical Parkinsonism"
    return ""
