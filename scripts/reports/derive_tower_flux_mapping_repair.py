#!/usr/bin/env python3
"""Derive a tower-only repair view for the August 2026 metadata mapping defect.

Read the immutable ESS-DIVE raw CSV, its dictionary, the staged typed tower TSV,
and the historical generic normalized TSV. Verify exact observation IDs, raw
values, units, half-hour windows, and the blank generic tower fields before
writing a new output directory. Historical inputs are never modified. The view
is site/time context, with no sequencing-sample or MAG flux attribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


OBSERVATION_TYPE = "gapfilled_tower_ch4_flux"
DOI = "10.15485/2500238"
UNIT = "nmol m-2 s-1"
SOURCE_FIELD = "FCH4_F"
TIME_FIELD = "source_datetime_start_local_or_timezone_unknown"
END_FIELD = "source_datetime_end_local_or_timezone_unknown"
TIMEZONE_STATUS = "source_timestamp_unzoned_do_not_convert_or_infer_utc"
OUTPUT_NAME = "fact_gapfilled_tower_ch4_flux_mapping_repair.tsv"
OUTPUT_FIELDS = [
    "lane_id", "observation_id", "observation_type", "value", "value_unit",
    "source_value_field", "source_date", TIME_FIELD, END_FIELD,
    "source_datetime_timezone_status", "site_code", "measurement_approach",
    "temporal_resolution", "source_value_status", "sample_join_status",
    "source_dataset_doi", "normalization_status", "claim_scope", "claim_boundary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-flux-csv", type=Path, required=True)
    parser.add_argument("--data-dictionary-csv", type=Path, required=True)
    parser.add_argument("--typed-tower-tsv", type=Path, required=True)
    parser.add_argument("--normalized-process-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tower-rows", type=int, required=True)
    return parser.parse_args()


def read_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader.fieldnames or []), list(reader)


def require_fields(fields: list[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def indexed_rows(
    rows: list[dict[str, str]], key_field: str, label: str
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get(key_field) or "").strip()
        if not key or key in index:
            raise ValueError(f"{label} has missing or duplicate {key_field}: {key!r}")
        index[key] = row
    return index


def source_decimal(value: str) -> Decimal | None:
    value = value.strip()
    if value in {"", "-9999"}:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid source flux value: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"non-finite source flux value: {value!r}")
    return number


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(
    raw_flux_csv: Path,
    data_dictionary_csv: Path,
    typed_tower_tsv: Path,
    normalized_process_tsv: Path,
    expected_tower_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return a validated tower-only projection and its source audit."""

    if expected_tower_rows <= 0:
        raise ValueError("expected_tower_rows must be positive")
    dictionary_fields, dictionary = read_rows(data_dictionary_csv, ",")
    require_fields(dictionary_fields, {"column_or_row_name", "unit", "definition"}, "dictionary")
    definitions = [row for row in dictionary if row["column_or_row_name"] == SOURCE_FIELD]
    if len(definitions) != 1 or definitions[0]["unit"] != UNIT:
        raise ValueError(f"dictionary must give exactly one {SOURCE_FIELD} definition in {UNIT}")
    if "gap-fill" not in definitions[0]["definition"].lower():
        raise ValueError(f"dictionary does not describe {SOURCE_FIELD} as gap-filled")

    raw_fields, raw_rows_all = read_rows(raw_flux_csv, ",")
    require_fields(raw_fields, {"Location", "TIMESTAMP_START", "TIMESTAMP_END", SOURCE_FIELD}, "raw flux")
    raw_rows = [row for row in raw_rows_all if row["Location"] != "-"]
    raw_by_id = indexed_rows(
        raw_rows, "TIMESTAMP_START", "raw flux timestamps"
    )
    typed_fields, typed_rows = read_rows(typed_tower_tsv, "\t")
    require_fields(
        typed_fields,
        {
            "lane_id", "flux_observation_id", "source_dataset_doi", TIME_FIELD,
            END_FIELD, "source_datetime_timezone_status", "site_code",
            "measurement_approach", "temporal_resolution", "methane_flux_nmol_m2_s",
            "source_value_status", "sample_join_status", "claim_boundary",
        },
        "typed tower",
    )
    typed_by_id = indexed_rows(typed_rows, "flux_observation_id", "typed tower")
    normalized_fields, normalized_all = read_rows(normalized_process_tsv, "\t")
    require_fields(
        normalized_fields,
        {"lane_id", "observation_id", "observation_type", "value", "source_date", "site_code", "sample_join_status", "claim_scope"},
        "normalized process",
    )
    normalized_rows = [
        row for row in normalized_all if row["observation_type"] == OBSERVATION_TYPE
    ]
    normalized_by_id = indexed_rows(normalized_rows, "observation_id", "normalized tower")
    if not (len(raw_by_id) == len(typed_by_id) == len(normalized_by_id) == expected_tower_rows):
        raise ValueError(
            "tower row counts differ: "
            f"raw={len(raw_by_id)}, typed={len(typed_by_id)}, "
            f"normalized={len(normalized_by_id)}, expected={expected_tower_rows}"
        )
    expected_ids = {
        "essdive_2500238_gapfilled_tower_" + timestamp for timestamp in raw_by_id
    }
    if set(typed_by_id) != expected_ids or set(normalized_by_id) != expected_ids:
        raise ValueError("raw, typed, and normalized tower observation IDs do not match exactly")

    repaired: list[dict[str, str]] = []
    valid_count = 0
    for observation_id in sorted(expected_ids):
        raw_start = observation_id.removeprefix("essdive_2500238_gapfilled_tower_")
        raw = raw_by_id[raw_start]
        typed = typed_by_id[observation_id]
        normalized = normalized_by_id[observation_id]
        if normalized["value"].strip() or normalized["source_date"].strip():
            raise ValueError(f"historical normalized tower row is not blank: {observation_id}")
        if raw["Location"] != typed["site_code"] or typed["site_code"] != normalized["site_code"]:
            raise ValueError(f"site mismatch for {observation_id}")
        if typed["lane_id"] != normalized["lane_id"] or typed["source_dataset_doi"] != DOI:
            raise ValueError(f"lane or DOI mismatch for {observation_id}")
        if typed["sample_join_status"] != normalized["sample_join_status"]:
            raise ValueError(f"sample join status mismatch for {observation_id}")
        if typed["sample_join_status"] != "unlinked_no_authoritative_sequence_sample_crosswalk":
            raise ValueError(f"unexpected sample linkage for {observation_id}")
        if typed["measurement_approach"] != "gap_filled_eddy_covariance" or typed["temporal_resolution"] != "half_hourly":
            raise ValueError(f"measurement contract mismatch for {observation_id}")
        if typed["source_datetime_timezone_status"] != TIMEZONE_STATUS:
            raise ValueError(f"timezone contract mismatch for {observation_id}")
        try:
            start = datetime.strptime(raw["TIMESTAMP_START"], "%Y%m%d%H%M")
            end = datetime.strptime(raw["TIMESTAMP_END"], "%Y%m%d%H%M")
        except ValueError as exc:
            raise ValueError(f"invalid raw timestamp for {observation_id}") from exc
        if end - start != timedelta(minutes=30):
            raise ValueError(f"non-half-hour window for {observation_id}")
        if typed[TIME_FIELD] != start.isoformat() or typed[END_FIELD] != end.isoformat():
            raise ValueError(f"typed timestamp mismatch for {observation_id}")
        raw_value = source_decimal(raw[SOURCE_FIELD])
        typed_value = source_decimal(typed["methane_flux_nmol_m2_s"])
        if raw_value != typed_value:
            raise ValueError(f"raw/typed methane flux mismatch for {observation_id}")
        expected_status = "reported_valid" if raw_value is not None else "source_missing_sentinel"
        if typed["source_value_status"] != expected_status:
            raise ValueError(f"source value status mismatch for {observation_id}")
        valid_count += int(raw_value is not None)
        repaired.append({
            "lane_id": typed["lane_id"],
            "observation_id": observation_id,
            "observation_type": OBSERVATION_TYPE,
            "value": typed["methane_flux_nmol_m2_s"] if raw_value is not None else "",
            "value_unit": UNIT,
            "source_value_field": SOURCE_FIELD,
            "source_date": start.date().isoformat(),
            TIME_FIELD: typed[TIME_FIELD],
            END_FIELD: typed[END_FIELD],
            "source_datetime_timezone_status": TIMEZONE_STATUS,
            "site_code": typed["site_code"],
            "measurement_approach": typed["measurement_approach"],
            "temporal_resolution": typed["temporal_resolution"],
            "source_value_status": typed["source_value_status"],
            "sample_join_status": typed["sample_join_status"],
            "source_dataset_doi": DOI,
            "normalization_status": "derived_repair_from_verified_typed_source",
            "claim_scope": normalized["claim_scope"],
            "claim_boundary": typed["claim_boundary"],
        })
    audit = {
        "repair_status": "validated_derived_view",
        "observation_type": OBSERVATION_TYPE,
        "rows": len(repaired),
        "valid_numeric_values": valid_count,
        "source_missing_values": len(repaired) - valid_count,
        "source_field": SOURCE_FIELD,
        "value_unit": UNIT,
        "time_contract": TIMEZONE_STATUS,
        "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
        "source_dataset_doi": DOI,
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256(path)}
            for label, path in (
                ("raw_flux_csv", raw_flux_csv),
                ("data_dictionary_csv", data_dictionary_csv),
                ("typed_tower_tsv", typed_tower_tsv),
                ("normalized_process_tsv", normalized_process_tsv),
            )
        },
        "claim_boundary": "Site/time gap-filled tower context only; no sample/MAG attribution, independent observed flux truth, calibrated MRV risk tier, or crediting decision.",
    }
    return repaired, audit


def main() -> int:
    args = parse_args()
    repaired, audit = derive(
        args.raw_flux_csv,
        args.data_dictionary_csv,
        args.typed_tower_tsv,
        args.normalized_process_tsv,
        args.expected_tower_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_tsv = args.output_dir / OUTPUT_NAME
    with output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(repaired)
    audit["output"] = {"path": str(output_tsv.resolve()), "sha256": sha256(output_tsv)}
    audit["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "validation.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: audit[key] for key in ("repair_status", "rows", "valid_numeric_values", "source_missing_values")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
