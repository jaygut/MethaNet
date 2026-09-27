"""End-to-end fixture for the ESS-DIVE tower normalization repair."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/reports/derive_tower_flux_mapping_repair.py"


def write_rows(path: Path, rows: list[dict[str, str]], delimiter: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def small_fixture(root: Path) -> dict[str, Path]:
    raw = root / "US_OWC_CH4_CO2_LE.csv"
    dictionary = root / "US_OWC_dd.csv"
    typed = root / "typed.tsv"
    normalized = root / "normalized.tsv"
    write_rows(
        raw,
        [
            {"Location": "-", "TIMESTAMP_START": "-", "TIMESTAMP_END": "-", "FCH4_F": "nmol m-2 s-1"},
            {"Location": "US-OWC", "TIMESTAMP_START": "201506010000", "TIMESTAMP_END": "201506010030", "FCH4_F": "2.500"},
            {"Location": "US-OWC", "TIMESTAMP_START": "201506010030", "TIMESTAMP_END": "201506010100", "FCH4_F": "-9999"},
        ],
        ",",
    )
    write_rows(
        dictionary,
        [{"column_or_row_name": "FCH4_F", "unit": "nmol m-2 s-1", "definition": "Methane flux gap-filled using an ANN model"}],
        ",",
    )
    typed_rows = []
    normalized_rows = []
    for index, (stamp, end, value, status) in enumerate(
        [
            ("201506010000", "2015-06-01T00:30:00", "2.5", "reported_valid"),
            ("201506010030", "2015-06-01T01:00:00", "", "source_missing_sentinel"),
        ]
    ):
        observation_id = "essdive_2500238_gapfilled_tower_" + stamp
        typed_rows.append({
            "lane_id": "mucc_v1_owc_wetland",
            "flux_observation_id": observation_id,
            "source_dataset_doi": "10.15485/2500238",
            "source_datetime_start_local_or_timezone_unknown": "2015-06-01T00:00:00" if index == 0 else "2015-06-01T00:30:00",
            "source_datetime_end_local_or_timezone_unknown": end,
            "source_datetime_timezone_status": "source_timestamp_unzoned_do_not_convert_or_infer_utc",
            "site_code": "US-OWC",
            "measurement_approach": "gap_filled_eddy_covariance",
            "temporal_resolution": "half_hourly",
            "methane_flux_nmol_m2_s": value,
            "source_value_status": status,
            "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
            "claim_boundary": "Site/time context only.",
        })
        normalized_rows.append({
            "lane_id": "mucc_v1_owc_wetland",
            "observation_id": observation_id,
            "observation_type": "gapfilled_tower_ch4_flux",
            "value": "",
            "source_date": "",
            "site_code": "US-OWC",
            "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
            "claim_scope": "source-staged process evidence",
        })
    write_rows(typed, typed_rows, "\t")
    write_rows(normalized, normalized_rows, "\t")
    return {"raw": raw, "dictionary": dictionary, "typed": typed, "normalized": normalized}


def test_tower_repair_restores_only_typed_values_with_units_and_status(tmp_path: Path) -> None:
    paths = small_fixture(tmp_path)
    output = tmp_path / "derived"
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--raw-flux-csv", str(paths["raw"]),
            "--data-dictionary-csv", str(paths["dictionary"]),
            "--typed-tower-tsv", str(paths["typed"]),
            "--normalized-process-tsv", str(paths["normalized"]),
            "--output-dir", str(output),
            "--expected-tower-rows", "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    audit = json.loads((output / "validation.json").read_text())
    assert audit["rows"] == 2
    assert audit["valid_numeric_values"] == 1
    assert audit["source_missing_values"] == 1
    with (output / "fact_gapfilled_tower_ch4_flux_mapping_repair.tsv").open() as handle:
        repaired = list(csv.DictReader(handle, delimiter="\t"))
    assert repaired[0]["value"] == "2.5"
    assert repaired[0]["value_unit"] == "nmol m-2 s-1"
    assert repaired[0]["source_date"] == "2015-06-01"
    assert repaired[1]["value"] == ""
    assert repaired[1]["source_value_status"] == "source_missing_sentinel"
    assert all(row["sample_join_status"].startswith("unlinked_") for row in repaired)
    with paths["normalized"].open() as handle:
        assert not any(row["value"] for row in csv.DictReader(handle, delimiter="\t"))


def test_tower_repair_rejects_raw_typed_value_conflict(tmp_path: Path) -> None:
    paths = small_fixture(tmp_path)
    raw_text = paths["raw"].read_text().replace("2.500", "9.000")
    paths["raw"].write_text(raw_text)
    # Run the CLI to exercise the same validation and prove no output is written.
    output = tmp_path / "derived"
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--raw-flux-csv", str(paths["raw"]),
            "--data-dictionary-csv", str(paths["dictionary"]),
            "--typed-tower-tsv", str(paths["typed"]),
            "--normalized-process-tsv", str(paths["normalized"]),
            "--output-dir", str(output),
            "--expected-tower-rows", "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "raw/typed methane flux mismatch" in completed.stderr
    assert not output.exists()
