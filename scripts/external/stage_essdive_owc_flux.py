#!/usr/bin/env python3
"""Stage ESS-DIVE US-OWC methane observations as governed MUCC atlas evidence.

This retrieves the authoritative EML package and the chamber/porewater workbook
for DOI 10.15485/1568865. It materializes source-observation tables without
inventing a link to metatranscriptome samples. An explicit sample/date/depth
crosswalk remains required before ecological or flux validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
ESS_DIVE_DOI = "10.15485/1568865"
ESS_DIVE_EML_URL = f"https://cn.dataone.org/cn/v2/resolve/doi:{ESS_DIVE_DOI}"
ESS_DIVE_WORKBOOK_PID = "ess-dive-2bec939e57730b2-20190925T210717215"
ESS_DIVE_WORKBOOK_URL = (
    f"https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/{ESS_DIVE_WORKBOOK_PID}"
)
CLAIM_BOUNDARY = (
    "ESS-DIVE chamber-flux and porewater observations are source-staged, unlinked "
    "ecological evidence. They do not establish a match to a MUCC sequencing sample, "
    "a MAG-level flux effect, final MRV score/A-E tier, crediting claim, or "
    "source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Retrieve source files again even if a local source-audit copy exists.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool) -> None:
    if destination.is_file() and destination.stat().st_size and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "MethaNet atlas source staging"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"empty response while retrieving {url}")
    destination.write_bytes(payload)


def normalize_column(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).strip()


def numeric_or_empty(value: object, missing_sentinel: float = -9999.0) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted) or float(converted) == missing_sentinel:
        return np.nan
    return float(converted)


def parse_eml_title_and_package_id(path: Path) -> tuple[str, str]:
    root = ET.parse(path).getroot()
    title = root.findtext(".//{*}dataset/{*}title", default="")
    return title, root.attrib.get("packageId", "")


def source_file_manifest(eml_path: Path, workbook_path: Path) -> pd.DataFrame:
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "lane_id": LANE_ID,
            "source_id": "ESS_DIVE_10.15485_1568865",
            "artifact_role": "dataset_metadata_eml",
            "source_url": ESS_DIVE_EML_URL,
            "dataone_pid": f"doi:{ESS_DIVE_DOI}",
            "local_path": str(eml_path),
            "bytes": eml_path.stat().st_size,
            "sha256": sha256(eml_path),
            "retrieved_utc": retrieved_utc,
            "validation_status": "retrieved_and_xml_parseable",
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "lane_id": LANE_ID,
            "source_id": "ESS_DIVE_10.15485_1568865",
            "artifact_role": "chamber_flux_and_porewater_workbook",
            "source_url": ESS_DIVE_WORKBOOK_URL,
            "dataone_pid": ESS_DIVE_WORKBOOK_PID,
            "local_path": str(workbook_path),
            "bytes": workbook_path.stat().st_size,
            "sha256": sha256(workbook_path),
            "retrieved_utc": retrieved_utc,
            "validation_status": "retrieved_and_excel_parseable",
            "claim_boundary": CLAIM_BOUNDARY,
        },
    ]
    return pd.DataFrame(rows)


def chamber_flux_table(workbook_path: Path) -> pd.DataFrame:
    source = pd.read_excel(workbook_path, sheet_name="Chamber", header=2)
    required = {
        "LOC_LATITUDE",
        "LOC_LONGITUDE",
        "CMB_DATE",
        "CMB_VEGTYPE",
        "CMB_FCH4",
        "CMB_FCH4_Flag",
        "CMB_COMMENT",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"ESS-DIVE Chamber sheet is missing expected columns: {missing}")
    rows: list[dict[str, Any]] = []
    for index, record in source.iterrows():
        stamp = str(record["CMB_DATE"]).split(".")[0].strip()
        parsed = pd.to_datetime(stamp, format="%Y%m%d%H%M", errors="coerce")
        flux_value = numeric_or_empty(record["CMB_FCH4"])
        flag = numeric_or_empty(record["CMB_FCH4_Flag"])
        rows.append(
            {
                "lane_id": LANE_ID,
                "flux_observation_id": f"essdive_1568865_chamber_{index + 1:04d}",
                "source_id": "ESS_DIVE_10.15485_1568865",
                "source_workbook_sheet": "Chamber",
                "source_datetime_local": parsed.isoformat() if pd.notna(parsed) else "",
                "source_date": parsed.date().isoformat() if pd.notna(parsed) else "",
                "site_code": "US-OWC",
                "location_code": str(record["CMB_COMMENT"]).strip(),
                "latitude": numeric_or_empty(record["LOC_LATITUDE"]),
                "longitude": numeric_or_empty(record["LOC_LONGITUDE"]),
                "dominant_species_or_patch": str(record.get("CMB_SPP", "")).strip(),
                "vegetation_type": str(record["CMB_VEGTYPE"]).strip(),
                "measurement_approach": str(record.get("CMB_APPROACH", "")).strip(),
                "chamber_area_cm2": numeric_or_empty(record.get("CMB_AREA")),
                "chamber_temperature_c": numeric_or_empty(record.get("CMB_TA")),
                "methane_flux_nmol_m2_s": flux_value,
                "methane_flux_statistic": str(record.get("CMB_FCH4_STATISTIC", "")).strip(),
                "methane_flux_quality_comment": str(record.get("CMB_CH4_COMMENT", "")).strip(),
                "methane_flux_quality_flag": "" if pd.isna(flag) else str(int(flag)),
                "co2_flux_umol_m2_s": numeric_or_empty(record.get("CMB_FCO2")),
                "paired_peeper_code": str(record.get("CMB_COMMENT", "")).strip(),
                "source_value_status": (
                    "reported_valid"
                    if pd.notna(flux_value) and (pd.isna(flag) or flag == 0.0)
                    else "reported_flagged"
                    if pd.notna(flux_value)
                    else "source_missing_sentinel"
                ),
                "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
                "ecological_resolution_tier": "site_plot_datetime_flux_observation",
                "next_join_requirement": (
                    "authoritative sequencing sample accession/date/depth and a documented "
                    "spatial-temporal mapping to this chamber observation"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def porewater_ch4_table(workbook_path: Path) -> pd.DataFrame:
    raw_peeper = pd.read_excel(workbook_path, sheet_name="Peeper", header=None)
    descriptive_headers = raw_peeper.iloc[0].tolist()
    source_headers = raw_peeper.iloc[2].tolist()
    peeper = raw_peeper.iloc[3:].copy()
    peeper.columns = source_headers
    locations = pd.read_excel(workbook_path, sheet_name="Peeper Locations")
    if "SOIL_H2O_DATE" not in peeper.columns or "LOC_VARIABLE" not in locations.columns:
        raise ValueError("ESS-DIVE Peeper sheets do not contain expected source identifiers")
    location_lookup: dict[tuple[str, int], dict[str, object]] = {}
    for _, record in locations.iterrows():
        source_variable = normalize_column(record["LOC_VARIABLE"])
        if "_CH4_" not in source_variable:
            continue
        comment = str(record.get("COMMENT", "")).strip()
        code_match = re.search(r"Peeper code:\s*(\S+)", comment)
        level_match = re.search(r"_CH4_\d+_(\d+)_\d+$", source_variable)
        if code_match and level_match:
            location_lookup[(code_match.group(1), int(level_match.group(1)))] = record.to_dict()
    rows: list[dict[str, Any]] = []
    ch4_columns: list[tuple[int, object, str, int]] = []
    for position, (column, descriptive_header) in enumerate(
        zip(peeper.columns, descriptive_headers, strict=True)
    ):
        match = re.search(r"CH4 concentration\s+(\S+)\s+level\s+(\d+)", str(descriptive_header))
        if match:
            ch4_columns.append((position, column, match.group(1), int(match.group(2))))
    if not ch4_columns:
        raise ValueError("ESS-DIVE Peeper sheet has no CH4 concentration descriptors")
    for row_index, record in peeper.iterrows():
        stamp = str(record["SOIL_H2O_DATE"]).split(".")[0].strip()
        parsed = pd.to_datetime(stamp, format="%Y%m%d", errors="coerce")
        for position, column, peeper_code, level in ch4_columns:
            location = location_lookup.get((peeper_code, level), {})
            source_variable = str(location.get("LOC_VARIABLE", column))
            source_value = numeric_or_empty(record.iloc[position])
            comment = str(location.get("COMMENT", "")).strip()
            code_match = re.search(r"Peeper code:\s*(\S+)", comment)
            height_m = numeric_or_empty(location.get("LOC_HEIGHT"))
            rows.append(
                {
                    "lane_id": LANE_ID,
                    "porewater_observation_id": (
                        f"essdive_1568865_peeper_{row_index + 1:03d}_{normalize_column(source_variable)}"
                    ),
                    "source_id": "ESS_DIVE_10.15485_1568865",
                    "source_workbook_sheet": "Peeper",
                    "source_variable": source_variable,
                    "source_date": parsed.date().isoformat() if pd.notna(parsed) else "",
                    "site_code": "US-OWC",
                    "peeper_code": code_match.group(1) if code_match else "",
                    "latitude": numeric_or_empty(location.get("LOC_LATITUDE")),
                    "longitude": numeric_or_empty(location.get("LOC_LONGITUDE.")),
                    "depth_m_relative_to_top_mineral_soil": height_m,
                    "depth_cm_relative_to_top_mineral_soil": (
                        np.nan if pd.isna(height_m) else float(height_m) * 100.0
                    ),
                    "profile_zero_reference": str(location.get("PROFILE_ZERO_REF", "")).strip(),
                    "porewater_ch4_mM": source_value,
                    "source_value_status": (
                        "reported_valid" if pd.notna(source_value) else "source_missing_sentinel"
                    ),
                    "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
                    "ecological_resolution_tier": "site_profile_date_depth_observation",
                    "next_join_requirement": (
                        "authoritative sequencing sample accession/date/depth and a documented "
                        "profile or spatial mapping to this porewater observation"
                    ),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    source_dir = run_dir / "source_audit/essdive_10.15485_1568865"
    eml_path = source_dir / "essdive_1568865_metadata.xml"
    workbook_path = source_dir / "ChamberFlux_PorewaterConcentration_2015_2018.xlsx"
    download(ESS_DIVE_EML_URL, eml_path, args.force_download)
    download(ESS_DIVE_WORKBOOK_URL, workbook_path, args.force_download)
    if not workbook_path.read_bytes().startswith(b"PK"):
        raise ValueError("ESS-DIVE workbook does not have an XLSX container signature")
    title, package_id = parse_eml_title_and_package_id(eml_path)
    flux = chamber_flux_table(workbook_path)
    porewater = porewater_ch4_table(workbook_path)
    if len(flux) != 275:
        raise ValueError(f"unexpected ESS-DIVE Chamber row count: {len(flux)}")
    if flux["flux_observation_id"].duplicated().any() or porewater["porewater_observation_id"].duplicated().any():
        raise ValueError("ESS-DIVE observation identifiers are not unique")

    environmental_dir = run_dir / "environmental_metadata"
    write_tsv(source_dir / "source_file_manifest.tsv", source_file_manifest(eml_path, workbook_path))
    write_tsv(environmental_dir / "fact_mucc_v1_essdive_chamber_flux.tsv", flux)
    write_tsv(environmental_dir / "fact_mucc_v1_essdive_porewater_ch4.tsv", porewater)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "source_doi": ESS_DIVE_DOI,
        "dataset_title": title,
        "eml_package_id": package_id,
        "chamber_flux_rows": int(len(flux)),
        "chamber_flux_valid_rows": int(flux["source_value_status"].eq("reported_valid").sum()),
        "porewater_ch4_rows": int(len(porewater)),
        "porewater_ch4_valid_rows": int(porewater["source_value_status"].eq("reported_valid").sum()),
        "sample_join_status": "blocked_no_authoritative_sequence_sample_crosswalk",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    report_path = run_dir / "reports/mucc_v1_essdive_flux_ingestion_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
