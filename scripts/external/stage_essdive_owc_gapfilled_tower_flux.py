#!/usr/bin/env python3
"""Stage ESS-DIVE US-OWC gap-filled tower CH4 fluxes as unlinked source context.

This stages the fixed ESS-DIVE DOI 10.15485/2500238 package, including its
data dictionary, location record, half-hourly gap-filled CH4/CO2 fluxes, and
file manifest.  The source overlaps the MUCC study period in 2015--2016, but
does not supply a sequencing-sample, plot, or depth correspondence.  It is
therefore *site/time contextual evidence*, never a MAG/sample flux join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
ESS_DIVE_DOI = "10.15485/2500238"
ESS_DIVE_EML_URL = f"https://cn.dataone.org/cn/v2/resolve/doi:{ESS_DIVE_DOI}"
SOURCE_ID = "ESS_DIVE_10.15485_2500238"
DATAONE_OBJECT_URL = "https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/{pid}"
EXPECTED_TITLE = (
    "Gap-filled methane and carbon dioxide fluxes across two ecosystem states at the "
    "US-OWC AmeriFlux site (2015−2016, 2020−2022)"
)
SOURCE_FILES = (
    {
        "artifact_role": "data_dictionary",
        "filename": "US_OWC_dd.csv",
        "pid": "ess-dive-9850463ecdd53af-20250115T172451595",
        "dataone_md5": "5e054159bbe98890e4de1eed6b188afa",
    },
    {
        "artifact_role": "site_location",
        "filename": "US_OWC_locations.csv",
        "pid": "ess-dive-4a56a5b994db4c2-20250115T172445795",
        "dataone_md5": "df6f2e985bc8e20abf11b8ce508ac8a7",
    },
    {
        "artifact_role": "gapfilled_half_hourly_flux",
        "filename": "US_OWC_CH4_CO2_LE.csv",
        "pid": "ess-dive-4fd9f6153874740-20250116T015748966",
        "dataone_md5": "2a08fbfb649ccc1a417cc5d764abd887",
    },
    {
        "artifact_role": "file_manifest",
        "filename": "US_OWC_flmd.csv",
        "pid": "ess-dive-03054c4ccec2794-20250116T204347770",
        "dataone_md5": "d72c602c91f2816a979b75258fd41c1c",
    },
)
REQUIRED_FLUX_COLUMNS = {
    "Location",
    "TIMESTAMP_START",
    "TIMESTAMP_END",
    "NEE_F",
    "RE",
    "GPP",
    "FCH4_F",
    "LE_F",
}
CLAIM_BOUNDARY = (
    "ESS-DIVE gap-filled eddy-covariance tower fluxes are source-staged, site/time "
    "context only. They do not establish a match to a MUCC sequencing sample, "
    "plot, depth, MAG-level flux effect, final MRV score/A-E tier, crediting claim, "
    "or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Retrieve source files again even when a source-audit copy already exists.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(url: str, destination: Path, force: bool) -> None:
    """Fetch a source payload with a small retry budget for DataONE transients."""

    if destination.is_file() and destination.stat().st_size and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MethaNet atlas source staging"},
    )
    failure: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload:
                raise ValueError(f"empty response while retrieving {url}")
            destination.write_bytes(payload)
            return
        except (
            Exception
        ) as error:  # pragma: no cover - retry depends on remote service.
            failure = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"could not retrieve {url}: {failure}") from failure


def parse_eml(path: Path) -> tuple[str, str, set[str]]:
    root = ET.parse(path).getroot()
    title = root.findtext(".//{*}dataset/{*}title", default="").strip()
    package_id = root.attrib.get("packageId", "")
    objects = {
        (element.text or "").strip()
        for element in [
            *root.findall(".//{*}objectName"),
            *root.findall(".//{*}entityName"),
        ]
        if (element.text or "").strip()
    }
    return title, package_id, objects


def numeric_or_empty(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or float(number) == -9999.0:
        return None
    return float(number)


def parse_timestamp(value: object) -> str:
    parsed = pd.to_datetime(str(value).strip(), format="%Y%m%d%H%M", errors="coerce")
    return parsed.isoformat() if pd.notna(parsed) else ""


def parse_location_table(path: Path) -> dict[str, float | str]:
    # Rows 2--3 carry data types and units, rather than observations.
    locations = pd.read_csv(path, skiprows=[1, 2], dtype=str).fillna("")
    required = {"Location_ID", "Latitude", "Longitude"}
    missing = sorted(required - set(locations.columns))
    if missing:
        raise ValueError(f"ESS-DIVE location table lacks expected columns: {missing}")
    matching = locations.loc[locations["Location_ID"].eq("US-OWC")]
    if len(matching) != 1:
        raise ValueError(f"expected one US-OWC location row, observed {len(matching)}")
    record = matching.iloc[0]
    return {
        "site_code": str(record["Location_ID"]),
        "latitude": numeric_or_empty(record["Latitude"]),
        "longitude": numeric_or_empty(record["Longitude"]),
        "elevation_m": numeric_or_empty(record.get("Elevation", "")),
        "site_description": str(record.get("Description", "")),
    }


def gapfilled_tower_flux_table(
    flux_path: Path, location: dict[str, float | str]
) -> pd.DataFrame:
    """Return a half-hourly source table without assigning any sample-level key."""

    source = pd.read_csv(flux_path, dtype=str).fillna("")
    missing = sorted(REQUIRED_FLUX_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"ESS-DIVE flux table lacks expected columns: {missing}")
    # ESS-DIVE's CSV carries one units row immediately after the header.
    source = source.loc[source["Location"].ne("-")].copy()
    if source.empty:
        raise ValueError("ESS-DIVE flux table is empty")
    if set(source["Location"].unique()) != {"US-OWC"}:
        raise ValueError("ESS-DIVE flux table includes an unexpected Location value")

    rows: list[dict[str, Any]] = []
    for _, record in source.iterrows():
        start = parse_timestamp(record["TIMESTAMP_START"])
        end = parse_timestamp(record["TIMESTAMP_END"])
        if not start or not end:
            raise ValueError(
                "ESS-DIVE flux table has an unparsable half-hourly timestamp"
            )
        flux = numeric_or_empty(record["FCH4_F"])
        rows.append(
            {
                "lane_id": LANE_ID,
                "flux_observation_id": (
                    "essdive_2500238_gapfilled_tower_" + str(record["TIMESTAMP_START"])
                ),
                "source_id": SOURCE_ID,
                "source_dataset_doi": ESS_DIVE_DOI,
                "source_datetime_start_local_or_timezone_unknown": start,
                "source_datetime_end_local_or_timezone_unknown": end,
                "source_datetime_timezone_status": (
                    "source_timestamp_unzoned_do_not_convert_or_infer_utc"
                ),
                "site_code": str(record["Location"]),
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "elevation_m": location["elevation_m"],
                "site_description": location["site_description"],
                "measurement_approach": "gap_filled_eddy_covariance",
                "temporal_resolution": "half_hourly",
                "methane_flux_nmol_m2_s": flux,
                "net_ecosystem_exchange_umol_m2_s": numeric_or_empty(record["NEE_F"]),
                "ecosystem_respiration_umol_m2_s": numeric_or_empty(record["RE"]),
                "gross_primary_productivity_umol_m2_s": numeric_or_empty(record["GPP"]),
                "latent_heat_w_m2": numeric_or_empty(record["LE_F"]),
                "source_value_status": "reported_valid"
                if flux is not None
                else "source_missing_sentinel",
                "uncertainty_status": (
                    "methods_describe_gapfill_and_cumulative_uncertainty_no_row_level_field"
                ),
                "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
                "ecological_resolution_tier": "site_tower_datetime_flux_observation",
                "next_join_requirement": (
                    "authoritative sequencing-sample accession, collection time, plot, depth, "
                    "assay/coverage evidence, and documented tower-context mapping"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    table = pd.DataFrame(rows)
    if table["flux_observation_id"].duplicated().any():
        raise ValueError(
            "ESS-DIVE gap-filled tower flux observation identifiers are not unique"
        )
    return table


def source_file_manifest(source_dir: Path, eml_path: Path) -> pd.DataFrame:
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = [
        {
            "lane_id": LANE_ID,
            "source_id": SOURCE_ID,
            "artifact_role": "dataset_metadata_eml",
            "source_url": ESS_DIVE_EML_URL,
            "dataone_pid": f"doi:{ESS_DIVE_DOI}",
            "expected_dataone_md5": "",
            "observed_md5": digest(eml_path, "md5"),
            "local_path": str(eml_path),
            "bytes": eml_path.stat().st_size,
            "sha256": digest(eml_path, "sha256"),
            "retrieved_utc": retrieved_utc,
            "validation_status": "retrieved_and_xml_parseable",
            "claim_boundary": CLAIM_BOUNDARY,
        }
    ]
    for source_file in SOURCE_FILES:
        path = source_dir / str(source_file["filename"])
        expected_md5 = str(source_file["dataone_md5"])
        observed_md5 = digest(path, "md5")
        if observed_md5 != expected_md5:
            raise ValueError(
                f"DataONE MD5 mismatch for {source_file['filename']}: "
                f"expected={expected_md5} observed={observed_md5}"
            )
        rows.append(
            {
                "lane_id": LANE_ID,
                "source_id": SOURCE_ID,
                "artifact_role": source_file["artifact_role"],
                "source_url": DATAONE_OBJECT_URL.format(pid=source_file["pid"]),
                "dataone_pid": source_file["pid"],
                "expected_dataone_md5": expected_md5,
                "observed_md5": observed_md5,
                "local_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest(path, "sha256"),
                "retrieved_utc": retrieved_utc,
                "validation_status": "retrieved_and_dataone_md5_validated",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    source_dir = run_dir / "source_audit/essdive_10.15485_2500238"
    eml_path = source_dir / "essdive_2500238_metadata.xml"
    download(ESS_DIVE_EML_URL, eml_path, args.force_download)
    for source_file in SOURCE_FILES:
        download(
            DATAONE_OBJECT_URL.format(pid=source_file["pid"]),
            source_dir / str(source_file["filename"]),
            args.force_download,
        )

    title, package_id, objects = parse_eml(eml_path)
    if title != EXPECTED_TITLE:
        raise ValueError(f"unexpected ESS-DIVE dataset title: {title!r}")
    expected_filenames = {str(item["filename"]) for item in SOURCE_FILES}
    if not expected_filenames.issubset(objects):
        raise ValueError(
            "ESS-DIVE EML does not enumerate all expected package objects: "
            f"missing={sorted(expected_filenames - objects)}"
        )
    data_dictionary_path = source_dir / "US_OWC_dd.csv"
    dictionary = pd.read_csv(data_dictionary_path, dtype=str).fillna("")
    dictionary_columns = set(dictionary.get("column_or_row_name", pd.Series(dtype=str)))
    if not REQUIRED_FLUX_COLUMNS.issubset(dictionary_columns):
        raise ValueError(
            "ESS-DIVE data dictionary does not describe all required flux columns"
        )
    location = parse_location_table(source_dir / "US_OWC_locations.csv")
    flux = gapfilled_tower_flux_table(source_dir / "US_OWC_CH4_CO2_LE.csv", location)
    years = pd.to_datetime(
        flux["source_datetime_start_local_or_timezone_unknown"], errors="raise"
    ).dt.year
    observed_years = sorted(int(year) for year in years.unique())
    if observed_years != [2015, 2016, 2020, 2021, 2022]:
        raise ValueError(
            f"unexpected ESS-DIVE tower-flux year coverage: {observed_years}"
        )

    environmental_dir = run_dir / "environmental_metadata"
    manifest = source_file_manifest(source_dir, eml_path)
    write_tsv(source_dir / "source_file_manifest.tsv", manifest)
    write_tsv(
        environmental_dir / "fact_mucc_v1_essdive_gapfilled_tower_ch4_flux.tsv",
        flux,
    )
    annual = (
        pd.DataFrame(
            {
                "year": years,
                "reported_valid": flux["source_value_status"].eq("reported_valid"),
            }
        )
        .groupby("year", as_index=False)
        .agg(rows=("reported_valid", "size"), valid_ch4_rows=("reported_valid", "sum"))
    )
    write_tsv(
        run_dir / "reports/mucc_v1_essdive_gapfilled_tower_flux_annual_coverage.tsv",
        annual,
    )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "source_doi": ESS_DIVE_DOI,
        "dataset_title": title,
        "eml_package_id": package_id,
        "gapfilled_half_hourly_tower_flux_rows": int(len(flux)),
        "gapfilled_half_hourly_tower_flux_valid_ch4_rows": int(
            flux["source_value_status"].eq("reported_valid").sum()
        ),
        "observed_years": observed_years,
        "mucc_temporal_overlap_years": [2015, 2016],
        "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    report_path = (
        run_dir / "reports/mucc_v1_essdive_gapfilled_tower_flux_ingestion_summary.json"
    )
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
