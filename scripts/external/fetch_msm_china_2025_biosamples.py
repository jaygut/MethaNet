#!/usr/bin/env python3
"""Fetch NCBI BioSample metadata linked from the MSM GigaDB DOI.

The MSM MAG payloads are hosted in eLMSG, but DataCite also references NCBI
BioProjects that expose run-level and BioSample metadata. This script harvests
those metadata into a flat table suitable for MethaNet sample-readiness gates.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BIOPROJECTS = (
    "PRJNA1136686",
    "PRJNA1159532",
    "PRJNA1268148",
    "PRJNA1268163",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MSM-linked NCBI BioSample metadata.")
    parser.add_argument(
        "--dataset-dir",
        default="data/external/msm_china_2025",
        help="Dataset staging directory.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Delay between NCBI requests.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Parse existing BioSample text reports without network fetches.",
    )
    return parser.parse_args()


def fetch_url(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "MethaNet-MSM-biosample-harvest/1.0"},
    )
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_runinfo(project: str, out_path: Path, skip_fetch: bool) -> None:
    if skip_fetch and out_path.exists():
        return
    url = f"https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={project}"
    text = fetch_url(url)
    out_path.write_text(text)


def load_runinfo(metadata_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(metadata_dir.glob("PRJNA*_sra_runinfo.csv")):
        if path.stat().st_size == 0:
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Run"):
                    rows.append(row)
    return rows


def parse_biosample_report(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Identifiers:"):
            for label in ("BioSample", "Sample name", "SRA"):
                match = re.search(rf"{re.escape(label)}: ([^;\\n]+)", line)
                if match:
                    attrs[label.lower().replace(" ", "_")] = match.group(1).strip()
        elif line.startswith("Organism:"):
            attrs["organism"] = line.split(":", 1)[1].strip()
        elif line.startswith("/") and "=" in line:
            key, value = line[1:].split("=", 1)
            attrs[key.strip().lower().replace(" ", "_").replace("-", "_")] = value.strip().strip('"')
        elif line.startswith("Accession:"):
            parts = line.split()
            if len(parts) >= 2:
                attrs["accession"] = parts[1]
    return attrs


def fetch_biosample_reports(rows: list[dict[str, str]], metadata_dir: Path, skip_fetch: bool, delay: float) -> None:
    report_dir = metadata_dir / "biosample_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for biosample in sorted({row.get("BioSample", "") for row in rows if row.get("BioSample")}):
        out_path = report_dir / f"{biosample}.txt"
        if skip_fetch and out_path.exists():
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        url = f"https://www.ncbi.nlm.nih.gov/biosample/{biosample}?report=xml&format=text"
        try:
            out_path.write_text(fetch_url(url))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            out_path.write_text(f"FETCH_ERROR\t{type(exc).__name__}: {exc}\n")
        time.sleep(delay)


def write_flat_table(rows: list[dict[str, str]], metadata_dir: Path) -> Path:
    report_dir = metadata_dir / "biosample_reports"
    biosample_attrs: dict[str, dict[str, str]] = {}
    for report in sorted(report_dir.glob("SAMN*.txt")):
        biosample_attrs[report.stem] = parse_biosample_report(report.read_text(errors="replace"))

    out_path = metadata_dir / "ncbi_biosample_environmental_metadata.tsv"
    fields = [
        "bioproject",
        "run",
        "experiment",
        "library_name",
        "sample",
        "biosample",
        "sra_sample",
        "scientific_name",
        "sample_name",
        "organism",
        "collection_date",
        "depth",
        "env_broad_scale",
        "env_local_scale",
        "env_medium",
        "geo_loc_name",
        "lat_lon",
        "elevation",
        "description",
        "library_strategy",
        "library_source",
        "library_selection",
        "library_layout",
        "platform",
        "model",
        "spots",
        "bases",
        "size_mb",
        "download_path",
        "metadata_resolution_tier",
        "metadata_status",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            attrs = biosample_attrs.get(row.get("BioSample", ""), {})
            writer.writerow(
                {
                    "bioproject": row.get("BioProject", ""),
                    "run": row.get("Run", ""),
                    "experiment": row.get("Experiment", ""),
                    "library_name": row.get("LibraryName", ""),
                    "sample": row.get("Sample", ""),
                    "biosample": row.get("BioSample", ""),
                    "sra_sample": row.get("Sample", ""),
                    "scientific_name": row.get("ScientificName", ""),
                    "sample_name": row.get("SampleName", ""),
                    "organism": attrs.get("organism", ""),
                    "collection_date": attrs.get("collection_date", ""),
                    "depth": attrs.get("depth", ""),
                    "env_broad_scale": attrs.get("broad_scale_environmental_context", ""),
                    "env_local_scale": attrs.get("local_scale_environmental_context", ""),
                    "env_medium": attrs.get("environmental_medium", ""),
                    "geo_loc_name": attrs.get("geographic_location", ""),
                    "lat_lon": attrs.get("latitude_and_longitude", ""),
                    "elevation": attrs.get("elevation", ""),
                    "description": attrs.get("description", ""),
                    "library_strategy": row.get("LibraryStrategy", ""),
                    "library_source": row.get("LibrarySource", ""),
                    "library_selection": row.get("LibrarySelection", ""),
                    "library_layout": row.get("LibraryLayout", ""),
                    "platform": row.get("Platform", ""),
                    "model": row.get("Model", ""),
                    "spots": row.get("spots", ""),
                    "bases": row.get("bases", ""),
                    "size_mb": row.get("size_MB", ""),
                    "download_path": row.get("download_path", ""),
                    "metadata_resolution_tier": "exact_biosample" if attrs else "sra_runinfo_only",
                    "metadata_status": "resolved" if attrs else "missing_biosample_report",
                }
            )
    return out_path


def main() -> int:
    args = parse_args()
    metadata_dir = Path(args.dataset_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for project in BIOPROJECTS:
        out_path = metadata_dir / f"{project}_sra_runinfo.csv"
        fetch_runinfo(project, out_path, args.skip_fetch)
        time.sleep(args.sleep)

    rows = load_runinfo(metadata_dir)
    fetch_biosample_reports(rows, metadata_dir, args.skip_fetch, args.sleep)
    out_path = write_flat_table(rows, metadata_dir)
    print(f"runs\t{len(rows)}")
    print(f"biosamples\t{len({row.get('BioSample') for row in rows if row.get('BioSample')})}")
    print(f"wrote\t{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
