#!/usr/bin/env python3
"""Stage authoritative NCBI BioProject links for MUCC expression sample labels.

NCBI BioProject titles supply a defensible sequence-project identity link for
some MUCC metatranscriptome labels. They do not supply an exact collection
datetime, depth-in-cm mapping, or a field-observation/flux pairing, so all of
those remain explicit unresolved fields in the staged crosswalk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_QUERY = '"Old Woman Creek"'
CLAIM_BOUNDARY = (
    "NCBI BioProject title matches identify sequence-project context only. They do not "
    "establish an exact collection datetime, depth in cm, environmental/flux pairing, "
    "ecological association, measured methane flux, final MRV score/A-E tier, crediting "
    "claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Retrieve NCBI E-utilities source responses again even if local copies exist.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str, destination: Path, force: bool) -> dict[str, Any]:
    if not destination.is_file() or not destination.stat().st_size or force:
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MethaNet MUCC sample-crosswalk recovery"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())
    return json.loads(destination.read_text())


def sample_label_from_project_title(title: str) -> str:
    match = re.search(r" - ([A-Za-z0-9_]+)$", title)
    return match.group(1) if match else ""


def candidate_project_labels(source_label: str) -> list[tuple[str, str]]:
    candidates = [(source_label, "exact_title_suffix")]
    if source_label.startswith("July_"):
        candidates.append(("Jul_" + source_label.removeprefix("July_"), "deterministic_July_to_Jul"))
    return candidates


def source_manifest_rows(source_dir: Path, retrieved_utc: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.json")):
        role = "bioproject_search" if path.name == "bioproject_search.json" else "bioproject_summary_batch"
        rows.append(
            {
                "lane_id": LANE_ID,
                "source_id": "NCBI_BioProject_Old_Woman_Creek_query",
                "artifact_role": role,
                "source_url": (
                    NCBI_ESEARCH_URL if role == "bioproject_search" else NCBI_ESUMMARY_URL
                ),
                "local_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "retrieved_utc": retrieved_utc,
                "validation_status": "retrieved_and_json_parseable",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def build_crosswalk(samples: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    records_by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        label = sample_label_from_project_title(str(record.get("project_title", "")))
        if label:
            records_by_label.setdefault(label, []).append(record)
    rows: list[dict[str, Any]] = []
    for _, source in samples.iterrows():
        source_label = str(source["source_sample_column"])
        sample_id = str(source["sample_id"])
        matching_record: dict[str, Any] | None = None
        mapping_method = ""
        candidate_label = ""
        for candidate, method in candidate_project_labels(source_label):
            candidate_records = records_by_label.get(candidate, [])
            if len(candidate_records) == 1:
                matching_record = candidate_records[0]
                mapping_method = method
                candidate_label = candidate
                break
            if len(candidate_records) > 1:
                mapping_method = "ambiguous_duplicate_project_title_suffix"
                candidate_label = candidate
                break
        if matching_record is None:
            rows.append(
                {
                    "lane_id": LANE_ID,
                    "sample_id": sample_id,
                    "source_sample_column": source_label,
                    "matched_project_title_sample_label": candidate_label,
                    "sample_label_mapping_method": mapping_method or "no_authoritative_BioProject_title_match",
                    "sample_project_link_status": "unmapped_preserved",
                    "bioproject_accession": "",
                    "bioproject_uid": "",
                    "bioproject_title": "",
                    "bioproject_description": "",
                    "bioproject_registration_date": "",
                    "bioproject_submitter": "",
                    "sequence_project_identity_status": "missing_authoritative_sequence_project_link",
                    "exact_collection_datetime_status": "not_available_from_BioProject_title",
                    "depth_cm_join_status": "unresolved_source_depth_code_to_cm_mapping",
                    "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
                    "next_validation_action": (
                        "recover Table S1/S4 or NCBI run/BioSample metadata with an exact sample label"
                    ),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            continue
        rows.append(
            {
                "lane_id": LANE_ID,
                "sample_id": sample_id,
                "source_sample_column": source_label,
                "matched_project_title_sample_label": candidate_label,
                "sample_label_mapping_method": mapping_method,
                "sample_project_link_status": "mapped_to_authoritative_NCBI_BioProject_title",
                "bioproject_accession": str(matching_record.get("project_acc", "")),
                "bioproject_uid": str(matching_record.get("uid", "")),
                "bioproject_title": str(matching_record.get("project_title", "")),
                "bioproject_description": str(matching_record.get("project_description", "")),
                "bioproject_registration_date": str(matching_record.get("registration_date", "")),
                "bioproject_submitter": ";".join(
                    str(item) for item in matching_record.get("submitter_organization_list", [])
                ),
                "sequence_project_identity_status": "linked_to_NCBI_BioProject_not_run_or_BioSample",
                "exact_collection_datetime_status": "not_available_from_BioProject_title",
                "depth_cm_join_status": "unresolved_source_depth_code_to_cm_mapping",
                "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
                "next_validation_action": (
                    "resolve NCBI run/BioSample accession and collection date/depth, then join to an "
                    "explicit ESS-DIVE spatial-temporal observation window"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["sample_id"].duplicated().any() or len(frame) != len(samples):
        raise ValueError("crosswalk must preserve each source expression sample exactly once")
    return frame


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    source_dir = run_dir / "source_audit/ncbi_bioproject_old_woman_creek"
    search_url = NCBI_ESEARCH_URL + "?" + urllib.parse.urlencode(
        {"db": "bioproject", "term": NCBI_QUERY, "retmode": "json", "retmax": 500}
    )
    search = fetch_json(search_url, source_dir / "bioproject_search.json", args.force_download)
    ids = search["esearchresult"].get("idlist", [])
    if not ids:
        raise ValueError("NCBI BioProject query returned no Old Woman Creek records")
    records: list[dict[str, Any]] = []
    for batch_number, start in enumerate(range(0, len(ids), 40), start=1):
        batch_ids = ids[start : start + 40]
        summary_url = NCBI_ESUMMARY_URL + "?" + urllib.parse.urlencode(
            {"db": "bioproject", "id": ",".join(batch_ids), "retmode": "json"}
        )
        summary = fetch_json(
            summary_url,
            source_dir / f"bioproject_summary_{batch_number:03d}.json",
            args.force_download,
        )["result"]
        records.extend(summary[item] for item in summary["uids"])
    samples = pd.read_csv(
        run_dir / "environmental_metadata/mucc_v1_sample_columns_scaffold.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    crosswalk = build_crosswalk(samples, records)
    environmental_dir = run_dir / "environmental_metadata"
    output_tsv = environmental_dir / "link_mucc_v1_sequence_bioproject_sample.tsv"
    write_tsv(output_tsv, crosswalk)
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    write_tsv(source_dir / "source_file_manifest.tsv", pd.DataFrame(source_manifest_rows(source_dir, retrieved_utc)))
    mapped = int(crosswalk["sample_project_link_status"].eq("mapped_to_authoritative_NCBI_BioProject_title").sum())
    summary = {
        "lane_id": LANE_ID,
        "crosswalk": str(output_tsv),
        "expression_samples": int(len(crosswalk)),
        "mapped_to_NCBI_BioProject": mapped,
        "unmapped_preserved": int(len(crosswalk) - mapped),
        "exact_collection_datetime_or_depth_cm_mapped": 0,
        "environment_or_flux_joined": 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (source_dir / "crosswalk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
