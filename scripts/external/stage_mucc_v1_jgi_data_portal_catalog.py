#!/usr/bin/env python3
"""Stage public JGI Data Portal catalog evidence for MUCC expression labels.

The public JGI Data Portal exposes source-label-specific records for this Old
Woman Creek metatranscriptome cohort.  It is useful identity and processing
provenance, but it is not a substitute for the unrecovered publication sample
table: the catalog does not report collection depth, field time, paired
chemistry, or a methane-flux association.  This script preserves that boundary
in every output row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
JGI_DATA_PORTAL_SEARCH_URL = "https://files.jgi.doe.gov/search/"
JGI_AWARD_ID = 504205
JGI_AWARD_DOI = "10.46936/10.25585/60001190"
CLAIM_BOUNDARY = (
    "JGI Data Portal catalog records establish source-label-to-named-JGI-record identity, "
    "processing/catalog provenance, and source-record geolocation only. They do not establish "
    "collection datetime, measured depth in cm, environmental/flux pairing, ecological association, "
    "measured methane flux, final MRV score/A-E tier, crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Maximum concurrent public JGI Data Portal queries (default: 3).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional deterministic smoke-test limit; zero resolves every source expression label.",
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


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def candidate_labels(value: str) -> list[tuple[str, str]]:
    candidates = [(value, "exact_JGI_Data_Portal_label")]
    if value.startswith("July_"):
        candidates.append(("Jul_" + value.removeprefix("July_"), "deterministic_July_to_Jul"))
    if value.startswith("Jul_"):
        candidates.append(("July_" + value.removeprefix("Jul_"), "deterministic_Jul_to_July"))
    return candidates


def cache_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def search_url(label: str) -> str:
    return JGI_DATA_PORTAL_SEARCH_URL + "?" + urllib.parse.urlencode(
        {"q": label, "t": "advanced", "x": "10"}
    )


def fetch_bytes(url: str, destination: Path, force: bool) -> tuple[bytes, str]:
    """Fetch and cache one public API response with bounded curl timeouts."""

    if destination.is_file() and destination.stat().st_size and not force:
        return destination.read_bytes(), url
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--connect-timeout",
        "10",
        "--max-time",
        "45",
        "--user-agent",
        "MethaNet MUCC JGI Data Portal catalog recovery",
        "--output",
        str(temporary),
        "--write-out",
        "%{url_effective}",
        url,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip() or f"curl exit code {completed.returncode}"
        raise RuntimeError(detail)
    if not temporary.is_file() or not temporary.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"empty JGI Data Portal response from {url}")
    temporary.replace(destination)
    return destination.read_bytes(), completed.stdout.strip() or url


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def _record_file_stats(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Return indexed file totals, bytes, and PURGED/non-PURGED/unknown counts."""

    files = record.get("files")
    if not isinstance(files, list):
        files = []
    file_count = len(files)
    bytes_total = 0
    purged = 0
    non_purged = 0
    unknown = 0
    for item in files:
        if not isinstance(item, dict):
            unknown += 1
            continue
        size = item.get("file_size")
        try:
            bytes_total += int(size) if size is not None else 0
        except (TypeError, ValueError):
            pass
        status = _as_text(item.get("file_status"))
        if status == "PURGED":
            purged += 1
        elif status:
            non_purged += 1
        else:
            unknown += 1
    if not file_count:
        try:
            file_count = int(record.get("file_total") or 0)
        except (TypeError, ValueError):
            file_count = 0
    if not bytes_total:
        try:
            bytes_total = int(record.get("fileSize") or 0)
        except (TypeError, ValueError):
            bytes_total = 0
    return file_count, bytes_total, purged, non_purged, unknown


def _coordinates(records: list[dict[str, Any]]) -> tuple[str, str, str]:
    values: set[tuple[str, str]] = set()
    for record in records:
        files = record.get("files")
        if not isinstance(files, list):
            continue
        for item in files:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            segment = metadata.get("sow_segment")
            if not isinstance(segment, dict):
                continue
            latitude = _as_text(segment.get("latitude_of_sample_collection"))
            longitude = _as_text(segment.get("longitude_of_sample_collection"))
            if latitude or longitude:
                values.add((latitude, longitude))
    if len(values) == 1:
        latitude, longitude = next(iter(values))
        return latitude, longitude, "one_coordinate_pair_reported_by_JGI_catalog"
    if not values:
        return "", "", "not_reported_by_JGI_Data_Portal_catalog"
    return "", "", "conflicting_coordinate_pairs_reported_by_JGI_catalog"


def record_values(records: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({_as_text(item.get(key)) for item in records if _as_text(item.get(key))})


def _exact_records(
    payload: dict[str, Any], source_label: str
) -> tuple[list[dict[str, Any]], str, str]:
    """Select a source-label-specific JGI analysis and expression record pair.

    JGI has two verified annotation-title conventions in this cohort.  The
    invariant is instead the award, product type, full source label, and a
    unique pair split by whether the catalog gives the analysis a taxon ID.
    """

    organisms = payload.get("organisms")
    if not isinstance(organisms, list):
        return [], "unresolved_invalid_JGI_Data_Portal_response", ""
    for candidate, method in candidate_labels(source_label):
        matches = [
            item
            for item in organisms
            if isinstance(item, dict)
            and _as_text(item.get("proposal_id")) == str(JGI_AWARD_ID)
            and _as_text(item.get("product_search_category")) == "Metatranscriptome"
            and candidate in _as_text(item.get("name"))
        ]
        analysis = [item for item in matches if _as_text(item.get("portal_detail_id"))]
        expression = [item for item in matches if not _as_text(item.get("portal_detail_id"))]
        if len(analysis) == 1 and len(expression) == 1:
            return [expression[0], analysis[0]], method, candidate
        if matches:
            return [], "unresolved_incomplete_exact_JGI_Data_Portal_record_pair", ""
    return [], "unresolved_no_exact_JGI_Data_Portal_record_pair", ""


def parse_catalog_record(sample: dict[str, str], payload: dict[str, Any], query_url: str) -> dict[str, str]:
    source_label = sample["source_sample_column"]
    records, method, matched_label = _exact_records(payload, source_label)
    base = {
        "lane_id": LANE_ID,
        "sample_id": sample["sample_id"],
        "source_sample_column": source_label,
        "jgi_data_portal_query_url": query_url,
        "jgi_data_portal_award_id": str(JGI_AWARD_ID),
        "jgi_data_portal_award_doi": JGI_AWARD_DOI,
        "collection_datetime_status": "not_reported_by_JGI_Data_Portal_catalog",
        "depth_cm_join_status": "not_reported_by_JGI_Data_Portal_catalog",
        "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if not records:
        return {
            **base,
            "jgi_data_portal_label_mapping_method": method,
            "jgi_data_portal_identity_status": method,
            "collection_datetime_status": method,
            "depth_cm_join_status": method,
            "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
            "jgi_data_portal_matched_catalog_label": "",
            "jgi_data_portal_expression_record_id": "",
            "jgi_data_portal_annotation_record_id": "",
            "jgi_data_portal_annotation_taxon_oid": "",
            "jgi_data_portal_visibility": "",
            "jgi_data_portal_data_utilization_status": "",
            "jgi_data_portal_processing_status": "",
            "jgi_data_portal_work_completion_date": "",
            "jgi_data_portal_indexed_file_count": "0",
            "jgi_data_portal_indexed_file_bytes": "0",
            "jgi_data_portal_purged_file_count": "0",
            "jgi_data_portal_nonpurged_file_count": "0",
            "jgi_data_portal_unknown_file_status_count": "0",
            "jgi_data_portal_file_access_status": "unresolved_no_exact_catalog_record_pair",
            "jgi_data_portal_latitude": "",
            "jgi_data_portal_longitude": "",
            "jgi_data_portal_coordinate_status": "not_reported_by_JGI_Data_Portal_catalog",
            "next_validation_action": "recover the authoritative publication Table S4-equivalent sample metadata and explicit environmental/flux links",
        }

    expression = next(item for item in records if not _as_text(item.get("portal_detail_id")))
    annotation = next(item for item in records if _as_text(item.get("portal_detail_id")))
    totals = [_record_file_stats(item) for item in records]
    file_count = sum(item[0] for item in totals)
    bytes_total = sum(item[1] for item in totals)
    purged = sum(item[2] for item in totals)
    non_purged = sum(item[3] for item in totals)
    unknown = sum(item[4] for item in totals)
    latitude, longitude, coordinate_status = _coordinates(records)
    visibility = record_values(records, "visibility")
    utilization = record_values(records, "data_utilization_status")
    processing = record_values(records, "status")
    completion_dates = record_values(records, "work_completion_date")
    annotation_taxon = _as_text(annotation.get("portal_detail_id"))
    access_status = (
        "all_indexed_assets_marked_PURGED; authenticated_JGI_download_required"
        if file_count and purged == file_count
        else "catalog_file_statuses_reported; authenticated_JGI_download_required"
    )
    return {
        **base,
        "jgi_data_portal_label_mapping_method": method,
        "jgi_data_portal_identity_status": "exact_source_label_to_JGI_Data_Portal_record_pair",
        "jgi_data_portal_matched_catalog_label": matched_label,
        "jgi_data_portal_expression_record_id": _as_text(expression.get("id")),
        "jgi_data_portal_annotation_record_id": _as_text(annotation.get("id")),
        "jgi_data_portal_annotation_taxon_oid": annotation_taxon,
        "jgi_data_portal_visibility": ";".join(visibility),
        "jgi_data_portal_data_utilization_status": ";".join(utilization),
        "jgi_data_portal_processing_status": ";".join(processing),
        "jgi_data_portal_work_completion_date": ";".join(completion_dates),
        "jgi_data_portal_indexed_file_count": str(file_count),
        "jgi_data_portal_indexed_file_bytes": str(bytes_total),
        "jgi_data_portal_purged_file_count": str(purged),
        "jgi_data_portal_nonpurged_file_count": str(non_purged),
        "jgi_data_portal_unknown_file_status_count": str(unknown),
        "jgi_data_portal_file_access_status": access_status,
        "jgi_data_portal_latitude": latitude,
        "jgi_data_portal_longitude": longitude,
        "jgi_data_portal_coordinate_status": coordinate_status,
        "next_validation_action": "recover the authoritative publication Table S4-equivalent sample metadata, then reconcile each sequence label to depth, chemistry, and flux windows",
    }


def error_record(sample: dict[str, str], query_url: str, error: Exception) -> dict[str, str]:
    payload = {"organisms": []}
    row = parse_catalog_record(sample, payload, query_url)
    row["jgi_data_portal_identity_status"] = "unresolved_JGI_Data_Portal_recovery_error"
    row["jgi_data_portal_label_mapping_method"] = f"{type(error).__name__}: {error}"
    row["jgi_data_portal_file_access_status"] = "unresolved_JGI_Data_Portal_recovery_error"
    row["collection_datetime_status"] = "unresolved_JGI_Data_Portal_recovery_error"
    row["depth_cm_join_status"] = "unresolved_JGI_Data_Portal_recovery_error"
    return row


def source_manifest_rows(source_dir: Path, retrieved_utc: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((source_dir / "responses").glob("*.json")):
        rows.append(
            {
                "lane_id": LANE_ID,
                "source_id": "JGI_Data_Portal_public_search_catalog",
                "artifact_role": "JGI_Data_Portal_source_label_query_response",
                "source_url": "retained_in_crosswalk_source_columns",
                "local_path": str(path),
                "bytes": str(path.stat().st_size),
                "sha256": sha256(path),
                "retrieved_utc": retrieved_utc,
                "validation_status": "retrieved_and_JSON_parseable",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    source_dir = run_dir / "source_audit/jgi_data_portal_catalog"
    samples = pd.read_csv(
        run_dir / "environmental_metadata/link_mucc_v1_sequence_bioproject_sample.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )[["sample_id", "source_sample_column"]]
    if args.max_samples:
        samples = samples.iloc[: args.max_samples].copy()
    if samples["sample_id"].duplicated().any():
        raise ValueError("source expression sample roster has duplicate sample_id values")

    def recover(sample: dict[str, str]) -> tuple[str, dict[str, str], str | None]:
        label = sample["source_sample_column"]
        urls: list[str] = []
        organisms: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            for candidate, _ in candidate_labels(label):
                url = search_url(candidate)
                urls.append(url)
                suffix = "" if candidate == label else f"__{cache_name(candidate)}"
                destination = source_dir / "responses" / f"{cache_name(sample['sample_id'])}{suffix}.json"
                try:
                    payload, _ = fetch_bytes(url, destination, args.force_download)
                    parsed = json.loads(payload.decode("utf-8"))
                    if not isinstance(parsed, dict):
                        raise ValueError("JGI Data Portal response is not a JSON object")
                    query_organisms = parsed.get("organisms")
                    if isinstance(query_organisms, list):
                        organisms.extend(
                            item for item in query_organisms if isinstance(item, dict)
                        )
                except Exception as error:
                    errors.append(f"{candidate}: {type(error).__name__}: {error}")
            if not organisms and errors:
                raise RuntimeError("; ".join(errors))
            row = parse_catalog_record(sample, {"organisms": organisms}, ";".join(urls))
            return sample["sample_id"], row, "; ".join(errors) if errors else None
        except Exception as error:
            return (
                sample["sample_id"],
                error_record(sample, ";".join(urls), error),
                f"{type(error).__name__}: {error}",
            )

    source_rows = samples.to_dict("records")
    rows: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(recover, sample): sample["sample_id"] for sample in source_rows}
        for future in as_completed(pending):
            sample_id, row, error = future.result()
            rows[sample_id] = row
            if error:
                errors[sample_id] = error
    crosswalk = pd.DataFrame([rows[sample_id] for sample_id in samples["sample_id"]])
    if len(crosswalk) != len(samples) or crosswalk["sample_id"].duplicated().any():
        raise ValueError("JGI Data Portal crosswalk must retain every source expression sample exactly once")
    output = run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_data_portal.tsv"
    write_tsv(output, crosswalk)
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    write_tsv(source_dir / "source_file_manifest.tsv", pd.DataFrame(source_manifest_rows(source_dir, retrieved_utc)))
    (source_dir / "recovery_errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
    exact = int(
        crosswalk["jgi_data_portal_identity_status"].eq(
            "exact_source_label_to_JGI_Data_Portal_record_pair"
        ).sum()
    )
    summary = {
        "lane_id": LANE_ID,
        "crosswalk": str(output),
        "expression_samples": int(len(crosswalk)),
        "exact_JGI_Data_Portal_record_pairs": exact,
        "unresolved_or_unmapped": int(len(crosswalk) - exact),
        "indexed_files_across_exact_pairs": int(
            pd.to_numeric(crosswalk["jgi_data_portal_indexed_file_count"], errors="coerce").fillna(0).sum()
        ),
        "purged_indexed_files_across_exact_pairs": int(
            pd.to_numeric(crosswalk["jgi_data_portal_purged_file_count"], errors="coerce").fillna(0).sum()
        ),
        "exact_collection_datetime_or_depth_cm_mapped": 0,
        "environment_or_flux_joined": 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (source_dir / "crosswalk_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
