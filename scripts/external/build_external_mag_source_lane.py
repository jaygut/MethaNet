#!/usr/bin/env python3
"""Build reusable MethaNet external MAG source-lane manifests.

This adapter starts from a normalized source manifest, such as a dataset-local
ready FASTA registry, and emits two standard handoff artifacts:

- a source-lane manifest that preserves every ready and gap row;
- a functional-run manifest compatible with the existing Apollo-3 array worker.

The source-specific work should happen before this script: parsing a paper
spreadsheet, resolving accessions, or downloading FASTAs. Everything after this
point is source-agnostic and keyed by `proteome_id`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRUE_VALUES = {"true", "1", "yes", "y"}
STANDARD_SOURCE_FIELDS = [
    "external_dataset_id",
    "source_project_id",
    "proteome_id",
    "mag_id",
    "source",
    "ecosystem",
    "domain",
    "source_group",
    "source_sample_ids",
    "mag_fasta",
    "proteome_faa",
    "local_fna_path",
    "local_faa_path",
    "local_ffn_path",
    "local_gff_path",
    "payload_status",
    "protein_prediction_status",
    "protein_count",
    "protein_qc_tier",
    "match_status",
    "functional_run_include",
    "esm2_include",
    "glm2_include",
    "analysis_unit_type",
    "mbag_mag_level_include",
    "claim_scope",
    "comparability_status",
    "denominator_status",
    "metadata_mapping_status",
    "mapped_ncbi_biosamples",
    "mapped_ncbi_bioprojects",
    "download_state",
    "gap_reason",
    "recommended_action",
    "total_bp",
    "contigs",
    "n50_bp",
    "gc_pct",
]
FUNCTIONAL_FIELDS = [
    "proteome_id",
    "mag_id",
    "source",
    "ecosystem",
    "domain",
    "source_group",
    "source_fasta_filename",
    "mag_fasta",
    "proteome_faa",
    "local_ffn_path",
    "local_gff_path",
    "protein_count",
    "protein_qc_tier",
    "match_status",
    "functional_run_include",
    "analysis_unit_type",
    "mbag_mag_level_include",
    "claim_scope",
    "comparability_status",
    "denominator_status",
    "metadata_mapping_status",
    "mapped_ncbi_biosamples",
    "mapped_ncbi_bioprojects",
    "source_sample_ids",
    "total_bp",
    "contigs",
    "n50_bp",
    "gc_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--proteome-manifest", type=Path)
    parser.add_argument("--gap-register", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-project-id", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--ecosystem", default="mangrove_sediment")
    parser.add_argument("--analysis-unit-type", default="mag_bin")
    parser.add_argument("--denominator-status", default="")
    parser.add_argument("--output-source-lane", type=Path, required=True)
    parser.add_argument("--output-functional-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--include-gaps",
        action="store_true",
        help="Append gap rows from --gap-register to the source-lane and functional manifests.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_unique_proteome_ids(rows: list[dict[str, str]], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    missing = 0
    for row in rows:
        proteome_id = str(row.get("proteome_id") or "").strip()
        if not proteome_id:
            missing += 1
        elif proteome_id in seen:
            duplicates.add(proteome_id)
        else:
            seen.add(proteome_id)
    if missing or duplicates:
        parts = []
        if missing:
            parts.append(f"{missing} rows missing proteome_id")
        if duplicates:
            parts.append("duplicate proteome_id values: " + ", ".join(sorted(duplicates)))
        raise SystemExit(f"{label} has invalid proteome_id values: {'; '.join(parts)}")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def file_ready(repo_root: Path, value: str | None) -> bool:
    path = resolve_path(repo_root, value)
    return bool(path and path.is_file() and path.stat().st_size > 0)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def protein_qc_tier(value: str | None) -> str:
    try:
        count = int(float(value or 0))
    except ValueError:
        count = 0
    if count <= 0:
        return "protein_count_missing"
    if count < 100:
        return "qc_warning_lt100_proteins"
    if count < 500:
        return "qc_warning_lt500_proteins"
    if count < 1000:
        return "qc_review_lt1000_proteins"
    return "protein_count_plausible"


def first_value(row: dict[str, str], *cols: str) -> str:
    for col in cols:
        value = row.get(col, "")
        if value:
            return value
    return ""


def normalize_analysis_unit_type(row: dict[str, str], default: str) -> str:
    return (row.get("analysis_unit_type") or default or "mag_bin").strip()


def normalize_ready_row(repo_root: Path, row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    source = args.source or row.get("source") or args.dataset_id
    fna = first_value(row, "mag_fasta", "local_fna_path", "fasta_path")
    faa = first_value(row, "proteome_faa", "local_faa_path", "faa_path")
    ffn = first_value(row, "local_ffn_path", "genes_ffn", "ffn_path")
    gff = first_value(row, "local_gff_path", "genes_gff", "gff_path")
    fna_ready = file_ready(repo_root, fna)
    faa_ready = file_ready(repo_root, faa)
    gff_ready = file_ready(repo_root, gff)
    protein_count = first_value(row, "protein_count", "predicted_proteins")
    protein_status = row.get("protein_prediction_status") or ("ready" if faa_ready else "pending")
    analysis_unit_type = normalize_analysis_unit_type(row, args.analysis_unit_type)
    functional_include = fna_ready and faa_ready and analysis_unit_type == "mag_bin"
    if fna_ready and faa_ready:
        payload_status = "fna_and_faa_ready"
    elif fna_ready:
        payload_status = row.get("payload_status") or "fna_ready_faa_missing_or_empty"
    else:
        payload_status = row.get("payload_status") or "missing_fna"
    missing = []
    if not fna_ready:
        missing.append("mag_fasta")
    if not faa_ready:
        missing.append("proteome_faa")
    match_status = "matched" if fna_ready and faa_ready else ("fna_only" if fna_ready else "missing_payload")
    comparability_status = (
        "ready_for_functional_annotation_pending_qc_reconciliation"
        if functional_include
        else "blocked_missing_" + "_".join(missing or ["required_payload"])
    )
    return {
        **row,
        "external_dataset_id": row.get("external_dataset_id") or args.dataset_id,
        "source_project_id": row.get("source_project_id") or args.source_project_id,
        "source": source,
        "ecosystem": row.get("ecosystem") or args.ecosystem,
        "source_group": row.get("source_group") or row.get("site_time_key", ""),
        "source_sample_ids": row.get("source_sample_ids") or row.get("site_time_key", ""),
        "mag_fasta": fna,
        "proteome_faa": faa,
        "local_fna_path": row.get("local_fna_path") or fna,
        "local_faa_path": row.get("local_faa_path") or faa,
        "local_ffn_path": ffn,
        "local_gff_path": gff,
        "payload_status": payload_status,
        "protein_prediction_status": protein_status,
        "protein_count": protein_count,
        "protein_qc_tier": row.get("protein_qc_tier") or protein_qc_tier(protein_count),
        "match_status": match_status,
        "functional_run_include": "true" if functional_include else "false",
        "esm2_include": "true" if faa_ready else "false",
        "glm2_include": "true" if fna_ready and faa_ready and gff_ready else "false",
        "analysis_unit_type": analysis_unit_type,
        "mbag_mag_level_include": "true" if functional_include else "false",
        "claim_scope": row.get("claim_scope") or "MAG/proteome functional potential after QC and annotation",
        "comparability_status": comparability_status,
        "denominator_status": row.get("denominator_status") or args.denominator_status,
        "metadata_mapping_status": row.get("metadata_mapping_status", ""),
        "mapped_ncbi_biosamples": row.get("mapped_ncbi_biosamples", ""),
        "mapped_ncbi_bioprojects": row.get("mapped_ncbi_bioprojects", ""),
        "download_state": row.get("download_state") or ("downloaded" if fna_ready else ""),
        "gap_reason": "",
        "recommended_action": row.get("recommended_action", ""),
        "total_bp": first_value(row, "total_bp", "downloaded_total_bp", "genome_size_bp"),
        "contigs": first_value(row, "contigs", "downloaded_contigs", "scaffolds"),
        "n50_bp": first_value(row, "n50_bp", "downloaded_n50_bp"),
        "gc_pct": first_value(row, "gc_pct", "downloaded_gc_pct"),
    }


def merge_proteome_rows(rows: list[dict[str, str]], proteome_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    require_unique_proteome_ids(proteome_rows, "proteome manifest")
    by_id = {row["proteome_id"]: row for row in proteome_rows if row.get("proteome_id")}
    merged: list[dict[str, str]] = []
    for row in rows:
        proteome_id = row.get("proteome_id", "")
        if proteome_id in by_id:
            merged.append({**row, **{k: v for k, v in by_id[proteome_id].items() if v != ""}})
        else:
            merged.append(row)
    return merged


def normalize_gap_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    source = args.source or row.get("source") or args.dataset_id
    return {
        **row,
        "external_dataset_id": row.get("external_dataset_id") or args.dataset_id,
        "source_project_id": row.get("source_project_id") or args.source_project_id,
        "source": source,
        "ecosystem": row.get("ecosystem") or args.ecosystem,
        "domain": row.get("domain", ""),
        "source_group": row.get("source_group", ""),
        "source_sample_ids": row.get("source_sample_ids", ""),
        "mag_fasta": row.get("mag_fasta", ""),
        "proteome_faa": row.get("proteome_faa", ""),
        "local_fna_path": row.get("local_fna_path", ""),
        "local_faa_path": row.get("local_faa_path", ""),
        "local_ffn_path": row.get("local_ffn_path", ""),
        "local_gff_path": row.get("local_gff_path", ""),
        "payload_status": "gap_no_local_fna",
        "protein_prediction_status": "blocked_no_fna",
        "protein_count": "",
        "protein_qc_tier": "protein_count_missing",
        "match_status": "missing_payload",
        "functional_run_include": "false",
        "esm2_include": "false",
        "glm2_include": "false",
        "analysis_unit_type": row.get("analysis_unit_type") or args.analysis_unit_type,
        "mbag_mag_level_include": "false",
        "claim_scope": "not currently analyzable; payload gap preserved for denominator accounting",
        "comparability_status": "blocked_" + (row.get("gap_reason") or "missing_payload"),
        "denominator_status": row.get("denominator_status") or args.denominator_status,
        "metadata_mapping_status": row.get("metadata_mapping_status", ""),
        "download_state": row.get("download_state", "not_logged"),
        "gap_reason": row.get("gap_reason", "missing_payload"),
        "recommended_action": row.get("recommended_action", "resolve_payload_or_keep_gap"),
    }


def ordered_fields(rows: list[dict[str, Any]], base_fields: list[str]) -> list[str]:
    extras: list[str] = []
    seen = set(base_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    return base_fields + extras


def functional_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: row.get(field, "") for field in FUNCTIONAL_FIELDS},
        "source_fasta_filename": row.get("source_fasta_filename") or Path(str(row.get("mag_fasta", ""))).name,
    }


def build_summary(rows: list[dict[str, Any]], functional_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "source_lane_rows": len(rows),
        "functional_manifest_rows": len(functional_rows),
        "functional_run_include": sum(truthy(row.get("functional_run_include")) for row in functional_rows),
        "esm2_include": sum(truthy(row.get("esm2_include")) for row in rows),
        "glm2_include": sum(truthy(row.get("glm2_include")) for row in rows),
        "payload_status_counts": dict(Counter(str(row.get("payload_status", "")) for row in rows)),
        "match_status_counts": dict(Counter(str(row.get("match_status", "")) for row in rows)),
        "gap_reason_counts": dict(Counter(str(row.get("gap_reason", "")) for row in rows if row.get("gap_reason"))),
        "outputs": {
            "source_lane": str(args.output_source_lane),
            "functional_manifest": str(args.output_functional_manifest),
        },
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    input_manifest = args.input_manifest if args.input_manifest.is_absolute() else repo_root / args.input_manifest
    raw_ready_rows = read_tsv(input_manifest)
    require_unique_proteome_ids(raw_ready_rows, "input manifest")
    if args.proteome_manifest:
        proteome_path = args.proteome_manifest if args.proteome_manifest.is_absolute() else repo_root / args.proteome_manifest
        raw_ready_rows = merge_proteome_rows(raw_ready_rows, read_tsv(proteome_path))
    ready_rows = [normalize_ready_row(repo_root, row, args) for row in raw_ready_rows]
    gap_rows: list[dict[str, Any]] = []
    if args.include_gaps and args.gap_register:
        gap_path = args.gap_register if args.gap_register.is_absolute() else repo_root / args.gap_register
        raw_gap_rows = read_tsv(gap_path)
        require_unique_proteome_ids(raw_gap_rows, "gap register")
        gap_rows = [normalize_gap_row(row, args) for row in raw_gap_rows]
    rows = ready_rows + gap_rows
    if not rows:
        raise SystemExit("No rows found for source-lane manifest.")
    require_unique_proteome_ids(rows, "combined source-lane manifest")

    source_fields = ordered_fields(rows, STANDARD_SOURCE_FIELDS)
    functional_rows = [functional_row(row) for row in rows]
    output_source = args.output_source_lane if args.output_source_lane.is_absolute() else repo_root / args.output_source_lane
    output_functional = (
        args.output_functional_manifest
        if args.output_functional_manifest.is_absolute()
        else repo_root / args.output_functional_manifest
    )
    write_tsv(output_source, rows, source_fields)
    write_tsv(output_functional, functional_rows, FUNCTIONAL_FIELDS)
    summary = build_summary(rows, functional_rows, args)
    if args.summary_json:
        summary_path = args.summary_json if args.summary_json.is_absolute() else repo_root / args.summary_json
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
