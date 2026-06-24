#!/usr/bin/env python3
"""Validate the MethaNet atlas lane registry contract.

This is a structural gate for the source-lane control plane. It checks that the
registry has the required columns, unique lane IDs, numeric denominators, and
resolvable registered paths before status/report builders consume it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "lane_id",
    "lane_role",
    "lane_status",
    "denominator_label",
    "denominator_units",
    "source_lane_manifest",
    "functional_manifest",
    "functional_per_mag_dirs",
    "functional_warehouse_dir",
    "consolidation_cohort_run_id",
    "consolidation_output_dir",
    "esm2_artifacts_dirs",
    "glm2_artifacts_dirs",
    "gap_register",
    "claim_scope",
    "notes",
]
RECOGNIZED_OPTIONAL_COLUMNS = [
    "source_provenance_dir",
    "source_provenance_checksums",
]

REQUIRED_PATH_COLUMNS = ["source_lane_manifest", "functional_manifest"]
OPTIONAL_EXISTING_PATH_COLUMNS = [
    "functional_per_mag_dirs",
    "functional_warehouse_dir",
    "esm2_artifacts_dirs",
    "glm2_artifacts_dirs",
    "gap_register",
    "source_provenance_dir",
    "source_provenance_checksums",
]
COMPLETE_LANE_STATUSES = {"complete", "closed", "current"}
EXPECTED_ARTIFACT_MARKERS = {
    "functional_warehouse_dir": [
        "cohort_table_manifest.tsv",
        "functional_atlas.duckdb",
    ],
    "esm2_artifacts_dirs": [
        "embedding_metadata.tsv",
        "genome_embeddings.npz",
        "embedding_stats.json",
        "embedding_checkpoints/embedding_stats_partial.json",
    ],
    "glm2_artifacts_dirs": [
        "feature_glm_mag_level.tsv",
        "features/glm2_smoke_window_embedding_summary.tsv",
    ],
}
SOURCE_MANIFEST_REQUIRED_COLUMNS = ["proteome_id"]
GAP_REGISTER_REQUIRED_COLUMNS = ["proteome_id"]
CHECKSUM_LEDGER_REQUIRED_COLUMNS = ["path", "size_bytes", "md5", "sha256"]
EXTERNAL_SOURCE_MANIFEST_RECOMMENDED_COLUMNS = [
    "proteome_id",
    "mag_id",
    "source",
    "ecosystem",
    "domain",
    "mag_fasta",
    "proteome_faa",
    "match_status",
    "functional_run_include",
    "analysis_unit_type",
    "claim_scope",
    "comparability_status",
]
FUNCTIONAL_MANIFEST_REQUIRED_COLUMNS = ["proteome_id", "functional_run_include"]
BOOLEAN_COLUMNS = [
    "functional_run_include",
    "esm2_include",
    "glm2_include",
    "mbag_mag_level_include",
    "assembly_context_include",
    "embedded_final_662",
]
BOOLEAN_VALUES = {"true", "false", "1", "0", "yes", "no", "y", "n"}
TRUE_VALUES = {"true", "1", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lane-registry", type=Path, default=Path("configs/methanet_atlas_lanes.tsv"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--allow-missing-optional-paths", action="store_true")
    return parser.parse_args()


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def split_paths(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def row_label(row_idx: int, row: dict[str, str]) -> str:
    return f"row {row_idx} lane_id={row.get('lane_id') or '<missing>'}"


def validate_manifest(
    path: Path,
    label: str,
    required_columns: list[str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    fields, rows = read_tsv(path)
    missing_columns = [column for column in required_columns if column not in fields]
    if missing_columns:
        errors.append(f"{label}: manifest missing required columns: {', '.join(missing_columns)}")
        return {"rows": len(rows), "functional_include_rows": 0, "fields": fields}
    if not rows:
        warnings.append(f"{label}: manifest has no data rows")

    seen: set[str] = set()
    functional_include_rows = 0
    for idx, row in enumerate(rows, start=2):
        row_context = f"{label} row {idx}"
        proteome_id = str(row.get("proteome_id") or "").strip()
        if not proteome_id:
            errors.append(f"{row_context}: missing proteome_id")
        elif proteome_id in seen:
            errors.append(f"{row_context}: duplicate proteome_id {proteome_id}")
        else:
            seen.add(proteome_id)

        for column in BOOLEAN_COLUMNS:
            if column not in fields:
                continue
            value = str(row.get(column) or "").strip().lower()
            if value and value not in BOOLEAN_VALUES:
                errors.append(f"{row_context}: invalid boolean value for {column}: {row.get(column)}")

        if "functional_run_include" in fields and truthy(row.get("functional_run_include")):
            functional_include_rows += 1
            if row.get("match_status") == "missing_payload":
                errors.append(f"{row_context}: functional_run_include=true but match_status=missing_payload")
            for payload_column in ["mag_fasta", "proteome_faa"]:
                if payload_column in fields and not str(row.get(payload_column) or "").strip():
                    errors.append(f"{row_context}: functional_run_include=true but {payload_column} is empty")
    return {"rows": len(rows), "functional_include_rows": functional_include_rows, "fields": fields}


def validate_gap_register(
    path: Path,
    label: str,
    source_rows: list[dict[str, str]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    fields, rows = read_tsv(path)
    missing_columns = [column for column in GAP_REGISTER_REQUIRED_COLUMNS if column not in fields]
    if missing_columns:
        errors.append(f"{label}: gap register missing required columns: {', '.join(missing_columns)}")
        return {"rows": len(rows), "fields": fields}
    if not rows:
        warnings.append(f"{label}: gap register has no data rows")
        return {"rows": 0, "fields": fields}

    seen: set[str] = set()
    source_by_id = {
        str(row.get("proteome_id") or "").strip(): row
        for row in source_rows
        if str(row.get("proteome_id") or "").strip()
    }
    source_missing_payload_ids = {
        proteome_id
        for proteome_id, row in source_by_id.items()
        if str(row.get("match_status") or "").strip() == "missing_payload"
    }
    source_non_runnable_ids = {
        proteome_id
        for proteome_id, row in source_by_id.items()
        if not truthy(row.get("functional_run_include"))
    }
    for idx, row in enumerate(rows, start=2):
        row_context = f"{label} row {idx}"
        proteome_id = str(row.get("proteome_id") or "").strip()
        if not proteome_id:
            errors.append(f"{row_context}: missing proteome_id")
            continue
        if proteome_id in seen:
            errors.append(f"{row_context}: duplicate proteome_id {proteome_id}")
        seen.add(proteome_id)
        if proteome_id not in source_by_id:
            errors.append(f"{row_context}: proteome_id {proteome_id} is absent from source-lane manifest")
            continue
        if source_non_runnable_ids and proteome_id not in source_non_runnable_ids:
            errors.append(f"{row_context}: proteome_id {proteome_id} is runnable in source-lane manifest")
        if source_missing_payload_ids and proteome_id not in source_missing_payload_ids:
            errors.append(f"{row_context}: proteome_id {proteome_id} is not match_status=missing_payload")

    if source_missing_payload_ids:
        missing_from_gap = sorted(source_missing_payload_ids - seen)
        if missing_from_gap:
            errors.append(
                f"{label}: source-lane missing-payload rows absent from gap register: "
                + ", ".join(missing_from_gap[:20])
                + (" ..." if len(missing_from_gap) > 20 else "")
            )
    return {"rows": len(rows), "fields": fields}


def file_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def validate_checksum_ledger(
    repo_root: Path,
    path: Path,
    label: str,
    errors: list[str],
    warnings: list[str],
    source_provenance_dir: Path | None = None,
) -> dict[str, Any]:
    fields, rows = read_tsv(path)
    missing_columns = [column for column in CHECKSUM_LEDGER_REQUIRED_COLUMNS if column not in fields]
    if missing_columns:
        errors.append(f"{label}: checksum ledger missing required columns: {', '.join(missing_columns)}")
        return {"rows": len(rows), "fields": fields}
    if not rows:
        warnings.append(f"{label}: checksum ledger has no data rows")
        return {"rows": 0, "fields": fields}

    seen_paths: set[str] = set()
    covered_source_paths = 0
    for idx, row in enumerate(rows, start=2):
        row_context = f"{label} row {idx}"
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            errors.append(f"{row_context}: missing path")
            continue
        if raw_path in seen_paths:
            errors.append(f"{row_context}: duplicate checksum path {raw_path}")
        seen_paths.add(raw_path)
        source_file = resolve(repo_root, raw_path)
        if not source_file.exists():
            errors.append(f"{row_context}: checksum path missing on disk: {raw_path}")
            continue
        if source_provenance_dir is not None:
            try:
                source_file.resolve().relative_to(source_provenance_dir.resolve())
                covered_source_paths += 1
            except ValueError:
                errors.append(
                    f"{row_context}: checksum path is outside registered source_provenance_dir: {raw_path}"
                )
        try:
            expected_size = int(str(row.get("size_bytes") or ""))
        except ValueError:
            errors.append(f"{row_context}: size_bytes is not an integer: {row.get('size_bytes')}")
            expected_size = -1
        observed_size = source_file.stat().st_size
        if expected_size >= 0 and expected_size != observed_size:
            errors.append(f"{row_context}: size_bytes mismatch for {raw_path}: expected {expected_size}, observed {observed_size}")
        observed_md5, observed_sha256 = file_hashes(source_file)
        expected_md5 = str(row.get("md5") or "").strip().lower()
        expected_sha256 = str(row.get("sha256") or "").strip().lower()
        if expected_md5 and expected_md5 != observed_md5:
            errors.append(f"{row_context}: md5 mismatch for {raw_path}")
        if expected_sha256 and expected_sha256 != observed_sha256:
            errors.append(f"{row_context}: sha256 mismatch for {raw_path}")
    if source_provenance_dir is not None and rows and covered_source_paths == 0:
        errors.append(f"{label}: checksum ledger does not cover files under source_provenance_dir")
    return {"rows": len(rows), "fields": fields}


def has_expected_marker(path: Path, marker_paths: list[str]) -> bool:
    return any((path / marker).exists() and (path / marker).stat().st_size > 0 for marker in marker_paths)


def validate(repo_root: Path, registry: Path, allow_missing_optional_paths: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    fields, rows = read_tsv(registry)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fields]
    recognized_columns = set(REQUIRED_COLUMNS) | set(RECOGNIZED_OPTIONAL_COLUMNS)
    extra_columns = [column for column in fields if column not in recognized_columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if extra_columns:
        warnings.append(f"Unexpected columns present: {', '.join(extra_columns)}")

    seen: set[str] = set()
    lane_ids: list[str] = []
    for idx, row in enumerate(rows, start=2):
        label = row_label(idx, row)
        lane_id = str(row.get("lane_id") or "").strip()
        if not lane_id:
            errors.append(f"{label}: missing lane_id")
        elif lane_id in seen:
            errors.append(f"{label}: duplicate lane_id")
        else:
            seen.add(lane_id)
            lane_ids.append(lane_id)

        denominator = -1
        try:
            denominator = int(str(row.get("denominator_units") or ""))
            if denominator < 0:
                errors.append(f"{label}: denominator_units must be non-negative")
        except ValueError:
            errors.append(f"{label}: denominator_units is not an integer")

        manifest_summaries: dict[str, dict[str, Any]] = {}
        source_manifest_rows: list[dict[str, str]] = []
        for column in REQUIRED_PATH_COLUMNS:
            value = str(row.get(column) or "").strip()
            if not value:
                errors.append(f"{label}: missing required path column {column}")
                continue
            for item in split_paths(value):
                resolved = resolve(repo_root, item)
                if not resolved.exists():
                    errors.append(f"{label}: missing registered path in {column}: {item}")
                    continue
                required_manifest_columns = (
                    SOURCE_MANIFEST_REQUIRED_COLUMNS
                    if column == "source_lane_manifest"
                    else FUNCTIONAL_MANIFEST_REQUIRED_COLUMNS
                )
                summary = validate_manifest(
                    resolved,
                    f"{label} {column}={item}",
                    required_manifest_columns,
                    errors,
                    warnings,
                )
                manifest_summaries[column] = summary
                if column == "source_lane_manifest":
                    _, source_manifest_rows = read_tsv(resolved)
        functional_summary = manifest_summaries.get("functional_manifest", {})
        source_summary = manifest_summaries.get("source_lane_manifest", {})
        source_rows = int(source_summary.get("rows", 0))
        functional_rows = int(functional_summary.get("rows", 0))
        functional_include_rows = int(functional_summary.get("functional_include_rows", 0))
        lane_role = str(row.get("lane_role") or "").strip().lower()
        source_fields = set(source_summary.get("fields", []))
        if lane_role.startswith("external") and source_fields:
            missing_external_fields = [
                column for column in EXTERNAL_SOURCE_MANIFEST_RECOMMENDED_COLUMNS if column not in source_fields
            ]
            if missing_external_fields:
                warnings.append(
                    f"{label}: external source-lane manifest missing normalized handoff columns: "
                    f"{', '.join(missing_external_fields)}"
                )
        if lane_role.startswith("external") and denominator >= 0 and source_rows > denominator:
            errors.append(
                f"{label}: denominator_units {denominator} is smaller than source lane manifest rows {source_rows}"
            )
        if denominator >= 0 and functional_rows > denominator:
            errors.append(
                f"{label}: denominator_units {denominator} is smaller than functional manifest rows {functional_rows}"
            )
        if denominator >= 0 and functional_include_rows > denominator:
            errors.append(
                f"{label}: denominator_units {denominator} is smaller than functional include rows {functional_include_rows}"
            )

        for column in OPTIONAL_EXISTING_PATH_COLUMNS:
            value = str(row.get(column) or "").strip()
            if not value:
                continue
            for item in split_paths(value):
                resolved = resolve(repo_root, item)
                if not resolved.exists():
                    message = f"{label}: missing optional registered path in {column}: {item}"
                    if allow_missing_optional_paths:
                        warnings.append(message)
                    else:
                        errors.append(message)
                    continue
                markers = EXPECTED_ARTIFACT_MARKERS.get(column, [])
                lane_status = str(row.get("lane_status") or "").strip().lower()
                if markers and lane_status in COMPLETE_LANE_STATUSES and not has_expected_marker(resolved, markers):
                    warnings.append(
                        f"{label}: {column} path lacks expected complete-lane marker "
                        f"({', '.join(markers)}): {item}"
                    )
                if column == "gap_register" and source_manifest_rows:
                    validate_gap_register(
                        resolved,
                        f"{label} {column}={item}",
                        source_manifest_rows,
                        errors,
                        warnings,
                    )
                if column == "source_provenance_checksums":
                    source_provenance_dirs = [
                        resolve(repo_root, item)
                        for item in split_paths(str(row.get("source_provenance_dir") or "").strip())
                    ]
                    source_provenance_dir = source_provenance_dirs[0] if len(source_provenance_dirs) == 1 else None
                    if len(source_provenance_dirs) > 1:
                        warnings.append(
                            f"{label}: multiple source_provenance_dir values registered; "
                            "checksum paths will be validated on disk but not constrained to one provenance root"
                        )
                    validate_checksum_ledger(
                        repo_root,
                        resolved,
                        f"{label} {column}={item}",
                        errors,
                        warnings,
                        source_provenance_dir,
                    )

    return {
        "registry": str(registry),
        "row_count": len(rows),
        "lane_ids": lane_ids,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    registry = resolve(repo_root, str(args.lane_registry))
    if not registry.exists():
        raise SystemExit(f"Lane registry missing: {registry}")
    report = validate(repo_root, registry, args.allow_missing_optional_paths)
    if args.output_json:
        output = resolve(repo_root, str(args.output_json))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        return 1
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Validated {report['row_count']} atlas lane registry rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
