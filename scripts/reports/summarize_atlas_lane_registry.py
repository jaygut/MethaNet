#!/usr/bin/env python3
"""Summarize MethaNet atlas lane readiness from a registry TSV.

The lane registry is the source of truth for multi-view atlas denominators and
artifact locations. This script inspects manifests and derived outputs without
modifying them, producing a compact status table suitable for reports,
dashboards, and handoff notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRUE_VALUES = {"true", "1", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lane-registry", type=Path, default=Path("configs/methanet_atlas_lanes.tsv"))
    parser.add_argument("--output-tsv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def resolve(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def resolve_many(repo_root: Path, value: str | None) -> list[Path]:
    if not value:
        return []
    return [path for item in str(value).split(";") if item.strip() for path in [resolve(repo_root, item.strip())] if path]


def read_tsv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def count_include(rows: list[dict[str, str]], col: str) -> int:
    if not rows:
        return 0
    if col not in rows[0]:
        return len(rows)
    return sum(1 for row in rows if truthy(row.get(col)))


def latest_run_status(proteome_dir: Path) -> str:
    runs = sorted([path for path in proteome_dir.iterdir() if path.is_dir()])
    if not runs:
        return "not_started"
    complete_runs = [path for path in runs if (path / "COMPLETE").exists() and (path / "curated/run_record.json").exists()]
    if complete_runs:
        return "complete"
    latest_run = runs[-1]
    if (latest_run / "FAILED").exists():
        return "failed"
    return "partial"


def functional_status_counts(per_mag_dirs: list[Path], manifest_rows: list[dict[str, str]]) -> Counter[str]:
    status_by_id: dict[str, str] = {}
    for per_mag_dir in per_mag_dirs:
        if not per_mag_dir.exists():
            continue
        for proteome_dir in sorted(path for path in per_mag_dir.iterdir() if path.is_dir()):
            proteome_id = proteome_dir.name
            status = latest_run_status(proteome_dir)
            previous = status_by_id.get(proteome_id)
            if previous == "complete":
                continue
            status_by_id[proteome_id] = status

    counts: Counter[str] = Counter()
    if manifest_rows and "functional_run_include" in manifest_rows[0]:
        expected_rows = [row for row in manifest_rows if truthy(row.get("functional_run_include"))]
    else:
        expected_rows = manifest_rows
    expected_ids = [row.get("proteome_id", "") for row in expected_rows if row.get("proteome_id")]
    if not expected_ids:
        counts.update(status_by_id.values())
        return counts
    for proteome_id in expected_ids:
        counts[status_by_id.get(proteome_id, "not_started")] += 1
    return counts


def functional_attempt_audit(per_mag_dirs: list[Path]) -> dict[str, int]:
    complete_attempts_by_id: Counter[str] = Counter()
    failed_attempts = 0
    partial_attempts = 0
    for per_mag_dir in per_mag_dirs:
        if not per_mag_dir.exists():
            continue
        for proteome_dir in sorted(path for path in per_mag_dir.iterdir() if path.is_dir()):
            proteome_id = proteome_dir.name
            for run_dir in sorted(path for path in proteome_dir.iterdir() if path.is_dir()):
                if (run_dir / "COMPLETE").exists() and (run_dir / "curated/run_record.json").exists():
                    complete_attempts_by_id[proteome_id] += 1
                elif (run_dir / "FAILED").exists():
                    failed_attempts += 1
                else:
                    partial_attempts += 1
    complete_attempts = sum(complete_attempts_by_id.values())
    duplicate_complete_attempts = sum(max(count - 1, 0) for count in complete_attempts_by_id.values())
    duplicate_complete_proteome_ids = sum(1 for count in complete_attempts_by_id.values() if count > 1)
    return {
        "functional_complete_run_attempts": complete_attempts,
        "functional_failed_run_attempts": failed_attempts,
        "functional_partial_run_attempts": partial_attempts,
        "duplicate_complete_attempts": duplicate_complete_attempts,
        "duplicate_complete_proteome_ids": duplicate_complete_proteome_ids,
    }


def count_esm2_units(paths: list[Path], expected_ids: set[str] | None = None) -> tuple[int, str]:
    metadata_ids: set[str] = set()
    stat_total = 0
    evidence: list[str] = []
    for path in paths:
        ids: set[str] = set()
        for metadata_path, evidence_label in [
            (path / "embedding_metadata.tsv", "embedding_metadata.tsv"),
            (path / "embedding_checkpoints/checkpoint_metadata.tsv", "checkpoint_metadata.tsv"),
        ]:
            metadata = read_tsv(metadata_path)
            if not metadata:
                continue
            path_ids = {
                row.get("proteome_id") or row.get("sample")
                for row in metadata
                if row.get("proteome_id") or row.get("sample")
            }
            if expected_ids is not None:
                path_ids = path_ids & expected_ids
            ids.update(path_ids)
            evidence.append(evidence_label)
        if ids:
            metadata_ids.update(ids)
            continue
        stats = read_json(path / "embedding_stats.json") or read_json(path / "embedding_checkpoints/embedding_stats_partial.json")
        value = stats.get("embedded_total_with_resume") or stats.get("embedded_new_this_run") or 0
        try:
            stat_total += int(value)
        except (TypeError, ValueError):
            pass
        if stats:
            evidence.append("embedding_stats")
    total = len(metadata_ids) + stat_total
    if expected_ids is not None:
        total = min(total, len(expected_ids))
    return total, ";".join(sorted(set(evidence)))


def count_glm2_units(paths: list[Path], expected_ids: set[str] | None = None) -> tuple[int, str]:
    ids: set[str] = set()
    evidence: list[str] = []
    for path in paths:
        mag_level = read_tsv(path / "feature_glm_mag_level.tsv")
        if mag_level:
            ids.update(row.get("proteome_id", "") for row in mag_level if row.get("proteome_id"))
            evidence.append("feature_glm_mag_level.tsv")
            continue
        windows = read_tsv(path / "features/glm2_smoke_window_embedding_summary.tsv")
        if windows:
            ids.update(row.get("proteome_id", "") for row in windows if row.get("proteome_id"))
            evidence.append("glm2_smoke_window_embedding_summary.tsv")
    if expected_ids is not None:
        ids = ids & expected_ids
    return len(ids), ";".join(sorted(set(evidence)))


def count_warehouse_dim_mag(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    manifest = read_tsv(path / "cohort_table_manifest.tsv")
    for row in manifest:
        if row.get("table") == "dim_mag":
            try:
                return int(float(row.get("rows", 0)))
            except (TypeError, ValueError):
                return 0
    return 0


def consolidation_command(repo_root: Path, row: dict[str, str], functional_include: int) -> str:
    cohort_run_id = row.get("consolidation_cohort_run_id") or row.get("lane_id") or "external_lane"
    output_dir = row.get("consolidation_output_dir") or row.get("functional_warehouse_dir") or ""
    parts = [
        "scripts/consolidate_functional_mag_cohort.py",
        "--repo-root",
        str(repo_root),
        "--cohort-run-id",
        cohort_run_id,
        "--manifest",
        row.get("functional_manifest", ""),
        "--expected-complete-count",
        str(functional_include),
    ]
    for per_mag_dir in str(row.get("functional_per_mag_dirs") or "").split(";"):
        if per_mag_dir.strip():
            parts.extend(["--per-mag-dir", per_mag_dir.strip()])
    if output_dir:
        parts.extend(["--output-dir", output_dir])
    parts.append("--build-duckdb")
    return " ".join(shlex.quote(part) for part in parts if part != "")


def summarize_lane(repo_root: Path, row: dict[str, str]) -> dict[str, Any]:
    source_manifest = resolve(repo_root, row.get("source_lane_manifest"))
    functional_manifest = resolve(repo_root, row.get("functional_manifest"))
    source_rows = read_tsv(source_manifest)
    functional_rows = read_tsv(functional_manifest)
    functional_expected_rows = (
        [row for row in functional_rows if truthy(row.get("functional_run_include"))]
        if functional_rows and "functional_run_include" in functional_rows[0]
        else functional_rows
    )
    expected_ids = {row.get("proteome_id", "") for row in functional_expected_rows if row.get("proteome_id")}
    if not expected_ids:
        expected_ids = {row.get("proteome_id", "") for row in source_rows if row.get("proteome_id")}
    per_mag_dirs = resolve_many(repo_root, row.get("functional_per_mag_dirs"))
    status_counts = functional_status_counts(per_mag_dirs, functional_rows)
    attempt_audit = functional_attempt_audit(per_mag_dirs)
    esm2_count, esm2_evidence = count_esm2_units(resolve_many(repo_root, row.get("esm2_artifacts_dirs")), expected_ids)
    glm2_count, glm2_evidence = count_glm2_units(resolve_many(repo_root, row.get("glm2_artifacts_dirs")), expected_ids)
    warehouse_count = count_warehouse_dim_mag(resolve(repo_root, row.get("functional_warehouse_dir")))
    functional_include = count_include(functional_rows, "functional_run_include")
    source_functional_include = (
        count_include(source_rows, "functional_run_include")
        if source_rows and "functional_run_include" in source_rows[0]
        else 0
    )
    if source_functional_include and functional_include > source_functional_include:
        functional_include = source_functional_include

    total = int(row.get("denominator_units") or len(source_rows) or len(functional_rows) or 0)
    manifest_gap_rows = max(total - functional_include, 0)
    sentinel_complete = int(status_counts.get("complete", 0))
    complete = max(sentinel_complete, warehouse_count)
    status_basis = "warehouse_dim_mag" if warehouse_count >= functional_include and warehouse_count > 0 else "live_per_mag_sentinels"
    partial = int(status_counts.get("partial", 0))
    failed = int(status_counts.get("failed", 0))
    not_started = int(status_counts.get("not_started", 0))
    if status_basis == "warehouse_dim_mag":
        partial = 0
        failed = 0
        not_started = max(functional_include - warehouse_count, 0)
    multiview = min(esm2_count, glm2_count, complete)
    consolidation_ready = functional_include > 0 and complete >= functional_include
    warehouse_current = warehouse_count >= functional_include and functional_include > 0
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": row.get("lane_id", ""),
        "lane_role": row.get("lane_role", ""),
        "lane_status": row.get("lane_status", ""),
        "denominator_label": row.get("denominator_label", ""),
        "denominator_units": total,
        "source_manifest_rows": len(source_rows),
        "functional_manifest_rows": len(functional_rows),
        "functional_include_rows": functional_include,
        "esm2_units": esm2_count,
        "glm2_units": glm2_count,
        "functional_complete": complete,
        "functional_partial": partial,
        "functional_failed": failed,
        "functional_not_started": not_started,
        **attempt_audit,
        "warehouse_dim_mag_rows": warehouse_count,
        "functional_status_basis": status_basis,
        "manifest_gap_rows": manifest_gap_rows,
        "tri_view_ready_units": multiview,
        "consolidation_ready": "true" if consolidation_ready else "false",
        "warehouse_current": "true" if warehouse_current else "false",
        "consolidation_command": consolidation_command(repo_root, row, functional_include),
        "gap_register": row.get("gap_register", ""),
        "source_provenance_dir": row.get("source_provenance_dir", ""),
        "source_provenance_checksums": row.get("source_provenance_checksums", ""),
        "claim_scope": row.get("claim_scope", ""),
        "esm2_evidence": esm2_evidence,
        "glm2_evidence": glm2_evidence,
        "notes": row.get("notes", ""),
    }


def markdown_report(rows: list[dict[str, Any]]) -> str:
    generated = rows[0]["generated_utc"] if rows else datetime.now(timezone.utc).isoformat()
    total_tri_view = sum(int(row.get("tri_view_ready_units") or 0) for row in rows)
    lines = [
        "# MethaNet Atlas Lane Registry Status",
        "",
        f"Generated UTC: `{generated}`",
        "",
        "This report is manifest-driven. It treats lane manifests as denominators, "
        "uses validated cohort warehouses where available, and preserves missing or "
        "pending evidence as explicit status rather than dropping rows.",
        "",
        f"Current tri-view ready units across registered lanes: **{total_tri_view:,}**.",
        "",
        "| Lane | Denominator | ESM2 | gLM2 | Functional complete | Tri-view ready | Status basis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {lane_id} | {denominator_units:,} | {esm2_units:,} | {glm2_units:,} | "
            "{functional_complete:,} | {tri_view_ready_units:,} | {functional_status_basis} |".format(
                **{
                    key: int(value) if str(value).isdigit() else value
                    for key, value in row.items()
                }
            )
        )

    lines.extend(
        [
            "",
            "## Lane Notes",
            "",
        ]
    )
    for row in rows:
        pending = int(row.get("functional_not_started") or 0) + int(row.get("functional_partial") or 0)
        failed = int(row.get("functional_failed") or 0)
        complete_attempts = int(row.get("functional_complete_run_attempts") or 0)
        failed_attempts = int(row.get("functional_failed_run_attempts") or 0)
        partial_attempts = int(row.get("functional_partial_run_attempts") or 0)
        duplicate_attempts = int(row.get("duplicate_complete_attempts") or 0)
        if row.get("functional_status_basis") == "warehouse_dim_mag":
            attempt_line = (
                f"- Warehouse `dim_mag` rows: `{int(row.get('warehouse_dim_mag_rows') or 0):,}`; "
                f"live attempt audit observed `{complete_attempts:,}` complete, "
                f"`{failed_attempts:,}` failed, and `{partial_attempts:,}` partial attempts, "
                "but warehouse rows are the status basis."
            )
        else:
            attempt_line = (
                f"- Attempt audit: `{complete_attempts:,}` complete, "
                f"`{failed_attempts:,}` failed, `{partial_attempts:,}` partial; "
                f"duplicate complete attempts: `{duplicate_attempts:,}`."
            )
        lines.extend(
            [
                f"### {row['lane_id']}",
                "",
                f"- Role: `{row['lane_role']}`; status: `{row['lane_status']}`.",
                f"- Functional include rows: `{int(row.get('functional_include_rows') or 0):,}`.",
                f"- Manifest gap rows: `{int(row.get('manifest_gap_rows') or 0):,}`.",
                f"- Pending/partial functional rows: `{pending:,}`; failed rows: `{failed:,}`.",
                attempt_line,
                f"- Claim scope: {row.get('claim_scope') or 'MAG/proteome molecular screening'}",
                f"- Source provenance: {row.get('source_provenance_dir') or 'not registered'}",
                f"- Source checksum ledger: {row.get('source_provenance_checksums') or 'not registered'}",
                f"- Notes: {row.get('notes') or 'None.'}",
                "",
            ]
        )

    lines.extend(["## Consolidation Plan", ""])
    for row in rows:
        if row.get("warehouse_current") == "true":
            lines.extend(
                [
                    f"- `{row['lane_id']}`: warehouse is current for the registered functional denominator.",
                    "",
                ]
            )
        elif row.get("consolidation_ready") == "true":
            lines.extend(
                [
                    f"- `{row['lane_id']}`: ready to consolidate with:",
                    "",
                    "```bash",
                    str(row.get("consolidation_command", "")),
                    "```",
                    "",
                ]
            )
        else:
            missing = int(row.get("functional_include_rows") or 0) - int(row.get("functional_complete") or 0)
            lines.extend(
                [
                    f"- `{row['lane_id']}`: not ready to consolidate; `{max(missing, 0):,}` functional include rows still lack completed curated outputs.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Claim Boundary",
            "",
            "Allowed now: MAG/proteome molecular screening, source-lane readiness, "
            "and evidence-completeness reporting.",
            "",
            "Not allowed from this report alone: final sample methane-risk tiers, "
            "measured methane flux, carbon-credit approval, or source-independent "
            "rumen-to-wetland/mangrove MRV transfer claims.",
            "",
            "Required for stronger sample/MRV claims: MAG-to-sample mapping, "
            "abundance/read coverage, environmental covariates, uncertainty "
            "propagation, and flux/process validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    registry = resolve(repo_root, str(args.lane_registry))
    lanes = read_tsv(registry)
    if not lanes:
        raise SystemExit(f"No lane rows found: {registry}")
    rows = [summarize_lane(repo_root, lane) for lane in lanes]
    fields = list(rows[0])
    if args.output_tsv:
        write_tsv(resolve(repo_root, str(args.output_tsv)) or args.output_tsv, rows, fields)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        path = resolve(repo_root, str(args.output_json)) or args.output_json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    if args.output_md:
        path = resolve(repo_root, str(args.output_md)) or args.output_md
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
