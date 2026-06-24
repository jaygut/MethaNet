#!/usr/bin/env python3
"""Compare two MethaNet atlas lane status snapshots.

This utility is intentionally registry/status driven: it compares the JSON
emitted by ``summarize_atlas_lane_registry.py`` and avoids lane-specific logic.
It is meant for long-running external source-lane integrations where operators
need to see what changed between readiness checks.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NUMERIC_FIELDS = [
    "denominator_units",
    "functional_manifest_rows",
    "functional_include_rows",
    "esm2_units",
    "glm2_units",
    "functional_complete",
    "functional_partial",
    "functional_failed",
    "functional_not_started",
    "functional_complete_run_attempts",
    "duplicate_complete_attempts",
    "warehouse_dim_mag_rows",
    "manifest_gap_rows",
    "tri_view_ready_units",
]

TEXT_FIELDS = [
    "lane_role",
    "lane_status",
    "functional_status_basis",
    "consolidation_ready",
    "warehouse_current",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-json", type=Path, required=True)
    parser.add_argument("--current-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def read_status(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Status JSON missing or empty: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"Status JSON must contain a list of lane rows: {path}")
    rows: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            raise SystemExit(f"Status JSON contains a non-object row: {path}")
        lane_id = str(row.get("lane_id") or "").strip()
        if not lane_id:
            raise SystemExit(f"Status JSON contains a row without lane_id: {path}")
        rows.append(row)
    return rows


def as_int(value: Any) -> int:
    try:
        return int(float(str(value or "0").replace(",", "")))
    except ValueError:
        return 0


def generated_utc(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        value = str(row.get("generated_utc") or "").strip()
        if value:
            return value
    return ""


def compare_rows(previous_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_by_lane = {str(row["lane_id"]): row for row in previous_rows}
    current_by_lane = {str(row["lane_id"]): row for row in current_rows}
    lane_ids = sorted(set(previous_by_lane) | set(current_by_lane))
    rows: list[dict[str, Any]] = []
    for lane_id in lane_ids:
        previous = previous_by_lane.get(lane_id, {})
        current = current_by_lane.get(lane_id, {})
        row: dict[str, Any] = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "previous_generated_utc": generated_utc(previous_rows),
            "current_generated_utc": generated_utc(current_rows),
            "lane_id": lane_id,
            "comparison_status": "existing",
        }
        if previous and not current:
            row["comparison_status"] = "removed"
        elif current and not previous:
            row["comparison_status"] = "added"

        for field in NUMERIC_FIELDS:
            previous_value = as_int(previous.get(field))
            current_value = as_int(current.get(field))
            row[f"previous_{field}"] = previous_value
            row[f"current_{field}"] = current_value
            row[f"delta_{field}"] = current_value - previous_value
        for field in TEXT_FIELDS:
            previous_value = str(previous.get(field) or "")
            current_value = str(current.get(field) or "")
            row[f"previous_{field}"] = previous_value
            row[f"current_{field}"] = current_value
            row[f"changed_{field}"] = "true" if previous_value != current_value else "false"
        rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["generated_utc", "lane_id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(rows: list[dict[str, Any]]) -> str:
    generated = rows[0]["generated_utc"] if rows else datetime.now(timezone.utc).isoformat()
    previous_generated = rows[0].get("previous_generated_utc", "") if rows else ""
    current_generated = rows[0].get("current_generated_utc", "") if rows else ""
    total_delta_tri = sum(as_int(row.get("delta_tri_view_ready_units")) for row in rows)
    total_delta_functional = sum(as_int(row.get("delta_functional_complete")) for row in rows)
    lines = [
        "# MethaNet Atlas Lane Status Delta",
        "",
        f"Generated UTC: `{generated}`",
        "",
        f"Previous snapshot UTC: `{previous_generated or 'unknown'}`",
        "",
        f"Current snapshot UTC: `{current_generated or 'unknown'}`",
        "",
        f"Functional complete delta across lanes: **{total_delta_functional:+,}**.",
        "",
        f"Tri-view ready delta across lanes: **{total_delta_tri:+,}**.",
        "",
        "| Lane | Status | Functional complete | ESM2 | gLM2 | Tri-view | Pending/partial | Failed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pending_delta = as_int(row.get("delta_functional_partial")) + as_int(row.get("delta_functional_not_started"))
        lines.append(
            "| {lane_id} | {comparison_status} | {delta_functional_complete:+,} | "
            "{delta_esm2_units:+,} | {delta_glm2_units:+,} | {delta_tri_view_ready_units:+,} | "
            "{pending_delta:+,} | {delta_functional_failed:+,} |".format(
                lane_id=row["lane_id"],
                comparison_status=row["comparison_status"],
                delta_functional_complete=as_int(row.get("delta_functional_complete")),
                delta_esm2_units=as_int(row.get("delta_esm2_units")),
                delta_glm2_units=as_int(row.get("delta_glm2_units")),
                delta_tri_view_ready_units=as_int(row.get("delta_tri_view_ready_units")),
                pending_delta=pending_delta,
                delta_functional_failed=as_int(row.get("delta_functional_failed")),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This delta report is operational evidence only. It tracks artifact and "
            "gate-readiness movement; it does not assign sample methane-risk tiers, "
            "measured flux, carbon-credit approval, or calibrated MRV scores.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    previous_rows = read_status(args.previous_json)
    current_rows = read_status(args.current_json)
    rows = compare_rows(previous_rows, current_rows)
    if args.output_tsv:
        write_tsv(args.output_tsv, rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
