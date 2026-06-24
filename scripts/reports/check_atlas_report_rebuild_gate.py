#!/usr/bin/env python3
"""Check whether the expanded MethaNet atlas report is ready to rebuild.

This gate is stricter than the consolidation gate. It checks tri-view readiness
for selected external lanes: ESM2, gLM2, and functional evidence must each cover
the registered runnable functional denominator before a final expanded atlas
rebuild is advertised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--lane-registry", default="configs/methanet_atlas_lanes.tsv")
    parser.add_argument(
        "--lane-id",
        action="append",
        default=[],
        help="External lane ID to require for final report readiness. May be repeated. Defaults to all external lanes.",
    )
    parser.add_argument("--output-dir", default="", help="Optional output directory for the report build command.")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def read_status_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Status JSON missing: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"Status JSON must contain a list of lane rows: {path}")
    return [row for row in data if isinstance(row, dict)]


def as_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def is_true(value: Any) -> bool:
    return str(value or "").strip().lower() == "true"


def external_rows(rows: list[dict[str, Any]], lane_ids: list[str]) -> list[dict[str, Any]]:
    if lane_ids:
        wanted = set(lane_ids)
        return [row for row in rows if str(row.get("lane_id") or "") in wanted]
    return [row for row in rows if str(row.get("lane_role") or "").startswith("external")]


def calibration_blockers(rows: list[dict[str, Any]]) -> list[str]:
    calibration = [row for row in rows if row.get("lane_role") == "calibration_core"]
    if len(calibration) != 1:
        return [f"expected exactly one calibration_core row; observed {len(calibration)}"]
    row = calibration[0]
    if not is_true(row.get("warehouse_current")):
        return [f"calibration lane {row.get('lane_id')} warehouse_current is not true"]
    return []


def lane_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    include_rows = as_int(row.get("functional_include_rows"))
    if include_rows <= 0:
        blockers.append("functional_include_rows is zero")
        return blockers
    expectations = [
        ("ESM2", "esm2_units"),
        ("gLM2", "glm2_units"),
        ("functional", "functional_complete"),
        ("tri-view", "tri_view_ready_units"),
    ]
    for label, field in expectations:
        observed = as_int(row.get(field))
        if observed < include_rows:
            blockers.append(f"{label} units {observed} are smaller than functional_include_rows {include_rows}")
    if as_int(row.get("functional_failed")):
        blockers.append(f"{as_int(row.get('functional_failed'))} functional rows are failed")
    pending = as_int(row.get("functional_not_started")) + as_int(row.get("functional_partial"))
    if pending:
        blockers.append(f"{pending} functional rows are pending/partial")
    return blockers


def report_command(lane_registry: str, output_dir: str = "") -> str:
    parts = [
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "--lane-registry",
        lane_registry,
    ]
    if output_dir:
        parts.extend(["--output-dir", output_dir])
    return " ".join(parts)


def main() -> int:
    args = parse_args()
    rows = read_status_rows(args.status_json)
    failures = 0

    calibration_errors = calibration_blockers(rows)
    if calibration_errors:
        failures += 1
        print("BLOCKED calibration_core: " + "; ".join(calibration_errors))

    selected = external_rows(rows, args.lane_id)
    if args.lane_id:
        found = {str(row.get("lane_id") or "") for row in selected}
        missing = sorted(set(args.lane_id) - found)
        for lane_id in missing:
            failures += 1
            print(f"BLOCKED {lane_id}: lane not found in status JSON")
    if not selected:
        failures += 1
        print("BLOCKED expanded atlas: no external lanes selected")

    for row in selected:
        lane_id = str(row.get("lane_id") or "<missing>")
        blockers = lane_blockers(row)
        if blockers:
            failures += 1
            print(f"BLOCKED {lane_id}: " + "; ".join(blockers))
        else:
            print(f"READY {lane_id}: tri-view report gate passed")

    if failures:
        return 1
    print("READY expanded atlas: all selected lanes passed final report gate")
    if args.print_command:
        print(report_command(args.lane_registry, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
