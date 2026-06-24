#!/usr/bin/env python3
"""Build an operator checklist from a MethaNet atlas lane status JSON.

The checklist is intentionally derived from the registry status snapshot rather
than live filesystem scans. That keeps the operational handoff reproducible:
refresh the registry status once, then every checklist row explains what still
blocks consolidation and expanded atlas rebuild for the same snapshot.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--lane-id", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def as_int(value: Any) -> int:
    try:
        return int(float(str(value or "0").replace(",", "")))
    except ValueError:
        return 0


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def read_status(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Status JSON missing or empty: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"Status JSON must contain a list of lane rows: {path}")
    rows: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            raise SystemExit(f"Status JSON contains a non-object lane row: {path}")
        if not row.get("lane_id"):
            raise SystemExit(f"Status JSON contains a row without lane_id: {path}")
        rows.append(row)
    return rows


def select_rows(rows: list[dict[str, Any]], lane_ids: list[str]) -> list[dict[str, Any]]:
    if not lane_ids:
        return rows
    wanted = set(lane_ids)
    selected = [row for row in rows if str(row.get("lane_id")) in wanted]
    missing = sorted(wanted - {str(row.get("lane_id")) for row in selected})
    if missing:
        raise SystemExit(f"Requested lane_id values were not found in status JSON: {', '.join(missing)}")
    return selected


def lane_actions(row: dict[str, Any]) -> list[str]:
    lane_id = str(row.get("lane_id") or "")
    functional_include = as_int(row.get("functional_include_rows"))
    functional_complete = as_int(row.get("functional_complete"))
    pending = as_int(row.get("functional_not_started")) + as_int(row.get("functional_partial"))
    failed = as_int(row.get("functional_failed"))
    esm2 = as_int(row.get("esm2_units"))
    glm2 = as_int(row.get("glm2_units"))
    tri_view = as_int(row.get("tri_view_ready_units"))
    actions: list[str] = []

    if failed:
        actions.append(f"Investigate and relaunch {failed:,} failed functional rows before consolidation.")
    if pending:
        actions.append(f"Wait for or relaunch {pending:,} pending/partial functional rows.")
    if functional_complete < functional_include:
        actions.append(
            f"Reach functional_complete={functional_include:,}; current {functional_complete:,}."
        )
    if esm2 < functional_include:
        actions.append(f"Complete ESM2 for {functional_include - esm2:,} selected rows.")
    if glm2 < functional_include:
        actions.append(f"Complete gLM2 for {functional_include - glm2:,} selected rows.")
    if tri_view < functional_include:
        actions.append(f"Reach tri-view coverage for {functional_include - tri_view:,} selected rows.")
    if not actions and not truthy(row.get("warehouse_current")):
        command = str(row.get("consolidation_command") or "").strip()
        if command:
            actions.append(f"Run registry-derived consolidation command for {lane_id}.")
        else:
            actions.append(f"Consolidate {lane_id}; no registry-derived command was present.")
    if not actions:
        actions.append("No action needed for this lane in the current snapshot.")
    return actions


def build_rows(status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generated = datetime.now(timezone.utc).isoformat()
    checklist: list[dict[str, Any]] = []
    for row in status_rows:
        functional_include = as_int(row.get("functional_include_rows"))
        functional_complete = as_int(row.get("functional_complete"))
        pending = as_int(row.get("functional_not_started")) + as_int(row.get("functional_partial"))
        failed = as_int(row.get("functional_failed"))
        esm2 = as_int(row.get("esm2_units"))
        glm2 = as_int(row.get("glm2_units"))
        tri_view = as_int(row.get("tri_view_ready_units"))
        consolidation_ready = truthy(row.get("consolidation_ready"))
        warehouse_current = truthy(row.get("warehouse_current"))
        report_ready = (
            functional_include > 0
            and failed == 0
            and pending == 0
            and functional_complete >= functional_include
            and esm2 >= functional_include
            and glm2 >= functional_include
            and tri_view >= functional_include
        )
        checklist.append(
            {
                "generated_utc": generated,
                "lane_id": row.get("lane_id", ""),
                "lane_role": row.get("lane_role", ""),
                "functional_include_rows": functional_include,
                "functional_complete": functional_complete,
                "functional_remaining": max(functional_include - functional_complete, 0),
                "functional_pending_or_partial": pending,
                "functional_failed": failed,
                "esm2_remaining": max(functional_include - esm2, 0),
                "glm2_remaining": max(functional_include - glm2, 0),
                "tri_view_remaining": max(functional_include - tri_view, 0),
                "consolidation_ready": consolidation_ready,
                "warehouse_current": warehouse_current,
                "expanded_atlas_report_ready": report_ready,
                "next_actions": lane_actions(row),
                "consolidation_command": row.get("consolidation_command", ""),
                "claim_boundary": (
                    "Operational checklist only; no final sample methane-risk tiers, "
                    "measured methane flux, carbon-credit approval, or calibrated MRV scoring."
                ),
            }
        )
    return checklist


def markdown_report(rows: list[dict[str, Any]]) -> str:
    generated = rows[0]["generated_utc"] if rows else datetime.now(timezone.utc).isoformat()
    lines = [
        "# MethaNet Atlas Lane Completion Checklist",
        "",
        f"Generated UTC: `{generated}`",
        "",
        "| Lane | Functional remaining | ESM2 remaining | gLM2 remaining | Tri-view remaining | Failed | Consolidation | Report rebuild |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {lane_id} | {functional_remaining:,} | {esm2_remaining:,} | {glm2_remaining:,} | "
            "{tri_view_remaining:,} | {functional_failed:,} | {consolidation} | {report} |".format(
                lane_id=row["lane_id"],
                functional_remaining=row["functional_remaining"],
                esm2_remaining=row["esm2_remaining"],
                glm2_remaining=row["glm2_remaining"],
                tri_view_remaining=row["tri_view_remaining"],
                functional_failed=row["functional_failed"],
                consolidation="ready" if row["consolidation_ready"] else "blocked",
                report="ready" if row["expanded_atlas_report_ready"] else "blocked",
            )
        )
    lines.extend(["", "## Next Actions", ""])
    for row in rows:
        lines.extend([f"### {row['lane_id']}", ""])
        for action in row["next_actions"]:
            lines.append(f"- {action}")
        if row["consolidation_ready"] and not row["warehouse_current"] and row.get("consolidation_command"):
            lines.extend(["", "```bash", str(row["consolidation_command"]), "```"])
        lines.append("")
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "This checklist is operational evidence only. It does not assign final "
            "sample methane-risk tiers, measured methane flux, carbon-credit "
            "approval, or calibrated MRV scores.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rows = build_rows(select_rows(read_status(args.status_json), args.lane_id))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(rows))
    if not args.output_json and not args.output_md:
        print(markdown_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
