#!/usr/bin/env python3
"""Audit exact identifier overlap across registered MethaNet atlas lanes.

This is a source-lane integration guard, not a genome similarity analysis. It
checks reusable manifest identifiers before warehouse/report rebuilds so obvious
duplicate submissions, reused accessions, or sample/project collisions are
visible as explicit evidence instead of being discovered in a downstream plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any


TRUE_VALUES = {"true", "1", "yes", "y"}
DEFAULT_FIELDS = [
    "proteome_id",
    "mag_id",
    "mapped_ncbi_biosamples",
    "mapped_ncbi_bioprojects",
    "mapped_node_projects",
    "ncbi_wgs_accession",
    "elmsg_genome_id",
    "source_fasta_filename",
    "local_fna_md5",
]
TOKEN_SPLIT = re.compile(r"[;,|]\s*|\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lane-registry", type=Path, default=Path("configs/methanet_atlas_lanes.tsv"))
    parser.add_argument(
        "--lane-id",
        action="append",
        default=[],
        help="Lane ID to include. May be repeated. Defaults to all external lanes.",
    )
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help="Manifest field to compare. Defaults to a conservative exact-ID field set.",
    )
    parser.add_argument("--output-summary-tsv", type=Path)
    parser.add_argument("--output-matches-tsv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def resolve(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def read_tsv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def selected_registry_rows(rows: list[dict[str, str]], lane_ids: list[str]) -> list[dict[str, str]]:
    if lane_ids:
        wanted = set(lane_ids)
        return [row for row in rows if row.get("lane_id") in wanted]
    return [row for row in rows if str(row.get("lane_role") or "").startswith("external")]


def selected_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    if "functional_run_include" not in rows[0]:
        return rows
    return [row for row in rows if truthy(row.get("functional_run_include"))]


def tokens_for_field(row: dict[str, str], field: str) -> set[str]:
    value = str(row.get(field) or "").strip()
    if not value:
        return set()
    if field in {"proteome_id", "mag_id", "local_fna_md5", "source_fasta_filename", "ncbi_wgs_accession", "elmsg_genome_id"}:
        return {value}
    return {token for token in TOKEN_SPLIT.split(value) if token}


def lane_token_index(rows: list[dict[str, str]], fields: list[str]) -> dict[str, dict[str, set[str]]]:
    index: dict[str, dict[str, set[str]]] = {}
    for field in fields:
        token_to_ids: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            proteome_id = row.get("proteome_id") or ""
            if not proteome_id:
                continue
            for token in tokens_for_field(row, field):
                token_to_ids[token].add(proteome_id)
        index[field] = dict(token_to_ids)
    return index


def lane_duplicate_counts(index: dict[str, dict[str, set[str]]]) -> dict[str, int]:
    return {field: sum(1 for ids in tokens.values() if len(ids) > 1) for field, tokens in index.items()}


def compare_lanes(
    lane_a: dict[str, Any],
    lane_b: dict[str, Any],
    fields: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    shared_by_field: Counter[str] = Counter()
    ids_a: set[str] = set()
    ids_b: set[str] = set()
    for field in fields:
        tokens_a = lane_a["index"].get(field, {})
        tokens_b = lane_b["index"].get(field, {})
        for token in sorted(set(tokens_a) & set(tokens_b)):
            left = sorted(tokens_a[token])
            right = sorted(tokens_b[token])
            shared_by_field[field] += 1
            ids_a.update(left)
            ids_b.update(right)
            matches.append(
                {
                    "lane_a": lane_a["lane_id"],
                    "lane_b": lane_b["lane_id"],
                    "field": field,
                    "shared_token": token,
                    "lane_a_proteome_ids": ";".join(left),
                    "lane_b_proteome_ids": ";".join(right),
                    "lane_a_match_count": len(left),
                    "lane_b_match_count": len(right),
                }
            )
    summary = {
        "lane_a": lane_a["lane_id"],
        "lane_b": lane_b["lane_id"],
        "lane_a_manifest_rows": lane_a["manifest_rows"],
        "lane_b_manifest_rows": lane_b["manifest_rows"],
        "lane_a_selected_rows": lane_a["selected_rows"],
        "lane_b_selected_rows": lane_b["selected_rows"],
        "shared_tokens_total": sum(shared_by_field.values()),
        "lane_a_proteome_ids_with_overlap": len(ids_a),
        "lane_b_proteome_ids_with_overlap": len(ids_b),
        "fields_with_overlap": ";".join(field for field in fields if shared_by_field[field]),
        "shared_tokens_by_field": json.dumps(dict(shared_by_field), sort_keys=True),
        "deduplication_action": "review_overlap" if shared_by_field else "no_exact_manifest_overlap_detected",
    }
    return summary, matches


def build_lane(repo_root: Path, registry_row: dict[str, str], fields: list[str]) -> dict[str, Any]:
    manifest = resolve(repo_root, registry_row.get("source_lane_manifest")) or resolve(
        repo_root, registry_row.get("functional_manifest")
    )
    rows = read_tsv(manifest)
    selected_rows = selected_manifest_rows(rows)
    index = lane_token_index(selected_rows, fields)
    return {
        "lane_id": registry_row.get("lane_id", ""),
        "lane_role": registry_row.get("lane_role", ""),
        "manifest_path": str(manifest or ""),
        "manifest_rows": len(rows),
        "selected_rows": len(selected_rows),
        "fields_present": sorted(field for field in fields if rows and field in rows[0]),
        "index": index,
        "duplicate_tokens_within_lane": lane_duplicate_counts(index),
    }


def markdown_report(summary_rows: list[dict[str, Any]], lanes: list[dict[str, Any]], matches: list[dict[str, Any]]) -> str:
    lines = [
        "# MethaNet Atlas Lane Overlap Audit",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "This audit checks exact manifest identifiers only. It is a pre-consolidation "
        "guard for source-lane overlap, not an ANI or genome-similarity result.",
        "",
        "| Lane | Manifest rows | Selected rows | Manifest |",
        "| --- | ---: | ---: | --- |",
    ]
    for lane in lanes:
        lines.append(
            f"| {lane['lane_id']} | {lane['manifest_rows']:,} | {lane['selected_rows']:,} | `{lane['manifest_path']}` |"
        )
    lines.extend(
        [
            "",
            "| Lane A | Lane B | Shared tokens | A IDs with overlap | B IDs with overlap | Action |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {lane_a} | {lane_b} | {shared_tokens_total:,} | "
            "{lane_a_proteome_ids_with_overlap:,} | {lane_b_proteome_ids_with_overlap:,} | "
            "`{deduplication_action}` |".format(**row)
        )
    if matches:
        lines.extend(["", "## Match Details", ""])
        for match in matches[:50]:
            lines.append(
                "- `{lane_a}` vs `{lane_b}` share `{field}` token `{shared_token}`: `{lane_a_proteome_ids}` <-> `{lane_b_proteome_ids}`".format(
                    **match
                )
            )
        if len(matches) > 50:
            lines.append(f"- Additional matches omitted from Markdown: {len(matches) - 50:,}. See TSV/JSON outputs.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Allowed now: exact manifest-overlap screening and deduplication planning.",
            "",
            "Not allowed from this audit alone: genome-level nonredundancy, ANI-based novelty, "
            "sample methane-risk tiers, measured methane flux, or carbon-crediting claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    registry = resolve(repo_root, str(args.lane_registry))
    registry_rows = read_tsv(registry)
    if not registry_rows:
        raise SystemExit(f"No lane rows found: {registry}")
    selected_rows = selected_registry_rows(registry_rows, args.lane_id)
    if args.lane_id:
        found = {row.get("lane_id") for row in selected_rows}
        missing = sorted(set(args.lane_id) - found)
        if missing:
            raise SystemExit("Lane ID(s) not found in registry: " + ", ".join(missing))
    if len(selected_rows) < 2:
        raise SystemExit("At least two lanes are required for overlap auditing")

    fields = args.field or DEFAULT_FIELDS
    lanes = [build_lane(repo_root, row, fields) for row in selected_rows]
    summary_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for lane_a, lane_b in combinations(lanes, 2):
        summary, matches = compare_lanes(lane_a, lane_b, fields)
        summary_rows.append(summary)
        match_rows.extend(matches)

    if args.output_summary_tsv:
        write_tsv(resolve(repo_root, str(args.output_summary_tsv)) or args.output_summary_tsv, summary_rows, list(summary_rows[0]))
    else:
        writer = csv.DictWriter(__import__("sys").stdout, delimiter="\t", fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    if args.output_matches_tsv:
        match_fields = [
            "lane_a",
            "lane_b",
            "field",
            "shared_token",
            "lane_a_proteome_ids",
            "lane_b_proteome_ids",
            "lane_a_match_count",
            "lane_b_match_count",
        ]
        write_tsv(resolve(repo_root, str(args.output_matches_tsv)) or args.output_matches_tsv, match_rows, match_fields)
    if args.output_json:
        path = resolve(repo_root, str(args.output_json)) or args.output_json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "fields": fields,
                    "lanes": [
                        {key: value for key, value in lane.items() if key != "index"}
                        for lane in lanes
                    ],
                    "summary": summary_rows,
                    "matches": match_rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.output_md:
        path = resolve(repo_root, str(args.output_md)) or args.output_md
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(summary_rows, lanes, match_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
