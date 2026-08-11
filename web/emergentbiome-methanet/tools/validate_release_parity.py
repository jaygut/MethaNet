#!/usr/bin/env python3
"""Fail publication when release counts or claim metadata drift across artifacts.

The release ledger is the authority. Every supplied downstream artifact must
agree with it; omitted optional artifacts are reported as skipped so callers
can use the same command before and after report/site generation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


COUNT_KEYS = (
    "registered_units",
    "esm2_units",
    "glm2_units",
    "functional_payload_units",
    "release_required_units",
    "explicit_non_runnable_gaps",
    "tri_view_ready_units",
    "schema_normalized_units",
    "schema_normalized_tri_view_units",
    "pipeline_normalized_tri_view_units",
    "mechanism_comparable_units",
    "annotation_complete_tri_view_units",
    "source_scaffold_tri_view_units",
    "blocking_units",
)

DECISION_KEYS = {
    "registered_units": "unit_rows",
    "esm2_units": "esm2_units",
    "glm2_units": "glm2_units",
    "functional_payload_units": "functional_payload_units",
    "release_required_units": "release_required_units",
    "explicit_non_runnable_gaps": "release_excluded_units",
    "tri_view_ready_units": "release_tri_view_ready_units",
    "schema_normalized_units": "schema_normalized_units",
    "schema_normalized_tri_view_units": "schema_normalized_tri_view_units",
    "pipeline_normalized_tri_view_units": "pipeline_normalized_tri_view_units",
    "mechanism_comparable_units": "canonical_mechanism_tri_view_units",
    "annotation_complete_tri_view_units": "annotation_complete_tri_view_units",
    "source_scaffold_tri_view_units": "source_scaffold_tri_view_units",
    "blocking_units": "blocking_units",
}

SUMMARY_COLUMNS = {
    "registered_units": "registry_denominator_units",
    "esm2_units": "esm2_units",
    "glm2_units": "glm2_units",
    "functional_payload_units": "functional_payload_units",
    "release_required_units": "release_required_units",
    "explicit_non_runnable_gaps": "release_excluded_units",
    "tri_view_ready_units": "release_tri_view_ready_units",
    "schema_normalized_units": "schema_normalized_units",
    "schema_normalized_tri_view_units": "schema_normalized_tri_view_units",
    "pipeline_normalized_tri_view_units": "pipeline_normalized_tri_view_units",
    "mechanism_comparable_units": "canonical_mechanism_tri_view_units",
    "annotation_complete_tri_view_units": "annotation_complete_tri_view_units",
    "source_scaffold_tri_view_units": "source_scaffold_tri_view_units",
}

REPORT_KEYS = {
    "registered_units": "atlas_registered_units",
    "esm2_units": "release_esm2_units",
    "glm2_units": "release_glm2_units",
    "functional_payload_units": "release_functional_payload_units",
    "release_required_units": "embedding_context_total",
    "tri_view_ready_units": "release_multiview_complete",
    "schema_normalized_units": "schema_normalized_units",
    "schema_normalized_tri_view_units": "schema_normalized_tri_view_units",
    "pipeline_normalized_tri_view_units": "pipeline_normalized_tri_view_units",
    "mechanism_comparable_units": "mechanism_comparable_tri_view",
    "annotation_complete_tri_view_units": "annotation_complete_harmonization_pending_tri_view",
    "source_scaffold_tri_view_units": "source_scaffold_tri_view",
}

CONFIG_KEYS = {
    "registered_units": ("plottedNodes", "warehouseReach"),
    "release_required_units": ("embeddingBearingUnits",),
    "tri_view_ready_units": ("triViewReady",),
    "schema_normalized_tri_view_units": ("schemaNormalizedTriView",),
    "pipeline_normalized_tri_view_units": ("pipelineNormalizedTriView",),
    "mechanism_comparable_units": ("mechanismComparableTriView",),
    "annotation_complete_tri_view_units": ("annotationCompletePendingTriView",),
    "source_scaffold_tri_view_units": ("sourceScaffoldTriView",),
}

REQUIRED_PROVENANCE = (
    "schema_version",
    "snapshot_date",
    "freeze_manifest_sha256",
    "release_state",
    "indexing_decision",
    "allowed_public_wording",
    "forbidden_public_wording",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-ledger", type=Path, required=True)
    parser.add_argument("--freeze-decision", type=Path, required=True)
    parser.add_argument("--freeze-summary", type=Path, required=True)
    parser.add_argument("--report-bundle-manifest", type=Path)
    parser.add_argument("--report-summary", type=Path)
    parser.add_argument("--report-html", type=Path)
    parser.add_argument("--digest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a count")
    return int(value)


def is_absolute_string(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value) is not None)


def walk_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)
    else:
        yield value


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def check(self, gate: str, passed: bool, detail: str) -> None:
        self.rows.append({"gate": gate, "status": "pass" if passed else "fail", "detail": detail})

    def equal(self, gate: str, actual: Any, expected: Any) -> None:
        self.check(gate, actual == expected, f"actual={actual!r}; expected={expected!r}")

    @property
    def failed(self) -> list[dict[str, str]]:
        return [row for row in self.rows if row["status"] == "fail"]


def validate_lane_summary(audit: Audit, ledger: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lanes = {str(row.get("lane_id")): row for row in ledger.get("lanes", [])}
    summary = {str(row.get("lane_id")): row for row in rows}
    audit.equal("freeze_summary.lane_ids", sorted(summary), sorted(lanes))
    lane_fields = (
        "registry_denominator_units",
        "release_required_units",
        "release_excluded_units",
        "esm2_units",
        "glm2_units",
        "functional_payload_units",
        "functional_complete",
        "functional_partial",
        "functional_failed",
        "functional_not_started",
        "functional_non_runnable_gap",
        "release_tri_view_ready_units",
        "schema_normalized_units",
        "schema_normalized_tri_view_units",
        "pipeline_normalized_tri_view_units",
        "canonical_mechanism_tri_view_units",
        "annotation_complete_tri_view_units",
        "source_scaffold_tri_view_units",
    )
    for lane_id in sorted(set(lanes) & set(summary)):
        for field in lane_fields:
            audit.equal(
                f"freeze_summary.{lane_id}.{field}",
                as_int(summary[lane_id].get(field, 0)),
                as_int(lanes[lane_id].get(field, 0)),
            )


def validate_report(
    audit: Audit,
    ledger: dict[str, Any],
    report_summary: dict[str, Any],
    report_manifest: dict[str, Any] | None,
) -> None:
    for ledger_key, report_key in REPORT_KEYS.items():
        audit.equal(
            f"report_summary.{report_key}",
            report_summary.get(report_key),
            ledger[ledger_key],
        )
    audit.equal(
        "report_summary.snapshot_date",
        report_summary.get("snapshot_date"),
        ledger["snapshot_date"],
    )
    absolute_values = sorted({value for value in walk_values(report_summary) if is_absolute_string(value)})
    audit.check(
        "report_summary.no_absolute_paths",
        not absolute_values,
        "no absolute paths" if not absolute_values else f"absolute paths: {absolute_values[:3]}",
    )
    if report_manifest is not None:
        audit.equal("report_manifest.summary", report_manifest.get("summary"), report_summary)
        audit.check(
            "report_manifest.claim_boundary",
            bool(str(report_manifest.get("claim_boundary", "")).strip()),
            "claim boundary is present",
        )


def validate_config(audit: Audit, ledger: dict[str, Any], path: Path) -> None:
    text = path.read_text()
    for ledger_key, config_keys in CONFIG_KEYS.items():
        for config_key in config_keys:
            match = re.search(rf"\b{re.escape(config_key)}\s*:\s*(\d+)\b", text)
            audit.check(f"config.{config_key}.present", match is not None, "numeric key is present")
            if match:
                audit.equal(f"config.{config_key}", int(match.group(1)), ledger[ledger_key])
    snapshot = re.search(r"\bsnapshot\s*:\s*[\"']([^\"']+)[\"']", text)
    audit.check("config.snapshot.present", snapshot is not None, "snapshot key is present")
    if snapshot:
        audit.equal("config.snapshot", snapshot.group(1), ledger["snapshot_date"])


def validate_atlas(audit: Audit, ledger: dict[str, Any], atlas: dict[str, Any]) -> None:
    meta = atlas.get("meta", {})
    points = atlas.get("points", [])
    audit.equal("atlas.meta.snapshot", meta.get("snapshot"), ledger["snapshot_date"])
    audit.equal("atlas.meta.n_points", meta.get("n_points"), ledger["release_required_units"])
    audit.equal("atlas.points.rows", len(points), ledger["release_required_units"])
    audit.equal("atlas.meta.excluded_gap_rows", meta.get("excluded_gap_rows"), ledger["explicit_non_runnable_gaps"])
    contracts = Counter(point.get("fc") for point in points)
    audit.equal("atlas.contract.mechanism_comparable", contracts[1], ledger["mechanism_comparable_units"])
    audit.equal("atlas.contract.annotation_complete", contracts[2], ledger["annotation_complete_tri_view_units"])
    audit.equal("atlas.contract.source_scaffold", contracts[3], ledger["source_scaffold_tri_view_units"])
    audit.equal("atlas.contract.pipeline_normalized", contracts[4], ledger["pipeline_normalized_tri_view_units"])
    audit.equal("atlas.contract.incomplete", contracts[0], ledger["blocking_units"])
    audit.equal(
        "atlas.contract.tri_view_total",
        contracts[1] + contracts[2] + contracts[3] + contracts[4],
        ledger["tri_view_ready_units"],
    )
    audit.check(
        "atlas.no_absolute_paths",
        not any(is_absolute_string(value) for value in walk_values(meta)),
        "atlas metadata contains no absolute paths",
    )


def validate_digest(audit: Audit, ledger: dict[str, Any], path: Path) -> None:
    text = path.read_text()
    match = re.search(
        r"<!-- METHANET_RELEASE_LEDGER_BEGIN -->\s*```json\s*(\{.*?\})\s*```\s*<!-- METHANET_RELEASE_LEDGER_END -->",
        text,
        flags=re.DOTALL,
    )
    audit.check("digest.machine_ledger.present", match is not None, "generated machine ledger block is present")
    if not match:
        return
    try:
        digest_ledger = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        audit.check("digest.machine_ledger.valid_json", False, str(exc))
        return
    expected = {key: ledger[key] for key in (*COUNT_KEYS, *REQUIRED_PROVENANCE)}
    audit.equal("digest.machine_ledger", digest_ledger, expected)


def validate_report_html(audit: Audit, ledger: dict[str, Any], path: Path) -> None:
    text = path.read_text()
    for key in (*COUNT_KEYS, "snapshot_date"):
        pattern = rf"data-release-key=[\"']{re.escape(key)}[\"'][^>]*>(.*?)</"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        audit.check(f"report_html.{key}.present", match is not None, "release-key marker is present")
        if match:
            visible = html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip().replace(",", "")
            audit.equal(f"report_html.{key}", visible, str(ledger[key]))


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=("gate", "status", "detail"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    audit = Audit()
    ledger = load_json(args.release_ledger)
    decision = load_json(args.freeze_decision)
    summary_rows = load_tsv(args.freeze_summary)

    for key in REQUIRED_PROVENANCE:
        audit.check(f"ledger.{key}.present", bool(str(ledger.get(key, "")).strip()), "provenance key is present")
    for key in COUNT_KEYS:
        audit.check(f"ledger.{key}.integer", isinstance(ledger.get(key), int), f"value={ledger.get(key)!r}")
    for ledger_key, decision_key in DECISION_KEYS.items():
        audit.equal(f"freeze_decision.{decision_key}", decision.get(decision_key), ledger[ledger_key])
    audit.equal("freeze_decision.release_state", bool(decision.get("freeze_allowed")), ledger["release_state"] == "ready")

    for ledger_key, column in SUMMARY_COLUMNS.items():
        audit.equal(
            f"freeze_summary.total.{column}",
            sum(as_int(row.get(column, 0)) for row in summary_rows),
            ledger[ledger_key],
        )
    validate_lane_summary(audit, ledger, summary_rows)

    report_summary = load_json(args.report_summary) if args.report_summary else None
    report_manifest = load_json(args.report_bundle_manifest) if args.report_bundle_manifest else None
    if report_summary is not None:
        validate_report(audit, ledger, report_summary, report_manifest)
    elif report_manifest is not None:
        embedded = report_manifest.get("summary")
        audit.check("report_manifest.summary.present", isinstance(embedded, dict), "embedded summary is present")
        if isinstance(embedded, dict):
            validate_report(audit, ledger, embedded, report_manifest)

    if args.config:
        validate_config(audit, ledger, args.config)
    if args.atlas:
        validate_atlas(audit, ledger, load_json(args.atlas))
    if args.digest:
        validate_digest(audit, ledger, args.digest)
    if args.report_html:
        validate_report_html(audit, ledger, args.report_html)

    if args.output:
        write_audit(args.output, audit.rows)
    result = {
        "gates": len(audit.rows),
        "passed": len(audit.rows) - len(audit.failed),
        "failed": len(audit.failed),
        "failures": audit.failed,
    }
    print(json.dumps(result, indent=2))
    return 1 if audit.failed else 0


if __name__ == "__main__":
    sys.exit(main())
