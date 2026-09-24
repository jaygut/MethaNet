#!/usr/bin/env python3
"""Validate tracked atlas contracts and optional local release artifacts.

The default mode uses only tracked files. Local mode reads small TSV and JSON
manifests and gate records; it never opens Parquet payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POINTER_PATHS = (
    "lane_registry",
    "freeze_manifest",
    "freeze_decision",
    "release_ledger",
    "tracked_ledger_snapshot",
    "source_report_bundle",
    "reconciled_public_report_bundle",
    "site_source",
)
REGISTRY_PATHS = (
    "source_lane_manifest",
    "functional_manifest",
    "functional_per_mag_dirs",
    "functional_warehouse_dir",
    "consolidation_output_dir",
    "esm2_artifacts_dirs",
    "glm2_artifacts_dirs",
    "gap_register",
    "source_provenance_dir",
    "source_provenance_checksums",
)
LANE_SUMS = {
    "registered_units": "registry_denominator_units",
    "esm2_units": "esm2_units",
    "glm2_units": "glm2_units",
    "functional_payload_units": "functional_payload_units",
    "release_required_units": "release_required_units",
    "explicit_non_runnable_gaps": "release_excluded_units",
    "tri_view_ready_units": "tri_view_ready_units",
    "schema_normalized_units": "schema_normalized_units",
    "schema_normalized_tri_view_units": "schema_normalized_tri_view_units",
    "pipeline_normalized_tri_view_units": "pipeline_normalized_tri_view_units",
    "mechanism_comparable_units": "canonical_mechanism_tri_view_units",
    "annotation_complete_tri_view_units": "annotation_complete_tri_view_units",
    "source_scaffold_tri_view_units": "source_scaffold_tri_view_units",
}
FLAGS = (
    "release_required",
    "release_excluded",
    "has_esm2",
    "has_glm2",
    "has_functional",
    "tri_view_ready",
    "schema_normalized",
    "mechanism_comparable",
    "sample_linked",
    "abundance_weighted",
    "environmentally_contextualized",
    "field_validated",
    "calibrated",
)
FUNCTIONAL_STATUS = {
    "complete": "functional_complete",
    "failed": "functional_failed",
    "partial": "functional_partial",
    "not_started": "functional_not_started",
    "non_runnable_gap": "functional_non_runnable_gap",
}
FORMAL_STATUS = {
    "complete_pipeline_normalized_tri_view_comparability_pending": "pipeline_normalized_tri_view_units",
    "complete_annotation_tri_view_harmonization_pending": "annotation_complete_tri_view_units",
    "complete_source_scaffold_tri_view": "source_scaffold_tri_view_units",
}


@dataclass
class Audit:
    checks: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def expect(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.errors.append(message)
        return condition

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def number(value: Any, audit: Audit, label: str) -> int:
    if isinstance(value, bool):
        audit.error(f"{label}: expected nonnegative integer")
        return 0
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        result = int(value)
    else:
        audit.error(f"{label}: expected nonnegative integer")
        return 0
    audit.expect(result >= 0, f"{label}: negative count")
    return result


def repo_path(root: Path, value: Any, audit: Audit, label: str) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        audit.error(f"{label}: expected repo-relative POSIX path")
        return None
    if PurePosixPath(value).is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        audit.error(f"{label}: unsafe path {value!r}")
        return None
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        audit.error(f"{label}: path escapes repository")
        return None
    return path


def read_json(path: Path | None, audit: Audit, label: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        audit.error(f"{label}: cannot read JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        audit.error(f"{label}: expected JSON object")
        return {}
    return value


def read_tsv(
    path: Path | None, audit: Audit, label: str, required: set[str]
) -> list[dict[str, str]]:
    if path is None:
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = required - set(reader.fieldnames or [])
            if missing:
                audit.error(f"{label}: missing columns {sorted(missing)}")
                return []
            return list(reader)
    except OSError as exc:
        audit.error(f"{label}: cannot read TSV: {exc}")
        return []


def check_sha(path: Path | None, expected: Any, audit: Audit, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        audit.error(f"{label}: invalid SHA-256 pin")
        return
    if path is None:
        return
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        audit.error(f"{label}: cannot hash file: {exc}")
        return
    audit.expect(digest.hexdigest() == expected, f"{label}: SHA-256 mismatch")


def string_list(
    value: Any, audit: Audit, label: str, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        audit.error(f"{label}: expected string list")
        return []
    if any(not isinstance(item, str) or not item.strip() for item in value):
        audit.error(f"{label}: blank or non-string member")
        return []
    audit.expect(len(value) == len(set(value)), f"{label}: duplicate members")
    return value


def validate_contract(
    contract: dict[str, Any],
    pointer: dict[str, Any],
    pointer_file: str,
    registry_ids: set[str],
    root: Path,
    audit: Audit,
) -> dict[str, str]:
    audit.expect(
        contract.get("schema_version") == pointer.get("schema_version"),
        "contract schema version differs from release pointer",
    )
    audit.expect(
        isinstance(contract.get("contract_id"), str)
        and bool(contract["contract_id"].strip()),
        "contract_id missing",
    )
    authorities = contract.get("authorities")
    if not isinstance(authorities, dict):
        audit.error("contract.authorities: expected object")
        return {}
    for name in ("current_release_pointer", "lane_registry", "tracked_release_ledger"):
        path = repo_path(
            root, authorities.get(name), audit, f"contract.authorities.{name}"
        )
        if path is not None:
            audit.expect(path.is_file(), f"contract.authorities.{name}: file missing")
    audit.expect(
        authorities.get("current_release_pointer") == pointer_file,
        "contract authority pointer mismatch",
    )
    for authority, source in (
        ("lane_registry", "lane_registry"),
        ("tracked_release_ledger", "tracked_ledger_snapshot"),
    ):
        audit.expect(
            authorities.get(authority) == pointer.get(source),
            f"contract authority {authority} differs from release pointer",
        )
    for name in ("warehouse_table_manifest", "warehouse_validation_gates"):
        value = authorities.get(name)
        audit.expect(
            isinstance(value, str)
            and value not in {"", ".", ".."}
            and "/" not in value
            and "\\" not in value,
            f"contract.authorities.{name}: expected basename",
        )

    identity = contract.get("identity")
    if not isinstance(identity, dict):
        audit.error("contract.identity: expected object")
        identity = {}
    audit.expect(
        identity.get("canonical_molecular_key") == ["lane_id", "proteome_id"],
        "contract identity must use lane_id + proteome_id",
    )
    string_list(
        identity.get("pipeline_warehouse_key"),
        audit,
        "contract.identity.pipeline_warehouse_key",
    )
    evidence = contract.get("evidence_contracts")
    if not isinstance(evidence, dict) or not evidence:
        audit.error("contract.evidence_contracts: expected nonempty object")
        evidence = {}
    lane_family: dict[str, str] = {}
    for family, details in evidence.items():
        label = f"contract.evidence_contracts.{family}"
        if not isinstance(details, dict):
            audit.error(f"{label}: expected object")
            continue
        lanes = string_list(details.get("lanes"), audit, f"{label}.lanes")
        audit.expect(
            bool(str(details.get("meaning") or "").strip()), f"{label}: missing meaning"
        )
        for lane in lanes:
            if lane in lane_family:
                audit.error(f"{label}: lane {lane} assigned to multiple contracts")
            lane_family[lane] = family
    audit.expect(
        set(lane_family) == registry_ids, "contract lanes differ from registry"
    )

    tables = contract.get("tables")
    if not isinstance(tables, list) or not tables:
        audit.error("contract.tables: expected nonempty list")
        tables = []
    seen: set[tuple[str, str]] = set()
    for index, table in enumerate(tables):
        label = f"contract.tables[{index}]"
        if not isinstance(table, dict):
            audit.error(f"{label}: expected object")
            continue
        family = table.get("evidence_contract")
        name = table.get("logical_table")
        audit.expect(
            family == "release" or family in evidence,
            f"{label}: unknown evidence contract",
        )
        if not isinstance(name, str) or not name.strip():
            audit.error(f"{label}: missing logical_table")
            continue
        key = (str(family), name)
        audit.expect(key not in seen, f"{label}: duplicate table {key}")
        seen.add(key)
        for field_name in ("grain", "key_status", "null_semantics", "claim_scope"):
            audit.expect(
                isinstance(table.get(field_name), str)
                and bool(table[field_name].strip()),
                f"{label}: missing {field_name}",
            )
        required = string_list(
            table.get("required_columns"), audit, f"{label}.required_columns"
        )
        primary = string_list(
            table.get("primary_key"),
            audit,
            f"{label}.primary_key",
            allow_empty=True,
        )
        statuses = string_list(
            table.get("status_columns"),
            audit,
            f"{label}.status_columns",
            allow_empty=True,
        )
        audit.expect(
            set(primary) <= set(required), f"{label}: primary key not required"
        )
        audit.expect(
            set(statuses) <= set(required), f"{label}: status column not required"
        )
        if not primary:
            audit.expect(
                str(table.get("key_status", "")).startswith("undeclared"),
                f"{label}: empty key lacks undeclared status",
            )
        foreign_keys = table.get("foreign_keys", [])
        if not isinstance(foreign_keys, list):
            audit.error(f"{label}.foreign_keys: expected list")
        else:
            for relation in foreign_keys:
                if not isinstance(relation, dict):
                    audit.error(f"{label}.foreign_keys: expected object")
                    continue
                string_list(
                    relation.get("columns"), audit, f"{label}.foreign_keys.columns"
                )
                audit.expect(
                    isinstance(relation.get("target"), str)
                    and bool(relation["target"].strip()),
                    f"{label}.foreign_keys: missing target",
                )
    audit.expect(
        ("release", "freeze_manifest") in seen, "contract lacks freeze_manifest"
    )
    semantics = contract.get("status_semantics")
    audit.expect(
        isinstance(semantics, dict) and bool(semantics),
        "contract.status_semantics: expected nonempty object",
    )
    graph = contract.get("graph_projection")
    if not isinstance(graph, dict):
        audit.error("contract.graph_projection: expected object")
    else:
        for name in ("node_classes", "relation_classes", "required_assertion_fields"):
            string_list(graph.get(name), audit, f"contract.graph_projection.{name}")
    return lane_family


def validate_registry(
    root: Path, path: Path | None, audit: Audit
) -> dict[str, dict[str, str]]:
    rows = read_tsv(
        path,
        audit,
        "lane registry",
        {"lane_id", "denominator_units", "functional_warehouse_dir"},
    )
    by_lane: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        label = f"lane registry row {index}"
        lane = str(row.get("lane_id") or "").strip()
        if not lane or lane in by_lane:
            audit.error(f"{label}: blank or duplicate lane_id {lane!r}")
            continue
        by_lane[lane] = row
        audit.expect(
            number(row.get("denominator_units"), audit, f"{label}.denominator_units")
            > 0,
            f"{label}: nonpositive denominator",
        )
        for name in REGISTRY_PATHS:
            for value in str(row.get(name) or "").split(";"):
                if value.strip():
                    repo_path(root, value.strip(), audit, f"{label}.{name}")
    audit.expect(bool(by_lane), "lane registry has no rows")
    return by_lane


def validate_ledger(
    pointer: dict[str, Any],
    ledger: dict[str, Any],
    registry: dict[str, dict[str, str]],
    audit: Audit,
) -> dict[str, dict[str, Any]]:
    audit.expect(
        pointer.get("schema_version") == ledger.get("schema_version"),
        "release schema version mismatch",
    )
    audit.expect(
        pointer.get("snapshot_date") == ledger.get("snapshot_date"),
        "release snapshot date mismatch",
    )
    audit.expect(
        pointer.get("freeze_manifest_sha256") == ledger.get("freeze_manifest_sha256"),
        "freeze manifest SHA differs between pointer and ledger",
    )
    audit.expect(
        "noindex" in str(pointer.get("release_scope", ""))
        and "noindex" in str(ledger.get("indexing_decision", "")),
        "controlled release is not consistently noindex",
    )
    audit.expect(ledger.get("release_state") == "ready", "active release is not ready")
    rows = ledger.get("lanes")
    if not isinstance(rows, list):
        audit.error("ledger.lanes: expected list")
        rows = []
    by_lane: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            audit.error("ledger.lanes: non-object row")
            continue
        lane = row.get("lane_id")
        if not isinstance(lane, str) or not lane or lane in by_lane:
            audit.error(f"ledger.lanes: blank or duplicate lane_id {lane!r}")
            continue
        by_lane[lane] = row
    audit.expect(set(by_lane) == set(registry), "ledger lanes differ from registry")
    for lane, row in by_lane.items():
        if lane not in registry:
            continue
        label = f"ledger.{lane}"
        denominator = number(
            registry[lane].get("denominator_units"),
            audit,
            f"registry.{lane}.denominator_units",
        )
        for name in ("expected_units", "registry_denominator_units"):
            audit.expect(
                number(row.get(name), audit, f"{label}.{name}") == denominator,
                f"{label}.{name}: denominator mismatch",
            )
        required = number(
            row.get("release_required_units"), audit, f"{label}.release_required_units"
        )
        excluded = number(
            row.get("release_excluded_units"), audit, f"{label}.release_excluded_units"
        )
        audit.expect(
            required + excluded == denominator, f"{label}: required + excluded mismatch"
        )
        partition = sum(
            number(row.get(name), audit, f"{label}.{name}")
            for name in FUNCTIONAL_STATUS.values()
        )
        audit.expect(
            partition == denominator, f"{label}: functional status partition mismatch"
        )
        audit.expect(
            number(
                row.get("registry_manifest_gap_rows"),
                audit,
                f"{label}.registry_manifest_gap_rows",
            )
            == number(
                row.get("functional_non_runnable_gap"),
                audit,
                f"{label}.functional_non_runnable_gap",
            ),
            f"{label}: gap rows mismatch",
        )
        audit.expect(
            number(
                row.get("release_tri_view_ready_units"),
                audit,
                f"{label}.release_tri_view_ready_units",
            )
            <= min(
                required,
                number(
                    row.get("tri_view_ready_units"),
                    audit,
                    f"{label}.tri_view_ready_units",
                ),
            ),
            f"{label}: release tri-views exceed denominators",
        )
    for total, lane_field in LANE_SUMS.items():
        observed = number(ledger.get(total), audit, f"ledger.{total}")
        expected = sum(
            number(row.get(lane_field), audit, f"ledger.{lane}.{lane_field}")
            for lane, row in by_lane.items()
        )
        audit.expect(observed == expected, f"ledger.{total}: lane sum mismatch")
    registered = number(
        ledger.get("registered_units"), audit, "ledger.registered_units"
    )
    required = number(
        ledger.get("release_required_units"), audit, "ledger.release_required_units"
    )
    excluded = number(
        ledger.get("explicit_non_runnable_gaps"),
        audit,
        "ledger.explicit_non_runnable_gaps",
    )
    tri_view = number(
        ledger.get("tri_view_ready_units"), audit, "ledger.tri_view_ready_units"
    )
    audit.expect(registered == required + excluded, "ledger release partition mismatch")
    for name in ("esm2_units", "glm2_units", "functional_payload_units"):
        count = number(ledger.get(name), audit, f"ledger.{name}")
        audit.expect(
            tri_view <= count <= registered, f"ledger.{name}: invalid tri-view bound"
        )
    audit.expect(
        number(
            ledger.get("schema_normalized_tri_view_units"),
            audit,
            "ledger.schema_normalized_tri_view_units",
        )
        <= tri_view,
        "ledger schema-normalized tri-views exceed data-complete tri-views",
    )
    audit.expect(
        number(
            ledger.get("mechanism_comparable_units"),
            audit,
            "ledger.mechanism_comparable_units",
        )
        <= number(
            ledger.get("pipeline_normalized_tri_view_units"),
            audit,
            "ledger.pipeline_normalized_tri_view_units",
        ),
        "ledger mechanism-comparable rows exceed pipeline-normalized rows",
    )
    if ledger.get("release_state") == "ready":
        audit.expect(
            sum(
                number(
                    row.get("release_tri_view_ready_units"),
                    audit,
                    f"ledger.{lane}.release_tri_view_ready_units",
                )
                for lane, row in by_lane.items()
            )
            == required,
            "ready release has incomplete required tri-views",
        )
        audit.expect(
            number(ledger.get("blocking_units"), audit, "ledger.blocking_units") == 0,
            "ready release has blocking units",
        )
    return by_lane


def validate_freeze(
    path: Path | None,
    decision: dict[str, Any],
    ledger: dict[str, Any],
    ledger_lanes: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    audit: Audit,
) -> int:
    freeze_table = next(
        (
            table
            for table in contract.get("tables", [])
            if isinstance(table, dict)
            and table.get("evidence_contract") == "release"
            and table.get("logical_table") == "freeze_manifest"
        ),
        {},
    )
    required = set(freeze_table.get("required_columns", []))
    required.update(FLAGS)
    required.update(
        {"formal_tri_view_status", "functional_status", "release_exclusion_reason"}
    )
    rows = read_tsv(path, audit, "freeze manifest", required)
    seen: set[tuple[str, str]] = set()
    totals: Counter[str] = Counter()
    lanes: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(rows, start=2):
        lane = str(row.get("lane_id") or "").strip()
        proteome = str(row.get("proteome_id") or "").strip()
        key = (lane, proteome)
        if not lane or not proteome or key in seen:
            audit.error(f"freeze row {index}: blank or duplicate lane_id/proteome_id")
        seen.add(key)
        if lane not in ledger_lanes:
            audit.error(f"freeze row {index}: unregistered lane {lane!r}")
        totals["registered_units"] += 1
        lanes[lane]["registry_denominator_units"] += 1
        flag: dict[str, bool] = {}
        for name in FLAGS:
            value = str(row.get(name) or "").lower().strip()
            if value not in {"true", "false"}:
                audit.error(f"freeze row {index}: invalid {name} flag")
            flag[name] = value == "true"
            if flag[name]:
                totals[name] += 1
                lanes[lane][name] += 1
        audit.expect(
            flag["release_required"] != flag["release_excluded"],
            f"freeze row {index}: release flags not complementary",
        )
        if flag["release_excluded"]:
            audit.expect(
                bool(str(row.get("release_exclusion_reason") or "").strip()),
                f"freeze row {index}: exclusion reason missing",
            )
        if flag["tri_view_ready"]:
            audit.expect(
                flag["has_esm2"] and flag["has_glm2"] and flag["has_functional"],
                f"freeze row {index}: tri-view lacks payload",
            )
        if flag["schema_normalized"]:
            audit.expect(
                flag["tri_view_ready"],
                f"freeze row {index}: normalized row lacks tri-view",
            )
        if flag["mechanism_comparable"]:
            audit.expect(
                flag["schema_normalized"],
                f"freeze row {index}: comparable row lacks normalized tri-view",
            )
        if flag["schema_normalized"] and flag["tri_view_ready"]:
            totals["schema_normalized_tri_view_units"] += 1
            lanes[lane]["schema_normalized_tri_view_units"] += 1
        if flag["release_required"] and flag["tri_view_ready"]:
            lanes[lane]["release_tri_view_ready_units"] += 1
        if flag["release_required"] and not flag["tri_view_ready"]:
            totals["blocking_units"] += 1
        status = str(row.get("functional_status") or "").strip()
        if status not in FUNCTIONAL_STATUS:
            audit.error(f"freeze row {index}: unknown functional_status {status!r}")
        else:
            lanes[lane][FUNCTIONAL_STATUS[status]] += 1
            if status == "non_runnable_gap":
                audit.expect(
                    flag["release_excluded"],
                    f"freeze row {index}: non-runnable gap is required",
                )
        formal = str(row.get("formal_tri_view_status") or "").strip()
        if formal in FORMAL_STATUS:
            field_name = FORMAL_STATUS[formal]
            totals[field_name] += 1
            lanes[lane][field_name] += 1
        elif formal != "incomplete_tri_view":
            audit.error(
                f"freeze row {index}: unknown formal_tri_view_status {formal!r}"
            )

    root_fields = {
        "release_required_units": "release_required",
        "explicit_non_runnable_gaps": "release_excluded",
        "esm2_units": "has_esm2",
        "glm2_units": "has_glm2",
        "functional_payload_units": "has_functional",
        "tri_view_ready_units": "tri_view_ready",
        "schema_normalized_units": "schema_normalized",
        "mechanism_comparable_units": "mechanism_comparable",
        "sample_linked_units": "sample_linked",
        "abundance_weighted_units": "abundance_weighted",
        "environmentally_contextualized_units": "environmentally_contextualized",
        "field_validated_units": "field_validated",
        "calibrated_units": "calibrated",
    }
    for ledger_field, counter_field in root_fields.items():
        if ledger_field in ledger:
            audit.expect(
                totals[counter_field]
                == number(ledger[ledger_field], audit, f"ledger.{ledger_field}"),
                f"freeze {ledger_field}: count differs from ledger",
            )
    for field_name in (
        "registered_units",
        "schema_normalized_tri_view_units",
        "pipeline_normalized_tri_view_units",
        "annotation_complete_tri_view_units",
        "source_scaffold_tri_view_units",
        "blocking_units",
    ):
        if field_name in ledger:
            audit.expect(
                totals[field_name]
                == number(ledger[field_name], audit, f"ledger.{field_name}"),
                f"freeze {field_name}: count differs from ledger",
            )
    for lane, counts in lanes.items():
        if lane not in ledger_lanes:
            continue
        aliases = {
            "release_required_units": "release_required",
            "release_excluded_units": "release_excluded",
            "esm2_units": "has_esm2",
            "glm2_units": "has_glm2",
            "functional_payload_units": "has_functional",
            "tri_view_ready_units": "tri_view_ready",
            "schema_normalized_units": "schema_normalized",
            "canonical_mechanism_tri_view_units": "mechanism_comparable",
            "registry_manifest_gap_rows": "functional_non_runnable_gap",
        }
        for expected_name, observed_name in aliases.items():
            counts[expected_name] = counts[observed_name]
        for field_name in {
            *LANE_SUMS.values(),
            "release_tri_view_ready_units",
            "registry_manifest_gap_rows",
            *FUNCTIONAL_STATUS.values(),
        }:
            if field_name in ledger_lanes[lane]:
                audit.expect(
                    counts[field_name]
                    == number(
                        ledger_lanes[lane][field_name],
                        audit,
                        f"ledger.{lane}.{field_name}",
                    ),
                    f"freeze {lane}.{field_name}: count differs from ledger",
                )
    audit.expect(
        number(decision.get("unit_rows"), audit, "freeze_decision.unit_rows")
        == len(rows),
        "freeze decision unit_rows differs from manifest",
    )
    audit.expect(
        number(
            decision.get("release_tri_view_ready_units"),
            audit,
            "freeze_decision.release_tri_view_ready_units",
        )
        == sum(counts["release_tri_view_ready_units"] for counts in lanes.values()),
        "freeze decision release tri-views differ from manifest",
    )
    return len(rows)


def validate_warehouses(
    root: Path,
    registry: dict[str, dict[str, str]],
    ledger_lanes: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    lane_family: dict[str, str],
    audit: Audit,
) -> dict[str, Any]:
    authorities = contract.get("authorities", {})
    if not isinstance(authorities, dict):
        return {}
    manifest_name = authorities.get("warehouse_table_manifest")
    gates_name = authorities.get("warehouse_validation_gates")
    if not isinstance(manifest_name, str) or not isinstance(gates_name, str):
        return {}
    summaries: dict[str, Any] = {}
    for lane, registry_row in registry.items():
        warehouse = repo_path(
            root,
            registry_row.get("functional_warehouse_dir"),
            audit,
            f"registry.{lane}.functional_warehouse_dir",
        )
        if warehouse is None:
            continue
        tables = read_tsv(
            warehouse / manifest_name,
            audit,
            f"warehouse {lane} tables",
            {"table", "path", "rows", "columns", "bytes"},
        )
        gates = read_tsv(
            warehouse / gates_name,
            audit,
            f"warehouse {lane} gates",
            {"gate", "status", "detail"},
        )
        audit.expect(bool(tables), f"warehouse {lane}: empty table manifest")
        audit.expect(bool(gates), f"warehouse {lane}: empty validation gates")
        by_table: dict[str, dict[str, str]] = {}
        for row in tables:
            table = str(row.get("table") or "").strip()
            if not table or table in by_table:
                audit.error(f"warehouse {lane}: blank or duplicate table {table!r}")
                continue
            by_table[table] = row
            audit.expect(
                bool(str(row.get("path") or "").strip()),
                f"warehouse {lane}.{table}: artifact path missing",
            )
            for column in ("rows", "columns", "bytes"):
                number(row.get(column), audit, f"warehouse {lane}.{table}.{column}")
        required = {
            item["logical_table"]
            for item in contract.get("tables", [])
            if isinstance(item, dict)
            and item.get("evidence_contract") == lane_family.get(lane)
        }
        audit.expect(
            required <= set(by_table),
            f"warehouse {lane}: missing contract tables {sorted(required - set(by_table))}",
        )
        family = lane_family.get(lane)
        ledger_row = ledger_lanes.get(lane, {})
        expected_dim_mag = number(
            ledger_row.get(
                "registry_denominator_units"
                if family == "source_scaffold"
                else "functional_complete"
            ),
            audit,
            f"ledger.{lane}.dim_mag_expected",
        )
        if "dim_mag" in by_table:
            audit.expect(
                number(
                    by_table["dim_mag"].get("rows"),
                    audit,
                    f"warehouse {lane}.dim_mag.rows",
                )
                == expected_dim_mag,
                f"warehouse {lane}: dim_mag count differs from ledger",
            )
        statuses: Counter[str] = Counter()
        nonpass: list[dict[str, str]] = []
        seen_gates: set[str] = set()
        for row in gates:
            name = str(row.get("gate") or "").strip()
            status = str(row.get("status") or "").strip().lower()
            if not name or name in seen_gates:
                audit.error(f"warehouse {lane}: blank or duplicate gate {name!r}")
            seen_gates.add(name)
            statuses[status] += 1
            if status != "pass":
                nonpass.append(
                    {
                        "gate": name,
                        "status": status,
                        "detail": str(row.get("detail") or "")[:160],
                    }
                )
                if status in {"", "fail", "failed", "error"}:
                    audit.error(f"warehouse {lane}: gate {name} is {status!r}")
        if nonpass:
            audit.warn(f"warehouse {lane}: {len(nonpass)} non-pass gates; see receipt")
        summaries[lane] = {
            "tables": len(by_table),
            "gates": len(gates),
            "gate_statuses": dict(sorted(statuses.items())),
            "nonpass_gates": nonpass[:15],
        }
    return summaries


def validate(
    repo_root: Path = ROOT,
    pointer_file: str = "configs/atlas_current_release.json",
    contract_file: str = "contracts/atlas_data_contract_v1.json",
    require_local: bool = False,
) -> dict[str, Any]:
    root = repo_root.resolve()
    audit = Audit()
    pointer_path = repo_path(root, pointer_file, audit, "current release pointer")
    contract_path = repo_path(root, contract_file, audit, "data contract")
    pointer = read_json(pointer_path, audit, "current release pointer")
    contract = read_json(contract_path, audit, "data contract")
    audit.expect(
        isinstance(pointer.get("release_id"), str)
        and bool(pointer["release_id"].strip()),
        "release pointer lacks release_id",
    )
    paths: dict[str, Path] = {}
    for name in POINTER_PATHS:
        path = repo_path(root, pointer.get(name), audit, f"pointer.{name}")
        if path is not None:
            paths[name] = path
    registry_path = paths.get("lane_registry")
    ledger_path = paths.get("tracked_ledger_snapshot")
    audit.expect(
        registry_path is not None and registry_path.is_file(),
        "tracked lane registry missing",
    )
    audit.expect(
        ledger_path is not None and ledger_path.is_file(),
        "tracked release ledger missing",
    )
    audit.expect(
        paths.get("site_source") is not None and paths["site_source"].is_dir(),
        "tracked site source missing",
    )
    check_sha(
        registry_path, pointer.get("lane_registry_sha256"), audit, "lane registry"
    )
    check_sha(
        ledger_path,
        pointer.get("release_ledger_sha256"),
        audit,
        "tracked release ledger",
    )
    registry = validate_registry(root, registry_path, audit)
    ledger = read_json(ledger_path, audit, "tracked release ledger")
    lane_family = validate_contract(
        contract, pointer, pointer_file, set(registry), root, audit
    )
    ledger_lanes = validate_ledger(pointer, ledger, registry, audit)
    local: dict[str, Any] = {}
    if require_local:
        for name in ("freeze_manifest", "freeze_decision", "release_ledger"):
            check_sha(paths.get(name), pointer.get(name + "_sha256"), audit, name)
        local_ledger = read_json(
            paths.get("release_ledger"), audit, "local release ledger"
        )
        audit.expect(
            local_ledger == ledger, "local ledger differs from tracked snapshot"
        )
        decision = read_json(paths.get("freeze_decision"), audit, "freeze decision")
        audit.expect(
            decision.get("freeze_allowed") is True, "freeze decision is not allowed"
        )
        for key, ledger_key in (
            ("unit_rows", "registered_units"),
            ("release_required_units", "release_required_units"),
            ("release_excluded_units", "explicit_non_runnable_gaps"),
        ):
            audit.expect(
                number(decision.get(key), audit, f"freeze_decision.{key}")
                == number(ledger.get(ledger_key), audit, f"ledger.{ledger_key}"),
                f"freeze decision {key} differs from ledger",
            )
        local["freeze_rows"] = validate_freeze(
            paths.get("freeze_manifest"),
            decision,
            ledger,
            ledger_lanes,
            contract,
            audit,
        )
        local["warehouse_summaries"] = validate_warehouses(
            root,
            registry,
            ledger_lanes,
            contract,
            lane_family,
            audit,
        )
    return {
        "status": "fail"
        if audit.errors
        else "pass_with_warnings"
        if audit.warnings
        else "pass",
        "mode": "require_local" if require_local else "contract_only",
        "release_id": pointer.get("release_id"),
        "snapshot_date": pointer.get("snapshot_date"),
        "lane_registry_sha256": pointer.get("lane_registry_sha256"),
        "release_ledger_sha256": pointer.get("release_ledger_sha256"),
        "freeze_manifest_sha256": pointer.get("freeze_manifest_sha256"),
        "checks": audit.checks,
        "error_count": len(audit.errors),
        "warning_count": len(audit.warnings),
        "errors": audit.errors[:30],
        "warnings": audit.warnings[:30],
        "registered_lanes": sorted(registry),
        **local,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--pointer", default="configs/atlas_current_release.json")
    parser.add_argument("--contract", default="contracts/atlas_data_contract_v1.json")
    parser.add_argument("--require-local", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    receipt = validate(args.repo_root, args.pointer, args.contract, args.require_local)
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        try:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            receipt["status"] = "fail"
            receipt["error_count"] += 1
            receipt["errors"].append(f"cannot write receipt: {exc}")
            rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    return int(receipt["status"] == "fail")


if __name__ == "__main__":
    raise SystemExit(main())
