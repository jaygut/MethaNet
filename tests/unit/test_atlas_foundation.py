"""Fresh-clone and local-manifest checks for the atlas foundation validator."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/reports/validate_atlas_foundation.py"
SPEC = importlib.util.spec_from_file_location("validate_atlas_foundation", SCRIPT)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_clone(tmp_path: Path) -> Path:
    """Copy tracked contract inputs; ignored warehouses are absent."""
    (tmp_path / "web/emergentbiome-methanet").mkdir(parents=True)
    for relative in (
        "configs/atlas_current_release.json",
        "configs/methanet_atlas_lanes.tsv",
        "contracts/atlas_data_contract_v1.json",
        "docs/releases/atlas_20260810_release_ledger.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


def _toy_repo(tmp_path: Path) -> Path:
    registry = tmp_path / "configs/methanet_atlas_lanes.tsv"
    freeze = tmp_path / "results/freeze/freeze_manifest.tsv"
    decision = tmp_path / "results/freeze/freeze_decision.json"
    warehouse = tmp_path / "results/warehouse"
    (tmp_path / "web/site").mkdir(parents=True)
    _tsv(
        registry,
        [
            {
                "lane_id": "lane_a",
                "denominator_units": "2",
                "functional_warehouse_dir": "results/warehouse",
            }
        ],
    )
    true = {
        "lane_id": "lane_a",
        "proteome_id": "unit_1",
        "functional_status": "complete",
        "release_required": "true",
        "release_excluded": "false",
        "release_exclusion_reason": "",
        "has_esm2": "true",
        "has_glm2": "true",
        "has_functional": "true",
        "tri_view_ready": "true",
        "schema_normalized": "true",
        "mechanism_comparable": "false",
        "sample_linked": "false",
        "abundance_weighted": "false",
        "environmentally_contextualized": "false",
        "field_validated": "false",
        "calibrated": "false",
        "formal_tri_view_status": (
            "complete_pipeline_normalized_tri_view_comparability_pending"
        ),
        "claim_scope": "molecular screening",
    }
    gap = {
        **true,
        "proteome_id": "unit_2",
        "functional_status": "non_runnable_gap",
        "release_required": "false",
        "release_excluded": "true",
        "release_exclusion_reason": "missing payload",
        "has_esm2": "false",
        "has_glm2": "false",
        "has_functional": "false",
        "tri_view_ready": "false",
        "schema_normalized": "false",
        "formal_tri_view_status": "incomplete_tri_view",
    }
    _tsv(freeze, [true, gap])
    _json(
        decision,
        {
            "freeze_allowed": True,
            "unit_rows": 2,
            "release_required_units": 1,
            "release_excluded_units": 1,
            "release_tri_view_ready_units": 1,
        },
    )
    lane = {
        "lane_id": "lane_a",
        "expected_units": 2,
        "registry_denominator_units": 2,
        "registry_manifest_gap_rows": 1,
        "release_required_units": 1,
        "release_excluded_units": 1,
        "release_tri_view_ready_units": 1,
        "functional_complete": 1,
        "functional_failed": 0,
        "functional_partial": 0,
        "functional_not_started": 0,
        "functional_non_runnable_gap": 1,
        "esm2_units": 1,
        "glm2_units": 1,
        "functional_payload_units": 1,
        "tri_view_ready_units": 1,
        "schema_normalized_units": 1,
        "schema_normalized_tri_view_units": 1,
        "pipeline_normalized_tri_view_units": 1,
        "canonical_mechanism_tri_view_units": 0,
        "annotation_complete_tri_view_units": 0,
        "source_scaffold_tri_view_units": 0,
    }
    ledger = {
        "schema_version": "1.0.0",
        "snapshot_date": "2026-08-10",
        "indexing_decision": "noindex_controlled_diligence",
        "release_state": "ready",
        "freeze_manifest_sha256": _sha(freeze),
        "lanes": [lane],
        "registered_units": 2,
        "release_required_units": 1,
        "explicit_non_runnable_gaps": 1,
        "esm2_units": 1,
        "glm2_units": 1,
        "functional_payload_units": 1,
        "tri_view_ready_units": 1,
        "schema_normalized_units": 1,
        "schema_normalized_tri_view_units": 1,
        "pipeline_normalized_tri_view_units": 1,
        "mechanism_comparable_units": 0,
        "annotation_complete_tri_view_units": 0,
        "source_scaffold_tri_view_units": 0,
        "blocking_units": 0,
        "sample_linked_units": 0,
        "abundance_weighted_units": 0,
        "environmentally_contextualized_units": 0,
        "field_validated_units": 0,
        "calibrated_units": 0,
    }
    _json(tmp_path / "docs/releases/ledger.json", ledger)
    _json(tmp_path / "results/freeze/release_ledger.json", ledger)
    _tsv(
        warehouse / "cohort_table_manifest.tsv",
        [
            {
                "table": "dim_mag",
                "path": "parquet/dim_mag/part-00000.parquet",
                "rows": "1",
                "columns": "3",
                "bytes": "10",
            }
        ],
    )
    _tsv(
        warehouse / "validation_gates.tsv",
        [
            {"gate": "selected_count", "status": "pass", "detail": "selected=1"},
            {
                "gate": "source_crosswalk",
                "status": "blocked",
                "detail": "exact sample join pending",
            },
        ],
    )
    _json(
        tmp_path / "contracts/atlas_data_contract_v1.json",
        {
            "schema_version": "1.0.0",
            "contract_id": "toy-atlas-contract",
            "authorities": {
                "current_release_pointer": "configs/atlas_current_release.json",
                "lane_registry": "configs/methanet_atlas_lanes.tsv",
                "tracked_release_ledger": "docs/releases/ledger.json",
                "warehouse_table_manifest": "cohort_table_manifest.tsv",
                "warehouse_validation_gates": "validation_gates.tsv",
            },
            "identity": {
                "canonical_molecular_key": ["lane_id", "proteome_id"],
                "pipeline_warehouse_key": ["cohort_run_id", "proteome_id"],
            },
            "status_semantics": {"registered_gap": "Retain explicit gaps."},
            "evidence_contracts": {
                "pipeline_normalized": {
                    "lanes": ["lane_a"],
                    "meaning": "Synthetic selected MAGs",
                }
            },
            "tables": [
                {
                    "logical_table": "freeze_manifest",
                    "evidence_contract": "release",
                    "grain": "registered unit",
                    "primary_key": ["lane_id", "proteome_id"],
                    "key_status": "validated_by_release_freeze",
                    "required_columns": [
                        "lane_id",
                        "proteome_id",
                        "functional_status",
                        "release_required",
                        "release_excluded",
                        "claim_scope",
                    ],
                    "status_columns": ["functional_status"],
                    "null_semantics": "Gaps remain explicit.",
                    "claim_scope": "Molecular screening.",
                },
                {
                    "logical_table": "dim_mag",
                    "evidence_contract": "pipeline_normalized",
                    "grain": "selected MAG",
                    "primary_key": ["cohort_run_id", "proteome_id"],
                    "key_status": "enforced_by_consolidator",
                    "required_columns": ["cohort_run_id", "proteome_id"],
                    "status_columns": [],
                    "null_semantics": "Missing is not zero.",
                    "claim_scope": "Molecular screening.",
                },
            ],
            "graph_projection": {
                "node_classes": ["mag_proteome"],
                "relation_classes": ["derived_from"],
                "required_assertion_fields": ["source_record_id"],
            },
        },
    )
    pointer = {
        "schema_version": "1.0.0",
        "release_id": "toy-release",
        "snapshot_date": "2026-08-10",
        "release_scope": "controlled_diligence_noindex",
        "lane_registry": "configs/methanet_atlas_lanes.tsv",
        "lane_registry_sha256": _sha(registry),
        "freeze_manifest": "results/freeze/freeze_manifest.tsv",
        "freeze_manifest_sha256": _sha(freeze),
        "freeze_decision": "results/freeze/freeze_decision.json",
        "freeze_decision_sha256": _sha(decision),
        "release_ledger": "results/freeze/release_ledger.json",
        "release_ledger_sha256": _sha(tmp_path / "docs/releases/ledger.json"),
        "tracked_ledger_snapshot": "docs/releases/ledger.json",
        "source_report_bundle": "results/report",
        "reconciled_public_report_bundle": "results/public-report",
        "site_source": "web/site",
    }
    _json(tmp_path / "configs/atlas_current_release.json", pointer)
    return tmp_path


def _repin_freeze(root: Path) -> None:
    """Keep hashes current so mutation tests isolate semantic failures."""
    pointer_path = root / "configs/atlas_current_release.json"
    pointer = json.loads(pointer_path.read_text())
    freeze_sha = _sha(root / pointer["freeze_manifest"])
    pointer["freeze_manifest_sha256"] = freeze_sha
    ledger_path = root / pointer["tracked_ledger_snapshot"]
    ledger = json.loads(ledger_path.read_text())
    ledger["freeze_manifest_sha256"] = freeze_sha
    _json(ledger_path, ledger)
    _json(root / pointer["release_ledger"], ledger)
    pointer["release_ledger_sha256"] = _sha(ledger_path)
    _json(pointer_path, pointer)


def test_contract_only_passes_without_ignored_artifacts(tmp_path: Path) -> None:
    root = _fresh_clone(tmp_path)
    result = VALIDATOR.validate(root)
    assert result["status"] == "pass"
    assert result["registered_lanes"] == [
        "futian_mangrove_2026_qi",
        "msm_china_2025",
        "mucc_v1_owc_wetland",
        "poc_core",
    ]
    assert "warehouse_summaries" not in result


def test_contract_only_rejects_registry_drift_and_unsafe_pointer(
    tmp_path: Path,
) -> None:
    root = _fresh_clone(tmp_path)
    registry = root / "configs/methanet_atlas_lanes.tsv"
    registry.write_text(registry.read_text() + "\n", encoding="utf-8")
    pointer = root / "configs/atlas_current_release.json"
    value = json.loads(pointer.read_text())
    value["site_source"] = "../outside"
    _json(pointer, value)
    result = VALIDATOR.validate(root)
    assert result["status"] == "fail"
    assert any("lane registry: SHA-256 mismatch" in error for error in result["errors"])
    assert any("unsafe path" in error for error in result["errors"])


def test_contract_only_rejects_duplicate_lane_and_table(tmp_path: Path) -> None:
    root = _fresh_clone(tmp_path)
    registry = root / "configs/methanet_atlas_lanes.tsv"
    lines = registry.read_text().splitlines()
    registry.write_text("\n".join([*lines, lines[1]]) + "\n")
    pointer_path = root / "configs/atlas_current_release.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["lane_registry_sha256"] = _sha(registry)
    _json(pointer_path, pointer)
    contract_path = root / "contracts/atlas_data_contract_v1.json"
    contract = json.loads(contract_path.read_text())
    contract["tables"].append(contract["tables"][0])
    _json(contract_path, contract)
    result = VALIDATOR.validate(root)
    assert result["status"] == "fail"
    assert any("duplicate lane_id" in error for error in result["errors"])
    assert any("duplicate table" in error for error in result["errors"])


def test_contract_only_rejects_reconciled_but_wrong_arithmetic(tmp_path: Path) -> None:
    root = _fresh_clone(tmp_path)
    ledger_path = root / "docs/releases/atlas_20260810_release_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["registered_units"] += 1
    _json(ledger_path, ledger)
    pointer_path = root / "configs/atlas_current_release.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["release_ledger_sha256"] = _sha(ledger_path)
    _json(pointer_path, pointer)
    result = VALIDATOR.validate(root)
    assert result["status"] == "fail"
    assert any(
        "ledger.registered_units: lane sum mismatch" in error
        for error in result["errors"]
    )


def test_contract_only_rejects_unrequired_primary_key(tmp_path: Path) -> None:
    root = _fresh_clone(tmp_path)
    contract_path = root / "contracts/atlas_data_contract_v1.json"
    contract = json.loads(contract_path.read_text())
    table = next(
        item
        for item in contract["tables"]
        if item["logical_table"] == "freeze_manifest"
    )
    table["required_columns"].remove("proteome_id")
    _json(contract_path, contract)
    result = VALIDATOR.validate(root)
    assert result["status"] == "fail"
    assert any("primary key not required" in error for error in result["errors"])


def test_local_mode_surfaces_nonpass_gates_without_parquet(tmp_path: Path) -> None:
    root = _toy_repo(tmp_path)
    result = VALIDATOR.validate(root, require_local=True)
    assert result["status"] == "pass_with_warnings"
    assert result["freeze_rows"] == 2
    assert result["warehouse_summaries"]["lane_a"]["gate_statuses"] == {
        "blocked": 1,
        "pass": 1,
    }
    assert (
        result["warehouse_summaries"]["lane_a"]["nonpass_gates"][0]["gate"]
        == "source_crosswalk"
    )


def test_local_mode_rejects_duplicate_identity_even_with_fresh_hash(
    tmp_path: Path,
) -> None:
    root = _toy_repo(tmp_path)
    freeze = root / "results/freeze/freeze_manifest.tsv"
    with freeze.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[1]["proteome_id"] = rows[0]["proteome_id"]
    _tsv(freeze, rows)
    _repin_freeze(root)
    result = VALIDATOR.validate(root, require_local=True)
    assert result["status"] == "fail"
    assert any("duplicate lane_id/proteome_id" in error for error in result["errors"])


def test_cli_writes_failed_receipt_and_exits_nonzero(tmp_path: Path) -> None:
    root = _toy_repo(tmp_path)
    pointer_path = root / "configs/atlas_current_release.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["site_source"] = "../../outside"
    _json(pointer_path, pointer)
    output = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(root),
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "fail"
    assert receipt["error_count"] > 0
