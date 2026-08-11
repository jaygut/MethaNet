from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "web/emergentbiome-methanet/tools/validate_release_parity.py"
SPEC = importlib.util.spec_from_file_location("validate_release_parity", SCRIPT)
assert SPEC and SPEC.loader
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


def ledger() -> dict:
    return {
        "schema_version": "1.0.0",
        "snapshot_date": "2026-08-10",
        "freeze_manifest_sha256": "a" * 64,
        "release_state": "ready",
        "indexing_decision": "noindex_controlled_diligence",
        "allowed_public_wording": "molecular screening",
        "forbidden_public_wording": "sample risk",
        "registered_units": 12,
        "esm2_units": 10,
        "glm2_units": 10,
        "functional_payload_units": 10,
        "release_required_units": 10,
        "explicit_non_runnable_gaps": 2,
        "tri_view_ready_units": 9,
        "schema_normalized_units": 5,
        "schema_normalized_tri_view_units": 5,
        "pipeline_normalized_tri_view_units": 0,
        "mechanism_comparable_units": 3,
        "annotation_complete_tri_view_units": 4,
        "source_scaffold_tri_view_units": 2,
        "blocking_units": 1,
        "lanes": [],
    }


def test_atlas_contract_parity_detects_drift() -> None:
    expected = ledger()
    atlas = {
        "meta": {"snapshot": "2026-08-10", "n_points": 10, "excluded_gap_rows": 2},
        "points": [{"fc": code} for code in ([1] * 3 + [2] * 4 + [3] * 2 + [0])],
    }
    audit = PARITY.Audit()
    PARITY.validate_atlas(audit, expected, atlas)
    assert not audit.failed

    atlas["points"][0]["fc"] = 2
    drift = PARITY.Audit()
    PARITY.validate_atlas(drift, expected, atlas)
    assert {row["gate"] for row in drift.failed} >= {
        "atlas.contract.mechanism_comparable",
        "atlas.contract.annotation_complete",
    }


def test_digest_requires_exact_generated_ledger(tmp_path: Path) -> None:
    expected = ledger()
    digest_values = {
        key: expected[key]
        for key in (*PARITY.COUNT_KEYS, *PARITY.REQUIRED_PROVENANCE)
    }
    digest = tmp_path / "DIGEST.md"
    digest.write_text(
        "<!-- METHANET_RELEASE_LEDGER_BEGIN -->\n```json\n"
        + json.dumps(digest_values)
        + "\n```\n<!-- METHANET_RELEASE_LEDGER_END -->\n"
    )
    audit = PARITY.Audit()
    PARITY.validate_digest(audit, expected, digest)
    assert not audit.failed

    digest.write_text("# stale hand-maintained digest\n")
    missing = PARITY.Audit()
    PARITY.validate_digest(missing, expected, digest)
    assert [row["gate"] for row in missing.failed] == ["digest.machine_ledger.present"]


def test_freeze_summary_lane_and_total_parity(tmp_path: Path) -> None:
    expected = ledger()
    expected["lanes"] = [{
        "lane_id": "lane_a",
        "registry_denominator_units": 12,
        "release_required_units": 10,
        "release_excluded_units": 2,
        "esm2_units": 10,
        "glm2_units": 10,
        "functional_payload_units": 10,
        "functional_complete": 9,
        "functional_partial": 1,
        "functional_failed": 0,
        "functional_not_started": 0,
        "functional_non_runnable_gap": 2,
        "release_tri_view_ready_units": 9,
        "schema_normalized_units": 5,
        "schema_normalized_tri_view_units": 5,
        "pipeline_normalized_tri_view_units": 0,
        "canonical_mechanism_tri_view_units": 3,
        "annotation_complete_tri_view_units": 4,
        "source_scaffold_tri_view_units": 2,
    }]
    path = tmp_path / "summary.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=expected["lanes"][0].keys())
        writer.writeheader()
        writer.writerows(expected["lanes"])
    rows = PARITY.load_tsv(path)
    audit = PARITY.Audit()
    PARITY.validate_lane_summary(audit, expected, rows)
    assert not audit.failed
