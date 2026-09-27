"""Synthetic release contract tests. No real warehouse assets required in CI."""
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdflib.namespace import RDF

from mvo.ingest import build_atlas, MAPPING
from mvo.model import M, digest, canonical
from mvo.validation import validate_graph


def tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def repin(repo):
    path = repo / "configs/atlas_current_release.json"
    pointer = json.loads(path.read_text())
    for key in ("lane_registry", "freeze_manifest", "freeze_decision", "release_ledger"):
        pointer[key + "_sha256"] = digest(repo / pointer[key])
    path.write_text(json.dumps(pointer))


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "MethaNet"
    repo.mkdir()
    data = []
    for index in range(2):
        row = {k: "source_value" for k in MAPPING["freeze_literal_fields"]}
        row.update({k: "true" for k in MAPPING["freeze_boolean_fields"]})
        row.update(lane_id="test", proteome_id=f"p{index}", mag_id=f"m{index}",
                   has_esm2="true" if index == 0 else "false", tri_view_ready="true" if index == 0 else "false",
                   release_excluded="false" if index == 0 else "true", mechanism_comparable="false",
                   release_exclusion_reason="missing" if index else "")
        data.append(row)
    tsv(repo / "freeze.tsv", data)
    tsv(repo / "lanes.tsv", [dict(lane_id="test", denominator_label="synthetic two units", functional_warehouse_dir="warehouse")])
    (repo / "decision.json").write_text('{}')
    ledger = dict(registered_units=2, esm2_units=1, glm2_units=2, functional_payload_units=2,
                  tri_view_ready_units=1, explicit_non_runnable_gaps=1, mechanism_comparable_units=0,
                  allowed_public_wording="Synthetic evidence only", lanes=[dict(lane_id="test", registry_denominator_units=2)])
    (repo / "ledger.json").write_text(json.dumps(ledger))
    warehouse = repo / "warehouse"
    warehouse.mkdir()
    attempt = warehouse / "attempts.parquet"
    pq.write_table(pa.Table.from_pylist([dict(cohort_run_id="c1", run_id="run1", proteome_id="p0", run_status="failed"),
                                       dict(cohort_run_id="c1", run_id="run2", proteome_id="out_of_release", run_status="partial")]), attempt)
    tsv(warehouse / "cohort_table_manifest.tsv", [dict(table="fact_run_status", path="warehouse/attempts.parquet", rows=2, columns=4, bytes=attempt.stat().st_size)])
    pointer = dict(release_id="synthetic-release", snapshot_date="2026-09-01", lane_registry="lanes.tsv",
                   freeze_manifest="freeze.tsv", freeze_decision="decision.json", release_ledger="ledger.json")
    (repo / "configs").mkdir()
    (repo / "configs/atlas_current_release.json").write_text(json.dumps(pointer))
    repin(repo)
    return repo


def test_import_denominator_failures_and_quarantine(repo):
    g, summary, sources = build_atlas(repo, "2026-09-26T00:00:00Z")
    assert summary["registered_records"] == 2
    assert summary["curation_attempts"] == 1
    assert len(summary["quarantined_attempts"]) == 1
    attempt = next(g.subjects(RDF.type, M.CurationAttempt))
    assert str(g.value(attempt, M.status)) == "failed"
    assert validate_graph(g)[0]
    assert all(s["sha256"] for s in sources)


def test_deterministic_import(repo):
    first = build_atlas(repo, "2026-09-26T00:00:00Z")
    second = build_atlas(repo, "2026-09-26T00:00:00Z")
    assert canonical(first[0]) == canonical(second[0])
    assert first[1:] == second[1:]


def test_tampered_pin_rejected(repo):
    (repo / "freeze.tsv").write_text((repo / "freeze.tsv").read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_atlas(repo, "2026-09-26T00:00:00Z")


def test_duplicate_registered_key_rejected(repo):
    text = (repo / "freeze.tsv").read_text()
    (repo / "freeze.tsv").write_text(text + text.splitlines()[1] + "\n")
    repin(repo)
    with pytest.raises(ValueError, match="Duplicate"):
        build_atlas(repo, "2026-09-26T00:00:00Z")


def test_ledger_drift_rejected(repo):
    ledger = json.loads((repo / "ledger.json").read_text())
    ledger["registered_units"] = 3
    (repo / "ledger.json").write_text(json.dumps(ledger))
    repin(repo)
    with pytest.raises(ValueError, match="denominator mismatch"):
        build_atlas(repo, "2026-09-26T00:00:00Z")


def test_warehouse_row_count_drift_rejected(repo):
    path = repo / "warehouse/cohort_table_manifest.tsv"
    text = path.read_text()
    path.write_text(text.replace("\t2\t4\t", "\t3\t4\t"))
    with pytest.raises(ValueError, match="row/byte mismatch"):
        build_atlas(repo, "2026-09-26T00:00:00Z")
