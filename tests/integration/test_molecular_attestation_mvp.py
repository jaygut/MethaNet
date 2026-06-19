from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_molecular_attestation_mvp_snapshot_builds(tmp_path: Path) -> None:
    output_dir = tmp_path / "attestation_snapshot"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/attestation/build_molecular_attestation_mvp.py"),
        "--repo-root",
        str(REPO_ROOT),
        "--snapshot-id",
        "pytest_attestation_mvp",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    validation = pd.read_csv(output_dir / "validation_gates.tsv", sep="\t")
    assert not validation[validation["status"].eq("fail")].to_dict("records")

    nodes = pd.read_parquet(output_dir / "graph_nodes.parquet")
    edges = pd.read_parquet(output_dir / "graph_edges.parquet")
    evidence = pd.read_parquet(output_dir / "evidence_atom.parquet")

    mag_nodes = nodes[nodes["node_type"].eq("MAG")]
    assert len(mag_nodes) == 662
    assert edges["rel_type"].isin(["HAS_EVIDENCE"]).any()
    assert evidence["source_artifact_id"].astype(str).str.startswith("artifact:").all()

    query_summary = pd.read_csv(output_dir / "query_results_summary.tsv", sep="\t")
    query_statuses = set(query_summary["status"])
    assert query_statuses == {"ok"} or query_statuses <= {"not_run_no_kuzu"}
    assert query_summary["query_name"].nunique() >= 12

    audit_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/attestation/audit_molecular_attestation_mvp.py"),
        "--snapshot-dir",
        str(output_dir),
    ]
    subprocess.run(audit_cmd, check=True, cwd=REPO_ROOT)

    static_audit = pd.read_csv(output_dir / "expert_static_audit_gates.tsv", sep="\t")
    assert not static_audit[static_audit["status"].eq("fail")].to_dict("records")

    stress = pd.read_csv(
        output_dir / "expert_audit_query_results_summary.tsv",
        sep="\t",
    )
    stress_statuses = set(stress["status"])
    assert stress_statuses == {"pass"} or stress_statuses <= {"skip"}
    assert {"simple", "complex"} <= set(stress["complexity"])
    assert (output_dir / "EXPERT_AUDIT_REPORT.md").exists()
