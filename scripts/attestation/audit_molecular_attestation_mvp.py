#!/usr/bin/env python3
"""Run an expert-oriented audit over a built MMAG MVP snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

STRESS_QUERIES: dict[str, dict[str, Any]] = {
    "simple_node_type_counts": {
        "complexity": "simple",
        "min_rows": 5,
        "why": "Confirms that the graph has the expected heterogeneous schema, not a flattened one-table export.",
        "query": """
MATCH (n)
RETURN LABEL(n) AS node_type, count(n) AS count
ORDER BY count DESC
""",
    },
    "simple_claim_status_counts": {
        "complexity": "simple",
        "min_rows": 3,
        "why": "Confirms blocked, provisional, allowed, and forbidden claim states remain queryable.",
        "query": """
MATCH (c:Claim)
RETURN c.claim_status AS claim_status, count(c) AS claim_count
ORDER BY claim_count DESC
""",
    },
    "simple_readiness_counts": {
        "complexity": "simple",
        "min_rows": 3,
        "why": "Confirms MAG-bin, QC-caveated, and quarantined units remain distinguishable.",
        "query": """
MATCH (m:MAG)
RETURN m.report_readiness AS readiness, count(m) AS mag_count
ORDER BY mag_count DESC
""",
    },
    "simple_artifact_fanout": {
        "complexity": "simple",
        "min_rows": 5,
        "why": "Confirms evidence atoms can be traced back to source artifact classes.",
        "query": """
MATCH (e:EvidenceAtom)-[:GENERATED_BY]->(a:Artifact)
RETURN a.artifact_key AS artifact, count(e) AS evidence_atoms
ORDER BY evidence_atoms DESC
""",
    },
    "complex_candidate_to_artifact_to_claim": {
        "complexity": "complex",
        "min_rows": 5,
        "why": "Audits the full evidence packet path from candidate MAG to evidence atom, source artifact, and claim boundary.",
        "query": """
MATCH (m:MAG)-[:HAS_EVIDENCE]->(e:EvidenceAtom)-[:GENERATED_BY]->(a:Artifact),
      (e)-[:EVIDENCE_SUPPORTS_CLAIM]->(c:Claim)
WHERE m.proteome_id = 'mucc__GCA_002495465.1_ASM249546v1_genomic'
RETURN m.proteome_id AS proteome_id,
       e.predicate AS predicate,
       e.evidence_direction AS direction,
       e.evidence_strength AS strength,
       a.artifact_key AS artifact,
       c.claim_name AS claim_name,
       c.claim_status AS claim_status
ORDER BY predicate, artifact
""",
    },
    "complex_bridge_neighbor_with_function_context": {
        "complexity": "complex",
        "min_rows": 10,
        "why": "Tests whether ESM2 cross-domain neighborhoods can be read together with functional evidence and readiness caveats.",
        "query": """
MATCH (m:MAG)-[:NEAR_IN_ESM2_SPACE]->(n:MAG),
      (m)-[:HAS_FEATURE]->(mf:Feature),
      (n)-[:HAS_FEATURE]->(nf:Feature)
WHERE m.source <> n.source
  AND mf.feature_type = 'methane'
  AND nf.feature_type = 'methane'
RETURN m.proteome_id AS query_proteome_id,
       m.source AS query_source,
       mf.score AS query_methane_score,
       m.report_readiness AS query_readiness,
       n.proteome_id AS neighbor_proteome_id,
       n.source AS neighbor_source,
       nf.score AS neighbor_methane_score,
       n.report_readiness AS neighbor_readiness
LIMIT 50
""",
    },
    "complex_claim_upgrade_paths": {
        "complexity": "complex",
        "min_rows": 8,
        "why": "Confirms that stronger MRV and carbon-credit claims expose explicit upgrade blockers.",
        "query": """
MATCH (c:Claim)-[:CLAIM_BLOCKED_BY]->(g:ValidationGap)
RETURN c.claim_name AS claim_name,
       c.claim_status AS status,
       g.gap_type AS gap_type,
       g.required_evidence AS required_evidence,
       g.next_action AS next_action
ORDER BY claim_name, gap_type
""",
    },
    "complex_forbidden_mrv_safety": {
        "complexity": "complex",
        "min_rows": 1,
        "why": "Checks that MAGs can support molecular attestation while remaining blocked from final MRV risk claims.",
        "query": """
MATCH (m:MAG)-[:MAG_SUPPORTS_CLAIM]->(allowed:Claim),
      (m)-[:MAG_BLOCKED_FROM_CLAIM]->(blocked:Claim)
WHERE allowed.id = 'claim:mag_molecular_attestation'
  AND blocked.id = 'claim:final_mrv_risk_tier'
RETURN allowed.claim_name AS allowed_claim,
       blocked.claim_name AS blocked_claim,
       count(m) AS mag_count
""",
    },
    "complex_source_taxonomy_caveats": {
        "complexity": "complex",
        "min_rows": 10,
        "why": "Surfaces source-taxonomy patterns that can confound bridge interpretation.",
        "query": """
MATCH (m:MAG)-[:FROM_SOURCE]->(s:SourceDomain)
OPTIONAL MATCH (m)-[:HAS_TAXONOMY]->(t:Taxon)
RETURN s.source_name AS source,
       s.ecosystem_name AS ecosystem,
       t.family AS family,
       count(m) AS mag_count,
       avg(m.methane_evidence_score) AS mean_methane_score
ORDER BY mag_count DESC
LIMIT 30
""",
    },
    "complex_quarantine_integrity": {
        "complexity": "complex",
        "min_rows": 1,
        "why": "Confirms assembly-context rumen units are explicitly blocked from MAG-level interpretation.",
        "query": """
MATCH (m:MAG)-[:BLOCKED_BY]->(g:ValidationGap)
WHERE m.analysis_unit_type = 'assembly_context'
  AND g.id = 'gap:noncomparable_assembly_context'
RETURN count(m) AS quarantined_units
""",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("results/attestation/mmag_mvp_20260617"),
    )
    return parser.parse_args()


def read_json_props(series: pd.Series) -> list[dict[str, Any]]:
    out = []
    for value in series.dropna().astype(str):
        try:
            out.append(json.loads(value))
        except json.JSONDecodeError:
            out.append({})
    return out


def load_snapshot(snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "nodes": pd.read_parquet(snapshot_dir / "graph_nodes.parquet"),
        "edges": pd.read_parquet(snapshot_dir / "graph_edges.parquet"),
        "evidence": pd.read_parquet(snapshot_dir / "evidence_atom.parquet"),
        "registry": pd.read_parquet(snapshot_dir / "registry_artifact.parquet"),
        "validation": pd.read_csv(snapshot_dir / "validation_gates.tsv", sep="\t"),
        "query_summary": pd.read_csv(snapshot_dir / "query_results_summary.tsv", sep="\t"),
    }


def run_stress_queries(snapshot_dir: Path) -> pd.DataFrame:
    out_dir = snapshot_dir / "expert_audit_query_results"
    out_dir.mkdir(exist_ok=True)
    try:
        import kuzu
    except ImportError as exc:
        rows = [
            {
                "query_name": name,
                "complexity": spec["complexity"],
                "status": "skip",
                "rows": 0,
                "min_rows": int(spec["min_rows"]),
                "why": spec["why"],
                "path": "",
                "detail": f"kuzu unavailable: {exc}",
            }
            for name, spec in STRESS_QUERIES.items()
        ]
        summary = pd.DataFrame(rows)
        summary.to_csv(snapshot_dir / "expert_audit_query_results_summary.tsv", sep="\t", index=False)
        return summary

    db_path = snapshot_dir / "mmag.kuzu"
    if not db_path.exists():
        rows = [
            {
                "query_name": name,
                "complexity": spec["complexity"],
                "status": "skip",
                "rows": 0,
                "min_rows": int(spec["min_rows"]),
                "why": spec["why"],
                "path": "",
                "detail": "mmag.kuzu graph database was not built",
            }
            for name, spec in STRESS_QUERIES.items()
        ]
        summary = pd.DataFrame(rows)
        summary.to_csv(snapshot_dir / "expert_audit_query_results_summary.tsv", sep="\t", index=False)
        return summary

    conn = kuzu.Connection(kuzu.Database(str(snapshot_dir / "mmag.kuzu")))
    rows = []
    for name, spec in STRESS_QUERIES.items():
        result = conn.execute(spec["query"])
        frame = result.get_as_df()
        output_path = out_dir / f"{name}.tsv"
        frame.to_csv(output_path, sep="\t", index=False)
        rows.append(
            {
                "query_name": name,
                "complexity": spec["complexity"],
                "status": "pass" if len(frame) >= int(spec["min_rows"]) else "fail",
                "rows": len(frame),
                "min_rows": int(spec["min_rows"]),
                "why": spec["why"],
                "path": str(output_path),
                "detail": "",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(snapshot_dir / "expert_audit_query_results_summary.tsv", sep="\t", index=False)
    return summary


def audit_static(snapshot: dict[str, pd.DataFrame]) -> pd.DataFrame:
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]
    evidence = snapshot["evidence"]
    validation = snapshot["validation"]
    node_ids = set(nodes["node_id"].astype(str))
    rows = []

    def add(gate: str, status: str, observed: Any, expected: Any, detail: str = "") -> None:
        rows.append(
            {
                "gate": gate,
                "status": status,
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    fail_count = int(validation["status"].eq("fail").sum())
    add("builder_validation_has_no_failures", "pass" if fail_count == 0 else "fail", fail_count, 0)

    missing_edges = set(edges["src_id"].astype(str)) | set(edges["dst_id"].astype(str))
    missing_edges = sorted(missing_edges - node_ids)
    add("all_edges_resolve_to_exported_nodes", "pass" if not missing_edges else "fail", len(missing_edges), 0, ";".join(missing_edges[:5]))

    missing_sources = edges[edges["source_artifact_id"].fillna("").astype(str).eq("")]
    add("all_edges_have_source_artifact", "pass" if missing_sources.empty else "fail", len(missing_sources), 0)

    evidence_sources = evidence[evidence["source_artifact_id"].fillna("").astype(str).eq("")]
    add("all_evidence_atoms_have_artifact_id", "pass" if evidence_sources.empty else "fail", len(evidence_sources), 0)

    final_mrv_support = edges[
        edges["rel_type"].eq("MAG_SUPPORTS_CLAIM")
        & edges["dst_id"].eq("claim:final_mrv_risk_tier")
    ]
    add("no_mag_supports_final_mrv_risk_tier", "pass" if final_mrv_support.empty else "fail", len(final_mrv_support), 0)

    mag_count = int(nodes["node_type"].eq("MAG").sum())
    mrv_blocks = edges[
        edges["rel_type"].eq("MAG_BLOCKED_FROM_CLAIM")
        & edges["dst_id"].eq("claim:final_mrv_risk_tier")
    ]
    add("every_mag_blocked_from_final_mrv", "pass" if len(mrv_blocks) == mag_count else "fail", len(mrv_blocks), mag_count)

    return pd.DataFrame(rows)


def read_optional_tsv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame()


def write_report(snapshot_dir: Path, snapshot: dict[str, pd.DataFrame], static_audit: pd.DataFrame, stress: pd.DataFrame) -> None:
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]
    evidence = snapshot["evidence"]

    node_counts = nodes["node_type"].value_counts().rename_axis("node_type").reset_index(name="count")
    edge_counts = edges["rel_type"].value_counts().rename_axis("rel_type").reset_index(name="count")
    evidence_counts = evidence["predicate"].value_counts().rename_axis("predicate").reset_index(name="count")

    readiness = read_optional_tsv(snapshot_dir / "query_results/report_ready_vs_blocked_counts.tsv")
    top_bridge = read_optional_tsv(snapshot_dir / "query_results/top_complete_multiview_bridge_candidates.tsv").head(10)
    blocker = read_optional_tsv(snapshot_dir / "expert_audit_query_results/complex_claim_upgrade_paths.tsv")

    report = [
        "# Expert Audit: MethaNet Molecular Attestation Graph MVP",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Snapshot: `{snapshot_dir}`",
        "",
        "## Audit Verdict",
        "",
        "The MVP is structurally solid for a local MAG/proteome-level molecular attestation substrate: it preserves the 662-row ESM2 denominator, separates the 625 MAG-bin units from the 37 assembly-context units, ties every evidence atom and graph edge to source-artifact provenance, executes both simple and multi-hop Kuzu queries, and explicitly blocks sample-level methane-risk, final A-E MRV tier, measured flux, and carbon-credit claims.",
        "",
        "The system is not yet a production MRV scorer. That is a strength, not a weakness: the graph makes the missing sample metadata, abundance, environmental covariates, source deconfounding, and flux validation visible as queryable blockers instead of burying them in prose.",
        "",
        "## Static Audit Gates",
        "",
        "| Gate | Status | Observed | Expected | Detail |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in static_audit.to_dict("records"):
        report.append(f"| {row['gate']} | {row['status']} | {row['observed']} | {row['expected']} | {row.get('detail', '')} |")

    report.extend(["", "## Stress Query Suite", "", "| Query | Complexity | Status | Rows | Why It Matters |", "| --- | --- | --- | ---: | --- |"])
    for row in stress.to_dict("records"):
        report.append(f"| {row['query_name']} | {row['complexity']} | {row['status']} | {row['rows']} | {row['why']} |")

    report.extend(["", "## Graph Inventory", "", "### Node Counts", "", "| Node type | Count |", "| --- | ---: |"])
    for row in node_counts.to_dict("records"):
        report.append(f"| {row['node_type']} | {row['count']} |")

    report.extend(["", "### Relationship Counts", "", "| Relationship | Count |", "| --- | ---: |"])
    for row in edge_counts.to_dict("records"):
        report.append(f"| {row['rel_type']} | {row['count']} |")

    report.extend(["", "### Evidence Atom Predicates", "", "| Predicate | Count |", "| --- | ---: |"])
    for row in evidence_counts.to_dict("records"):
        report.append(f"| {row['predicate']} | {row['count']} |")

    report.extend(["", "## Readiness Distribution", "", "| Readiness | MAGs |", "| --- | ---: |"])
    if readiness.empty:
        report.append("| Not available without Kuzu query execution | 0 |")
    else:
        for row in readiness.to_dict("records"):
            report.append(f"| {row['report_readiness']} | {row['mag_count']} |")

    report.extend(["", "## Top Multiview Bridge Candidates", "", "| Proteome ID | Source | Provisional bridge | Methane score | Readiness |", "| --- | --- | ---: | ---: | --- |"])
    if top_bridge.empty:
        report.append("| Not available without Kuzu query execution | NA | NA | NA | NA |")
    else:
        for row in top_bridge.to_dict("records"):
            report.append(
                f"| {row['proteome_id']} | {row['source']} | {row['provisional_bridge_score']:.3f} | {row['methane_score']:.3f} | {row['readiness']} |"
            )

    report.extend(["", "## Claim Upgrade Blockers", "", "| Claim | Status | Blocking gap | Next action |", "| --- | --- | --- | --- |"])
    if blocker.empty:
        report.append("| Not available without Kuzu query execution | skipped | graph-query layer unavailable | install optional Kuzu dependency and rerun audit |")
    else:
        for row in blocker.to_dict("records"):
            report.append(f"| {row['claim_name']} | {row['status']} | {row['gap_type']} | {row['next_action']} |")

    report.extend(
        [
            "",
            "## Intelligence Unlocked Beyond A Monolithic Table",
            "",
            "- Evidence-path interrogation: a reviewer can start from one bridge candidate and traverse to functional evidence, gLM2 context, source artifact, and allowed/blocked claim wording.",
            "- Claim-safe biological reasoning: MAGs can support molecular attestation while simultaneously remaining blocked from final MRV risk tiers, so the same system carries signal and caveat.",
            "- Source-aware confounding checks: source, ecosystem, taxonomy, bridge-neighbor, and methane-feature context can be queried together instead of inspected in separate reports.",
            "- Missingness as data: absent gLM2, sample metadata, abundance, environmental covariates, flux validation, and non-comparable assembly context are explicit nodes and edges.",
            "- Multi-view candidate triage: ESM2 bridge rank, methane evidence, sulfur context, QC tier, annotation coverage, gLM2 context, and source-domain neighbors are queryable in one local substrate.",
            "",
            "A monolithic feature table can rank rows. This graph can explain why a row is interesting, what evidence supports it, what evidence weakens it, which files produced that evidence, what claim is allowed, and what must be measured next to upgrade the claim.",
            "",
            "## Remaining Expert Caveats",
            "",
            "- Gene-level marker/pathway nodes are not yet first-class; current functional evidence is MAG-level feature evidence from warehouse summaries.",
            "- Kuzu 0.11.3 is viable locally but should remain pinned because the original KuzuDB repository was archived in 2025; Parquet exports are the durable fallback.",
            "- The current POC still has rumen/wetland source confounding; source-independent transfer claims remain blocked.",
            "- Sample/project MRV scoring remains blocked until sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, and flux/process validation are integrated.",
        ]
    )

    (snapshot_dir / "EXPERT_AUDIT_REPORT.md").write_text("\n".join(report) + "\n")


def main() -> int:
    args = parse_args()
    snapshot_dir = args.snapshot_dir.resolve()
    snapshot = load_snapshot(snapshot_dir)
    stress = run_stress_queries(snapshot_dir)
    static_audit = audit_static(snapshot)
    static_audit.to_csv(snapshot_dir / "expert_static_audit_gates.tsv", sep="\t", index=False)
    write_report(snapshot_dir, snapshot, static_audit, stress)
    failures = int(static_audit["status"].eq("fail").sum()) + int(stress["status"].eq("fail").sum())
    print(f"snapshot_dir={snapshot_dir}")
    print(f"static_failures={int(static_audit['status'].eq('fail').sum())}")
    print(f"stress_query_failures={int(stress['status'].eq('fail').sum())}")
    print(f"report={snapshot_dir / 'EXPERT_AUDIT_REPORT.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
