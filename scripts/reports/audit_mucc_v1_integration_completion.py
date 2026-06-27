#!/usr/bin/env python3
"""Audit MUCC v1 integration completion against the active MethaNet goal."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def exists_nonempty(repo_root: Path, rel: str) -> bool:
    path = repo_root / rel
    return path.is_file() and path.stat().st_size > 0


def count_rows(repo_root: Path, rel: str) -> int:
    path = repo_root / rel
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def count_truthy(repo_root: Path, rel: str, column: str) -> int:
    path = repo_root / rel
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    return sum(1 for row in read_tsv(path) if str(row.get(column, "")).lower() in {"true", "1", "yes", "y"})


def first_existing_rel(repo_root: Path, rel_paths: list[str]) -> str:
    for rel in rel_paths:
        path = repo_root / rel
        if path.is_file() and path.stat().st_size > 0:
            return rel
    return rel_paths[0]


def all_ready(repo_root: Path, rel_dirs: list[str]) -> bool:
    return all(
        exists_nonempty(repo_root, f"{rel}/embedding_metadata.tsv")
        and exists_nonempty(repo_root, f"{rel}/genome_embeddings.npz")
        for rel in rel_dirs
    )


def count_nonempty_files(path: Path, pattern: str) -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file() and item.stat().st_size > 0)


def count_nonempty_triplets(base: Path) -> int:
    faa_dir = base / "prodigal/proteomes_faa"
    gff_dir = base / "prodigal/genes_gff"
    ffn_dir = base / "prodigal/genes_ffn"
    stems = {
        item.stem
        for item in faa_dir.glob("*.faa")
        if item.is_file() and item.stat().st_size > 0
    }
    return sum(
        1
        for stem in stems
        if (gff_dir / f"{stem}.gff").is_file()
        and (gff_dir / f"{stem}.gff").stat().st_size > 0
        and (ffn_dir / f"{stem}.ffn").is_file()
        and (ffn_dir / f"{stem}.ffn").stat().st_size > 0
    )


def status_row(
    requirement: str,
    status: str,
    evidence: str,
    blocker: str = "",
    next_action: str = "",
) -> dict[str, str]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "blocker": blocker,
        "next_action": next_action,
    }


def build(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    esm2_dirs = [
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard001/artifacts",
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard002/artifacts",
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard003/artifacts",
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard004/artifacts",
    ]
    prodigal_faa = count_nonempty_files(repo_root / BASE / "prodigal/proteomes_faa", "*.faa")
    prodigal_gff = count_nonempty_files(repo_root / BASE / "prodigal/genes_gff", "*.gff")
    prodigal_ffn = count_nonempty_files(repo_root / BASE / "prodigal/genes_ffn", "*.ffn")
    prodigal_triplets = count_nonempty_triplets(repo_root / BASE)
    glm2_manifest_rel = first_existing_rel(
        repo_root,
        [
            str(BASE / "manifests/mucc_v1_glm2_ready_manifest.tsv"),
            str(BASE / "manifests/mucc_v1_glm2_ready_manifest.partial_file_scan.tsv"),
        ],
    )
    glm2_gap_rel = first_existing_rel(
        repo_root,
        [
            str(BASE / "manifests/mucc_v1_glm2_ready_gap_register.tsv"),
            str(BASE / "manifests/mucc_v1_glm2_ready_gap_register.partial_file_scan.tsv"),
        ],
    )
    glm2_ready = count_truthy(repo_root, glm2_manifest_rel, "glm2_include")
    glm2_blocked = count_rows(repo_root, glm2_gap_rel)
    mrv_readiness_rows = count_rows(
        repo_root,
        str(BASE / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv"),
    )
    mrv_readiness_candidate_rows = count_rows(
        repo_root,
        str(BASE / "candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv"),
    )
    warehouse_dim_mag_rows = 0
    warehouse_manifest = repo_root / BASE / "cohort_warehouse/cohort_table_manifest.tsv"
    if warehouse_manifest.is_file():
        for row in read_tsv(warehouse_manifest):
            if row.get("table") == "dim_mag":
                warehouse_dim_mag_rows = int(float(row.get("rows") or 0))
                break
    stop_condition_rel = str(BASE / "reports/mucc_v1_stop_condition_external_compute_blockers_20260626.tsv")
    stop_condition_rows = count_rows(repo_root, stop_condition_rel)
    esm2_complete = all_ready(repo_root, esm2_dirs)
    legacy_neighbor_rows = count_rows(
        repo_root,
        str(
            BASE
            / "bridge_reanchoring/legacy_poc_mucc_neighbor_validation/wetland_reference_neighbor_summary.tsv"
        ),
    )
    final_neighbor_rows = count_rows(
        repo_root,
        str(BASE / "bridge_reanchoring/wetland_reference_neighbor_summary.tsv"),
    )

    rows = [
        status_row(
            "source provenance ledger",
            "complete",
            f"{BASE / 'source_audit/source_provenance_ledger.tsv'} rows={count_rows(repo_root, str(BASE / 'source_audit/source_provenance_ledger.tsv'))}",
        ),
        status_row(
            "lane manifest",
            "complete",
            f"{BASE / 'manifests/mucc_v1_source_lane_manifest.tsv'} rows={count_rows(repo_root, str(BASE / 'manifests/mucc_v1_source_lane_manifest.tsv'))}; source-scaffold warehouse dim_mag rows={warehouse_dim_mag_rows}",
        ),
        status_row(
            "MAG catalog",
            "complete_with_denominator_gap",
            f"{BASE / 'manifests/mucc_v1_mag_catalog_full.tsv'} rows={count_rows(repo_root, str(BASE / 'manifests/mucc_v1_mag_catalog_full.tsv'))}",
            "local MAGs.zip has 2,508 FASTA rows versus published/deposit 2,502 HQ/MQ denominator",
            "reconcile six-entry denominator mismatch before stronger denominator claims",
        ),
        status_row(
            "expression support tables",
            "complete",
            f"{BASE / 'expression/feature_mucc_v1_expression_mag_summary.tsv'} rows={count_rows(repo_root, str(BASE / 'expression/feature_mucc_v1_expression_mag_summary.tsv'))}; {BASE / 'expression/feature_mucc_v1_gene_expression_mag_summary.tsv'} rows={count_rows(repo_root, str(BASE / 'expression/feature_mucc_v1_gene_expression_mag_summary.tsv'))}",
        ),
        status_row(
            "functional feature tables",
            "partial",
            f"source annotation/status tables exist; source-scaffold warehouse dim_mag rows={warehouse_dim_mag_rows}; MRV readiness scaffold rows={mrv_readiness_rows}; Prodigal non-empty files FAA/GFF/FFN={prodigal_faa}/{prodigal_gff}/{prodigal_ffn}; non-empty triplets={prodigal_triplets}; gLM2 manifest={glm2_manifest_rel}; gLM2-ready rows={glm2_ready} blocked={glm2_blocked}",
            "curated MethaNet functional runs and production gLM2 context are not complete",
            "run production gLM2 and curated functional feature generation, then promote final contextual feature tables",
        ),
        status_row(
            "sample/depth readiness scaffold",
            "complete_scaffold_only",
            f"{BASE / 'environmental_metadata/feature_sample_risk_readiness_scaffold.tsv'} rows={count_rows(repo_root, str(BASE / 'environmental_metadata/feature_sample_risk_readiness_scaffold.tsv'))}",
            "environmental/sample/depth metadata remain scaffolded, not resolved sample-level MRV evidence",
            "join source environmental, depth, abundance/coverage, geochemistry, and flux/process context with explicit tiers",
        ),
        status_row(
            "wetland-neighbor bridge tables",
            "incomplete",
            f"legacy POC validation summary rows={legacy_neighbor_rows}; final MUCC neighbor summary rows={final_neighbor_rows}; MUCC ESM2 shards complete={esm2_complete}",
            "MUCC ESM2 shard artifacts are not complete, so final MUCC v1 neighbor table cannot be built",
            "after shard artifacts exist, consolidate MUCC ESM2 outputs and run wetland neighbor builder",
        ),
        status_row(
            "candidate cards",
            "complete_scaffold_only",
            f"{BASE / 'candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv'} rows={count_rows(repo_root, str(BASE / 'candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv'))}; {BASE / 'candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv'} rows={mrv_readiness_candidate_rows}",
            "strategic review cards are not final bridge/mechanism cards",
            "promote only after ESM2 neighbor, curated functional, and gLM2 gates pass",
        ),
        status_row(
            "validation gap register",
            "complete",
            f"{BASE / 'reports/validation_gap_register.tsv'} rows={count_rows(repo_root, str(BASE / 'reports/validation_gap_register.tsv'))}",
        ),
        status_row(
            "integration report",
            "complete_current_snapshot",
            f"{BASE / 'reports/INTEGRATION_REPORT.md'} and ai_docs/mucc_v1_wetland_mbag_integration/integration_report_20260626.md",
            "snapshot remains conditional while ESM2/gLM2/neighbor gates are pending",
            "refresh after each compute gate changes state",
        ),
        status_row(
            "claim boundaries",
            "complete",
            f"{BASE / 'reports/claim_boundary_matrix.tsv'} rows={count_rows(repo_root, str(BASE / 'reports/claim_boundary_matrix.tsv'))}",
        ),
    ]

    fields = ["requirement", "status", "evidence", "blocker", "next_action"]
    write_tsv(args.output_tsv, rows, fields)
    fully_complete_statuses = {"complete"}
    partial_statuses = {
        "partial",
        "incomplete",
        "complete_scaffold_only",
        "complete_current_snapshot",
        "complete_with_denominator_gap",
    }
    summary = {
        "output_tsv": str(args.output_tsv),
        "requirements": len(rows),
        "fully_complete": sum(1 for row in rows if row["status"] in fully_complete_statuses),
        "partial_or_incomplete": sum(1 for row in rows if row["status"] in partial_statuses),
        "mucc_esm2_shards_complete": esm2_complete,
        "prodigal_triplets": {
            "faa": prodigal_faa,
            "gff": prodigal_gff,
            "ffn": prodigal_ffn,
            "complete_nonempty_triplets": prodigal_triplets,
        },
        "glm2_ready_manifest": glm2_manifest_rel,
        "glm2_ready_rows": glm2_ready,
        "glm2_blocked_rows": glm2_blocked,
        "mrv_readiness_rows": mrv_readiness_rows,
        "mrv_readiness_candidate_rows": mrv_readiness_candidate_rows,
        "warehouse_dim_mag_rows": warehouse_dim_mag_rows,
        "stop_condition_blocker_rows": stop_condition_rows,
        "stop_condition_blocker_ledger": stop_condition_rel,
        "final_goal_complete": False,
        "claim_boundary": (
            "Audit statuses do not authorize final MRV scores, A-E tiers, measured-flux claims, "
            "crediting claims, or source-independent transfer claims."
        ),
    }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=BASE / "reports/mucc_v1_integration_completion_audit.tsv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE / "reports/mucc_v1_integration_completion_audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
