#!/usr/bin/env python3
"""Build claim-safe MUCC v1 MAG-level MRV readiness features.

The output is a warehouse-facing feature primitive for triage only. It combines
source DRAM-style annotations, processed expression support, source protein
readiness, and Prodigal-derived gLM2 input readiness without assigning final
MRV scores, A-E tiers, measured flux, crediting status, or transfer claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
CLAIM_BOUNDARY = (
    "MAG-level source-scaffold readiness only; no final MRV score, A-E tier, "
    "measured methane flux, carbon-crediting claim, or source-independent "
    "transfer claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    return parser.parse_args()


def read_tsv(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return {
            row[key]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get(key)
        }


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def fmt(value: float) -> str:
    return f"{value:.6g}"


def qc_label(completeness: float, contamination: float) -> str:
    if completeness >= 90.0 and contamination <= 5.0:
        return "source_qc_strong"
    if completeness >= 70.0 and contamination <= 10.0:
        return "source_qc_reviewable"
    if completeness > 0.0:
        return "source_qc_limited"
    return "source_qc_missing"


def expression_label(
    mag_expr: bool,
    gene_expr: bool,
    methane_expr: int,
    sulfur_expr: int,
    methyl_expr: int,
    substrate_expr: int,
) -> str:
    if any(value > 0 for value in [methane_expr, sulfur_expr, methyl_expr, substrate_expr]):
        return "processed_gene_expression_supports_marker_terms"
    if gene_expr:
        return "processed_gene_expression_available"
    if mag_expr:
        return "processed_mag_expression_available"
    return "no_processed_expression_support_mapped"


def readiness_label(
    has_source_features: bool,
    has_expression: bool,
    direct_esm2_ready: bool,
    prodigal_glm2_ready: bool,
) -> str:
    if has_source_features and has_expression and direct_esm2_ready and prodigal_glm2_ready:
        return "mrv_feature_scaffold_ready_pending_embedding_outputs"
    if has_source_features and prodigal_glm2_ready and not direct_esm2_ready:
        return "context_feature_scaffold_ready_missing_direct_source_esm2_protein"
    if has_source_features and direct_esm2_ready:
        return "feature_scaffold_ready_pending_context_or_expression"
    if has_source_features:
        return "source_annotation_scaffold_only"
    return "catalog_only_pending_feature_evidence"


def allowed_wording(label: str) -> str:
    if label == "mrv_feature_scaffold_ready_pending_embedding_outputs":
        return (
            "This MAG has source-derived functional terms, processed expression support, "
            "direct source-protein ESM2 input readiness, and Prodigal-derived gLM2 input "
            "readiness; it is ready for MRV feature review after embedding outputs complete."
        )
    if label == "context_feature_scaffold_ready_missing_direct_source_esm2_protein":
        return (
            "This MAG has source-derived functional terms and Prodigal-derived gLM2 input "
            "readiness, but lacks direct source-protein support for the current ESM2 lane."
        )
    if label == "feature_scaffold_ready_pending_context_or_expression":
        return (
            "This MAG has source-derived functional terms and direct source-protein ESM2 "
            "input readiness, but still needs expression/context evidence before stronger review."
        )
    if label == "source_annotation_scaffold_only":
        return "This MAG has source-derived functional scaffold evidence only."
    return "This MAG is present in the local MUCC catalog but lacks mapped feature evidence."


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir

    catalog = read_tsv(run_dir / "manifests/mucc_v1_mag_catalog_full.tsv", "mag_id")
    gene_ann = read_tsv(
        run_dir / "functional_features/feature_mucc_v1_gene_annotation_mag_summary.tsv",
        "mag_id",
    )
    dram = read_tsv(
        run_dir / "functional_features/feature_mucc_v1_source_dram_mag_summary.tsv",
        "mag_id",
    )
    mag_expr = read_tsv(run_dir / "expression/feature_mucc_v1_expression_mag_summary.tsv", "mag_id")
    gene_expr = read_tsv(
        run_dir / "expression/feature_mucc_v1_gene_expression_mag_summary.tsv",
        "mag_id",
    )
    esm2 = read_tsv(run_dir / "manifests/mucc_v1_esm2_input_manifest.tsv", "mag_id")
    glm2 = read_tsv(run_dir / "manifests/mucc_v1_glm2_ready_manifest.tsv", "mag_id")

    rows: list[dict[str, Any]] = []
    for mag_id in sorted(catalog):
        ann = gene_ann.get(mag_id, {})
        src = dram.get(mag_id, {})
        mex = mag_expr.get(mag_id, {})
        gex = gene_expr.get(mag_id, {})
        esm = esm2.get(mag_id, {})
        gl = glm2.get(mag_id, {})

        methane_terms = safe_int(ann.get("methane_term_rows")) + safe_int(src.get("methane_term_rows"))
        sulfur_terms = safe_int(ann.get("sulfur_term_rows")) + safe_int(src.get("sulfur_term_rows"))
        methyl_terms = safe_int(ann.get("methyl_substrate_term_rows")) + safe_int(
            src.get("methyl_substrate_term_rows")
        )
        substrate_terms = safe_int(ann.get("substrate_term_rows")) + safe_int(
            src.get("substrate_term_rows")
        )
        source_feature_rows = safe_int(ann.get("gene_annotation_rows")) + safe_int(
            src.get("source_dram_rows")
        )
        completeness = safe_float(ann.get("bin_completeness"))
        contamination = safe_float(ann.get("bin_contamination"))
        mag_expr_support = mag_id in mag_expr and safe_float(mex.get("sum_expression")) > 0.0
        gene_expr_support = safe_int(gex.get("gene_expression_rows")) > 0
        direct_esm2_ready = truthy(esm.get("functional_run_include")) and safe_int(
            esm.get("protein_count")
        ) > 0
        prodigal_glm2_ready = truthy(gl.get("glm2_include")) and safe_int(gl.get("protein_count")) > 0
        label = readiness_label(
            source_feature_rows > 0,
            mag_expr_support or gene_expr_support,
            direct_esm2_ready,
            prodigal_glm2_ready,
        )
        marker_breadth = sum(
            value > 0 for value in [methane_terms, sulfur_terms, methyl_terms, substrate_terms]
        )
        expression_marker_breadth = sum(
            safe_int(gex.get(field)) > 0
            for field in [
                "methane_expressed_gene_rows",
                "sulfur_expressed_gene_rows",
                "methyl_substrate_expressed_gene_rows",
                "substrate_expressed_gene_rows",
            ]
        )
        review_priority_score = (
            marker_breadth * 2.0
            + expression_marker_breadth * 2.0
            + min(safe_float(gex.get("expression_sum")) / 10000.0, 5.0)
            + min(completeness / 20.0, 5.0)
            - min(contamination / 5.0, 5.0)
            + (1.0 if direct_esm2_ready else 0.0)
            + (1.0 if prodigal_glm2_ready else 0.0)
        )

        rows.append(
            {
                "lane_id": LANE_ID,
                "proteome_id": catalog[mag_id].get("proteome_id", f"mucc_v1__{mag_id}"),
                "mag_id": mag_id,
                "domain": ann.get("domain", ""),
                "phylum": ann.get("phylum", ""),
                "class": ann.get("class", ""),
                "order": ann.get("order", ""),
                "family": ann.get("family", ""),
                "genus": ann.get("genus", ""),
                "bin_completeness": ann.get("bin_completeness", ""),
                "bin_contamination": ann.get("bin_contamination", ""),
                "source_qc_label": qc_label(completeness, contamination),
                "source_feature_rows": source_feature_rows,
                "methane_term_rows": methane_terms,
                "sulfur_term_rows": sulfur_terms,
                "methyl_substrate_term_rows": methyl_terms,
                "substrate_term_rows": substrate_terms,
                "marker_breadth_count": marker_breadth,
                "processed_mag_expression_support": str(mag_expr_support).lower(),
                "processed_mag_expression_occupancy_fraction": mex.get("occupancy_fraction", "0"),
                "processed_gene_expression_support": str(gene_expr_support).lower(),
                "processed_gene_expression_sum": gex.get("expression_sum", "0"),
                "methane_expressed_gene_rows": safe_int(gex.get("methane_expressed_gene_rows")),
                "sulfur_expressed_gene_rows": safe_int(gex.get("sulfur_expressed_gene_rows")),
                "methyl_substrate_expressed_gene_rows": safe_int(
                    gex.get("methyl_substrate_expressed_gene_rows")
                ),
                "substrate_expressed_gene_rows": safe_int(gex.get("substrate_expressed_gene_rows")),
                "expression_marker_breadth_count": expression_marker_breadth,
                "expression_support_label": expression_label(
                    mag_expr_support,
                    gene_expr_support,
                    safe_int(gex.get("methane_expressed_gene_rows")),
                    safe_int(gex.get("sulfur_expressed_gene_rows")),
                    safe_int(gex.get("methyl_substrate_expressed_gene_rows")),
                    safe_int(gex.get("substrate_expressed_gene_rows")),
                ),
                "direct_source_esm2_input_ready": str(direct_esm2_ready).lower(),
                "prodigal_glm2_input_ready": str(prodigal_glm2_ready).lower(),
                "mrv_readiness_label": label,
                "review_priority_score": fmt(review_priority_score),
                "allowed_claim_wording": allowed_wording(label),
                "blocking_gap": (
                    "completed ESM2/gLM2 outputs, wetland-neighbor bridge tables, environmental "
                    "sample/depth joins, abundance/read coverage, uncertainty propagation, and "
                    "flux/process validation"
                ),
                "next_validation_action": (
                    "after embeddings complete, join nearest-neighbor evidence and promote only "
                    "reviewable rows into mechanism cards with source-aware caveats"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    fields = list(rows[0])
    feature_path = run_dir / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv"
    write_tsv(feature_path, rows, fields)

    card_fields = [
        "card_id",
        "candidate_set",
        "set_rank",
        "lane_id",
        "proteome_id",
        "mag_id",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "source_qc_label",
        "mrv_readiness_label",
        "review_priority_score",
        "marker_breadth_count",
        "expression_marker_breadth_count",
        "methane_term_rows",
        "sulfur_term_rows",
        "methyl_substrate_term_rows",
        "substrate_term_rows",
        "processed_gene_expression_sum",
        "allowed_claim_wording",
        "blocking_gap",
        "next_validation_action",
        "claim_boundary",
    ]
    candidates = [
        row
        for row in rows
        if row["mrv_readiness_label"] == "mrv_feature_scaffold_ready_pending_embedding_outputs"
        and safe_int(row["marker_breadth_count"]) >= 2
        and safe_int(row["expression_marker_breadth_count"]) >= 1
    ]
    candidates = sorted(
        candidates,
        key=lambda row: (
            safe_float(row["review_priority_score"]),
            safe_int(row["methane_term_rows"]),
            safe_float(row["processed_gene_expression_sum"]),
        ),
        reverse=True,
    )[:100]
    card_rows = []
    for rank, row in enumerate(candidates, start=1):
        card_rows.append(
            {
                "card_id": f"mucc_v1_mrv_readiness_review_{rank:03d}",
                "candidate_set": "mrv_feature_readiness_review",
                "set_rank": rank,
                **{field: row.get(field, "") for field in card_fields if field not in {"card_id", "candidate_set", "set_rank"}},
            }
        )
    card_path = run_dir / "candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv"
    write_tsv(card_path, card_rows, card_fields)

    summary = {
        "lane_id": LANE_ID,
        "feature_table": str(feature_path.relative_to(repo_root)),
        "candidate_cards": str(card_path.relative_to(repo_root)),
        "mag_rows": len(rows),
        "candidate_card_rows": len(card_rows),
        "mrv_feature_scaffold_ready_pending_embedding_outputs": sum(
            1
            for row in rows
            if row["mrv_readiness_label"] == "mrv_feature_scaffold_ready_pending_embedding_outputs"
        ),
        "context_feature_scaffold_ready_missing_direct_source_esm2_protein": sum(
            1
            for row in rows
            if row["mrv_readiness_label"]
            == "context_feature_scaffold_ready_missing_direct_source_esm2_protein"
        ),
        "catalog_only_pending_feature_evidence": sum(
            1 for row in rows if row["mrv_readiness_label"] == "catalog_only_pending_feature_evidence"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    summary_path = run_dir / "reports/mucc_v1_mrv_readiness_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
