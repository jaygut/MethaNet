#!/usr/bin/env python3
"""Materialize the MUCC v1 source and molecular-reference lane as a queryable warehouse.

This does not create final MethaNet-curated mechanism calls. It packages the
validated MUCC source manifests, expression support, source-derived feature
scaffolds, completed molecular-reference evidence when present, MRV readiness,
and candidate review cards as partitioned Parquet tables plus a DuckDB catalog
so atlas tooling can query the lane natively while claim boundaries remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
COHORT_RUN_ID = "mucc_v1_owc_wetland_20260626"
CLAIM_BOUNDARY = (
    "MUCC v1 source-scaffold and promoted molecular-reference warehouse; no final MRV risk score, A-E tier, "
    "measured methane flux, carbon-crediting claim, or source-independent "
    "transfer claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--cohort-run-id", default=COHORT_RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=BASE / "cohort_warehouse")
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def read_tsv_df(pd: Any, path: Path) -> Any:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(path, sep="\t", dtype=str, compression=compression).fillna("")


def read_csv_df(pd: Any, path: Path) -> Any:
    return pd.read_csv(path, dtype=str).fillna("")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_cohort(pd: Any, df: Any, cohort_run_id: str) -> Any:
    if "cohort_run_id" not in df.columns:
        df.insert(0, "cohort_run_id", cohort_run_id)
    else:
        df["cohort_run_id"] = cohort_run_id
    return df


def ensure_claim(df: Any, claim_boundary: str = CLAIM_BOUNDARY) -> Any:
    if "claim_boundary" not in df.columns:
        df["claim_boundary"] = claim_boundary
    return df


def build_dim_mag(pd: Any, run_dir: Path, cohort_run_id: str) -> Any:
    manifest = read_tsv_df(pd, run_dir / "manifests/mucc_v1_source_lane_manifest.tsv")
    readiness = read_tsv_df(
        pd,
        run_dir / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv",
    )
    cols = [
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "source_group",
        "analysis_unit_type",
        "local_fna_path",
        "proteome_faa",
        "local_faa_path",
        "local_ffn_path",
        "local_gff_path",
        "payload_status",
        "protein_prediction_status",
        "protein_count",
        "match_status",
        "functional_run_include",
        "esm2_include",
        "glm2_include",
        "mbag_mag_level_include",
        "comparability_status",
        "denominator_status",
        "metadata_mapping_status",
        "gap_reason",
        "recommended_action",
    ]
    dim = manifest[[col for col in cols if col in manifest.columns]].copy()
    readiness_cols = [
        "proteome_id",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "bin_completeness",
        "bin_contamination",
        "source_qc_label",
        "mrv_readiness_label",
        "review_priority_score",
    ]
    dim = dim.merge(
        readiness[[col for col in readiness_cols if col in readiness.columns]],
        on="proteome_id",
        how="left",
    ).fillna("")
    taxonomy_projection_path = (
        run_dir / "functional_features/feature_mucc_v1_taxonomy_projection.tsv"
    )
    if taxonomy_projection_path.is_file():
        projection = read_tsv_df(pd, taxonomy_projection_path)
        projection_cols = [
            "proteome_id",
            "atlas_taxonomy_lineage",
            "atlas_taxonomy_projection_status",
            "atlas_taxonomy_policy",
            "source_kbase_rank_disagreement_count",
            "kbase_rank_fallback_count",
            "atlas_taxonomy_available_rank_count",
            "atlas_taxonomy_rank_provenance_json",
        ]
        for rank in (
            "domain",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ):
            projection_cols.extend([f"atlas_{rank}", f"atlas_{rank}_provenance"])
        dim = dim.merge(
            projection[[col for col in projection_cols if col in projection.columns]],
            on="proteome_id",
            how="left",
            validate="one_to_one",
        ).fillna("")
    dim = add_cohort(pd, dim, cohort_run_id)
    return ensure_claim(dim)


def table_specs(run_dir: Path) -> list[tuple[str, Path, str]]:
    specs = [
        (
            "source_lane_manifest",
            run_dir / "manifests/mucc_v1_source_lane_manifest.tsv",
            "tsv",
        ),
        (
            "functional_manifest",
            run_dir / "manifests/mucc_v1_functional_manifest.tsv",
            "tsv",
        ),
        (
            "glm2_ready_manifest",
            run_dir / "manifests/mucc_v1_glm2_ready_manifest.tsv",
            "tsv",
        ),
        (
            "glm2_ready_gap_register",
            run_dir / "manifests/mucc_v1_glm2_ready_gap_register.tsv",
            "tsv",
        ),
        ("mag_catalog", run_dir / "manifests/mucc_v1_mag_catalog_full.tsv", "tsv"),
        (
            "feature_mucc_v1_zenodo_source_qc",
            run_dir / "functional_features/feature_mucc_v1_zenodo_source_qc.tsv",
            "tsv",
        ),
        (
            "feature_mrv_readiness_mag_level",
            run_dir / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv",
            "tsv",
        ),
        (
            "feature_source_dram_mag_summary",
            run_dir / "functional_features/feature_mucc_v1_source_dram_mag_summary.tsv",
            "tsv",
        ),
        (
            "feature_gene_annotation_mag_summary",
            run_dir
            / "functional_features/feature_mucc_v1_gene_annotation_mag_summary.tsv",
            "tsv",
        ),
        (
            "feature_mag_expression_summary",
            run_dir / "expression/feature_mucc_v1_expression_mag_summary.tsv",
            "tsv",
        ),
        (
            "feature_gene_expression_mag_summary",
            run_dir / "expression/feature_mucc_v1_gene_expression_mag_summary.tsv",
            "tsv",
        ),
        (
            "fact_mag_expression_sample",
            run_dir / "expression/fact_mucc_v1_expression_mag_sample.tsv.gz",
            "tsv",
        ),
        (
            "fact_gene_expression_mag_sample",
            run_dir / "expression/fact_mucc_v1_gene_expression_mag_sample.tsv.gz",
            "tsv",
        ),
        (
            "candidate_mrv_readiness_cards",
            run_dir / "candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv",
            "tsv",
        ),
        (
            "candidate_strategic_review_cards",
            run_dir / "candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv",
            "tsv",
        ),
        (
            "validation_gap_register",
            run_dir / "reports/validation_gap_register.tsv",
            "tsv",
        ),
        (
            "claim_boundary_matrix",
            run_dir / "reports/claim_boundary_matrix.tsv",
            "tsv",
        ),
    ]
    # Promotion artifacts are intentionally optional: the base source-scaffold
    # warehouse can be rebuilt before ESM2/gLM2 complete, while a promoted lane
    # gains these queryable evidence layers once they exist.
    specs.extend(
        [
            (
                "feature_mucc_v1_embedding_status",
                run_dir / "functional_features/feature_mucc_v1_embedding_status.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_glm2_context",
                run_dir / "functional_features/feature_mucc_v1_glm2_context.parquet",
                "parquet",
            ),
            (
                "feature_mucc_v1_sample_ecological_readiness",
                run_dir
                / "environmental_metadata/feature_mucc_v1_sample_ecological_readiness.tsv",
                "tsv",
            ),
            (
                "link_mucc_v1_sequence_authoritative_ecology",
                run_dir
                / "environmental_metadata/link_mucc_v1_sequence_authoritative_ecology.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_authoritative_ecology_readiness",
                run_dir
                / "environmental_metadata/feature_mucc_v1_authoritative_ecology_readiness.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_sample_methods_design_context",
                run_dir
                / "environmental_metadata/feature_mucc_v1_sample_methods_design_context.tsv",
                "tsv",
            ),
            (
                "link_mucc_v1_sequence_bioproject_sample",
                run_dir
                / "environmental_metadata/link_mucc_v1_sequence_bioproject_sample.tsv",
                "tsv",
            ),
            (
                "link_mucc_v1_sequence_sra_sample",
                run_dir / "environmental_metadata/link_mucc_v1_sequence_sra_sample.tsv",
                "tsv",
            ),
            (
                "link_mucc_v1_sequence_jgi_sample",
                run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_sample.tsv",
                "tsv",
            ),
            (
                "link_mucc_v1_sequence_jgi_data_portal",
                run_dir
                / "environmental_metadata/link_mucc_v1_sequence_jgi_data_portal.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_ncbi_bioproject_source_files",
                run_dir
                / "source_audit/ncbi_bioproject_old_woman_creek/source_file_manifest.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_ncbi_sra_source_files",
                run_dir
                / "source_audit/ncbi_sra_sample_crosswalk/source_file_manifest.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_jgi_sample_qc_source_files",
                run_dir
                / "source_audit/jgi_owc_sample_crosswalk/source_file_manifest.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_jgi_data_portal_source_files",
                run_dir
                / "source_audit/jgi_data_portal_catalog/source_file_manifest.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_authoritative_ecology_crosswalk_source_files",
                run_dir
                / "source_audit/authoritative_ecological_crosswalk/source_file_manifest.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_external_source_registry",
                run_dir
                / "environmental_metadata/fact_mucc_v1_external_source_registry.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_essdive_chamber_flux",
                run_dir
                / "environmental_metadata/fact_mucc_v1_essdive_chamber_flux.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_essdive_porewater_ch4",
                run_dir
                / "environmental_metadata/fact_mucc_v1_essdive_porewater_ch4.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_essdive_source_files",
                run_dir
                / "source_audit/essdive_10.15485_1568865/source_file_manifest.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_essdive_gapfilled_tower_ch4_flux",
                run_dir
                / "environmental_metadata/fact_mucc_v1_essdive_gapfilled_tower_ch4_flux.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_essdive_gapfilled_tower_source_files",
                run_dir
                / "source_audit/essdive_10.15485_2500238/source_file_manifest.tsv",
                "tsv",
            ),
            (
                "mucc_v1_source_metadata_recovery_ledger",
                run_dir / "source_audit/mucc_v1_source_metadata_recovery_ledger.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_kbase_public_catalog_reconciliation",
                run_dir
                / "source_audit/kbase_public_workspace_147022/mucc_v1_kbase_public_catalog_reconciliation.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_taxonomy_projection",
                run_dir / "functional_features/feature_mucc_v1_taxonomy_projection.tsv",
                "tsv",
            ),
            (
                "mucc_v1_taxonomy_projection_summary",
                run_dir / "reconciliation/mucc_v1_taxonomy_projection_summary.tsv",
                "tsv",
            ),
            (
                "mucc_v1_methods_sample_design_reconciliation",
                run_dir
                / "reconciliation/mucc_v1_methods_sample_design_reconciliation.tsv",
                "tsv",
            ),
            (
                "candidate_mucc_v1_promoted_molecular_reference_cards",
                run_dir
                / "candidate_cards/mucc_v1_promoted_molecular_reference_cards.tsv",
                "tsv",
            ),
            (
                "mucc_v1_denominator_reconciliation",
                run_dir / "reconciliation/mucc_v1_denominator_reconciliation.tsv",
                "tsv",
            ),
            (
                "mucc_v1_mag_reconciliation",
                run_dir / "reconciliation/mucc_v1_mag_reconciliation.tsv",
                "tsv",
            ),
            (
                "mucc_v1_zenodo_source_qc_reconciliation",
                run_dir / "reconciliation/mucc_v1_zenodo_source_qc_reconciliation.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_zenodo_source_qc_manifest",
                run_dir / "source_audit/zenodo_mucc_v1_qc_source_manifest.tsv",
                "tsv",
            ),
            (
                "mucc_v1_integrated_atlas_promotion_gates",
                run_dir / "reports/mucc_v1_integrated_atlas_promotion_gates.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_source_aware_neighbor_summary",
                run_dir
                / "bridge_reanchoring/integrated_atlas/wetland_reference_neighbor_summary.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_source_aware_neighbor_edges",
                run_dir
                / "bridge_reanchoring/integrated_atlas/wetland_reference_neighbor_edges.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_network_analysis_status",
                run_dir / "network_analysis/network_analysis_status.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_flashweave_edges",
                run_dir / "network_analysis/fact_mucc_v1_flashweave_edges.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_flashweave_edge_atlas_context",
                run_dir
                / "network_analysis/fact_mucc_v1_flashweave_edge_atlas_context.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_flashweave_edge_stability",
                run_dir / "network_analysis/fact_mucc_v1_flashweave_edge_stability.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_flashweave_metadata_edges",
                run_dir / "network_analysis/fact_mucc_v1_flashweave_metadata_edges.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_wgcna_secondary_module_membership",
                run_dir
                / "network_analysis/feature_mucc_v1_wgcna_secondary_module_membership.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_wgcna_secondary_module_summary",
                run_dir
                / "network_analysis/feature_mucc_v1_wgcna_secondary_module_summary.tsv",
                "tsv",
            ),
            (
                "fact_mucc_v1_wgcna_secondary_module_eigengenes",
                run_dir
                / "network_analysis/fact_mucc_v1_wgcna_secondary_module_eigengenes.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_flashweave_node_summary",
                run_dir
                / "network_analysis/feature_mucc_v1_flashweave_node_summary.tsv",
                "tsv",
            ),
            (
                "feature_mucc_v1_flashweave_node_atlas_context",
                run_dir
                / "network_analysis/feature_mucc_v1_flashweave_node_atlas_context.tsv",
                "tsv",
            ),
            (
                "mucc_v1_flashweave_atlas_explorer_summary",
                run_dir
                / "network_analysis/mucc_v1_flashweave_atlas_explorer_summary.tsv",
                "tsv",
            ),
            (
                "mucc_v1_flashweave_network_validation_gates",
                run_dir
                / "network_analysis/mucc_v1_flashweave_network_validation_gates.tsv",
                "tsv",
            ),
        ]
    )
    return specs


def write_partitioned_table(
    df: Any, output_dir: Path, table: str, cohort_run_id: str
) -> dict[str, Any]:
    table_dir = output_dir / "parquet" / table / f"cohort_run_id={cohort_run_id}"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "part-00000.parquet"
    df.to_parquet(path, index=False)
    return {
        "table": table,
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "bytes": path.stat().st_size,
    }


def build_duckdb(output_dir: Path, manifest_rows: list[dict[str, Any]]) -> Path:
    import duckdb

    db_path = output_dir / "functional_atlas.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for row in manifest_rows:
        table = row["table"]
        parquet_glob = str(output_dir / "parquet" / table / "*" / "*.parquet").replace(
            "'", "''"
        )
        con.execute(
            f"CREATE VIEW \"{table}\" AS SELECT * FROM read_parquet('{parquet_glob}')"
        )
    con.close()
    return db_path


def validate(pd: Any, tables: dict[str, Any], run_dir: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(gate: str, status: str, detail: str) -> None:
        checks.append({"gate": gate, "status": status, "detail": detail})

    dim_mag = tables["dim_mag"]
    readiness = tables["feature_mrv_readiness_mag_level"]
    source_manifest = tables["source_lane_manifest"]
    mag_catalog = tables["mag_catalog"]
    source_qc = tables.get("feature_mucc_v1_zenodo_source_qc")
    mag_expr = tables["feature_mag_expression_summary"]
    gene_expr = tables["feature_gene_expression_mag_summary"]
    source_dram = tables["feature_source_dram_mag_summary"]
    glm2_ready = tables["glm2_ready_manifest"]
    embedding_status = tables.get("feature_mucc_v1_embedding_status")
    glm2_context = tables.get("feature_mucc_v1_glm2_context")
    neighbor_summary = tables.get("fact_mucc_v1_source_aware_neighbor_summary")
    network_status = tables.get("fact_mucc_v1_network_analysis_status")
    flashweave_edges = tables.get("fact_mucc_v1_flashweave_edges")
    flashweave_nodes = tables.get("feature_mucc_v1_flashweave_node_summary")
    flashweave_gates = tables.get("mucc_v1_flashweave_network_validation_gates")
    explorer_edges = tables.get("fact_mucc_v1_flashweave_edge_atlas_context")
    explorer_nodes = tables.get("feature_mucc_v1_flashweave_node_atlas_context")
    explorer_summary = tables.get("mucc_v1_flashweave_atlas_explorer_summary")
    wgcna_membership = tables.get("feature_mucc_v1_wgcna_secondary_module_membership")
    wgcna_summary = tables.get("feature_mucc_v1_wgcna_secondary_module_summary")
    wgcna_eigengenes = tables.get("fact_mucc_v1_wgcna_secondary_module_eigengenes")
    wgcna_expected_features = 0
    wgcna_manifest_path = (
        run_dir / "network_analysis/wgcna_secondary_input_manifest.json"
    )
    if wgcna_manifest_path.is_file():
        try:
            wgcna_expected_features = int(
                json.loads(wgcna_manifest_path.read_text()).get(
                    "selected_MAG_features", 0
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            wgcna_expected_features = 0
    essdive_chamber = tables.get("fact_mucc_v1_essdive_chamber_flux")
    essdive_porewater = tables.get("fact_mucc_v1_essdive_porewater_ch4")
    essdive_gapfilled_tower = tables.get(
        "fact_mucc_v1_essdive_gapfilled_tower_ch4_flux"
    )
    source_recovery = tables.get("mucc_v1_source_metadata_recovery_ledger")
    kbase_catalog = tables.get("feature_mucc_v1_kbase_public_catalog_reconciliation")
    taxonomy_projection = tables.get("feature_mucc_v1_taxonomy_projection")
    methods_context = tables.get("feature_mucc_v1_sample_methods_design_context")
    methods_reconciliation = tables.get("mucc_v1_methods_sample_design_reconciliation")
    bioproject_crosswalk = tables.get("link_mucc_v1_sequence_bioproject_sample")
    sra_crosswalk = tables.get("link_mucc_v1_sequence_sra_sample")
    jgi_crosswalk = tables.get("link_mucc_v1_sequence_jgi_sample")
    jgi_data_portal = tables.get("link_mucc_v1_sequence_jgi_data_portal")

    add(
        "source_scaffold_warehouse_materialized",
        "pass",
        f"warehouse tables={len(tables)}; dim_mag rows={len(dim_mag)}; DuckDB catalog is built by this script",
    )
    add(
        "dim_mag_rows",
        "pass" if len(dim_mag) == 2508 else "fail",
        f"dim_mag rows={len(dim_mag)} expected=2508",
    )
    add(
        "mag_catalog_rows",
        "pass" if len(mag_catalog) == 2508 else "fail",
        (
            "checksum-validated Zenodo MAGs.zip archive roster rows=2508; direct source-QC "
            "reconciliation defines the published 2502 HQ/MQ cohort separately"
        ),
    )
    if source_qc is None:
        add(
            "published_HQMQ_QC_roster_reconciliation",
            "fail",
            "Direct Zenodo per-MAG QC evidence table is missing from the warehouse build.",
        )
    else:
        hqmq_rows = (
            source_qc["published_mq_hq_membership_status"]
            .eq("meets_published_MQHQ_CheckM_threshold")
            .sum()
        )
        archive_scope_rows = (
            source_qc["published_mq_hq_membership_status"]
            .eq("does_not_meet_published_MQHQ_CheckM_threshold")
            .sum()
        )
        consistent_rows = (
            source_qc["source_qc_value_consistency_status"]
            .eq("direct_source_qc_values_consistent_across_annotation_rows")
            .sum()
        )
        add(
            "published_HQMQ_QC_roster_reconciliation",
            "pass"
            if len(source_qc) == 2508
            and hqmq_rows == 2502
            and archive_scope_rows == 6
            and consistent_rows == 2508
            else "fail",
            (
                f"direct Zenodo source QC rows={len(source_qc)}; published-MQ/HQ threshold "
                f"members={hqmq_rows}; archive members outside that quality scope={archive_scope_rows}; "
                f"repeated-QC-consistent rows={consistent_rows}"
            ),
        )
    add(
        "source_dram_mapping",
        "pass" if len(source_dram) == 2508 else "fail",
        f"source DRAM summary rows={len(source_dram)} expected=2508",
    )
    add(
        "processed_expression_support",
        "pass" if len(mag_expr) == 1948 and len(gene_expr) == 2508 else "warn",
        f"processed MAG expression rows={len(mag_expr)}; gene-expression summary rows={len(gene_expr)}",
    )
    direct_esm2_ready = (
        source_manifest["esm2_include"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .sum()
        if "esm2_include" in source_manifest.columns
        else 0
    )
    add(
        "direct_source_esm2_input_readiness",
        "warn" if direct_esm2_ready == 2501 else "fail",
        f"direct source-protein ESM2-ready rows={direct_esm2_ready}; seven local FASTA entries remain excluded under current parser",
    )
    glm2_ready_rows = (
        glm2_ready["glm2_include"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .sum()
        if "glm2_include" in glm2_ready.columns
        else 0
    )
    add(
        "prodigal_glm2_input_readiness",
        "pass" if glm2_ready_rows == 2508 else "fail",
        f"Prodigal-derived gLM2-ready rows={glm2_ready_rows} expected=2508",
    )
    add(
        "mrv_readiness_rows",
        "pass" if len(readiness) == 2508 else "fail",
        f"feature_mrv_readiness_mag_level rows={len(readiness)} expected=2508",
    )
    add(
        "proteome_id_unique",
        "pass" if dim_mag["proteome_id"].nunique() == len(dim_mag) else "fail",
        f"unique proteome_id={dim_mag['proteome_id'].nunique()} rows={len(dim_mag)}",
    )
    add(
        "claim_boundary_present",
        "pass" if "claim_boundary" in readiness.columns else "fail",
        "MRV readiness table carries claim_boundary column",
    )
    add(
        "final_claim_lock",
        "pass",
        "Warehouse does not authorize final MRV scores, A-E tiers, measured flux, crediting, or transfer claims.",
    )
    if embedding_status is None:
        add(
            "promoted_embedding_status",
            "blocked",
            "promotion embedding-status table is not materialized",
        )
    else:
        esm2_complete = (
            embedding_status["esm2_embedding_status"]
            .eq("complete_finite_embedding")
            .sum()
        )
        glm2_complete = (
            embedding_status["glm2_context_status"]
            .eq("complete_multiwindow_stability_summary")
            .sum()
        )
        add(
            "promoted_embedding_status",
            "pass"
            if len(embedding_status) == 2508
            and esm2_complete == 2501
            and glm2_complete == 2508
            else "fail",
            f"embedding rows={len(embedding_status)}; ESM2 complete={esm2_complete}; gLM2 complete={glm2_complete}",
        )
    if glm2_context is not None:
        add(
            "gLM2_context_summary",
            "pass" if len(glm2_context) == 2508 else "fail",
            f"gLM2 contextual summary rows={len(glm2_context)} expected=2508",
        )
    add(
        "wetland_neighbor_bridge_table",
        "pass"
        if neighbor_summary is not None and len(neighbor_summary) == 2501
        else "blocked",
        (
            f"source-aware MUCC neighbor summary rows={len(neighbor_summary)} expected=2501"
            if neighbor_summary is not None
            else "Final MUCC wetland-neighbor bridge table is not materialized."
        ),
    )
    add(
        "ecological_network_input",
        "ready" if network_status is not None and len(network_status) else "blocked",
        "FlashWeave/WGCNA input contract is materialized"
        if network_status is not None and len(network_status)
        else "network input contract is not materialized",
    )
    if flashweave_edges is None or flashweave_nodes is None:
        add(
            "flashweave_conditional_network",
            "blocked",
            "FlashWeave result tables are not materialized",
        )
    else:
        add(
            "flashweave_conditional_network",
            "pass"
            if len(flashweave_edges) == 694 and len(flashweave_nodes) == 300
            else "fail",
            f"MAG-to-MAG edges={len(flashweave_edges)} expected=694; node summaries={len(flashweave_nodes)} expected=300",
        )
    if flashweave_gates is not None:
        stability = flashweave_gates.loc[
            flashweave_gates["gate"].eq("edge_stability"), "status"
        ].tolist()
        add(
            "flashweave_edge_stability",
            stability[0] if stability else "blocked",
            "bootstrap or grouped leave-out stability is an explicit ecological validation gate",
        )
    if wgcna_membership is None or wgcna_summary is None or wgcna_eigengenes is None:
        add(
            "WGCNA_secondary_descriptive_modules",
            "ready",
            "WGCNA is optional and secondary until a compatible R/WGCNA runtime writes validated outputs",
        )
    else:
        non_grey_modules = wgcna_summary["module"].ne("grey").sum()
        add(
            "WGCNA_secondary_descriptive_modules",
            "pass"
            if wgcna_expected_features > 0
            and len(wgcna_membership) == wgcna_expected_features
            and wgcna_membership["proteome_id"].nunique() == wgcna_expected_features
            and len(wgcna_eigengenes) == 133
            and non_grey_modules >= 1
            else "fail",
            (
                f"secondary module memberships={len(wgcna_membership)} expected="
                f"{wgcna_expected_features}; eigengene sample rows="
                f"{len(wgcna_eigengenes)}; non-grey descriptive modules={non_grey_modules}; "
                "WGCNA does not replace primary FlashWeave conditional associations"
            ),
        )
    if explorer_edges is None or explorer_nodes is None or explorer_summary is None:
        add(
            "flashweave_atlas_explorer",
            "blocked",
            "Ocean-M-inspired taxonomy/marker/stability explorer tables are not materialized",
        )
    else:
        explorer_eligible = (
            explorer_edges["network_explorer_visibility_status"]
            .eq("stability_and_taxonomy_filter_eligible")
            .sum()
        )
        add(
            "flashweave_atlas_explorer",
            "pass"
            if len(explorer_edges) == 694
            and len(explorer_nodes) == 300
            and explorer_eligible == 126
            else "fail",
            (
                f"Ocean-M-inspired edge/node context rows={len(explorer_edges)}/{len(explorer_nodes)}; "
                f"stability-and-taxonomy filter eligible={explorer_eligible}; all remain exploratory "
                "conditional associations without exact environmental or flux joins"
            ),
        )
    if essdive_chamber is None or essdive_porewater is None:
        add(
            "ESS_DIVE_source_observations",
            "blocked",
            "ESS-DIVE source observations are not staged",
        )
    else:
        chamber_valid = (
            essdive_chamber["source_value_status"].eq("reported_valid").sum()
        )
        porewater_valid = (
            essdive_porewater["source_value_status"].eq("reported_valid").sum()
        )
        add(
            "ESS_DIVE_source_observations",
            "pass"
            if len(essdive_chamber) == 275 and len(essdive_porewater) == 5280
            else "fail",
            (
                f"chamber rows={len(essdive_chamber)} valid={chamber_valid}; "
                f"porewater rows={len(essdive_porewater)} valid={porewater_valid}; "
                "all rows remain unlinked to sequencing samples"
            ),
        )
    if essdive_gapfilled_tower is None:
        add(
            "ESS_DIVE_gapfilled_tower_flux_source_context",
            "blocked",
            "ESS-DIVE DOI 10.15485/2500238 gap-filled tower flux context is not staged",
        )
    else:
        valid = (
            essdive_gapfilled_tower["source_value_status"].eq("reported_valid").sum()
        )
        unlinked = (
            essdive_gapfilled_tower["sample_join_status"]
            .eq("unlinked_no_authoritative_sequence_sample_crosswalk")
            .all()
        )
        years = sorted(
            essdive_gapfilled_tower["source_datetime_start_local_or_timezone_unknown"]
            .str.slice(0, 4)
            .unique()
        )
        add(
            "ESS_DIVE_gapfilled_tower_flux_source_context",
            "pass"
            if valid and unlinked and years == ["2015", "2016", "2020", "2021", "2022"]
            else "fail",
            (
                f"half-hourly rows={len(essdive_gapfilled_tower)}; valid CH4 rows={valid}; "
                f"year coverage={','.join(years)}; all source context remains unlinked to sequencing samples"
            ),
        )
    if source_recovery is None:
        add(
            "published_source_metadata_recovery",
            "blocked",
            "authoritative supplement-recovery ledger is not materialized",
        )
    else:
        table_s1 = source_recovery.loc[
            source_recovery["artifact_role"].eq(
                "published_Table_S1_to_S13_accession_spreadsheet"
            )
        ]
        table_s1_status = (
            table_s1.iloc[0]["availability_status"] if len(table_s1) == 1 else "missing"
        )
        add(
            "published_source_metadata_recovery",
            "pass" if table_s1_status == "parseable_xlsx_container" else "blocked",
            (
                f"Table S1-S13 spreadsheet status={table_s1_status}; NCBI accession, exact "
                "sample date/depth, and paper HQ/MQ membership remain unresolved until a parseable payload is available"
            ),
        )
    if kbase_catalog is None:
        add(
            "KBase_public_catalog_reconciliation",
            "blocked",
            "public KBase Workspace roster reconciliation is not materialized",
        )
    else:
        kbase_matches = (
            kbase_catalog["kbase_roster_reconciliation_status"]
            .eq("exact_MAG_id_match_public_KBase_GenomeSet")
            .sum()
        )
        kbase_absent = (
            kbase_catalog["kbase_roster_reconciliation_status"]
            .eq("Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet")
            .sum()
        )
        supplemental_taxonomy = (
            kbase_catalog["taxonomy_reconciliation_status"]
            .eq("KBase_Gtdb_taxonomy_supplemental_only")
            .sum()
        )
        add(
            "KBase_public_catalog_reconciliation",
            "partial"
            if len(kbase_catalog) == 2508
            and kbase_matches == 2494
            and kbase_absent == 14
            else "fail",
            (
                f"public KBase exact MAG-ID matches={kbase_matches}; Zenodo archive MAGs absent "
                f"from KBase={kbase_absent}; KBase supplementary GTDB-only values={supplemental_taxonomy}; "
                "KBase membership does not resolve the published 2,502 HQ/MQ subset"
            ),
        )
    if taxonomy_projection is None:
        add(
            "taxonomy_projection",
            "blocked",
            "rank-aware source/KBase taxonomy projection is not materialized",
        )
    else:
        taxonomy_available = taxonomy_projection["atlas_taxonomy_lineage"].ne("").sum()
        taxonomy_unavailable = (
            taxonomy_projection["atlas_taxonomy_projection_status"]
            .eq("taxonomy_unavailable_in_source_and_KBase")
            .sum()
        )
        projection_keys = taxonomy_projection["proteome_id"].nunique()
        add(
            "taxonomy_projection",
            "partial"
            if len(taxonomy_projection) == 2508
            and projection_keys == 2508
            and taxonomy_available == 2503
            and taxonomy_unavailable == 5
            else "fail",
            (
                f"rank-aware projection rows={len(taxonomy_projection)}; unique proteome_id="
                f"{projection_keys}; atlas lineage available={taxonomy_available}; "
                f"unavailable in both sources={taxonomy_unavailable}; source remains primary and "
                "KBase only fills missing ranks"
            ),
        )
    if methods_context is None or methods_reconciliation is None:
        add(
            "published_methods_design_context",
            "blocked",
            "SuF1-derived sample-design context and reconciliation table are not materialized",
        )
    else:
        direct_depth = (
            methods_context["methods_design_context_status"]
            .eq("validated_2018_methods_context_direct_depth_code")
            .sum()
        )
        d6_pending = (
            methods_context["methods_design_context_status"]
            .eq("validated_2018_cohort_but_raw_depth_code_reconciliation_pending")
            .sum()
        )
        exact_table_blocked = (
            methods_context["exact_table_s4_row_status"]
            .eq("blocked_published_Tables_S1_S13_XLSX_malformed")
            .sum()
        )
        add(
            "published_methods_design_context",
            "partial"
            if len(methods_context) == 133
            and direct_depth == 91
            and d6_pending == 18
            and exact_table_blocked == 133
            else "fail",
            (
                f"methods-derived context rows={len(methods_context)}; direct coded-depth rows={direct_depth}; "
                f"D6/D5 reconciliation pending={d6_pending}; all rows retain Table S4 block={exact_table_blocked}; "
                "no methods-derived row is an ecological or flux join"
            ),
        )
    if bioproject_crosswalk is None:
        add(
            "NCBI_BioProject_sequence_project_crosswalk",
            "blocked",
            "NCBI BioProject sample-title crosswalk is not staged",
        )
    else:
        mapped = (
            bioproject_crosswalk["sample_project_link_status"]
            .eq("mapped_to_authoritative_NCBI_BioProject_title")
            .sum()
        )
        add(
            "NCBI_BioProject_sequence_project_crosswalk",
            "partial" if mapped else "blocked",
            (
                f"mapped sequence-project rows={mapped}/{len(bioproject_crosswalk)}; exact collection "
                "datetime/depth and field-observation joins remain unresolved"
            ),
        )
    if sra_crosswalk is None:
        add(
            "NCBI_SRA_sample_identity_and_collection_date_crosswalk",
            "blocked",
            "NCBI SRA package/BioSample crosswalk is not staged",
        )
    else:
        exact = (
            sra_crosswalk["sra_sample_identity_status"]
            .eq("exact_source_label_to_NCBI_SRA_package")
            .sum()
        )
        exact_dates = (
            sra_crosswalk["sra_collection_date_status"]
            .eq("exact_collection_date_from_NCBI_SRA_sample_attributes")
            .sum()
        )
        no_depth = (
            sra_crosswalk["sra_depth_cm_join_status"]
            .eq("not_reported_by_NCBI_SRA_sample_attributes")
            .sum()
        )
        rna_seq = sra_crosswalk["sra_library_strategy"].eq("RNA-Seq").sum()
        wgs = sra_crosswalk["sra_library_strategy"].eq("WGS").sum()
        add(
            "NCBI_SRA_sample_identity_and_collection_date_crosswalk",
            "partial"
            if len(sra_crosswalk) == 133
            and exact == 130
            and exact_dates == 107
            and no_depth == 130
            else "fail",
            (
                f"entity-validated SRA package rows={exact}/{len(sra_crosswalk)}; exact collection "
                f"dates={exact_dates}; depth not reported={no_depth}; library strategies RNA-Seq={rna_seq}, "
                f"WGS={wgs}; no record authorizes a depth, environmental, or flux join"
            ),
        )
    if jgi_crosswalk is None:
        add(
            "JGI_Sample_QC_identity_crosswalk",
            "blocked",
            "JGI Sample QC identity crosswalk is not staged",
        )
    else:
        exact = (
            jgi_crosswalk["jgi_sample_identity_status"]
            .eq("exact_source_label_to_JGI_Sample_QC_record")
            .sum()
        )
        final_portals = jgi_crosswalk["jgi_final_deliverable_portal_id"].ne("").sum()
        add(
            "JGI_Sample_QC_identity_crosswalk",
            "partial" if final_portals else "blocked",
            (
                f"final JGI delivery portals={final_portals}/{len(jgi_crosswalk)}; "
                f"exact JGI Sample QC identities={exact}/{len(jgi_crosswalk)}; "
                "portal identity and any operational receipt/QC dates are not collection date, depth, environmental, or flux joins"
            ),
        )
    if jgi_data_portal is None:
        add(
            "JGI_Data_Portal_catalog_identity_crosswalk",
            "blocked",
            "JGI Data Portal source-label catalog crosswalk is not staged",
        )
    else:
        exact = (
            jgi_data_portal["jgi_data_portal_identity_status"]
            .eq("exact_source_label_to_JGI_Data_Portal_record_pair")
            .sum()
        )
        aliases = (
            jgi_data_portal["jgi_data_portal_label_mapping_method"]
            .eq("deterministic_July_to_Jul")
            .sum()
        )
        no_depth = (
            jgi_data_portal["depth_cm_join_status"]
            .eq("not_reported_by_JGI_Data_Portal_catalog")
            .sum()
        )
        add(
            "JGI_Data_Portal_catalog_identity_crosswalk",
            "partial"
            if len(jgi_data_portal) == 133
            and exact == 107
            and aliases == 27
            and no_depth == 107
            else "fail",
            (
                f"exact source-label-specific JGI catalog record pairs={exact}/{len(jgi_data_portal)}; "
                f"controlled July-to-Jul aliases={aliases}; JGI catalog records with no reported "
                f"depth={no_depth}; catalog identity/processing evidence does not authorize a "
                "collection-time, ecological, environmental, or flux join"
            ),
        )
    return checks


def markdown_report(
    output_dir: Path,
    cohort_run_id: str,
    manifest_rows: list[dict[str, Any]],
    checks: list[dict[str, str]],
    duckdb_path: Path,
) -> str:
    status = "pass" if all(row["status"] != "fail" for row in checks) else "fail"
    lines = [
        "# MUCC v1 Source and Molecular-Reference Warehouse",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Cohort run ID: `{cohort_run_id}`",
        f"Validation status: `{status}`",
        "",
        "This warehouse materializes validated MUCC v1 source-scaffold and, when "
        "available, promoted molecular-reference artifacts as partitioned Parquet tables "
        "plus a DuckDB catalog. It is queryable atlas infrastructure, not a final curated "
        "MethaNet mechanism warehouse.",
        "",
        f"DuckDB catalog: `{duckdb_path}`",
        "",
        "## Tables",
        "",
        "| Table | Rows | Columns |",
        "| --- | ---: | ---: |",
    ]
    for row in manifest_rows:
        lines.append(f"| `{row['table']}` | {row['rows']} | {row['columns']} |")
    lines.extend(
        [
            "",
            "## Validation Gates",
            "",
            "| Gate | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for row in checks:
        lines.append(f"| `{row['gate']}` | {row['status']} | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            CLAIM_BOUNDARY,
            "",
            "The warehouse preserves missing, pending, scaffolded, and blocked evidence as "
            "explicit rows. Exact sample/depth/environment joins, abundance/read coverage, "
            "uncertainty propagation, and flux/process validation remain ecological completion gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> int:
    import pandas as pd

    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    output_dir = resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, Any] = {"dim_mag": build_dim_mag(pd, run_dir, args.cohort_run_id)}
    for table, path, kind in table_specs(run_dir):
        if not path.exists():
            continue
        if kind == "tsv":
            df = read_tsv_df(pd, path)
        elif kind == "csv":
            df = read_csv_df(pd, path)
        elif kind == "parquet":
            df = pd.read_parquet(path).fillna("")
        else:
            raise ValueError(f"unsupported table kind: {kind}")
        df = add_cohort(pd, df, args.cohort_run_id)
        tables[table] = ensure_claim(df)

    manifest_rows = [
        write_partitioned_table(df, output_dir, table, args.cohort_run_id)
        for table, df in sorted(tables.items())
    ]
    write_tsv(output_dir / "cohort_table_manifest.tsv", manifest_rows)
    duckdb_path = build_duckdb(output_dir, manifest_rows)
    checks = validate(pd, tables, run_dir)
    write_tsv(output_dir / "validation_gates.tsv", checks)
    (output_dir / "DATA_ARCHITECTURE_VALIDATION.md").write_text(
        markdown_report(
            output_dir, args.cohort_run_id, manifest_rows, checks, duckdb_path
        )
    )
    summary = {
        "output_dir": str(output_dir.relative_to(repo_root)),
        "cohort_run_id": args.cohort_run_id,
        "tables": len(manifest_rows),
        "duckdb": str(duckdb_path.relative_to(repo_root)),
        "validation_status": "pass"
        if all(row["status"] != "fail" for row in checks)
        else "fail",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / "warehouse_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["validation_status"] == "pass" else 1


def main() -> int:
    return build(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
