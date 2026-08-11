#!/usr/bin/env python3
"""Build a source-backed, claim-safe dashboard artifact for the MUCC wetland lane.

The resulting JSON is a bounded snapshot for the portable Data Analytics
dashboard reader. It deliberately visualizes integration coverage, provenance,
and validation gates rather than modelling methane flux or assigning MRV risk.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
CLAIM_BOUNDARY = (
    "This dashboard is a source-backed molecular-reference and integration-status view. "
    "It does not establish measured methane flux, a causal microbial interaction, a final "
    "MRV score or A–E tier, carbon-crediting eligibility, or source-independent ecological transfer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "reports/mucc_v1_wetland_atlas_dashboard_artifact.json",
        help="Artifact JSON path, relative to --repo-root unless absolute.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()[0]


def rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cursor = con.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, record, strict=True)) for record in cursor.fetchall()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_number(value: Any) -> int | float:
    if isinstance(value, (int, float)):
        return value
    return float(value) if "." in str(value) else int(value)


def make_source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    tables: list[str],
    generated_at: str,
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "sql": sql,
            "description": description,
            "executed_at": generated_at,
            "tables_used": tables,
            "metric_definitions": metric_definitions or [],
        },
    }


def build_artifact(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    warehouse_path = run_dir / "cohort_warehouse/functional_atlas.duckdb"
    audit_path = run_dir / "reports/mucc_v1_integration_completion_audit.tsv"
    if not warehouse_path.is_file():
        raise FileNotFoundError(f"Warehouse not found: {warehouse_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Integration audit not found: {audit_path}")

    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        summary = {
            "warehouse_table_count": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main'",
                )
            ),
            "archive_mag_roster": int(scalar(con, "SELECT COUNT(*) FROM dim_mag")),
            "archive_mag_distinct": int(
                scalar(con, "SELECT COUNT(DISTINCT proteome_id) FROM dim_mag")
            ),
            "published_hqmq_headline": int(
                scalar(
                    con,
                    """
                    SELECT CAST(value AS INTEGER)
                    FROM mucc_v1_denominator_reconciliation
                    WHERE metric = 'published_high_medium_quality_mag_headline'
                    """,
                )
            ),
            "zenodo_source_qc_rows": int(
                scalar(con, "SELECT COUNT(*) FROM feature_mucc_v1_zenodo_source_qc")
            ),
            "zenodo_source_qc_consistent_rows": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*) FROM feature_mucc_v1_zenodo_source_qc
                    WHERE source_qc_value_consistency_status = 'direct_source_qc_values_consistent_across_annotation_rows'
                    """,
                )
            ),
            "published_hqmq_qc_reconciled_mag_count": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*) FROM feature_mucc_v1_zenodo_source_qc
                    WHERE published_mq_hq_membership_status = 'meets_published_MQHQ_CheckM_threshold'
                    """,
                )
            ),
            "archive_mag_outside_published_hqmq_qc_scope": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*) FROM feature_mucc_v1_zenodo_source_qc
                    WHERE published_mq_hq_membership_status = 'does_not_meet_published_MQHQ_CheckM_threshold'
                    """,
                )
            ),
            "kbase_public_exact_mag_matches": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_kbase_public_catalog_reconciliation
                    WHERE kbase_roster_reconciliation_status = 'exact_MAG_id_match_public_KBase_GenomeSet'
                    """,
                )
            ),
            "zenodo_archive_mag_absent_from_kbase": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_kbase_public_catalog_reconciliation
                    WHERE kbase_roster_reconciliation_status = 'Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet'
                    """,
                )
            ),
            "kbase_supplemental_taxonomy_only": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_kbase_public_catalog_reconciliation
                    WHERE taxonomy_reconciliation_status = 'KBase_Gtdb_taxonomy_supplemental_only'
                    """,
                )
            ),
            "kbase_source_taxonomy_differences": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_kbase_public_catalog_reconciliation
                    WHERE taxonomy_reconciliation_status = 'source_and_KBase_Gtdb_taxonomy_differ'
                    """,
                )
            ),
            "atlas_taxonomy_available_mag_count": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_taxonomy_projection
                    WHERE atlas_taxonomy_lineage <> ''
                    """,
                )
            ),
            "atlas_taxonomy_unavailable_both_sources": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_taxonomy_projection
                    WHERE atlas_taxonomy_projection_status = 'taxonomy_unavailable_in_source_and_KBase'
                    """,
                )
            ),
            "atlas_taxonomy_kbase_rank_fallbacks": int(
                scalar(
                    con,
                    """
                    SELECT SUM(CAST(kbase_rank_fallback_count AS INTEGER))
                    FROM feature_mucc_v1_taxonomy_projection
                    """,
                )
            ),
            "atlas_taxonomy_source_kbase_rank_disagreements": int(
                scalar(
                    con,
                    """
                    SELECT SUM(CAST(source_kbase_rank_disagreement_count AS INTEGER))
                    FROM feature_mucc_v1_taxonomy_projection
                    """,
                )
            ),
            "expression_supported_mag_count": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mrv_readiness_mag_level
                    WHERE processed_mag_expression_support = 'true'
                    """,
                )
            ),
            "source_expression_sample_count": int(
                scalar(
                    con,
                    "SELECT COUNT(DISTINCT sample_id) FROM fact_mag_expression_sample",
                )
            ),
            "bioproject_crosswalk_rows": int(
                scalar(
                    con, "SELECT COUNT(*) FROM link_mucc_v1_sequence_bioproject_sample"
                )
            ),
            "bioproject_crosswalk_distinct_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(DISTINCT sample_id) FROM link_mucc_v1_sequence_bioproject_sample",
                )
            ),
            "bioproject_mapped_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_bioproject_sample
                    WHERE sample_project_link_status = 'mapped_to_authoritative_NCBI_BioProject_title'
                    """,
                )
            ),
            "sra_sample_crosswalk_rows": int(
                scalar(con, "SELECT COUNT(*) FROM link_mucc_v1_sequence_sra_sample")
            ),
            "sra_sample_crosswalk_distinct_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(DISTINCT sample_id) FROM link_mucc_v1_sequence_sra_sample",
                )
            ),
            "sra_exact_package_identity_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_sra_sample
                    WHERE sra_sample_identity_status = 'exact_source_label_to_NCBI_SRA_package'
                    """,
                )
            ),
            "sra_exact_collection_date_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_sra_sample
                    WHERE sra_collection_date_status = 'exact_collection_date_from_NCBI_SRA_sample_attributes'
                    """,
                )
            ),
            "sra_depth_not_reported_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_sra_sample
                    WHERE sra_depth_cm_join_status = 'not_reported_by_NCBI_SRA_sample_attributes'
                    """,
                )
            ),
            "sra_declared_rna_seq_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM link_mucc_v1_sequence_sra_sample WHERE sra_library_strategy = 'RNA-Seq'",
                )
            ),
            "sra_declared_wgs_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM link_mucc_v1_sequence_sra_sample WHERE sra_library_strategy = 'WGS'",
                )
            ),
            "jgi_sample_crosswalk_rows": int(
                scalar(con, "SELECT COUNT(*) FROM link_mucc_v1_sequence_jgi_sample")
            ),
            "jgi_sample_crosswalk_distinct_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(DISTINCT sample_id) FROM link_mucc_v1_sequence_jgi_sample",
                )
            ),
            "jgi_exact_identity_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_jgi_sample
                    WHERE jgi_sample_identity_status = 'exact_source_label_to_JGI_Sample_QC_record'
                    """,
                )
            ),
            "jgi_final_delivery_portal_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_jgi_sample
                    WHERE jgi_final_deliverable_portal_id <> ''
                    """,
                )
            ),
            "jgi_data_portal_crosswalk_rows": int(
                scalar(
                    con, "SELECT COUNT(*) FROM link_mucc_v1_sequence_jgi_data_portal"
                )
            ),
            "jgi_data_portal_crosswalk_distinct_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(DISTINCT sample_id) FROM link_mucc_v1_sequence_jgi_data_portal",
                )
            ),
            "jgi_data_portal_exact_identity_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_jgi_data_portal
                    WHERE jgi_data_portal_identity_status = 'exact_source_label_to_JGI_Data_Portal_record_pair'
                    """,
                )
            ),
            "jgi_data_portal_july_alias_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM link_mucc_v1_sequence_jgi_data_portal
                    WHERE jgi_data_portal_label_mapping_method = 'deterministic_July_to_Jul'
                    """,
                )
            ),
            "jgi_data_portal_indexed_files": int(
                scalar(
                    con,
                    """
                    SELECT SUM(CAST(jgi_data_portal_indexed_file_count AS INTEGER))
                    FROM link_mucc_v1_sequence_jgi_data_portal
                    """,
                )
            ),
            "jgi_data_portal_purged_indexed_files": int(
                scalar(
                    con,
                    """
                    SELECT SUM(CAST(jgi_data_portal_purged_file_count AS INTEGER))
                    FROM link_mucc_v1_sequence_jgi_data_portal
                    """,
                )
            ),
            "methods_design_context_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM feature_mucc_v1_sample_methods_design_context",
                )
            ),
            "methods_2018_cohort_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_sample_methods_design_context
                    WHERE methods_cohort = 'field_2018_metatranscriptome_design'
                    """,
                )
            ),
            "methods_direct_depth_context_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_sample_methods_design_context
                    WHERE methods_design_context_status = 'validated_2018_methods_context_direct_depth_code'
                    """,
                )
            ),
            "methods_d6_reconciliation_pending_samples": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_sample_methods_design_context
                    WHERE methods_design_context_status = 'validated_2018_cohort_but_raw_depth_code_reconciliation_pending'
                    """,
                )
            ),
            "flashweave_mag_mag_edges": int(
                scalar(
                    con, "SELECT COUNT(*) FROM fact_mucc_v1_flashweave_edge_stability"
                )
            ),
            "flashweave_stable_edges": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM fact_mucc_v1_flashweave_edge_stability
                    WHERE stability_class = 'stable_at_or_above_threshold'
                    """,
                )
            ),
            "flashweave_iterations": int(
                scalar(
                    con,
                    "SELECT MAX(CAST(iterations AS INTEGER)) FROM fact_mucc_v1_flashweave_edge_stability",
                )
            ),
            "flashweave_explorer_edge_rows": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM fact_mucc_v1_flashweave_edge_atlas_context",
                )
            ),
            "flashweave_explorer_stability_taxonomy_eligible_edges": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM fact_mucc_v1_flashweave_edge_atlas_context
                    WHERE network_explorer_visibility_status = 'stability_and_taxonomy_filter_eligible'
                    """,
                )
            ),
            "flashweave_explorer_taxonomy_conflict_exposure_edges": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM fact_mucc_v1_flashweave_edge_atlas_context
                    WHERE endpoint_taxonomy_conflict_exposure = 'true'
                    """,
                )
            ),
            "wgcna_module_memberships": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM feature_mucc_v1_wgcna_secondary_module_membership",
                )
            ),
            "wgcna_non_grey_modules": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_wgcna_secondary_module_summary
                    WHERE module <> 'grey'
                    """,
                )
            ),
            "wgcna_unassigned_grey_mag_count": int(
                scalar(
                    con,
                    """
                    SELECT COALESCE(SUM(CAST(mag_count AS INTEGER)), 0)
                    FROM feature_mucc_v1_wgcna_secondary_module_summary
                    WHERE module = 'grey'
                    """,
                )
            ),
            "wgcna_eigengene_samples": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM fact_mucc_v1_wgcna_secondary_module_eigengenes",
                )
            ),
            "essdive_chamber_records": int(
                scalar(con, "SELECT COUNT(*) FROM fact_mucc_v1_essdive_chamber_flux")
            ),
            "essdive_chamber_valid": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM fact_mucc_v1_essdive_chamber_flux
                    WHERE source_value_status = 'reported_valid'
                    """,
                )
            ),
            "essdive_porewater_records": int(
                scalar(con, "SELECT COUNT(*) FROM fact_mucc_v1_essdive_porewater_ch4")
            ),
            "essdive_porewater_valid": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM fact_mucc_v1_essdive_porewater_ch4
                    WHERE source_value_status = 'reported_valid'
                    """,
                )
            ),
            "essdive_gapfilled_tower_records": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM fact_mucc_v1_essdive_gapfilled_tower_ch4_flux",
                )
            ),
            "essdive_gapfilled_tower_valid": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM fact_mucc_v1_essdive_gapfilled_tower_ch4_flux
                    WHERE source_value_status = 'reported_valid'
                    """,
                )
            ),
            "exact_ecological_sample_joins": int(
                scalar(
                    con,
                    """
                    SELECT COUNT(*)
                    FROM feature_mucc_v1_sample_ecological_readiness
                    WHERE sample_ecological_validation_status NOT LIKE 'blocked%'
                    """,
                )
            ),
            "ecological_readiness_sample_count": int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM feature_mucc_v1_sample_ecological_readiness",
                )
            ),
        }
        summary["expression_mag_fraction"] = (
            summary["expression_supported_mag_count"] / summary["archive_mag_roster"]
        )
        summary["bioproject_mapped_fraction"] = (
            summary["bioproject_mapped_samples"] / summary["bioproject_crosswalk_rows"]
        )
        summary["bioproject_unmapped_samples"] = (
            summary["bioproject_crosswalk_rows"] - summary["bioproject_mapped_samples"]
        )
        summary["sra_exact_package_identity_fraction"] = (
            summary["sra_exact_package_identity_samples"]
            / summary["sra_sample_crosswalk_rows"]
        )
        summary["sra_exact_collection_date_fraction"] = (
            summary["sra_exact_collection_date_samples"]
            / summary["sra_sample_crosswalk_rows"]
        )
        summary["jgi_exact_identity_fraction"] = (
            summary["jgi_exact_identity_samples"] / summary["jgi_sample_crosswalk_rows"]
        )
        summary["jgi_final_delivery_portal_fraction"] = (
            summary["jgi_final_delivery_portal_samples"]
            / summary["jgi_sample_crosswalk_rows"]
        )
        summary["jgi_data_portal_exact_identity_fraction"] = (
            summary["jgi_data_portal_exact_identity_samples"]
            / summary["jgi_data_portal_crosswalk_rows"]
        )
        summary["methods_2018_cohort_fraction"] = (
            summary["methods_2018_cohort_samples"]
            / summary["methods_design_context_samples"]
        )
        summary["methods_direct_depth_context_fraction"] = (
            summary["methods_direct_depth_context_samples"]
            / summary["methods_design_context_samples"]
        )
        summary["flashweave_stable_edge_fraction"] = (
            summary["flashweave_stable_edges"] / summary["flashweave_mag_mag_edges"]
        )
        summary["flashweave_explorer_stability_taxonomy_eligible_fraction"] = (
            summary["flashweave_explorer_stability_taxonomy_eligible_edges"]
            / summary["flashweave_explorer_edge_rows"]
        )
        summary["essdive_chamber_valid_fraction"] = (
            summary["essdive_chamber_valid"] / summary["essdive_chamber_records"]
        )
        summary["essdive_porewater_valid_fraction"] = (
            summary["essdive_porewater_valid"] / summary["essdive_porewater_records"]
        )
        summary["essdive_gapfilled_tower_valid_fraction"] = (
            summary["essdive_gapfilled_tower_valid"]
            / summary["essdive_gapfilled_tower_records"]
        )
        summary["mag_archive_duplicate_ids"] = (
            summary["archive_mag_roster"] - summary["archive_mag_distinct"]
        )
        summary["bioproject_crosswalk_duplicate_samples"] = (
            summary["bioproject_crosswalk_rows"]
            - summary["bioproject_crosswalk_distinct_samples"]
        )
        summary["sra_sample_crosswalk_duplicate_samples"] = (
            summary["sra_sample_crosswalk_rows"]
            - summary["sra_sample_crosswalk_distinct_samples"]
        )
        summary["jgi_sample_crosswalk_duplicate_samples"] = (
            summary["jgi_sample_crosswalk_rows"]
            - summary["jgi_sample_crosswalk_distinct_samples"]
        )
        summary["jgi_data_portal_crosswalk_duplicate_samples"] = (
            summary["jgi_data_portal_crosswalk_rows"]
            - summary["jgi_data_portal_crosswalk_distinct_samples"]
        )

        coverage = [
            {
                "evidence_layer": "Checksum-validated MAG archive roster",
                "numerator": summary["archive_mag_distinct"],
                "denominator": summary["archive_mag_roster"],
                "coverage_fraction": 1.0,
                "coverage_label": "checksum-validated archive membership",
            },
            {
                "evidence_layer": "Paper-defined HQ/MQ MAG quality roster",
                "numerator": summary["published_hqmq_qc_reconciled_mag_count"],
                "denominator": summary["archive_mag_roster"],
                "coverage_fraction": summary["published_hqmq_qc_reconciled_mag_count"]
                / summary["archive_mag_roster"],
                "coverage_label": "direct record-specific Zenodo QC; CheckM completeness >=50% and contamination <10%; six archive MAGs remain explicit outside this scope",
            },
            {
                "evidence_layer": "Public KBase GenomeSet identity reconciliation",
                "numerator": summary["kbase_public_exact_mag_matches"],
                "denominator": summary["archive_mag_roster"],
                "coverage_fraction": summary["kbase_public_exact_mag_matches"]
                / summary["archive_mag_roster"],
                "coverage_label": "exact MAG-ID matches only; does not assign published HQ/MQ membership",
            },
            {
                "evidence_layer": "Conflict-preserving atlas taxonomy projection",
                "numerator": summary["atlas_taxonomy_available_mag_count"],
                "denominator": summary["archive_mag_roster"],
                "coverage_fraction": summary["atlas_taxonomy_available_mag_count"]
                / summary["archive_mag_roster"],
                "coverage_label": "source ranks primary; KBase fills only missing ranks; all conflicts remain queryable",
            },
            {
                "evidence_layer": "Processed MAG expression support",
                "numerator": summary["expression_supported_mag_count"],
                "denominator": summary["archive_mag_roster"],
                "coverage_fraction": summary["expression_mag_fraction"],
                "coverage_label": "MAGs with processed expression support",
            },
            {
                "evidence_layer": "Sequence-to-BioProject title crosswalk",
                "numerator": summary["bioproject_mapped_samples"],
                "denominator": summary["bioproject_crosswalk_rows"],
                "coverage_fraction": summary["bioproject_mapped_fraction"],
                "coverage_label": "source sample columns matched to BioProject titles",
            },
            {
                "evidence_layer": "Sequence-to-exact NCBI SRA package",
                "numerator": summary["sra_exact_package_identity_samples"],
                "denominator": summary["sra_sample_crosswalk_rows"],
                "coverage_fraction": summary["sra_exact_package_identity_fraction"],
                "coverage_label": "entity-validated SRA/BioSample/experiment/run package identity; not a depth, environment, or flux join",
            },
            {
                "evidence_layer": "Exact NCBI SRA collection dates",
                "numerator": summary["sra_exact_collection_date_samples"],
                "denominator": summary["sra_sample_crosswalk_rows"],
                "coverage_fraction": summary["sra_exact_collection_date_fraction"],
                "coverage_label": "date-only attributes; no time-of-day, depth, field chemistry, or flux-window crosswalk",
            },
            {
                "evidence_layer": "Sequence-to-JGI final delivery portal",
                "numerator": summary["jgi_final_delivery_portal_samples"],
                "denominator": summary["jgi_sample_crosswalk_rows"],
                "coverage_fraction": summary["jgi_final_delivery_portal_fraction"],
                "coverage_label": "JGI project/portal identity; exact JGI Sample identity is tracked separately and neither is an ecological join",
            },
            {
                "evidence_layer": "Sequence-to-exact JGI Sample QC identity",
                "numerator": summary["jgi_exact_identity_samples"],
                "denominator": summary["jgi_sample_crosswalk_rows"],
                "coverage_fraction": summary["jgi_exact_identity_fraction"],
                "coverage_label": "identity and receipt/QC records only; not collection date, depth, environment, or flux",
            },
            {
                "evidence_layer": "Sequence-to-exact JGI Data Portal record pair",
                "numerator": summary["jgi_data_portal_exact_identity_samples"],
                "denominator": summary["jgi_data_portal_crosswalk_rows"],
                "coverage_fraction": summary["jgi_data_portal_exact_identity_fraction"],
                "coverage_label": "independent public catalog identity and processing evidence; 27 controlled July-to-Jul aliases; no collection time, depth, chemistry, or flux join",
            },
            {
                "evidence_layer": "Supplemental-methods 2018 cohort context",
                "numerator": summary["methods_2018_cohort_samples"],
                "denominator": summary["methods_design_context_samples"],
                "coverage_fraction": summary["methods_2018_cohort_fraction"],
                "coverage_label": "SuF1 methods design; not a Table S4 sample/field-observation join",
            },
            {
                "evidence_layer": "Direct supplemental-methods depth-code context",
                "numerator": summary["methods_direct_depth_context_samples"],
                "denominator": summary["methods_design_context_samples"],
                "coverage_fraction": summary["methods_direct_depth_context_fraction"],
                "coverage_label": "91 coded 2018 rows only; July/September raw D6 versus published D5 remains unreconciled",
            },
            {
                "evidence_layer": "Valid ESS-DIVE chamber CH4 records",
                "numerator": summary["essdive_chamber_valid"],
                "denominator": summary["essdive_chamber_records"],
                "coverage_fraction": summary["essdive_chamber_valid_fraction"],
                "coverage_label": "reported-valid source records; none joined to sequencing samples",
            },
            {
                "evidence_layer": "Valid ESS-DIVE porewater CH4 records",
                "numerator": summary["essdive_porewater_valid"],
                "denominator": summary["essdive_porewater_records"],
                "coverage_fraction": summary["essdive_porewater_valid_fraction"],
                "coverage_label": "reported-valid source records; none joined to sequencing samples",
            },
            {
                "evidence_layer": "Valid ESS-DIVE gap-filled tower CH4 records",
                "numerator": summary["essdive_gapfilled_tower_valid"],
                "denominator": summary["essdive_gapfilled_tower_records"],
                "coverage_fraction": summary["essdive_gapfilled_tower_valid_fraction"],
                "coverage_label": "half-hourly site/time tower context; 2015-2016 overlaps MUCC, but no sample/plot/depth join exists",
            },
            {
                "evidence_layer": "FlashWeave edge selection stability",
                "numerator": summary["flashweave_stable_edges"],
                "denominator": summary["flashweave_mag_mag_edges"],
                "coverage_fraction": summary["flashweave_stable_edge_fraction"],
                "coverage_label": "20 scaffold-stratified subsamples; exploratory only",
            },
            {
                "evidence_layer": "Ocean-M-inspired stability and taxonomy filter",
                "numerator": summary[
                    "flashweave_explorer_stability_taxonomy_eligible_edges"
                ],
                "denominator": summary["flashweave_explorer_edge_rows"],
                "coverage_fraction": summary[
                    "flashweave_explorer_stability_taxonomy_eligible_fraction"
                ],
                "coverage_label": "conditional associations with >=0.70 selection frequency and atlas taxonomy at both endpoints; exploratory only",
            },
            {
                "evidence_layer": "Exact sample/environment/flux joins",
                "numerator": summary["exact_ecological_sample_joins"],
                "denominator": summary["ecological_readiness_sample_count"],
                "coverage_fraction": (
                    summary["exact_ecological_sample_joins"]
                    / summary["ecological_readiness_sample_count"]
                    if summary["ecological_readiness_sample_count"]
                    else 0.0
                ),
                "coverage_label": (
                    "authoritative crosswalk has validated exact sample/date/depth/assay/coverage/environment/flux/uncertainty fields; eligible for grouped ecological validation only"
                    if summary["exact_ecological_sample_joins"]
                    else "blocked pending authoritative date, depth, assay/coverage, environment, flux-window, and uncertainty crosswalk"
                ),
            },
        ]
        candidate_rows = rows(
            con,
            """
            SELECT
                mag_id,
                class,
                source_qc_label,
                CAST(processed_mag_expression_occupancy_fraction AS DOUBLE) AS expression_occupancy_fraction,
                CAST(review_priority_score AS DOUBLE) AS review_priority_score,
                next_validation_action
            FROM candidate_mucc_v1_promoted_molecular_reference_cards
            ORDER BY CAST(review_priority_score AS DOUBLE) DESC, mag_id
            LIMIT 10
            """,
        )
        for index, candidate in enumerate(candidate_rows, start=1):
            candidate["rank"] = index
        gate_rows = rows(
            con,
            """
            SELECT gate, status, detail
            FROM mucc_v1_integrated_atlas_promotion_gates
            ORDER BY CASE status WHEN 'pass' THEN 1 WHEN 'partial' THEN 2 ELSE 3 END, gate
            """,
        )
        network_explorer_rows = rows(
            con,
            """
            SELECT
                source_mag_id,
                target_mag_id,
                source_atlas_phylum,
                source_atlas_class,
                target_atlas_phylum,
                target_atlas_class,
                association_sign,
                CAST(association_weight AS DOUBLE) AS association_weight,
                CAST(absolute_association_weight AS DOUBLE) AS absolute_association_weight,
                CAST(selection_frequency AS DOUBLE) AS selection_frequency,
                stability_class,
                network_explorer_visibility_status,
                CAST(source_marker_breadth_count AS INTEGER) AS source_marker_breadth_count,
                CAST(target_marker_breadth_count AS INTEGER) AS target_marker_breadth_count,
                CAST(source_methane_term_rows AS INTEGER) AS source_methane_term_rows,
                CAST(target_methane_term_rows AS INTEGER) AS target_methane_term_rows,
                endpoint_taxonomy_conflict_exposure
            FROM fact_mucc_v1_flashweave_edge_atlas_context
            ORDER BY
                CASE network_explorer_visibility_status
                    WHEN 'stability_and_taxonomy_filter_eligible' THEN 1
                    WHEN 'stability_eligible_taxonomy_incomplete' THEN 2
                    ELSE 3
                END,
                CAST(absolute_association_weight AS DOUBLE) DESC,
                source_mag_id,
                target_mag_id
            LIMIT 100
            """,
        )
        wgcna_module_rows = rows(
            con,
            """
            SELECT
                module,
                CAST(mag_count AS INTEGER) AS mag_count,
                module_assignment_status,
                method_role
            FROM feature_mucc_v1_wgcna_secondary_module_summary
            ORDER BY CASE WHEN module = 'grey' THEN 2 ELSE 1 END,
                CAST(mag_count AS INTEGER) DESC,
                module
            """,
        )
    finally:
        con.close()

    audit_rows = [
        {
            "requirement": row["requirement"],
            "status": row["status"],
            "blocker": row["blocker"],
            "next_action": row["next_action"],
        }
        for row in read_tsv(audit_path)
        if row["status"]
        not in {"complete", "complete_source_staged", "complete_exploratory_only"}
    ]
    data_quality_rows = [
        {
            "check": "Archive roster key uniqueness",
            "result": "pass" if summary["mag_archive_duplicate_ids"] == 0 else "fail",
            "observed": summary["mag_archive_duplicate_ids"],
            "expected": 0,
            "impact": "Duplicate proteome IDs would invalidate MAG-level denominators.",
        },
        {
            "check": "BioProject crosswalk sample key uniqueness",
            "result": "pass"
            if summary["bioproject_crosswalk_duplicate_samples"] == 0
            else "fail",
            "observed": summary["bioproject_crosswalk_duplicate_samples"],
            "expected": 0,
            "impact": "Duplicate sample keys would make source-project coverage unreliable.",
        },
        {
            "check": "JGI Sample QC crosswalk sample key uniqueness",
            "result": "pass"
            if summary["jgi_sample_crosswalk_duplicate_samples"] == 0
            else "fail",
            "observed": summary["jgi_sample_crosswalk_duplicate_samples"],
            "expected": 0,
            "impact": "Duplicate sample keys would make JGI identity coverage unreliable.",
        },
        {
            "check": "JGI Data Portal crosswalk sample key uniqueness",
            "result": (
                "pass"
                if summary["jgi_data_portal_crosswalk_duplicate_samples"] == 0
                else "fail"
            ),
            "observed": summary["jgi_data_portal_crosswalk_duplicate_samples"],
            "expected": 0,
            "impact": "Duplicate sample keys would make JGI catalog identity coverage unreliable.",
        },
        {
            "check": "NCBI SRA crosswalk sample key uniqueness",
            "result": "pass"
            if summary["sra_sample_crosswalk_duplicate_samples"] == 0
            else "fail",
            "observed": summary["sra_sample_crosswalk_duplicate_samples"],
            "expected": 0,
            "impact": "Duplicate sample keys would make the SRA package and date crosswalk ambiguous.",
        },
        {
            "check": "NCBI SRA declared WGS packages requiring assay reconciliation",
            "result": "requires_reconciliation",
            "observed": summary["sra_declared_wgs_samples"],
            "expected": 0,
            "impact": "Declared WGS packages cannot be silently pooled with declared RNA-Seq packages for expression or ecological validation.",
        },
        {
            "check": "Published versus archive MAG denominator",
            "result": (
                "pass_exact_per_MAG_QC_reconciliation"
                if summary["published_hqmq_qc_reconciled_mag_count"]
                == summary["published_hqmq_headline"]
                and summary["zenodo_source_qc_consistent_rows"]
                == summary["archive_mag_roster"]
                else "fail_or_incomplete_QC_reconciliation"
            ),
            "observed": summary["archive_mag_outside_published_hqmq_qc_scope"],
            "expected": summary["archive_mag_roster"]
            - summary["published_hqmq_headline"],
            "impact": "Direct record-specific Zenodo QC reconciles all 2,508 archive MAGs: 2,502 meet the paper-defined HQ/MQ threshold and six remain explicit outside it.",
        },
        {
            "check": "Public KBase GenomeSet versus Zenodo archive",
            "result": "documented_subset_not_quality_assignment",
            "observed": summary["zenodo_archive_mag_absent_from_kbase"],
            "expected": 0,
            "impact": "Fourteen archive MAGs are absent from public KBase; KBase presence/absence cannot assign the paper's HQ/MQ subset.",
        },
        {
            "check": "Atlas taxonomy projection coverage",
            "result": "partial_conflict_preserving_projection",
            "observed": summary["atlas_taxonomy_unavailable_both_sources"],
            "expected": 0,
            "impact": "Five MAGs remain without taxonomy in both source and KBase; all available source/KBase rank differences remain explicit rather than silently overwritten.",
        },
        {
            "check": "Supplemental-methods D6 versus published D5 reconciliation",
            "result": "partial",
            "observed": summary["methods_d6_reconciliation_pending_samples"],
            "expected": 0,
            "impact": "The 18 July/September raw D6 labels cannot become exact depth or flux joins until Table S4 or another authoritative row-level source reconciles the notation.",
        },
        {
            "check": "Exact sequence-to-ecology joins",
            "result": "blocked",
            "observed": summary["exact_ecological_sample_joins"],
            "expected": summary["ecological_readiness_sample_count"],
            "impact": "No flux, depth, or environmental association may be interpreted at sample/MAG level.",
        },
    ]

    warehouse_rel = relative(repo_root, warehouse_path)
    audit_rel = relative(repo_root, audit_path)
    dashboard_sql = """
    WITH mag AS (
      SELECT COUNT(*) AS archive_mag_roster,
             COUNT(DISTINCT proteome_id) AS archive_mag_distinct
      FROM dim_mag
    ), expression AS (
      SELECT COUNT(*) AS expression_supported_mag_count
    FROM feature_mrv_readiness_mag_level
      WHERE processed_mag_expression_support = 'true'
    ), project AS (
      SELECT COUNT(*) AS crosswalk_rows,
             COUNT(DISTINCT sample_id) AS distinct_samples,
             COUNT(*) FILTER (WHERE sample_project_link_status = 'mapped_to_authoritative_NCBI_BioProject_title') AS mapped_samples
      FROM link_mucc_v1_sequence_bioproject_sample
    ), sra AS (
      SELECT COUNT(*) AS crosswalk_rows,
             COUNT(DISTINCT sample_id) AS distinct_samples,
             COUNT(*) FILTER (WHERE sra_sample_identity_status = 'exact_source_label_to_NCBI_SRA_package') AS exact_package_identity_samples,
             COUNT(*) FILTER (WHERE sra_collection_date_status = 'exact_collection_date_from_NCBI_SRA_sample_attributes') AS exact_collection_date_samples,
             COUNT(*) FILTER (WHERE sra_depth_cm_join_status = 'not_reported_by_NCBI_SRA_sample_attributes') AS depth_not_reported_samples,
             COUNT(*) FILTER (WHERE sra_library_strategy = 'RNA-Seq') AS declared_rna_seq_samples,
             COUNT(*) FILTER (WHERE sra_library_strategy = 'WGS') AS declared_wgs_samples
      FROM link_mucc_v1_sequence_sra_sample
    ), network AS (
      SELECT COUNT(*) AS mag_mag_edges,
             COUNT(*) FILTER (WHERE stability_class = 'stable_at_or_above_threshold') AS stable_edges,
             MAX(CAST(iterations AS INTEGER)) AS iterations
      FROM fact_mucc_v1_flashweave_edge_stability
    ), network_explorer AS (
      SELECT COUNT(*) AS explorer_edge_rows,
             COUNT(*) FILTER (WHERE network_explorer_visibility_status = 'stability_and_taxonomy_filter_eligible') AS explorer_stability_taxonomy_eligible_edges,
             COUNT(*) FILTER (WHERE endpoint_taxonomy_conflict_exposure = 'true') AS explorer_taxonomy_conflict_exposure_edges
      FROM fact_mucc_v1_flashweave_edge_atlas_context
    ), wgcna AS (
      SELECT COUNT(*) AS wgcna_module_memberships,
             COUNT(*) FILTER (WHERE module <> 'grey') AS wgcna_non_grey_modules,
             COALESCE(SUM(CAST(mag_count AS INTEGER)) FILTER (WHERE module = 'grey'), 0) AS wgcna_unassigned_grey_mag_count,
             (SELECT COUNT(*) FROM fact_mucc_v1_wgcna_secondary_module_eigengenes) AS wgcna_eigengene_samples
      FROM feature_mucc_v1_wgcna_secondary_module_summary
    ), methods_context AS (
      SELECT COUNT(*) AS methods_context_samples,
             COUNT(*) FILTER (WHERE methods_cohort = 'field_2018_metatranscriptome_design') AS methods_2018_cohort_samples,
             COUNT(*) FILTER (WHERE methods_design_context_status = 'validated_2018_methods_context_direct_depth_code') AS methods_direct_depth_context_samples,
             COUNT(*) FILTER (WHERE methods_design_context_status = 'validated_2018_cohort_but_raw_depth_code_reconciliation_pending') AS methods_d6_reconciliation_pending_samples
      FROM feature_mucc_v1_sample_methods_design_context
    ), kbase AS (
      SELECT
        COUNT(*) FILTER (WHERE kbase_roster_reconciliation_status = 'exact_MAG_id_match_public_KBase_GenomeSet') AS kbase_public_exact_mag_matches,
        COUNT(*) FILTER (WHERE kbase_roster_reconciliation_status = 'Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet') AS zenodo_archive_mag_absent_from_kbase,
        COUNT(*) FILTER (WHERE taxonomy_reconciliation_status = 'KBase_Gtdb_taxonomy_supplemental_only') AS kbase_supplemental_taxonomy_only,
        COUNT(*) FILTER (WHERE taxonomy_reconciliation_status = 'source_and_KBase_Gtdb_taxonomy_differ') AS kbase_source_taxonomy_differences
      FROM feature_mucc_v1_kbase_public_catalog_reconciliation
    ), taxonomy_projection AS (
      SELECT
        COUNT(*) FILTER (WHERE atlas_taxonomy_lineage <> '') AS atlas_taxonomy_available_mag_count,
        COUNT(*) FILTER (WHERE atlas_taxonomy_projection_status = 'taxonomy_unavailable_in_source_and_KBase') AS atlas_taxonomy_unavailable_both_sources,
        SUM(CAST(kbase_rank_fallback_count AS INTEGER)) AS atlas_taxonomy_kbase_rank_fallbacks,
        SUM(CAST(source_kbase_rank_disagreement_count AS INTEGER)) AS atlas_taxonomy_source_kbase_rank_disagreements
      FROM feature_mucc_v1_taxonomy_projection
    ), qc AS (
      SELECT
        COUNT(*) AS zenodo_source_qc_rows,
        COUNT(*) FILTER (WHERE source_qc_value_consistency_status = 'direct_source_qc_values_consistent_across_annotation_rows') AS zenodo_source_qc_consistent_rows,
        COUNT(*) FILTER (WHERE published_mq_hq_membership_status = 'meets_published_MQHQ_CheckM_threshold') AS published_hqmq_qc_reconciled_mag_count,
        COUNT(*) FILTER (WHERE published_mq_hq_membership_status = 'does_not_meet_published_MQHQ_CheckM_threshold') AS archive_mag_outside_published_hqmq_qc_scope
      FROM feature_mucc_v1_zenodo_source_qc
    )
    SELECT * FROM mag CROSS JOIN expression CROSS JOIN project CROSS JOIN sra CROSS JOIN network CROSS JOIN network_explorer CROSS JOIN wgcna CROSS JOIN methods_context CROSS JOIN kbase CROSS JOIN taxonomy_projection CROSS JOIN qc
    """.strip()
    coverage_sql = """
    SELECT
      'Checksum-validated MAG archive roster' AS evidence_layer,
      COUNT(DISTINCT proteome_id) AS numerator,
      COUNT(*) AS denominator
    FROM dim_mag
    UNION ALL
    SELECT 'Paper-defined HQ/MQ MAG quality roster',
      COUNT(*) FILTER (WHERE published_mq_hq_membership_status = 'meets_published_MQHQ_CheckM_threshold'), COUNT(*)
    FROM feature_mucc_v1_zenodo_source_qc
    UNION ALL
    SELECT 'Public KBase GenomeSet identity reconciliation',
      COUNT(*) FILTER (WHERE kbase_roster_reconciliation_status = 'exact_MAG_id_match_public_KBase_GenomeSet'), COUNT(*)
    FROM feature_mucc_v1_kbase_public_catalog_reconciliation
    UNION ALL
    SELECT 'Conflict-preserving atlas taxonomy projection',
      COUNT(*) FILTER (WHERE atlas_taxonomy_lineage <> ''), COUNT(*)
    FROM feature_mucc_v1_taxonomy_projection
    UNION ALL
    SELECT 'Processed MAG expression support', COUNT(*), (SELECT COUNT(*) FROM dim_mag)
    FROM feature_mrv_readiness_mag_level WHERE processed_mag_expression_support = 'true'
    UNION ALL
    SELECT 'Sequence-to-BioProject title crosswalk',
      COUNT(*) FILTER (WHERE sample_project_link_status = 'mapped_to_authoritative_NCBI_BioProject_title'), COUNT(*)
    FROM link_mucc_v1_sequence_bioproject_sample
    UNION ALL
    SELECT 'Sequence-to-exact NCBI SRA package',
      COUNT(*) FILTER (WHERE sra_sample_identity_status = 'exact_source_label_to_NCBI_SRA_package'), COUNT(*)
    FROM link_mucc_v1_sequence_sra_sample
    UNION ALL
    SELECT 'Exact NCBI SRA collection dates',
      COUNT(*) FILTER (WHERE sra_collection_date_status = 'exact_collection_date_from_NCBI_SRA_sample_attributes'), COUNT(*)
    FROM link_mucc_v1_sequence_sra_sample
    UNION ALL
    SELECT 'Sequence-to-exact JGI Sample QC identity',
      COUNT(*) FILTER (WHERE jgi_sample_identity_status = 'exact_source_label_to_JGI_Sample_QC_record'), COUNT(*)
    FROM link_mucc_v1_sequence_jgi_sample
    UNION ALL
    SELECT 'Sequence-to-exact JGI Data Portal record pair',
      COUNT(*) FILTER (WHERE jgi_data_portal_identity_status = 'exact_source_label_to_JGI_Data_Portal_record_pair'), COUNT(*)
    FROM link_mucc_v1_sequence_jgi_data_portal
    UNION ALL
    SELECT 'Supplemental-methods 2018 cohort context',
      COUNT(*) FILTER (WHERE methods_cohort = 'field_2018_metatranscriptome_design'), COUNT(*)
    FROM feature_mucc_v1_sample_methods_design_context
    UNION ALL
    SELECT 'Direct supplemental-methods depth-code context',
      COUNT(*) FILTER (WHERE methods_design_context_status = 'validated_2018_methods_context_direct_depth_code'), COUNT(*)
    FROM feature_mucc_v1_sample_methods_design_context
    UNION ALL
    SELECT 'FlashWeave edge selection stability',
      COUNT(*) FILTER (WHERE stability_class = 'stable_at_or_above_threshold'), COUNT(*)
    FROM fact_mucc_v1_flashweave_edge_stability
    UNION ALL
    SELECT 'Ocean-M-inspired stability and taxonomy filter',
      COUNT(*) FILTER (WHERE network_explorer_visibility_status = 'stability_and_taxonomy_filter_eligible'), COUNT(*)
    FROM fact_mucc_v1_flashweave_edge_atlas_context
    """.strip()
    candidate_sql = """
    SELECT mag_id, class, source_qc_label,
      CAST(processed_mag_expression_occupancy_fraction AS DOUBLE) AS expression_occupancy_fraction,
      CAST(review_priority_score AS DOUBLE) AS review_priority_score,
      next_validation_action
    FROM candidate_mucc_v1_promoted_molecular_reference_cards
    ORDER BY CAST(review_priority_score AS DOUBLE) DESC, mag_id
    LIMIT 10
    """.strip()
    network_explorer_sql = """
    SELECT source_mag_id, target_mag_id, source_atlas_phylum, source_atlas_class,
      target_atlas_phylum, target_atlas_class, association_sign,
      CAST(association_weight AS DOUBLE) AS association_weight,
      CAST(absolute_association_weight AS DOUBLE) AS absolute_association_weight,
      CAST(selection_frequency AS DOUBLE) AS selection_frequency, stability_class,
      network_explorer_visibility_status,
      CAST(source_marker_breadth_count AS INTEGER) AS source_marker_breadth_count,
      CAST(target_marker_breadth_count AS INTEGER) AS target_marker_breadth_count,
      CAST(source_methane_term_rows AS INTEGER) AS source_methane_term_rows,
      CAST(target_methane_term_rows AS INTEGER) AS target_methane_term_rows,
      endpoint_taxonomy_conflict_exposure
    FROM fact_mucc_v1_flashweave_edge_atlas_context
    ORDER BY CASE network_explorer_visibility_status
        WHEN 'stability_and_taxonomy_filter_eligible' THEN 1
        WHEN 'stability_eligible_taxonomy_incomplete' THEN 2
        ELSE 3 END,
      CAST(absolute_association_weight AS DOUBLE) DESC, source_mag_id, target_mag_id
    LIMIT 100
    """.strip()
    wgcna_module_sql = """
    SELECT
      module,
      CAST(mag_count AS INTEGER) AS mag_count,
      module_assignment_status,
      method_role,
      claim_boundary
    FROM feature_mucc_v1_wgcna_secondary_module_summary
    ORDER BY CASE WHEN module = 'grey' THEN 2 ELSE 1 END, CAST(mag_count AS INTEGER) DESC, module
    """.strip()
    gate_sql = """
    SELECT gate, status, detail
    FROM mucc_v1_integrated_atlas_promotion_gates
    ORDER BY CASE status WHEN 'pass' THEN 1 WHEN 'partial' THEN 2 ELSE 3 END, gate
    """.strip()
    audit_sql = (
        "SELECT requirement, status, blocker, next_action "
        f"FROM read_csv_auto('{audit_rel}', delim = '\\t', header = true) "
        "WHERE status NOT IN ('complete', 'complete_source_staged', 'complete_exploratory_only')"
    )
    sources = [
        make_source(
            "warehouse_summary",
            "MUCC v1 DuckDB warehouse — core reconciliation",
            warehouse_rel,
            dashboard_sql,
            "Computes dashboard headline measures from the promoted MUCC v1 warehouse.",
            [
                "dim_mag",
                "feature_mrv_readiness_mag_level",
                "feature_mucc_v1_kbase_public_catalog_reconciliation",
                "feature_mucc_v1_taxonomy_projection",
                "link_mucc_v1_sequence_bioproject_sample",
                "link_mucc_v1_sequence_sra_sample",
                "link_mucc_v1_sequence_jgi_sample",
                "link_mucc_v1_sequence_jgi_data_portal",
                "fact_mucc_v1_flashweave_edge_stability",
                "fact_mucc_v1_flashweave_edge_atlas_context",
                "feature_mucc_v1_wgcna_secondary_module_summary",
                "feature_mucc_v1_sample_methods_design_context",
                "fact_mucc_v1_essdive_gapfilled_tower_ch4_flux",
            ],
            generated_at,
            [
                "Archive MAG roster: rows in dim_mag; its canonical key is proteome_id.",
                "BioProject mapping: expression source columns matched to an authoritative NCBI BioProject title; not a run/BioSample, depth, date, or flux join.",
                "SRA package identity: entity-validated expression-label matches to NCBI SRA/BioSample/experiment/run records; exact collection dates are date-only and depth, environmental, and flux joins remain unresolved.",
                "JGI Data Portal record pairs: independent public catalog identity and processing provenance for 107 source labels, including 27 controlled July-to-Jul aliases; catalog evidence does not supply field depth, chemistry, or flux links.",
                "Declared SRA library strategies are preserved: WGS packages require assay reconciliation before any pooled-expression or ecological analysis.",
                "Exact ecological joins are counted only after the strict authoritative crosswalk contract validates sample identity, full collection time, depth, assay reconciliation, MAG abundance/read coverage, environment, flux window, replicate, and uncertainty identifiers.",
                "Stable FlashWeave edges: baseline MAG-to-MAG edges selected in at least 70% of 20 scaffold-stratified 80% subsamples.",
                "Explorer eligibility: stable conditional associations with taxonomy available at both endpoints; it is a navigation filter, not an ecological validation result.",
                "WGCNA module assignments are a completed source-method-aligned secondary comparison on all 133 processed-expression samples and 1,948 MAGs; they never replace FlashWeave conditional edges.",
                "The publication reports 132 WGCNA samples after an unidentified outlier screen; retaining all 133 makes this a controlled comparator, not an exact reproduction or ecological validation.",
                "Supplemental-methods context: the intact SuF1 establishes a 2018 cohort design, not an exact sample/date/depth/environment/flux crosswalk.",
                "ESS-DIVE DOI 10.15485/2500238 supplies half-hourly gap-filled tower CH4 context. Its temporal overlap with MUCC does not establish a sequencing-sample, plot, or depth correspondence.",
            ],
        ),
        make_source(
            "evidence_coverage",
            "MUCC v1 warehouse — evidence coverage and stability",
            warehouse_rel,
            coverage_sql,
            "Reconciles numerators and denominators for displayed evidence coverage. Exact ecological joins remain zero unless a strict author/publisher-supplied crosswalk has validated every required field.",
            [
                "dim_mag",
                "feature_mucc_v1_kbase_public_catalog_reconciliation",
                "feature_mucc_v1_taxonomy_projection",
                "feature_mrv_readiness_mag_level",
                "link_mucc_v1_sequence_bioproject_sample",
                "link_mucc_v1_sequence_sra_sample",
                "link_mucc_v1_sequence_jgi_sample",
                "link_mucc_v1_sequence_jgi_data_portal",
                "fact_mucc_v1_flashweave_edge_stability",
                "fact_mucc_v1_flashweave_edge_atlas_context",
                "fact_mucc_v1_essdive_chamber_flux",
                "fact_mucc_v1_essdive_porewater_ch4",
                "fact_mucc_v1_essdive_gapfilled_tower_ch4_flux",
                "feature_mucc_v1_sample_ecological_readiness",
                "feature_mucc_v1_sample_methods_design_context",
            ],
            generated_at,
        ),
        make_source(
            "candidate_review_queue",
            "MUCC v1 warehouse — promoted molecular-reference candidates",
            warehouse_rel,
            candidate_sql,
            "Ranks the 100 reviewable molecular-reference cards by the existing priority score. The score is a review queue, not a risk, flux, or crediting score.",
            ["candidate_mucc_v1_promoted_molecular_reference_cards"],
            generated_at,
            [
                "Review priority score: existing molecular-reference triage score; do not treat as an ecological or MRV-risk prediction.",
                "Expression occupancy: fraction of 133 source expression samples with processed MAG expression support.",
            ],
        ),
        make_source(
            "conditional_network_explorer",
            "MUCC v1 warehouse — conditional association explorer",
            warehouse_rel,
            network_explorer_sql,
            "Returns the first 100 edge-table rows, prioritized by stability and taxonomy completeness. Association signs and marker counts support filtering only; they do not establish interactions, mechanisms, ecology, or flux effects.",
            [
                "fact_mucc_v1_flashweave_edge_atlas_context",
                "feature_mucc_v1_flashweave_node_atlas_context",
                "feature_mucc_v1_taxonomy_projection",
            ],
            generated_at,
            [
                "Stable and taxonomy filter eligible: selection frequency at least 0.70 in the existing 20 scaffold-stratified subsamples, with rank-aware atlas taxonomy at both endpoints.",
                "Taxonomy conflict exposure: at least one endpoint has an explicit source/KBase rank disagreement; inspect rank provenance before taxonomic stratification.",
                "Marker term counts summarize source annotation context and do not demonstrate activity, a causal interaction, or a methane-flux mechanism.",
            ],
        ),
        make_source(
            "wgcna_secondary_modules",
            "MUCC v1 warehouse — secondary WGCNA module summary",
            warehouse_rel,
            wgcna_module_sql,
            "Returns the completed descriptive WGCNA module sizes for comparison with, never replacement of, the primary FlashWeave conditional association analysis.",
            [
                "feature_mucc_v1_wgcna_secondary_module_membership",
                "feature_mucc_v1_wgcna_secondary_module_summary",
                "fact_mucc_v1_wgcna_secondary_module_eigengenes",
            ],
            generated_at,
            [
                "Module membership: WGCNA 1.74 source-method-aligned signed-hybrid modules (power 7, minimum module size 50, merge height 0.3) from log1p CPM transformed processed relative expression, on 133 samples and 1,948 MAGs.",
                "The source analysis excluded one unidentified outlier to reach 132 samples; this current comparator retains all 133 and cannot reproduce source trait correlations without the authoritative Table S4/Table S12 data.",
                "Grey is unassigned. Module counts are descriptive coexpression structure only and do not establish direct interaction, ecology, or methane flux.",
            ],
        ),
        make_source(
            "promotion_gates",
            "MUCC v1 warehouse — promotion gates",
            warehouse_rel,
            gate_sql,
            "Lists the current molecular-reference promotion gates and their explicit statuses.",
            ["mucc_v1_integrated_atlas_promotion_gates"],
            generated_at,
        ),
        make_source(
            "integration_audit",
            "MUCC v1 integration audit — unresolved requirements",
            audit_rel,
            audit_sql,
            "Reads incomplete or partial requirements from the current deterministic integration audit.",
            [audit_rel],
            generated_at,
        ),
        {
            "id": "borton_2026",
            "label": "Borton et al. — Mapping the soil microbiome functions shaping wetland methane emissions",
            "href": "https://journals.asm.org/doi/10.1128/msystems.00680-25",
        },
        {
            "id": "zenodo_mucc_v1",
            "label": "Zenodo — MUCC v1 dataset archive",
            "href": "https://doi.org/10.5281/zenodo.8194033",
        },
        {
            "id": "essdive_owc",
            "label": "ESS-DIVE — Old Woman Creek methane and carbon dioxide observations",
            "href": "https://data.ess-dive.lbl.gov/view/doi:10.15485/1568865",
        },
        {
            "id": "essdive_owc_gapfilled_tower",
            "label": "ESS-DIVE — US-OWC gap-filled methane and carbon dioxide tower fluxes",
            "href": "https://doi.org/10.15485/2500238",
        },
        {
            "id": "jgi_data_portal_award_504205",
            "label": "JGI Data Portal — Old Woman Creek wetland-soils project catalog",
            "href": "https://genome.jgi.doe.gov/portal/Frogenwetlasoils/Frogenwetlasoils.info.html",
        },
        {
            "id": "flashweave_paper",
            "label": "FlashWeave method paper",
            "href": "https://doi.org/10.1016/j.cels.2019.08.002",
        },
        {
            "id": "ocean_m_paper",
            "label": "Ocean-M database — visualization and network-explorer inspiration",
            "href": "https://academic.oup.com/nar/article/54/D1/D813/8307366",
        },
    ]
    return {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "MUCC v1 Old Woman Creek wetland atlas",
            "description": "Integration coverage, provenance, and validation gates for the external wetland atlas lane.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "warehouse_tables",
                    "description": "Promoted source tables currently queryable in the MUCC v1 warehouse.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Warehouse tables",
                            "field": "warehouse_table_count",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "mag_roster",
                    "description": "Checksum-validated archive roster with direct record-specific Zenodo QC reconciliation to the paper-defined HQ/MQ subset.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Archive MAG roster",
                            "field": "archive_mag_roster",
                            "format": "number",
                        },
                        {
                            "label": "Published HQ/MQ headline",
                            "field": "published_hqmq_headline",
                            "format": "number",
                        },
                        {
                            "label": "HQ/MQ MAGs reconciled",
                            "field": "published_hqmq_qc_reconciled_mag_count",
                            "format": "number",
                        },
                        {
                            "label": "Archive MAGs outside HQ/MQ scope",
                            "field": "archive_mag_outside_published_hqmq_qc_scope",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "kbase_reconciliation",
                    "description": "Public KBase Workspace identity and supplementary-taxonomy coverage. KBase membership is not a quality tier and source/KBase taxonomy differences remain explicit.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Exact KBase MAG-ID matches",
                            "field": "kbase_public_exact_mag_matches",
                            "format": "number",
                        },
                        {
                            "label": "Zenodo MAGs absent from KBase",
                            "field": "zenodo_archive_mag_absent_from_kbase",
                            "format": "number",
                        },
                        {
                            "label": "KBase taxonomy-only MAGs",
                            "field": "kbase_supplemental_taxonomy_only",
                            "format": "number",
                        },
                        {
                            "label": "Source/KBase taxonomy differences",
                            "field": "kbase_source_taxonomy_differences",
                            "format": "number",
                        },
                        {
                            "label": "Atlas taxonomy available",
                            "field": "atlas_taxonomy_available_mag_count",
                            "format": "number",
                        },
                        {
                            "label": "Unassigned in both sources",
                            "field": "atlas_taxonomy_unavailable_both_sources",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "bioproject_crosswalk",
                    "description": "Expression sample columns linked to authoritative BioProject titles; not a sample/date/depth/flux join.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "BioProject title links",
                            "field": "bioproject_mapped_samples",
                            "format": "number",
                        },
                        {
                            "label": "Coverage of expression sample columns",
                            "field": "bioproject_mapped_fraction",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "sra_package_crosswalk",
                    "description": "Entity-validated NCBI SRA package identities and collection-date attributes. Dates are date-only; no depth, environmental, or flux window is supplied.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Exact SRA packages",
                            "field": "sra_exact_package_identity_samples",
                            "format": "number",
                        },
                        {
                            "label": "Exact collection dates",
                            "field": "sra_exact_collection_date_samples",
                            "format": "number",
                        },
                        {
                            "label": "Depth not reported",
                            "field": "sra_depth_not_reported_samples",
                            "format": "number",
                        },
                        {
                            "label": "Declared WGS packages",
                            "field": "sra_declared_wgs_samples",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "jgi_data_portal_crosswalk",
                    "description": "Independent JGI catalog identity and processing provenance. Public catalog records do not provide collection time, depth, field chemistry, or a flux window; catalog assets marked PURGED require authenticated retrieval.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Exact JGI catalog record pairs",
                            "field": "jgi_data_portal_exact_identity_samples",
                            "format": "number",
                        },
                        {
                            "label": "Controlled July-to-Jul aliases",
                            "field": "jgi_data_portal_july_alias_samples",
                            "format": "number",
                        },
                        {
                            "label": "Indexed catalog assets",
                            "field": "jgi_data_portal_indexed_files",
                            "format": "number",
                        },
                        {
                            "label": "Assets marked PURGED",
                            "field": "jgi_data_portal_purged_indexed_files",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "supplemental_methods_context",
                    "description": "The intact supplemental methods PDF establishes cohort-level 2018 design context. It does not resolve the malformed Table S4 or authorize an environmental/flux join.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "2018 methods-context samples",
                            "field": "methods_2018_cohort_samples",
                            "format": "number",
                        },
                        {
                            "label": "Direct coded-depth context",
                            "field": "methods_direct_depth_context_samples",
                            "format": "number",
                        },
                        {
                            "label": "D6/D5 reconciliation pending",
                            "field": "methods_d6_reconciliation_pending_samples",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "flashweave_stability",
                    "description": "Edges recovered in at least 70% of 20 scaffold-stratified subsamples. This is reproducibility evidence, not ecological validation.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Stable FlashWeave edges",
                            "field": "flashweave_stable_edges",
                            "format": "number",
                        },
                        {
                            "label": "Selection-stable fraction",
                            "field": "flashweave_stable_edge_fraction",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "wgcna_secondary_modules",
                    "description": "Source-method-aligned secondary coexpression modules on the full processed-expression cohort. They complement but never replace FlashWeave conditional associations.",
                    "dataset": "summary",
                    "sourceId": "wgcna_secondary_modules",
                    "metrics": [
                        {
                            "label": "Module memberships",
                            "field": "wgcna_module_memberships",
                            "format": "number",
                        },
                        {
                            "label": "Non-grey modules",
                            "field": "wgcna_non_grey_modules",
                            "format": "number",
                        },
                        {
                            "label": "Unassigned grey MAGs",
                            "field": "wgcna_unassigned_grey_mag_count",
                            "format": "number",
                        },
                        {
                            "label": "Eigengene samples",
                            "field": "wgcna_eigengene_samples",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "conditional_network_explorer",
                    "description": "Ocean-M-inspired filtering surface for exploratory conditional associations. Taxonomy, marker context, prevalence, and stability are queryable; no ecological or flux conclusion is permitted.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Explorer association edges",
                            "field": "flashweave_explorer_edge_rows",
                            "format": "number",
                        },
                        {
                            "label": "Stable + taxonomy eligible",
                            "field": "flashweave_explorer_stability_taxonomy_eligible_edges",
                            "format": "number",
                        },
                        {
                            "label": "Taxonomy conflict exposure",
                            "field": "flashweave_explorer_taxonomy_conflict_exposure_edges",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "ecological_join_guardrail",
                    "description": "Exact sequencing-to-environmental-to-flux joins are a hard prerequisite for ecological interpretation; an eligible row requires the strict authoritative crosswalk and is still not an MRV result.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Exact ecological joins",
                            "field": "exact_ecological_sample_joins",
                            "format": "number",
                        },
                        {
                            "label": "Expression sample columns",
                            "field": "ecological_readiness_sample_count",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "essdive_gapfilled_tower_context",
                    "description": "Half-hourly gap-filled eddy-covariance CH4 context from the US-OWC tower. The 2015-2016 period overlaps MUCC, but this is not a sequencing-sample, plot, or depth join.",
                    "dataset": "summary",
                    "sourceId": "warehouse_summary",
                    "metrics": [
                        {
                            "label": "Tower-flux source rows",
                            "field": "essdive_gapfilled_tower_records",
                            "format": "number",
                        },
                        {
                            "label": "Valid tower CH4 rows",
                            "field": "essdive_gapfilled_tower_valid",
                            "format": "number",
                        },
                    ],
                },
            ],
            "charts": [
                {
                    "id": "evidence_coverage_chart",
                    "title": "Evidence coverage by atlas layer",
                    "subtitle": "Percent uses the layer-specific denominator shown in the source table. Ecological eligibility requires a strict authoritative crosswalk and never by itself supports flux or MRV claims.",
                    "type": "bar",
                    "dataset": "coverage",
                    "sourceId": "evidence_coverage",
                    "encodings": {
                        "x": {
                            "field": "evidence_layer",
                            "type": "nominal",
                            "label": "Evidence layer",
                        },
                        "y": {
                            "field": "coverage_fraction",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Coverage",
                        },
                        "tooltip": [
                            {
                                "field": "numerator",
                                "type": "quantitative",
                                "label": "Numerator",
                            },
                            {
                                "field": "denominator",
                                "type": "quantitative",
                                "label": "Denominator",
                            },
                            {
                                "field": "coverage_label",
                                "type": "text",
                                "label": "Definition",
                            },
                        ],
                    },
                    "layout": "full",
                },
                {
                    "id": "candidate_priority_chart",
                    "title": "Top molecular-reference review candidates",
                    "subtitle": "Existing review-priority score ranks evidence review work only; it is not an ecological, methane, or MRV-risk score.",
                    "type": "bar",
                    "dataset": "candidate_review_queue",
                    "sourceId": "candidate_review_queue",
                    "encodings": {
                        "x": {"field": "mag_id", "type": "nominal", "label": "MAG"},
                        "y": {
                            "field": "review_priority_score",
                            "type": "quantitative",
                            "label": "Review priority score",
                        },
                        "tooltip": [
                            {"field": "class", "type": "text", "label": "Class"},
                            {
                                "field": "source_qc_label",
                                "type": "text",
                                "label": "Source QC",
                            },
                            {
                                "field": "expression_occupancy_fraction",
                                "type": "quantitative",
                                "format": "percent",
                                "label": "Expression occupancy",
                            },
                        ],
                    },
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "candidate_review_table",
                    "title": "Candidate review queue",
                    "subtitle": "Top 10 of 100 promoted molecular-reference cards; retain source QC and required next validation action in review.",
                    "dataset": "candidate_review_queue",
                    "sourceId": "candidate_review_queue",
                    "defaultSort": {
                        "field": "review_priority_score",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "rank", "label": "Rank", "format": "number"},
                        {"field": "mag_id", "label": "MAG"},
                        {"field": "class", "label": "Class"},
                        {"field": "source_qc_label", "label": "Source QC"},
                        {
                            "field": "expression_occupancy_fraction",
                            "label": "Expression occupancy",
                            "format": "percent",
                        },
                        {
                            "field": "review_priority_score",
                            "label": "Review priority",
                            "format": "number",
                        },
                        {
                            "field": "next_validation_action",
                            "label": "Required next validation",
                        },
                    ],
                    "layout": "full",
                },
                {
                    "id": "conditional_network_explorer_table",
                    "title": "Ocean-M-inspired conditional network explorer",
                    "subtitle": "Top 100 of 694 conditional MAG associations, prioritized by stability and endpoint taxonomy. Filter markers and taxonomy in the queryable warehouse; this table is exploratory, not ecological validation.",
                    "dataset": "conditional_network_explorer_edges",
                    "sourceId": "conditional_network_explorer",
                    "defaultSort": {
                        "field": "absolute_association_weight",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "source_mag_id", "label": "Source MAG"},
                        {"field": "target_mag_id", "label": "Target MAG"},
                        {"field": "source_atlas_phylum", "label": "Source phylum"},
                        {"field": "target_atlas_phylum", "label": "Target phylum"},
                        {"field": "association_sign", "label": "Sign"},
                        {
                            "field": "absolute_association_weight",
                            "label": "|Association|",
                            "format": "number",
                        },
                        {
                            "field": "selection_frequency",
                            "label": "Selection frequency",
                            "format": "percent",
                        },
                        {
                            "field": "network_explorer_visibility_status",
                            "label": "Explorer status",
                        },
                        {
                            "field": "source_marker_breadth_count",
                            "label": "Source marker breadth",
                            "format": "number",
                        },
                        {
                            "field": "target_marker_breadth_count",
                            "label": "Target marker breadth",
                            "format": "number",
                        },
                        {
                            "field": "endpoint_taxonomy_conflict_exposure",
                            "label": "Taxonomy conflict exposure",
                        },
                    ],
                    "layout": "full",
                },
                {
                    "id": "wgcna_secondary_module_table",
                    "title": "Secondary WGCNA descriptive modules",
                    "subtitle": "WGCNA 1.74 signed-hybrid comparator from 133 samples and 1,948 MAGs, aligned to reported source parameters but retaining the unidentified source outlier. This is descriptive coexpression only, not an interaction or ecological result.",
                    "dataset": "wgcna_secondary_modules",
                    "sourceId": "wgcna_secondary_modules",
                    "defaultSort": {"field": "mag_count", "direction": "desc"},
                    "columns": [
                        {"field": "module", "label": "Module"},
                        {
                            "field": "mag_count",
                            "label": "MAG memberships",
                            "format": "number",
                        },
                        {
                            "field": "module_assignment_status",
                            "label": "Assignment status",
                        },
                        {"field": "method_role", "label": "Method role"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "incomplete_requirements_table",
                    "title": "Ecological validation and source-recovery blockers",
                    "subtitle": "Items that remain incomplete or partial in the deterministic integration audit.",
                    "dataset": "incomplete_requirements",
                    "sourceId": "integration_audit",
                    "defaultSort": {"field": "requirement", "direction": "asc"},
                    "columns": [
                        {"field": "requirement", "label": "Requirement"},
                        {"field": "status", "label": "Status"},
                        {"field": "blocker", "label": "Blocker"},
                        {"field": "next_action", "label": "Next action"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "data_quality_table",
                    "title": "Data quality and reconciliation checks",
                    "subtitle": "Key uniqueness and denominator checks performed on the warehouse snapshot.",
                    "dataset": "data_quality",
                    "sourceId": "warehouse_summary",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "columns": [
                        {"field": "check", "label": "Check"},
                        {"field": "result", "label": "Result"},
                        {"field": "observed", "label": "Observed", "format": "number"},
                        {"field": "expected", "label": "Expected", "format": "number"},
                        {"field": "impact", "label": "Interpretation"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "promotion_gates_table",
                    "title": "Promotion gate ledger",
                    "subtitle": "Current gate states for the promoted molecular-reference lane.",
                    "dataset": "promotion_gates",
                    "sourceId": "promotion_gates",
                    "defaultSort": {"field": "gate", "direction": "asc"},
                    "columns": [
                        {"field": "gate", "label": "Gate"},
                        {"field": "status", "label": "Status"},
                        {"field": "detail", "label": "Detail"},
                    ],
                    "layout": "full",
                },
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# MUCC v1 Old Woman Creek wetland atlas\n\nSource-backed integration coverage, evidence provenance, and ecological-validation gates for the external wetland lane.",
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "warehouse_tables",
                        "mag_roster",
                        "bioproject_crosswalk",
                        "sra_package_crosswalk",
                        "jgi_data_portal_crosswalk",
                        "flashweave_stability",
                        "wgcna_secondary_modules",
                        "conditional_network_explorer",
                        "essdive_gapfilled_tower_context",
                        "ecological_join_guardrail",
                    ],
                },
                {
                    "id": "coverage_chart",
                    "type": "chart",
                    "chartId": "evidence_coverage_chart",
                },
                {
                    "id": "candidate_chart",
                    "type": "chart",
                    "chartId": "candidate_priority_chart",
                },
                {
                    "id": "candidate_table",
                    "type": "table",
                    "tableId": "candidate_review_table",
                },
                {
                    "id": "conditional_network_table",
                    "type": "table",
                    "tableId": "conditional_network_explorer_table",
                },
                {
                    "id": "wgcna_module_table",
                    "type": "table",
                    "tableId": "wgcna_secondary_module_table",
                },
                {
                    "id": "gaps_table",
                    "type": "table",
                    "tableId": "incomplete_requirements_table",
                },
                {
                    "id": "quality_table",
                    "type": "table",
                    "tableId": "data_quality_table",
                },
                {
                    "id": "gates_table",
                    "type": "table",
                    "tableId": "promotion_gates_table",
                },
                {
                    "id": "claim_boundary",
                    "type": "markdown",
                    "body": f"## Claim boundary\n\n{CLAIM_BOUNDARY}",
                },
                {
                    "id": "source_notes",
                    "type": "markdown",
                    "body": "## Primary source record\n\nThe warehouse carries direct source links for the mSystems study, its Zenodo archive, JGI project catalog, ESS-DIVE chamber/porewater observations, ESS-DIVE gap-filled tower fluxes, and the FlashWeave method. The dashboard is a snapshot generated from the local DuckDB warehouse, not a live connection.",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "partial",
            "accessIssues": [
                {
                    "id": "exact_sample_ecology_crosswalk_unavailable",
                    "scope": "ecological_validation",
                    "sourceId": "integration_audit",
                    "dataset": "incomplete_requirements",
                    "message": "130 expression labels have entity-validated SRA packages and 107 have exact collection dates, but no record provides an authoritative depth, environmental, or flux-window crosswalk. ESS-DIVE chamber, porewater, and 2015-2016-overlapping gap-filled tower data remain unlinked to sequencing samples.",
                    "actionLabel": "Review integration audit",
                },
            ],
            "datasets": {
                "summary": [summary],
                "coverage": coverage,
                "candidate_review_queue": candidate_rows,
                "conditional_network_explorer_edges": network_explorer_rows,
                "wgcna_secondary_modules": wgcna_module_rows,
                "promotion_gates": gate_rows,
                "incomplete_requirements": audit_rows,
                "data_quality": data_quality_rows,
            },
        },
        "sources": sources,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    output_path = resolve(repo_root, args.output)
    artifact = build_artifact(repo_root, run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "lane_id": LANE_ID,
                "snapshot_status": artifact["snapshot"]["status"],
                "warehouse_tables": artifact["snapshot"]["datasets"]["summary"][0][
                    "warehouse_table_count"
                ],
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
