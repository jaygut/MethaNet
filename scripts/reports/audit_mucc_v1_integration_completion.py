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
    return sum(
        1
        for row in read_tsv(path)
        if str(row.get(column, "")).lower() in {"true", "1", "yes", "y"}
    )


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
    return sum(
        1 for item in path.glob(pattern) if item.is_file() and item.stat().st_size > 0
    )


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
    prodigal_faa = count_nonempty_files(
        repo_root / BASE / "prodigal/proteomes_faa", "*.faa"
    )
    prodigal_gff = count_nonempty_files(
        repo_root / BASE / "prodigal/genes_gff", "*.gff"
    )
    prodigal_ffn = count_nonempty_files(
        repo_root / BASE / "prodigal/genes_ffn", "*.ffn"
    )
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
            str(
                BASE / "manifests/mucc_v1_glm2_ready_gap_register.partial_file_scan.tsv"
            ),
        ],
    )
    glm2_ready = count_truthy(repo_root, glm2_manifest_rel, "glm2_include")
    glm2_blocked = count_rows(repo_root, glm2_gap_rel)
    glm2_context_rel = "results/contextual_genomics/mucc_v1_owc_glm2_multiwindow_20260713/validation/glm2_multiwindow_reduce_report.json"
    glm2_context_path = repo_root / glm2_context_rel
    glm2_context_complete = False
    if glm2_context_path.is_file():
        try:
            glm2_context_complete = (
                json.loads(glm2_context_path.read_text()).get("status") == "pass"
            )
        except json.JSONDecodeError:
            glm2_context_complete = False
    mrv_readiness_rows = count_rows(
        repo_root,
        str(BASE / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv"),
    )
    mrv_readiness_candidate_rows = count_rows(
        repo_root,
        str(BASE / "candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv"),
    )
    warehouse_dim_mag_rows = 0
    warehouse_table_rows = 0
    warehouse_manifest = repo_root / BASE / "cohort_warehouse/cohort_table_manifest.tsv"
    if warehouse_manifest.is_file():
        warehouse_rows = read_tsv(warehouse_manifest)
        warehouse_table_rows = len(warehouse_rows)
        for row in warehouse_rows:
            if row.get("table") == "dim_mag":
                warehouse_dim_mag_rows = int(float(row.get("rows") or 0))
                break
    stop_condition_rel = str(
        BASE / "reports/mucc_v1_stop_condition_external_compute_blockers_20260626.tsv"
    )
    stop_condition_rows = count_rows(repo_root, stop_condition_rel)
    esm2_complete = all_ready(repo_root, esm2_dirs)
    legacy_neighbor_rows = count_rows(
        repo_root,
        str(
            BASE
            / "bridge_reanchoring/legacy_poc_mucc_neighbor_validation/wetland_reference_neighbor_summary.tsv"
        ),
    )
    final_neighbor_rel = str(
        BASE
        / "bridge_reanchoring/integrated_atlas/wetland_reference_neighbor_summary.tsv"
    )
    final_neighbor_rows = count_rows(repo_root, final_neighbor_rel)
    promoted_card_rel = str(
        BASE / "candidate_cards/mucc_v1_promoted_molecular_reference_cards.tsv"
    )
    promoted_card_rows = count_rows(repo_root, promoted_card_rel)
    promotion_summary_rel = str(BASE / "reports/mucc_v1_integrated_atlas_summary.json")
    network_status_rel = str(BASE / "network_analysis/network_analysis_status.tsv")
    network_status_rows = count_rows(repo_root, network_status_rel)
    flashweave_edge_rel = str(
        BASE / "network_analysis/fact_mucc_v1_flashweave_edges.tsv"
    )
    flashweave_node_rel = str(
        BASE / "network_analysis/feature_mucc_v1_flashweave_node_summary.tsv"
    )
    flashweave_gate_rel = str(
        BASE / "network_analysis/mucc_v1_flashweave_network_validation_gates.tsv"
    )
    flashweave_summary_rel = str(
        BASE / "network_analysis/mucc_v1_flashweave_network_summary.json"
    )
    explorer_edge_rel = str(
        BASE / "network_analysis/fact_mucc_v1_flashweave_edge_atlas_context.tsv"
    )
    explorer_node_rel = str(
        BASE / "network_analysis/feature_mucc_v1_flashweave_node_atlas_context.tsv"
    )
    explorer_summary_rel = str(
        BASE / "network_analysis/mucc_v1_flashweave_atlas_explorer_summary.tsv"
    )
    flashweave_edge_rows = count_rows(repo_root, flashweave_edge_rel)
    flashweave_node_rows = count_rows(repo_root, flashweave_node_rel)
    explorer_edge_rows = count_rows(repo_root, explorer_edge_rel)
    explorer_node_rows = count_rows(repo_root, explorer_node_rel)
    explorer_eligible_rows = (
        sum(
            1
            for row in read_tsv(repo_root / explorer_edge_rel)
            if row.get("network_explorer_visibility_status")
            == "stability_and_taxonomy_filter_eligible"
        )
        if exists_nonempty(repo_root, explorer_edge_rel)
        else 0
    )
    explorer_taxonomy_conflict_rows = count_truthy(
        repo_root, explorer_edge_rel, "endpoint_taxonomy_conflict_exposure"
    )
    flashweave_stability_status = "blocked"
    if exists_nonempty(repo_root, flashweave_gate_rel):
        for row in read_tsv(repo_root / flashweave_gate_rel):
            if row.get("gate") == "edge_stability":
                flashweave_stability_status = row.get("status", "blocked")
                break
    wgcna_summary_rel = str(
        BASE / "network_analysis/mucc_v1_wgcna_secondary_summary.json"
    )
    wgcna_status = "not_materialized"
    wgcna_samples = 0
    wgcna_features = 0
    wgcna_non_grey_modules = 0
    wgcna_soft_power = ""
    wgcna_soft_power_selection = ""
    wgcna_source_alignment = ""
    wgcna_outlier_status = ""
    wgcna_runtime = ""
    wgcna_path = repo_root / wgcna_summary_rel
    if wgcna_path.is_file() and wgcna_path.stat().st_size:
        try:
            wgcna = json.loads(wgcna_path.read_text())
            wgcna_status = str(wgcna.get("status", wgcna_status))
            wgcna_samples = int(wgcna.get("samples", 0))
            wgcna_features = int(wgcna.get("mag_features", 0))
            wgcna_non_grey_modules = int(wgcna.get("non_grey_module_count", 0))
            wgcna_soft_power = str(wgcna.get("soft_power", ""))
            wgcna_soft_power_selection = str(wgcna.get("soft_power_selection", ""))
            wgcna_source_alignment = str(
                wgcna.get("source_parameter_alignment_status", "")
            )
            wgcna_outlier_status = str(
                wgcna.get("source_outlier_reconciliation_status", "")
            )
            wgcna_runtime = f"R={wgcna.get('r_version', '')}; WGCNA={wgcna.get('wgcna_version', '')}"
        except (TypeError, ValueError, json.JSONDecodeError):
            wgcna_status = "malformed_summary"
    essdive_chamber_rel = str(
        BASE / "environmental_metadata/fact_mucc_v1_essdive_chamber_flux.tsv"
    )
    essdive_porewater_rel = str(
        BASE / "environmental_metadata/fact_mucc_v1_essdive_porewater_ch4.tsv"
    )
    essdive_gapfilled_tower_rel = str(
        BASE
        / "environmental_metadata/fact_mucc_v1_essdive_gapfilled_tower_ch4_flux.tsv"
    )
    essdive_chamber_rows = count_rows(repo_root, essdive_chamber_rel)
    essdive_porewater_rows = count_rows(repo_root, essdive_porewater_rel)
    essdive_gapfilled_tower_rows = count_rows(repo_root, essdive_gapfilled_tower_rel)
    essdive_gapfilled_tower_valid_rows = 0
    if exists_nonempty(repo_root, essdive_gapfilled_tower_rel):
        essdive_gapfilled_tower_valid_rows = sum(
            1
            for row in read_tsv(repo_root / essdive_gapfilled_tower_rel)
            if row.get("source_value_status") == "reported_valid"
        )
    bioproject_crosswalk_rel = str(
        BASE / "environmental_metadata/link_mucc_v1_sequence_bioproject_sample.tsv"
    )
    bioproject_crosswalk_rows = count_rows(repo_root, bioproject_crosswalk_rel)
    bioproject_mapped_rows = 0
    if exists_nonempty(repo_root, bioproject_crosswalk_rel):
        bioproject_mapped_rows = sum(
            1
            for row in read_tsv(repo_root / bioproject_crosswalk_rel)
            if row.get("sample_project_link_status")
            == "mapped_to_authoritative_NCBI_BioProject_title"
        )
    sra_crosswalk_rel = str(
        BASE / "environmental_metadata/link_mucc_v1_sequence_sra_sample.tsv"
    )
    sra_crosswalk_rows = count_rows(repo_root, sra_crosswalk_rel)
    sra_exact_identity_rows = 0
    sra_exact_collection_date_rows = 0
    sra_no_depth_rows = 0
    sra_rna_seq_rows = 0
    sra_wgs_rows = 0
    if exists_nonempty(repo_root, sra_crosswalk_rel):
        sra_crosswalk = read_tsv(repo_root / sra_crosswalk_rel)
        sra_exact_identity_rows = sum(
            1
            for row in sra_crosswalk
            if row.get("sra_sample_identity_status")
            == "exact_source_label_to_NCBI_SRA_package"
        )
        sra_exact_collection_date_rows = sum(
            1
            for row in sra_crosswalk
            if row.get("sra_collection_date_status")
            == "exact_collection_date_from_NCBI_SRA_sample_attributes"
        )
        sra_no_depth_rows = sum(
            1
            for row in sra_crosswalk
            if row.get("sra_depth_cm_join_status")
            == "not_reported_by_NCBI_SRA_sample_attributes"
        )
        sra_rna_seq_rows = sum(
            1 for row in sra_crosswalk if row.get("sra_library_strategy") == "RNA-Seq"
        )
        sra_wgs_rows = sum(
            1 for row in sra_crosswalk if row.get("sra_library_strategy") == "WGS"
        )
    jgi_crosswalk_rel = str(
        BASE / "environmental_metadata/link_mucc_v1_sequence_jgi_sample.tsv"
    )
    jgi_crosswalk_rows = count_rows(repo_root, jgi_crosswalk_rel)
    jgi_exact_identity_rows = 0
    jgi_final_portal_rows = 0
    if exists_nonempty(repo_root, jgi_crosswalk_rel):
        jgi_crosswalk = read_tsv(repo_root / jgi_crosswalk_rel)
        jgi_exact_identity_rows = sum(
            1
            for row in jgi_crosswalk
            if row.get("jgi_sample_identity_status")
            == "exact_source_label_to_JGI_Sample_QC_record"
        )
        jgi_final_portal_rows = sum(
            1 for row in jgi_crosswalk if row.get("jgi_final_deliverable_portal_id")
        )
    jgi_data_portal_rel = str(
        BASE / "environmental_metadata/link_mucc_v1_sequence_jgi_data_portal.tsv"
    )
    jgi_data_portal_rows = count_rows(repo_root, jgi_data_portal_rel)
    jgi_data_portal_exact_rows = 0
    jgi_data_portal_july_alias_rows = 0
    jgi_data_portal_no_depth_rows = 0
    if exists_nonempty(repo_root, jgi_data_portal_rel):
        jgi_data_portal = read_tsv(repo_root / jgi_data_portal_rel)
        jgi_data_portal_exact_rows = sum(
            1
            for row in jgi_data_portal
            if row.get("jgi_data_portal_identity_status")
            == "exact_source_label_to_JGI_Data_Portal_record_pair"
        )
        jgi_data_portal_july_alias_rows = sum(
            1
            for row in jgi_data_portal
            if row.get("jgi_data_portal_label_mapping_method")
            == "deterministic_July_to_Jul"
        )
        jgi_data_portal_no_depth_rows = sum(
            1
            for row in jgi_data_portal
            if row.get("depth_cm_join_status")
            == "not_reported_by_JGI_Data_Portal_catalog"
        )
    source_recovery_rel = str(
        BASE / "source_audit/mucc_v1_source_metadata_recovery_ledger.tsv"
    )
    table_s1_recovery_status = "not_recovery_audited"
    methods_pdf_recovery_status = "not_recovery_audited"
    kbase_public_catalog_status = "not_recovery_audited"
    zenodo_release_pin_status = "not_recovery_audited"
    if exists_nonempty(repo_root, source_recovery_rel):
        for row in read_tsv(repo_root / source_recovery_rel):
            if (
                row.get("artifact_role")
                == "published_Table_S1_to_S13_accession_spreadsheet"
            ):
                table_s1_recovery_status = row.get(
                    "availability_status", table_s1_recovery_status
                )
            if (
                row.get("artifact_role")
                == "published_Supplemental_Information_methods_PDF"
            ):
                methods_pdf_recovery_status = row.get(
                    "availability_status", methods_pdf_recovery_status
                )
            if row.get("artifact_role") == (
                "published_MAG_collection_public_roster_and_supplementary_taxonomy"
            ):
                kbase_public_catalog_status = row.get(
                    "availability_status", kbase_public_catalog_status
                )
            if row.get("artifact_role") == (
                "Zenodo_MUCC_v1_release_pin_and_concept_version_audit"
            ):
                zenodo_release_pin_status = row.get(
                    "availability_status", zenodo_release_pin_status
                )
    kbase_reconciliation_rel = str(
        BASE / "source_audit/kbase_public_workspace_147022/"
        "mucc_v1_kbase_public_catalog_reconciliation.tsv"
    )
    kbase_reconciliation_rows = count_rows(repo_root, kbase_reconciliation_rel)
    kbase_exact_mag_rows = 0
    kbase_archive_absent_rows = 0
    kbase_taxonomy_only_rows = 0
    if exists_nonempty(repo_root, kbase_reconciliation_rel):
        kbase_reconciliation = read_tsv(repo_root / kbase_reconciliation_rel)
        kbase_exact_mag_rows = sum(
            1
            for row in kbase_reconciliation
            if row.get("kbase_roster_reconciliation_status")
            == "exact_MAG_id_match_public_KBase_GenomeSet"
        )
        kbase_archive_absent_rows = sum(
            1
            for row in kbase_reconciliation
            if row.get("kbase_roster_reconciliation_status")
            == "Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet"
        )
        kbase_taxonomy_only_rows = sum(
            1
            for row in kbase_reconciliation
            if row.get("taxonomy_reconciliation_status")
            == "KBase_Gtdb_taxonomy_supplemental_only"
        )
    source_qc_rel = str(
        BASE / "functional_features/feature_mucc_v1_zenodo_source_qc.tsv"
    )
    source_qc_rows = count_rows(repo_root, source_qc_rel)
    source_qc_hqmq_rows = 0
    source_qc_archive_scope_rows = 0
    source_qc_consistent_rows = 0
    if exists_nonempty(repo_root, source_qc_rel):
        source_qc = read_tsv(repo_root / source_qc_rel)
        source_qc_hqmq_rows = sum(
            1
            for row in source_qc
            if row.get("published_mq_hq_membership_status")
            == "meets_published_MQHQ_CheckM_threshold"
        )
        source_qc_archive_scope_rows = sum(
            1
            for row in source_qc
            if row.get("published_mq_hq_membership_status")
            == "does_not_meet_published_MQHQ_CheckM_threshold"
        )
        source_qc_consistent_rows = sum(
            1
            for row in source_qc
            if row.get("source_qc_value_consistency_status")
            == "direct_source_qc_values_consistent_across_annotation_rows"
        )
    taxonomy_projection_rel = str(
        BASE / "functional_features/feature_mucc_v1_taxonomy_projection.tsv"
    )
    taxonomy_projection_rows = count_rows(repo_root, taxonomy_projection_rel)
    atlas_taxonomy_available_rows = 0
    atlas_taxonomy_unavailable_rows = 0
    if exists_nonempty(repo_root, taxonomy_projection_rel):
        taxonomy_projection = read_tsv(repo_root / taxonomy_projection_rel)
        atlas_taxonomy_available_rows = sum(
            1 for row in taxonomy_projection if row.get("atlas_taxonomy_lineage")
        )
        atlas_taxonomy_unavailable_rows = sum(
            1
            for row in taxonomy_projection
            if row.get("atlas_taxonomy_projection_status")
            == "taxonomy_unavailable_in_source_and_KBase"
        )
    methods_context_rel = str(
        BASE
        / "environmental_metadata/feature_mucc_v1_sample_methods_design_context.tsv"
    )
    methods_context_rows = count_rows(repo_root, methods_context_rel)
    methods_direct_depth_rows = 0
    methods_d6_pending_rows = 0
    methods_legacy_rows = 0
    if exists_nonempty(repo_root, methods_context_rel):
        methods_context = read_tsv(repo_root / methods_context_rel)
        methods_direct_depth_rows = sum(
            1
            for row in methods_context
            if row.get("methods_design_context_status")
            == "validated_2018_methods_context_direct_depth_code"
        )
        methods_d6_pending_rows = sum(
            1
            for row in methods_context
            if row.get("methods_design_context_status")
            == "validated_2018_cohort_but_raw_depth_code_reconciliation_pending"
        )
        methods_legacy_rows = sum(
            1
            for row in methods_context
            if row.get("methods_design_context_status")
            == "legacy_sample_label_context_only"
        )
    authoritative_ecology_link_rel = str(
        BASE / "environmental_metadata/link_mucc_v1_sequence_authoritative_ecology.tsv"
    )
    authoritative_ecology_readiness_rel = str(
        BASE
        / "environmental_metadata/feature_mucc_v1_authoritative_ecology_readiness.tsv"
    )
    authoritative_ecology_link_rows = count_rows(
        repo_root, authoritative_ecology_link_rel
    )
    authoritative_ecology_validated_rows = 0
    if exists_nonempty(repo_root, authoritative_ecology_link_rel):
        authoritative_ecology_validated_rows = sum(
            1
            for row in read_tsv(repo_root / authoritative_ecology_link_rel)
            if row.get("mapping_validation_status")
            == "validated_authoritative_sample_environment_flux_mapping"
        )
    authoritative_ecology_readiness_rows = count_rows(
        repo_root, authoritative_ecology_readiness_rel
    )
    authoritative_ecology_ready_samples = 0
    if exists_nonempty(repo_root, authoritative_ecology_readiness_rel):
        authoritative_ecology_ready_samples = sum(
            1
            for row in read_tsv(repo_root / authoritative_ecology_readiness_rel)
            if row.get("authoritative_ecology_readiness_status")
            == "ready_for_grouped_ecological_validation"
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
            f"{BASE / 'manifests/mucc_v1_source_lane_manifest.tsv'} rows={count_rows(repo_root, str(BASE / 'manifests/mucc_v1_source_lane_manifest.tsv'))}; warehouse tables={warehouse_table_rows}; dim_mag rows={warehouse_dim_mag_rows}",
        ),
        status_row(
            "MAG catalog",
            (
                "complete"
                if source_qc_rows == 2508
                and source_qc_hqmq_rows == 2502
                and source_qc_archive_scope_rows == 6
                and source_qc_consistent_rows == 2508
                else "incomplete"
            ),
            (
                f"{BASE / 'manifests/mucc_v1_mag_catalog_full.tsv'} rows="
                f"{count_rows(repo_root, str(BASE / 'manifests/mucc_v1_mag_catalog_full.tsv'))}; "
                f"Zenodo release pin={zenodo_release_pin_status}; direct source-QC rows={source_qc_rows}; "
                f"paper-defined HQ/MQ members={source_qc_hqmq_rows}; archive members outside HQ/MQ QC "
                f"scope={source_qc_archive_scope_rows}; repeated-QC-consistent rows={source_qc_consistent_rows}"
            ),
            "The 2,508-member archive remains broader than the paper-defined 2,502 HQ/MQ cohort; warehouse consumers must use published_mq_hq_membership_status rather than assuming all archive MAGs qualify.",
            "retain all archive members with explicit QC-scope status; exclude the six non-qualifying MAGs only when an analysis requires the paper-defined HQ/MQ denominator",
        ),
        status_row(
            "public KBase catalog reconciliation",
            (
                "partial"
                if kbase_public_catalog_status
                == "public_api_roster_reconciled_quality_scope_unresolved"
                and kbase_reconciliation_rows == 2508
                and kbase_exact_mag_rows == 2494
                and kbase_archive_absent_rows == 14
                else "incomplete"
            ),
            (
                f"{kbase_reconciliation_rel}; public API status={kbase_public_catalog_status}; "
                f"exact MAG-ID matches={kbase_exact_mag_rows}; archive MAGs absent from KBase="
                f"{kbase_archive_absent_rows}; KBase GTDB-only taxonomy values={kbase_taxonomy_only_rows}"
            ),
            "KBase metadata has no per-MAG completeness, contamination, CheckM, quality-tier, or N50 fields; the 2,494-member subset is not the published 2,502 HQ/MQ roster",
            "retain KBase as public identity/supplementary taxonomy evidence; use direct Zenodo QC for HQ/MQ membership and obtain original supplement/author data only for accession-level provenance gaps",
        ),
        status_row(
            "rank-aware atlas taxonomy projection",
            (
                "partial"
                if taxonomy_projection_rows == 2508
                and atlas_taxonomy_available_rows == 2503
                and atlas_taxonomy_unavailable_rows == 5
                else "incomplete"
            ),
            (
                f"{taxonomy_projection_rel}; rows={taxonomy_projection_rows}; "
                f"source-primary/KBase-fallback atlas taxonomy={atlas_taxonomy_available_rows}; "
                f"unavailable in both sources={atlas_taxonomy_unavailable_rows}"
            ),
            "KBase taxonomy is a supplementary 2024 GTDB annotation; source/KBase rank conflicts remain explicit and this does not resolve MAG quality or sample ecology",
            "use the projected taxonomy for MAG-level stratification only; obtain exact sample/environment/flux metadata before ecological or MRV use",
        ),
        status_row(
            "expression support tables",
            "complete",
            f"{BASE / 'expression/feature_mucc_v1_expression_mag_summary.tsv'} rows={count_rows(repo_root, str(BASE / 'expression/feature_mucc_v1_expression_mag_summary.tsv'))}; {BASE / 'expression/feature_mucc_v1_gene_expression_mag_summary.tsv'} rows={count_rows(repo_root, str(BASE / 'expression/feature_mucc_v1_gene_expression_mag_summary.tsv'))}",
        ),
        status_row(
            "functional feature tables",
            "partial",
            f"source annotation/status tables exist; warehouse dim_mag rows={warehouse_dim_mag_rows}; MRV readiness scaffold rows={mrv_readiness_rows}; Prodigal FAA/GFF/FFN={prodigal_faa}/{prodigal_gff}/{prodigal_ffn}; triplets={prodigal_triplets}; gLM2-ready rows={glm2_ready} blocked={glm2_blocked}; completed gLM2 context={glm2_context_complete}",
            "MethaNet-curated cross-lane mechanism facts are not yet materialized; source DRAM remains a scaffold",
            "run or map curated methane, sulfur, substrate, and coverage facts before mechanism-level promotion",
        ),
        status_row(
            "published accession/QC roster recovery",
            "complete"
            if table_s1_recovery_status == "parseable_xlsx_container"
            else "partial",
            (
                f"{source_recovery_rel}; published Table S1-S13 status={table_s1_recovery_status}; "
                f"direct Zenodo QC reconciles {source_qc_hqmq_rows}/2502 HQ/MQ MAGs with "
                f"{source_qc_archive_scope_rows} explicit archive-scope exceptions"
            ),
            "Europe PMC-retrieved inner XLSX is malformed, so the supplementary accession/bin-stat worksheet remains unavailable; the direct Zenodo annotation payload resolves QC membership but not every intended supplemental accession field.",
            "obtain a fresh publisher-original or author-provided roster only if accession-level provenance or the original Table S3 column set is required",
        ),
        status_row(
            "published supplemental methods design context",
            (
                "partial"
                if methods_pdf_recovery_status
                == "parseable_pdf_methods_design_confirmed"
                and methods_context_rows == 133
                and methods_direct_depth_rows == 91
                and methods_d6_pending_rows == 18
                and methods_legacy_rows == 24
                else "incomplete"
            ),
            (
                f"{methods_context_rel}; SuF1 status={methods_pdf_recovery_status}; rows={methods_context_rows}; "
                f"direct coded-depth context={methods_direct_depth_rows}; D6/D5 reconciliation pending={methods_d6_pending_rows}; "
                f"legacy label-only rows={methods_legacy_rows}"
            ),
            "the intact methods PDF establishes design-level cohort context only; it does not replace the malformed Table S4 sample/chemistry/flux workbook",
            "use Table S4 or another authoritative row-level source to reconcile D6/D5 and establish collection date, depth, environmental, and flux join keys",
        ),
        status_row(
            "NCBI sequence-project crosswalk",
            "partial" if bioproject_mapped_rows else "incomplete",
            f"{bioproject_crosswalk_rel}; mapped BioProject rows={bioproject_mapped_rows}/{bioproject_crosswalk_rows}",
            "a BioProject title match is not a run/BioSample accession, collection datetime, depth-in-cm, or flux join",
            "recover run/BioSample metadata and match the remaining expression labels before field-observation reconciliation",
        ),
        status_row(
            "NCBI SRA package and collection-date crosswalk",
            (
                "partial"
                if sra_crosswalk_rows == 133
                and sra_exact_identity_rows == 130
                and sra_exact_collection_date_rows == 107
                and sra_no_depth_rows == 130
                else "incomplete"
            ),
            (
                f"{sra_crosswalk_rel}; exact entity-validated packages={sra_exact_identity_rows}/"
                f"{sra_crosswalk_rows}; exact collection dates={sra_exact_collection_date_rows}; "
                f"depth not reported={sra_no_depth_rows}; declared RNA-Seq={sra_rna_seq_rows}; "
                f"declared WGS={sra_wgs_rows}; unresolved labels="
                f"{sra_crosswalk_rows - sra_exact_identity_rows}"
            ),
            "SRA package identity and dates do not establish depth, field chemistry, porewater/chamber-flux windows, abundance/read coverage, or assay equivalence; WGS-declared records cannot be silently pooled with RNA-Seq expression",
            "recover an authoritative depth/environment/flux mapping and reconcile assay provenance before ecological validation; retain three unresolved labels explicitly",
        ),
        status_row(
            "JGI Sample QC identity crosswalk",
            "partial" if jgi_final_portal_rows else "incomplete",
            (
                f"{jgi_crosswalk_rel}; final JGI delivery portal rows={jgi_final_portal_rows}/{jgi_crosswalk_rows}; "
                f"exact JGI Sample QC identity rows={jgi_exact_identity_rows}/{jgi_crosswalk_rows}"
            ),
            "JGI Sample QC receipt/QC timestamps establish operational intake/QC metadata only; they are not collection datetime, depth-in-cm, environmental, or flux evidence",
            "obtain collection date/depth and exact field-observation link keys from a published or authorized sample metadata source",
        ),
        status_row(
            "JGI Data Portal catalog identity crosswalk",
            (
                "partial"
                if jgi_data_portal_rows == 133
                and jgi_data_portal_exact_rows == 107
                and jgi_data_portal_july_alias_rows == 27
                and jgi_data_portal_no_depth_rows == 107
                else "incomplete"
            ),
            (
                f"{jgi_data_portal_rel}; exact JGI catalog record pairs="
                f"{jgi_data_portal_exact_rows}/{jgi_data_portal_rows}; controlled July-to-Jul "
                f"aliases={jgi_data_portal_july_alias_rows}; catalog rows with no reported depth="
                f"{jgi_data_portal_no_depth_rows}"
            ),
            "public catalog/processing and source-record geolocation evidence only; indexed assets may be purged and authenticated download is required, while collection time, depth, environmental, and flux fields remain absent",
            "obtain the authoritative Table S4-equivalent field metadata and reconcile its rows to the exact JGI/NCBI records before ecological validation",
        ),
        status_row(
            "ESS-DIVE source observations",
            "complete_source_staged"
            if essdive_chamber_rows == 275 and essdive_porewater_rows == 5280
            else "incomplete",
            f"chamber flux rows={essdive_chamber_rows}; porewater CH4 rows={essdive_porewater_rows}",
            "all observations remain explicitly unlinked to sequencing samples",
            "join only from an authoritative sample/date/depth and spatial-temporal mapping",
        ),
        status_row(
            "ESS-DIVE gap-filled tower flux source context",
            (
                "complete_source_context_unlinked"
                if essdive_gapfilled_tower_rows and essdive_gapfilled_tower_valid_rows
                else "incomplete"
            ),
            (
                f"{essdive_gapfilled_tower_rel}; half-hourly rows={essdive_gapfilled_tower_rows}; "
                f"valid CH4 rows={essdive_gapfilled_tower_valid_rows}; 2015-2016 overlaps the MUCC study period"
            ),
            "site/time tower context lacks a sequencing-sample, plot, depth, and documented tower-context mapping",
            "retain as unlinked site/time context; join only after an authoritative sequence sample/time/plot/depth mapping and a documented tower-context rule are available",
        ),
        status_row(
            "authoritative sequence-to-ecology crosswalk contract",
            (
                "complete"
                if authoritative_ecology_ready_samples == 133
                and authoritative_ecology_readiness_rows == 133
                else "partial"
                if authoritative_ecology_ready_samples
                else "incomplete"
            ),
            (
                f"{authoritative_ecology_link_rel}; rows={authoritative_ecology_link_rows}; "
                f"validated mappings={authoritative_ecology_validated_rows}; "
                f"{authoritative_ecology_readiness_rel}; readiness rows="
                f"{authoritative_ecology_readiness_rows}; samples eligible for grouped ecological validation="
                f"{authoritative_ecology_ready_samples}/133"
            ),
            "No author/publisher-supplied complete sequence-to-environment/flux mapping has been staged. "
            "The contract refuses inferred joins from labels, dates, sites, or depth codes.",
            "Request a corrected publisher/author Table S4-equivalent canonical TSV, run "
            "stage_mucc_v1_authoritative_ecological_crosswalk.py, then refresh promotion, warehouse, "
            "dashboard, and this audit.",
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
            "complete" if final_neighbor_rows == 2501 else "incomplete",
            f"legacy POC validation summary rows={legacy_neighbor_rows}; source-aware MUCC neighbor summary rows={final_neighbor_rows}; MUCC ESM2 shards complete={esm2_complete}",
            ""
            if final_neighbor_rows == 2501
            else "final source-aware neighbor table is absent or incomplete",
            "retain source-aware similarity wording and confirm legacy POC embedding checkpoint provenance before cross-run comparison claims",
        ),
        status_row(
            "candidate cards",
            "complete" if promoted_card_rows else "complete_scaffold_only",
            f"strategic review rows={count_rows(repo_root, str(BASE / 'candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv'))}; MRV scaffold cards={mrv_readiness_candidate_rows}; promoted molecular-reference cards={promoted_card_rows}",
            "cards remain review artifacts, not final mechanism, flux, or risk claims",
            "add curated mechanism evidence and ecological validation metadata before escalating any card",
        ),
        status_row(
            "validation gap register",
            "complete",
            f"{BASE / 'reports/validation_gap_register.tsv'} rows={count_rows(repo_root, str(BASE / 'reports/validation_gap_register.tsv'))}",
        ),
        status_row(
            "integration report",
            "complete"
            if exists_nonempty(repo_root, promotion_summary_rel)
            else "complete_current_snapshot",
            f"promotion summary={promotion_summary_rel}; network status rows={network_status_rows}",
            "ecological validation remains blocked pending exact sample/depth/environment/flux joins",
            "refresh after external measurement joins or network-stability validation",
        ),
        status_row(
            "FlashWeave conditional association network",
            "complete_exploratory_only"
            if flashweave_edge_rows == 694 and flashweave_node_rows == 300
            else "incomplete",
            f"MAG-to-MAG edges={flashweave_edge_rows}; node summaries={flashweave_node_rows}; summary={flashweave_summary_rel}",
            f"edge stability={flashweave_stability_status}; exact environmental and flux covariates are unresolved",
            "run bootstrap/grouped leave-out stability after authoritative sample/depth/environment/flux joins",
        ),
        status_row(
            "WGCNA secondary descriptive module comparison",
            (
                "complete_exploratory_only"
                if wgcna_status == "completed_secondary_descriptive_modules"
                and wgcna_samples == 133
                and wgcna_features == 1948
                and wgcna_non_grey_modules >= 1
                else "incomplete"
            ),
            (
                f"summary={wgcna_summary_rel}; status={wgcna_status}; {wgcna_runtime}; "
                f"samples={wgcna_samples}; features={wgcna_features}; non-grey modules="
                f"{wgcna_non_grey_modules}; softPower={wgcna_soft_power}; "
                f"selection={wgcna_soft_power_selection}; source_alignment={wgcna_source_alignment}; "
                f"outlier_status={wgcna_outlier_status}"
            ),
            "WGCNA modules are descriptive coexpression structure, without direct-association conditioning, exact environmental covariates, or flux linkage.",
            "keep FlashWeave primary; correlate eigengenes only after the authoritative ecological crosswalk and grouped validation are available.",
        ),
        status_row(
            "Ocean-M-inspired FlashWeave atlas explorer",
            (
                "complete_exploratory_only"
                if explorer_edge_rows == 694
                and explorer_node_rows == 300
                and explorer_eligible_rows == 126
                else "incomplete"
            ),
            (
                f"edge context={explorer_edge_rel} rows={explorer_edge_rows}; node context="
                f"{explorer_node_rel} rows={explorer_node_rows}; summary={explorer_summary_rel}; "
                f"stability-and-taxonomy eligible={explorer_eligible_rows}; taxonomy-conflict exposure="
                f"{explorer_taxonomy_conflict_rows}"
            ),
            "the explorer supplies MAG-level taxonomy/marker/stability context only; exact environmental and flux covariates remain unresolved",
            "use rank provenance when filtering and rerun prevalence, null/permutation, and grouped leave-out checks after authoritative ecological joins",
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
        "blocked",
        "incomplete",
        "complete_source_staged",
        "complete_source_context_unlinked",
        "complete_scaffold_only",
        "complete_current_snapshot",
        "complete_with_denominator_gap",
        "complete_archive_roster_with_published_quality_scope_gap",
        "complete_exploratory_only",
    }
    summary = {
        "output_tsv": str(args.output_tsv),
        "requirements": len(rows),
        "fully_complete": sum(
            1 for row in rows if row["status"] in fully_complete_statuses
        ),
        "partial_or_incomplete": sum(
            1 for row in rows if row["status"] in partial_statuses
        ),
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
        "glm2_context_complete": glm2_context_complete,
        "mrv_readiness_rows": mrv_readiness_rows,
        "mrv_readiness_candidate_rows": mrv_readiness_candidate_rows,
        "warehouse_dim_mag_rows": warehouse_dim_mag_rows,
        "warehouse_table_rows": warehouse_table_rows,
        "ESS_DIVE_chamber_flux_rows": essdive_chamber_rows,
        "ESS_DIVE_porewater_ch4_rows": essdive_porewater_rows,
        "ESS_DIVE_gapfilled_tower_flux_rows": essdive_gapfilled_tower_rows,
        "ESS_DIVE_gapfilled_tower_flux_valid_CH4_rows": essdive_gapfilled_tower_valid_rows,
        "NCBI_BioProject_crosswalk_rows": bioproject_crosswalk_rows,
        "NCBI_BioProject_mapped_rows": bioproject_mapped_rows,
        "NCBI_SRA_crosswalk_rows": sra_crosswalk_rows,
        "NCBI_SRA_exact_sample_identity_rows": sra_exact_identity_rows,
        "NCBI_SRA_exact_collection_date_rows": sra_exact_collection_date_rows,
        "NCBI_SRA_depth_not_reported_rows": sra_no_depth_rows,
        "NCBI_SRA_declared_RNA_Seq_rows": sra_rna_seq_rows,
        "NCBI_SRA_declared_WGS_rows": sra_wgs_rows,
        "JGI_Sample_QC_crosswalk_rows": jgi_crosswalk_rows,
        "JGI_final_delivery_portal_rows": jgi_final_portal_rows,
        "JGI_exact_sample_identity_rows": jgi_exact_identity_rows,
        "JGI_Data_Portal_crosswalk_rows": jgi_data_portal_rows,
        "JGI_Data_Portal_exact_record_pair_rows": jgi_data_portal_exact_rows,
        "JGI_Data_Portal_July_to_Jul_alias_rows": jgi_data_portal_july_alias_rows,
        "JGI_Data_Portal_depth_not_reported_rows": jgi_data_portal_no_depth_rows,
        "authoritative_ecology_crosswalk_link_rows": authoritative_ecology_link_rows,
        "authoritative_ecology_validated_mapping_rows": authoritative_ecology_validated_rows,
        "authoritative_ecology_readiness_rows": authoritative_ecology_readiness_rows,
        "authoritative_ecology_ready_samples": authoritative_ecology_ready_samples,
        "published_supplemental_methods_pdf_status": methods_pdf_recovery_status,
        "methods_design_context_rows": methods_context_rows,
        "methods_design_direct_depth_rows": methods_direct_depth_rows,
        "methods_design_D6_reconciliation_pending_rows": methods_d6_pending_rows,
        "methods_design_legacy_label_only_rows": methods_legacy_rows,
        "published_table_s1_recovery_status": table_s1_recovery_status,
        "Zenodo_release_pin_status": zenodo_release_pin_status,
        "KBase_public_catalog_status": kbase_public_catalog_status,
        "KBase_public_catalog_reconciliation_rows": kbase_reconciliation_rows,
        "KBase_public_exact_MAG_ID_rows": kbase_exact_mag_rows,
        "Zenodo_archive_MAGs_absent_from_KBase": kbase_archive_absent_rows,
        "KBase_Gtdb_taxonomy_only_rows": kbase_taxonomy_only_rows,
        "Zenodo_direct_source_QC_rows": source_qc_rows,
        "Zenodo_published_HQMQ_MAGs_reconciled": source_qc_hqmq_rows,
        "Zenodo_archive_MAGs_outside_HQMQ_QC_scope": source_qc_archive_scope_rows,
        "Zenodo_repeated_source_QC_consistent_rows": source_qc_consistent_rows,
        "taxonomy_projection_rows": taxonomy_projection_rows,
        "atlas_taxonomy_available_rows": atlas_taxonomy_available_rows,
        "atlas_taxonomy_unavailable_in_both_sources_rows": atlas_taxonomy_unavailable_rows,
        "stop_condition_blocker_rows": stop_condition_rows,
        "stop_condition_blocker_ledger": stop_condition_rel,
        "source_aware_neighbor_rows": final_neighbor_rows,
        "promoted_molecular_reference_card_rows": promoted_card_rows,
        "network_input_status_rows": network_status_rows,
        "flashweave_mag_mag_edge_rows": flashweave_edge_rows,
        "flashweave_node_summary_rows": flashweave_node_rows,
        "flashweave_edge_stability_status": flashweave_stability_status,
        "WGCNA_secondary_status": wgcna_status,
        "WGCNA_secondary_samples": wgcna_samples,
        "WGCNA_secondary_MAG_features": wgcna_features,
        "WGCNA_secondary_non_grey_modules": wgcna_non_grey_modules,
        "WGCNA_secondary_soft_power": wgcna_soft_power,
        "WGCNA_secondary_soft_power_selection": wgcna_soft_power_selection,
        "WGCNA_secondary_source_parameter_alignment": wgcna_source_alignment,
        "WGCNA_secondary_source_outlier_reconciliation_status": wgcna_outlier_status,
        "flashweave_atlas_explorer_edge_rows": explorer_edge_rows,
        "flashweave_atlas_explorer_node_rows": explorer_node_rows,
        "flashweave_atlas_explorer_stability_taxonomy_eligible_rows": explorer_eligible_rows,
        "flashweave_atlas_explorer_taxonomy_conflict_exposure_rows": explorer_taxonomy_conflict_rows,
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
