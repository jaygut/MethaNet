#!/usr/bin/env python3
"""Promote the MUCC v1 source scaffold into a governed atlas reference lane.

This is deliberately a *promotion* step, not a risk-score generator.  It
reconciles the published and local denominators, verifies completed ESM2/gLM2
artifacts, exposes their coverage as queryable features, connects the current
sample scaffold to explicit ecological-validation gates, and materializes
reviewable molecular-reference cards.  Missing ecological joins remain visible
as blocked evidence rather than being imputed or silently dropped.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
PUBLISHED_HQMQ_MAGS = 2502
CLAIM_BOUNDARY = (
    "Promoted MUCC v1 molecular-reference evidence only. MAG/proteome functional "
    "potential, processed transcriptional support, ESM2 similarity, and gLM2 window "
    "stability do not establish measured methane flux, final MRV score/A-E tier, "
    "carbon-crediting eligibility, or source-independent ecological transfer."
)

DEFAULT_ESM2_DIRS = [
    Path(
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard001/artifacts"
    ),
    Path(
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard002/artifacts"
    ),
    Path(
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard003/artifacts"
    ),
    Path(
        "results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard004/artifacts"
    ),
]
DEFAULT_GLM2_DIR = Path(
    "results/contextual_genomics/mucc_v1_owc_glm2_multiwindow_20260713"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--esm2-artifact-dir", type=Path, action="append", default=[])
    parser.add_argument("--glm2-dir", type=Path, default=DEFAULT_GLM2_DIR)
    parser.add_argument(
        "--neighbor-summary",
        type=Path,
        default=BASE
        / "bridge_reanchoring/integrated_atlas/wetland_reference_neighbor_summary.tsv",
    )
    parser.add_argument(
        "--network-manifest",
        type=Path,
        default=BASE / "network_analysis/flashweave_input_manifest.json",
    )
    parser.add_argument("--top-cards", type=int, default=100)
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_esm2(repo_root: Path, artifact_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    observed_dims: set[int] = set()
    for artifact_dir in artifact_dirs:
        meta_path = artifact_dir / "embedding_metadata.tsv"
        npz_path = artifact_dir / "genome_embeddings.npz"
        if not meta_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(f"missing ESM2 artifact pair in {artifact_dir}")
        meta = read_tsv(meta_path)
        if "proteome_id" not in meta.columns:
            if "sample" not in meta.columns:
                raise ValueError(f"{meta_path} lacks both proteome_id and sample")
            meta["proteome_id"] = meta["sample"].astype(str)
        matrix = np.load(npz_path, allow_pickle=False)["embeddings"]
        if matrix.ndim != 2 or len(meta) != matrix.shape[0]:
            raise ValueError(
                f"ESM2 metadata/embedding mismatch in {artifact_dir}: "
                f"metadata={len(meta)} embedding_shape={matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"ESM2 artifact contains non-finite values: {artifact_dir}"
            )
        observed_dims.add(int(matrix.shape[1]))
        stats_path = artifact_dir / "embedding_stats.json"
        stats: dict[str, Any] = {}
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text())
        meta = meta.copy()
        meta["esm2_embedding_dim"] = str(matrix.shape[1])
        meta["esm2_model_name"] = str(stats.get("model_name", ""))
        meta["esm2_completed_utc"] = str(stats.get("completed_utc", ""))
        meta["esm2_artifact_dir"] = relative(repo_root, artifact_dir)
        frames.append(meta)
    if len(observed_dims) != 1:
        raise ValueError(
            f"ESM2 dimensions differ across shards: {sorted(observed_dims)}"
        )
    esm2 = pd.concat(frames, ignore_index=True)
    if esm2["proteome_id"].duplicated().any():
        duplicates = esm2.loc[esm2["proteome_id"].duplicated(), "proteome_id"].head(10)
        raise ValueError(f"duplicate ESM2 proteome IDs: {duplicates.tolist()}")
    esm2["esm2_embedding_status"] = "complete_finite_embedding"
    return esm2


def build_embedding_status(
    manifest: pd.DataFrame,
    esm2: pd.DataFrame,
    glm2: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "esm2_include",
        "protein_count",
    ]
    base = manifest[[column for column in columns if column in manifest.columns]].copy()
    base = base.merge(
        esm2[
            [
                "proteome_id",
                "esm2_embedding_status",
                "esm2_embedding_dim",
                "esm2_model_name",
                "esm2_completed_utc",
                "esm2_artifact_dir",
            ]
        ],
        how="left",
        on="proteome_id",
    )
    glm2_columns = [
        "proteome_id",
        "mag_id",
        "n_native",
        "n_shuffle",
        "native_within_mag_dispersion",
        "native_dispersion_sd",
        "native_vs_shuffle_centroid_dist",
        "native_vs_shuffle_matched_dist",
        "matched_minus_dispersion_raw",
        "permutation_p",
        "glm_run_id",
        "model_name",
        "model_revision",
    ]
    base = base.merge(
        glm2[[column for column in glm2_columns if column in glm2.columns]],
        how="left",
        on=[column for column in ["proteome_id", "mag_id"] if column in glm2.columns],
    )
    expected_esm2 = truthy(base.get("esm2_include", pd.Series(False, index=base.index)))
    observed_esm2 = base["esm2_embedding_status"].eq("complete_finite_embedding")
    if not observed_esm2[expected_esm2].all():
        missing = (
            base.loc[expected_esm2 & ~observed_esm2, "proteome_id"].head(10).tolist()
        )
        raise ValueError(f"ESM2-ready MAGs without completed embeddings: {missing}")
    base["esm2_embedding_status"] = np.where(
        observed_esm2,
        "complete_finite_embedding",
        "not_available_no_direct_source_protein",
    )
    numeric_p = pd.to_numeric(base.get("permutation_p"), errors="coerce")
    native = pd.to_numeric(base.get("n_native"), errors="coerce")
    shuffled = pd.to_numeric(base.get("n_shuffle"), errors="coerce")
    glm2_valid = native.ge(1) & shuffled.ge(1) & numeric_p.notna()
    if not glm2_valid.all():
        missing = base.loc[~glm2_valid, "proteome_id"].head(10).tolist()
        raise ValueError(f"MUCC MAGs without valid gLM2 summary rows: {missing}")
    base["glm2_context_status"] = "complete_multiwindow_stability_summary"
    base["embedding_coverage_class"] = np.where(
        observed_esm2,
        "esm2_and_glm2_complete",
        "glm2_complete_esm2_source_protein_gap",
    )
    base.insert(0, "lane_id", LANE_ID)
    base["glm2_artifact_dir"] = relative(Path.cwd(), DEFAULT_GLM2_DIR)
    base["claim_boundary"] = CLAIM_BOUNDARY
    return base


def denominator_reconciliation(
    repo_root: Path,
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    expression: pd.DataFrame,
    esm2: pd.DataFrame,
    glm2: pd.DataFrame,
    samples: pd.DataFrame,
    kbase_catalog: pd.DataFrame,
    source_qc: pd.DataFrame,
) -> pd.DataFrame:
    direct_protein = int(truthy(manifest["esm2_include"]).sum())
    hqmq_rows = int(
        source_qc["published_mq_hq_membership_status"]
        .eq("meets_published_MQHQ_CheckM_threshold")
        .sum()
    )
    archive_scope_rows = int(
        source_qc["published_mq_hq_membership_status"]
        .eq("does_not_meet_published_MQHQ_CheckM_threshold")
        .sum()
    )
    qc_consistent_rows = int(
        source_qc["source_qc_value_consistency_status"]
        .eq("direct_source_qc_values_consistent_across_annotation_rows")
        .sum()
    )
    qc_exact = (
        len(source_qc) == len(catalog)
        and hqmq_rows == PUBLISHED_HQMQ_MAGS
        and archive_scope_rows == len(catalog) - PUBLISHED_HQMQ_MAGS
        and qc_consistent_rows == len(catalog)
    )
    rows = [
        {
            "lane_id": LANE_ID,
            "metric": "published_high_medium_quality_mag_headline",
            "value": PUBLISHED_HQMQ_MAGS,
            "status": "authoritative_published_headline",
            "evidence": "mSystems 10.1128/msystems.00680-25 and Zenodo MUCC v1 description",
        },
        {
            "lane_id": LANE_ID,
            "metric": "local_MAGs_zip_fasta_entries",
            "value": len(catalog),
            "status": "checksum_validated_Zenodo_archive_roster",
            "evidence": relative(
                repo_root, BASE / "manifests/mucc_v1_mag_catalog_full.tsv"
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "Zenodo_archive_minus_published_HQMQ_headline",
            "value": len(catalog) - PUBLISHED_HQMQ_MAGS,
            "status": (
                "exact_per_MAG_source_QC_scope_reconciled"
                if qc_exact
                else "per_MAG_source_QC_reconciliation_incomplete_review_required"
            ),
            "evidence": (
                "The record-specific Zenodo DRAM payload supplies per-MAG CheckM completeness and contamination. "
                "The paper's adapted MIMARKS screen (completeness >=50%, contamination <10%) identifies "
                f"{hqmq_rows} of {len(catalog)} archive MAGs; {archive_scope_rows} remain outside the published HQ/MQ scope."
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "direct_Zenodo_source_QC_reconciled_MAGs",
            "value": len(source_qc),
            "status": "complete_repeated_source_QC_values_consistent"
            if qc_consistent_rows == len(catalog)
            else "source_QC_value_conflict_or_missing_review_required",
            "evidence": relative(
                repo_root,
                BASE / "functional_features/feature_mucc_v1_zenodo_source_qc.tsv",
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "published_HQMQ_MAGs_reconciled_from_direct_Zenodo_QC",
            "value": hqmq_rows,
            "status": "exact_authoritative_headline_reconciliation"
            if qc_exact
            else "published_headline_reconciliation_incomplete",
            "evidence": "Borton et al. Methods/Figure S1/Table S3 quality definition applied to record-specific Zenodo DRAM QC fields.",
        },
        {
            "lane_id": LANE_ID,
            "metric": "archive_MAGs_outside_published_HQMQ_QC_scope",
            "value": archive_scope_rows,
            "status": "explicitly_retained_not_dropped"
            if qc_exact
            else "source_QC_scope_review_required",
            "evidence": relative(
                repo_root,
                BASE / "reconciliation/mucc_v1_zenodo_source_qc_reconciliation.tsv",
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "source_annotation_covered_MAGs",
            "value": len(manifest),
            "status": "source_scaffold_coverage",
            "evidence": relative(
                repo_root, BASE / "manifests/mucc_v1_source_lane_manifest.tsv"
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "processed_expression_supported_MAGs",
            "value": len(expression),
            "status": "processed_transcriptional_support_not_process_rate",
            "evidence": relative(
                repo_root,
                BASE / "expression/feature_mucc_v1_expression_mag_summary.tsv",
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "direct_source_protein_MAGs",
            "value": direct_protein,
            "status": "validated_ESM2_input_coverage",
            "evidence": relative(
                repo_root, BASE / "manifests/mucc_v1_esm2_input_manifest.tsv"
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "completed_finite_ESM2_MAGs",
            "value": len(esm2),
            "status": "validated_embedding_coverage",
            "evidence": "all completed MUCC ESM2 shard metadata and NPZ files",
        },
        {
            "lane_id": LANE_ID,
            "metric": "completed_gLM2_context_MAGs",
            "value": len(glm2),
            "status": "validated_multiwindow_context_coverage",
            "evidence": relative(
                repo_root,
                DEFAULT_GLM2_DIR / "validation/glm2_multiwindow_reduce_report.json",
            ),
        },
        {
            "lane_id": LANE_ID,
            "metric": "processed_expression_sample_columns",
            "value": len(samples),
            "status": "parsed_sample_scaffold_not_exact_environmental_join",
            "evidence": relative(
                repo_root,
                BASE / "environmental_metadata/mucc_v1_sample_columns_scaffold.tsv",
            ),
        },
    ]
    if not kbase_catalog.empty:
        kbase_matches = int(
            kbase_catalog["kbase_roster_reconciliation_status"]
            .eq("exact_MAG_id_match_public_KBase_GenomeSet")
            .sum()
        )
        kbase_absent = int(
            kbase_catalog["kbase_roster_reconciliation_status"]
            .eq("Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet")
            .sum()
        )
        kbase_taxonomy_only = int(
            kbase_catalog["taxonomy_reconciliation_status"]
            .eq("KBase_Gtdb_taxonomy_supplemental_only")
            .sum()
        )
        rows.extend(
            [
                {
                    "lane_id": LANE_ID,
                    "metric": "public_KBase_GenomeSet_exact_MAG_ID_members",
                    "value": kbase_matches,
                    "status": "public_identity_subset_quality_scope_unresolved",
                    "evidence": relative(
                        repo_root,
                        BASE / "source_audit/kbase_public_workspace_147022/"
                        "mucc_v1_kbase_public_catalog_reconciliation.tsv",
                    ),
                },
                {
                    "lane_id": LANE_ID,
                    "metric": "Zenodo_archive_minus_public_KBase_GenomeSet",
                    "value": kbase_absent,
                    "status": "public_KBase_subset_difference_not_a_quality_assignment",
                    "evidence": (
                        "The public KBase GenomeSet has exact MAG-ID matches for a strict subset "
                        "of the Zenodo archive; neither KBase presence nor absence identifies the "
                        "published HQ/MQ subset."
                    ),
                },
                {
                    "lane_id": LANE_ID,
                    "metric": "KBase_Gtdb_taxonomy_supplemental_only_MAGs",
                    "value": kbase_taxonomy_only,
                    "status": "supplementary_taxonomy_coverage_not_source_taxonomy_override",
                    "evidence": "KBase Workspace metadata, GTDB source version 214.1; source and KBase taxonomy values remain separately preserved.",
                },
            ]
        )
    frame = pd.DataFrame(rows)
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def staged_essdive_counts(run_dir: Path) -> tuple[int, int]:
    chamber_path = (
        run_dir / "environmental_metadata/fact_mucc_v1_essdive_chamber_flux.tsv"
    )
    porewater_path = (
        run_dir / "environmental_metadata/fact_mucc_v1_essdive_porewater_ch4.tsv"
    )
    chamber_rows = len(read_tsv(chamber_path)) if chamber_path.is_file() else 0
    porewater_rows = len(read_tsv(porewater_path)) if porewater_path.is_file() else 0
    return chamber_rows, porewater_rows


def staged_essdive_gapfilled_tower_counts(run_dir: Path) -> tuple[int, int]:
    path = (
        run_dir
        / "environmental_metadata/fact_mucc_v1_essdive_gapfilled_tower_ch4_flux.tsv"
    )
    if not path.is_file():
        return 0, 0
    flux = read_tsv(path)
    valid = int(flux["source_value_status"].eq("reported_valid").sum())
    return len(flux), valid


def staged_bioproject_crosswalk_counts(run_dir: Path) -> tuple[int, int]:
    path = (
        run_dir / "environmental_metadata/link_mucc_v1_sequence_bioproject_sample.tsv"
    )
    if not path.is_file():
        return 0, 0
    crosswalk = read_tsv(path)
    mapped = int(
        crosswalk["sample_project_link_status"]
        .eq("mapped_to_authoritative_NCBI_BioProject_title")
        .sum()
    )
    return len(crosswalk), mapped


def staged_jgi_sample_crosswalk_counts(run_dir: Path) -> tuple[int, int, int]:
    path = run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_sample.tsv"
    if not path.is_file():
        return 0, 0, 0
    crosswalk = read_tsv(path)
    exact = int(
        crosswalk["jgi_sample_identity_status"]
        .eq("exact_source_label_to_JGI_Sample_QC_record")
        .sum()
    )
    final_portal = int(crosswalk["jgi_final_deliverable_portal_id"].ne("").sum())
    return len(crosswalk), exact, final_portal


def staged_jgi_data_portal_crosswalk_counts(run_dir: Path) -> tuple[int, int, int, int]:
    path = run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_data_portal.tsv"
    if not path.is_file():
        return 0, 0, 0, 0
    crosswalk = read_tsv(path)
    exact = int(
        crosswalk["jgi_data_portal_identity_status"]
        .eq("exact_source_label_to_JGI_Data_Portal_record_pair")
        .sum()
    )
    july_aliases = int(
        crosswalk["jgi_data_portal_label_mapping_method"]
        .eq("deterministic_July_to_Jul")
        .sum()
    )
    no_depth = int(
        crosswalk["depth_cm_join_status"]
        .eq("not_reported_by_JGI_Data_Portal_catalog")
        .sum()
    )
    return len(crosswalk), exact, july_aliases, no_depth


def staged_sra_sample_crosswalk_counts(run_dir: Path) -> tuple[int, int, int, int]:
    path = run_dir / "environmental_metadata/link_mucc_v1_sequence_sra_sample.tsv"
    if not path.is_file():
        return 0, 0, 0, 0
    crosswalk = read_tsv(path)
    exact_identity = int(
        crosswalk["sra_sample_identity_status"]
        .eq("exact_source_label_to_NCBI_SRA_package")
        .sum()
    )
    exact_collection_date = int(
        crosswalk["sra_collection_date_status"]
        .eq("exact_collection_date_from_NCBI_SRA_sample_attributes")
        .sum()
    )
    unresolved = int(len(crosswalk) - exact_identity)
    return len(crosswalk), exact_identity, exact_collection_date, unresolved


def staged_sra_library_strategy_counts(run_dir: Path) -> tuple[int, int]:
    """Return declared SRA library strategies without asserting assay equivalence."""
    path = run_dir / "environmental_metadata/link_mucc_v1_sequence_sra_sample.tsv"
    if not path.is_file():
        return 0, 0
    crosswalk = read_tsv(path)
    return (
        int(crosswalk["sra_library_strategy"].eq("RNA-Seq").sum()),
        int(crosswalk["sra_library_strategy"].eq("WGS").sum()),
    )


def staged_methods_design_counts(run_dir: Path) -> tuple[int, int, int, int]:
    path = (
        run_dir
        / "environmental_metadata/feature_mucc_v1_sample_methods_design_context.tsv"
    )
    if not path.is_file():
        return 0, 0, 0, 0
    context = read_tsv(path)
    direct = int(
        context["methods_design_context_status"]
        .eq("validated_2018_methods_context_direct_depth_code")
        .sum()
    )
    d6_pending = int(
        context["methods_design_context_status"]
        .eq("validated_2018_cohort_but_raw_depth_code_reconciliation_pending")
        .sum()
    )
    legacy = int(
        context["methods_design_context_status"]
        .eq("legacy_sample_label_context_only")
        .sum()
    )
    return len(context), direct, d6_pending, legacy


def staged_authoritative_ecological_crosswalk_counts(
    run_dir: Path,
) -> tuple[int, int, int, int]:
    """Count a staged explicit ecology mapping without inferring any joins."""

    links_path = (
        run_dir
        / "environmental_metadata/link_mucc_v1_sequence_authoritative_ecology.tsv"
    )
    readiness_path = (
        run_dir
        / "environmental_metadata/feature_mucc_v1_authoritative_ecology_readiness.tsv"
    )
    if not links_path.is_file() and not readiness_path.is_file():
        return 0, 0, 0, 0
    if not links_path.is_file() or not readiness_path.is_file():
        raise ValueError(
            "authoritative ecological crosswalk stage is incomplete: both link and readiness files are required"
        )
    links = read_tsv(links_path)
    readiness = read_tsv(readiness_path)
    required_link_columns = {"sample_id", "mapping_validation_status"}
    required_readiness_columns = {
        "sample_id",
        "authoritative_ecology_readiness_status",
    }
    if not required_link_columns.issubset(
        links.columns
    ) or not required_readiness_columns.issubset(readiness.columns):
        raise ValueError(
            "authoritative ecological crosswalk stage has an invalid schema"
        )
    if readiness["sample_id"].duplicated().any():
        raise ValueError(
            "authoritative ecological readiness must contain one row per sample"
        )
    valid = int(
        links["mapping_validation_status"]
        .eq("validated_authoritative_sample_environment_flux_mapping")
        .sum()
    )
    ready = int(
        readiness["authoritative_ecology_readiness_status"]
        .eq("ready_for_grouped_ecological_validation")
        .sum()
    )
    return len(links), valid, len(readiness), ready


def build_sample_ecological_readiness(
    samples: pd.DataFrame,
    readiness: pd.DataFrame,
    run_dir: Path,
) -> pd.DataFrame:
    merged = samples.merge(
        readiness,
        how="left",
        on=[
            column
            for column in ["sample_id", "source_sample_column"]
            if column in readiness.columns
        ],
    )
    if "lane_id" in merged.columns:
        merged["lane_id"] = LANE_ID
    else:
        merged.insert(0, "lane_id", LANE_ID)
    merged["sample_identity_status"] = "source_expression_column_parsed"
    crosswalk_path = (
        run_dir / "environmental_metadata/link_mucc_v1_sequence_bioproject_sample.tsv"
    )
    if crosswalk_path.is_file():
        crosswalk = read_tsv(crosswalk_path)
        link_columns = [
            "sample_id",
            "sample_project_link_status",
            "bioproject_accession",
            "sequence_project_identity_status",
        ]
        merged = merged.merge(crosswalk[link_columns], on="sample_id", how="left")
        mapped = merged["sample_project_link_status"].eq(
            "mapped_to_authoritative_NCBI_BioProject_title"
        )
        merged.loc[mapped, "sample_identity_status"] = (
            "source_expression_column_linked_to_NCBI_BioProject"
        )
    else:
        merged["sample_project_link_status"] = "pending_NCBI_BioProject_crosswalk"
        merged["bioproject_accession"] = ""
        merged["sequence_project_identity_status"] = (
            "pending_authoritative_sequence_project_link"
        )
    merged["exact_collection_datetime_status"] = "pending_sample_crosswalk"
    merged["depth_cm_join_status"] = "pending_authoritative_depth_mapping"
    jgi_crosswalk_path = (
        run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_sample.tsv"
    )
    if jgi_crosswalk_path.is_file():
        jgi_crosswalk = read_tsv(jgi_crosswalk_path)
        jgi_columns = [
            "sample_id",
            "jgi_final_deliverable_portal_id",
            "jgi_sample_id",
            "jgi_sample_name",
            "jgi_sample_receipt_datetime",
            "jgi_sample_qc_datetime",
            "jgi_sample_qc_result",
            "jgi_sample_label_mapping_method",
            "jgi_sample_identity_status",
            "collection_datetime_status",
            "depth_cm_join_status",
        ]
        jgi_crosswalk = jgi_crosswalk[
            [column for column in jgi_columns if column in jgi_crosswalk.columns]
        ].rename(
            columns={
                "collection_datetime_status": "jgi_collection_datetime_status",
                "depth_cm_join_status": "jgi_depth_cm_join_status",
            }
        )
        merged = merged.merge(jgi_crosswalk, on="sample_id", how="left")
        resolved_portal = merged["jgi_final_deliverable_portal_id"].ne("")
        merged.loc[resolved_portal, "sample_identity_status"] = (
            "source_expression_column_linked_to_JGI_final_deliverable_portal"
        )
        exact_jgi = merged["jgi_sample_identity_status"].eq(
            "exact_source_label_to_JGI_Sample_QC_record"
        )
        merged.loc[exact_jgi, "sample_identity_status"] = (
            "source_expression_column_linked_to_exact_JGI_Sample_QC_record"
        )
        merged.loc[exact_jgi, "exact_collection_datetime_status"] = merged.loc[
            exact_jgi, "jgi_collection_datetime_status"
        ]
        merged.loc[exact_jgi, "depth_cm_join_status"] = merged.loc[
            exact_jgi, "jgi_depth_cm_join_status"
        ]
    else:
        merged["jgi_sample_identity_status"] = "pending_JGI_Sample_QC_crosswalk"
        merged["jgi_sample_id"] = ""
    jgi_data_portal_path = (
        run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_data_portal.tsv"
    )
    if jgi_data_portal_path.is_file():
        jgi_data_portal = read_tsv(jgi_data_portal_path)
        jgi_data_portal_columns = [
            "sample_id",
            "jgi_data_portal_label_mapping_method",
            "jgi_data_portal_identity_status",
            "jgi_data_portal_matched_catalog_label",
            "jgi_data_portal_expression_record_id",
            "jgi_data_portal_annotation_record_id",
            "jgi_data_portal_annotation_taxon_oid",
            "jgi_data_portal_visibility",
            "jgi_data_portal_data_utilization_status",
            "jgi_data_portal_processing_status",
            "jgi_data_portal_work_completion_date",
            "jgi_data_portal_indexed_file_count",
            "jgi_data_portal_indexed_file_bytes",
            "jgi_data_portal_purged_file_count",
            "jgi_data_portal_nonpurged_file_count",
            "jgi_data_portal_unknown_file_status_count",
            "jgi_data_portal_file_access_status",
            "jgi_data_portal_latitude",
            "jgi_data_portal_longitude",
            "jgi_data_portal_coordinate_status",
        ]
        jgi_data_portal = jgi_data_portal[
            [
                column
                for column in jgi_data_portal_columns
                if column in jgi_data_portal.columns
            ]
        ]
        merged = merged.merge(jgi_data_portal, on="sample_id", how="left")
        exact_data_portal = merged["jgi_data_portal_identity_status"].eq(
            "exact_source_label_to_JGI_Data_Portal_record_pair"
        )
        without_sample_qc = ~merged["jgi_sample_identity_status"].eq(
            "exact_source_label_to_JGI_Sample_QC_record"
        )
        merged.loc[exact_data_portal & without_sample_qc, "sample_identity_status"] = (
            "source_expression_column_linked_to_exact_JGI_Data_Portal_record_pair"
        )
    else:
        merged["jgi_data_portal_identity_status"] = (
            "pending_JGI_Data_Portal_catalog_crosswalk"
        )
        merged["jgi_data_portal_expression_record_id"] = ""
    sra_crosswalk_path = (
        run_dir / "environmental_metadata/link_mucc_v1_sequence_sra_sample.tsv"
    )
    if sra_crosswalk_path.is_file():
        sra_crosswalk = read_tsv(sra_crosswalk_path)
        sra_columns = [
            "sample_id",
            "sra_matched_source_label",
            "sra_label_mapping_method",
            "sra_search_status",
            "sra_package_selection_status",
            "sra_study_accession",
            "sra_bioproject_accession",
            "sra_experiment_accession",
            "sra_run_accessions",
            "sra_biosample_accession",
            "sra_sample_accession",
            "sra_experiment_title",
            "sra_sample_title",
            "sra_library_strategy",
            "sra_library_source",
            "sra_library_layout",
            "sra_instrument_model",
            "sra_run_publication_datetime",
            "sra_collection_date",
            "sra_collection_date_status",
            "sra_geo_loc_name",
            "sra_lat_lon",
            "sra_isolation_source",
            "sra_gold_ecosystem_classification",
            "sra_reported_depth_value",
            "sra_depth_cm_join_status",
            "sra_sample_identity_status",
        ]
        merged = merged.merge(
            sra_crosswalk[
                [column for column in sra_columns if column in sra_crosswalk.columns]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        exact_sra = merged["sra_sample_identity_status"].eq(
            "exact_source_label_to_NCBI_SRA_package"
        )
        exact_jgi = merged["jgi_sample_identity_status"].eq(
            "exact_source_label_to_JGI_Sample_QC_record"
        )
        merged.loc[exact_sra & exact_jgi, "sample_identity_status"] = (
            "source_expression_column_linked_to_exact_NCBI_SRA_package_and_JGI_Sample_QC_record"
        )
        merged.loc[exact_sra & ~exact_jgi, "sample_identity_status"] = (
            "source_expression_column_linked_to_exact_NCBI_SRA_package"
        )
        exact_collection_date = merged["sra_collection_date_status"].eq(
            "exact_collection_date_from_NCBI_SRA_sample_attributes"
        )
        merged.loc[exact_collection_date, "exact_collection_datetime_status"] = (
            "exact_collection_date_only_from_NCBI_SRA_sample_attributes"
        )
        merged.loc[exact_sra, "depth_cm_join_status"] = merged.loc[
            exact_sra, "sra_depth_cm_join_status"
        ]
        merged["sra_assay_reconciliation_status"] = (
            "unresolved_no_exact_NCBI_SRA_package"
        )
        rna_seq = exact_sra & merged["sra_library_strategy"].eq("RNA-Seq")
        wgs = exact_sra & merged["sra_library_strategy"].eq("WGS")
        merged.loc[rna_seq, "sra_assay_reconciliation_status"] = (
            "declared_NCBI_SRA_RNA_Seq_metatranscriptomic_package"
        )
        merged.loc[wgs, "sra_assay_reconciliation_status"] = (
            "declared_NCBI_SRA_WGS_package_assay_reconciliation_required"
        )
    else:
        merged["sra_sample_identity_status"] = "pending_NCBI_SRA_sample_crosswalk"
        merged["sra_collection_date_status"] = "pending_NCBI_SRA_sample_crosswalk"
        merged["sra_depth_cm_join_status"] = "pending_NCBI_SRA_sample_crosswalk"
        merged["sra_assay_reconciliation_status"] = "pending_NCBI_SRA_sample_crosswalk"
    methods_context_path = (
        run_dir
        / "environmental_metadata/feature_mucc_v1_sample_methods_design_context.tsv"
    )
    if methods_context_path.is_file():
        methods_context = read_tsv(methods_context_path)
        method_columns = [
            "sample_id",
            "method_source",
            "method_evidence_pointer",
            "methods_cohort",
            "methods_collection_year",
            "methods_collection_month",
            "methods_site_or_landcover",
            "methods_core_label",
            "raw_depth_code",
            "raw_replicate_label",
            "nominal_depth_interval_cm",
            "methods_depth_assignment_status",
            "methods_design_context_status",
            "exact_table_s4_row_status",
        ]
        merged = merged.merge(
            methods_context[
                [
                    column
                    for column in method_columns
                    if column in methods_context.columns
                ]
            ],
            on="sample_id",
            how="left",
        )
    else:
        merged["methods_design_context_status"] = (
            "pending_published_SuF1_methods_recovery"
        )
        merged["exact_table_s4_row_status"] = "pending_published_Tables_S1_S13_recovery"
    chamber_rows, porewater_rows = staged_essdive_counts(run_dir)
    tower_rows, tower_valid = staged_essdive_gapfilled_tower_counts(run_dir)
    if chamber_rows and porewater_rows:
        merged["environment_measurement_join_status"] = (
            "ESS_DIVE_records_staged_unlinked_to_sequence_samples"
        )
        merged["flux_window_join_status"] = (
            "ESS_DIVE_chamber_and_porewater_observations_staged_pending_crosswalk"
            if not tower_rows
            else "ESS_DIVE_chamber_porewater_and_gapfilled_tower_observations_staged_pending_crosswalk"
        )
    else:
        merged["environment_measurement_join_status"] = (
            "pending_ESS_DIVE_AmeriFlux_NERR_or_supplement_join"
        )
        merged["flux_window_join_status"] = (
            "pending_porewater_or_chamber_time_window_crosswalk"
        )
    merged["sample_ecological_validation_status"] = (
        "blocked_missing_exact_sample_environment_flux_join"
    )
    merged["validation_split_eligibility"] = (
        "not_eligible_until_date_depth_and_flux_windows_resolve"
    )
    merged["next_validation_action"] = (
        "Request a corrected publisher/author Table S4-equivalent authoritative sample/date/depth crosswalk, "
        "reconcile the raw July/September D6 versus published D5 notation, then join environmental and flux "
        "observations at an explicit resolution tier."
    )
    merged["authoritative_ecology_link_status"] = "not_staged"
    authoritative_links_path = (
        run_dir
        / "environmental_metadata/link_mucc_v1_sequence_authoritative_ecology.tsv"
    )
    authoritative_readiness_path = (
        run_dir
        / "environmental_metadata/feature_mucc_v1_authoritative_ecology_readiness.tsv"
    )
    if authoritative_links_path.is_file() or authoritative_readiness_path.is_file():
        staged_authoritative_ecological_crosswalk_counts(run_dir)
        authoritative_links = read_tsv(authoritative_links_path)
        authoritative_readiness = read_tsv(authoritative_readiness_path)
        valid_status = "validated_authoritative_sample_environment_flux_mapping"
        ready_status = "ready_for_grouped_ecological_validation"
        valid_links = authoritative_links.loc[
            authoritative_links["mapping_validation_status"].eq(valid_status)
        ].copy()
        ready_samples = authoritative_readiness.loc[
            authoritative_readiness["authoritative_ecology_readiness_status"].eq(
                ready_status
            ),
            "sample_id",
        ]
        valid_samples = set(valid_links["sample_id"])
        if set(ready_samples) != valid_samples:
            raise ValueError(
                "authoritative ecological readiness disagrees with validated crosswalk link rows"
            )
        link_columns = [
            "sample_id",
            "mapping_id",
            "authoritative_sample_id",
            "collection_datetime",
            "site_id",
            "core_or_plot_id",
            "depth_cm",
            "depth_reference",
            "sequence_assay_type",
            "assay_reconciliation_status",
            "mag_abundance_or_read_coverage_record_id",
            "mag_abundance_or_read_coverage_units",
            "environment_source",
            "environment_record_id",
            "environment_measurement_datetime",
            "environment_measurement_units",
            "flux_source",
            "flux_observation_id",
            "flux_measurement_type",
            "flux_units",
            "flux_window_start_datetime",
            "flux_window_end_datetime",
            "replicate_id",
            "uncertainty_record_id",
            "uncertainty_method",
            "source_url",
        ]
        selected_links = (
            valid_links[
                [column for column in link_columns if column in valid_links.columns]
            ]
            .sort_values("mapping_id")
            .drop_duplicates("sample_id", keep="first")
            .rename(
                columns={
                    column: f"authoritative_ecology_{column}"
                    for column in link_columns
                    if column != "sample_id" and column in valid_links.columns
                }
            )
        )
        merged = merged.merge(
            selected_links, on="sample_id", how="left", validate="one_to_one"
        )
        eligible = merged["sample_id"].isin(valid_samples)
        merged.loc[eligible, "authoritative_ecology_link_status"] = (
            "validated_authoritative_sample_environment_flux_mapping"
        )
        merged.loc[eligible, "exact_collection_datetime_status"] = (
            "exact_authoritative_collection_datetime"
        )
        merged.loc[eligible, "depth_cm_join_status"] = "exact_authoritative_depth_cm"
        merged.loc[eligible, "environment_measurement_join_status"] = (
            "exact_authoritative_sample_to_environment_record"
        )
        merged.loc[eligible, "flux_window_join_status"] = (
            "exact_authoritative_sample_to_flux_window"
        )
        merged.loc[eligible, "sample_ecological_validation_status"] = (
            "eligible_for_grouped_ecological_validation_not_final_MRV"
        )
        merged.loc[eligible, "validation_split_eligibility"] = (
            "eligible_for_grouped_date_depth_site_validation_split"
        )
        merged.loc[eligible, "next_validation_action"] = (
            "Run preregistered grouped ecological validation with the exact mapped abundance/read coverage, "
            "environment, flux window, replicate, and uncertainty records; final MRV remains out of scope."
        )
    merged["claim_boundary"] = CLAIM_BOUNDARY
    return merged


def external_source_registry(run_dir: Path) -> pd.DataFrame:
    chamber_rows, porewater_rows = staged_essdive_counts(run_dir)
    tower_rows, tower_valid = staged_essdive_gapfilled_tower_counts(run_dir)
    bioproject_rows, bioproject_mapped = staged_bioproject_crosswalk_counts(run_dir)
    jgi_rows, jgi_exact, jgi_final_portals = staged_jgi_sample_crosswalk_counts(run_dir)
    jgi_data_rows, jgi_data_exact, jgi_data_aliases, jgi_data_no_depth = (
        staged_jgi_data_portal_crosswalk_counts(run_dir)
    )
    sra_rows, sra_exact, sra_dates, sra_unresolved = staged_sra_sample_crosswalk_counts(
        run_dir
    )
    sra_rna_seq, sra_wgs = staged_sra_library_strategy_counts(run_dir)
    methods_rows, methods_direct, methods_d6_pending, methods_legacy = (
        staged_methods_design_counts(run_dir)
    )
    (
        authoritative_link_rows,
        authoritative_valid_links,
        authoritative_readiness_rows,
        authoritative_ready_samples,
    ) = staged_authoritative_ecological_crosswalk_counts(run_dir)
    essdive_status = (
        f"source_records_staged_unlinked_to_sequence_samples; chamber_rows={chamber_rows}; "
        f"porewater_rows={porewater_rows}"
        if chamber_rows and porewater_rows
        else "required_for_observed_flux_validation"
    )
    essdive_tower_status = (
        f"source_context_staged_unlinked_to_sequence_samples; half_hourly_rows={tower_rows}; "
        f"valid_CH4_rows={tower_valid}; 2015-2016_temporal_overlap_does_not_authorize_a_join"
        if tower_rows
        else "not_staged"
    )
    rows = [
        (
            "NCBI_BioProject_Old_Woman_Creek",
            "NCBI BioProject titles for Old Woman Creek",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "sequence-project identity from published sample-label title suffixes",
            "exact source sample label; does not provide date/depth/flux link alone",
            (
                f"source_titles_staged; mapped_samples={bioproject_mapped}/{bioproject_rows}; "
                "remaining rows preserved as unmatched"
                if bioproject_rows
                else "not_staged"
            ),
        ),
        (
            "NCBI_SRA_MUCC_expression_packages",
            "NCBI SRA experiment packages matched to MUCC expression labels",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "SRA/BioSample/experiment/run accessions plus reported collection date, location, and library metadata",
            "exact expression label matched to an entity-validated NCBI SRA package",
            (
                f"exact_SRA_packages={sra_exact}/{sra_rows}; exact_collection_dates={sra_dates}; "
                f"declared_RNA_Seq={sra_rna_seq}; declared_WGS={sra_wgs}; unresolved={sra_unresolved}; "
                "depth/environment/flux joins remain unresolved and WGS-declared packages require assay reconciliation"
                if sra_rows
                else "not_staged"
            ),
        ),
        (
            "JGI_Genome_Portal_Sample_QC",
            "JGI Genome Portal public Sample QC exports",
            "https://genome.jgi.doe.gov/",
            "exact JGI Sample identity and operational receipt/QC timestamps",
            "NCBI-linked BioProject plus exact source expression-label match",
            (
                f"final_JGI_portal_links={jgi_final_portals}/{jgi_rows}; "
                f"exact_JGI_Sample_QC_records={jgi_exact}/{jgi_rows}; "
                "collection date/depth/environment/flux remain unresolved"
                if jgi_rows
                else "not_staged"
            ),
        ),
        (
            "JGI_Data_Portal_award_504205",
            "JGI Data Portal public source-label catalog records",
            "https://files.jgi.doe.gov/search/",
            "independent source-label-specific analysis/expression record identity and processing provenance",
            "exact source label plus immutable JGI award 504205 and one analysis/taxon plus one expression record",
            (
                f"exact_JGI_catalog_record_pairs={jgi_data_exact}/{jgi_data_rows}; "
                f"controlled_July_to_Jul_aliases={jgi_data_aliases}; catalog_depth_not_reported="
                f"{jgi_data_no_depth}; collection time/environment/flux remain unresolved"
                if jgi_data_rows
                else "not_staged"
            ),
        ),
        (
            "paper_supplement_tables",
            "ASM Supplemental Information PDF and Tables S1-S13",
            "https://journals.asm.org/doi/10.1128/msystems.00680-25",
            "methods-level 2018 cohort design; Table S4 should provide sample-level metadata",
            "source_sample_column plus authoritative sample/date/depth crosswalk",
            (
                f"SuF1 methods context staged={methods_rows}/133; direct coded-depth context={methods_direct}; "
                f"D6/D5 reconciliation pending={methods_d6_pending}; legacy label-only rows={methods_legacy}; "
                "SuF2 Tables S1-S13 XLSX is malformed, so exact sample/date/depth/environment/flux joins remain blocked"
                if methods_rows
                else "published supplement methods/table recovery not staged"
            ),
        ),
        (
            "ESS_DIVE_porewater_and_chamber_flux",
            "ESS-DIVE OWC porewater CH4/CO2 and chamber flux data",
            "https://doi.org/10.15485/1568865",
            "porewater gas, chamber methane flux, time/depth/site context",
            "site + collection date/time + depth or explicit sample identifier",
            essdive_status,
        ),
        (
            "ESS_DIVE_gapfilled_tower_flux",
            "ESS-DIVE US-OWC gap-filled eddy-covariance methane and carbon dioxide fluxes",
            "https://doi.org/10.15485/2500238",
            "half-hourly gap-filled tower CH4/CO2 fluxes at the US-OWC site",
            "authoritative sequencing sample time/plot/depth plus documented tower-context mapping",
            essdive_tower_status,
        ),
        (
            "authoritative_MUCC_sequence_to_ecology_crosswalk",
            "Author/publisher-supplied MUCC sequence-to-ecology crosswalk",
            "retained_in_crosswalk_source_url_column",
            "exact sample/date/depth/assay/coverage/environment/flux-window/uncertainty linkage",
            "source_sample_column plus all fields required by the strict crosswalk contract",
            (
                f"validated link rows={authoritative_valid_links}/{authoritative_link_rows}; "
                f"eligible samples={authoritative_ready_samples}/{authoritative_readiness_rows}; "
                "eligible rows remain grouped-ecological-validation inputs, not final MRV"
                if authoritative_link_rows or authoritative_readiness_rows
                else "contract_not_materialized; no inferred ecology mapping is permitted"
            ),
        ),
        (
            "AmeriFlux_US_OWC",
            "AmeriFlux site US-OWC",
            "https://ameriflux.lbl.gov/sites/siteinfo/US-OWC",
            "site-scale eddy covariance and meteorological context",
            "time window only; never substitute for exact core/depth observation",
            "site_level_context_only_until_window_join",
        ),
        (
            "NERR_CDMO",
            "NERR Centralized Data Management Office",
            "https://cdmo.baruch.sc.edu/",
            "hydrological, meteorological, water-quality context",
            "station + time window; preserve station and resolution tier",
            "contextual_covariates_pending_station_time_join",
        ),
        (
            "MassIVE_MSV000093935",
            "OWC LC-MS metabolomics",
            "https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000093935",
            "metabolite evidence, including substrate context",
            "sample or paired extraction metadata; no join assumed",
            "metabolite_context_pending_sample_crosswalk",
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "source_id",
            "source_name",
            "landing_page_or_doi",
            "expected_measurements",
            "required_join_key",
            "current_join_status",
        ],
    )
    frame.insert(0, "lane_id", LANE_ID)
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def promotion_gates(
    repo_root: Path,
    run_dir: Path,
    catalog: pd.DataFrame,
    esm2: pd.DataFrame,
    glm2: pd.DataFrame,
    source_qc: pd.DataFrame,
    neighbor_summary: Path,
    network_manifest: Path,
    ecological: pd.DataFrame,
) -> pd.DataFrame:
    neighbor_rows = 0
    if neighbor_summary.is_file() and neighbor_summary.stat().st_size:
        neighbor_rows = len(read_tsv(neighbor_summary))
    network_state = "not_materialized"
    network_detail = "network input manifest is not materialized"
    if network_manifest.is_file() and network_manifest.stat().st_size:
        network_state = json.loads(network_manifest.read_text()).get(
            "status", "materialized"
        )
        network_detail = f"network input state={network_state}"
    network_summary = (
        run_dir / "network_analysis/mucc_v1_flashweave_network_summary.json"
    )
    if network_summary.is_file() and network_summary.stat().st_size:
        completed = json.loads(network_summary.read_text())
        if int(completed.get("mag_mag_edges", 0)) > 0:
            network_state = "completed_exploratory_flashweave"
            network_detail = (
                f"completed MAG-to-MAG edges={completed['mag_mag_edges']}; "
                f"metadata-involving edges={completed.get('metadata_involving_edges', 0)}"
            )
    stability_summary = (
        run_dir / "network_analysis/mucc_v1_flashweave_stability_summary.json"
    )
    stability_status = "blocked"
    stability_detail = (
        "scaffold-stratified edge-selection stability is not materialized"
    )
    if stability_summary.is_file() and stability_summary.stat().st_size:
        stability = json.loads(stability_summary.read_text())
        iterations = int(stability.get("iterations", 0))
        stable_edges = int(stability.get("edges_at_or_above_stability_threshold", 0))
        baseline_edges = int(stability.get("baseline_mag_mag_edges", 0))
        stability_status = "pass" if iterations >= 20 else "limited"
        stability_detail = (
            f"iterations={iterations}; edges at or above threshold={stable_edges}/{baseline_edges}; "
            "scaffold labels remain distinct from exact ecological covariates"
        )
    wgcna_summary_path = (
        run_dir / "network_analysis/mucc_v1_wgcna_secondary_summary.json"
    )
    wgcna_status = "ready"
    wgcna_detail = "secondary WGCNA module comparison is not materialized"
    if wgcna_summary_path.is_file() and wgcna_summary_path.stat().st_size:
        wgcna = json.loads(wgcna_summary_path.read_text())
        samples = int(wgcna.get("samples", 0))
        features = int(wgcna.get("mag_features", 0))
        non_grey_modules = int(wgcna.get("non_grey_module_count", 0))
        wgcna_expected_features = 0
        wgcna_manifest_path = (
            run_dir / "network_analysis/wgcna_secondary_input_manifest.json"
        )
        if wgcna_manifest_path.is_file() and wgcna_manifest_path.stat().st_size:
            wgcna_expected_features = int(
                json.loads(wgcna_manifest_path.read_text()).get(
                    "selected_MAG_features", 0
                )
            )
        wgcna_status = (
            "pass"
            if wgcna.get("status") == "completed_secondary_descriptive_modules"
            and samples == 133
            and wgcna_expected_features > 0
            and features == wgcna_expected_features
            and non_grey_modules >= 1
            else "fail"
        )
        wgcna_detail = (
            f"R={wgcna.get('r_version', '')}; WGCNA={wgcna.get('wgcna_version', '')}; "
            f"samples={samples}; MAG features={features}/{wgcna_expected_features}; non-grey modules={non_grey_modules}; "
            f"softPower={wgcna.get('soft_power', '')}; "
            f"selection={wgcna.get('soft_power_selection', '')}; secondary descriptive comparison only"
        )
    chamber_rows, porewater_rows = staged_essdive_counts(run_dir)
    tower_rows, tower_valid = staged_essdive_gapfilled_tower_counts(run_dir)
    bioproject_rows, bioproject_mapped = staged_bioproject_crosswalk_counts(run_dir)
    jgi_rows, jgi_exact, jgi_final_portals = staged_jgi_sample_crosswalk_counts(run_dir)
    jgi_data_rows, jgi_data_exact, jgi_data_aliases, jgi_data_no_depth = (
        staged_jgi_data_portal_crosswalk_counts(run_dir)
    )
    sra_rows, sra_exact, sra_dates, sra_unresolved = staged_sra_sample_crosswalk_counts(
        run_dir
    )
    sra_rna_seq, sra_wgs = staged_sra_library_strategy_counts(run_dir)
    methods_rows, methods_direct, methods_d6_pending, methods_legacy = (
        staged_methods_design_counts(run_dir)
    )
    (
        authoritative_link_rows,
        authoritative_valid_links,
        authoritative_readiness_rows,
        authoritative_ready_samples,
    ) = staged_authoritative_ecological_crosswalk_counts(run_dir)
    recovery_ledger = (
        run_dir / "source_audit/mucc_v1_source_metadata_recovery_ledger.tsv"
    )
    table_s1_status = "not_recovery_audited"
    if recovery_ledger.is_file() and recovery_ledger.stat().st_size:
        recovered = read_tsv(recovery_ledger)
        table_s1 = recovered.loc[
            recovered["artifact_role"].eq(
                "published_Table_S1_to_S13_accession_spreadsheet"
            ),
            "availability_status",
        ].tolist()
        if len(table_s1) == 1:
            table_s1_status = table_s1[0]
    hqmq_rows = int(
        source_qc["published_mq_hq_membership_status"]
        .eq("meets_published_MQHQ_CheckM_threshold")
        .sum()
    )
    archive_scope_rows = int(
        source_qc["published_mq_hq_membership_status"]
        .eq("does_not_meet_published_MQHQ_CheckM_threshold")
        .sum()
    )
    qc_consistent_rows = int(
        source_qc["source_qc_value_consistency_status"]
        .eq("direct_source_qc_values_consistent_across_annotation_rows")
        .sum()
    )
    rows = [
        (
            "source_payload_provenance",
            "pass",
            "validated source ledger and staged payloads exist",
        ),
        (
            "published_vs_local_MAG_denominator",
            "pass"
            if len(source_qc) == len(catalog)
            and hqmq_rows == PUBLISHED_HQMQ_MAGS
            and archive_scope_rows == len(catalog) - PUBLISHED_HQMQ_MAGS
            and qc_consistent_rows == len(catalog)
            else "warn",
            (
                f"published HQ/MQ headline={PUBLISHED_HQMQ_MAGS}; checksum-validated Zenodo archive/local "
                f"roster={len(catalog)}; direct Zenodo source-QC reconciled HQ/MQ={hqmq_rows}; "
                f"archive members outside published HQ/MQ QC scope={archive_scope_rows}; "
                f"repeated-QC-consistent MAGs={qc_consistent_rows}"
            ),
        ),
        ("ESM2_embedding_coverage", "pass", f"completed finite ESM2 MAGs={len(esm2)}"),
        ("gLM2_context_coverage", "pass", f"completed gLM2 MAGs={len(glm2)}"),
        (
            "gLM2_independence_boundary",
            "pass",
            "gLM2 window stability is explicitly not treated as independent ESM2 validation",
        ),
        (
            "source_aware_neighbor_bridge",
            "pass" if neighbor_rows else "blocked",
            f"source-aware neighbor summary rows={neighbor_rows}",
        ),
        (
            "FlashWeave_conditional_network",
            "pass"
            if network_state == "completed_exploratory_flashweave"
            else "ready"
            if network_state == "ready_to_run_flashweave"
            else "blocked",
            network_detail,
        ),
        (
            "FlashWeave_edge_selection_stability",
            stability_status,
            stability_detail,
        ),
        (
            "WGCNA_secondary_descriptive_modules",
            wgcna_status,
            wgcna_detail,
        ),
        (
            "ESS_DIVE_flux_and_porewater_source_staging",
            "pass" if chamber_rows and porewater_rows else "blocked",
            f"staged chamber rows={chamber_rows}; porewater rows={porewater_rows}; no sequence-sample crosswalk assumed",
        ),
        (
            "ESS_DIVE_gapfilled_tower_flux_source_context",
            "pass" if tower_rows and tower_valid else "blocked",
            (
                f"staged half-hourly gap-filled tower rows={tower_rows}; valid CH4 rows={tower_valid}; "
                "2015-2016 temporal overlap remains unlinked to sequencing samples"
            ),
        ),
        (
            "NCBI_BioProject_sequence_project_crosswalk",
            "partial" if bioproject_mapped else "blocked",
            f"authoritative title-suffix sequence-project links={bioproject_mapped}/{bioproject_rows}",
        ),
        (
            "JGI_Sample_QC_identity_crosswalk",
            "partial" if jgi_final_portals else "blocked",
            (
                f"final JGI delivery portals={jgi_final_portals}/{jgi_rows}; "
                f"exact JGI Sample QC identity links={jgi_exact}/{jgi_rows}; "
                "Sample QC receipt timestamp is not collection datetime, depth, environment, or flux evidence"
            ),
        ),
        (
            "JGI_Data_Portal_catalog_identity_crosswalk",
            "partial"
            if jgi_data_rows == 133
            and jgi_data_exact == 107
            and jgi_data_aliases == 27
            and jgi_data_no_depth == 107
            else "blocked",
            (
                f"exact source-label-specific JGI Data Portal record pairs={jgi_data_exact}/"
                f"{jgi_data_rows}; controlled July-to-Jul aliases={jgi_data_aliases}; "
                f"catalog depth not reported={jgi_data_no_depth}; catalog/processing evidence is not "
                "collection-time, environmental, or flux evidence"
            ),
        ),
        (
            "NCBI_SRA_sample_identity_and_collection_date_crosswalk",
            "partial" if sra_exact else "blocked",
            (
                f"exact entity-validated NCBI SRA packages={sra_exact}/{sra_rows}; "
                f"exact collection dates={sra_dates}; unresolved labels={sra_unresolved}; "
                "no NCBI SRA record reports a usable depth value or flux/environmental crosswalk"
            ),
        ),
        (
            "NCBI_SRA_declared_library_strategy_reconciliation",
            "partial" if sra_rna_seq else "blocked",
            (
                f"declared RNA-Seq packages={sra_rna_seq}; declared WGS packages={sra_wgs}; "
                "declared strategy is source metadata only and WGS packages cannot be silently pooled "
                "with RNA-Seq expression for ecological validation"
            ),
        ),
        (
            "published_accession_roster_recovery",
            "pass" if table_s1_status == "parseable_xlsx_container" else "blocked",
            f"published Table S1-S13 source status={table_s1_status}",
        ),
        (
            "published_methods_design_context",
            "partial" if methods_rows == len(ecological) else "blocked",
            (
                f"methods-derived context rows={methods_rows}/{len(ecological)}; direct coded-depth rows={methods_direct}; "
                f"D6/D5 raw-code reconciliation pending={methods_d6_pending}; legacy label-only rows={methods_legacy}; "
                "context is not an exact environmental or flux join"
            ),
        ),
        (
            "authoritative_sequence_to_ecology_crosswalk_contract",
            "pass"
            if authoritative_ready_samples == len(ecological) and len(ecological)
            else "partial"
            if authoritative_ready_samples
            else "ready",
            (
                f"authoritative link rows={authoritative_link_rows}; validated link rows={authoritative_valid_links}; "
                f"readiness rows={authoritative_readiness_rows}; samples eligible for grouped ecological validation="
                f"{authoritative_ready_samples}/{len(ecological)}; only an author/publisher-supplied exact mapping "
                "may change this gate"
            ),
        ),
        (
            "exact_sample_depth_environment_flux_join",
            "pass"
            if authoritative_ready_samples == len(ecological) and len(ecological)
            else "partial"
            if authoritative_ready_samples
            else "blocked",
            (
                f"samples with validated exact crosswalks={authoritative_ready_samples}/{len(ecological)}; "
                "eligibility is not ecological-model validation, a MAG-level flux effect, or final MRV"
            ),
        ),
        (
            "final_MRV_risk_and_flux_claim_lock",
            "pass",
            "no final A-E tier, measured flux, crediting, or transfer claim is authorized",
        ),
    ]
    frame = pd.DataFrame(rows, columns=["gate", "status", "detail"])
    frame.insert(0, "lane_id", LANE_ID)
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def build_candidate_cards(
    readiness: pd.DataFrame,
    embedding: pd.DataFrame,
    neighbor: pd.DataFrame,
    top_cards: int,
) -> pd.DataFrame:
    frame = readiness.merge(
        embedding[
            [
                "proteome_id",
                "esm2_embedding_status",
                "glm2_context_status",
                "embedding_coverage_class",
                "permutation_p",
            ]
        ],
        on="proteome_id",
        how="left",
    )
    if not neighbor.empty:
        keep = [column for column in neighbor.columns if column.startswith("nearest_")]
        frame = frame.merge(
            neighbor[["proteome_id", *keep]], on="proteome_id", how="left"
        )
    frame["review_priority_numeric"] = pd.to_numeric(
        frame.get("review_priority_score"), errors="coerce"
    ).fillna(-np.inf)
    frame = frame.loc[
        frame["esm2_embedding_status"].eq("complete_finite_embedding")
    ].copy()
    frame = frame.sort_values("review_priority_numeric", ascending=False).head(
        top_cards
    )
    frame.insert(
        0,
        "card_id",
        [f"mucc_v1_promoted_reference_{i:03d}" for i in range(1, len(frame) + 1)],
    )
    frame.insert(1, "candidate_set", "promoted_wetland_molecular_reference_review")
    frame["promotion_status"] = "reviewable_molecular_reference_candidate"
    frame["next_validation_action"] = (
        "Review source-aware neighbours alongside curated mechanism features; require sample/depth "
        "and flux joins before any ecological or MRV-risk interpretation."
    )
    frame["claim_boundary"] = CLAIM_BOUNDARY
    columns = [
        "card_id",
        "candidate_set",
        "promotion_status",
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
        "processed_mag_expression_support",
        "processed_mag_expression_occupancy_fraction",
        "expression_support_label",
        "esm2_embedding_status",
        "glm2_context_status",
        "embedding_coverage_class",
        "permutation_p",
        "review_priority_score",
        "next_validation_action",
        "claim_boundary",
    ]
    columns.extend(column for column in frame.columns if column.startswith("nearest_"))
    return frame[[column for column in columns if column in frame.columns]]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    esm2_dirs = [
        resolve(repo_root, path)
        for path in (args.esm2_artifact_dir or DEFAULT_ESM2_DIRS)
    ]
    glm2_dir = resolve(repo_root, args.glm2_dir)
    neighbor_summary = resolve(repo_root, args.neighbor_summary)
    network_manifest = resolve(repo_root, args.network_manifest)

    catalog = read_tsv(run_dir / "manifests/mucc_v1_mag_catalog_full.tsv")
    manifest = read_tsv(run_dir / "manifests/mucc_v1_source_lane_manifest.tsv")
    expression = read_tsv(
        run_dir / "expression/feature_mucc_v1_expression_mag_summary.tsv"
    )
    readiness = read_tsv(
        run_dir / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv"
    )
    samples = read_tsv(
        run_dir / "environmental_metadata/mucc_v1_sample_columns_scaffold.tsv"
    )
    sample_readiness = read_tsv(
        run_dir / "environmental_metadata/feature_sample_risk_readiness_scaffold.tsv"
    )
    kbase_catalog_path = (
        run_dir / "source_audit/kbase_public_workspace_147022/"
        "mucc_v1_kbase_public_catalog_reconciliation.tsv"
    )
    kbase_catalog = (
        read_tsv(kbase_catalog_path) if kbase_catalog_path.is_file() else pd.DataFrame()
    )
    source_qc_path = (
        run_dir / "functional_features/feature_mucc_v1_zenodo_source_qc.tsv"
    )
    if not source_qc_path.is_file():
        raise FileNotFoundError(
            "record-specific Zenodo source QC reconciliation is required for promotion"
        )
    source_qc = read_tsv(source_qc_path)
    if source_qc["mag_id"].duplicated().any() or len(source_qc) != len(catalog):
        raise ValueError(
            "Zenodo source QC reconciliation is not a unique full MAG-catalog coverage"
        )
    glm2_path = glm2_dir / "features/glm2_multiwindow_summary.parquet"
    glm2_report = glm2_dir / "validation/glm2_multiwindow_reduce_report.json"
    if not glm2_path.is_file() or not glm2_report.is_file():
        raise FileNotFoundError("gLM2 summary/report are required for promotion")
    if json.loads(glm2_report.read_text()).get("status") != "pass":
        raise ValueError("gLM2 reduce report is not passing")
    glm2 = pd.read_parquet(glm2_path)
    if glm2["proteome_id"].duplicated().any() or len(glm2) != len(manifest):
        raise ValueError("gLM2 summary is not a unique, full MUCC manifest coverage")
    esm2 = load_esm2(repo_root, esm2_dirs)
    embedding = build_embedding_status(manifest, esm2, glm2)
    if len(embedding) != len(manifest) or embedding["proteome_id"].duplicated().any():
        raise ValueError(
            "embedding status did not preserve the complete unique MUCC manifest"
        )

    reconciliation_dir = run_dir / "reconciliation"
    feature_dir = run_dir / "functional_features"
    environmental_dir = run_dir / "environmental_metadata"
    report_dir = run_dir / "reports"
    card_dir = run_dir / "candidate_cards"
    write_tsv(feature_dir / "feature_mucc_v1_embedding_status.tsv", embedding)
    glm2_feature = glm2.copy()
    glm2_feature.insert(0, "lane_id", LANE_ID)
    glm2_feature["glm2_context_status"] = "complete_multiwindow_stability_summary"
    glm2_feature["claim_boundary"] = CLAIM_BOUNDARY
    glm2_feature.to_parquet(
        feature_dir / "feature_mucc_v1_glm2_context.parquet", index=False
    )
    reconciliation = denominator_reconciliation(
        repo_root,
        catalog,
        manifest,
        expression,
        esm2,
        glm2,
        samples,
        kbase_catalog,
        source_qc,
    )
    write_tsv(
        reconciliation_dir / "mucc_v1_denominator_reconciliation.tsv", reconciliation
    )
    mag_reconciliation = embedding[
        [
            "lane_id",
            "proteome_id",
            "mag_id",
            "esm2_embedding_status",
            "glm2_context_status",
            "embedding_coverage_class",
            "claim_boundary",
        ]
    ].copy()
    archive_columns = [
        column
        for column in ["mag_id", "archive_member", "zip_crc", "source_mag_fasta_status"]
        if column in catalog.columns
    ]
    mag_reconciliation = mag_reconciliation.merge(
        catalog[archive_columns], on="mag_id", how="left", validate="one_to_one"
    )
    mag_reconciliation["archive_roster_membership_status"] = np.where(
        mag_reconciliation["archive_member"].notna(),
        "checksum_validated_Zenodo_MAGs_zip_member",
        "unexpected_missing_from_checksum_validated_archive_roster",
    )
    source_qc_columns = [
        "mag_id",
        "source_annotation_rows",
        "source_qc_mapping_status",
        "bin_completeness",
        "bin_contamination",
        "source_qc_value_consistency_status",
        "source_qc_evidence_status",
        "published_mq_hq_membership_status",
        "published_quality_definition",
        "quality_definition_source",
        "source_record_doi",
    ]
    mag_reconciliation = mag_reconciliation.merge(
        source_qc[
            [column for column in source_qc_columns if column in source_qc.columns]
        ],
        on="mag_id",
        how="left",
        validate="one_to_one",
    )
    mag_reconciliation = mag_reconciliation.rename(
        columns={
            "published_mq_hq_membership_status": "published_hqmq_membership_status"
        }
    )
    mag_reconciliation["published_hqmq_headline_denominator"] = PUBLISHED_HQMQ_MAGS
    mag_reconciliation["local_MAGs_zip_denominator"] = len(catalog)
    if not kbase_catalog.empty:
        kbase_columns = [
            "mag_id",
            "kbase_roster_reconciliation_status",
            "kbase_genomeset_membership_ref",
            "kbase_genome_latest_ref",
            "kbase_latest_object_version",
            "kbase_gtdb_source_version",
            "kbase_gtdb_lineage",
            "source_bin_taxonomy",
            "taxonomy_reconciliation_status",
            "kbase_genome_quality_fields_status",
        ]
        mag_reconciliation = mag_reconciliation.merge(
            kbase_catalog[
                [column for column in kbase_columns if column in kbase_catalog.columns]
            ],
            on="mag_id",
            how="left",
            validate="one_to_one",
        )
    taxonomy_projection_path = (
        run_dir / "functional_features/feature_mucc_v1_taxonomy_projection.tsv"
    )
    taxonomy_projection = (
        read_tsv(taxonomy_projection_path)
        if taxonomy_projection_path.is_file()
        else pd.DataFrame()
    )
    if not taxonomy_projection.empty:
        taxonomy_columns = [
            "mag_id",
            "atlas_taxonomy_lineage",
            "atlas_taxonomy_projection_status",
            "source_kbase_rank_disagreement_count",
            "kbase_rank_fallback_count",
            "atlas_taxonomy_available_rank_count",
        ]
        mag_reconciliation = mag_reconciliation.merge(
            taxonomy_projection[
                [
                    column
                    for column in taxonomy_columns
                    if column in taxonomy_projection.columns
                ]
            ],
            on="mag_id",
            how="left",
            validate="one_to_one",
        )
    write_tsv(reconciliation_dir / "mucc_v1_mag_reconciliation.tsv", mag_reconciliation)

    ecological = build_sample_ecological_readiness(samples, sample_readiness, run_dir)
    write_tsv(
        environmental_dir / "feature_mucc_v1_sample_ecological_readiness.tsv",
        ecological,
    )
    write_tsv(
        environmental_dir / "fact_mucc_v1_external_source_registry.tsv",
        external_source_registry(run_dir),
    )
    if neighbor_summary.is_file() and neighbor_summary.stat().st_size:
        neighbor = read_tsv(neighbor_summary)
    else:
        neighbor = pd.DataFrame(columns=["proteome_id"])
    cards = build_candidate_cards(readiness, embedding, neighbor, args.top_cards)
    write_tsv(card_dir / "mucc_v1_promoted_molecular_reference_cards.tsv", cards)
    gates = promotion_gates(
        repo_root,
        run_dir,
        catalog,
        esm2,
        glm2,
        source_qc,
        neighbor_summary,
        network_manifest,
        ecological,
    )
    write_tsv(report_dir / "mucc_v1_integrated_atlas_promotion_gates.tsv", gates)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "promotion_state": "molecular_reference_promoted_ecological_validation_pending",
        "published_hqmq_MAG_headline": PUBLISHED_HQMQ_MAGS,
        "local_MAGs_zip_entries": int(len(catalog)),
        "direct_Zenodo_source_QC_MAGs": int(len(source_qc)),
        "published_HQMQ_MAGs_reconciled_from_direct_Zenodo_QC": int(
            source_qc["published_mq_hq_membership_status"]
            .eq("meets_published_MQHQ_CheckM_threshold")
            .sum()
        ),
        "archive_MAGs_outside_published_HQMQ_QC_scope": int(
            source_qc["published_mq_hq_membership_status"]
            .eq("does_not_meet_published_MQHQ_CheckM_threshold")
            .sum()
        ),
        "public_KBase_GenomeSet_exact_MAG_ID_members": int(
            kbase_catalog["kbase_roster_reconciliation_status"]
            .eq("exact_MAG_id_match_public_KBase_GenomeSet")
            .sum()
        )
        if not kbase_catalog.empty
        else 0,
        "Zenodo_archive_MAGs_absent_from_public_KBase": int(
            kbase_catalog["kbase_roster_reconciliation_status"]
            .eq("Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet")
            .sum()
        )
        if not kbase_catalog.empty
        else 0,
        "atlas_taxonomy_available_MAGs": int(
            taxonomy_projection["atlas_taxonomy_lineage"].ne("").sum()
        )
        if not taxonomy_projection.empty
        else 0,
        "atlas_taxonomy_unavailable_in_both_sources_MAGs": int(
            taxonomy_projection["atlas_taxonomy_projection_status"]
            .eq("taxonomy_unavailable_in_source_and_KBase")
            .sum()
        )
        if not taxonomy_projection.empty
        else 0,
        "esm2_completed_finite_MAGs": int(len(esm2)),
        "glm2_completed_MAGs": int(len(glm2)),
        "expression_supported_MAGs": int(len(expression)),
        "sample_scaffold_rows": int(len(ecological)),
        "source_aware_neighbor_rows": int(len(neighbor)),
        "promoted_candidate_cards": int(len(cards)),
        "ESS_DIVE_chamber_flux_source_rows": staged_essdive_counts(run_dir)[0],
        "ESS_DIVE_porewater_source_rows": staged_essdive_counts(run_dir)[1],
        "ESS_DIVE_gapfilled_tower_flux_source_rows": staged_essdive_gapfilled_tower_counts(
            run_dir
        )[0],
        "ESS_DIVE_gapfilled_tower_flux_valid_CH4_rows": staged_essdive_gapfilled_tower_counts(
            run_dir
        )[1],
        "NCBI_BioProject_crosswalk_rows": staged_bioproject_crosswalk_counts(run_dir)[
            0
        ],
        "NCBI_BioProject_mapped_samples": staged_bioproject_crosswalk_counts(run_dir)[
            1
        ],
        "NCBI_SRA_crosswalk_rows": staged_sra_sample_crosswalk_counts(run_dir)[0],
        "NCBI_SRA_exact_package_matches": staged_sra_sample_crosswalk_counts(run_dir)[
            1
        ],
        "NCBI_SRA_exact_collection_dates": staged_sra_sample_crosswalk_counts(run_dir)[
            2
        ],
        "NCBI_SRA_declared_RNA_Seq_packages": staged_sra_library_strategy_counts(
            run_dir
        )[0],
        "NCBI_SRA_declared_WGS_packages": staged_sra_library_strategy_counts(run_dir)[
            1
        ],
        "JGI_Data_Portal_crosswalk_rows": staged_jgi_data_portal_crosswalk_counts(
            run_dir
        )[0],
        "JGI_Data_Portal_exact_record_pairs": staged_jgi_data_portal_crosswalk_counts(
            run_dir
        )[1],
        "JGI_Data_Portal_controlled_July_to_Jul_aliases": staged_jgi_data_portal_crosswalk_counts(
            run_dir
        )[2],
        "JGI_Data_Portal_depth_not_reported": staged_jgi_data_portal_crosswalk_counts(
            run_dir
        )[3],
        "authoritative_ecology_crosswalk_link_rows": staged_authoritative_ecological_crosswalk_counts(
            run_dir
        )[0],
        "authoritative_ecology_validated_mapping_rows": staged_authoritative_ecological_crosswalk_counts(
            run_dir
        )[1],
        "authoritative_ecology_ready_samples": staged_authoritative_ecological_crosswalk_counts(
            run_dir
        )[3],
        "ecological_validation_state": (
            "all_samples_eligible_for_grouped_ecological_validation_not_final_MRV"
            if staged_authoritative_ecological_crosswalk_counts(run_dir)[3]
            == len(ecological)
            and len(ecological)
            else "some_samples_eligible_for_grouped_ecological_validation_not_final_MRV"
            if staged_authoritative_ecological_crosswalk_counts(run_dir)[3]
            else "source_observations_staged_pending_exact_sample_depth_environment_and_flux_joins"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (report_dir / "mucc_v1_integrated_atlas_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
