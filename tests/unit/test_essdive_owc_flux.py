from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


def _load_script(name: str, relative_path: str) -> object:
    repo_root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location(name, repo_root / relative_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_workbook(path: Path) -> None:
    chamber = pd.DataFrame(
        [
            {
                "LOC_LATITUDE": 41.2,
                "LOC_LONGITUDE": -83.1,
                "CMB_DATE": 202001011200,
                "CMB_VEGTYPE": "Typha",
                "CMB_FCH4": 12.5,
                "CMB_FCH4_Flag": 0,
                "CMB_COMMENT": "OW1",
                "CMB_SPP": "Typha angustifolia",
                "CMB_APPROACH": "chamber",
            },
            {
                "LOC_LATITUDE": 41.2,
                "LOC_LONGITUDE": -83.1,
                "CMB_DATE": 202001011300,
                "CMB_VEGTYPE": "Typha",
                "CMB_FCH4": -9999,
                "CMB_FCH4_Flag": 1,
                "CMB_COMMENT": "OW1",
            },
        ]
    )
    peeper = pd.DataFrame(
        [
            ["collection date", "CH4 concentration OW1 level 1"],
            ["", ""],
            ["SOIL_H2O_DATE", "SOIL_H2O_CH4_1_1_1"],
            [20200101, 0.8],
        ]
    )
    locations = pd.DataFrame(
        [
            {
                "LOC_VARIABLE": "SOIL_H2O_CH4_1_1_1",
                "COMMENT": "Peeper code: OW1",
                "LOC_HEIGHT": 0.1,
                "LOC_LATITUDE": 41.2,
                "LOC_LONGITUDE.": -83.1,
                "PROFILE_ZERO_REF": "Top of mineral soil",
            }
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        chamber.to_excel(writer, sheet_name="Chamber", index=False, startrow=2)
        peeper.to_excel(writer, sheet_name="Peeper", index=False, header=False)
        locations.to_excel(writer, sheet_name="Peeper Locations", index=False)


def test_stage_essdive_preserves_source_grain_and_unlinked_status(
    tmp_path: Path,
) -> None:
    module = _load_script(
        "stage_essdive_owc_flux",
        "scripts/external/stage_essdive_owc_flux.py",
    )
    workbook = tmp_path / "essdive.xlsx"
    _write_workbook(workbook)

    chamber = module.chamber_flux_table(workbook)
    porewater = module.porewater_ch4_table(workbook)

    assert len(chamber) == 2
    assert chamber.loc[0, "methane_flux_nmol_m2_s"] == 12.5
    assert chamber.loc[0, "source_value_status"] == "reported_valid"
    assert chamber.loc[1, "source_value_status"] == "source_missing_sentinel"
    assert set(chamber["sample_join_status"]) == {
        "unlinked_no_authoritative_sequence_sample_crosswalk"
    }
    assert len(porewater) == 1
    assert porewater.loc[0, "peeper_code"] == "OW1"
    assert porewater.loc[0, "depth_cm_relative_to_top_mineral_soil"] == 10.0
    assert porewater.loc[0, "porewater_ch4_mM"] == 0.8
    assert porewater.loc[0, "sample_join_status"] == (
        "unlinked_no_authoritative_sequence_sample_crosswalk"
    )


def test_source_recovery_rejects_truncated_xlsx_payload(tmp_path: Path) -> None:
    module = _load_script(
        "build_mucc_v1_source_recovery_ledger",
        "scripts/reports/build_mucc_v1_source_recovery_ledger.py",
    )
    malformed = tmp_path / "truncated.xlsx"
    malformed.write_bytes(b"PK\x03\x04not-a-complete-zip")

    status, detail = module.validate_xlsx_container(malformed)

    assert status == "malformed_no_central_directory"
    assert "central directory" in detail.lower()


def test_publisher_listing_size_evidence_does_not_validate_a_workbook(
    tmp_path: Path,
) -> None:
    module = _load_script(
        "build_mucc_v1_source_recovery_ledger_publisher_size",
        "scripts/reports/build_mucc_v1_source_recovery_ledger.py",
    )
    malformed = tmp_path / "publisher-sized-malformed.xlsx"
    malformed.write_bytes(b"x" * module.PUBLISHER_LISTED_SUPPLEMENT_TABLE_BYTES)

    matches, detail = module.publisher_listing_size_evidence(malformed)

    assert matches is True
    assert "does not repair" in detail
    status, _ = module.validate_xlsx_container(malformed)
    assert status == "malformed_not_zip_container"


def test_methods_design_context_keeps_d6_reconciliation_explicit() -> None:
    module = _load_script(
        "build_mucc_v1_source_recovery_ledger_methods_context",
        "scripts/reports/build_mucc_v1_source_recovery_ledger.py",
    )
    samples = [
        {
            "sample_id": "aug",
            "source_sample_column": "Aug_M1_C1_D5_A",
            "collection_year": "2018",
            "month_label": "Aug",
            "site_or_landcover": "M1",
            "core": "C1",
            "depth_code": "D5",
            "replicate": "A",
        },
        {
            "sample_id": "july_direct",
            "source_sample_column": "July_N3_C1_D3_A",
            "collection_year": "2018",
            "month_label": "July",
            "site_or_landcover": "N3",
            "core": "C1",
            "depth_code": "D3",
            "replicate": "A",
        },
        {
            "sample_id": "july_d6",
            "source_sample_column": "July_N3_C1_D6_A",
            "collection_year": "2018",
            "month_label": "July",
            "site_or_landcover": "N3",
            "core": "C1",
            "depth_code": "D6",
            "replicate": "A",
        },
        {
            "sample_id": "legacy",
            "source_sample_column": "AugMudDeep1_2015",
            "collection_year": "2015",
            "month_label": "Aug",
            "site_or_landcover": "mud",
            "core": "",
            "depth_code": "deep",
            "replicate": "1",
        },
    ]

    context = module.methods_design_context_rows(samples)

    assert context[0]["nominal_depth_interval_cm"] == "20-25"
    assert context[0]["methods_depth_assignment_status"] == (
        "methods_design_direct_5cm_depth_code"
    )
    assert context[1]["nominal_depth_interval_cm"] == "10-15"
    assert context[2]["nominal_depth_interval_cm"] == "20-25"
    assert context[2]["methods_design_context_status"] == (
        "validated_2018_cohort_but_raw_depth_code_reconciliation_pending"
    )
    assert context[2]["environment_flux_join_status"].startswith("blocked_")
    assert context[3]["nominal_depth_interval_cm"] == ""
    assert (
        context[3]["methods_design_context_status"]
        == "legacy_sample_label_context_only"
    )


def test_source_recovery_validates_zenodo_archive_inventory(tmp_path: Path) -> None:
    module = _load_script(
        "build_mucc_v1_source_recovery_ledger_inventory",
        "scripts/reports/build_mucc_v1_source_recovery_ledger.py",
    )
    catalog = tmp_path / "catalog.tsv"
    catalog.write_text(
        "mag_id\tarchive_member\tzip_crc\tsource_mag_fasta_status\n"
        "OWC_0001\tMAGs/OWC_0001.fa.gz\t2c95ae66\tdownloaded_validated_in_MAGs.zip\n"
    )

    status, detail, count = module.validate_zenodo_archive_roster(catalog)

    assert status == "archive_roster_recovered_checksum_validated"
    assert count == 1
    assert "unique MAG archive members" in detail


def test_source_recovery_validates_exact_zenodo_hqmq_qc_scope(tmp_path: Path) -> None:
    module = _load_script(
        "build_mucc_v1_source_recovery_ledger_qc",
        "scripts/reports/build_mucc_v1_source_recovery_ledger.py",
    )
    qc_path = tmp_path / "source_qc.tsv"
    rows = [
        "mag_id\tbin_completeness\tbin_contamination\tsource_qc_value_consistency_status\tpublished_mq_hq_membership_status\n"
    ]
    rows.extend(
        f"OWC_{index:04d}\t50\t9\tdirect_source_qc_values_consistent_across_annotation_rows\tmeets_published_MQHQ_CheckM_threshold\n"
        for index in range(2502)
    )
    rows.extend(
        f"OUT_{index}\t50\t10\tdirect_source_qc_values_consistent_across_annotation_rows\tdoes_not_meet_published_MQHQ_CheckM_threshold\n"
        for index in range(6)
    )
    qc_path.write_text("".join(rows))

    status, detail, _, _, counts = module.zenodo_source_qc_evidence(qc_path)

    assert status == (
        "exact_published_2502_HQMQ_and_six_archive_scope_difference_reconciled"
    )
    assert "2,502" in detail
    assert counts == {
        "rows": 2508,
        "hqmq": 2502,
        "archive_scope": 6,
        "consistent": 2508,
    }


def test_kbase_public_catalog_reconciliation_preserves_quality_boundary() -> None:
    module = _load_script(
        "stage_mucc_v1_kbase_public_catalog",
        "scripts/external/stage_mucc_v1_kbase_public_catalog.py",
    )
    catalog = [
        {
            "mag_id": "OWC_0001",
            "proteome_id": "mucc_v1__OWC_0001",
            "archive_member": "MAGs/OWC_0001.fa.gz",
        },
        {
            "mag_id": "OWC_0002",
            "proteome_id": "mucc_v1__OWC_0002",
            "archive_member": "MAGs/OWC_0002.fa.gz",
        },
    ]
    annotations = {
        "OWC_0001": {"bin_taxonomy": "d__Bacteria;p__Example"},
        "OWC_0002": {"bin_taxonomy": "d__Archaea;p__SourceOnly"},
    }
    genome_info = [
        17,
        "OWC_0001.fa_genome",
        "KBaseGenomes.Genome-11.1",
        "2024-04-01T00:00:00+0000",
        2,
        "kbase_user",
        147022,
        "public_workspace",
        "checksum",
        100,
        {"GTDB_lineage": "d__Bacteria;p__Example", "GTDB_source_ver": "214.1"},
    ]

    rows = module.reconciliation_rows(
        catalog,
        annotations,
        [genome_info],
        {"OWC_0001.fa_genome": {"ref": "147022/17/1"}},
        147022,
        "public_workspace",
        "147022/7498/2",
    )

    matched, absent = rows
    assert matched["kbase_roster_reconciliation_status"] == (
        "exact_MAG_id_match_public_KBase_GenomeSet"
    )
    assert matched["taxonomy_reconciliation_status"] == (
        "source_and_KBase_Gtdb_taxonomy_exact"
    )
    assert matched["kbase_genome_quality_fields_status"] == (
        "no_completeness_contamination_CheckM_quality_or_N50_metadata"
    )
    assert matched["published_hqmq_membership_status"] == (
        "unresolved_do_not_infer_from_KBase_membership"
    )
    assert absent["kbase_roster_reconciliation_status"] == (
        "Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet"
    )
    assert absent["published_hqmq_membership_status"] == (
        "unresolved_do_not_infer_from_KBase_absence"
    )


def test_taxonomy_projection_uses_source_rank_then_kbase_fallback() -> None:
    module = _load_script(
        "build_mucc_v1_taxonomy_projection",
        "scripts/reports/build_mucc_v1_taxonomy_projection.py",
    )
    rows = module.build_rows(
        [
            {
                "lane_id": "mucc_v1_owc_wetland",
                "proteome_id": "mucc_v1__OWC_0001",
                "mag_id": "OWC_0001",
                "source_bin_taxonomy": "d__Bacteria;p__SourcePhylum;g__",
                "kbase_gtdb_lineage": "d__Bacteria;p__KBasePhylum;g__KBaseGenus",
                "kbase_gtdb_source_version": "214.1",
                "kbase_roster_reconciliation_status": (
                    "exact_MAG_id_match_public_KBase_GenomeSet"
                ),
                "taxonomy_reconciliation_status": (
                    "source_and_KBase_Gtdb_taxonomy_differ"
                ),
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["atlas_phylum"] == "p__SourcePhylum"
    assert row["atlas_phylum_provenance"] == "source_annotation_primary"
    assert row["atlas_genus"] == "g__KBaseGenus"
    assert row["atlas_genus_provenance"] == "KBase_GTDB_214_1_fallback"
    assert row["source_kbase_rank_disagreement_count"] == "1"
    assert row["kbase_rank_fallback_count"] == "1"
    assert row["atlas_taxonomy_projection_status"] == (
        "source_primary_with_explicit_KBase_rank_disagreements"
    )


def test_flashweave_explorer_keeps_one_unambiguous_atlas_context_per_endpoint() -> None:
    module = _load_script(
        "build_mucc_v1_flashweave_atlas_explorer",
        "scripts/reports/build_mucc_v1_flashweave_atlas_explorer.py",
    )
    edges = pd.DataFrame(
        [
            {
                "lane_id": "lane",
                "analysis_id": "flashweave_direct_association",
                "source_proteome_id": "mucc_v1__OWC_0001",
                "target_proteome_id": "mucc_v1__OWC_0002",
                "association_weight": "0.8",
                "association_sign": "positive",
                "absolute_association_weight": "0.8",
                "edge_directionality": "undirected",
                "fdr_controlled_during_inference": "true",
                "edge_level_q_value": "not_emitted_by_flashweave_edgelist",
                "selection_count": "15",
                "iterations": "20",
                "selection_frequency": "0.75",
                "stability_class": "stable_at_or_above_threshold",
            }
        ]
    )
    nodes = pd.DataFrame(
        [
            {
                "proteome_id": "mucc_v1__OWC_0001",
                "mag_id": "OWC_0001",
                "atlas_taxonomy_lineage": "d__Archaea;p__Halobacteriota",
                "atlas_taxonomy_projection_status": "source_primary",
                "atlas_domain": "d__Archaea",
                "atlas_phylum": "p__Halobacteriota",
                "atlas_class": "c__Methanosarcinia",
                "atlas_taxonomy_available": "true",
                "source_kbase_rank_disagreement_count": "1",
                "kbase_rank_fallback_count": "0",
                "marker_breadth_count": "4",
                "methane_term_rows": "10",
            },
            {
                "proteome_id": "mucc_v1__OWC_0002",
                "mag_id": "OWC_0002",
                "atlas_taxonomy_lineage": "d__Bacteria;p__Desulfobacterota",
                "atlas_taxonomy_projection_status": "source_primary",
                "atlas_domain": "d__Bacteria",
                "atlas_phylum": "p__Desulfobacterota",
                "atlas_class": "c__Desulfovibrionia",
                "atlas_taxonomy_available": "true",
                "source_kbase_rank_disagreement_count": "0",
                "kbase_rank_fallback_count": "1",
                "marker_breadth_count": "2",
                "methane_term_rows": "0",
            },
        ]
    )

    context = module.edge_context(edges, nodes)

    row = context.iloc[0]
    assert row["source_mag_id"] == "OWC_0001"
    assert row["target_mag_id"] == "OWC_0002"
    assert row["network_explorer_visibility_status"] == (
        "stability_and_taxonomy_filter_eligible"
    )
    assert row["endpoint_taxonomy_conflict_exposure"] == "true"
    assert "source_mag_id_x" not in context.columns
    assert "stability_class_x" not in context.columns


def test_sra_crosswalk_accepts_exact_owc_package_and_rejects_entity_conflict() -> None:
    module = _load_script(
        "stage_mucc_v1_ncbi_sra_sample_crosswalk",
        "scripts/external/stage_mucc_v1_ncbi_sra_sample_crosswalk.py",
    )
    payload = b"""
    <EXPERIMENT_PACKAGE_SET>
      <EXPERIMENT_PACKAGE>
        <EXPERIMENT accession="SRX1">
          <TITLE>Old Woman Creek 2018 metatranscriptomes Jul_M1_C1_D1_A</TITLE>
          <STUDY_REF accession="SRP1"><IDENTIFIERS>
            <EXTERNAL_ID namespace="BioProject">PRJNA1</EXTERNAL_ID>
          </IDENTIFIERS></STUDY_REF>
          <LIBRARY_DESCRIPTOR>
            <LIBRARY_STRATEGY>RNA-Seq</LIBRARY_STRATEGY>
            <LIBRARY_SOURCE>METATRANSCRIPTOMIC</LIBRARY_SOURCE>
            <LIBRARY_LAYOUT><PAIRED/></LIBRARY_LAYOUT>
          </LIBRARY_DESCRIPTOR>
        </EXPERIMENT>
        <STUDY accession="SRP1"><IDENTIFIERS>
          <EXTERNAL_ID namespace="BioProject">PRJNA1</EXTERNAL_ID>
        </IDENTIFIERS></STUDY>
        <SAMPLE accession="SRS1"><IDENTIFIERS>
          <EXTERNAL_ID namespace="BioSample">SAMN1</EXTERNAL_ID>
        </IDENTIFIERS><TITLE>Old Woman Creek sample - Jul_M1_C1_D1_A</TITLE>
          <SAMPLE_ATTRIBUTES>
            <SAMPLE_ATTRIBUTE><TAG>collection_date</TAG>
              <VALUE>2018-07-11</VALUE></SAMPLE_ATTRIBUTE>
            <SAMPLE_ATTRIBUTE><TAG>geo_loc_name</TAG>
              <VALUE>USA: Ohio</VALUE></SAMPLE_ATTRIBUTE>
            <SAMPLE_ATTRIBUTE><TAG>isolation_source</TAG>
              <VALUE>freshwater estuary soil</VALUE></SAMPLE_ATTRIBUTE>
          </SAMPLE_ATTRIBUTES>
        </SAMPLE>
        <RUN_SET><RUN accession="SRR1" published="2020-10-28 00:36:00"/></RUN_SET>
      </EXPERIMENT_PACKAGE>
      <EXPERIMENT_PACKAGE>
        <EXPERIMENT accession="SRX2"><TITLE>Jul_M1_C1_D1_A</TITLE></EXPERIMENT>
        <SAMPLE accession="SRS2"><TITLE>Jul_M1_C1_D1_A</TITLE>
          <SAMPLE_ATTRIBUTES>
            <SAMPLE_ATTRIBUTE><TAG>geo_loc_name</TAG>
              <VALUE>United Kingdom: Norwich</VALUE></SAMPLE_ATTRIBUTE>
            <SAMPLE_ATTRIBUTE><TAG>isolation_source</TAG>
              <VALUE>plant leaves</VALUE></SAMPLE_ATTRIBUTE>
          </SAMPLE_ATTRIBUTES>
        </SAMPLE>
      </EXPERIMENT_PACKAGE>
    </EXPERIMENT_PACKAGE_SET>
    """

    record, method, status = module.select_exact_package(payload, "July_M1_C1_D1_A")

    assert status == "exact_source_label_to_NCBI_SRA_package"
    assert method == "deterministic_July_to_Jul"
    assert record is not None
    assert record["sra_run_accessions"] == "SRR1"
    assert record["sra_collection_date"] == "2018-07-11"
    assert record["sra_collection_date_status"] == (
        "exact_collection_date_from_NCBI_SRA_sample_attributes"
    )


def test_flashweave_stability_sampling_preserves_each_scaffold_stratum() -> None:
    module = _load_script(
        "run_mucc_v1_flashweave_stability",
        "scripts/reports/run_mucc_v1_flashweave_stability.py",
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "month_label": ["Aug", "Aug", "Aug", "Jul"],
            "site_or_landcover": ["M1", "M1", "M1", "M2"],
            "depth_context_code": ["coded_D1", "coded_D1", "coded_D2", "coded_D1"],
        }
    )

    selected = module.stratified_sample_indices(
        metadata,
        0.5,
        module.np.random.default_rng(7),
    )

    selected_strata = set(
        metadata.iloc[selected][
            ["month_label", "site_or_landcover", "depth_context_code"]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    all_strata = set(
        metadata[["month_label", "site_or_landcover", "depth_context_code"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    assert selected_strata == all_strata


def test_bioproject_crosswalk_preserves_unmatched_samples() -> None:
    module = _load_script(
        "stage_mucc_v1_ncbi_bioproject_crosswalk",
        "scripts/external/stage_mucc_v1_ncbi_bioproject_crosswalk.py",
    )
    samples = pd.DataFrame(
        {
            "sample_id": [
                "owc_expr__Aug_M1_C1_D1_A",
                "owc_expr__July_M1_C1_D1_A",
                "old",
            ],
            "source_sample_column": ["Aug_M1_C1_D1_A", "July_M1_C1_D1_A", "No_match"],
        }
    )
    records = [
        {
            "uid": "1",
            "project_acc": "PRJNA1",
            "project_title": "OWC metatranscriptome - Aug_M1_C1_D1_A",
            "project_description": "test",
            "registration_date": "2020/11/12 00:00",
            "submitter_organization_list": ["JGI"],
        },
        {
            "uid": "2",
            "project_acc": "PRJNA2",
            "project_title": "OWC metatranscriptome - Jul_M1_C1_D1_A",
            "project_description": "test",
            "registration_date": "2020/11/12 00:00",
            "submitter_organization_list": ["JGI"],
        },
    ]

    crosswalk = module.build_crosswalk(samples, records)

    assert crosswalk["sample_project_link_status"].tolist() == [
        "mapped_to_authoritative_NCBI_BioProject_title",
        "mapped_to_authoritative_NCBI_BioProject_title",
        "unmapped_preserved",
    ]
    assert crosswalk.loc[1, "sample_label_mapping_method"] == (
        "deterministic_July_to_Jul"
    )
    assert crosswalk.loc[2, "bioproject_accession"] == ""


def test_jgi_crosswalk_preserves_identity_without_ecological_overclaim() -> None:
    module = _load_script(
        "stage_mucc_v1_jgi_sample_crosswalk",
        "scripts/external/stage_mucc_v1_jgi_sample_crosswalk.py",
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["owc_expr__Aug_M1_C1_D1_A", "unmapped"],
            "source_sample_column": ["Aug_M1_C1_D1_A", "No_match"],
            "bioproject_accession": ["PRJNA1", ""],
            "bioproject_uid": ["1", ""],
        }
    )
    recovered = {
        "PRJNA1": {
            "jgi_sequencing_project_id": "1256939",
            "jgi_final_deliverable_portal_id": "OldWomM1_C1_D1_A_FD",
            "jgi_portal_final_url": "https://genome.jgi.doe.gov/portal/OldWomM1_C1_D1_A_FD/OldWomM1_C1_D1_A_FD.info.html",
            "jgi_portal_project_name": "Aug_M1_C1_D1_A",
            "jgi_qc_rows": [
                {
                    "Sample Id": "227332",
                    "Sample Name": "Aug_M1_C1_D1_A",
                    "Sample Receipt Date": "2019-09-05T11:07:56",
                    "Sample Qc Date": "2019-09-11T00:00:00",
                    "Sample Qc Result": "pass",
                }
            ],
        }
    }

    crosswalk = module.build_crosswalk(samples, recovered)

    assert crosswalk.loc[0, "jgi_sample_id"] == "227332"
    assert crosswalk.loc[0, "jgi_sample_identity_status"] == (
        "exact_source_label_to_JGI_Sample_QC_record"
    )
    assert crosswalk.loc[0, "collection_datetime_status"] == (
        "not_reported_by_JGI_Sample_QC_export"
    )
    assert crosswalk.loc[0, "depth_cm_join_status"] == (
        "not_reported_by_JGI_Sample_QC_export"
    )
    assert crosswalk.loc[1, "jgi_sample_identity_status"] == "unmapped_preserved"


def test_jgi_crosswalk_handles_bioproject_without_qc_export() -> None:
    module = _load_script(
        "stage_mucc_v1_jgi_sample_crosswalk_without_qc",
        "scripts/external/stage_mucc_v1_jgi_sample_crosswalk.py",
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["owc_expr__Aug_M1_C1_D1_A"],
            "source_sample_column": ["Aug_M1_C1_D1_A"],
            "bioproject_accession": ["PRJNA1"],
            "bioproject_uid": ["1"],
        }
    )

    crosswalk = module.build_crosswalk(
        samples,
        {
            "PRJNA1": {
                "jgi_sequencing_project_id": "1256939",
                "jgi_final_deliverable_portal_id": "",
                "jgi_portal_final_url": "",
                "jgi_portal_project_name": "",
            }
        },
    )

    assert crosswalk.loc[0, "jgi_sample_identity_status"] == (
        "unresolved_no_JGI_Sample_QC_export"
    )


def test_jgi_portal_id_can_be_recovered_from_cached_info_html() -> None:
    module = _load_script(
        "stage_mucc_v1_jgi_sample_crosswalk_cached_info",
        "scripts/external/stage_mucc_v1_jgi_sample_crosswalk.py",
    )

    portal_id = module.portal_id_from_info_html(
        b'<a href="/portal/OldWomM1_C1_D1_A_FD/OldWomM1_C1_D1_A_FD.info.html">Info</a>'
    )

    assert portal_id == "OldWomM1_C1_D1_A_FD"


def test_jgi_qc_endpoint_uses_the_sequencing_project_id() -> None:
    module = _load_script(
        "stage_mucc_v1_jgi_sample_crosswalk_qc_url",
        "scripts/external/stage_mucc_v1_jgi_sample_crosswalk.py",
    )

    _, qc_url = module.jgi_urls("1256939")

    assert "exportQClist" in qc_url
    assert "spProjects=1256939" in qc_url
    assert "exportQCs" not in qc_url


def test_jgi_data_portal_catalog_retains_identity_without_ecological_overclaim() -> (
    None
):
    module = _load_script(
        "stage_mucc_v1_jgi_data_portal_catalog",
        "scripts/external/stage_mucc_v1_jgi_data_portal_catalog.py",
    )
    source_label = "Aug_M1_C1_D1_A"
    name = f"Old Woman Creek 2018 metatranscriptomes {source_label}"
    payload = {
        "organisms": [
            {
                "id": "IMG_AP-1256701",
                "name": name,
                "proposal_id": 504205,
                "product_search_category": "Metatranscriptome",
                "visibility": "public",
                "data_utilization_status": "Unrestricted",
                "status": "Complete",
                "work_completion_date": "2020-03-01",
                "file_total": 1,
                "fileSize": 20,
                "files": [
                    {
                        "file_status": "PURGED",
                        "file_size": 20,
                        "metadata": {
                            "sow_segment": {
                                "latitude_of_sample_collection": 41.3776,
                                "longitude_of_sample_collection": -82.512,
                            }
                        },
                    }
                ],
            },
            {
                "id": "IMG_AP-1256700",
                "name": f"{name} Annotation",
                "proposal_id": 504205,
                "product_search_category": "Metatranscriptome",
                "visibility": "public",
                "data_utilization_status": "Unrestricted",
                "status": "Complete",
                "work_completion_date": "2020-02-25",
                "portal_detail_id": 3300037648,
                "file_total": 2,
                "fileSize": 30,
                "files": [
                    {"file_status": "PURGED", "file_size": 10},
                    {"file_status": "PURGED", "file_size": 20},
                ],
            },
        ]
    }

    row = module.parse_catalog_record(
        {
            "sample_id": "owc_expr__Aug_M1_C1_D1_A",
            "source_sample_column": source_label,
        },
        payload,
        "https://files.jgi.doe.gov/search/?q=Aug_M1_C1_D1_A",
    )

    assert row["jgi_data_portal_identity_status"] == (
        "exact_source_label_to_JGI_Data_Portal_record_pair"
    )
    assert row["jgi_data_portal_expression_record_id"] == "IMG_AP-1256701"
    assert row["jgi_data_portal_annotation_taxon_oid"] == "3300037648"
    assert row["jgi_data_portal_indexed_file_count"] == "3"
    assert row["jgi_data_portal_purged_file_count"] == "3"
    assert row["jgi_data_portal_latitude"] == "41.3776"
    assert row["collection_datetime_status"] == (
        "not_reported_by_JGI_Data_Portal_catalog"
    )
    assert row["depth_cm_join_status"] == "not_reported_by_JGI_Data_Portal_catalog"
    assert row["environment_flux_join_status"] == (
        "unlinked_pending_exact_spatiotemporal_crosswalk"
    )


def test_authoritative_ecology_crosswalk_requires_explicit_complete_evidence() -> None:
    module = _load_script(
        "stage_mucc_v1_authoritative_ecological_crosswalk",
        "scripts/external/stage_mucc_v1_authoritative_ecological_crosswalk.py",
    )
    source_label = "Aug_M1_C1_D1_A"
    sample_id = "owc_expr__Aug_M1_C1_D1_A"
    scaffold = pd.DataFrame(
        [{"sample_id": sample_id, "source_sample_column": source_label}]
    )
    complete = {
        "mapping_id": "authoritative_0001",
        "source_sample_column": source_label,
        "authoritative_sample_id": "OWC-2018-001",
        "collection_datetime": "2018-08-01T10:00:00-04:00",
        "site_id": "M1",
        "core_or_plot_id": "C1",
        "depth_cm": "5",
        "depth_reference": "top_of_mineral_soil",
        "sequence_assay_type": "metatranscriptome",
        "assay_reconciliation_status": "validated_metatranscriptome",
        "mag_abundance_or_read_coverage_record_id": "coverage-001",
        "mag_abundance_or_read_coverage_units": "reads_per_MAG",
        "environment_source": "ESS_DIVE_10.15485_1568865",
        "environment_record_id": "essdive_1568865_peeper_004_SOIL_H2O_CH4_1_1_1",
        "environment_measurement_datetime": "2018-08-01T10:15:00-04:00",
        "environment_measurement_units": "mM",
        "flux_source": "ESS_DIVE_10.15485_1568865",
        "flux_observation_id": "essdive_1568865_chamber_0001",
        "flux_measurement_type": "chamber_CH4_flux",
        "flux_units": "nmol_m2_s",
        "flux_window_start_datetime": "2018-08-01T09:00:00-04:00",
        "flux_window_end_datetime": "2018-08-01T11:00:00-04:00",
        "replicate_id": "A",
        "uncertainty_record_id": "uncertainty-001",
        "uncertainty_method": "reported_standard_error",
        "source_evidence_status": "authoritative_complete",
        "missingness_status": "complete",
        "source_url": "https://doi.org/10.1128/msystems.00680-25",
    }

    links, readiness = module.validate_crosswalk(
        pd.DataFrame([complete]),
        scaffold,
        {"essdive_1568865_chamber_0001"},
        {"essdive_1568865_peeper_004_SOIL_H2O_CH4_1_1_1"},
    )

    assert links.loc[0, "mapping_validation_status"] == (
        "validated_authoritative_sample_environment_flux_mapping"
    )
    assert readiness.loc[0, "authoritative_ecology_readiness_status"] == (
        "ready_for_grouped_ecological_validation"
    )

    tower_complete = complete.copy()
    tower_complete["mapping_id"] = "authoritative_tower_0001"
    tower_complete["flux_source"] = "ESS_DIVE_10.15485_2500238"
    tower_complete["flux_observation_id"] = (
        "essdive_2500238_gapfilled_tower_201506010000"
    )
    links, _ = module.validate_crosswalk(
        pd.DataFrame([tower_complete]),
        scaffold,
        {"essdive_1568865_chamber_0001"},
        {"essdive_1568865_peeper_004_SOIL_H2O_CH4_1_1_1"},
        {"essdive_2500238_gapfilled_tower_201506010000"},
    )

    assert links.loc[0, "mapping_validation_status"] == (
        "validated_authoritative_sample_environment_flux_mapping"
    )


def test_authoritative_ecology_crosswalk_preserves_partial_mapping_as_blocked() -> None:
    module = _load_script(
        "stage_mucc_v1_authoritative_ecological_crosswalk_partial",
        "scripts/external/stage_mucc_v1_authoritative_ecological_crosswalk.py",
    )
    scaffold = pd.DataFrame(
        [
            {
                "sample_id": "owc_expr__Aug_M1_C1_D1_A",
                "source_sample_column": "Aug_M1_C1_D1_A",
            }
        ]
    )
    row = {column: "" for column in module.REQUIRED_COLUMNS}
    row.update(
        {
            "mapping_id": "authoritative_partial_0001",
            "source_sample_column": "Aug_M1_C1_D1_A",
            "source_evidence_status": "authoritative_partial",
            "missingness_status": "flux_window_missing",
        }
    )

    links, readiness = module.validate_crosswalk(
        pd.DataFrame([row]), scaffold, set(), set()
    )

    assert links.loc[0, "mapping_validation_status"] == (
        "explicit_authoritative_mapping_partial"
    )
    assert readiness.loc[0, "authoritative_ecology_readiness_status"] == (
        "blocked_no_validated_authoritative_mapping"
    )


def test_promotion_uses_only_validated_authoritative_ecology_mapping(
    tmp_path: Path,
) -> None:
    module = _load_script(
        "promote_mucc_v1_integrated_atlas_authoritative_ecology",
        "scripts/reports/promote_mucc_v1_integrated_atlas.py",
    )
    environmental = tmp_path / "environmental_metadata"
    environmental.mkdir()
    sample_id = "owc_expr__Aug_M1_C1_D1_A"
    source_label = "Aug_M1_C1_D1_A"
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "mapping_id": "authoritative_0001",
                "authoritative_sample_id": "OWC-2018-001",
                "collection_datetime": "2018-08-01T10:00:00-04:00",
                "depth_cm": "5",
                "environment_record_id": "environment-001",
                "flux_observation_id": "flux-001",
                "mapping_validation_status": (
                    "validated_authoritative_sample_environment_flux_mapping"
                ),
            }
        ]
    ).to_csv(
        environmental / "link_mucc_v1_sequence_authoritative_ecology.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "authoritative_ecology_readiness_status": (
                    "ready_for_grouped_ecological_validation"
                ),
            }
        ]
    ).to_csv(
        environmental / "feature_mucc_v1_authoritative_ecology_readiness.tsv",
        sep="\t",
        index=False,
    )

    ecological = module.build_sample_ecological_readiness(
        pd.DataFrame([{"sample_id": sample_id, "source_sample_column": source_label}]),
        pd.DataFrame([{"sample_id": sample_id, "source_sample_column": source_label}]),
        tmp_path,
    )

    row = ecological.loc[0]
    assert row["authoritative_ecology_link_status"] == (
        "validated_authoritative_sample_environment_flux_mapping"
    )
    assert row["sample_ecological_validation_status"] == (
        "eligible_for_grouped_ecological_validation_not_final_MRV"
    )
    assert row["authoritative_ecology_mapping_id"] == "authoritative_0001"


def test_lane_summary_counts_passing_multiwindow_glm2_report(tmp_path: Path) -> None:
    module = _load_script(
        "summarize_atlas_lane_registry",
        "scripts/reports/summarize_atlas_lane_registry.py",
    )
    glm_dir = tmp_path / "glm"
    report = glm_dir / "validation/glm2_multiwindow_reduce_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "pass", "n_mags": 3}))

    count, evidence = module.count_glm2_units([glm_dir], {"a", "b"})

    assert count == 2
    assert evidence == "glm2_multiwindow_reduce_report.json"
