from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_script() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location(
        "stage_mucc_v1_zenodo_qc_reconciliation",
        repo_root / "scripts/external/stage_mucc_v1_zenodo_qc_reconciliation.py",
    )
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_quality_status_matches_paper_boundary() -> None:
    module = _load_script()

    assert module.source_quality_status(50.0, 9.999) == (
        "meets_published_MQHQ_CheckM_threshold"
    )
    assert module.source_quality_status(49.99, 0.0) == (
        "does_not_meet_published_MQHQ_CheckM_threshold"
    )
    assert module.source_quality_status(50.0, 10.0) == (
        "does_not_meet_published_MQHQ_CheckM_threshold"
    )
    assert module.source_quality_status(None, 1.0) == "source_qc_value_missing"


def test_output_rows_preserve_nonqualifying_archive_member() -> None:
    module = _load_script()
    qc = module.SourceQC()
    qc.add(
        {
            "bin_completeness": "48.5",
            "bin_contamination": "2.0",
            "bin_taxonomy": "d__Bacteria;p__Example",
        },
        "OWC_0001",
        True,
    )
    qc.add(
        {
            "bin_completeness": "48.50",
            "bin_contamination": "2",
            "bin_taxonomy": "d__Bacteria;p__Example",
        },
        "alias_bin",
        False,
    )

    rows = module.output_rows(
        [{"mag_id": "OWC_0001", "proteome_id": "mucc_v1__OWC_0001"}],
        {"OWC_0001": qc},
    )

    assert rows[0]["source_qc_value_consistency_status"] == (
        "direct_source_qc_values_consistent_across_annotation_rows"
    )
    assert rows[0]["source_qc_mapping_status"] == (
        "direct_catalog_fasta_and_source_bin_crosswalk_QC"
    )
    assert rows[0]["published_mq_hq_membership_status"] == (
        "does_not_meet_published_MQHQ_CheckM_threshold"
    )
    assert module.reconciliation_rows(rows)[0]["reconciliation_status"] == (
        "archive_member_outside_published_2502_HQMQ_QC_scope_reconciled_from_direct_Zenodo_QC"
    )
