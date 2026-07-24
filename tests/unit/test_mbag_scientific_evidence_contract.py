from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import sys

import numpy as np
import pandas as pd


def _load_script(relative_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location(module_name, repo_root / relative_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


report = _load_script(
    "scripts/reports/build_mbag_nextgen_molecular_niche_atlas.py",
    "build_mbag_nextgen_molecular_niche_atlas_scientific_test",
)
freeze = _load_script(
    "scripts/reports/build_methanet_3view_payload_freeze.py",
    "build_methanet_3view_payload_freeze_scientific_test",
)


def test_scientific_contract_separates_completeness_from_comparability() -> None:
    atlas = pd.DataFrame(
        {
            "lane_id": [
                "poc_core",
                "msm_china_2025",
                "futian_mangrove_2026_qi",
                "mucc_v1_owc_wetland",
                "futian_mangrove_2026_qi",
            ],
            "has_esm2": [True, True, True, True, True],
            "has_glm2": [True, True, True, True, True],
            "has_functional": [True, True, True, True, False],
            "native_window_count": [1, 1, 1, 10, 1],
            "shuffled_control_count": [1, 1, 1, 10, 1],
            "protein_cap_applied": [False, True, False, False, False],
        }
    )

    out = report.apply_scientific_evidence_contract(atlas)

    assert (
        out.loc[0, "formal_tri_view_status"]
        == "complete_canonical_mechanism_tri_view"
    )
    assert (
        out.loc[1, "formal_tri_view_status"]
        == "complete_annotation_tri_view_harmonization_pending"
    )
    assert (
        out.loc[2, "mechanism_equivalence_status"]
        == "not_yet_mechanism_equivalent"
    )
    assert (
        out.loc[3, "formal_tri_view_status"]
        == "complete_source_scaffold_tri_view"
    )
    assert out.loc[3, "glm2_protocol_class"].startswith("multiwindow")
    assert out.loc[4, "formal_tri_view_status"] == "incomplete_tri_view"
    assert out["mechanism_equivalent_tri_view"].sum() == 1


def test_noncomparable_raw_hit_rows_are_quarantined_from_public_rates() -> None:
    atlas = pd.DataFrame(
        {
            "lane_id": ["poc_core", "msm_china_2025"],
            "has_esm2": [True, True],
            "has_glm2": [True, True],
            "has_functional": [True, True],
            "native_window_count": [1, 1],
            "shuffled_control_count": [1, 1],
            "protein_cap_applied": [False, False],
            "prodigal_proteins": [2_000, 2_000],
            "methane_evidence_score": [3, 4_000],
            "sulfur_competition_score": [2, 3_000],
            "cazy_family_count": [8, 800],
            "merops_family_count": [4, 400],
            "kofam_annotated_gene_fraction": [0.5, 1.0],
            "metabolic_modules_present": [4, 40],
            "broad_function_evidence_count": [12, 1_200],
            "checkm2_completeness": [90, 90],
            "checkm2_contamination": [2, 2],
            "nearest_poc_similarity": [0.99, 0.99],
            "cross_domain_neighbor_fraction": [0.1, 0.1],
            "mixing_coeff": [0.2, 0.2],
            "glm_context_delta": [2.0, 2.0],
        }
    )
    atlas = report.apply_scientific_evidence_contract(atlas)

    out = report.add_molecular_metrics(atlas)

    assert np.isfinite(out.loc[0, "methane_marker_density_per_1k"])
    assert np.isnan(out.loc[1, "methane_marker_density_per_1k"])
    assert np.isnan(out.loc[1, "molecular_attestation_index"])
    assert np.isfinite(
        out.loc[1, "legacy_noncomparable_attestation_index_quarantined"]
    )
    assert (
        out.loc[1, "rate_metric_status"]
        == "quarantined_raw_hit_row_numerator_not_marker_density"
    )


def test_freeze_contract_does_not_promote_external_per_mag_annotations(
    tmp_path: Path,
) -> None:
    poc = freeze.functional_evidence_contract(
        tmp_path,
        {"lane_id": "poc_core", "lane_role": "calibration_core"},
    )
    expansion = freeze.functional_evidence_contract(
        tmp_path,
        {
            "lane_id": "msm_china_2025",
            "lane_role": "external_mangrove",
        },
    )

    assert poc["mechanism_equivalence_status"] == "mechanism_equivalent"
    assert (
        expansion["functional_evidence_class"]
        == "annotation_complete_feature_aggregation_pending"
    )
    assert (
        expansion["mechanism_equivalence_status"]
        == "not_yet_mechanism_equivalent"
    )


def test_taxonomy_synonym_normalization_handles_gtdb_prefix() -> None:
    assert report.normalized_phylum("p__Proteobacteria") == "Pseudomonadota"
    assert report.normalized_phylum("p__Pseudomonadota") == "Pseudomonadota"
    assert report.normalized_phylum(np.nan) == ""
