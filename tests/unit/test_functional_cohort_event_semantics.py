from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "scripts/consolidate_functional_mag_cohort.py"
    spec = importlib.util.spec_from_file_location("functional_cohort_event_semantics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dim_mag() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "cohort_run_id": "cohort",
            "run_id": "run",
            "proteome_id": "p1",
            "mag_id": "m1",
            "prodigal_proteins": 10,
            "analysis_unit_type": "mag_bin",
            "claim_scope": "MAG functional potential",
        }]
    )


def test_coverage_uses_accepted_kofam_and_best_ranked_diamond_events():
    module = load_module()
    tables = {
        "fact_kofam_hits": pd.DataFrame([
            {"proteome_id": "p1", "gene_id": "g1", "accepted_hit": True},
            {"proteome_id": "p1", "gene_id": "g2", "accepted_hit": False},
        ]),
        "fact_mcycdb_hits": pd.DataFrame([
            {"proteome_id": "p1", "gene_id": "g3", "hit_rank_bitscore": 1},
            {"proteome_id": "p1", "gene_id": "g3", "hit_rank_bitscore": 2},
        ]),
    }
    coverage = module.build_coverage(pd, dim_mag(), tables)
    kofam = coverage.loc[coverage["annotation_tool"].eq("KOfam")].iloc[0]
    mcyc = coverage.loc[coverage["annotation_tool"].eq("MCycDB")].iloc[0]

    assert kofam["row_count"] == 2
    assert kofam["accepted_or_present_event_count"] == 1
    assert kofam["annotated_gene_count"] == 1
    assert kofam["event_semantics"] == "accepted_hit_only"
    assert mcyc["row_count"] == 2
    assert mcyc["accepted_or_present_event_count"] == 1
    assert mcyc["annotated_gene_count"] == 1
    assert mcyc["event_semantics"] == "best_ranked_hit_per_gene"


def test_mechanism_features_exclude_secondary_diamond_hits_and_include_mcycdb():
    module = load_module()
    tables = {
        "fact_kofam_hits": pd.DataFrame([
            {"proteome_id": "p1", "gene_id": "g1", "ko_definition": "methanogenesis", "accepted_hit": True},
            {"proteome_id": "p1", "gene_id": "g2", "ko_definition": "sulfate reduction", "accepted_hit": True},
            {"proteome_id": "p1", "gene_id": "g2", "ko_definition": "sulfur metabolism", "accepted_hit": True},
        ]),
        "fact_mcycdb_hits": pd.DataFrame([
            {"proteome_id": "p1", "gene_id": "g3", "hit_rank_bitscore": 1},
            {"proteome_id": "p1", "gene_id": "g3", "hit_rank_bitscore": 2},
        ]),
        "fact_scycdb_hits": pd.DataFrame([
            {"proteome_id": "p1", "gene_id": "g4", "hit_rank_bitscore": 1},
            {"proteome_id": "p1", "gene_id": "g4", "hit_rank_bitscore": 2},
        ]),
    }
    coverage = module.build_coverage(pd, dim_mag(), tables)
    methane, sulfur, _mrv = module.build_mechanism_features(pd, dim_mag(), tables, coverage)

    assert methane.iloc[0]["accepted_kofam_methane_hits"] == 1
    assert methane.iloc[0]["mcycdb_best_hit_count"] == 1
    assert methane.iloc[0]["methane_evidence_score"] == 1
    assert sulfur.iloc[0]["scycdb_best_hit_count"] == 1
    assert sulfur.iloc[0]["sulfur_competition_score"] == 2
    assert sulfur.iloc[0]["sulfur_associated_screening_breadth"] == 2
    assert sulfur.iloc[0]["score_validation_status"] == "unvalidated_screening_alias"
