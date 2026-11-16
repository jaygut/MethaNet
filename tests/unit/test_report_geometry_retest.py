from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts/reports"))
    script = repo_root / "scripts/reports/build_mbag_nextgen_molecular_niche_atlas.py"
    spec = spec_from_file_location("build_mbag_nextgen_molecular_niche_atlas_retest", script)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_geometry_retest_summary_keeps_only_cited_checks(tmp_path: Path) -> None:
    load = _load_report_module().load_geometry_retest
    summary = tmp_path / "key_numbers.json"
    summary.write_text(
        json.dumps(
            {
                "A_pilot": {"ecosystem_by_domain": {"wetland": {"Bacteria": 0}, "rumen": {"Bacteria": 5}}},
                "B2_collinearity": {
                    "pairwise": {"raw": {"share_pairs_above_0.99": 0.8}},
                    "effective_dimensionality": [{"transform": "raw (uncentered second moment)", "offset_share_of_squared_norm": 0.99}],
                },
                "B3_B5_neighbourhoods_and_mutual_pairs": {
                    "knn_coherence_all_edges": {"raw": {"phylum": 0.75, "family": 0.3}},
                    "raw_mangrove_wetland_phylum_share_retest_taxonomy": 0.78,
                    "zscore_rumen_pairs_share_same": {"phylum": 0.3},
                    "mutual_pair_counts": {"raw": {"any pair involving rumen": 10}, "whitened": {"any pair involving rumen": 20}},
                },
                "B4_B6_nearest_core_and_candidates": {
                    "raw": {
                        "bacterial_records_to_archaeal_core": 3,
                        "bacterial_records_outside_core": 9,
                        "same_phylum": 1,
                        "phylum_known": 4,
                        "top10_hub_share": 0.5,
                    }
                },
            }
        )
    )
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "B3_knn_taxonomic_coherence.tsv").write_text(
        "transform\trank\tscope\trandom_expectation_same_habitat_pairing\n"
        "raw\tphylum\tall neighbour edges\t0.1\n"
        "raw\tfamily\tall neighbour edges\t0.01\n"
    )
    (tables / "B6_candidate_nearest_core.tsv").write_text(
        "proteome_id\tpublished_similarity\na\t0.99999994\nb\t0.9979\n"
    )

    out = load(summary)

    assert out["core_wetland_bacteria"] == 0
    assert out["nearest_core_same_phylum_share"] == 0.25
    assert (out["rumen_mutual_pair_min"], out["rumen_mutual_pair_max"]) == (10, 20)
    assert out["random_same_phylum_share"] == 0.1
    assert out["candidate_self_copies"] == 1
    assert out["offset_share"] == 0.99
    assert load(tmp_path / "missing.json") == {}
