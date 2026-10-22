from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


def _load_atlas_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts/reports/build_mbag_expanded_multiview_atlas.py"
    spec = spec_from_file_location("build_mbag_expanded_multiview_atlas_overlay", script)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_table(warehouse: Path, table: str, rows: list[dict]) -> None:
    path = warehouse / "parquet" / table / "cohort_run_id=test" / "part-00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _per_mag_fallback() -> pd.DataFrame:
    # Raw per-MAG fallback values: MCycDB + METABOLIC rows under the score name.
    return pd.DataFrame(
        [
            {"proteome_id": "in_warehouse", "methane_evidence_score": 3655, "sulfur_competition_score": 9000},
            {"proteome_id": "not_in_warehouse", "methane_evidence_score": 4071, "sulfur_competition_score": 8000},
        ]
    )


def test_overlay_uses_warehouse_scores_and_gtdb_release(tmp_path: Path) -> None:
    module = _load_atlas_module()
    warehouse = tmp_path / "cohort_warehouse"
    _write_table(
        warehouse,
        "feature_mrv_mag_level",
        [{"proteome_id": "in_warehouse", "methane_evidence_score": 3, "sulfur_competition_score": 435}],
    )
    _write_table(
        warehouse,
        "dim_mag",
        [{"proteome_id": "in_warehouse", "gtdb_release": "R232", "gtdb_classification": "d__Bacteria;p__Desulfobacterota"}],
    )

    out = module.overlay_pipeline_warehouse_semantics(_per_mag_fallback(), [warehouse])
    by_id = out.set_index("proteome_id")

    assert by_id.loc["in_warehouse", "methane_evidence_score"] == 3
    assert by_id.loc["in_warehouse", "sulfur_competition_score"] == 435
    assert by_id.loc["in_warehouse", "methane_evidence_score_source"] == "lane_warehouse_feature_mrv_mag_level"
    assert by_id.loc["in_warehouse", "gtdb_release"] == "R232"
    assert by_id.loc["in_warehouse", "gtdb_classification"].startswith("d__Bacteria")
    # A row the warehouse does not cover keeps its value and is labelled as a raw fallback.
    assert by_id.loc["not_in_warehouse", "methane_evidence_score"] == 4071
    assert by_id.loc["not_in_warehouse", "methane_evidence_score_source"] == "per_mag_raw_row_count_fallback"
    assert pd.isna(by_id.loc["not_in_warehouse", "gtdb_release"])


def test_overlay_without_pipeline_tables_leaves_rows_unchanged(tmp_path: Path) -> None:
    module = _load_atlas_module()
    fallback = _per_mag_fallback()

    out = module.overlay_pipeline_warehouse_semantics(fallback, [tmp_path / "empty_warehouse"])

    assert out["methane_evidence_score"].tolist() == [3655, 4071]
    assert set(out["methane_evidence_score_source"]) == {"per_mag_raw_row_count_fallback"}
    assert module.overlay_pipeline_warehouse_semantics(fallback, []).equals(fallback)


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts/reports"))
    script = repo_root / "scripts/reports/build_mbag_nextgen_molecular_niche_atlas.py"
    spec = spec_from_file_location("build_mbag_nextgen_molecular_niche_atlas_release_text", script)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gtdb_release_text_follows_the_audited_lanes() -> None:
    text = _load_report_module().gtdb_release_coverage_text
    only_core = [{"lane_id": "poc_core", "gtdb_release": "R232"}, {"lane_id": "msm_china_2025", "gtdb_release": "missing"}]
    assert text(only_core) == "GTDB release metadata exists only for the reference core"
    filled = [
        {"lane_id": "poc_core", "gtdb_release": "R232"},
        {"lane_id": "msm_china_2025", "gtdb_release": "R232"},
        {"lane_id": "futian_mangrove_2026_qi", "gtdb_release": "R232"},
        {"lane_id": "futian_mangrove_2026_qi", "gtdb_release": "missing"},
        {"lane_id": "mucc_v1_owc_wetland", "gtdb_release": "missing"},
    ]
    assert text(filled) == (
        "GTDB release metadata (R232) exists for the reference core, the MSM mangrove lane "
        "and the Futian mangrove lane and is missing for the Old Woman Creek lane"
    )
