from __future__ import annotations

import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


def _load_script() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts/reports/build_mucc_v1_network_analysis_inputs.py"
    spec = spec_from_file_location("build_mucc_v1_network_analysis_inputs", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wgcna_uses_full_expression_scope_without_changing_flashweave_scope(
    tmp_path: Path,
) -> None:
    module = _load_script()
    run_dir = tmp_path / "run"
    expression_dir = run_dir / "expression"
    metadata_dir = run_dir / "environmental_metadata"
    feature_dir = run_dir / "functional_features"
    expression_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    proteomes = [f"p{index:02d}" for index in range(26)]
    fact_rows = [
        {
            "sample_id": sample_id,
            "proteome_id": proteome_id,
            "mag_id": f"mag_{proteome_id}",
            "expression_value": 1 / len(proteomes),
        }
        for sample_id in ["s1", "s2"]
        for proteome_id in proteomes
    ]
    pd.DataFrame(fact_rows).to_csv(
        expression_dir / "fact_mucc_v1_expression_mag_sample.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "month_label": "July",
                "site_or_landcover": "M1",
                "depth_code": "D1",
                "depth_class_inferred": "surface",
            },
            {
                "sample_id": "s2",
                "month_label": "August",
                "site_or_landcover": "M1",
                "depth_code": "D3",
                "depth_class_inferred": "mid",
            },
        ]
    ).to_csv(
        metadata_dir / "mucc_v1_sample_columns_scaffold.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        [{"proteome_id": proteome_id, "class": "Example"} for proteome_id in proteomes]
    ).to_csv(
        feature_dir / "feature_mucc_v1_mrv_readiness_mag_level.tsv",
        sep="\t",
        index=False,
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_mucc_v1_network_analysis_inputs.py",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            "run",
            "--max-features",
            "25",
            "--wgcna-max-features",
            "0",
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    network = run_dir / "network_analysis"
    with (network / "network_analysis_status.tsv").open(newline="") as handle:
        status = {
            row["analysis_id"]: row for row in csv.DictReader(handle, delimiter="\t")
        }
    assert status["flashweave_direct_association"]["MAG_features"] == "25"
    assert status["wgcna_secondary_module_discovery"]["MAG_features"] == "26"
    with (network / "wgcna_secondary_expression.tsv").open(newline="") as handle:
        wgcna_input = list(csv.reader(handle, delimiter="\t"))
    assert len(wgcna_input) == 3
    assert len(wgcna_input[0]) == 27
    manifest = json.loads((network / "wgcna_secondary_input_manifest.json").read_text())
    assert manifest["selected_MAG_features"] == 26
    assert manifest["selection"] == "all_nonzero_processed_expression_MAGs"
    assert manifest["source_method_alignment"] == {
        "comparability_status": "source_method_aligned_partial_not_exact_reproduction",
        "current_samples_retained": 2,
        "merge_cut_height": 0.3,
        "min_module_size": 50,
        "network_type": "signed_hybrid",
        "outlier_reconciliation_status": (
            "blocked_source_outlier_sample_id_not_available"
        ),
        "publication": "https://journals.asm.org/doi/10.1128/msystems.00680-25",
        "soft_power": 7,
        "source_reported_samples_after_outlier_screening": 132,
    }
    runner = (network / "run_wgcna_secondary.R").read_text()
    assert 'networkType = "signed hybrid"' in runner
    assert "softPower <- 7" in runner
    assert "minModuleSize = 50, mergeCutHeight = 0.3" in runner
