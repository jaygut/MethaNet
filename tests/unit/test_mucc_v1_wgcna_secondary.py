from __future__ import annotations

import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_script() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts/reports/summarize_mucc_v1_wgcna_secondary.py"
    spec = spec_from_file_location("summarize_mucc_v1_wgcna_secondary", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_wgcna_summarizer_requires_exact_input_coverage_and_updates_status(
    tmp_path: Path,
) -> None:
    module = _load_script()
    network = tmp_path / "run/network_analysis"
    _write_tsv(
        network / "wgcna_secondary_expression.tsv",
        [
            {"sample_id": "s1", "p1": "0.2", "p2": "0.8"},
            {"sample_id": "s2", "p1": "0.3", "p2": "0.7"},
        ],
    )
    _write_tsv(
        network / "wgcna_secondary_modules.tsv",
        [
            {"proteome_id": "p1", "module": "blue"},
            {"proteome_id": "p2", "module": "grey"},
        ],
    )
    _write_tsv(
        network / "wgcna_secondary_module_eigengenes.tsv",
        [
            {"sample_id": "s1", "MEblue": "0.1", "MEgrey": "0.0"},
            {"sample_id": "s2", "MEblue": "-0.1", "MEgrey": "0.0"},
        ],
    )
    _write_tsv(
        network / "wgcna_secondary_run_metadata.tsv",
        [
            {
                "r_version": "4.4.3",
                "wgcna_version": "1.74",
                "samples": "2",
                "mag_features": "2",
                "soft_power": "6",
                "selected_sft_r_sq": "0.2",
                "module_count_including_grey": "2",
                "non_grey_module_count": "1",
                "unassigned_grey_mag_count": "1",
                "network_type": "signed hybrid",
                "min_module_size": "50",
                "merge_cut_height": "0.3",
                "source_reported_wgcna_samples": "132",
                "source_outlier_reconciliation_status": (
                    "blocked_source_outlier_sample_id_not_available"
                ),
                "source_parameter_alignment_status": (
                    "aligned_power_7_signed_hybrid_min_module_50_merge_height_0.3"
                ),
            }
        ],
    )
    _write_tsv(
        network / "network_analysis_status.tsv",
        [
            {
                "lane_id": "mucc_v1_owc_wetland",
                "analysis_id": "wgcna_secondary_module_discovery",
                "role": "secondary",
                "status": "ready",
                "execution_state": "runtime_available",
                "samples": "2",
                "MAG_features": "2",
                "method_parameters": "",
            }
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "summarize_mucc_v1_wgcna_secondary.py",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            "run",
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    summary = json.loads((network / "mucc_v1_wgcna_secondary_summary.json").read_text())
    assert summary["non_grey_module_count"] == 1
    assert summary["source_reported_wgcna_samples"] == "132"
    assert (
        summary["source_outlier_reconciliation_status"]
        == "blocked_source_outlier_sample_id_not_available"
    )
    with (network / "network_analysis_status.tsv").open(newline="") as handle:
        status = list(csv.DictReader(handle, delimiter="\t"))
    assert status[0]["status"] == "completed_secondary_descriptive_modules"
    with (network / "feature_mucc_v1_wgcna_secondary_module_membership.tsv").open(
        newline=""
    ) as handle:
        membership = list(csv.DictReader(handle, delimiter="\t"))
    assert membership[0]["proteome_id"] == "p1"
