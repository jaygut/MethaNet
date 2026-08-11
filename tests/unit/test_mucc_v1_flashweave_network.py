from __future__ import annotations

import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_script() -> object:
    pytest.importorskip("networkx")
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts/reports/summarize_mucc_v1_flashweave_network.py"
    spec = spec_from_file_location("summarize_mucc_v1_flashweave_network", path)
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


def test_summarizer_preserves_mag_and_metadata_edges_separately(tmp_path: Path) -> None:
    module = _load_script()
    run_dir = tmp_path / "run"
    network = run_dir / "network_analysis"
    network.mkdir(parents=True)
    (network / "flashweave_direct_associations.edgelist").write_text(
        "# header\tmucc__a,mucc__b,month_label_July,site_or_landcover_A\n"
        "# meta mask\tfalse,false,true,true\n"
        "mucc__a\tmucc__b\t0.5\n"
        "mucc__a\tmonth_label_July\t0.2\n"
        "month_label_July\tsite_or_landcover_A\t-0.1\n"
    )
    _write_tsv(
        network / "network_node_manifest.tsv",
        [
            {
                "proteome_id": "mucc__a",
                "mag_id": "a",
                "domain": "d__Archaea",
                "phylum": "p__Euryarchaeota",
                "class": "c__Methanobacteria",
                "mrv_readiness_label": (
                    "molecular_reference_ready_ecological_validation_pending"
                ),
            },
            {
                "proteome_id": "mucc__b",
                "mag_id": "b",
                "domain": "d__Bacteria",
                "phylum": "p__Example",
                "class": "c__Example",
                "mrv_readiness_label": "source_annotation_scaffold_only",
            },
        ],
    )
    _write_tsv(
        network / "flashweave_conditioning_metadata.tsv",
        [{"sample_id": "s1", "month_label": "July", "site_or_landcover": "A"}],
    )
    (network / "flashweave_input_manifest.json").write_text(
        json.dumps(
            {
                "status": "ready_to_run_flashweave",
                "samples": 1,
                "selected_MAG_features": 2,
                "conditioning_metadata": ["month_label", "site_or_landcover"],
                "flashweave": {"FDR": True},
            }
        )
    )
    _write_tsv(
        network / "network_analysis_status.tsv",
        [
            {
                "lane_id": "mucc_v1_owc_wetland",
                "analysis_id": "wgcna_secondary_module_discovery",
                "role": "secondary",
                "status": "ready",
            }
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "summarize_mucc_v1_flashweave_network.py",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            "run",
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    with (network / "fact_mucc_v1_flashweave_edges.tsv").open(newline="") as handle:
        mag_edges = list(csv.DictReader(handle, delimiter="\t"))
    metadata_edge_path = network / "fact_mucc_v1_flashweave_metadata_edges.tsv"
    with metadata_edge_path.open(newline="") as handle:
        metadata_edges = list(csv.DictReader(handle, delimiter="\t"))
    assert len(mag_edges) == 1
    assert mag_edges[0]["source_mag_id"] == "a"
    assert len(metadata_edges) == 2
    assert {row["edge_class"] for row in metadata_edges} == {
        "mag_to_conditioning_metadata",
        "conditioning_metadata_to_metadata",
    }
    summary_path = network / "mucc_v1_flashweave_network_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["mag_mag_edges"] == 1
    assert summary["metadata_involving_edges"] == 2
