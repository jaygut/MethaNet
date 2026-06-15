from __future__ import annotations

import csv
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_manifest_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "build_mag_unit_scope_manifests.py"
    spec = spec_from_file_location("build_mag_unit_scope_manifests", script_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_unit_scope_classifies_mag_bin_and_assembly_context(tmp_path: Path) -> None:
    module = _load_manifest_module()
    repo_root = tmp_path
    cohort_dir = tmp_path / "results" / "functional_metagenomics" / "fgx_test"
    manifest = tmp_path / "manifest.tsv"

    rows = [
        {
            "proteome_id": "mucc__3300004775_9",
            "sample": "mucc__3300004775_9",
            "source": "mucc",
            "ecosystem": "wetland",
            "mag_id": "3300004775_9",
            "mag_fasta": "data/assemblies/3300004775_9.fasta",
            "mag_fasta_basename": "3300004775_9.fasta",
            "proteome_faa": "data/proteomes/mucc__3300004775_9.faa",
            "proteome_faa_basename": "mucc__3300004775_9.faa",
            "match_status": "matched",
            "functional_run_include": "True",
            "n_proteins_used": "1800",
        },
        {
            "proteome_id": "rumen__10674_0001_idba_bin.10",
            "sample": "rumen__10674_0001_idba_bin.10",
            "source": "rumen",
            "ecosystem": "rumen",
            "mag_id": "10674_0001_idba_bin.10",
            "mag_fasta": "data/rumen/10674_0001_idba_bin.10.fa.gz",
            "mag_fasta_basename": "10674_0001_idba_bin.10.fa.gz",
            "proteome_faa": "data/proteomes/rumen__10674_0001_idba_bin.10.faa",
            "proteome_faa_basename": "rumen__10674_0001_idba_bin.10.faa",
            "match_status": "matched",
            "functional_run_include": "True",
            "n_proteins_used": "2100",
        },
        {
            "proteome_id": "rumen__10676_0001_idba",
            "sample": "rumen__10676_0001_idba",
            "source": "rumen",
            "ecosystem": "rumen",
            "mag_id": "10676_0001_idba",
            "mag_fasta": "data/rumen/10676_0001_idba.fa.gz",
            "mag_fasta_basename": "10676_0001_idba.fa.gz",
            "proteome_faa": "data/proteomes/rumen__10676_0001_idba.faa",
            "proteome_faa_basename": "rumen__10676_0001_idba.faa",
            "match_status": "matched",
            "functional_run_include": "True",
            "n_proteins_used": "6000",
        },
    ]
    _write_manifest(manifest, rows)

    completed_run = (
        cohort_dir
        / "per_mag"
        / "rumen__10674_0001_idba_bin.10"
        / "fgx_1_rumen__10674_0001_idba_bin.10"
    )
    (completed_run / "curated").mkdir(parents=True)
    (completed_run / "COMPLETE").write_text("")
    (completed_run / "curated" / "run_record.json").write_text(
        json.dumps(
            {
                "run_id": "fgx_1_rumen__10674_0001_idba_bin.10",
                "status": "complete",
                "proteome_id": "rumen__10674_0001_idba_bin.10",
                "mag_id": "10674_0001_idba_bin.10",
            }
        )
    )

    observed = module.build_rows(repo_root, manifest, cohort_dir)
    by_id = {row["proteome_id"]: row for row in observed}

    assert by_id["mucc__3300004775_9"]["analysis_unit_type"] == "mag_bin"
    assert by_id["mucc__3300004775_9"]["mbag_mag_level_include"] == "true"
    assert by_id["rumen__10674_0001_idba_bin.10"]["analysis_unit_type"] == "mag_bin"
    assert by_id["rumen__10674_0001_idba_bin.10"]["latest_run_status"] == "complete"
    assert by_id["rumen__10676_0001_idba"]["analysis_unit_type"] == "assembly_context"
    assert by_id["rumen__10676_0001_idba"]["mbag_mag_level_include"] == "false"
    assert by_id["rumen__10676_0001_idba"]["assembly_context_include"] == "true"

    remaining = [
        row["proteome_id"]
        for row in observed
        if row["analysis_unit_type"] == "mag_bin" and row["latest_run_status"] != "complete"
    ]
    assert remaining == ["mucc__3300004775_9"]
