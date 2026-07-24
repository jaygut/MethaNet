from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_script(relative_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / relative_path
    spec = spec_from_file_location(module_name, script_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_build_external_source_lane_preserves_ready_and_gap_rows(tmp_path: Path) -> None:
    module = _load_script("scripts/external/build_external_mag_source_lane.py", "build_external_mag_source_lane")
    genome = tmp_path / "genomes_fna" / "mag1.fna"
    faa = tmp_path / "proteomes_faa" / "mag1.faa"
    gff = tmp_path / "genes_gff" / "mag1.gff"
    ffn = tmp_path / "genes_ffn" / "mag1.ffn"
    for path, text in [(genome, ">c1\nATGC\n"), (faa, ">p1\nMA\n"), (gff, "##gff-version 3\n"), (ffn, ">g1\nATG\n")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    input_manifest = tmp_path / "ready.tsv"
    proteome_manifest = tmp_path / "proteomes.tsv"
    gap_register = tmp_path / "gaps.tsv"
    source_lane = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    summary_json = tmp_path / "summary.json"

    _write_tsv(
        input_manifest,
        [
            {
                "proteome_id": "dataset__mag1",
                "mag_id": "mag1",
                "mag_fasta": str(genome),
                "source_group": "site_a",
                "domain": "d__Bacteria",
            }
        ],
    )
    _write_tsv(
        proteome_manifest,
        [
            {
                "proteome_id": "dataset__mag1",
                "mag_id": "mag1",
                "local_faa_path": str(faa),
                "local_ffn_path": str(ffn),
                "local_gff_path": str(gff),
                "protein_prediction_status": "predicted",
                "protein_count": "1200",
            }
        ],
    )
    _write_tsv(
        gap_register,
        [
            {
                "proteome_id": "dataset__mag2",
                "mag_id": "mag2",
                "gap_reason": "download_failed",
                "download_state": "failed",
            }
        ],
    )

    args = [
        "--repo-root",
        str(tmp_path),
        "--dataset-id",
        "dataset",
        "--source-project-id",
        "source_project",
        "--source",
        "dataset_source",
        "--input-manifest",
        str(input_manifest),
        "--proteome-manifest",
        str(proteome_manifest),
        "--gap-register",
        str(gap_register),
        "--include-gaps",
        "--output-source-lane",
        str(source_lane),
        "--output-functional-manifest",
        str(functional_manifest),
        "--summary-json",
        str(summary_json),
    ]
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["build_external_mag_source_lane.py", *args]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    lane_rows = _read_tsv(source_lane)
    functional_rows = _read_tsv(functional_manifest)
    by_id = {row["proteome_id"]: row for row in lane_rows}
    assert len(lane_rows) == 2
    assert by_id["dataset__mag1"]["match_status"] == "matched"
    assert by_id["dataset__mag1"]["functional_run_include"] == "true"
    assert by_id["dataset__mag1"]["esm2_include"] == "true"
    assert by_id["dataset__mag1"]["glm2_include"] == "true"
    assert by_id["dataset__mag2"]["match_status"] == "missing_payload"
    assert by_id["dataset__mag2"]["functional_run_include"] == "false"
    assert len(functional_rows) == 2
    summary = json.loads(summary_json.read_text())
    assert summary["functional_run_include"] == 1
    assert summary["gap_reason_counts"]["download_failed"] == 1


def test_build_source_provenance_checksums_preserves_urls_and_excludes_output(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/external/build_source_provenance_checksums.py",
        "build_source_provenance_checksums",
    )
    source_dir = tmp_path / "data/external/example/source_docs"
    output = source_dir / "source_file_checksums.tsv"
    paper = source_dir / "paper.pdf"
    nested = source_dir / "provider" / "metadata.json"
    ignored = source_dir / "local_ingestion_scripts" / "recover.py"
    for path, content in [
        (paper, b"paper bytes\n"),
        (nested, b"{\"ok\": true}\n"),
        (ignored, b"scratch helper\n"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _write_tsv(
        output,
        [
            {
                "artifact": "old",
                "path": str(paper.relative_to(tmp_path)),
                "size_bytes": "0",
                "md5": "",
                "sha256": "",
                "source_url": "https://example.org/paper.pdf",
            }
        ],
    )

    rows = module.collect_rows(
        tmp_path,
        source_dir,
        output,
        artifact_prefix="example",
        excludes=["local_ingestion_scripts"],
    )
    module.write_tsv(output, rows)
    written = _read_tsv(output)
    by_path = {row["path"]: row for row in written}

    assert len(written) == 2
    assert str(output.relative_to(tmp_path)) not in by_path
    assert str(ignored.relative_to(tmp_path)) not in by_path
    paper_row = by_path[str(paper.relative_to(tmp_path))]
    assert paper_row["source_url"] == "https://example.org/paper.pdf"
    assert paper_row["artifact"] == "example__paper"
    assert paper_row["md5"] == hashlib.md5(paper.read_bytes()).hexdigest()
    assert paper_row["sha256"] == hashlib.sha256(paper.read_bytes()).hexdigest()
    assert by_path[str(nested.relative_to(tmp_path))]["artifact"] == "example__provider__metadata"


def test_build_external_source_lane_rejects_duplicate_ready_proteome_ids(tmp_path: Path) -> None:
    module = _load_script("scripts/external/build_external_mag_source_lane.py", "build_external_mag_source_lane_duplicate_ready")
    input_manifest = tmp_path / "ready.tsv"
    source_lane = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    _write_tsv(
        input_manifest,
        [
            {"proteome_id": "dataset__mag1", "mag_id": "mag1"},
            {"proteome_id": "dataset__mag1", "mag_id": "mag1_duplicate"},
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_external_mag_source_lane.py",
            "--repo-root",
            str(tmp_path),
            "--dataset-id",
            "dataset",
            "--input-manifest",
            str(input_manifest),
            "--output-source-lane",
            str(source_lane),
            "--output-functional-manifest",
            str(functional_manifest),
        ]
        try:
            module.main()
        except SystemExit as exc:
            assert "input manifest has invalid proteome_id values" in str(exc)
            assert "duplicate proteome_id values: dataset__mag1" in str(exc)
        else:
            raise AssertionError("duplicate ready proteome_id should abort source-lane build")
    finally:
        sys.argv = old_argv


def test_build_external_source_lane_rejects_ready_gap_proteome_id_collision(tmp_path: Path) -> None:
    module = _load_script("scripts/external/build_external_mag_source_lane.py", "build_external_mag_source_lane_ready_gap_collision")
    input_manifest = tmp_path / "ready.tsv"
    gap_register = tmp_path / "gaps.tsv"
    source_lane = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    _write_tsv(input_manifest, [{"proteome_id": "dataset__mag1", "mag_id": "mag1"}])
    _write_tsv(
        gap_register,
        [
            {
                "proteome_id": "dataset__mag1",
                "mag_id": "mag1",
                "gap_reason": "download_failed",
            }
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_external_mag_source_lane.py",
            "--repo-root",
            str(tmp_path),
            "--dataset-id",
            "dataset",
            "--input-manifest",
            str(input_manifest),
            "--gap-register",
            str(gap_register),
            "--include-gaps",
            "--output-source-lane",
            str(source_lane),
            "--output-functional-manifest",
            str(functional_manifest),
        ]
        try:
            module.main()
        except SystemExit as exc:
            assert "combined source-lane manifest has invalid proteome_id values" in str(exc)
            assert "duplicate proteome_id values: dataset__mag1" in str(exc)
        else:
            raise AssertionError("ready/gap proteome_id collision should abort source-lane build")
    finally:
        sys.argv = old_argv


def test_build_external_source_lane_respects_row_analysis_unit_type(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/external/build_external_mag_source_lane.py",
        "build_external_mag_source_lane_analysis_unit_type",
    )
    genome = tmp_path / "genomes_fna" / "assembly.fna"
    faa = tmp_path / "proteomes_faa" / "assembly.faa"
    genome.parent.mkdir(parents=True, exist_ok=True)
    faa.parent.mkdir(parents=True, exist_ok=True)
    genome.write_text(">c1\nATGC\n")
    faa.write_text(">p1\nMA\n")

    input_manifest = tmp_path / "ready.tsv"
    source_lane = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    _write_tsv(
        input_manifest,
        [
            {
                "proteome_id": "dataset__assembly_context",
                "mag_id": "assembly",
                "mag_fasta": str(genome),
                "proteome_faa": str(faa),
                "analysis_unit_type": "assembly_context",
            }
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_external_mag_source_lane.py",
            "--repo-root",
            str(tmp_path),
            "--dataset-id",
            "dataset",
            "--analysis-unit-type",
            "mag_bin",
            "--input-manifest",
            str(input_manifest),
            "--output-source-lane",
            str(source_lane),
            "--output-functional-manifest",
            str(functional_manifest),
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    lane_rows = _read_tsv(source_lane)
    functional_rows = _read_tsv(functional_manifest)
    assert lane_rows[0]["analysis_unit_type"] == "assembly_context"
    assert lane_rows[0]["functional_run_include"] == "false"
    assert lane_rows[0]["mbag_mag_level_include"] == "false"
    assert lane_rows[0]["comparability_status"] == "blocked_missing_required_payload"
    assert functional_rows[0]["analysis_unit_type"] == "assembly_context"
    assert functional_rows[0]["functional_run_include"] == "false"


def test_lane_registry_summary_uses_warehouse_for_closed_lane(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/summarize_atlas_lane_registry.py", "summarize_atlas_lane_registry")
    manifest = tmp_path / "manifest.tsv"
    warehouse = tmp_path / "warehouse"
    esm = tmp_path / "esm"
    glm = tmp_path / "glm"
    registry = tmp_path / "lanes.tsv"

    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "mag_id": "m1", "functional_run_include": "true"},
            {"proteome_id": "p2", "mag_id": "m2", "functional_run_include": "true"},
        ],
    )
    _write_tsv(esm / "embedding_metadata.tsv", [{"proteome_id": "p1"}, {"proteome_id": "p2"}, {"proteome_id": "other"}])
    _write_tsv(glm / "feature_glm_mag_level.tsv", [{"proteome_id": "p1"}, {"proteome_id": "p2"}, {"proteome_id": "other"}])
    _write_tsv(warehouse / "cohort_table_manifest.tsv", [{"table": "dim_mag", "rows": "2"}])
    partial_run = tmp_path / "per_mag" / "p1" / "run1"
    partial_run.mkdir(parents=True)

    _write_tsv(
        registry,
        [
            {
                "lane_id": "closed",
                "lane_role": "calibration_core",
                "lane_status": "complete",
                "denominator_label": "test",
                "denominator_units": "2",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": str(tmp_path / "per_mag"),
                "functional_warehouse_dir": str(warehouse),
                "esm2_artifacts_dirs": str(esm),
                "glm2_artifacts_dirs": str(glm),
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )
    rows = [module.summarize_lane(tmp_path, row) for row in _read_tsv(registry)]
    assert rows[0]["esm2_units"] == 2
    assert rows[0]["glm2_units"] == 2
    assert rows[0]["functional_complete"] == 2
    assert rows[0]["functional_partial"] == 0
    assert rows[0]["functional_status_basis"] == "warehouse_dim_mag"
    assert rows[0]["manifest_gap_rows"] == 0
    assert rows[0]["tri_view_ready_units"] == 2
    assert rows[0]["consolidation_ready"] == "true"
    assert rows[0]["warehouse_current"] == "true"
    command_parts = shlex.split(rows[0]["consolidation_command"])
    assert command_parts[:3] == [
        "scripts/consolidate_functional_mag_cohort.py",
        "--repo-root",
        str(tmp_path),
    ]
    assert "--manifest" in command_parts
    assert command_parts[command_parts.index("--manifest") + 1] == str(manifest)
    assert "--expected-complete-count" in command_parts
    assert command_parts[command_parts.index("--expected-complete-count") + 1] == "2"
    assert "--per-mag-dir" in command_parts
    assert command_parts[command_parts.index("--per-mag-dir") + 1] == str(tmp_path / "per_mag")
    assert "--output-dir" in command_parts
    assert command_parts[command_parts.index("--output-dir") + 1] == str(warehouse)
    assert "--build-duckdb" in command_parts


def test_lane_registry_consolidation_command_keeps_multiple_per_mag_dirs(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/summarize_atlas_lane_registry.py",
        "summarize_atlas_lane_registry_consolidation_command",
    )
    row = {
        "lane_id": "external_lane",
        "consolidation_cohort_run_id": "external_2026",
        "functional_manifest": "manifests/functional.tsv",
        "functional_per_mag_dirs": "runs/archaea/per_mag;runs/bacteria_001/per_mag;runs/bacteria_002/per_mag",
        "consolidation_output_dir": "warehouses/external",
    }

    command_parts = shlex.split(module.consolidation_command(tmp_path, row, 3156))

    assert command_parts[:3] == [
        "scripts/consolidate_functional_mag_cohort.py",
        "--repo-root",
        str(tmp_path),
    ]
    assert command_parts[command_parts.index("--cohort-run-id") + 1] == "external_2026"
    assert command_parts[command_parts.index("--manifest") + 1] == "manifests/functional.tsv"
    assert command_parts[command_parts.index("--expected-complete-count") + 1] == "3156"
    per_mag_values = [
        command_parts[idx + 1]
        for idx, part in enumerate(command_parts)
        if part == "--per-mag-dir"
    ]
    assert per_mag_values == [
        "runs/archaea/per_mag",
        "runs/bacteria_001/per_mag",
        "runs/bacteria_002/per_mag",
    ]
    assert command_parts[command_parts.index("--output-dir") + 1] == "warehouses/external"
    assert command_parts[-1] == "--build-duckdb"


def test_atlas_lane_consolidation_gate_reports_blockers_and_ready_commands(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/check_atlas_lane_consolidation_gate.py",
        "check_atlas_lane_consolidation_gate",
    )
    blocked = {
        "lane_id": "blocked_lane",
        "functional_include_rows": 3,
        "functional_complete": 1,
        "functional_failed": 1,
        "functional_not_started": 1,
        "functional_partial": 0,
        "consolidation_ready": "false",
        "warehouse_current": "false",
        "consolidation_command": "scripts/consolidate_functional_mag_cohort.py --blocked",
    }
    ready = {
        "lane_id": "ready_lane",
        "functional_include_rows": 2,
        "functional_complete": 2,
        "functional_failed": 0,
        "functional_not_started": 0,
        "functional_partial": 0,
        "consolidation_ready": "true",
        "warehouse_current": "false",
        "consolidation_command": "scripts/consolidate_functional_mag_cohort.py --ready",
    }

    assert module.lane_blockers(blocked) == [
        "1 functional rows are failed",
        "1 functional rows are pending/partial",
        "functional_complete 1 is smaller than functional_include_rows 3",
        "consolidation_ready is not true",
    ]
    assert module.lane_blockers(ready) == []


def test_atlas_lane_consolidation_gate_selection_skips_current_warehouses(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/check_atlas_lane_consolidation_gate.py",
        "check_atlas_lane_consolidation_gate_selection",
    )
    current = {"lane_id": "current_lane", "warehouse_current": "true"}
    active = {"lane_id": "active_lane", "warehouse_current": "false"}

    assert module.selected_rows([current, active], [], include_current=False) == [active]
    assert module.selected_rows([current, active], [], include_current=True) == [current, active]
    assert module.selected_rows([current, active], ["current_lane"], include_current=False) == [current]


def test_atlas_report_rebuild_gate_blocks_incomplete_triview_lanes(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/check_atlas_report_rebuild_gate.py",
        "check_atlas_report_rebuild_gate",
    )
    incomplete = {
        "lane_id": "external_lane",
        "lane_role": "external_mangrove",
        "functional_include_rows": 10,
        "esm2_units": 10,
        "glm2_units": 9,
        "functional_complete": 8,
        "tri_view_ready_units": 8,
        "functional_failed": 1,
        "functional_not_started": 1,
        "functional_partial": 0,
    }

    blockers = module.lane_blockers(incomplete)

    assert "gLM2 units 9 are smaller than functional_include_rows 10" in blockers
    assert "functional units 8 are smaller than functional_include_rows 10" in blockers
    assert "tri-view units 8 are smaller than functional_include_rows 10" in blockers
    assert "1 functional rows are failed" in blockers
    assert "1 functional rows are pending/partial" in blockers


def test_atlas_report_rebuild_gate_accepts_complete_triview_lanes(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/check_atlas_report_rebuild_gate.py",
        "check_atlas_report_rebuild_gate_ready",
    )
    calibration = {
        "lane_id": "poc_core",
        "lane_role": "calibration_core",
        "warehouse_current": "true",
    }
    ready = {
        "lane_id": "external_lane",
        "lane_role": "external_mangrove",
        "functional_include_rows": 10,
        "esm2_units": 10,
        "glm2_units": 10,
        "functional_complete": 10,
        "tri_view_ready_units": 10,
        "functional_failed": 0,
        "functional_not_started": 0,
        "functional_partial": 0,
    }

    assert module.calibration_blockers([calibration, ready]) == []
    assert module.lane_blockers(ready) == []
    assert module.report_command("configs/methanet_atlas_lanes.tsv") == (
        "scripts/reports/build_mbag_expanded_multiview_atlas.py "
        "--lane-registry configs/methanet_atlas_lanes.tsv"
    )


def test_atlas_lane_readiness_wrapper_writes_summary_for_blocked_gates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / "scripts/reports/check_atlas_lane_readiness.sh"
    registry = tmp_path / "configs/methanet_atlas_lanes.tsv"
    output_dir = tmp_path / "results/reports"
    refresher = tmp_path / "fake_refresh.sh"
    consolidation_gate = tmp_path / "fake_consolidation_gate.py"
    report_gate = tmp_path / "fake_report_gate.py"
    overlap_audit = tmp_path / "fake_overlap_audit.py"
    squeue = tmp_path / "fake_squeue.sh"
    python_bin = sys.executable

    registry.parent.mkdir(parents=True)
    registry.write_text("lane_id\nexternal_lane\n")
    refresher.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "base=\"$OUTPUT_DIR/atlas_lane_registry_status_${STAMP}\"\n"
        "cat > \"${base}.json\" <<'JSON'\n"
        "[{\"lane_id\":\"external_lane\",\"functional_include_rows\":2,\"functional_complete\":1,"
        "\"functional_not_started\":1,\"functional_partial\":0,\"functional_failed\":0,"
        "\"esm2_units\":1,\"glm2_units\":1,\"tri_view_ready_units\":1},"
        "{\"lane_id\":\"second_lane\",\"functional_include_rows\":1,\"functional_complete\":1,"
        "\"functional_not_started\":0,\"functional_partial\":0,\"functional_failed\":0,"
        "\"esm2_units\":1,\"glm2_units\":1,\"tri_view_ready_units\":1}]\n"
        "JSON\n"
        "printf '# status\\n' > \"${base}.md\"\n"
        "printf '{}\\n' > \"${base}.validation.json\"\n"
        "printf 'validation_json\\t%s\\n' \"${base}.validation.json\"\n"
        "printf 'status_json\\t%s\\n' \"${base}.json\"\n"
        "printf 'status_md\\t%s\\n' \"${base}.md\"\n"
    )
    consolidation_gate.write_text(
        "import sys\n"
        "print('BLOCKED external_lane: still running')\n"
        "sys.exit(1)\n"
    )
    report_gate.write_text(
        "import sys\n"
        "print('BLOCKED external_lane: tri-view incomplete')\n"
        "sys.exit(1)\n"
    )
    overlap_audit.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "args = sys.argv\n"
        "for flag, text in [\n"
        "    ('--output-summary-tsv', 'lane_a\\tlane_b\\tshared_tokens_total\\n'),\n"
        "    ('--output-matches-tsv', 'lane_a\\tlane_b\\tfield\\tshared_token\\n'),\n"
        "    ('--output-json', '{\"summary\": []}\\n'),\n"
        "    ('--output-md', '# overlap\\n'),\n"
        "]:\n"
        "    if flag in args:\n"
        "        Path(args[args.index(flag) + 1]).write_text(text)\n"
        "print('READY overlap: no exact manifest overlap detected')\n"
    )
    squeue.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'JOBID NAME STATE TIME TIME_LIMIT NODES NODELIST(REASON)\\n'\n"
        "printf '123 test_job RUNNING 1:23 8:00:00 1 node-a\\n'\n"
    )
    for path in [refresher, consolidation_gate, report_gate, overlap_audit, squeue]:
        path.chmod(0o755)

    result = subprocess.run(
        [str(wrapper)],
        cwd=repo_root,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "REPO_ROOT": str(tmp_path),
            "LANE_REGISTRY": str(registry),
            "OUTPUT_DIR": str(output_dir),
            "STAMP": "20260621_1701",
            "PYTHON": python_bin,
            "REFRESHER": str(refresher),
            "CONSOLIDATION_GATE": str(consolidation_gate),
            "REPORT_GATE": str(report_gate),
            "OVERLAP_AUDIT": str(overlap_audit),
            "COMPLETION_CHECKLIST": str(repo_root / "scripts/reports/build_atlas_lane_completion_checklist.py"),
            "SQUEUE": str(squeue),
            "INCLUDE_SLURM": "1",
            "SLURM_USER": "tester",
            "LANE_IDS": "external_lane;second_lane",
            "STRICT_GATES": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    summary = output_dir / "atlas_lane_readiness_20260621_1701.md"
    assert result.returncode == 0
    assert "readiness_summary" in result.stdout
    assert "readiness_json" in result.stdout
    assert "overlap_md" in result.stdout
    assert "consolidation_gate_rc\t1" in result.stdout
    assert "report_gate_rc\t1" in result.stdout
    assert "overlap_audit_rc\t0" in result.stdout
    assert "completion_checklist_rc\t0" in result.stdout
    text = summary.read_text()
    assert "Lane selection: `external_lane second_lane`" in text
    assert "| Consolidation | BLOCKED |" in text
    assert "| Expanded atlas report | BLOCKED |" in text
    assert "| Exact manifest overlap audit | PASS |" in text
    assert "BLOCKED external_lane: still running" in text
    assert "BLOCKED external_lane: tri-view incomplete" in text
    assert "READY overlap: no exact manifest overlap detected" in text
    assert "## Slurm Queue Snapshot" in text
    assert "## Completion Checklist" in text
    assert "test_job RUNNING" in text
    assert "does not assign final sample methane-risk tiers" in text
    readiness = json.loads((output_dir / "atlas_lane_readiness_20260621_1701.json").read_text())
    assert readiness["lane_selection"] == ["external_lane", "second_lane"]
    assert readiness["gates"]["consolidation"]["result"] == "blocked"
    assert readiness["gates"]["expanded_atlas_report"]["result"] == "blocked"
    assert readiness["gates"]["exact_manifest_overlap"]["result"] == "pass"
    assert readiness["completion_checklist"]["result"] == "pass"
    assert readiness["slurm"]["captured"] is True
    assert "test_job RUNNING" in readiness["slurm"]["output"]


def test_atlas_lane_readiness_wrapper_reports_quiet_successful_overlap_audit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / "scripts/reports/check_atlas_lane_readiness.sh"
    registry = tmp_path / "configs/methanet_atlas_lanes.tsv"
    output_dir = tmp_path / "results/reports"
    refresher = tmp_path / "fake_refresh.sh"
    gate = tmp_path / "fake_gate.py"
    overlap_audit = tmp_path / "quiet_overlap_audit.py"

    registry.parent.mkdir(parents=True)
    registry.write_text("lane_id\nexternal_lane\n")
    refresher.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "base=\"$OUTPUT_DIR/atlas_lane_registry_status_${STAMP}\"\n"
        "cat > \"${base}.json\" <<'JSON'\n"
        "[{\"lane_id\":\"external_lane\",\"functional_include_rows\":1,\"functional_complete\":1,"
        "\"functional_not_started\":0,\"functional_partial\":0,\"functional_failed\":0,"
        "\"esm2_units\":1,\"glm2_units\":1,\"tri_view_ready_units\":1},"
        "{\"lane_id\":\"second_lane\",\"functional_include_rows\":1,\"functional_complete\":1,"
        "\"functional_not_started\":0,\"functional_partial\":0,\"functional_failed\":0,"
        "\"esm2_units\":1,\"glm2_units\":1,\"tri_view_ready_units\":1}]\n"
        "JSON\n"
        "printf '# status\\n' > \"${base}.md\"\n"
        "printf '{}\\n' > \"${base}.validation.json\"\n"
        "printf 'validation_json\\t%s\\n' \"${base}.validation.json\"\n"
        "printf 'status_json\\t%s\\n' \"${base}.json\"\n"
        "printf 'status_md\\t%s\\n' \"${base}.md\"\n"
    )
    gate.write_text("print('READY gate')\n")
    overlap_audit.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "args = sys.argv\n"
        "for flag, text in [\n"
        "    ('--output-summary-tsv', 'lane_a\\tlane_b\\tshared_tokens_total\\n'),\n"
        "    ('--output-matches-tsv', 'lane_a\\tlane_b\\tfield\\tshared_token\\n'),\n"
        "    ('--output-json', '{\"summary\": []}\\n'),\n"
        "    ('--output-md', '# overlap\\n'),\n"
        "]:\n"
        "    if flag in args:\n"
        "        Path(args[args.index(flag) + 1]).write_text(text)\n"
    )
    for path in [refresher, gate, overlap_audit]:
        path.chmod(0o755)

    result = subprocess.run(
        [str(wrapper)],
        cwd=repo_root,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "REPO_ROOT": str(tmp_path),
            "LANE_REGISTRY": str(registry),
            "OUTPUT_DIR": str(output_dir),
            "STAMP": "20260621_1702",
            "PYTHON": sys.executable,
            "REFRESHER": str(refresher),
            "CONSOLIDATION_GATE": str(gate),
            "REPORT_GATE": str(gate),
            "OVERLAP_AUDIT": str(overlap_audit),
            "COMPLETION_CHECKLIST": str(repo_root / "scripts/reports/build_atlas_lane_completion_checklist.py"),
            "INCLUDE_SLURM": "0",
            "LANE_IDS": "external_lane;second_lane",
            "STRICT_GATES": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    summary = output_dir / "atlas_lane_readiness_20260621_1702.md"
    assert result.returncode == 0
    assert "overlap_audit_rc\t0" in result.stdout
    assert "readiness_json" in result.stdout
    assert "PASS: exact manifest-overlap audit completed" in summary.read_text()
    readiness = json.loads((output_dir / "atlas_lane_readiness_20260621_1702.json").read_text())
    assert readiness["gates"]["exact_manifest_overlap"]["output"].startswith("PASS: exact manifest-overlap audit completed")


def test_atlas_lane_readiness_wrapper_can_emit_status_delta(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / "scripts/reports/check_atlas_lane_readiness.sh"
    registry = tmp_path / "configs/methanet_atlas_lanes.tsv"
    output_dir = tmp_path / "results/reports"
    previous_status = tmp_path / "previous_status.json"
    refresher = tmp_path / "fake_refresh.sh"
    gate = tmp_path / "fake_gate.py"

    registry.parent.mkdir(parents=True)
    registry.write_text("lane_id\nexternal_lane\n")
    previous_status.write_text(
        json.dumps(
            [
                {
                    "generated_utc": "2026-06-21T17:00:00+00:00",
                    "lane_id": "external_lane",
                    "functional_complete": 2,
                    "functional_not_started": 8,
                    "esm2_units": 2,
                    "glm2_units": 0,
                    "tri_view_ready_units": 0,
                }
            ]
        )
    )
    refresher.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "base=\"$OUTPUT_DIR/atlas_lane_registry_status_${STAMP}\"\n"
        "cat > \"${base}.json\" <<'JSON'\n"
        "[{\"generated_utc\":\"2026-06-21T18:00:00+00:00\",\"lane_id\":\"external_lane\","
        "\"functional_complete\":5,\"functional_not_started\":5,\"esm2_units\":5,"
        "\"glm2_units\":3,\"tri_view_ready_units\":3}]\n"
        "JSON\n"
        "printf '# status\\n' > \"${base}.md\"\n"
        "printf '{}\\n' > \"${base}.validation.json\"\n"
        "printf 'validation_json\\t%s\\n' \"${base}.validation.json\"\n"
        "printf 'status_json\\t%s\\n' \"${base}.json\"\n"
        "printf 'status_md\\t%s\\n' \"${base}.md\"\n"
    )
    gate.write_text("print('READY gate')\n")
    for path in [refresher, gate]:
        path.chmod(0o755)

    result = subprocess.run(
        [str(wrapper)],
        cwd=repo_root,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "REPO_ROOT": str(tmp_path),
            "LANE_REGISTRY": str(registry),
            "OUTPUT_DIR": str(output_dir),
            "STAMP": "20260621_1703",
            "PYTHON": sys.executable,
            "REFRESHER": str(refresher),
            "CONSOLIDATION_GATE": str(gate),
            "REPORT_GATE": str(gate),
            "OVERLAP_AUDIT": str(gate),
            "STATUS_COMPARE": str(repo_root / "scripts/reports/compare_atlas_lane_status.py"),
            "COMPLETION_CHECKLIST": str(repo_root / "scripts/reports/build_atlas_lane_completion_checklist.py"),
            "PREVIOUS_STATUS_JSON": str(previous_status),
            "RUN_OVERLAP_AUDIT": "0",
            "INCLUDE_SLURM": "0",
            "STRICT_GATES": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "status_delta_md" in result.stdout
    assert "status_delta_rc\t0" in result.stdout
    summary = output_dir / "atlas_lane_readiness_20260621_1703.md"
    delta_md = output_dir / "atlas_lane_registry_delta_20260621_1703.md"
    assert "## Status Delta" in summary.read_text()
    assert "PASS: status delta generated" in summary.read_text()
    delta_text = delta_md.read_text()
    assert "Functional complete delta across lanes: **+3**." in delta_text
    assert "Tri-view ready delta across lanes: **+3**." in delta_text
    readiness = json.loads((output_dir / "atlas_lane_readiness_20260621_1703.json").read_text())
    assert readiness["status_delta"]["result"] == "pass"
    assert readiness["completion_checklist"]["result"] == "pass"
    assert readiness["artifacts"]["previous_status_json"] == str(previous_status)
    assert readiness["artifacts"]["status_delta_markdown"] == str(delta_md)


def test_atlas_lane_readiness_wrapper_auto_selects_previous_status_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / "scripts/reports/check_atlas_lane_readiness.sh"
    registry = tmp_path / "configs/methanet_atlas_lanes.tsv"
    output_dir = tmp_path / "results/reports"
    refresher = tmp_path / "fake_refresh.sh"
    gate = tmp_path / "fake_gate.py"

    registry.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    registry.write_text("lane_id\nexternal_lane\n")
    previous_status = output_dir / "atlas_lane_registry_status_20260621_165000.json"
    validation_decoy = output_dir / "atlas_lane_registry_status_20990101_000000.validation.json"
    previous_status.write_text(
        json.dumps(
            [
                {
                    "generated_utc": "2026-06-21T16:50:00+00:00",
                    "lane_id": "external_lane",
                    "functional_complete": 1,
                    "functional_not_started": 9,
                    "tri_view_ready_units": 0,
                }
            ]
        )
    )
    validation_decoy.write_text("{}\n")
    refresher.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "base=\"$OUTPUT_DIR/atlas_lane_registry_status_${STAMP}\"\n"
        "cat > \"${base}.json\" <<'JSON'\n"
        "[{\"generated_utc\":\"2026-06-21T17:10:00+00:00\",\"lane_id\":\"external_lane\","
        "\"functional_complete\":4,\"functional_not_started\":6,\"tri_view_ready_units\":2}]\n"
        "JSON\n"
        "printf '# status\\n' > \"${base}.md\"\n"
        "printf '{}\\n' > \"${base}.validation.json\"\n"
        "printf 'validation_json\\t%s\\n' \"${base}.validation.json\"\n"
        "printf 'status_json\\t%s\\n' \"${base}.json\"\n"
        "printf 'status_md\\t%s\\n' \"${base}.md\"\n"
    )
    gate.write_text("print('READY gate')\n")
    for path in [refresher, gate]:
        path.chmod(0o755)

    result = subprocess.run(
        [str(wrapper)],
        cwd=repo_root,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "REPO_ROOT": str(tmp_path),
            "LANE_REGISTRY": str(registry),
            "OUTPUT_DIR": str(output_dir),
            "STAMP": "20260621_171000",
            "PYTHON": sys.executable,
            "REFRESHER": str(refresher),
            "CONSOLIDATION_GATE": str(gate),
            "REPORT_GATE": str(gate),
            "OVERLAP_AUDIT": str(gate),
            "STATUS_COMPARE": str(repo_root / "scripts/reports/compare_atlas_lane_status.py"),
            "COMPLETION_CHECKLIST": str(repo_root / "scripts/reports/build_atlas_lane_completion_checklist.py"),
            "PREVIOUS_STATUS_JSON": "auto",
            "RUN_OVERLAP_AUDIT": "0",
            "INCLUDE_SLURM": "0",
            "STRICT_GATES": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"Using latest previous status JSON: {previous_status}" in result.stdout
    readiness = json.loads((output_dir / "atlas_lane_readiness_20260621_171000.json").read_text())
    assert readiness["artifacts"]["previous_status_json"] == str(previous_status)
    assert readiness["status_delta"]["result"] == "pass"
    assert readiness["completion_checklist"]["result"] == "pass"
    delta_md = output_dir / "atlas_lane_registry_delta_20260621_171000.md"
    text = delta_md.read_text()
    assert "Functional complete delta across lanes: **+3**." in text
    assert "Tri-view ready delta across lanes: **+2**." in text


def test_atlas_lane_readiness_wrapper_default_stamp_has_seconds(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / "scripts/reports/check_atlas_lane_readiness.sh"
    registry = tmp_path / "configs/methanet_atlas_lanes.tsv"
    output_dir = tmp_path / "results/reports"
    refresher = tmp_path / "fake_refresh.sh"
    gate = tmp_path / "fake_gate.py"

    registry.parent.mkdir(parents=True)
    registry.write_text("lane_id\nexternal_lane\n")
    refresher.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "base=\"$OUTPUT_DIR/atlas_lane_registry_status_${STAMP}\"\n"
        "printf '[]\\n' > \"${base}.json\"\n"
        "printf '# status\\n' > \"${base}.md\"\n"
        "printf '{}\\n' > \"${base}.validation.json\"\n"
        "printf 'validation_json\\t%s\\n' \"${base}.validation.json\"\n"
        "printf 'status_json\\t%s\\n' \"${base}.json\"\n"
        "printf 'status_md\\t%s\\n' \"${base}.md\"\n"
    )
    gate.write_text("print('READY gate')\n")
    for path in [refresher, gate]:
        path.chmod(0o755)

    result = subprocess.run(
        [str(wrapper)],
        cwd=repo_root,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "REPO_ROOT": str(tmp_path),
            "LANE_REGISTRY": str(registry),
            "OUTPUT_DIR": str(output_dir),
            "PYTHON": sys.executable,
            "REFRESHER": str(refresher),
            "CONSOLIDATION_GATE": str(gate),
            "REPORT_GATE": str(gate),
            "OVERLAP_AUDIT": str(gate),
            "COMPLETION_CHECKLIST": str(repo_root / "scripts/reports/build_atlas_lane_completion_checklist.py"),
            "RUN_OVERLAP_AUDIT": "0",
            "INCLUDE_SLURM": "0",
            "STRICT_GATES": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    summary_line = next(line for line in result.stdout.splitlines() if line.startswith("readiness_summary\t"))
    readiness_path = Path(summary_line.split("\t", 1)[1])
    assert re.search(r"atlas_lane_readiness_\d{8}_\d{6}\.md$", readiness_path.name)
    assert readiness_path.exists()


def test_atlas_lane_overlap_audit_flags_exact_source_token_overlap(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/audit_atlas_lane_overlap.py", "audit_atlas_lane_overlap")
    registry = tmp_path / "configs/methanet_atlas_lanes.tsv"
    lane_a = tmp_path / "data/external/lane_a/source_lane.tsv"
    lane_b = tmp_path / "data/external/lane_b/source_lane.tsv"
    summary_tsv = tmp_path / "results/reports/overlap.summary.tsv"
    matches_tsv = tmp_path / "results/reports/overlap.matches.tsv"
    report_md = tmp_path / "results/reports/overlap.md"
    report_json = tmp_path / "results/reports/overlap.json"

    _write_tsv(
        lane_a,
        [
            {
                "proteome_id": "lane_a__mag1",
                "mag_id": "a_mag1",
                "functional_run_include": "true",
                "mapped_ncbi_bioprojects": "PRJNA1",
            },
            {
                "proteome_id": "lane_a__mag2",
                "mag_id": "a_mag2",
                "functional_run_include": "true",
                "mapped_ncbi_bioprojects": "PRJNA2",
            },
        ],
    )
    _write_tsv(
        lane_b,
        [
            {
                "proteome_id": "lane_b__mag1",
                "mag_id": "b_mag1",
                "functional_run_include": "true",
                "mapped_ncbi_bioprojects": "PRJNA1;PRJNA3",
            },
            {
                "proteome_id": "lane_b__gap",
                "mag_id": "b_gap",
                "functional_run_include": "false",
                "mapped_ncbi_bioprojects": "PRJNA2",
            },
        ],
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "lane_a",
                "lane_role": "external_mangrove",
                "source_lane_manifest": str(lane_a),
                "functional_manifest": str(lane_a),
            },
            {
                "lane_id": "lane_b",
                "lane_role": "external_mangrove",
                "source_lane_manifest": str(lane_b),
                "functional_manifest": str(lane_b),
            },
        ],
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "audit_atlas_lane_overlap.py",
            "--repo-root",
            str(tmp_path),
            "--lane-registry",
            str(registry),
            "--lane-id",
            "lane_a",
            "--lane-id",
            "lane_b",
            "--output-summary-tsv",
            str(summary_tsv),
            "--output-matches-tsv",
            str(matches_tsv),
            "--output-json",
            str(report_json),
            "--output-md",
            str(report_md),
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    summary = _read_tsv(summary_tsv)
    matches = _read_tsv(matches_tsv)
    assert summary[0]["shared_tokens_total"] == "1"
    assert summary[0]["fields_with_overlap"] == "mapped_ncbi_bioprojects"
    assert summary[0]["deduplication_action"] == "review_overlap"
    assert matches[0]["shared_token"] == "PRJNA1"
    assert matches[0]["lane_a_proteome_ids"] == "lane_a__mag1"
    assert matches[0]["lane_b_proteome_ids"] == "lane_b__mag1"
    assert "PRJNA2" not in {row["shared_token"] for row in matches}
    assert "not an ANI or genome-similarity result" in report_md.read_text()
    assert json.loads(report_json.read_text())["summary"][0]["shared_tokens_total"] == 1


def test_atlas_lane_overlap_audit_reports_no_exact_manifest_overlap(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/audit_atlas_lane_overlap.py", "audit_atlas_lane_overlap_none")
    lane_a = {
        "lane_id": "lane_a",
        "manifest_rows": 1,
        "selected_rows": 1,
        "index": module.lane_token_index(
            [{"proteome_id": "a1", "mag_id": "mag_a", "mapped_ncbi_bioprojects": "PRJNA1"}],
            module.DEFAULT_FIELDS,
        ),
    }
    lane_b = {
        "lane_id": "lane_b",
        "manifest_rows": 1,
        "selected_rows": 1,
        "index": module.lane_token_index(
            [{"proteome_id": "b1", "mag_id": "mag_b", "mapped_ncbi_bioprojects": "PRJNA2"}],
            module.DEFAULT_FIELDS,
        ),
    }

    summary, matches = module.compare_lanes(lane_a, lane_b, module.DEFAULT_FIELDS)

    assert matches == []
    assert summary["shared_tokens_total"] == 0
    assert summary["deduplication_action"] == "no_exact_manifest_overlap_detected"


def test_lane_registry_validator_accepts_valid_registry_and_rejects_duplicates(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry")
    manifest = tmp_path / "manifest.tsv"
    per_mag = tmp_path / "per_mag"
    esm = tmp_path / "esm"
    glm = tmp_path / "glm"
    warehouse = tmp_path / "warehouse"
    for path in [per_mag, esm, glm, warehouse]:
        path.mkdir(parents=True)
    _write_tsv(manifest, [{"proteome_id": "p1", "functional_run_include": "true"}])
    valid_registry = tmp_path / "valid_lanes.tsv"
    row = {
        "lane_id": "lane1",
        "lane_role": "external_mangrove",
        "lane_status": "active",
        "denominator_label": "test",
        "denominator_units": "1",
        "source_lane_manifest": str(manifest),
        "functional_manifest": str(manifest),
        "functional_per_mag_dirs": str(per_mag),
        "functional_warehouse_dir": str(warehouse),
        "consolidation_cohort_run_id": "lane1",
        "consolidation_output_dir": str(warehouse),
        "esm2_artifacts_dirs": str(esm),
        "glm2_artifacts_dirs": str(glm),
        "gap_register": "",
        "claim_scope": "MAG/proteome molecular screening",
        "notes": "",
    }
    _write_tsv(valid_registry, [row])

    report = module.validate(tmp_path, valid_registry)
    assert report["valid"] is True
    assert report["errors"] == []

    duplicate_registry = tmp_path / "duplicate_lanes.tsv"
    _write_tsv(duplicate_registry, [row, row])
    duplicate_report = module.validate(tmp_path, duplicate_registry)
    assert duplicate_report["valid"] is False
    assert any("duplicate lane_id" in error for error in duplicate_report["errors"])


def test_lane_registry_validator_warns_when_complete_lane_artifacts_lack_markers(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_markers")
    manifest = tmp_path / "manifest.tsv"
    per_mag = tmp_path / "per_mag"
    esm = tmp_path / "esm"
    glm = tmp_path / "glm"
    warehouse = tmp_path / "warehouse"
    for path in [per_mag, esm, glm, warehouse]:
        path.mkdir(parents=True)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_fasta": "p1.fna",
                "proteome_faa": "p1.faa",
                "functional_run_include": "true",
            }
        ],
    )
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        registry,
        [
            {
                "lane_id": "closed_missing_markers",
                "lane_role": "calibration_core",
                "lane_status": "complete",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": str(per_mag),
                "functional_warehouse_dir": str(warehouse),
                "consolidation_cohort_run_id": "closed_missing_markers",
                "consolidation_output_dir": str(warehouse),
                "esm2_artifacts_dirs": str(esm),
                "glm2_artifacts_dirs": str(glm),
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is True
    assert any("functional_warehouse_dir path lacks expected complete-lane marker" in warning for warning in report["warnings"])
    assert any("esm2_artifacts_dirs path lacks expected complete-lane marker" in warning for warning in report["warnings"])
    assert any("glm2_artifacts_dirs path lacks expected complete-lane marker" in warning for warning in report["warnings"])


def test_lane_registry_validator_allows_active_placeholder_artifact_dirs(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_active_placeholders")
    manifest = tmp_path / "manifest.tsv"
    esm = tmp_path / "esm"
    glm = tmp_path / "glm"
    warehouse = tmp_path / "warehouse"
    for path in [esm, glm, warehouse]:
        path.mkdir(parents=True)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "source": "test_source",
                "ecosystem": "mangrove_sediment",
                "domain": "d__Bacteria",
                "mag_fasta": "p1.fna",
                "proteome_faa": "p1.faa",
                "match_status": "matched",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "claim_scope": "MAG/proteome molecular screening",
                "comparability_status": "ready",
            }
        ],
    )
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        registry,
        [
            {
                "lane_id": "active_placeholders",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": str(warehouse),
                "consolidation_cohort_run_id": "active_placeholders",
                "consolidation_output_dir": str(warehouse),
                "esm2_artifacts_dirs": str(esm),
                "glm2_artifacts_dirs": str(glm),
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is True
    assert report["warnings"] == []


def test_lane_registry_validator_accepts_registered_source_provenance_paths(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_provenance_paths")
    manifest = tmp_path / "manifest.tsv"
    source_docs = tmp_path / "source_docs"
    source_file = source_docs / "paper_accessions.png"
    checksum_ledger = source_docs / "source_file_checksums.tsv"
    source_docs.mkdir(parents=True)
    source_file.write_bytes(b"accession screenshot bytes\n")
    payload = source_file.read_bytes()
    _write_tsv(
        checksum_ledger,
        [
            {
                "artifact": "paper_accessions",
                "path": str(source_file),
                "size_bytes": str(source_file.stat().st_size),
                "md5": hashlib.md5(payload).hexdigest(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_url": "https://example.org/paper",
            }
        ],
    )
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "source": "test_source",
                "ecosystem": "mangrove_sediment",
                "domain": "d__Bacteria",
                "mag_fasta": "p1.fna",
                "proteome_faa": "p1.faa",
                "match_status": "matched",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "claim_scope": "MAG/proteome molecular screening",
                "comparability_status": "ready",
            }
        ],
    )
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        registry,
        [
            {
                "lane_id": "external_with_provenance",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "external_with_provenance",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "source_provenance_dir": str(source_docs),
                "source_provenance_checksums": str(checksum_ledger),
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)

    assert report["valid"] is True
    assert report["warnings"] == []
    assert report["extra_columns"] == []


def test_lane_registry_validator_rejects_source_provenance_checksum_mismatch(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/validate_atlas_lane_registry.py",
        "validate_atlas_lane_registry_provenance_mismatch",
    )
    manifest = tmp_path / "manifest.tsv"
    source_docs = tmp_path / "source_docs"
    source_file = source_docs / "dataset.xlsx"
    checksum_ledger = source_docs / "source_file_checksums.tsv"
    source_docs.mkdir(parents=True)
    source_file.write_bytes(b"real payload\n")
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "source": "test_source",
                "ecosystem": "mangrove_sediment",
                "domain": "d__Bacteria",
                "mag_fasta": "p1.fna",
                "proteome_faa": "p1.faa",
                "match_status": "matched",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "claim_scope": "MAG/proteome molecular screening",
                "comparability_status": "ready",
            }
        ],
    )
    _write_tsv(
        checksum_ledger,
        [
            {
                "artifact": "bad_source_file",
                "path": str(source_file),
                "size_bytes": "1",
                "md5": "0" * 32,
                "sha256": "0" * 64,
                "source_url": "https://example.org/source",
            }
        ],
    )
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        registry,
        [
            {
                "lane_id": "external_with_bad_provenance",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "external_with_bad_provenance",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "source_provenance_dir": str(source_docs),
                "source_provenance_checksums": str(checksum_ledger),
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)

    assert report["valid"] is False
    assert any("size_bytes mismatch" in error for error in report["errors"])
    assert any("md5 mismatch" in error for error in report["errors"])
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_lane_registry_validator_rejects_checksum_paths_outside_source_provenance_dir(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/validate_atlas_lane_registry.py",
        "validate_atlas_lane_registry_provenance_outside_dir",
    )
    manifest = tmp_path / "manifest.tsv"
    source_docs = tmp_path / "source_docs"
    other_docs = tmp_path / "other_docs"
    source_file = other_docs / "wrong_dataset.json"
    checksum_ledger = source_docs / "source_file_checksums.tsv"
    source_docs.mkdir(parents=True)
    other_docs.mkdir(parents=True)
    source_file.write_bytes(b"wrong source payload\n")
    payload = source_file.read_bytes()
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "source": "test_source",
                "ecosystem": "mangrove_sediment",
                "domain": "d__Bacteria",
                "mag_fasta": "p1.fna",
                "proteome_faa": "p1.faa",
                "match_status": "matched",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "claim_scope": "MAG/proteome molecular screening",
                "comparability_status": "ready",
            }
        ],
    )
    _write_tsv(
        checksum_ledger,
        [
            {
                "artifact": "wrong_dataset",
                "path": str(source_file),
                "size_bytes": str(source_file.stat().st_size),
                "md5": hashlib.md5(payload).hexdigest(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_url": "https://example.org/wrong",
            }
        ],
    )
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        registry,
        [
            {
                "lane_id": "external_with_misregistered_provenance",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "external_with_misregistered_provenance",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "source_provenance_dir": str(source_docs),
                "source_provenance_checksums": str(checksum_ledger),
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)

    assert report["valid"] is False
    assert any("checksum path is outside registered source_provenance_dir" in error for error in report["errors"])
    assert any("checksum ledger does not cover files under source_provenance_dir" in error for error in report["errors"])


def test_lane_registry_validator_warns_when_external_source_manifest_is_not_normalized(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_external_columns")
    manifest = tmp_path / "manifest.tsv"
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "functional_run_include": "true",
            }
        ],
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "external_minimal",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "external_minimal",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is True
    assert any("external source-lane manifest missing normalized handoff columns" in warning for warning in report["warnings"])


def test_lane_registry_validator_rejects_malformed_lane_manifest(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_manifest")
    source_manifest = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        source_manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": "m1.fna",
                "proteome_faa": "m1.faa",
                "functional_run_include": "true",
            },
            {
                "proteome_id": "p1",
                "mag_id": "m1_dup",
                "mag_fasta": "m1_dup.fna",
                "proteome_faa": "m1_dup.faa",
                "functional_run_include": "maybe",
            },
        ],
    )
    _write_tsv(
        functional_manifest,
        [
            {
                "proteome_id": "p2",
                "mag_id": "m2",
                "mag_fasta": "",
                "proteome_faa": "m2.faa",
                "functional_run_include": "true",
            }
        ],
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "lane1",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "2",
                "source_lane_manifest": str(source_manifest),
                "functional_manifest": str(functional_manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "lane1",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is False
    assert any("duplicate proteome_id p1" in error for error in report["errors"])
    assert any("invalid boolean value for functional_run_include: maybe" in error for error in report["errors"])
    assert any("functional_run_include=true but mag_fasta is empty" in error for error in report["errors"])


def test_lane_registry_validator_rejects_denominator_smaller_than_functional_manifest(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_denominator")
    manifest = tmp_path / "functional.tsv"
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_fasta": "p1.fna",
                "proteome_faa": "p1.faa",
                "functional_run_include": "true",
            },
            {
                "proteome_id": "p2",
                "mag_fasta": "p2.fna",
                "proteome_faa": "p2.faa",
                "functional_run_include": "true",
            },
        ],
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "lane1",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "lane1",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is False
    assert any("denominator_units 1 is smaller than functional manifest rows 2" in error for error in report["errors"])
    assert any("denominator_units 1 is smaller than functional include rows 2" in error for error in report["errors"])


def test_lane_registry_validator_rejects_denominator_smaller_than_source_manifest(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_source_denominator")
    source_manifest = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        source_manifest,
        [
            {"proteome_id": "p1", "functional_run_include": "true", "match_status": "matched"},
            {"proteome_id": "p2", "functional_run_include": "false", "match_status": "missing_payload"},
        ],
    )
    _write_tsv(
        functional_manifest,
        [
            {
                "proteome_id": "p1",
                "mag_fasta": "p1.fna",
                "proteome_faa": "p1.faa",
                "functional_run_include": "true",
            }
        ],
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "lane1",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(source_manifest),
                "functional_manifest": str(functional_manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "lane1",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is False
    assert any("denominator_units 1 is smaller than source lane manifest rows 2" in error for error in report["errors"])


def test_lane_registry_validator_checks_gap_register_against_source_manifest(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/validate_atlas_lane_registry.py", "validate_atlas_lane_registry_gap_register")
    source_manifest = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    gap_register = tmp_path / "gaps.tsv"
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        source_manifest,
        [
            {
                "proteome_id": "ready",
                "mag_id": "m1",
                "mag_fasta": "ready.fna",
                "proteome_faa": "ready.faa",
                "functional_run_include": "true",
                "match_status": "matched",
            },
            {
                "proteome_id": "gap",
                "mag_id": "m2",
                "functional_run_include": "false",
                "match_status": "missing_payload",
            },
        ],
    )
    _write_tsv(
        functional_manifest,
        [
            {
                "proteome_id": "ready",
                "mag_fasta": "ready.fna",
                "proteome_faa": "ready.faa",
                "functional_run_include": "true",
            }
        ],
    )
    _write_tsv(gap_register, [{"proteome_id": "gap", "gap_reason": "download_failed"}])
    _write_tsv(
        registry,
        [
            {
                "lane_id": "lane1",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "2",
                "source_lane_manifest": str(source_manifest),
                "functional_manifest": str(functional_manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "lane1",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": str(gap_register),
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is True
    assert report["errors"] == []


def test_lane_registry_validator_rejects_gap_register_rows_not_matching_manifest_gaps(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/validate_atlas_lane_registry.py",
        "validate_atlas_lane_registry_bad_gap_register",
    )
    source_manifest = tmp_path / "source_lane.tsv"
    functional_manifest = tmp_path / "functional.tsv"
    gap_register = tmp_path / "gaps.tsv"
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        source_manifest,
        [
            {
                "proteome_id": "ready",
                "mag_id": "m1",
                "mag_fasta": "ready.fna",
                "proteome_faa": "ready.faa",
                "functional_run_include": "true",
                "match_status": "matched",
            },
            {
                "proteome_id": "gap",
                "mag_id": "m2",
                "functional_run_include": "false",
                "match_status": "missing_payload",
            },
        ],
    )
    _write_tsv(
        functional_manifest,
        [
            {
                "proteome_id": "ready",
                "mag_fasta": "ready.fna",
                "proteome_faa": "ready.faa",
                "functional_run_include": "true",
            }
        ],
    )
    _write_tsv(
        gap_register,
        [
            {"proteome_id": "ready", "gap_reason": "should_not_be_gap"},
            {"proteome_id": "unknown", "gap_reason": "not_in_manifest"},
        ],
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "lane1",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "2",
                "source_lane_manifest": str(source_manifest),
                "functional_manifest": str(functional_manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "consolidation_cohort_run_id": "lane1",
                "consolidation_output_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": str(gap_register),
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    report = module.validate(tmp_path, registry)
    assert report["valid"] is False
    assert any("proteome_id ready is runnable in source-lane manifest" in error for error in report["errors"])
    assert any("proteome_id ready is not match_status=missing_payload" in error for error in report["errors"])
    assert any("proteome_id unknown is absent from source-lane manifest" in error for error in report["errors"])
    assert any("source-lane missing-payload rows absent from gap register: gap" in error for error in report["errors"])


def test_lane_registry_summary_excludes_gap_rows_from_functional_queue_counts(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/summarize_atlas_lane_registry.py", "summarize_atlas_lane_registry_gap")
    manifest = tmp_path / "manifest.tsv"
    registry = tmp_path / "lanes.tsv"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "ready", "mag_id": "m1", "functional_run_include": "true"},
            {"proteome_id": "gap", "mag_id": "m2", "functional_run_include": "false"},
        ],
    )
    partial_run = tmp_path / "per_mag" / "ready" / "run1"
    partial_run.mkdir(parents=True)
    _write_tsv(
        registry,
        [
            {
                "lane_id": "active",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "2",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": str(tmp_path / "per_mag"),
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )
    row = module.summarize_lane(tmp_path, _read_tsv(registry)[0])
    assert row["functional_include_rows"] == 1
    assert row["manifest_gap_rows"] == 1
    assert row["functional_partial"] == 1
    assert row["functional_not_started"] == 0


def test_lane_registry_summary_deduplicates_esm2_units_across_artifact_dirs(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/summarize_atlas_lane_registry.py", "summarize_atlas_lane_registry_esm_dedupe")
    manifest = tmp_path / "manifest.tsv"
    registry = tmp_path / "lanes.tsv"
    esm1 = tmp_path / "esm1"
    esm2 = tmp_path / "esm2"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "mag_id": "m1", "functional_run_include": "true"},
            {"proteome_id": "p2", "mag_id": "m2", "functional_run_include": "true"},
            {"proteome_id": "p3", "mag_id": "m3", "functional_run_include": "true"},
        ],
    )
    _write_tsv(esm1 / "embedding_metadata.tsv", [{"proteome_id": "p1"}, {"proteome_id": "p2"}])
    _write_tsv(esm2 / "embedding_metadata.tsv", [{"proteome_id": "p2"}, {"proteome_id": "p3"}])
    _write_tsv(
        registry,
        [
            {
                "lane_id": "active",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "3",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": f"{esm1};{esm2}",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )
    row = module.summarize_lane(tmp_path, _read_tsv(registry)[0])
    assert row["esm2_units"] == 3
    assert row["esm2_evidence"] == "embedding_metadata.tsv"


def test_lane_registry_summary_caps_stats_based_esm2_units_to_selected_denominator(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/summarize_atlas_lane_registry.py", "summarize_atlas_lane_registry_esm_stats_cap")
    manifest = tmp_path / "manifest.tsv"
    registry = tmp_path / "lanes.tsv"
    esm1 = tmp_path / "esm_resume"
    esm2 = tmp_path / "esm_shard"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "mag_id": "m1", "functional_run_include": "true"},
            {"proteome_id": "p2", "mag_id": "m2", "functional_run_include": "true"},
            {"proteome_id": "gap", "mag_id": "m_gap", "functional_run_include": "false"},
        ],
    )
    (esm1 / "embedding_checkpoints").mkdir(parents=True)
    (esm2 / "embedding_checkpoints").mkdir(parents=True)
    (esm1 / "embedding_checkpoints/embedding_stats_partial.json").write_text(
        json.dumps({"embedded_total_with_resume": 3})
    )
    (esm2 / "embedding_checkpoints/embedding_stats_partial.json").write_text(
        json.dumps({"embedded_total_with_resume": 3})
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "active",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "3",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": f"{esm1};{esm2}",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    row = module.summarize_lane(tmp_path, _read_tsv(registry)[0])

    assert row["functional_include_rows"] == 2
    assert row["manifest_gap_rows"] == 1
    assert row["esm2_units"] == 2
    assert row["esm2_evidence"] == "embedding_stats"


def test_lane_registry_summary_deduplicates_esm2_checkpoint_metadata(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/summarize_atlas_lane_registry.py",
        "summarize_atlas_lane_registry_esm_checkpoint_dedupe",
    )
    manifest = tmp_path / "manifest.tsv"
    registry = tmp_path / "lanes.tsv"
    esm_full = tmp_path / "esm_full"
    esm_shard = tmp_path / "esm_shard"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "mag_id": "m1", "functional_run_include": "true"},
            {"proteome_id": "p2", "mag_id": "m2", "functional_run_include": "true"},
            {"proteome_id": "p3", "mag_id": "m3", "functional_run_include": "true"},
        ],
    )
    _write_tsv(
        esm_full / "embedding_checkpoints/checkpoint_metadata.tsv",
        [{"proteome_id": "p1"}, {"proteome_id": "p2"}],
    )
    _write_tsv(
        esm_shard / "embedding_checkpoints/checkpoint_metadata.tsv",
        [{"proteome_id": "p2"}, {"proteome_id": "p3"}],
    )
    _write_tsv(
        registry,
        [
            {
                "lane_id": "active",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "3",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": f"{esm_full};{esm_shard}",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    row = module.summarize_lane(tmp_path, _read_tsv(registry)[0])

    assert row["esm2_units"] == 3
    assert row["esm2_evidence"] == "checkpoint_metadata.tsv"


def test_lane_registry_summary_reports_duplicate_complete_attempts_without_inflating_units(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/summarize_atlas_lane_registry.py", "summarize_atlas_lane_registry_attempts")
    manifest = tmp_path / "manifest.tsv"
    registry = tmp_path / "lanes.tsv"
    per_mag = tmp_path / "per_mag"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "mag_id": "m1", "functional_run_include": "true"},
            {"proteome_id": "p2", "mag_id": "m2", "functional_run_include": "true"},
        ],
    )
    for run_name in ["run1", "run2"]:
        run_dir = per_mag / "p1" / run_name
        (run_dir / "curated").mkdir(parents=True)
        (run_dir / "COMPLETE").write_text("")
        (run_dir / "curated/run_record.json").write_text("{}")
    (per_mag / "p2" / "run1").mkdir(parents=True)
    _write_tsv(
        registry,
        [
            {
                "lane_id": "attempts",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "2",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": str(per_mag),
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    row = module.summarize_lane(tmp_path, _read_tsv(registry)[0])
    assert row["functional_complete"] == 1
    assert row["functional_complete_run_attempts"] == 2
    assert row["duplicate_complete_attempts"] == 1
    assert row["duplicate_complete_proteome_ids"] == 1
    assert row["functional_partial"] == 1


def test_lane_registry_summary_treats_newer_retry_as_partial_not_failed(tmp_path: Path) -> None:
    module = _load_script("scripts/reports/summarize_atlas_lane_registry.py", "summarize_atlas_lane_registry_retry")
    manifest = tmp_path / "manifest.tsv"
    registry = tmp_path / "lanes.tsv"
    per_mag = tmp_path / "per_mag"
    _write_tsv(
        manifest,
        [{"proteome_id": "retrying", "mag_id": "m1", "functional_run_include": "true"}],
    )
    failed_run = per_mag / "retrying" / "run_20260621_120000"
    failed_run.mkdir(parents=True)
    (failed_run / "FAILED").write_text("")
    active_retry = per_mag / "retrying" / "run_20260621_130000"
    active_retry.mkdir(parents=True)
    _write_tsv(
        registry,
        [
            {
                "lane_id": "retry_lane",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "1",
                "source_lane_manifest": str(manifest),
                "functional_manifest": str(manifest),
                "functional_per_mag_dirs": str(per_mag),
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    row = module.summarize_lane(tmp_path, _read_tsv(registry)[0])
    assert row["functional_complete"] == 0
    assert row["functional_failed"] == 0
    assert row["functional_partial"] == 1
    assert row["functional_failed_run_attempts"] == 1
    assert row["functional_partial_run_attempts"] == 1


def test_lane_registry_status_markdown_reports_full_attempt_audit() -> None:
    module = _load_script(
        "scripts/reports/summarize_atlas_lane_registry.py",
        "summarize_atlas_lane_registry_attempt_markdown",
    )
    text = module.markdown_report(
        [
            {
                "generated_utc": "2026-06-21T18:00:00+00:00",
                "lane_id": "retry_lane",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_units": 1,
                "esm2_units": 0,
                "glm2_units": 0,
                "functional_complete": 0,
                "tri_view_ready_units": 0,
                "functional_status_basis": "live_per_mag_sentinels",
                "functional_include_rows": 1,
                "manifest_gap_rows": 0,
                "functional_not_started": 0,
                "functional_partial": 1,
                "functional_failed": 0,
                "functional_complete_run_attempts": 0,
                "functional_failed_run_attempts": 1,
                "functional_partial_run_attempts": 1,
                "duplicate_complete_attempts": 0,
                "claim_scope": "MAG/proteome molecular screening",
                "source_provenance_dir": "",
                "source_provenance_checksums": "",
                "notes": "",
                "warehouse_current": "false",
                "consolidation_ready": "false",
            }
        ]
    )

    assert "Pending/partial functional rows: `1`; failed rows: `0`." in text
    assert "Attempt audit: `0` complete, `1` failed, `1` partial" in text


def test_expanded_atlas_report_loader_preserves_external_lane_status_rows(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "build_mbag_expanded_multiview_atlas",
    )
    manifest = tmp_path / "source_lane.tsv"
    esm = tmp_path / "esm"
    glm = tmp_path / "glm"
    per_mag = tmp_path / "per_mag"
    (per_mag / "ready" / "run1").mkdir(parents=True)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "ready",
                "mag_id": "mag_ready",
                "domain": "d__Bacteria",
                "functional_run_include": "true",
                "match_status": "matched",
            },
            {
                "proteome_id": "gap",
                "mag_id": "mag_gap",
                "domain": "d__Bacteria",
                "functional_run_include": "false",
                "match_status": "missing_payload",
            },
        ],
    )
    _write_tsv(esm / "embedding_metadata.tsv", [{"proteome_id": "ready", "n_proteins_used": "10"}])
    _write_tsv(
        glm / "features/glm2_smoke_window_embedding_summary.tsv",
        [
            {
                "proteome_id": "ready",
                "mag_id": "mag_ready",
                "window_id": "ready_native",
                "window_type": "native",
                "embedding_std": "0.5",
                "token_count": "100",
                "embedding_finite": "true",
                "model_name": "glm",
                "model_revision": "test",
            },
            {
                "proteome_id": "ready",
                "mag_id": "mag_ready",
                "window_id": "ready_shuffled",
                "window_type": "shuffled",
                "embedding_std": "0.2",
                "token_count": "100",
                "embedding_finite": "true",
                "model_name": "glm",
                "model_revision": "test",
            },
        ],
    )
    lane = module.pd.Series(
        {
            "lane_id": "external_test",
            "denominator_label": "External test lane",
            "source_lane_manifest": str(manifest),
            "functional_manifest": str(manifest),
            "functional_per_mag_dirs": str(per_mag),
            "esm2_artifacts_dirs": str(esm),
            "glm2_artifacts_dirs": str(glm),
            "claim_scope": "MAG/proteome molecular screening",
        }
    )

    frame, status, stats = module.load_external_lane_features(tmp_path, lane)
    by_id = {row["proteome_id"]: row for row in frame.to_dict(orient="records")}
    status_by_id = {row["proteome_id"]: row["functional_status"] for row in status.to_dict(orient="records")}

    assert len(frame) == 2
    assert by_id["ready"]["lane_id"] == "external_test"
    assert by_id["ready"]["has_esm2"] is True
    assert by_id["ready"]["has_glm2"] is True
    assert by_id["ready"]["has_functional"] is False
    assert by_id["ready"]["atlas_inclusion_status"] == "mangrove_esm_glm_only"
    assert status_by_id["ready"] == "partial"
    assert by_id["gap"]["atlas_inclusion_status"] == "manifest_gap_missing_payload"
    assert status_by_id["gap"] == "missing_payload"
    assert stats == {}


def test_expanded_atlas_discovers_multiwindow_glm2_summary(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "build_mbag_expanded_multiview_atlas_multiwindow",
    )
    glm_dir = tmp_path / "glm"
    summary = glm_dir / "features/glm2_multiwindow_summary.parquet"
    summary.parent.mkdir(parents=True)
    module.pd.DataFrame(
        [
            {
                "proteome_id": "wetland__mag1",
                "mag_id": "mag1",
                "n_native": 10,
                "n_shuffle": 10,
                "native_within_mag_dispersion": 0.12,
                "native_vs_shuffle_centroid_dist": 0.25,
                "native_vs_shuffle_matched_dist": 0.31,
                "matched_minus_dispersion_raw": 0.19,
                "permutation_p": 0.01,
                "model_name": "gLM2",
                "model_revision": "test",
            }
        ]
    ).to_parquet(summary, index=False)

    frame = module.aggregate_glm_dirs([glm_dir])

    assert frame["proteome_id"].tolist() == ["wetland__mag1"]
    assert bool(frame.loc[0, "has_glm2"]) is True
    assert frame.loc[0, "native_window_count"] == 10
    assert frame.loc[0, "shuffled_control_count"] == 10
    assert frame.loc[0, "glm_context_delta"] == 0.19
    assert (
        frame.loc[0, "context_qc_tier"]
        == "multiwindow_native_plus_shuffled_stability"
    )


def test_freeze_discovers_multiwindow_glm2_manifest(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_methanet_3view_payload_freeze.py",
        "build_methanet_3view_payload_freeze_multiwindow",
    )
    glm_dir = tmp_path / "glm"
    _write_tsv(
        glm_dir / "manifests/glm2_multiwindow_manifest.tsv",
        [
            {
                "proteome_id": "wetland__mag1",
                "window_id": "native_1",
            },
            {
                "proteome_id": "wetland__mag1",
                "window_id": "shuffle_1",
            },
        ],
    )

    assert module.glm2_ids([glm_dir], {"wetland__mag1", "missing"}) == {
        "wetland__mag1"
    }


def test_expanded_atlas_loads_source_scaffold_warehouse_without_mechanism_promotion(
    tmp_path: Path,
) -> None:
    module = _load_script(
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "build_mbag_expanded_multiview_atlas_source_scaffold",
    )
    manifest = tmp_path / "source_lane.tsv"
    esm = tmp_path / "esm"
    glm = tmp_path / "glm"
    warehouse = tmp_path / "warehouse"
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "mucc__mag1",
                "mag_id": "mag1",
                "ecosystem": "freshwater_wetland",
                "domain": "d__Archaea",
                "protein_count": "1000",
                "functional_run_include": "true",
                "match_status": "matched",
            }
        ],
    )
    _write_tsv(
        esm / "embedding_metadata.tsv",
        [{"proteome_id": "mucc__mag1", "n_proteins_used": "1000"}],
    )
    glm_summary = glm / "features/glm2_multiwindow_summary.parquet"
    glm_summary.parent.mkdir(parents=True)
    module.pd.DataFrame(
        [
            {
                "proteome_id": "mucc__mag1",
                "mag_id": "mag1",
                "n_native": 10,
                "n_shuffle": 10,
                "native_within_mag_dispersion": 0.1,
                "matched_minus_dispersion_raw": 0.2,
            }
        ]
    ).to_parquet(glm_summary, index=False)
    readiness_path = (
        warehouse
        / "parquet/feature_mrv_readiness_mag_level/cohort_run_id=test/part-00000.parquet"
    )
    readiness_path.parent.mkdir(parents=True)
    module.pd.DataFrame(
        [
            {
                "proteome_id": "mucc__mag1",
                "mag_id": "mag1",
                "bin_completeness": "92",
                "bin_contamination": "2",
                "source_feature_rows": "1500",
                "methane_term_rows": "12",
                "sulfur_term_rows": "8",
                "substrate_term_rows": "30",
                "review_priority_score": "21",
                "allowed_claim_wording": "Source-scaffold review only.",
                "blocking_gap": "Canonical annotations and ecological joins.",
                "next_validation_action": "Run canonical annotation workflow.",
            }
        ]
    ).to_parquet(readiness_path, index=False)
    dram_path = (
        warehouse
        / "parquet/feature_source_dram_mag_summary/cohort_run_id=test/part-00000.parquet"
    )
    dram_path.parent.mkdir(parents=True)
    module.pd.DataFrame(
        [
            {
                "proteome_id": "mucc__mag1",
                "mag_id": "mag1",
                "source_dram_rows": "1200",
                "ko_rows": "400",
                "cazy_rows": "20",
                "pfam_rows": "700",
                "peptidase_rows": "40",
            }
        ]
    ).to_parquet(dram_path, index=False)
    _write_tsv(
        warehouse / "cohort_table_manifest.tsv",
        [
            {
                "table": "feature_mrv_readiness_mag_level",
                "path": str(readiness_path),
                "rows": "1",
            },
            {
                "table": "feature_source_dram_mag_summary",
                "path": str(dram_path),
                "rows": "1",
            },
        ],
    )
    lane = module.pd.Series(
        {
            "lane_id": "mucc_test",
            "lane_role": "external_wetland",
            "denominator_label": "MUCC test",
            "source_lane_manifest": str(manifest),
            "functional_manifest": str(manifest),
            "functional_warehouse_dir": str(warehouse),
            "esm2_artifacts_dirs": str(esm),
            "glm2_artifacts_dirs": str(glm),
            "claim_scope": "source-scaffold molecular reference only",
        }
    )

    frame, status, _ = module.load_external_lane_features(tmp_path, lane)

    assert len(frame) == 1
    assert len(status) == 1
    assert bool(frame.loc[0, "has_esm2"]) is True
    assert bool(frame.loc[0, "has_glm2"]) is True
    assert bool(frame.loc[0, "has_functional"]) is True
    assert frame.loc[0, "source_category"] == "wetland"
    assert frame.loc[0, "functional_evidence_class"] == "source_annotation_scaffold"
    assert (
        frame.loc[0, "mechanism_equivalence_status"]
        == "not_canonical_mechanism_equivalent"
    )
    assert frame.loc[0, "methane_evidence_score"] == 12
    assert frame.loc[0, "substrate_evidence_count"] == 30


def test_expanded_atlas_report_loader_treats_newer_retry_as_partial(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "build_mbag_expanded_multiview_atlas_retry",
    )
    manifest = tmp_path / "source_lane.tsv"
    per_mag = tmp_path / "per_mag"
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "retrying",
                "mag_id": "mag_retry",
                "domain": "d__Bacteria",
                "functional_run_include": "true",
                "match_status": "matched",
            }
        ],
    )
    failed_run = per_mag / "retrying" / "run_20260621_120000"
    failed_run.mkdir(parents=True)
    (failed_run / "FAILED").write_text("")
    active_retry = per_mag / "retrying" / "run_20260621_130000"
    active_retry.mkdir(parents=True)

    _, status = module.discover_external_functional(
        module.read_tsv(manifest),
        [per_mag],
        "external_retry",
    )

    assert status.iloc[0]["proteome_id"] == "retrying"
    assert status.iloc[0]["functional_status"] == "partial"


def test_expanded_atlas_embedding_context_deduplicates_registered_artifact_dirs(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "build_mbag_expanded_multiview_atlas_dedupe",
    )
    esm1 = tmp_path / "esm1"
    esm2 = tmp_path / "esm2"
    esm1.mkdir()
    esm2.mkdir()
    np.savez_compressed(
        esm1 / "genome_embeddings.npz",
        embeddings=np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32),
        proteome_id=np.array(["p1", "p2"]),
        source=np.array(["poc", "mangrove"]),
        ecosystem=np.array(["rumen", "mangrove_sediment"]),
    )
    np.savez_compressed(
        esm2 / "genome_embeddings.npz",
        embeddings=np.array([[0.8, 0.2, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        proteome_id=np.array(["p2", "p3"]),
        source=np.array(["mangrove", "mangrove"]),
        ecosystem=np.array(["mangrove_sediment", "mangrove_sediment"]),
    )
    atlas = module.pd.DataFrame(
        [
            {"proteome_id": "p1", "atlas_inclusion_status": "poc_core_complete", "has_functional": True, "has_glm2": True},
            {"proteome_id": "p2", "atlas_inclusion_status": "mangrove_multiview_complete", "has_functional": True, "has_glm2": True},
            {"proteome_id": "p3", "atlas_inclusion_status": "mangrove_esm_glm_only", "has_functional": False, "has_glm2": True},
        ]
    )

    emb_meta, edge_df, embeddings = module.build_embedding_context_from_inputs(
        [
            {"path": esm1, "key_candidates": ["proteome_id"], "cohort_label": "first"},
            {"path": esm2, "key_candidates": ["proteome_id"], "cohort_label": "second"},
        ],
        atlas,
        k=1,
    )

    assert list(emb_meta["proteome_id"]) == ["p1", "p2", "p3"]
    assert embeddings.shape == (3, 3)
    assert emb_meta["proteome_id"].duplicated().sum() == 0
    assert set(edge_df["source"]).issubset({"p1", "p2", "p3"})


def test_expanded_atlas_builder_requires_lane_registry_by_default(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/reports/build_mbag_expanded_multiview_atlas.py"),
            "--repo-root",
            str(tmp_path),
            "--lane-registry",
            str(tmp_path / "missing_lanes.tsv"),
            "--output-dir",
            str(tmp_path / "report"),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Lane registry missing:" in result.stderr
    assert "--allow-legacy-defaults" in result.stderr


def test_expanded_atlas_report_registry_gate_rejects_ambiguous_control_plane(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "build_mbag_expanded_multiview_atlas_registry_gate",
    )
    registry_path = tmp_path / "lanes.tsv"
    registry = module.pd.DataFrame(
        [
            {
                "lane_id": "dup",
                "lane_role": "calibration_core",
                "denominator_units": "1",
                "source_lane_manifest": str(tmp_path / "missing_source.tsv"),
                "functional_manifest": str(tmp_path / "missing_functional.tsv"),
                "functional_warehouse_dir": str(tmp_path / "warehouse"),
                "esm2_artifacts_dirs": str(tmp_path / "esm"),
                "glm2_artifacts_dirs": str(tmp_path / "glm"),
            },
            {
                "lane_id": "dup",
                "lane_role": "experimental_future_role",
                "denominator_units": "not_int",
                "source_lane_manifest": "",
                "functional_manifest": "",
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
            },
        ]
    )

    try:
        module.validate_report_lane_registry(tmp_path, registry_path, registry)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("ambiguous registry should fail expanded-atlas report gate")

    assert "duplicate lane_id values: dup" in message
    assert "unsupported report lane_role values: experimental_future_role" in message
    assert "at least one external_mangrove" in message
    assert "denominator_units is not an integer" in message
    assert "registered path is missing for source_lane_manifest" in message


def test_expanded_atlas_report_registry_gate_accepts_registered_control_plane(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_mbag_expanded_multiview_atlas.py",
        "build_mbag_expanded_multiview_atlas_registry_gate_valid",
    )
    for path in [
        tmp_path / "poc_source.tsv",
        tmp_path / "poc_functional.tsv",
        tmp_path / "external_source.tsv",
        tmp_path / "external_functional.tsv",
    ]:
        _write_tsv(path, [{"proteome_id": "p1"}])
    for path in [tmp_path / "warehouse", tmp_path / "esm", tmp_path / "glm"]:
        path.mkdir()
    registry = module.pd.DataFrame(
        [
            {
                "lane_id": "poc",
                "lane_role": "calibration_core",
                "denominator_units": "1",
                "source_lane_manifest": str(tmp_path / "poc_source.tsv"),
                "functional_manifest": str(tmp_path / "poc_functional.tsv"),
                "functional_warehouse_dir": str(tmp_path / "warehouse"),
                "esm2_artifacts_dirs": str(tmp_path / "esm"),
                "glm2_artifacts_dirs": str(tmp_path / "glm"),
            },
            {
                "lane_id": "external",
                "lane_role": "external_mangrove",
                "denominator_units": "1",
                "source_lane_manifest": str(tmp_path / "external_source.tsv"),
                "functional_manifest": str(tmp_path / "external_functional.tsv"),
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
            },
        ]
    )

    module.validate_report_lane_registry(tmp_path, tmp_path / "lanes.tsv", registry)


def test_esm2_shard_submitter_renders_worker_commands_and_skips_empty_shards(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_one = shard_dir / "lane.shard_001.tsv"
    shard_two = shard_dir / "lane.shard_002.tsv"
    shard_one.write_text("proteome_id\tesm2_include\np1\ttrue\n")
    shard_two.write_text("proteome_id\tesm2_include\n")
    shard_manifest = shard_dir / "lane.shard_manifest.tsv"
    _write_tsv(
        shard_manifest,
        [
            {
                "shard": "001",
                "path": str(shard_one),
                "rows": "1",
                "start_index_1based": "1",
                "end_index_1based": "1",
            },
            {
                "shard": "002",
                "path": str(shard_two),
                "rows": "0",
                "start_index_1based": "",
                "end_index_1based": "",
            },
        ],
    )

    result = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts/submit_manifest_esm2_shards_apollo3.sh"),
        ],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "SHARD_MANIFEST": str(shard_manifest),
            "OUTPUT_DIR_TEMPLATE": str(tmp_path / "esm2_shard{shard}" / "artifacts"),
            "ESM2_INCLUDE_COL": "esm2_include",
            "JOB_NAME_PREFIX": "test_esm2",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Prepared ESM2 shard 001 (1 rows):" in result.stdout
    assert "test_esm2_s001" in result.stdout
    assert f"ESM2_MANIFEST={shard_one}" in result.stdout
    assert f"ESM2_OUTPUT_DIR={tmp_path / 'esm2_shard001' / 'artifacts'}" in result.stdout
    assert "ESM2_INCLUDE_COL=esm2_include" in result.stdout
    assert "shards_prepared\t1" in result.stdout
    assert "empty_shards_skipped\t1" in result.stdout


def test_esm2_shard_submitter_rejects_duplicate_shard_ids(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_one = shard_dir / "lane.shard_001.tsv"
    shard_two = shard_dir / "lane.shard_002.tsv"
    shard_one.write_text("proteome_id\tesm2_include\np1\ttrue\n")
    shard_two.write_text("proteome_id\tesm2_include\np2\ttrue\n")
    shard_manifest = shard_dir / "lane.shard_manifest.tsv"
    _write_tsv(
        shard_manifest,
        [
            {"shard": "001", "path": str(shard_one), "rows": "1"},
            {"shard": "001", "path": str(shard_two), "rows": "1"},
        ],
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/submit_manifest_esm2_shards_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "SHARD_MANIFEST": str(shard_manifest),
            "OUTPUT_DIR_TEMPLATE": str(tmp_path / "esm2_shard{shard}" / "artifacts"),
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Duplicate shard id in manifest" in result.stderr
    assert "001" in result.stderr
    assert "Prepared ESM2 shard" not in result.stdout


def test_esm2_shard_submitter_rejects_nonnumeric_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_one = shard_dir / "lane.shard_001.tsv"
    shard_one.write_text("proteome_id\tesm2_include\np1\ttrue\n")
    shard_manifest = shard_dir / "lane.shard_manifest.tsv"
    _write_tsv(shard_manifest, [{"shard": "001", "path": str(shard_one), "rows": "one"}])

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/submit_manifest_esm2_shards_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "SHARD_MANIFEST": str(shard_manifest),
            "OUTPUT_DIR_TEMPLATE": str(tmp_path / "esm2_shard{shard}" / "artifacts"),
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Shard rows must be a non-negative integer" in result.stderr
    assert "one" in result.stderr
    assert "Prepared ESM2 shard" not in result.stdout


def test_functional_submitter_renders_command_after_manifest_preflight(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "functional.tsv"
    worker = tmp_path / "run_functional_worker.sh"
    worker.write_text("#!/usr/bin/env bash\nexit 0\n")
    worker.chmod(0o755)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": "genomes/p1.fna",
                "proteome_faa": "proteomes/p1.faa",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "true",
                "match_status": "matched",
                "claim_scope": "MAG/proteome molecular screening",
            },
            {
                "proteome_id": "gap",
                "mag_id": "m2",
                "mag_fasta": "",
                "proteome_faa": "",
                "functional_run_include": "false",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "false",
                "match_status": "missing_payload",
                "claim_scope": "gap row",
            },
        ],
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/submit_functional_mag_batches_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "MANIFEST": str(manifest),
            "ARRAY_WORKER": str(worker),
            "RESULT_ROOT": str(tmp_path / "results"),
            "COHORT_RUN_ID": "test_lane",
            "DRY_RUN": "1",
            "CONCURRENCY": "2",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Prepared cohort batch command for 1 included MAGs" in result.stdout
    assert "--array=1-1%2" in result.stdout
    assert f"MANIFEST={manifest}" in result.stdout
    assert f"RESULT_BASE={tmp_path / 'results'}/per_mag" in result.stdout
    assert "DRY_RUN=1: not submitting." in result.stdout


def test_functional_submitter_rejects_duplicate_included_proteome_ids(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "functional.tsv"
    worker = tmp_path / "run_functional_worker.sh"
    worker.write_text("#!/usr/bin/env bash\nexit 0\n")
    worker.chmod(0o755)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": "genomes/p1.fna",
                "proteome_faa": "proteomes/p1.faa",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "true",
                "match_status": "matched",
            },
            {
                "proteome_id": "p1",
                "mag_id": "m1_dup",
                "mag_fasta": "genomes/p1_dup.fna",
                "proteome_faa": "proteomes/p1_dup.faa",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "true",
                "match_status": "matched",
            },
        ],
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/submit_functional_mag_batches_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "MANIFEST": str(manifest),
            "ARRAY_WORKER": str(worker),
            "RESULT_ROOT": str(tmp_path / "results"),
            "COHORT_RUN_ID": "test_lane",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "duplicate included proteome_id: p1" in result.stderr
    assert "Functional manifest preflight failed" in result.stderr
    assert "Prepared cohort batch command" not in result.stdout


def test_functional_submitter_rejects_missing_payload_in_included_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "functional.tsv"
    worker = tmp_path / "run_functional_worker.sh"
    worker.write_text("#!/usr/bin/env bash\nexit 0\n")
    worker.chmod(0o755)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": "",
                "proteome_faa": "proteomes/p1.faa",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "true",
                "match_status": "missing_payload",
            }
        ],
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/submit_functional_mag_batches_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "MANIFEST": str(manifest),
            "ARRAY_WORKER": str(worker),
            "RESULT_ROOT": str(tmp_path / "results"),
            "COHORT_RUN_ID": "test_lane",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "included row 2 proteome_id=p1 has empty mag_fasta" in result.stderr
    assert "Prepared cohort batch command" not in result.stdout


def test_functional_submitter_rejects_included_missing_payload_status(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "functional.tsv"
    worker = tmp_path / "run_functional_worker.sh"
    worker.write_text("#!/usr/bin/env bash\nexit 0\n")
    worker.chmod(0o755)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": "genomes/p1.fna",
                "proteome_faa": "proteomes/p1.faa",
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "true",
                "match_status": "missing_payload",
            }
        ],
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/submit_functional_mag_batches_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "MANIFEST": str(manifest),
            "ARRAY_WORKER": str(worker),
            "RESULT_ROOT": str(tmp_path / "results"),
            "COHORT_RUN_ID": "test_lane",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "included row 2 proteome_id=p1 has match_status=missing_payload" in result.stderr
    assert "Prepared cohort batch command" not in result.stdout


def test_functional_array_worker_marks_failed_runner_attempt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "functional.tsv"
    genome = tmp_path / "genomes" / "p1.fna"
    proteome = tmp_path / "proteomes" / "p1.faa"
    result_base = tmp_path / "per_mag"
    runner = tmp_path / "failing_runner.sh"
    genome.parent.mkdir(parents=True)
    proteome.parent.mkdir(parents=True)
    genome.write_text(">c1\nATGC\n")
    proteome.write_text(">p1\nMA\n")
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$RESULT_ROOT/logs\"\n"
        "printf 'runner saw %s\\n' \"$PROTEOME_ID\" > \"$RESULT_ROOT/logs/driver.out\"\n"
        "exit 7\n"
    )
    runner.chmod(0o755)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": str(genome),
                "proteome_faa": str(proteome),
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "true",
                "match_status": "matched",
                "claim_scope": "MAG/proteome molecular screening",
            }
        ],
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/slurm/run_functional_mag_array_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "MANIFEST": str(manifest),
            "RESULT_BASE": os.path.relpath(result_base, repo_root),
            "RUNNER": str(runner),
            "TASK_INDEX": "1",
            "RUN_ID": "test_run",
            "COHORT_RUN_ID": "test_lane",
        },
        text=True,
        capture_output=True,
    )

    run_dir = result_base / "p1" / "test_run"
    assert result.returncode == 7
    assert (run_dir / "FAILED").exists()
    assert not (run_dir / "COMPLETE").exists()
    failure = {}
    for line in (run_dir / "failure.tsv").read_text().splitlines():
        key, value = line.split("\t", 1)
        failure[key] = value
    assert failure["proteome_id"] == "p1"
    assert failure["mag_id"] == "m1"
    assert failure["run_id"] == "test_run"
    assert failure["exit_code"] == "7"
    assert "runner saw p1" in (run_dir / "logs" / "driver.out").read_text()


def test_functional_array_worker_dry_run_absolutizes_relative_result_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "functional.tsv"
    genome = tmp_path / "genomes" / "p1.fna"
    proteome = tmp_path / "proteomes" / "p1.faa"
    runner = tmp_path / "runner.sh"
    genome.parent.mkdir(parents=True)
    proteome.parent.mkdir(parents=True)
    genome.write_text(">c1\nATGC\n")
    proteome.write_text(">p1\nMA\n")
    runner.write_text("#!/usr/bin/env bash\nexit 0\n")
    runner.chmod(0o755)
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": str(genome),
                "proteome_faa": str(proteome),
                "functional_run_include": "true",
                "analysis_unit_type": "mag_bin",
                "mbag_mag_level_include": "true",
                "match_status": "matched",
                "claim_scope": "MAG/proteome molecular screening",
            }
        ],
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/slurm/run_functional_mag_array_apollo3.sh")],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "MANIFEST": str(manifest),
            "RESULT_ROOT": "results/functional_metagenomics/relative_result_root_check",
            "RUNNER": str(runner),
            "TASK_INDEX": "1",
            "RUN_ID": "test_run",
            "COHORT_RUN_ID": "test_lane",
            "ARRAY_DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert f"result_root\t{repo_root / 'results/functional_metagenomics/relative_result_root_check'}" in result.stdout
    assert "result_root\tresults/functional_metagenomics/relative_result_root_check" not in result.stdout


def test_split_manifest_shards_filters_sorts_and_records_empty_shards(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "source_lane.tsv"
    output_dir = tmp_path / "shards"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p3", "esm2_include": "true", "payload": "c"},
            {"proteome_id": "p1", "esm2_include": "true", "payload": "a"},
            {"proteome_id": "p2", "esm2_include": "false", "payload": "b"},
        ],
    )

    result = subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/external/split_manifest_shards.py"),
            "--input",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--prefix",
            "lane",
            "--shards",
            "3",
            "--include-col",
            "esm2_include",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    shard_manifest = _read_tsv(output_dir / "lane.shard_manifest.tsv")
    assert "input_rows\t2" in result.stdout
    assert "shards\t3" in result.stdout
    assert [row["rows"] for row in shard_manifest] == ["1", "1", "0"]
    assert [row["start_index_1based"] for row in shard_manifest] == ["1", "2", ""]
    assert [row["end_index_1based"] for row in shard_manifest] == ["1", "2", ""]
    assert [row["proteome_id"] for row in _read_tsv(output_dir / "lane.shard_001.tsv")] == ["p1"]
    assert [row["proteome_id"] for row in _read_tsv(output_dir / "lane.shard_002.tsv")] == ["p3"]
    assert _read_tsv(output_dir / "lane.shard_003.tsv") == []


def test_split_manifest_shards_rejects_duplicate_selected_ids(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "source_lane.tsv"
    output_dir = tmp_path / "shards"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "esm2_include": "true"},
            {"proteome_id": "p1", "esm2_include": "true"},
        ],
    )

    result = subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/external/split_manifest_shards.py"),
            "--input",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--prefix",
            "lane",
            "--shards",
            "2",
            "--include-col",
            "esm2_include",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Selected shard rows have invalid proteome_id values" in result.stderr
    assert "duplicate proteome_id values: p1" in result.stderr
    assert not (output_dir / "lane.shard_manifest.tsv").exists()


def test_split_manifest_shards_allows_duplicates_excluded_by_filter(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = tmp_path / "source_lane.tsv"
    output_dir = tmp_path / "shards"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "esm2_include": "false"},
            {"proteome_id": "p1", "esm2_include": "true"},
        ],
    )

    subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/external/split_manifest_shards.py"),
            "--input",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--prefix",
            "lane",
            "--shards",
            "2",
            "--include-col",
            "esm2_include",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    shard_manifest = _read_tsv(output_dir / "lane.shard_manifest.tsv")
    assert [row["rows"] for row in shard_manifest] == ["1", "0"]
    assert [row["proteome_id"] for row in _read_tsv(output_dir / "lane.shard_001.tsv")] == ["p1"]


def test_predict_external_mag_proteomes_filters_and_reuses_existing_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_prodigal = bin_dir / "prodigal"
    fake_prodigal.write_text("#!/usr/bin/env bash\nexit 99\n")
    fake_prodigal.chmod(0o755)

    genome = tmp_path / "genomes" / "p1.fna"
    faa = tmp_path / "proteomes" / "p1.faa"
    gff = tmp_path / "gff" / "m1.gff"
    genome.parent.mkdir(parents=True)
    faa.parent.mkdir(parents=True)
    gff.parent.mkdir(parents=True)
    genome.write_text(">contig1\nATGC\n")
    faa.write_text(">prot1\nMA\n>prot2\nMT\n")
    gff.write_text("##gff-version 3\n")

    manifest = tmp_path / "source_lane.tsv"
    output = tmp_path / "proteome_manifest.tsv"
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p2",
                "mag_id": "m2",
                "mag_fasta": str(tmp_path / "missing.fna"),
                "proteome_faa": str(tmp_path / "missing.faa"),
                "protein_prediction_include": "false",
            },
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": str(genome),
                "proteome_faa": str(faa),
                "protein_prediction_include": "true",
            },
        ],
    )

    result = subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/external/predict_external_mag_proteomes.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--output-manifest",
            str(output),
            "--proteome-dir",
            str(tmp_path / "proteomes"),
            "--ffn-dir",
            str(tmp_path / "ffn"),
            "--gff-dir",
            str(tmp_path / "gff"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--include-col",
            "protein_prediction_include",
            "--workers",
            "1",
        ],
        cwd=repo_root,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=True,
    )

    rows = _read_tsv(output)
    assert "existing\t2\tm1" in result.stdout
    assert len(rows) == 1
    assert rows[0]["proteome_id"] == "p1"
    assert rows[0]["protein_prediction_status"] == "existing"
    assert rows[0]["protein_count"] == "2"
    assert rows[0]["note"] == "Existing FAA/GFF reused."


def test_predict_external_mag_proteomes_rejects_duplicate_selected_proteome_ids(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_prodigal = bin_dir / "prodigal"
    fake_prodigal.write_text("#!/usr/bin/env bash\nexit 99\n")
    fake_prodigal.chmod(0o755)

    manifest = tmp_path / "source_lane.tsv"
    output = tmp_path / "proteome_manifest.tsv"
    _write_tsv(
        manifest,
        [
            {"proteome_id": "p1", "mag_id": "m1", "mag_fasta": str(tmp_path / "p1.fna")},
            {"proteome_id": "p1", "mag_id": "m1_dup", "mag_fasta": str(tmp_path / "p1_dup.fna")},
        ],
    )

    result = subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/external/predict_external_mag_proteomes.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--output-manifest",
            str(output),
            "--proteome-dir",
            str(tmp_path / "proteomes"),
            "--ffn-dir",
            str(tmp_path / "ffn"),
            "--gff-dir",
            str(tmp_path / "gff"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--workers",
            "1",
        ],
        cwd=repo_root,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "selected protein-prediction rows has invalid proteome_id values" in result.stderr
    assert "duplicate proteome_id values: p1" in result.stderr
    assert not output.exists()


def test_predict_external_mag_proteomes_allows_duplicates_excluded_by_include_filter(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_prodigal = bin_dir / "prodigal"
    fake_prodigal.write_text("#!/usr/bin/env bash\nexit 99\n")
    fake_prodigal.chmod(0o755)

    genome = tmp_path / "genomes" / "p1.fna"
    faa = tmp_path / "proteomes" / "p1.faa"
    gff = tmp_path / "gff" / "m1.gff"
    genome.parent.mkdir(parents=True)
    faa.parent.mkdir(parents=True)
    gff.parent.mkdir(parents=True)
    genome.write_text(">contig1\nATGC\n")
    faa.write_text(">prot1\nMA\n")
    gff.write_text("##gff-version 3\n")

    manifest = tmp_path / "source_lane.tsv"
    output = tmp_path / "proteome_manifest.tsv"
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "p1",
                "mag_id": "m1_excluded",
                "mag_fasta": str(tmp_path / "excluded.fna"),
                "proteome_faa": "",
                "protein_prediction_include": "false",
            },
            {
                "proteome_id": "p1",
                "mag_id": "m1",
                "mag_fasta": str(genome),
                "proteome_faa": str(faa),
                "protein_prediction_include": "true",
            },
        ],
    )

    subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/external/predict_external_mag_proteomes.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--output-manifest",
            str(output),
            "--proteome-dir",
            str(tmp_path / "proteomes"),
            "--ffn-dir",
            str(tmp_path / "ffn"),
            "--gff-dir",
            str(tmp_path / "gff"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--include-col",
            "protein_prediction_include",
            "--workers",
            "1",
        ],
        cwd=repo_root,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=True,
    )

    rows = _read_tsv(output)
    assert len(rows) == 1
    assert rows[0]["proteome_id"] == "p1"
    assert rows[0]["mag_id"] == "m1"


def test_glm2_source_lane_manifest_resolves_local_payload_paths(tmp_path: Path) -> None:
    module = _load_script("scripts/contextual_genomics/build_glm2_smoke_inputs.py", "build_glm2_smoke_inputs_source_lane")
    genome = tmp_path / "genomes" / "mag1.fna"
    faa = tmp_path / "proteomes" / "mag1.faa"
    gff = tmp_path / "gff" / "mag1.gff"
    for path in [genome, faa, gff]:
        path.parent.mkdir(parents=True, exist_ok=True)
    genome.write_text(">contig1\nATGAAATTTGGGCCC\n")
    faa.write_text(">gene1\nMKFGP\n")
    gff.write_text("##gff-version 3\ncontig1\tProdigal\tCDS\t1\t15\t.\t+\t0\tID=gene1;protein_id=gene1\n")

    manifest = tmp_path / "source_lane.tsv"
    _write_tsv(
        manifest,
        [
            {
                "proteome_id": "lane__mag1",
                "mag_id": "mag1",
                "source": "lane",
                "ecosystem": "mangrove_sediment",
                "domain": "d__Bacteria",
                "local_fna_path": str(genome),
                "local_faa_path": str(faa),
                "local_gff_path": str(gff),
                "protein_count": "1",
            }
        ],
    )

    selected = module.select_manifest_mags(
        tmp_path,
        manifest,
        "source_lane",
        "lane_glm2",
        tmp_path / "missing_bridge.tsv",
        False,
        0,
    )

    assert len(selected) == 1
    row = selected[0]
    assert row["proteome_id"] == "lane__mag1"
    assert row["sample"] == "lane__mag1"
    assert row["source_fna"] == str(genome)
    assert row["source_faa"] == str(faa)
    assert row["source_gff"] == str(gff)
    assert row["mag_fasta_basename"] == "mag1.fna"
    assert row["n_proteins_used"] == "1"
    assert row["embedded_final_662"] == "false"
    assert row["cohort_run_id"] == "lane"
    assert row["smoke_group"] == "lane_glm2"


def test_glm2_payload_submitter_renders_worker_commands_and_skips_unprepared_dirs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    prepared = tmp_path / "glm2_prepared"
    unprepared = tmp_path / "glm2_empty"
    (prepared / "prepared_inputs").mkdir(parents=True)
    (unprepared / "prepared_inputs").mkdir(parents=True)
    (prepared / "prepared_inputs/glm2_sequences.jsonl").write_text("{}\n")

    result = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts/submit_glm2_payload_dirs_apollo3.sh"),
        ],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "GLM2_RESULTS_DIRS": f"{prepared};{unprepared}",
            "JOB_NAME_PREFIX": "test_glm2",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert f"Prepared gLM2 payload {prepared}:" in result.stdout
    assert "test_glm2_glm2_prepared" in result.stdout
    assert f"RESULTS_DIR={prepared}" in result.stdout
    assert f"--output={prepared}/logs/slurm-%j.out" in result.stdout
    assert f"Skipping unprepared gLM2 payload: {unprepared}" in result.stdout
    assert "payloads_prepared\t1" in result.stdout
    assert "unprepared_payloads_skipped\t1" in result.stdout


def test_glm2_payload_submitter_rejects_duplicate_results_dirs_before_rendering(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    prepared = tmp_path / "glm2_prepared"
    (prepared / "prepared_inputs").mkdir(parents=True)
    (prepared / "prepared_inputs/glm2_sequences.jsonl").write_text("{}\n")

    result = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts/submit_glm2_payload_dirs_apollo3.sh"),
        ],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "GLM2_RESULTS_DIRS": f"{prepared};{prepared}",
            "JOB_NAME_PREFIX": "test_glm2",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Duplicate gLM2 results directory in launch list" in result.stderr
    assert str(prepared) in result.stderr
    assert "Prepared gLM2 payload" not in result.stdout


def test_lane_registry_refresh_wrapper_renders_timestamped_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    registry = tmp_path / "lanes.tsv"
    output_dir = tmp_path / "reports"
    _write_tsv(
        registry,
        [
            {
                "lane_id": "lane",
                "lane_role": "external_mangrove",
                "lane_status": "active",
                "denominator_label": "test",
                "denominator_units": "0",
                "source_lane_manifest": "",
                "functional_manifest": "",
                "functional_per_mag_dirs": "",
                "functional_warehouse_dir": "",
                "esm2_artifacts_dirs": "",
                "glm2_artifacts_dirs": "",
                "gap_register": "",
                "claim_scope": "MAG/proteome molecular screening",
                "notes": "",
            }
        ],
    )

    result = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts/reports/refresh_atlas_lane_registry_status.sh"),
        ],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(repo_root),
            "LANE_REGISTRY": str(registry),
            "OUTPUT_DIR": str(output_dir),
            "STAMP": "20990102_0304",
            "DRY_RUN": "1",
            "PYTHON": str(repo_root / ".venv/bin/python"),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Prepared lane-registry validation command:" in result.stdout
    assert "Prepared lane-registry refresh command:" in result.stdout
    assert f"--lane-registry {registry}" in result.stdout
    assert f"--output-json {output_dir}/atlas_lane_registry_status_20990102_0304.validation.json" in result.stdout
    assert "--allow-missing-optional-paths" in result.stdout
    assert f"--output-tsv {output_dir}/atlas_lane_registry_status_20990102_0304.tsv" in result.stdout
    assert f"--output-json {output_dir}/atlas_lane_registry_status_20990102_0304.json" in result.stdout
    assert f"--output-md {output_dir}/atlas_lane_registry_status_20990102_0304.md" in result.stdout
    assert "DRY_RUN=1: not writing status artifacts." in result.stdout


def test_compare_atlas_lane_status_reports_progress_deltas(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/compare_atlas_lane_status.py",
        "compare_atlas_lane_status",
    )
    previous = tmp_path / "previous.json"
    current = tmp_path / "current.json"
    output_md = tmp_path / "delta.md"
    output_tsv = tmp_path / "delta.tsv"
    previous.write_text(
        json.dumps(
            [
                {
                    "generated_utc": "2026-06-21T17:00:00+00:00",
                    "lane_id": "external_lane",
                    "functional_complete": 10,
                    "functional_not_started": 90,
                    "functional_partial": 0,
                    "functional_failed": 0,
                    "esm2_units": 20,
                    "glm2_units": 5,
                    "tri_view_ready_units": 5,
                    "functional_status_basis": "live_per_mag_sentinels",
                }
            ]
        )
    )
    current.write_text(
        json.dumps(
            [
                {
                    "generated_utc": "2026-06-21T18:00:00+00:00",
                    "lane_id": "external_lane",
                    "functional_complete": 15,
                    "functional_not_started": 85,
                    "functional_partial": 0,
                    "functional_failed": 1,
                    "esm2_units": 40,
                    "glm2_units": 10,
                    "tri_view_ready_units": 10,
                    "functional_status_basis": "live_per_mag_sentinels",
                }
            ]
        )
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "compare_atlas_lane_status.py",
            "--previous-json",
            str(previous),
            "--current-json",
            str(current),
            "--output-tsv",
            str(output_tsv),
            "--output-md",
            str(output_md),
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    rows = _read_tsv(output_tsv)
    assert rows[0]["lane_id"] == "external_lane"
    assert rows[0]["delta_functional_complete"] == "5"
    assert rows[0]["delta_esm2_units"] == "20"
    assert rows[0]["delta_glm2_units"] == "5"
    assert rows[0]["delta_tri_view_ready_units"] == "5"
    assert rows[0]["delta_functional_failed"] == "1"
    text = output_md.read_text()
    assert "Functional complete delta across lanes: **+5**." in text
    assert "Tri-view ready delta across lanes: **+5**." in text
    assert "| external_lane | existing | +5 | +20 | +5 | +5 | -5 | +1 |" in text
    assert "does not assign sample methane-risk tiers" in text


def test_build_atlas_lane_completion_checklist_reports_remaining_actions(tmp_path: Path) -> None:
    module = _load_script(
        "scripts/reports/build_atlas_lane_completion_checklist.py",
        "build_atlas_lane_completion_checklist",
    )
    status = tmp_path / "status.json"
    output_json = tmp_path / "checklist.json"
    output_md = tmp_path / "checklist.md"
    status.write_text(
        json.dumps(
            [
                {
                    "lane_id": "external_lane",
                    "lane_role": "external_mangrove",
                    "functional_include_rows": 10,
                    "functional_complete": 7,
                    "functional_not_started": 2,
                    "functional_partial": 1,
                    "functional_failed": 0,
                    "esm2_units": 10,
                    "glm2_units": 6,
                    "tri_view_ready_units": 6,
                    "consolidation_ready": "false",
                    "warehouse_current": "false",
                    "consolidation_command": "scripts/consolidate_functional_mag_cohort.py --lane external_lane",
                }
            ]
        )
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_atlas_lane_completion_checklist.py",
            "--status-json",
            str(status),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    rows = json.loads(output_json.read_text())
    assert rows[0]["lane_id"] == "external_lane"
    assert rows[0]["functional_remaining"] == 3
    assert rows[0]["functional_pending_or_partial"] == 3
    assert rows[0]["glm2_remaining"] == 4
    assert rows[0]["tri_view_remaining"] == 4
    assert rows[0]["consolidation_ready"] is False
    assert rows[0]["expanded_atlas_report_ready"] is False
    assert "Wait for or relaunch 3 pending/partial functional rows." in rows[0]["next_actions"]
    assert "Complete gLM2 for 4 selected rows." in rows[0]["next_actions"]
    text = output_md.read_text()
    assert "| external_lane | 3 | 0 | 4 | 4 | 0 | blocked | blocked |" in text
    assert "does not assign final sample methane-risk tiers" in text
