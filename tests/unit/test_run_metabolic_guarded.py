from __future__ import annotations

import importlib.util
import subprocess
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/run_metabolic_guarded.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_metabolic_guarded", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_test_workbook(path: Path, worksheet_count: int = 6, mag_id: str = "m1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>{mag_id}.Hmm.presence</t></si></sst>',
        )
        for index in range(1, worksheet_count + 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", "<worksheet/>")


def test_run_uses_private_working_directory_and_accepts_complete_workbook(tmp_path, monkeypatch):
    module = load_module()
    output_dir = tmp_path / "run" / "metabolic"
    observed = {}

    def fake_run(command, cwd, check):
        observed["command"] = command
        observed["cwd"] = Path(cwd)
        write_test_workbook(output_dir / "METABOLIC_result.xlsx")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    workbook = module.run_metabolic(
        perl="perl",
        metabolic_script=tmp_path / "METABOLIC-G.pl",
        input_genomes=tmp_path / "staged_fasta",
        output_dir=output_dir,
        threads=16,
        expected_mag_id="m1",
    )

    assert workbook == output_dir / "METABOLIC_result.xlsx"
    assert observed["cwd"].parent == output_dir.parent / "tmp"
    assert observed["cwd"].name.startswith("metabolic_work_")
    assert observed["cwd"] != Path.cwd()


def test_zero_exit_without_workbook_fails_and_reports_preserved_intermediates(tmp_path, monkeypatch):
    module = load_module()
    output_dir = tmp_path / "run" / "metabolic"
    tables = output_dir / "METABOLIC_result_each_spreadsheet"
    tables.mkdir(parents=True)
    for index in range(1, 7):
        (tables / f"worksheet{index}.tsv").write_text("header\nvalue\n")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, cwd, check: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(module.OutputContractError, match="nonempty intermediate worksheet TSVs=6"):
        module.run_metabolic(
            perl="perl",
            metabolic_script=tmp_path / "METABOLIC-G.pl",
            input_genomes=tmp_path / "staged_fasta",
            output_dir=output_dir,
            threads=16,
            expected_mag_id="m1",
        )


def test_structurally_incomplete_workbook_is_rejected(tmp_path):
    module = load_module()
    workbook = tmp_path / "METABOLIC_result.xlsx"
    write_test_workbook(workbook, worksheet_count=5)
    with pytest.raises(module.OutputContractError, match="expected at least 6 worksheets"):
        module.validate_workbook(workbook, "m1")


def test_workbook_from_another_mag_is_rejected(tmp_path):
    module = load_module()
    workbook = tmp_path / "METABOLIC_result.xlsx"
    write_test_workbook(workbook, mag_id="other_mag")
    with pytest.raises(module.OutputContractError, match="workbook MAG identity mismatch"):
        module.validate_workbook(workbook, "expected_mag")


def test_r_syntactic_prefix_for_numeric_mag_id_is_accepted(tmp_path):
    module = load_module()
    workbook = tmp_path / "METABOLIC_result.xlsx"
    write_test_workbook(workbook, mag_id="X3300004775_9")
    module.validate_workbook(workbook, "3300004775_9")


def test_consolidator_quarantines_cross_mag_workbook(tmp_path):
    consolidator_path = REPO_ROOT / "scripts/consolidate_functional_mag_cohort.py"
    spec = importlib.util.spec_from_file_location("consolidator_integrity", consolidator_path)
    assert spec and spec.loader
    consolidator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consolidator)

    run_dir = tmp_path / "per_mag" / "proteome" / "run_20260101"
    (run_dir / "curated").mkdir(parents=True)
    (run_dir / "metabolic").mkdir()
    (run_dir / "COMPLETE").touch()
    (run_dir / "curated/run_record.json").write_text(
        '{"status":"complete","run_id":"run_20260101","mag_id":"expected_mag"}'
    )
    (run_dir / "curated/parquet_manifest.tsv").write_text("table\tpath\n")
    write_test_workbook(run_dir / "metabolic/METABOLIC_result.xlsx", mag_id="other_mag")

    attempts, selected = consolidator.discover_runs(tmp_path / "per_mag", "cohort", tmp_path)
    assert attempts[0]["run_status"] == "failed_integrity"
    assert attempts[0]["integrity_status"] == "fail"
    assert selected == {}
