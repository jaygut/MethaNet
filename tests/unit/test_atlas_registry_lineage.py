"""Small end-to-end fixture for high-consequence atlas manifest joins."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/atlas_registry_minimal"
VALIDATOR = REPO_ROOT / "scripts/reports/validate_atlas_lane_registry.py"


def run_validator(fixture_root: Path, output_path: Path) -> tuple[int, dict]:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--repo-root",
            str(fixture_root),
            "--lane-registry",
            "configs/methanet_atlas_lanes.tsv",
            "--output-json",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, json.loads(output_path.read_text())


def rewrite_tsv(path: Path, update: Callable[[list[dict[str, str]]], None]) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    update(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_minimal_atlas_registry_fixture_preserves_assembly_context(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE_ROOT, fixture)
    code, report = run_validator(fixture, tmp_path / "report.json")
    assert code == 0
    assert report["valid"] is True
    assert report["row_count"] == 1
    reconciliation = report["manifest_reconciliation"][0]
    assert reconciliation["population_counts"] == {
        "source_manifest": 3,
        "functional_manifest": 2,
        "functional_included": 2,
        "source_mag_level_included": 2,
    }
    assert reconciliation["linkage"] == {
        "checked": True,
        "orphan_ids": 0,
        "field_mismatches": {},
    }


def test_manifest_lineage_rejects_orphans_and_identity_drift(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE_ROOT, fixture)

    def corrupt(rows: list[dict[str, str]]) -> None:
        rows[0]["mag_id"] = "wrong_mag"
        rows[0]["mbag_mag_level_include"] = "false"
        rows[1]["proteome_id"] = "orphan"

    rewrite_tsv(fixture / "functional.tsv", corrupt)
    code, report = run_validator(fixture, tmp_path / "report.json")
    assert code == 1
    assert any("functional proteome_id values absent" in error for error in report["errors"])
    assert any("source/functional mag_id mismatches" in error for error in report["errors"])
    assert any("source/functional mbag_mag_level_include mismatches" in error for error in report["errors"])


def test_manifest_lineage_rejects_unreconciled_denominator(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE_ROOT, fixture)
    registry = fixture / "configs/methanet_atlas_lanes.tsv"
    rewrite_tsv(registry, lambda rows: rows[0].__setitem__("denominator_units", "4"))
    code, report = run_validator(fixture, tmp_path / "report.json")
    assert code == 1
    assert any("does not match any registered manifest population" in error for error in report["errors"])
