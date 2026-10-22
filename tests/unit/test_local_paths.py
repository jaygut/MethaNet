"""Machine-local roots resolve from the environment or .methanet.env, never from tracked files."""
from __future__ import annotations

import csv
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_paths

COHORT_MANIFEST = (
    REPO_ROOT
    / "results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse/cohort_table_manifest.tsv"
)


def test_expand_local_path_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("METHANET_SCRATCH", str(tmp_path / "scratch"))
    expand = local_paths.expand_local_path
    assert expand("$METHANET_ROOT/results/a.tsv", tmp_path) == tmp_path / "results/a.tsv"
    assert expand("${METHANET_ROOT}/results/a.tsv", tmp_path) == tmp_path / "results/a.tsv"
    assert expand("$METHANET_SCRATCH/methanet_db", tmp_path) == tmp_path / "scratch/methanet_db"
    assert expand("$HOME/.cache", tmp_path) == Path.home() / ".cache"
    assert expand("results/a.tsv", tmp_path) == tmp_path / "results/a.tsv"
    assert expand("/srv/data/a.tsv", tmp_path) == Path("/srv/data/a.tsv")


def test_portable_path_round_trip(tmp_path, monkeypatch):
    repo = (tmp_path / "repo").resolve()
    monkeypatch.setenv("METHANET_SCRATCH", str(tmp_path / "scratch"))
    target = repo / "results" / "x.parquet"
    assert local_paths.portable_path(target, repo) == "$METHANET_ROOT/results/x.parquet"
    assert local_paths.portable_path(repo, repo) == "$METHANET_ROOT"
    assert local_paths.portable_path(tmp_path / "scratch" / "db", repo) == "$METHANET_SCRATCH/db"
    assert local_paths.expand_local_path(local_paths.portable_path(target, repo), repo) == target


def test_local_env_file_is_read_after_the_environment(tmp_path, monkeypatch):
    (tmp_path / ".methanet.env").write_text(
        "# local settings\nexport HPC_HOST='login.example'\nMETHANET_ROOT=/data/MethaNet\n"
    )
    monkeypatch.delenv("HPC_HOST", raising=False)
    assert local_paths.local_setting("HPC_HOST", tmp_path) == "login.example"
    assert local_paths.local_setting("METHANET_ROOT", tmp_path) == "/data/MethaNet"
    monkeypatch.setenv("HPC_HOST", "override.example")
    assert local_paths.local_setting("HPC_HOST", tmp_path) == "override.example"
    assert local_paths.local_setting("HPC_USER", tmp_path, default="fallback") == "fallback"


def test_local_env_file_is_ignored():
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", ".methanet.env"], check=False
    )
    assert result.returncode == 0
    example = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", ".methanet.env.example"], check=False
    )
    assert example.returncode == 1


def test_fetch_script_requires_an_hpc_host(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(REPO_ROOT / "scripts/fetch_apolo_blue_catalyst_artifacts.sh", scripts)
    env = {k: v for k, v in os.environ.items() if k not in {"HPC_HOST", "APOLO_HOST"}}
    env["APOLO_EXPORT_DIR"] = "/export"
    result = subprocess.run(
        ["bash", str(scripts / "fetch_apolo_blue_catalyst_artifacts.sh"), "--stamp", "20260101_000000"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "HPC_HOST" in result.stderr
    assert not (tmp_path / "results").exists()


def test_tracked_data_files_carry_no_account_home_paths():
    files = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "*.tsv", "*.json", "*.csv", "*.yaml", "*.yml", "*.log"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    account_home = re.compile(r"/home/[^/\s|\"]+/")
    offenders = [
        name for name in files
        if (REPO_ROOT / name).is_file()
        and account_home.search((REPO_ROOT / name).read_text(errors="replace"))
    ]
    assert offenders == []


def test_cohort_manifest_paths_resolve_inside_the_repository():
    with COHORT_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows
    for row in rows:
        assert row["path"].startswith("$METHANET_ROOT/")
        assert local_paths.expand_local_path(row["path"], REPO_ROOT).is_relative_to(REPO_ROOT)


def test_lane_validator_resolves_the_root_token(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "validate_atlas_lane_registry_local_paths",
        REPO_ROOT / "scripts/reports/validate_atlas_lane_registry.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.resolve(tmp_path, "$METHANET_ROOT/configs/x.tsv") == tmp_path / "configs/x.tsv"
    assert module.resolve(tmp_path, "configs/x.tsv") == tmp_path / "configs/x.tsv"
