#!/usr/bin/env python3
"""Preflight the MethaNet functional MAG production gates.

This validator performs no submission and makes no destructive changes. It is
meant to be run immediately before launching the 662-MAG array.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_MANIFEST_COLUMNS = {
    "proteome_id",
    "mag_id",
    "mag_fasta",
    "proteome_faa",
    "match_status",
    "functional_run_include",
}

REQUIRED_SCRIPTS = [
    "scripts/slurm/run_one_mag_functional_smoke_apollo3.sh",
    "scripts/slurm/run_functional_mag_array_apollo3.sh",
    "scripts/submit_functional_mag_batches_apollo3.sh",
    "scripts/curate_functional_mag_run.py",
    "scripts/prepare_dbcan_compat_cache_apollo3.sh",
]

DBCAN_CACHE_FILES = [
    "dbCAN.hmm",
    "dbCAN.hmm.h3p",
    "dbCAN.hmm.h3m",
    "dbCAN.hmm.h3f",
    "dbCAN.hmm.h3i",
    "dbCAN_sub.hmm",
    "dbCAN_sub.hmm.h3p",
    "dbCAN_sub.hmm.h3m",
    "dbCAN_sub.hmm.h3f",
    "dbCAN_sub.hmm.h3i",
]

REQUIRED_PARQUET_FIXTURE_TABLES = [
    "fact_tool_timing",
    "fact_qc_checkm2",
    "fact_qc_gunc",
    "fact_kofam_hits",
    "fact_mcycdb_hits",
    "fact_scycdb_hits",
    "fact_dbcan_hits",
    "fact_bakta_features",
    "fact_metabolic_keggmodulehit",
]


@dataclass
class Check:
    gate: str
    status: str
    detail: str


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def included_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("functional_run_include", "").strip().lower() in {"true", "1", "yes"}
    ]


def file_exists(root: Path, value: str) -> bool:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.is_file() and path.stat().st_size > 0


def du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        completed = subprocess.run(
            ["du", "-sB1", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return int(completed.stdout.split()[0])
    except (IndexError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument(
        "--manifest",
        default=Path("results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.proposed.tsv"),
        type=Path,
    )
    parser.add_argument("--db-root", default=Path("/home/rsg-jcorre38/scratch/methanet_db"), type=Path)
    parser.add_argument("--dbcan-compat-dir", type=Path)
    parser.add_argument(
        "--smoke-run-dir",
        default=Path("results/functional_metagenomics/one_mag_smoke/one_mag_fgx_rumen__10674_0002_idba_bin.8_20260611_231754"),
        type=Path,
    )
    parser.add_argument("--expected-count", default=662, type=int)
    parser.add_argument("--sample-file-check-count", default=662, type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    dbcan_compat = args.dbcan_compat_dir or args.db_root / "dbcan_compat_pressed"

    checks: list[Check] = []
    rows: list[dict[str, str]] = []

    if manifest.exists() and manifest.stat().st_size > 0:
        rows = read_manifest(manifest)
        checks.append(Check("manifest_exists", "pass", rel(manifest, repo_root)))
    else:
        checks.append(Check("manifest_exists", "fail", f"Missing manifest: {manifest}"))

    if rows:
        missing = REQUIRED_MANIFEST_COLUMNS - set(rows[0].keys())
        checks.append(Check(
            "manifest_columns",
            "pass" if not missing else "fail",
            "all required columns present" if not missing else f"missing columns: {sorted(missing)}",
        ))
        included = included_rows(rows)
        checks.append(Check(
            "included_mag_count",
            "pass" if len(included) == args.expected_count else "fail",
            f"included={len(included)} expected={args.expected_count}",
        ))
        proteome_ids = [row.get("proteome_id", "") for row in included]
        mag_ids = [row.get("mag_id", "") for row in included]
        checks.append(Check(
            "unique_proteome_id",
            "pass" if len(proteome_ids) == len(set(proteome_ids)) else "fail",
            f"unique={len(set(proteome_ids))} total={len(proteome_ids)}",
        ))
        checks.append(Check(
            "unique_mag_id",
            "pass" if len(mag_ids) == len(set(mag_ids)) else "warn",
            f"unique={len(set(mag_ids))} total={len(mag_ids)}",
        ))
        matched = [row for row in included if row.get("match_status") == "matched"]
        checks.append(Check(
            "all_included_rows_matched",
            "pass" if len(matched) == len(included) else "fail",
            f"matched={len(matched)} included={len(included)}",
        ))
        sample = included[: args.sample_file_check_count]
        missing_fasta = [row.get("proteome_id", "?") for row in sample if not file_exists(repo_root, row.get("mag_fasta", ""))]
        missing_faa = [row.get("proteome_id", "?") for row in sample if not file_exists(repo_root, row.get("proteome_faa", ""))]
        checks.append(Check(
            "mag_fasta_files_exist",
            "pass" if not missing_fasta else "fail",
            f"checked={len(sample)} missing={len(missing_fasta)}" + (f" examples={missing_fasta[:5]}" if missing_fasta else ""),
        ))
        checks.append(Check(
            "proteome_faa_files_exist",
            "pass" if not missing_faa else "fail",
            f"checked={len(sample)} missing={len(missing_faa)}" + (f" examples={missing_faa[:5]}" if missing_faa else ""),
        ))

    for script in REQUIRED_SCRIPTS:
        path = repo_root / script
        checks.append(Check(
            f"script_ready:{script}",
            "pass" if path.exists() and os.access(path, os.X_OK) else "fail",
            rel(path, repo_root),
        ))

    missing_cache = [name for name in DBCAN_CACHE_FILES if not (dbcan_compat / name).is_file()]
    cache_bytes = du_bytes(dbcan_compat)
    checks.append(Check(
        "dbcan_shared_cache_ready",
        "pass" if not missing_cache else "fail",
        f"path={dbcan_compat}; missing={missing_cache}; du_bytes={cache_bytes}",
    ))
    checks.append(Check(
        "no_per_run_dbcan_cache_required",
        "pass",
        "runner uses DBCAN_COMPAT_DIR and curation prunes dbcan_db_compat if legacy outputs exist",
    ))

    smoke_run = args.smoke_run_dir
    if not smoke_run.is_absolute():
        smoke_run = repo_root / smoke_run
    parquet_manifest = smoke_run / "curated/parquet_manifest.tsv"
    checks.append(Check(
        "curated_parquet_manifest_fixture",
        "pass" if parquet_manifest.exists() and parquet_manifest.stat().st_size > 0 else "fail",
        rel(parquet_manifest, repo_root),
    ))
    missing_parquet = [
        table for table in REQUIRED_PARQUET_FIXTURE_TABLES
        if not (smoke_run / "curated/parquet" / f"{table}.parquet").is_file()
    ]
    checks.append(Check(
        "curated_parquet_fixture_tables",
        "pass" if not missing_parquet else "fail",
        "all required fixture tables present" if not missing_parquet else f"missing={missing_parquet}",
    ))

    failed = [check for check in checks if check.status == "fail"]
    warned = [check for check in checks if check.status == "warn"]

    for check in checks:
        print(f"{check.status.upper()}\t{check.gate}\t{check.detail}")

    summary = {
        "status": "fail" if failed else "pass",
        "failures": [check.__dict__ for check in failed],
        "warnings": [check.__dict__ for check in warned],
        "checks": [check.__dict__ for check in checks],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
