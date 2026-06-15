#!/usr/bin/env python3
"""Summarize live MSM China 2025 functional annotation status."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MANIFEST = Path(
    "results/functional_metagenomics/msm_china_2025_20260615/manifests/"
    "msm_china_2025_functional_mag_manifest.tsv"
)
DEFAULT_RESULT_ROOT = Path("results/functional_metagenomics/msm_china_2025_20260615")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def latest_run_dir(per_mag_dir: Path) -> Path | None:
    if not per_mag_dir.is_dir():
        return None
    run_dirs = [path for path in per_mag_dir.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs[0] if run_dirs else None


def run_status(run_dir: Path | None) -> str:
    if run_dir is None:
        return "not_started"
    if (run_dir / "COMPLETE").is_file():
        return "complete"
    if (run_dir / "FAILED").is_file():
        return "failed"
    if (run_dir / "status.tsv").is_file():
        return "partial"
    return "attempt_created"


def last_step(run_dir: Path | None) -> tuple[str, str]:
    if run_dir is None:
        return "", ""
    status_path = run_dir / "status.tsv"
    if not status_path.is_file() or status_path.stat().st_size == 0:
        return "", ""
    try:
        rows = read_rows(status_path)
    except csv.Error:
        return "", ""
    if not rows:
        return "", ""
    row = rows[-1]
    return row.get("step", ""), row.get("status", "")


def count_curated_tables(run_dir: Path | None) -> int:
    if run_dir is None:
        return 0
    manifest = run_dir / "curated/parquet_manifest.tsv"
    if not manifest.is_file() or manifest.stat().st_size == 0:
        return 0
    try:
        return len(read_rows(manifest))
    except csv.Error:
        return 0


def command_output(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return f"{cmd[0]} unavailable: {exc}"
    return completed.stdout.rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--jobs", default="8804,8810,8813")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    result_root = args.result_root if args.result_root.is_absolute() else repo_root / args.result_root
    outdir = args.outdir or result_root / "status"
    outdir = outdir if outdir.is_absolute() else repo_root / outdir
    per_mag_root = result_root / "per_mag"

    rows = read_rows(manifest)
    detail: list[dict[str, object]] = []
    for row in rows:
        run_dir = latest_run_dir(per_mag_root / row["proteome_id"])
        status = run_status(run_dir)
        step, step_status = last_step(run_dir)
        detail.append({
            "proteome_id": row["proteome_id"],
            "mag_id": row["mag_id"],
            "source_group": row.get("source_group", ""),
            "domain": row.get("domain", ""),
            "protein_count": row.get("protein_count", ""),
            "run_status": status,
            "last_step": step,
            "last_step_status": step_status,
            "curated_parquet_table_count": count_curated_tables(run_dir),
            "run_dir": str(run_dir.relative_to(repo_root)) if run_dir else "",
        })

    status_counts = Counter(str(row["run_status"]) for row in detail)
    step_counts = Counter(
        f"{row['last_step']}:{row['last_step_status']}"
        for row in detail
        if row["last_step"]
    )
    summary = [
        {"metric": "snapshot_utc", "value": datetime.now(timezone.utc).isoformat(), "interpretation": "status generation time"},
        {"metric": "manifest_rows", "value": len(rows), "interpretation": "expected MAG/proteome functional units"},
        {"metric": "complete", "value": status_counts.get("complete", 0), "interpretation": "COMPLETE sentinels"},
        {"metric": "failed", "value": status_counts.get("failed", 0), "interpretation": "FAILED sentinels"},
        {"metric": "partial", "value": status_counts.get("partial", 0), "interpretation": "status.tsv exists but no terminal sentinel"},
        {"metric": "attempt_created", "value": status_counts.get("attempt_created", 0), "interpretation": "run folder exists but no terminal/status row"},
        {"metric": "not_started", "value": status_counts.get("not_started", 0), "interpretation": "no run folder yet"},
        {"metric": "curated_manifests_present", "value": sum(int(row["curated_parquet_table_count"]) > 0 for row in detail), "interpretation": "runs with curated parquet manifest"},
    ]

    for key, count in sorted(step_counts.items()):
        summary.append({"metric": "last_step_status", "value": key, "interpretation": count})

    jobs = [job.strip() for job in args.jobs.split(",") if job.strip()]
    scheduler = {}
    if jobs:
        scheduler["squeue"] = command_output([
            "squeue",
            "-j",
            ",".join(jobs),
            "-o",
            "%.18i %.10P %.24j %.2t %.10M %.10l %.5D %.5C %.10m %.20R",
        ])
        scheduler["sacct"] = command_output([
            "sacct",
            "-j",
            ",".join(jobs),
            "--format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS",
            "--parsable2",
        ])

    write_tsv(outdir / "msm_china_2025_functional_status_summary.tsv", summary)
    write_tsv(outdir / "msm_china_2025_functional_status_detail.tsv", detail)
    (outdir / "msm_china_2025_scheduler_status.json").write_text(
        json.dumps(scheduler, indent=2, sort_keys=True) + "\n"
    )

    print(f"wrote\t{outdir}")
    for row in summary[:8]:
        print(f"{row['metric']}\t{row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
