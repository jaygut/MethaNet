#!/usr/bin/env python3
"""Create compact manifests for a MethaNet functional MAG run.

The script is intentionally conservative: by default it only writes curated
metadata and does not delete or compress source outputs. Use --prune-success or
--compress-logs only after the extraction tables have been validated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


RETAIN_SUCCESS_RAW = [
    "summary.tsv",
    "status.tsv",
    "timings.tsv",
    "input_stats.tsv",
    "submission.tsv",
    "checkm2/quality_report.tsv",
    "gunc/GUNC.progenomes_3.maxCSS_level.tsv",
    "gtdbtk/gtdbtk.json",
    "kofam/*.kofam.detail.tsv",
    "mcycdb/*.diamond.tsv",
    "scycdb/*.diamond.tsv",
    "dbcan/overview.txt",
    "bakta/*.tsv",
    "bakta/*.json",
    "bakta/*.gff3",
    "metabolic/METABOLIC_result.xlsx",
]

DELETE_AFTER_EXTRACTION = [
    "dbcan_db_compat",
    "staged_fasta",
    "checkm2/protein_files",
    "gunc/gene_calls",
    "metabolic/intermediate_files",
    "metabolic/Each_HMM_Amino_Acid_Sequence",
    "tmp",
]

DIAMOND_COLUMNS = [
    "gene_id",
    "subject_id",
    "pident",
    "alignment_length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
    "query_coverage",
    "subject_coverage",
]

KOFAM_COLUMNS = [
    "threshold_marker",
    "gene_id",
    "ko_id",
    "threshold",
    "score",
    "evalue",
    "ko_definition",
]


def read_kv_tsv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def count_rows(path: Path, header: bool = True) -> int | None:
    if not path.exists() or path.is_dir():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as handle:
        n = sum(1 for line in handle if line.strip())
    return max(0, n - 1) if header else n


def coerce_num(value: Any) -> Any:
    if value in (None, "", "N/A"):
        return None if value in (None, "") else value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass"}:
        return True
    if text in {"false", "0", "no", "fail"}:
        return False
    return None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def file_size(path: Path) -> int | None:
    if path.exists() and path.is_file():
        return path.stat().st_size
    return None


def dir_size(path: Path) -> int | None:
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
        return int(completed.stdout.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    if path.is_file():
        stat = path.stat()
        return stat.st_blocks * 512 if hasattr(stat, "st_blocks") else stat.st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            total += stat.st_blocks * 512 if hasattr(stat, "st_blocks") else stat.st_size
    return total


def parse_taxonomy(classification: str | None) -> dict[str, str | None]:
    ranks: dict[str, str | None] = {
        "domain": None,
        "phylum": None,
        "class": None,
        "order": None,
        "family": None,
        "genus": None,
        "species": None,
    }
    if not classification:
        return ranks
    prefixes = {
        "d__": "domain",
        "p__": "phylum",
        "c__": "class",
        "o__": "order",
        "f__": "family",
        "g__": "genus",
        "s__": "species",
    }
    for token in classification.split(";"):
        token = token.strip()
        for prefix, rank in prefixes.items():
            if token.startswith(prefix):
                ranks[rank] = token
                break
    return ranks


def first_glob(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def build_outputs(run_dir: Path, repo_root: Path, mag_id: str) -> dict[str, dict[str, Any]]:
    paths: dict[str, tuple[Path, bool, str | None, str]] = {
        "summary": (run_dir / "summary.tsv", False, None, "manifest_keep"),
        "timings": (run_dir / "timings.tsv", True, "fact_tool_timing", "manifest_keep"),
        "input_stats": (run_dir / "input_stats.tsv", False, "fact_input_stats", "manifest_keep"),
        "checkm2_quality": (run_dir / "checkm2/quality_report.tsv", True, "fact_qc_checkm2", "raw_selected_keep"),
        "gunc_maxcss": (run_dir / "gunc/GUNC.progenomes_3.maxCSS_level.tsv", True, "fact_qc_gunc", "raw_selected_keep"),
        "kofam_detail": (run_dir / f"kofam/{mag_id}.kofam.detail.tsv", False, "fact_kofam_hits", "raw_selected_keep"),
        "mcycdb_diamond": (run_dir / f"mcycdb/{mag_id}.diamond.tsv", False, "fact_mcycdb_hits", "raw_selected_keep"),
        "scycdb_diamond": (run_dir / f"scycdb/{mag_id}.diamond.tsv", False, "fact_scycdb_hits", "raw_selected_keep"),
        "dbcan_overview": (run_dir / "dbcan/overview.txt", True, "fact_dbcan_hits", "raw_selected_keep"),
        "bakta_tsv": (run_dir / f"bakta/{mag_id}.tsv", True, "fact_bakta_features", "raw_selected_keep"),
        "bakta_json": (run_dir / f"bakta/{mag_id}.json", True, None, "raw_selected_keep_compress"),
        "metabolic_workbook": (run_dir / "metabolic/METABOLIC_result.xlsx", True, "fact_metabolic_tables", "raw_selected_keep"),
    }
    outputs: dict[str, dict[str, Any]] = {}
    for name, (path, header, table, retention) in paths.items():
        outputs[name] = {
            "path": rel(path, repo_root),
            "rows": count_rows(path, header=header),
            "bytes": file_size(path),
            "warehouse_table": table,
            "retention_class": retention,
        }
    dbcan_compat = run_dir / "dbcan_db_compat"
    outputs["dbcan_compat_indexes"] = {
        "path": rel(dbcan_compat, repo_root),
        "rows": None,
        "bytes": dir_size(dbcan_compat),
        "warehouse_table": None,
        "retention_class": "delete_after_extraction_or_centralize",
    }
    return outputs


def collect_file_manifest(run_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            rows.append({
                "path": rel(path, repo_root),
                "bytes": path.stat().st_size,
                "suffix": path.suffix,
            })
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def add_identity_columns(df: Any, record: dict[str, Any], table_name: str) -> Any:
    df.insert(0, "table_name", table_name)
    df.insert(0, "mag_id", record["mag_id"])
    df.insert(0, "proteome_id", record["proteome_id"])
    df.insert(0, "run_id", record["run_id"])
    df.insert(0, "cohort_run_id", record["cohort_run_id"])
    return df


def write_parquet_shards(run_dir: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for --write-parquet") from exc
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for --write-parquet") from exc

    mag_id = record["mag_id"]
    out_dir = run_dir / "curated" / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []

    def read_bakta_tsv(path: Path) -> Any:
        header: list[str] | None = None
        rows: list[list[str]] = []
        with path.open() as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("#Sequence Id"):
                    header = line.lstrip("#").split("\t")
                    continue
                if line.startswith("#"):
                    continue
                rows.append(line.split("\t"))
        if header is None:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=header)

    def write_df(name: str, df: Any) -> None:
        if df is None or df.empty:
            return
        df = add_identity_columns(df, record, name)
        target = out_dir / f"{name}.parquet"
        df.to_parquet(target, index=False, compression="zstd")
        written.append({
            "table": name,
            "path": str(target),
            "rows": int(len(df)),
            "bytes": target.stat().st_size,
        })

    summary_path = run_dir / "summary.tsv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, sep="\t", names=["metric", "value"], header=0)
        write_df("run_summary_metrics", summary)

    input_stats_path = run_dir / "input_stats.tsv"
    if input_stats_path.exists():
        input_stats = pd.read_csv(input_stats_path, sep="\t")
        write_df("fact_input_stats", input_stats)

    timings_path = run_dir / "timings.tsv"
    if timings_path.exists():
        timings = pd.read_csv(timings_path, sep="\t")
        write_df("fact_tool_timing", timings)

    checkm2_path = run_dir / "checkm2/quality_report.tsv"
    if checkm2_path.exists():
        write_df("fact_qc_checkm2", pd.read_csv(checkm2_path, sep="\t"))

    gunc_path = run_dir / "gunc/GUNC.progenomes_3.maxCSS_level.tsv"
    if gunc_path.exists():
        write_df("fact_qc_gunc", pd.read_csv(gunc_path, sep="\t"))

    kofam_path = run_dir / f"kofam/{mag_id}.kofam.detail.tsv"
    if kofam_path.exists():
        kofam = pd.read_csv(kofam_path, sep="\t", names=KOFAM_COLUMNS, skiprows=2)
        kofam["accepted_hit"] = kofam["threshold_marker"].astype(str).eq("*")
        write_df("fact_kofam_hits", kofam)

    for tool in ("mcycdb", "scycdb"):
        diamond_path = run_dir / f"{tool}/{mag_id}.diamond.tsv"
        if diamond_path.exists():
            diamond = pd.read_csv(diamond_path, sep="\t", names=DIAMOND_COLUMNS)
            diamond = diamond.drop_duplicates()
            diamond["hit_rank_bitscore"] = (
                diamond.sort_values(["gene_id", "bitscore", "evalue"], ascending=[True, False, True])
                .groupby("gene_id")
                .cumcount()
                + 1
            )
            write_df(f"fact_{tool}_hits", diamond)

    dbcan_path = run_dir / "dbcan/overview.txt"
    if dbcan_path.exists():
        write_df("fact_dbcan_hits", pd.read_csv(dbcan_path, sep="\t"))

    bakta_path = run_dir / f"bakta/{mag_id}.tsv"
    if bakta_path.exists():
        write_df("fact_bakta_features", read_bakta_tsv(bakta_path))

    metabolic_xlsx = run_dir / "metabolic/METABOLIC_result.xlsx"
    if metabolic_xlsx.exists():
        sheets = pd.read_excel(metabolic_xlsx, sheet_name=None)
        for sheet_name, sheet_df in sheets.items():
            safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(sheet_name)).strip("_")
            write_df(f"fact_metabolic_{safe_name}", sheet_df)

    return written


def gzip_log(path: Path) -> Path:
    target = path.with_suffix(path.suffix + ".gz")
    if target.exists():
        return target
    with path.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()
    return target


def prune_success(run_dir: Path, dry_run: bool) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for rel_path in DELETE_AFTER_EXTRACTION:
        path = run_dir / rel_path
        if path.exists():
            actions.append({"action": "delete", "path": str(path)})
            if not dry_run:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--proteome-id", required=True)
    parser.add_argument("--mag-id", required=True)
    parser.add_argument("--cohort-run-id", default="one_mag_smoke_20260611")
    parser.add_argument("--job-id")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--mag-fasta", type=Path)
    parser.add_argument("--proteome-faa", type=Path)
    parser.add_argument("--compress-logs", action="store_true")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--prune-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    repo_root = args.repo_root.resolve()
    curated = run_dir / "curated"
    curated.mkdir(exist_ok=True)

    summary = read_kv_tsv(run_dir / "summary.tsv")
    input_stats = read_kv_tsv(run_dir / "input_stats.tsv")
    checkm2_rows = read_tsv_rows(run_dir / "checkm2/quality_report.tsv")
    gunc_rows = read_tsv_rows(run_dir / "gunc/GUNC.progenomes_3.maxCSS_level.tsv")
    checkm2 = checkm2_rows[0] if checkm2_rows else {}
    gunc = gunc_rows[0] if gunc_rows else {}

    gtdb: dict[str, Any] = {}
    gtdb_summary = first_glob(run_dir, "gtdbtk/**/*.summary.tsv")
    if gtdb_summary:
        rows = read_tsv_rows(gtdb_summary)
        gtdb = rows[0] if rows else {}
    gtdb_json = run_dir / "gtdbtk/gtdbtk.json"
    if not gtdb and gtdb_json.exists():
        try:
            data = json.loads(gtdb_json.read_text())
            gtdb = data.get(args.mag_id, data) if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            gtdb = {}

    classification = (
        gtdb.get("classification")
        or gtdb.get("taxonomy")
        or summary.get("gtdbtk_classification")
    )
    ranks = parse_taxonomy(classification)
    completeness = coerce_num(checkm2.get("Completeness") or checkm2.get("completeness"))
    contamination = coerce_num(checkm2.get("Contamination") or checkm2.get("contamination"))
    caveats: list[str] = []
    if isinstance(completeness, (int, float)) and completeness < 70:
        caveats.append("MAG completeness is below 70%; absence calls require coverage caveats.")
    if isinstance(contamination, (int, float)) and contamination > 5:
        caveats.append("MAG contamination exceeds 5%; inspect pathway evidence for chimerism.")

    outputs = build_outputs(run_dir, repo_root, args.mag_id)
    work_fasta = run_dir / f"input/{args.mag_id}.fasta"
    elapsed = coerce_num(summary.get("total_elapsed_seconds"))

    record = {
        "schema_version": "0.1.0",
        "cohort_run_id": args.cohort_run_id,
        "run_id": run_dir.name,
        "proteome_id": args.proteome_id,
        "mag_id": args.mag_id,
        "status": "complete" if (run_dir / "COMPLETE").exists() else "partial",
        "inputs": {
            "mag_fasta": str(args.mag_fasta.resolve()) if args.mag_fasta else None,
            "proteome_faa": str(args.proteome_faa.resolve()) if args.proteome_faa else None,
            "work_fasta": str(work_fasta.resolve()) if work_fasta.exists() else None,
            "input_sha256": sha256_file(args.mag_fasta) if args.mag_fasta else None,
        },
        "job": {
            "scheduler": "slurm",
            "job_id": args.job_id,
            "cpus": args.cpus,
            "elapsed_seconds": elapsed,
            "exit_code": "0:0" if (run_dir / "COMPLETE").exists() else None,
        },
        "summary_metrics": {
            "input_contigs": coerce_num(input_stats.get("input_contigs") or summary.get("input_contigs")),
            "input_total_bp": coerce_num(input_stats.get("input_total_bp") or summary.get("input_total_bp")),
            "input_n50_bp": coerce_num(input_stats.get("input_n50_bp") or summary.get("input_n50_bp")),
            "prodigal_proteins": coerce_num(summary.get("prodigal_proteins")),
            "kofam_rows": outputs["kofam_detail"]["rows"],
            "mcycdb_hits": outputs["mcycdb_diamond"]["rows"],
            "scycdb_hits": outputs["scycdb_diamond"]["rows"],
            "dbcan_overview_rows": outputs["dbcan_overview"]["rows"],
            "bakta_feature_rows": outputs["bakta_tsv"]["rows"],
        },
        "qc": {
            "completeness": completeness,
            "contamination": contamination,
            "gunc_pass": parse_bool(gunc.get("pass.GUNC") or gunc.get("pass_gunc")),
            "gunc_css": coerce_num(gunc.get("clade_separation_score") or gunc.get("CSS")),
            "gunc_rrs": coerce_num(gunc.get("reference_representation_score") or gunc.get("RRS")),
            "qc_tier": "medium_low_completeness" if isinstance(completeness, (int, float)) and completeness < 70 else "pass_review",
            "caveats": caveats,
        },
        "taxonomy": {
            "gtdb_release": "R232",
            "classification": classification,
            **ranks,
            "ani": coerce_num(gtdb.get("closest_placement_ani") or gtdb.get("ani")),
            "af": coerce_num(gtdb.get("closest_placement_af") or gtdb.get("af")),
        },
        "outputs": outputs,
        "retention": {
            "retain_success_raw": RETAIN_SUCCESS_RAW,
            "delete_after_extraction": DELETE_AFTER_EXTRACTION + ["empty logs/*.out and logs/*.err"],
            "logs_to_keep": [
                "logs/slurm-*.out.gz",
                "logs/slurm-*.err.gz",
                "logs/driver.out.gz",
                "logs/driver.err.gz",
                "logs/*.time.txt",
                "logs/by_tool/*.stderr.gz only when non-empty/warning/failed",
            ],
        },
    }

    if args.compress_logs:
        for log in (run_dir / "logs").glob("*.out"):
            if log.stat().st_size > 0:
                gzip_log(log)
        for log in (run_dir / "logs").glob("*.err"):
            if log.stat().st_size > 0:
                gzip_log(log)

    parquet_shards: list[dict[str, Any]] = []
    if args.write_parquet:
        parquet_shards = write_parquet_shards(run_dir, record)
        (curated / "parquet_manifest.tsv").write_text("")
        write_tsv(curated / "parquet_manifest.tsv", parquet_shards)
        record["outputs"]["curated_parquet_manifest"] = {
            "path": rel(curated / "parquet_manifest.tsv", repo_root),
            "rows": len(parquet_shards),
            "bytes": file_size(curated / "parquet_manifest.tsv"),
            "warehouse_table": None,
            "retention_class": "curated_keep",
        }

    prune_actions: list[dict[str, str]] = []
    if args.prune_success and record["status"] == "complete":
        prune_actions = prune_success(run_dir, dry_run=args.dry_run)
    (curated / "prune_plan.json").write_text(json.dumps(prune_actions, indent=2) + "\n")

    (curated / "run_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    write_tsv(curated / "file_manifest.tsv", collect_file_manifest(run_dir, repo_root))
    print(curated / "run_record.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
