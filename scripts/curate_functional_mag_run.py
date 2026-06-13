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
    "eggnog/*.emapper.annotations",
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
        "eggnog_annotations": (run_dir / f"eggnog/{mag_id}.emapper.annotations", False, "fact_eggnog_annotations", "raw_selected_keep"),
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
    df.insert(0, "source_tool", source_tool_for_table(table_name))
    df.insert(0, "table_name", table_name)
    df.insert(0, "mag_id", record["mag_id"])
    df.insert(0, "proteome_id", record["proteome_id"])
    df.insert(0, "run_id", record["run_id"])
    df.insert(0, "cohort_run_id", record["cohort_run_id"])
    return df


def source_tool_for_table(table_name: str) -> str:
    if table_name.startswith("fact_cazy_") or table_name.startswith("fact_merops_"):
        return "METABOLIC-G"
    if table_name.startswith("fact_metabolic_"):
        return "METABOLIC-G"
    if table_name.startswith("fact_kofam_"):
        return "KOfam"
    if table_name.startswith("fact_mcycdb_"):
        return "MCycDB"
    if table_name.startswith("fact_scycdb_"):
        return "SCycDB"
    if table_name.startswith("fact_dbcan_"):
        return "dbCAN"
    if table_name.startswith("fact_eggnog_"):
        return "eggNOG-mapper"
    if table_name.startswith("fact_bakta_") or table_name == "dim_gene":
        return "Bakta"
    if table_name.startswith("fact_qc_checkm2"):
        return "CheckM2"
    if table_name.startswith("fact_qc_gunc"):
        return "GUNC"
    if table_name.startswith("fact_taxonomy_gtdbtk"):
        return "GTDB-Tk"
    if table_name.startswith("fact_tool_timing"):
        return "MethaNet"
    if table_name.startswith("fact_input_stats") or table_name.startswith("run_summary"):
        return "MethaNet"
    return "MethaNet"


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

    def read_eggnog_annotations(path: Path) -> Any:
        header: list[str] | None = None
        rows: list[list[str]] = []
        with path.open(errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("##"):
                    continue
                if line.startswith("#"):
                    candidate = line.lstrip("#").split("\t")
                    if candidate and candidate[0] in {"query", "query_name"}:
                        header = candidate
                    continue
                if header is None:
                    continue
                parts = line.split("\t")
                if len(parts) < len(header):
                    parts.extend([""] * (len(header) - len(parts)))
                rows.append(parts[:len(header)])
        if header is None:
            return pd.DataFrame()
        out = pd.DataFrame(rows, columns=header)
        rename = {
            "query": "gene_id",
            "seed_ortholog": "seed_ortholog",
            "evalue": "evalue",
            "score": "score",
            "eggNOG_OGs": "eggnog_ogs",
            "max_annot_lvl": "max_annotation_level",
            "COG_category": "cog_category",
            "Description": "description",
            "Preferred_name": "preferred_name",
            "GOs": "go_terms",
            "EC": "ec_numbers",
            "KEGG_ko": "kegg_ko",
            "KEGG_Pathway": "kegg_pathway",
            "KEGG_Module": "kegg_module",
            "KEGG_Reaction": "kegg_reaction",
            "KEGG_rclass": "kegg_rclass",
            "BRITE": "brite",
            "KEGG_TC": "kegg_tc",
            "CAZy": "cazy",
            "BiGG_Reaction": "bigg_reaction",
            "PFAMs": "pfams",
        }
        out = out.rename(columns={old: new for old, new in rename.items() if old in out.columns})
        for col in ("evalue", "score"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

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

    def prefixed_col(df: Any, suffix: str) -> str | None:
        matches = [col for col in df.columns if str(col).endswith(suffix)]
        return matches[0] if matches else None

    def clean_optional_text(value: Any) -> Any:
        if value is None:
            return None
        text = str(value)
        if text in {"", "nan", "None", "NA"}:
            return None
        return text

    def split_gene_list(value: Any) -> Any:
        text = clean_optional_text(value)
        if text is None:
            return None
        return ";".join(part for part in (item.strip() for item in text.split(";")) if part and part != "None")

    def parse_hit_count(value: Any) -> Any:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value

    def normalize_metabolic_sheet(sheet_name: str, df: Any) -> tuple[str, Any] | None:
        safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(sheet_name)).strip("_")
        if df is None or df.empty:
            return None
        df = df.copy()
        df = df.dropna(how="all")
        if df.empty:
            return None

        if safe_name == "hmmhitnum":
            presence_col = prefixed_col(df, ".Hmm.presence")
            hit_count_col = prefixed_col(df, ".Hit.numbers")
            hits_col = prefixed_col(df, ".Hits")
            out = pd.DataFrame({
                "function_category": df.get("Category"),
                "function_name": df.get("Function"),
                "gene_abbreviation": df.get("Gene.abbreviation"),
                "gene_name": df.get("Gene.name"),
                "hmm_file": df.get("Hmm.file"),
                "ko_id": df.get("Corresponding.KO"),
                "reaction": df.get("Reaction"),
                "substrate": df.get("Substrate"),
                "product": df.get("Product"),
                "hmm_detecting_threshold": df.get("Hmm.detecting.threshold"),
                "presence": df[presence_col] if presence_col else None,
                "hit_count": df[hit_count_col].map(parse_hit_count) if hit_count_col else None,
                "hit_genes": df[hits_col].map(split_gene_list) if hits_col else None,
                "evidence_source": "METABOLIC_result.xlsx:HMMHitNum",
            })
            return "fact_metabolic_hmm_hits", out

        if safe_name == "functionhit":
            presence_col = prefixed_col(df, ".Function.presence")
            out = pd.DataFrame({
                "function_category": df.get("Category"),
                "function_name": df.get("Function"),
                "gene_abbreviation": df.get("Gene.abbreviation"),
                "presence": df[presence_col] if presence_col else None,
                "evidence_source": "METABOLIC_result.xlsx:FunctionHit",
            })
            return "fact_metabolic_function_presence", out

        if safe_name == "keggmodulehit":
            presence_col = prefixed_col(df, ".Module.presence")
            out = pd.DataFrame({
                "module_id": df.get("Module.ID"),
                "module_name": df.get("Module"),
                "module_category": df.get("Module.Category"),
                "presence": df[presence_col] if presence_col else None,
                "hit_count": None,
                "hit_genes": None,
                "evidence_source": "METABOLIC_result.xlsx:KEGGModuleHit",
            })
            return "fact_metabolic_module_presence", out

        if safe_name == "keggmodulestephit":
            presence_col = prefixed_col(df, ".Module.step.presence")
            out = pd.DataFrame({
                "module_step_id": df.get("Module.step"),
                "module_name": df.get("Module"),
                "ko_id": df.get("KO.id"),
                "module_category": df.get("Module.Category"),
                "presence": df[presence_col] if presence_col else None,
                "evidence_source": "METABOLIC_result.xlsx:KEGGModuleStepHit",
            })
            return "fact_metabolic_module_step_presence", out

        if safe_name == "dbcan2hit":
            hit_count_col = prefixed_col(df, ".Hit.numbers")
            hits_col = prefixed_col(df, ".Hits")
            out = pd.DataFrame({
                "cazy_family": df.get("CAZyme.ID"),
                "hit_count": df[hit_count_col].map(parse_hit_count) if hit_count_col else None,
                "hit_genes": df[hits_col].map(split_gene_list) if hits_col else None,
                "evidence_source": "METABOLIC_result.xlsx:dbCAN2Hit",
            })
            return "fact_cazy_hits", out

        if safe_name == "meropshit":
            hit_count_col = prefixed_col(df, ".Hit.numbers")
            hits_col = prefixed_col(df, ".Hits")
            out = pd.DataFrame({
                "merops_peptidase_id": df.get("MEROPS.peptidase.ID"),
                "hit_count": df[hit_count_col].map(parse_hit_count) if hit_count_col else None,
                "hit_genes": df[hits_col].map(split_gene_list) if hits_col else None,
                "evidence_source": "METABOLIC_result.xlsx:MEROPSHit",
            })
            return "fact_merops_hits", out

        return f"fact_metabolic_{safe_name}", df

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

    eggnog_path = run_dir / f"eggnog/{mag_id}.emapper.annotations"
    if eggnog_path.exists():
        write_df("fact_eggnog_annotations", read_eggnog_annotations(eggnog_path))

    bakta_path = run_dir / f"bakta/{mag_id}.tsv"
    if bakta_path.exists():
        write_df("fact_bakta_features", read_bakta_tsv(bakta_path))

    metabolic_xlsx = run_dir / "metabolic/METABOLIC_result.xlsx"
    if metabolic_xlsx.exists():
        sheets = pd.read_excel(metabolic_xlsx, sheet_name=None)
        for sheet_name, sheet_df in sheets.items():
            normalized = normalize_metabolic_sheet(sheet_name, sheet_df)
            if normalized:
                table_name, table_df = normalized
                write_df(table_name, table_df)

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
