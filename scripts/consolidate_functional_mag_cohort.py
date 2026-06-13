#!/usr/bin/env python3
"""Consolidate MethaNet per-MAG functional outputs into cohort tables.

This script is intentionally read-only with respect to per-MAG run folders. It
selects the latest completed curated run per proteome, normalizes legacy
METABOLIC wide workbook shards, writes cohort-level Parquet tables, and records
all failed/partial attempts in a status table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTITY_COLUMNS = ["cohort_run_id", "run_id", "proteome_id", "mag_id", "source_tool"]

EXPECTED_COMPLETE_TABLES = {
    "run_summary_metrics",
    "fact_input_stats",
    "fact_tool_timing",
    "fact_qc_checkm2",
    "fact_qc_gunc",
    "fact_kofam_hits",
    "fact_mcycdb_hits",
    "fact_scycdb_hits",
    "fact_dbcan_hits",
    "fact_bakta_features",
}

METABOLIC_TABLE_ALIASES = {
    "fact_metabolic_hmmhitnum": "fact_metabolic_hmm_hits",
    "fact_metabolic_functionhit": "fact_metabolic_function_presence",
    "fact_metabolic_keggmodulehit": "fact_metabolic_module_presence",
    "fact_metabolic_keggmodulestephit": "fact_metabolic_module_step_presence",
    "fact_metabolic_dbcan2hit": "fact_cazy_hits",
    "fact_metabolic_meropshit": "fact_merops_hits",
}

PRIMARY_KEYS = {
    "dim_mag": ["cohort_run_id", "proteome_id"],
    "dim_gene": ["cohort_run_id", "proteome_id", "gene_id"],
    "fact_run_status": ["cohort_run_id", "proteome_id", "run_id"],
    "fact_tool_timing": ["cohort_run_id", "proteome_id", "run_id", "step"],
    "fact_qc_checkm2": ["cohort_run_id", "proteome_id", "Name"],
    "fact_qc_gunc": ["cohort_run_id", "proteome_id"],
    "fact_taxonomy_gtdbtk": ["cohort_run_id", "proteome_id"],
    "fact_kofam_hits": ["cohort_run_id", "proteome_id", "gene_id", "ko_id"],
    "fact_eggnog_annotations": ["cohort_run_id", "proteome_id", "gene_id"],
    "fact_mcycdb_hits": ["cohort_run_id", "proteome_id", "gene_id", "subject_id"],
    "fact_scycdb_hits": ["cohort_run_id", "proteome_id", "gene_id", "subject_id"],
    "fact_metabolic_hmm_hits": ["cohort_run_id", "proteome_id", "function_category", "function_name", "gene_abbreviation", "hmm_file"],
    "fact_metabolic_function_presence": ["cohort_run_id", "proteome_id", "function_category", "function_name", "gene_abbreviation"],
    "fact_metabolic_module_presence": ["cohort_run_id", "proteome_id", "module_id"],
    "fact_metabolic_module_step_presence": ["cohort_run_id", "proteome_id", "module_step_id", "ko_id"],
    "fact_cazy_hits": ["cohort_run_id", "proteome_id", "cazy_family"],
    "fact_merops_hits": ["cohort_run_id", "proteome_id", "merops_peptidase_id"],
    "feature_annotation_coverage": ["cohort_run_id", "proteome_id", "annotation_tool"],
    "feature_methane_mechanism": ["cohort_run_id", "proteome_id"],
    "feature_sulfur_competition": ["cohort_run_id", "proteome_id"],
    "feature_mrv_mag_level": ["cohort_run_id", "proteome_id"],
}


def source_tool_for_table(table_name: str) -> str:
    if table_name.startswith("fact_metabolic_") or table_name in {"fact_cazy_hits", "fact_merops_hits"}:
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
    return "MethaNet"


def import_dataframe_libs() -> tuple[Any, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required; activate methanet-fgx before running this script") from exc
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit("pyarrow is required; activate methanet-fgx before running this script") from exc
    return pd, pyarrow


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def last_status_line(path: Path) -> str:
    if not path.exists():
        return "missing_status"
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else "empty_status"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def coerce_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "present", "*"}:
        return True
    if text in {"false", "0", "no", "absent", "none", ""}:
        return False
    return None


def safe_float(value: Any) -> float | None:
    if value in (None, "", "NA", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, "", "NA", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "nan", "None", "NA", "N/A"}:
        return None
    return text


def prefixed_col(columns: list[str], suffix: str) -> str | None:
    matches = [col for col in columns if str(col).endswith(suffix)]
    return matches[0] if matches else None


def split_gene_list(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    genes = [item.strip() for item in text.split(";") if item.strip() and item.strip() != "None"]
    return ";".join(genes) if genes else None


def discover_runs(per_mag_dir: Path, cohort_run_id: str, repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}

    for run_dir in sorted(path for path in per_mag_dir.glob("*/*") if path.is_dir()):
        proteome_id = run_dir.parent.name
        run_record_path = run_dir / "curated/run_record.json"
        parquet_manifest_path = run_dir / "curated/parquet_manifest.tsv"
        record = read_json(run_record_path) if run_record_path.exists() else {}
        status = record.get("status")
        if not status:
            status = "complete" if (run_dir / "COMPLETE").exists() else "partial"
        last_status = last_status_line(run_dir / "status.tsv")
        if "failed" in last_status and not run_record_path.exists():
            status = "failed"
        run_id = record.get("run_id") or run_dir.name
        mag_id = record.get("mag_id") or proteome_id.removeprefix("mucc__").removeprefix("rumen__")
        attempt = {
            "cohort_run_id": cohort_run_id,
            "run_id": run_id,
            "proteome_id": proteome_id,
            "mag_id": mag_id,
            "source_tool": "MethaNet",
            "run_dir": rel(run_dir, repo_root),
            "run_status": status,
            "last_status": last_status,
            "has_run_record": run_record_path.exists(),
            "has_file_manifest": (run_dir / "curated/file_manifest.tsv").exists(),
            "has_parquet_manifest": parquet_manifest_path.exists(),
            "has_complete_sentinel": (run_dir / "COMPLETE").exists(),
            "mtime_epoch": run_dir.stat().st_mtime,
        }
        attempts.append(attempt)
        if status == "complete" and parquet_manifest_path.exists() and run_record_path.exists():
            current = selected.get(proteome_id)
            if current is None or attempt["mtime_epoch"] > current["attempt"]["mtime_epoch"]:
                selected[proteome_id] = {
                    "attempt": attempt,
                    "run_dir": run_dir,
                    "record": record,
                    "parquet_manifest": read_tsv(parquet_manifest_path),
                }

    return attempts, selected


def add_missing_identity(pd: Any, df: Any, table_name: str, record: dict[str, Any]) -> Any:
    df = df.copy()
    values = {
        "cohort_run_id": record.get("cohort_run_id"),
        "run_id": record.get("run_id"),
        "proteome_id": record.get("proteome_id"),
        "mag_id": record.get("mag_id"),
        "source_tool": source_tool_for_table(table_name),
    }
    if "table_name" not in df.columns:
        df.insert(0, "table_name", table_name)
    for col in reversed(IDENTITY_COLUMNS):
        if col not in df.columns:
            df.insert(0, col, values[col])
        else:
            df[col] = df[col].fillna(values[col])
    return df


def normalize_metabolic_legacy(pd: Any, table_name: str, df: Any) -> tuple[str, Any]:
    if table_name not in METABOLIC_TABLE_ALIASES:
        return table_name, df
    columns = list(df.columns)

    if table_name == "fact_metabolic_hmmhitnum":
        presence_col = prefixed_col(columns, ".Hmm.presence")
        hit_count_col = prefixed_col(columns, ".Hit.numbers")
        hits_col = prefixed_col(columns, ".Hits")
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
            "hit_count": df[hit_count_col].map(safe_int) if hit_count_col else None,
            "hit_genes": df[hits_col].map(split_gene_list) if hits_col else None,
            "evidence_source": "METABOLIC_result.xlsx:HMMHitNum",
        })
        return METABOLIC_TABLE_ALIASES[table_name], out

    if table_name == "fact_metabolic_functionhit":
        presence_col = prefixed_col(columns, ".Function.presence")
        out = pd.DataFrame({
            "function_category": df.get("Category"),
            "function_name": df.get("Function"),
            "gene_abbreviation": df.get("Gene.abbreviation"),
            "presence": df[presence_col] if presence_col else None,
            "evidence_source": "METABOLIC_result.xlsx:FunctionHit",
        })
        return METABOLIC_TABLE_ALIASES[table_name], out

    if table_name == "fact_metabolic_keggmodulehit":
        presence_col = prefixed_col(columns, ".Module.presence")
        out = pd.DataFrame({
            "module_id": df.get("Module.ID"),
            "module_name": df.get("Module"),
            "module_category": df.get("Module.Category"),
            "presence": df[presence_col] if presence_col else None,
            "hit_count": None,
            "hit_genes": None,
            "evidence_source": "METABOLIC_result.xlsx:KEGGModuleHit",
        })
        return METABOLIC_TABLE_ALIASES[table_name], out

    if table_name == "fact_metabolic_keggmodulestephit":
        presence_col = prefixed_col(columns, ".Module.step.presence")
        out = pd.DataFrame({
            "module_step_id": df.get("Module.step"),
            "module_name": df.get("Module"),
            "ko_id": df.get("KO.id"),
            "module_category": df.get("Module.Category"),
            "presence": df[presence_col] if presence_col else None,
            "evidence_source": "METABOLIC_result.xlsx:KEGGModuleStepHit",
        })
        return METABOLIC_TABLE_ALIASES[table_name], out

    if table_name == "fact_metabolic_dbcan2hit":
        hit_count_col = prefixed_col(columns, ".Hit.numbers")
        hits_col = prefixed_col(columns, ".Hits")
        out = pd.DataFrame({
            "cazy_family": df.get("CAZyme.ID"),
            "hit_count": df[hit_count_col].map(safe_int) if hit_count_col else None,
            "hit_genes": df[hits_col].map(split_gene_list) if hits_col else None,
            "evidence_source": "METABOLIC_result.xlsx:dbCAN2Hit",
        })
        return METABOLIC_TABLE_ALIASES[table_name], out

    if table_name == "fact_metabolic_meropshit":
        hit_count_col = prefixed_col(columns, ".Hit.numbers")
        hits_col = prefixed_col(columns, ".Hits")
        out = pd.DataFrame({
            "merops_peptidase_id": df.get("MEROPS.peptidase.ID"),
            "hit_count": df[hit_count_col].map(safe_int) if hit_count_col else None,
            "hit_genes": df[hits_col].map(split_gene_list) if hits_col else None,
            "evidence_source": "METABOLIC_result.xlsx:MEROPSHit",
        })
        return METABOLIC_TABLE_ALIASES[table_name], out

    return table_name, df


def load_selected_tables(pd: Any, selected: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    tables: dict[str, list[Any]] = defaultdict(list)
    for proteome_id, item in sorted(selected.items()):
        record = item["record"]
        record.setdefault("proteome_id", proteome_id)
        for row in item["parquet_manifest"]:
            table_name = row.get("table")
            path = row.get("path")
            if not table_name or not path:
                continue
            parquet_path = Path(path)
            if not parquet_path.exists():
                continue
            df = pd.read_parquet(parquet_path)
            table_name, df = normalize_metabolic_legacy(pd, table_name, df)
            df = add_missing_identity(pd, df, table_name, record)
            df["source_tool"] = source_tool_for_table(table_name)
            df["table_name"] = table_name
            tables[table_name].append(df)
    return tables


def concat_tables(pd: Any, tables: dict[str, list[Any]]) -> dict[str, Any]:
    return {name: pd.concat(frames, ignore_index=True, sort=False) for name, frames in tables.items() if frames}


def load_manifest(pd: Any, manifest_path: Path) -> Any:
    if not manifest_path.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_path, sep="\t")


def build_dim_mag(pd: Any, selected: dict[str, dict[str, Any]], manifest: Any, cohort_run_id: str) -> Any:
    manifest_by_proteome = {}
    if not manifest.empty and "proteome_id" in manifest.columns:
        manifest_by_proteome = manifest.set_index("proteome_id", drop=False).to_dict(orient="index")
    rows = []
    for proteome_id, item in sorted(selected.items()):
        record = item["record"]
        taxonomy = record.get("taxonomy", {})
        qc = record.get("qc", {})
        summary = record.get("summary_metrics", {})
        inputs = record.get("inputs", {})
        manifest_row = manifest_by_proteome.get(proteome_id, {})
        rows.append({
            "cohort_run_id": cohort_run_id,
            "run_id": record.get("run_id"),
            "proteome_id": proteome_id,
            "mag_id": record.get("mag_id"),
            "source_tool": "MethaNet",
            "sample": manifest_row.get("sample"),
            "source_group": manifest_row.get("source_group") or manifest_row.get("source"),
            "mag_fasta": manifest_row.get("mag_fasta") or inputs.get("mag_fasta"),
            "proteome_faa": manifest_row.get("proteome_faa") or inputs.get("proteome_faa"),
            "source_analysis_accession": manifest_row.get("source_analysis_accession"),
            "analysis_alias": manifest_row.get("analysis_alias"),
            "mucc_img_like_id": manifest_row.get("mucc_img_like_id"),
            "input_contigs": summary.get("input_contigs"),
            "input_total_bp": summary.get("input_total_bp"),
            "input_n50_bp": summary.get("input_n50_bp"),
            "prodigal_proteins": summary.get("prodigal_proteins"),
            "checkm2_completeness": qc.get("completeness"),
            "checkm2_contamination": qc.get("contamination"),
            "gunc_pass": qc.get("gunc_pass"),
            "qc_tier": qc.get("qc_tier"),
            "gtdb_release": taxonomy.get("gtdb_release"),
            "gtdb_classification": taxonomy.get("classification"),
            "domain": taxonomy.get("domain"),
            "phylum": taxonomy.get("phylum"),
            "class": taxonomy.get("class"),
            "order": taxonomy.get("order"),
            "family": taxonomy.get("family"),
            "genus": taxonomy.get("genus"),
            "species": taxonomy.get("species"),
        })
    return pd.DataFrame(rows)


def build_taxonomy(pd: Any, dim_mag: Any) -> Any:
    keep = [
        "cohort_run_id", "run_id", "proteome_id", "mag_id", "gtdb_release",
        "gtdb_classification", "domain", "phylum", "class", "order", "family",
        "genus", "species",
    ]
    cols = [col for col in keep if col in dim_mag.columns]
    out = dim_mag[cols].copy()
    out["source_tool"] = "GTDB-Tk"
    return out[["cohort_run_id", "run_id", "proteome_id", "mag_id", "source_tool"] + [c for c in out.columns if c not in IDENTITY_COLUMNS]]


def choose_col(df: Any, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_dim_gene(pd: Any, fact_bakta: Any | None) -> Any:
    if fact_bakta is None or fact_bakta.empty:
        return pd.DataFrame(columns=IDENTITY_COLUMNS + ["gene_id"])
    gene_col = choose_col(fact_bakta, ["Locus Tag", "locus_tag", "ID", "Name", "gene_id"])
    seq_col = choose_col(fact_bakta, ["Sequence Id", "sequence_id", "contig"])
    type_col = choose_col(fact_bakta, ["Type", "type"])
    start_col = choose_col(fact_bakta, ["Start", "start"])
    stop_col = choose_col(fact_bakta, ["Stop", "stop", "End"])
    strand_col = choose_col(fact_bakta, ["Strand", "strand"])
    product_col = choose_col(fact_bakta, ["Product", "product"])
    gene_name_col = choose_col(fact_bakta, ["Gene", "gene"])
    rows = fact_bakta.copy()
    if gene_col is None:
        rows["gene_id"] = rows.index.astype(str)
    else:
        rows["gene_id"] = rows[gene_col].astype(str)
    out = pd.DataFrame({
        "cohort_run_id": rows["cohort_run_id"],
        "run_id": rows["run_id"],
        "proteome_id": rows["proteome_id"],
        "mag_id": rows["mag_id"],
        "source_tool": "Bakta",
        "gene_id": rows["gene_id"],
        "contig_id": rows[seq_col] if seq_col else None,
        "feature_type": rows[type_col] if type_col else None,
        "start": rows[start_col] if start_col else None,
        "end": rows[stop_col] if stop_col else None,
        "strand": rows[strand_col] if strand_col else None,
        "gene_name": rows[gene_name_col] if gene_name_col else None,
        "product": rows[product_col] if product_col else None,
    })
    out = out[out["gene_id"].notna() & (out["gene_id"].astype(str) != "")]
    return out.drop_duplicates(["cohort_run_id", "proteome_id", "gene_id"])


def present_mask(series: Any) -> Any:
    return series.astype(str).str.lower().isin(["present", "true", "1", "yes"])


def build_coverage(pd: Any, dim_mag: Any, tables: dict[str, Any]) -> Any:
    rows = []
    total_by_mag = {
        row["proteome_id"]: safe_float(row.get("prodigal_proteins"))
        for _, row in dim_mag.iterrows()
    }
    specs = [
        ("KOfam", "fact_kofam_hits", "gene_id"),
        ("eggNOG", "fact_eggnog_annotations", "gene_id"),
        ("MCycDB", "fact_mcycdb_hits", "gene_id"),
        ("SCycDB", "fact_scycdb_hits", "gene_id"),
        ("dbCAN", "fact_dbcan_hits", None),
        ("Bakta", "fact_bakta_features", None),
        ("METABOLIC-HMM", "fact_metabolic_hmm_hits", None),
        ("METABOLIC-function", "fact_metabolic_function_presence", None),
        ("METABOLIC-module", "fact_metabolic_module_presence", None),
        ("METABOLIC-module-step", "fact_metabolic_module_step_presence", None),
        ("METABOLIC-CAZy", "fact_cazy_hits", None),
        ("METABOLIC-MEROPS", "fact_merops_hits", None),
    ]
    for _, mag in dim_mag.iterrows():
        proteome_id = mag["proteome_id"]
        total = total_by_mag.get(proteome_id)
        for tool, table, gene_col in specs:
            df = tables.get(table)
            if df is None or df.empty:
                rows.append({
                    "cohort_run_id": mag["cohort_run_id"],
                    "run_id": mag["run_id"],
                    "proteome_id": proteome_id,
                    "mag_id": mag["mag_id"],
                    "source_tool": "MethaNet",
                    "annotation_tool": tool,
                    "source_table": table,
                    "row_count": 0,
                    "annotated_gene_count": None,
                    "prodigal_proteins": total,
                    "annotated_gene_fraction": None,
                })
                continue
            sub = df[df["proteome_id"] == proteome_id]
            annotated = None
            if gene_col and gene_col in sub.columns:
                annotated = int(sub[gene_col].dropna().astype(str).nunique())
            rows.append({
                "cohort_run_id": mag["cohort_run_id"],
                "run_id": mag["run_id"],
                "proteome_id": proteome_id,
                "mag_id": mag["mag_id"],
                "source_tool": "MethaNet",
                "annotation_tool": tool,
                "source_table": table,
                "row_count": int(len(sub)),
                "annotated_gene_count": annotated,
                "prodigal_proteins": total,
                "annotated_gene_fraction": (annotated / total) if annotated is not None and total else None,
            })
    return pd.DataFrame(rows)


def text_contains(df: Any, columns: list[str], pattern: str) -> Any:
    if df is None or df.empty:
        return None
    mask = None
    regex = re.compile(pattern, re.IGNORECASE)
    for col in columns:
        if col not in df.columns:
            continue
        current = df[col].astype(str).map(lambda value: bool(regex.search(value)))
        mask = current if mask is None else (mask | current)
    return mask


def build_mechanism_features(pd: Any, dim_mag: Any, tables: dict[str, Any], coverage: Any) -> tuple[Any, Any, Any]:
    methane_rows = []
    sulfur_rows = []
    mrv_rows = []
    kofam = tables.get("fact_kofam_hits")
    scycdb = tables.get("fact_scycdb_hits")
    metabolic_hmm = tables.get("fact_metabolic_hmm_hits")
    metabolic_function = tables.get("fact_metabolic_function_presence")
    metabolic_module = tables.get("fact_metabolic_module_presence")
    cazy = tables.get("fact_cazy_hits")
    merops = tables.get("fact_merops_hits")

    for _, mag in dim_mag.iterrows():
        proteome_id = mag["proteome_id"]
        ident = {
            "cohort_run_id": mag["cohort_run_id"],
            "run_id": mag["run_id"],
            "proteome_id": proteome_id,
            "mag_id": mag["mag_id"],
            "source_tool": "MethaNet",
        }
        ksub = kofam[kofam["proteome_id"] == proteome_id] if kofam is not None and not kofam.empty else pd.DataFrame()
        accepted_kofam = ksub[ksub["accepted_hit"].astype(bool)] if "accepted_hit" in ksub.columns else ksub
        methane_kofam_mask = text_contains(accepted_kofam, ["ko_definition", "gene_id", "ko_id"], r"methan|methyl.?coenzyme|mcr|hdr|f420|formylmethanofuran")
        methane_kofam = accepted_kofam[methane_kofam_mask] if methane_kofam_mask is not None else pd.DataFrame()
        mhmm = metabolic_hmm[metabolic_hmm["proteome_id"] == proteome_id] if metabolic_hmm is not None and not metabolic_hmm.empty else pd.DataFrame()
        mhmm_mask = text_contains(mhmm, ["function_category", "function_name", "gene_abbreviation", "gene_name", "reaction"], r"methan|methyl.?coenzyme|mcr|hdr|f420|formylmethanofuran")
        mhmm_present = mhmm[mhmm_mask & present_mask(mhmm["presence"])] if mhmm_mask is not None and "presence" in mhmm.columns else pd.DataFrame()
        methane_rows.append({
            **ident,
            "accepted_kofam_methane_hits": int(len(methane_kofam)),
            "metabolic_methane_hmm_present": int(len(mhmm_present)),
            "methane_evidence_score": int(len(methane_kofam) + len(mhmm_present)),
            "evidence_basis": "keyword screen over accepted KOfam definitions and present METABOLIC HMM functions",
        })

        ssub = scycdb[scycdb["proteome_id"] == proteome_id] if scycdb is not None and not scycdb.empty else pd.DataFrame()
        sulfur_scyc = int(len(ssub))
        sulfur_kofam_mask = text_contains(accepted_kofam, ["ko_definition", "gene_id", "ko_id"], r"sulfur|sulfate|sulfite|sulfide|thiosulfate|sulfonate|sulfurtransferase")
        sulfur_kofam = accepted_kofam[sulfur_kofam_mask] if sulfur_kofam_mask is not None else pd.DataFrame()
        mfunc = metabolic_function[metabolic_function["proteome_id"] == proteome_id] if metabolic_function is not None and not metabolic_function.empty else pd.DataFrame()
        mfunc_mask = text_contains(mfunc, ["function_category", "function_name", "gene_abbreviation"], r"sulfur|sulfate|sulfite|sulfide|thiosulfate|sulfonate")
        mfunc_present = mfunc[mfunc_mask & present_mask(mfunc["presence"])] if mfunc_mask is not None and "presence" in mfunc.columns else pd.DataFrame()
        sulfur_rows.append({
            **ident,
            "scycdb_hit_count": sulfur_scyc,
            "accepted_kofam_sulfur_hits": int(len(sulfur_kofam)),
            "metabolic_sulfur_function_present": int(len(mfunc_present)),
            "sulfur_competition_score": int(sulfur_scyc + len(sulfur_kofam) + len(mfunc_present)),
            "evidence_basis": "SCycDB hits plus keyword screen over accepted KOfam and present METABOLIC functions",
        })

        cov_sub = coverage[coverage["proteome_id"] == proteome_id]
        kcov = cov_sub[cov_sub["annotation_tool"] == "KOfam"]["annotated_gene_fraction"].dropna()
        cazy_sub = cazy[cazy["proteome_id"] == proteome_id] if cazy is not None and not cazy.empty else pd.DataFrame()
        merops_sub = merops[merops["proteome_id"] == proteome_id] if merops is not None and not merops.empty else pd.DataFrame()
        module_sub = metabolic_module[metabolic_module["proteome_id"] == proteome_id] if metabolic_module is not None and not metabolic_module.empty else pd.DataFrame()
        module_present = int(present_mask(module_sub["presence"]).sum()) if "presence" in module_sub.columns else 0
        mrv_rows.append({
            **ident,
            "checkm2_completeness": mag.get("checkm2_completeness"),
            "checkm2_contamination": mag.get("checkm2_contamination"),
            "gunc_pass": mag.get("gunc_pass"),
            "kofam_annotated_gene_fraction": float(kcov.iloc[0]) if len(kcov) else None,
            "metabolic_modules_present": module_present,
            "cazy_family_count": int(cazy_sub["cazy_family"].nunique()) if "cazy_family" in cazy_sub.columns else 0,
            "merops_family_count": int(merops_sub["merops_peptidase_id"].nunique()) if "merops_peptidase_id" in merops_sub.columns else 0,
            "methane_evidence_score": int(len(methane_kofam) + len(mhmm_present)),
            "sulfur_competition_score": int(sulfur_scyc + len(sulfur_kofam) + len(mfunc_present)),
            "absence_interpretation_caveat": "absence requires QC and annotation coverage review",
        })

    return pd.DataFrame(methane_rows), pd.DataFrame(sulfur_rows), pd.DataFrame(mrv_rows)


def write_partitioned_tables(tables: dict[str, Any], output_dir: Path, cohort_run_id: str) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for table_name, df in sorted(tables.items()):
        if df is None or df.empty:
            continue
        table_dir = output_dir / "parquet" / table_name / f"cohort_run_id={cohort_run_id}"
        table_dir.mkdir(parents=True, exist_ok=True)
        target = table_dir / "part-00000.parquet"
        df.to_parquet(target, index=False, compression="zstd")
        manifest.append({
            "table": table_name,
            "path": str(target),
            "rows": int(len(df)),
            "columns": len(df.columns),
            "bytes": target.stat().st_size,
        })
    return manifest


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_duckdb_catalog(output_dir: Path, manifest: list[dict[str, Any]]) -> Path | None:
    try:
        import duckdb
    except ImportError:
        return None
    db_path = output_dir / "functional_atlas.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for row in manifest:
        table = row["table"]
        parquet_glob = str(output_dir / "parquet" / table / "*" / "*.parquet")
        escaped_glob = parquet_glob.replace("'", "''")
        con.execute(f'CREATE VIEW "{table}" AS SELECT * FROM read_parquet(\'{escaped_glob}\')')
    con.close()
    return db_path


def validate_tables(tables: dict[str, Any], attempts: list[dict[str, Any]], selected: dict[str, dict[str, Any]], expected_count: int | None) -> tuple[list[dict[str, Any]], str]:
    checks: list[dict[str, Any]] = []

    def add(gate: str, status: str, detail: str) -> None:
        checks.append({"gate": gate, "status": status, "detail": detail})

    dim_mag = tables.get("dim_mag")
    completed_count = len(selected)
    if expected_count:
        status = "pass" if completed_count == expected_count else "warn"
        add("selected_completed_mag_count", status, f"selected={completed_count}; expected={expected_count}")
    else:
        add("selected_completed_mag_count", "pass", f"selected={completed_count}")

    if dim_mag is not None and "proteome_id" in dim_mag.columns:
        duplicate = int(dim_mag["proteome_id"].duplicated().sum())
        add("one_row_per_proteome_id_dim_mag", "pass" if duplicate == 0 else "fail", f"duplicates={duplicate}; rows={len(dim_mag)}")

    for table, df in sorted(tables.items()):
        missing = [col for col in IDENTITY_COLUMNS if col not in df.columns]
        add(f"identity_columns:{table}", "pass" if not missing else "fail", f"missing={missing}; rows={len(df)}")
        key = PRIMARY_KEYS.get(table)
        if key and all(col in df.columns for col in key):
            dup = int(df.duplicated(key).sum())
            add(f"primary_key_duplicates:{table}", "pass" if dup == 0 else "warn", f"key={key}; duplicate_rows={dup}")

    wide_pattern = re.compile(r"^X.+\\.(Module|Function|Hmm|Hit|Hits)")
    for table in [
        "fact_metabolic_hmm_hits",
        "fact_metabolic_function_presence",
        "fact_metabolic_module_presence",
        "fact_metabolic_module_step_presence",
        "fact_cazy_hits",
        "fact_merops_hits",
    ]:
        df = tables.get(table)
        if df is None:
            add(f"metabolic_long_table_present:{table}", "fail", "missing")
            continue
        wide = [col for col in df.columns if wide_pattern.match(str(col))]
        add(f"no_tool_native_wide_columns:{table}", "pass" if not wide else "fail", f"wide_columns={wide}")

    kofam = tables.get("fact_kofam_hits")
    if kofam is not None:
        add("kofam_accepted_hit_flag", "pass" if "accepted_hit" in kofam.columns else "fail", "accepted_hit column present" if "accepted_hit" in kofam.columns else "missing accepted_hit")

    for table in ["fact_mcycdb_hits", "fact_scycdb_hits"]:
        df = tables.get(table)
        add(f"diamond_best_hit_rank:{table}", "pass" if df is not None and "hit_rank_bitscore" in df.columns else "fail", "hit_rank_bitscore present" if df is not None and "hit_rank_bitscore" in df.columns else "missing")

    coverage = tables.get("feature_annotation_coverage")
    if coverage is not None and dim_mag is not None:
        expected_pairs = len(dim_mag) * coverage["annotation_tool"].nunique()
        add("annotation_coverage_measured", "pass" if len(coverage) >= expected_pairs else "warn", f"rows={len(coverage)}; expected_min={expected_pairs}")

    required_by_run = EXPECTED_COMPLETE_TABLES | {
        "fact_metabolic_hmm_hits",
        "fact_metabolic_function_presence",
        "fact_metabolic_module_presence",
        "fact_metabolic_module_step_presence",
        "fact_taxonomy_gtdbtk",
    }
    for proteome_id in selected:
        missing = []
        for table in required_by_run:
            df = tables.get(table)
            if df is None or df[df["proteome_id"] == proteome_id].empty:
                missing.append(table)
        add(f"complete_mag_required_tables:{proteome_id}", "pass" if not missing else "fail", f"missing={missing}")

    partial_or_failed = [row for row in attempts if row["run_status"] != "complete"]
    run_status = tables.get("fact_run_status")
    preserved = run_status is not None and len(run_status) == len(attempts)
    add("failed_partial_attempts_preserved", "pass" if preserved else "fail", f"attempts={len(attempts)}; status_rows={0 if run_status is None else len(run_status)}; noncomplete={len(partial_or_failed)}")

    failures = [row for row in checks if row["status"] == "fail"]
    warnings = [row for row in checks if row["status"] == "warn"]
    if failures:
        decision = "NO-LAUNCH: fix failed data-architecture gates before full 662-MAG launch."
    elif warnings:
        decision = "CONDITIONAL NO-LAUNCH: resolve warnings or explicitly accept them before full 662-MAG launch."
    else:
        decision = "LAUNCH-READY: data-format gates passed for the inspected calibration outputs."
    return checks, decision


def markdown_report(
    output_dir: Path,
    cohort_run_id: str,
    attempts: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    decision: str,
    duckdb_path: Path | None,
) -> str:
    status_counts = Counter(row["run_status"] for row in attempts)
    table_lines = "\n".join(
        f"| {row['table']} | {row['rows']} | {row['columns']} | {row['bytes']} |"
        for row in manifest_rows
    )
    gate_counts = Counter(row["status"] for row in validation)
    failed = [row for row in validation if row["status"] == "fail"]
    warned = [row for row in validation if row["status"] == "warn"]
    issues = failed + warned
    issue_lines = "\n".join(f"- `{row['status']}` `{row['gate']}`: {row['detail']}" for row in issues[:80])
    if not issue_lines:
        issue_lines = "- None."
    return f"""# Functional MAG Cohort Data Architecture Validation

Generated: {datetime.now(timezone.utc).isoformat()}

## Scope

- Cohort run: `{cohort_run_id}`
- Run attempts inspected: {len(attempts)}
- Completed curated MAGs selected: {len(selected)}
- Attempt status counts: {dict(status_counts)}
- Output root: `{output_dir}`
- DuckDB catalog: `{duckdb_path if duckdb_path else 'not built'}`

## Decision

{decision}

## Table Model Written

| table | rows | columns | bytes |
|---|---:|---:|---:|
{table_lines}

## Validation Summary

- Gate counts: {dict(gate_counts)}

## Issues

{issue_lines}

## Final Architecture Notes

- Per-MAG run folders remain immutable evidence bundles.
- The cohort layer selects the latest completed curated run per `proteome_id`.
- Failed, partial, and superseded attempts are preserved in `fact_run_status`.
- Legacy METABOLIC workbook-derived wide columns are normalized into long analytical tables.
- Cohort tables are written as Parquet-first partitions under `parquet/<table>/cohort_run_id=<id>/`.
- DuckDB is used as a lightweight SQL catalog over the Parquet files when available.
- Absence claims must be filtered or caveated with CheckM2 completeness, contamination, GUNC status, and annotation coverage.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--cohort-run-id", default="fgx_662_apollo3_20260612")
    parser.add_argument(
        "--cohort-dir",
        default=Path("results/functional_metagenomics/fgx_662_apollo3_20260612"),
        type=Path,
    )
    parser.add_argument(
        "--manifest",
        default=Path("results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.proposed.tsv"),
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-complete-count", type=int)
    parser.add_argument("--build-duckdb", action="store_true")
    args = parser.parse_args()

    pd, _ = import_dataframe_libs()
    repo_root = args.repo_root.resolve()
    cohort_dir = args.cohort_dir if args.cohort_dir.is_absolute() else repo_root / args.cohort_dir
    per_mag_dir = cohort_dir / "per_mag"
    output_dir = args.output_dir or cohort_dir / "cohort_warehouse"
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest

    attempts, selected = discover_runs(per_mag_dir, args.cohort_run_id, repo_root)
    raw_tables = load_selected_tables(pd, selected)
    tables = concat_tables(pd, raw_tables)

    manifest = load_manifest(pd, manifest_path)
    dim_mag = build_dim_mag(pd, selected, manifest, args.cohort_run_id)
    tables["dim_mag"] = dim_mag
    tables["fact_taxonomy_gtdbtk"] = build_taxonomy(pd, dim_mag)
    tables["dim_gene"] = build_dim_gene(pd, tables.get("fact_bakta_features"))
    tables["fact_run_status"] = pd.DataFrame(attempts).drop(columns=["mtime_epoch"], errors="ignore")
    coverage = build_coverage(pd, dim_mag, tables)
    tables["feature_annotation_coverage"] = coverage
    methane, sulfur, mrv = build_mechanism_features(pd, dim_mag, tables, coverage)
    tables["feature_methane_mechanism"] = methane
    tables["feature_sulfur_competition"] = sulfur
    tables["feature_mrv_mag_level"] = mrv

    validation, decision = validate_tables(tables, attempts, selected, args.expected_complete_count)
    table_manifest = write_partitioned_tables(tables, output_dir, args.cohort_run_id)
    write_tsv(output_dir / "cohort_table_manifest.tsv", table_manifest)
    write_tsv(output_dir / "validation_gates.tsv", validation)
    duckdb_path = build_duckdb_catalog(output_dir, table_manifest) if args.build_duckdb else None
    report = markdown_report(output_dir, args.cohort_run_id, attempts, selected, table_manifest, validation, decision, duckdb_path)
    (output_dir / "DATA_ARCHITECTURE_VALIDATION.md").write_text(report)
    print(output_dir / "DATA_ARCHITECTURE_VALIDATION.md")
    print(decision)
    return 1 if any(row["status"] == "fail" for row in validation) else 0


if __name__ == "__main__":
    raise SystemExit(main())
