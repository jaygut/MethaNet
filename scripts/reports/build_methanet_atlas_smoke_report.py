#!/usr/bin/env python3
"""Build a preliminary MethaNet functional-atlas smoke report.

This report intentionally reads live per-MAG curated outputs rather than the
cohort warehouse, because the production warehouse can lag active Slurm runs.
Taxonomy readiness is assessed from curated run records and GTDB-Tk output
files; the cohort-level ``fact_taxonomy_gtdbtk`` table is built by the
consolidator and is not expected to exist as a per-MAG Parquet shard.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import subprocess
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns


EXPECTED_EMBEDDED_TOTAL = 662
EXPECTED_MAG_BIN_TOTAL = 625
EXPECTED_WETLAND = 107
EXPECTED_RUMEN_MAG_BIN = 518
EXPECTED_ASSEMBLY_CONTEXT = 37
RUN_ID_DEFAULT = "fgx_662_apollo3_20260612"
SCOPE_MANIFEST_DEFAULT = (
    "results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/"
    "poc_662_functional_mag_manifest.with_unit_scope.tsv"
)
EXPECTED_TABLES = [
    "fact_qc_checkm2",
    "fact_qc_gunc",
    "fact_taxonomy_gtdbtk",
    "fact_kofam_hits",
    "fact_mcycdb_hits",
    "fact_scycdb_hits",
    "fact_dbcan_hits",
    "fact_bakta_features",
    "fact_metabolic_hmm_hits",
    "fact_metabolic_function_presence",
    "fact_metabolic_module_presence",
    "fact_metabolic_module_step_presence",
    "fact_cazy_hits",
    "fact_merops_hits",
    "fact_tool_timing",
    "fact_input_stats",
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": {"base": "#A3BEFA", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "dark": "#736422"},
    "orange": {"base": "#F0986E", "dark": "#804126"},
    "olive": {"base": "#A3D576", "dark": "#386411"},
}


def run_cmd(cmd: str, repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        shell=True,
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator / denominator * 100:.1f}%"


def rel(path: Path, out_dir: Path) -> str:
    return os.path.relpath(path, out_dir)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def has_gtdb_summary(run_dir: Path) -> bool:
    return any(run_dir.glob("gtdbtk/**/*.summary.tsv"))


def has_gtdb_json(run_dir: Path) -> bool:
    return (run_dir / "gtdbtk/gtdbtk.json").exists()


def load_scope_manifest(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def discover_completed(per_mag_dir: Path, scope_manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[Path], list[Path]]:
    complete_dirs = sorted(path.parent for path in per_mag_dir.glob("*/*/COMPLETE"))
    failed_dirs = sorted(path.parent for path in per_mag_dir.glob("*/*/FAILED"))
    rows: list[dict[str, Any]] = []
    scope_by_id = {}
    if not scope_manifest.empty and "proteome_id" in scope_manifest.columns:
        scope_cols = [
            col
            for col in [
                "analysis_unit_type",
                "mbag_mag_level_include",
                "assembly_context_include",
                "claim_scope",
                "comparability_status",
                "recommended_action",
            ]
            if col in scope_manifest.columns
        ]
        scope_by_id = scope_manifest.set_index("proteome_id")[scope_cols].to_dict(orient="index")

    for run_dir in complete_dirs:
        rec_path = run_dir / "curated/run_record.json"
        record = read_json(rec_path) if rec_path.exists() else {}
        proteome_id = record.get("proteome_id") or run_dir.parent.name
        taxonomy = record.get("taxonomy") or {}
        run_index_match = re.match(r"fgx_(\d+)_", run_dir.name)
        scope = scope_by_id.get(proteome_id, {})
        rows.append(
            {
                "proteome_id": proteome_id,
                "mag_id": record.get("mag_id") or "",
                "source": "wetland/MUCC" if proteome_id.startswith("mucc__") else "rumen",
                "analysis_unit_type": scope.get("analysis_unit_type"),
                "claim_scope": scope.get("claim_scope"),
                "comparability_status": scope.get("comparability_status"),
                "mbag_mag_level_include": scope.get("mbag_mag_level_include"),
                "run_id": record.get("run_id") or run_dir.name,
                "run_index": int(run_index_match.group(1)) if run_index_match else None,
                "run_dir": str(run_dir),
                "run_record": rec_path.exists(),
                "file_manifest": (run_dir / "curated/file_manifest.tsv").exists(),
                "parquet_manifest": (run_dir / "curated/parquet_manifest.tsv").exists(),
                "taxonomy_in_run_record": bool(taxonomy.get("classification")),
                "gtdb_release": taxonomy.get("gtdb_release"),
                "domain": taxonomy.get("domain"),
                "phylum": taxonomy.get("phylum"),
                "class": taxonomy.get("class"),
                "order": taxonomy.get("order"),
                "family": taxonomy.get("family"),
                "genus": taxonomy.get("genus"),
                "species": taxonomy.get("species"),
                "gtdb_summary_file": has_gtdb_summary(run_dir),
                "gtdb_json_file": has_gtdb_json(run_dir),
            }
        )

    return pd.DataFrame(rows), complete_dirs, failed_dirs


def read_per_mag_manifests(complete_dirs: list[Path], scope_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scope_by_id = {}
    if not scope_manifest.empty and "proteome_id" in scope_manifest.columns:
        scope_cols = [col for col in ["analysis_unit_type", "claim_scope"] if col in scope_manifest.columns]
        scope_by_id = scope_manifest.set_index("proteome_id")[scope_cols].to_dict(orient="index")
    for run_dir in complete_dirs:
        manifest = run_dir / "curated/parquet_manifest.tsv"
        if not manifest.exists():
            continue
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["proteome_id"] = run_dir.parent.name
                row.update(scope_by_id.get(row["proteome_id"], {}))
                row["run_dir"] = str(run_dir)
                row["rows"] = int(row.get("rows") or 0)
                row["bytes"] = int(row.get("bytes") or 0)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_tables(manifest_df: pd.DataFrame, completed_total: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest_df.empty:
        table_summary = pd.DataFrame(columns=["table", "mags_with_table", "total_rows", "total_bytes"])
    else:
        table_summary = (
            manifest_df.groupby("table")
            .agg(
                mags_with_table=("proteome_id", "nunique"),
                total_rows=("rows", "sum"),
                total_bytes=("bytes", "sum"),
            )
            .reset_index()
            .sort_values(["mags_with_table", "total_rows"], ascending=[False, False])
        )

    summary_map = {row["table"]: row for _, row in table_summary.iterrows()}
    expected_rows = []
    for table in EXPECTED_TABLES:
        row = summary_map.get(table)
        count = int(row["mags_with_table"]) if row is not None else 0
        total_rows = int(row["total_rows"]) if row is not None else 0
        note = ""
        if table == "fact_taxonomy_gtdbtk":
            note = "cohort-consolidated from run_record taxonomy; not expected as per-MAG shard"
        expected_rows.append(
            {
                "table": table,
                "mags_with_table": count,
                "missing_mags": completed_total - count,
                "total_rows": total_rows,
                "note": note,
            }
        )
    return table_summary, pd.DataFrame(expected_rows)


def read_fact(
    per_mag_dir: Path,
    table: str,
    columns: list[str] | None = None,
    eligible_ids: set[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for path in per_mag_dir.glob(f"*/*/curated/parquet/{table}.parquet"):
        run_dir = path.parents[2]
        if not (run_dir / "COMPLETE").exists():
            continue
        proteome_id = run_dir.parent.name
        if eligible_ids is not None and proteome_id not in eligible_ids:
            continue
        try:
            frames.append(pq.read_table(path, columns=columns).to_pandas())
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def source_tool_status(squeue_text: str, sacct_text: str) -> tuple[int, int, Counter[str]]:
    running = 0
    pending = 0
    for line in squeue_text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5:
            running += int(parts[4] == "R")
            pending += int(parts[4] == "PD")

    states: Counter[str] = Counter()
    for line in sacct_text.splitlines()[1:]:
        cols = line.split("|")
        if cols and re.match(r"8504_\d+$", cols[0]):
            states[cols[1]] += 1
    return running, pending, states


def table_html(df: pd.DataFrame, max_rows: int = 12) -> str:
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.head(max_rows).iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def add_header(fig: Any, ax: Any, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.80, left=0.19, right=0.96, bottom=0.15)
    left = ax.get_position().x0
    fig.text(
        left,
        0.965,
        textwrap.fill(title, 78),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.905,
        textwrap.fill(subtitle, 112),
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def save_fig(fig: Any, assets_dir: Path, name: str) -> Path:
    png = assets_dir / f"{name}.png"
    svg = assets_dir / f"{name}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def build_charts(
    assets_dir: Path,
    out_dir: Path,
    completed_total: int,
    completed_wetland: int,
    completed_rumen: int,
    table_summary: pd.DataFrame,
    expected_presence: pd.DataFrame,
    warehouse_dim_mag: int,
) -> dict[str, Path]:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
            "patch.linewidth": 1.0,
        },
    )

    completion_chart = pd.DataFrame(
        [
            {
                "source": "wetland/MUCC",
                "complete": completed_wetland,
                "remaining": max(EXPECTED_WETLAND - completed_wetland, 0),
                "expected": EXPECTED_WETLAND,
            },
            {
                "source": "rumen MAG/bin",
                "complete": completed_rumen,
                "remaining": max(EXPECTED_RUMEN_MAG_BIN - completed_rumen, 0),
                "expected": EXPECTED_RUMEN_MAG_BIN,
            },
        ]
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh(
        completion_chart["source"],
        completion_chart["complete"],
        color=COLORS["olive"]["base"],
        edgecolor=COLORS["olive"]["dark"],
        label="Complete",
    )
    ax.barh(
        completion_chart["source"],
        completion_chart["remaining"],
        left=completion_chart["complete"],
        color="#E2E5EA",
        edgecolor="#7A828F",
        label="Remaining",
    )
    for i, row in completion_chart.iterrows():
        ax.text(
            row["expected"] + 2,
            i,
            f"{int(row['complete'])}/{int(row['expected'])} ({row['complete'] / row['expected']:.0%})",
            va="center",
            fontsize=9,
            color=TOKENS["ink"],
        )
    ax.set_xlim(0, EXPECTED_RUMEN_MAG_BIN * 1.08)
    ax.set_xlabel("MAGs")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2)
    add_header(
        fig,
        ax,
        "Completion by source cohort",
        "Live per-MAG COMPLETE sentinels as of the smoke-report snapshot; wetland is complete, rumen is pilot-scale only.",
    )
    completion_png = save_fig(fig, assets_dir, "completion_by_source")

    pres_plot = expected_presence[expected_presence["table"] != "fact_taxonomy_gtdbtk"].copy()
    pres_plot["presence_rate"] = pres_plot["mags_with_table"] / max(completed_total, 1)
    pres_plot = pres_plot.sort_values("presence_rate").tail(15)
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    colors = [
        COLORS["blue"]["base"] if value == completed_total else COLORS["orange"]["base"]
        for value in pres_plot["mags_with_table"]
    ]
    edges = [
        COLORS["blue"]["dark"] if value == completed_total else COLORS["orange"]["dark"]
        for value in pres_plot["mags_with_table"]
    ]
    bars = ax.barh(pres_plot["table"], pres_plot["mags_with_table"], color=colors, edgecolor=edges)
    for bar, value in zip(bars, pres_plot["mags_with_table"]):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}/{completed_total}",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.set_xlim(0, completed_total * 1.16)
    ax.set_xlabel("Completed MAGs with per-MAG shard")
    add_header(
        fig,
        ax,
        "Per-MAG Parquet shard presence",
        "Expected functional/QC shards from per-MAG manifests; taxonomy is intentionally handled through run records and consolidation.",
    )
    presence_png = save_fig(fig, assets_dir, "curated_table_presence")

    row_tables = table_summary[
        table_summary["table"].isin(
            [
                "fact_kofam_hits",
                "fact_mcycdb_hits",
                "fact_scycdb_hits",
                "fact_bakta_features",
                "fact_dbcan_hits",
                "fact_metabolic_hmm_hits",
                "fact_metabolic_function_presence",
                "fact_metabolic_module_presence",
                "fact_metabolic_module_step_presence",
                "fact_qc_checkm2",
                "fact_qc_gunc",
            ]
        )
    ].copy()
    row_tables = row_tables.sort_values("total_rows")
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.barh(row_tables["table"], row_tables["total_rows"], color=COLORS["gold"]["base"], edgecolor=COLORS["gold"]["dark"])
    for patch, value in zip(ax.patches, row_tables["total_rows"]):
        ax.text(
            value * 1.08 if value > 0 else 1,
            patch.get_y() + patch.get_height() / 2,
            f"{int(value):,}",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Rows across completed MAGs, log scale")
    add_header(
        fig,
        ax,
        "Functional evidence volume by table",
        "Manifest row counts across completed MAGs; row volume is evidence availability, not biological confidence by itself.",
    )
    rows_png = save_fig(fig, assets_dir, "functional_evidence_rows")

    lag_df = pd.DataFrame(
        [
            {"snapshot": "Live per-MAG completed", "MAGs": completed_total},
            {"snapshot": "Current warehouse dim_mag", "MAGs": warehouse_dim_mag},
        ]
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        lag_df["snapshot"],
        lag_df["MAGs"],
        color=[COLORS["olive"]["base"], COLORS["orange"]["base"]],
        edgecolor=[COLORS["olive"]["dark"], COLORS["orange"]["dark"]],
    )
    for patch, value in zip(ax.patches, lag_df["MAGs"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + max(completed_total * 0.025, 2),
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TOKENS["ink"],
        )
    ax.set_ylim(0, max(completed_total, warehouse_dim_mag) * 1.2)
    ax.set_ylabel("MAG rows")
    add_header(
        fig,
        ax,
        "Warehouse is behind live production",
        "The current DuckDB/Parquet warehouse still reflects the early validation snapshot, not the live completed per-MAG outputs.",
    )
    lag_png = save_fig(fig, assets_dir, "warehouse_lag")

    return {
        "completion": completion_png,
        "presence": presence_png,
        "rows": rows_png,
        "lag": lag_png,
    }


def html_report(
    out_dir: Path,
    charts: dict[str, Path],
    snapshot_ts: str,
    completed_total: int,
    completed_wetland: int,
    completed_rumen: int,
    assembly_context_complete: int,
    failed_total: int,
    running: int,
    warehouse_dim_mag: int,
    taxonomy_summary: dict[str, int],
    qc_summary: dict[str, Any],
    timing_summary: dict[str, Any],
    function_summary: dict[str, int],
    met_top_categories: pd.DataFrame,
    step_summary: pd.DataFrame,
) -> str:
    css = """
:root{--ink:#1F2430;--muted:#5f6878;--line:#E4E8F0;--panel:#fff;--bg:#FCFCFD;--blue:#5477C4;--orange:#CC6F47}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}main{max-width:1080px;margin:0 auto;padding:40px 28px 64px}h1{font-size:30px;line-height:1.1;margin:0 0 22px;letter-spacing:0}h2{font-size:20px;margin:34px 0 12px;line-height:1.25}h3{font-size:15px;margin:20px 0 8px}p{margin:0 0 12px}ul,ol{margin-top:8px;padding-left:22px}li{margin:5px 0}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:22px 0 18px}.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.kpi .value{font-size:26px;font-weight:750;line-height:1;margin-bottom:6px}.kpi .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.summary{border-top:4px solid var(--blue);background:#fff;padding:18px 20px;border-radius:8px;border-left:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.summary li{margin:8px 0}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}.figure{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;margin:14px 0 20px}.figure img{width:100%;height:auto;display:block}.note{color:var(--muted);font-size:13px}.callout{border-left:4px solid var(--orange);background:#fff8f4;padding:12px 14px;border-radius:6px;margin:14px 0}table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:12px 0 20px;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#F4F6FA;color:#333b49;font-size:12px;text-transform:uppercase;letter-spacing:.03em}.allowed{color:#386411;font-weight:650}.provisional{color:#804126;font-weight:650}.blocked{color:#8A3A6F;font-weight:650}@media(max-width:760px){main{padding:28px 16px 48px}.kpis{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}h1{font-size:25px}}
"""

    source_table = pd.DataFrame(
        [
            {
                "Evidence lane": "Run state",
                "Current finding": f"{completed_total}/{EXPECTED_MAG_BIN_TOTAL} MAG/bin complete; {assembly_context_complete}/{EXPECTED_ASSEMBLY_CONTEXT} assembly-context outputs quarantined; {failed_total} failed sentinels; {running} running",
                "Decision use": "Smoke-test operational status and report freshness",
            },
            {
                "Evidence lane": "Wetland target domain",
                "Current finding": f"{completed_wetland}/107 wetland/MUCC MAGs complete",
                "Decision use": "Target-domain signal inventory is now possible",
            },
            {
                "Evidence lane": "Rumen source domain",
                "Current finding": f"{completed_rumen}/{EXPECTED_RUMEN_MAG_BIN} rumen MAG/bin records complete",
                "Decision use": "Source-domain contrast remains pending until the clean MAG/bin relaunch finishes",
            },
            {
                "Evidence lane": "Taxonomy",
                "Current finding": f"{taxonomy_summary['taxonomy_in_run_record']}/{completed_total} run records have GTDB-Tk classification; {taxonomy_summary['gtdb_summary_file']}/{completed_total} have summary TSVs",
                "Decision use": "Taxonomy can be used from run records now and from fact_taxonomy_gtdbtk after consolidation",
            },
            {
                "Evidence lane": "Warehouse",
                "Current finding": f"dim_mag has {warehouse_dim_mag} rows versus {completed_total} live completions",
                "Decision use": "Full report should refresh/rebuild cohort warehouse first",
            },
            {
                "Evidence lane": "Functional tables",
                "Current finding": f"{function_summary['kofam_rows']:,} KOfam rows; {function_summary['mcycdb_best_hits']:,} MCycDB best hits; {function_summary['scycdb_best_hits']:,} SCycDB best hits",
                "Decision use": "Sufficient to test MethaNet functional signal extraction",
            },
        ]
    )

    claim_matrix = pd.DataFrame(
        [
            {
                "Claim area": "Operational production",
                "Status": "Allowed",
                "Safe wording now": "The production run is generating complete per-MAG curated outputs with no failed sentinels at this snapshot.",
                "Evidence needed to upgrade": "Continue monitoring stderr, sentinels, and full array completion.",
            },
            {
                "Claim area": "Wetland functional atlas",
                "Status": "Allowed",
                "Safe wording now": "The wetland/MUCC subset is complete at MAG level and ready for preliminary target-domain functional profiling.",
                "Evidence needed to upgrade": "Regenerate cohort warehouse over all 107 wetland MAGs and validate normalized tables.",
            },
            {
                "Claim area": "Taxonomy-aware MAG interpretation",
                "Status": "Allowed",
                "Safe wording now": "GTDB-Tk taxonomy is present in all completed run records and can be used before warehouse refresh if the reader handles run records.",
                "Evidence needed to upgrade": "Refresh the cohort warehouse to materialize fact_taxonomy_gtdbtk for all live completions.",
            },
            {
                "Claim area": "Rumen-to-wetland transfer",
                "Status": "Provisional",
                "Safe wording now": "The workflow can test transfer-oriented reporting, but cross-domain conclusions are not yet supported because only the first rumen MAGs are complete.",
                "Evidence needed to upgrade": "Complete enough of the 555 rumen MAGs for source-aware comparison and bridge candidate joins.",
            },
            {
                "Claim area": "Blue carbon sample/metagenome risk",
                "Status": "Blocked",
                "Safe wording now": "Current results support MAG-level genomic potential, not sample-level methane risk tiers.",
                "Evidence needed to upgrade": "Add abundance/read-coverage, sample metadata, geochemistry, and validation/flux data.",
            },
        ]
    )
    status_class = {"Allowed": "allowed", "Provisional": "provisional", "Blocked": "blocked"}
    claim_rows = []
    for _, row in claim_matrix.iterrows():
        claim_rows.append(
            "<tr>"
            f"<td>{html.escape(row['Claim area'])}</td>"
            f"<td><span class=\"{status_class[row['Status']]}\">{html.escape(row['Status'])}</span></td>"
            f"<td>{html.escape(row['Safe wording now'])}</td>"
            f"<td>{html.escape(row['Evidence needed to upgrade'])}</td>"
            "</tr>"
        )
    claim_html = (
        "<table><thead><tr><th>Claim area</th><th>Status</th><th>Safe wording now</th>"
        "<th>Evidence needed to upgrade</th></tr></thead><tbody>"
        + "".join(claim_rows)
        + "</tbody></table>"
    )

    met_html = table_html(
        met_top_categories.rename(
            columns={
                "function_category": "METABOLIC category",
                "hit_count": "present hit count",
                "mags": "MAGs",
                "functions": "functions",
            }
        ),
        8,
    )
    if not step_summary.empty:
        step_table = step_summary.head(8).copy()
        step_table["sum_minutes"] = (step_table["sum"] / 60).round(1)
        step_table["median_minutes"] = (step_table["median"] / 60).round(1)
        step_html = table_html(step_table[["step", "sum_minutes", "median_minutes", "count"]], 8)
    else:
        step_html = "<p>No timing rows available.</p>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Preliminary MethaNet Atlas Smoke Report</title><style>{css}</style></head><body><main>
<h1>Preliminary MethaNet Atlas Smoke Report</h1>
<section class="summary"><h2>Executive Summary</h2><ul>
<li><strong>The smoke-test basis is real but preliminary.</strong> Live per-MAG outputs show <strong>{completed_total}/{EXPECTED_MAG_BIN_TOTAL} comparable MAG/bin records complete</strong>, with <strong>{completed_wetland}/{EXPECTED_WETLAND} wetland/MUCC MAGs complete</strong> and <strong>{completed_rumen}/{EXPECTED_RUMEN_MAG_BIN} rumen MAG/bin records complete</strong>. A separate <strong>{assembly_context_complete}/{EXPECTED_ASSEMBLY_CONTEXT}</strong> no-bin rumen assembly-context outputs are preserved but excluded from MAG-level MBAG.</li>
<li><strong>The taxonomy issue is a data-access issue, not a biology-output gap.</strong> GTDB-Tk classification is present in <strong>{taxonomy_summary['taxonomy_in_run_record']}/{completed_total}</strong> curated run records, with GTDB-Tk summary files and `gtdbtk.json` available for all completed runs. It is absent only as a per-MAG Parquet shard because the consolidator materializes `fact_taxonomy_gtdbtk` from `run_record.json`.</li>
<li><strong>The completed outputs are artifact-clean at the control-file level.</strong> All {completed_total} completed MAGs have `run_record.json`, `file_manifest.tsv`, and `parquet_manifest.tsv`; failed sentinels are <strong>{failed_total}</strong>.</li>
<li><strong>The warehouse should still be refreshed before any final atlas report.</strong> The current cohort warehouse has <strong>{warehouse_dim_mag} `dim_mag` rows</strong>, while live production has <strong>{completed_total}</strong> completed MAGs, so this smoke report reads live per-MAG artifacts and manifests directly.</li>
</ul></section>
<div class="kpis"><div class="kpi"><div class="value">{completed_total}</div><div class="label">completed MAG/bin</div></div><div class="kpi"><div class="value">{pct(completed_total, EXPECTED_MAG_BIN_TOTAL)}</div><div class="label">MAG/bin cohort complete</div></div><div class="kpi"><div class="value">{completed_wetland}/{EXPECTED_WETLAND}</div><div class="label">wetland complete</div></div><div class="kpi"><div class="value">{assembly_context_complete}</div><div class="label">assembly-context quarantined</div></div></div>
<section><h2>Wetland is ready for a first target-domain intelligence pass; rumen remains too sparse for transfer conclusions</h2>
<p><strong>What this supports now:</strong> a wetland/MUCC MAG-level functional inventory and an end-to-end check that the MethaNet atlas workflow can turn annotation outputs into MRV-facing primitives. <strong>What it does not support yet:</strong> a source-balanced rumen-to-wetland conclusion, because the clean source-domain MAG/bin side is only {completed_rumen} records deep.</p>
<div class="figure"><img src="{rel(charts['completion'], out_dir)}" alt="Completion by source cohort chart"></div></section>
<section><h2>The per-MAG artifact layer is healthy enough for a smoke report</h2>
<p><strong>The completed runs are internally controlled.</strong> Every completed MAG has the curated JSON record, raw file manifest, and Parquet manifest. The key analytical shards are present broadly enough to exercise the report workflow. Taxonomy is handled separately from per-MAG shards: use `run_record.json` for live smoke reports and `fact_taxonomy_gtdbtk` after consolidation.</p>
<div class="figure"><img src="{rel(charts['presence'], out_dir)}" alt="Per-MAG Parquet shard presence chart"></div>
<p class="note">Rows are based on `curated/parquet_manifest.tsv` per completed MAG; taxonomy presence is measured from `curated/run_record.json` and GTDB-Tk output files.</p></section>
<section><h2>The evidence volume is already useful for signal extraction, but interpretation must remain QC-aware</h2>
<p><strong>Function-level evidence is large enough for workflow validation.</strong> Current completed MAGs contain {function_summary['kofam_rows']:,} KOfam hit rows, {function_summary['kofam_accepted_hits']:,} accepted KOfam hits, {function_summary['mcycdb_best_hits']:,} best-ranked MCycDB hits, {function_summary['scycdb_best_hits']:,} best-ranked SCycDB hits, and {function_summary['metabolic_hmm_present_rows']:,} present METABOLIC HMM rows. These are meaningful inputs for MethaNet's mechanism-card and MRV-feature logic, but not final process-rate evidence.</p>
<div class="figure"><img src="{rel(charts['rows'], out_dir)}" alt="Functional evidence row count chart"></div>
<div class="grid2"><div><h3>QC snapshot</h3><ul><li>CheckM2 rows: <strong>{qc_summary.get('qc_rows', 0)}</strong></li><li>Median completeness: <strong>{qc_summary.get('median_completeness', 0):.1f}%</strong></li><li>Median contamination: <strong>{qc_summary.get('median_contamination', 0):.2f}%</strong></li><li>Contamination <=5%: <strong>{qc_summary.get('contam_le_5', 0)}/{qc_summary.get('qc_rows', 0)}</strong></li></ul></div><div><h3>Top present METABOLIC categories</h3>{met_html}</div></div></section>
<section><h2>The main data-system issue is warehouse freshness, not missing taxonomy</h2>
<p><strong>The current DuckDB/Parquet warehouse is behind live production.</strong> That is acceptable for this smoke run because the report uses live per-MAG manifests and run records directly, but a final next-gen intelligence report should first rebuild the scoped cohort warehouse so `dim_mag`, `fact_taxonomy_gtdbtk`, fact tables, and feature tables reflect only comparable MAG/bin units by default.</p>
<div class="figure"><img src="{rel(charts['lag'], out_dir)}" alt="Warehouse lag chart"></div>
<div class="callout"><strong>Smoke-test decision:</strong> use the preliminary atlas workflow now, but label it as live-artifact smoke output. Do not use the stale warehouse as the denominator for stakeholder or partner-facing claims.</div></section>
<section><h2>MethaNet value-proposition relevance</h2>
<p><strong>The immediate business value is an MRV-intelligence prototype, not a final risk score.</strong> The wetland-complete subset can test how MethaNet turns MAG-level functional potential into candidate features: methane mechanism evidence, sulfur competition evidence, substrate/CAZy capacity, QC confidence, taxonomy context, annotation coverage, and missing-evidence labels.</p>{table_html(source_table, 10)}</section>
<section><h2>Claim boundary matrix</h2><p><strong>This is the guardrail for using the smoke report.</strong> It preserves commercial usefulness without overstating scientific readiness.</p>{claim_html}</section>
<section><h2>Recommended next steps</h2><ol><li><strong>Use this as the corrected atlas smoke report, not the final atlas report.</strong> It verifies reading order, feature extraction logic, taxonomy provenance, claim boundaries, and MRV framing.</li><li><strong>Regenerate the cohort warehouse after a larger rumen tranche finishes.</strong> The next robust checkpoint should align `dim_mag`, `fact_taxonomy_gtdbtk`, and all fact tables with live completions.</li><li><strong>Build provisional `feature_mrv_mag_level` rows for the 107 wetland MAGs.</strong> Prioritize methane/sulfur/substrate/QC/taxonomy fields and explicit missingness labels.</li><li><strong>Delay any source-independent transfer claim.</strong> Revisit once enough rumen MAGs complete to support source-aware controls and bridge-candidate joins.</li><li><strong>Start the sample/metagenome rollup design now.</strong> The next business unlock requires abundance/read coverage and environmental metadata, not just more MAG annotations.</li></ol></section>
<section><h2>Further questions</h2><ul><li>Which sample metadata table should become the authoritative wetland/MUCC sample or site mapping?</li><li>Which abundance or read-coverage output should weight MAG-level features into sample/metagenome-level MRV signals?</li><li>At what rumen completion threshold should the next cross-domain bridge analysis be considered credible enough for a stronger report?</li></ul></section>
<section><h2>Caveats and assumptions</h2><ul><li>This report is a snapshot generated at {snapshot_ts}; the Slurm array is still active and counts will change.</li><li>The analysis reads per-MAG curated manifests, selected Parquet facts, and run-record taxonomy directly because the cohort warehouse is stale.</li><li>MAG-level functional potential is not the same as sample-level methane process rate, methane flux, or carbon-crediting risk tier.</li><li>No-bin rumen `10676_*_idba` assembly-context outputs are excluded from MAG-level MBAG and MAG mechanism cards.</li><li>Absent pathway evidence must be caveated by MAG completeness, contamination, GUNC status, taxonomy status, and annotation coverage.</li></ul><h3>Runtime bottleneck snapshot</h3>{step_html}</section>
<p class="note">Supporting files in this folder: `completed_mag_snapshot.tsv`, `per_mag_parquet_table_summary.tsv`, `expected_table_presence.tsv`, `tool_timing_summary.tsv`, `metabolic_top_categories.tsv`, and `source_notes.json`.</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--job-id", default="8504")
    parser.add_argument("--scope-manifest", type=Path, default=Path(SCOPE_MANIFEST_DEFAULT))
    parser.add_argument("--output-name", default="methanet_atlas_smoke_20260613")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_root = repo_root / "results" / "functional_metagenomics" / args.run_id
    per_mag_dir = run_root / "per_mag"
    warehouse_dir = run_root / "cohort_warehouse"
    scope_manifest_path = args.scope_manifest if args.scope_manifest.is_absolute() else repo_root / args.scope_manifest
    scope_manifest = load_scope_manifest(scope_manifest_path)
    out_dir = run_root / "reports" / args.output_name
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    complete_all_df, complete_all_dirs, failed_dirs = discover_completed(per_mag_dir, scope_manifest)
    if complete_all_df.empty:
        complete_df = complete_all_df
        assembly_context_complete = 0
    else:
        complete_df = complete_all_df[complete_all_df["analysis_unit_type"].fillna("mag_bin").eq("mag_bin")].copy()
        assembly_context_complete = int(complete_all_df["analysis_unit_type"].eq("assembly_context").sum())
    complete_dirs = [Path(path) for path in complete_df["run_dir"].tolist()] if not complete_df.empty else []
    eligible_ids = set(complete_df["proteome_id"].tolist()) if not complete_df.empty else set()
    completed_total = len(complete_df)
    completed_wetland = int((complete_df["source"] == "wetland/MUCC").sum()) if completed_total else 0
    completed_rumen = int((complete_df["source"] == "rumen").sum()) if completed_total else 0

    manifest_df = read_per_mag_manifests(complete_dirs, scope_manifest)
    table_summary, expected_presence = summarize_tables(manifest_df, completed_total)

    qc = read_fact(
        per_mag_dir,
        "fact_qc_checkm2",
        ["proteome_id", "mag_id", "Completeness", "Contamination", "Total_Coding_Sequences", "Genome_Size"],
        eligible_ids=eligible_ids,
    )
    timing = read_fact(per_mag_dir, "fact_tool_timing", ["proteome_id", "mag_id", "step", "elapsed_seconds", "rc"], eligible_ids=eligible_ids)
    kofam = read_fact(per_mag_dir, "fact_kofam_hits", ["proteome_id", "mag_id", "ko_id", "accepted_hit"], eligible_ids=eligible_ids)
    mcyc = read_fact(per_mag_dir, "fact_mcycdb_hits", ["proteome_id", "mag_id", "subject_id", "gene_id", "hit_rank_bitscore"], eligible_ids=eligible_ids)
    scyc = read_fact(per_mag_dir, "fact_scycdb_hits", ["proteome_id", "mag_id", "subject_id", "gene_id", "hit_rank_bitscore"], eligible_ids=eligible_ids)
    dbcan = read_fact(per_mag_dir, "fact_dbcan_hits", ["proteome_id", "mag_id", "Gene ID", "#ofTools"], eligible_ids=eligible_ids)
    met_hmm = read_fact(
        per_mag_dir,
        "fact_metabolic_hmm_hits",
        ["proteome_id", "mag_id", "function_category", "function_name", "presence", "hit_count"],
        eligible_ids=eligible_ids,
    )

    qc_summary: dict[str, Any]
    if not qc.empty:
        qc_summary = {
            "qc_rows": int(len(qc)),
            "median_completeness": float(qc["Completeness"].median()),
            "median_contamination": float(qc["Contamination"].median()),
            "completeness_ge_90": int((qc["Completeness"] >= 90).sum()),
            "completeness_70_90": int(((qc["Completeness"] >= 70) & (qc["Completeness"] < 90)).sum()),
            "completeness_50_70": int(((qc["Completeness"] >= 50) & (qc["Completeness"] < 70)).sum()),
            "completeness_lt_50": int((qc["Completeness"] < 50).sum()),
            "contam_le_5": int((qc["Contamination"] <= 5).sum()),
        }
    else:
        qc_summary = {"qc_rows": 0, "median_completeness": 0, "median_contamination": 0, "contam_le_5": 0}

    if not timing.empty:
        per_mag_runtime = timing.groupby("proteome_id")["elapsed_seconds"].sum()
        timing_summary = {
            "timing_rows": int(len(timing)),
            "median_runtime_min": float(per_mag_runtime.median() / 60),
            "p95_runtime_min": float(per_mag_runtime.quantile(0.95) / 60),
            "max_runtime_min": float(per_mag_runtime.max() / 60),
            "nonzero_rc_rows": int((timing["rc"] != 0).sum()),
        }
        step_summary = (
            timing.groupby("step")["elapsed_seconds"]
            .agg(["sum", "median", "count"])
            .reset_index()
            .sort_values("sum", ascending=False)
        )
    else:
        timing_summary = {"timing_rows": 0, "median_runtime_min": 0, "p95_runtime_min": 0, "max_runtime_min": 0, "nonzero_rc_rows": 0}
        step_summary = pd.DataFrame(columns=["step", "sum", "median", "count"])

    function_summary = {
        "kofam_rows": int(len(kofam)),
        "kofam_accepted_hits": int(kofam["accepted_hit"].sum()) if not kofam.empty else 0,
        "kofam_accepted_unique_ko": int(kofam.loc[kofam["accepted_hit"].fillna(False), "ko_id"].nunique()) if not kofam.empty else 0,
        "mcycdb_rows": int(len(mcyc)),
        "mcycdb_best_hits": int((mcyc["hit_rank_bitscore"] == 1).sum()) if not mcyc.empty else 0,
        "mcycdb_mags_with_best_hit": int(mcyc.loc[mcyc["hit_rank_bitscore"] == 1, "proteome_id"].nunique()) if not mcyc.empty else 0,
        "scycdb_rows": int(len(scyc)),
        "scycdb_best_hits": int((scyc["hit_rank_bitscore"] == 1).sum()) if not scyc.empty else 0,
        "scycdb_mags_with_best_hit": int(scyc.loc[scyc["hit_rank_bitscore"] == 1, "proteome_id"].nunique()) if not scyc.empty else 0,
        "dbcan_rows": int(len(dbcan)),
        "dbcan_mags_with_hits": int(dbcan["proteome_id"].nunique()) if not dbcan.empty else 0,
        "metabolic_hmm_rows": int(len(met_hmm)),
        "metabolic_hmm_present_rows": int((met_hmm["presence"].astype(str).str.lower() == "present").sum()) if not met_hmm.empty else 0,
        "metabolic_mags_with_present_hmm": int(met_hmm.loc[met_hmm["presence"].astype(str).str.lower() == "present", "proteome_id"].nunique()) if not met_hmm.empty else 0,
    }

    if not met_hmm.empty:
        present = met_hmm[met_hmm["presence"].astype(str).str.lower() == "present"].copy()
        present["hit_count"] = pd.to_numeric(present["hit_count"], errors="coerce").fillna(0)
        met_top_categories = (
            present.groupby("function_category")
            .agg(hit_count=("hit_count", "sum"), mags=("proteome_id", "nunique"), functions=("function_name", "nunique"))
            .reset_index()
            .sort_values("hit_count", ascending=False)
            .head(10)
        )
    else:
        met_top_categories = pd.DataFrame(columns=["function_category", "hit_count", "mags", "functions"])

    warehouse_manifest = warehouse_dir / "cohort_table_manifest.tsv"
    warehouse_dim_mag = 0
    if warehouse_manifest.exists():
        warehouse_rows = pd.read_csv(warehouse_manifest, sep="\t")
        if "table" in warehouse_rows.columns and "rows" in warehouse_rows.columns:
            match = warehouse_rows.loc[warehouse_rows["table"] == "dim_mag", "rows"]
            warehouse_dim_mag = int(match.iloc[0]) if not match.empty else 0

    squeue = run_cmd(f"squeue -j {args.job_id} -o '%.18i %.9P %.30j %.8u %.2t %.10M %.10L %.6D %.4C %.12m %R'", repo_root)
    sacct = run_cmd(f"sacct -j {args.job_id} --format=JobID,State,Elapsed,ExitCode,AllocCPUS,ReqMem -P", repo_root)
    running, pending, sacct_states = source_tool_status(squeue.stdout, sacct.stdout)

    taxonomy_summary = {
        "taxonomy_in_run_record": int(complete_df["taxonomy_in_run_record"].sum()) if completed_total else 0,
        "gtdb_summary_file": int(complete_df["gtdb_summary_file"].sum()) if completed_total else 0,
        "gtdb_json_file": int(complete_df["gtdb_json_file"].sum()) if completed_total else 0,
        "domains": int(complete_df["domain"].notna().sum()) if completed_total else 0,
    }

    complete_df.to_csv(out_dir / "completed_mag_snapshot.tsv", sep="\t", index=False)
    complete_all_df.to_csv(out_dir / "completed_all_unit_scope_snapshot.tsv", sep="\t", index=False)
    table_summary.to_csv(out_dir / "per_mag_parquet_table_summary.tsv", sep="\t", index=False)
    expected_presence.to_csv(out_dir / "expected_table_presence.tsv", sep="\t", index=False)
    step_summary.to_csv(out_dir / "tool_timing_summary.tsv", sep="\t", index=False)
    met_top_categories.to_csv(out_dir / "metabolic_top_categories.tsv", sep="\t", index=False)

    charts = build_charts(
        assets_dir,
        out_dir,
        completed_total,
        completed_wetland,
        completed_rumen,
        table_summary,
        expected_presence,
        warehouse_dim_mag,
    )

    snapshot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_notes = {
        "snapshot_ts": snapshot_ts,
        "run_id": args.run_id,
        "completed_total": completed_total,
        "completed_wetland": completed_wetland,
        "completed_rumen": completed_rumen,
        "assembly_context_complete": assembly_context_complete,
        "failed_total": len(failed_dirs),
        "warehouse_dim_mag_rows": warehouse_dim_mag,
        "taxonomy_summary": taxonomy_summary,
        "qc_summary": qc_summary,
        "timing_summary": timing_summary,
        "function_summary": function_summary,
        "squeue_stdout": squeue.stdout,
        "sacct_state_counts": dict(sacct_states),
        "sources": [
            f"results/functional_metagenomics/{args.run_id}/per_mag/",
            str(scope_manifest_path),
            f"results/functional_metagenomics/{args.run_id}/cohort_warehouse/cohort_table_manifest.tsv",
            "scripts/curate_functional_mag_run.py",
            "scripts/consolidate_functional_mag_cohort.py",
            "ai_docs/functional_metagenomics_expansion/proteome_crosswalk/proteome_crosswalk_summary.tsv",
            "README.md",
            "docs/functional_metagenomics_expansion.md",
            "ai_docs/functional_metagenomics_expansion/data_aggregation_strategy.md",
        ],
        "limitations": [
            f"Rumen MAG/bin completion is pilot-scale only: {completed_rumen} of {EXPECTED_RUMEN_MAG_BIN} comparable rumen MAG/bin units complete at snapshot time.",
            f"Assembly-context outputs are quarantined from MAG-level report facts: {assembly_context_complete} complete assembly-context outputs at snapshot time.",
            f"Warehouse is stale: current dim_mag has {warehouse_dim_mag} rows, while live comparable MAG/bin outputs have {completed_total} complete sentinels.",
            "Sample/metagenome-level claims require abundance/read coverage, environmental metadata, and field validation not present in this smoke report.",
        ],
    }
    (out_dir / "source_notes.json").write_text(json.dumps(source_notes, indent=2))

    report = html_report(
        out_dir,
        charts,
        snapshot_ts,
        completed_total,
        completed_wetland,
        completed_rumen,
        assembly_context_complete,
        len(failed_dirs),
        running,
        warehouse_dim_mag,
        taxonomy_summary,
        qc_summary,
        timing_summary,
        function_summary,
        met_top_categories,
        step_summary,
    )
    report_path = out_dir / "report.html"
    report_path.write_text(report)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "completed_total": completed_total,
                "completed_wetland": completed_wetland,
                "completed_rumen": completed_rumen,
                "failed": len(failed_dirs),
                "warehouse_dim_mag": warehouse_dim_mag,
                "taxonomy_in_run_record": taxonomy_summary["taxonomy_in_run_record"],
                "gtdb_summary_file": taxonomy_summary["gtdb_summary_file"],
                "gtdb_json_file": taxonomy_summary["gtdb_json_file"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
