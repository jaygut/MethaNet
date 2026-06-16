#!/usr/bin/env python3
"""Build a MethaNet bridge-candidate functional signature atlas.

This figure explains why the top ESM-2 bridge candidates are interesting:
it combines latent bridge affinity, gLM2 native-minus-shuffled genomic
context, per-MAG functional signatures, taxonomy, and QC. It is deliberately
MAG/proteome-grain and does not assign sample-level methane risk.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_methanet_multiview_signal_map import (  # noqa: E402
    DEFAULT_ESM_DIR,
    DEFAULT_FUNCTIONAL_RUN_DIRS,
    DEFAULT_GLM_DIR,
    DEFAULT_MSM_MANIFEST,
    DEFAULT_POC_MANIFEST,
    build_joined_table,
    discover_functional_status,
    load_esm_artifacts,
    load_glm_features,
    load_manifest,
    resolve,
)


DEFAULT_OUTPUT_DIR = Path(
    "results/figures/methanet_bridge_candidate_signature_atlas_20260616"
)

METHANE_TERMS = [
    "methan",
    "methyl",
    "coenzyme m",
    "coenzyme b",
    "mcr",
    "mtr",
    "mta",
    "mtb",
    "mtt",
    "hdr",
    "mvh",
    "fwd",
    "ftr",
]

SULFUR_TERMS = [
    "sulfur",
    "sulphur",
    "sulfate",
    "sulfite",
    "sulfide",
    "thiosulfate",
    "sulfurtransferase",
    "dsr",
    "apr",
    "sat",
    "sox",
]

SUBSTRATE_TERMS = [
    "carbohydrate",
    "glycolysis",
    "gluconeogenesis",
    "acetate",
    "formate",
    "methyl",
    "carbon fixation",
    "co2",
    "one-carbon",
    "hydrogen",
]

ELECTRON_TERMS = [
    "hydrogenase",
    "ferredoxin",
    "dehydrogenase",
    "electron",
    "redox",
    "formate",
]

SIGNATURE_COLUMNS = [
    ("bridge_affinity_norm", "ESM-2\nbridge\naffinity"),
    ("glm_context_norm", "gLM2\nnative -\nshuffled"),
    ("methane_signature_norm", "Methane\ncycle\nsignal"),
    ("sulfur_signature_norm", "Sulfur\ncycle\nsignal"),
    ("substrate_signature_norm", "Substrate /\nCAZy\ncapacity"),
    ("broad_metabolism_norm", "Broad\nmetabolic\nsupport"),
    ("qc_signal_norm", "QC\nconfidence"),
]

READINESS_COLORS = {
    "triangulated_now": "#047857",
    "glm_only_wait_function": "#f59e0b",
    "function_only_wait_glm": "#7c3aed",
    "latent_only_pending": "#94a3b8",
}

DOMAIN_COLORS = {
    "Archaea": "#7c3aed",
    "Bacteria": "#2563eb",
    "Unknown": "#64748b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--esm-artifact-dir", type=Path, default=DEFAULT_ESM_DIR)
    parser.add_argument("--glm-integration-dir", type=Path, default=DEFAULT_GLM_DIR)
    parser.add_argument("--poc-manifest", type=Path, default=DEFAULT_POC_MANIFEST)
    parser.add_argument("--msm-manifest", type=Path, default=DEFAULT_MSM_MANIFEST)
    parser.add_argument("--functional-warehouse-dir", type=Path)
    parser.add_argument("--functional-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--snapshot-label",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Label printed on the figure/report.",
    )
    return parser.parse_args()


def short_id(proteome_id: str) -> str:
    label = (
        proteome_id.replace("rumen__", "r__")
        .replace("mucc__", "m__")
        .replace("_idba_bin.", "_bin.")
        .replace("_ASM249546v1_genomic", "")
        .replace("_genomic", "")
    )
    return label if len(label) <= 38 else label[:35] + "..."


def present_mask(df: pd.DataFrame, column: str = "presence") -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower().isin(["present", "yes", "true", "1"])


def count_term_rows(df: pd.DataFrame, columns: list[str], terms: list[str]) -> int:
    if df.empty:
        return 0
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return 0
    pattern = re.compile("|".join(re.escape(t) for t in terms), flags=re.IGNORECASE)
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    return int(text.str.contains(pattern, regex=True).sum())


def read_table(manifest: pd.DataFrame, table: str) -> pd.DataFrame:
    if manifest.empty or "table" not in manifest.columns or "path" not in manifest.columns:
        return pd.DataFrame()
    row = manifest[manifest["table"].eq(table)]
    if row.empty:
        return pd.DataFrame()
    path = Path(str(row.iloc[0]["path"]))
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def latest_complete_run(proteome_id: str, run_dirs: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in run_dirs:
        candidates.extend(root.glob(f"per_mag/{proteome_id}/*"))
    complete = [
        p
        for p in candidates
        if p.is_dir()
        and (p / "COMPLETE").exists()
        and (p / "curated/parquet_manifest.tsv").exists()
    ]
    if not complete:
        return None
    return sorted(complete, key=lambda p: p.stat().st_mtime)[-1]


def taxonomy_from_record(record: dict[str, Any], fallback_domain: str) -> dict[str, str]:
    tax = record.get("taxonomy") or {}
    return {
        "domain_tax": str(tax.get("domain") or fallback_domain or "Unknown").replace("d__", ""),
        "phylum": str(tax.get("phylum") or "").replace("p__", ""),
        "class": str(tax.get("class") or "").replace("c__", ""),
        "order": str(tax.get("order") or "").replace("o__", ""),
        "family": str(tax.get("family") or "").replace("f__", ""),
        "genus": str(tax.get("genus") or "").replace("g__", ""),
        "species": str(tax.get("species") or "").replace("s__", ""),
    }


def summarize_functional_run(
    proteome_id: str, run_dir: Path | None, fallback_domain: str
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "proteome_id": proteome_id,
        "functional_run_dir": str(run_dir) if run_dir else "",
        "functional_signature_status": "pending",
        "checkm2_completeness": np.nan,
        "checkm2_contamination": np.nan,
        "gunc_pass": np.nan,
        "protein_coding_genes": np.nan,
        "kofam_accepted_hits": np.nan,
        "mcycdb_best_hits": np.nan,
        "scycdb_best_hits": np.nan,
        "dbcan_hits": np.nan,
        "metabolic_present_functions": np.nan,
        "metabolic_present_modules": np.nan,
        "metabolic_hmm_present": np.nan,
        "methane_term_hits": np.nan,
        "sulfur_term_hits": np.nan,
        "substrate_term_hits": np.nan,
        "electron_term_hits": np.nan,
    }
    base.update(taxonomy_from_record({}, fallback_domain))
    if run_dir is None:
        return base

    manifest_path = run_dir / "curated/parquet_manifest.tsv"
    record_path = run_dir / "curated/run_record.json"
    if not manifest_path.exists():
        return base
    manifest = pd.read_csv(manifest_path, sep="\t")
    record: dict[str, Any] = {}
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
        except Exception:
            record = {}
    base.update(taxonomy_from_record(record, fallback_domain))
    base["functional_signature_status"] = "complete"

    checkm2 = read_table(manifest, "fact_qc_checkm2")
    if not checkm2.empty:
        base["checkm2_completeness"] = pd.to_numeric(
            checkm2.get("Completeness"), errors="coerce"
        ).max()
        base["checkm2_contamination"] = pd.to_numeric(
            checkm2.get("Contamination"), errors="coerce"
        ).max()
        base["protein_coding_genes"] = pd.to_numeric(
            checkm2.get("Total_Coding_Sequences"), errors="coerce"
        ).max()

    gunc = read_table(manifest, "fact_qc_gunc")
    if not gunc.empty and "pass.GUNC" in gunc.columns:
        base["gunc_pass"] = bool(gunc["pass.GUNC"].fillna(False).astype(bool).any())

    kofam = read_table(manifest, "fact_kofam_hits")
    if not kofam.empty:
        accepted = kofam.get("accepted_hit", False)
        accepted_mask = accepted.fillna(False).astype(bool) if hasattr(accepted, "fillna") else False
        accepted_df = kofam[accepted_mask] if hasattr(accepted_mask, "__len__") else kofam.iloc[0:0]
        base["kofam_accepted_hits"] = int(len(accepted_df))
        base["methane_term_hits"] = count_term_rows(
            accepted_df, ["ko_definition", "ko_id"], METHANE_TERMS
        )
        base["sulfur_term_hits"] = count_term_rows(
            accepted_df, ["ko_definition", "ko_id"], SULFUR_TERMS
        )
        base["substrate_term_hits"] = count_term_rows(
            accepted_df, ["ko_definition", "ko_id"], SUBSTRATE_TERMS
        )
        base["electron_term_hits"] = count_term_rows(
            accepted_df, ["ko_definition", "ko_id"], ELECTRON_TERMS
        )

    mcyc = read_table(manifest, "fact_mcycdb_hits")
    if not mcyc.empty:
        rank = pd.to_numeric(mcyc.get("hit_rank_bitscore"), errors="coerce")
        base["mcycdb_best_hits"] = int((rank == 1).sum())

    scyc = read_table(manifest, "fact_scycdb_hits")
    if not scyc.empty:
        rank = pd.to_numeric(scyc.get("hit_rank_bitscore"), errors="coerce")
        base["scycdb_best_hits"] = int((rank == 1).sum())

    dbcan = read_table(manifest, "fact_dbcan_hits")
    if not dbcan.empty:
        base["dbcan_hits"] = int(len(dbcan))

    functions = read_table(manifest, "fact_metabolic_function_presence")
    if not functions.empty:
        present = functions[present_mask(functions)]
        base["metabolic_present_functions"] = int(len(present))
        base["methane_term_hits"] = int(base["methane_term_hits"] or 0) + count_term_rows(
            present, ["function_category", "function_name", "gene_abbreviation"], METHANE_TERMS
        )
        base["sulfur_term_hits"] = int(base["sulfur_term_hits"] or 0) + count_term_rows(
            present, ["function_category", "function_name", "gene_abbreviation"], SULFUR_TERMS
        )
        base["substrate_term_hits"] = int(base["substrate_term_hits"] or 0) + count_term_rows(
            present, ["function_category", "function_name", "gene_abbreviation"], SUBSTRATE_TERMS
        )
        base["electron_term_hits"] = int(base["electron_term_hits"] or 0) + count_term_rows(
            present, ["function_category", "function_name", "gene_abbreviation"], ELECTRON_TERMS
        )

    modules = read_table(manifest, "fact_metabolic_module_presence")
    if not modules.empty:
        present = modules[present_mask(modules)]
        base["metabolic_present_modules"] = int(len(present))
        base["substrate_term_hits"] = int(base["substrate_term_hits"] or 0) + count_term_rows(
            present, ["module_id", "module_name", "module_category"], SUBSTRATE_TERMS
        )
        base["methane_term_hits"] = int(base["methane_term_hits"] or 0) + count_term_rows(
            present, ["module_id", "module_name", "module_category"], METHANE_TERMS
        )

    hmm = read_table(manifest, "fact_metabolic_hmm_hits")
    if not hmm.empty:
        hit_count = pd.to_numeric(hmm.get("hit_count"), errors="coerce").fillna(0)
        present = hmm[(present_mask(hmm)) | (hit_count > 0)]
        base["metabolic_hmm_present"] = int(len(present))
        base["methane_term_hits"] = int(base["methane_term_hits"] or 0) + count_term_rows(
            present,
            ["function_category", "function_name", "gene_abbreviation", "gene_name", "ko_id"],
            METHANE_TERMS,
        )
        base["sulfur_term_hits"] = int(base["sulfur_term_hits"] or 0) + count_term_rows(
            present,
            ["function_category", "function_name", "gene_abbreviation", "gene_name", "ko_id"],
            SULFUR_TERMS,
        )
        base["electron_term_hits"] = int(base["electron_term_hits"] or 0) + count_term_rows(
            present,
            ["function_category", "function_name", "gene_abbreviation", "gene_name", "ko_id"],
            ELECTRON_TERMS,
        )

    bakta = read_table(manifest, "fact_bakta_features")
    if not bakta.empty and math.isnan(float(base.get("protein_coding_genes", np.nan))):
        if "Type" in bakta.columns:
            base["protein_coding_genes"] = int(bakta["Type"].astype(str).str.lower().eq("cds").sum())
        else:
            base["protein_coding_genes"] = int(len(bakta))
    return base


def max_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty or float(finite.max()) <= 0:
        return values * np.nan
    return (values / float(finite.max())).clip(0, 1)


def log_scale(series: pd.Series) -> pd.Series:
    return max_scale(np.log1p(pd.to_numeric(series, errors="coerce")))


def build_signature_table(args: argparse.Namespace) -> pd.DataFrame:
    repo_root = args.repo_root.resolve()
    esm_dir = resolve(repo_root, args.esm_artifact_dir)
    glm_dir = resolve(repo_root, args.glm_integration_dir)
    poc_manifest = resolve(repo_root, args.poc_manifest)
    warehouse_dir = resolve(repo_root, args.functional_warehouse_dir)
    run_dirs = [
        resolve(repo_root, p)
        for p in (args.functional_run_dir or DEFAULT_FUNCTIONAL_RUN_DIRS)
    ]
    run_dirs = [p for p in run_dirs if p is not None]

    projection, _, bridge = load_esm_artifacts(esm_dir)
    glm = load_glm_features(glm_dir)
    manifest = load_manifest(poc_manifest)
    functional = discover_functional_status(run_dirs)
    joined = build_joined_table(
        projection=projection,
        bridge=bridge,
        glm=glm,
        manifest=manifest,
        functional=functional,
        warehouse_dir=warehouse_dir,
    )
    top = joined[joined["is_top_bridge_candidate"]].sort_values("rank").head(args.top_n).copy()
    if top.empty:
        raise SystemExit("No top bridge candidates were found in the ESM-2 artifacts.")

    summaries = []
    for _, row in top.iterrows():
        pid = str(row["proteome_id"])
        run_dir = latest_complete_run(pid, run_dirs)
        summaries.append(summarize_functional_run(pid, run_dir, str(row.get("domain") or "Unknown")))
    summary = pd.DataFrame(summaries)
    out = top.merge(summary, on="proteome_id", how="left")

    out["display_id"] = out["proteome_id"].astype(str).map(short_id)
    out["taxonomy_label"] = out.apply(
        lambda r: (
            str(r.get("genus") or r.get("family") or r.get("order") or r.get("domain_tax") or r.get("domain") or "Unknown")
        ),
        axis=1,
    )
    out["taxonomy_label"] = out["taxonomy_label"].replace({"nan": "Unknown", "": "Unknown"})

    out["methane_signature_raw"] = (
        pd.to_numeric(out["mcycdb_best_hits"], errors="coerce").fillna(0)
        + 4 * pd.to_numeric(out["methane_term_hits"], errors="coerce").fillna(0)
    )
    out["sulfur_signature_raw"] = (
        pd.to_numeric(out["scycdb_best_hits"], errors="coerce").fillna(0)
        + 3 * pd.to_numeric(out["sulfur_term_hits"], errors="coerce").fillna(0)
    )
    out["substrate_signature_raw"] = (
        pd.to_numeric(out["dbcan_hits"], errors="coerce").fillna(0)
        + 2 * pd.to_numeric(out["substrate_term_hits"], errors="coerce").fillna(0)
    )
    out["broad_metabolism_raw"] = (
        pd.to_numeric(out["kofam_accepted_hits"], errors="coerce").fillna(0)
        + pd.to_numeric(out["metabolic_present_functions"], errors="coerce").fillna(0)
        + pd.to_numeric(out["metabolic_present_modules"], errors="coerce").fillna(0)
        + pd.to_numeric(out["metabolic_hmm_present"], errors="coerce").fillna(0)
    )
    completeness = pd.to_numeric(out["checkm2_completeness"], errors="coerce")
    contamination = pd.to_numeric(out["checkm2_contamination"], errors="coerce")
    gunc_bonus = out["gunc_pass"].fillna(False).astype(bool).astype(float) * 0.08
    out["qc_signal_raw"] = ((completeness / 100.0) - (contamination / 10.0) + gunc_bonus).clip(0, 1)

    out["bridge_affinity_norm"] = max_scale(out["mixing_coeff"].fillna(0))
    delta = pd.to_numeric(out["glm_context_delta"], errors="coerce")
    out["glm_context_norm"] = max_scale(delta.clip(lower=0))
    out["methane_signature_norm"] = log_scale(out["methane_signature_raw"])
    out["sulfur_signature_norm"] = log_scale(out["sulfur_signature_raw"])
    out["substrate_signature_norm"] = log_scale(out["substrate_signature_raw"])
    out["broad_metabolism_norm"] = log_scale(out["broad_metabolism_raw"])
    out["qc_signal_norm"] = out["qc_signal_raw"]

    def archetype(row: pd.Series) -> str:
        domain = str(row.get("domain") or row.get("domain_tax") or "")
        eco = str(row.get("ecosystem") or "").lower()
        if not bool(row.get("has_glm")) and not bool(row.get("has_functional")):
            return "latent-priority gap: needs gLM2 + functional evidence"
        if not bool(row.get("has_glm")):
            return "functional evidence present; gLM2 context pending"
        if not bool(row.get("has_functional")):
            return "gLM2 context present; functional evidence pending"
        if "wetland" in eco:
            return "target-domain wetland anchor with context support"
        if "bacteria" in domain.lower():
            return "bacterial bridge-like comparator"
        return "context-supported archaeal bridge"

    out["bridge_archetype"] = out.apply(archetype, axis=1)
    out["claim_boundary"] = (
        "MAG/proteome bridge-candidate mechanism screen; not sample-level risk or measured methane flux."
    )
    return out


def draw_status_chip(
    ax: plt.Axes,
    x: float,
    y: float,
    ok: bool,
    label: str,
    *,
    width: float = 0.085,
    height: float = 0.28,
) -> None:
    color = "#047857" if ok else "#e2e8f0"
    edge = "#065f46" if ok else "#cbd5e1"
    text_color = "#ffffff" if ok else "#475569"
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            facecolor=color,
            edgecolor=edge,
            lw=0.9,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=6.7, color=text_color, weight="bold")


def make_figure(table: pd.DataFrame, output_png: Path, output_pdf: Path, snapshot_label: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
        }
    )
    n = len(table)
    fig = plt.figure(figsize=(24, 14.6), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        16,
        height_ratios=[0.78, 7.2, 2.25],
        hspace=0.34,
        wspace=0.68,
    )
    title_ax = fig.add_subplot(gs[0, :])
    meta_ax = fig.add_subplot(gs[1, :4])
    heat_ax = fig.add_subplot(gs[1, 4:10])
    delta_ax = fig.add_subplot(gs[1, 10:12])
    status_ax = fig.add_subplot(gs[1, 12:])
    story_ax = fig.add_subplot(gs[2, :])

    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.78,
        "MethaNet Bridge Candidate Functional Signature Atlas",
        fontsize=24,
        weight="bold",
        color="#061b3a",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.0,
        0.24,
        textwrap.fill(
            "Why the top bridge candidates behave as bridges: latent ESM-2 affinity aligned with gLM2 genomic context, methane/sulfur/substrate machinery, broad metabolic support, taxonomy, and QC.",
            width=135,
        ),
        fontsize=11.8,
        color="#0f766e",
        style="italic",
        ha="left",
        va="center",
    )
    title_ax.text(1.0, 0.52, f"snapshot: {snapshot_label}", ha="right", va="center", fontsize=10, color="#475569")

    ordered = table.sort_values("rank").reset_index(drop=True)
    y_positions = np.arange(n)
    row_colors = [
        READINESS_COLORS.get(str(x), "#94a3b8")
        for x in ordered["readiness_class"].fillna("latent_only_pending")
    ]

    meta_ax.set_xlim(0, 1)
    meta_ax.set_ylim(n - 0.5, -0.5)
    meta_ax.axis("off")
    meta_ax.set_title("A. Candidate identity and bridge archetype", loc="left", fontsize=12, weight="bold", pad=10)
    for i, row in ordered.iterrows():
        meta_ax.add_patch(
            Rectangle((0.0, i - 0.42), 1.0, 0.84, facecolor="#f8fafc", edgecolor="#e2e8f0", lw=0.8)
        )
        meta_ax.add_patch(Rectangle((0.0, i - 0.42), 0.018, 0.84, facecolor=row_colors[i], edgecolor="none"))
        domain = str(row.get("domain") or row.get("domain_tax") or "Unknown")
        rank_box = FancyBboxPatch(
            (0.035, i - 0.225),
            0.105,
            0.45,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            facecolor=row_colors[i],
            edgecolor="#ffffff",
            lw=1.2,
        )
        meta_ax.add_patch(rank_box)
        meta_ax.text(
            0.0875,
            i,
            f"#{int(row['rank'])}",
            ha="center",
            va="center",
            fontsize=8.8,
            color="#ffffff",
            weight="bold",
        )
        meta_ax.add_patch(
            Circle((0.165, i), 0.047, facecolor=DOMAIN_COLORS.get(domain, "#64748b"), edgecolor="#ffffff", lw=1)
        )
        meta_ax.text(0.205, i - 0.15, row["display_id"], ha="left", va="center", fontsize=8.5, color="#0f172a", weight="bold")
        tax = str(row.get("taxonomy_label") or "Unknown")
        arch = str(row.get("bridge_archetype") or "")
        meta_ax.text(0.205, i + 0.04, textwrap.shorten(tax, width=38, placeholder="..."), ha="left", va="center", fontsize=7.4, color="#334155")
        meta_ax.text(0.205, i + 0.23, textwrap.shorten(arch, width=55, placeholder="..."), ha="left", va="center", fontsize=7.0, color="#475569")

    heat = ordered[[c for c, _ in SIGNATURE_COLUMNS]].to_numpy(dtype=float)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "methanet_bridge",
        ["#f8fafc", "#dbeafe", "#5eead4", "#0f766e", "#062f4f"],
    )
    masked = np.ma.masked_invalid(heat)
    cmap.set_bad("#e5e7eb")
    im = heat_ax.imshow(masked, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    heat_ax.set_title("B. Normalized signature intensity", loc="left", fontsize=12, weight="bold", pad=10)
    heat_ax.set_xticks(np.arange(len(SIGNATURE_COLUMNS)), [label for _, label in SIGNATURE_COLUMNS], fontsize=8.2)
    heat_ax.set_yticks(y_positions, [""] * n)
    heat_ax.tick_params(axis="x", length=0, pad=8)
    heat_ax.tick_params(axis="y", length=0)
    for i in range(n):
        for j in range(len(SIGNATURE_COLUMNS)):
            val = heat[i, j]
            if np.isfinite(val):
                text_color = "#ffffff" if val > 0.58 else "#0f172a"
                heat_ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.3, color=text_color, weight="bold")
            else:
                heat_ax.text(j, i, "NA", ha="center", va="center", fontsize=7.0, color="#64748b")
    heat_ax.set_xticks(np.arange(-0.5, len(SIGNATURE_COLUMNS), 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    heat_ax.grid(which="minor", color="#ffffff", linewidth=1.5)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=heat_ax, fraction=0.025, pad=0.025)
    cbar.set_label("relative intensity within top candidates", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    delta = pd.to_numeric(ordered["glm_context_delta"], errors="coerce")
    y = np.arange(n)
    delta_ax.barh(
        y,
        delta.fillna(0),
        color=np.where(delta.notna(), "#0f766e", "#cbd5e1"),
        edgecolor="#ffffff",
        height=0.62,
    )
    delta_ax.set_ylim(n - 0.5, -0.5)
    xmax = max(1.0, float(delta.max(skipna=True) or 1.0) * 1.18)
    delta_ax.set_xlim(0, xmax)
    delta_ax.set_yticks([])
    delta_ax.set_xlabel("native - shuffled", fontsize=8.6)
    delta_ax.set_title("C. gLM2 context\norder sensitivity", loc="left", fontsize=12, weight="bold", pad=10)
    delta_ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    for i, val in enumerate(delta):
        label = "pending" if not np.isfinite(val) else f"{val:.1f}"
        delta_ax.text(
            (0.04 * xmax) if not np.isfinite(val) else min(float(val) + 0.25, xmax * 0.94),
            i,
            label,
            va="center",
            ha="left",
            fontsize=7.8,
            color="#334155",
        )

    status_ax.set_xlim(0, 1)
    status_ax.set_ylim(n - 0.5, -0.95)
    status_ax.axis("off")
    status_ax.set_title("D. Evidence audit: layer status and raw counts", loc="left", fontsize=12, weight="bold", pad=10)
    for x, label in [(0.055, "ESM"), (0.155, "gLM"), (0.255, "FUN"), (0.355, "QC")]:
        status_ax.text(x, -0.66, label, ha="center", va="center", fontsize=7.2, color="#475569", weight="bold")
    status_ax.text(0.49, -0.66, "raw evidence snapshot", ha="left", va="center", fontsize=7.2, color="#475569", weight="bold")
    for i, row in ordered.iterrows():
        status_ax.add_patch(Rectangle((0.0, i - 0.42), 1.0, 0.84, facecolor="#ffffff", edgecolor="#e2e8f0", lw=0.8))
        draw_status_chip(status_ax, 0.055, i, True, "E")
        draw_status_chip(status_ax, 0.155, i, bool(row.get("has_glm")), "G")
        draw_status_chip(status_ax, 0.255, i, bool(row.get("has_functional")), "F")
        qc_ok = bool(
            pd.notna(row.get("checkm2_completeness"))
            and float(row.get("checkm2_completeness")) >= 80
            and (
                pd.isna(row.get("checkm2_contamination"))
                or float(row.get("checkm2_contamination")) <= 5
            )
        )
        draw_status_chip(status_ax, 0.355, i, qc_ok, "Q")
        comp = row.get("checkm2_completeness")
        cont = row.get("checkm2_contamination")
        genes = row.get("protein_coding_genes")
        raw = (
            f"QC {float(comp):.0f}/{float(cont):.1f}  "
            if pd.notna(comp) and pd.notna(cont)
            else "QC pending  "
        )
        raw += f"CDS {int(genes):,}" if pd.notna(genes) else "CDS pending"
        status_ax.text(0.49, i - 0.11, raw, ha="left", va="center", fontsize=7.6, color="#0f172a", weight="bold")
        raw2 = (
            f"MCyc {int(row['mcycdb_best_hits']) if pd.notna(row.get('mcycdb_best_hits')) else 0} | "
            f"SCyc {int(row['scycdb_best_hits']) if pd.notna(row.get('scycdb_best_hits')) else 0} | "
            f"KO+ {int(row['kofam_accepted_hits']) if pd.notna(row.get('kofam_accepted_hits')) else 0}"
        )
        status_ax.text(0.49, i + 0.13, raw2, ha="left", va="center", fontsize=7.2, color="#475569")

    story_ax.axis("off")
    story_ax.set_xlim(0, 1)
    story_ax.set_ylim(0, 1)
    story_ax.set_title("Strategic interpretation: what makes a top bridge candidate distinctive", loc="left", fontsize=12, weight="bold", pad=8)
    cards = [
        (
            "1. Geometry finds transfer candidates",
            "Top bridges sit at the rumen-wetland boundary in ESM-2 proteome space, so they are not generic MAGs: they are latent candidates for mechanism transfer or comparator review.",
            "#eff6ff",
            "#93c5fd",
        ),
        (
            "2. gLM2 asks whether gene order supports the signal",
            "Positive native-minus-shuffled context means the candidate's ordered genomic neighborhood carries information beyond a bag of genes. This is the strongest clue that architecture, not only annotation count, matters.",
            "#ecfdf5",
            "#6ee7b7",
        ),
        (
            "3. Functional tables explain the mechanism",
            "MCycDB, SCycDB, KOfam, METABOLIC, dbCAN, taxonomy, and QC separate context-supported archaeal bridges, bacterial comparators, wetland anchors, and pending-evidence gaps.",
            "#fff7ed",
            "#fdba74",
        ),
    ]
    for idx, (heading, body, fc, ec) in enumerate(cards):
        x0 = 0.015 + idx * 0.327
        box = FancyBboxPatch(
            (x0, 0.22),
            0.305,
            0.62,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor=fc,
            edgecolor=ec,
            lw=1.2,
        )
        story_ax.add_patch(box)
        story_ax.text(x0 + 0.018, 0.73, heading, ha="left", va="top", fontsize=10.2, weight="bold", color="#0f172a")
        story_ax.text(x0 + 0.018, 0.62, textwrap.fill(body, width=44), ha="left", va="top", fontsize=8.5, color="#334155", linespacing=1.25)
    story_ax.text(
        0.5,
        0.06,
        "Claim boundary: this is MAG/proteome molecular screening for candidate cards and validation planning; it is not sample-level methane flux, final MRV tiering, or carbon-credit approval.",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#7c2d12",
        bbox=dict(boxstyle="round,pad=0.40", fc="#fff7ed", ec="#fdba74", lw=1),
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, args: argparse.Namespace, table: pd.DataFrame) -> None:
    summary = {
        "top_bridge_candidates": int(len(table)),
        "with_glm2": int(table["has_glm"].fillna(False).astype(bool).sum()),
        "with_functional": int(table["has_functional"].fillna(False).astype(bool).sum()),
        "with_all_three_layers": int(
            (
                table["has_esm2"].fillna(False).astype(bool)
                & table["has_glm"].fillna(False).astype(bool)
                & table["has_functional"].fillna(False).astype(bool)
            ).sum()
        ),
        "archetypes": table["bridge_archetype"].value_counts(dropna=False).to_dict(),
        "snapshot_label": args.snapshot_label,
    }
    readme = f"""# MethaNet Bridge Candidate Functional Signature Atlas

Generated: {args.snapshot_label}

This artifact explains the top ESM-2 bridge candidates by joining latent
geometry, gLM2 native-minus-shuffled genomic-context signal, per-MAG
functional signatures, taxonomy, and QC. It is designed to support bridge
candidate cards and validation planning.

## Files

- `bridge_candidate_signature_atlas.png`: high-resolution visual atlas.
- `bridge_candidate_signature_atlas.pdf`: vector-backed report figure.
- `bridge_candidate_signature_table.tsv`: auditable table keyed by `proteome_id`.
- `ARTIFACT_REVIEW.md`: brief QA notes and interpretation.

## Reproducible Command

Run from the MethaNet repo root:

```bash
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
conda activate methanet-fgx
python scripts/reports/build_bridge_candidate_signature_atlas.py \\
  --output-dir {args.output_dir} \\
  --snapshot-label {args.snapshot_label}
```

## Summary

```json
{json.dumps(summary, indent=2)}
```

## Claim Boundary

This is MAG/proteome-level molecular screening. It supports prioritization,
mechanism review, and partner-demo evidence. It does not support final
sample-level methane-risk tiers, measured methane flux claims, carbon-credit
approval, or registry-grade MRV without abundance, environmental context,
uncertainty propagation, and external validation.
"""
    (output_dir / "README.md").write_text(readme)
    review = f"""# Artifact Review

## What Is Actionable

- The atlas separates top bridge candidates into explicit archetypes rather
  than treating them as a single ranked list.
- Panel A now uses high-contrast rank pills so candidate numbers remain visible
  at report scale.
- Panel D now uses a table-style audit lane with ESM/gLM/functional/QC status
  chips plus raw QC, CDS, MCycDB, SCycDB, and KOfam accepted-hit counts.
- Candidates with ESM-2 bridge affinity, positive gLM2 native-minus-shuffled
  signal, complete functional outputs, and high QC are ready for bridge
  mechanism cards.
- Pending rows remain visible as evidence gaps, preserving the audit trail.

## What Still Needs Care

- MCycDB and SCycDB best-hit counts are screening signals, not process-rate
  measurements.
- The gLM2 signal is a contextual feature layer; it complements ESM-2 and
  annotations, and does not replace either.
- Sample-level methane permanence risk still requires MAG abundance/read
  coverage, sample metadata, environmental covariates, uncertainty propagation,
  and flux/process validation.

## Current Counts

```json
{json.dumps({
    "rows": int(len(table)),
    "complete_functional_rows": int(table["has_functional"].fillna(False).astype(bool).sum()),
    "glm2_rows": int(table["has_glm"].fillna(False).astype(bool).sum()),
}, indent=2)}
```
"""
    (output_dir / "ARTIFACT_REVIEW.md").write_text(review)


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.repo_root.resolve(), args.output_dir)
    table = build_signature_table(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "bridge_candidate_signature_table.tsv", sep="\t", index=False)
    make_figure(
        table,
        output_png=output_dir / "bridge_candidate_signature_atlas.png",
        output_pdf=output_dir / "bridge_candidate_signature_atlas.pdf",
        snapshot_label=args.snapshot_label,
    )
    write_readme(output_dir, args, table)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
