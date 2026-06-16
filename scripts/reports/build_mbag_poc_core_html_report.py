#!/usr/bin/env python3
"""Build a self-contained MBAG POC core HTML intelligence report.

The report integrates the closed 625-MAG POC core:

- ESM-2 proteome geometry and bridge artifacts.
- Functional cohort warehouse facts/features.
- gLM2 contextual-genomics features.

It deliberately keeps the artifact at MAG/proteome grain. It does not assign
sample-level methane-risk tiers, measured methane flux, or crediting decisions.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ESM_DIR = Path(
    "results/blue_catalyst_poc/runs/"
    "apolo_full_20260228_080644_embed_20260305_061952/artifacts"
)
DEFAULT_WAREHOUSE_DIR = Path(
    "results/functional_metagenomics/fgx_662_apollo3_20260612/"
    "cohort_warehouse_poc_magbin_union_20260616_075022"
)
DEFAULT_GLM_DIR = Path(
    "results/contextual_genomics/"
    "glm2_integration_20260616_poc_catchup_20260616_073441"
)
DEFAULT_SIGNAL_DIR = Path(
    "results/figures/methanet_multiview_signal_map_20260616_poc_closed_v2"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/reports/mbag_poc_core_molecular_intelligence_20260616"
)

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

COLORS = {
    "blue": {
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "gold": {
        "xlight": "#FFF4C2",
        "light": "#FFEA8F",
        "base": "#FFE15B",
        "mid": "#B8A037",
        "dark": "#736422",
    },
    "orange": {
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
    "pink": {
        "xlight": "#FCDAD6",
        "light": "#F5BACC",
        "base": "#F390CA",
        "mid": "#BD569B",
        "dark": "#8A3A6F",
    },
}

SOURCE_COLORS = {
    "rumen": COLORS["orange"]["base"],
    "wetland": COLORS["blue"]["base"],
    "quarantined": NEUTRAL["base"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--esm-dir", type=Path, default=DEFAULT_ESM_DIR)
    parser.add_argument("--warehouse-dir", type=Path, default=DEFAULT_WAREHOUSE_DIR)
    parser.add_argument("--glm-dir", type=Path, default=DEFAULT_GLM_DIR)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--snapshot-label",
        default=datetime.now().strftime("%B %d, %Y"),
        help="Human-readable report date label.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
    *,
    title_width: int = 86,
    subtitle_width: int = 118,
    top: float = 0.82,
) -> None:
    title = textwrap.fill(title.strip(), width=title_width, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=subtitle_width, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(top=max(0.58, top - 0.038 * (title_lines - 1) - 0.028 * (subtitle_lines - 1)))
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(
        left,
        0.925 - 0.042 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9.3,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    sns.despine(ax=ax)


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def fmt_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{int(round(float(value))):,}"


def fmt_pct(value: Any, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def short_id(proteome_id: str, n: int = 34) -> str:
    out = (
        str(proteome_id)
        .replace("rumen__", "r__")
        .replace("mucc__", "m__")
        .replace("_idba_bin.", "_bin.")
        .replace("_ASM249546v1_genomic", "")
        .replace("_genomic", "")
    )
    return out if len(out) <= n else out[: n - 3] + "..."


def clean_tax(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if text in {"", "nan", "None"}:
        return "Unresolved"
    for prefix in ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]:
        text = text.replace(prefix, "")
    return text or "Unresolved"


def read_warehouse_table(warehouse_dir: Path, table: str) -> pd.DataFrame:
    manifest = pd.read_csv(warehouse_dir / "cohort_table_manifest.tsv", sep="\t")
    match = manifest.loc[manifest["table"].eq(table)]
    if match.empty:
        return pd.DataFrame()
    return pd.read_parquet(Path(match.iloc[0]["path"]))


def norm01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return pd.Series(np.where(np.isfinite(values), 0.5, np.nan), index=series.index)
    return ((values - lo) / (hi - lo)).clip(0, 1)


def log_norm(series: pd.Series) -> pd.Series:
    return norm01(np.log1p(pd.to_numeric(series, errors="coerce").clip(lower=0)))


def classify_qc(row: pd.Series) -> str:
    completeness = float(row.get("checkm2_completeness") or 0.0)
    contamination = float(row.get("checkm2_contamination") or 999.0)
    gunc_pass = bool(row.get("gunc_pass"))
    if completeness >= 90 and contamination <= 5 and gunc_pass:
        return "high_qc"
    if completeness >= 70 and contamination <= 10 and gunc_pass:
        return "reviewable_qc"
    if completeness >= 50 and contamination <= 10:
        return "caution_qc"
    return "qc_limited"


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "pass"})


def classify_bridge_card(row: pd.Series) -> str:
    qc = row.get("qc_tier_report")
    context = float(row.get("glm_context_delta") or 0.0)
    methane = float(row.get("methane_evidence_score") or 0.0)
    if qc in {"high_qc", "reviewable_qc"} and context > 0 and methane > 0:
        return "review-ready bridge card"
    if context > 0 and methane > 0:
        return "promising with QC caveat"
    if methane > 0:
        return "functional support; context review needed"
    return "hypothesis only"


def build_citations() -> pd.DataFrame:
    rows = [
        {
            "id": "esm2",
            "label": "ESM / ESM2",
            "citation": "Lin et al., Science 2023, evolutionary-scale structure prediction with protein language models.",
            "url": "https://www.science.org/doi/10.1126/science.ade2574",
            "report_use": "Protein/proteome latent-geometry layer.",
        },
        {
            "id": "glm",
            "label": "gLM contextual genomics",
            "citation": "Hwang et al., Nature Communications 2024, contextualized protein embeddings from metagenomic gene neighborhoods.",
            "url": "https://www.nature.com/articles/s41467-024-46947-9",
            "report_use": "Gene-neighborhood context and native-vs-shuffled control framing.",
        },
        {
            "id": "glm2",
            "label": "gLM2 / OMG",
            "citation": "TattaBio gLM2 model family and OMG/open metagenomic language modeling resources.",
            "url": "https://huggingface.co/tattabio/gLM2_650M",
            "report_use": "Contextual-genomics implementation used by the local POC payload.",
        },
        {
            "id": "mcycdb",
            "label": "MCycDB",
            "citation": "Zheng et al., Molecular Ecology Resources 2022, curated methane cycling gene database.",
            "url": "https://github.com/qichao1984/MCycDB",
            "report_use": "Methane-cycle marker evidence.",
        },
        {
            "id": "scycdb",
            "label": "SCycDB",
            "citation": "SCycDB sulfur cycling gene database and profiling resources.",
            "url": "https://github.com/qichao1984/SCycDB",
            "report_use": "Sulfur-cycle competition/context evidence.",
        },
        {
            "id": "dbcan",
            "label": "dbCAN",
            "citation": "Zhang et al., Nucleic Acids Research 2023, dbCAN3 for CAZyme and substrate annotation.",
            "url": "https://academic.oup.com/nar/article/51/W1/W115/7161199",
            "report_use": "CAZy and substrate-processing capacity.",
        },
        {
            "id": "kofam",
            "label": "KOfam / KofamKOALA",
            "citation": "Aramaki et al., Bioinformatics 2020, KOfam HMM profiles and adaptive thresholds.",
            "url": "https://academic.oup.com/bioinformatics/article/36/7/2251/5631907",
            "report_use": "Broad KO functional evidence and accepted-hit coverage.",
        },
        {
            "id": "metabolic",
            "label": "METABOLIC",
            "citation": "Zhou et al., Microbiome 2022, METABOLIC for microbial genome-scale metabolic and biogeochemical traits.",
            "url": "https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01213-8",
            "report_use": "METABOLIC function, module, HMM, CAZy, and MEROPS long-form facts.",
        },
        {
            "id": "checkm2",
            "label": "CheckM2",
            "citation": "Chklovski et al., Nature Methods 2023, machine-learning genome quality assessment.",
            "url": "https://www.nature.com/articles/s41592-023-01940-w",
            "report_use": "MAG completeness and contamination confidence.",
        },
        {
            "id": "gunc",
            "label": "GUNC",
            "citation": "Orakov et al., Nature Microbiology 2021, detection of genome chimerism and contamination using lineage consistency.",
            "url": "https://www.nature.com/articles/s41564-021-00929-7",
            "report_use": "Genome contamination/chimerism consistency flag.",
        },
        {
            "id": "gtdbtk",
            "label": "GTDB-Tk",
            "citation": "Chaumeil et al., Bioinformatics 2022, GTDB-Tk classification workflow.",
            "url": "https://academic.oup.com/bioinformatics/article/38/23/5315/6758240",
            "report_use": "Taxonomic context and source-aware interpretation.",
        },
        {
            "id": "bakta",
            "label": "Bakta",
            "citation": "Schwengers et al., Microbial Genomics 2021, rapid standardized bacterial genome annotation.",
            "url": "https://www.microbiologyresearch.org/content/journal/mgen/10.1099/mgen.0.000685",
            "report_use": "Gene/feature calls and protein-count context.",
        },
        {
            "id": "icvcm",
            "label": "ICVCM Core Carbon Principles",
            "citation": "Integrity Council for the Voluntary Carbon Market, Core Carbon Principles and Assessment Framework.",
            "url": "https://icvcm.org/core-carbon-principles/",
            "report_use": "Carbon-market claim boundary and need for verifiable impact.",
        },
        {
            "id": "verra_vm0033",
            "label": "Verra VM0033",
            "citation": "Verra VM0033 methodology for tidal wetland and seagrass restoration.",
            "url": "https://verra.org/methodologies/vm0033-methodology-for-tidal-wetland-and-seagrass-restoration-v2-1/",
            "report_use": "Blue-carbon MRV context; MethaNet supports screening, not credit approval.",
        },
        {
            "id": "ipcc_wetlands",
            "label": "IPCC Wetlands Supplement",
            "citation": "IPCC 2013 Supplement to the 2006 Guidelines for National Greenhouse Gas Inventories: Wetlands.",
            "url": "https://www.ipcc-nggip.iges.or.jp/public/wetlands/",
            "report_use": "Wetlands GHG accounting context and need for environmental/process evidence.",
        },
    ]
    return pd.DataFrame(rows)


def load_sources(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    esm_dir = resolve(repo_root, args.esm_dir)
    warehouse_dir = resolve(repo_root, args.warehouse_dir)
    glm_dir = resolve(repo_root, args.glm_dir)
    signal_dir = resolve(repo_root, args.signal_dir)

    joined = pd.read_csv(signal_dir / "joined_multiview_signal_table.tsv", sep="\t")
    top_bridge = pd.read_csv(signal_dir / "top_bridge_multiview_candidates.tsv", sep="\t")
    msm_status = pd.read_csv(signal_dir / "msm_mangrove_payload_status.tsv", sep="\t")
    readiness_summary = json.loads((signal_dir / "readiness_summary.json").read_text())
    glm_summary = json.loads((glm_dir / "validation_summary.json").read_text())
    warehouse_manifest = pd.read_csv(warehouse_dir / "cohort_table_manifest.tsv", sep="\t")
    validation_gates = pd.read_csv(warehouse_dir / "validation_gates.tsv", sep="\t")

    dim_mag = read_warehouse_table(warehouse_dir, "dim_mag")
    feature_mrv = read_warehouse_table(warehouse_dir, "feature_mrv_mag_level")
    methane = read_warehouse_table(warehouse_dir, "feature_methane_mechanism")
    sulfur = read_warehouse_table(warehouse_dir, "feature_sulfur_competition")
    coverage = read_warehouse_table(warehouse_dir, "feature_annotation_coverage")
    taxonomy = read_warehouse_table(warehouse_dir, "fact_taxonomy_gtdbtk")

    poc = joined[joined["readiness_class"].eq("triangulated_now")].copy()
    poc = poc.merge(
        dim_mag[
            [
                "proteome_id",
                "mag_id",
                "domain",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
                "prodigal_proteins",
                "qc_tier",
            ]
        ].rename(columns={"domain": "taxonomy_domain"}),
        on="proteome_id",
        how="left",
    )
    poc = poc.rename(columns={"mag_id_x": "mag_id_manifest", "mag_id_y": "mag_id_glm"})
    poc["tax_domain_clean"] = poc["taxonomy_domain"].map(clean_tax)
    poc["tax_family_clean"] = poc["family"].map(clean_tax)
    poc["tax_genus_clean"] = poc["genus"].map(clean_tax)
    poc["display_id"] = poc["proteome_id"].map(short_id)
    poc["qc_tier_report"] = poc.apply(classify_qc, axis=1)
    poc["bridge_card_status"] = poc.apply(classify_bridge_card, axis=1)
    poc["source_label"] = np.where(poc["source"].eq("mucc"), "Wetland/MUCC", "Rumen")
    poc["is_top_bridge_candidate"] = poc["is_top_bridge_candidate"].fillna(False).astype(bool)
    poc["bridge_rank_label"] = np.where(
        poc["is_top_bridge_candidate"],
        "#" + pd.to_numeric(poc["rank"], errors="coerce").astype("Int64").astype(str),
        "",
    )

    # Report-scale normalized signals.
    poc["esm_bridge_affinity_norm"] = norm01(pd.to_numeric(poc["mixing_coeff"], errors="coerce").fillna(0))
    poc["glm_context_norm"] = norm01(pd.to_numeric(poc["glm_context_delta"], errors="coerce").clip(lower=0))
    poc["methane_signal_norm"] = log_norm(poc["methane_evidence_score"])
    poc["sulfur_signal_norm"] = log_norm(poc["sulfur_competition_score"])
    poc["substrate_signal_norm"] = log_norm(poc["cazy_family_count"].fillna(0) + poc["metabolic_modules_present"].fillna(0))
    poc["broad_function_norm"] = log_norm(
        poc["metabolic_modules_present"].fillna(0)
        + poc["cazy_family_count"].fillna(0)
        + poc["merops_family_count"].fillna(0)
        + poc["kofam_annotated_gene_fraction"].fillna(0) * 100
    )
    completeness = pd.to_numeric(poc["checkm2_completeness"], errors="coerce").fillna(0) / 100
    contamination = pd.to_numeric(poc["checkm2_contamination"], errors="coerce").fillna(20) / 20
    gunc_bool = bool_series(poc["gunc_pass"])
    poc["qc_signal_norm"] = (completeness - contamination + gunc_bool.astype(float) * 0.1).clip(0, 1)

    top = poc[poc["is_top_bridge_candidate"]].sort_values("rank").head(args.top_n).copy()
    top["allowed_claim_wording"] = top.apply(
        lambda row: (
            "Review-ready MAG/proteome bridge candidate with ESM-2, gLM2, functional, QC, and taxonomy evidence; "
            "not a sample-level methane-risk score."
            if row["bridge_card_status"] == "review-ready bridge card"
            else "Bridge candidate should be reviewed with explicit QC/context caveats before stronger interpretation."
        ),
        axis=1,
    )
    top["next_validation_action"] = top.apply(
        lambda row: (
            "Prioritize candidate card review, marker-neighborhood inspection, source-aware null tests, and sample/abundance mapping."
            if row["qc_tier_report"] in {"high_qc", "reviewable_qc"}
            else "Prioritize QC review before biological upgrading; retain as provisional molecular signal."
        ),
        axis=1,
    )

    return {
        "repo_root": repo_root,
        "esm_dir": esm_dir,
        "warehouse_dir": warehouse_dir,
        "glm_dir": glm_dir,
        "signal_dir": signal_dir,
        "joined": joined,
        "poc": poc,
        "top_bridge": top_bridge,
        "top": top,
        "msm_status": msm_status,
        "readiness_summary": readiness_summary,
        "glm_summary": glm_summary,
        "warehouse_manifest": warehouse_manifest,
        "validation_gates": validation_gates,
        "dim_mag": dim_mag,
        "feature_mrv": feature_mrv,
        "methane": methane,
        "sulfur": sulfur,
        "coverage": coverage,
        "taxonomy": taxonomy,
        "citations": build_citations(),
    }


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    poc = data["poc"]
    joined = data["joined"]
    top = data["top"]
    validation = data["validation_gates"]
    msm = data["msm_status"]
    warehouse_manifest = data["warehouse_manifest"]

    source_counts = poc["source"].value_counts().to_dict()
    qc_counts = poc["qc_tier_report"].value_counts().to_dict()
    gate_counts = validation["status"].value_counts().to_dict() if "status" in validation.columns else {}
    table_rows = warehouse_manifest.set_index("table")["rows"].to_dict()
    coverage = data["coverage"]
    kcov = coverage.loc[coverage["annotation_tool"].eq("KOfam"), "annotated_gene_fraction"].dropna()

    return {
        "poc_core_mags": int(len(poc)),
        "embedded_total": int(len(joined)),
        "quarantined_esm_rows": int((joined["readiness_class"] == "non_poc_or_unscoped").sum()),
        "rumen_mags": int(source_counts.get("rumen", 0)),
        "wetland_mags": int(source_counts.get("mucc", 0)),
        "top_bridge_count": int(len(top)),
        "top_bridge_all_three": int((top["has_esm2"] & top["has_glm"] & top["has_functional"]).sum()),
        "gates_pass": int(gate_counts.get("pass", 0)),
        "gates_warn": int(gate_counts.get("warn", 0)),
        "gates_fail": int(gate_counts.get("fail", 0)),
        "dim_gene_rows": int(table_rows.get("dim_gene", 0)),
        "kofam_rows": int(table_rows.get("fact_kofam_hits", 0)),
        "mcycdb_rows": int(table_rows.get("fact_mcycdb_hits", 0)),
        "scycdb_rows": int(table_rows.get("fact_scycdb_hits", 0)),
        "dbcan_rows": int(table_rows.get("fact_dbcan_hits", 0)),
        "metabolic_hmm_rows": int(table_rows.get("fact_metabolic_hmm_hits", 0)),
        "median_completeness": float(pd.to_numeric(poc["checkm2_completeness"], errors="coerce").median()),
        "median_contamination": float(pd.to_numeric(poc["checkm2_contamination"], errors="coerce").median()),
        "gunc_pass_fraction": float(bool_series(poc["gunc_pass"]).mean()),
        "median_kofam_coverage": float(kcov.median()) if len(kcov) else np.nan,
        "high_or_reviewable_qc": int(poc["qc_tier_report"].isin(["high_qc", "reviewable_qc"]).sum()),
        "msm_rows": int(len(msm)),
        "msm_glm": int(msm["has_glm"].fillna(False).astype(bool).sum()),
        "msm_functional": int(msm["has_functional"].fillna(False).astype(bool).sum()),
        "msm_glm_functional": int((msm["has_glm"].fillna(False).astype(bool) & msm["has_functional"].fillna(False).astype(bool)).sum()),
    }


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)
    return path


def plot_evidence_ledger(data: dict[str, Any], summary: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    rows = pd.DataFrame(
        [
            {"cohort": "POC MAG-bin core", "state": "ESM-2 + gLM2 + functional", "count": summary["poc_core_mags"]},
            {"cohort": "POC ESM-2 context", "state": "Quarantined non-MAG/unscoped", "count": summary["quarantined_esm_rows"]},
            {"cohort": "MSM mangrove expansion", "state": "gLM2 + functional", "count": summary["msm_glm_functional"]},
            {"cohort": "MSM mangrove expansion", "state": "gLM2 only, awaiting function", "count": summary["msm_glm"] - summary["msm_glm_functional"]},
        ]
    )
    order = ["POC MAG-bin core", "POC ESM-2 context", "MSM mangrove expansion"]
    state_order = ["ESM-2 + gLM2 + functional", "Quarantined non-MAG/unscoped", "gLM2 + functional", "gLM2 only, awaiting function"]
    palette = {
        "ESM-2 + gLM2 + functional": COLORS["olive"]["base"],
        "Quarantined non-MAG/unscoped": NEUTRAL["base"],
        "gLM2 + functional": COLORS["blue"]["base"],
        "gLM2 only, awaiting function": COLORS["gold"]["base"],
    }
    wide = rows.pivot_table(index="cohort", columns="state", values="count", fill_value=0).reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    left = np.zeros(len(wide))
    y = np.arange(len(wide))
    for state in state_order:
        values = wide[state].values if state in wide else np.zeros(len(wide))
        bars = ax.barh(y, values, left=left, label=state, color=palette[state], edgecolor=TOKENS["ink"], linewidth=0.8)
        for bar, value, start in zip(bars, values, left):
            if value > 0:
                ax.text(start + value / 2, bar.get_y() + bar.get_height() / 2, fmt_int(value), ha="center", va="center", fontsize=9, color=TOKENS["ink"], fontweight="semibold")
        left += values
    ax.set_yticks(y, wide.index)
    ax.set_xlabel("MAG/proteome units")
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=TOKENS["axis"], ncol=1, fontsize=8.3)
    ax.grid(axis="x", color=TOKENS["grid"])
    add_chart_header(
        fig,
        ax,
        "Evidence ledger for the closed POC core and mangrove expansion",
        "Counts are analysis units, not samples. The 625-unit POC core is fully triangulated; mangrove/MSM is gLM2-complete but still functionally in progress.",
    )
    return save_figure(fig, path)


def plot_esm2_geometry(data: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    joined = data["joined"].copy()
    joined["plot_group"] = np.where(joined["readiness_class"].eq("triangulated_now"), joined["source"], "quarantined")
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    for group, sub in joined.groupby("plot_group"):
        label = {"rumen": "Rumen MAG-bin", "mucc": "Wetland/MUCC MAG-bin", "quarantined": "Quarantined ESM2 context"}.get(group, group)
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=42 if group != "quarantined" else 28,
            c=SOURCE_COLORS.get(group if group != "mucc" else "wetland", NEUTRAL["base"]),
            edgecolors="#ffffff",
            linewidths=0.35,
            alpha=0.82 if group != "quarantined" else 0.48,
            label=label,
            zorder=2 if group != "quarantined" else 1,
        )
    top = data["top"].sort_values("rank").copy()
    top["methane_log"] = np.log1p(pd.to_numeric(top["methane_evidence_score"], errors="coerce").fillna(0))
    top["sulfur_log"] = np.log1p(pd.to_numeric(top["sulfur_competition_score"], errors="coerce").fillna(0))
    top = top.assign(
        methane_log=np.log1p(pd.to_numeric(top["methane_evidence_score"], errors="coerce").fillna(0)),
        sulfur_log=np.log1p(pd.to_numeric(top["sulfur_competition_score"], errors="coerce").fillna(0)),
    )
    ax.scatter(top["umap_1"], top["umap_2"], s=170, facecolors="none", edgecolors=COLORS["pink"]["dark"], linewidths=1.6, label="Top ESM-2 bridge candidates", zorder=5)
    for _, row in top.iterrows():
        ax.text(row["umap_1"] + 0.15, row["umap_2"] + 0.12, f"#{int(row['rank'])}", fontsize=8.5, fontweight="bold", color=COLORS["pink"]["dark"], zorder=6)
    ax.set_xlabel("ESM-2 UMAP 1")
    ax.set_ylabel("ESM-2 UMAP 2")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2, fontsize=8.5)
    ax.grid(color=TOKENS["grid"], linewidth=0.7)
    add_chart_header(
        fig,
        ax,
        "ESM-2 proteome geometry now has complete functional and gLM2 evidence for every MAG-bin point",
        "UMAP coordinates communicate the original POC geometry; bridge interpretation should use high-dimensional embeddings plus functional and gLM2 evidence, not the 2D map alone.",
        top=0.80,
    )
    return save_figure(fig, path)


def plot_glm_context(data: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    poc = data["poc"].copy()
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    order = ["Wetland/MUCC", "Rumen"]
    palette = {"Wetland/MUCC": COLORS["blue"]["base"], "Rumen": COLORS["orange"]["base"]}
    sns.boxplot(
        data=poc,
        x="source_label",
        y="glm_context_delta",
        order=order,
        palette=palette,
        hue="source_label",
        legend=False,
        ax=ax,
        linewidth=1,
        fliersize=1.5,
    )
    sns.stripplot(
        data=poc.sample(min(len(poc), 500), random_state=20260616),
        x="source_label",
        y="glm_context_delta",
        order=order,
        color=NEUTRAL["dark"],
        alpha=0.18,
        jitter=0.22,
        size=2.3,
        ax=ax,
    )
    top = data["top"].copy()
    xmap = {"Wetland/MUCC": 0, "Rumen": 1}
    for _, row in top.iterrows():
        ax.scatter(xmap[row["source_label"]], row["glm_context_delta"], s=95, facecolors=COLORS["gold"]["base"], edgecolors=COLORS["gold"]["dark"], linewidths=0.9, zorder=4)
        ax.text(xmap[row["source_label"]] + 0.06, row["glm_context_delta"], f"#{int(row['rank'])}", fontsize=8, color=COLORS["gold"]["dark"], fontweight="bold", va="center")
    ax.axhline(0, color=TOKENS["ink"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("")
    ax.set_ylabel("gLM2 native-minus-shuffled context delta")
    ax.grid(axis="y", color=TOKENS["grid"])
    add_chart_header(
        fig,
        ax,
        "gLM2 adds an order-sensitive context check to the bridge hypothesis",
        "Positive values mean the native gene-window embedding is more structured than its shuffled control under this summary feature; this is genomic-context evidence, not activity evidence.",
    )
    return save_figure(fig, path)


def plot_functional_landscape(data: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    poc = data["poc"].copy()
    poc["methane_log"] = np.log1p(pd.to_numeric(poc["methane_evidence_score"], errors="coerce").fillna(0))
    poc["sulfur_log"] = np.log1p(pd.to_numeric(poc["sulfur_competition_score"], errors="coerce").fillna(0))
    poc["substrate_size"] = 30 + np.sqrt(pd.to_numeric(poc["cazy_family_count"], errors="coerce").fillna(0) + 1) * 34
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    for source, sub in poc.groupby("source_label"):
        color = COLORS["blue"]["base"] if source == "Wetland/MUCC" else COLORS["orange"]["base"]
        edge = COLORS["blue"]["dark"] if source == "Wetland/MUCC" else COLORS["orange"]["dark"]
        ax.scatter(
            sub["methane_log"],
            sub["sulfur_log"],
            s=sub["substrate_size"],
            c=color,
            edgecolors=edge,
            linewidths=0.45,
            alpha=0.56,
            label=source,
        )
    top = data["top"].sort_values("rank").copy()
    top["methane_log"] = np.log1p(pd.to_numeric(top["methane_evidence_score"], errors="coerce").fillna(0))
    top["sulfur_log"] = np.log1p(pd.to_numeric(top["sulfur_competition_score"], errors="coerce").fillna(0))
    ax.scatter(top["methane_log"], top["sulfur_log"], s=190, facecolors="none", edgecolors=COLORS["pink"]["dark"], linewidths=1.5, label="Top bridge candidates", zorder=5)
    for _, row in top.iterrows():
        ax.text(row["methane_log"] + 0.03, row["sulfur_log"] + 0.03, f"#{int(row['rank'])}", fontsize=8, fontweight="bold", color=COLORS["pink"]["dark"])
    ax.set_xlabel("Methane-cycle evidence score, log scale")
    ax.set_ylabel("Sulfur-cycle competition score, log scale")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=3, fontsize=8.5)
    ax.grid(color=TOKENS["grid"])
    add_chart_header(
        fig,
        ax,
        "Functional fingerprints separate methane evidence from sulfur-competition context",
        "Each point is a MAG-bin. Marker counts are screening features from MCycDB, SCycDB, accepted KOfam, and METABOLIC-derived long tables; size reflects CAZy substrate breadth.",
        top=0.80,
    )
    return save_figure(fig, path)


def plot_bridge_heatmap(data: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    top = data["top"].sort_values("rank").copy()
    metrics = [
        ("esm_bridge_affinity_norm", "ESM-2\nbridge"),
        ("glm_context_norm", "gLM2\ncontext"),
        ("methane_signal_norm", "Methane\nsignal"),
        ("sulfur_signal_norm", "Sulfur\ncontext"),
        ("substrate_signal_norm", "Substrate\nbreadth"),
        ("broad_function_norm", "Broad\nfunction"),
        ("qc_signal_norm", "QC\nconfidence"),
    ]
    matrix = top[[m for m, _ in metrics]].to_numpy(float)
    labels = [f"#{int(r['rank'])} {short_id(r['proteome_id'], 30)}" for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    cmap = sns.color_palette("crest", as_cmap=True)
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if val > 0.58 else TOKENS["ink"], fontweight="semibold")
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7.5, color=NEUTRAL["dark"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="within-report normalized intensity")
    add_chart_header(
        fig,
        ax,
        "Top bridge candidates are no longer latent-only: each has a complete multiview evidence card",
        "The heatmap intentionally normalizes within the top-candidate set to show relative review priorities; it is not a calibrated risk score.",
        top=0.78,
    )
    return save_figure(fig, path)


def plot_qc_coverage(data: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    poc = data["poc"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.1, 0.9]})
    ax = axes[0]
    qc_palette = {
        "high_qc": COLORS["olive"]["base"],
        "reviewable_qc": COLORS["blue"]["base"],
        "caution_qc": COLORS["gold"]["base"],
        "qc_limited": COLORS["orange"]["base"],
    }
    for tier, sub in poc.groupby("qc_tier_report"):
        ax.scatter(
            sub["checkm2_completeness"],
            sub["checkm2_contamination"],
            s=40,
            color=qc_palette.get(tier, NEUTRAL["base"]),
            edgecolors="#ffffff",
            linewidths=0.35,
            alpha=0.76,
            label=tier.replace("_", " "),
        )
    ax.axvline(70, color=TOKENS["ink"], linestyle=":", linewidth=0.9)
    ax.axhline(10, color=TOKENS["ink"], linestyle=":", linewidth=0.9)
    ax.set_xlabel("CheckM2 completeness (%)")
    ax.set_ylabel("CheckM2 contamination (%)")
    ax.legend(frameon=False, fontsize=7.7, loc="upper right")
    ax.grid(color=TOKENS["grid"])

    cov = data["coverage"].copy()
    kcov = cov[cov["annotation_tool"].eq("KOfam")].merge(poc[["proteome_id", "source_label"]], on="proteome_id", how="inner")
    ax2 = axes[1]
    sns.histplot(
        data=kcov,
        x="annotated_gene_fraction",
        hue="source_label",
        palette={"Wetland/MUCC": COLORS["blue"]["base"], "Rumen": COLORS["orange"]["base"]},
        bins=24,
        multiple="layer",
        alpha=0.45,
        ax=ax2,
        edgecolor=TOKENS["ink"],
        linewidth=0.4,
    )
    if ax2.legend_ is not None:
        ax2.legend_.set_title("Source")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_xlabel("KOfam accepted annotation coverage")
    ax2.set_ylabel("MAG count")
    ax2.grid(axis="y", color=TOKENS["grid"])
    fig.suptitle("")
    add_chart_header(
        fig,
        axes[0],
        "QC and annotation coverage make absence calls auditable",
        "Left: genome quality confidence. Right: KOfam accepted-hit gene coverage by source. Absence of a pathway should only be interpreted after these confidence layers are checked.",
        top=0.76,
    )
    return save_figure(fig, path)


def plot_taxonomy_context(data: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    poc = data["poc"].copy()
    families = poc["tax_family_clean"].replace({"Unresolved": "Unresolved"}).value_counts().head(10).index.tolist()
    sub = poc[poc["tax_family_clean"].isin(families)].copy()
    counts = sub.groupby(["tax_family_clean", "source_label"]).size().reset_index(name="count")
    totals = counts.groupby("tax_family_clean")["count"].sum().sort_values(ascending=True)
    counts["tax_family_clean"] = pd.Categorical(counts["tax_family_clean"], categories=totals.index, ordered=True)
    wide = counts.pivot_table(
        index="tax_family_clean",
        columns="source_label",
        values="count",
        fill_value=0,
        observed=False,
    ).reindex(totals.index)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    left = np.zeros(len(wide))
    for source, color, edge in [
        ("Wetland/MUCC", COLORS["blue"]["base"], COLORS["blue"]["dark"]),
        ("Rumen", COLORS["orange"]["base"], COLORS["orange"]["dark"]),
    ]:
        values = wide[source].values if source in wide else np.zeros(len(wide))
        ax.barh(wide.index, values, left=left, label=source, color=color, edgecolor=edge, linewidth=0.8)
        left += values
    ax.set_xlabel("MAG-bin count")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2, fontsize=8.5)
    ax.grid(axis="x", color=TOKENS["grid"])
    add_chart_header(
        fig,
        ax,
        "Taxonomy keeps bridge interpretation source-aware",
        "The POC is biologically informative but source/ecosystem-confounded: rumen and wetland/MUCC come from different source programs. Candidate claims must remain source-aware until additional sources are added.",
    )
    return save_figure(fig, path)


def plot_maturity_ladder(data: dict[str, Any], path: Path) -> Path:
    use_chart_theme()
    stages = pd.DataFrame(
        [
            {"stage": "MAG molecular screening", "status": "complete now", "x": 1},
            {"stage": "Bridge attestation cards", "status": "ready now", "x": 2},
            {"stage": "Sample mapping + abundance", "status": "next gap", "x": 3},
            {"stage": "Environment + flux validation", "status": "future validation", "x": 4},
            {"stage": "Final MRV risk tiers", "status": "blocked until calibrated", "x": 5},
        ]
    )
    palette = {
        "complete now": COLORS["olive"]["base"],
        "ready now": COLORS["blue"]["base"],
        "next gap": COLORS["gold"]["base"],
        "future validation": COLORS["orange"]["base"],
        "blocked until calibrated": COLORS["pink"]["base"],
    }
    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    ax.plot(stages["x"], np.ones(len(stages)), color=NEUTRAL["base"], linewidth=2.0, zorder=1)
    for _, row in stages.iterrows():
        ax.scatter(row["x"], 1, s=650, color=palette[row["status"]], edgecolor=TOKENS["ink"], linewidth=1.0, zorder=3)
        ax.text(row["x"], 1, str(int(row["x"])), ha="center", va="center", fontsize=11, fontweight="bold", color=TOKENS["ink"])
        ax.text(row["x"], 0.74, textwrap.fill(row["stage"], width=18), ha="center", va="top", fontsize=8.5, color=TOKENS["ink"], fontweight="semibold")
        ax.text(row["x"], 1.24, textwrap.fill(row["status"], width=18), ha="center", va="bottom", fontsize=8.2, color=TOKENS["muted"])
    ax.set_xlim(0.45, 5.55)
    ax.set_ylim(0.35, 1.55)
    ax.axis("off")
    add_chart_header(
        fig,
        ax,
        "The report upgrades molecular intelligence, not credit-grade MRV scoring",
        "The POC now reaches a strong MAG/proteome evidence stage. Sample-level risk still requires abundance, environmental context, uncertainty propagation, and flux/process validation.",
        top=0.72,
    )
    return save_figure(fig, path)


def build_figures(data: dict[str, Any], summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    fig_dir = output_dir / "assets" / "figures"
    figures = {
        "evidence_ledger": plot_evidence_ledger(data, summary, fig_dir / "figure_01_evidence_ledger.png"),
        "esm2_geometry": plot_esm2_geometry(data, fig_dir / "figure_02_esm2_bridge_geometry.png"),
        "glm_context": plot_glm_context(data, fig_dir / "figure_03_glm_context.png"),
        "functional_landscape": plot_functional_landscape(data, fig_dir / "figure_04_functional_landscape.png"),
        "bridge_heatmap": plot_bridge_heatmap(data, fig_dir / "figure_05_bridge_signature_heatmap.png"),
        "qc_coverage": plot_qc_coverage(data, fig_dir / "figure_06_qc_coverage.png"),
        "taxonomy_context": plot_taxonomy_context(data, fig_dir / "figure_07_taxonomy_context.png"),
        "maturity_ladder": plot_maturity_ladder(data, fig_dir / "figure_08_mrv_maturity_ladder.png"),
    }
    return figures


def build_claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim": "POC MAG-bin molecular atlas is complete",
                "status": "supported",
                "allowed_wording": "The 625 MAG-bin POC core has ESM-2, gLM2, functional, QC, taxonomy, and annotation-coverage evidence.",
                "blocking_gap": "",
                "next_action": "Use as denominator for MBAG POC reporting.",
            },
            {
                "claim": "Bridge candidates are review-ready molecular hypotheses",
                "status": "supported_with_caveats",
                "allowed_wording": "Top bridge candidates can be reviewed with complete multiview evidence and explicit QC/source caveats.",
                "blocking_gap": "Source/ecosystem confounding and lack of external validation.",
                "next_action": "Run source-aware nulls, bootstrap stability, and candidate-card review.",
            },
            {
                "claim": "MethaNet can score sample-level methane permanence risk",
                "status": "blocked",
                "allowed_wording": "The current atlas provides MAG-level risk primitives, not sample-level risk scores.",
                "blocking_gap": "Sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, and flux/process validation.",
                "next_action": "Build sample risk readiness layer.",
            },
            {
                "claim": "MethaNet supports carbon-credit approval",
                "status": "not_allowed",
                "allowed_wording": "MethaNet can support screening, monitoring design, and validation prioritization.",
                "blocking_gap": "Registry methodology integration, field validation, and third-party verification.",
                "next_action": "Keep crediting language out of current molecular report.",
            },
        ]
    )


def build_gap_register(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap": "Sample/metagenome rollup",
                "why_it_matters": "Turns MAG potential into project-relevant ecological interpretation.",
                "current_state": "Not integrated in POC report.",
                "priority": "P0",
                "next_step": "Create dim_sample and link_sample_mag with provenance confidence.",
            },
            {
                "gap": "Abundance/read coverage",
                "why_it_matters": "Prevents rare MAGs from dominating sample-risk interpretation.",
                "current_state": "Not integrated.",
                "priority": "P0",
                "next_step": "Add fact_mag_abundance and marker abundance features.",
            },
            {
                "gap": "Environmental methane permissiveness",
                "why_it_matters": "Gene potential only matters if redox, sulfate, salinity, substrate, and hydrology permit methane production.",
                "current_state": "Not joined.",
                "priority": "P1",
                "next_step": "Join sample/site environmental covariates with resolution tiers.",
            },
            {
                "gap": "MSM/mangrove functional completion",
                "why_it_matters": "Expands MethaNet from POC wetland/rumen bridge evidence into a larger mangrove target domain.",
                "current_state": f"{summary['msm_functional']:,} of {summary['msm_rows']:,} MSM MAGs had functional completion in the latest signal-map artifact.",
                "priority": "P1",
                "next_step": "Finish active arrays and build MSM functional warehouse.",
            },
            {
                "gap": "MSM ESM-2 parity",
                "why_it_matters": "Needed before mangroves are a true ESM-2/function/gLM tri-view peer to the POC core.",
                "current_state": "Not present in current POC embedding artifact.",
                "priority": "P2",
                "next_step": "Generate MSM ESM-2 proteome embeddings with the same configuration or document a deliberate alternative.",
            },
        ]
    )


def write_tables(data: dict[str, Any], summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    table_dir = output_dir / "tables"
    source_dir = output_dir / "sources"
    table_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    poc_keep = [
        "proteome_id",
        "mag_id_manifest",
        "source",
        "ecosystem",
        "tax_domain_clean",
        "tax_family_clean",
        "tax_genus_clean",
        "n_proteins_used",
        "umap_1",
        "umap_2",
        "mixing_coeff",
        "rank",
        "has_esm2",
        "has_glm",
        "has_functional",
        "glm_context_delta",
        "glm_context_ratio",
        "checkm2_completeness",
        "checkm2_contamination",
        "gunc_pass",
        "qc_tier_report",
        "kofam_annotated_gene_fraction",
        "metabolic_modules_present",
        "cazy_family_count",
        "merops_family_count",
        "methane_evidence_score",
        "sulfur_competition_score",
        "triangulation_signal",
        "bridge_card_status",
        "claim_boundary",
    ]
    poc_path = table_dir / "mbag_poc_core_feature_table.tsv"
    data["poc"][[c for c in poc_keep if c in data["poc"].columns]].to_csv(poc_path, sep="\t", index=False)

    candidate_cols = [
        "rank",
        "proteome_id",
        "source",
        "ecosystem",
        "tax_domain_clean",
        "tax_family_clean",
        "checkm2_completeness",
        "checkm2_contamination",
        "gunc_pass",
        "qc_tier_report",
        "mixing_coeff",
        "glm_context_delta",
        "methane_evidence_score",
        "sulfur_competition_score",
        "cazy_family_count",
        "metabolic_modules_present",
        "bridge_card_status",
        "allowed_claim_wording",
        "next_validation_action",
    ]
    card_path = table_dir / "bridge_candidate_cards.tsv"
    data["top"][[c for c in candidate_cols if c in data["top"].columns]].to_csv(card_path, sep="\t", index=False)

    claim_path = table_dir / "claim_boundary_matrix.tsv"
    build_claim_matrix().to_csv(claim_path, sep="\t", index=False)

    gap_path = table_dir / "validation_gap_register.tsv"
    build_gap_register(summary).to_csv(gap_path, sep="\t", index=False)

    citation_path = source_dir / "citation_inventory.tsv"
    data["citations"].to_csv(citation_path, sep="\t", index=False)

    source_inventory = pd.DataFrame(
        [
            {"source": "ESM-2 artifacts", "path": str(data["esm_dir"]), "role": "Embedding geometry, bridge candidates, projections"},
            {"source": "Functional warehouse", "path": str(data["warehouse_dir"]), "role": "Parquet/DuckDB functional facts and feature tables"},
            {"source": "gLM2 integration", "path": str(data["glm_dir"]), "role": "Contextual genomic MAG/window feature layer"},
            {"source": "Corrected signal map", "path": str(data["signal_dir"]), "role": "Joined POC multiview readiness and MSM status"},
        ]
    )
    source_path = source_dir / "source_inventory.tsv"
    source_inventory.to_csv(source_path, sep="\t", index=False)

    return {
        "poc_feature_table": poc_path,
        "candidate_cards": card_path,
        "claim_boundary_matrix": claim_path,
        "validation_gap_register": gap_path,
        "citation_inventory": citation_path,
        "source_inventory": source_path,
    }


def render_table_html(df: pd.DataFrame, *, max_rows: int = 12) -> str:
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, escape=False, classes="data-table", border=0)


def render_metric_card(label: str, value: str, sub: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-value">{value}</div>
      <div class="metric-label">{label}</div>
      <div class="metric-sub">{sub}</div>
    </div>
    """


def render_figure_block(title: str, text: str, image_uri: str, alt: str) -> str:
    return f"""
    <figure class="figure-block">
      <img src="{image_uri}" alt="{alt}" />
      <figcaption>
        <strong>{title}</strong>
        <span>{text}</span>
      </figcaption>
    </figure>
    """


def build_html(
    data: dict[str, Any],
    summary: dict[str, Any],
    figures: dict[str, Path],
    table_paths: dict[str, Path],
    args: argparse.Namespace,
    output_dir: Path,
) -> str:
    figure_uri = {key: image_data_uri(path) for key, path in figures.items()}
    top = data["top"].sort_values("rank")
    claim_matrix = build_claim_matrix()
    gap_register = build_gap_register(summary)
    citations = data["citations"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metric_cards = "\n".join(
        [
            render_metric_card("Closed POC core", fmt_int(summary["poc_core_mags"]), "MAG-bin units with ESM-2 + gLM2 + functional evidence"),
            render_metric_card("Top bridge candidates", f"{summary['top_bridge_all_three']}/{summary['top_bridge_count']}", "all three evidence layers present"),
            render_metric_card("Validation gates", f"{summary['gates_pass']} pass", f"{summary['gates_warn']} warnings / {summary['gates_fail']} failures"),
            render_metric_card("Functional evidence scale", fmt_int(summary["kofam_rows"]), "KOfam hit rows in the POC warehouse"),
            render_metric_card("Median QC", f"{summary['median_completeness']:.1f}% / {summary['median_contamination']:.1f}%", "CheckM2 completeness / contamination"),
            render_metric_card("MSM expansion", f"{summary['msm_functional']:,}/{summary['msm_rows']:,}", "mangrove MAGs with functional + gLM2 so far"),
        ]
    )

    candidate_table = top[
        [
            "rank",
            "display_id",
            "source_label",
            "tax_family_clean",
            "qc_tier_report",
            "glm_context_delta",
            "methane_evidence_score",
            "sulfur_competition_score",
            "bridge_card_status",
        ]
    ].copy()
    candidate_table.columns = [
        "Rank",
        "Candidate",
        "Source",
        "Taxonomic family",
        "QC tier",
        "gLM2 context delta",
        "Methane signal",
        "Sulfur context",
        "Report status",
    ]
    candidate_table["gLM2 context delta"] = candidate_table["gLM2 context delta"].map(lambda x: f"{float(x):.2f}")
    candidate_table["Methane signal"] = candidate_table["Methane signal"].map(fmt_int)
    candidate_table["Sulfur context"] = candidate_table["Sulfur context"].map(fmt_int)

    references_html = "\n".join(
        f"""<li id="ref-{row.id}"><a href="{row.url}">{row.label}</a>: {row.citation} <em>Used for:</em> {row.report_use}</li>"""
        for _, row in citations.iterrows()
    )

    css = """
    :root {
      --ink: #132033;
      --muted: #667085;
      --line: #DFE5EF;
      --surface: #F7F8FB;
      --panel: #FFFFFF;
      --blue: #5477C4;
      --blue-soft: #EAF1FE;
      --gold: #B8A037;
      --gold-soft: #FFF4C2;
      --orange: #CC6F47;
      --orange-soft: #FFEDDE;
      --olive: #71B436;
      --olive-soft: #D8ECBD;
      --pink: #BD569B;
      --pink-soft: #FCDAD6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: Inter, Aptos, "Segoe UI", Roboto, Arial, sans-serif;
      line-height: 1.55;
    }
    .page { max-width: 1180px; margin: 0 auto; padding: 42px 26px 70px; }
    header.hero {
      padding: 54px 48px 42px;
      background:
        radial-gradient(circle at 82% 18%, rgba(255, 244, 194, 0.78), transparent 28%),
        linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 48%, #ECFDF3 100%);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 22px 80px rgba(19, 32, 51, 0.09);
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--blue);
      font-weight: 800;
      font-size: 0.78rem;
      margin-bottom: 16px;
    }
    h1 {
      font-size: clamp(2.35rem, 6vw, 5.6rem);
      line-height: 0.98;
      letter-spacing: 0;
      margin: 0 0 20px;
      max-width: 980px;
    }
    .subtitle {
      max-width: 860px;
      color: #334155;
      font-size: 1.15rem;
      margin: 0;
    }
    .claim-lock {
      margin-top: 26px;
      display: inline-flex;
      gap: 10px;
      align-items: center;
      border: 1px solid #FDBA74;
      background: #FFF7ED;
      color: #7C2D12;
      padding: 11px 14px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.92rem;
    }
    section {
      margin-top: 34px;
      padding: 34px 34px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 14px 44px rgba(19, 32, 51, 0.055);
    }
    section.flat {
      background: transparent;
      border: 0;
      box-shadow: none;
      padding: 16px 2px;
    }
    h2 {
      font-size: clamp(1.55rem, 3vw, 2.35rem);
      line-height: 1.12;
      letter-spacing: 0;
      margin: 0 0 14px;
    }
    h3 {
      font-size: 1.22rem;
      margin: 26px 0 8px;
    }
    p { margin: 0 0 16px; color: #334155; }
    .summary-list {
      display: grid;
      gap: 14px;
      margin-top: 18px;
    }
    .summary-item {
      border-left: 5px solid var(--blue);
      padding: 12px 14px;
      background: #F8FAFC;
      border-radius: 14px;
    }
    .summary-item strong { color: var(--ink); }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 24px;
    }
    .metric-card {
      background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 18px 16px;
      min-height: 138px;
    }
    .metric-value {
      font-size: 2rem;
      font-weight: 850;
      line-height: 1.03;
      color: var(--ink);
      font-variant-numeric: tabular-nums;
    }
    .metric-label {
      margin-top: 10px;
      font-weight: 800;
      color: #1F2937;
    }
    .metric-sub {
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 4px;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .callout {
      border-radius: 18px;
      border: 1px solid #BFDBFE;
      background: #EFF6FF;
      padding: 18px 20px;
      color: #1E3A8A;
      margin: 18px 0;
    }
    .warning {
      border-color: #FDBA74;
      background: #FFF7ED;
      color: #7C2D12;
    }
    .figure-block {
      margin: 24px 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: #FFFFFF;
    }
    .figure-block img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 14px;
      background: #FCFCFD;
    }
    figcaption {
      display: grid;
      gap: 4px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    figcaption strong { color: var(--ink); }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      margin-top: 18px;
    }
    .data-table th {
      text-align: left;
      color: var(--ink);
      background: #F1F5F9;
      border-bottom: 1px solid var(--line);
      padding: 10px 9px;
      font-weight: 800;
    }
    .data-table td {
      border-bottom: 1px solid #EEF2F7;
      padding: 9px;
      color: #334155;
      vertical-align: top;
    }
    .pill-row { display: flex; flex-wrap: wrap; gap: 9px; margin: 14px 0 4px; }
    .pill {
      border-radius: 999px;
      padding: 7px 11px;
      background: #F1F5F9;
      color: #334155;
      font-weight: 750;
      font-size: 0.84rem;
    }
    ol, ul { color: #334155; padding-left: 1.35rem; }
    li { margin: 8px 0; }
    .refs {
      columns: 2;
      column-gap: 34px;
      font-size: 0.86rem;
    }
    .artifact-list code {
      font-family: "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.82rem;
      background: #F1F5F9;
      padding: 2px 5px;
      border-radius: 6px;
    }
    footer {
      margin-top: 36px;
      color: var(--muted);
      font-size: 0.86rem;
      text-align: center;
    }
    @media (max-width: 820px) {
      .page { padding: 22px 14px 42px; }
      header.hero { padding: 36px 24px 30px; }
      section { padding: 25px 18px; }
      .metric-grid, .two-col { grid-template-columns: 1fr; }
      .refs { columns: 1; }
    }
    """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MethaNet MBAG POC Molecular Intelligence Report</title>
  <style>{css}</style>
</head>
<body>
<main class="page">
  <header class="hero">
    <div class="eyebrow">MethaNet Bridge Attestation Graph · POC Core</div>
    <h1>Molecular intelligence for methane-risk screening in blue carbon systems</h1>
    <p class="subtitle">A closed 625-MAG proof-of-concept integrating ESM-2 proteome geometry, gLM2 genomic context, and functional annotations into an auditable bridge-candidate evidence layer.</p>
    <div class="claim-lock">Claim boundary: MAG/proteome molecular screening, not final MRV risk scoring.</div>
  </header>

  <section>
    <h2>Executive Summary</h2>
    <div class="summary-list">
      <div class="summary-item"><strong>The POC core is now closed.</strong> All {summary['poc_core_mags']:,} comparable rumen + wetland/MUCC MAG-bin units have ESM-2, gLM2, functional, QC, taxonomy, and annotation-coverage evidence. The remaining {summary['quarantined_esm_rows']} ESM-2 rows are correctly quarantined as non-MAG/unscoped context.</div>
      <div class="summary-item"><strong>The bridge-candidate story has moved from latent geometry to molecular attestation.</strong> The top {summary['top_bridge_count']} ESM-2 bridge candidates now have complete gLM2 and functional evidence, enabling candidate cards with mechanism, genomic-context, QC, and source-boundary caveats.</div>
      <div class="summary-item"><strong>This is fundable because it converts metagenomic complexity into a product primitive.</strong> MethaNet can now demonstrate how molecular fingerprints become monitoring-priority hypotheses, MRV feature primitives, and partner-facing evidence packets for blue carbon methane-risk screening.</div>
      <div class="summary-item"><strong>The science remains honest.</strong> The report does not claim measured methane flux, final A-E risk tiers, carbon-credit approval, or source-independent rumen-to-wetland transfer. Those require sample mapping, abundance, environmental covariates, uncertainty propagation, and validation.</div>
    </div>
    <div class="metric-grid">{metric_cards}</div>
  </section>

  <section>
    <h2>The problem: blue carbon needs methane-risk intelligence before methane becomes a surprise</h2>
    <p><strong>Blue carbon projects are valued for durable climate benefit, but methane can erode that benefit in an uneven, site-specific way.</strong> Direct flux measurement is expensive and sparse, while microbial risk signatures can be observed at scale from metagenomes. MethaNet's opportunity is to provide a molecular attestation layer: a way to decide which samples, sites, and microbial mechanisms deserve deeper monitoring before they become accounting or credibility problems.</p>
    <p>The current artifact is intentionally upstream of final MRV scoring. It asks a more defensible first question: <strong>do rumen methane-system proteomes and blue-carbon wetland MAGs share molecular bridge signatures that can be explained with independent functional and genomic-context evidence?</strong></p>
    {render_figure_block(
        "Evidence ledger",
        "The POC core is complete at MAG/proteome grain. MSM/mangrove is included as a target-domain expansion, not as full ESM-2 parity yet.",
        figure_uri["evidence_ledger"],
        "Evidence ledger for POC and MSM expansion"
    )}
  </section>

  <section>
    <h2>The approach: three molecular views, one auditable bridge layer</h2>
    <p><strong>ESM-2 gives the latent proteome geometry.</strong> It asks which MAG/proteome units look similar in learned protein-sequence/function space. This is useful for discovering bridge candidates but cannot prove mechanism by itself.</p>
    <p><strong>Functional annotation explains the mechanism.</strong> KOfam, MCycDB, SCycDB, dbCAN/CAZy, METABOLIC, MEROPS, Bakta, CheckM2, GUNC, and GTDB-Tk convert each MAG into methane, sulfur, substrate, QC, and taxonomy evidence.</p>
    <p><strong>gLM2 asks whether genomic context supports the signal.</strong> Native gene-window embeddings are compared against shuffled controls so the report can distinguish ordered genomic architecture from a simple bag of genes.</p>
    <div class="pill-row">
      <span class="pill">ESM-2 proteome geometry</span>
      <span class="pill">gLM2 genomic context</span>
      <span class="pill">MCycDB methane markers</span>
      <span class="pill">SCycDB sulfur context</span>
      <span class="pill">dbCAN/CAZy substrates</span>
      <span class="pill">METABOLIC modules</span>
      <span class="pill">CheckM2/GUNC/GTDB-Tk confidence</span>
    </div>
    {render_figure_block(
        "ESM-2 bridge geometry",
        "The original 2D geometry is a communication layer; statistical and biological interpretation must use the full multiview feature system.",
        figure_uri["esm2_geometry"],
        "ESM-2 geometry with top bridge candidates"
    )}
  </section>

  <section>
    <h2>The core result: bridge candidates are now evidence-complete, not just visually interesting</h2>
    <p><strong>The previous smoke-test weakness is gone.</strong> The top bridge candidates are no longer missing functional or gLM2 evidence. Each can now be represented as a candidate card with latent bridge rank, gLM2 context, methane/sulfur/substrate features, QC, taxonomy, claim boundary, and next validation action.</p>
    {render_figure_block(
        "Bridge candidate signature matrix",
        "The matrix is normalized within the top-candidate set for review prioritization. It is not calibrated methane risk.",
        figure_uri["bridge_heatmap"],
        "Top bridge candidate signature heatmap"
    )}
    <h3>Top bridge candidate audit table</h3>
    <p>This table is deliberately conservative: it shows review status and direct evidence fields, not a final risk tier.</p>
    {render_table_html(candidate_table, max_rows=10)}
  </section>

  <section>
    <h2>gLM2 adds a genomic-context test that ESM-2 alone cannot provide</h2>
    <p><strong>Protein similarity can nominate a bridge; gene order can make the nomination more biologically plausible.</strong> The gLM2 native-minus-shuffled feature is a compact check that the native gene-window context carries structure beyond randomized gene order. It does not prove expression or flux, but it makes the bridge card more interpretable.</p>
    {render_figure_block(
        "gLM2 context by ecosystem",
        "Top bridge candidates are highlighted on the source distributions. Positive deltas are contextual evidence, not activity evidence.",
        figure_uri["glm_context"],
        "gLM2 contextual evidence distribution"
    )}
  </section>

  <section>
    <h2>Functional evidence turns bridge candidates into mechanism hypotheses</h2>
    <p><strong>The most valuable output is not a single score; it is a mechanism ledger.</strong> Methane markers, sulfur competition, substrate breadth, broad metabolism, taxonomy, and QC together say what kind of biological follow-up a candidate deserves. This is how MethaNet moves from "interesting embedding point" to "reviewable molecular attestation packet."</p>
    {render_figure_block(
        "Functional mechanism landscape",
        "Methane and sulfur axes are shown on log scales because the database hit counts span orders of magnitude. Point size reflects CAZy/substrate breadth.",
        figure_uri["functional_landscape"],
        "Functional mechanism landscape"
    )}
    {render_figure_block(
        "QC and annotation coverage",
        "These confidence layers prevent over-reading absent pathways. A missing function is weaker evidence in low-completeness or low-coverage MAGs.",
        figure_uri["qc_coverage"],
        "QC and annotation coverage evidence"
    )}
  </section>

  <section>
    <h2>Source-aware interpretation is a strength, not a caveat to hide</h2>
    <p><strong>The POC is powerful but source-confounded.</strong> Rumen and wetland/MUCC units come from different source programs, so the report should not claim source-independent transfer. Instead, it should show that MethaNet has the architecture needed to test transfer rigorously as additional sources are added.</p>
    {render_figure_block(
        "Taxonomy and source context",
        "Taxonomy helps decide whether a bridge candidate is a methanogen-like anchor, a bacterial comparator, or a broader metabolic context signal.",
        figure_uri["taxonomy_context"],
        "Taxonomy and source context"
    )}
  </section>

  <section>
    <h2>The MRV translation: strong molecular primitives, not final risk tiers</h2>
    <p><strong>This report supports a credible fundraising story because it is both ambitious and bounded.</strong> The POC now demonstrates a molecular evidence layer that can become a dashboard/API primitive for screening, prioritization, and monitoring design. The next product layer is sample risk readiness, not final A-E scoring.</p>
    {render_figure_block(
        "MRV maturity ladder",
        "The current report reaches MAG molecular screening and bridge-card readiness. Sample-level risk needs the additional layers shown to the right.",
        figure_uri["maturity_ladder"],
        "MRV maturity ladder"
    )}
    <div class="callout warning"><strong>Forbidden claims:</strong> this artifact must not be used to claim carbon-credit approval, measured methane flux, final sample-level A-E risk tiers, or source-independent rumen-to-wetland transfer.</div>
  </section>

  <section>
    <h2>Recommended next steps</h2>
    <ol>
      <li><strong>Use this POC report as the core fundraising artifact.</strong> It is now complete enough to show MethaNet's differentiated molecular-intelligence architecture.</li>
      <li><strong>Build candidate cards from the top bridge table.</strong> Pair each candidate with mechanism evidence, QC, taxonomy, gLM2 context, allowed wording, and validation action.</li>
      <li><strong>Finish MSM/mangrove functional processing and build the MSM warehouse.</strong> MSM currently has gLM2 for all {summary['msm_rows']:,} MAGs but functional completion is still partial.</li>
      <li><strong>Generate MSM ESM-2 embeddings if mangrove tri-view parity is needed.</strong> Until then, MSM remains a target-domain expansion layer rather than a full POC-equivalent ESM-2/function/gLM cohort.</li>
      <li><strong>Start the sample risk readiness layer.</strong> Add sample mapping, MAG abundance/read coverage, environmental covariates, and uncertainty fields before any sample-level MRV score.</li>
    </ol>
  </section>

  <section>
    <h2>Claim-boundary matrix</h2>
    <p><strong>The strongest reports are explicit about what they do not prove.</strong> This matrix keeps MethaNet's fundraising narrative scientifically durable.</p>
    {render_table_html(claim_matrix, max_rows=8)}
  </section>

  <section>
    <h2>Further questions that would upgrade the claim</h2>
    <p>These are not weaknesses in the POC; they are the next evidence layers that turn a molecular atlas into MRV-grade intelligence.</p>
    {render_table_html(gap_register, max_rows=8)}
  </section>

  <section>
    <h2>Caveats and assumptions</h2>
    <ul>
      <li>The report is MAG/proteome-grain. It does not represent whole sample or whole metagenome methane flux.</li>
      <li>Functional annotations are genomic potential and database evidence, not expression, activity, or process-rate measurements.</li>
      <li>ESM-2 and gLM2 are learned representation layers; they nominate and contextualize hypotheses, but do not replace direct marker evidence or validation.</li>
      <li>The POC is source-aware but source-confounded: rumen and wetland/MUCC are different source domains.</li>
      <li>Mangrove/MSM is included as an expansion layer; it does not yet have POC-equivalent ESM-2 parity in this artifact.</li>
    </ul>
  </section>

  <section>
    <h2>Methods and literature grounding</h2>
    <p>The visible narrative keeps methods brief, but the report is grounded in primary and official sources for protein language models, contextual genomic modeling, functional annotation, genome QC, taxonomy, and blue-carbon MRV boundaries.</p>
    <ol class="refs">{references_html}</ol>
  </section>

  <section>
    <h2>Artifact package</h2>
    <p>The HTML page is self-contained with embedded PNG images. Supporting tables and source inventories are included for audit and reuse.</p>
    <ul class="artifact-list">
      <li><code>{table_paths['poc_feature_table'].relative_to(output_dir)}</code>: 625-row POC feature table.</li>
      <li><code>{table_paths['candidate_cards'].relative_to(output_dir)}</code>: top bridge candidate cards.</li>
      <li><code>{table_paths['claim_boundary_matrix'].relative_to(output_dir)}</code>: allowed/prohibited claim register.</li>
      <li><code>{table_paths['validation_gap_register'].relative_to(output_dir)}</code>: evidence gaps for MRV upgrade.</li>
      <li><code>{table_paths['citation_inventory'].relative_to(output_dir)}</code>: literature and official-source inventory.</li>
    </ul>
  </section>

  <footer>
    Generated {generated}. Snapshot label: {args.snapshot_label}. MethaNet POC core report built from local ESM-2, gLM2, and functional warehouse artifacts.
  </footer>
</main>
</body>
</html>
"""
    return html


def write_readme(output_dir: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    readme = f"""# MethaNet MBAG POC Core Molecular Intelligence Report

Generated: {datetime.now(timezone.utc).isoformat()}

This package contains a self-contained HTML intelligence report for the closed
625-MAG POC core. The report integrates ESM-2 proteome geometry, gLM2
contextual genomics, and the functional cohort warehouse.

## Main Artifact

- `report.html`

## Headline Counts

- POC MAG-bin core: {summary['poc_core_mags']}
- Quarantined ESM2 non-MAG/unscoped rows: {summary['quarantined_esm_rows']}
- Top bridge candidates with all three POC layers: {summary['top_bridge_all_three']} / {summary['top_bridge_count']}
- Validation gates: {summary['gates_pass']} pass, {summary['gates_warn']} warn, {summary['gates_fail']} fail

## Claim Boundary

This artifact supports MAG/proteome molecular screening, bridge-candidate
prioritization, MRV feature primitives, and validation planning. It does not
claim measured methane flux, final sample-level risk tiers, carbon-credit
approval, or source-independent transfer.

## Reproducible Command

```bash
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
conda activate methanet-fgx
python scripts/reports/build_mbag_poc_core_html_report.py \\
  --output-dir {args.output_dir}
```
"""
    (output_dir / "README.md").write_text(readme)


def write_manifest(
    output_dir: Path,
    data: dict[str, Any],
    summary: dict[str, Any],
    figures: dict[str, Path],
    table_paths: dict[str, Path],
) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": str(output_dir / "report.html"),
        "summary": summary,
        "inputs": {
            "esm_dir": str(data["esm_dir"]),
            "warehouse_dir": str(data["warehouse_dir"]),
            "glm_dir": str(data["glm_dir"]),
            "signal_dir": str(data["signal_dir"]),
        },
        "figures": {key: str(path) for key, path in figures.items()},
        "tables": {key: str(path) for key, path in table_paths.items()},
        "claim_boundary": "MAG/proteome molecular screening; not final MRV risk scoring.",
    }
    (output_dir / "report_bundle_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_sources(args)
    summary = summarize(data)
    figures = build_figures(data, summary, output_dir)
    table_paths = write_tables(data, summary, output_dir)
    html = build_html(data, summary, figures, table_paths, args, output_dir)
    (output_dir / "report.html").write_text(html)
    write_readme(output_dir, summary, args)
    write_manifest(output_dir, data, summary, figures, table_paths)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
