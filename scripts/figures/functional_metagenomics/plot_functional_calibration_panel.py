#!/usr/bin/env python3
"""Build a preliminary MethaNet functional-genomics calibration figure panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

BLUE = {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
PINK = {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}
NEUTRAL = {"xlight": "#F4F5F7", "light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"],
        "axes.labelcolor": TOKENS["ink"],
        "xtick.color": TOKENS["muted"],
        "ytick.color": TOKENS["muted"],
        "grid.color": TOKENS["grid"],
        "grid.linewidth": 0.8,
        "font.family": ["DejaVu Sans", "sans-serif"],
        "axes.titleweight": "bold",
        "axes.titlecolor": TOKENS["ink"],
    })


def load_records(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("per_mag/*/*/curated/run_record.json")):
        record = json.loads(p.read_text())
        if record.get("status") != "complete":
            continue
        summary = record.get("summary_metrics", {})
        qc = record.get("qc", {})
        tax = record.get("taxonomy", {})
        rows.append({
            "run_dir": str(p.parents[1]),
            "proteome_id": record["proteome_id"],
            "mag_id": record["mag_id"],
            "elapsed_min": record.get("job", {}).get("elapsed_seconds", np.nan) / 60,
            "contigs": summary.get("input_contigs"),
            "bp": summary.get("input_total_bp"),
            "n50": summary.get("input_n50_bp"),
            "proteins": summary.get("prodigal_proteins"),
            "kofam_rows": summary.get("kofam_rows"),
            "mcycdb_hits": summary.get("mcycdb_hits"),
            "scycdb_hits": summary.get("scycdb_hits"),
            "dbcan_rows": summary.get("dbcan_overview_rows"),
            "bakta_rows": summary.get("bakta_feature_rows"),
            "completeness": qc.get("completeness"),
            "contamination": qc.get("contamination"),
            "gunc_pass": qc.get("gunc_pass"),
            "domain": tax.get("domain"),
            "phylum": tax.get("phylum"),
            "genus": tax.get("genus"),
            "species": tax.get("species"),
        })
    return pd.DataFrame(rows)


def read_parquet_table(root: Path, table: str, completed_dirs: set[str]) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob(f"per_mag/*/*/curated/parquet/{table}.parquet")):
        if str(p.parents[2]) in completed_dirs:
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", color=TOKENS["ink"])


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def annotate_bars(ax, fmt="{:.0f}", offset=2) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + offset, patch.get_y() + patch.get_height() / 2, fmt.format(width),
                va="center", ha="left", fontsize=8, color=TOKENS["muted"])


def build_panel(root: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_records(root)
    if df.empty:
        raise SystemExit(f"No completed run records found under {root}")
    completed_dirs = set(df["run_dir"])
    kofam = read_parquet_table(root, "fact_kofam_hits", completed_dirs)
    mcyc = read_parquet_table(root, "fact_mcycdb_hits", completed_dirs)
    scyc = read_parquet_table(root, "fact_scycdb_hits", completed_dirs)
    dbcan = read_parquet_table(root, "fact_dbcan_hits", completed_dirs)

    if not kofam.empty and "accepted_hit" in kofam:
        accepted_ko = kofam.loc[kofam["accepted_hit"].astype(bool)].groupby("proteome_id")["ko_id"].nunique()
        df = df.merge(accepted_ko.rename("accepted_ko_count"), on="proteome_id", how="left")
    else:
        df["accepted_ko_count"] = np.nan

    for table_name, table, col in [
        ("mcyc", mcyc, "mcyc_best_genes"),
        ("scyc", scyc, "scyc_best_genes"),
    ]:
        if not table.empty:
            best = table.loc[table.get("hit_rank_bitscore", 1).eq(1)] if "hit_rank_bitscore" in table else table
            counts = best.groupby("proteome_id")["gene_id"].nunique()
            df = df.merge(counts.rename(col), on="proteome_id", how="left")
        else:
            df[col] = np.nan

    df["qc_tier"] = np.select(
        [
            (df["completeness"] >= 90) & (df["contamination"] <= 5) & (df["gunc_pass"] == True),
            (df["completeness"] >= 70) & (df["contamination"] <= 5) & (df["gunc_pass"] == True),
            (df["completeness"] >= 50) & (df["contamination"] <= 10),
        ],
        ["high", "medium", "review"],
        default="caution",
    )

    fig = plt.figure(figsize=(16, 12), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.72, 1.1, 1.1])
    fig.suptitle("MethaNet functional-genomics calibration: compact, QC-aware signal is emerging",
                 x=0.02, ha="left", fontsize=18, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.02, 0.955,
             f"Completed successful MAGs: {len(df)} | Source: curated run records + Parquet shards | Cohort folder: {root.name}",
             ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    kpis = [
        ("Successful MAGs", f"{len(df)}", "All have run records and curated Parquet"),
        ("Median runtime", f"{df['elapsed_min'].median():.1f} min", f"Range {df['elapsed_min'].min():.1f}-{df['elapsed_min'].max():.1f} min"),
        ("Median completeness", f"{df['completeness'].median():.1f}%", f"{(df['completeness'] >= 70).sum()} MAGs >=70%"),
        ("GUNC pass", f"{int((df['gunc_pass'] == True).sum())}/{len(df)}", "Genome-level contamination screen"),
        ("Median accepted KOs", f"{df['accepted_ko_count'].median():.0f}", "Accepted threshold-marked KOfam calls"),
    ]
    for i, (title, value, subtitle) in enumerate(kpis):
        x = 0.01 + i * 0.195
        rect = plt.Rectangle((x, 0.12), 0.18, 0.72, transform=ax0.transAxes,
                             facecolor=TOKENS["panel"], edgecolor=TOKENS["axis"], linewidth=1)
        ax0.add_patch(rect)
        ax0.text(x + 0.015, 0.68, title, transform=ax0.transAxes, fontsize=9.5, color=TOKENS["muted"], va="center")
        ax0.text(x + 0.015, 0.45, value, transform=ax0.transAxes, fontsize=21, color=TOKENS["ink"], fontweight="bold", va="center")
        ax0.text(x + 0.015, 0.25, subtitle, transform=ax0.transAxes, fontsize=8.5, color=TOKENS["muted"], va="center")

    ax1 = fig.add_subplot(gs[1, 0])
    sns.histplot(df["elapsed_min"], bins=10, color=BLUE["base"], edgecolor=BLUE["dark"], ax=ax1)
    ax1.axvline(df["elapsed_min"].median(), color=BLUE["dark"], linestyle="--", linewidth=1)
    ax1.set_title("Runtime is stable", loc="left")
    ax1.set_xlabel("Elapsed minutes per MAG")
    ax1.set_ylabel("MAG count")
    add_panel_label(ax1, "A")
    clean_axis(ax1)

    ax2 = fig.add_subplot(gs[1, 1])
    palette = {"high": OLIVE["base"], "medium": BLUE["base"], "review": GOLD["base"], "caution": ORANGE["base"]}
    sns.scatterplot(data=df, x="completeness", y="contamination", hue="qc_tier", palette=palette,
                    s=58, edgecolor=TOKENS["ink"], linewidth=0.35, ax=ax2)
    ax2.axvline(70, color=NEUTRAL["mid"], linestyle="--", linewidth=1)
    ax2.axhline(5, color=NEUTRAL["mid"], linestyle="--", linewidth=1)
    ax2.set_title("QC tiers separate strong and review MAGs", loc="left")
    ax2.set_xlabel("CheckM2 completeness (%)")
    ax2.set_ylabel("CheckM2 contamination (%)")
    ax2.legend(title="QC tier", frameon=False, fontsize=8, title_fontsize=8, loc="upper left")
    add_panel_label(ax2, "B")
    clean_axis(ax2)

    ax3 = fig.add_subplot(gs[1, 2])
    tax_counts = df["genus"].fillna("unresolved").replace({"g__": "unresolved"}).value_counts().sort_values()
    ax3.barh(tax_counts.index, tax_counts.values, color=GOLD["base"], edgecolor=GOLD["dark"], linewidth=0.8)
    ax3.set_title("Taxonomy is coherent", loc="left")
    ax3.set_xlabel("MAG count")
    ax3.set_ylabel("")
    annotate_bars(ax3, "{:.0f}", offset=max(tax_counts.max() * 0.02, 0.6))
    ax3.set_xlim(0, tax_counts.max() * 1.18)
    add_panel_label(ax3, "C")
    clean_axis(ax3)

    ax4 = fig.add_subplot(gs[2, 0])
    yield_long = df[["accepted_ko_count", "mcyc_best_genes", "scyc_best_genes", "dbcan_rows"]].rename(columns={
        "accepted_ko_count": "Accepted KOs",
        "mcyc_best_genes": "MCycDB best-hit genes",
        "scyc_best_genes": "SCycDB best-hit genes",
        "dbcan_rows": "dbCAN rows",
    }).melt(var_name="Layer", value_name="Count").dropna()
    order = ["Accepted KOs", "MCycDB best-hit genes", "SCycDB best-hit genes", "dbCAN rows"]
    sns.boxplot(data=yield_long, y="Layer", x="Count", order=order, color=BLUE["light"],
                fliersize=2, linewidth=0.9, ax=ax4)
    sns.stripplot(data=yield_long, y="Layer", x="Count", order=order, color=BLUE["dark"],
                  size=2.5, alpha=0.45, ax=ax4)
    ax4.set_title("Functional yield is broad", loc="left")
    ax4.set_xlabel("Calls or best-hit genes per MAG")
    ax4.set_ylabel("")
    ax4.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    add_panel_label(ax4, "D")
    clean_axis(ax4)

    ax5 = fig.add_subplot(gs[2, 1])
    if not dbcan.empty:
        import re
        families = []
        for col in [c for c in ["HMMER", "dbCAN_sub", "DIAMOND"] if c in dbcan.columns]:
            for val in dbcan[col].dropna().astype(str):
                families.extend(re.findall(r"\b(?:GH|GT|CE|CBM|PL|AA)\d+\b", val))
        fam_counts = pd.Series(families).value_counts().head(10).sort_values()
        if not fam_counts.empty:
            ax5.barh(fam_counts.index, fam_counts.values, color=OLIVE["base"], edgecolor=OLIVE["dark"], linewidth=0.8)
            annotate_bars(ax5, "{:.0f}", offset=max(fam_counts.max() * 0.02, 1))
            ax5.set_xlim(0, fam_counts.max() * 1.18)
    ax5.set_title("CAZy calls highlight substrate capacity", loc="left")
    ax5.set_xlabel("Family evidence count")
    ax5.set_ylabel("")
    add_panel_label(ax5, "E")
    clean_axis(ax5)

    ax6 = fig.add_subplot(gs[2, 2])
    metrics = df[["proteome_id", "accepted_ko_count", "mcyc_best_genes", "scyc_best_genes", "dbcan_rows", "completeness"]].copy()
    metrics = metrics.set_index("proteome_id")
    scaled = metrics.rank(pct=True)
    scaled = scaled.sort_values("completeness", ascending=False).head(min(32, len(scaled)))
    sns.heatmap(scaled.T, cmap=sns.light_palette(BLUE["mid"], as_cmap=True), cbar_kws={"label": "Percentile"},
                linewidths=0.2, linecolor=TOKENS["grid"], ax=ax6)
    ax6.set_title("MAG-level signal is multi-layered", loc="left")
    ax6.set_xlabel("Successful MAGs, sorted by completeness")
    ax6.set_ylabel("")
    ax6.set_xticks([])
    add_panel_label(ax6, "F")

    png = out_dir / "functional_calibration_panel.png"
    pdf = out_dir / "functional_calibration_panel.pdf"
    csv = out_dir / "functional_calibration_panel_source.csv"
    fig.savefig(png, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=TOKENS["surface"])
    df.to_csv(csv, index=False)
    plt.close(fig)
    return png, pdf, csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/functional_metagenomics/fgx_662_apollo3_20260612"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/functional_metagenomics/fgx_662_apollo3_20260612/figures/preliminary_functional_panel"))
    args = parser.parse_args()
    png, pdf, csv = build_panel(args.root, args.out_dir)
    print(f"PNG\t{png}")
    print(f"PDF\t{pdf}")
    print(f"CSV\t{csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
