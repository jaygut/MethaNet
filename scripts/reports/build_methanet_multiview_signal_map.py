#!/usr/bin/env python3
"""Build a MethaNet multi-view signal map from ESM-2, gLM2, and functional evidence.

The figure is designed as a pattern-finding artifact, not a final MRV score:
it highlights MAG/proteome units where ESM-2 bridge geometry, gLM2
gene-order-sensitive context, and functional annotation readiness intersect.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


DEFAULT_ESM_DIR = Path(
    "results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts"
)
DEFAULT_GLM_DIR = Path("results/contextual_genomics/glm2_integration_20260615_092737")
DEFAULT_FUNCTIONAL_RUN_DIRS = [
    Path("results/functional_metagenomics/fgx_662_apollo3_20260612"),
    Path("results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean"),
    Path("results/functional_metagenomics/msm_china_2025_20260615"),
]
DEFAULT_POC_MANIFEST = Path(
    "results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/"
    "poc_662_functional_mag_manifest.mag_bin_only.tsv"
)
DEFAULT_MSM_MANIFEST = Path(
    "results/functional_metagenomics/msm_china_2025_20260615/"
    "manifests/msm_china_2025_functional_mag_manifest.tsv"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/figures/methanet_multiview_signal_map_20260615"
)


ECOSYSTEM_COLORS = {
    "rumen": "#4f46e5",
    "wetland": "#0f766e",
    "mangrove": "#0ea5e9",
    "unknown": "#64748b",
    "other": "#64748b",
}

READINESS_COLORS = {
    "triangulated_now": "#047857",
    "glm_only_wait_function": "#f59e0b",
    "function_only_wait_glm": "#7c3aed",
    "latent_only_pending": "#94a3b8",
    "non_poc_or_unscoped": "#cbd5e1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--esm-artifact-dir", type=Path, default=DEFAULT_ESM_DIR)
    parser.add_argument("--glm-integration-dir", type=Path, default=DEFAULT_GLM_DIR)
    parser.add_argument(
        "--functional-run-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Functional run directory containing per_mag/. May be repeated. "
            "Defaults to current POC + MSM production roots."
        ),
    )
    parser.add_argument(
        "--functional-warehouse-dir",
        type=Path,
        help="Optional consolidated functional cohort warehouse with Parquet features.",
    )
    parser.add_argument("--poc-manifest", type=Path, default=DEFAULT_POC_MANIFEST)
    parser.add_argument("--msm-manifest", type=Path, default=DEFAULT_MSM_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--snapshot-label",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Label printed on the figure/report.",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def normalize_proteome_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample" in df.columns and "proteome_id" not in df.columns:
        df = df.rename(columns={"sample": "proteome_id"})
    if "proteome_id" in df.columns:
        df["proteome_id"] = df["proteome_id"].astype(str)
    return df


def load_esm_artifacts(esm_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    projection = normalize_proteome_id(read_tsv(esm_dir / "embedding_projection_clusters.tsv"))
    metadata = normalize_proteome_id(read_tsv(esm_dir / "embedding_metadata.tsv"))
    bridge = normalize_proteome_id(read_tsv(esm_dir / "bridge_top_candidates.tsv"))
    if "rank" not in bridge.columns and not bridge.empty:
        bridge["rank"] = np.arange(1, len(bridge) + 1)
    keep_meta = [
        col
        for col in [
            "proteome_id",
            "source",
            "ecosystem",
            "domain",
            "source_analysis_accession",
            "n_proteins_used",
        ]
        if col in metadata.columns
    ]
    if keep_meta and not projection.empty:
        projection = projection.merge(
            metadata[keep_meta].drop_duplicates("proteome_id"),
            on="proteome_id",
            how="left",
            suffixes=("", "_metadata"),
        )
        for col in ["source", "ecosystem", "domain", "n_proteins_used"]:
            meta_col = f"{col}_metadata"
            if meta_col in projection.columns:
                projection[col] = projection[col].fillna(projection[meta_col])
                projection = projection.drop(columns=[meta_col])
    return projection, metadata, bridge


def load_glm_features(glm_dir: Path) -> pd.DataFrame:
    feature_path = glm_dir / "feature_glm_mag_level.tsv"
    glm = normalize_proteome_id(read_tsv(feature_path))
    if glm.empty:
        return glm
    for col in ["native_embedding_std_mean", "shuffled_embedding_std_mean"]:
        glm[col] = pd.to_numeric(glm.get(col), errors="coerce")
    glm["glm_context_delta"] = (
        glm["native_embedding_std_mean"] - glm["shuffled_embedding_std_mean"]
    )
    glm["glm_context_ratio"] = (
        glm["native_embedding_std_mean"]
        / glm["shuffled_embedding_std_mean"].replace(0, np.nan)
    )
    native = pd.to_numeric(glm.get("native_window_count"), errors="coerce").fillna(0)
    shuffled = pd.to_numeric(glm.get("shuffled_control_count"), errors="coerce").fillna(0)
    glm["glm_has_native_and_control"] = (native > 0) & (shuffled > 0)
    glm["glm_all_finite"] = glm.get("all_embeddings_finite", True).astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    return glm


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = normalize_proteome_id(read_tsv(path))
    if manifest.empty:
        return manifest
    keep = [
        col
        for col in [
            "proteome_id",
            "mag_id",
            "source",
            "ecosystem",
            "domain",
            "analysis_unit_type",
            "claim_scope",
            "mbag_mag_level_include",
            "comparability_status",
            "recommended_action",
        ]
        if col in manifest.columns
    ]
    return manifest[keep].drop_duplicates("proteome_id")


def build_payload_status_table(
    manifest: pd.DataFrame,
    glm: pd.DataFrame,
    functional: pd.DataFrame,
    *,
    payload_name: str,
    cohort_label: str,
    has_esm2: bool,
) -> pd.DataFrame:
    """Build a cohort-level availability table for non-ESM expansion payloads."""

    if manifest.empty:
        return pd.DataFrame()
    status = manifest.copy()
    status["cohort_label"] = cohort_label
    status["has_esm2"] = bool(has_esm2)
    if "ecosystem" not in status.columns:
        status["ecosystem"] = cohort_label
    if "source" not in status.columns:
        status["source"] = cohort_label

    glm_payload = glm[glm.get("payload_name", "").astype(str).eq(payload_name)].copy()
    if not glm_payload.empty:
        keep = [
            col
            for col in [
                "proteome_id",
                "payload_name",
                "native_window_count",
                "shuffled_control_count",
                "glm_context_delta",
                "glm_context_ratio",
                "glm_has_native_and_control",
                "glm_all_finite",
            ]
            if col in glm_payload.columns
        ]
        status = status.merge(glm_payload[keep].drop_duplicates("proteome_id"), on="proteome_id", how="left")
    status["has_glm"] = status.get("glm_context_delta", pd.Series(np.nan, index=status.index)).notna()

    if not functional.empty:
        status = status.merge(functional, on="proteome_id", how="left")
    status["has_functional"] = status.get("functional_complete", False).fillna(False).astype(bool)
    status["readiness_class"] = "latent_or_manifest_only"
    status.loc[status["has_glm"] & status["has_functional"], "readiness_class"] = "triangulated_now"
    status.loc[status["has_glm"] & ~status["has_functional"], "readiness_class"] = "glm_only_wait_function"
    status.loc[~status["has_glm"] & status["has_functional"], "readiness_class"] = "function_only_wait_glm"
    status["claim_boundary"] = "MAG/proteome molecular screening; target-domain expansion is not sample-level risk."
    return status


def discover_functional_status(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_root in run_dirs:
        per_mag = run_root / "per_mag"
        if not per_mag.exists():
            continue
        for run_dir in sorted(path for path in per_mag.glob("*/*") if path.is_dir()):
            proteome_id = run_dir.parent.name
            run_record = run_dir / "curated/run_record.json"
            record: dict[str, Any] = {}
            if run_record.exists():
                try:
                    record = json.loads(run_record.read_text())
                except Exception:
                    record = {}
            status = record.get("status")
            if not status:
                if (run_dir / "COMPLETE").exists():
                    status = "complete"
                elif (run_dir / "FAILED").exists():
                    status = "failed"
                else:
                    status = "partial_or_pending"
            rows.append(
                {
                    "proteome_id": str(record.get("proteome_id") or proteome_id),
                    "functional_run_root": str(run_root),
                    "functional_run_id": record.get("run_id") or run_dir.name,
                    "functional_mag_id": record.get("mag_id") or proteome_id,
                    "functional_status": str(status).lower(),
                    "functional_complete": bool(
                        str(status).lower() == "complete" and (run_dir / "COMPLETE").exists()
                    ),
                    "has_parquet_manifest": (run_dir / "curated/parquet_manifest.tsv").exists(),
                    "mtime_epoch": run_dir.stat().st_mtime,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["proteome_id", "functional_complete"])
    status = pd.DataFrame(rows).sort_values("mtime_epoch")
    return status.drop_duplicates("proteome_id", keep="last").drop(columns=["mtime_epoch"])


def read_warehouse_feature(warehouse_dir: Path | None, table: str) -> pd.DataFrame:
    if warehouse_dir is None:
        return pd.DataFrame()
    table_root = warehouse_dir / "parquet" / table
    if not table_root.exists():
        return pd.DataFrame()
    parts = sorted(table_root.glob("*/*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = []
    for part in parts:
        try:
            frames.append(pd.read_parquet(part))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return normalize_proteome_id(pd.concat(frames, ignore_index=True))


def build_joined_table(
    projection: pd.DataFrame,
    bridge: pd.DataFrame,
    glm: pd.DataFrame,
    manifest: pd.DataFrame,
    functional: pd.DataFrame,
    warehouse_dir: Path | None,
) -> pd.DataFrame:
    joined = projection.copy()
    if joined.empty:
        raise SystemExit("No ESM-2 projection rows were found; cannot build the POC signal map.")
    joined = normalize_proteome_id(joined)

    if not manifest.empty:
        joined = joined.merge(
            manifest,
            on="proteome_id",
            how="left",
            suffixes=("", "_manifest"),
        )
        joined["in_poc_magbin_manifest"] = joined["analysis_unit_type"].notna()
        for col in ["source", "ecosystem", "domain"]:
            alt = f"{col}_manifest"
            if alt in joined.columns:
                joined[col] = joined[col].fillna(joined[alt])
                joined = joined.drop(columns=[alt])
    else:
        joined["in_poc_magbin_manifest"] = True

    glm_keep = [
        col
        for col in [
            "proteome_id",
            "mag_id",
            "payload_name",
            "native_window_count",
            "shuffled_control_count",
            "all_embeddings_finite",
            "embedding_dim",
            "native_embedding_std_mean",
            "shuffled_embedding_std_mean",
            "glm_context_delta",
            "glm_context_ratio",
            "glm_has_native_and_control",
            "glm_all_finite",
            "context_qc_tier",
        ]
        if col in glm.columns
    ]
    if glm_keep:
        # Prefer MAG-bin rows for the POC view; assembly context remains visible only by status.
        glm_join = glm[glm_keep].copy()
        payload = glm_join.get("payload_name", "").astype(str)
        glm_join["_glm_priority"] = np.select(
            [
                payload.isin(["poc_magbin_available", "poc_magbin_catchup"]),
                payload.eq("poc_assembly_context_available"),
            ],
            [0, 2],
            default=1,
        )
        glm_join = glm_join.sort_values(["proteome_id", "_glm_priority"]).drop_duplicates(
            "proteome_id", keep="first"
        )
        joined = joined.merge(glm_join.drop(columns=["_glm_priority"]), on="proteome_id", how="left")

    if not functional.empty:
        joined = joined.merge(functional, on="proteome_id", how="left")
    joined["functional_complete"] = joined.get("functional_complete", False).fillna(False).astype(bool)

    feature_mrv = read_warehouse_feature(warehouse_dir, "feature_mrv_mag_level")
    feature_methane = read_warehouse_feature(warehouse_dir, "feature_methane_mechanism")
    feature_sulfur = read_warehouse_feature(warehouse_dir, "feature_sulfur_competition")
    for table_name, frame in [
        ("mrv", feature_mrv),
        ("methane", feature_methane),
        ("sulfur", feature_sulfur),
    ]:
        if frame.empty:
            continue
        cols = ["proteome_id"] + [
            c
            for c in frame.columns
            if c not in joined.columns and c not in {"cohort_run_id", "run_id", "mag_id", "source_tool"}
        ]
        keep = [c for c in cols if c in frame.columns]
        joined = joined.merge(
            frame[keep].drop_duplicates("proteome_id"),
            on="proteome_id",
            how="left",
            suffixes=("", f"_{table_name}"),
        )

    if not bridge.empty:
        bridge_keep = [
            col
            for col in [
                "proteome_id",
                "rank",
                "wetland_projection",
                "bridging_score",
                "mixing_coeff",
            ]
            if col in bridge.columns
        ]
        bridge_join = bridge[bridge_keep].copy()
        bridge_join["is_top_bridge_candidate"] = True
        joined = joined.merge(
            bridge_join.drop_duplicates("proteome_id"),
            on="proteome_id",
            how="left",
            suffixes=("", "_bridge_top"),
        )
    joined["is_top_bridge_candidate"] = joined.get("is_top_bridge_candidate", False).fillna(False)

    # Numeric cleanup and derived scores.
    for col in [
        "bridging_score",
        "mixing_coeff",
        "wetland_projection",
        "glm_context_delta",
        "glm_context_ratio",
        "native_embedding_std_mean",
        "shuffled_embedding_std_mean",
        "umap_1",
        "umap_2",
    ]:
        if col in joined.columns:
            joined[col] = pd.to_numeric(joined[col], errors="coerce")

    joined["has_glm"] = joined["glm_context_delta"].notna()
    joined["has_functional"] = joined["functional_complete"].fillna(False).astype(bool)
    joined.loc[~joined["in_poc_magbin_manifest"], ["has_glm", "has_functional"]] = False
    joined["has_esm2"] = True
    joined["ecosystem"] = joined.get("ecosystem", "unknown").fillna("unknown").astype(str)
    joined["source"] = joined.get("source", "unknown").fillna("unknown").astype(str)
    joined["domain"] = joined.get("domain", "Unknown").fillna("Unknown").astype(str)

    joined["readiness_class"] = "latent_only_pending"
    joined.loc[joined["has_glm"] & joined["has_functional"], "readiness_class"] = "triangulated_now"
    joined.loc[joined["has_glm"] & ~joined["has_functional"], "readiness_class"] = "glm_only_wait_function"
    joined.loc[~joined["has_glm"] & joined["has_functional"], "readiness_class"] = "function_only_wait_glm"
    if "analysis_unit_type" in joined.columns:
        unit_type = joined["analysis_unit_type"].fillna("non_poc_or_unscoped").astype(str)
        non_mag = unit_type.ne("mag_bin")
        joined.loc[non_mag, "readiness_class"] = "non_poc_or_unscoped"

    delta = joined["glm_context_delta"]
    if delta.notna().sum() > 1:
        joined["glm_context_delta_z"] = (delta - delta.mean()) / delta.std(ddof=0)
    else:
        joined["glm_context_delta_z"] = np.nan
    joined["glm_context_delta_z"] = joined["glm_context_delta_z"].replace([np.inf, -np.inf], np.nan)

    bridge_component = joined.get("mixing_coeff", pd.Series(0.0, index=joined.index)).fillna(0.0)
    context_component = joined["glm_context_delta_z"].clip(lower=0).fillna(0.0)
    functional_component = joined["has_functional"].astype(float)
    joined["triangulation_signal"] = (
        0.45 * bridge_component + 0.35 * context_component + 0.20 * functional_component
    )
    joined["claim_boundary"] = (
        "MAG/proteome molecular screening; not sample-level methane risk or flux."
    )
    return joined


def _plot_status_strip(ax: plt.Axes, counts: Counter[str]) -> None:
    labels = [
        ("triangulated_now", "ESM-2 + gLM2 + functional"),
        ("glm_only_wait_function", "ESM-2 + gLM2"),
        ("function_only_wait_glm", "ESM-2 + functional"),
        ("latent_only_pending", "ESM-2 only"),
    ]
    total = sum(counts.values()) or 1
    left = 0.0
    for key, label in labels:
        width = counts.get(key, 0) / total
        ax.barh([0], [width], left=[left], color=READINESS_COLORS[key], edgecolor="white", height=0.55)
        if width > 0.075:
            ax.text(left + width / 2, 0, str(counts.get(key, 0)), ha="center", va="center", color="white", weight="bold")
        left += width
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Layer coverage across ESM-2 POC", loc="left", fontsize=11, weight="bold")
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=READINESS_COLORS[key], label=label)
        for key, label in labels
    ]
    ax.legend(handles=legend_handles, ncol=2, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -1.1))


def _scatter_esm(ax: plt.Axes, joined: pd.DataFrame, top_n: int) -> None:
    order = ["latent_only_pending", "function_only_wait_glm", "glm_only_wait_function", "triangulated_now"]
    for cls in order:
        sub = joined[joined["readiness_class"].eq(cls)]
        if sub.empty:
            continue
        colors = [
            ECOSYSTEM_COLORS.get(str(x).lower(), ECOSYSTEM_COLORS["other"])
            for x in sub["ecosystem"]
        ]
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=np.where(sub["has_glm"], 42, 20),
            c=colors,
            marker="o",
            alpha=0.28 if cls == "latent_only_pending" else 0.80,
            edgecolors=READINESS_COLORS.get(cls, "#334155"),
            linewidths=0.75,
            label=cls,
        )
    top = joined[joined["is_top_bridge_candidate"]].sort_values("rank").head(top_n)
    if not top.empty:
        ax.scatter(
            top["umap_1"],
            top["umap_2"],
            s=180,
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=2.2,
            zorder=7,
        )
        for _, row in top.iterrows():
            if pd.notna(row.get("umap_1")) and pd.notna(row.get("umap_2")):
                ax.text(
                    row["umap_1"],
                    row["umap_2"],
                    str(int(row["rank"])),
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                    color="#111827",
                    zorder=8,
                )
    ax.set_title("A. ESM-2 bridge geometry with gLM2 / functional readiness", loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(color="#e2e8f0", linewidth=0.7, alpha=0.7)


def _scatter_context(ax: plt.Axes, joined: pd.DataFrame, top_n: int) -> None:
    sub = joined[joined["has_glm"]].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No gLM2 features available", ha="center", va="center")
        return
    x = sub.get("mixing_coeff", pd.Series(0.0, index=sub.index)).fillna(0.0)
    y = sub["glm_context_delta"].fillna(0.0)
    med_x = float(np.nanmedian(x)) if len(x) else 0.0
    med_y = float(np.nanmedian(y)) if len(y) else 0.0
    rng = np.random.default_rng(20260615)
    sub["mixing_coeff_plot"] = sub.get("mixing_coeff", 0).fillna(0.0).astype(float)
    zeroish = sub["mixing_coeff_plot"].abs() < 1e-9
    sub.loc[zeroish, "mixing_coeff_plot"] = rng.uniform(-0.012, 0.012, int(zeroish.sum()))
    for eco, eco_sub in sub.groupby(sub["ecosystem"].str.lower()):
        colors = ECOSYSTEM_COLORS.get(eco, ECOSYSTEM_COLORS["other"])
        ax.scatter(
            eco_sub["mixing_coeff_plot"],
            eco_sub["glm_context_delta"],
            s=np.where(eco_sub["has_functional"], 54, 30),
            c=colors,
            edgecolors=np.where(eco_sub["has_functional"], "#0f172a", "#cbd5e1"),
            linewidths=0.7,
            alpha=0.78,
            label=eco,
        )
    ax.axvline(med_x, color="#94a3b8", linestyle="--", linewidth=1)
    ax.axhline(med_y, color="#94a3b8", linestyle="--", linewidth=1)
    ax.text(
        0.98,
        0.96,
        "context-supported\nbridge zone",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#064e3b",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ecfdf5", ec="#a7f3d0", lw=0.8),
    )
    top = joined[joined["is_top_bridge_candidate"] & joined["has_glm"]].sort_values("rank").head(top_n)
    top = top.merge(sub[["proteome_id", "mixing_coeff_plot"]], on="proteome_id", how="left")
    for _, row in top.iterrows():
        ax.text(
            float(row.get("mixing_coeff_plot") or row.get("mixing_coeff") or 0.0),
            float(row.get("glm_context_delta") or 0.0),
            str(int(row["rank"])),
            fontsize=8,
            weight="bold",
            color="#111827",
        )
    ax.set_title("B. Latent bridge affinity vs gLM2 order-sensitive context", loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("ESM-2 bridge affinity (zero values lightly jittered)")
    ax.set_ylabel("gLM2 native-minus-shuffled context delta")
    ax.grid(color="#e2e8f0", linewidth=0.7, alpha=0.7)


def _bridge_panel(ax: plt.Axes, joined: pd.DataFrame, top_n: int) -> None:
    top = joined[joined["is_top_bridge_candidate"]].sort_values("rank").head(top_n).copy()
    if top.empty:
        ax.text(0.5, 0.5, "No bridge candidate table available", ha="center", va="center")
        ax.axis("off")
        return
    top["display"] = (
        top["proteome_id"]
        .str.replace("rumen__", "r__", regex=False)
        .str.replace("mucc__", "m__", regex=False)
        .str.replace("_idba_bin.", "_bin.", regex=False)
        .str.replace("_ASM249546v1_genomic", "", regex=False)
        .str.replace("_genomic", "", regex=False)
    )
    top["display"] = top["display"].str.slice(0, 33)
    y = np.arange(len(top))[::-1]
    score = top["triangulation_signal"].fillna(0.0)
    ax.barh(y, score, color=[READINESS_COLORS.get(c, "#94a3b8") for c in top["readiness_class"]], alpha=0.88)
    labels = [f"#{int(row['rank'])} {row['display']}" for _, row in top.iterrows()]
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", labelsize=8.0, pad=2)
    for yi, (_, row) in zip(y, top.iterrows()):
        status = [
            "gLM+" if row["has_glm"] else "gLM-",
            "func+" if row["has_functional"] else "func-",
        ]
        ax.text(
            float(row["triangulation_signal"] or 0.0) + 0.015,
            yi,
            "  ".join(status),
            va="center",
            ha="left",
            fontsize=7.8,
            color="#334155",
        )
    xmax = max(1.25, float(score.max() * 1.33 if len(score) else 1.25))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Provisional triangulation signal", fontsize=9)
    ax.set_title("C. Top latent bridge candidates: where evidence triangulates now", loc="left", fontsize=12, weight="bold")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7, alpha=0.7)


def _readiness_matrix(ax: plt.Axes, joined: pd.DataFrame, msm_status: pd.DataFrame | None = None) -> None:
    categories = ["rumen", "wetland"]
    metrics = [
        ("ESM-2", "has_esm2"),
        ("gLM2", "has_glm"),
        ("Functional", "has_functional"),
        ("All 3", "all3"),
        ("Top Bridge", "is_top_bridge_candidate"),
    ]
    data = []
    labels = []
    for eco in categories:
        sub = joined[joined["ecosystem"].str.lower().eq(eco)]
        denom = max(len(sub), 1)
        row = []
        for _, col in metrics:
            if col == "all3":
                value = (sub["has_esm2"] & sub["has_glm"] & sub["has_functional"]).sum() / denom
            else:
                value = sub[col].fillna(False).astype(bool).sum() / denom
            row.append(value)
        data.append(row)
        labels.append(f"{eco} (n={len(sub)})")
    if msm_status is not None and not msm_status.empty:
        sub = msm_status
        denom = max(len(sub), 1)
        data.append(
            [
                0.0,
                sub["has_glm"].fillna(False).astype(bool).sum() / denom,
                sub["has_functional"].fillna(False).astype(bool).sum() / denom,
                (sub["has_glm"].fillna(False).astype(bool) & sub["has_functional"].fillna(False).astype(bool)).sum()
                / denom,
                0.0,
            ]
        )
        labels.append(f"MSM mangrove (n={len(sub)})")
    matrix = np.asarray(data)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), [m[0] for m in metrics], rotation=24, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_color = "#ffffff" if matrix[i, j] >= 0.58 else "#0f172a"
            ax.text(j, i, f"{matrix[i, j]*100:.0f}%", ha="center", va="center", fontsize=9, color=text_color, weight="bold")
    ax.set_title("D. Evidence coverage by ecosystem / payload", loc="left", fontsize=12, weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="coverage fraction")


def make_figure(
    joined: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    top_n: int,
    snapshot_label: str,
    msm_status: pd.DataFrame | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
        }
    )
    fig = plt.figure(figsize=(20, 12.4), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        6,
        height_ratios=[0.58, 3.25, 2.65, 0.80],
        hspace=0.72,
        wspace=0.70,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.80,
        "MethaNet Multi-View Bridge Signal Map",
        fontsize=23,
        weight="bold",
        color="#061b3a",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.0,
        0.25,
        textwrap.fill(
            "Spotting MAG/proteome candidates where ESM-2 geometry, gLM2 genomic context, and functional readiness converge",
            width=112,
        ),
        fontsize=11.8,
        color="#0f766e",
        style="italic",
        ha="left",
        va="center",
    )
    title_ax.text(
        1.0,
        0.50,
        f"snapshot: {snapshot_label}",
        fontsize=10,
        color="#475569",
        ha="right",
        va="center",
    )

    ax_a = fig.add_subplot(gs[1, :3])
    ax_b = fig.add_subplot(gs[1, 3:])
    ax_c = fig.add_subplot(gs[2, :3])
    ax_d = fig.add_subplot(gs[2, 3:])
    ax_strip = fig.add_subplot(gs[3, :])

    _scatter_esm(ax_a, joined, top_n)
    _scatter_context(ax_b, joined, top_n)
    _bridge_panel(ax_c, joined, top_n)
    _readiness_matrix(ax_d, joined, msm_status)
    _plot_status_strip(ax_strip, Counter(joined["readiness_class"]))

    ecosystem_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=label, markersize=8)
        for label, color in [
            ("rumen", ECOSYSTEM_COLORS["rumen"]),
            ("wetland", ECOSYSTEM_COLORS["wetland"]),
            ("mangrove/MSM", ECOSYSTEM_COLORS["mangrove"]),
        ]
    ]
    fig.legend(
        handles=ecosystem_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.945),
        frameon=False,
        ncol=3,
        fontsize=9,
    )

    fig.text(
        0.5,
        0.015,
        "Claim boundary: MAG/proteome molecular screening and bridge prioritization only; sample-level risk requires abundance, environment, uncertainty, and flux validation.",
        ha="center",
        va="center",
        fontsize=10,
        color="#7c2d12",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff7ed", ec="#fdba74", lw=1),
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    readme = f"""# MethaNet Multi-View Signal Map

Generated: {args.snapshot_label}

Purpose: identify MAG/proteome candidates where ESM-2 bridge geometry,
gLM2 genomic context, and functional annotation readiness converge. This is a
pattern-finding and prioritization visualization, not a final MRV scoring
artifact.

## Files

- `methanet_multiview_signal_map.png`: high-resolution figure.
- `methanet_multiview_signal_map.pdf`: vector-backed report figure.
- `joined_multiview_signal_table.tsv`: joined plotting table keyed by `proteome_id`.
- `top_bridge_multiview_candidates.tsv`: top bridge candidates with gLM2 and functional status.
- `msm_mangrove_payload_status.tsv`: MSM target-domain payload status, when provided.
- `readiness_summary.json`: counts and paths used.
- `ARTIFACT_REVIEW.md`: layout QA notes, current counts, and claim boundary.

## Reproducible Command

Run from the MethaNet repo root:

```bash
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
conda activate methanet-fgx
python scripts/reports/build_methanet_multiview_signal_map.py \\
  --output-dir {args.output_dir} \\
  --snapshot-label {args.snapshot_label}
```

## Strategic Read

The key pattern to look for is the upper-right zone of panel B: high ESM-2
bridge affinity plus positive native-minus-shuffled gLM2 context signal. When
those points also have functional completion, they are the strongest candidates
for MethaNet bridge attestation cards.

## Current Summary

```json
{json.dumps(summary, indent=2)}
```

## Claim Boundary

This figure supports MAG/proteome-level molecular screening, bridge candidate
prioritization, and validation planning. It does not support final sample-level
methane-risk tiers, measured flux claims, carbon-credit approval, or replacement
of abundance, environmental, uncertainty, and flux validation layers.
"""
    (output_dir / "README.md").write_text(readme)
    review = f"""# Artifact Review

## Layout QA

- Re-rendered with wider canvas, expanded grid spacing, shorter axis labels, and y-axis candidate labels to prevent text overflow in the bridge-candidate panel.
- Bottom row was rebalanced so panel C is narrower and panel D has enough room for ecosystem labels and the colorbar.
- Panel B now uses compact language for the latent bridge affinity axis.
- Panel D uses smaller rotated labels and contrast-aware cell text.
- The legend and claim boundary are separated from the analytical panels.

## Current Counts

```json
{json.dumps({
    "esm_rows": summary["esm_rows"],
    "glm_feature_rows": summary["glm_feature_rows"],
    "functional_status_rows": summary["functional_status_rows"],
    "readiness_counts": summary["readiness_counts"],
    "top_bridge_with_all_three": summary["top_bridge_with_all_three"],
    "msm_with_glm_and_functional": summary["msm_with_glm_and_functional"],
}, indent=2)}
```

## Claim Boundary

This is MAG/proteome molecular screening and bridge prioritization. It does not
assign final sample-level methane-risk tiers, measured flux, or carbon-credit
approval.
"""
    (output_dir / "ARTIFACT_REVIEW.md").write_text(review)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    esm_dir = resolve(repo_root, args.esm_artifact_dir)
    glm_dir = resolve(repo_root, args.glm_integration_dir)
    output_dir = resolve(repo_root, args.output_dir)
    poc_manifest = resolve(repo_root, args.poc_manifest)
    msm_manifest_path = resolve(repo_root, args.msm_manifest)
    warehouse_dir = resolve(repo_root, args.functional_warehouse_dir)
    run_dirs = [resolve(repo_root, p) for p in (args.functional_run_dir or DEFAULT_FUNCTIONAL_RUN_DIRS)]
    run_dirs = [p for p in run_dirs if p is not None]

    projection, metadata, bridge = load_esm_artifacts(esm_dir)
    glm = load_glm_features(glm_dir)
    manifest = load_manifest(poc_manifest)
    msm_manifest = load_manifest(msm_manifest_path)
    functional = discover_functional_status(run_dirs)
    msm_status = build_payload_status_table(
        msm_manifest,
        glm,
        functional,
        payload_name="msm_magbin_full",
        cohort_label="MSM mangrove",
        has_esm2=False,
    )

    joined = build_joined_table(
        projection=projection,
        bridge=bridge,
        glm=glm,
        manifest=manifest,
        functional=functional,
        warehouse_dir=warehouse_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = output_dir / "joined_multiview_signal_table.tsv"
    joined.to_csv(joined_path, sep="\t", index=False)

    top_bridge = joined[joined["is_top_bridge_candidate"]].sort_values("rank").head(args.top_n)
    top_bridge.to_csv(output_dir / "top_bridge_multiview_candidates.tsv", sep="\t", index=False)
    if not msm_status.empty:
        msm_status.to_csv(output_dir / "msm_mangrove_payload_status.tsv", sep="\t", index=False)

    summary = {
        "esm_rows": int(len(projection)),
        "glm_feature_rows": int(len(glm)),
        "functional_status_rows": int(len(functional)),
        "joined_rows": int(len(joined)),
        "readiness_counts": {k: int(v) for k, v in Counter(joined["readiness_class"]).items()},
        "top_bridge_rows": int(len(top_bridge)),
        "top_bridge_with_glm": int(top_bridge["has_glm"].sum()) if not top_bridge.empty else 0,
        "top_bridge_with_functional": int(top_bridge["has_functional"].sum()) if not top_bridge.empty else 0,
        "top_bridge_with_all_three": int(
            (top_bridge["has_esm2"] & top_bridge["has_glm"] & top_bridge["has_functional"]).sum()
        )
        if not top_bridge.empty
        else 0,
        "msm_rows": int(len(msm_status)),
        "msm_with_glm": int(msm_status["has_glm"].sum()) if not msm_status.empty else 0,
        "msm_with_functional": int(msm_status["has_functional"].sum()) if not msm_status.empty else 0,
        "msm_with_glm_and_functional": int(
            (msm_status["has_glm"] & msm_status["has_functional"]).sum()
        )
        if not msm_status.empty
        else 0,
        "claim_boundary": "MAG/proteome molecular screening; not sample-level risk or flux.",
        "inputs": {
            "esm_artifact_dir": str(esm_dir),
            "glm_integration_dir": str(glm_dir),
            "functional_run_dirs": [str(p) for p in run_dirs],
            "functional_warehouse_dir": str(warehouse_dir) if warehouse_dir else None,
            "poc_manifest": str(poc_manifest),
            "msm_manifest": str(msm_manifest_path),
        },
    }
    (output_dir / "readiness_summary.json").write_text(json.dumps(summary, indent=2))

    make_figure(
        joined,
        output_png=output_dir / "methanet_multiview_signal_map.png",
        output_pdf=output_dir / "methanet_multiview_signal_map.pdf",
        top_n=args.top_n,
        snapshot_label=args.snapshot_label,
        msm_status=msm_status,
    )
    write_readme(output_dir, args, summary)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
