#!/usr/bin/env python3
"""Build an interim Blue Catalyst analytics brief from partial checkpoint artifacts.

This script does two things in one pass:
1) Copies the currently available run artifacts into a dedicated local snapshot folder.
2) Aggregates checkpoint embeddings and generates rich analytics outputs/figures.

Typical usage (after fetching artifacts from Apolo to local):

  python scripts/blue_catalyst_partial_intel_report.py \
    --source-artifacts-dir results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts \
    --output-root results/blue_catalyst_poc
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

try:
    import umap as umap_pkg

    HAS_UMAP = True
except ModuleNotFoundError:
    HAS_UMAP = False
    umap_pkg = None

try:
    import plotly.express as px

    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False
    px = None


REQUIRED_META_COLS = [
    "sample",
    "source",
    "ecosystem",
    "domain",
    "source_analysis_accession",
    "n_proteins_used",
]


@dataclass
class LoadedEmbeddings:
    matrix: np.ndarray
    metadata: pd.DataFrame
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy partial Blue Catalyst artifacts into a dedicated snapshot and generate "
            "interim analytics figures/tables."
        )
    )
    parser.add_argument(
        "--source-artifacts-dir",
        required=True,
        type=Path,
        help="Path to a run artifact directory (contains embedding_checkpoints/)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/blue_catalyst_poc"),
        help="Root folder where interim snapshot + analytics are written",
    )
    parser.add_argument(
        "--snapshot-name",
        type=str,
        default="",
        help="Optional custom snapshot folder name",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Neighbors for bridge/mixing analysis",
    )
    parser.add_argument(
        "--max-heatmap-samples",
        type=int,
        default=220,
        help="Maximum genomes plotted in cosine-distance heatmap",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Permutation count for PERMANOVA-style separation test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def infer_run_id_from_source(source_artifacts_dir: Path) -> str:
    runs_dir = source_artifacts_dir.parent
    if runs_dir.name == "runs":
        return source_artifacts_dir.parent.name
    return source_artifacts_dir.parent.name


def ensure_snapshot_dirs(output_root: Path, snapshot_name: str) -> dict[str, Path]:
    snapshot_dir = output_root / "interim_snapshots" / snapshot_name
    paths = {
        "snapshot": snapshot_dir,
        "fetched": snapshot_dir / "fetched_artifacts",
        "analytics": snapshot_dir / "analytics",
        "figures": snapshot_dir / "figures",
        "interactive": snapshot_dir / "interactive",
        "tables": snapshot_dir / "tables",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def copy_available_artifacts(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []

    files_to_copy = [
        "embedding_input_inventory.tsv",
        "embedding_coassembly_excluded.tsv",
        "sample_source_counts.tsv",
        "proteome_sample_manifest.tsv",
        "prjeb31266_selected_subset.tsv",
        "prjeb31266_analysis_manifest.tsv",
        "coassembly_excluded_from_primary.tsv",
        "prjeb31266_source_subset_seed.tsv",
        "embedding_metadata.tsv",
        "embedding_stats.json",
        "genome_embeddings.npz",
        "artifact_manifest.tsv",
        "artifact_manifest.json",
        "artifact_registry.tsv",
        "advanced_analytics_summary.json",
        "poc_metrics.json",
        "bridging_genomes_top.tsv",
        "embedding_projection_clusters.tsv",
    ]

    for name in files_to_copy:
        src = source_dir / name
        if src.exists() and src.is_file():
            dst = target_dir / name
            shutil.copy2(src, dst)
            copied.append(name)

    src_checkpoints = source_dir / "embedding_checkpoints"
    if src_checkpoints.exists() and src_checkpoints.is_dir():
        dst_checkpoints = target_dir / "embedding_checkpoints"
        if dst_checkpoints.exists():
            shutil.rmtree(dst_checkpoints)
        shutil.copytree(src_checkpoints, dst_checkpoints)
        copied.append("embedding_checkpoints/")

    return copied


def load_from_final_npz(artifacts_dir: Path) -> LoadedEmbeddings | None:
    npz_path = artifacts_dir / "genome_embeddings.npz"
    if not npz_path.exists():
        return None

    bundle = np.load(npz_path, allow_pickle=True)
    matrix = bundle["embeddings"].astype(np.float32)

    meta = pd.DataFrame(
        {
            "sample": bundle["sample"].astype(str),
            "source": bundle["source"].astype(str),
            "ecosystem": bundle["ecosystem"].astype(str),
            "domain": bundle["domain"].astype(str),
            "n_proteins_used": bundle["n_proteins_used"].astype(int),
        }
    )
    if "source_analysis_accession" in bundle.files:
        meta["source_analysis_accession"] = bundle["source_analysis_accession"].astype(str)
    else:
        meta["source_analysis_accession"] = ""

    return LoadedEmbeddings(matrix=matrix, metadata=meta, mode="final_npz")


def aggregate_checkpoint_batches(artifacts_dir: Path) -> LoadedEmbeddings:
    checkpoint_dir = artifacts_dir / "embedding_checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    npz_files = sorted(checkpoint_dir.glob("embedding_batch_*.npz"))
    if not npz_files:
        raise RuntimeError(f"No checkpoint npz files found in: {checkpoint_dir}")

    matrices: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []

    for npz_path in npz_files:
        tsv_path = checkpoint_dir / f"{npz_path.stem}.tsv"
        if not tsv_path.exists():
            raise RuntimeError(f"Checkpoint metadata TSV missing: {tsv_path}")

        block = np.load(npz_path, allow_pickle=True)["embeddings"].astype(np.float32)
        block_meta = pd.read_csv(tsv_path, sep="\t")

        if len(block_meta) != block.shape[0]:
            raise RuntimeError(
                "Checkpoint row mismatch: "
                f"{npz_path.name} has {block.shape[0]} embeddings but "
                f"{tsv_path.name} has {len(block_meta)} rows"
            )

        matrices.append(block)
        metas.append(block_meta)

    matrix = np.vstack(matrices)
    meta = pd.concat(metas, ignore_index=True)

    if {"checkpoint_batch", "checkpoint_order"}.issubset(meta.columns):
        meta = meta.sort_values(["checkpoint_batch", "checkpoint_order"]).reset_index(drop=True)

    for col in REQUIRED_META_COLS:
        if col not in meta.columns:
            if col == "n_proteins_used":
                meta[col] = 0
            else:
                meta[col] = ""

    meta["n_proteins_used"] = pd.to_numeric(meta["n_proteins_used"], errors="coerce").fillna(0).astype(int)

    if meta["sample"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate sample IDs detected in checkpoint metadata")

    if len(meta) != matrix.shape[0]:
        raise RuntimeError(
            "Aggregated checkpoint mismatch: "
            f"rows={len(meta)} embeddings={matrix.shape[0]}"
        )

    return LoadedEmbeddings(matrix=matrix, metadata=meta, mode="checkpoint_aggregate")


def attach_inventory_metadata(meta: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    inventory_fp = artifacts_dir / "embedding_input_inventory.tsv"
    if not inventory_fp.exists():
        return meta

    inventory = pd.read_csv(inventory_fp, sep="\t")
    keep_cols = [c for c in ["sample", "size_bytes", "exists", "already_checkpointed"] if c in inventory.columns]
    if "sample" not in keep_cols:
        return meta

    inv = inventory[keep_cols].drop_duplicates(subset=["sample"])
    out = meta.merge(inv, on="sample", how="left")
    out = out.rename(columns={"size_bytes": "proteome_size_bytes"})
    return out


def compute_permanova(labels: np.ndarray, d2: np.ndarray, permutations: int, rng: np.random.Generator) -> dict[str, float]:
    unique = np.unique(labels)
    n = len(labels)
    g = len(unique)

    if g < 2 or n <= g:
        return {"pseudo_f": float("nan"), "p_value": float("nan"), "permutations": permutations}

    def pseudo_f(local_labels: np.ndarray) -> float:
        ss_total = float(d2.sum()) / (2.0 * n)
        ss_within = 0.0
        for val in np.unique(local_labels):
            idx = np.flatnonzero(local_labels == val)
            ng = len(idx)
            if ng <= 1:
                continue
            ss_within += float(d2[np.ix_(idx, idx)].sum()) / (2.0 * ng)

        ss_between = ss_total - ss_within
        df_between = max(1, g - 1)
        df_within = max(1, n - g)
        if ss_within <= 0:
            return float("inf")
        return (ss_between / df_between) / (ss_within / df_within)

    observed = pseudo_f(labels)
    ge = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        if pseudo_f(permuted) >= observed:
            ge += 1

    pval = (ge + 1) / (permutations + 1)
    return {"pseudo_f": float(observed), "p_value": float(pval), "permutations": permutations}


def plot_projection(ax: Any, df: pd.DataFrame, x: str, y: str, title: str) -> None:
    palette = {"rumen": "#9b2226", "wetland": "#0a9396"}
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="ecosystem",
        style="source",
        palette=palette,
        s=42,
        alpha=0.8,
        ax=ax,
        linewidth=0,
    )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def run_analytics(
    loaded: LoadedEmbeddings,
    out_figures: Path,
    out_interactive: Path,
    out_tables: Path,
    max_heatmap_samples: int,
    knn_k: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    meta = loaded.metadata.copy().reset_index(drop=True)
    matrix = loaded.matrix

    finite_mask = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite_mask]
    meta = meta.loc[finite_mask].reset_index(drop=True)

    n = len(meta)
    if n < 5:
        raise RuntimeError(f"Need at least 5 embeddings for analytics; got {n}")

    pca = PCA(n_components=min(20, n, matrix.shape[1]), random_state=seed)
    pcs = pca.fit_transform(matrix)

    if HAS_UMAP and n >= 15:
        um = umap_pkg.UMAP(
            n_neighbors=min(20, max(2, n - 1)),
            min_dist=0.15,
            metric="cosine",
            random_state=seed,
        )
        U = um.fit_transform(matrix)
        umap_mode = "umap"
    else:
        U = pcs[:, :2]
        umap_mode = "pca_fallback"

    tsne_perplexity = min(30, max(5, (n - 1) // 3))
    T = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=seed, init="pca").fit_transform(matrix)

    proj = meta.copy()
    proj["pc1"] = pcs[:, 0]
    proj["pc2"] = pcs[:, 1]
    proj["umap1"] = U[:, 0]
    proj["umap2"] = U[:, 1]
    proj["tsne1"] = T[:, 0]
    proj["tsne2"] = T[:, 1]

    ecosystems = proj["ecosystem"].astype(str).values
    nbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, n), metric="cosine").fit(matrix)
    neigh_idx = nbrs.kneighbors(matrix, return_distance=False)

    opp_frac = []
    for i in range(n):
        nn = neigh_idx[i][1:]
        same = np.mean(ecosystems[nn] == ecosystems[i]) if len(nn) else 1.0
        opp_frac.append(1.0 - same)
    proj["opp_neighbor_fraction"] = np.array(opp_frac, dtype=float)
    proj["bridge_score"] = 1.0 - np.abs(proj["opp_neighbor_fraction"] - 0.5) * 2.0

    bridge_top = proj.sort_values("bridge_score", ascending=False).head(30)
    bridge_fp = out_tables / "bridge_genomes_top30_partial.tsv"
    bridge_top.to_csv(bridge_fp, sep="\t", index=False)

    label_counts = proj["ecosystem"].value_counts()
    if len(label_counts) >= 2 and (label_counts >= 2).all():
        sil_samples = silhouette_samples(matrix, proj["ecosystem"].values, metric="cosine")
        sil_global = float(silhouette_score(matrix, proj["ecosystem"].values, metric="cosine"))
        proj["silhouette"] = sil_samples
    else:
        sil_global = float("nan")
        proj["silhouette"] = np.nan

    d = cosine_distances(matrix)
    d2 = d * d
    perm = compute_permanova(proj["ecosystem"].astype(str).values, d2, permutations, rng)

    if set(["rumen", "wetland"]).issubset(set(proj["ecosystem"].unique())):
        rum_c = matrix[proj["ecosystem"] == "rumen"].mean(axis=0)
        wet_c = matrix[proj["ecosystem"] == "wetland"].mean(axis=0)
        axis_vec = wet_c - rum_c
        norm = np.linalg.norm(axis_vec)
        if norm > 0:
            axis_vec = axis_vec / norm
            proj["ecosystem_axis_projection"] = matrix @ axis_vec
        else:
            proj["ecosystem_axis_projection"] = 0.0
    else:
        proj["ecosystem_axis_projection"] = 0.0

    sns.set_theme(style="whitegrid", context="talk")

    # Figure 1: 2x2 projections + bridge distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_projection(axes[0, 0], proj, "pc1", "pc2", "PCA landscape")
    plot_projection(axes[0, 1], proj, "umap1", "umap2", f"UMAP landscape ({umap_mode})")
    plot_projection(axes[1, 0], proj, "tsne1", "tsne2", "t-SNE landscape")
    sns.histplot(
        data=proj,
        x="opp_neighbor_fraction",
        hue="ecosystem",
        bins=20,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Bridge mixing coefficient distribution")
    axes[1, 1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_figures / "projection_landscapes_panel.png", dpi=220)
    plt.close(fig)

    # Figure 2: PCA scree
    fig, ax = plt.subplots(figsize=(10, 6))
    var = pca.explained_variance_ratio_ * 100.0
    ax.plot(np.arange(1, len(var) + 1), var, marker="o", color="#005f73")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA variance spectrum")
    fig.tight_layout()
    fig.savefig(out_figures / "pca_scree.png", dpi=220)
    plt.close(fig)

    # Figure 3: Cosine heatmap (sampled)
    if n > max_heatmap_samples:
        idx = rng.choice(n, size=max_heatmap_samples, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)
    hm = d[np.ix_(idx, idx)]
    hm_labels = proj.iloc[idx]["ecosystem"].astype(str).values
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(hm, cmap="mako", ax=ax, cbar_kws={"label": "cosine distance"})
    ax.set_title("Pairwise cosine distance heatmap (sampled)")
    ax.set_xlabel(f"Genomes (n={len(idx)}; ecosystems={pd.Series(hm_labels).value_counts().to_dict()})")
    ax.set_ylabel("Genomes")
    fig.tight_layout()
    fig.savefig(out_figures / "cosine_distance_heatmap_sampled.png", dpi=220)
    plt.close(fig)

    # Figure 4: silhouette profile
    if proj["silhouette"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=proj, x="ecosystem", y="silhouette", palette="Set2", ax=ax)
        sns.stripplot(data=proj.sample(min(len(proj), 250), random_state=seed), x="ecosystem", y="silhouette", color="black", size=2, alpha=0.4, ax=ax)
        ax.set_title("Per-genome silhouette by ecosystem")
        fig.tight_layout()
        fig.savefig(out_figures / "silhouette_by_ecosystem.png", dpi=220)
        plt.close(fig)

    # Figure 5: ecosystem axis trajectory
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=proj, x="ecosystem_axis_projection", hue="ecosystem", fill=True, common_norm=False, alpha=0.35, ax=ax)
    ax.set_title("Projection on rumen → wetland trajectory axis")
    fig.tight_layout()
    fig.savefig(out_figures / "ecosystem_trajectory_axis.png", dpi=220)
    plt.close(fig)

    # Interactive plots
    if HAS_PLOTLY:
        hover_cols = [
            c
            for c in [
                "sample",
                "source",
                "domain",
                "source_analysis_accession",
                "n_proteins_used",
                "proteome_size_bytes",
                "bridge_score",
            ]
            if c in proj.columns
        ]
        p_umap = px.scatter(
            proj,
            x="umap1",
            y="umap2",
            color="ecosystem",
            symbol="source",
            hover_data=hover_cols,
            title="Interim UMAP landscape",
        )
        p_umap.write_html(out_interactive / "umap_interactive.html", include_plotlyjs="cdn")

        p_tsne = px.scatter(
            proj,
            x="tsne1",
            y="tsne2",
            color="ecosystem",
            symbol="source",
            hover_data=hover_cols,
            title="Interim t-SNE landscape",
        )
        p_tsne.write_html(out_interactive / "tsne_interactive.html", include_plotlyjs="cdn")

    proj_fp = out_tables / "embedding_projection_partial.tsv"
    proj.to_csv(proj_fp, sep="\t", index=False)

    summary = {
        "n_embeddings": int(n),
        "embedding_dim": int(matrix.shape[1]),
        "ecosystem_counts": proj["ecosystem"].value_counts().to_dict(),
        "source_counts": proj["source"].value_counts().to_dict(),
        "domain_counts": proj["domain"].value_counts().to_dict(),
        "unique_rumen_analyses": int(
            proj.loc[proj["source"] == "rumen", "source_analysis_accession"].astype(str).nunique()
        ),
        "silhouette_global": sil_global,
        "permanova": perm,
        "pca_variance_top5": (pca.explained_variance_ratio_[:5] * 100.0).round(4).tolist(),
        "umap_mode": umap_mode,
        "tsne_perplexity": int(tsne_perplexity),
        "knn_k": int(knn_k),
        "bridge_score_top_sample": bridge_top[["sample", "bridge_score", "ecosystem", "source"]].head(10).to_dict(orient="records"),
    }
    return summary


def write_readme(snapshot_dir: Path, source_artifacts_dir: Path, summary: dict[str, Any], copied: list[str]) -> None:
    readme = snapshot_dir / "README.md"
    lines = [
        "# Blue Catalyst Interim Analytics Snapshot",
        "",
        f"- Source artifacts: `{source_artifacts_dir}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Embeddings analyzed: `{summary['n_embeddings']}`",
        f"- Ecosystem counts: `{summary['ecosystem_counts']}`",
        f"- Source counts: `{summary['source_counts']}`",
        f"- Unique rumen analyses: `{summary['unique_rumen_analyses']}`",
        "",
        "## Copied artifact items",
        "",
    ]
    lines.extend([f"- {x}" for x in copied])
    lines.extend(
        [
            "",
            "## Output folders",
            "",
            "- `fetched_artifacts/` raw copied artifacts/checkpoints",
            "- `tables/` projection table + bridge rankings",
            "- `figures/` static PNG figures",
            "- `interactive/` Plotly HTML figures (if plotly installed)",
            "- `analytics/analytics_summary.json` machine-readable summary",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print(
        "[DEPRECATED] scripts/blue_catalyst_partial_intel_report.py is no longer the canonical local analytics entrypoint.\n"
        "Use notebooks/blue_catalyst_partial_report_local.ipynb (execute via Jupyter or nbconvert)."
    )
    return 2

    args = parse_args()
    source_dir = args.source_artifacts_dir.expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source artifacts directory not found: {source_dir}")

    run_id = infer_run_id_from_source(source_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = args.snapshot_name.strip() if args.snapshot_name else f"{run_id}_interim_{stamp}"
    paths = ensure_snapshot_dirs(args.output_root.expanduser().resolve(), snapshot_name)

    copied = copy_available_artifacts(source_dir, paths["fetched"])

    loaded = load_from_final_npz(paths["fetched"])
    if loaded is None:
        loaded = aggregate_checkpoint_batches(paths["fetched"])

    loaded.metadata = attach_inventory_metadata(loaded.metadata, paths["fetched"])

    # Save explicit partial consolidated artifacts for downstream reproducibility.
    partial_npz = paths["analytics"] / "genome_embeddings.partial.npz"
    np.savez_compressed(
        partial_npz,
        embeddings=loaded.matrix.astype(np.float32),
        sample=loaded.metadata["sample"].astype(str).values,
        source=loaded.metadata["source"].astype(str).values,
        ecosystem=loaded.metadata["ecosystem"].astype(str).values,
        domain=loaded.metadata["domain"].astype(str).values,
        source_analysis_accession=loaded.metadata["source_analysis_accession"].astype(str).values,
        n_proteins_used=loaded.metadata["n_proteins_used"].astype(int).values,
    )
    loaded.metadata.to_csv(paths["analytics"] / "embedding_metadata.partial.tsv", sep="\t", index=False)

    summary = run_analytics(
        loaded=loaded,
        out_figures=paths["figures"],
        out_interactive=paths["interactive"],
        out_tables=paths["tables"],
        max_heatmap_samples=args.max_heatmap_samples,
        knn_k=args.knn_k,
        permutations=args.permutations,
        seed=args.seed,
    )
    summary["load_mode"] = loaded.mode
    summary["snapshot_name"] = snapshot_name
    summary["source_artifacts_dir"] = str(source_dir)
    summary["partial_npz"] = str(partial_npz)

    analytics_summary_path = paths["analytics"] / "analytics_summary.json"
    analytics_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_readme(paths["snapshot"], source_dir, summary, copied)

    print("[OK] Interim Blue Catalyst analytics snapshot complete")
    print(f"Snapshot dir: {paths['snapshot']}")
    print(f"Summary JSON: {analytics_summary_path}")
    print(f"Static figures: {paths['figures']}")
    print(f"Tables: {paths['tables']}")
    if HAS_PLOTLY:
        print(f"Interactive figures: {paths['interactive']}")
    else:
        print("Interactive figures skipped (plotly not installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
