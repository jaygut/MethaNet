#!/usr/bin/env python3
"""Build the next-generation MethaNet MBAG molecular niche atlas.

This report consolidates the closed rumen/wetland POC core with the current
MSM/mangrove payloads. The reader-facing artifact foregrounds molecular
niche-space structure, bridge-neighborhood evidence, candidate signatures,
and claim boundaries. It does not assign sample-level MRV scores, flux claims,
or carbon-crediting conclusions.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import shutil
import subprocess
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_mbag_expanded_multiview_atlas as legacy


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFOGRAPHIC = Path(
    "results/visual_artifacts/methanet_agentic_workflow_moat_20260616/"
    "methanet_agentic_workflow_moat_v2.png"
)
DEFAULT_SAMPLE_RISK_ABSTRACT = Path(
    "ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/"
    "infographics/methanet_sample_risk_readiness_graphical_abstract_20260619/"
    "methanet_mag_to_sample_risk_readiness_graphical_abstract.png"
)
CLAIM_BOUNDARY = (
    "MAG/proteome-level molecular screening and bridge-candidate prioritization only; "
    "not a final MRV risk score, not measured methane flux, and not carbon-credit approval."
)

COLORS = {
    "rumen": "#c56a13",
    "wetland": "#0284a8",
    "mangrove": "#168a48",
    "pending": "#d89b14",
    "ink": "#172033",
    "muted": "#607083",
    "line": "#d8e2eb",
    "surface": "#f5f8fb",
    "panel": "#ffffff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--poc-esm-dir", type=Path, default=legacy.DEFAULT_POC_ESM_DIR)
    parser.add_argument("--poc-warehouse-dir", type=Path, default=legacy.DEFAULT_POC_WAREHOUSE_DIR)
    parser.add_argument("--poc-glm-dir", type=Path, default=legacy.DEFAULT_POC_GLM_DIR)
    parser.add_argument("--msm-root", type=Path, default=legacy.DEFAULT_MSM_ROOT)
    parser.add_argument("--msm-esm-dir", type=Path, default=legacy.DEFAULT_MSM_ESM_DIR)
    parser.add_argument("--msm-glm-dir", type=Path, default=legacy.DEFAULT_MSM_GLM_DIR)
    parser.add_argument("--infographic", type=Path, default=DEFAULT_INFOGRAPHIC)
    parser.add_argument("--sample-risk-abstract", type=Path, default=DEFAULT_SAMPLE_RISK_ABSTRACT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--knn", type=int, default=35)
    parser.add_argument("--top-n-poc", type=int, default=10)
    parser.add_argument("--top-n-mangrove", type=int, default=16)
    parser.add_argument("--graph-node-cap", type=int, default=360)
    parser.add_argument("--skip-phate", action="store_true", help="Skip PHATE if runtime is constrained.")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP if runtime is constrained.")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE if runtime is constrained.")
    return parser.parse_args()


def resolve(root: Path, path: Path | str | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else root / path


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def norm01(values: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    lo = float(s.min(skipna=True))
    hi = float(s.max(skipna=True))
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - lo) / (hi - lo)).fillna(0)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def short_id(value: Any, width: int = 32) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    keep = max(8, (width - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def source_label(category: str) -> str:
    return {
        "rumen": "Rumen",
        "wetland": "Wetland/MUCC",
        "mangrove": "Mangrove/MSM",
        "context": "Embedding context",
    }.get(category, category)


def frame_records(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> list[dict[str, Any]]:
    deduped_cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    use = df.loc[:, ~df.columns.duplicated()][deduped_cols].copy()
    if max_rows is not None:
        use = use.head(max_rows)
    return json.loads(use.replace({np.nan: None}).to_json(orient="records"))


def compute_diffusion_map(embeddings: np.ndarray, k: int, random_state: int = 20260619) -> np.ndarray:
    """Compute a two-dimensional diffusion map from a sparse cosine kNN graph."""
    del random_state
    x = normalize(embeddings)
    n = x.shape[0]
    n_neighbors = min(max(k + 1, 4), n)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    sigma = np.maximum(distances[:, -1], 1e-6)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            denom = max(sigma[i] * sigma[j], 1e-8)
            weight = math.exp(-float(dist * dist) / denom)
            rows.append(i)
            cols.append(int(j))
            vals.append(weight)
    w = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    w = w.maximum(w.T)
    degree = np.asarray(w.sum(axis=1)).ravel()
    degree = np.where(degree > 0, degree, 1.0)
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degree))
    sym = d_inv_sqrt @ w @ d_inv_sqrt
    eig_count = min(4, n - 1)
    eigvals, eigvecs = eigsh(sym, k=eig_count, which="LA")
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if eigvecs.shape[1] < 3:
        coords = PCA(n_components=2, random_state=20260619).fit_transform(x)
    else:
        coords = eigvecs[:, 1:3] * eigvals[1:3]
    return np.asarray(coords, dtype=float)


def compute_manifold_coordinates(
    embeddings: np.ndarray,
    k: int,
    skip_umap: bool,
    skip_phate: bool,
    skip_tsne: bool,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    x = normalize(embeddings)
    coords: dict[str, np.ndarray] = {
        "pca": PCA(n_components=2, random_state=20260619).fit_transform(x),
        "diffusion": compute_diffusion_map(embeddings, k=k),
    }
    methods: list[dict[str, str]] = [
        {
            "method": "diffusion",
            "status": "computed",
            "role": "Primary molecular niche-space map. Built from the cosine kNN affinity graph.",
        },
        {
            "method": "pca",
            "status": "computed",
            "role": "Linear sanity-check projection, not the bridge-evidence substrate.",
        },
    ]
    if not skip_umap:
        try:
            import umap

            coords["umap"] = umap.UMAP(
                n_neighbors=max(15, min(45, k * 2)),
                min_dist=0.08,
                metric="cosine",
                random_state=20260619,
                low_memory=True,
            ).fit_transform(x)
            methods.append({"method": "umap", "status": "computed", "role": "Nonlinear neighborhood-preserving comparison."})
        except Exception as exc:
            methods.append({"method": "umap", "status": f"unavailable: {type(exc).__name__}: {exc}", "role": "optional"})
    if not skip_phate:
        try:
            import phate

            coords["phate"] = phate.PHATE(
                n_components=2,
                knn=max(5, k),
                decay=40,
                random_state=20260619,
                n_jobs=1,
                verbose=0,
            ).fit_transform(x)
            methods.append({"method": "phate", "status": "computed", "role": "Biology-oriented diffusion-potential comparison."})
        except Exception as exc:
            methods.append({"method": "phate", "status": f"unavailable: {type(exc).__name__}: {exc}", "role": "optional"})
    if not skip_tsne:
        try:
            coords["tsne"] = TSNE(
                n_components=2,
                perplexity=35,
                metric="cosine",
                init="pca",
                learning_rate="auto",
                max_iter=1000,
                random_state=20260619,
            ).fit_transform(x)
            methods.append({"method": "tsne", "status": "computed", "role": "Local-neighborhood visual comparison; not used for ranking."})
        except Exception as exc:
            methods.append({"method": "tsne", "status": f"unavailable: {type(exc).__name__}: {exc}", "role": "optional"})

    out = pd.DataFrame(index=np.arange(x.shape[0]))
    for method, arr in coords.items():
        out[f"{method}_1"] = arr[:, 0]
        out[f"{method}_2"] = arr[:, 1]
    return out, methods


def rebuild_scoped_embedding_context(
    emb_meta: pd.DataFrame,
    embeddings: np.ndarray,
    atlas: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Restrict bridge geometry to report-scoped MAG/proteome units and rebuild kNN evidence."""
    scoped_ids = set(atlas["proteome_id"].astype(str))
    keep_mask = emb_meta["proteome_id"].astype(str).isin(scoped_ids).to_numpy()
    emb_meta = emb_meta.loc[keep_mask].reset_index(drop=True).copy()
    embeddings = embeddings[keep_mask]

    status_cols = [
        "proteome_id",
        "source_category",
        "atlas_inclusion_status",
        "has_functional",
        "has_glm2",
        "has_esm2",
    ]
    status = atlas[[c for c in status_cols if c in atlas.columns]].drop_duplicates("proteome_id")
    emb_meta = emb_meta.drop(
        columns=[
            "atlas_inclusion_status",
            "has_functional",
            "has_glm2",
            "has_esm2",
            "cross_domain_neighbor_count",
            "cross_domain_neighbor_fraction",
            "nearest_poc_similarity",
            "nearest_mangrove_similarity",
            "pca_1",
            "pca_2",
        ],
        errors="ignore",
    )
    emb_meta = emb_meta.merge(status, on="proteome_id", how="left", suffixes=("", "_atlas"))
    if "source_category_atlas" in emb_meta.columns:
        emb_meta["source_category"] = emb_meta["source_category_atlas"].fillna(emb_meta.get("source_category", "context"))
        emb_meta = emb_meta.drop(columns=["source_category_atlas"])
    emb_meta["atlas_inclusion_status"] = emb_meta["atlas_inclusion_status"].fillna("not_in_report_scope")
    emb_meta["has_functional"] = emb_meta["has_functional"].fillna(False).astype(bool)
    emb_meta["has_glm2"] = emb_meta["has_glm2"].fillna(False).astype(bool)
    emb_meta["has_esm2"] = True

    reduced = PCA(n_components=2, random_state=20260619).fit_transform(normalize(embeddings))
    emb_meta["pca_1"] = reduced[:, 0]
    emb_meta["pca_2"] = reduced[:, 1]

    n_neighbors = min(k + 1, len(emb_meta))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    edges: list[dict[str, Any]] = []
    cross_counts = np.zeros(len(emb_meta), dtype=int)
    for i, (ds, js) in enumerate(zip(distances, indices)):
        src_cat = str(emb_meta.iloc[i]["source_category"])
        kept = 0
        for dist, j in zip(ds, js):
            if i == j:
                continue
            dst_cat = str(emb_meta.iloc[j]["source_category"])
            if dst_cat != src_cat:
                cross_counts[i] += 1
            edges.append(
                {
                    "source": emb_meta.iloc[i]["proteome_id"],
                    "target": emb_meta.iloc[j]["proteome_id"],
                    "source_category": src_cat,
                    "target_category": dst_cat,
                    "cosine_distance": float(dist),
                    "similarity": float(1.0 - dist),
                    "cross_domain": bool(dst_cat != src_cat),
                    "rank": kept + 1,
                }
            )
            kept += 1
            if kept >= k:
                break
    edge_df = pd.DataFrame(edges)
    edge_pairs = {(r.source, r.target) for r in edge_df.itertuples()}
    edge_df["reciprocal"] = [((r.target, r.source) in edge_pairs) for r in edge_df.itertuples()]
    emb_meta["cross_domain_neighbor_count"] = cross_counts
    emb_meta["cross_domain_neighbor_fraction"] = cross_counts / max(k, 1)

    poc_core_ids = set(atlas.loc[atlas["atlas_inclusion_status"].eq("poc_core_complete"), "proteome_id"].astype(str))
    msm_ids = set(atlas.loc[atlas["source_category"].eq("mangrove"), "proteome_id"].astype(str))
    poc_idx = emb_meta.index[emb_meta["proteome_id"].astype(str).isin(poc_core_ids)].to_numpy()
    msm_idx = emb_meta.index[emb_meta["proteome_id"].astype(str).isin(msm_ids)].to_numpy()
    emb_meta["nearest_poc_similarity"] = 0.0
    emb_meta["nearest_mangrove_similarity"] = 0.0
    if len(poc_idx):
        nn_poc = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[poc_idx])
        d_poc, _ = nn_poc.kneighbors(embeddings)
        emb_meta["nearest_poc_similarity"] = 1.0 - d_poc[:, 0]
    if len(msm_idx):
        nn_msm = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[msm_idx])
        d_msm, _ = nn_msm.kneighbors(embeddings)
        emb_meta["nearest_mangrove_similarity"] = 1.0 - d_msm[:, 0]
    return emb_meta, edge_df, embeddings


def add_molecular_metrics(atlas: pd.DataFrame) -> pd.DataFrame:
    atlas = atlas.copy()
    protein_count = pd.to_numeric(atlas.get("prodigal_proteins"), errors="coerce")
    protein_count = protein_count.fillna(pd.to_numeric(atlas.get("n_proteins_used"), errors="coerce"))
    has_real_denominator = protein_count.notna() & protein_count.gt(0)
    atlas["protein_count_for_rates"] = protein_count.where(has_real_denominator)
    atlas["rate_metric_status"] = np.where(has_real_denominator, "rate_normalized", "not_rate_normalized")

    methane_from_scores = pd.to_numeric(atlas.get("methane_evidence_score"), errors="coerce")
    methane_from_raw = (
        pd.to_numeric(atlas.get("mcycdb_hits", 0), errors="coerce").fillna(0)
        + pd.to_numeric(atlas.get("metabolic_hmm_rows", 0), errors="coerce").fillna(0)
    )
    sulfur_from_scores = pd.to_numeric(atlas.get("sulfur_competition_score"), errors="coerce")
    sulfur_from_raw = (
        pd.to_numeric(atlas.get("scycdb_hits", 0), errors="coerce").fillna(0)
        + pd.to_numeric(atlas.get("metabolic_functions_present", 0), errors="coerce").fillna(0)
    )
    methane_count = methane_from_scores.where(methane_from_scores.notna(), methane_from_raw).fillna(0)
    sulfur_count = sulfur_from_scores.where(sulfur_from_scores.notna(), sulfur_from_raw).fillna(0)
    substrate_count = (
        pd.to_numeric(atlas.get("cazy_family_count", 0), errors="coerce").fillna(0)
        + pd.to_numeric(atlas.get("merops_family_count", 0), errors="coerce").fillna(0)
    )
    atlas["methane_marker_count"] = methane_count
    atlas["sulfur_context_count"] = sulfur_count
    atlas["substrate_breadth_count"] = substrate_count
    atlas["methane_marker_density_per_1k"] = np.where(has_real_denominator, 1000 * methane_count / protein_count, np.nan)
    atlas["sulfur_context_density_per_1k"] = np.where(has_real_denominator, 1000 * sulfur_count / protein_count, np.nan)
    atlas["substrate_breadth_per_1k"] = np.where(has_real_denominator, 1000 * substrate_count / protein_count, np.nan)
    atlas["methane_sulfur_balance"] = np.log1p(atlas["methane_marker_density_per_1k"]) - np.log1p(
        atlas["sulfur_context_density_per_1k"]
    )
    atlas["annotation_breadth_index"] = (
        0.34 * norm01(pd.to_numeric(atlas.get("kofam_annotated_gene_fraction", 0), errors="coerce").fillna(0))
        + 0.22 * norm01(np.log1p(pd.to_numeric(atlas.get("metabolic_modules_present", 0), errors="coerce").fillna(0)))
        + 0.22 * norm01(np.log1p(pd.to_numeric(atlas.get("cazy_family_count", 0), errors="coerce").fillna(0)))
        + 0.22 * norm01(np.log1p(pd.to_numeric(atlas.get("merops_family_count", 0), errors="coerce").fillna(0)))
    )
    qc_raw = (
        pd.to_numeric(atlas.get("checkm2_completeness", 0), errors="coerce").fillna(0)
        - 5 * pd.to_numeric(atlas.get("checkm2_contamination", 0), errors="coerce").fillna(0)
    ).clip(lower=0)
    atlas["qc_confidence_index"] = (qc_raw / 100).clip(0, 1)
    atlas["bridge_affinity_index"] = norm01(
        pd.to_numeric(atlas.get("nearest_poc_similarity", 0), errors="coerce").fillna(0)
        + 0.6 * pd.to_numeric(atlas.get("cross_domain_neighbor_fraction", 0), errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(atlas.get("mixing_coeff", 0), errors="coerce").fillna(0)
    )
    atlas["methane_signal_index"] = norm01(np.log1p(methane_count))
    atlas["sulfur_context_index"] = norm01(np.log1p(sulfur_count))
    atlas["substrate_breadth_index"] = norm01(np.log1p(substrate_count))
    atlas["glm_context_index"] = norm01(pd.to_numeric(atlas.get("glm_context_delta", 0), errors="coerce").fillna(0))
    atlas["molecular_attestation_index"] = (
        0.24 * atlas["bridge_affinity_index"]
        + 0.14 * atlas["glm_context_index"]
        + 0.19 * atlas["methane_signal_index"]
        + 0.13 * atlas["sulfur_context_index"]
        + 0.12 * atlas["substrate_breadth_index"]
        + 0.10 * atlas["annotation_breadth_index"]
        + 0.08 * atlas["qc_confidence_index"]
    )
    atlas.loc[~atlas["has_functional"], "molecular_attestation_index"] = np.nan
    return atlas


def classify_review_tier(row: pd.Series) -> str:
    if not bool(row.get("has_functional", False)):
        return "functional pending"
    score = safe_float(row.get("molecular_attestation_index"))
    qc = safe_float(row.get("qc_confidence_index"))
    if score >= 0.72 and qc >= 0.55:
        return "high-priority review"
    if score >= 0.52:
        return "mechanism review"
    if score >= 0.34:
        return "screening signal"
    return "low-current signal"


def read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def first_nonempty(row: pd.Series, cols: list[str]) -> str:
    for col in cols:
        value = row.get(col)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


def add_source_provenance_context(atlas: pd.DataFrame, repo_root: Path, msm_root: Path) -> pd.DataFrame:
    """Attach report-facing source/provenance readiness fields without implying sample-level MRV."""
    atlas = atlas.copy()
    defaults = {
        "rumen": {
            "source_paper_doi": "10.1038/s41587-019-0202-3",
            "source_dataset_doi": "10.7488/ds/2470",
            "primary_accession_type": "ENA analysis accession",
            "provenance_resolution_tier": "exact_analysis_accession",
            "metadata_caveat": "Exact MAG/proteome accession provenance; environmental context is mostly cohort-level cattle rumen.",
            "sample_rollup_status": "reference_context_not_blue_carbon_sample_rollup",
            "next_metadata_action": "Retain as methane-domain reference; do not treat as blue-carbon sample context.",
        },
        "wetland": {
            "source_paper_doi": "10.1038/s41467-025-56133-0",
            "source_dataset_doi": "10.5281/zenodo.14532347",
            "primary_accession_type": "MUCC/NCBI/Zenodo source record",
            "provenance_resolution_tier": "mixed_mucc_resolution",
            "metadata_caveat": "Strong paper/dataset provenance, but sample resolution is mixed across NCBI BioSample, OWC site/project, and MUCC source-bucket rows.",
            "sample_rollup_status": "blocked_mixed_sample_resolution",
            "next_metadata_action": "Recover MAG-to-sample/BioSample mapping for source-bucket rows before sample-level rollup.",
        },
        "mangrove": {
            "source_paper_doi": "10.1093/gigascience/giaf081",
            "source_dataset_doi": "10.5524/102702",
            "primary_accession_type": "GigaDB/NCBI BioSample group mapping",
            "provenance_resolution_tier": "source_group_biosample_context",
            "metadata_caveat": "Strong source provenance and sediment-sample context, but per-MAG sample assignment and 966-vs-1428 denominator reconciliation remain pending.",
            "sample_rollup_status": "blocked_mag_to_sample_reconciliation",
            "next_metadata_action": "Resolve MAG-to-sample links and paper-final MAG denominator before sample/site MRV features.",
        },
    }
    for col in [
        "source_paper_doi",
        "source_dataset_doi",
        "primary_accession",
        "primary_accession_type",
        "provenance_resolution_tier",
        "metadata_caveat",
        "sample_rollup_status",
        "next_metadata_action",
        "source_sample_ids",
        "mapped_ncbi_biosamples",
        "site_label",
        "source_bucket",
    ]:
        if col not in atlas.columns:
            atlas[col] = ""
    for cat, vals in defaults.items():
        mask = atlas["source_category"].astype(str).eq(cat)
        for col, value in vals.items():
            atlas.loc[mask & atlas[col].fillna("").astype(str).eq(""), col] = value

    env_path = repo_root / "results/functional_metagenomics/environmental_metadata_recovery_20260612/cohort_662_environmental_metadata_crosswalk.tsv"
    env = read_optional_tsv(env_path)
    if not env.empty and "proteome_id" in env.columns:
        rows = []
        for _, row in env.drop_duplicates("proteome_id").iterrows():
            rows.append(
                {
                    "proteome_id": str(row["proteome_id"]),
                    "primary_accession": first_nonempty(
                        row,
                        [
                            "source_analysis_accession",
                            "analysis_accession",
                            "ncbi_assembly_accession",
                            "ncbi_biosample_accession",
                            "biosample_accession",
                        ],
                    ),
                    "provenance_resolution_tier": first_nonempty(row, ["metadata_resolution", "context_level"]),
                    "site_label": first_nonempty(row, ["site_label", "biosample_attr_geo_loc_name", "country"]),
                    "source_bucket": first_nonempty(row, ["mucc_source_bucket"]),
                    "mapped_ncbi_biosamples": first_nonempty(row, ["ncbi_biosample_accession", "biosample_accession"]),
                }
            )
        ctx = pd.DataFrame(rows)
        atlas = atlas.merge(ctx, on="proteome_id", how="left", suffixes=("", "_ctx"))
        for col in ["primary_accession", "provenance_resolution_tier", "site_label", "source_bucket", "mapped_ncbi_biosamples"]:
            ctx_col = f"{col}_ctx"
            if ctx_col in atlas.columns:
                ctx_nonempty = atlas[ctx_col].fillna("").astype(str).str.strip().ne("")
                atlas[col] = atlas[col].where(~ctx_nonempty, atlas[ctx_col].fillna(""))
                atlas = atlas.drop(columns=[ctx_col])

    msm_manifest = read_optional_tsv(msm_root / "manifests/msm_china_2025_functional_mag_manifest.tsv")
    if not msm_manifest.empty and "proteome_id" in msm_manifest.columns:
        keep_cols = [
            "proteome_id",
            "metadata_mapping_status",
            "mapped_ncbi_biosamples",
            "mapped_ncbi_bioprojects",
            "source_sample_ids",
            "source_group",
        ]
        keep = [c for c in keep_cols if c in msm_manifest.columns]
        msm_ctx = msm_manifest[keep].drop_duplicates("proteome_id").copy()
        rename = {
            "metadata_mapping_status": "provenance_resolution_tier_ctx",
            "source_group": "source_bucket_ctx",
            "mapped_ncbi_biosamples": "mapped_ncbi_biosamples_ctx",
            "source_sample_ids": "source_sample_ids_ctx",
        }
        msm_ctx = msm_ctx.rename(columns=rename)
        atlas = atlas.merge(msm_ctx, on="proteome_id", how="left")
        for col in ["provenance_resolution_tier", "mapped_ncbi_biosamples", "source_sample_ids", "source_bucket"]:
            ctx_col = f"{col}_ctx"
            if ctx_col in atlas.columns:
                ctx_nonempty = atlas[ctx_col].fillna("").astype(str).str.strip().ne("")
                atlas[col] = atlas[col].where(~ctx_nonempty, atlas[ctx_col].fillna(""))
                atlas = atlas.drop(columns=[ctx_col])
        if "mapped_ncbi_bioprojects" in atlas.columns:
            atlas["primary_accession"] = atlas["primary_accession"].where(
                atlas["primary_accession"].fillna("").astype(str).ne(""),
                atlas["mapped_ncbi_bioprojects"].fillna(""),
            )
    source_bucket_only = atlas["provenance_resolution_tier"].astype(str).str.contains("source_bucket", case=False, na=False)
    numeric_primary = atlas["primary_accession"].fillna("").astype(str).str.fullmatch(r"\d+(\.0)?")
    atlas.loc[source_bucket_only & numeric_primary, "primary_accession"] = ""
    atlas["functional_annotation_status"] = np.where(
        atlas["has_functional"].fillna(False).astype(bool),
        "functional annotations complete",
        "functional annotations pending; taxonomy and mechanism fields not interpreted yet",
    )
    atlas["plot_annotation_status"] = np.where(
        atlas["has_functional"].fillna(False).astype(bool),
        atlas["review_tier"].fillna("screening signal"),
        "ESM-2 + gLM2 only; function/QC/taxonomy pending",
    )
    return atlas


def build_candidate_cards(atlas: pd.DataFrame, top_n_poc: int, top_n_mangrove: int) -> pd.DataFrame:
    poc = atlas[atlas["source_category"].isin(["rumen", "wetland"])].copy()
    poc_top = poc[poc.get("rank").notna()].sort_values("rank").head(top_n_poc).copy()
    poc_top["candidate_set"] = "POC bridge candidate"
    msm = atlas[atlas["source_category"].eq("mangrove") & atlas["has_functional"] & atlas["has_glm2"]].copy()
    msm_top = (
        msm.sort_values(["molecular_attestation_index", "nearest_poc_similarity"], ascending=False)
        .head(top_n_mangrove)
        .copy()
    )
    msm_top["candidate_set"] = "Mangrove nearest-neighborhood candidate"
    msm_top["rank"] = np.arange(1, len(msm_top) + 1)
    cards = pd.concat([poc_top, msm_top], ignore_index=True, sort=False)
    cards["review_tier"] = cards.apply(classify_review_tier, axis=1)
    cards["card_id"] = (
        cards["candidate_set"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        + "_"
        + cards["rank"].fillna(0).astype(int).astype(str)
    )
    cards["allowed_claim_wording"] = (
        "Reviewable MAG/proteome bridge hypothesis supported by embedding-neighborhood, genomic-context, "
        "functional, QC, and taxonomy evidence where present."
    )
    cards["blocking_gap"] = (
        "sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, "
        "phylogeny/source controls, and flux/process validation"
    )
    cards["next_validation_action"] = (
        "inspect marker neighborhoods, compare phylogeny versus embedding proximity, run source-aware nulls, "
        "and connect to sample metadata before MRV scoring"
    )
    return cards


def build_evidence_flow(atlas: pd.DataFrame) -> dict[str, Any]:
    df = atlas.copy()
    df["review_tier"] = df.apply(classify_review_tier, axis=1)
    df["evidence_state"] = np.select(
        [
            df["atlas_inclusion_status"].eq("poc_core_complete"),
            df["source_category"].eq("mangrove") & df["has_functional"],
            df["source_category"].eq("mangrove") & ~df["has_functional"],
        ],
        ["POC tri-view complete", "Mangrove tri-view complete", "Mangrove function pending"],
        default="Other",
    )
    stage_cols = ["source_display", "evidence_state", "review_tier"]
    nodes: dict[tuple[int, str], dict[str, Any]] = {}
    links: dict[tuple[str, str], int] = defaultdict(int)
    for _, row in df.iterrows():
        labels = [str(row.get(col, "")) for col in stage_cols]
        for stage, label in enumerate(labels):
            nodes[(stage, label)] = {"id": f"{stage}:{label}", "name": label, "stage": stage}
        links[(f"0:{labels[0]}", f"1:{labels[1]}")] += 1
        links[(f"1:{labels[1]}", f"2:{labels[2]}")] += 1
    return {"nodes": list(nodes.values()), "links": [{"source": a, "target": b, "value": v} for (a, b), v in links.items()]}


def build_source_provenance_readiness(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "lane": "Rumen reference",
            "report_units": int(summary.get("poc_rumen_total", 0)),
            "metadata_universe": "555 embedded POC rumen proteomes",
            "primary_source": "Stewart et al. 2019; ENA PRJEB31266",
            "resolution_now": "555/555 exact ERZ analysis-accession matches",
            "use_now": "Methane-domain reference provenance and source-aware bridge comparison",
            "blocking_gap": "Animal/sample-level environmental metadata is mostly cohort-level; not blue-carbon sample context",
        },
        {
            "lane": "Wetland/MUCC target",
            "report_units": int(summary.get("poc_wetland_total", 0)),
            "metadata_universe": "107 embedded wetland/MUCC Methanoregula proteomes",
            "primary_source": "Bechtold et al. 2025; MUCC v2.0.0 Zenodo 10.5281/zenodo.14532347",
            "resolution_now": "20 exact NCBI assembly/BioSample; 23 OWC bin plus site/project; 64 source-bucket rows",
            "use_now": "Target-domain provenance, wetland source-bucket context, and metadata-readiness triage",
            "blocking_gap": "Uniform MAG-to-sample BioSample mapping for JGI/PPR/STM/source-bucket rows",
        },
        {
            "lane": "Mangrove/MSM target",
            "report_units": int(summary.get("msm_total", 0)),
            "metadata_universe": "1428 local MAG candidates; paper reports 966 final MAGs",
            "primary_source": "Pan et al. 2025; GigaDB 10.5524/102702; NCBI PRJNA1150796",
            "resolution_now": "82 sediment sample rows; 71 exact BioSample rows; group-level sample lists in manifest",
            "use_now": "Target-domain molecular screening lane and sample-readiness prioritization",
            "blocking_gap": "MAG-to-sample assignment and 966-vs-1428 denominator reconciliation before sample/site rollups",
        },
    ]


def build_report_validation_gates(atlas: pd.DataFrame, payload: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []

    def add(gate: str, passed: bool, details: str) -> None:
        gates.append({"gate": gate, "status": "pass" if passed else "fail", "details": details})

    duplicate_nodes = int(atlas["proteome_id"].astype(str).duplicated().sum())
    add("one_row_per_proteome_id_atlas", duplicate_nodes == 0, f"duplicate_rows={duplicate_nodes}; rows={len(atlas)}")

    non_scoped = atlas["atlas_inclusion_status"].astype(str).eq("non_poc_or_unscoped").sum()
    add("no_unscoped_rows_in_atlas_payload", int(non_scoped) == 0, f"non_scoped_rows={int(non_scoped)}")

    poc = atlas[atlas["atlas_inclusion_status"].astype(str).eq("poc_core_complete")].copy()
    zeroed_methane = poc[
        pd.to_numeric(poc.get("methane_evidence_score", 0), errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(poc.get("methane_marker_count", 0), errors="coerce").fillna(0).eq(0)
    ]
    zeroed_sulfur = poc[
        pd.to_numeric(poc.get("sulfur_competition_score", 0), errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(poc.get("sulfur_context_count", 0), errors="coerce").fillna(0).eq(0)
    ]
    add(
        "poc_methane_scores_preserved",
        zeroed_methane.empty,
        f"zeroed_examples={zeroed_methane['proteome_id'].head(5).tolist()}",
    )
    add(
        "poc_sulfur_scores_preserved",
        zeroed_sulfur.empty,
        f"zeroed_examples={zeroed_sulfur['proteome_id'].head(5).tolist()}",
    )

    fake_denominator = poc[
        pd.to_numeric(poc.get("prodigal_proteins", np.nan), errors="coerce").fillna(0).le(1)
        | poc.get("rate_metric_status", pd.Series(index=poc.index, dtype=object)).astype(str).ne("rate_normalized")
    ]
    add(
        "poc_rate_denominators_real",
        fake_denominator.empty,
        f"examples={fake_denominator['proteome_id'].head(5).tolist()}",
    )

    for payload_name in ["niche", "candidate_graph"]:
        nodes = payload.get(payload_name, {}).get("nodes", [])
        links = payload.get(payload_name, {}).get("links", [])
        node_ids = {str(row.get("proteome_id")) for row in nodes}
        missing = [
            (str(row.get("source")), str(row.get("target")))
            for row in links
            if str(row.get("source")) not in node_ids or str(row.get("target")) not in node_ids
        ]
        add(
            f"{payload_name}_links_have_visible_endpoints",
            not missing,
            f"missing_endpoint_examples={missing[:5]}",
        )

    return gates


def build_candidate_circos(cards: pd.DataFrame) -> dict[str, Any]:
    pillar_defs = [
        {
            "id": "bridge",
            "label": "Bridge affinity composite",
            "short": "Bridge",
            "metric": "bridge_affinity_index",
            "threshold": 0.55,
            "source": "ESM-2 high-dimensional neighborhood composite: nearest POC similarity, cross-domain neighbors, and mixing coefficient",
        },
        {
            "id": "glm",
            "label": "gLM2 context",
            "short": "gLM2",
            "metric": "glm_context_index",
            "threshold": 0.55,
            "source": "native-minus-shuffled genomic-context perturbation",
        },
        {
            "id": "methane",
            "label": "Methane machinery",
            "short": "CH4",
            "metric": "methane_signal_index",
            "threshold": 0.55,
            "source": "MCycDB plus METABOLIC methane-related evidence",
        },
        {
            "id": "sulfur",
            "label": "Sulfur context",
            "short": "Sulfur",
            "metric": "sulfur_context_index",
            "threshold": 0.55,
            "source": "SCycDB plus METABOLIC sulfur/competition evidence",
        },
        {
            "id": "substrate",
            "label": "Carbon substrate breadth",
            "short": "Carbon",
            "metric": "substrate_breadth_index",
            "threshold": 0.55,
            "source": "dbCAN/CAZy and MEROPS substrate-processing evidence",
        },
        {
            "id": "coverage",
            "label": "Annotation breadth",
            "short": "Coverage",
            "metric": "annotation_breadth_index",
            "threshold": 0.50,
            "source": "KOfam, METABOLIC modules, CAZy, and MEROPS coverage",
        },
        {
            "id": "qc",
            "label": "QC support",
            "short": "QC",
            "metric": "qc_confidence_index",
            "threshold": 0.55,
            "source": "CheckM2, GUNC, and curated run evidence",
        },
    ]
    group_defs = [
        {"id": "rumen", "label": "POC rumen", "color": COLORS["rumen"]},
        {"id": "wetland", "label": "POC wetland", "color": COLORS["wetland"]},
        {"id": "mangrove", "label": "Mangrove/MSM", "color": COLORS["mangrove"]},
    ]
    records: list[dict[str, Any]] = []
    for group in group_defs:
        sub = cards[cards["source_category"].astype(str).eq(group["id"])].copy()
        n = int(len(sub))
        for pillar in pillar_defs:
            values = pd.to_numeric(sub.get(pillar["metric"], pd.Series(dtype=float)), errors="coerce").fillna(0)
            high_count = int(values.ge(float(pillar["threshold"])).sum()) if n else 0
            records.append(
                {
                    "group": group["id"],
                    "group_label": group["label"],
                    "pillar": pillar["id"],
                    "pillar_label": pillar["label"],
                    "pillar_short": pillar["short"],
                    "source": pillar["source"],
                    "candidate_count": n,
                    "high_count": high_count,
                    "high_share": high_count / n if n else 0.0,
                    "average_value": float(values.mean()) if n else 0.0,
                    "threshold": float(pillar["threshold"]),
                }
            )
    return {"groups": group_defs, "pillars": pillar_defs, "records": records}


def build_signature_matrix(cards: pd.DataFrame) -> dict[str, Any]:
    metric_defs = [
        {"id": "bridge_affinity_index", "label": "Bridge", "source": "ESM-2 neighborhood geometry"},
        {"id": "glm_context_index", "label": "gLM2", "source": "native-minus-shuffled genomic context"},
        {"id": "methane_signal_index", "label": "CH4", "source": "MCycDB and METABOLIC methane markers"},
        {"id": "sulfur_context_index", "label": "Sulfur", "source": "SCycDB and METABOLIC sulfur context"},
        {"id": "substrate_breadth_index", "label": "Carbon", "source": "dbCAN/CAZy and MEROPS substrate breadth"},
        {"id": "annotation_breadth_index", "label": "Coverage", "source": "KOfam, METABOLIC, CAZy, MEROPS breadth"},
        {"id": "qc_confidence_index", "label": "QC", "source": "CheckM2 and GUNC confidence"},
    ]
    records: list[dict[str, Any]] = []
    for _, row in cards.iterrows():
        source_code = {"rumen": "R", "wetland": "W", "mangrove": "M"}.get(str(row.get("source_category")), "C")
        display_label = f"{source_code}{safe_int(row.get('rank')):02d}  {short_id(row['proteome_id'], 34)}"
        for metric in metric_defs:
            records.append(
                {
                    "proteome_id": row["proteome_id"],
                    "candidate_set": row.get("candidate_set", ""),
                    "rank": safe_int(row.get("rank")),
                    "source_category": row.get("source_category", ""),
                    "label": display_label,
                    "metric": metric["id"],
                    "metric_label": metric["label"],
                    "metric_source": metric["source"],
                    "value": safe_float(row.get(metric["id"])),
                }
            )
    return {"metrics": [m["id"] for m in metric_defs], "metric_defs": metric_defs, "records": records}


def build_payloads(
    atlas: pd.DataFrame,
    emb_meta: pd.DataFrame,
    edge_df: pd.DataFrame,
    cards: pd.DataFrame,
    summary: dict[str, Any],
    graph_node_cap: int,
    manifold_methods: list[dict[str, str]],
) -> dict[str, Any]:
    card_ids = set(cards["proteome_id"].astype(str))
    graph_edges = edge_df.sort_values(["cross_domain", "similarity"], ascending=[False, False]).copy()
    context_cols = [
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "cohort_label",
        "source_display",
        "atlas_inclusion_status",
        "analysis_unit_type",
        "mbag_mag_level_include",
        "claim_scope",
        "functional_annotation_status",
        "plot_annotation_status",
        "review_tier",
        "molecular_attestation_index",
        "bridge_affinity_index",
        "methane_marker_count",
        "sulfur_context_count",
        "substrate_breadth_count",
        "methane_marker_density_per_1k",
        "sulfur_context_density_per_1k",
        "substrate_breadth_per_1k",
        "methane_sulfur_balance",
        "nearest_poc_similarity",
        "qc_confidence_index",
        "rate_metric_status",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "qc_tier",
        "checkm2_completeness",
        "checkm2_contamination",
        "gunc_pass",
        "prodigal_proteins",
        "input_total_bp",
        "gtdb_release",
        "gtdb_classification",
        "methane_evidence_score",
        "sulfur_competition_score",
        "absence_interpretation_caveat",
        "annotation_breadth_index",
        "glm_context_delta",
        "source_paper_doi",
        "source_dataset_doi",
        "primary_accession",
        "primary_accession_type",
        "provenance_resolution_tier",
        "metadata_caveat",
        "sample_rollup_status",
        "next_metadata_action",
        "source_sample_ids",
        "mapped_ncbi_biosamples",
        "site_label",
        "source_bucket",
        "allowed_claim_wording",
        "blocking_gap",
        "next_validation_action",
    ]
    context_cols = [c for c in context_cols if c in atlas.columns]
    focus_ids = set(card_ids)
    focus_ids.update(graph_edges[graph_edges["source"].isin(card_ids)]["target"].head(160).astype(str))
    focus_ids.update(graph_edges[graph_edges["target"].isin(card_ids)]["source"].head(160).astype(str))
    focus_ids.update(
        atlas[atlas["source_category"].eq("mangrove") & atlas["has_functional"]]
        .sort_values("molecular_attestation_index", ascending=False)
        .head(140)["proteome_id"]
        .astype(str)
    )
    selected_ids = set(list(focus_ids)[:graph_node_cap])
    graph_coord_cols = [
        "source_category",
        "has_functional",
        "has_glm2",
        "diffusion_1",
        "diffusion_2",
        "umap_1",
        "umap_2",
        "phate_1",
        "phate_2",
        "tsne_1",
        "tsne_2",
        "pca_1",
        "pca_2",
    ]
    graph_node_cols = list(dict.fromkeys([c for c in context_cols + graph_coord_cols if c in atlas.columns]))
    graph_nodes = atlas[atlas["proteome_id"].astype(str).isin(selected_ids)][graph_node_cols].copy()
    graph_nodes["label"] = graph_nodes["proteome_id"].map(short_id)
    graph_links = graph_edges[
        graph_edges["source"].isin(graph_nodes["proteome_id"]) & graph_edges["target"].isin(graph_nodes["proteome_id"])
    ].head(1500)

    niche_cols = [
        "proteome_id",
        "mag_id",
        "source_category",
        "source_display",
        "atlas_inclusion_status",
        "analysis_unit_type",
        "claim_scope",
        "review_tier",
        "plot_annotation_status",
        "functional_annotation_status",
        "has_functional",
        "has_glm2",
        "molecular_attestation_index",
        "bridge_affinity_index",
        "methane_marker_count",
        "sulfur_context_count",
        "substrate_breadth_count",
        "methane_marker_density_per_1k",
        "sulfur_context_density_per_1k",
        "substrate_breadth_per_1k",
        "methane_sulfur_balance",
        "nearest_poc_similarity",
        "qc_confidence_index",
        "rate_metric_status",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "qc_tier",
        "checkm2_completeness",
        "checkm2_contamination",
        "gunc_pass",
        "prodigal_proteins",
        "methane_evidence_score",
        "sulfur_competition_score",
        "annotation_breadth_index",
        "glm_context_delta",
        "source_paper_doi",
        "source_dataset_doi",
        "primary_accession",
        "primary_accession_type",
        "provenance_resolution_tier",
        "metadata_caveat",
        "sample_rollup_status",
        "next_metadata_action",
        "source_sample_ids",
        "mapped_ncbi_biosamples",
        "site_label",
        "source_bucket",
        "allowed_claim_wording",
        "blocking_gap",
        "next_validation_action",
        "diffusion_1",
        "diffusion_2",
        "umap_1",
        "umap_2",
        "phate_1",
        "phate_2",
        "tsne_1",
        "tsne_2",
        "pca_1",
        "pca_2",
    ]
    niche_cols = [c for c in niche_cols if c in atlas.columns]
    niche_nodes = atlas[niche_cols].copy()
    niche_nodes["label"] = niche_nodes["proteome_id"].map(short_id)
    niche_node_ids = set(niche_nodes["proteome_id"].astype(str))
    niche_links = graph_edges[
        graph_edges["cross_domain"]
        & graph_edges["source"].astype(str).isin(niche_node_ids)
        & graph_edges["target"].astype(str).isin(niche_node_ids)
    ].head(2200)

    completed_mangrove = atlas[atlas["source_category"].eq("mangrove") & atlas["has_functional"]].copy()
    mangrove_cols = [
        "proteome_id",
        "mag_id",
        "phylum",
        "class",
        "qc_tier",
        "review_tier",
        "methane_marker_density_per_1k",
        "sulfur_context_density_per_1k",
        "methane_evidence_score",
        "sulfur_competition_score",
        "substrate_breadth_per_1k",
        "substrate_breadth_count",
        "methane_sulfur_balance",
        "annotation_breadth_index",
        "qc_confidence_index",
        "glm_context_delta",
        "nearest_poc_similarity",
        "molecular_attestation_index",
    ]
    return {
        "summary": summary,
        "niche": {
            "methods": manifold_methods,
            "nodes": frame_records(niche_nodes, niche_cols + ["label"]),
            "links": frame_records(
                niche_links,
                ["source", "target", "source_category", "target_category", "similarity", "cross_domain", "reciprocal", "rank"],
            ),
        },
        "candidate_graph": {
            "nodes": frame_records(
                graph_nodes,
                [
                    "proteome_id",
                    "label",
                    "source_category",
                    "source_display",
                    "atlas_inclusion_status",
                    "analysis_unit_type",
                    "claim_scope",
                    "functional_annotation_status",
                    "plot_annotation_status",
                    "review_tier",
                    "molecular_attestation_index",
                    "bridge_affinity_index",
                    "methane_marker_count",
                    "sulfur_context_count",
                    "substrate_breadth_count",
                    "methane_marker_density_per_1k",
                    "sulfur_context_density_per_1k",
                    "substrate_breadth_per_1k",
                    "methane_sulfur_balance",
                    "nearest_poc_similarity",
                    "qc_confidence_index",
                    "rate_metric_status",
                    "domain",
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                    "species",
                    "qc_tier",
                    "checkm2_completeness",
                    "checkm2_contamination",
                    "gunc_pass",
                    "methane_evidence_score",
                    "sulfur_competition_score",
                    "source_paper_doi",
                    "primary_accession",
                    "provenance_resolution_tier",
                    "glm_context_delta",
                    "diffusion_1",
                    "diffusion_2",
                    "pca_1",
                    "pca_2",
                ],
            ),
            "links": frame_records(
                graph_links,
                ["source", "target", "source_category", "target_category", "similarity", "cross_domain", "reciprocal", "rank"],
            ),
        },
        "matrix": build_signature_matrix(cards),
        "mangrove": frame_records(completed_mangrove.sort_values("molecular_attestation_index", ascending=False), mangrove_cols),
        "circos": build_candidate_circos(cards),
        "cards": frame_records(
            cards,
            [
                "card_id",
                "candidate_set",
                "rank",
                "proteome_id",
                "mag_id",
                "source_category",
                "source_display",
                "domain",
                "phylum",
                "class",
                "qc_tier",
                "review_tier",
                "checkm2_completeness",
                "checkm2_contamination",
                "glm_context_delta",
                "nearest_poc_similarity",
                "bridge_affinity_index",
                "methane_marker_count",
                "sulfur_context_count",
                "substrate_breadth_count",
                "methane_evidence_score",
                "sulfur_competition_score",
                "rate_metric_status",
                "methane_marker_density_per_1k",
                "sulfur_context_density_per_1k",
                "substrate_breadth_per_1k",
                "methane_sulfur_balance",
                "annotation_breadth_index",
                "qc_confidence_index",
                "molecular_attestation_index",
                "source_paper_doi",
                "source_dataset_doi",
                "primary_accession",
                "primary_accession_type",
                "provenance_resolution_tier",
                "metadata_caveat",
                "sample_rollup_status",
                "next_metadata_action",
                "allowed_claim_wording",
                "blocking_gap",
                "next_validation_action",
            ],
        ),
    }


def plot_niche(payload: dict[str, Any], path: Path) -> Path:
    nodes = pd.DataFrame(payload["niche"]["nodes"])
    fig, ax = plt.subplots(figsize=(11, 7.3), facecolor=COLORS["surface"])
    ax.set_facecolor("#fbfdff")
    for cat, sub in nodes.groupby("source_category"):
        ax.scatter(
            sub["diffusion_1"],
            sub["diffusion_2"],
            s=12 + 50 * pd.to_numeric(sub.get("molecular_attestation_index", 0), errors="coerce").fillna(0),
            color=COLORS.get(cat, "#94a3b8"),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
            label=source_label(cat),
        )
    ax.set_title("Molecular niche-space map, diffusion coordinates", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("diffusion coordinate 1")
    ax.set_ylabel("diffusion coordinate 2")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mangrove(payload: dict[str, Any], path: Path) -> Path:
    df = pd.DataFrame(payload["mangrove"])
    fig, ax = plt.subplots(figsize=(10.5, 6.7), facecolor=COLORS["surface"])
    ax.set_facecolor("#fbfdff")
    if not df.empty:
        x = pd.to_numeric(df["methane_marker_density_per_1k"], errors="coerce").fillna(0)
        y = pd.to_numeric(df["sulfur_context_density_per_1k"], errors="coerce").fillna(0)
        s = 18 + 160 * norm01(np.log1p(pd.to_numeric(df["substrate_breadth_per_1k"], errors="coerce").fillna(0)))
        c = pd.to_numeric(df["nearest_poc_similarity"], errors="coerce").fillna(0)
        sc = ax.scatter(np.log1p(x), np.log1p(y), s=s, c=c, cmap="YlGnBu", alpha=0.74, edgecolor="white", linewidth=0.35)
        fig.colorbar(sc, ax=ax, label="nearest POC embedding similarity")
    ax.set_title("Mangrove molecular fingerprint landscape", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("methane marker density per 1k proteins, log1p")
    ax.set_ylabel("sulfur context density per 1k proteins, log1p")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_matrix(payload: dict[str, Any], path: Path) -> Path:
    records = pd.DataFrame(payload["matrix"]["records"])
    if records.empty:
        path.touch()
        return path
    mat = records.pivot_table(index="label", columns="metric", values="value", aggfunc="first").fillna(0)
    metric_defs = payload["matrix"].get("metric_defs") or [{"id": m, "label": m} for m in payload["matrix"]["metrics"]]
    metric_ids = [m["id"] for m in metric_defs]
    metric_labels = [m.get("label", m["id"]) for m in metric_defs]
    mat = mat[metric_ids]
    fig, ax = plt.subplots(figsize=(11.6, max(6.4, 0.38 * len(mat))), facecolor=COLORS["surface"])
    im = ax.imshow(mat.values, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(mat.columns)), metric_labels, fontsize=9)
    ax.set_yticks(np.arange(len(mat.index)), mat.index, fontsize=7.5)
    ax.set_title("MBAG candidate evidence matrix", loc="left", fontsize=13, weight="bold")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-0.5, len(mat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_fallbacks(payload: dict[str, Any], figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "niche": plot_niche(payload, figure_dir / "fallback_01_molecular_niche_space.png"),
        "matrix": plot_matrix(payload, figure_dir / "fallback_02_candidate_signature_matrix.png"),
        "mangrove": plot_mangrove(payload, figure_dir / "fallback_03_mangrove_fingerprints.png"),
    }


def save_payloads(payload: dict[str, Any], data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, obj in payload.items():
        path = data_dir / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
        out[name] = path
    return out


def render_html(
    summary: dict[str, Any],
    payload: dict[str, Any],
    fallbacks: dict[str, Path],
    infographic: Path,
    sample_risk_abstract: Path,
    d3_source: str,
    source_readiness: list[dict[str, Any]],
) -> str:
    fallback_uri = {k: data_uri(v) for k, v in fallbacks.items()}
    infographic_uri = data_uri(infographic) if infographic.exists() else ""
    sample_risk_abstract_uri = data_uri(sample_risk_abstract) if sample_risk_abstract.exists() else ""
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    metric_cards = "\n".join(
        [
            f"<div class='metric'><b>{summary['multiview_complete']:,}</b><span>tri-view MAG/proteome units explorable now</span></div>",
            f"<div class='metric'><b>{summary['msm_functional']:,}/{summary['msm_total']:,}</b><span>mangrove MAGs with ESM-2, gLM2, and function</span></div>",
            f"<div class='metric'><b>{summary['msm_esm2']:,}/{summary['msm_total']:,}</b><span>mangrove proteome embeddings ready</span></div>",
            f"<div class='metric'><b>{summary['msm_glm2']:,}/{summary['msm_total']:,}</b><span>mangrove genomic-context profiles ready</span></div>",
            f"<div class='metric'><b>{summary['knn_edges']:,}</b><span>high-dimensional embedding-neighborhood links audited</span></div>",
        ]
    )
    citations = [
        ("ESM-2 protein language model", "https://www.science.org/doi/10.1126/science.ade2574"),
        ("Diffusion maps", "https://www.math.pku.edu.cn/teachers/yaoy/Fall2011/Lafon06.pdf"),
        ("PHATE for biological manifolds", "https://www.nature.com/articles/s41587-019-0336-3"),
        ("Bacterial metabolic niche space", "https://www.nature.com/articles/s41467-020-18695-z"),
        ("UMAP", "https://umap-learn.readthedocs.io/"),
    ]
    citation_html = " ".join(f"<a href='{url}'>{html.escape(label)}</a>" for label, url in citations)
    provenance_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['lane']))}</td>"
        f"<td>{safe_int(row['report_units']):,}</td>"
        f"<td>{html.escape(str(row['metadata_universe']))}</td>"
        f"<td>{html.escape(str(row['primary_source']))}</td>"
        f"<td>{html.escape(str(row['resolution_now']))}</td>"
        f"<td>{html.escape(str(row['use_now']))}</td>"
        f"<td>{html.escape(str(row['blocking_gap']))}</td>"
        "</tr>"
        for row in source_readiness
    )
    css = """
    :root{--ink:#16202a;--muted:#64748b;--panel:#ffffff;--surface:#f6fafb;--line:#dbe5ee;--rumen:#c56a13;--wetland:#0284a8;--mangrove:#168a48;--gold:#d89b14}
    *{box-sizing:border-box} body{margin:0;background:var(--surface);color:var(--ink);font-family:Inter,Aptos,Segoe UI,Arial,sans-serif;line-height:1.48}
    header{padding:58px 7vw 34px;background:linear-gradient(135deg,#071b25,#0d3c42 52%,#10523f);color:white}
    .eyebrow{letter-spacing:.12em;text-transform:uppercase;color:#b8f7d8;font-size:12px;font-weight:750}
    h1{font-size:clamp(34px,5vw,62px);line-height:1.02;margin:.35em 0 .25em;max-width:1120px}
    h2{font-size:26px;margin:0 0 12px} h3{font-size:18px;margin:18px 0 8px}
    .subtitle{max-width:980px;font-size:18px;color:#ddfff2}.claim{display:inline-block;margin-top:18px;border:1px solid #85dff1;padding:9px 12px;border-radius:999px;color:#e8fbff}
    main{max-width:1280px;margin:auto;padding:28px 24px 76px}.section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 10px 30px rgba(15,23,42,.05)}
    .metric-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px}.metric{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px}.metric b{display:block;font-size:28px}.metric span{font-size:12px;color:var(--muted)}
    .viz{min-height:540px;border:1px solid var(--line);border-radius:12px;background:#fbfdff;position:relative;overflow:hidden}.viz.tall{min-height:700px}.viz.medium{min-height:470px}.viz.matrix{min-height:760px;overflow:auto}.viz.circos{min-height:720px}
    .grid2{display:grid;grid-template-columns:1.35fr .65fr;gap:18px}.grid2-even{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    .signature-stack{display:grid;grid-template-columns:1fr;gap:22px}.signature-panel{min-width:0}
    .chart-note{font-size:13px;color:var(--muted);margin:8px 0 10px}
    .sample-score-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.sample-score-card{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px}.sample-score-card h3{font-size:15px;margin:0 0 8px}.sample-score-card p{font-size:13px;color:var(--muted);margin:0}
    .sample-risk-abstract{width:100%;margin:18px 0 10px;border:1px solid var(--line);border-radius:14px;background:#fbfdff;box-shadow:0 10px 28px rgba(15,23,42,.06)}
    .figure-caption{font-size:13px;color:var(--muted);line-height:1.52;margin:8px 2px 18px}
    .approach-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}.approach-card{border:1px solid var(--line);border-radius:12px;background:#f8fafc;padding:14px}.approach-card b{display:block;margin-bottom:6px}.approach-card span{font-size:13px;color:var(--muted)}
    .side-card{border:1px solid var(--line);border-radius:12px;background:#f8fafc;padding:16px;min-height:540px}.side-card .muted{font-size:12px;color:var(--muted)}
    .infographic{width:100%;max-height:620px;object-fit:contain;background:#061a22;border-radius:14px;border:1px solid #0c3440}.legend-note{font-size:13px;color:var(--muted);margin-top:10px}
    .tooltip{position:absolute;pointer-events:none;background:#0f172a;color:white;padding:8px 10px;border-radius:8px;font-size:12px;max-width:320px;opacity:0;z-index:20}
    .toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px}.toolbar button{border:1px solid var(--line);background:white;border-radius:999px;padding:7px 10px;cursor:pointer}.toolbar button.active{background:#0f766e;color:white;border-color:#0f766e}
    .legend{display:flex;gap:13px;flex-wrap:wrap;color:var(--muted);font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
    .fallback{margin-top:12px}.fallback img{max-width:100%;border:1px solid var(--line);border-radius:10px}.note{color:var(--muted)}.warn{background:#fff7ed;border-left:4px solid var(--gold);padding:12px;border-radius:10px}
    .pill{display:inline-block;padding:5px 8px;border:1px solid var(--line);border-radius:999px;margin:4px 4px 0 0;color:#334155;background:#f8fafc;font-size:12px}
    .readiness-table{width:100%;border-collapse:collapse;font-size:13px}.readiness-table th{background:#eef7f5;text-align:left}.readiness-table th,.readiness-table td{border:1px solid var(--line);padding:9px;vertical-align:top}.readiness-table td:nth-child(2){font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
    a{color:#075985} .closing{font-size:18px;line-height:1.58}
    @media (max-width:1020px){.metric-grid{grid-template-columns:repeat(2,1fr)}.approach-grid{grid-template-columns:1fr 1fr}.sample-score-grid{grid-template-columns:1fr 1fr}.grid2,.grid2-even{grid-template-columns:1fr}.viz.tall{min-height:560px}}
    @media (max-width:720px){.approach-grid{grid-template-columns:1fr}.metric-grid{grid-template-columns:1fr}.sample-score-grid{grid-template-columns:1fr}}
    """
    js = """
    const ATLAS = __PAYLOAD__;
    const COLORS = {rumen:'#c56a13', wetland:'#0284a8', mangrove:'#168a48', context:'#94a3b8', pending:'#d89b14'};
    const fmt = d3.format(',');
    function tooltip(){ return d3.select('body').append('div').attr('class','tooltip'); }
    function clean(v, fallback='not available'){ return (v === null || v === undefined || String(v).trim()==='' || String(v)==='NaN') ? fallback : String(v); }
    function finiteNum(v){ const x=Number(v); return Number.isFinite(x); }
    function fixed(v, digits=2, fallback='not available'){ return finiteNum(v) ? Number(v).toFixed(digits) : fallback; }
    function rateText(d, key){ return d.rate_metric_status === 'rate_normalized' && finiteNum(d[key]) ? `${Number(d[key]).toFixed(2)} /1k proteins` : 'not rate-normalized'; }
    function taxonomyText(d){
      const parts=[d.domain,d.phylum,d.class,d.order,d.family,d.genus,d.species].filter(v=>v && String(v).trim());
      if(parts.length){ return parts.join(' · '); }
      return d.has_functional ? 'taxonomy unresolved' : 'taxonomy pending functional run';
    }
    function tipText(d){
      return `${clean(d.proteome_id)}<br>
      <span>${clean(d.mag_id,'MAG/bin id not available')}</span><br>
      Lane: ${clean(d.source_display || d.source_category)} · Status: ${clean(d.plot_annotation_status || d.review_tier || d.atlas_inclusion_status)}<br>
      Scope: ${clean(d.analysis_unit_type,'MAG/proteome unit')} · ${clean(d.claim_scope,'MAG/proteome molecular screening')}<br>
      Taxonomy: ${taxonomyText(d)}<br>
      Evidence: methane ${fmt(Number(d.methane_marker_count || d.methane_evidence_score || 0))}; sulfur ${fmt(Number(d.sulfur_context_count || d.sulfur_competition_score || 0))}; substrate ${fmt(Number(d.substrate_breadth_count || 0))}<br>
      Rates: CH4 ${rateText(d,'methane_marker_density_per_1k')}; sulfur ${rateText(d,'sulfur_context_density_per_1k')}<br>
      QC: ${fixed(d.checkm2_completeness,1)}% complete / ${fixed(d.checkm2_contamination,2)}% contam · ${clean(d.qc_tier,'QC tier not available')}<br>
      Provenance: ${clean(d.provenance_resolution_tier)} · ${clean(d.primary_accession || d.source_bucket)}`;
    }
    const cardMap = new Map((ATLAS.cards || []).map(d => [d.proteome_id, d]));
    function updateCard(id){
      const d = cardMap.get(id) || (ATLAS.niche.nodes || []).find(x => x.proteome_id === id) || (ATLAS.candidate_graph.nodes || []).find(x => x.proteome_id === id);
      const box = d3.select('#candidate-card');
      if(!d){ return; }
      box.html(`<h3>${d.candidate_set || 'Molecular atlas node'}</h3>
        <p>${d.proteome_id}</p>
        <p class='muted'>${clean(d.source_display || d.source_category)} · ${clean(d.plot_annotation_status || d.review_tier)}<br>${taxonomyText(d)}</p>
        <p>Bridge affinity: ${fixed(d.bridge_affinity_index || d.nearest_poc_similarity,3)}<br>
        Methane evidence score: ${fmt(Number(d.methane_marker_count || d.methane_evidence_score || 0))}<br>
        Sulfur context score: ${fmt(Number(d.sulfur_context_count || d.sulfur_competition_score || 0))}<br>
        Substrate breadth: ${fmt(Number(d.substrate_breadth_count || 0))} CAZy/MEROPS signals<br>
        Rate status: ${clean(d.rate_metric_status)}<br>
        Provisional internal MBAG review index: ${fixed(d.molecular_attestation_index,3)}</p>
        <p>QC: ${fixed(d.checkm2_completeness,1)}% complete / ${fixed(d.checkm2_contamination,2)}% contamination · ${clean(d.qc_tier,'QC tier not available')}</p>
        <p>Provenance: ${clean(d.provenance_resolution_tier)} · ${clean(d.primary_accession || d.source_bucket)}<br><span class='muted'>${clean(d.metadata_caveat,'No extra metadata caveat recorded.')}</span></p>
        <p>Allowed claim: ${d.allowed_claim_wording || 'MAG/proteome-level molecular screening only.'}</p>
        <p class='muted'>Blocks final MRV scoring: ${d.blocking_gap || 'sample mapping, abundance, environment, uncertainty, and validation.'}<br>Next metadata action: ${clean(d.next_metadata_action,'Resolve sample and validation context.')}</p>`);
    }
    function availableMethods(){
      const node = (ATLAS.niche.nodes || [])[0] || {};
      return ['diffusion','umap','phate','tsne','pca'].filter(m => Number.isFinite(+node[`${m}_1`]) && Number.isFinite(+node[`${m}_2`]));
    }
    function renderNiche(method='diffusion'){
      const el=d3.select('#niche-map'); el.selectAll('*').remove();
      const data=ATLAS.niche.nodes, links=ATLAS.niche.links, w=el.node().clientWidth, h=690, m={t:26,r:30,b:50,l:58}, tip=tooltip();
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]);
      const x=d3.scaleLinear().domain(d3.extent(data,d=>+d[`${method}_1`])).nice().range([m.l,w-m.r]);
      const y=d3.scaleLinear().domain(d3.extent(data,d=>+d[`${method}_2`])).nice().range([h-m.b,m.t]);
      const byId=new Map(data.map(d=>[d.proteome_id,d]));
      svg.append('g').attr('opacity',.18).selectAll('line').data(links.filter(d=>byId.has(d.source)&&byId.has(d.target)).slice(0,900)).join('line')
        .attr('x1',d=>x(+byId.get(d.source)[`${method}_1`])).attr('y1',d=>y(+byId.get(d.source)[`${method}_2`]))
        .attr('x2',d=>x(+byId.get(d.target)[`${method}_1`])).attr('y2',d=>y(+byId.get(d.target)[`${method}_2`]))
        .attr('stroke',d=>d.cross_domain?'#334155':'#cbd5e1').attr('stroke-width',d=>d.cross_domain?0.8:0.35);
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x).ticks(5)); svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
      svg.append('text').attr('x',w/2).attr('y',h-12).attr('text-anchor','middle').attr('font-size',12).text(`${method} coordinate 1`);
      svg.append('text').attr('transform','rotate(-90)').attr('x',-h/2).attr('y',18).attr('text-anchor','middle').attr('font-size',12).text(`${method} coordinate 2`);
      svg.append('g').selectAll('circle').data(data).join('circle')
        .attr('cx',d=>x(+d[`${method}_1`])).attr('cy',d=>y(+d[`${method}_2`]))
        .attr('r',d=>d.has_functional ? 3.2 + 7*Number(d.molecular_attestation_index||0) : 2.4)
        .attr('fill',d=>COLORS[d.source_category] || COLORS.context).attr('stroke','white').attr('stroke-width',.45).attr('opacity',d=>d.has_functional?.78:.28).style('cursor','pointer')
        .on('mouseover',(e,d)=>tip.style('opacity',1).html(tipText(d))).on('mousemove',e=>tip.style('left',`${e.pageX+12}px`).style('top',`${e.pageY+12}px`)).on('mouseout',()=>tip.style('opacity',0)).on('click',(e,d)=>updateCard(d.proteome_id));
    }
    function renderMethodButtons(){
      const methods=availableMethods(), box=d3.select('#method-buttons');
      box.selectAll('button').data(methods).join('button').attr('class',(d,i)=>i===0?'active':null).text(d=>d==='diffusion'?'Diffusion map':d.toUpperCase()).on('click',function(e,d){box.selectAll('button').classed('active',false); d3.select(this).classed('active',true); renderNiche(d);});
      renderNiche(methods[0] || 'diffusion');
    }
    function renderMatrix(){
      const rec=ATLAS.matrix.records || [], metricDefs=ATLAS.matrix.metric_defs || (ATLAS.matrix.metrics || []).map(d=>({id:d,label:d})), metrics=metricDefs.map(d=>d.id);
      const rows=Array.from(new Set(rec.map(d=>d.label))), el=d3.select('#signature-matrix'); el.selectAll('*').remove();
      const w=Math.max(1120, el.node().clientWidth), cellH=30, h=112+rows.length*cellH, m={t:76,r:36,b:36,l:360};
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('width',w).attr('height',h);
      const x=d3.scaleBand().domain(metrics).range([m.l,w-m.r]).padding(.08), y=d3.scaleBand().domain(rows).range([m.t,h-m.b]).padding(.08);
      const c=d3.scaleSequential(d3.interpolateYlGnBu).domain([0,1]), rowMeta=new Map(rec.map(d=>[d.label,d]));
      svg.append('text').attr('x',m.l).attr('y',24).attr('font-size',13).attr('fill','#64748b').text('Rows are review cards; columns are normalized evidence pillars. Click a cell to inspect the candidate.');
      svg.append('g').selectAll('rect.row').data(rows).join('rect').attr('class','row').attr('x',m.l-10).attr('y',d=>y(d)).attr('width',w-m.l-m.r+10).attr('height',y.bandwidth()).attr('fill',(d,i)=>i%2?'#f8fafc':'#ffffff');
      svg.selectAll('rect.cell').data(rec).join('rect').attr('class','cell').attr('x',d=>x(d.metric)).attr('y',d=>y(d.label)).attr('width',x.bandwidth()).attr('height',y.bandwidth()).attr('rx',4).attr('fill',d=>c(d.value)).attr('stroke','white').attr('stroke-width',1.6).style('cursor','pointer').on('click',(e,d)=>updateCard(d.proteome_id)).append('title').text(d=>`${d.label}\\n${d.metric_label}: ${d.value.toFixed(2)}\\n${d.metric_source || ''}`);
      svg.append('g').selectAll('rect.strip').data(rows).join('rect').attr('class','strip').attr('x',m.l-24).attr('y',d=>y(d)).attr('width',9).attr('height',y.bandwidth()).attr('rx',4).attr('fill',d=>COLORS[(rowMeta.get(d)||{}).source_category] || '#64748b');
      svg.append('g').attr('transform',`translate(0,${m.t-12})`).call(d3.axisTop(x).tickFormat(d=>((metricDefs.find(m=>m.id===d)||{}).label)||d)).call(g=>g.select('.domain').remove()).selectAll('text').attr('font-weight',700).attr('font-size',13).attr('fill','#172033');
      svg.append('g').attr('transform',`translate(${m.l-30},0)`).call(d3.axisLeft(y)).call(g=>g.select('.domain').remove()).call(g=>g.selectAll('.tick line').remove()).selectAll('text').attr('font-size',12).attr('fill',d=>COLORS[(rowMeta.get(d)||{}).source_category] || '#334155');
    }
    function renderMangrove(){
      const data=ATLAS.mangrove, el=d3.select('#mangrove-scatter'), w=el.node().clientWidth, h=520, m={t:22,r:30,b:58,l:72}, tip=tooltip();
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]); const x=d3.scaleLinear().domain(d3.extent(data,d=>Math.log1p(d.methane_marker_density_per_1k||0))).nice().range([m.l,w-m.r]);
      const y=d3.scaleLinear().domain(d3.extent(data,d=>Math.log1p(d.sulfur_context_density_per_1k||0))).nice().range([h-m.b,m.t]);
      const r=d3.scaleSqrt().domain(d3.extent(data,d=>d.substrate_breadth_per_1k||0)).range([3,16]); const col=d3.scaleSequential(d3.interpolateYlGnBu).domain(d3.extent(data,d=>d.nearest_poc_similarity||0));
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x)); svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y));
      svg.append('text').attr('x',w/2).attr('y',h-12).attr('text-anchor','middle').text('methane marker density per 1k proteins, log1p');
      svg.append('text').attr('transform','rotate(-90)').attr('x',-h/2).attr('y',20).attr('text-anchor','middle').text('sulfur context density per 1k proteins, log1p');
      svg.selectAll('circle').data(data).join('circle').attr('cx',d=>x(Math.log1p(d.methane_marker_density_per_1k||0))).attr('cy',d=>y(Math.log1p(d.sulfur_context_density_per_1k||0))).attr('r',d=>r(d.substrate_breadth_per_1k||0)).attr('fill',d=>col(d.nearest_poc_similarity||0)).attr('stroke','white').attr('stroke-width',.6).attr('opacity',.77).style('cursor','pointer').on('mouseover',(e,d)=>tip.style('opacity',1).html(tipText({...d,label:d.proteome_id,source_display:'Mangrove/MSM'}))).on('mousemove',e=>tip.style('left',`${e.pageX+12}px`).style('top',`${e.pageY+12}px`)).on('mouseout',()=>tip.style('opacity',0)).on('click',(e,d)=>updateCard(d.proteome_id));
    }
    function renderCircos(){
      const data=ATLAS.circos || {}, records=data.records || [], pillars=data.pillars || [], groups=data.groups || [];
      const el=d3.select('#candidate-circos'); el.selectAll('*').remove();
      const w=Math.max(760, el.node().clientWidth), h=720, cx=w/2, cy=310, outer=220, inner=88, ringStep=(outer-inner)/Math.max(groups.length,1), labelR=outer+38;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]); const g=svg.append('g').attr('transform',`translate(${cx},${cy})`);
      const angle=d3.scaleBand().domain(pillars.map(d=>d.id)).range([0,Math.PI*2]).padding(.16);
      const arc=d3.arc(); const groupColor=new Map(groups.map(d=>[d.id,d.color]));
      groups.forEach((grp,gi)=>{
        const r0=inner+gi*ringStep+4, r1=r0+ringStep-10;
        g.append('circle').attr('r',r0).attr('fill','none').attr('stroke','#dbe5ee').attr('stroke-dasharray','2 4');
        g.append('circle').attr('r',r1).attr('fill','none').attr('stroke','#edf3f7');
      });
      g.selectAll('path.bar').data(records).join('path').attr('class','bar')
        .attr('d',d=>{const gi=groups.findIndex(g=>g.id===d.group); const base=inner+gi*ringStep+6; const maxLen=ringStep-16; return arc({innerRadius:base, outerRadius:base+4+Math.max(1,maxLen*d.average_value), startAngle:angle(d.pillar), endAngle:angle(d.pillar)+angle.bandwidth()});})
        .attr('fill',d=>groupColor.get(d.group) || '#64748b').attr('opacity',d=>0.36+0.58*Math.max(d.high_share,d.average_value)).attr('stroke','white').attr('stroke-width',.8)
        .append('title').text(d=>`${d.group_label} · ${d.pillar_label}\\nAverage support: ${d.average_value.toFixed(2)}\\nHigh-support candidates: ${d.high_count}/${d.candidate_count}\\nEvidence source: ${d.source}`);
      g.selectAll('text.pillar').data(pillars).join('text').attr('class','pillar').attr('font-size',11).attr('font-weight',700).attr('fill','#334155')
        .attr('x',d=>{const a=angle(d.id)+angle.bandwidth()/2-Math.PI/2; return Math.cos(a)*labelR;})
        .attr('y',d=>{const a=angle(d.id)+angle.bandwidth()/2-Math.PI/2; return Math.sin(a)*labelR+4;})
        .attr('text-anchor',d=>{const a=angle(d.id)+angle.bandwidth()/2-Math.PI/2; const c=Math.cos(a); return Math.abs(c)<.18?'middle':c>0?'start':'end';})
        .text(d=>d.short);
      g.append('circle').attr('r',inner-22).attr('fill','#f8fafc').attr('stroke','#dbe5ee');
      g.append('text').attr('text-anchor','middle').attr('y',-8).attr('font-weight',800).attr('font-size',16).text('MBAG');
      g.append('text').attr('text-anchor','middle').attr('y',14).attr('font-size',11).attr('fill','#64748b').text('candidate pillars');
      const legend=svg.append('g').attr('transform',`translate(${Math.max(24,cx-310)},22)`);
      groups.forEach((grp,i)=>{const count=records.find(r=>r.group===grp.id)?.candidate_count || 0; const row=legend.append('g').attr('transform',`translate(${i*205},0)`); row.append('rect').attr('width',12).attr('height',12).attr('rx',3).attr('fill',grp.color); row.append('text').attr('x',18).attr('y',10).attr('font-size',12).attr('fill','#334155').text(`${grp.label} cards n=${count}`);});
      const footer=svg.append('text').attr('x',cx).attr('y',h-48).attr('text-anchor','middle').attr('font-size',12).attr('fill','#64748b');
      footer.append('tspan').attr('x',cx).text('Ring length shows average support across selected candidate cards.');
      footer.append('tspan').attr('x',cx).attr('dy','1.35em').text('Opacity increases when more cards clear the support threshold; wetland is single-card in this render.');
    }
    renderMethodButtons(); renderMatrix(); renderMangrove(); renderCircos(); updateCard((ATLAS.cards[0]||{}).proteome_id);
    """.replace("__PAYLOAD__", payload_json)

    infographic_block = ""
    if infographic_uri:
        infographic_block = f"""
        <section class="section">
          <h2>The Operating Model Behind The Atlas</h2>
          <img class="infographic" src="{infographic_uri}" alt="MethaNet agent-assisted molecular intelligence workflow infographic">
          <p class="legend-note">The infographic summarizes the operating flywheel: concept framing, compute-agnostic environment setup, reference database assembly, MAG/proteome processing, multiview feature generation, report-ready interpretation, and future API/MCP delivery. This is MethaNet's execution moat, while the biological claims below still come only from audited ESM-2, functional annotation, gLM2, QC, taxonomy, and provenance evidence.</p>
        </section>
        """

    sample_risk_abstract_block = ""
    if sample_risk_abstract_uri:
        sample_risk_abstract_block = f"""
        <img class="sample-risk-abstract" src="{sample_risk_abstract_uri}" alt="Graphical abstract showing MethaNet MAG/proteome molecular fingerprints flowing through sample linkage, abundance weighting, environmental covariates, uncertainty, and validation gates into sample risk readiness labels before any final MRV risk score.">
        <p class="figure-caption">Graphical abstract. MethaNet's current evidence layer operates at MAG/proteome grain: ESM-2 and gLM2 context, methane/sulfur/substrate annotations, QC, taxonomy, and provenance define molecular fingerprints for bridge-candidate review. The next product layer links those fingerprints to physical samples or metagenomes, weights them by MAG/read abundance and unbinned marker evidence, adds environmental permissiveness covariates, uncertainty, and flux/process validation status, then emits sample risk readiness labels such as not_scoreable, needs_metadata, needs_abundance, needs_environment, needs_flux_validation, monitor_more, or scoreable_provisional when evidence is sufficient. This is an evidence-to-readiness workflow for monitoring and validation design, not a final MRV risk score, measured methane-flux result, A-E risk tier, or carbon-credit approval.</p>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MethaNet Molecular Niche Atlas</title>
  <style>{css}</style>
</head>
<body>
<header>
  <div class="eyebrow">MethaNet Bridge Attestation Graph · Molecular Niche Atlas</div>
  <h1>Molecular intelligence for blue-carbon methane risk starts here</h1>
  <p class="subtitle">From latent proteome geometry to auditable bridge evidence: a MAG/proteome atlas built to prioritize where methane-risk mechanisms deserve monitoring, validation, and product attention.</p>
  <div class="claim">{html.escape(CLAIM_BOUNDARY)}</div>
</header>
<main>
  <section class="section">
    <h2>Executive Summary</h2>
    <p>MethaNet's near-term product is not another annotation report; it is a molecular attestation layer for methane permanence-risk screening. MBAG, the MethaNet Bridge Attestation Graph, asks whether a latent ESM-2 bridge candidate is supported by independent functional biology, genomic context, QC, taxonomy, and provenance. That turns raw MAG outputs into reviewable evidence objects a partner can understand and a technical reviewer can challenge.</p>
    <p>The market problem is timing. Blue-carbon projects need earlier warning on methane permanence risk, but field flux data are expensive, sparse, and often retrospective. MBAG does not replace flux validation; it creates an auditable molecular intelligence layer that helps decide where monitoring, diligence, and experimental validation should focus before a project depends on late-stage surprises.</p>
    <p>The atlas is now large enough to inspect cross-ecosystem molecular patterns, not just report run progress. This snapshot contains {summary['multiview_complete']:,} tri-view MAG/proteome units: {summary['poc_core_total']:,} MAG/bin-comparable POC rumen + wetland/MUCC units and {summary['msm_multiview']:,} mangrove/MSM units with ESM-2, gLM2, and functional evidence.</p>
    <p>Mangrove/MSM is now a target-domain screening lane, not yet a sample-level MRV layer. All {summary['msm_total']:,} mangrove proteomes have ESM-2 embeddings and gLM2 context, while {summary['msm_functional']:,} have functional evidence in this latest render. The remaining {summary['msm_function_pending']:,} are retained as pending-functional records, so downstream readers can separate absence of evidence from evidence of absence.</p>
    <p>The strongest value today is decision-grade triage. This report identifies bridge-like molecular neighborhoods, candidate mechanism signatures, provenance readiness, and monitoring-priority hypotheses. It supports partner-facing proof points and MRV feature primitives, while explicitly stopping short of final sample-level risk tiers, measured methane flux, or crediting decisions.</p>
    <div class="metric-grid">{metric_cards}</div>
  </section>
  {infographic_block}
  <section class="section">
    <h2>Source Provenance And Environmental Readiness</h2>
    <p>Environmental metadata now makes MBAG provenance-aware, not sample-scoreable. The report can explain where each evidence lane comes from and how close it is to sample/site rollup, but it cannot yet convert MAG-level molecular potential into final environmental methane-risk scores. That distinction is strategically valuable: it tells partners what is already auditable and tells the team exactly what metadata gaps must close next.</p>
    <table class="readiness-table">
      <thead><tr><th>Evidence lane</th><th>Report units</th><th>Metadata universe</th><th>Primary source</th><th>Resolution now</th><th>Use now</th><th>Blocking gap</th></tr></thead>
      <tbody>{provenance_rows_html}</tbody>
    </table>
    <p class="note">Denominator note: the POC crosswalk and environmental reconciliation cover the broader 662 embedded POC proteome universe, while this report renders the 625 MAG/bin-comparable POC units plus 1,428 mangrove candidates. Rows that are pending, mixed-resolution, or not yet sample-linkable remain visible as readiness states rather than being treated as biological absence.</p>
  </section>
  <section class="section">
    <h2>MBAG: Three Molecular Views, One Attestation Layer</h2>
    <p>The MethaNet Bridge Attestation Graph is designed to turn molecular similarity into a reviewable evidence trail. A 2D embedding bridge can be a powerful discovery signal, but it is not a mechanism. MBAG therefore treats each candidate as a convergence test across three molecular views plus explicit guardrails.</p>
    <div class="approach-grid">
      <div class="approach-card"><b>ESM-2 proteome geometry</b><span>Protein-language embeddings provide a high-dimensional hypothesis engine for MAG/proteome similarity. Bridge evidence is read from neighborhoods and graph links, not from a decorative UMAP alone.</span></div>
      <div class="approach-card"><b>Functional annotations</b><span>Alignment/HMM-based calls from MCycDB, SCycDB, KOfam, dbCAN/CAZy, METABOLIC, MEROPS, and Bakta provide interpretable molecular machinery: methane routes, sulfur competition, substrate breadth, and pathway context.</span></div>
      <div class="approach-card"><b>gLM2 genomic context</b><span>Native-minus-shuffled context checks test whether gene-neighborhood architecture carries extra signal beyond a bag of annotated genes. This is a complementary context layer, not a replacement for curated functional evidence.</span></div>
      <div class="approach-card"><b>QC and provenance guardrails</b><span>CheckM2, GUNC, GTDB-Tk, annotation coverage, source labels, and missingness protect against attractive artifacts. Weak evidence remains visible instead of being silently dropped.</span></div>
    </div>
    <p class="note">For partner communication, the useful output is a candidate card, a feature primitive, or a monitoring-priority hypothesis. The proprietary scoring details can remain internal; the report exposes the evidence pillars and claim boundaries needed for trust.</p>
  </section>
  <section class="section">
    <h2>Why Molecular Niche Space Is The Right Lens</h2>
    <p>The core question is geometric and biological at the same time. Methane permanence risk is not driven by one marker in isolation; it emerges from molecular machinery, substrate access, redox and sulfur context, genome quality, ecological source, and eventually sample abundance and environmental conditions. Molecular niche space gives MethaNet a way to ask whether MAGs occupy similar biological neighborhoods before claiming anything about site-level methane behavior.</p>
    <p>In this report, ESM-2 defines the latent neighborhood structure, functional annotations explain what those neighborhoods encode, and gLM2 tests whether local genomic organization adds contextual support. The right interpretation is not "this MAG emits methane"; it is "this MAG deserves review because multiple molecular views point to a methane-relevant niche hypothesis."</p>
    <p class="note">Scientific anchors: {citation_html}. Diffusion coordinates are used as the primary niche-space view because they are built directly from a kNN affinity graph; UMAP, PHATE, t-SNE, and PCA are included as comparison views when available, never as standalone proof of bridge status.</p>
  </section>
  <section class="section">
    <h2>Molecular Niche-Space Bridge Map</h2>
    <p>The bridge map is a navigation layer, not the final decision rule. It displays MAG/proteome units in nonlinear molecular niche space and overlays high-dimensional cosine kNN links from the original 1,280-dimensional ESM-2 embedding space. Points are colored by source ecosystem, point size follows the provisional internal MBAG review index, and dim points mark units that still lack functional evidence.</p>
    <p>The practical use is fast triage: find cross-domain neighborhoods where a wetland or mangrove MAG sits near methane-relevant rumen biology, then ask whether the functional, gLM2, QC, and taxonomy layers agree or contradict that bridge hypothesis.</p>
    <div class="toolbar" id="method-buttons"></div>
    <div class="legend"><span><i class="dot" style="background:var(--rumen)"></i>Rumen</span><span><i class="dot" style="background:var(--wetland)"></i>Wetland/MUCC</span><span><i class="dot" style="background:var(--mangrove)"></i>Mangrove/MSM</span></div>
    <div class="grid2">
      <div id="niche-map" class="viz tall"></div>
      <aside id="candidate-card" class="side-card"><p class="note">Click a node or signature cell to inspect bridge evidence.</p></aside>
    </div>
    <details class="fallback"><summary>Static fallback</summary><img src="{fallback_uri['niche']}" alt="Molecular niche-space fallback"></details>
  </section>
  <section class="section">
    <h2>Candidate Signatures Turn Audit Tables Into Review Cards</h2>
    <p>MBAG candidates should be inspected as signatures, not as a long spreadsheet. The TSV evidence remains in the artifact package for audit, but the report view emphasizes the evidence pillars a scientist, partner, or investor can reason about: bridge geometry, gLM2 context, methane machinery, sulfur context, carbon/substrate breadth, annotation coverage, and QC support.</p>
    <p>The matrix shows candidate-level support. The radial evidence-pillar view summarizes selected candidate-card support for POC rumen, POC wetland, and mangrove/MSM candidates; the wetland value is a single-card value in this render. These are review signals, not full-lane summaries and not final methane-risk scores.</p>
    <div class="signature-stack">
      <div class="signature-panel">
        <h3>MBAG candidate evidence matrix</h3>
        <p class="chart-note">Rows are candidate cards and columns are normalized evidence pillars. The source-colored strip at left separates POC rumen, POC wetland, and mangrove/MSM candidates.</p>
        <div id="signature-matrix" class="viz medium matrix"></div>
      </div>
      <div class="signature-panel">
        <h3>Evidence pillar wheel</h3>
        <p class="chart-note">Each ring is a selected candidate source group; longer arcs indicate stronger average support for that evidence pillar across the selected candidate cards.</p>
        <div id="candidate-circos" class="viz medium circos"></div>
      </div>
    </div>
    <details class="fallback"><summary>Static fallback</summary><img src="{fallback_uri['matrix']}" alt="Candidate signature matrix fallback"></details>
  </section>
  <section class="section">
    <h2>Mangrove Molecular Fingerprint Landscape</h2>
    <p>Mangrove turns the report from a POC bridge story into a target-domain screening surface. This panel uses only mangrove MAGs with completed functional outputs. The axes are normalized by protein count, so the pattern reflects marker density rather than raw annotation volume: x is methane marker density per 1,000 proteins, y is sulfur-context density per 1,000 proteins, size is CAZy/MEROPS substrate breadth per 1,000 proteins, and color is nearest POC embedding similarity.</p>
    <p>Commercially, this is the first layer of a monitoring-priority map: candidates with strong methane machinery, weak sulfur competition context, broad substrate capacity, and good QC are not "high-risk sites" yet, but they are the kind of molecular signal that should shape sampling, flux validation, and partner diligence.</p>
    <div id="mangrove-scatter" class="viz"></div>
    <details class="fallback"><summary>Static fallback</summary><img src="{fallback_uri['mangrove']}" alt="Mangrove molecular fingerprint fallback"></details>
  </section>
  <section class="section">
    <h2>From MAG Fingerprints To Environmental Methane-Risk Scoring</h2>
    <p>The atlas becomes commercially transformative when MAG/proteome fingerprints can be rolled up to physical samples, metagenomes, sites, and monitoring periods. The current report is deliberately MAG-level: it says which genomes encode methane-relevant potential, sulfur competition context, substrate-processing capacity, QC confidence, and bridge-neighborhood evidence. Environmental methane-risk scoring begins only when those molecular fingerprints are weighted by abundance, connected to sample provenance, and interpreted alongside measured or carefully tiered environmental covariates.</p>
    {sample_risk_abstract_block}
    <p>A defensible sample score should therefore combine four layers. First, the molecular layer estimates methane machinery, sulfur competition, substrate breadth, genomic context, and annotation confidence across all MAGs and unbinned marker evidence in the sample. Second, the community layer weights those signals by read coverage, relative or absolute abundance, pathway redundancy, and the fraction of community signal left unassembled or unbinned. Third, the environmental layer captures whether local conditions make methane expression plausible: salinity, sulfate, redox or oxygen proxies, pH, temperature, organic carbon quantity and quality, sediment depth, vegetation, tidal/hydrologic regime, season, and restoration status. Fourth, the validation layer anchors the score against chamber fluxes, dissolved methane, porewater chemistry, incubation assays, or repeated field observations.</p>
    <div class="sample-score-grid">
      <div class="sample-score-card"><h3>1. Link molecules to samples</h3><p>Resolve MAG-to-sample and MAG-to-site provenance, keep resolution tiers, and preserve unlinked MAGs as not scoreable rather than forcing a risk estimate.</p></div>
      <div class="sample-score-card"><h3>2. Weight by community abundance</h3><p>Turn genome potential into sample capacity using MAG coverage, marker abundance, unbinned functional reads, and uncertainty from incomplete assembly.</p></div>
      <div class="sample-score-card"><h3>3. Add environmental permissiveness</h3><p>Use measured metadata first, modeled covariates second, and mark every salinity, sulfate, redox, substrate, depth, and vegetation field by evidence tier.</p></div>
      <div class="sample-score-card"><h3>4. Calibrate with field evidence</h3><p>Use flux, porewater, geochemistry, and temporal resampling to learn which molecular signatures predict methane risk under real blue-carbon conditions.</p></div>
    </div>
    <p>This is why field work is not a downstream formality; it is the learning engine that turns MBAG from a molecular atlas into a progressively stronger risk system. Dense sampling across mangroves, salt marshes, freshwater wetlands, restored sites, degraded sites, salinity gradients, depth profiles, seasons, and management regimes will expand the molecular niche map, expose source-specific blind spots, and calibrate which bridge signatures matter in blue-carbon settings. Every new sample can strengthen the atlas if it arrives with clean provenance, abundance, environmental measurements, and a validation target.</p>
    <p>The immediate product interpretation is a sample risk readiness layer, not a final A-E risk tier. A sample can be labeled scoreable, monitor more, needs metadata, needs abundance, needs environmental covariates, or needs flux validation. That language is honest, useful for partners, and strategically powerful: it converts MethaNet's current molecular evidence into a concrete sampling and diligence plan while building the evidence base needed for future calibrated methane-risk scoring.</p>
  </section>
  <section class="section">
    <h2>Strategic Readout: From Molecular Attestation To MRV Product Readiness</h2>
    <p class="closing">Bottom line: MBAG is no longer just a visualization; it is a defensible evidence product. The expanded atlas turns thousands of MAG/proteome artifacts into a queryable evidence surface where bridge-neighborhood structure, methane/sulfur/substrate annotations, genomic-context perturbation, QC, taxonomy, and provenance can be inspected together. That is already valuable for partner proof, candidate prioritization, monitoring design, and future dashboard/API feature delivery.</p>
    <p class="closing">Now, MethaNet can show that molecular bridge candidates are not selected from embedding geometry alone; they are audited against independent functional machinery, genome context, quality controls, taxonomy, and provenance readiness. Next, those MAG/proteome signatures must be converted into sample-linked, abundance-aware, uncertainty-aware MRV feature primitives. Not yet: no final risk tiers, no measured methane flux, and no carbon-crediting conclusion should be inferred from this artifact.</p>
    <p class="closing">The strategic implication is clear and bounded. MethaNet is building an upstream molecular attestation layer for blue-carbon methane diligence: a system that can flag where permanence risk deserves attention, explain why a candidate is plausible or weak, and preserve the evidence chain for technical review. The stronger MRV claim comes only after MAG-to-sample links, abundance/read coverage, environmental covariates, phylogeny/source controls, uncertainty propagation, and validation against flux or process measurements are in place.</p>
    <div class="warn">Forbidden claims remain locked: no final A-E risk tiers, no measured methane flux, no carbon-credit approval, and no source-independent rumen-to-wetland/mangrove transfer claims from this artifact alone.</div>
  </section>
</main>
<script>{d3_source}</script>
<script>{js}</script>
</body>
</html>"""


def write_outputs(
    output_dir: Path,
    atlas: pd.DataFrame,
    emb_meta: pd.DataFrame,
    edge_df: pd.DataFrame,
    cards: pd.DataFrame,
    status: pd.DataFrame,
    payload_paths: dict[str, Path],
    fallback_paths: dict[str, Path],
    summary: dict[str, Any],
    manifold_methods: list[dict[str, str]],
    source_readiness: list[dict[str, Any]],
    validation_gates: list[dict[str, Any]],
    infographic_path: Path,
    sample_risk_abstract_path: Path,
    html_text: str,
) -> None:
    table_dir = output_dir / "tables"
    source_dir = output_dir / "sources"
    table_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(table_dir / "atlas_multiview_feature_table.tsv", sep="\t", index=False)
    emb_meta.to_csv(table_dir / "embedding_context_table.tsv", sep="\t", index=False)
    edge_df.to_csv(table_dir / "bridge_knn_edges.tsv", sep="\t", index=False)
    cards.to_csv(table_dir / "candidate_cards.tsv", sep="\t", index=False)
    status.to_csv(table_dir / "mangrove_payload_status.tsv", sep="\t", index=False)
    pd.DataFrame(manifold_methods).to_csv(table_dir / "manifold_method_status.tsv", sep="\t", index=False)
    pd.DataFrame(source_readiness).to_csv(table_dir / "source_provenance_readiness.tsv", sep="\t", index=False)
    pd.DataFrame(validation_gates).to_csv(table_dir / "report_validation_gates.tsv", sep="\t", index=False)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim": "Expanded atlas supports MAG/proteome molecular screening",
                "status": "allowed",
                "allowed_wording": "MethaNet can inspect multiview molecular evidence for completed MAG/proteome units.",
                "blocking_gap": "none for MAG/proteome screening",
            },
            {
                "claim": "Bridge candidates are review-ready hypotheses",
                "status": "allowed with caveats",
                "allowed_wording": "Candidates can be prioritized for source-aware review using ESM-2, gLM2, function, QC, and taxonomy evidence.",
                "blocking_gap": "source-aware nulls, phylogeny comparison, sample linkage",
            },
            {
                "claim": "Final MRV risk score or A-E tier",
                "status": "forbidden",
                "allowed_wording": "Not available from this artifact.",
                "blocking_gap": "sample mapping, abundance, environmental covariates, uncertainty, and field/process validation",
            },
        ]
    )
    claim_matrix.to_csv(table_dir / "claim_boundary_matrix.tsv", sep="\t", index=False)
    if infographic_path.exists():
        shutil.copy2(infographic_path, output_dir / "assets/figures/methanet_agentic_workflow_moat_v2.png")
    if sample_risk_abstract_path.exists():
        shutil.copy2(
            sample_risk_abstract_path,
            output_dir / "assets/figures/figure_04_mag_to_sample_risk_readiness_graphical_abstract.png",
        )
    report = output_dir / "report.html"
    report.write_text(html_text)
    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "report": str(report),
        "summary": summary,
        "tables": {
            "atlas_multiview_feature_table": str(table_dir / "atlas_multiview_feature_table.tsv"),
            "embedding_context_table": str(table_dir / "embedding_context_table.tsv"),
            "bridge_knn_edges": str(table_dir / "bridge_knn_edges.tsv"),
            "candidate_cards": str(table_dir / "candidate_cards.tsv"),
            "mangrove_payload_status": str(table_dir / "mangrove_payload_status.tsv"),
            "manifold_method_status": str(table_dir / "manifold_method_status.tsv"),
            "source_provenance_readiness": str(table_dir / "source_provenance_readiness.tsv"),
            "report_validation_gates": str(table_dir / "report_validation_gates.tsv"),
            "claim_boundary_matrix": str(table_dir / "claim_boundary_matrix.tsv"),
        },
        "interactive_payloads": {k: str(v) for k, v in payload_paths.items()},
        "fallback_figures": {k: str(v) for k, v in fallback_paths.items()},
        "infographic": str(infographic_path),
        "sample_risk_graphical_abstract": str(sample_risk_abstract_path),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / "report_bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (output_dir / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # MethaNet Next-Generation Molecular Niche Atlas

            Generated: {summary['generated_at_utc']}

            Main artifact: `report.html`

            ## Snapshot

            - POC core tri-view complete: {summary['poc_core_total']:,}/{summary['poc_core_total']:,}
            - Mangrove/MSM ESM-2 complete: {summary['msm_esm2']:,}/{summary['msm_total']:,}
            - Mangrove/MSM gLM2 complete: {summary['msm_glm2']:,}/{summary['msm_total']:,}
            - Mangrove/MSM functional complete: {summary['msm_functional']:,}/{summary['msm_total']:,}
            - Expanded tri-view atlas: {summary['multiview_complete']:,}

            ## Claim Boundary

            {CLAIM_BOUNDARY}

            ## Regenerate

            ```bash
            source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
            conda activate methanet-fgx
            python scripts/reports/build_mbag_nextgen_molecular_niche_atlas.py
            ```
            """
        )
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve(repo_root, args.output_dir) if args.output_dir else repo_root / f"results/reports/mbag_nextgen_molecular_niche_atlas_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    poc_esm_dir = resolve(repo_root, args.poc_esm_dir)
    poc_warehouse_dir = resolve(repo_root, args.poc_warehouse_dir)
    poc_glm_dir = resolve(repo_root, args.poc_glm_dir)
    msm_root = resolve(repo_root, args.msm_root)
    msm_esm_dir = resolve(repo_root, args.msm_esm_dir)
    msm_glm_dir = resolve(repo_root, args.msm_glm_dir)
    infographic = resolve(repo_root, args.infographic)
    sample_risk_abstract = resolve(repo_root, args.sample_risk_abstract)
    assert (
        poc_esm_dir
        and poc_warehouse_dir
        and poc_glm_dir
        and msm_root
        and msm_esm_dir
        and msm_glm_dir
        and infographic
        and sample_risk_abstract
    )

    poc = legacy.load_poc_features(poc_warehouse_dir, poc_glm_dir, poc_esm_dir)
    msm, msm_status, msm_esm_stats = legacy.load_msm_features(msm_root, msm_esm_dir, msm_glm_dir)
    atlas = pd.concat([poc, msm], ignore_index=True, sort=False)

    emb_meta, edge_df, embeddings = legacy.build_embedding_context(poc_esm_dir, msm_esm_dir, atlas, args.knn)
    emb_meta, edge_df, embeddings = rebuild_scoped_embedding_context(emb_meta, embeddings, atlas, args.knn)
    manifold_df, manifold_methods = compute_manifold_coordinates(
        embeddings,
        args.knn,
        skip_umap=args.skip_umap,
        skip_phate=args.skip_phate,
        skip_tsne=args.skip_tsne,
    )
    manifold_df = manifold_df.drop(columns=[c for c in manifold_df.columns if c in emb_meta.columns], errors="ignore")
    emb_meta = pd.concat([emb_meta.reset_index(drop=True), manifold_df.reset_index(drop=True)], axis=1)
    emb_meta = emb_meta.loc[:, ~emb_meta.columns.duplicated()]
    atlas = legacy.add_report_metrics(atlas, emb_meta)
    manifold_cols = [c for c in emb_meta.columns if c.endswith("_1") or c.endswith("_2")]
    manifold_cols = [c for c in manifold_cols if c not in atlas.columns]
    if manifold_cols:
        atlas = atlas.merge(
            emb_meta[["proteome_id"] + manifold_cols].drop_duplicates("proteome_id"),
            on="proteome_id",
            how="left",
        )
    atlas = add_molecular_metrics(atlas)
    atlas["review_tier"] = atlas.apply(classify_review_tier, axis=1)
    atlas["allowed_claim_wording"] = (
        "MAG/proteome-level molecular screening and monitoring-priority hypothesis only; "
        "requires sample, abundance, environmental, uncertainty, and validation layers before MRV scoring."
    )
    atlas["blocking_gap"] = (
        "sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, "
        "phylogeny/source controls, and flux/process validation"
    )
    atlas["next_validation_action"] = (
        "connect to sample metadata, run source-aware nulls, compare phylogeny versus embedding proximity, "
        "and validate against flux or process measurements before risk scoring"
    )
    atlas = add_source_provenance_context(atlas, repo_root, msm_root)
    cards = build_candidate_cards(atlas, args.top_n_poc, args.top_n_mangrove)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(repo_root),
        "poc_core_total": int(poc["has_functional"].sum()),
        "poc_rumen_total": int(poc["source_category"].eq("rumen").sum()),
        "poc_wetland_total": int(poc["source_category"].eq("wetland").sum()),
        "msm_total": int(len(msm)),
        "msm_esm2": int(msm["has_esm2"].sum()),
        "msm_glm2": int(msm["has_glm2"].sum()),
        "msm_functional": int(msm["has_functional"].sum()),
        "msm_multiview": int((msm["has_esm2"] & msm["has_glm2"] & msm["has_functional"]).sum()),
        "msm_function_pending": int((~msm["has_functional"]).sum()),
        "msm_esm_embedded_total_with_resume": int(msm_esm_stats.get("embedded_total_with_resume") or 0),
        "msm_esm_pending_remaining": int(msm_esm_stats.get("pending_remaining") or 0),
        "multiview_complete": int(
            poc["has_functional"].sum() + (msm["has_esm2"] & msm["has_glm2"] & msm["has_functional"]).sum()
        ),
        "embedding_context_total": int(len(emb_meta)),
        "knn_edges": int(len(edge_df)),
    }
    source_readiness = build_source_provenance_readiness(summary)

    payload = build_payloads(atlas, emb_meta, edge_df, cards, summary, args.graph_node_cap, manifold_methods)
    validation_gates = build_report_validation_gates(atlas, payload)
    failed_gates = [gate for gate in validation_gates if gate["status"] != "pass"]
    if failed_gates:
        raise SystemExit(f"Report validation gates failed: {failed_gates}")
    summary["report_validation_gates"] = len(validation_gates)
    summary["report_validation_failures"] = len(failed_gates)
    summary["graph_node_count"] = len(payload["candidate_graph"]["nodes"])
    summary["graph_edge_count"] = len(payload["candidate_graph"]["links"])
    payload_paths = save_payloads(payload, output_dir / "assets/data")
    fallback_paths = build_fallbacks(payload, output_dir / "assets/figures")
    d3_path, d3_source = legacy.fetch_d3(output_dir / "assets/js")
    summary["interactive_runtime_asset"] = str(d3_path)
    html_text = render_html(summary, payload, fallback_paths, infographic, sample_risk_abstract, d3_source, source_readiness)
    write_outputs(
        output_dir=output_dir,
        atlas=atlas,
        emb_meta=emb_meta,
        edge_df=edge_df,
        cards=cards,
        status=msm_status,
        payload_paths=payload_paths,
        fallback_paths=fallback_paths,
        summary=summary,
        manifold_methods=manifold_methods,
        source_readiness=source_readiness,
        validation_gates=validation_gates,
        infographic_path=infographic,
        sample_risk_abstract_path=sample_risk_abstract,
        html_text=html_text,
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary, "manifold_methods": manifold_methods}, indent=2))


if __name__ == "__main__":
    main()
