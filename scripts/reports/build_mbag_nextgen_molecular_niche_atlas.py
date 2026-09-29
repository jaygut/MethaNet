#!/usr/bin/env python3
"""Build the next-generation MethaNet MBAG molecular niche atlas.

This report consolidates the closed rumen/wetland POC core with the registered
mangrove expansion payloads. The reader-facing artifact foregrounds molecular
niche-space structure, bridge-neighborhood evidence, candidate signatures,
and claim boundaries. It does not assign sample-level MRV scores, flux claims,
or carbon-crediting conclusions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
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
import matplotlib.colors
import matplotlib.patches
import matplotlib.ticker
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
from atlas_embedding_contract import validate_embedding_contract


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFOGRAPHIC = Path(
    "ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/"
    "infographics/methanet_agentic_workflow_moat_20260616/"
    "methanet_agentic_workflow_moat_v3.png"
)
DEFAULT_SAMPLE_RISK_ABSTRACT = Path(
    "ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/"
    "infographics/methanet_sample_risk_readiness_graphical_abstract_20260619/"
    "methanet_mag_to_sample_risk_readiness_graphical_abstract.png"
)
CLAIM_BOUNDARY = (
    "Current evidence supports genome-level molecular screening, candidate review and "
    "measurement planning. Calibrated sample risk and any crediting use require sample "
    "linkage, abundance, environmental context, uncertainty and field validation."
)
# The published report no longer renders the unlabeled graphical abstract; the
# four gated layers are shown as a labeled status list instead.
RENDER_SAMPLE_RISK_ABSTRACT = False

# Lane colors mirror the landing page's four source hues, darkened for a light page.
COLORS = {
    "rumen": "#db2777",
    "wetland": "#65a30d",
    "mangrove": "#0891b2",
    "msm": "#0891b2",
    "futian": "#6366f1",
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
    parser.add_argument("--embedding-contract", type=Path,
                        default=Path("configs/atlas_embedding_contract_20260929.json"),
                        help="Hash-bound configuration evidence for every ESM-2 geometry input.")
    parser.add_argument("--poc-esm-dir", type=Path, default=legacy.DEFAULT_POC_ESM_DIR)
    parser.add_argument("--poc-warehouse-dir", type=Path, default=legacy.DEFAULT_POC_WAREHOUSE_DIR)
    parser.add_argument("--poc-glm-dir", type=Path, default=legacy.DEFAULT_POC_GLM_DIR)
    parser.add_argument("--msm-root", type=Path, default=legacy.DEFAULT_MSM_ROOT)
    parser.add_argument("--msm-esm-dir", type=Path, default=legacy.DEFAULT_MSM_ESM_DIR)
    parser.add_argument("--msm-glm-dir", type=Path, default=legacy.DEFAULT_MSM_GLM_DIR)
    parser.add_argument(
        "--lane-registry",
        type=Path,
        default=Path("configs/methanet_atlas_lanes_20260929.tsv"),
        help="Atlas lane registry TSV. When present, report inputs are derived from this registry.",
    )
    parser.add_argument(
        "--allow-legacy-defaults",
        action="store_true",
        help="Deprecated compatibility flag; unverified historical geometry is no longer accepted.",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=None,
        help="Optional 3-view freeze_manifest.tsv to annotate report rows and preserve the exact payload snapshot.",
    )
    parser.add_argument(
        "--release-ledger",
        type=Path,
        default=None,
        help="Release-ledger JSON generated beside the freeze manifest.",
    )
    parser.add_argument(
        "--infographic",
        type=Path,
        default=None,
        help=(
            "Optional operating-model infographic. Omitted by default so stale "
            "headline counts cannot enter a scientific release implicitly."
        ),
    )
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


def asset_href(path: Path, output_dir: Path) -> str:
    """Return a browser-safe relative asset path for files inside the report bundle."""
    try:
        return html.escape(path.resolve().relative_to(output_dir.resolve()).as_posix())
    except ValueError:
        return html.escape(path.as_posix())


def copy_report_asset(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, destination)
    return destination


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
        "rumen": "Rumen reference",
        "wetland": "Wetland",
        "mangrove": "Mangrove",
        "msm": "Mangrove, China coast (MSM)",
        "futian": "Mangrove, Futian (Shenzhen)",
        "context": "Embedding context",
    }.get(category, category)


# Reader-facing lane names, shared by tables, tooltips and charts.
LANE_DISPLAY = {
    "poc_core": "Reference core (POC)",
    "msm_china_2025": "Mangrove, China coast (MSM)",
    "futian_mangrove_2026_qi": "Mangrove, Futian (Shenzhen)",
    "mucc_v1_owc_wetland": "Wetland, Old Woman Creek (MUCC v1)",
}
LANE_ORDER = ["poc_core", "msm_china_2025", "futian_mangrove_2026_qi", "mucc_v1_owc_wetland"]


def apply_public_lane_display(atlas: pd.DataFrame) -> pd.DataFrame:
    """Name each record's source lane precisely (Futian is not MSM)."""
    atlas = atlas.copy()
    lane = atlas.get("lane_id", pd.Series("", index=atlas.index)).fillna("").astype(str)
    category = atlas["source_category"].fillna("").astype(str)
    display = lane.map(LANE_DISPLAY)
    key = lane.map({"msm_china_2025": "msm", "futian_mangrove_2026_qi": "futian", "mucc_v1_owc_wetland": "wetland"})
    poc = lane.eq("poc_core")
    display.loc[poc & category.eq("rumen")] = "Rumen reference (POC core)"
    display.loc[poc & category.eq("wetland")] = "Wetland reference (POC core)"
    key.loc[poc] = category.loc[poc]
    fallback = atlas["source_display"] if "source_display" in atlas.columns else category
    atlas["source_display"] = display.fillna(fallback)
    atlas["lane_key"] = key.fillna(category)
    return atlas


def frame_records(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> list[dict[str, Any]]:
    deduped_cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    use = df.loc[:, ~df.columns.duplicated()][deduped_cols].copy()
    if max_rows is not None:
        use = use.head(max_rows)
    return json.loads(use.replace({np.nan: None}).to_json(orient="records"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


PUBLIC_REPORT_EXCLUDED_FIELDS = {
    "freeze_manifest",
    "interactive_data_asset",
    "interactive_runtime_asset",
    "lane_registry",
    "mapped_ncbi_biosamples",
    "primary_accession",
    "primary_accession_type",
    "source_dataset_doi",
    "source_paper_doi",
    "source_bucket",
    "source_group",
    "source_sample_ids",
    "sample_context_key",
}


def public_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Publish only the compact view-model needed for the interactive report."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): strip(item)
                for key, item in value.items()
                if str(key) not in PUBLIC_REPORT_EXCLUDED_FIELDS
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    clean = strip(json_safe(payload))

    def records(name: str, fields: list[str], parent: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        source = (parent or clean).get(name, [])
        if not isinstance(source, list):
            return []
        return [
            {field: row[field] for field in fields if field in row}
            for row in source
            if isinstance(row, dict)
        ]

    summary_fields = [
        "atlas_registered_units",
        "embedding_context_total",
        "release_multiview_complete",
        "glm2_single_window_units",
        "glm2_multiwindow_units",
    ]
    niche_fields = [
        "proteome_id",
        "mag_id",
        "source_category",
        "lane_key",
        "source_display",
        "analysis_unit_type",
        "claim_scope",
        "plot_annotation_status",
        "review_tier",
        "functional_comparability_tier",
        "functional_numerator_provenance",
        "public_attestation_score_status",
        "glm2_protocol_class",
        "formal_tri_view_status",
        "has_functional",
        "has_glm2",
        "nearest_poc_id",
        "nearest_poc_similarity",
        "qc_tier",
        "checkm2_completeness",
        "checkm2_contamination",
        "domain",
        "phylum",
        "class",
        "processed_gene_expression_support",
        "methane_expressed_gene_rows",
        "sulfur_expressed_gene_rows",
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
        "case_study_set",
        "case_study_rank",
        "is_case_study",
    ]
    card_fields = [
        "card_id",
        "candidate_set",
        "rank",
        "proteome_id",
        "mag_id",
        "source_category",
        "lane_key",
        "source_display",
        "domain",
        "phylum",
        "class",
        "qc_tier",
        "review_tier",
        "functional_evidence_class",
        "functional_harmonization_status",
        "mechanism_equivalence_status",
        "functional_comparability_tier",
        "functional_numerator_provenance",
        "public_attestation_score_status",
        "glm2_protocol_class",
        "glm2_metric_comparability_status",
        "formal_tri_view_status",
        "checkm2_completeness",
        "checkm2_contamination",
        "nearest_poc_similarity",
        "nearest_poc_id",
        "bridge_affinity_index",
        "rate_metric_status",
        "qc_confidence_index",
        "molecular_attestation_index",
        "source_scaffold_review_score",
        "provenance_resolution_tier",
        "metadata_caveat",
        "sample_rollup_status",
        "next_metadata_action",
        "allowed_claim_wording",
        "blocking_gap",
        "next_validation_action",
        "processed_gene_expression_support",
        "methane_expressed_gene_rows",
        "sulfur_expressed_gene_rows",
    ]
    audit = clean.get("scientific_audit", {})
    mucc = audit.get("mucc_validation_readiness", {}) if isinstance(audit, dict) else {}
    niche = clean.get("niche", {}) if isinstance(clean.get("niche", {}), dict) else {}
    sample_linkage = clean.get("sample_linkage", {}) if isinstance(clean.get("sample_linkage", {}), dict) else {}

    return {
        "summary": {field: clean.get("summary", {}).get(field) for field in summary_fields},
        "scientific_audit": {
            "mucc_validation_readiness": {
                field: mucc.get(field)
                for field in ["exact_sample_environment_flux_links", "expression_sample_columns"]
            }
        },
        "evidence_contract": records(
            "evidence_contract",
            ["lane_id", "lane", "registered_units", "data_complete_tri_view_units", "mechanism_comparable_tri_view_units", "functional_contract"],
        ),
        "niche": {
            "methods": records("methods", ["method", "status", "role"], niche),
            "nodes": records("nodes", niche_fields, niche),
            "case_study_count": niche.get("case_study_count", 0),
            "links": records(
                "links",
                ["source", "target", "source_category", "target_category", "similarity", "cross_domain", "reciprocal", "rank", "evidence_type"],
                niche,
            ),
        },
        "matrix": clean.get("matrix", {}),
        "circos": clean.get("circos", {}),
        "sample_linkage": {
            "contexts": records(
                "contexts",
                [
                    "sample_context_label",
                    "chart_label",
                    "lane_key",
                    "sample_linkage_bucket",
                    "units",
                    "tri_view_units",
                    "sample_context_resolution",
                    "linked_sample_context_count",
                    "environmental_context_fields_present",
                    "sample_context_blocking_gap",
                ],
                sample_linkage,
            )
        },
        "cards": records("cards", card_fields),
    }


def public_release_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the path-free report summary used by assets and parity gates."""
    clean: dict[str, Any] = {}
    for key, value in json_safe(summary).items():
        if key in PUBLIC_REPORT_EXCLUDED_FIELDS:
            continue
        if isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)):
            continue
        clean[key] = value
    return clean


def compute_diffusion_map(embeddings: np.ndarray, k: int, random_state: int = 20260619) -> np.ndarray:
    """Compute a two-dimensional diffusion map from a sparse cosine kNN graph.

    A seeded start vector and a sign convention make the coordinates
    reproducible; eigenvector signs are otherwise arbitrary between runs.
    """
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
    v0 = np.random.default_rng(random_state).random(n)
    eigvals, eigvecs = eigsh(sym, k=eig_count, which="LA", v0=v0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    for column in range(eigvecs.shape[1]):
        if eigvecs[np.argmax(np.abs(eigvecs[:, column])), column] < 0:
            eigvecs[:, column] = -eigvecs[:, column]
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
            "role": "Spectral view of the cosine kNN affinity graph; coordinates are exploratory and do not establish ecological transfer.",
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
            methods.append({"method": "umap", "status": "computed", "role": "Default navigation view; preserves local neighborhoods."})
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
            "nearest_poc_id",
            "nearest_mangrove_similarity",
            "nearest_mangrove_id",
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
    emb_meta["nearest_poc_id"] = ""
    emb_meta["nearest_mangrove_similarity"] = 0.0
    emb_meta["nearest_mangrove_id"] = ""
    if len(poc_idx):
        nn_poc = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[poc_idx])
        d_poc, i_poc = nn_poc.kneighbors(embeddings)
        emb_meta["nearest_poc_similarity"] = 1.0 - d_poc[:, 0]
        emb_meta["nearest_poc_id"] = emb_meta.iloc[poc_idx[i_poc[:, 0]]]["proteome_id"].to_numpy()
    if len(msm_idx):
        nn_msm = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[msm_idx])
        d_msm, i_msm = nn_msm.kneighbors(embeddings)
        emb_meta["nearest_mangrove_similarity"] = 1.0 - d_msm[:, 0]
        emb_meta["nearest_mangrove_id"] = emb_meta.iloc[msm_idx[i_msm[:, 0]]]["proteome_id"].to_numpy()
    return emb_meta, edge_df, embeddings


def apply_scientific_evidence_contract(atlas: pd.DataFrame) -> pd.DataFrame:
    """Apply the report's conservative, lane-aware evidence contract.

    ``tri_view_ready`` means that ESM-2, gLM2, and a functional payload all
    exist.  It deliberately does *not* mean the functional numerators or gLM2
    summary metrics are comparable across pipelines.
    """
    atlas = atlas.copy()
    lane = atlas.get("lane_id", pd.Series("", index=atlas.index)).fillna("").astype(str)
    has_functional = atlas.get(
        "has_functional", pd.Series(False, index=atlas.index)
    ).map(legacy.truthy)
    has_esm2 = atlas.get("has_esm2", pd.Series(False, index=atlas.index)).map(legacy.truthy)
    has_glm2 = atlas.get("has_glm2", pd.Series(False, index=atlas.index)).map(legacy.truthy)
    poc = lane.eq("poc_core")
    expansion = lane.isin({"msm_china_2025", "futian_mangrove_2026_qi"})
    mucc = lane.eq("mucc_v1_owc_wetland")

    atlas["functional_evidence_class"] = np.select(
        [poc, expansion, mucc],
        [
            "canonical_curated_mechanism_features",
            "annotation_complete_feature_aggregation_pending",
            "source_annotation_scaffold",
        ],
        default="unclassified_functional_evidence",
    )
    atlas["functional_harmonization_status"] = np.select(
        [
            ~has_functional,
            poc,
            expansion,
            mucc,
        ],
        [
            "functional_output_incomplete",
            "canonical_feature_contract",
            "raw_annotation_outputs_complete_common_mechanism_aggregation_pending",
            "common_screening_axes_harmonized_with_source_scaffold_caveat",
        ],
        default="functional_contract_unclassified",
    )
    atlas["mechanism_equivalence_status"] = np.select(
        [
            ~has_functional,
            poc,
            expansion,
            mucc,
        ],
        [
            "not_applicable_functional_incomplete",
            "mechanism_equivalent",
            "not_yet_mechanism_equivalent",
            "not_canonical_mechanism_equivalent",
        ],
        default="not_demonstrated",
    )
    atlas["functional_comparability_tier"] = np.select(
        [
            ~has_functional,
            poc,
            expansion,
            mucc,
        ],
        [
            "functional_incomplete",
            "canonical_mechanism_comparable",
            "annotation_complete_harmonization_pending",
            "source_scaffold_non_equivalent",
        ],
        default="unclassified",
    )
    atlas["functional_numerator_provenance"] = np.select(
        [poc, expansion, mucc],
        [
            "curated_accepted_or_present_mechanism_features",
            "raw_many_to_many_annotation_hit_rows_plus_all_hmm_rows",
            "source_dram_term_rows_and_processed_expression_detection",
        ],
        default="unclassified",
    )

    native = pd.to_numeric(
        atlas.get("native_window_count", pd.Series(np.nan, index=atlas.index)),
        errors="coerce",
    )
    shuffled = pd.to_numeric(
        atlas.get("shuffled_control_count", pd.Series(np.nan, index=atlas.index)),
        errors="coerce",
    )
    atlas["glm2_protocol_class"] = np.select(
        [
            ~has_glm2,
            native.ge(10) & shuffled.ge(10),
            native.ge(1) & shuffled.ge(1),
        ],
        [
            "glm2_not_available",
            "multiwindow_10_native_plus_10_shuffled",
            "paired_single_native_plus_single_shuffled",
        ],
        default="glm2_protocol_metadata_incomplete",
    )
    atlas["glm2_metric_comparability_status"] = np.where(
        has_glm2,
        "comparable_within_protocol_class_only",
        "not_available",
    )
    atlas["esm2_protocol_class"] = np.where(
        has_esm2,
        "esm2_650m_layer33_proteome_mean_pool_max_6000_proteins",
        "esm2_not_available",
    )
    cap_applied = atlas.get(
        "protein_cap_applied", pd.Series(np.nan, index=atlas.index)
    )
    cap_true = cap_applied.fillna("").astype(str).str.lower().isin({"true", "1"})
    atlas["esm2_protein_cap_status"] = np.select(
        [~has_esm2, cap_true, poc],
        [
            "not_embedded",
            "cap_6000_applied",
            "cap_not_applied_observed_poc_max_below_6000",
        ],
        default="cap_not_applied_or_not_flagged",
    )

    atlas["tri_view_ready"] = has_esm2 & has_glm2 & has_functional
    atlas["formal_tri_view_status"] = np.select(
        [
            ~atlas["tri_view_ready"],
            poc,
            expansion,
            mucc,
        ],
        [
            "incomplete_tri_view",
            "complete_canonical_mechanism_tri_view",
            "complete_annotation_tri_view_harmonization_pending",
            "complete_source_scaffold_tri_view",
        ],
        default="complete_tri_view_contract_unclassified",
    )
    atlas["mechanism_equivalent_tri_view"] = (
        atlas["tri_view_ready"] & poc
    )
    atlas["public_attestation_score_status"] = np.select(
        [
            ~has_functional,
            poc,
            expansion,
            mucc,
        ],
        [
            "not_available_functional_incomplete",
            "poc_internal_screen_only",
            "quarantined_pending_common_feature_rebuild",
            "not_available_source_scaffold_non_equivalent",
        ],
        default="not_available_contract_unclassified",
    )
    # A freeze-backed release is authoritative over historical lane defaults.
    # This keeps a completed payload, pipeline normalization, and demonstrated
    # cross-lane mechanism comparability as separate evidence states.
    freeze_contract = atlas.get(
        "freeze_functional_evidence_class", pd.Series("", index=atlas.index)
    ).fillna("").astype(str)
    frozen = freeze_contract.str.strip().ne("")
    for live_col, freeze_col in {
        "functional_evidence_class": "freeze_functional_evidence_class",
        "functional_harmonization_status": "freeze_functional_harmonization_status",
        "mechanism_equivalence_status": "freeze_mechanism_equivalence_status",
        "formal_tri_view_status": "freeze_formal_tri_view_status",
    }.items():
        if freeze_col in atlas.columns:
            atlas[live_col] = np.where(
                frozen.to_numpy(),
                atlas[freeze_col].to_numpy(),
                atlas[live_col].to_numpy(),
            )
    if "freeze_mechanism_equivalent_tri_view" in atlas.columns:
        frozen_mechanism = truthy_series(
            atlas["freeze_mechanism_equivalent_tri_view"]
        )
        atlas["mechanism_equivalent_tri_view"] = np.where(
            frozen.to_numpy(),
            frozen_mechanism.to_numpy(),
            atlas["mechanism_equivalent_tri_view"].to_numpy(),
        )
    atlas.loc[frozen, "functional_comparability_tier"] = np.select(
        [
            atlas.loc[frozen, "formal_tri_view_status"].eq("incomplete_tri_view"),
            atlas.loc[frozen, "formal_tri_view_status"].eq(
                "complete_canonical_mechanism_tri_view"
            ),
            atlas.loc[frozen, "formal_tri_view_status"].eq(
                "complete_pipeline_normalized_tri_view_comparability_pending"
            ),
            atlas.loc[frozen, "formal_tri_view_status"].eq(
                "complete_annotation_tri_view_harmonization_pending"
            ),
            atlas.loc[frozen, "formal_tri_view_status"].eq(
                "complete_source_scaffold_tri_view"
            ),
        ],
        [
            "functional_incomplete",
            "canonical_mechanism_comparable",
            "pipeline_normalized_comparability_pending",
            "annotation_complete_harmonization_pending",
            "source_scaffold_non_equivalent",
        ],
        default="unclassified",
    )
    normalized_pipeline = frozen & atlas["functional_evidence_class"].isin(
        {"normalized_screening_warehouse", "canonical_pipeline_normalized_features"}
    )
    source_scaffold = frozen & atlas["functional_evidence_class"].eq(
        "source_annotation_scaffold"
    )
    annotation_pending = frozen & atlas["functional_evidence_class"].eq(
        "annotation_complete_feature_aggregation_pending"
    )
    atlas.loc[normalized_pipeline, "functional_numerator_provenance"] = (
        "accepted KOfam genes and present METABOLIC events; best-ranked "
        "MCycDB/SCycDB hits exposed separately"
    )
    atlas.loc[source_scaffold, "functional_numerator_provenance"] = (
        "source DRAM term rows and processed expression detection"
    )
    atlas.loc[annotation_pending, "functional_numerator_provenance"] = (
        "curated annotation bundle present; normalized cohort aggregation pending"
    )
    atlas.loc[normalized_pipeline, "public_attestation_score_status"] = (
        "pipeline_normalized_screening_only_comparability_pending"
    )
    atlas.loc[source_scaffold, "public_attestation_score_status"] = (
        "not_available_source_scaffold_non_equivalent"
    )
    atlas.loc[annotation_pending, "public_attestation_score_status"] = (
        "quarantined_pending_normalized_cohort_aggregation"
    )
    return atlas


def add_molecular_metrics(atlas: pd.DataFrame) -> pd.DataFrame:
    """Compute public metrics while quarantining non-comparable numerators."""
    atlas = atlas.copy()

    def numbers(column: str, default: float = 0.0) -> pd.Series:
        values = (
            atlas[column]
            if column in atlas.columns
            else pd.Series(default, index=atlas.index)
        )
        return pd.to_numeric(values, errors="coerce")

    evidence_groups = atlas["functional_evidence_class"].fillna("unclassified")
    mechanism_equivalent = atlas["mechanism_equivalence_status"].eq(
        "mechanism_equivalent"
    )
    has_functional = atlas["has_functional"].map(legacy.truthy)

    protein_count = numbers("prodigal_proteins", np.nan)
    protein_count = protein_count.fillna(numbers("n_proteins_used", np.nan))
    has_real_denominator = protein_count.notna() & protein_count.gt(0)
    comparable_rate = has_real_denominator & mechanism_equivalent & has_functional
    atlas["protein_count_for_rates"] = protein_count.where(has_real_denominator)
    atlas["rate_metric_status"] = np.select(
        [
            ~has_functional,
            ~has_real_denominator,
            mechanism_equivalent,
            atlas["functional_comparability_tier"].eq(
                "pipeline_normalized_comparability_pending"
            ),
            atlas["functional_comparability_tier"].eq(
                "annotation_complete_harmonization_pending"
            ),
            atlas["functional_comparability_tier"].eq(
                "source_scaffold_non_equivalent"
            ),
        ],
        [
            "not_available_functional_incomplete",
            "not_available_missing_protein_denominator",
            "comparable_curated_feature_density",
            "pipeline_normalized_screening_not_cross_lane_mechanism_rate",
            "quarantined_raw_hit_row_numerator_not_marker_density",
            "source_scaffold_term_density_non_equivalent",
        ],
        default="not_comparable",
    )

    methane_from_scores = numbers("methane_evidence_score", np.nan)
    methane_from_raw = (
        numbers("mcycdb_hits").fillna(0)
        + numbers("metabolic_hmm_rows").fillna(0)
    )
    sulfur_from_scores = numbers("sulfur_competition_score", np.nan)
    sulfur_from_raw = (
        numbers("scycdb_hits").fillna(0)
        + numbers("metabolic_functions_present").fillna(0)
    )
    raw_methane_count = methane_from_scores.where(
        methane_from_scores.notna(), methane_from_raw
    ).fillna(0)
    raw_sulfur_count = sulfur_from_scores.where(
        sulfur_from_scores.notna(), sulfur_from_raw
    ).fillna(0)
    canonical_substrate_count = (
        numbers("cazy_family_count").fillna(0)
        + numbers("merops_family_count").fillna(0)
    )
    raw_substrate_count = pd.to_numeric(
        atlas.get(
            "substrate_evidence_count", pd.Series(np.nan, index=atlas.index)
        ),
        errors="coerce",
    ).fillna(canonical_substrate_count)
    atlas["raw_methane_annotation_row_count"] = raw_methane_count
    atlas["raw_sulfur_annotation_row_count"] = raw_sulfur_count
    atlas["raw_substrate_annotation_row_count"] = raw_substrate_count

    atlas["methane_marker_count"] = raw_methane_count.where(mechanism_equivalent)
    atlas["sulfur_context_count"] = raw_sulfur_count.where(mechanism_equivalent)
    atlas["substrate_breadth_count"] = raw_substrate_count.where(
        mechanism_equivalent
    )
    atlas["methane_marker_density_per_1k"] = (
        1000 * atlas["methane_marker_count"] / protein_count
    ).where(comparable_rate)
    atlas["sulfur_context_density_per_1k"] = (
        1000 * atlas["sulfur_context_count"] / protein_count
    ).where(comparable_rate)
    atlas["substrate_breadth_per_1k"] = (
        1000 * atlas["substrate_breadth_count"] / protein_count
    ).where(comparable_rate)
    atlas["methane_sulfur_balance"] = (
        np.log1p(atlas["methane_marker_density_per_1k"])
        - np.log1p(atlas["sulfur_context_density_per_1k"])
    )

    canonical_annotation_breadth = (
        0.34
        * norm01(
            numbers("kofam_annotated_gene_fraction").fillna(0)
        )
        + 0.22
        * norm01(
            np.log1p(
                numbers("metabolic_modules_present").fillna(0)
            )
        )
        + 0.22
        * norm01(
            np.log1p(
                numbers("cazy_family_count").fillna(0)
            )
        )
        + 0.22
        * norm01(
            np.log1p(
                numbers("merops_family_count").fillna(0)
            )
        )
    )
    pipeline_annotation_breadth = legacy.norm01_by_group(
        np.log1p(
            pd.to_numeric(
                atlas.get(
                    "broad_function_evidence_count",
                    pd.Series(np.nan, index=atlas.index),
                ),
                errors="coerce",
            ).fillna(0)
        ),
        evidence_groups,
    )
    atlas["annotation_coverage_index_within_pipeline"] = (
        pipeline_annotation_breadth
    )
    atlas.loc[mechanism_equivalent, "annotation_coverage_index_within_pipeline"] = (
        canonical_annotation_breadth.loc[mechanism_equivalent]
    )
    atlas["annotation_breadth_index"] = canonical_annotation_breadth.where(
        mechanism_equivalent
    )

    qc_raw = (
        numbers("checkm2_completeness").fillna(0)
        - 5
        * numbers("checkm2_contamination").fillna(0)
    ).clip(lower=0)
    atlas["qc_confidence_index"] = (qc_raw / 100).clip(0, 1)
    atlas["bridge_affinity_index"] = norm01(
        numbers("nearest_poc_similarity").fillna(0)
        + 0.6
        * numbers("cross_domain_neighbor_fraction").fillna(0)
        + 0.25
        * numbers("mixing_coeff").fillna(0)
    )

    legacy_methane_index = legacy.norm01_by_group(
        np.log1p(raw_methane_count), evidence_groups
    )
    legacy_sulfur_index = legacy.norm01_by_group(
        np.log1p(raw_sulfur_count), evidence_groups
    )
    legacy_substrate_index = legacy.norm01_by_group(
        np.log1p(raw_substrate_count), evidence_groups
    )
    atlas["pipeline_specific_methane_signal_index"] = legacy_methane_index
    atlas["pipeline_specific_sulfur_signal_index"] = legacy_sulfur_index
    atlas["pipeline_specific_substrate_signal_index"] = legacy_substrate_index
    glm_raw = numbers("glm_context_delta").fillna(0)
    atlas["glm_context_index_within_protocol"] = legacy.norm01_by_group(
        glm_raw, atlas["glm2_protocol_class"]
    )
    atlas["glm_context_index"] = atlas[
        "glm_context_index_within_protocol"
    ].where(mechanism_equivalent)
    atlas["methane_signal_index"] = legacy_methane_index.where(
        mechanism_equivalent
    )
    atlas["sulfur_context_index"] = legacy_sulfur_index.where(
        mechanism_equivalent
    )
    atlas["substrate_breadth_index"] = legacy_substrate_index.where(
        mechanism_equivalent
    )

    legacy_index = (
        0.24 * atlas["bridge_affinity_index"]
        + 0.14 * atlas["glm_context_index_within_protocol"]
        + 0.19 * legacy_methane_index
        + 0.13 * legacy_sulfur_index
        + 0.12 * legacy_substrate_index
        + 0.10 * atlas["annotation_coverage_index_within_pipeline"]
        + 0.08 * atlas["qc_confidence_index"]
    )
    atlas["legacy_noncomparable_attestation_index_quarantined"] = (
        legacy_index.where(has_functional & ~mechanism_equivalent)
    )
    atlas["molecular_attestation_index"] = legacy_index.where(
        has_functional & mechanism_equivalent
    )

    # Reproduce the former public score exactly enough to quantify its source
    # bias.  The old contract grouped POC, MSM, and Futian together as one
    # "canonical" class while excluding the MUCC source scaffold.
    former_score_eligible = (
        has_functional
        & atlas["lane_id"].astype(str).isin(
            {"poc_core", "msm_china_2025", "futian_mangrove_2026_qi"}
        )
    )
    former_group = pd.Series(
        np.where(
            atlas["lane_id"].astype(str).eq("mucc_v1_owc_wetland"),
            "former_source_scaffold",
            "former_combined_poc_mangrove_bucket",
        ),
        index=atlas.index,
    )
    former_methane_index = legacy.norm01_by_group(
        np.log1p(raw_methane_count), former_group
    )
    former_sulfur_index = legacy.norm01_by_group(
        np.log1p(raw_sulfur_count), former_group
    )
    former_substrate_index = legacy.norm01_by_group(
        np.log1p(raw_substrate_count), former_group
    )
    former_glm_index = legacy.norm01_by_group(glm_raw, former_group)
    former_attestation_index = (
        0.24 * atlas["bridge_affinity_index"]
        + 0.14 * former_glm_index
        + 0.19 * former_methane_index
        + 0.13 * former_sulfur_index
        + 0.12 * former_substrate_index
        + 0.10 * canonical_annotation_breadth
        + 0.08 * atlas["qc_confidence_index"]
    ).where(former_score_eligible)
    atlas["legacy_published_methane_signal_index"] = former_methane_index.where(
        former_score_eligible
    )
    atlas["legacy_published_attestation_index_quarantined"] = (
        former_attestation_index
    )
    atlas["legacy_noncomparable_attestation_index_quarantined"] = (
        former_attestation_index.where(~mechanism_equivalent)
    )
    return atlas


def classify_review_tier(row: pd.Series) -> str:
    if not bool(row.get("has_functional", False)):
        return "functional pending"
    if (
        row.get("functional_comparability_tier")
        == "annotation_complete_harmonization_pending"
    ):
        return "annotation complete; harmonization pending"
    if row.get("functional_comparability_tier") == "source_scaffold_non_equivalent":
        return "source-scaffold review"
    if row.get("functional_comparability_tier") == "pipeline_normalized_comparability_pending":
        return "shared-pipeline screening; cross-route comparison pending"
    if row.get("mechanism_equivalence_status") != "mechanism_equivalent":
        return "evidence contract unresolved"
    score = safe_float(row.get("molecular_attestation_index"))
    qc = safe_float(row.get("qc_confidence_index"))
    if score >= 0.72 and qc >= 0.55:
        return "POC internal high-priority review"
    if score >= 0.52:
        return "POC internal mechanism review"
    if score >= 0.34:
        return "POC internal screening signal"
    return "POC internal low-current signal"


def read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def truthy_series(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna("").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def apply_freeze_manifest(
    atlas: pd.DataFrame,
    status: pd.DataFrame,
    freeze_manifest_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if freeze_manifest_path is None:
        return atlas, status, {}
    freeze = read_optional_tsv(freeze_manifest_path)
    if freeze.empty or "proteome_id" not in freeze.columns:
        return atlas, status, {"freeze_manifest": str(freeze_manifest_path), "freeze_manifest_rows": 0}

    freeze = freeze.copy()
    freeze["proteome_id"] = freeze["proteome_id"].astype(str)
    for col in [
        "has_esm2",
        "has_glm2",
        "has_functional",
        "tri_view_ready",
        "schema_normalized",
        "mechanism_equivalent_tri_view",
        "release_required",
        "release_excluded",
    ]:
        if col in freeze.columns:
            freeze[col] = truthy_series(freeze[col])
    keep_cols = [
        c
        for c in [
            "lane_id",
            "proteome_id",
            "has_esm2",
            "has_glm2",
            "has_functional",
            "tri_view_ready",
            "schema_normalized",
            "functional_evidence_class",
            "functional_harmonization_status",
            "mechanism_equivalence_status",
            "formal_tri_view_status",
            "mechanism_equivalent_tri_view",
            "release_required",
            "release_excluded",
            "release_exclusion_reason",
            "release_exclusion_scope",
            "release_exclusion_approved_by",
            "release_exclusion_approved_at_utc",
            "functional_status",
            "functional_status_basis",
            "selected_run_dir",
            "claim_scope",
        ]
        if c in freeze.columns
    ]
    freeze = freeze[keep_cols].drop_duplicates([c for c in ["lane_id", "proteome_id"] if c in keep_cols])
    rename = {
        "has_esm2": "freeze_has_esm2",
        "has_glm2": "freeze_has_glm2",
        "has_functional": "freeze_has_functional",
        "tri_view_ready": "freeze_tri_view_ready",
        "schema_normalized": "freeze_schema_normalized",
        "functional_evidence_class": "freeze_functional_evidence_class",
        "functional_harmonization_status": "freeze_functional_harmonization_status",
        "mechanism_equivalence_status": "freeze_mechanism_equivalence_status",
        "formal_tri_view_status": "freeze_formal_tri_view_status",
        "mechanism_equivalent_tri_view": "freeze_mechanism_equivalent_tri_view",
        "release_required": "freeze_release_required",
        "release_excluded": "freeze_release_excluded",
        "release_exclusion_reason": "freeze_release_exclusion_reason",
        "release_exclusion_scope": "freeze_release_exclusion_scope",
        "release_exclusion_approved_by": "freeze_release_exclusion_approved_by",
        "release_exclusion_approved_at_utc": "freeze_release_exclusion_approved_at_utc",
        "functional_status": "freeze_functional_status",
        "functional_status_basis": "freeze_functional_status_basis",
        "selected_run_dir": "freeze_selected_run_dir",
        "claim_scope": "freeze_claim_scope",
    }
    freeze = freeze.rename(columns=rename)
    join_cols = ["lane_id", "proteome_id"] if "lane_id" in atlas.columns and "lane_id" in freeze.columns else ["proteome_id"]
    atlas = atlas.merge(freeze, on=join_cols, how="left")
    enforced_status_cols = {
        "has_esm2": "freeze_has_esm2",
        "has_glm2": "freeze_has_glm2",
        "has_functional": "freeze_has_functional",
    }
    for live_col, freeze_col in enforced_status_cols.items():
        if live_col in atlas.columns and freeze_col in atlas.columns:
            freeze_mask = atlas[freeze_col].notna()
            atlas.loc[freeze_mask, live_col] = truthy_series(atlas.loc[freeze_mask, freeze_col]).to_numpy()
    for live_col, freeze_col in {
        "functional_evidence_class": "freeze_functional_evidence_class",
        "functional_harmonization_status": "freeze_functional_harmonization_status",
        "mechanism_equivalence_status": "freeze_mechanism_equivalence_status",
        "formal_tri_view_status": "freeze_formal_tri_view_status",
    }.items():
        if freeze_col in atlas.columns:
            freeze_mask = atlas[freeze_col].fillna("").astype(str).str.strip().ne("")
            atlas[live_col] = np.where(
                freeze_mask.to_numpy(),
                atlas[freeze_col].to_numpy(),
                atlas.get(live_col, pd.Series("", index=atlas.index)).to_numpy(),
            )
    if "functional_status" in atlas.columns and "freeze_functional_status" in atlas.columns:
        freeze_status_mask = atlas["freeze_functional_status"].notna()
        atlas.loc[freeze_status_mask, "functional_status"] = atlas.loc[freeze_status_mask, "freeze_functional_status"]
    if not status.empty:
        status_join_cols = ["lane_id", "proteome_id"] if "lane_id" in status.columns and "lane_id" in freeze.columns else ["proteome_id"]
        status = status.merge(freeze, on=status_join_cols, how="left")
        for live_col, freeze_col in enforced_status_cols.items():
            if live_col in status.columns and freeze_col in status.columns:
                freeze_mask = status[freeze_col].notna()
                status.loc[freeze_mask, live_col] = truthy_series(status.loc[freeze_mask, freeze_col]).to_numpy()
        if "functional_status" in status.columns and "freeze_functional_status" in status.columns:
            freeze_status_mask = status["freeze_functional_status"].notna()
            status.loc[freeze_status_mask, "functional_status"] = status.loc[freeze_status_mask, "freeze_functional_status"]
    metadata = {
        "freeze_manifest": str(freeze_manifest_path),
        "freeze_manifest_rows": int(len(freeze)),
        "freeze_tri_view_ready_rows": int(
            truthy_series(atlas.get("freeze_tri_view_ready", pd.Series(dtype=object))).sum()
        ),
        "freeze_release_excluded_rows": int(
            truthy_series(atlas.get("freeze_release_excluded", pd.Series(dtype=object))).sum()
        ),
        "freeze_release_required_rows": int(
            truthy_series(atlas.get("freeze_release_required", pd.Series(dtype=object))).sum()
        ),
        "freeze_release_required_tri_view_ready_rows": int(
            (
                truthy_series(atlas.get("freeze_release_required", pd.Series(dtype=object)))
                & truthy_series(atlas.get("freeze_tri_view_ready", pd.Series(dtype=object)))
            ).sum()
        ),
        "freeze_status_enforced": True,
    }
    return atlas, status, metadata


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
            "metadata_caveat": "Exact accession for each genome; environmental context is at cohort level (cattle rumen).",
            "sample_rollup_status": "reference_context_not_blue_carbon_sample_rollup",
            "next_metadata_action": "Keep as a methane reference; never treat as blue-carbon sample context.",
        },
        "wetland": {
            "source_paper_doi": "10.1038/s41467-025-56133-0",
            "source_dataset_doi": "10.5281/zenodo.14532347",
            "primary_accession_type": "MUCC/NCBI/Zenodo source record",
            "provenance_resolution_tier": "mixed_mucc_resolution",
            "metadata_caveat": "Paper and dataset provenance are clear; sample resolution is mixed (NCBI BioSample, Old Woman Creek site or project, or source bucket only).",
            "sample_rollup_status": "blocked_mixed_sample_resolution",
            "next_metadata_action": "Map source-bucket genomes to their BioSamples before any sample-level summary.",
        },
        "mangrove": {
            "source_paper_doi": "10.1093/gigascience/giaf081",
            "source_dataset_doi": "10.5524/102702",
            "primary_accession_type": "GigaDB/NCBI BioSample group mapping",
            "provenance_resolution_tier": "source_group_biosample_context",
            "metadata_caveat": "Clear source provenance and sediment-sample context; each genome still needs a sample assignment, and the 1,428 deposited genomes must be reconciled with the paper's 966 final genomes.",
            "sample_rollup_status": "blocked_mag_to_sample_reconciliation",
            "next_metadata_action": "Assign genomes to samples and reconcile the genome count before sample or site features.",
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
    if "lane_id" in atlas.columns:
        mucc_v1_mask = atlas["lane_id"].astype(str).eq(
            "mucc_v1_owc_wetland"
        )
        mucc_v1_defaults = {
            "source_paper_doi": "10.1128/msystems.00680-25",
            "source_dataset_doi": "10.5281/zenodo.8194033",
            "primary_accession_type": (
                "checksum-validated Zenodo MUCC v1 MAG and annotation record"
            ),
            "provenance_resolution_tier": (
                "exact_mag_archive_qc_source_scaffold"
            ),
            "metadata_caveat": (
                "Exact genome, QC and source-annotation provenance, with processed "
                "expression data; exact sample, date and depth links and "
                "methane-process joins are not yet available."
            ),
            "sample_rollup_status": (
                "blocked_exact_sample_depth_environment_flux_join"
            ),
            "next_metadata_action": (
                "Join exact sample, date, depth, abundance, environment and flux "
                "before any ecological or MRV use."
            ),
        }
        for col, value in mucc_v1_defaults.items():
            atlas.loc[mucc_v1_mask, col] = value
        futian_mask = atlas["lane_id"].astype(str).eq("futian_mangrove_2026_qi")
        futian_defaults = {
            "source_paper_doi": "10.1038/s41597-026-07291-3",
            "source_dataset_doi": "10.6084/m9.figshare.30883646.v3",
            "primary_accession_type": "Figshare/source-manifest rMAG payload",
            "provenance_resolution_tier": "site_month_habitat_context",
            "metadata_caveat": (
                "Clear source provenance with Futian site, month, depth and habitat metadata; "
                "each genome is placed at site and month, not yet in a single depth sample."
            ),
            "sample_rollup_status": "blocked_depth_resolved_mag_to_sample_and_abundance",
            "next_metadata_action": (
                "Link genomes to depth-resolved samples, add abundance and pair with "
                "validation measurements before sample or site features."
            ),
        }
        for col, value in futian_defaults.items():
            atlas.loc[futian_mask, col] = value

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
        "documented source gap; not plotted",
    )
    return atlas


def count_delimited_ids(value: Any) -> int:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return 0
    pieces = [p.strip() for token in text.split(";") for p in token.split(",")]
    return len({p for p in pieces if p})


def add_sample_linkage_context(atlas: pd.DataFrame, repo_root: Path, msm_root: Path) -> pd.DataFrame:
    """Add sample/context rollup fields while preserving MAG-level claim boundaries."""
    atlas = atlas.copy()
    for col in [
        "sample_context_key",
        "sample_context_label",
        "sample_context_resolution",
        "environmental_context_status",
        "sample_context_blocking_gap",
        "sample_site_name",
        "sampling_month_iso",
    ]:
        if col not in atlas.columns:
            atlas[col] = ""
    for col in [
        "linked_sample_context_count",
        "environmental_context_fields_present",
        "mean_ph",
        "mean_salinity_psu",
        "mean_toc_mg_g",
    ]:
        if col not in atlas.columns:
            atlas[col] = np.nan

    if "source_group" not in atlas.columns:
        atlas["source_group"] = ""
    if "lane_id" not in atlas.columns:
        atlas["lane_id"] = np.where(atlas["source_category"].astype(str).eq("mangrove"), "mangrove_unknown", "poc_core")

    futian_sample_path = repo_root / "data/external/futian_mangrove_2026_qi/metadata/futian_65_sample_metadata.tsv"
    futian_samples = read_optional_tsv(futian_sample_path)
    if not futian_samples.empty and "site_time_key" in futian_samples.columns:
        env_cols = [
            c
            for c in [
                "ph",
                "salinity_psu",
                "toc_mg_g",
                "ammonium_mg_kg",
                "nitrate_mg_kg",
                "tn_mg_g",
                "tp_mg_g",
                "ts_mg_g",
            ]
            if c in futian_samples.columns
        ]
        value_cols = [c for c in ["ph", "salinity_psu", "toc_mg_g"] if c in futian_samples.columns]
        for col in env_cols:
            futian_samples[col] = pd.to_numeric(futian_samples[col], errors="coerce")
        for col in value_cols:
            futian_samples[col] = pd.to_numeric(futian_samples[col], errors="coerce")
        futian_samples["environmental_context_fields_present"] = futian_samples[env_cols].notna().sum(axis=1) if env_cols else 0
        agg_spec: dict[str, Any] = {
            "linked_sample_context_count_ctx": ("sample_name", "nunique"),
            "sample_site_name_ctx": ("site_name", lambda x: "; ".join(sorted({str(v) for v in x.dropna() if str(v).strip()}))[:120]),
            "sampling_month_iso_ctx": ("sampling_month_iso", lambda x: first_nonempty(pd.Series({"x": next(iter([v for v in x.dropna() if str(v).strip()]), "")}), ["x"])),
            "environmental_context_fields_present_ctx": ("environmental_context_fields_present", "max"),
        }
        if "depth_cm" in futian_samples.columns:
            agg_spec["depth_context_count_ctx"] = ("depth_cm", "nunique")
        for col in value_cols:
            agg_spec[f"mean_{col}_ctx"] = (col, "mean")
        futian_agg = futian_samples.groupby("site_time_key", dropna=False).agg(**agg_spec).reset_index()
        futian_agg = futian_agg.rename(columns={"site_time_key": "source_group"})
        atlas = atlas.merge(futian_agg, on="source_group", how="left")
        futian_mask = atlas["lane_id"].astype(str).eq("futian_mangrove_2026_qi")
        source_group = atlas["source_group"].fillna("").astype(str)
        atlas.loc[futian_mask & atlas["site_label"].fillna("").astype(str).eq(""), "site_label"] = source_group
        atlas.loc[futian_mask & atlas["source_bucket"].fillna("").astype(str).eq(""), "source_bucket"] = source_group
        atlas.loc[futian_mask, "sample_context_key"] = source_group
        atlas.loc[futian_mask, "sample_context_label"] = np.where(
            atlas.loc[futian_mask, "sample_site_name_ctx"].fillna("").astype(str).ne(""),
            atlas.loc[futian_mask, "sample_site_name_ctx"].fillna("").astype(str)
            + " · "
            + source_group.loc[futian_mask],
            source_group.loc[futian_mask],
        )
        atlas.loc[futian_mask, "sample_context_resolution"] = "site_month_multi_depth_context"
        has_futian_context = futian_mask & pd.to_numeric(
            atlas.get("linked_sample_context_count_ctx", pd.Series(index=atlas.index)), errors="coerce"
        ).fillna(0).gt(0)
        atlas.loc[futian_mask, "environmental_context_status"] = "site_month_context_present_depth_assignment_pending"
        atlas.loc[futian_mask & ~has_futian_context, "environmental_context_status"] = "site_month_context_without_sample_metadata_match"
        atlas.loc[futian_mask, "sample_context_blocking_gap"] = (
            "Still needed: depth-resolved MAG-to-sample assignment, abundance, and flux or process validation."
        )
        ctx_map = {
            "linked_sample_context_count": "linked_sample_context_count_ctx",
            "sample_site_name": "sample_site_name_ctx",
            "sampling_month_iso": "sampling_month_iso_ctx",
            "environmental_context_fields_present": "environmental_context_fields_present_ctx",
            "mean_ph": "mean_ph_ctx",
            "mean_salinity_psu": "mean_salinity_psu_ctx",
            "mean_toc_mg_g": "mean_toc_mg_g_ctx",
        }
        for target, ctx in ctx_map.items():
            if ctx in atlas.columns:
                atlas.loc[futian_mask, target] = atlas.loc[futian_mask, ctx]
                atlas = atlas.drop(columns=[ctx])
        if "depth_context_count_ctx" in atlas.columns:
            atlas = atlas.drop(columns=["depth_context_count_ctx"])

    msm_sample_path = msm_root / "gigadb_wasabi/metadata_sediment_samples.txt"
    if not msm_sample_path.exists():
        msm_sample_path = repo_root / "data/external/msm_china_2025/gigadb_wasabi/metadata_sediment_samples.txt"
    msm_samples = read_optional_tsv(msm_sample_path)
    if not msm_samples.empty and "group" in msm_samples.columns:
        msm_samples = msm_samples.copy()
        msm_samples["msm_group_key"] = msm_samples["group"].astype(str).str.strip()
        msm_agg = (
            msm_samples.groupby("msm_group_key", dropna=False)
            .agg(
                linked_sample_context_count_ctx=("sample_id", "nunique"),
                sample_site_name_ctx=("sample_loc", lambda x: "; ".join(sorted({str(v).strip() for v in x.dropna() if str(v).strip()}))[:140]),
                sampling_month_iso_ctx=("collect_date", lambda x: "; ".join(sorted({str(v).strip() for v in x.dropna() if str(v).strip()}))[:80]),
                environmental_context_fields_present_ctx=(
                    "sample_id",
                    lambda x: 4,
                ),
            )
            .reset_index()
        )
        atlas["msm_group_key"] = atlas["source_group"].fillna("").astype(str).str.replace("_MAGs", "", regex=False)
        atlas = atlas.merge(msm_agg, on="msm_group_key", how="left")
        msm_mask = atlas["lane_id"].astype(str).eq("msm_china_2025")
        source_group = atlas["source_group"].fillna("").astype(str)
        atlas.loc[msm_mask, "sample_context_key"] = source_group
        atlas.loc[msm_mask, "sample_context_label"] = np.where(
            atlas.loc[msm_mask, "sample_site_name_ctx"].fillna("").astype(str).ne(""),
            source_group.loc[msm_mask] + " · " + atlas.loc[msm_mask, "sample_site_name_ctx"].fillna("").astype(str),
            source_group.loc[msm_mask],
        )
        atlas.loc[msm_mask, "sample_context_resolution"] = "source_group_multi_sample_biosample_context"
        atlas.loc[msm_mask, "environmental_context_status"] = "source_group_context_present_mag_to_sample_assignment_pending"
        atlas.loc[msm_mask, "sample_context_blocking_gap"] = (
            "Still needed: a sample for each MAG, reconciliation of 1,428 deposited with 966 final MAGs, abundance, and validation."
        )
        for target, ctx in {
            "linked_sample_context_count": "linked_sample_context_count_ctx",
            "sample_site_name": "sample_site_name_ctx",
            "sampling_month_iso": "sampling_month_iso_ctx",
            "environmental_context_fields_present": "environmental_context_fields_present_ctx",
        }.items():
            if ctx in atlas.columns:
                atlas.loc[msm_mask, target] = atlas.loc[msm_mask, ctx]
                atlas = atlas.drop(columns=[ctx])
        atlas = atlas.drop(columns=["msm_group_key"], errors="ignore")

    poc_mask = atlas["source_category"].astype(str).isin(["rumen", "wetland"])
    atlas.loc[poc_mask & atlas["sample_context_key"].fillna("").astype(str).eq(""), "sample_context_key"] = atlas.loc[
        poc_mask, "primary_accession"
    ].fillna("").astype(str)
    atlas.loc[poc_mask & atlas["sample_context_label"].fillna("").astype(str).eq(""), "sample_context_label"] = atlas.loc[
        poc_mask, "site_label"
    ].fillna("").astype(str)
    atlas.loc[atlas["sample_context_key"].fillna("").astype(str).eq(""), "sample_context_key"] = atlas[
        "source_bucket"
    ].fillna("").astype(str)
    atlas.loc[atlas["sample_context_label"].fillna("").astype(str).eq(""), "sample_context_label"] = atlas[
        "sample_context_key"
    ].fillna("").astype(str)
    atlas["linked_sample_context_count"] = pd.to_numeric(atlas["linked_sample_context_count"], errors="coerce").fillna(
        atlas["source_sample_ids"].map(count_delimited_ids) if "source_sample_ids" in atlas.columns else 0
    )
    if "environmental_context_fields_present" in atlas.columns:
        atlas["environmental_context_fields_present"] = pd.to_numeric(
            atlas["environmental_context_fields_present"], errors="coerce"
        ).fillna(0)
    for field in ["mean_ph", "mean_salinity_psu", "mean_toc_mg_g"]:
        atlas[field] = pd.to_numeric(atlas[field], errors="coerce")
    return atlas


def build_candidate_cards(atlas: pd.DataFrame, top_n_poc: int, top_n_mangrove: int) -> pd.DataFrame:
    poc = atlas[atlas["atlas_inclusion_status"].eq("poc_core_complete")].copy()
    # Recompute selection from the current geometry. Archived pilot bridge ranks
    # belong to the withdrawn pooling configuration and must never be reused.
    poc_top = poc.sort_values(
        ["cross_domain_neighbor_fraction", "qc_confidence_index", "proteome_id"],
        ascending=[False, False, True],
    ).head(top_n_poc).copy()
    poc_top["rank"] = np.arange(1, len(poc_top) + 1)
    poc_top["candidate_set"] = "POC geometry-led review candidate"
    msm = atlas[
        atlas["source_category"].eq("mangrove")
        & atlas["has_functional"]
        & atlas["has_glm2"]
    ].copy()
    msm_top = (
        msm.sort_values(
            [
                "bridge_affinity_index",
                "qc_confidence_index",
                "nearest_poc_similarity",
            ],
            ascending=False,
        )
        .head(top_n_mangrove)
        .copy()
    )
    msm_top["candidate_set"] = (
        "Mangrove geometry-led candidate; functional harmonization pending"
    )
    msm_top["rank"] = np.arange(1, len(msm_top) + 1)
    scaffold = atlas[
        atlas.get(
            "functional_evidence_class",
            pd.Series("", index=atlas.index),
        ).eq("source_annotation_scaffold")
        & atlas["has_esm2"]
        & atlas["has_glm2"]
        & atlas["has_functional"]
    ].copy()
    scaffold_top = (
        scaffold.sort_values(
            ["source_scaffold_review_score", "nearest_poc_similarity"],
            ascending=False,
        )
        .head(top_n_poc)
        .copy()
    )
    scaffold_top["candidate_set"] = "MUCC v1 source-scaffold review candidate"
    scaffold_top["rank"] = np.arange(1, len(scaffold_top) + 1)
    cards = pd.concat(
        [poc_top, msm_top, scaffold_top], ignore_index=True, sort=False
    )
    cards["review_tier"] = cards.apply(classify_review_tier, axis=1)
    cards["card_id"] = (
        cards["candidate_set"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        + "_"
        + cards["rank"].fillna(0).astype(int).astype(str)
    )
    defaults = {
        "allowed_claim_wording": (
            "A genome-level hypothesis for review; read each evidence view "
            "only within its documented contract."
        ),
        "blocking_gap": (
            "sample mapping, abundance, environmental covariates, uncertainty, "
            "phylogeny and source controls, and flux or process validation"
        ),
        "next_validation_action": (
            "inspect marker neighborhoods, compare phylogeny versus embedding "
            "proximity, run source-aware nulls, and connect to sample metadata "
            "before MRV scoring"
        ),
    }
    for column, default in defaults.items():
        if column not in cards.columns:
            cards[column] = default
        else:
            existing = cards[column].fillna("").astype(str).str.strip()
            cards.loc[existing.eq(""), column] = default
    poc_mask = cards["candidate_set"].astype(str).eq("POC geometry-led review candidate")
    mangrove_mask = cards["candidate_set"].astype(str).str.startswith(
        "Mangrove geometry-led"
    )
    scaffold_mask = cards["candidate_set"].astype(str).str.startswith(
        "MUCC v1 source-scaffold"
    )
    cards.loc[poc_mask, "allowed_claim_wording"] = (
        "Reference-core screening hypothesis with shared-pipeline features. "
        "Cross-route comparison, ecological transfer, methane flux and sample "
        "risk each need their own evidence."
    )
    cards.loc[mangrove_mask, "allowed_claim_wording"] = (
        "Mangrove review candidate chosen by embedding geometry and QC, with "
        "shared-pipeline screening events. Mechanism strength and cross-route "
        "ranking are withheld until comparability checks pass."
    )
    cards.loc[scaffold_mask, "allowed_claim_wording"] = (
        "Old Woman Creek candidate with source annotations and, where present, "
        "processed expression detection. Mechanism scoring and flux linkage "
        "need a shared functional contract and exact sample links."
    )
    return cards


def build_evidence_flow(atlas: pd.DataFrame) -> dict[str, Any]:
    df = atlas.copy()
    df["review_tier"] = df.apply(classify_review_tier, axis=1)
    df["evidence_state"] = np.select(
        [
            df["atlas_inclusion_status"].eq("poc_core_complete"),
            df.get(
                "formal_tri_view_status",
                pd.Series("", index=df.index),
            ).eq("complete_source_scaffold_tri_view"),
            df["source_category"].eq("mangrove") & df["has_functional"],
            df["source_category"].eq("mangrove") & ~df["has_functional"],
        ],
        [
            "POC canonical tri-view complete",
            "Wetland source-scaffold tri-view complete",
            "Mangrove canonical tri-view complete",
            "Mangrove function pending",
        ],
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


def build_external_source_readiness(atlas: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "lane_id" not in atlas.columns:
        return rows
    external = atlas[~atlas["lane_id"].astype(str).eq("poc_core")].copy()
    external_ids = external["lane_id"].astype(str)
    present = set(external_ids)
    for lane_id in [lane for lane in LANE_ORDER if lane in present] + sorted(present - set(LANE_ORDER)):
        frame = external[external_ids.eq(lane_id)]
        report_units = int(len(frame))
        tri_view = int((frame["has_esm2"] & frame["has_glm2"] & frame["has_functional"]).sum())
        if lane_id == "msm_china_2025":
            rows.append(
                {
                    "lane": LANE_DISPLAY[lane_id],
                    "report_units": report_units,
                    "metadata_universe": "1,428 deposited MAGs; the source paper reports 966 final MAGs",
                    "primary_source": "Pan et al. 2025, GigaScience",
                    "resolution_now": f"{tri_view:,}/{report_units:,} with all three views; 82 sediment-sample rows; 71 exact BioSample rows",
                    "use_now": "Mangrove screening and sample-readiness priorities",
                    "blocking_gap": "Assign each MAG to its sample and reconcile 1,428 deposited with 966 final MAGs before sample or site summaries",
                }
            )
        elif lane_id == "futian_mangrove_2026_qi":
            rows.append(
                {
                    "lane": LANE_DISPLAY[lane_id],
                    "report_units": report_units,
                    "metadata_universe": "3,404 registered MAGs: 3,156 complete, 248 documented gaps",
                    "primary_source": "Qi et al. 2026, Scientific Data",
                    "resolution_now": f"{tri_view:,}/{report_units:,} with all three views; 65 exact sediment-sample metadata rows",
                    "use_now": "Mangrove and mudflat screening; design of time, depth and habitat sampling",
                    "blocking_gap": "Depth-resolved MAG-to-sample assignment, abundance, and flux or process validation",
                }
            )
        elif lane_id == "mucc_v1_owc_wetland":
            rows.append(
                {
                    "lane": LANE_DISPLAY[lane_id],
                    "report_units": report_units,
                    "metadata_universe": (
                        "2,508 checksum-validated archive MAGs; 2,502 pass the "
                        "paper's quality screen; 7 lack a source protein file"
                    ),
                    "primary_source": "Borton et al. 2026, mSystems",
                    "resolution_now": (
                        f"{tri_view:,}/{report_units:,} with all three views (source "
                        "annotations); processed expression for 1,948 MAGs across "
                        "133 sample columns; chamber-flux, porewater and tower-flux "
                        "records staged as site and time context"
                    ),
                    "use_now": (
                        "Wetland reference screening and candidate review under "
                        "the source-annotation contract"
                    ),
                    "blocking_gap": (
                        "A shared mechanism-feature contract and an authoritative "
                        "sample, date, depth, environment and flux crosswalk "
                        "(exact joins: 0 of 133); expression normalization units"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "lane": LANE_DISPLAY.get(lane_id, f"Registered source lane {lane_id}"),
                    "report_units": report_units,
                    "metadata_universe": "registered source payload",
                    "primary_source": lane_id,
                    "resolution_now": f"{tri_view:,}/{report_units:,} tri-view units in the current report input",
                    "use_now": "Source-aware molecular screening lane",
                    "blocking_gap": "Resolve lane-specific sample mapping, abundance, environmental covariates, and validation",
                }
            )
    return rows


def build_source_provenance_readiness(summary: dict[str, Any], atlas: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    base = [
        {
            "lane": "Rumen reference (POC core)",
            "report_units": int(summary.get("poc_rumen_total", 0)),
            "metadata_universe": "555 rumen proteomes in the proof-of-concept cohort; 518 in the atlas reference core",
            "primary_source": "Stewart et al. 2019, Nature Biotechnology",
            "resolution_now": "555/555 exact ENA analysis-accession matches",
            "use_now": "Methane reference with exact provenance; source-aware comparison",
            "blocking_gap": "Animal and sample metadata are cohort-level; any blue-carbon reading needs target-site sample context",
        },
        {
            "lane": "Wetland reference (POC core)",
            "report_units": int(summary.get("poc_wetland_total", 0)),
            "metadata_universe": "107 wetland Methanoregula proteomes in the proof-of-concept cohort",
            "primary_source": "Bechtold et al. 2025, Nature Communications",
            "resolution_now": "20 with exact NCBI assembly and BioSample; 23 with an Old Woman Creek bin and site; 64 with a source bucket only",
            "use_now": "Wetland reference with source context; metadata triage",
            "blocking_gap": "Map every MAG to its BioSample, including source-bucket rows",
        },
    ]
    external_rows = build_external_source_readiness(atlas) if atlas is not None else []
    if external_rows:
        base.extend(external_rows)
    else:
        base.append(
            {
                "lane": LANE_DISPLAY["msm_china_2025"],
                "report_units": int(summary.get("msm_total", 0)),
                "metadata_universe": "1,428 deposited MAGs; the source paper reports 966 final MAGs",
                "primary_source": "Pan et al. 2025, GigaScience",
                "resolution_now": "82 sediment-sample rows; 71 exact BioSample rows; group-level sample lists",
                "use_now": "Mangrove screening and sample-readiness priorities",
                "blocking_gap": "Assign each MAG to its sample and reconcile 1,428 deposited with 966 final MAGs before sample or site summaries",
            }
        )
    return base


def build_report_validation_gates(atlas: pd.DataFrame, payload: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []

    def add(gate: str, passed: bool, details: str) -> None:
        gates.append({"gate": gate, "status": "pass" if passed else "fail", "details": details})

    if "lane_id" in atlas.columns:
        duplicate_nodes = int(atlas[["lane_id", "proteome_id"]].astype(str).duplicated().sum())
        add("one_row_per_lane_id_proteome_id_atlas", duplicate_nodes == 0, f"duplicate_rows={duplicate_nodes}; rows={len(atlas)}")
    else:
        duplicate_nodes = int(atlas["proteome_id"].astype(str).duplicated().sum())
        add("one_row_per_proteome_id_atlas", duplicate_nodes == 0, f"duplicate_rows={duplicate_nodes}; rows={len(atlas)}")

    non_scoped = atlas["atlas_inclusion_status"].astype(str).eq("non_poc_or_unscoped").sum()
    add("no_unscoped_rows_in_atlas_payload", int(non_scoped) == 0, f"non_scoped_rows={int(non_scoped)}")
    wrong_habitat_flags = [link for link in payload.get("niche", {}).get("links", [])
        if legacy.truthy(link.get("cross_domain")) !=
        (str(link.get("source_category")) != str(link.get("target_category")))]
    add("niche_link_habitat_flags_match_source_categories", not wrong_habitat_flags,
        f"mislabeled_links={len(wrong_habitat_flags)}")

    poc = atlas[atlas["atlas_inclusion_status"].astype(str).eq("poc_core_complete")].copy()
    poc_methane_source = pd.to_numeric(
        poc.get("methane_evidence_score", pd.Series(np.nan, index=poc.index)),
        errors="coerce",
    )
    poc_methane_raw = pd.to_numeric(
        poc.get(
            "raw_methane_annotation_row_count",
            pd.Series(np.nan, index=poc.index),
        ),
        errors="coerce",
    )
    poc_sulfur_source = pd.to_numeric(
        poc.get("sulfur_competition_score", pd.Series(np.nan, index=poc.index)),
        errors="coerce",
    )
    poc_sulfur_raw = pd.to_numeric(
        poc.get(
            "raw_sulfur_annotation_row_count",
            pd.Series(np.nan, index=poc.index),
        ),
        errors="coerce",
    )
    # The freeze currently declares no row mechanism-equivalent. Preserve POC
    # source scores in explicitly raw/quarantined fields while withholding the
    # public marker-count fields unless a later evidence contract establishes
    # common mechanism semantics.
    lost_methane = poc[
        poc_methane_source.notna()
        & ~np.isclose(
            poc_methane_source.fillna(0).to_numpy(),
            poc_methane_raw.fillna(0).to_numpy(),
        )
    ]
    lost_sulfur = poc[
        poc_sulfur_source.notna()
        & ~np.isclose(
            poc_sulfur_source.fillna(0).to_numpy(),
            poc_sulfur_raw.fillna(0).to_numpy(),
        )
    ]
    add(
        "poc_methane_scores_preserved",
        lost_methane.empty,
        f"lost_or_changed_raw_examples={lost_methane['proteome_id'].head(5).tolist()}",
    )
    add(
        "poc_sulfur_scores_preserved",
        lost_sulfur.empty,
        f"lost_or_changed_raw_examples={lost_sulfur['proteome_id'].head(5).tolist()}",
    )

    poc_mechanism_equivalent = poc.get(
        "mechanism_equivalence_status", pd.Series("", index=poc.index)
    ).astype(str).eq("mechanism_equivalent")
    poc_rate_status = poc.get(
        "rate_metric_status", pd.Series("", index=poc.index)
    ).astype(str)
    poc_public_rate_present = (
        pd.to_numeric(
            poc.get("methane_marker_density_per_1k", pd.Series(np.nan, index=poc.index)),
            errors="coerce",
        ).notna()
        | pd.to_numeric(
            poc.get("sulfur_context_density_per_1k", pd.Series(np.nan, index=poc.index)),
            errors="coerce",
        ).notna()
        | pd.to_numeric(
            poc.get("substrate_breadth_per_1k", pd.Series(np.nan, index=poc.index)),
            errors="coerce",
        ).notna()
    )
    fake_denominator = poc[
        (poc_mechanism_equivalent & poc_rate_status.ne("comparable_curated_feature_density"))
        | (~poc_mechanism_equivalent & poc_rate_status.eq("comparable_curated_feature_density"))
        | (~poc_mechanism_equivalent & poc_public_rate_present)
    ]
    add(
        "poc_rate_semantics_follow_evidence_contract",
        fake_denominator.empty,
        f"examples={fake_denominator['proteome_id'].head(5).tolist()}",
    )

    required_semantic_columns = {
        "functional_evidence_class",
        "functional_harmonization_status",
        "mechanism_equivalence_status",
        "formal_tri_view_status",
    }
    missing_semantic_columns = sorted(
        required_semantic_columns - set(atlas.columns)
    )
    semantic_blank_rows = 0
    if not missing_semantic_columns:
        semantic_blank_rows = int(
            atlas[list(required_semantic_columns)]
            .fillna("")
            .astype(str)
            .apply(lambda col: col.str.strip().eq(""))
            .any(axis=1)
            .sum()
        )
    add(
        "tri_view_semantic_contract_complete",
        not missing_semantic_columns and semantic_blank_rows == 0,
        (
            f"missing_columns={missing_semantic_columns}; "
            f"rows_with_blank_semantics={semantic_blank_rows}"
        ),
    )
    scaffold_rows = atlas[
        atlas.get(
            "functional_evidence_class", pd.Series("", index=atlas.index)
        ).eq("source_annotation_scaffold")
    ]
    invalid_scaffold = scaffold_rows[
        scaffold_rows.get(
            "mechanism_equivalence_status",
            pd.Series("", index=scaffold_rows.index),
        ).eq("mechanism_equivalent")
        | pd.to_numeric(
            scaffold_rows.get(
                "molecular_attestation_index",
                pd.Series(np.nan, index=scaffold_rows.index),
            ),
            errors="coerce",
        ).notna()
    ]
    add(
        "source_scaffold_not_promoted_to_canonical_mechanism_score",
        invalid_scaffold.empty,
        (
            f"source_scaffold_rows={len(scaffold_rows)}; "
            f"invalid_rows={len(invalid_scaffold)}"
        ),
    )
    non_poc = atlas[~atlas["lane_id"].astype(str).eq("poc_core")]
    invalid_non_poc_equivalence = non_poc[
        non_poc["mechanism_equivalence_status"].eq("mechanism_equivalent")
        | pd.to_numeric(
            non_poc["molecular_attestation_index"], errors="coerce"
        ).notna()
    ]
    add(
        "only_poc_rows_are_mechanism_comparable_or_publicly_scored",
        invalid_non_poc_equivalence.empty,
        (
            f"non_poc_rows={len(non_poc)}; "
            f"invalid_rows={len(invalid_non_poc_equivalence)}"
        ),
    )
    invalid_non_poc_rates = non_poc[
        pd.to_numeric(
            non_poc["methane_marker_density_per_1k"], errors="coerce"
        ).notna()
        | pd.to_numeric(
            non_poc["sulfur_context_density_per_1k"], errors="coerce"
        ).notna()
        | pd.to_numeric(
            non_poc["substrate_breadth_per_1k"], errors="coerce"
        ).notna()
    ]
    add(
        "noncomparable_cross_lane_marker_densities_are_quarantined",
        invalid_non_poc_rates.empty,
        f"invalid_non_poc_rate_rows={len(invalid_non_poc_rates)}",
    )
    # Incomplete rows can retain source-side raw annotation counts while carrying
    # no active functional payload. They are explicit blocked states and never
    # enter public functional density interpretation. Apply the numerator gate
    # to non-POC rows that have an active functional payload.
    functional_non_poc = non_poc[
        non_poc["has_functional"].map(legacy.truthy)
    ].copy()
    raw_ratio = (
        pd.to_numeric(
            functional_non_poc["raw_methane_annotation_row_count"], errors="coerce"
        )
        / pd.to_numeric(
            functional_non_poc["protein_count_for_rates"], errors="coerce"
        )
    )
    explicit_noncomparable_rate_states = {
        "pipeline_normalized_screening_not_cross_lane_mechanism_rate",
        "quarantined_raw_hit_row_numerator_not_marker_density",
        "source_scaffold_term_density_non_equivalent",
    }
    unquarantined_raw_hit_rows = functional_non_poc[
        raw_ratio.gt(1)
        & ~functional_non_poc["rate_metric_status"].astype(str).isin(
            explicit_noncomparable_rate_states
        )
    ]
    add(
        "raw_hit_row_numerators_cannot_be_labeled_marker_density",
        unquarantined_raw_hit_rows.empty,
        (
            f"functional_non_poc_raw_rows_per_protein_gt_1={int(raw_ratio.gt(1).sum())}; "
            f"incomplete_nonfunctional_rows_excluded={int((~non_poc['has_functional'].map(legacy.truthy)).sum())}; "
            f"unquarantined_rows={len(unquarantined_raw_hit_rows)}"
        ),
    )
    invalid_glm_protocol = atlas[
        atlas["has_glm2"].map(legacy.truthy)
        & ~atlas["glm2_protocol_class"].isin(
            {
                "paired_single_native_plus_single_shuffled",
                "multiwindow_10_native_plus_10_shuffled",
            }
        )
    ]
    add(
        "glm2_protocol_class_documented_for_every_glm2_row",
        invalid_glm_protocol.empty,
        f"invalid_rows={len(invalid_glm_protocol)}",
    )
    if "freeze_tri_view_ready" in atlas.columns:
        freeze_present = atlas["freeze_tri_view_ready"].notna()
        computed_tri = (
            atlas["has_esm2"] & atlas["has_glm2"] & atlas["has_functional"]
        )
        frozen_tri = truthy_series(atlas["freeze_tri_view_ready"])
        mismatch = int((freeze_present & computed_tri.ne(frozen_tri)).sum())
        add(
            "freeze_tri_view_reconciles_with_report_rows",
            mismatch == 0,
            (
                f"freeze_annotated_rows={int(freeze_present.sum())}; "
                f"mismatched_rows={mismatch}"
            ),
        )
        semantic_mismatch = 0
        for field in [
            "functional_evidence_class",
            "functional_harmonization_status",
            "mechanism_equivalence_status",
            "formal_tri_view_status",
        ]:
            freeze_field = f"freeze_{field}"
            if freeze_field not in atlas.columns:
                semantic_mismatch += len(atlas)
                continue
            present = atlas[freeze_field].notna()
            semantic_mismatch += int(
                (
                    present
                    & atlas[freeze_field].astype(str).ne(atlas[field].astype(str))
                ).sum()
            )
        add(
            "freeze_semantic_contract_reconciles_with_report_rows",
            semantic_mismatch == 0,
            f"semantic_mismatches={semantic_mismatch}",
        )

    niche_nodes_df = pd.DataFrame(payload.get("niche", {}).get("nodes", []))
    if not niche_nodes_df.empty:
        atlas_embedding_rows = int(truthy_series(atlas.get("has_esm2", pd.Series(dtype=object))).sum())
        finite_niche_rows = int(
            (
                pd.to_numeric(niche_nodes_df.get("diffusion_1"), errors="coerce").notna()
                & pd.to_numeric(niche_nodes_df.get("diffusion_2"), errors="coerce").notna()
            ).sum()
        )
        add(
            "niche_payload_contains_all_embedding_bearing_units",
            finite_niche_rows == atlas_embedding_rows,
            f"finite_diffusion_nodes={finite_niche_rows}; atlas_has_esm2_rows={atlas_embedding_rows}; total_niche_rows={len(niche_nodes_df)}",
        )
        mangrove_case_nodes = int(
            (
                truthy_series(niche_nodes_df.get("is_case_study", pd.Series(dtype=object)))
                & niche_nodes_df.get("source_category", pd.Series(dtype=object)).astype(str).eq("mangrove")
            ).sum()
        )
        case_links = [
            row
            for row in payload.get("niche", {}).get("links", [])
            if str(row.get("evidence_type")) == "case_study_nearest_poc"
        ]
        add(
            "mangrove_case_study_nearest_poc_links_present",
            len(case_links) >= mangrove_case_nodes,
            f"case_links={len(case_links)}; mangrove_case_nodes={mangrove_case_nodes}",
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

    sample_contexts = payload.get("sample_linkage", {}).get("contexts", [])
    add(
        "sample_linkage_context_payload_present",
        len(sample_contexts) > 0,
        f"context_rows={len(sample_contexts)}; claim_scope=sample-readiness only, not methane risk scoring",
    )
    evidence_contract = payload.get("evidence_contract", [])
    add(
        "evidence_contract_payload_has_all_registered_lanes",
        len(evidence_contract) == int(atlas["lane_id"].nunique()),
        (
            f"evidence_contract_rows={len(evidence_contract)}; "
            f"registered_lanes={int(atlas['lane_id'].nunique())}"
        ),
    )

    return gates


def expression_supported(value: Any) -> bool:
    """Processed-expression support arrives as a boolean string or a count."""
    if legacy.truthy(value):
        return True
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(number) and number > 0)


def build_candidate_circos(cards: pd.DataFrame) -> dict[str, Any]:
    """Summarize evidence availability, not cross-pipeline biological strength."""
    pillar_defs = [
        {"id": "esm2", "label": "ESM-2 available", "short": "ESM-2"},
        {"id": "glm2", "label": "gLM2 available", "short": "gLM2"},
        {
            "id": "functional",
            "label": "Functional payload complete",
            "short": "Function",
        },
        {
            "id": "mechanism",
            "label": "Common mechanism contract eligible",
            "short": "Comparable",
        },
        {
            "id": "expression",
            "label": "Processed expression detection",
            "short": "Expression",
        },
        {"id": "qc", "label": "Genome QC available", "short": "QC"},
        {"id": "taxonomy", "label": "Phylum resolved", "short": "Taxonomy"},
        {
            "id": "sample",
            "label": "Sample-context key available",
            "short": "Sample context",
        },
    ]
    # Ring order, inside to outside, follows group_defs.
    group_defs = [
        {"id": "poc", "label": "Reference-core candidates (POC)", "color": COLORS["rumen"]},
        {"id": "mangrove", "label": "Mangrove candidates", "color": COLORS["mangrove"]},
        {"id": "mucc", "label": "Old Woman Creek candidates", "color": COLORS["wetland"]},
    ]

    def group_mask(group_id: str) -> pd.Series:
        candidate_set = cards["candidate_set"].fillna("").astype(str)
        if group_id == "poc":
            return candidate_set.eq("POC geometry-led review candidate")
        if group_id == "mangrove":
            return candidate_set.str.startswith("Mangrove geometry-led")
        return candidate_set.str.startswith("MUCC v1 source-scaffold")

    def evidence_value(frame: pd.DataFrame, pillar_id: str) -> pd.Series:
        if pillar_id == "esm2":
            return frame.get("has_esm2", False).map(legacy.truthy).astype(float)
        if pillar_id == "glm2":
            return frame.get("has_glm2", False).map(legacy.truthy).astype(float)
        if pillar_id == "functional":
            return frame.get("has_functional", False).map(legacy.truthy).astype(float)
        if pillar_id == "mechanism":
            return frame.get(
                "mechanism_equivalence_status", pd.Series("", index=frame.index)
            ).astype(str).eq("mechanism_equivalent").astype(float)
        if pillar_id == "expression":
            return frame.get(
                "processed_gene_expression_support",
                pd.Series(0, index=frame.index),
            ).map(expression_supported).astype(float)
        if pillar_id == "qc":
            completeness = pd.to_numeric(
                frame.get(
                    "checkm2_completeness", pd.Series(np.nan, index=frame.index)
                ),
                errors="coerce",
            )
            contamination = pd.to_numeric(
                frame.get(
                    "checkm2_contamination", pd.Series(np.nan, index=frame.index)
                ),
                errors="coerce",
            )
            return (completeness.notna() & contamination.notna()).astype(float)
        if pillar_id == "taxonomy":
            return frame.get("phylum", pd.Series("", index=frame.index)).fillna(
                ""
            ).astype(str).str.strip().ne("").astype(float)
        return frame.get(
            "sample_context_key", pd.Series("", index=frame.index)
        ).fillna("").astype(str).str.strip().ne("").astype(float)

    records: list[dict[str, Any]] = []
    for group in group_defs:
        sub = cards[group_mask(group["id"])].copy()
        n = int(len(sub))
        for pillar in pillar_defs:
            values = evidence_value(sub, pillar["id"]) if n else pd.Series(dtype=float)
            high_count = int(values.ge(1).sum()) if n else 0
            records.append(
                {
                    "group": group["id"],
                    "group_label": group["label"],
                    "pillar": pillar["id"],
                    "pillar_label": pillar["label"],
                    "pillar_short": pillar["short"],
                    "source": (
                        "Share of cards in this group with the evidence available or eligible. "
                        "It shows review readiness, not biological signal strength."
                    ),
                    "candidate_count": n,
                    "high_count": high_count,
                    "high_share": high_count / n if n else 0.0,
                    "average_value": float(values.mean()) if n else 0.0,
                    "threshold": 1.0,
                }
            )
    return {"groups": group_defs, "pillars": pillar_defs, "records": records}


def build_signature_matrix(cards: pd.DataFrame) -> dict[str, Any]:
    metric_defs = [
        {"id": "esm2_available", "label": "ESM-2", "source": "ESM-2 payload availability"},
        {"id": "glm2_available", "label": "gLM2", "source": "gLM2 payload availability"},
        {
            "id": "functional_available",
            "label": "Function",
            "source": "completed functional payload availability",
        },
        {
            "id": "mechanism_comparable",
            "label": "Comparable",
            "source": "eligibility for the common mechanism-feature contract",
        },
        {
            "id": "expression_available",
            "label": "Expression",
            "source": "processed expression detection in the MUCC source scaffold",
        },
        {"id": "qc_available", "label": "QC", "source": "CheckM2 QC field availability"},
        {"id": "taxonomy_available", "label": "Taxonomy", "source": "resolved phylum label"},
        {
            "id": "sample_context_available",
            "label": "Sample context",
            "source": "sample and context availability. Exact flux linkage requires matched source measurements.",
        },
    ]

    def evidence_values(row: pd.Series) -> dict[str, float]:
        return {
            "esm2_available": float(legacy.truthy(row.get("has_esm2"))),
            "glm2_available": float(legacy.truthy(row.get("has_glm2"))),
            "functional_available": float(
                legacy.truthy(row.get("has_functional"))
            ),
            "mechanism_comparable": float(
                row.get("mechanism_equivalence_status")
                == "mechanism_equivalent"
            ),
            "expression_available": float(
                expression_supported(row.get("processed_gene_expression_support"))
            ),
            "qc_available": float(
                pd.notna(row.get("checkm2_completeness"))
                and pd.notna(row.get("checkm2_contamination"))
            ),
            "taxonomy_available": float(
                bool(str(row.get("phylum") or "").strip())
            ),
            "sample_context_available": float(
                bool(str(row.get("sample_context_key") or "").strip())
            ),
        }

    records: list[dict[str, Any]] = []
    for _, row in cards.iterrows():
        # One code per candidate set keeps row labels unique: P = reference core,
        # M = mangrove, O = Old Woman Creek.
        candidate_set = str(row.get("candidate_set", ""))
        set_code = "P" if candidate_set == "POC geometry-led review candidate" else "M" if candidate_set.startswith("Mangrove") else "O"
        display_label = f"{set_code}{safe_int(row.get('rank')):02d}  {short_id(row['proteome_id'], 34)}"
        values = evidence_values(row)
        for metric in metric_defs:
            records.append(
                {
                    "proteome_id": row["proteome_id"],
                    "candidate_set": row.get("candidate_set", ""),
                    "rank": safe_int(row.get("rank")),
                    "source_category": row.get("source_category", ""),
                    "lane_key": row.get("lane_key", row.get("source_category", "")),
                    "label": display_label,
                    "metric": metric["id"],
                    "metric_label": metric["label"],
                    "metric_source": metric["source"],
                    "value": values[metric["id"]],
                }
            )
    return {"metrics": [m["id"] for m in metric_defs], "metric_defs": metric_defs, "records": records}


def build_evidence_contract_summary(atlas: pd.DataFrame) -> list[dict[str, Any]]:
    labels = LANE_DISPLAY
    rows: list[dict[str, Any]] = []
    for lane_id in [
        "poc_core",
        "msm_china_2025",
        "futian_mangrove_2026_qi",
        "mucc_v1_owc_wetland",
    ]:
        frame = atlas[atlas["lane_id"].astype(str).eq(lane_id)].copy()
        if frame.empty:
            continue
        tri = (
            frame["has_esm2"].map(legacy.truthy)
            & frame["has_glm2"].map(legacy.truthy)
            & frame["has_functional"].map(legacy.truthy)
        )
        rows.append(
            {
                "lane_id": lane_id,
                "lane": labels.get(lane_id, lane_id),
                "registered_units": int(len(frame)),
                "esm2_units": int(frame["has_esm2"].map(legacy.truthy).sum()),
                "glm2_units": int(frame["has_glm2"].map(legacy.truthy).sum()),
                "functional_payload_units": int(
                    frame["has_functional"].map(legacy.truthy).sum()
                ),
                "data_complete_tri_view_units": int(tri.sum()),
                "mechanism_comparable_tri_view_units": int(
                    (
                        tri
                        & frame["mechanism_equivalence_status"].eq(
                            "mechanism_equivalent"
                        )
                    ).sum()
                ),
                "annotation_complete_harmonization_pending_units": int(
                    frame["functional_comparability_tier"].eq(
                        "annotation_complete_harmonization_pending"
                    ).sum()
                ),
                "source_scaffold_units": int(
                    frame["functional_comparability_tier"].eq(
                        "source_scaffold_non_equivalent"
                    ).sum()
                ),
                "single_window_glm2_units": int(
                    frame["glm2_protocol_class"].eq(
                        "paired_single_native_plus_single_shuffled"
                    ).sum()
                ),
                "multiwindow_glm2_units": int(
                    frame["glm2_protocol_class"].eq(
                        "multiwindow_10_native_plus_10_shuffled"
                    ).sum()
                ),
                "esm2_cap_applied_units": int(
                    frame["esm2_protein_cap_status"].eq("cap_6000_applied").sum()
                ),
                "functional_contract": str(
                    frame["functional_comparability_tier"].mode().iloc[0]
                ),
            }
        )
    return rows


def reciprocal_unique_cross_pairs(edge_df: pd.DataFrame) -> pd.DataFrame:
    cross = edge_df[
        edge_df["cross_domain"].map(legacy.truthy)
        & edge_df["reciprocal"].map(legacy.truthy)
    ].copy()
    if cross.empty:
        return cross
    cross["pair_key"] = cross.apply(
        lambda row: "||".join(
            sorted([str(row["source"]), str(row["target"])])
        ),
        axis=1,
    )
    return cross.drop_duplicates("pair_key").copy()


def category_pair_counts(
    pairs: pd.DataFrame,
    source_category_by_id: dict[str, str],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in pairs.itertuples(index=False):
        categories = sorted(
            {
                source_category_by_id.get(str(row.source), ""),
                source_category_by_id.get(str(row.target), ""),
            }
        )
        counts["↔".join(categories)] += 1
    return dict(counts)


def build_nearest_core_context_audit(
    atlas: pd.DataFrame,
    emb_meta: pd.DataFrame,
    cards: pd.DataFrame,
) -> dict[str, Any]:
    """Count one-way raw-cosine nearest-core links at MAG/proteome grain.

    These links answer a different question from reciprocal top-k neighbors in
    the full atlas. The POC core is the reference set, not an independent
    validation cohort, and a nearest match is not functional transfer.
    """
    core = atlas.loc[
        atlas["atlas_inclusion_status"].eq("poc_core_complete"),
        ["proteome_id", "source_category"],
    ].copy()
    if core["proteome_id"].duplicated().any():
        raise ValueError("POC reference core contains duplicate proteome_id values")
    core_category = core.set_index("proteome_id")["source_category"].to_dict()

    neighbors = emb_meta["nearest_poc_id"].map(core_category)
    if neighbors.isna().any():
        raise ValueError("Nearest POC reference is missing from the POC core")
    targets = cards[cards["source_category"].isin(["wetland", "mangrove"])]
    candidate_neighbors = targets["nearest_poc_id"].map(core_category)
    if candidate_neighbors.isna().any():
        raise ValueError("A target candidate lacks a POC core reference")
    # Core members match themselves, so denominators for "points to rumen"
    # count only records outside the reference core.
    in_core = emb_meta["proteome_id"].astype(str).isin(set(core_category))
    wetland_outside = emb_meta["source_category"].eq("wetland") & ~in_core
    similarity = pd.to_numeric(emb_meta["nearest_poc_similarity"], errors="coerce")
    targets_outside = ~targets["proteome_id"].astype(str).isin(set(core_category))

    return {
        "unit_grain": "MAG/proteome embedding record",
        "metric": "raw ESM-2 cosine similarity",
        "comparison": "single nearest member of the POC reference core",
        "reference_core_units": int(len(core)),
        "reference_core_rumen_units": int(core["source_category"].eq("rumen").sum()),
        "reference_core_wetland_units": int(core["source_category"].eq("wetland").sum()),
        "wetland_embedding_units": int(emb_meta["source_category"].eq("wetland").sum()),
        "wetland_nearest_rumen_units": int((emb_meta["source_category"].eq("wetland") & neighbors.eq("rumen")).sum()),
        "mangrove_embedding_units": int(emb_meta["source_category"].eq("mangrove").sum()),
        "mangrove_nearest_rumen_units": int((emb_meta["source_category"].eq("mangrove") & neighbors.eq("rumen")).sum()),
        "target_candidate_cards": int(len(targets)),
        "target_candidate_nearest_rumen_cards": int(candidate_neighbors.eq("rumen").sum()),
        "wetland_outside_core_units": int(wetland_outside.sum()),
        "wetland_outside_core_nearest_rumen_units": int((wetland_outside & neighbors.eq("rumen")).sum()),
        "outside_core_units": int((~in_core).sum()),
        "outside_core_nearest_similarity_median": float(similarity[~in_core].median()),
        "target_candidate_cards_outside_core": int(targets_outside.sum()),
        "target_candidate_outside_core_nearest_rumen_cards": int((targets_outside & candidate_neighbors.eq("rumen")).sum()),
        "source_tables": ["tables/embedding_context_table.tsv", "tables/candidate_cards.tsv"],
        "interpretation": "One-way nearest-reference links nominate records for review; they do not establish reciprocal neighborhoods, source-independent transfer, pathway activity, or methane flux.",
    }


def build_embedding_geometry_audit(
    emb_meta: pd.DataFrame,
    edge_df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int,
) -> dict[str, Any]:
    """Quantify anisotropy and a simple dimension-standardized sensitivity."""
    x_raw = np.asarray(embeddings, dtype=np.float32)
    x = normalize(x_raw).astype(np.float32, copy=False)
    rng = np.random.default_rng(20260724)
    pair_n = min(100_000, max(10_000, len(x) * 10))
    left = rng.integers(0, len(x), size=pair_n)
    right = rng.integers(0, len(x), size=pair_n)
    same = left == right
    right[same] = (right[same] + 1) % len(x)
    pair_similarity = np.einsum("ij,ij->i", x[left], x[right])
    centroid = x.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    centroid_similarity = x @ centroid

    cross = edge_df[edge_df["cross_domain"].map(legacy.truthy)].copy()
    raw_pairs = reciprocal_unique_cross_pairs(edge_df)
    source_category_by_id = (
        emb_meta.set_index("proteome_id")["source_category"]
        .fillna("")
        .astype(str)
        .to_dict()
    )
    raw_pair_counts = category_pair_counts(raw_pairs, source_category_by_id)

    mean = x_raw.mean(axis=0, keepdims=True)
    scale = x_raw.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    z = (x_raw - mean) / scale
    n_neighbors = min(k + 1, len(z))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(z)
    distances, indices = nn.kneighbors(z)
    z_edges: list[tuple[int, int, float]] = []
    for i, (row_dist, row_idx) in enumerate(zip(distances, indices)):
        kept = 0
        for dist, j in zip(row_dist, row_idx):
            if i == int(j):
                continue
            z_edges.append((i, int(j), float(1.0 - dist)))
            kept += 1
            if kept >= k:
                break
    z_edge_set = {(i, j) for i, j, _ in z_edges}
    z_cross = [
        (i, j, sim)
        for i, j, sim in z_edges
        if str(emb_meta.iloc[i]["source_category"])
        != str(emb_meta.iloc[j]["source_category"])
    ]
    z_reciprocal_pairs: set[tuple[int, int]] = set()
    for i, j, _ in z_cross:
        if (j, i) in z_edge_set:
            z_reciprocal_pairs.add(tuple(sorted((i, j))))
    z_pair_counts: dict[str, int] = defaultdict(int)
    for i, j in z_reciprocal_pairs:
        cats = sorted(
            {
                str(emb_meta.iloc[i]["source_category"]),
                str(emb_meta.iloc[j]["source_category"]),
            }
        )
        z_pair_counts["↔".join(cats)] += 1

    return {
        "embedding_units": int(len(x)),
        "dimensions": int(x.shape[1]),
        "knn_k": int(k),
        "raw_directed_edges": int(len(edge_df)),
        "raw_cross_domain_directed_edges": int(len(cross)),
        "raw_cross_edge_similarity_mean": float(
            pd.to_numeric(cross["similarity"], errors="coerce").mean()
        ),
        "raw_cross_edge_similarity_median": float(
            pd.to_numeric(cross["similarity"], errors="coerce").median()
        ),
        "raw_cross_edge_similarity_min": float(
            pd.to_numeric(cross["similarity"], errors="coerce").min()
        ),
        "raw_reciprocal_unique_cross_pairs": int(len(raw_pairs)),
        "raw_reciprocal_pair_counts": raw_pair_counts,
        "random_pair_similarity_mean": float(pair_similarity.mean()),
        "random_pair_similarity_median": float(np.median(pair_similarity)),
        "similarity_to_global_centroid_median": float(
            np.median(centroid_similarity)
        ),
        "dimension_zscore_cross_domain_directed_edges": int(len(z_cross)),
        "dimension_zscore_cross_edge_similarity_median": float(
            np.median([row[2] for row in z_cross]) if z_cross else np.nan
        ),
        "dimension_zscore_reciprocal_unique_cross_pairs": int(
            len(z_reciprocal_pairs)
        ),
        "dimension_zscore_reciprocal_pair_counts": dict(z_pair_counts),
        "interpretation": (
            "Raw ESM-2 cosine similarity and dimension-standardized neighborhoods "
            "are reported separately. Biological transfer requires independent evidence. The graph supports "
            "neighborhood navigation and routes mechanism attestation to its dedicated evidence contract."
        ),
    }


def clean_taxon_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalized_phylum(value: Any) -> str:
    text = clean_taxon_label(value)
    if text.startswith("p__"):
        text = text[3:]
    aliases = {
        "Proteobacteria": "Pseudomonadota",
        "Actinobacteriota": "Actinomycetota",
        "Patescibacteria": "Patescibacteriota",
    }
    return aliases.get(text, text)


def build_taxonomy_bridge_audit(
    atlas: pd.DataFrame,
    edge_df: pd.DataFrame,
) -> dict[str, Any]:
    pairs = reciprocal_unique_cross_pairs(edge_df)
    lookup = (
        atlas.drop_duplicates("proteome_id")
        .set_index("proteome_id")[["source_category", "phylum", "gtdb_release"]]
        .to_dict("index")
    )
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        left = lookup.get(str(pair.source), {})
        right = lookup.get(str(pair.target), {})
        categories = sorted(
            [str(left.get("source_category", "")), str(right.get("source_category", ""))]
        )
        if categories != ["mangrove", "wetland"]:
            continue
        left_phylum = clean_taxon_label(left.get("phylum"))
        right_phylum = clean_taxon_label(right.get("phylum"))
        usable = bool(left_phylum and right_phylum)
        rows.append(
            {
                "source": str(pair.source),
                "target": str(pair.target),
                "left_phylum": left_phylum,
                "right_phylum": right_phylum,
                "both_phyla_usable": usable,
                "raw_exact_match": usable and left_phylum == right_phylum,
                "synonym_normalized_match": (
                    usable
                    and normalized_phylum(left_phylum)
                    == normalized_phylum(right_phylum)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    usable = frame[frame["both_phyla_usable"]] if not frame.empty else frame
    release_counts = (
        atlas.groupby(["lane_id", "gtdb_release"], dropna=False)
        .size()
        .reset_index(name="units")
        .fillna({"gtdb_release": "missing"})
        .to_dict("records")
    )
    return {
        "target_target_reciprocal_pairs": int(len(frame)),
        "pairs_with_both_phyla_usable": int(len(usable)),
        "raw_exact_name_matches_all_pairs": int(
            frame["raw_exact_match"].sum() if not frame.empty else 0
        ),
        "raw_exact_name_share_usable": float(
            usable["raw_exact_match"].mean() if len(usable) else np.nan
        ),
        "synonym_normalized_matches_all_pairs": int(
            frame["synonym_normalized_match"].sum() if not frame.empty else 0
        ),
        "synonym_normalized_share_usable": float(
            usable["synonym_normalized_match"].mean() if len(usable) else np.nan
        ),
        "gtdb_release_by_lane": release_counts,
        "interpretation": (
            "Reciprocal mangrove↔wetland neighborhoods are substantially "
            "taxonomically structured. GTDB release metadata is present only "
            "for the POC lane, so source and taxonomy-release effects cannot be "
            "cleanly separated in this freeze."
        ),
    }


def build_functional_metric_audit(atlas: pd.DataFrame) -> dict[str, Any]:
    labels = LANE_DISPLAY
    rows: list[dict[str, Any]] = []
    lane_ids = atlas["lane_id"].astype(str)
    ordered = [lane for lane in LANE_ORDER if lane in set(lane_ids)] + sorted(set(lane_ids) - set(LANE_ORDER))
    for lane_id in ordered:
        frame = atlas[lane_ids.eq(lane_id)]
        functional = frame[frame["has_functional"].map(legacy.truthy)].copy()
        raw = pd.to_numeric(
            functional["raw_methane_annotation_row_count"], errors="coerce"
        )
        proteins = pd.to_numeric(
            functional["protein_count_for_rates"], errors="coerce"
        )
        ratio = raw / proteins
        rows.append(
            {
                "lane_id": lane_id,
                "lane": labels.get(lane_id, lane_id),
                "functional_units": int(len(functional)),
                "numerator_provenance": str(
                    functional["functional_numerator_provenance"].mode().iloc[0]
                )
                if len(functional)
                else "",
                "raw_methane_row_count_median": float(raw.median())
                if len(functional)
                else np.nan,
                "protein_count_median": float(proteins.median())
                if len(functional)
                else np.nan,
                "raw_rows_per_protein_gt_1_units": int(ratio.gt(1).sum()),
                "raw_rows_per_protein_gt_1_share": float(ratio.gt(1).mean())
                if len(functional)
                else np.nan,
                "public_rate_metric_status": str(
                    functional["rate_metric_status"].mode().iloc[0]
                )
                if len(functional)
                else "not_available",
            }
        )
    legacy_index = pd.to_numeric(
        atlas["legacy_published_attestation_index_quarantined"],
        errors="coerce",
    )
    methane_index = pd.to_numeric(
        atlas["legacy_published_methane_signal_index"], errors="coerce"
    )
    valid = legacy_index.notna() & methane_index.notna()
    correlation = (
        float(legacy_index[valid].corr(methane_index[valid]))
        if int(valid.sum()) > 2
        else np.nan
    )
    top = atlas.assign(_legacy_index=legacy_index).nlargest(500, "_legacy_index")
    top_counts = (
        top["source_category"].astype(str).value_counts().sort_index().to_dict()
    )
    return {
        "lane_metrics": rows,
        "legacy_score_methane_component_pearson_r": correlation,
        "legacy_top_500_source_counts": {
            str(key): int(value) for key, value in top_counts.items()
        },
        "legacy_top_500_mangrove_share": float(
            top["source_category"].astype(str).eq("mangrove").mean()
        ),
        "public_action": (
            "Cross-route methane, sulfur and substrate densities and any universal "
            "ranking are quarantined. Raw annotation counts appear only as "
            "diagnostics until a shared feature table is built and validated."
        ),
    }


def build_mucc_validation_readiness(
    atlas: pd.DataFrame,
    repo_root: Path,
) -> dict[str, Any]:
    mucc = atlas[atlas["lane_id"].astype(str).eq("mucc_v1_owc_wetland")].copy()
    processed_support = mucc.get(
        "processed_mag_expression_support",
        pd.Series(False, index=mucc.index),
    )
    processed_support = processed_support.map(legacy.truthy) | pd.to_numeric(
        processed_support, errors="coerce"
    ).fillna(0).gt(0)
    out: dict[str, Any] = {
        "mag_units": int(len(mucc)),
        "processed_expression_supported_mags": int(processed_support.sum()),
        "methane_expression_detected_mags": int(
            pd.to_numeric(
                mucc.get(
                    "methane_expressed_gene_rows", pd.Series(0, index=mucc.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        ),
        "sulfur_expression_detected_mags": int(
            pd.to_numeric(
                mucc.get(
                    "sulfur_expressed_gene_rows", pd.Series(0, index=mucc.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        ),
        "expression_units_status": (
            "processed detection/occupancy support; source normalization units "
            "remain unresolved, so no activity magnitude or flux inference"
        ),
    }
    db = (
        repo_root
        / "results/functional_metagenomics/mucc_v1_owc_wetland_20260626/"
        "cohort_warehouse/functional_atlas.duckdb"
    )
    if not db.exists():
        out["warehouse_status"] = "not_available"
        return out
    try:
        import duckdb

        con = duckdb.connect(str(db), read_only=True)

        def scalar(query: str) -> int:
            return int(con.execute(query).fetchone()[0] or 0)

        out.update(
            {
                "warehouse_status": "available",
                "expression_sample_columns": scalar(
                    "select count(*) from feature_mucc_v1_sample_ecological_readiness"
                ),
                "chamber_flux_rows": scalar(
                    "select count(*) from fact_mucc_v1_essdive_chamber_flux"
                ),
                "chamber_flux_valid_rows": scalar(
                    "select count(*) from fact_mucc_v1_essdive_chamber_flux "
                    "where source_value_status='reported_valid'"
                ),
                "porewater_rows": scalar(
                    "select count(*) from fact_mucc_v1_essdive_porewater_ch4"
                ),
                "porewater_valid_rows": scalar(
                    "select count(*) from fact_mucc_v1_essdive_porewater_ch4 "
                    "where source_value_status='reported_valid'"
                ),
                "tower_flux_rows": scalar(
                    "select count(*) from fact_mucc_v1_essdive_gapfilled_tower_ch4_flux"
                ),
                "exact_sample_environment_flux_links": scalar(
                    "select count(*) from feature_mucc_v1_sample_ecological_readiness "
                    "where authoritative_ecology_link_status not in ('','not_staged')"
                ),
                "ecological_join_blocked_samples": scalar(
                    "select count(*) from feature_mucc_v1_sample_ecological_readiness "
                    "where sample_ecological_validation_status like 'blocked%'"
                ),
                "flashweave_edges": scalar(
                    "select count(*) from fact_mucc_v1_flashweave_edge_stability"
                ),
                "flashweave_stable_edges": scalar(
                    "select count(*) from fact_mucc_v1_flashweave_edge_stability "
                    "where stability_class<>'below_stability_threshold'"
                ),
                "wgcna_non_grey_modules": scalar(
                    "select count(*) from feature_mucc_v1_wgcna_secondary_module_summary "
                    "where module<>'grey'"
                ),
            }
        )
        con.close()
    except Exception as exc:
        out["warehouse_status"] = (
            f"audit_query_unavailable:{type(exc).__name__}:{exc}"
        )
    out["claim_boundary"] = (
        "Expression is orthogonal processed detection evidence. ESS-DIVE flux "
        "and porewater observations are staged site/time context only: 0/133 "
        "sequencing samples currently have an authoritative exact "
        "sample-depth-environment-flux join."
    )
    return out


def build_scientific_findings(
    atlas: pd.DataFrame,
    evidence_contract: list[dict[str, Any]],
    geometry: dict[str, Any],
    taxonomy: dict[str, Any],
    functional: dict[str, Any],
    mucc: dict[str, Any],
) -> list[dict[str, Any]]:
    comparable = int(
        atlas["mechanism_equivalent_tri_view"].map(legacy.truthy).sum()
    )
    tri_view = int(atlas["tri_view_ready"].map(legacy.truthy).sum())
    annotation_pending = int(
        (
            atlas["formal_tri_view_status"]
            == "complete_annotation_tri_view_harmonization_pending"
        ).sum()
    )
    source_scaffold = int(
        (atlas["formal_tri_view_status"] == "complete_source_scaffold_tri_view").sum()
    )
    raw_pairs = geometry.get("raw_reciprocal_pair_counts", {})
    z_pairs = geometry.get("dimension_zscore_reciprocal_pair_counts", {})
    pipeline_normalized = int(
        atlas["functional_comparability_tier"].eq("pipeline_normalized_comparability_pending").sum()
    )
    return [
        {
            "severity": "Evidence states",
            "finding": "Complete data and comparable data are separate states.",
            "result": (
                f"{tri_view:,} records have all three views. {pipeline_normalized:,} carry "
                f"shared-pipeline screening events, {source_scaffold:,} carry source "
                f"annotations, {annotation_pending:,} await shared aggregation and "
                f"{comparable:,} are mechanism-comparable across routes."
            ),
            "report_action": (
                "Every count is reported with its state. No record receives a "
                "cross-route methane ranking."
            ),
        },
        {
            "severity": "Functional counting",
            "finding": "Raw annotation-hit rows are not a methane-gene count.",
            "result": (
                "One gene can produce several hits, and hits per protein differ by "
                "lane and tool; the source lane uses DRAM terms instead."
            ),
            "report_action": functional["public_action"],
        },
        {
            "severity": "Ranking",
            "finding": "A combined index built on raw hit rows tracked lane-specific counting.",
            "result": (
                f"Its methane component correlated with the index at Pearson r = "
                f"{functional['legacy_score_methane_component_pearson_r']:.3f}; "
                + (
                    "all of its top 500 records were mangrove."
                    if functional["legacy_top_500_mangrove_share"] >= 0.9995
                    else f"{100 * functional['legacy_top_500_mangrove_share']:.1f}% of its top 500 records were mangrove."
                )
            ),
            "report_action": (
                "The index is quarantined. Mangrove candidates are chosen by "
                "embedding geometry and QC only."
            ),
        },
        {
            "severity": "Embedding geometry",
            "finding": "Raw and dimension-standardized neighbor counts are sensitivity diagnostics, not transfer validation.",
            "result": (
                f"Mutual top-{safe_int(geometry.get('knn_k'))} pairs in raw space: "
                f"{raw_pairs.get('mangrove↔wetland', 0):,} mangrove–wetland, "
                f"{raw_pairs.get('rumen↔wetland', 0):,} rumen–wetland, "
                f"{raw_pairs.get('mangrove↔rumen', 0):,} rumen–mangrove. "
                f"After standardizing each dimension: "
                f"{z_pairs.get('mangrove↔wetland', 0):,}, "
                f"{z_pairs.get('rumen↔wetland', 0):,} and "
                f"{z_pairs.get('mangrove↔rumen', 0):,}."
            ),
            "report_action": (
                "Embedding links are neighborhoods to explore. Shared function "
                "across sources needs independent validation."
            ),
        },
        {
            "severity": "Taxonomy",
            "finding": "Neighbor pairs often share a phylum, and taxonomy releases differ by source.",
            "result": (
                f"{100 * taxonomy['synonym_normalized_share_usable']:.1f}% of usable "
                "mutual mangrove–wetland pairs share a phylum after conservative "
                "synonym normalization; GTDB release metadata exists only for the "
                "reference core."
            ),
            "report_action": (
                "Part of the neighborhood continuity is taxonomic. Harmonized "
                "taxonomy and phylogeny-aware null models come before any "
                "functional reading."
            ),
        },
        {
            "severity": "Old Woman Creek",
            "finding": "Expression and field data exist; exact joins do not yet.",
            "result": (
                f"{mucc.get('methane_expression_detected_mags', 0):,} MAGs have "
                f"processed methane-gene detection and "
                f"{mucc.get('sulfur_expression_detected_mags', 0):,} have sulfur-gene detection. "
                f"Exact sample, environment and flux joins: "
                f"{mucc.get('exact_sample_environment_flux_links', 0)} of "
                f"{mucc.get('expression_sample_columns', 133)}."
            ),
            "report_action": (
                "Expression counts as detection support only. Activity levels and "
                "flux attribution wait for the exact joins."
            ),
        },
    ]


def sample_linkage_bucket(row: pd.Series) -> str:
    status = str(row.get("sample_rollup_status", "")).lower()
    resolution = str(row.get("sample_context_resolution", row.get("provenance_resolution_tier", ""))).lower()
    lane = str(row.get("lane_id", ""))
    source_category = str(row.get("source_category", ""))
    if source_category == "rumen":
        return "reference_context"
    if source_category == "wetland":
        if "biosample" in resolution or "owc" in resolution:
            return "mixed_wetland_context"
        return "source_bucket_only"
    if lane == "futian_mangrove_2026_qi":
        return "site_month_context"
    if lane == "msm_china_2025":
        return "sample_set_context"
    if "missing" in str(row.get("atlas_inclusion_status", "")).lower():
        return "payload_gap"
    if "blocked" in status:
        return "context_pending"
    return "context_pending"


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def sample_context_chart_label(row: pd.Series) -> str:
    """Short, readable chart label; the full source label stays in the tooltip."""
    label = str(row.get("sample_context_label", "") or "")
    lane = str(row.get("lane_id", ""))
    if int(row.get("tri_view_units", 0) or 0) == 0:
        return f"{LANE_DISPLAY.get(lane, 'Source')} · source gaps"
    if lane == "futian_mangrove_2026_qi" and " · " in label:
        site, key = label.split(" · ", 1)
        match = re.search(r"_(\d{4})(\d{2})$", key)
        when = f"{MONTHS[int(match.group(2)) - 1]} {match.group(1)}" if match else key
        return f"Futian · {site} · {when}"
    if lane == "msm_china_2025" and " · " in label:
        group, places = label.split(" · ", 1)
        number = re.sub(r"\D", "", group) or group
        names = [p.split(":")[-1].strip() for p in places.split(";") if p.strip()]
        names = [n for n in names if n and n.lower() != "china"]
        head = names[0] if names else places
        more = f" +{len(names) - 1}" if len(names) > 1 else ""
        return f"MSM group {number} · {head}{more}"
    return label


def build_sample_linkage_payload(atlas: pd.DataFrame) -> dict[str, Any]:
    """Summarize MAG/proteome signatures at the strongest current sample-context grain."""
    df = atlas.copy()
    if df.empty:
        return {"groups": [], "contexts": [], "status_defs": []}
    for col in ["has_esm2", "has_glm2", "has_functional"]:
        if col in df.columns:
            df[col] = truthy_series(df[col])
        else:
            df[col] = False
    df["tri_view_ready"] = df["has_esm2"] & df["has_glm2"] & df["has_functional"]
    df["sample_linkage_bucket"] = df.apply(sample_linkage_bucket, axis=1)
    df["sample_context_label"] = df["sample_context_label"].fillna("").astype(str)
    df["sample_context_key"] = df["sample_context_key"].fillna("").astype(str)
    empty_context = df["sample_context_label"].str.strip().eq("")
    df.loc[empty_context, "sample_context_label"] = (
        df.loc[empty_context, "source_display"].fillna("").astype(str) + " · no sample context"
    )
    df["sample_context_sort"] = df["source_display"].fillna("").astype(str) + " · " + df["sample_context_label"].astype(str)
    metric_cols = [
        "molecular_attestation_index",
        "bridge_affinity_index",
        "methane_signal_index",
        "sulfur_context_index",
        "substrate_breadth_index",
        "annotation_breadth_index",
        "qc_confidence_index",
        "methane_marker_density_per_1k",
        "sulfur_context_density_per_1k",
        "substrate_breadth_per_1k",
        "linked_sample_context_count",
        "environmental_context_fields_present",
        "mean_ph",
        "mean_salinity_psu",
        "mean_toc_mg_g",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    lane_group = (
        df.groupby(["source_display", "sample_linkage_bucket", "sample_rollup_status"], dropna=False)
        .agg(
            units=("proteome_id", "count"),
            tri_view_units=("tri_view_ready", "sum"),
            esm2_units=("has_esm2", "sum"),
            glm2_units=("has_glm2", "sum"),
            functional_units=("has_functional", "sum"),
            average_molecular_attestation=("molecular_attestation_index", "mean"),
            average_methane_signal=("methane_signal_index", "mean"),
            average_sulfur_context=("sulfur_context_index", "mean"),
            average_substrate_breadth=("substrate_breadth_index", "mean"),
        )
        .reset_index()
    )
    context_group_cols = [
        "lane_id",
        "source_display",
        "sample_context_key",
        "sample_context_label",
        "sample_context_resolution",
        "sample_linkage_bucket",
        "sample_rollup_status",
        "environmental_context_status",
        "sample_context_blocking_gap",
    ]
    context_group_cols = [c for c in context_group_cols if c in df.columns]
    contexts = (
        df[df["source_category"].astype(str).eq("mangrove")]
        .groupby(context_group_cols, dropna=False)
        .agg(
            units=("proteome_id", "count"),
            tri_view_units=("tri_view_ready", "sum"),
            esm2_units=("has_esm2", "sum"),
            glm2_units=("has_glm2", "sum"),
            functional_units=("has_functional", "sum"),
            average_molecular_attestation=("molecular_attestation_index", "mean"),
            average_bridge_affinity=("bridge_affinity_index", "mean"),
            average_methane_signal=("methane_signal_index", "mean"),
            average_sulfur_context=("sulfur_context_index", "mean"),
            average_substrate_breadth=("substrate_breadth_index", "mean"),
            average_annotation_breadth=("annotation_breadth_index", "mean"),
            average_qc_confidence=("qc_confidence_index", "mean"),
            methane_density_mean=("methane_marker_density_per_1k", "mean"),
            sulfur_density_mean=("sulfur_context_density_per_1k", "mean"),
            substrate_density_mean=("substrate_breadth_per_1k", "mean"),
            linked_sample_context_count=("linked_sample_context_count", "max"),
            environmental_context_fields_present=("environmental_context_fields_present", "max"),
            mean_ph=("mean_ph", "mean"),
            mean_salinity_psu=("mean_salinity_psu", "mean"),
            mean_toc_mg_g=("mean_toc_mg_g", "mean"),
        )
        .reset_index()
    )
    if not contexts.empty:
        contexts["sample_readiness_label"] = np.select(
            [
                contexts["tri_view_units"].gt(0)
                & contexts["linked_sample_context_count"].fillna(0).gt(0)
                & contexts["environmental_context_fields_present"].fillna(0).gt(0),
                contexts["tri_view_units"].gt(0) & contexts["linked_sample_context_count"].fillna(0).gt(0),
                contexts["esm2_units"].gt(0) & contexts["glm2_units"].gt(0),
            ],
            [
                "molecular_context_plus_environment_ready_for_abundance_linkage",
                "molecular_context_ready_needs_environment_or_abundance",
                "embedding_context_ready_needs_functional_completion",
            ],
            default="not_scoreable_payload_or_metadata_gap",
        )
        contexts["chart_label"] = contexts.apply(sample_context_chart_label, axis=1)
        contexts["lane_key"] = contexts["lane_id"].astype(str).map(
            {"msm_china_2025": "msm", "futian_mangrove_2026_qi": "futian"}
        ).fillna("mangrove")
        # Complete contexts first, by size; source-gap rows last.
        contexts = contexts.assign(_gap=contexts["tri_view_units"].eq(0)).sort_values(
            ["_gap", "units"], ascending=[True, False]
        ).drop(columns="_gap")
    status_defs = [
        {
            "id": "site_month_context",
            "label": "Futian site-month context",
            "meaning": "MAGs can be grouped to site and month with environmental sample metadata, but depth-resolved MAG-to-sample assignment is still blocked.",
        },
        {
            "id": "sample_set_context",
            "label": "MSM sample-set/BioSample context",
            "meaning": "MAGs can be grouped to source sample sets and BioSample sets, but individual MAG-to-sample assignment is unresolved.",
        },
        {
            "id": "mixed_wetland_context",
            "label": "Mixed wetland context",
            "meaning": "Some wetland/MUCC rows have BioSample or site/project context, but uniform sample rollup is not available.",
        },
        {
            "id": "reference_context",
            "label": "Rumen reference context",
            "meaning": "Reference-domain molecular comparison only; not a blue-carbon environmental sample.",
        },
    ]
    return {
        "groups": frame_records(lane_group, list(lane_group.columns)),
        "contexts": frame_records(contexts, list(contexts.columns), max_rows=120),
        "status_defs": status_defs,
    }


def build_payloads(
    atlas: pd.DataFrame,
    emb_meta: pd.DataFrame,
    edge_df: pd.DataFrame,
    cards: pd.DataFrame,
    summary: dict[str, Any],
    graph_node_cap: int,
    manifold_methods: list[dict[str, str]],
    scientific_audit: dict[str, Any],
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
        "nearest_poc_id",
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
        "source_group",
        "sample_context_key",
        "sample_context_label",
        "sample_context_resolution",
        "environmental_context_status",
        "sample_context_blocking_gap",
        "linked_sample_context_count",
        "environmental_context_fields_present",
        "sample_site_name",
        "sampling_month_iso",
        "mean_ph",
        "mean_salinity_psu",
        "mean_toc_mg_g",
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
        .sort_values(
            ["bridge_affinity_index", "qc_confidence_index"],
            ascending=False,
        )
        .head(140)["proteome_id"]
        .astype(str)
    )
    selected_ids = set(list(focus_ids)[:graph_node_cap])
    graph_coord_cols = [
        "source_category",
        "has_functional",
        "has_esm2",
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
        "lane_key",
        "source_display",
        "atlas_inclusion_status",
        "analysis_unit_type",
        "claim_scope",
        "review_tier",
        "plot_annotation_status",
        "functional_annotation_status",
        "functional_evidence_class",
        "functional_harmonization_status",
        "mechanism_equivalence_status",
        "functional_comparability_tier",
        "functional_numerator_provenance",
        "public_attestation_score_status",
        "glm2_protocol_class",
        "glm2_metric_comparability_status",
        "esm2_protocol_class",
        "esm2_protein_cap_status",
        "formal_tri_view_status",
        "has_functional",
        "has_glm2",
        "molecular_attestation_index",
        "legacy_noncomparable_attestation_index_quarantined",
        "source_scaffold_review_score",
        "bridge_affinity_index",
        "methane_marker_count",
        "sulfur_context_count",
        "substrate_breadth_count",
        "methane_marker_density_per_1k",
        "sulfur_context_density_per_1k",
        "substrate_breadth_per_1k",
        "methane_sulfur_balance",
        "nearest_poc_similarity",
        "nearest_poc_id",
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
        "source_group",
        "sample_context_key",
        "sample_context_label",
        "sample_context_resolution",
        "environmental_context_status",
        "sample_context_blocking_gap",
        "linked_sample_context_count",
        "environmental_context_fields_present",
        "sample_site_name",
        "sampling_month_iso",
        "mean_ph",
        "mean_salinity_psu",
        "mean_toc_mg_g",
        "allowed_claim_wording",
        "blocking_gap",
        "next_validation_action",
        "processed_gene_expression_support",
        "methane_expressed_gene_rows",
        "sulfur_expressed_gene_rows",
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
    card_meta = cards[["proteome_id", "candidate_set", "rank"]].rename(
        columns={"candidate_set": "case_study_set", "rank": "case_study_rank"}
    )
    niche_nodes = niche_nodes.merge(card_meta, on="proteome_id", how="left")
    niche_nodes["is_case_study"] = niche_nodes["case_study_set"].notna()
    niche_nodes["label"] = niche_nodes["proteome_id"].map(short_id)
    niche_node_ids = set(niche_nodes["proteome_id"].astype(str))
    niche_links = graph_edges[
        graph_edges["cross_domain"]
        & graph_edges["source"].astype(str).isin(niche_node_ids)
        & graph_edges["target"].astype(str).isin(niche_node_ids)
    ].head(2200).copy()
    niche_links["evidence_type"] = "cross_domain_knn"
    source_category_by_id = atlas.set_index("proteome_id")["source_category"].astype(str).to_dict()
    case_links: list[dict[str, Any]] = []
    for row in cards.itertuples(index=False):
        source_id = str(getattr(row, "proteome_id"))
        target_id = str(getattr(row, "nearest_poc_id", "") or "")
        if not target_id or source_id == target_id or source_id not in niche_node_ids or target_id not in niche_node_ids:
            continue
        case_links.append(
            {
                "source": source_id,
                "target": target_id,
                "source_category": source_category_by_id.get(source_id, ""),
                "target_category": source_category_by_id.get(target_id, ""),
                "similarity": safe_float(getattr(row, "nearest_poc_similarity", np.nan)),
                "cross_domain": source_category_by_id.get(source_id, "") != source_category_by_id.get(target_id, ""),
                "reciprocal": False,
                "rank": safe_int(getattr(row, "rank", 0)),
                "evidence_type": "case_study_nearest_poc",
            }
        )
    if case_links:
        niche_links = pd.concat([pd.DataFrame(case_links), niche_links], ignore_index=True, sort=False)
        niche_links = niche_links.drop_duplicates(["source", "target", "evidence_type"])

    completed_mangrove = atlas[atlas["source_category"].eq("mangrove") & atlas["has_functional"]].copy()
    mangrove_cols = [
        "proteome_id",
        "mag_id",
        "lane_id",
        "source_display",
        "source_group",
        "source_bucket",
        "atlas_inclusion_status",
        "analysis_unit_type",
        "claim_scope",
        "functional_annotation_status",
        "plot_annotation_status",
        "has_esm2",
        "has_glm2",
        "has_functional",
        "rate_metric_status",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "qc_tier",
        "review_tier",
        "checkm2_completeness",
        "checkm2_contamination",
        "gunc_pass",
        "methane_marker_count",
        "sulfur_context_count",
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
        "nearest_poc_id",
        "nearest_poc_similarity",
        "molecular_attestation_index",
        "legacy_noncomparable_attestation_index_quarantined",
        "functional_comparability_tier",
        "functional_numerator_provenance",
        "public_attestation_score_status",
        "glm2_protocol_class",
        "glm2_metric_comparability_status",
        "esm2_protocol_class",
        "esm2_protein_cap_status",
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
        "sample_context_key",
        "sample_context_label",
        "sample_context_resolution",
        "environmental_context_status",
        "sample_context_blocking_gap",
        "linked_sample_context_count",
        "environmental_context_fields_present",
        "mean_ph",
        "mean_salinity_psu",
        "mean_toc_mg_g",
        "allowed_claim_wording",
        "blocking_gap",
        "next_validation_action",
        "processed_gene_expression_support",
        "methane_expressed_gene_rows",
        "sulfur_expressed_gene_rows",
    ]
    return {
        "summary": summary,
        "scientific_audit": scientific_audit,
        "evidence_contract": scientific_audit.get("evidence_contract", []),
        "niche": {
            "methods": manifold_methods,
            "nodes": frame_records(niche_nodes, niche_cols + ["label", "case_study_set", "case_study_rank", "is_case_study"]),
            "case_study_count": int(niche_nodes["is_case_study"].sum()),
            "links": frame_records(
                niche_links,
                [
                    "source",
                    "target",
                    "source_category",
                    "target_category",
                    "similarity",
                    "cross_domain",
                    "reciprocal",
                    "rank",
                    "evidence_type",
                ],
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
                    "functional_evidence_class",
                    "functional_harmonization_status",
                    "mechanism_equivalence_status",
                    "formal_tri_view_status",
                    "plot_annotation_status",
                    "review_tier",
                    "molecular_attestation_index",
                    "source_scaffold_review_score",
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
        "mangrove": frame_records(
            completed_mangrove.sort_values(
                ["bridge_affinity_index", "qc_confidence_index"],
                ascending=False,
            ),
            mangrove_cols,
        ),
        "circos": build_candidate_circos(cards),
        "sample_linkage": build_sample_linkage_payload(atlas),
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
                "functional_evidence_class",
                "functional_harmonization_status",
                "mechanism_equivalence_status",
                "functional_comparability_tier",
                "functional_numerator_provenance",
                "public_attestation_score_status",
                "glm2_protocol_class",
                "glm2_metric_comparability_status",
                "esm2_protocol_class",
                "esm2_protein_cap_status",
                "formal_tri_view_status",
                "checkm2_completeness",
                "checkm2_contamination",
                "glm_context_delta",
                "nearest_poc_similarity",
                "nearest_poc_id",
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
                "legacy_noncomparable_attestation_index_quarantined",
                "source_scaffold_review_score",
                "source_paper_doi",
                "source_dataset_doi",
                "primary_accession",
                "primary_accession_type",
                "provenance_resolution_tier",
                "metadata_caveat",
                "sample_rollup_status",
                "sample_context_key",
                "sample_context_label",
                "sample_context_resolution",
                "environmental_context_status",
                "sample_context_blocking_gap",
                "next_metadata_action",
                "allowed_claim_wording",
                "blocking_gap",
                "next_validation_action",
                "processed_gene_expression_support",
                "methane_expressed_gene_rows",
                "sulfur_expressed_gene_rows",
            ],
        ),
    }


def plot_niche(payload: dict[str, Any], path: Path) -> Path:
    nodes = pd.DataFrame(payload["niche"]["nodes"])
    method = "umap" if {"umap_1", "umap_2"} <= set(nodes.columns) else "diffusion"
    xy = nodes[[f"{method}_1", f"{method}_2"]].apply(pd.to_numeric, errors="coerce")
    nodes = nodes[xy.notna().all(axis=1)]
    key_col = "lane_key" if "lane_key" in nodes.columns else "source_category"
    fig, ax = plt.subplots(figsize=(11, 7.3), facecolor=COLORS["surface"])
    ax.set_facecolor("#fbfdff")
    for key in ["rumen", "wetland", "msm", "futian", "mangrove"]:
        sub = nodes[nodes[key_col].astype(str).eq(key)]
        if sub.empty:
            continue
        ax.scatter(
            pd.to_numeric(sub[f"{method}_1"]),
            pd.to_numeric(sub[f"{method}_2"]),
            s=12,
            color=COLORS.get(key, "#94a3b8"),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.2,
            label=f"{source_label(key)} ({len(sub):,})",
        )
    name = {"umap": "UMAP", "diffusion": "Diffusion map"}[method]
    ax.set_title(
        f"Atlas map: {len(nodes):,} genome records, {name} projection of ESM-2 embeddings",
        loc="left", fontsize=13, weight="bold",
    )
    ax.set_xlabel(f"{name} 1 (unitless)")
    ax.set_ylabel(f"{name} 2 (unitless)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_evidence_contract(payload: dict[str, Any], path: Path) -> Path:
    df = pd.DataFrame(payload["evidence_contract"])
    fig, ax = plt.subplots(figsize=(11.2, 6.7), facecolor=COLORS["surface"])
    ax.set_facecolor("#fbfdff")
    if not df.empty:
        y = np.arange(len(df))
        height = 0.22
        series = [
            ("registered_units", "Registered", "#cbd5e1"),
            ("data_complete_tri_view_units", "All three views", "#0f766e"),
            (
                "mechanism_comparable_tri_view_units",
                "Comparable across routes",
                "#d89b14",
            ),
        ]
        # Match the interactive chart: complete-record bars show the annotation route.
        route_colors = np.where(
            df.get("functional_contract", pd.Series("", index=df.index)).astype(str).eq("source_scaffold_non_equivalent"),
            "#a16207",
            "#0f766e",
        )
        for offset, (column, label, color) in zip(
            [-height, 0, height], series
        ):
            values = pd.to_numeric(df[column], errors="coerce").fillna(0)
            bars = ax.barh(
                y + offset,
                values,
                height=height * 0.9,
                color=route_colors if column == "data_complete_tri_view_units" else color,
                edgecolor="#475569",
                linewidth=0.45,
                label=label,
            )
            ax.bar_label(
                bars,
                labels=[f"{int(value):,}" for value in values],
                padding=3,
                fontsize=8,
                color=COLORS["ink"],
            )
        ax.set_yticks(y, df["lane"])
        ax.invert_yaxis()
        ax.legend(
            handles=[
                matplotlib.patches.Patch(facecolor="#cbd5e1", edgecolor="#475569", label="Registered"),
                matplotlib.patches.Patch(facecolor="#0f766e", edgecolor="#475569", label="All three views, shared pipeline"),
                matplotlib.patches.Patch(facecolor="#a16207", edgecolor="#475569", label="All three views, source annotations"),
                matplotlib.patches.Patch(facecolor="#d89b14", edgecolor="#475569", label="Comparable across routes"),
            ],
            frameon=False, ncol=2, loc="lower right", fontsize=9,
        )
    ax.set_title("Genome records by source lane and evidence state", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("Genome records")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _pos: f"{int(value):,}"))
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.125,
        0.015,
        "All three views = ESM-2 + gLM2 + functional annotation. No record is yet comparable across annotation routes.",
        fontsize=9,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_matrix(payload: dict[str, Any], path: Path) -> Path:
    records = pd.DataFrame(payload["matrix"]["records"])
    if records.empty:
        path.touch()
        return path
    row_order = list(dict.fromkeys(records["label"]))
    mat = records.pivot_table(index="label", columns="metric", values="value", aggfunc="first").fillna(0)
    mat = mat.reindex(row_order)
    metric_defs = payload["matrix"].get("metric_defs") or [{"id": m, "label": m} for m in payload["matrix"]["metrics"]]
    metric_ids = [m["id"] for m in metric_defs]
    metric_labels = [m.get("label", m["id"]) for m in metric_defs]
    mat = mat[metric_ids]
    fig, ax = plt.subplots(figsize=(11.6, max(6.4, 0.38 * len(mat))), facecolor=COLORS["surface"])
    binary = matplotlib.colors.ListedColormap(["#e2e8f0", "#0f766e"])
    ax.imshow(mat.values, aspect="auto", vmin=0, vmax=1, cmap=binary)
    ax.set_xticks(np.arange(len(mat.columns)), metric_labels, fontsize=9)
    ax.set_yticks(np.arange(len(mat.index)), mat.index, fontsize=7.5)
    ax.set_title("Candidate evidence matrix (P = reference core, M = mangrove, O = Old Woman Creek)", loc="left", fontsize=12, weight="bold")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-0.5, len(mat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.legend(
        handles=[
            matplotlib.patches.Patch(color="#0f766e", label="Available or eligible"),
            matplotlib.patches.Patch(color="#e2e8f0", label="Not available"),
        ],
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_fallbacks(payload: dict[str, Any], figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "niche": plot_niche(payload, figure_dir / "fallback_01_molecular_niche_space.png"),
        "matrix": plot_matrix(payload, figure_dir / "fallback_02_candidate_signature_matrix.png"),
        "evidence_contract": plot_evidence_contract(
            payload,
            figure_dir / "fallback_03_evidence_contract.png",
        ),
    }


def save_payloads(payload: dict[str, Any], data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = json_safe(payload)
    out = {}
    for name, obj in payload.items():
        path = data_dir / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
        out[name] = path
    bundle_path = data_dir / "atlas_bundle.js"
    bundle_path.write_text(
        "window.METHANET_ATLAS = "
        + json.dumps(payload, ensure_ascii=False, allow_nan=False)
        + ";\n",
        encoding="utf-8",
    )
    out["atlas_bundle_js"] = bundle_path
    return out


def render_html(
    summary: dict[str, Any],
    payload: dict[str, Any],
    fallbacks: dict[str, Path],
    infographic: Path | None,
    sample_risk_abstract: Path,
    d3_path: Path,
    output_dir: Path,
    source_readiness: list[dict[str, Any]],
) -> str:
    fallback_uri = {k: asset_href(v, output_dir) for k, v in fallbacks.items()}
    infographic_uri = (
        asset_href(infographic, output_dir)
        if infographic is not None and infographic.exists()
        else ""
    )
    sample_risk_abstract_uri = (
        asset_href(sample_risk_abstract, output_dir)
        if RENDER_SAMPLE_RISK_ABSTRACT and sample_risk_abstract.is_file()
        else ""
    )
    d3_href = asset_href(d3_path, output_dir)
    # Keep the published report self-contained. The detailed payload files remain
    # in the internal report bundle, while the public HTML embeds the minimum
    # interactive state needed by its figures and candidate cards.
    atlas_payload_json = (
        json.dumps(public_report_payload(payload), ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    release_required_payload_total = int(
        summary.get("mangrove_release_required_payload_total", summary["mangrove_ready_payload_total"])
    )
    release_functional = int(summary.get("mangrove_release_functional", summary["msm_functional"]))
    release_multiview = int(summary.get("release_multiview_complete", summary["multiview_complete"]))
    release_pending = int(summary.get("mangrove_release_function_pending", summary["msm_function_pending"]))
    release_excluded = int(summary.get("mangrove_release_excluded_units", 0))
    release_note = ""
    if release_excluded:
        release_note = (
            f" This release denominator explicitly excludes {release_excluded:,} incomplete unit(s); "
            "the excluded rows remain in the freeze manifest, status tables, and report bundle rather than being dropped."
        )
    audit = payload.get("scientific_audit", {})
    evidence_contract = audit.get("evidence_contract", [])
    geometry = audit.get("embedding_geometry", {})
    nearest_core = audit.get("nearest_core_context", {})
    taxonomy_audit = audit.get("taxonomy", {})
    functional_audit = audit.get("functional_metric_provenance", {})
    mucc_audit = audit.get("mucc_validation_readiness", {})
    findings = audit.get("findings", [])
    mucc_exact = safe_int(mucc_audit.get("exact_sample_environment_flux_links"))
    mucc_columns = safe_int(mucc_audit.get("expression_sample_columns"))
    metric_cards = "\n".join(
        [
            f"<div class='metric'><b>{summary['atlas_registered_units']:,}</b><span>registered genome records, including {summary['explicit_non_runnable_gaps']:,} documented source gaps</span></div>",
            f"<div class='metric'><b>{release_multiview:,}</b><span>records with all three evidence views: ESM-2, gLM2 and functional annotation</span></div>",
            f"<div class='metric'><b>{summary['pipeline_normalized_tri_view_units']:,}</b><span>screened for methane-cycle genes through the shared annotation pipeline</span></div>",
            f"<div class='metric'><b>{summary['source_scaffold_tri_view']:,}</b><span>screened with the Old Woman Creek source annotations</span></div>",
            f"<div class='metric'><b>{summary['mechanism_comparable_tri_view']:,}</b><span>records with a methane mechanism score comparable across annotation routes</span></div>",
            f"<div class='metric'><b>{mucc_exact}/{mucc_columns}</b><span>Old Woman Creek sequencing samples with an exact sample, environment and flux join</span></div>",
        ]
    )
    release_ledger_cards = "\n".join(
        [
            f"<div class='metric'><b data-release-key='snapshot_date'>{summary['snapshot_date']}</b><span>release snapshot date</span></div>",
            f"<div class='metric'><b data-release-key='registered_units'>{summary['atlas_registered_units']:,}</b><span>registered genome records, including documented gaps</span></div>",
            f"<div class='metric'><b data-release-key='esm2_units'>{summary['release_esm2_units']:,}</b><span>ESM-2 protein embeddings</span></div>",
            f"<div class='metric'><b data-release-key='glm2_units'>{summary['release_glm2_units']:,}</b><span>gLM2 genomic-context embeddings, under two protocols</span></div>",
            f"<div class='metric'><b data-release-key='functional_payload_units'>{summary['release_functional_payload_units']:,}</b><span>functional annotation payloads, across both routes</span></div>",
            f"<div class='metric'><b data-release-key='release_required_units'>{summary['embedding_context_total']:,}</b><span>records the release requires</span></div>",
            f"<div class='metric'><b data-release-key='explicit_non_runnable_gaps'>{summary['explicit_non_runnable_gaps']:,}</b><span>documented source gaps that cannot be run</span></div>",
            f"<div class='metric'><b data-release-key='tri_view_ready_units'>{summary['release_multiview_complete']:,}</b><span>records with all three evidence views</span></div>",
            f"<div class='metric'><b data-release-key='schema_normalized_units'>{summary['schema_normalized_units']:,}</b><span>functional payloads in the shared schema</span></div>",
            f"<div class='metric'><b data-release-key='schema_normalized_tri_view_units'>{summary['schema_normalized_tri_view_units']:,}</b><span>three-view records in the shared schema</span></div>",
            f"<div class='metric'><b data-release-key='pipeline_normalized_tri_view_units'>{summary['pipeline_normalized_tri_view_units']:,}</b><span>three-view records from the shared pipeline</span></div>",
            f"<div class='metric'><b data-release-key='mechanism_comparable_units'>{summary['mechanism_comparable_tri_view']:,}</b><span>records comparable across annotation routes</span></div>",
            f"<div class='metric'><b data-release-key='annotation_complete_tri_view_units'>{summary['annotation_complete_harmonization_pending_tri_view']:,}</b><span>annotated records awaiting shared aggregation</span></div>",
            f"<div class='metric'><b data-release-key='source_scaffold_tri_view_units'>{summary['source_scaffold_tri_view']:,}</b><span>three-view records with source annotations</span></div>",
            f"<div class='metric'><b data-release-key='blocking_units'>{summary['blocking_units']:,}</b><span>unresolved release blockers</span></div>",
        ]
    )
    indexing = str(summary.get("indexing_decision", ""))
    indexing_text = "off (noindex)" if indexing.startswith("noindex") else html.escape(indexing.replace("_", " "))
    release_state = html.escape(str(summary.get("release_state", "")).replace("_", " "))
    try:
        snapshot_day = datetime.strptime(str(summary["snapshot_date"]), "%Y-%m-%d")
        snapshot_text = f"{snapshot_day.day} {snapshot_day.strftime('%B %Y')}"
    except ValueError:
        snapshot_text = html.escape(str(summary["snapshot_date"]))
    references = [
        ("Lin et al. 2023, Science: the ESM-2 protein language model", "https://www.science.org/doi/10.1126/science.ade2574"),
        ("Scientific Reports 2025: medium-sized protein language models in transfer learning", "https://www.nature.com/articles/s41598-025-05674-x"),
        ("Coifman and Lafon 2006: diffusion maps", "https://doi.org/10.1016/j.acha.2006.04.006"),
        ("McInnes, Healy and Melville 2018: UMAP", "https://arxiv.org/abs/1802.03426"),
        ("van der Maaten and Hinton 2008: t-SNE", "https://www.jmlr.org/papers/v9/vandermaaten08a.html"),
        ("Communications Biology 2022: evaluating dimension-reduction methods", "https://www.nature.com/articles/s42003-022-03628-x"),
        ("Old Woman Creek wetland microbiome study, mSystems", "https://journals.asm.org/doi/10.1128/msystems.00680-25"),
    ]
    references_html = "".join(
        f"<li><a href='{url}'>{html.escape(label)}</a></li>" for label, url in references
    )
    source_links = {
        "Stewart et al. 2019, Nature Biotechnology": "https://doi.org/10.1038/s41587-019-0202-3",
        "Bechtold et al. 2025, Nature Communications": "https://doi.org/10.1038/s41467-025-56133-0",
        "Pan et al. 2025, GigaScience": "https://doi.org/10.1093/gigascience/giaf081",
        "Qi et al. 2026, Scientific Data": "https://doi.org/10.1038/s41597-026-07291-3",
        "Borton et al. 2026, mSystems": "https://doi.org/10.1128/msystems.00680-25",
    }

    def source_cell(name: Any) -> str:
        text = html.escape(str(name))
        url = source_links.get(str(name))
        return f"<a href='{url}'>{text}</a>" if url else text

    functional_contract_labels = {
        "pipeline_normalized_comparability_pending": "Shared pipeline; cross-route comparison pending",
        "source_scaffold_non_equivalent": "Source annotations (DRAM terms, genes, expression); separate contract",
        "annotation_complete_harmonization_pending": "Annotated; shared aggregation pending",
        "canonical_mechanism_comparable": "Comparable mechanism features",
        "functional_incomplete": "No functional payload",
    }
    numerator_labels = {
        "accepted KOfam genes and present METABOLIC events; best-ranked MCycDB/SCycDB hits exposed separately": (
            "Accepted KOfam genes and METABOLIC events; best MCycDB and SCycDB hits kept separate"
        ),
        "source DRAM term rows and processed expression detection": (
            "Source DRAM terms, plus processed expression detection"
        ),
        "curated annotation bundle present; normalized cohort aggregation pending": (
            "Curated annotations; shared aggregation pending"
        ),
        "raw_many_to_many_annotation_hit_rows_plus_all_hmm_rows": "Raw annotation-hit rows plus all HMM rows",
        "source_dram_term_rows_and_processed_expression_detection": "Source DRAM terms, plus processed expression detection",
        "curated_accepted_or_present_mechanism_features": "Curated accepted or present mechanism features",
    }
    public_rate_labels = {
        "pipeline_normalized_screening_not_cross_lane_mechanism_rate": "Screening events only; no cross-route rate",
        "source_scaffold_term_density_non_equivalent": "Source-term density; not comparable across routes",
        "quarantined_raw_hit_row_numerator_not_marker_density": "Raw hit rows quarantined; not a marker density",
        "comparable_curated_feature_density": "Comparable curated-feature density",
    }
    provenance_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['lane']))}</td>"
        f"<td class='num'>{safe_int(row['report_units']):,}</td>"
        f"<td>{html.escape(str(row['metadata_universe']))}</td>"
        f"<td>{source_cell(row['primary_source'])}</td>"
        f"<td>{html.escape(str(row['resolution_now']))}</td>"
        f"<td>{html.escape(str(row['use_now']))}</td>"
        f"<td>{html.escape(str(row['blocking_gap']))}</td>"
        "</tr>"
        for row in source_readiness
    )
    evidence_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['lane']))}</td>"
        f"<td class='num'>{safe_int(row['registered_units']):,}</td>"
        f"<td class='num'>{safe_int(row['esm2_units']):,}</td>"
        f"<td class='num'>{safe_int(row['glm2_units']):,}</td>"
        f"<td class='num'>{safe_int(row['functional_payload_units']):,}</td>"
        f"<td class='num'>{safe_int(row['data_complete_tri_view_units']):,}</td>"
        f"<td class='num'>{safe_int(row['mechanism_comparable_tri_view_units']):,}</td>"
        f"<td>{html.escape(functional_contract_labels.get(str(row['functional_contract']), str(row['functional_contract']).replace('_', ' ')))}</td>"
        "</tr>"
        for row in evidence_contract
    )
    findings_rows_html = "\n".join(
        "<tr>"
        f"<td><span class='status-tag'>{html.escape(str(row['severity']))}</span></td>"
        f"<td><b>{html.escape(str(row['finding']))}</b></td>"
        f"<td>{html.escape(str(row['result']))}</td>"
        f"<td>{html.escape(str(row['report_action']))}</td>"
        "</tr>"
        for row in findings
    )
    functional_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['lane']))}</td>"
        f"<td class='num'>{safe_int(row['functional_units']):,}</td>"
        f"<td>{html.escape(numerator_labels.get(str(row['numerator_provenance']), str(row['numerator_provenance'])))}</td>"
        f"<td class='num'>{safe_float(row['raw_methane_row_count_median']):,.1f}</td>"
        f"<td class='num'>{safe_float(row['protein_count_median']):,.1f}</td>"
        f"<td class='num'>{100 * safe_float(row['raw_rows_per_protein_gt_1_share']):.1f}%</td>"
        f"<td>{html.escape(public_rate_labels.get(str(row['public_rate_metric_status']), str(row['public_rate_metric_status']).replace('_', ' ')))}</td>"
        "</tr>"
        for row in functional_audit.get("lane_metrics", [])
    )
    raw_pairs = geometry.get("raw_reciprocal_pair_counts", {}) or {}
    z_pairs = geometry.get("dimension_zscore_reciprocal_pair_counts", {}) or {}
    raw_rumen_wetland = safe_int(raw_pairs.get("rumen↔wetland"))
    raw_rumen_mangrove = safe_int(raw_pairs.get("mangrove↔rumen"))
    z_rumen_total = safe_int(z_pairs.get("rumen↔wetland")) + safe_int(z_pairs.get("mangrove↔rumen"))
    z_rumen_text = (
        "and no mutual pair involving a rumen genome remains"
        if z_rumen_total == 0
        else f"and {z_rumen_total:,} mutual pairs involving a rumen genome remain"
    )
    knn_k = safe_int(geometry.get("knn_k"))
    dimensions = safe_int(geometry.get("dimensions"), 1280)
    outside_cards = safe_int(nearest_core.get("target_candidate_cards_outside_core"))
    outside_cards_rumen = safe_int(nearest_core.get("target_candidate_outside_core_nearest_rumen_cards"))
    nearest_median = safe_float(nearest_core.get("outside_core_nearest_similarity_median"))
    random_median = safe_float(geometry.get("random_pair_similarity_median"))
    cross_edges = safe_int(geometry.get("raw_cross_domain_directed_edges"))
    drawn_neighbor_links = sum(
        1 for link in payload.get("niche", {}).get("links", []) if link.get("evidence_type") != "case_study_nearest_poc"
    )
    drawn_case_links = sum(
        1 for link in payload.get("niche", {}).get("links", []) if link.get("evidence_type") == "case_study_nearest_poc"
    )
    case_study_count = safe_int(payload.get("niche", {}).get("case_study_count"))
    candidates_rumen_text = (
        f"all {outside_cards:,} selected candidates outside the core"
        if outside_cards and outside_cards_rumen == outside_cards
        else f"{outside_cards_rumen:,} of the {outside_cards:,} selected candidates outside the core"
    )
    neighbor_link_rows = [
        link for link in payload.get("niche", {}).get("links", []) if link.get("evidence_type") != "case_study_nearest_poc"
    ]
    mangrove_wetland_only = bool(neighbor_link_rows) and all(
        {str(link.get("source_category")), str(link.get("target_category"))} == {"mangrove", "wetland"}
        for link in neighbor_link_rows
    )
    mutual_share = (
        sum(1 for link in neighbor_link_rows if link.get("reciprocal")) / len(neighbor_link_rows)
        if neighbor_link_rows else 0.0
    )
    neighbor_mix_text = (
        (" All of them join mangrove and wetland records" if mangrove_wetland_only else "")
        + (f", and {100 * mutual_share:.1f}% are mutual." if mangrove_wetland_only else f" {100 * mutual_share:.1f}% are mutual.")
    )
    top500_share = safe_float(functional_audit.get("legacy_top_500_mangrove_share"))
    top500_text = "all of its top 500 records were mangrove" if top500_share >= 0.9995 else f"{100 * top500_share:.1f}% of its top 500 records were mangrove"
    css = """
    :root{--ink:#16202a;--muted:#5b6b7c;--panel:#ffffff;--surface:#f6fafb;--line:#dbe5ee;--rumen:#db2777;--wetland:#65a30d;--mangrove:#0891b2;--futian:#6366f1;--gold:#d89b14;--teal:#0f766e}
    *{box-sizing:border-box} html,body{max-width:100%;overflow-x:hidden} body{margin:0;background:var(--surface);color:var(--ink);font-family:Inter,Aptos,Segoe UI,Arial,sans-serif;line-height:1.5}
    header{padding:58px 7vw 34px;background:linear-gradient(135deg,#071b25,#0d3c42 52%,#10523f);color:white}
    .eyebrow{letter-spacing:.12em;text-transform:uppercase;color:#b8f7d8;font-size:12px;font-weight:750}
    h1{font-size:clamp(34px,5vw,62px);line-height:1.02;margin:.35em 0 .25em;max-width:1120px}
    h2{font-size:26px;margin:0 0 12px;line-height:1.2} h3{font-size:18px;margin:18px 0 8px}
    .subtitle{max-width:980px;font-size:18px;color:#ddfff2}.claim{display:inline-block;margin-top:18px;border:1px solid #85dff1;padding:9px 14px;border-radius:14px;color:#e8fbff;max-width:980px}
    main{width:100%;max-width:1280px;margin:auto;padding:28px 24px 76px}.section{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 10px 30px rgba(15,23,42,.05)}
    .section>p{max-width:980px}
    .metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}.metric{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px}.metric b{display:block;font-size:28px;font-variant-numeric:tabular-nums}.metric span{font-size:12.5px;color:var(--muted)}
    .viz{min-height:540px;border:1px solid var(--line);border-radius:12px;background:#fbfdff;position:relative;overflow:hidden}.viz.tall{min-height:700px}.viz.medium{min-height:470px}.viz.graph{min-height:560px}.viz.graph>svg{width:100%;display:block}.viz.matrix{min-height:760px;overflow:auto}.viz.circos{min-height:620px}
    .viz svg{display:block;width:100%;height:auto}.viz.matrix svg,.viz.chart-scroll svg{width:auto;max-width:none}.viz.chart-scroll{overflow-x:auto}
    .grid2{display:grid;grid-template-columns:1.35fr .65fr;gap:18px}.grid2-even{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    .signature-stack{display:grid;grid-template-columns:1fr;gap:22px}.signature-panel{min-width:0}
    .figure-caption{font-size:13px;color:var(--muted);line-height:1.55;margin:8px 2px 18px;max-width:980px}
    .runtime-error{position:absolute;inset:18px;border:1px dashed #d89b14;border-radius:12px;background:#fff7ed;color:#7c2d12;padding:18px;font-size:14px;line-height:1.5}
    .approach-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}.approach-card{border:1px solid var(--line);border-radius:12px;background:#f8fafc;padding:14px}.approach-card b{display:block;margin-bottom:6px}.approach-card span{font-size:13px;color:var(--muted)}
    .decision-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}.decision-card{border:1px solid var(--line);border-radius:12px;background:#f8fafc;padding:15px}.decision-card b{display:block;margin-bottom:7px}.decision-card span{font-size:13px;color:var(--muted)}
    .layer-list{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.layer{border:1px solid var(--line);border-radius:12px;background:#f8fafc;padding:14px}.layer h3{font-size:15px;margin:10px 0 6px}.layer p{font-size:13px;color:var(--muted);margin:0 0 8px}.layer .now-line{color:var(--ink);margin:0}
    .state{display:inline-block;font-size:11px;font-weight:700;border-radius:999px;padding:3px 8px;border:1px solid}.state.partial{color:#92400e;border-color:#f3cf8b;background:#fff8eb}.state.missing{color:#475569;border-color:#cbd5e1;background:#f1f5f9}
    .side-card{border:1px solid var(--line);border-radius:12px;background:#f8fafc;padding:16px;min-height:540px;font-size:14px}.side-card .muted{font-size:12.5px;color:var(--muted)}.side-card h3{margin:4px 0 6px;overflow-wrap:anywhere}
    .card-facts{display:grid;grid-template-columns:minmax(0,1fr);gap:2px;margin:10px 0}.card-facts dt{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-top:8px}.card-facts dd{margin:0;overflow-wrap:anywhere}
    .tooltip{position:absolute;pointer-events:none;background:#0f172a;color:white;padding:8px 10px;border-radius:8px;font-size:12px;line-height:1.45;max-width:340px;opacity:0;z-index:20}
    .toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 6px}.toolbar button{border:1px solid var(--line);background:white;border-radius:999px;padding:7px 12px;cursor:pointer;font:inherit;font-size:14px}.toolbar button.active{background:var(--teal);color:white;border-color:var(--teal)}
    .legend{display:flex;gap:8px 16px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin:10px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}.halo{display:inline-block;width:11px;height:11px;border-radius:50%;border:1.6px solid var(--gold);margin-right:6px;vertical-align:-2px}.line{display:inline-block;width:18px;border-top:2px solid;margin-right:6px;vertical-align:middle}
    .fallback{margin-top:12px}.fallback img{max-width:100%;border:1px solid var(--line);border-radius:10px}.note{color:var(--muted)}.warn{background:#fff7ed;border-left:4px solid var(--gold);padding:12px;border-radius:10px;max-width:980px}
    .refs{margin:6px 0 0;padding-left:20px;color:var(--muted);font-size:13px;line-height:1.7}
    .readiness-table{display:block;width:100%;max-width:100%;overflow-x:auto;border-collapse:collapse;font-size:13px}.readiness-table th{background:#eef7f5;text-align:left}.readiness-table th,.readiness-table td{border:1px solid var(--line);padding:9px;vertical-align:top}.readiness-table td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
    .status-tag{display:inline-block;border:1px solid #cbd5e1;background:#f1f5f9;color:#334155;border-radius:999px;padding:3px 8px;font-size:11px;font-weight:700;white-space:nowrap}
    a{color:#075985} .closing{font-size:18px;line-height:1.58}
    @media (max-width:1020px){.metric-grid{grid-template-columns:repeat(2,1fr)}.approach-grid,.decision-grid,.layer-list{grid-template-columns:1fr 1fr}.grid2,.grid2-even{grid-template-columns:1fr}.viz.tall{min-height:520px}.side-card{min-height:0}}
    :focus-visible{outline:3px solid #0ea5e9;outline-offset:3px}
    @media (max-width:720px){header{padding:38px 18px 26px}main{padding:14px 10px 48px}.section{padding:16px 12px;margin:12px 0}.approach-grid,.decision-grid,.layer-list{grid-template-columns:1fr}.metric-grid{grid-template-columns:1fr}.viz{min-height:0}.viz.tall,.viz.medium,.viz.graph,.viz.circos{min-height:0}.viz.matrix{min-height:620px}.closing{font-size:16px}h2{font-size:22px}}
    @media (prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
    """
    js = """
    (function(){
    const ATLAS = window.METHANET_ATLAS;
    const COLORS = {rumen:'#db2777', wetland:'#65a30d', mangrove:'#0891b2', msm:'#0891b2', futian:'#6366f1', context:'#94a3b8', pending:'#d89b14'};
    const ROUTE_COLORS = {pipeline:'#0f766e', source:'#a16207'};
    const METHOD_NAMES = {umap:'UMAP', diffusion:'Diffusion map', phate:'PHATE', tsne:'t-SNE', pca:'PCA'};
    const PROJECTION_NOTES = {
      umap:'UMAP keeps local neighborhoods readable. Distances between far-apart clusters are not meaningful.',
      diffusion:'The diffusion map summarizes the cosine-neighbor graph after pooling reconciliation; habitat and source structure still require controls.',
      tsne:'t-SNE preserves local neighborhoods. Cluster sizes and the gaps between clusters are not meaningful.',
      pca:'PCA is a linear view of the largest directions of variance. It shows broad structure only.',
      phate:'PHATE emphasizes transitions between neighborhoods.'
    };
    const LABELS = {
      pipeline_normalized_comparability_pending:'shared pipeline; cross-route comparison pending',
      source_scaffold_non_equivalent:'source annotations; separate contract',
      functional_incomplete:'no functional payload',
      annotation_complete_harmonization_pending:'annotated; shared aggregation pending',
      paired_single_native_plus_single_shuffled:'single window (1 native, 1 shuffled)',
      multiwindow_10_native_plus_10_shuffled:'multi-window (10 native, 10 shuffled)',
      glm2_not_available:'not available',
      pass_review:'passes review',
      medium_low_completeness:'medium or low completeness',
      mapped_to_ncbi_biosample:'mapped to an NCBI BioSample',
      exact_mag_archive_qc_source_scaffold:'exact archive, QC and source annotations',
      exact_analysis_accession:'exact ENA analysis accession',
      site_month_habitat_context:'site, month and habitat context',
      exact_ncbi_assembly_biosample:'exact NCBI assembly and BioSample',
      'POC geometry-led review candidate':'Reference-core candidate (reconciled geometry)',
      'Mangrove geometry-led candidate; functional harmonization pending':'Mangrove candidate (embedding geometry and QC)',
      'MUCC v1 source-scaffold review candidate':'Old Woman Creek candidate (source annotations)',
      'accepted KOfam genes and present METABOLIC events; best-ranked MCycDB/SCycDB hits exposed separately':'accepted KOfam genes and METABOLIC events; best MCycDB and SCycDB hits kept separate',
      'source DRAM term rows and processed expression detection':'source DRAM terms, plus processed expression detection'
    };
    const fmt = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString('en-US') : String(v);
    const panelIds = ['mbag-knowledge-graph','niche-map','signature-matrix','candidate-circos','evidence-contract-chart','sample-linkage'];
    function renderFailure(err){
      console.error(err);
      panelIds.forEach(id => {
        const el = document.getElementById(id);
        if(!el){ return; }
        el.innerHTML = `<div class="runtime-error"><b>This interactive figure did not load.</b><br>${esc(err && err.message ? err.message : err)}<br><span>The static versions below each figure show the same data.</span></div>`;
      });
    }
    function requireAtlas(){
      if(!window.d3){ throw new Error('The D3 charting library is unavailable.'); }
      if(!ATLAS || !ATLAS.niche || !ATLAS.summary){ throw new Error('The embedded atlas data is unavailable.'); }
    }
    function panelWidth(el, minWidth=760){ return Math.max(minWidth, (el.node() && el.node().clientWidth) || minWidth); }
    function tooltip(){ return d3.select('body').append('div').attr('class','tooltip'); }
    function present(v){ return !(v === null || v === undefined || String(v).trim()==='' || ['NaN','None','null','undefined'].includes(String(v))); }
    function esc(v){ return String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function label(v){ if(!present(v)){ return ''; } const key=String(v); return LABELS[key] || key.replace(/_/g,' '); }
    // JSON null marks a source-lane gap, not a coordinate at the origin.
    function finiteNum(v){
      return v !== null && v !== undefined && typeof v !== 'boolean' &&
        (typeof v !== 'string' || v.trim() !== '') && Number.isFinite(Number(v));
    }
    function nodeColor(d){ return COLORS[d.lane_key] || COLORS[d.source_category] || COLORS.context; }
    function isMucc(d){ return String(d.source_display || '').indexOf('MUCC v1') >= 0; }
    function expressionSupported(d){ const v=String(d.processed_gene_expression_support).toLowerCase(); return ['true','1','yes'].includes(v) || Number(d.processed_gene_expression_support) > 0; }
    // An unassayed gene is not an absent gene: only the Old Woman Creek source carries expression data.
    function expressionText(d){
      if(expressionSupported(d) || isMucc(d)){
        return `methane ${fmt(Number(d.methane_expressed_gene_rows || 0))} rows, sulfur ${fmt(Number(d.sulfur_expressed_gene_rows || 0))} rows (processed source tables; detection, not activity)`;
      }
      return 'not assayed in this source';
    }
    function qcText(d){
      if(!finiteNum(d.checkm2_completeness)){ return ''; }
      const contamination = finiteNum(d.checkm2_contamination) ? `${Number(d.checkm2_contamination).toFixed(2)}% contamination` : 'contamination not reported';
      const tier = label(d.qc_tier);
      return `${Number(d.checkm2_completeness).toFixed(1)}% complete, ${contamination}${tier ? ' · ' + tier : ''}`;
    }
    function taxonomyText(d){
      const parts=[d.domain,d.phylum,d.class,d.order,d.family,d.genus,d.species].filter(v=>present(v) && String(v)!=='Unknown');
      return parts.length ? parts.join(' · ') : 'taxonomy not resolved';
    }
    function referenceText(d){
      if(!present(d.nearest_poc_id)){ return ''; }
      if(d.nearest_poc_id === d.proteome_id){ return 'this record is itself in the reference core'; }
      const sim = finiteNum(d.nearest_poc_similarity) ? ` (raw cosine ${Number(d.nearest_poc_similarity).toFixed(3)})` : '';
      return `${esc(d.nearest_poc_id)}${sim}`;
    }
    function row(name, value){ return present(value) ? `<span class="tip-k">${name}:</span> ${value}<br>` : ''; }
    function tipText(d){
      return `<b>${esc(d.proteome_id)}</b><br>`
        + row('Lane', esc(d.source_display || d.source_category))
        + row('Status', esc(d.plot_annotation_status || d.review_tier || ''))
        + row('Taxonomy', esc(taxonomyText(d)))
        + row('Functional route', esc(label(d.functional_comparability_tier)))
        + row('gLM2 protocol', esc(label(d.glm2_protocol_class)))
        + row('Closest reference-core genome', referenceText(d))
        + row('Expression', esc(expressionText(d)))
        + row('Genome quality', esc(qcText(d)));
    }
    const cardMap = new Map((ATLAS.cards || []).map(d => [d.proteome_id, d]));
    function fact(name, value){ return present(value) ? `<dt>${name}</dt><dd>${value}</dd>` : ''; }
    function updateCard(id){
      const d = cardMap.get(id) || (ATLAS.niche.nodes || []).find(x => x.proteome_id === id);
      const box = d3.select('#candidate-card');
      if(!d){ return; }
      const mechanism = d.mechanism_equivalence_status === 'mechanism_equivalent'
        ? 'available within the comparable contract'
        : 'not available: the annotation routes are not yet comparable';
      box.html(`<p class='muted'>${esc(label(d.candidate_set) || 'Atlas record')}</p>
        <h3>${esc(d.proteome_id)}</h3>
        <p class='muted'>${esc(d.source_display || d.source_category)} · ${esc(d.review_tier || d.plot_annotation_status || '')}<br>${esc(taxonomyText(d))}</p>
        <dl class='card-facts'>
          ${fact('Closest reference-core genome', referenceText(d))}
          ${fact('Functional route', esc(label(d.functional_comparability_tier)))}
          ${fact('What is counted', esc(label(d.functional_numerator_provenance)))}
          ${fact('Cross-route methane score', mechanism)}
          ${fact('gLM2 protocol', present(d.glm2_protocol_class) ? esc(label(d.glm2_protocol_class)) + '; compared within protocol only' : '')}
          ${fact('Expression', esc(expressionText(d)))}
          ${fact('Genome quality', esc(qcText(d)))}
          ${fact('Provenance', esc(label(d.provenance_resolution_tier)))}
        </dl>
        ${present(d.metadata_caveat) ? `<p class='muted'>${esc(d.metadata_caveat)}</p>` : ''}
        <p><b>Allowed claim.</b> ${esc(d.allowed_claim_wording || 'Genome-level molecular screening only.')}</p>
        <p class='muted'><b>Needed before any risk score:</b> ${esc(d.blocking_gap || 'sample mapping, abundance, environment, uncertainty and validation')}.${present(d.next_metadata_action) ? ' <b>Next:</b> ' + esc(d.next_metadata_action) : ''}</p>`);
    }
    function availableMethods(){
      const nodes = ATLAS.niche.nodes || [];
      return ['umap','diffusion','phate','tsne','pca'].filter(m =>
        nodes.some(node => finiteNum(node[`${m}_1`]) && finiteNum(node[`${m}_2`])));
    }
    function arrowDefs(svg){
      const defs=svg.append('defs');
      defs.append('marker').attr('id','mbag-arrow').attr('viewBox','0 -5 10 10').attr('refX',9).attr('refY',0).attr('markerWidth',6).attr('markerHeight',6).attr('orient','auto').append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#94a3b8');
      defs.append('marker').attr('id','mbag-arrow-gate').attr('viewBox','0 -5 10 10').attr('refX',9).attr('refY',0).attr('markerWidth',6).attr('markerHeight',6).attr('orient','auto').append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#d89b14');
    }
    const GRAPH_PALETTE={evidence:['#0f766e','#effaf8','#99d6cf'],guardrail:['#475569','#f7fafc','#cbd5e1'],core:['#0d3c42','#e7fff4','#9ee6c0'],output:['#4338ca','#eef2ff','#c7d2fe'],gate:['#a16207','#fff9e8','#efca7d']};
    function renderKnowledgeGraph(){
      const el=d3.select('#mbag-knowledge-graph'); el.selectAll('*').remove();
      const summary=ATLAS.summary || {}, audit=ATLAS.scientific_audit || {}, mucc=audit.mucc_validation_readiness || {};
      const glm2=(summary.glm2_single_window_units || 0) + (summary.glm2_multiwindow_units || 0);
      const available=(el.node() && el.node().clientWidth) || 1120;
      if(available < 760){ renderKnowledgeGraphNarrow(el, summary, mucc, glm2, available); return; }
      const w=Math.max(1060, available), h=690, margin=44;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('preserveAspectRatio','xMidYMid meet').attr('role','img').attr('aria-label','Evidence model for one genome record: evidence present today and the validation path still required');
      arrowDefs(svg);
      const leftW=270, rightW=300, coreW=290, gateW=Math.max(250, Math.min(300, (w-margin*2-64)/3));
      const gateGap=Math.max(32, Math.min(220, (w-margin*2-gateW*3)/2));
      const gateX=(w-gateW*3-gateGap*2)/2;
      const nodes=[
        {id:'esm2',x:margin,y:54,w:leftW,h:100,kind:'evidence',eyebrow:'protein embedding',lines:['ESM-2 neighborhoods'],meta:[`${fmt(summary.embedding_context_total || 0)} embedded records`]},
        {id:'function',x:margin,y:214,w:leftW,h:100,kind:'evidence',eyebrow:'functional annotation',lines:['Methane-cycle screening'],meta:[`${fmt(summary.release_multiview_complete || 0)} records, two routes`]},
        {id:'glm2',x:margin,y:374,w:leftW,h:100,kind:'evidence',eyebrow:'genomic context',lines:['gLM2 embeddings'],meta:[`${fmt(glm2)} records, two protocols`]},
        {id:'qc',x:w-margin-rightW,y:54,w:rightW,h:106,kind:'guardrail',eyebrow:'reliability checks',lines:['QC, taxonomy,', 'provenance'],meta:['claim limits travel with the evidence']},
        {id:'core',x:(w-coreW)/2,y:222,w:coreW,h:112,kind:'core',eyebrow:'evidence graph',lines:['Genome record'],meta:[`${fmt(summary.atlas_registered_units || 0)} registered records`]},
        {id:'card',x:w-margin-rightW,y:222,w:rightW,h:106,kind:'output',eyebrow:'decision output',lines:['Evidence card', 'and next action'],meta:['review, diligence, study design']},
        {id:'sample',x:gateX,y:530,w:gateW,h:96,kind:'gate',eyebrow:'still required',lines:['Exact sample links'],meta:[`${fmt(mucc.exact_sample_environment_flux_links || 0)} of ${fmt(mucc.expression_sample_columns || 0)} Old Woman Creek samples joined`]},
        {id:'context',x:gateX+gateW+gateGap,y:530,w:gateW,h:96,kind:'gate',eyebrow:'still required',lines:['Abundance and', 'environment'],meta:['who is present, under which conditions']},
        {id:'field',x:gateX+2*(gateW+gateGap),y:530,w:gateW,h:96,kind:'gate',eyebrow:'still required',lines:['Field or process', 'measurement'],meta:['flux, calibration, uncertainty']}
      ];
      const byId=new Map(nodes.map(d=>[d.id,d]));
      const links=[
        ['esm2','core','solid'],['function','core','solid'],['glm2','core','solid'],['qc','core','solid'],
        ['core','card','solid'],['core','sample','gate'],['sample','context','gate'],['context','field','gate']
      ];
      function center(node){ return [node.x+node.w/2,node.y+node.h/2]; }
      function edgePoint(from,to){
        const [fx,fy]=center(from), [tx,ty]=center(to), dx=tx-fx, dy=ty-fy;
        const sx=dx===0 ? Infinity : (from.w/2)/Math.abs(dx), sy=dy===0 ? Infinity : (from.h/2)/Math.abs(dy);
        const scale=Math.min(sx,sy);
        return [fx+dx*scale, fy+dy*scale];
      }
      svg.append('g').selectAll('line').data(links).join('line')
        .attr('x1',d=>edgePoint(byId.get(d[0]),byId.get(d[1]))[0]).attr('y1',d=>edgePoint(byId.get(d[0]),byId.get(d[1]))[1])
        .attr('x2',d=>edgePoint(byId.get(d[1]),byId.get(d[0]))[0]).attr('y2',d=>edgePoint(byId.get(d[1]),byId.get(d[0]))[1])
        .attr('stroke',d=>d[2]==='gate'?'#d89b14':'#8ca2b3').attr('stroke-width',d=>d[2]==='gate'?2.2:1.7)
        .attr('stroke-dasharray',d=>d[2]==='gate'?'7 6':null).attr('marker-end',d=>d[2]==='gate'?'url(#mbag-arrow-gate)':'url(#mbag-arrow)');
      const group=svg.append('g').selectAll('g.node').data(nodes).join('g').attr('class','node').attr('transform',d=>`translate(${d.x},${d.y})`);
      group.append('rect').attr('width',d=>d.w).attr('height',d=>d.h).attr('rx',14).attr('fill',d=>GRAPH_PALETTE[d.kind][1]).attr('stroke',d=>GRAPH_PALETTE[d.kind][2]).attr('stroke-width',1.3).attr('stroke-dasharray',d=>d.kind==='gate'?'6 4':null);
      group.append('rect').attr('width',5).attr('height',d=>d.h-20).attr('x',11).attr('y',10).attr('rx',3).attr('fill',d=>GRAPH_PALETTE[d.kind][0]);
      group.append('text').attr('x',27).attr('y',25).attr('font-size',10).attr('font-weight',800).attr('letter-spacing','.08em').attr('fill',d=>GRAPH_PALETTE[d.kind][0]).text(d=>d.eyebrow.toUpperCase());
      group.append('text').attr('x',27).attr('y',51).attr('font-size',15).attr('font-weight',800).attr('fill','#172033')
        .selectAll('tspan').data(d=>d.lines.map((line,i)=>({line,i}))).join('tspan').attr('x',27).attr('dy',d=>d.i===0?0:18).text(d=>d.line);
      group.append('text').attr('x',27).attr('y',d=>d.lines.length>1?88:74).attr('font-size',12).attr('fill','#475569')
        .selectAll('tspan').data(d=>d.meta.map((line,i)=>({line,i}))).join('tspan').attr('x',27).attr('dy',d=>d.i===0?0:15).text(d=>d.line);
      const legend=svg.append('g').attr('transform',`translate(${margin},${h-26})`);
      legend.append('line').attr('x1',0).attr('x2',28).attr('y1',0).attr('y2',0).attr('stroke','#8ca2b3').attr('stroke-width',1.7);
      legend.append('text').attr('x',36).attr('y',4).attr('font-size',12).attr('fill','#475569').text('evidence present today');
      legend.append('line').attr('x1',210).attr('x2',238).attr('y1',0).attr('y2',0).attr('stroke','#d89b14').attr('stroke-width',2.2).attr('stroke-dasharray','7 6');
      legend.append('text').attr('x',246).attr('y',4).attr('font-size',12).attr('fill','#475569').text('validation still required');
    }
    function renderKnowledgeGraphNarrow(el, summary, mucc, glm2, available){
      const w=Math.max(300, available), pad=12, boxW=w-pad*2, lineH=19;
      const blocks=[
        {kind:'evidence', title:'Evidence present today', lines:[`ESM-2 protein embeddings: ${fmt(summary.embedding_context_total || 0)}`, `Methane-cycle screening: ${fmt(summary.release_multiview_complete || 0)}`, `gLM2 genomic context: ${fmt(glm2)}`, 'QC, taxonomy and provenance checks']},
        {kind:'core', title:'Genome record', lines:[`${fmt(summary.atlas_registered_units || 0)} registered records`]},
        {kind:'output', title:'Evidence card and next action', lines:['review, diligence, study design']},
        {kind:'gate', title:'Validation still required', lines:[`Exact sample links: ${fmt(mucc.exact_sample_environment_flux_links || 0)} of ${fmt(mucc.expression_sample_columns || 0)} joined`, 'Abundance and environment', 'Field or process measurement']}
      ];
      let y=pad; const gap=34;
      blocks.forEach(b=>{ b.y=y; b.h=38+b.lines.length*lineH; y+=b.h+gap; });
      const h=y-gap+pad;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('role','img').attr('aria-label','Evidence model for one genome record: evidence present today and the validation path still required');
      arrowDefs(svg);
      blocks.slice(0,-1).forEach((b,i)=>{
        const next=blocks[i+1], gate=next.kind==='gate';
        svg.append('line').attr('x1',w/2).attr('x2',w/2).attr('y1',b.y+b.h+3).attr('y2',next.y-4)
          .attr('stroke',gate?'#d89b14':'#8ca2b3').attr('stroke-width',gate?2.2:1.7).attr('stroke-dasharray',gate?'7 6':null)
          .attr('marker-end',gate?'url(#mbag-arrow-gate)':'url(#mbag-arrow)');
      });
      const g=svg.append('g').selectAll('g').data(blocks).join('g').attr('transform',d=>`translate(${pad},${d.y})`);
      g.append('rect').attr('width',boxW).attr('height',d=>d.h).attr('rx',12).attr('fill',d=>GRAPH_PALETTE[d.kind][1]).attr('stroke',d=>GRAPH_PALETTE[d.kind][2]).attr('stroke-dasharray',d=>d.kind==='gate'?'6 4':null);
      g.append('text').attr('x',14).attr('y',25).attr('font-size',14).attr('font-weight',800).attr('fill',d=>GRAPH_PALETTE[d.kind][0]).text(d=>d.title);
      g.append('text').attr('x',14).attr('y',47).attr('font-size',13).attr('fill','#334155')
        .selectAll('tspan').data(d=>d.lines.map((line,i)=>({line,i}))).join('tspan').attr('x',14).attr('dy',d=>d.i===0?0:lineH).text(d=>d.line);
    }
    function renderNiche(method='umap'){
      const el=d3.select('#niche-map'); el.selectAll('*').remove();
      d3.selectAll('.tooltip.niche-tip').remove();
      const data=(ATLAS.niche.nodes || []), links=(ATLAS.niche.links || []);
      const w=panelWidth(el,560), h=w < 700 ? Math.round(w*1.08) : 720, m={t:40,r:22,b:50,l:52};
      const tip=tooltip().classed('niche-tip',true), name=METHOD_NAMES[method] || method;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('role','img').attr('aria-label',`${name} projection of ${fmt(data.length)} genome records colored by source lane, with candidate and neighbor links`);
      // A fixed pseudo-random draw order keeps any one lane from hiding another.
      const order=id=>{let hsh=0; for(let i=0;i<id.length;i++){hsh=(hsh*31+id.charCodeAt(i))|0;} return hsh;};
      const plotted=data.filter(d=>finiteNum(d[`${method}_1`]) && finiteNum(d[`${method}_2`])).sort((a,b)=>order(String(a.proteome_id))-order(String(b.proteome_id)));
      const x=d3.scaleLinear().domain(d3.extent(plotted,d=>+d[`${method}_1`])).nice().range([m.l,w-m.r]);
      const y=d3.scaleLinear().domain(d3.extent(plotted,d=>+d[`${method}_2`])).nice().range([h-m.b,m.t]);
      const byId=new Map(data.map(d=>[d.proteome_id,d]));
      const drawable=links.filter(d=>byId.has(d.source)&&byId.has(d.target)).filter(d=>[byId.get(d.source),byId.get(d.target)].every(node=>
          finiteNum(node[`${method}_1`]) && finiteNum(node[`${method}_2`])));
      const neighborLinks=drawable.filter(d=>d.evidence_type!=='case_study_nearest_poc');
      const caseLinks=drawable.filter(d=>d.evidence_type==='case_study_nearest_poc');
      const px=d=>x(+d[`${method}_1`]), py=d=>y(+d[`${method}_2`]);
      svg.append('g').selectAll('line').data(neighborLinks).join('line')
        .attr('x1',d=>px(byId.get(d.source))).attr('y1',d=>py(byId.get(d.source)))
        .attr('x2',d=>px(byId.get(d.target))).attr('y2',d=>py(byId.get(d.target)))
        .attr('stroke',d=>d.reciprocal?'#0f766e':'#94a3b8').attr('stroke-width',d=>d.reciprocal?.9:.5).attr('opacity',d=>d.reciprocal?.34:.2);
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x).ticks(5)).call(g=>g.selectAll('text').attr('fill','#64748b'));
      svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).call(g=>g.selectAll('text').attr('fill','#64748b'));
      svg.append('text').attr('x',(m.l+w-m.r)/2).attr('y',h-12).attr('text-anchor','middle').attr('font-size',12).attr('fill','#475569').text(`${name} 1 (unitless)`);
      svg.append('text').attr('transform','rotate(-90)').attr('x',-(m.t+h-m.b)/2).attr('y',14).attr('text-anchor','middle').attr('font-size',12).attr('fill','#475569').text(`${name} 2 (unitless)`);
      svg.append('text').attr('x',m.l).attr('y',24).attr('font-size',13).attr('font-weight',700).attr('fill','#172033')
        .text(`${fmt(plotted.length)} genome records · ${name}`);
      svg.append('g').selectAll('circle').data(plotted).join('circle')
        .attr('cx',px).attr('cy',py).attr('r',2.6)
        .attr('fill',nodeColor).attr('stroke','white').attr('stroke-width',.3).attr('opacity',.72).style('cursor','pointer')
        .attr('tabindex',d=>d.is_case_study?0:null).attr('role',d=>d.is_case_study?'button':null)
        .attr('aria-label',d=>d.is_case_study?`Show the evidence card for candidate ${d.proteome_id}`:null)
        .on('mouseover',(e,d)=>tip.style('opacity',1).html(tipText(d))).on('mousemove',e=>tip.style('left',`${e.pageX+12}px`).style('top',`${e.pageY+12}px`)).on('mouseout',()=>tip.style('opacity',0)).on('click',(e,d)=>updateCard(d.proteome_id))
        .on('keydown',(e,d)=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();updateCard(d.proteome_id);}});
      svg.append('g').selectAll('line').data(caseLinks).join('line')
        .attr('x1',d=>px(byId.get(d.source))).attr('y1',d=>py(byId.get(d.source)))
        .attr('x2',d=>px(byId.get(d.target))).attr('y2',d=>py(byId.get(d.target)))
        .attr('stroke','#d89b14').attr('stroke-width',1.5).attr('opacity',.8).style('pointer-events','none');
      svg.append('g').selectAll('circle.case-halo').data(plotted.filter(d=>d.is_case_study)).join('circle')
        .attr('class','case-halo').attr('cx',px).attr('cy',py)
        .attr('r',8).attr('fill','none').attr('stroke','#d89b14').attr('stroke-width',1.5).attr('opacity',.85).style('pointer-events','none');
    }
    function renderMethodButtons(){
      const methods=availableMethods(), box=d3.select('#method-buttons');
      const note=d3.select('#projection-note');
      box.attr('role','group').attr('aria-label','Map projection');
      function choose(method){ note.text(PROJECTION_NOTES[method] || ''); renderNiche(method); }
      box.selectAll('button').data(methods).join('button').attr('type','button')
        .attr('class',(d,i)=>i===0?'active':null).attr('aria-pressed',(d,i)=>i===0?'true':'false')
        .text(d=>METHOD_NAMES[d] || d)
        .on('click',function(e,d){box.selectAll('button').classed('active',false).attr('aria-pressed','false'); d3.select(this).classed('active',true).attr('aria-pressed','true'); choose(d);});
      choose(methods[0] || 'umap');
    }
    function renderMatrix(){
      const rec=ATLAS.matrix.records || [], metricDefs=ATLAS.matrix.metric_defs || (ATLAS.matrix.metrics || []).map(d=>({id:d,label:d})), metrics=metricDefs.map(d=>d.id);
      const rows=Array.from(new Set(rec.map(d=>d.label))), el=d3.select('#signature-matrix'); el.selectAll('*').remove();
      const w=panelWidth(el,980), cellH=28, m={t:78,r:24,b:24,l:330}, h=m.t+rows.length*cellH+m.b;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('width',w).attr('height',h).attr('role','img').attr('aria-label','Candidate evidence matrix: which evidence each candidate has');
      const x=d3.scaleBand().domain(metrics).range([m.l,w-m.r]).padding(.08), y=d3.scaleBand().domain(rows).range([m.t,h-m.b]).padding(.08);
      const on='#0f766e', off='#e2e8f0', rowMeta=new Map(rec.map(d=>[d.label,d]));
      const yes=d=>Number(d.value)>=1;
      svg.append('g').selectAll('rect.row').data(rows).join('rect').attr('class','row').attr('x',m.l-10).attr('y',d=>y(d)).attr('width',w-m.l-m.r+10).attr('height',y.bandwidth()).attr('fill',(d,i)=>i%2?'#f8fafc':'#ffffff');
      svg.selectAll('rect.cell').data(rec).join('rect').attr('class','cell').attr('x',d=>x(d.metric)).attr('y',d=>y(d.label)).attr('width',x.bandwidth()).attr('height',y.bandwidth()).attr('rx',4).attr('fill',d=>yes(d)?on:off).attr('stroke','white').attr('stroke-width',1.6).style('cursor','pointer').attr('tabindex',0).attr('role','button').attr('aria-label',d=>`${d.label}; ${d.metric_label}: ${yes(d)?'available':'not available'}`).on('click',(e,d)=>updateCard(d.proteome_id)).on('keydown',(e,d)=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();updateCard(d.proteome_id);}}).append('title').text(d=>`${d.label}\\n${d.metric_label}: ${yes(d)?'available or eligible':'not available'}\\n${d.metric_source || ''}`);
      svg.append('g').selectAll('rect.strip').data(rows).join('rect').attr('class','strip').attr('x',m.l-24).attr('y',d=>y(d)).attr('width',9).attr('height',y.bandwidth()).attr('rx',4).attr('fill',d=>nodeColor(rowMeta.get(d)||{}));
      svg.append('g').attr('transform',`translate(0,${m.t-10})`).call(d3.axisTop(x).tickFormat(d=>((metricDefs.find(md=>md.id===d)||{}).label)||d)).call(g=>g.select('.domain').remove()).selectAll('text').attr('font-weight',700).attr('font-size',12.5).attr('fill','#172033');
      svg.append('g').attr('transform',`translate(${m.l-30},0)`).call(d3.axisLeft(y)).call(g=>g.select('.domain').remove()).call(g=>g.selectAll('.tick line').remove()).selectAll('text').attr('font-size',12).attr('fill','#334155');
      const legend=svg.append('g').attr('transform',`translate(${m.l},18)`);
      [[on,'available or eligible'],[off,'not available']].forEach((item,i)=>{
        const g=legend.append('g').attr('transform',`translate(${i*190},0)`);
        g.append('rect').attr('width',14).attr('height',14).attr('rx',3).attr('fill',item[0]).attr('stroke','#cbd5e1');
        g.append('text').attr('x',20).attr('y',11).attr('font-size',12).attr('fill','#334155').text(item[1]);
      });
    }
    function renderEvidenceContract(){
      const data=ATLAS.evidence_contract || [], el=d3.select('#evidence-contract-chart');
      el.selectAll('*').remove();
      if(!data.length){ el.append('div').attr('class','runtime-error').html('The evidence-contract summary is unavailable.'); return; }
      const w=panelWidth(el,760), m={t:92,r:70,b:96,l:250}, bandH=78, h=m.t+data.length*bandH+m.b;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('width',w).attr('height',h).attr('role','img').attr('aria-label','Genome records by source lane: registered, with all three evidence views, and comparable across annotation routes');
      const route=d=>d.functional_contract==='source_scaffold_non_equivalent'?'source':'pipeline';
      const y0=d3.scaleBand().domain(data.map(d=>d.lane)).range([m.t,h-m.b]).padding(.22);
      const y1=d3.scaleBand().domain(['registered','complete']).range([0,y0.bandwidth()]).padding(.14);
      const x=d3.scaleLinear().domain([0,d3.max(data,d=>d.registered_units)||1]).nice().range([m.l,w-m.r]);
      const comparable=d3.sum(data,d=>Number(d.mechanism_comparable_tri_view_units||0)), complete=d3.sum(data,d=>Number(d.data_complete_tri_view_units||0));
      svg.append('text').attr('x',16).attr('y',26).attr('font-weight',800).attr('font-size',14).attr('fill','#172033').text('Genome records by source lane and evidence state');
      svg.append('text').attr('x',16).attr('y',46).attr('font-size',12).attr('fill','#64748b').text('All three views = ESM-2 + gLM2 + functional annotation. Bar color shows the annotation route.');
      svg.append('text').attr('x',16).attr('y',64).attr('font-size',12).attr('fill','#64748b').text(`Comparable across routes: ${fmt(comparable)} of ${fmt(complete)} complete records.`);
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(',')));
      svg.append('text').attr('x',(m.l+w-m.r)/2).attr('y',h-m.b+38).attr('text-anchor','middle').attr('font-size',12).attr('fill','#475569').text('Genome records');
      svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y0)).call(g=>g.select('.domain').remove()).call(g=>g.selectAll('text').attr('font-size',12).attr('fill','#172033'));
      const rows=svg.append('g').selectAll('g').data(data).join('g').attr('transform',d=>`translate(0,${y0(d.lane)})`);
      const bars=d=>[{key:'registered',value:Number(d.registered_units||0),color:'#cbd5e1'},{key:'complete',value:Number(d.data_complete_tri_view_units||0),color:ROUTE_COLORS[route(d)]}];
      rows.selectAll('rect').data(bars).join('rect').attr('x',m.l).attr('y',d=>y1(d.key)).attr('width',d=>Math.max(0,x(d.value)-m.l)).attr('height',y1.bandwidth()).attr('rx',4).attr('fill',d=>d.color);
      rows.selectAll('text.value').data(bars).join('text').attr('class','value').attr('x',d=>x(d.value)+5).attr('y',d=>y1(d.key)+y1.bandwidth()/2+4).attr('font-size',11).attr('fill','#334155').text(d=>fmt(d.value));
      const legend=svg.append('g').attr('transform',`translate(${m.l},${h-30})`);
      [['#cbd5e1','Registered'],[ROUTE_COLORS.pipeline,'All three views, shared pipeline'],[ROUTE_COLORS.source,'All three views, source annotations']].forEach((item,i)=>{
        const g=legend.append('g').attr('transform',`translate(${[0,110,360][i]},0)`);
        g.append('rect').attr('width',12).attr('height',12).attr('rx',3).attr('fill',item[0]);
        g.append('text').attr('x',18).attr('y',10).attr('font-size',11.5).attr('fill','#334155').text(item[1]);
      });
    }
    function renderSampleLinkage(){
      const contexts=((ATLAS.sample_linkage || {}).contexts || []);
      const el=d3.select('#sample-linkage'); el.selectAll('*').remove();
      d3.selectAll('.tooltip.sample-tip').remove();
      if(!contexts.length){ el.append('p').attr('class','note').style('padding','18px').text('No sample-context groups are available yet.'); return; }
      const w=panelWidth(el,760), rowH=25, m={t:92,r:60,b:92,l:290}, h=m.t+contexts.length*rowH+m.b, tip=tooltip().classed('sample-tip',true);
      const labelOf=d=>d.chart_label || d.sample_context_label;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('width',w).attr('height',h).attr('role','img').attr('aria-label','Mangrove genome records grouped by their best available sample context');
      const x=d3.scaleLinear().domain([0,d3.max(contexts,d=>d.units||0)||1]).nice().range([m.l,w-m.r]);
      const y=d3.scaleBand().domain(contexts.map(labelOf)).range([m.t,h-m.b]).padding(.2);
      const gap=d=>Number(d.tri_view_units||0)===0;
      const colorOf=d=>gap(d)?'#cbd5e1':(COLORS[d.lane_key]||COLORS.mangrove);
      const complete=contexts.filter(d=>!gap(d) && Number(d.tri_view_units)===Number(d.units)).length;
      svg.append('text').attr('x',16).attr('y',26).attr('font-weight',800).attr('font-size',14).attr('fill','#172033').text('Mangrove records by best available sample context');
      svg.append('text').attr('x',16).attr('y',46).attr('font-size',12).attr('fill','#64748b').text('Futian: site and month, several depths each. MSM: source sample groups.');
      svg.append('text').attr('x',16).attr('y',64).attr('font-size',12).attr('fill','#64748b').text(`No record is yet assigned to one physical sample. ${complete} of ${contexts.length} groups are complete in all three views.`);
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(','))).call(g=>g.selectAll('text').attr('font-size',11));
      svg.append('text').attr('x',(m.l+w-m.r)/2).attr('y',h-m.b+36).attr('text-anchor','middle').attr('font-size',12).attr('fill','#475569').text('Genome records');
      svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y)).call(g=>g.select('.domain').remove()).call(g=>g.selectAll('.tick text').attr('font-size',11).attr('fill','#334155'));
      svg.selectAll('rect.context').data(contexts).join('rect').attr('class','context')
        .attr('x',m.l).attr('y',d=>y(labelOf(d))).attr('width',d=>Math.max(1,x(d.units||0)-m.l)).attr('height',y.bandwidth()).attr('rx',4)
        .attr('fill',colorOf).attr('opacity',.88)
        .on('mouseover',(e,d)=>tip.style('opacity',1).html(`<b>${esc(d.sample_context_label || labelOf(d))}</b><br>${fmt(d.units||0)} records · ${fmt(d.tri_view_units||0)} with all three views<br>${fmt(d.linked_sample_context_count||0)} linked sample contexts · ${fmt(d.environmental_context_fields_present||0)} environmental fields<br>${esc(d.sample_context_blocking_gap || '')}<br><b>Readiness only; no risk score.</b>`))
        .on('mousemove',e=>tip.style('left',`${e.pageX+12}px`).style('top',`${e.pageY+12}px`)).on('mouseout',()=>tip.style('opacity',0));
      svg.selectAll('text.count').data(contexts).join('text').attr('class','count').attr('x',d=>x(d.units||0)+5).attr('y',d=>y(labelOf(d))+y.bandwidth()/2+4).attr('font-size',11).attr('fill','#334155').text(d=>fmt(d.units||0));
      const legend=svg.append('g').attr('transform',`translate(${m.l},${h-34})`);
      [[COLORS.futian,'Futian site and month'],[COLORS.msm,'MSM sample group'],['#cbd5e1','Source gaps']].forEach((item,i)=>{
        const g=legend.append('g').attr('transform',`translate(${i*185},0)`);
        g.append('rect').attr('width',12).attr('height',12).attr('rx',3).attr('fill',item[0]);
        g.append('text').attr('x',18).attr('y',10).attr('font-size',11.5).attr('fill','#334155').text(item[1]);
      });
    }
    function renderCircos(){
      const data=ATLAS.circos || {}, records=data.records || [], pillars=data.pillars || [], groups=data.groups || [];
      const el=d3.select('#candidate-circos'); el.selectAll('*').remove();
      const w=panelWidth(el,340), outer=Math.max(96, Math.min(210, w/2-104)), inner=outer*0.41, cx=w/2, cy=outer+56, h=cy+outer+56+groups.length*20+18;
      const ringStep=(outer-inner)/Math.max(groups.length,1), labelR=outer+30;
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]).attr('role','img').attr('aria-label','Evidence coverage by candidate group: the share of cards in each group with each type of evidence');
      const g=svg.append('g').attr('transform',`translate(${cx},${cy})`);
      const angle=d3.scaleBand().domain(pillars.map(d=>d.id)).range([0,Math.PI*2]).padding(.16);
      const arc=d3.arc(); const groupColor=new Map(groups.map(d=>[d.id,d.color]));
      groups.forEach((grp,gi)=>{
        const r0=inner+gi*ringStep+ringStep*0.08, r1=r0+ringStep*0.84;
        g.append('circle').attr('r',r0).attr('fill','none').attr('stroke','#dbe5ee').attr('stroke-dasharray','2 4');
        g.append('circle').attr('r',r1).attr('fill','none').attr('stroke','#edf3f7');
      });
      g.selectAll('path.bar').data(records.filter(d=>Number(d.average_value)>0)).join('path').attr('class','bar')
        .attr('d',d=>{const gi=groups.findIndex(gr=>gr.id===d.group); const base=inner+gi*ringStep+ringStep*0.12; const maxLen=ringStep*0.76; return arc({innerRadius:base, outerRadius:base+maxLen*d.average_value, startAngle:angle(d.pillar), endAngle:angle(d.pillar)+angle.bandwidth()});})
        .attr('fill',d=>groupColor.get(d.group) || '#64748b').attr('opacity',.85).attr('stroke','white').attr('stroke-width',.8)
        .append('title').text(d=>`${d.group_label} · ${d.pillar_label}\\n${d.high_count} of ${d.candidate_count} cards (${Math.round(100*d.average_value)}%)\\n${d.source}`);
      g.selectAll('text.pillar').data(pillars).join('text').attr('class','pillar').attr('font-size',11.5).attr('font-weight',700).attr('fill','#334155')
        .attr('x',d=>{const a=angle(d.id)+angle.bandwidth()/2-Math.PI/2; return Math.cos(a)*labelR;})
        .attr('y',d=>{const a=angle(d.id)+angle.bandwidth()/2-Math.PI/2; return Math.sin(a)*labelR+4;})
        .attr('text-anchor',d=>{const a=angle(d.id)+angle.bandwidth()/2-Math.PI/2; const c=Math.cos(a); return Math.abs(c)<.18?'middle':c>0?'start':'end';})
        .text(d=>d.short);
      // The center label needs room; on small screens the caption carries it.
      if(inner >= 70){
        g.append('circle').attr('r',inner-22).attr('fill','#f8fafc').attr('stroke','#dbe5ee');
        g.append('text').attr('text-anchor','middle').attr('y',-4).attr('font-weight',800).attr('font-size',12.5).text('Share of cards');
        g.append('text').attr('text-anchor','middle').attr('y',14).attr('font-size',11).attr('fill','#64748b').text('with the evidence');
      }
      const counts=new Map(); records.forEach(r=>counts.set(r.group, r.candidate_count));
      const legend=svg.append('g').attr('transform',`translate(${Math.max(12,cx-200)},${cy+outer+50})`);
      groups.forEach((grp,i)=>{
        const row=legend.append('g').attr('transform',`translate(0,${i*20})`);
        row.append('rect').attr('width',12).attr('height',12).attr('rx',3).attr('fill',grp.color);
        row.append('text').attr('x',18).attr('y',10).attr('font-size',12).attr('fill','#334155').text(`${i===0?'Inner':i===groups.length-1?'Outer':'Middle'} ring: ${grp.label}, ${fmt(counts.get(grp.id)||0)} cards`);
      });
    }
    function startReport(){
      requireAtlas();
      renderKnowledgeGraph();
      renderMethodButtons();
      renderMatrix();
      renderEvidenceContract();
      renderCircos();
      renderSampleLinkage();
      updateCard((ATLAS.cards[0]||{}).proteome_id);
    }
    try { startReport(); } catch(err) { renderFailure(err); }
    })();
    """
    infographic_block = ""
    if infographic_uri:
        infographic_block = f"""
        <section class="section">
          <h2>The Operating Model Behind The Atlas</h2>
          <img class="infographic" src="{infographic_uri}" alt="Molecular intelligence research workflow infographic">
        </section>
        """

    sample_risk_abstract_block = ""
    if sample_risk_abstract_uri:
        sample_risk_abstract_block = f"""
        <img class="sample-risk-abstract" src="{sample_risk_abstract_uri}" alt="Illustrative concept art for the path from genome evidence to sample-risk readiness; not data.">
        <p class="figure-caption">Illustrative concept art, not data.</p>
        """

    wetland_outside = safe_int(nearest_core.get("wetland_outside_core_units"))
    wetland_outside_rumen = safe_int(nearest_core.get("wetland_outside_core_nearest_rumen_units"))
    mangrove_units = safe_int(nearest_core.get("mangrove_embedding_units"))
    mangrove_rumen = safe_int(nearest_core.get("mangrove_nearest_rumen_units"))
    mucc_methane = safe_int(mucc_audit.get("methane_expression_detected_mags"))
    mucc_sulfur = safe_int(mucc_audit.get("sulfur_expression_detected_mags"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>EmergentBiome Molecular Atlas | Technical evidence report</title>
  <style>{css}</style>
</head>
<body>
<header>
  <div class="eyebrow">EmergentBiome · Technical evidence report</div>
  <h1>EmergentBiome Molecular Atlas</h1>
  <p class="subtitle">Methods, evidence states and limits of the {snapshot_text} atlas release: {release_multiview:,} genome records screened for methane-cycle genes, and the validation each claim still needs.</p>
  <div class="claim">{html.escape(CLAIM_BOUNDARY)}</div>
</header>
<main>
  <section class="section">
    <h2>Executive summary</h2>
    <p><b>The atlas is a frozen, source-audited map of microbial genomes from methane-relevant environments.</b> It registers {summary['atlas_registered_units']:,} genome records (metagenome-assembled genomes and their proteomes) from a rumen reference set, freshwater wetlands and mangrove sediments. {release_multiview:,} carry all three evidence views: an ESM-2 protein embedding, a gLM2 genomic-context embedding and a functional annotation. The remaining {summary['explicit_non_runnable_gaps']:,} are documented source gaps and stay in the release as explicit rows.</p>
    <p>Every record was screened for methane-cycle genes through one of two annotation routes. {summary['pipeline_normalized_tri_view_units']:,} records went through a shared pipeline (accepted KOfam genes and METABOLIC events, with the best MCycDB and SCycDB hits kept separate). {summary['source_scaffold_tri_view']:,} Old Woman Creek wetland records keep the source's own DRAM annotations and processed expression data. Because the two routes count genes differently, the number of records with a methane mechanism score comparable across routes is {summary['mechanism_comparable_tri_view']:,}.</p>
    <p><b>What the atlas supports today:</b> genome-level screening, candidate review with source and quality context, and measurement planning. <b>What it does not support yet:</b> measured methane flux, calibrated site risk, A–E risk tiers or carbon-credit decisions. Those need exact sample links, abundance, environmental covariates, uncertainty and paired flux or process measurements, and none of these is joined to the atlas records yet.</p>
    <div class="metric-grid">{metric_cards}</div>
  </section>
  <section class="section">
    <h2>Verified release snapshot</h2>
    <p>Each figure below is rendered from the machine-checked release ledger; the build stops if any count disagrees with it. Payload availability, shared-schema normalization, comparability across annotation routes and field validation are separate rungs, so their counts differ by design.</p>
    <div class="metric-grid">{release_ledger_cards}</div>
    <p class="note">Release state: <b>{release_state}</b>. Search indexing: <b>{indexing_text}</b>.</p>
  </section>
  <section class="section">
    <h2>The EmergentBiome evidence graph</h2>
    <p>Each genome record sits at the center of an evidence model. Embeddings, functional annotations, genomic context, quality checks and validation requirements attach to it as separate, typed relationships. A neighbor in embedding space is therefore never mistaken for a measured function, and a missing measurement stays visible as a gap.</p>
    <p>The queryable implementation covers a 662-record proof of concept. A formal ontology slice, added on 26 September 2026, holds selected evidence for 145 atlas records and is described with the evidence cases on the landing page. Neither yet spans the full atlas.</p>
    <div id="mbag-knowledge-graph" class="viz graph"></div>
    <p class="figure-caption">Evidence model for one genome record. Solid arrows: evidence present today, feeding an evidence card. Dashed amber arrows: the validation path still required, from exact sample links to abundance and environment, then to field or process measurement. The diagram shows structure, not causation.</p>
    <div class="decision-grid">
      <div class="decision-card"><b>Candidate review</b><span>See each candidate's evidence with its source, quality, annotation route and claim limit.</span></div>
      <div class="decision-card"><b>Measurement planning</b><span>Find which missing link, whether sample identity, abundance, environment or flux, would most change a decision.</span></div>
      <div class="decision-card"><b>Validation priorities</b><span>Rank sites and samples for field work by how much a measurement would resolve.</span></div>
      <div class="decision-card"><b>Audit trail</b><span>Keep a traceable path from molecule to measurement that calibrated models can use later.</span></div>
    </div>
  </section>
  <section class="section">
    <h2>Evidence integrity and current scope</h2>
    <p>Before release, the warehouse was checked against its source tables, per-genome outputs, embedding protocols, taxonomy fields and the Old Woman Creek expression tables. The table lists the findings that shape how the atlas can be used.</p>
    <table class="readiness-table findings-table">
      <thead><tr><th>Area</th><th>Finding</th><th>Result</th><th>Consequence for use</th></tr></thead>
      <tbody>{findings_rows_html}</tbody>
    </table>
    <p class="note">Evidence matures in four stages: payload availability, signal within one protocol, comparability across annotation routes, then sample and ecological validation. Detailed audit tables stay in the internal warehouse; this report shows the resulting evidence states and the validation agenda.</p>
  </section>
  {infographic_block}
  <section class="section">
    <h2>The tri-view evidence contract</h2>
    <p>A tri-view record carries an ESM-2 embedding, a gLM2 embedding and a functional annotation. Its evidence state records whether those payloads support a common quantitative interpretation. That state travels with the record into the freeze manifest, the candidate cards and the release checks.</p>
    <table class="readiness-table">
      <thead><tr><th>Source lane</th><th>Registered</th><th>ESM-2</th><th>gLM2</th><th>Functional annotation</th><th>All three views</th><th>Comparable across routes</th><th>Functional contract</th></tr></thead>
      <tbody>{evidence_rows_html}</tbody>
    </table>
    <div id="evidence-contract-chart" class="viz medium chart-scroll"></div>
    <details class="fallback"><summary>Static version of this chart</summary><img src="{fallback_uri['evidence_contract']}" alt="Genome records by source lane: registered, with all three evidence views, and comparable across annotation routes"></details>
    <p class="note">gLM2 runs under two protocols and is compared only within a protocol: {summary['glm2_single_window_units']:,} records use one native and one shuffled gene-order window, and {summary['glm2_multiwindow_units']:,} Old Woman Creek records use ten of each. ESM-2 uses one 650-million-parameter model with a cap of 6,000 proteins per genome; the {summary['esm2_cap_applied_units']:,} capped records are flagged.</p>
  </section>
  <section class="section">
    <h2>Source provenance and environmental readiness</h2>
    <p>Where each source comes from, how precisely its genomes can be placed in a sample today, and the next link needed. Together these gaps set a concrete agenda for abundance mapping, environmental context and field validation.</p>
    <table class="readiness-table">
      <thead><tr><th>Source lane</th><th>Records</th><th>Source universe</th><th>Primary source</th><th>Resolution now</th><th>Use now</th><th>Blocking gap</th></tr></thead>
      <tbody>{provenance_rows_html}</tbody>
    </table>
    <p class="note">The proof-of-concept crosswalk covers a 662-proteome cohort; the atlas reference core holds 625 of those genomes ({safe_int(nearest_core.get('reference_core_rumen_units')):,} rumen, {safe_int(nearest_core.get('reference_core_wetland_units')):,} wetland) alongside the registered mangrove and Old Woman Creek records. Pending, source-gap, mixed-resolution and unlinked rows stay visible as explicit states.</p>
  </section>
  <section class="section">
    <h2>Three molecular views in one evidence graph</h2>
    <p>Each view answers a different question, has its own comparison set and carries its own validation gaps. A mechanism claim needs convergent evidence from the right view under a compatible protocol; nearness on a map is only a starting point.</p>
    <div class="approach-grid">
      <div class="approach-card"><b>ESM-2 protein embeddings</b><span>A protein language model summarizes each genome's proteins as one vector. Neighborhoods suggest genomes worth comparing; they do not establish shared function.</span></div>
      <div class="approach-card"><b>Functional annotation</b><span>The reference core and both mangrove sources carry shared-pipeline screening events. Old Woman Creek keeps its source annotations and expression data. No ranking crosses the two routes.</span></div>
      <div class="approach-card"><b>gLM2 genomic context</b><span>Available for {summary.get('release_glm2_units', summary['external_glm2'] + summary['poc_core_total']):,} records. Native and shuffled gene-order windows run under two protocols, so scores are compared within a protocol.</span></div>
      <div class="approach-card"><b>Quality and provenance checks</b><span>Genome completeness and contamination, taxonomy, annotation coverage, source labels and missingness guard against attractive artifacts. Weak evidence stays visible instead of being dropped.</span></div>
    </div>
    <p class="note">For every record the report shows what evidence exists, its protocol, what each functional count measures and the claim the record supports. A methane mechanism score across routes becomes possible once a shared feature table is built and validated.</p>
  </section>
  <section class="section">
    <h2>ESM-2 geometry with measured limitations</h2>
    <p>The ESM-2 representation places {safe_int(geometry.get('embedding_units')):,} records in a {dimensions:,}-dimensional space. Raw cosine similarity in this space is strongly anisotropic: two random records have a mean cosine of {safe_float(geometry.get('random_pair_similarity_mean')):.4f} (median {random_median:.4f}), and the median similarity to the global centroid is {safe_float(geometry.get('similarity_to_global_centroid_median')):.4f}. Cross-habitat neighbor edges therefore sit in a saturated range, with a median raw cosine of {safe_float(geometry.get('raw_cross_edge_similarity_median')):.6f}. The atlas uses this geometry to navigate neighborhoods and keeps functional and validation evidence separate.</p>
    <p>A stricter test counts mutual neighbors: pairs in which each record is among the other's {knn_k} closest across the full atlas. Raw space holds {safe_int(raw_pairs.get('mangrove↔wetland')):,} mutual mangrove–wetland pairs, {raw_rumen_wetland:,} rumen–wetland pair{'' if raw_rumen_wetland == 1 else 's'} and {raw_rumen_mangrove:,} rumen–mangrove pair{'' if raw_rumen_mangrove == 1 else 's'}. After each dimension is standardized, {safe_int(z_pairs.get('mangrove↔wetland')):,} mangrove–wetland pairs remain {z_rumen_text}.</p>
    <p>A separate, one-way comparison asks which member of the {safe_int(nearest_core.get('reference_core_units')):,}-genome reference core ({safe_int(nearest_core.get('reference_core_rumen_units')):,} rumen, {safe_int(nearest_core.get('reference_core_wetland_units')):,} wetland) is closest to each record. Outside the core, {wetland_outside_rumen:,} of {wetland_outside:,} wetland and {mangrove_rumen:,} of {mangrove_units:,} mangrove records point to a rumen genome, as do {candidates_rumen_text}. The median closest-match similarity outside the core is {nearest_median:.3f}; the random-pair atlas median is {random_median:.3f}. These values describe representation geometry, not a calibrated measure of biological equivalence. The core is mostly rumen, so a rumen match nominates a record for review; it does not establish shared biology, transfer between sources or methane flux.</p>
    <p>Taxonomy explains part of the mangrove–wetland continuity. Among mutual pairs with usable phylum labels, {100 * safe_float(taxonomy_audit.get('raw_exact_name_share_usable')):.1f}% match exactly and {100 * safe_float(taxonomy_audit.get('synonym_normalized_share_usable')):.1f}% match after conservative synonym normalization. GTDB release metadata exists only for the reference core, so source and taxonomy-release effects are confounded. Harmonized taxonomy and phylogeny-aware null models are needed before neighborhood enrichment can be read as functional convergence.</p>
    <p>UMAP is the default map view because it keeps local neighborhoods readable. Diffusion, t-SNE and PCA are offered for comparison; source structure and projection distortion must be assessed in the reconciled release. No projection is evidence on its own, and link membership is always computed in the full representation.</p>
    <p class="note"><b>References</b></p>
    <ol class="refs">{references_html}</ol>
  </section>
  <section class="section">
    <h2>Molecular niche-space map</h2>
    <p>Every record with an ESM-2 embedding appears on the map, colored by source lane. Positions come from a two-dimensional projection; links come from the full {dimensions:,}-dimensional representation, so switching projections moves points but never changes which links exist.</p>
    <p>Gold lines join the {drawn_case_links:,} selected candidates outside the reference core to their closest reference-core genome. Teal lines are mutual neighbor pairs and gray lines one-way neighbor pairs; the map draws the {drawn_neighbor_links:,} most similar of the {cross_edges:,} directed cross-habitat neighbor edges.{neighbor_mix_text} Gold rings mark the {case_study_count:,} case-study candidates: select one to read its evidence card, or hover over any point.</p>
    <div class="toolbar" id="method-buttons"></div>
    <p class="note" id="projection-note"></p>
    <div class="legend"><span><i class="dot" style="background:var(--rumen)"></i>Rumen reference</span><span><i class="dot" style="background:var(--wetland)"></i>Wetland (mostly Old Woman Creek)</span><span><i class="dot" style="background:var(--mangrove)"></i>Mangrove, China coast (MSM)</span><span><i class="dot" style="background:var(--futian)"></i>Mangrove, Futian (Shenzhen)</span><span><i class="halo"></i>Case-study candidate</span><span><i class="line" style="border-color:var(--gold)"></i>Candidate to closest core genome</span><span><i class="line" style="border-color:var(--teal)"></i>Mutual neighbors</span><span><i class="line" style="border-color:#94a3b8"></i>One-way neighbors</span></div>
    <div class="grid2">
      <div id="niche-map" class="viz tall"></div>
      <aside id="candidate-card" class="side-card" aria-live="polite"><p class="note">Select a case-study candidate on the map, or a cell in the matrix below, to read its evidence card.</p></aside>
    </div>
    <details class="fallback"><summary>Static version of this map</summary><img src="{fallback_uri['niche']}" alt="UMAP projection of the atlas records colored by source lane"></details>
  </section>
  <section class="section">
    <h2>Candidate evidence cards</h2>
    <p>The candidate layer asks which evidence exists for each review hypothesis, and which comparisons that evidence can support. Reference-core cards (P) are reselected from the corrected cross-habitat neighbor fraction, QC and stable record identity. Mangrove cards (M) are ranked by embedding geometry and QC. Old Woman Creek cards (O) carry the source's annotations and, where present, processed expression detection.</p>
    <p>The matrix and the wheel show availability and eligibility, not strength: ESM-2, gLM2, functional annotation, comparability across routes, expression, QC, taxonomy and sample context. Mechanism strength, activity and any causal link to flux need their own direct evidence.</p>
    <div class="signature-stack">
      <div class="signature-panel">
        <h3>Candidate evidence matrix</h3>
        <div id="signature-matrix" class="viz medium matrix"></div>
      </div>
      <div class="signature-panel">
        <h3>Evidence coverage by candidate group</h3>
        <div id="candidate-circos" class="viz medium circos"></div>
        <p class="figure-caption">Each ring is a candidate group, from the reference core (inner) to Old Woman Creek (outer). A wedge's length is the share of that group's cards with the evidence; an empty position means no card in the group has it. Expression exists only in the Old Woman Creek source, so its absence elsewhere means not assayed rather than not expressed.</p>
      </div>
    </div>
    <details class="fallback"><summary>Static version of the matrix</summary><img src="{fallback_uri['matrix']}" alt="Candidate evidence matrix: which evidence each candidate has"></details>
  </section>
  <section class="section">
    <h2>Functional metric harmonization</h2>
    <p>A methane-marker density is only meaningful if every source counts genes the same way. Raw annotation-hit rows do not: one gene can produce several hits, and the number of hits per protein differs by source and tool. The shared pipeline therefore records accepted KOfam genes and METABOLIC events as explicit events and keeps the best MCycDB and SCycDB hits separate; MCycDB hits do not enter a methane score until a validated family map exists. Old Woman Creek keeps its own source annotations under a separate contract.</p>
    <table class="readiness-table">
      <thead><tr><th>Source lane</th><th>Annotated records</th><th>What is counted now</th><th>Raw methane hit rows (median)</th><th>Proteins (median)</th><th>Records with over one raw hit per protein</th><th>Public status</th></tr></thead>
      <tbody>{functional_rows_html}</tbody>
    </table>
    <p>The raw-row columns are diagnostics that show why raw hits cannot be compared. A combined index built on them is quarantined and plays no part in any ranking here: its methane component tracked the index closely (Pearson r = {safe_float(functional_audit.get('legacy_score_methane_component_pearson_r')):.3f}), and {top500_text}, a pattern consistent with counting differences between sources rather than biology. The event tables support screening and audit within one route; they do not authorize a ranking across routes.</p>
    <div class="warn"><b>Before a cross-route methane score is enabled:</b> lock tool and database versions, validate the gene-family mappings and denominator behavior, quantify missingness and QC sensitivity, and pass source-aware, taxonomy-aware, null, stability and ablation tests.</div>
  </section>
  <section class="section">
    <h2>Old Woman Creek adds expression evidence and a field-validation lane</h2>
    <p>The Old Woman Creek wetland lane (MUCC v1) adds processed metatranscriptome detection across {mucc_columns:,} source sample columns. Expression support exists for {safe_int(mucc_audit.get('processed_expression_supported_mags')):,} MAGs: {mucc_methane:,} have at least one processed methane-associated expressed-gene row and {mucc_sulfur:,} have sulfur-associated rows. These are detection signals from deposited processed tables; comparing activity levels needs expression normalization first.</p>
    <p>The warehouse also stages {safe_int(mucc_audit.get('chamber_flux_rows')):,} chamber-flux rows ({safe_int(mucc_audit.get('chamber_flux_valid_rows')):,} valid in the source), {safe_int(mucc_audit.get('porewater_rows')):,} porewater rows ({safe_int(mucc_audit.get('porewater_valid_rows')):,} valid) and {safe_int(mucc_audit.get('tower_flux_rows')):,} half-hourly, gap-filled tower-flux rows. It includes {safe_int(mucc_audit.get('flashweave_edges')):,} exploratory FlashWeave associations, {safe_int(mucc_audit.get('flashweave_stable_edges')):,} of which pass the current stability filter, and {safe_int(mucc_audit.get('wgcna_non_grey_modules')):,} descriptive WGCNA modules (excluding the unassigned grey module).</p>
    <p>The decisive gap is linkage: {mucc_exact:,} of {mucc_columns:,} sequencing samples have an authoritative exact join to sample, depth, environment and flux, and {safe_int(mucc_audit.get('ecological_join_blocked_samples')):,} remain blocked for ecological validation. Until that join exists, the flux records are site and time context, and no MAG, expression signature or network edge can be attributed to a measured flux.</p>
  </section>
  <section class="section">
    <h2>Sample-linkage readiness</h2>
    <p>This chart groups mangrove records at the most precise environmental context available today. Futian records are grouped by site and month, each context spanning several depth samples with chemistry metadata. MSM records are grouped by the source's sample groups and BioSample sets. No record is yet assigned to a single physical sample.</p>
    <p>These groups show a field team where abundance mapping, metadata reconciliation and validation measurements would pay off first. Sample-level methane-risk estimates become possible only after per-genome abundance, exact sample assignment, environmental conditions and flux or process validation are in place.</p>
    <div id="sample-linkage" class="viz medium chart-scroll"></div>
  </section>
  <section class="section">
    <h2>From molecular evidence to environmental readiness</h2>
    <p>The atlas becomes decision-grade when validated genome features roll up to physical samples, sites and monitoring periods. Today the reference core and both mangrove sources carry shared-pipeline screening events, and Old Woman Creek carries source annotations and expression detection; none is yet a mechanism score comparable across routes. A defensible sample-level score needs four gated layers.</p>
    {sample_risk_abstract_block}
    <div class="layer-list">
      <div class="layer"><span class="state partial">Partial</span><h3>1. Link molecules to samples</h3><p>Assign each genome to its sample and site, keep resolution tiers, and show unlinked genomes as explicit states.</p><p class="now-line">Now: site-month or sample-group context for mangroves; {mucc_exact:,} of {mucc_columns:,} exact joins at Old Woman Creek.</p></div>
      <div class="layer"><span class="state missing">Not yet</span><h3>2. Weight by community abundance</h3><p>Turn genome potential into sample capacity with genome coverage, marker abundance, unbinned functional reads and assembly uncertainty.</p><p class="now-line">Now: no abundance joined to the atlas records.</p></div>
      <div class="layer"><span class="state partial">Partial</span><h3>3. Add environmental conditions</h3><p>Measured metadata first, modeled covariates second, with every salinity, sulfate, redox, substrate, depth and vegetation field marked by evidence tier.</p><p class="now-line">Now: environmental fields for some mangrove sample contexts, not yet joined to genomes.</p></div>
      <div class="layer"><span class="state missing">Not yet</span><h3>4. Calibrate with field evidence</h3><p>Anchor predictions to chamber flux, dissolved methane, porewater chemistry, incubations or repeated observations, with explicit joins in time and place.</p><p class="now-line">Now: no genome-to-flux pairs.</p></div>
    </div>
    <p>Field work is how the atlas learns. Sampling across habitats, restoration stages, salinity gradients, depths and seasons would widen the map, expose source-specific blind spots and test candidate signatures under blue-carbon conditions, provided each sample arrives with clean provenance, abundance, environmental measurements and a validation target.</p>
    <p>The next operational output is a sample-readiness layer. Once samples are mapped, each would be labeled scoreable, monitor more, needs metadata, needs abundance, needs environmental covariates or needs flux validation. Current records do not support calibrated sample-risk scores; the labels would guide sampling while the evidence grows.</p>
  </section>
  <section class="section">
    <h2>What comes next</h2>
    <p class="closing">The atlas rests on a source-audited warehouse of {summary['atlas_registered_units']:,} registered genome records across four source lanes. It already supports payload auditing, neighborhood exploration, protocol-aware candidate review, expression-detection queries, metadata-gap priorities and validation-study design.</p>
    <p class="closing">Its evidence states are explicit: {summary['pipeline_normalized_tri_view_units']:,} records carry shared-pipeline screening events, {summary['source_scaffold_tri_view']:,} carry Old Woman Creek source annotations, and {summary['mechanism_comparable_tri_view']:,} pass the cross-route comparability gate. Keeping those states separate protects decisions from pipeline artifacts.</p>
    <p class="closing">The next build should deliver one mechanism-feature table shared by all sources, harmonized taxonomy with phylogeny-aware null models, calibrated gLM2 protocols, exact sample and abundance mappings, and field or process validation with uncertainty. Together these enable cross-route mechanism ranking and calibrated sample-risk modeling.</p>
    <div class="warn">Current evidence supports molecular screening, evidence-card review and measurement planning. Final A–E risk tiers, measured methane-flux claims, carbon-credit decisions and claims of transfer between independent sources all require further validation.</div>
  </section>
</main>
<script src="{d3_href}"></script>
<script>window.METHANET_ATLAS = {atlas_payload_json};</script>
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
    payload: dict[str, Any],
    payload_paths: dict[str, Path],
    fallback_paths: dict[str, Path],
    summary: dict[str, Any],
    manifold_methods: list[dict[str, str]],
    source_readiness: list[dict[str, Any]],
    validation_gates: list[dict[str, Any]],
    infographic_path: Path | None,
    sample_risk_abstract_path: Path,
    release_ledger_path: Path | None,
    html_text: str,
) -> None:
    table_dir = output_dir / "tables"
    source_dir = output_dir / "sources"
    audit_dir = output_dir / "audit"
    table_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(table_dir / "atlas_multiview_feature_table.tsv", sep="\t", index=False)
    emb_meta.to_csv(table_dir / "embedding_context_table.tsv", sep="\t", index=False)
    edge_df.to_csv(table_dir / "bridge_knn_edges.tsv", sep="\t", index=False)
    pd.DataFrame(payload.get("niche", {}).get("links", [])).to_csv(
        table_dir / "bridge_evidence_links.tsv", sep="\t", index=False
    )
    pd.DataFrame(payload.get("sample_linkage", {}).get("groups", [])).to_csv(
        table_dir / "sample_linkage_group_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(payload.get("sample_linkage", {}).get("contexts", [])).to_csv(
        table_dir / "sample_linkage_context_summary.tsv", sep="\t", index=False
    )
    cards.to_csv(table_dir / "candidate_cards.tsv", sep="\t", index=False)
    status.to_csv(table_dir / "mangrove_payload_status.tsv", sep="\t", index=False)
    if "freeze_release_excluded" in atlas.columns:
        release_exclusions = atlas[truthy_series(atlas["freeze_release_excluded"])].copy()
    else:
        release_exclusions = pd.DataFrame()
    release_exclusions.to_csv(table_dir / "release_exclusions.tsv", sep="\t", index=False)
    pd.DataFrame(manifold_methods).to_csv(table_dir / "manifold_method_status.tsv", sep="\t", index=False)
    pd.DataFrame(source_readiness).to_csv(table_dir / "source_provenance_readiness.tsv", sep="\t", index=False)
    pd.DataFrame(validation_gates).to_csv(table_dir / "report_validation_gates.tsv", sep="\t", index=False)
    pd.DataFrame(payload.get("evidence_contract", [])).to_csv(
        table_dir / "evidence_contract_summary.tsv", sep="\t", index=False
    )
    scientific_audit = payload.get("scientific_audit", {})
    pd.DataFrame(scientific_audit.get("findings", [])).to_csv(
        table_dir / "scientific_reconciliation_findings.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        scientific_audit.get("functional_metric_provenance", {}).get(
            "lane_metrics", []
        )
    ).to_csv(
        table_dir / "functional_metric_provenance_audit.tsv",
        sep="\t",
        index=False,
    )
    (source_dir / "scientific_audit.json").write_text(
        json.dumps(
            json_safe(scientific_audit),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if release_ledger_path is not None and release_ledger_path.exists():
        shutil.copy2(release_ledger_path, source_dir / "release_ledger.json")
    claim_matrix = pd.DataFrame(
        [
            {
                "claim": "Expanded atlas supports MAG/proteome molecular screening",
                "status": "allowed",
                "allowed_wording": "EmergentBiome can inspect multiview molecular evidence for completed MAG/proteome units.",
                "blocking_gap": "none for MAG/proteome screening",
            },
            {
                "claim": "Bridge candidates are review-ready hypotheses",
                "status": "allowed with caveats",
                "allowed_wording": "Candidates can be prioritized for source-aware review using ESM-2 geometry, protocol-stratified gLM2 availability, evidence-contract state, QC, and taxonomy.",
                "blocking_gap": "common mechanism-feature rebuild, source-aware nulls, phylogeny comparison, sample linkage",
            },
            {
                "claim": (
                    f"All {summary['release_multiview_complete']:,} tri-view units "
                    "are fully mechanism harmonized"
                ),
                "status": "forbidden in current release",
                "allowed_wording": (
                    f"{summary['release_multiview_complete']:,} units are data-complete "
                    f"tri-views; {summary['mechanism_comparable_tri_view']:,} are "
                    "currently mechanism-comparable."
                ),
                "blocking_gap": "common accepted/present mechanism-feature rebuild across MSM, Futian, and MUCC",
            },
            {
                "claim": "MUCC expression supports transcriptional detection",
                "status": "allowed with caveats",
                "allowed_wording": "Processed MUCC tables support gene-expression detection/occupancy for named MAGs and source sample columns.",
                "blocking_gap": "expression normalization units, exact environmental/flux crosswalk, abundance and uncertainty",
            },
            {
                "claim": "Cross-lane molecular attestation ranking",
                "status": "quarantined",
                "allowed_wording": "No universal cross-lane mechanism rank is published in this release.",
                "blocking_gap": "common feature aggregation, denominator validation, protocol calibration, and source-balance tests",
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
    public_audit_files = [
        table_dir / "evidence_contract_summary.tsv",
        table_dir / "scientific_reconciliation_findings.tsv",
        table_dir / "functional_metric_provenance_audit.tsv",
        table_dir / "report_validation_gates.tsv",
        table_dir / "claim_boundary_matrix.tsv",
        table_dir / "release_exclusions.tsv",
        source_dir / "scientific_audit.json",
    ]
    for audit_file in public_audit_files:
        shutil.copy2(audit_file, audit_dir / audit_file.name)
    if infographic_path is not None and infographic_path.exists():
        shutil.copy2(infographic_path, output_dir / "assets/figures/methanet_agentic_workflow_moat_v3.png")
    if RENDER_SAMPLE_RISK_ABSTRACT and sample_risk_abstract_path.exists():
        shutil.copy2(
            sample_risk_abstract_path,
            output_dir / "assets/figures/figure_04_mag_to_sample_risk_readiness_graphical_abstract.png",
        )
    report = output_dir / "report.html"
    report.write_text(html_text)
    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "report": str(report),
        "summary": payload["summary"],
        "tables": {
            "atlas_multiview_feature_table": str(table_dir / "atlas_multiview_feature_table.tsv"),
            "embedding_context_table": str(table_dir / "embedding_context_table.tsv"),
            "bridge_knn_edges": str(table_dir / "bridge_knn_edges.tsv"),
            "bridge_evidence_links": str(table_dir / "bridge_evidence_links.tsv"),
            "sample_linkage_group_summary": str(table_dir / "sample_linkage_group_summary.tsv"),
            "sample_linkage_context_summary": str(table_dir / "sample_linkage_context_summary.tsv"),
            "candidate_cards": str(table_dir / "candidate_cards.tsv"),
            "mangrove_payload_status": str(table_dir / "mangrove_payload_status.tsv"),
            "release_exclusions": str(table_dir / "release_exclusions.tsv"),
            "manifold_method_status": str(table_dir / "manifold_method_status.tsv"),
            "source_provenance_readiness": str(table_dir / "source_provenance_readiness.tsv"),
            "report_validation_gates": str(table_dir / "report_validation_gates.tsv"),
            "claim_boundary_matrix": str(table_dir / "claim_boundary_matrix.tsv"),
            "evidence_contract_summary": str(
                table_dir / "evidence_contract_summary.tsv"
            ),
            "scientific_reconciliation_findings": str(
                table_dir / "scientific_reconciliation_findings.tsv"
            ),
            "functional_metric_provenance_audit": str(
                table_dir / "functional_metric_provenance_audit.tsv"
            ),
            "scientific_audit": str(source_dir / "scientific_audit.json"),
        },
        "interactive_payloads": {k: str(v) for k, v in payload_paths.items()},
        "fallback_figures": {k: str(v) for k, v in fallback_paths.items()},
        "infographic": str(infographic_path) if infographic_path is not None else None,
        "sample_risk_graphical_abstract": str(sample_risk_abstract_path),
        "claim_boundary": CLAIM_BOUNDARY,
        "release_ledger": (
            str(source_dir / "release_ledger.json")
            if release_ledger_path is not None
            else None
        ),
    }
    (output_dir / "report_bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (output_dir / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # EmergentBiome Molecular Atlas

            Generated: {summary['generated_at_utc']}

            Main artifact: `report.html`

            ## Snapshot

            - POC core tri-view complete: {summary['poc_core_total']:,}/{summary['poc_core_total']:,}
            - Registered mangrove ESM-2 complete: {summary['msm_esm2']:,}/{summary['mangrove_ready_payload_total']:,}
            - Registered mangrove gLM2 complete: {summary['msm_glm2']:,}/{summary['mangrove_ready_payload_total']:,}
            - Registered mangrove functional complete: {summary['msm_functional']:,}/{summary['mangrove_ready_payload_total']:,}
            - Release-required registered mangrove functional complete: {summary['mangrove_release_functional']:,}/{summary['mangrove_release_required_payload_total']:,}
            - Release-excluded units preserved: {summary['mangrove_release_excluded_units']:,}
            - Registered mangrove source-lane gap rows preserved: {summary['mangrove_gap_rows']:,}
            - Expanded release-required data-complete tri-view atlas: {summary['release_multiview_complete']:,}
            - Pipeline-normalized tri-view, comparability pending: {summary['pipeline_normalized_tri_view_units']:,}
            - Cross-lane mechanism-comparable tri-view: {summary['mechanism_comparable_tri_view']:,}
            - Annotation-complete, harmonization-pending tri-view: {summary['annotation_complete_harmonization_pending_tri_view']:,}
            - MUCC v1 source-scaffold tri-view: {summary['source_scaffold_tri_view']:,}
            - MUCC v1 ESM-2/gLM2/source-functional tri-view: {summary['mucc_v1_tri_view']:,}/{summary['mucc_v1_total']:,}
            - Registered ESM-2 embedding context: {summary['embedding_context_total']:,}

            Data-complete tri-view does not imply common mechanism comparability.
            MSM/Futian functional outputs are complete but await a shared
            accepted/present feature rebuild. MUCC v1 source-scaffold rows support
            source-aware screening and expression detection but are not canonical
            mechanism-equivalent rows.

            ## Claim Boundary

            {CLAIM_BOUNDARY}

            ## Regenerate

            ```bash
            MPLCONFIGDIR=/tmp/methanet_mpl NUMBA_CACHE_DIR=/tmp/methanet_numba \\
            .venv/bin/python scripts/reports/build_mbag_nextgen_molecular_niche_atlas.py \\
              --lane-registry configs/methanet_atlas_lanes_20260929.tsv \\
              --embedding-contract configs/atlas_embedding_contract_20260929.json \\
              --freeze-manifest results/reports/methanet_3view_payload_freeze_<UTCSTAMP>/freeze_manifest.tsv \\
              --skip-phate --output-dir results/reports/<new_report_dir>
            ```

            Use `--skip-phate` (or an environment without PHATE) so the published
            projection buttons stay UMAP, diffusion map, t-SNE and PCA.
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
    registry_path = resolve(repo_root, args.lane_registry)
    freeze_manifest = resolve(repo_root, args.freeze_manifest)
    release_ledger_path = resolve(repo_root, args.release_ledger)
    if release_ledger_path is None and freeze_manifest is not None:
        release_ledger_path = freeze_manifest.parent / "release_ledger.json"
    if freeze_manifest is not None and (
        release_ledger_path is None or not release_ledger_path.exists()
    ):
        raise SystemExit(
            "A freeze-backed report requires the sibling release_ledger.json or "
            "an explicit --release-ledger path."
        )
    release_ledger = (
        json.loads(release_ledger_path.read_text())
        if release_ledger_path is not None and release_ledger_path.exists()
        else None
    )
    infographic = resolve(repo_root, args.infographic)
    sample_risk_abstract = resolve(repo_root, args.sample_risk_abstract)
    assert (
        poc_esm_dir
        and poc_warehouse_dir
        and poc_glm_dir
        and msm_root
        and msm_esm_dir
        and msm_glm_dir
        and sample_risk_abstract
    )

    lane_ledger: list[dict[str, Any]] = []
    registry_metadata: dict[str, Any] = {}
    if registry_path and registry_path.exists():
        registry = legacy.read_lane_registry(registry_path)
        legacy.validate_report_lane_registry(repo_root, registry_path, registry)
        atlas, poc, msm, msm_status, lane_ledger, esm_inputs, msm_esm_stats = legacy.load_registry_backed_atlas(
            repo_root,
            registry,
            args,
        )
        registry_metadata = {
            "lane_registry": str(registry_path),
            "lane_registry_rows": int(len(registry)),
            "input_mode": "lane_registry",
        }
        embedding_configuration = validate_embedding_contract(
            repo_root, resolve(repo_root, args.embedding_contract), esm_inputs
        )
        emb_meta, edge_df, embeddings = legacy.build_embedding_context_from_inputs(esm_inputs, atlas, args.knn)
    else:
        if not args.allow_legacy_defaults:
            raise SystemExit(
                "Lane registry is required for current nextgen atlas rebuilds. "
                "Supply the reconciled lane registry and a verified embedding contract."
            )
        raise SystemExit("Historical mixed-configuration geometry is withdrawn. Use a registry and verified embedding contract.")

    atlas, msm_status, freeze_metadata = apply_freeze_manifest(atlas, msm_status, freeze_manifest)
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
    atlas = apply_scientific_evidence_contract(atlas)
    atlas["mixing_coeff"] = atlas["proteome_id"].map(
        emb_meta.set_index("proteome_id")["cross_domain_neighbor_fraction"]
    )
    atlas = legacy.add_report_metrics(atlas, emb_meta)
    # The legacy metric helper derives useful within-contract diagnostics but
    # predates the pipeline-normalized freeze state and maps unknown complete
    # rows to source scaffold. Reapply the authoritative freeze contract after
    # metric calculation so rendering and release accounting cannot inherit
    # that historical fallback.
    atlas = apply_scientific_evidence_contract(atlas)
    neighbor_cols = [c for c in ["nearest_poc_id", "nearest_mangrove_id"] if c in emb_meta.columns and c not in atlas.columns]
    if neighbor_cols:
        atlas = atlas.merge(
            emb_meta[["proteome_id"] + neighbor_cols].drop_duplicates("proteome_id"),
            on="proteome_id",
            how="left",
        )
    manifold_cols = [c for c in emb_meta.columns if c.endswith("_1") or c.endswith("_2")]
    if manifold_cols:
        atlas = atlas.drop(columns=manifold_cols, errors="ignore")
        atlas = atlas.merge(
            emb_meta[["proteome_id"] + manifold_cols].drop_duplicates("proteome_id"),
            on="proteome_id",
            how="left",
        )
    atlas = add_molecular_metrics(atlas)
    atlas["review_tier"] = atlas.apply(classify_review_tier, axis=1)
    row_defaults = {
        "allowed_claim_wording": (
            "Genome-level screening and monitoring-priority hypothesis only; "
            "sample, abundance, environmental, uncertainty and validation "
            "layers come before any MRV score."
        ),
        "blocking_gap": (
            "sample mapping, abundance, environmental covariates, uncertainty, "
            "phylogeny and source controls, and flux or process validation"
        ),
        "next_validation_action": (
            "connect to sample metadata, run source-aware null models, compare "
            "phylogeny with embedding proximity, and validate against flux or "
            "process measurements before risk scoring"
        ),
    }
    for column, default in row_defaults.items():
        if column not in atlas.columns:
            atlas[column] = default
        else:
            existing = atlas[column].fillna("").astype(str).str.strip()
            atlas.loc[existing.eq(""), column] = default
    atlas = add_source_provenance_context(atlas, repo_root, msm_root)
    atlas = add_sample_linkage_context(atlas, repo_root, msm_root)
    atlas = apply_public_lane_display(atlas)

    # Re-split report frames after freeze/provenance enrichment so release
    # accounting uses the audited atlas, not stale loader slices.
    poc = atlas[atlas["atlas_inclusion_status"].astype(str).eq("poc_core_complete")].copy()
    msm = atlas[atlas["source_category"].astype(str).eq("mangrove")].copy()
    external = atlas[~atlas["lane_id"].astype(str).eq("poc_core")].copy()
    mucc_v1 = atlas[
        atlas["lane_id"].astype(str).eq("mucc_v1_owc_wetland")
    ].copy()

    cards = build_candidate_cards(atlas, args.top_n_poc, args.top_n_mangrove)
    if "functional_run_include" in msm.columns:
        mangrove_ready_mask = truthy_series(msm["functional_run_include"])
    else:
        mangrove_ready_mask = pd.Series([True] * len(msm), index=msm.index)
    if "freeze_release_required" in msm.columns:
        mangrove_release_required_mask = truthy_series(msm["freeze_release_required"])
    else:
        mangrove_release_required_mask = pd.Series([True] * len(msm), index=msm.index)
    if "freeze_release_excluded" in msm.columns:
        mangrove_release_excluded_mask = truthy_series(msm["freeze_release_excluded"])
    else:
        mangrove_release_excluded_mask = pd.Series([False] * len(msm), index=msm.index)
    mangrove_release_ready_mask = mangrove_ready_mask & mangrove_release_required_mask
    mangrove_tri_view_mask = msm["has_esm2"] & msm["has_glm2"] & msm["has_functional"]
    mangrove_release_tri_view_mask = mangrove_release_ready_mask & mangrove_tri_view_mask
    all_tri_view_mask = (
        atlas["has_esm2"] & atlas["has_glm2"] & atlas["has_functional"]
    )
    if "freeze_release_required" in atlas.columns:
        all_release_required_mask = truthy_series(
            atlas["freeze_release_required"]
        )
    else:
        all_release_required_mask = atlas.get(
            "functional_run_include",
            pd.Series([True] * len(atlas), index=atlas.index),
        ).map(legacy.truthy)
    all_release_tri_view_mask = all_release_required_mask & all_tri_view_mask
    canonical_tri_view_mask = atlas.get(
        "formal_tri_view_status",
        pd.Series("", index=atlas.index),
    ).eq("complete_canonical_mechanism_tri_view")
    annotation_complete_tri_view_mask = atlas.get(
        "formal_tri_view_status",
        pd.Series("", index=atlas.index),
    ).eq("complete_annotation_tri_view_harmonization_pending")
    source_scaffold_tri_view_mask = atlas.get(
        "formal_tri_view_status",
        pd.Series("", index=atlas.index),
    ).eq("complete_source_scaffold_tri_view")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(repo_root),
        "poc_core_total": int(poc["has_functional"].sum()),
        "poc_rumen_total": int(poc["source_category"].eq("rumen").sum()),
        "poc_wetland_total": int(poc["source_category"].eq("wetland").sum()),
        "msm_total": int(len(msm)),
        "mangrove_ready_payload_total": int(mangrove_ready_mask.sum()),
        "mangrove_gap_rows": int((~mangrove_ready_mask).sum()),
        "msm_esm2": int(msm["has_esm2"].sum()),
        "msm_glm2": int(msm["has_glm2"].sum()),
        "msm_functional": int(msm["has_functional"].sum()),
        "msm_multiview": int(mangrove_tri_view_mask.sum()),
        "mangrove_release_required_payload_total": int(mangrove_release_ready_mask.sum()),
        "mangrove_release_excluded_units": int(mangrove_release_excluded_mask.sum()),
        "mangrove_release_functional": int((mangrove_release_ready_mask & msm["has_functional"]).sum()),
        "mangrove_release_multiview": int(mangrove_release_tri_view_mask.sum()),
        "mangrove_release_function_pending": int((mangrove_release_ready_mask & ~msm["has_functional"]).sum()),
        "msm_function_pending": int((mangrove_ready_mask & ~msm["has_functional"]).sum()),
        "msm_esm_embedded_total_with_resume": int(msm_esm_stats.get("embedded_total_with_resume") or 0),
        "msm_esm_pending_remaining": int(msm_esm_stats.get("pending_remaining") or 0),
        "external_total": int(len(external)),
        "external_esm2": int(external["has_esm2"].sum()),
        "external_glm2": int(external["has_glm2"].sum()),
        "external_functional": int(external["has_functional"].sum()),
        "external_multiview": int(
            (
                external["has_esm2"]
                & external["has_glm2"]
                & external["has_functional"]
            ).sum()
        ),
        "mucc_v1_total": int(len(mucc_v1)),
        "mucc_v1_esm2": int(mucc_v1["has_esm2"].sum()),
        "mucc_v1_glm2": int(mucc_v1["has_glm2"].sum()),
        "mucc_v1_functional": int(mucc_v1["has_functional"].sum()),
        "mucc_v1_tri_view": int(
            (
                mucc_v1["has_esm2"]
                & mucc_v1["has_glm2"]
                & mucc_v1["has_functional"]
            ).sum()
        ),
        "multiview_complete": int(all_tri_view_mask.sum()),
        "release_multiview_complete": int(all_release_tri_view_mask.sum()),
        "canonical_mechanism_tri_view": int(canonical_tri_view_mask.sum()),
        "mechanism_comparable_tri_view": int(canonical_tri_view_mask.sum()),
        "annotation_complete_harmonization_pending_tri_view": int(
            annotation_complete_tri_view_mask.sum()
        ),
        "source_scaffold_tri_view": int(source_scaffold_tri_view_mask.sum()),
        "atlas_registered_units": int(len(atlas)),
        "embedding_context_total": int(len(emb_meta)),
        "knn_edges": int(len(edge_df)),
        "glm2_single_window_units": int(
            atlas["glm2_protocol_class"]
            .eq("paired_single_native_plus_single_shuffled")
            .sum()
        ),
        "glm2_multiwindow_units": int(
            atlas["glm2_protocol_class"]
            .eq("multiwindow_10_native_plus_10_shuffled")
            .sum()
        ),
        "esm2_cap_applied_units": int(
            atlas["esm2_protein_cap_status"].eq("cap_6000_applied").sum()
        ),
    }
    summary.update(registry_metadata)
    summary.update(freeze_metadata)
    if release_ledger is not None:
        schema_normalized_mask = truthy_series(
            atlas.get("freeze_schema_normalized", pd.Series(False, index=atlas.index))
        )
        observed_release = {
            "registered_units": int(len(atlas)),
            "esm2_units": int(atlas["has_esm2"].sum()),
            "glm2_units": int(atlas["has_glm2"].sum()),
            "functional_payload_units": int(atlas["has_functional"].sum()),
            "release_required_units": int(all_release_required_mask.sum()),
            "explicit_non_runnable_gaps": int(
                truthy_series(
                    atlas.get("freeze_release_excluded", pd.Series(False, index=atlas.index))
                ).sum()
            ),
            "tri_view_ready_units": int(all_release_tri_view_mask.sum()),
            "schema_normalized_units": int(schema_normalized_mask.sum()),
            "schema_normalized_tri_view_units": int(
                (schema_normalized_mask & all_tri_view_mask).sum()
            ),
            "pipeline_normalized_tri_view_units": int(
                atlas.get("formal_tri_view_status", pd.Series("", index=atlas.index))
                .eq("complete_pipeline_normalized_tri_view_comparability_pending")
                .sum()
            ),
            "mechanism_comparable_units": int(canonical_tri_view_mask.sum()),
            "annotation_complete_tri_view_units": int(
                annotation_complete_tri_view_mask.sum()
            ),
            "source_scaffold_tri_view_units": int(
                source_scaffold_tri_view_mask.sum()
            ),
            "blocking_units": int(
                (all_release_required_mask & ~all_tri_view_mask).sum()
            ),
        }
        mismatches = {
            key: {"observed": value, "ledger": release_ledger.get(key)}
            for key, value in observed_release.items()
            if release_ledger.get(key) != value
        }
        if mismatches:
            formal_status_counts = (
                atlas.get("formal_tri_view_status", pd.Series("", index=atlas.index))
                .fillna("")
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
            )
            freeze_formal_status_counts = (
                atlas.get("freeze_formal_tri_view_status", pd.Series("", index=atlas.index))
                .fillna("")
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
            )
            raise SystemExit(
                "Release-ledger reconciliation failed: "
                + json.dumps(
                    {
                        "mismatches": mismatches,
                        "formal_tri_view_status_counts": formal_status_counts,
                        "freeze_formal_tri_view_status_counts": freeze_formal_status_counts,
                    },
                    sort_keys=True,
                )
            )
        summary.update(
            {
                "release_esm2_units": observed_release["esm2_units"],
                "release_glm2_units": observed_release["glm2_units"],
                "release_functional_payload_units": observed_release[
                    "functional_payload_units"
                ],
                "schema_normalized_units": observed_release[
                    "schema_normalized_units"
                ],
                "schema_normalized_tri_view_units": observed_release[
                    "schema_normalized_tri_view_units"
                ],
                "pipeline_normalized_tri_view_units": observed_release[
                    "pipeline_normalized_tri_view_units"
                ],
                "explicit_non_runnable_gaps": observed_release[
                    "explicit_non_runnable_gaps"
                ],
                "blocking_units": observed_release["blocking_units"],
                "snapshot_date": release_ledger["snapshot_date"],
                "release_state": release_ledger["release_state"],
                "release_ledger_schema_version": release_ledger["schema_version"],
                "release_ledger_sha256": hashlib.sha256(
                    release_ledger_path.read_bytes()
                ).hexdigest(),
                "freeze_manifest_sha256": release_ledger[
                    "freeze_manifest_sha256"
                ],
                "indexing_decision": release_ledger["indexing_decision"],
                "allowed_public_wording": release_ledger[
                    "allowed_public_wording"
                ],
                "forbidden_public_wording": release_ledger[
                    "forbidden_public_wording"
                ],
            }
        )
    if "lane_id" in atlas.columns:
        lane_label_by_id = {
            "poc_core": "POC core",
            "msm_china_2025": "Mangrove/MSM local MAG candidates",
            "futian_mangrove_2026_qi": "Phase 1 dereplicated rMAGs at 99% ANI",
            "mucc_v1_owc_wetland": (
                "MUCC v1 Old Woman Creek source-scaffold references"
            ),
        }
        lane_order = [lane_id for lane_id in lane_label_by_id if lane_id in set(atlas["lane_id"].astype(str))]
        lane_order.extend(
            sorted(
                set(atlas["lane_id"].astype(str)) - set(lane_order),
            )
        )
        lane_ledger = [
            legacy.lane_counts(
                atlas[atlas["lane_id"].astype(str).eq(lane_id)].copy(),
                lane_label_by_id.get(lane_id, lane_id),
            )
            for lane_id in lane_order
        ]
    summary["lane_ledger"] = lane_ledger
    source_readiness = build_source_provenance_readiness(summary, atlas)

    evidence_contract = build_evidence_contract_summary(atlas)
    geometry_audit = build_embedding_geometry_audit(
        emb_meta, edge_df, embeddings, args.knn
    )
    nearest_core_audit = build_nearest_core_context_audit(atlas, emb_meta, cards)
    taxonomy_audit = build_taxonomy_bridge_audit(atlas, edge_df)
    functional_metric_audit = build_functional_metric_audit(atlas)
    mucc_validation = build_mucc_validation_readiness(atlas, repo_root)
    scientific_findings = build_scientific_findings(
        atlas,
        evidence_contract,
        geometry_audit,
        taxonomy_audit,
        functional_metric_audit,
        mucc_validation,
    )
    scientific_audit = {
        "evidence_contract": evidence_contract,
        "embedding_geometry": geometry_audit,
        "nearest_core_context": nearest_core_audit,
        "taxonomy": taxonomy_audit,
        "functional_metric_provenance": functional_metric_audit,
        "mucc_validation_readiness": mucc_validation,
        "findings": scientific_findings,
    }
    summary["cross_domain_knn_edges"] = int(
        geometry_audit["raw_cross_domain_directed_edges"]
    )
    summary["reciprocal_unique_cross_domain_pairs"] = int(
        geometry_audit["raw_reciprocal_unique_cross_pairs"]
    )

    public_summary = public_release_summary(summary)
    payload = build_payloads(
        atlas,
        emb_meta,
        edge_df,
        cards,
        public_summary,
        args.graph_node_cap,
        manifold_methods,
        scientific_audit,
    )
    validation_gates = build_report_validation_gates(atlas, payload)
    failed_gates = [gate for gate in validation_gates if gate["status"] != "pass"]
    if failed_gates:
        raise SystemExit(f"Report validation gates failed: {failed_gates}")
    summary["report_validation_gates"] = len(validation_gates)
    summary["report_validation_failures"] = len(failed_gates)
    summary["graph_node_count"] = len(payload["candidate_graph"]["nodes"])
    summary["graph_edge_count"] = len(payload["candidate_graph"]["links"])
    payload_paths = save_payloads(payload, output_dir / "assets/data")
    (output_dir / "embedding_configuration.json").write_text(
        json.dumps(embedding_configuration, indent=2) + "\n"
    )
    fallback_paths = build_fallbacks(payload, output_dir / "assets/figures")
    d3_path, _d3_source = legacy.fetch_d3(output_dir / "assets/js")
    infographic_bundle = (
        copy_report_asset(
            infographic,
            output_dir / "assets/figures/methanet_agentic_workflow_moat_v3.png",
        )
        if infographic is not None
        else None
    )
    sample_risk_abstract_bundle = (
        copy_report_asset(
            sample_risk_abstract,
            output_dir / "assets/figures/figure_04_mag_to_sample_risk_readiness_graphical_abstract.png",
        )
        if RENDER_SAMPLE_RISK_ABSTRACT
        else Path("")
    )
    summary["interactive_runtime_asset"] = str(d3_path)
    summary["interactive_data_asset"] = str(payload_paths["atlas_bundle_js"])
    html_text = render_html(
        summary,
        payload,
        fallback_paths,
        infographic_bundle,
        sample_risk_abstract_bundle,
        d3_path,
        output_dir,
        source_readiness,
    )
    notice = ('<aside style="padding:14px 5%;background:#e8f4ef;color:#163e37;font:14px/1.5 sans-serif">'
        '<strong>Geometry reconciled 29 September 2026.</strong> Displayed ESM-2 geometry uses final-layer (33) pooling; '
        'the original pilot was recomputed from retained inputs. Molecular payload counts retain the 10 August snapshot. '
        'Historical June model-revision metadata is incomplete; layer choice is supported by code lineage and numerical controls. '
        'Neighbor links are exploratory sequence-representation similarities; functional transfer and flux prediction remain unvalidated.</aside>')
    html_text = re.sub(r'(<body[^>]*>)', lambda m: m.group(1) + notice, html_text, count=1)
    write_outputs(
        output_dir=output_dir,
        atlas=atlas,
        emb_meta=emb_meta,
        edge_df=edge_df,
        cards=cards,
        status=msm_status,
        payload=payload,
        payload_paths=payload_paths,
        fallback_paths=fallback_paths,
        summary=summary,
        manifold_methods=manifold_methods,
        source_readiness=source_readiness,
        validation_gates=validation_gates,
        infographic_path=infographic,
        sample_risk_abstract_path=sample_risk_abstract,
        release_ledger_path=release_ledger_path,
        html_text=html_text,
    )
    embedding_configuration["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    embedding_configuration["niche_sha256"] = hashlib.sha256(
        (output_dir / "assets/data/niche.json").read_bytes()
    ).hexdigest()
    embedding_configuration["scientific_audit_sha256"] = hashlib.sha256(
        (output_dir / "audit/scientific_audit.json").read_bytes()
    ).hexdigest()
    embedding_configuration["report_sha256"] = hashlib.sha256(
        (output_dir / "report.html").read_bytes()
    ).hexdigest()
    (output_dir / "embedding_configuration.json").write_text(
        json.dumps(embedding_configuration, indent=2) + "\n"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary, "manifold_methods": manifold_methods}, indent=2))


if __name__ == "__main__":
    main()
