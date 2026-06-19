#!/usr/bin/env python3
"""Build an expanded MethaNet MBAG multi-view atlas HTML report.

The atlas consolidates the closed rumen + wetland/MUCC POC core with the
current mangrove/MSM payloads. It is intentionally a MAG/proteome molecular
screening artifact: no sample-level MRV tiers, measured methane flux, or
carbon-crediting decisions are assigned here.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import subprocess
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POC_ESM_DIR = Path(
    "results/blue_catalyst_poc/runs/"
    "apolo_full_20260228_080644_embed_20260305_061952/artifacts"
)
DEFAULT_POC_WAREHOUSE_DIR = Path(
    "results/functional_metagenomics/fgx_662_apollo3_20260612/"
    "cohort_warehouse_poc_magbin_union_20260616_075022"
)
DEFAULT_POC_GLM_DIR = Path(
    "results/contextual_genomics/"
    "glm2_integration_20260616_poc_catchup_20260616_073441"
)
DEFAULT_MSM_ROOT = Path("results/functional_metagenomics/msm_china_2025_20260615")
DEFAULT_MSM_ESM_DIR = Path(
    "results/blue_catalyst_poc/runs/msm_china_2025_esm2_20260616_082112/artifacts"
)
DEFAULT_MSM_GLM_DIR = Path("results/contextual_genomics/glm2_msm_magbin_full_20260615_092737")

COLORS = {
    "rumen": "#d97706",
    "wetland": "#0891b2",
    "mangrove": "#16a34a",
    "context": "#94a3b8",
    "pending": "#f59e0b",
    "ink": "#16202a",
    "muted": "#64748b",
    "grid": "#dbe3ef",
    "panel": "#ffffff",
    "surface": "#f7fafc",
}

CLAIM_BOUNDARY = (
    "MAG/proteome molecular screening and bridge-candidate prioritization; "
    "not final MRV risk scoring, measured methane flux, or carbon-credit approval."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--poc-esm-dir", type=Path, default=DEFAULT_POC_ESM_DIR)
    parser.add_argument("--poc-warehouse-dir", type=Path, default=DEFAULT_POC_WAREHOUSE_DIR)
    parser.add_argument("--poc-glm-dir", type=Path, default=DEFAULT_POC_GLM_DIR)
    parser.add_argument("--msm-root", type=Path, default=DEFAULT_MSM_ROOT)
    parser.add_argument("--msm-esm-dir", type=Path, default=DEFAULT_MSM_ESM_DIR)
    parser.add_argument("--msm-glm-dir", type=Path, default=DEFAULT_MSM_GLM_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output report directory. Defaults to a timestamped results/reports path.",
    )
    parser.add_argument("--top-n-poc", type=int, default=10)
    parser.add_argument("--top-n-mangrove", type=int, default=12)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--graph-node-cap", type=int, default=280)
    return parser.parse_args()


def resolve(root: Path, path: Path | str | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else root / path


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def image_data_uri(path: Path) -> str:
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
        if value is None or value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def short_id(value: Any, width: int = 34) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    keep = max(8, (width - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def source_category(source: Any, ecosystem: Any = "") -> str:
    text = f"{source} {ecosystem}".lower()
    if "mangrove" in text or "msm" in text:
        return "mangrove"
    if "mucc" in text or "wetland" in text:
        return "wetland"
    if "rumen" in text:
        return "rumen"
    return "context"


def source_label(category: str) -> str:
    return {
        "rumen": "Rumen",
        "wetland": "Wetland/MUCC",
        "mangrove": "Mangrove/MSM",
        "context": "ESM-2 context",
    }.get(category, category)


def load_npz_embeddings(path: Path, key_candidates: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    embeddings = np.asarray(arr["embeddings"], dtype="float32")
    key = next((k for k in key_candidates if k in arr.files), None)
    if key is None:
        raise SystemExit(f"No identifier key among {key_candidates} found in {path}")
    df = pd.DataFrame({"proteome_id": arr[key].astype(str)})
    for col in [
        "sample",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "source_group",
        "source_analysis_accession",
        "n_proteins_used",
        "n_valid_proteins_seen",
        "protein_cap_applied",
    ]:
        if col in arr.files:
            df[col] = arr[col]
    if "sample" in df.columns and key != "sample":
        df["sample"] = df["sample"].astype(str)
    return df, embeddings


def load_poc_features(warehouse_dir: Path, glm_dir: Path, esm_dir: Path) -> pd.DataFrame:
    import duckdb

    db = warehouse_dir / "functional_atlas.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    query = """
    select
      d.cohort_run_id,
      d.run_id,
      d.proteome_id,
      d.mag_id,
      d.source_group as source,
      case when d.source_group = 'mucc' then 'wetland' else d.source_group end as ecosystem,
      d.source_tool,
      d.sample,
      d.mag_fasta,
      d.proteome_faa,
      d.source_analysis_accession,
      d.analysis_alias,
      d.mucc_img_like_id,
      d.analysis_unit_type,
      d.mbag_mag_level_include,
      d.assembly_context_include,
      d.claim_scope,
      d.comparability_status,
      d.comparability_reason,
      d.recommended_action,
      d.input_contigs,
      d.input_total_bp,
      d.input_n50_bp,
      d.prodigal_proteins,
      d.gtdb_release,
      d.gtdb_classification,
      d.domain,
      d.phylum,
      d.class,
      d."order",
      d.family,
      d.genus,
      d.species,
      d.qc_tier,
      d.checkm2_completeness,
      d.checkm2_contamination,
      d.gunc_pass,
      coalesce(f.kofam_annotated_gene_fraction, 0) as kofam_annotated_gene_fraction,
      coalesce(f.metabolic_modules_present, 0) as metabolic_modules_present,
      coalesce(f.cazy_family_count, 0) as cazy_family_count,
      coalesce(f.merops_family_count, 0) as merops_family_count,
      coalesce(f.methane_evidence_score, 0) as methane_evidence_score,
      coalesce(f.sulfur_competition_score, 0) as sulfur_competition_score,
      f.absence_interpretation_caveat
    from dim_mag d
    left join feature_mrv_mag_level f using (cohort_run_id, run_id, proteome_id, mag_id)
    """
    poc = con.execute(query).fetchdf()
    con.close()

    glm = read_tsv(glm_dir / "feature_glm_mag_level.tsv")
    if not glm.empty:
        glm["proteome_id"] = glm["proteome_id"].astype(str)
        glm["glm_context_delta"] = (
            pd.to_numeric(glm.get("native_embedding_std_mean"), errors="coerce")
            - pd.to_numeric(glm.get("shuffled_embedding_std_mean"), errors="coerce")
        )
        glm_keep = [
            "proteome_id",
            "native_window_count",
            "shuffled_control_count",
            "all_embeddings_finite",
            "embedding_dim",
            "native_embedding_std_mean",
            "shuffled_embedding_std_mean",
            "glm_context_delta",
            "context_qc_tier",
        ]
        poc = poc.merge(glm[[c for c in glm_keep if c in glm.columns]], on="proteome_id", how="left")

    projection = read_tsv(esm_dir / "embedding_projection_clusters.tsv")
    if not projection.empty:
        projection = projection.rename(columns={"sample": "proteome_id"})
        keep = [
            "proteome_id",
            "cluster",
            "umap_1",
            "umap_2",
            "tsne_1",
            "tsne_2",
            "bridging_score",
            "mixing_coeff",
        ]
        poc = poc.merge(projection[[c for c in keep if c in projection.columns]], on="proteome_id", how="left")

    bridge = read_tsv(esm_dir / "bridge_top_candidates.tsv")
    if not bridge.empty:
        bridge = bridge.rename(columns={"sample": "proteome_id"})
        poc = poc.merge(
            bridge[["proteome_id", "rank", "wetland_projection"]].drop_duplicates("proteome_id"),
            on="proteome_id",
            how="left",
        )

    poc["cohort_label"] = "POC core"
    poc["source_category"] = poc.apply(lambda r: source_category(r["source"], r["ecosystem"]), axis=1)
    poc["has_esm2"] = True
    poc["has_glm2"] = poc["native_window_count"].fillna(0).astype(float) > 0
    poc["has_functional"] = True
    poc["is_poc_core"] = True
    poc["atlas_inclusion_status"] = "poc_core_complete"
    if "claim_scope" not in poc.columns:
        poc["claim_scope"] = ""
    poc["claim_scope"] = poc["claim_scope"].fillna("").replace("", "MAG/proteome molecular screening")
    return poc


def aggregate_msm_glm(msm_glm_dir: Path) -> pd.DataFrame:
    path = msm_glm_dir / "features/glm2_smoke_window_embedding_summary.tsv"
    glm = read_tsv(path)
    if glm.empty:
        return pd.DataFrame(columns=["proteome_id"])
    glm["proteome_id"] = glm["proteome_id"].astype(str)
    glm["is_shuffled"] = glm["window_type"].astype(str).str.contains("shuffled", case=False, na=False)
    glm["embedding_std"] = pd.to_numeric(glm["embedding_std"], errors="coerce")
    glm["token_count"] = pd.to_numeric(glm["token_count"], errors="coerce")
    native = (
        glm[~glm["is_shuffled"]]
        .groupby("proteome_id")
        .agg(
            mag_id=("mag_id", "first"),
            native_window_count=("window_id", "count"),
            native_embedding_std_mean=("embedding_std", "mean"),
            max_token_count=("token_count", "max"),
            native_all_finite=("embedding_finite", lambda x: x.astype(str).str.lower().isin(["true", "1"]).all()),
            model_name=("model_name", "first"),
            model_revision=("model_revision", "first"),
        )
        .reset_index()
    )
    shuffled = (
        glm[glm["is_shuffled"]]
        .groupby("proteome_id")
        .agg(
            shuffled_control_count=("window_id", "count"),
            shuffled_embedding_std_mean=("embedding_std", "mean"),
            shuffled_all_finite=("embedding_finite", lambda x: x.astype(str).str.lower().isin(["true", "1"]).all()),
        )
        .reset_index()
    )
    out = native.merge(shuffled, on="proteome_id", how="outer")
    out["glm_context_delta"] = out["native_embedding_std_mean"] - out["shuffled_embedding_std_mean"]
    out["all_embeddings_finite"] = out["native_all_finite"].fillna(False) & out["shuffled_all_finite"].fillna(False)
    out["has_glm2"] = out["native_window_count"].fillna(0).astype(float).gt(0) & out[
        "shuffled_control_count"
    ].fillna(0).astype(float).gt(0)
    out["embedding_dim"] = 1280
    out["context_qc_tier"] = "two_window_native_plus_shuffled"
    return out


def discover_msm_functional(msm_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = msm_root / "manifests/msm_china_2025_functional_mag_manifest.tsv"
    manifest = read_tsv(manifest_path)
    if manifest.empty:
        raise SystemExit(f"Missing MSM manifest: {manifest_path}")
    manifest["proteome_id"] = manifest["proteome_id"].astype(str)
    per_mag = msm_root / "per_mag"

    selected: dict[str, Path] = {}
    partial: set[str] = set()
    failed: set[str] = set()
    for mag_dir in sorted(p for p in per_mag.glob("*") if p.is_dir()):
        runs = sorted([p for p in mag_dir.iterdir() if p.is_dir()])
        complete_runs = [p for p in runs if (p / "COMPLETE").exists() and (p / "curated/run_record.json").exists()]
        if complete_runs:
            selected[mag_dir.name] = complete_runs[-1]
            continue
        if any((p / "FAILED").exists() for p in runs):
            failed.add(mag_dir.name)
        elif runs:
            partial.add(mag_dir.name)

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        proteome_id = str(item["proteome_id"])
        run_dir = selected.get(proteome_id)
        if run_dir is None:
            run_status = "failed" if proteome_id in failed else "partial" if proteome_id in partial else "not_started"
            status_rows.append(
                {
                    "proteome_id": proteome_id,
                    "mag_id": item.get("mag_id", proteome_id),
                    "functional_status": run_status,
                    "has_functional": False,
                    "run_id": "",
                    "run_dir": "",
                    "has_curated_manifest": False,
                }
            )
            continue

        record = read_json(run_dir / "curated/run_record.json")
        parquet_manifest = read_tsv(run_dir / "curated/parquet_manifest.tsv")
        table_rows = {}
        if not parquet_manifest.empty:
            table_rows = {
                str(r["table"]): safe_int(r["rows"])
                for _, r in parquet_manifest.iterrows()
                if "table" in r
            }
        summary = record.get("summary_metrics", {}) or {}
        qc = record.get("qc", {}) or {}
        tax = record.get("taxonomy", {}) or {}

        row = {
            "cohort_run_id": record.get("cohort_run_id", "msm_china_2025_20260615"),
            "run_id": record.get("run_id", run_dir.name),
            "proteome_id": proteome_id,
            "mag_id": record.get("mag_id") or item.get("mag_id") or proteome_id,
            "source": "msm_china_2025",
            "ecosystem": "mangrove_sediment",
            "domain": tax.get("domain") or item.get("domain") or "unknown",
            "phylum": tax.get("phylum", ""),
            "class": tax.get("class", ""),
            "order": tax.get("order", ""),
            "family": tax.get("family", ""),
            "genus": tax.get("genus", ""),
            "species": tax.get("species", ""),
            "qc_tier": qc.get("qc_tier", ""),
            "checkm2_completeness": safe_float(qc.get("completeness")),
            "checkm2_contamination": safe_float(qc.get("contamination")),
            "gunc_pass": bool(qc.get("gunc_pass", False)),
            "gunc_css": safe_float(qc.get("gunc_css")),
            "input_total_bp": safe_int(summary.get("input_total_bp")),
            "prodigal_proteins": safe_int(summary.get("prodigal_proteins")),
            "kofam_rows": safe_int(summary.get("kofam_rows") or table_rows.get("fact_kofam_hits")),
            "mcycdb_hits": safe_int(summary.get("mcycdb_hits") or table_rows.get("fact_mcycdb_hits")),
            "scycdb_hits": safe_int(summary.get("scycdb_hits") or table_rows.get("fact_scycdb_hits")),
            "dbcan_overview_rows": safe_int(summary.get("dbcan_overview_rows") or table_rows.get("fact_dbcan_hits")),
            "bakta_feature_rows": safe_int(summary.get("bakta_feature_rows") or table_rows.get("fact_bakta_features")),
            "metabolic_hmm_rows": safe_int(table_rows.get("fact_metabolic_hmm_hits")),
            "metabolic_modules_present": safe_int(table_rows.get("fact_metabolic_module_presence")),
            "metabolic_functions_present": safe_int(table_rows.get("fact_metabolic_function_presence")),
            "cazy_family_count": safe_int(table_rows.get("fact_cazy_hits") or summary.get("dbcan_overview_rows")),
            "merops_family_count": safe_int(table_rows.get("fact_merops_hits")),
            "has_functional": True,
            "functional_status": "complete",
            "run_dir": str(run_dir),
            "has_curated_manifest": True,
        }
        row["methane_evidence_score"] = row["mcycdb_hits"] + row["metabolic_hmm_rows"]
        row["sulfur_competition_score"] = row["scycdb_hits"] + row["metabolic_functions_present"]
        denom = max(row["prodigal_proteins"], 1)
        row["kofam_annotated_gene_fraction"] = min(row["kofam_rows"] / denom, 1.0)
        rows.append(row)
        status_rows.append(
            {
                "proteome_id": proteome_id,
                "mag_id": row["mag_id"],
                "functional_status": "complete",
                "has_functional": True,
                "run_id": row["run_id"],
                "run_dir": str(run_dir),
                "has_curated_manifest": True,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def load_msm_features(msm_root: Path, msm_esm_dir: Path, msm_glm_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = read_tsv(msm_root / "manifests/msm_china_2025_functional_mag_manifest.tsv")
    manifest["proteome_id"] = manifest["proteome_id"].astype(str)
    functional, status = discover_msm_functional(msm_root)
    glm = aggregate_msm_glm(msm_glm_dir)
    esm_meta = read_tsv(msm_esm_dir / "embedding_metadata.tsv")
    if "sample" in esm_meta.columns and "proteome_id" not in esm_meta.columns:
        esm_meta = esm_meta.rename(columns={"sample": "proteome_id"})
    esm_meta["proteome_id"] = esm_meta["proteome_id"].astype(str)

    base_cols = [
        c
        for c in [
            "proteome_id",
            "mag_id",
            "source",
            "ecosystem",
            "domain",
            "source_group",
            "n_proteins_used",
            "n_valid_proteins_seen",
            "protein_cap_applied",
        ]
        if c in esm_meta.columns
    ]
    msm = manifest[["proteome_id", "mag_id"]].drop_duplicates("proteome_id").merge(
        esm_meta[base_cols].drop_duplicates("proteome_id"),
        on="proteome_id",
        how="left",
        suffixes=("", "_esm"),
    )
    if "mag_id_esm" in msm.columns:
        msm["mag_id"] = msm["mag_id"].fillna(msm["mag_id_esm"])
        msm = msm.drop(columns=["mag_id_esm"])
    msm = msm.merge(status, on=["proteome_id", "mag_id"], how="left")
    msm = msm.merge(functional.drop(columns=["mag_id"], errors="ignore"), on="proteome_id", how="left", suffixes=("", "_functional"))
    msm = msm.merge(glm.drop(columns=["mag_id"], errors="ignore"), on="proteome_id", how="left")

    msm["source"] = "msm_china_2025"
    msm["ecosystem"] = msm["ecosystem"].fillna("mangrove_sediment")
    msm["cohort_label"] = "Mangrove/MSM"
    msm["source_category"] = "mangrove"
    msm["has_esm2"] = True
    msm["has_glm2"] = msm.get("has_glm2", False).fillna(False).astype(bool)
    msm["has_functional"] = msm["has_functional"].fillna(False).astype(bool)
    msm["atlas_inclusion_status"] = np.where(
        msm["has_functional"],
        "mangrove_multiview_complete",
        "mangrove_esm_glm_only",
    )
    msm["claim_scope"] = "MAG/proteome molecular screening"
    for col in [
        "checkm2_completeness",
        "checkm2_contamination",
        "kofam_annotated_gene_fraction",
        "metabolic_modules_present",
        "cazy_family_count",
        "merops_family_count",
        "methane_evidence_score",
        "sulfur_competition_score",
        "mcycdb_hits",
        "scycdb_hits",
        "dbcan_overview_rows",
        "kofam_rows",
        "metabolic_hmm_rows",
    ]:
        if col not in msm.columns:
            msm[col] = 0
        msm[col] = pd.to_numeric(msm[col], errors="coerce").fillna(0)
    stats = read_json(msm_esm_dir / "embedding_stats.json")
    return msm, status, stats


def build_embedding_context(
    poc_esm_dir: Path,
    msm_esm_dir: Path,
    atlas: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    poc_meta, poc_emb = load_npz_embeddings(poc_esm_dir / "genome_embeddings.npz", ["sample", "proteome_id"])
    poc_meta["cohort_label"] = "POC ESM-2 context"
    msm_meta, msm_emb = load_npz_embeddings(msm_esm_dir / "genome_embeddings.npz", ["proteome_id", "sample"])
    msm_meta["cohort_label"] = "Mangrove/MSM"

    emb_meta = pd.concat([poc_meta, msm_meta], ignore_index=True, sort=False)
    embeddings = np.vstack([poc_emb, msm_emb])
    emb_meta["source_category"] = emb_meta.apply(lambda r: source_category(r.get("source"), r.get("ecosystem")), axis=1)
    status = atlas[["proteome_id", "atlas_inclusion_status", "has_functional", "has_glm2"]].drop_duplicates("proteome_id")
    emb_meta = emb_meta.merge(status, on="proteome_id", how="left")
    emb_meta["atlas_inclusion_status"] = emb_meta["atlas_inclusion_status"].fillna("non_poc_or_unscoped")
    emb_meta["has_functional"] = emb_meta["has_functional"].fillna(False).astype(bool)
    emb_meta["has_glm2"] = emb_meta["has_glm2"].fillna(False).astype(bool)
    emb_meta["has_esm2"] = True

    reduced = PCA(n_components=2, random_state=20260619).fit_transform(normalize(embeddings))
    emb_meta["pca_1"] = reduced[:, 0]
    emb_meta["pca_2"] = reduced[:, 1]

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(emb_meta)), metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    edges: list[dict[str, Any]] = []
    cross_counts = np.zeros(len(emb_meta), dtype=int)
    for i, (ds, js) in enumerate(zip(distances, indices)):
        src_cat = emb_meta.iloc[i]["source_category"]
        kept = 0
        for dist, j in zip(ds, js):
            if i == j:
                continue
            dst_cat = emb_meta.iloc[j]["source_category"]
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

    poc_core_ids = set(atlas.loc[atlas["atlas_inclusion_status"].eq("poc_core_complete"), "proteome_id"])
    msm_ids = set(atlas.loc[atlas["source_category"].eq("mangrove"), "proteome_id"])
    poc_idx = emb_meta.index[emb_meta["proteome_id"].isin(poc_core_ids)].to_numpy()
    msm_idx = emb_meta.index[emb_meta["proteome_id"].isin(msm_ids)].to_numpy()
    emb_norm = normalize(embeddings)
    emb_meta["nearest_poc_similarity"] = 0.0
    emb_meta["nearest_mangrove_similarity"] = 0.0
    if len(poc_idx) and len(msm_idx):
        nn_poc = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[poc_idx])
        d_poc, i_poc = nn_poc.kneighbors(embeddings)
        emb_meta["nearest_poc_similarity"] = 1.0 - d_poc[:, 0]
        nn_msm = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[msm_idx])
        d_msm, _ = nn_msm.kneighbors(embeddings)
        emb_meta["nearest_mangrove_similarity"] = 1.0 - d_msm[:, 0]

    return emb_meta, edge_df, embeddings


def add_report_metrics(atlas: pd.DataFrame, emb_meta: pd.DataFrame) -> pd.DataFrame:
    atlas = atlas.merge(
        emb_meta[
            [
                "proteome_id",
                "pca_1",
                "pca_2",
                "cross_domain_neighbor_count",
                "cross_domain_neighbor_fraction",
                "nearest_poc_similarity",
                "nearest_mangrove_similarity",
            ]
        ].drop_duplicates("proteome_id"),
        on="proteome_id",
        how="left",
    )
    atlas["source_display"] = atlas["source_category"].map(source_label)
    atlas["qc_confidence_raw"] = (
        pd.to_numeric(atlas["checkm2_completeness"], errors="coerce").fillna(0)
        - 4 * pd.to_numeric(atlas["checkm2_contamination"], errors="coerce").fillna(0)
    ).clip(lower=0)
    atlas["esm_bridge_norm"] = norm01(
        pd.to_numeric(atlas.get("mixing_coeff", 0), errors="coerce").fillna(0)
        + pd.to_numeric(atlas.get("cross_domain_neighbor_fraction", 0), errors="coerce").fillna(0)
        + pd.to_numeric(atlas.get("nearest_poc_similarity", 0), errors="coerce").fillna(0) * atlas["source_category"].eq("mangrove").astype(float)
    )
    atlas["glm_context_norm"] = norm01(pd.to_numeric(atlas.get("glm_context_delta", 0), errors="coerce").fillna(0))
    atlas["methane_signal_norm"] = norm01(np.log1p(pd.to_numeric(atlas["methane_evidence_score"], errors="coerce").fillna(0)))
    atlas["sulfur_signal_norm"] = norm01(np.log1p(pd.to_numeric(atlas["sulfur_competition_score"], errors="coerce").fillna(0)))
    atlas["substrate_signal_norm"] = norm01(np.log1p(pd.to_numeric(atlas["cazy_family_count"], errors="coerce").fillna(0)))
    atlas["broad_function_norm"] = norm01(
        np.log1p(
            pd.to_numeric(atlas.get("kofam_rows", 0), errors="coerce").fillna(0)
            + pd.to_numeric(atlas.get("metabolic_modules_present", 0), errors="coerce").fillna(0)
            + pd.to_numeric(atlas.get("merops_family_count", 0), errors="coerce").fillna(0)
        )
    )
    atlas["qc_signal_norm"] = norm01(atlas["qc_confidence_raw"])
    atlas["atlas_review_score"] = (
        0.22 * atlas["esm_bridge_norm"]
        + 0.16 * atlas["glm_context_norm"]
        + 0.20 * atlas["methane_signal_norm"]
        + 0.14 * atlas["sulfur_signal_norm"]
        + 0.10 * atlas["substrate_signal_norm"]
        + 0.08 * atlas["broad_function_norm"]
        + 0.10 * atlas["qc_signal_norm"]
    )
    atlas.loc[~atlas["has_functional"], "atlas_review_score"] = np.nan
    return atlas


def build_candidate_cards(atlas: pd.DataFrame, top_n_poc: int, top_n_mangrove: int) -> pd.DataFrame:
    poc = atlas[atlas["source_category"].isin(["rumen", "wetland"])].copy()
    poc_top = poc[poc.get("rank").notna()].sort_values("rank").head(top_n_poc).copy()
    poc_top["candidate_set"] = "POC ESM-2 bridge candidate"

    msm = atlas[
        atlas["source_category"].eq("mangrove")
        & atlas["has_functional"]
        & atlas["has_glm2"]
    ].copy()
    msm_top = msm.sort_values(["atlas_review_score", "nearest_poc_similarity"], ascending=False).head(top_n_mangrove).copy()
    msm_top["candidate_set"] = "Mangrove nearest-neighborhood candidate"
    msm_top["rank"] = np.arange(1, len(msm_top) + 1)

    cards = pd.concat([poc_top, msm_top], ignore_index=True, sort=False)
    cards["card_id"] = cards["candidate_set"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_") + "_" + cards["rank"].fillna(0).astype(int).astype(str)
    cards["allowed_claim_wording"] = cards.apply(
        lambda r: (
            "Reviewable MAG/proteome bridge hypothesis with ESM-2, gLM2, functional, QC, and taxonomy evidence."
            if bool(r.get("has_functional")) and bool(r.get("has_glm2"))
            else "Candidate remains incomplete and should be interpreted only as a latent/context signal."
        ),
        axis=1,
    )
    cards["blocking_gap"] = cards.apply(
        lambda r: "sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, and flux/process validation",
        axis=1,
    )
    cards["next_validation_action"] = cards.apply(
        lambda r: "inspect marker neighborhoods, source-aware nulls, taxonomy/phylogeny context, and sample metadata linkage",
        axis=1,
    )
    return cards


def build_evidence_ledger(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "cohort": "POC core",
            "total": summary["poc_core_total"],
            "esm2": summary["poc_core_total"],
            "glm2": summary["poc_core_total"],
            "functional": summary["poc_core_total"],
            "multiview": summary["poc_core_total"],
            "pending_function": 0,
        },
        {
            "cohort": "Mangrove/MSM",
            "total": summary["msm_total"],
            "esm2": summary["msm_esm2"],
            "glm2": summary["msm_glm2"],
            "functional": summary["msm_functional"],
            "multiview": summary["msm_multiview"],
            "pending_function": summary["msm_total"] - summary["msm_functional"],
        },
    ]


def js_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def frame_records(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> list[dict[str, Any]]:
    use = df[[c for c in cols if c in df.columns]].copy()
    if max_rows is not None:
        use = use.head(max_rows)
    return json.loads(use.replace({np.nan: None}).to_json(orient="records"))


def build_d3_payloads(
    atlas: pd.DataFrame,
    emb_meta: pd.DataFrame,
    edge_df: pd.DataFrame,
    cards: pd.DataFrame,
    summary: dict[str, Any],
    graph_node_cap: int,
) -> dict[str, Any]:
    card_ids = set(cards["proteome_id"].astype(str))
    neighbor_ids = set(card_ids)
    card_edge = edge_df[edge_df["source"].isin(card_ids) | edge_df["target"].isin(card_ids)]
    neighbor_ids.update(card_edge["source"].head(120).astype(str))
    neighbor_ids.update(card_edge["target"].head(120).astype(str))

    mangrove_top = (
        atlas[atlas["source_category"].eq("mangrove") & atlas["has_functional"]]
        .sort_values(["nearest_poc_similarity", "atlas_review_score"], ascending=False)
        .head(90)["proteome_id"]
        .astype(str)
    )
    neighbor_ids.update(mangrove_top)
    selected_ids = set(list(neighbor_ids)[:graph_node_cap])
    graph_nodes = emb_meta[emb_meta["proteome_id"].isin(selected_ids)].copy()
    graph_nodes = graph_nodes.merge(
        atlas[
            [
                "proteome_id",
                "mag_id",
                "source_display",
                "atlas_inclusion_status",
                "atlas_review_score",
                "methane_evidence_score",
                "sulfur_competition_score",
                "cazy_family_count",
                "checkm2_completeness",
                "checkm2_contamination",
                "qc_tier",
                "phylum",
                "class",
                "glm_context_delta",
            ]
        ],
        on="proteome_id",
        how="left",
    )
    graph_nodes["label"] = graph_nodes["proteome_id"].map(lambda x: short_id(x, 30))
    graph_edges = edge_df[
        edge_df["source"].isin(graph_nodes["proteome_id"]) & edge_df["target"].isin(graph_nodes["proteome_id"])
    ].copy()
    graph_edges = graph_edges.sort_values(["cross_domain", "similarity"], ascending=[False, False]).head(1200)

    matrix_metrics = [
        "esm_bridge_norm",
        "glm_context_norm",
        "methane_signal_norm",
        "sulfur_signal_norm",
        "substrate_signal_norm",
        "broad_function_norm",
        "qc_signal_norm",
    ]
    matrix_df = cards.copy()
    matrix_payload = []
    for _, row in matrix_df.iterrows():
        for metric in matrix_metrics:
            matrix_payload.append(
                {
                    "proteome_id": row["proteome_id"],
                    "candidate_set": row.get("candidate_set", ""),
                    "rank": int(row.get("rank") or 0),
                    "label": short_id(row["proteome_id"], 28),
                    "metric": metric,
                    "value": safe_float(row.get(metric), 0.0),
                }
            )

    completed_msm = atlas[atlas["source_category"].eq("mangrove") & atlas["has_functional"]].copy()
    scatter_cols = [
        "proteome_id",
        "mag_id",
        "phylum",
        "class",
        "qc_tier",
        "methane_evidence_score",
        "sulfur_competition_score",
        "cazy_family_count",
        "checkm2_completeness",
        "checkm2_contamination",
        "glm_context_delta",
        "nearest_poc_similarity",
        "atlas_review_score",
    ]

    cats = ["rumen", "wetland", "mangrove"]
    idx = {c: i for i, c in enumerate(cats)}
    chord_matrix = [[0 for _ in cats] for _ in cats]
    strong_edges = edge_df[edge_df["similarity"] >= edge_df["similarity"].quantile(0.92)]
    for _, row in strong_edges.iterrows():
        a = row["source_category"]
        b = row["target_category"]
        if a in idx and b in idx and a != b:
            chord_matrix[idx[a]][idx[b]] += 1

    cards_payload = frame_records(
        cards,
        [
            "card_id",
            "candidate_set",
            "rank",
            "proteome_id",
            "mag_id",
            "source_display",
            "domain",
            "phylum",
            "class",
            "qc_tier",
            "checkm2_completeness",
            "checkm2_contamination",
            "glm_context_delta",
            "methane_evidence_score",
            "sulfur_competition_score",
            "cazy_family_count",
            "nearest_poc_similarity",
            "atlas_review_score",
            "allowed_claim_wording",
            "blocking_gap",
            "next_validation_action",
        ],
    )

    return {
        "ledger": build_evidence_ledger(summary),
        "graph": {
            "nodes": frame_records(
                graph_nodes,
                [
                    "proteome_id",
                    "label",
                    "source_category",
                    "source_display",
                    "cohort_label",
                    "atlas_inclusion_status",
                    "has_functional",
                    "has_glm2",
                    "pca_1",
                    "pca_2",
                    "atlas_review_score",
                    "methane_evidence_score",
                    "sulfur_competition_score",
                    "cazy_family_count",
                    "checkm2_completeness",
                    "checkm2_contamination",
                    "qc_tier",
                    "phylum",
                    "class",
                    "glm_context_delta",
                ],
            ),
            "links": frame_records(
                graph_edges,
                [
                    "source",
                    "target",
                    "source_category",
                    "target_category",
                    "similarity",
                    "cosine_distance",
                    "cross_domain",
                    "reciprocal",
                    "rank",
                ],
            ),
        },
        "matrix": {
            "metrics": matrix_metrics,
            "records": matrix_payload,
        },
        "mangrove": frame_records(completed_msm.sort_values("atlas_review_score", ascending=False), scatter_cols),
        "radial": {"categories": cats, "matrix": chord_matrix},
        "cards": cards_payload,
    }


def save_json_payloads(payloads: dict[str, Any], data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, payload in payloads.items():
        path = data_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        out[name] = path
    return out


def plot_evidence_ledger(payload: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=COLORS["surface"])
    ax.set_facecolor(COLORS["panel"])
    cohorts = [p["cohort"] for p in payload]
    y = np.arange(len(cohorts))
    totals = np.array([p["total"] for p in payload], dtype=float)
    colors = [COLORS["rumen"], COLORS["wetland"], COLORS["mangrove"], COLORS["pending"]]
    labels = ["ESM-2", "gLM2", "Functional", "Function pending"]
    vals = [
        np.array([p["esm2"] for p in payload], dtype=float),
        np.array([p["glm2"] for p in payload], dtype=float),
        np.array([p["functional"] for p in payload], dtype=float),
        np.array([p["pending_function"] for p in payload], dtype=float),
    ]
    left = np.zeros(len(payload))
    for v, c, lab in zip(vals, colors, labels):
        widths = np.divide(v, totals, out=np.zeros_like(v), where=totals > 0)
        ax.barh(y, widths, left=left, height=0.45, color=c, label=lab)
        left += widths
    ax.set_yticks(y, cohorts)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of cohort")
    ax.set_title("Expanded atlas evidence ledger", loc="left", fontsize=14, weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    for i, p in enumerate(payload):
        ax.text(1.01, i, f"{p['multiview']:,}/{p['total']:,} multiview", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bridge_graph(payload: dict[str, Any], path: Path) -> Path:
    nodes = pd.DataFrame(payload["nodes"])
    links = pd.DataFrame(payload["links"])
    fig, ax = plt.subplots(figsize=(9.4, 7.0), facecolor=COLORS["surface"])
    ax.set_facecolor("#f8fbfd")
    if not links.empty and not nodes.empty:
        pos = nodes.set_index("proteome_id")[["pca_1", "pca_2"]].to_dict("index")
        for _, e in links[links["cross_domain"]].head(400).iterrows():
            if e["source"] in pos and e["target"] in pos:
                a, b = pos[e["source"]], pos[e["target"]]
                ax.plot([a["pca_1"], b["pca_1"]], [a["pca_2"], b["pca_2"]], color="#94a3b8", alpha=0.18, lw=0.7)
    for cat, sub in nodes.groupby("source_category"):
        ax.scatter(
            sub["pca_1"],
            sub["pca_2"],
            s=30 + 90 * pd.to_numeric(sub.get("atlas_review_score", 0), errors="coerce").fillna(0),
            color=COLORS.get(cat, COLORS["context"]),
            edgecolor="white",
            linewidth=0.4,
            alpha=0.82,
            label=source_label(cat),
        )
    ax.set_title("High-dimensional ESM-2 kNN bridge-neighborhood graph", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("PCA 1 of ESM-2 embeddings")
    ax.set_ylabel("PCA 2 of ESM-2 embeddings")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_signature_matrix(payload: dict[str, Any], path: Path) -> Path:
    records = pd.DataFrame(payload["records"])
    if records.empty:
        return path
    matrix = records.pivot_table(index="label", columns="metric", values="value", aggfunc="first").fillna(0)
    matrix = matrix[payload["metrics"]]
    fig, ax = plt.subplots(figsize=(10.2, max(5.0, 0.32 * len(matrix))), facecolor=COLORS["surface"])
    im = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), [c.replace("_", "\n") for c in matrix.columns], fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else "#172033")
    ax.set_title("Candidate signature matrix", loc="left", fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mangrove_patterns(records: list[dict[str, Any]], path: Path) -> Path:
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9.5, 6.0), facecolor=COLORS["surface"])
    if not df.empty:
        x = np.log1p(pd.to_numeric(df["methane_evidence_score"], errors="coerce").fillna(0))
        y = np.log1p(pd.to_numeric(df["sulfur_competition_score"], errors="coerce").fillna(0))
        s = 18 + 34 * np.sqrt(pd.to_numeric(df["cazy_family_count"], errors="coerce").fillna(0) + 1)
        c = pd.to_numeric(df["nearest_poc_similarity"], errors="coerce").fillna(0)
        sc = ax.scatter(x, y, s=s, c=c, cmap="YlGnBu", alpha=0.72, edgecolor="white", linewidth=0.35)
        fig.colorbar(sc, ax=ax, label="nearest POC ESM-2 similarity")
    ax.set_title("Mangrove completed MAG molecular pattern space", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("methane evidence, log1p")
    ax.set_ylabel("sulfur context, log1p")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_radial(payload: dict[str, Any], path: Path) -> Path:
    cats = payload["categories"]
    mat = np.array(payload["matrix"], dtype=float)
    totals = mat.sum(axis=1) + mat.sum(axis=0)
    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw={"projection": "polar"}, facecolor=COLORS["surface"])
    ax.set_axis_off()
    starts = np.linspace(0, 2 * np.pi, len(cats), endpoint=False)
    width = 2 * np.pi / len(cats) * 0.72
    for start, cat, total in zip(starts, cats, totals):
        ax.bar(start, 1, width=width, bottom=0.72, color=COLORS.get(cat, COLORS["context"]), alpha=0.86)
        ax.text(start, 1.88, f"{source_label(cat)}\n{int(total):,} links", ha="center", va="center", fontsize=9)
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if i >= j or mat[i, j] + mat[j, i] == 0:
                continue
            theta = np.linspace(starts[i], starts[j], 60)
            r = 0.5 + 0.1 * np.sin(np.linspace(0, np.pi, 60))
            ax.plot(theta, r, color="#334155", alpha=min(0.75, 0.08 + (mat[i, j] + mat[j, i]) / max(mat.sum(), 1)), lw=1.2)
    ax.set_title("Evidence-qualified cross-domain kNN bridge links", fontsize=13, weight="bold", pad=24)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_fallbacks(payloads: dict[str, Any], figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "evidence_ledger": plot_evidence_ledger(payloads["ledger"], figure_dir / "fallback_01_evidence_ledger.png"),
        "bridge_graph": plot_bridge_graph(payloads["graph"], figure_dir / "fallback_02_bridge_graph.png"),
        "signature_matrix": plot_signature_matrix(payloads["matrix"], figure_dir / "fallback_03_signature_matrix.png"),
        "mangrove_patterns": plot_mangrove_patterns(payloads["mangrove"], figure_dir / "fallback_04_mangrove_patterns.png"),
        "radial_bridge": plot_radial(payloads["radial"], figure_dir / "fallback_05_radial_bridge.png"),
    }


def fetch_d3(js_dir: Path) -> tuple[Path, str]:
    js_dir.mkdir(parents=True, exist_ok=True)
    path = js_dir / "d3.v7.min.js"
    if not path.exists() or path.stat().st_size < 100_000:
        with urlopen("https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js", timeout=30) as handle:
            path.write_bytes(handle.read())
    return path, path.read_text()


def html_table(df: pd.DataFrame, cols: list[str], max_rows: int = 16) -> str:
    use = df[[c for c in cols if c in df.columns]].head(max_rows).copy()
    if use.empty:
        return "<p>No rows available.</p>"
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in use.columns)
    rows = []
    for _, row in use.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row.fillna("").tolist())
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_html(
    output_dir: Path,
    summary: dict[str, Any],
    payloads: dict[str, Any],
    fallbacks: dict[str, Path],
    cards: pd.DataFrame,
    gap_register: pd.DataFrame,
    d3_source: str,
) -> str:
    figure_uri = {name: image_data_uri(path) for name, path in fallbacks.items()}
    generated = summary["generated_at_utc"]
    metric_cards = "\n".join(
        [
            f"<div class='metric'><b>{summary['multiview_complete']:,}</b><span>multiview-complete MAGs now explorable</span></div>",
            f"<div class='metric'><b>{summary['poc_core_total']:,}/{summary['poc_core_total']:,}</b><span>POC rumen + wetland/MUCC complete</span></div>",
            f"<div class='metric'><b>{summary['msm_functional']:,}/{summary['msm_total']:,}</b><span>mangrove functional MAGs complete so far</span></div>",
            f"<div class='metric'><b>{summary['msm_esm2']:,}/{summary['msm_total']:,}</b><span>mangrove ESM-2 embeddings complete</span></div>",
            f"<div class='metric'><b>{summary['msm_glm2']:,}/{summary['msm_total']:,}</b><span>mangrove gLM2 contextual layer complete</span></div>",
            f"<div class='metric'><b>{summary['graph_node_count']:,}</b><span>D3 bridge-neighborhood nodes in interactive graph</span></div>",
        ]
    )
    card_table = html_table(
        cards,
        [
            "candidate_set",
            "rank",
            "proteome_id",
            "source_display",
            "phylum",
            "class",
            "methane_evidence_score",
            "sulfur_competition_score",
            "glm_context_delta",
            "nearest_poc_similarity",
            "atlas_review_score",
        ],
        24,
    )
    gap_table = html_table(gap_register, list(gap_register.columns), 12)
    payload_json = js_json(payloads)

    css = """
    :root{--ink:#16202a;--muted:#617084;--panel:#ffffff;--surface:#f6fafb;--line:#dce6ef;--rumen:#d97706;--wetland:#0891b2;--mangrove:#16a34a;--gold:#f59e0b}
    *{box-sizing:border-box} body{margin:0;background:var(--surface);color:var(--ink);font-family:Inter,Aptos,Segoe UI,Arial,sans-serif;line-height:1.48}
    header{padding:56px 7vw 34px;background:linear-gradient(135deg,#061a22,#0b3b43 52%,#0d513e);color:white}
    .eyebrow{letter-spacing:.12em;text-transform:uppercase;color:#a7f3d0;font-size:12px;font-weight:700}
    h1{font-size:clamp(34px,5vw,62px);line-height:1.02;margin:.35em 0 .25em;max-width:1100px}
    h2{font-size:26px;margin:0 0 12px} h3{font-size:17px;margin:20px 0 8px}
    .subtitle{max-width:960px;font-size:18px;color:#d8fff3}.claim{display:inline-block;margin-top:18px;border:1px solid #7dd3fc;padding:9px 12px;border-radius:999px;color:#e0f7ff}
    main{max-width:1240px;margin:auto;padding:28px 24px 70px}.section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 10px 30px rgba(15,23,42,.05)}
    .metric-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:16px}.metric{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px}.metric b{display:block;font-size:28px}.metric span{font-size:12px;color:var(--muted)}
    .grid2{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}.viz{min-height:430px;border:1px solid var(--line);border-radius:12px;background:#fbfdff;position:relative;overflow:hidden}.viz.small{min-height:320px}
    .side-card{border:1px solid var(--line);border-radius:12px;background:#f8fafc;padding:16px;min-height:430px}.side-card .muted{font-size:12px}
    .tooltip{position:absolute;pointer-events:none;background:#0f172a;color:white;padding:8px 10px;border-radius:8px;font-size:12px;max-width:300px;opacity:0;z-index:20}
    .legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
    table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}th{background:#f1f5f9;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
    .fallback{margin-top:12px}.fallback img{max-width:100%;border:1px solid var(--line);border-radius:10px}.note{color:var(--muted)}.warn{background:#fff7ed;border-left:4px solid var(--gold);padding:12px;border-radius:10px}
    .pill{display:inline-block;padding:5px 8px;border:1px solid var(--line);border-radius:999px;margin:4px 4px 0 0;color:#334155;background:#f8fafc;font-size:12px}
    @media (max-width:980px){.metric-grid{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
    """

    js = """
    const ATLAS = __PAYLOAD__;
    const COLORS = {rumen:'#d97706', wetland:'#0891b2', mangrove:'#16a34a', context:'#94a3b8', pending:'#f59e0b'};
    const fmt = d3.format(',');
    function tipText(d){return `<b>${d.label || d.proteome_id}</b><br>${d.source_display || d.source_category || ''}<br>status: ${d.atlas_inclusion_status || ''}<br>methane: ${fmt(d.methane_evidence_score || 0)} · sulfur: ${fmt(d.sulfur_competition_score || 0)}<br>QC: ${(d.checkm2_completeness || 0).toFixed(1)}% / ${(d.checkm2_contamination || 0).toFixed(1)}%`; }
    function tooltip(){ return d3.select('body').append('div').attr('class','tooltip'); }
    const cardMap = new Map((ATLAS.cards || []).map(d => [d.proteome_id, d]));
    function updateCard(id){
      const d = cardMap.get(id) || (ATLAS.graph.nodes || []).find(x => x.proteome_id === id);
      const box = d3.select('#candidate-card');
      if(!d){ return; }
      box.html(`<h3>${d.candidate_set || 'Atlas node'}</h3>
        <p><b>${d.proteome_id}</b></p>
        <p class='muted'>${d.source_display || d.source_category || ''} · ${d.domain || ''} · ${d.phylum || ''} · ${d.class || ''}</p>
        <p><b>Methane signal:</b> ${fmt(d.methane_evidence_score || 0)}<br>
        <b>Sulfur context:</b> ${fmt(d.sulfur_competition_score || 0)}<br>
        <b>gLM2 context delta:</b> ${Number(d.glm_context_delta || 0).toFixed(2)}<br>
        <b>Nearest POC similarity:</b> ${Number(d.nearest_poc_similarity || 0).toFixed(3)}<br>
        <b>Review score:</b> ${Number(d.atlas_review_score || 0).toFixed(3)}</p>
        <p><b>Allowed claim:</b> ${d.allowed_claim_wording || 'MAG/proteome molecular screening only.'}</p>
        <p class='muted'><b>Blocks final MRV:</b> ${d.blocking_gap || 'sample mapping, abundance, environment, uncertainty, and validation.'}</p>`);
    }
    function renderLedger(){
      const data = ATLAS.ledger, el = d3.select('#ledger'), w = el.node().clientWidth, h = 340, m = {t:20,r:130,b:40,l:140};
      const svg = el.append('svg').attr('viewBox',[0,0,w,h]); const x = d3.scaleLinear().domain([0,1]).range([m.l,w-m.r]);
      const y = d3.scaleBand().domain(data.map(d=>d.cohort)).range([m.t,h-m.b]).padding(.35);
      const keys = ['esm2','glm2','functional','pending_function'], labs = {esm2:'ESM-2',glm2:'gLM2',functional:'Functional',pending_function:'Function pending'};
      const cols = {esm2:'#0891b2',glm2:'#7c3aed',functional:'#16a34a',pending_function:'#f59e0b'};
      data.forEach(d => { let left = 0; keys.forEach(k => { const frac=(d[k]||0)/d.total; svg.append('rect').attr('x',x(left)).attr('y',y(d.cohort)).attr('width',Math.max(0,x(left+frac)-x(left))).attr('height',y.bandwidth()).attr('fill',cols[k]).attr('opacity',.88); if(frac>.1){svg.append('text').attr('x',x(left+frac/2)).attr('y',y(d.cohort)+y.bandwidth()/2+4).attr('text-anchor','middle').attr('font-size',11).attr('fill','white').text(labs[k]);} left+=frac; }); svg.append('text').attr('x',w-m.r+10).attr('y',y(d.cohort)+y.bandwidth()/2+4).attr('font-size',12).text(`${fmt(d.multiview)}/${fmt(d.total)} multiview`); });
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x).tickFormat(d3.format('.0%'))); svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y));
    }
    function renderGraph(){
      const data = JSON.parse(JSON.stringify(ATLAS.graph)); const el = d3.select('#bridge-graph'), w = el.node().clientWidth, h = 520; const tip = tooltip();
      const svg = el.append('svg').attr('viewBox',[0,0,w,h]); const link = svg.append('g').selectAll('line').data(data.links).join('line').attr('stroke',d=>d.cross_domain?'#475569':'#cbd5e1').attr('stroke-width',d=>d.cross_domain?1.2:.5).attr('opacity',d=>d.cross_domain?.34:.12);
      const node = svg.append('g').selectAll('circle').data(data.nodes).join('circle').attr('r',d=>5+9*Math.max(0,Number(d.atlas_review_score||0))).attr('fill',d=>COLORS[d.source_category]||COLORS.context).attr('stroke','white').attr('stroke-width',1).style('cursor','pointer').on('mouseover',(e,d)=>tip.style('opacity',1).html(tipText(d))).on('mousemove',e=>tip.style('left',`${e.pageX+12}px`).style('top',`${e.pageY+12}px`)).on('mouseout',()=>tip.style('opacity',0)).on('click',(e,d)=>updateCard(d.proteome_id));
      const label = svg.append('g').selectAll('text').data(data.nodes.filter(d=>cardMap.has(d.proteome_id))).join('text').text(d=>d.label).attr('font-size',9).attr('fill','#334155');
      const sim = d3.forceSimulation(data.nodes).force('link',d3.forceLink(data.links).id(d=>d.proteome_id).distance(d=>d.cross_domain?70:34).strength(.18)).force('charge',d3.forceManyBody().strength(-65)).force('center',d3.forceCenter(w/2,h/2)).force('collision',d3.forceCollide(10)).on('tick',()=>{link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y); node.attr('cx',d=>d.x).attr('cy',d=>d.y); label.attr('x',d=>d.x+9).attr('y',d=>d.y+3);});
    }
    function renderHeatmap(){
      const rec = ATLAS.matrix.records, metrics = ATLAS.matrix.metrics, rows = Array.from(new Set(rec.map(d=>d.label))), el=d3.select('#signature-matrix'), w=el.node().clientWidth, cellH=22, h=80+rows.length*cellH, m={t:28,r:20,b:88,l:220};
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]); const x=d3.scaleBand().domain(metrics).range([m.l,w-m.r]).padding(.04), y=d3.scaleBand().domain(rows).range([m.t,h-m.b]).padding(.04), c=d3.scaleSequential(d3.interpolateViridis).domain([0,1]);
      svg.selectAll('rect').data(rec).join('rect').attr('x',d=>x(d.metric)).attr('y',d=>y(d.label)).attr('width',x.bandwidth()).attr('height',y.bandwidth()).attr('fill',d=>c(d.value)).style('cursor','pointer').on('click',(e,d)=>updateCard(d.proteome_id)).append('title').text(d=>`${d.label}\\n${d.metric}: ${d.value.toFixed(2)}`);
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x)).selectAll('text').attr('transform','rotate(-36)').style('text-anchor','end'); svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y));
    }
    function renderMangrove(){
      const data=ATLAS.mangrove, el=d3.select('#mangrove-scatter'), w=el.node().clientWidth, h=480, m={t:18,r:28,b:54,l:62}, tip=tooltip();
      const svg=el.append('svg').attr('viewBox',[0,0,w,h]); const x=d3.scaleLinear().domain(d3.extent(data,d=>Math.log1p(d.methane_evidence_score||0))).nice().range([m.l,w-m.r]); const y=d3.scaleLinear().domain(d3.extent(data,d=>Math.log1p(d.sulfur_competition_score||0))).nice().range([h-m.b,m.t]); const r=d3.scaleSqrt().domain(d3.extent(data,d=>d.cazy_family_count||0)).range([3,16]); const col=d3.scaleSequential(d3.interpolateYlGnBu).domain(d3.extent(data,d=>d.nearest_poc_similarity||0));
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x)); svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y)); svg.append('text').attr('x',w/2).attr('y',h-12).attr('text-anchor','middle').text('methane evidence log1p'); svg.append('text').attr('transform','rotate(-90)').attr('x',-h/2).attr('y',18).attr('text-anchor','middle').text('sulfur context log1p');
      svg.selectAll('circle').data(data).join('circle').attr('cx',d=>x(Math.log1p(d.methane_evidence_score||0))).attr('cy',d=>y(Math.log1p(d.sulfur_competition_score||0))).attr('r',d=>r(d.cazy_family_count||0)).attr('fill',d=>col(d.nearest_poc_similarity||0)).attr('stroke','white').attr('stroke-width',.6).attr('opacity',.78).style('cursor','pointer').on('mouseover',(e,d)=>tip.style('opacity',1).html(tipText({...d,label:d.proteome_id,source_display:'Mangrove/MSM'}))).on('mousemove',e=>tip.style('left',`${e.pageX+12}px`).style('top',`${e.pageY+12}px`)).on('mouseout',()=>tip.style('opacity',0)).on('click',(e,d)=>updateCard(d.proteome_id));
    }
    function renderRadial(){
      const el=d3.select('#radial'), w=el.node().clientWidth, h=460, outer=Math.min(w,h)/2-44, inner=outer-18; const svg=el.append('svg').attr('viewBox',[0,0,w,h]).append('g').attr('transform',`translate(${w/2},${h/2})`);
      const chord=d3.chord().padAngle(.08).sortSubgroups(d3.descending)(ATLAS.radial.matrix); const arc=d3.arc().innerRadius(inner).outerRadius(outer); const ribbon=d3.ribbon().radius(inner); const cats=ATLAS.radial.categories;
      svg.append('g').selectAll('path').data(chord.groups).join('path').attr('d',arc).attr('fill',d=>COLORS[cats[d.index]]).attr('stroke','white');
      svg.append('g').attr('fill-opacity',.45).selectAll('path').data(chord).join('path').attr('d',ribbon).attr('fill',d=>COLORS[cats[d.source.index]]).attr('stroke','#fff').append('title').text(d=>`${cats[d.source.index]} to ${cats[d.target.index]}: ${d.source.value}`);
      svg.append('g').selectAll('text').data(chord.groups).join('text').attr('dy','.35em').attr('transform',d=>{const a=(d.startAngle+d.endAngle)/2; return `rotate(${a*180/Math.PI-90}) translate(${outer+16}) ${a>Math.PI?'rotate(180)':''}`}).attr('text-anchor',d=>((d.startAngle+d.endAngle)/2)>Math.PI?'end':'start').text(d=>cats[d.index]);
    }
    renderLedger(); renderGraph(); renderHeatmap(); renderMangrove(); renderRadial(); updateCard((ATLAS.cards[0]||{}).proteome_id);
    """.replace("__PAYLOAD__", payload_json)

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MethaNet Expanded MBAG Multi-View Atlas</title>
  <style>{css}</style>
</head>
<body>
<header>
  <div class="eyebrow">MethaNet Bridge Attestation Graph · Expanded Multi-View Atlas</div>
  <h1>Rumen, wetland, and mangrove molecular intelligence in one bridge-evidence atlas</h1>
  <p class="subtitle">A D3-powered MAG/proteome atlas integrating high-dimensional ESM-2 neighborhood evidence, functional annotations, gLM2 genomic context, QC, taxonomy, and explicit claim boundaries.</p>
  <div class="claim">{html.escape(CLAIM_BOUNDARY)}</div>
</header>
<main>
  <section class="section">
    <h2>Executive Summary</h2>
    <p><b>The expanded atlas now joins the closed POC core with the computed mangrove payloads available so far.</b> The multiview-complete analytical layer contains {summary['multiview_complete']:,} MAG/proteome units: {summary['poc_core_total']:,} POC rumen + wetland/MUCC units and {summary['msm_multiview']:,} mangrove/MSM units with ESM-2, gLM2, and functional evidence.</p>
    <p><b>Mangrove is no longer just a future lane.</b> All {summary['msm_total']:,} mangrove proteomes have ESM-2 embeddings and gLM2 context; {summary['msm_functional']:,} have completed functional outputs in this snapshot. The remaining {summary['msm_total'] - summary['msm_functional']:,} are preserved as function-pending rows instead of being hidden.</p>
    <div class="metric-grid">{metric_cards}</div>
  </section>

  <section class="section">
    <h2>Atlas Evidence Ledger</h2>
    <p class="note">This panel separates data availability from biological interpretation. ESM-2 and gLM2 are complete for the mangrove payload; functional evidence is computed-so-far and will expand as active arrays finish.</p>
    <div id="ledger" class="viz small"></div>
    <details class="fallback"><summary>Static fallback</summary><img src="{figure_uri['evidence_ledger']}" alt="Evidence ledger fallback"></details>
  </section>

  <section class="section">
    <h2>High-Dimensional Bridge Evidence Graph</h2>
    <p>The bridge graph is built from cosine kNN relationships in the original 1,280-dimensional ESM-2 proteome embedding space. UMAP/t-SNE-style projections are deliberately not used as the bridge-calling substrate.</p>
    <div class="legend"><span><i class="dot" style="background:var(--rumen)"></i>Rumen</span><span><i class="dot" style="background:var(--wetland)"></i>Wetland/MUCC</span><span><i class="dot" style="background:var(--mangrove)"></i>Mangrove/MSM</span></div>
    <div class="grid2">
      <div id="bridge-graph" class="viz"></div>
      <aside id="candidate-card" class="side-card"><p class="note">Click a graph node or heatmap row to inspect evidence.</p></aside>
    </div>
    <details class="fallback"><summary>Static fallback</summary><img src="{figure_uri['bridge_graph']}" alt="Bridge graph fallback"></details>
  </section>

  <section class="section">
    <h2>Candidate Signature Matrix</h2>
    <p>Rows combine the original POC ESM-2 bridge candidates with mangrove nearest-neighborhood candidates selected from completed multiview payloads. Values are normalized review signals, not calibrated risk scores.</p>
    <div id="signature-matrix" class="viz"></div>
    <details class="fallback"><summary>Static fallback</summary><img src="{figure_uri['signature_matrix']}" alt="Candidate signature matrix fallback"></details>
    <h3>Candidate Audit Table</h3>
    {card_table}
  </section>

  <section class="section">
    <h2>Mangrove Molecular Pattern Explorer</h2>
    <p>This view only uses mangrove MAGs with completed functional outputs. It asks what high-level molecular patterns are already visible across methane evidence, sulfur context, CAZy/substrate breadth, QC, taxonomy, and nearest POC embedding affinity.</p>
    <div id="mangrove-scatter" class="viz"></div>
    <details class="fallback"><summary>Static fallback</summary><img src="{figure_uri['mangrove_patterns']}" alt="Mangrove molecular pattern fallback"></details>
  </section>

  <section class="section">
    <h2>Radial Cross-Domain Bridge View</h2>
    <p>The radial panel aggregates evidence-qualified high-similarity kNN links among rumen, wetland/MUCC, and mangrove sectors. It is a navigation device for molecular-neighborhood review, not proof of ecological transfer.</p>
    <div id="radial" class="viz"></div>
    <details class="fallback"><summary>Static fallback</summary><img src="{figure_uri['radial_bridge']}" alt="Radial bridge fallback"></details>
  </section>

  <section class="section">
    <h2>Strategic Interpretation</h2>
    <p><b>What is strong now:</b> MethaNet has a complete POC core and a large mangrove expansion where ESM-2 and gLM2 are already complete. The {summary['msm_multiview']:,} completed mangrove MAGs let the team inspect target-domain molecular fingerprints rather than waiting for the full tranche.</p>
    <p><b>What remains provisional:</b> mangrove pattern interpretation is still MAG/proteome-level functional potential. It cannot be rolled up to blue-carbon sample risk until MAG-to-sample mapping, abundance/read coverage, environmental covariates, uncertainty, and validation exist.</p>
    <p><b>Why this matters commercially:</b> the atlas converts raw metagenomic outputs into an auditable bridge-evidence interface: candidate cards, MRV feature primitives, monitoring-priority hypotheses, and a credible partner-facing proof artifact with claim locks built in.</p>
    <div class="warn"><b>Forbidden claims:</b> no final A-E risk tiers, no measured methane flux, no carbon-credit approval, and no source-independent rumen-to-wetland/mangrove transfer claims from this artifact alone.</div>
  </section>

  <section class="section">
    <h2>Validation Gap Register</h2>
    {gap_table}
  </section>

  <section class="section">
    <h2>Methods And Provenance</h2>
    <p>Generated at {generated}. Git HEAD: {summary['git_head']}. The report reads existing local artifacts only and does not submit jobs. D3 is embedded into this HTML package for offline visual exploration.</p>
    <p class="note">Primary analytical key: <code>proteome_id</code>. Claim boundary: {html.escape(CLAIM_BOUNDARY)}</p>
    <span class="pill">ESM-2 1280D proteome embeddings</span><span class="pill">high-dimensional kNN cosine graph</span><span class="pill">gLM2 native-minus-shuffled context</span><span class="pill">MCycDB / SCycDB / KOfam / dbCAN / METABOLIC</span><span class="pill">CheckM2 / GUNC / GTDB-Tk</span>
  </section>
</main>
<script>{d3_source}</script>
<script>{js}</script>
</body>
</html>"""
    return body


def build_gap_register(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap": "Mangrove functional completion",
                "current_state": f"{summary['msm_functional']:,}/{summary['msm_total']:,} complete",
                "affected_claim": "full mangrove tri-view atlas parity",
                "next_action": "rerender atlas after arrays finish and validate all curated manifests",
            },
            {
                "gap": "Sample/metagenome rollup",
                "current_state": "MAG/proteome-level evidence only",
                "affected_claim": "sample-level methane permanence risk",
                "next_action": "join MAGs to samples, read coverage/abundance, and environmental covariates",
            },
            {
                "gap": "Source-aware bridge validation",
                "current_state": "high-dimensional kNN graph and candidate cards generated",
                "affected_claim": "source-independent transfer",
                "next_action": "run source-aware nulls, bootstrap rank stability, and phylogeny-versus-embedding checks",
            },
            {
                "gap": "Flux/process validation",
                "current_state": "not present in molecular atlas",
                "affected_claim": "measured methane flux or calibrated A-E risk tier",
                "next_action": "connect to chamber/flux/geochemistry/incubation evidence before external MRV scoring",
            },
        ]
    )


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
    gap_register: pd.DataFrame,
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
    gap_register.to_csv(table_dir / "validation_gap_register.tsv", sep="\t", index=False)
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
            "claim_boundary_matrix": str(table_dir / "claim_boundary_matrix.tsv"),
            "validation_gap_register": str(table_dir / "validation_gap_register.tsv"),
        },
        "d3_payloads": {k: str(v) for k, v in payload_paths.items()},
        "fallback_figures": {k: str(v) for k, v in fallback_paths.items()},
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / "report_bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (output_dir / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # MethaNet Expanded MBAG Multi-View Atlas

            Generated: {summary['generated_at_utc']}

            Main artifact: `report.html`

            ## Snapshot

            - POC core multiview complete: {summary['poc_core_total']:,}/{summary['poc_core_total']:,}
            - Mangrove/MSM ESM-2 complete: {summary['msm_esm2']:,}/{summary['msm_total']:,}
            - Mangrove/MSM gLM2 complete: {summary['msm_glm2']:,}/{summary['msm_total']:,}
            - Mangrove/MSM functional complete: {summary['msm_functional']:,}/{summary['msm_total']:,}
            - Expanded multiview complete atlas: {summary['multiview_complete']:,}

            ## Claim Boundary

            {CLAIM_BOUNDARY}

            ## Regenerate

            ```bash
            source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
            conda activate methanet-fgx
            python scripts/reports/build_mbag_expanded_multiview_atlas.py
            ```
            """
        )
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve(repo_root, args.output_dir) if args.output_dir else repo_root / f"results/reports/mbag_expanded_multiview_atlas_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    poc_esm_dir = resolve(repo_root, args.poc_esm_dir)
    poc_warehouse_dir = resolve(repo_root, args.poc_warehouse_dir)
    poc_glm_dir = resolve(repo_root, args.poc_glm_dir)
    msm_root = resolve(repo_root, args.msm_root)
    msm_esm_dir = resolve(repo_root, args.msm_esm_dir)
    msm_glm_dir = resolve(repo_root, args.msm_glm_dir)
    assert poc_esm_dir and poc_warehouse_dir and poc_glm_dir and msm_root and msm_esm_dir and msm_glm_dir

    poc = load_poc_features(poc_warehouse_dir, poc_glm_dir, poc_esm_dir)
    msm, msm_status, msm_esm_stats = load_msm_features(msm_root, msm_esm_dir, msm_glm_dir)
    atlas = pd.concat([poc, msm], ignore_index=True, sort=False)

    emb_meta, edge_df, _ = build_embedding_context(poc_esm_dir, msm_esm_dir, atlas, args.knn)
    atlas = add_report_metrics(atlas, emb_meta)
    cards = build_candidate_cards(atlas, args.top_n_poc, args.top_n_mangrove)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(repo_root),
        "poc_core_total": int(poc["has_functional"].sum()),
        "msm_total": int(len(msm)),
        "msm_esm2": int(msm["has_esm2"].sum()),
        "msm_glm2": int(msm["has_glm2"].sum()),
        "msm_functional": int(msm["has_functional"].sum()),
        "msm_multiview": int((msm["has_esm2"] & msm["has_glm2"] & msm["has_functional"]).sum()),
        "msm_function_pending": int((~msm["has_functional"]).sum()),
        "msm_esm_embedded_total_with_resume": int(msm_esm_stats.get("embedded_total_with_resume") or 0),
        "msm_esm_pending_remaining": int(msm_esm_stats.get("pending_remaining") or 0),
        "multiview_complete": int(poc["has_functional"].sum() + (msm["has_esm2"] & msm["has_glm2"] & msm["has_functional"]).sum()),
        "embedding_context_total": int(len(emb_meta)),
        "knn_edges": int(len(edge_df)),
    }

    payloads = build_d3_payloads(atlas, emb_meta, edge_df, cards, summary, args.graph_node_cap)
    summary["graph_node_count"] = len(payloads["graph"]["nodes"])
    summary["graph_edge_count"] = len(payloads["graph"]["links"])
    payload_paths = save_json_payloads(payloads, output_dir / "assets/data")
    fallbacks = build_fallbacks(payloads, output_dir / "assets/figures")
    d3_path, d3_source = fetch_d3(output_dir / "assets/js")
    summary["d3_asset"] = str(d3_path)
    gap_register = build_gap_register(summary)
    html_text = render_html(output_dir, summary, payloads, fallbacks, cards, gap_register, d3_source)
    write_outputs(output_dir, atlas, emb_meta, edge_df, cards, msm_status, payload_paths, fallbacks, summary, gap_register, html_text)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
