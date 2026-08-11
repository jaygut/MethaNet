#!/usr/bin/env python3
"""
V9 canonical joined table builder.
Loads raw ESM2 vectors for all 7,710 paired units, computes a joint PCA+UMAP,
joins with the frozen atlas feature table, MUCC warehouse tables, and the
audit-core / KOfam-panel subset flags. Writes:
  v9out/tables/atlas_unit_ledger_v9.tsv         (7710 rows, unified schema + coords)
  v9out/embeddings/atlas_joint_embedding_7710.npz (raw vectors + pca + umap)
"""
import numpy as np, pandas as pd, glob, os, json, hashlib, sys, time
from pathlib import Path
np.random.seed(20250601)

BASE = str(Path(__file__).resolve().parents[2])
OUT = "v9out"
os.makedirs(f"{OUT}/tables", exist_ok=True)
os.makedirs(f"{OUT}/embeddings", exist_ok=True)

t0 = time.time()

def load_npz_records(path):
    d = np.load(path, allow_pickle=True)
    n = d["embeddings"].shape[0]
    rec = {"embeddings": d["embeddings"]}
    for k in d.files:
        if k != "embeddings":
            rec[k] = d[k]
    return rec, n

frozen_paths = [
    f"{BASE}/results/blue_catalyst_poc/runs/msm_china_2025_esm2_20260616_082112/artifacts/genome_embeddings.npz",
] + sorted(glob.glob(f"{BASE}/results/blue_catalyst_poc/runs/futian_mangrove_2026_esm2_phase1_shard*/artifacts/genome_embeddings.npz")) + [
    f"{BASE}/results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts/genome_embeddings.npz",
]
mucc_paths = sorted(glob.glob(f"{BASE}/results/blue_catalyst_poc/runs/mucc_v1_owc_wetland_esm2_20260626_gpu_v2_shard*/artifacts/genome_embeddings.npz"))

all_vecs, all_meta = [], []
for p in frozen_paths + mucc_paths:
    rec, n = load_npz_records(p)
    all_vecs.append(rec["embeddings"])
    # id key varies by run: prefer proteome_id, then mag_id, then sample
    # (apolo_full has none of proteome_id/mag_id, only 'sample', which
    # matches embedding_context_table.proteome_id exactly for its rumen+mucc rows)
    if "proteome_id" in rec:
        idvals = rec["proteome_id"]
    elif "mag_id" in rec:
        idvals = rec["mag_id"]
    elif "sample" in rec:
        idvals = rec["sample"]
    else:
        raise ValueError(f"no id-like key found in {p}: {list(rec.keys())}")
    meta = pd.DataFrame({
        "proteome_id": idvals,
        "mag_id": rec.get("mag_id", idvals),
        "source": rec.get("source", np.array(["unknown"] * n)),
        "ecosystem": rec.get("ecosystem", np.array([""] * n)),
        "domain": rec.get("domain", np.array([""] * n)),
    })
    meta["_shard_file"] = os.path.basename(os.path.dirname(os.path.dirname(p)))
    all_meta.append(meta)
    print(f"loaded {p}: n={n}")

vecs = np.vstack(all_vecs).astype(np.float32)
meta = pd.concat(all_meta, ignore_index=True)
print("raw stacked shape:", vecs.shape, "meta rows:", len(meta))

before = len(meta)
dup_mask = meta.duplicated(subset=["proteome_id"], keep="first")
n_dup = dup_mask.sum()
vecs = vecs[~dup_mask.values]
meta = meta[~dup_mask.values].reset_index(drop=True)
print(f"deduplicated by proteome_id: {before} -> {len(meta)} ({n_dup} dropped as duplicate ids)")

ectx = pd.read_csv(f"{BASE}/results/reports/mbag_nextgen_molecular_niche_atlas_20260629_interim_2364/tables/embedding_context_table.tsv", sep="\t")
atlas_ids = set(ectx["proteome_id"])
mucc_ids = set(meta.loc[meta["source"].isin(["mucc_v1_owc_wetland"]), "proteome_id"])
print("atlas (frozen) ids:", len(atlas_ids), "| mucc warehouse ids:", len(mucc_ids), "| overlap:", len(atlas_ids & mucc_ids))

keep_ids = atlas_ids | mucc_ids
keep_mask = meta["proteome_id"].isin(keep_ids).values
vecs7710 = vecs[keep_mask]
meta7710 = meta[keep_mask].reset_index(drop=True)
print("final paired-embedding set:", vecs7710.shape, len(meta7710))
assert len(meta7710) == 7710, f"expected 7710, got {len(meta7710)}"

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ESM2 embeddings carry a small number of extreme-variance rogue dimensions
# (a documented artifact of protein language model embeddings); standardize
# per-dimension before PCA/UMAP so those dimensions do not dominate the
# projection. This changes only the visualization basis, not the raw vectors
# saved to atlas_joint_embedding_7710.npz (embeddings key is unscaled).
scaler = StandardScaler()
vecs7710_scaled = scaler.fit_transform(vecs7710)

pca = PCA(n_components=2, random_state=20250601)
pca_coords = pca.fit_transform(vecs7710_scaled)
print("PCA explained variance ratio (standardized):", pca.explained_variance_ratio_)

import umap
reducer = umap.UMAP(n_components=2, random_state=20250601, n_neighbors=30, min_dist=0.1)
umap_coords = reducer.fit_transform(vecs7710_scaled)
print("UMAP done, shape:", umap_coords.shape)

meta7710["pca_1_v9"] = pca_coords[:, 0]
meta7710["pca_2_v9"] = pca_coords[:, 1]
meta7710["umap_1_v9"] = umap_coords[:, 0]
meta7710["umap_2_v9"] = umap_coords[:, 1]
meta7710["pca_var_pc1"] = pca.explained_variance_ratio_[0]
meta7710["pca_var_pc2"] = pca.explained_variance_ratio_[1]

np.savez_compressed(f"{OUT}/embeddings/atlas_joint_embedding_7710.npz",
    embeddings=vecs7710,
    proteome_id=meta7710["proteome_id"].values,
    pca=pca_coords, umap=umap_coords,
    pca_explained_variance_ratio=pca.explained_variance_ratio_)

amf = pd.read_csv(f"{BASE}/results/reports/mbag_nextgen_molecular_niche_atlas_20260629_interim_2364/tables/atlas_multiview_feature_table.tsv", sep="\t", low_memory=False)

wh = f"{BASE}/results/functional_metagenomics/mucc_v1_owc_wetland_20260626/cohort_warehouse/parquet"

audit_core_path_candidates = glob.glob(f"{BASE}/results/reports/mbag_manuscript_v*/tables/audit_core_437_manifest.tsv")
audit_ids = set()
if audit_core_path_candidates:
    ac = pd.read_csv(sorted(audit_core_path_candidates)[-1], sep="\t")
    idcol = "proteome_id" if "proteome_id" in ac.columns else ac.columns[0]
    audit_ids = set(ac[idcol])
print("audit core ids loaded:", len(audit_ids))

kofam_path = f"{BASE}/results/reports/mbag_manuscript_v7_20260716/tables/methane_cycle_gene_panel_v5.parquet"
kof = pd.read_parquet(kofam_path)
# kof.mag_dir matches embedding_context_table.proteome_id / this build's
# proteome_id key directly (verified: 2,711 of 2,730 KOfam rows match; the
# remaining 19 are POC_mucc-lane rows on a different id vintage than the
# 107-row mucc atlas lane, consistent with the prior V5/V6 quarantine record)
kofam_ids_proteome = set(kof["mag_dir"].astype(str))
print("kofam panel rows:", len(kof), "id col: mag_dir (matches proteome_id)")

led = meta7710.copy()
led["is_frozen_atlas"] = led["proteome_id"].isin(atlas_ids)
led["is_mucc_warehouse"] = led["proteome_id"].isin(mucc_ids)

keepcols = ["proteome_id","mag_id","domain","phylum","class","order","family","genus","species",
            "qc_tier","checkm2_completeness","checkm2_contamination","gunc_pass",
            "kofam_annotated_gene_fraction","comparability_status"]
if "freeze_tri_view_ready" in amf.columns:
    keepcols.append("freeze_tri_view_ready")
amf_small = amf[[c for c in keepcols if c in amf.columns]].drop_duplicates(subset=["proteome_id"])
led = led.merge(amf_small, on="proteome_id", how="left", suffixes=("","_amf"))

led["in_audit_core"] = led["proteome_id"].isin(audit_ids)
led["in_kofam_panel"] = led["proteome_id"].astype(str).isin(kofam_ids_proteome)

def evidence_tier(row):
    if row["is_mucc_warehouse"]:
        return "mucc_source_scaffold"
    if str(row.get("comparability_status","")) == "comparable_mag_bin":
        return "canonical_mechanism_table"
    if row.get("freeze_tri_view_ready", False) == True:
        return "freeze_equivalent_functional"
    return "embedding_only"
led["evidence_tier"] = led.apply(evidence_tier, axis=1)

out_cols = ["proteome_id","mag_id","source","ecosystem","domain","is_frozen_atlas","is_mucc_warehouse",
            "evidence_tier","in_audit_core","in_kofam_panel",
            "pca_1_v9","pca_2_v9","umap_1_v9","umap_2_v9","pca_var_pc1","pca_var_pc2",
            "phylum","class","order","family","genus","species",
            "qc_tier","checkm2_completeness","checkm2_contamination","gunc_pass","kofam_annotated_gene_fraction"]
out_cols = [c for c in out_cols if c in led.columns]
led_out = led[out_cols]
led_out.to_csv(f"{OUT}/tables/atlas_unit_ledger_v9.tsv", sep="\t", index=False)
print("atlas_unit_ledger_v9 shape:", led_out.shape)
print(led_out["evidence_tier"].value_counts())
print(led_out["source"].value_counts())

print(f"\ntotal wall time: {time.time()-t0:.1f}s")
print("DONE")
