"""Data-loading and feature summarization for MBAG smoke analyses."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from methanet.mbag.core import compute_reliability_weight


METHANE_TERMS = re.compile(
    r"methan|methyl|mcr|mtr|hdr|mvh|fwd|ftr|mer|mta|mtb|mtt|pmo|mmo",
    re.IGNORECASE,
)
SULFUR_TERMS = re.compile(r"sulfur|sulphur|sulfate|sulfite|sulfide|dsr|apr|sat|sox", re.IGNORECASE)


@dataclass(frozen=True)
class MBAGPaths:
    """Repository-relative default paths used by the MBAG smoke workflow."""

    repo_root: Path
    cohort_run_id: str = "fgx_662_apollo3_20260612"

    @property
    def crosswalk(self) -> Path:
        return self.repo_root / "ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv"

    @property
    def functional_manifest(self) -> Path:
        return self.repo_root / "results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv"

    @property
    def embedding_artifact_dir(self) -> Path:
        return self.repo_root / "results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts"

    @property
    def per_mag_dir(self) -> Path:
        return self.repo_root / "results/functional_metagenomics" / self.cohort_run_id / "per_mag"


@dataclass(frozen=True)
class EmbeddingArtifacts:
    """Loaded ESM2 artifacts needed by MBAG."""

    metadata: pd.DataFrame
    embeddings: np.ndarray
    projection: pd.DataFrame
    bridge_top: pd.DataFrame
    knn_neighborhoods: pd.DataFrame


@dataclass(frozen=True)
class FunctionalDataset:
    """Compact functional profiles and feature-token counters."""

    profiles: pd.DataFrame
    feature_counters: dict[str, dict[str, Counter[str]]]
    run_status: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "NA", "N/A"):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed"}:
        return True
    if text in {"false", "0", "no", "fail", "failed"}:
        return False
    return None


def _load_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for MBAG functional Parquet loading") from exc
    try:
        return pq.read_table(path, columns=columns).to_pandas()
    except Exception:
        return pd.DataFrame()


def load_crosswalk(path: Path, manifest_path: Path | None = None) -> pd.DataFrame:
    """Load the 662-row canonical crosswalk, optionally enriched with FASTA paths."""

    crosswalk = pd.read_csv(path, sep="\t")
    if "sample" in crosswalk.columns and "proteome_id" not in crosswalk.columns:
        crosswalk = crosswalk.rename(columns={"sample": "proteome_id"})
    if "proteome_id" not in crosswalk.columns:
        raise ValueError(f"{path} does not contain proteome_id")

    out = crosswalk.copy()
    if manifest_path and manifest_path.exists():
        manifest = pd.read_csv(manifest_path, sep="\t")
        keep = [
            col
            for col in [
                "proteome_id",
                "mag_id",
                "mag_fasta",
                "mag_fasta_basename",
                "proteome_faa",
                "proteome_faa_basename",
                "match_key",
                "match_status",
                "functional_run_include",
                "analysis_unit_type",
                "mbag_mag_level_include",
                "assembly_context_include",
                "claim_scope",
                "comparability_status",
                "comparability_reason",
                "recommended_action",
            ]
            if col in manifest.columns
        ]
        out = out.merge(manifest[keep], on="proteome_id", how="left", suffixes=("", "_manifest"))
        if "mag_id" not in out.columns and "mag_id_candidate" in out.columns:
            out["mag_id"] = out["mag_id_candidate"]
    elif "mag_id_candidate" in out.columns:
        out["mag_id"] = out["mag_id_candidate"]

    if out["proteome_id"].duplicated().any():
        dupes = out.loc[out["proteome_id"].duplicated(), "proteome_id"].head().tolist()
        raise ValueError(f"Duplicate proteome_id values in crosswalk: {dupes}")
    return out


def load_embedding_artifacts(artifact_dir: Path) -> EmbeddingArtifacts:
    """Load ESM2 embedding matrix plus projection and bridge tables."""

    npz_path = artifact_dir / "genome_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    bundle = np.load(npz_path, allow_pickle=True)
    embeddings = np.asarray(bundle["embeddings"], dtype=float)
    metadata = pd.DataFrame(
        {
            "proteome_id": bundle["sample"].astype(str),
            "sample": bundle["sample"].astype(str),
            "source": bundle["source"].astype(str),
            "ecosystem": bundle["ecosystem"].astype(str),
            "domain": bundle["domain"].astype(str),
            "source_analysis_accession": bundle["source_analysis_accession"],
            "n_proteins_used": bundle["n_proteins_used"],
        }
    )
    projection = pd.read_csv(artifact_dir / "embedding_projection_clusters.tsv", sep="\t")
    if "sample" in projection.columns:
        projection = projection.rename(columns={"sample": "proteome_id"})
    bridge_top = pd.read_csv(artifact_dir / "bridge_top_candidates.tsv", sep="\t")
    if "sample" in bridge_top.columns:
        bridge_top = bridge_top.rename(columns={"sample": "proteome_id"})
    knn_path = artifact_dir / "bridge_knn_neighborhoods.tsv"
    knn = pd.read_csv(knn_path, sep="\t") if knn_path.exists() else pd.DataFrame()
    return EmbeddingArtifacts(metadata, embeddings, projection, bridge_top, knn)


def discover_completed_runs(per_mag_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Discover all run attempts and select the latest complete run per proteome."""

    rows: list[dict[str, Any]] = []
    selected: dict[str, Path] = {}
    for run_dir in sorted(path for path in per_mag_dir.glob("*/*") if path.is_dir()):
        proteome_id = run_dir.parent.name
        record_path = run_dir / "curated/run_record.json"
        manifest_path = run_dir / "curated/parquet_manifest.tsv"
        record = _read_json(record_path)
        status = record.get("status")
        if not status:
            status = "complete" if (run_dir / "COMPLETE").exists() else "failed" if (run_dir / "FAILED").exists() else "partial"
        run_id = record.get("run_id") or run_dir.name
        mag_id = record.get("mag_id") or proteome_id.removeprefix("mucc__").removeprefix("rumen__")
        row = {
            "proteome_id": proteome_id,
            "mag_id": mag_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run_status": status,
            "has_complete_sentinel": (run_dir / "COMPLETE").exists(),
            "has_failed_sentinel": (run_dir / "FAILED").exists(),
            "has_run_record": record_path.exists(),
            "has_file_manifest": (run_dir / "curated/file_manifest.tsv").exists(),
            "has_parquet_manifest": manifest_path.exists(),
            "mtime_epoch": run_dir.stat().st_mtime,
        }
        rows.append(row)
        if status == "complete" and row["has_complete_sentinel"] and row["has_parquet_manifest"]:
            current = selected.get(proteome_id)
            if current is None or run_dir.stat().st_mtime > current.stat().st_mtime:
                selected[proteome_id] = run_dir
    return pd.DataFrame(rows), selected


def _taxonomy_resolution(taxonomy: dict[str, Any]) -> str:
    for rank in ["species", "genus", "family", "order", "class", "phylum", "domain"]:
        value = str(taxonomy.get(rank) or "").strip()
        if value and value not in {"s__", "g__", "f__", "N/A", "nan"}:
            return f"resolved_{rank}"
    return "unresolved"


def _feature_add(counter: Counter[str], prefix: str, values: pd.Series, limit: int = 200) -> None:
    for value, count in values.dropna().astype(str).value_counts().head(limit).items():
        text = value.strip()
        if text and text not in {"-", "None", "nan"}:
            counter[f"{prefix}:{text}"] += int(count)


def _contains_terms(frame: pd.DataFrame, columns: list[str], pattern: re.Pattern[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    text = pd.Series([""] * len(frame), index=frame.index)
    for col in columns:
        if col in frame.columns:
            text = text + " " + frame[col].fillna("").astype(str)
    return text.str.contains(pattern, regex=True, na=False)


def _summarize_one_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Counter[str]]]:
    record = _read_json(run_dir / "curated/run_record.json")
    proteome_id = record.get("proteome_id") or run_dir.parent.name
    mag_id = record.get("mag_id") or proteome_id.removeprefix("mucc__").removeprefix("rumen__")
    run_id = record.get("run_id") or run_dir.name
    taxonomy = record.get("taxonomy") or {}
    counters = {
        "methane": Counter(),
        "sulfur": Counter(),
        "substrate": Counter(),
        "broad": Counter(),
    }

    parquet_dir = run_dir / "curated/parquet"
    qc = _load_parquet(parquet_dir / "fact_qc_checkm2.parquet")
    gunc = _load_parquet(parquet_dir / "fact_qc_gunc.parquet")
    completeness = _safe_float(qc["Completeness"].iloc[0]) if "Completeness" in qc and not qc.empty else None
    contamination = _safe_float(qc["Contamination"].iloc[0]) if "Contamination" in qc and not qc.empty else None
    total_genes = _safe_float(qc["Total_Coding_Sequences"].iloc[0]) if "Total_Coding_Sequences" in qc and not qc.empty else None
    gunc_pass = _safe_bool(gunc["pass.GUNC"].iloc[0]) if "pass.GUNC" in gunc and not gunc.empty else None

    kofam = _load_parquet(
        parquet_dir / "fact_kofam_hits.parquet",
        columns=["gene_id", "ko_id", "ko_definition", "accepted_hit"],
    )
    if not kofam.empty and "accepted_hit" in kofam:
        accepted = kofam[kofam["accepted_hit"].fillna(False).astype(bool)].copy()
    else:
        accepted = pd.DataFrame()
    if not accepted.empty:
        _feature_add(counters["broad"], "ko", accepted["ko_id"], limit=300)
        methane_mask = _contains_terms(accepted, ["ko_id", "ko_definition"], METHANE_TERMS)
        sulfur_mask = _contains_terms(accepted, ["ko_id", "ko_definition"], SULFUR_TERMS)
        _feature_add(counters["methane"], "ko", accepted.loc[methane_mask, "ko_id"], limit=100)
        _feature_add(counters["sulfur"], "ko", accepted.loc[sulfur_mask, "ko_id"], limit=100)

    mcyc = _load_parquet(
        parquet_dir / "fact_mcycdb_hits.parquet",
        columns=["gene_id", "subject_id", "hit_rank_bitscore", "bitscore"],
    )
    mcyc_best = mcyc[mcyc["hit_rank_bitscore"].eq(1)] if "hit_rank_bitscore" in mcyc else pd.DataFrame()
    if not mcyc_best.empty:
        _feature_add(counters["methane"], "mcyc", mcyc_best["subject_id"], limit=250)

    scyc = _load_parquet(
        parquet_dir / "fact_scycdb_hits.parquet",
        columns=["gene_id", "subject_id", "hit_rank_bitscore", "bitscore"],
    )
    scyc_best = scyc[scyc["hit_rank_bitscore"].eq(1)] if "hit_rank_bitscore" in scyc else pd.DataFrame()
    if not scyc_best.empty:
        _feature_add(counters["sulfur"], "scyc", scyc_best["subject_id"], limit=250)

    dbcan = _load_parquet(parquet_dir / "fact_dbcan_hits.parquet", columns=["Gene ID", "HMMER", "dbCAN_sub", "DIAMOND"])
    if not dbcan.empty:
        for col in ["HMMER", "dbCAN_sub", "DIAMOND"]:
            if col in dbcan:
                _feature_add(counters["substrate"], f"dbcan_{col.lower()}", dbcan[col], limit=150)

    metabolic_hmm = _load_parquet(
        parquet_dir / "fact_metabolic_hmm_hits.parquet",
        columns=["function_category", "function_name", "gene_abbreviation", "presence", "hit_count"],
    )
    if not metabolic_hmm.empty and "presence" in metabolic_hmm:
        present_hmm = metabolic_hmm[metabolic_hmm["presence"].astype(str).str.lower().eq("present")]
    else:
        present_hmm = pd.DataFrame()
    if not present_hmm.empty:
        methane_mask = _contains_terms(present_hmm, ["function_category", "function_name", "gene_abbreviation"], METHANE_TERMS)
        sulfur_mask = _contains_terms(present_hmm, ["function_category", "function_name", "gene_abbreviation"], SULFUR_TERMS)
        _feature_add(counters["methane"], "metabolic_hmm", present_hmm.loc[methane_mask, "gene_abbreviation"], limit=100)
        _feature_add(counters["sulfur"], "metabolic_hmm", present_hmm.loc[sulfur_mask, "gene_abbreviation"], limit=100)
        _feature_add(counters["broad"], "metabolic_hmm", present_hmm["function_category"], limit=150)

    modules = _load_parquet(
        parquet_dir / "fact_metabolic_module_presence.parquet",
        columns=["module_id", "module_name", "module_category", "presence"],
    )
    if not modules.empty and "presence" in modules:
        present_modules = modules[modules["presence"].astype(str).str.lower().eq("present")]
        _feature_add(counters["broad"], "module", present_modules["module_id"], limit=200)

    cazy = _load_parquet(parquet_dir / "fact_cazy_hits.parquet", columns=["cazy_family", "hit_count"])
    if not cazy.empty and "cazy_family" in cazy:
        _feature_add(counters["substrate"], "cazy", cazy["cazy_family"], limit=200)

    merops = _load_parquet(parquet_dir / "fact_merops_hits.parquet", columns=["merops_peptidase_id", "hit_count"])
    if not merops.empty and "merops_peptidase_id" in merops:
        _feature_add(counters["substrate"], "merops", merops["merops_peptidase_id"], limit=120)

    protein_count = total_genes or 0.0
    accepted_gene_count = accepted["gene_id"].nunique() if "gene_id" in accepted else 0
    dbcan_gene_count = dbcan["Gene ID"].nunique() if "Gene ID" in dbcan else 0
    annotation_cov = max(
        accepted_gene_count / protein_count if protein_count else 0.0,
        dbcan_gene_count / protein_count if protein_count else 0.0,
    )
    metabolic_cov = len(present_hmm) / len(metabolic_hmm) if len(metabolic_hmm) else 0.0
    coverage_required = max(min(annotation_cov * 2.0, 1.0), min(metabolic_cov * 4.0, 1.0))
    taxonomy_status = _taxonomy_resolution(taxonomy)
    reliability = compute_reliability_weight(
        completeness,
        contamination,
        gunc_pass,
        coverage_required,
        taxonomy_status,
    )

    row = {
        "proteome_id": proteome_id,
        "mag_id": mag_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "functional_status": "complete",
        "completeness": completeness,
        "contamination": contamination,
        "gunc_pass": gunc_pass,
        "taxonomy_status": taxonomy_status,
        "taxonomy_domain": taxonomy.get("domain"),
        "taxonomy_family": taxonomy.get("family"),
        "taxonomy_genus": taxonomy.get("genus"),
        "total_genes": int(protein_count) if protein_count else None,
        "kofam_accepted_hits": int(len(accepted)),
        "kofam_accepted_unique_ko": int(accepted["ko_id"].nunique()) if "ko_id" in accepted else 0,
        "mcycdb_best_hits": int(len(mcyc_best)),
        "mcycdb_best_genes": int(mcyc_best["gene_id"].nunique()) if "gene_id" in mcyc_best else 0,
        "scycdb_best_hits": int(len(scyc_best)),
        "scycdb_best_genes": int(scyc_best["gene_id"].nunique()) if "gene_id" in scyc_best else 0,
        "dbcan_hits": int(len(dbcan)),
        "dbcan_genes": int(dbcan_gene_count),
        "metabolic_present_hmm": int(len(present_hmm)),
        "metabolic_total_hmm": int(len(metabolic_hmm)),
        "metabolic_present_modules": int(len(present_modules)) if "present_modules" in locals() else 0,
        "cazy_families": int(cazy["cazy_family"].nunique()) if "cazy_family" in cazy else 0,
        "merops_families": int(merops["merops_peptidase_id"].nunique()) if "merops_peptidase_id" in merops else 0,
        "methane_feature_count": int(sum(counters["methane"].values())),
        "sulfur_feature_count": int(sum(counters["sulfur"].values())),
        "substrate_feature_count": int(sum(counters["substrate"].values())),
        "broad_feature_count": int(sum(counters["broad"].values())),
        "coverage_required": float(coverage_required),
        "annotation_coverage_proxy": float(annotation_cov),
        "metabolic_coverage_proxy": float(metabolic_cov),
        "reliability_weight": reliability,
    }
    return row, counters


def build_functional_dataset(selected_runs: dict[str, Path]) -> FunctionalDataset:
    """Summarize selected completed runs into MBAG profile rows and counters."""

    rows: list[dict[str, Any]] = []
    counters: dict[str, dict[str, Counter[str]]] = {}
    status_rows = [
        {"proteome_id": proteome_id, "selected_run_dir": str(run_dir), "functional_status": "complete"}
        for proteome_id, run_dir in selected_runs.items()
    ]
    for proteome_id, run_dir in sorted(selected_runs.items()):
        row, feature_counters = _summarize_one_run(run_dir)
        rows.append(row)
        counters[proteome_id] = feature_counters
    return FunctionalDataset(pd.DataFrame(rows), counters, pd.DataFrame(status_rows))
