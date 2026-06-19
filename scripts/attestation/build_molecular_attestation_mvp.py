#!/usr/bin/env python3
"""Build MethaNet's local molecular attestation MVP snapshot.

The builder is read-only with respect to production functional outputs. It
projects existing cohort warehouse, ESM2 bridge, gLM2 context, and claim-boundary
artifacts into:

- Parquet graph node/edge exports;
- an evidence atom table with source artifact provenance;
- a Kuzu embedded property graph when the Python package is available;
- a small canonical graph query library and validation report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COHORT_ROWS = 662
EXPECTED_MAG_BIN_ROWS = 625
EXPECTED_ASSEMBLY_CONTEXT_ROWS = 37

DEFAULT_WAREHOUSE = (
    "results/functional_metagenomics/fgx_662_apollo3_20260612/"
    "cohort_warehouse_poc_magbin_union_20260616_075022"
)
DEFAULT_BRIDGE_CARDS = (
    "results/functional_metagenomics/fgx_662_apollo3_20260612/reports/"
    "mbag_smoke_full_docx_scoped_20260614_0554_final/"
    "bridge_attestation_cards_smoke.tsv"
)
DEFAULT_ESM2_EDGES = (
    "results/functional_metagenomics/fgx_662_apollo3_20260612/reports/"
    "mbag_smoke_full_docx_scoped_20260614_0554_final/"
    "mbag_esm2_knn_edges.parquet"
)
DEFAULT_GLM_FEATURES = (
    "results/contextual_genomics/"
    "glm2_integration_20260616_poc_catchup_20260616_073441/"
    "feature_glm_mag_level.tsv"
)
DEFAULT_UNIT_SCOPE = (
    "results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/"
    "poc_662_functional_mag_manifest.with_unit_scope.tsv"
)
DEFAULT_CROSSWALK = (
    "ai_docs/functional_metagenomics_expansion/proteome_crosswalk/"
    "embedded_662_proteome_id_crosswalk.tsv"
)


@dataclass(frozen=True)
class SourceArtifacts:
    """Artifact IDs for tables used by evidence atoms."""

    by_key: dict[str, str]

    def get(self, key: str) -> str:
        return self.by_key.get(key, self.by_key["attestation_builder"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--warehouse-dir", type=Path, default=None)
    parser.add_argument("--crosswalk", type=Path, default=None)
    parser.add_argument("--unit-scope-manifest", type=Path, default=None)
    parser.add_argument("--bridge-cards", type=Path, default=None)
    parser.add_argument("--esm2-edges", type=Path, default=None)
    parser.add_argument("--glm-feature-table", type=Path, default=None)
    parser.add_argument("--skip-kuzu", action="store_true")
    return parser.parse_args()


def repo_path(repo_root: Path, value: str | Path | None, default: str) -> Path:
    path = Path(value) if value is not None else Path(default)
    return path if path.is_absolute() else repo_root / path


def safe_str(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value)
    return default if text == "nan" else text


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if pd.notna(parsed) else default


def safe_bool_text(value: Any) -> str:
    text = safe_str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed"}:
        return "true"
    if text in {"false", "0", "no", "fail", "failed"}:
        return "false"
    return ""


def short_hash(text: str, prefix: str = "") -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def hash_file(path: Path, max_bytes: int = 8_000_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    if path.stat().st_size > max_bytes:
        return "not_hashed_large_file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, sep="\t", **kwargs)


def parquet_path_for(warehouse_dir: Path, table: str) -> Path:
    candidates = sorted((warehouse_dir / "parquet" / table).glob("cohort_run_id=*/part-*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No Parquet shard found for {table} under {warehouse_dir}")
    return candidates[0]


def load_inputs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    repo_root = args.repo_root.resolve()
    warehouse_dir = repo_path(repo_root, args.warehouse_dir, DEFAULT_WAREHOUSE)
    crosswalk_path = repo_path(repo_root, args.crosswalk, DEFAULT_CROSSWALK)
    unit_scope_path = repo_path(repo_root, args.unit_scope_manifest, DEFAULT_UNIT_SCOPE)
    bridge_cards_path = repo_path(repo_root, args.bridge_cards, DEFAULT_BRIDGE_CARDS)
    esm2_edges_path = repo_path(repo_root, args.esm2_edges, DEFAULT_ESM2_EDGES)
    glm_path = repo_path(repo_root, args.glm_feature_table, DEFAULT_GLM_FEATURES)

    crosswalk = read_table(crosswalk_path)
    if "sample" in crosswalk.columns and "proteome_id" not in crosswalk.columns:
        crosswalk = crosswalk.rename(columns={"sample": "proteome_id"})
    unit_scope = read_table(unit_scope_path)
    dim_mag = read_table(parquet_path_for(warehouse_dir, "dim_mag"))
    taxonomy = read_table(parquet_path_for(warehouse_dir, "fact_taxonomy_gtdbtk"))
    methane = read_table(parquet_path_for(warehouse_dir, "feature_methane_mechanism"))
    sulfur = read_table(parquet_path_for(warehouse_dir, "feature_sulfur_competition"))
    mrv = read_table(parquet_path_for(warehouse_dir, "feature_mrv_mag_level"))
    coverage = read_table(parquet_path_for(warehouse_dir, "feature_annotation_coverage"))
    run_status = read_table(parquet_path_for(warehouse_dir, "fact_run_status"))
    bridge_cards = read_table(bridge_cards_path) if bridge_cards_path.exists() else pd.DataFrame()
    esm2_edges = read_table(esm2_edges_path) if esm2_edges_path.exists() else pd.DataFrame()
    glm = read_table(glm_path) if glm_path.exists() else pd.DataFrame()

    table_paths = {
        "crosswalk": crosswalk_path,
        "unit_scope": unit_scope_path,
        "bridge_cards": bridge_cards_path,
        "esm2_edges": esm2_edges_path,
        "glm_features": glm_path,
        "dim_mag": parquet_path_for(warehouse_dir, "dim_mag"),
        "taxonomy": parquet_path_for(warehouse_dir, "fact_taxonomy_gtdbtk"),
        "methane": parquet_path_for(warehouse_dir, "feature_methane_mechanism"),
        "sulfur": parquet_path_for(warehouse_dir, "feature_sulfur_competition"),
        "mrv": parquet_path_for(warehouse_dir, "feature_mrv_mag_level"),
        "coverage": parquet_path_for(warehouse_dir, "feature_annotation_coverage"),
        "run_status": parquet_path_for(warehouse_dir, "fact_run_status"),
    }

    return {
        "crosswalk": crosswalk,
        "unit_scope": unit_scope,
        "dim_mag": dim_mag,
        "taxonomy": taxonomy,
        "methane": methane,
        "sulfur": sulfur,
        "mrv": mrv,
        "coverage": coverage,
        "run_status": run_status,
        "bridge_cards": bridge_cards,
        "esm2_edges": esm2_edges,
        "glm": glm,
        "_table_paths": pd.DataFrame(
            [{"key": key, "path": str(path)} for key, path in table_paths.items()]
        ),
    }


def build_artifact_registry(
    repo_root: Path,
    output_dir: Path,
    table_paths: pd.DataFrame,
) -> tuple[pd.DataFrame, SourceArtifacts]:
    rows: list[dict[str, Any]] = []
    for row in table_paths.to_dict("records"):
        path = Path(row["path"])
        artifact_id = f"artifact:{row['key']}:{short_hash(str(path.resolve()))}"
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_key": row["key"],
                "artifact_uri": str(path),
                "artifact_class": "source_evidence",
                "artifact_format": path.suffix.lstrip(".") or "directory",
                "source_system": "local_filesystem",
                "source_tool": row["key"],
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                if path.exists()
                else "",
                "sha256": hash_file(path),
                "license_or_use_caveat": "internal MethaNet evidence artifact; claim boundaries apply",
            }
        )
    builder_id = f"artifact:attestation_builder:{short_hash(__file__)}"
    rows.append(
        {
            "artifact_id": builder_id,
            "artifact_key": "attestation_builder",
            "artifact_uri": str(repo_root / "scripts/attestation/build_molecular_attestation_mvp.py"),
            "artifact_class": "derived_builder",
            "artifact_format": "py",
            "source_system": "local_script",
            "source_tool": "build_molecular_attestation_mvp",
            "size_bytes": 0,
            "modified_at": "",
            "sha256": "",
            "license_or_use_caveat": "version-controlled source code",
        }
    )
    registry = pd.DataFrame(rows)
    registry.to_parquet(output_dir / "registry_artifact.parquet", index=False)
    registry.to_csv(output_dir / "registry_artifact.tsv", sep="\t", index=False)
    return registry, SourceArtifacts(dict(zip(registry["artifact_key"], registry["artifact_id"])))


def latest_run_status(run_status: pd.DataFrame) -> pd.DataFrame:
    if run_status.empty:
        return pd.DataFrame(columns=["proteome_id", "run_status", "run_id"])
    status = run_status.copy()
    status["_complete_rank"] = status["run_status"].eq("complete").astype(int)
    status = status.sort_values(["proteome_id", "_complete_rank", "run_id"])
    return status.groupby("proteome_id", as_index=False).tail(1).drop(columns=["_complete_rank"])


def first_by_proteome(frame: pd.DataFrame, sort_cols: list[str] | None = None) -> pd.DataFrame:
    if frame.empty or "proteome_id" not in frame.columns:
        return pd.DataFrame(columns=["proteome_id"])
    data = frame.copy()
    if sort_cols:
        present = [col for col in sort_cols if col in data.columns]
        if present:
            data = data.sort_values(present)
    return data.drop_duplicates("proteome_id", keep="last")


def build_mag_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    crosswalk = inputs["crosswalk"].copy()
    unit = inputs["unit_scope"].copy()
    keep_unit = [
        col
        for col in [
            "proteome_id",
            "mag_id",
            "source",
            "ecosystem",
            "domain",
            "analysis_unit_type",
            "mbag_mag_level_include",
            "assembly_context_include",
            "claim_scope",
            "comparability_status",
            "comparability_reason",
            "recommended_action",
        ]
        if col in unit.columns
    ]
    mag = crosswalk.merge(unit[keep_unit], on="proteome_id", how="left", suffixes=("", "_unit"))

    dim = inputs["dim_mag"].copy()
    dim_keep = [
        col
        for col in [
            "proteome_id",
            "run_id",
            "mag_id",
            "source_group",
            "qc_tier",
            "checkm2_completeness",
            "checkm2_contamination",
            "gunc_pass",
            "gtdb_release",
            "gtdb_classification",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]
        if col in dim.columns
    ]
    mag = mag.merge(dim[dim_keep], on="proteome_id", how="left", suffixes=("", "_dim"))

    status = latest_run_status(inputs["run_status"])
    mag = mag.merge(
        status[["proteome_id", "run_status", "has_complete_sentinel", "has_parquet_manifest"]]
        if not status.empty
        else status,
        on="proteome_id",
        how="left",
    )
    mag["functional_status"] = mag["run_status"].fillna("missing_or_not_started")
    mag.loc[mag["analysis_unit_type"].eq("assembly_context"), "functional_status"] = mag.loc[
        mag["analysis_unit_type"].eq("assembly_context"), "functional_status"
    ].replace({"missing_or_not_started": "assembly_context_quarantined"})

    bridge = first_by_proteome(inputs["bridge_cards"], ["rank"])
    if not bridge.empty:
        bridge_keep = [
            col
            for col in [
                "proteome_id",
                "rank",
                "candidate_set",
                "mbag_score_provisional",
                "mbag_score_status",
                "cross_domain_neighbor_fraction",
                "ot_best_coupling",
                "ot_partner",
                "evidence_tier",
                "blocking_caveats",
                "allowed_claim_wording",
                "next_validation_action",
            ]
            if col in bridge.columns
        ]
        mag = mag.merge(bridge[bridge_keep], on="proteome_id", how="left", suffixes=("", "_bridge"))

    mrv = first_by_proteome(inputs["mrv"])
    if not mrv.empty:
        mag = mag.merge(
            mrv[
                [
                    "proteome_id",
                    "kofam_annotated_gene_fraction",
                    "metabolic_modules_present",
                    "cazy_family_count",
                    "merops_family_count",
                    "methane_evidence_score",
                    "sulfur_competition_score",
                    "absence_interpretation_caveat",
                ]
            ],
            on="proteome_id",
            how="left",
        )

    glm = inputs["glm"].copy()
    if not glm.empty:
        glm = glm[glm["payload_name"].astype(str).str.contains("poc_", na=False)].copy()
        glm = first_by_proteome(glm, ["payload_name"])
        mag = mag.merge(
            glm[
                [
                    "proteome_id",
                    "glm_run_id",
                    "model_name",
                    "native_window_count",
                    "shuffled_control_count",
                    "all_embeddings_finite",
                    "embedding_dim",
                    "context_qc_tier",
                ]
            ],
            on="proteome_id",
            how="left",
        )

    for col in ["source", "ecosystem", "domain"]:
        alt = f"{col}_unit"
        if col not in mag.columns and alt in mag.columns:
            mag[col] = mag[alt]
    mag["source"] = mag["source"].fillna(mag.get("source_group", "")).fillna("unknown")
    mag["ecosystem"] = mag["ecosystem"].fillna(mag["source"]).fillna("unknown")
    mag["analysis_unit_type"] = mag["analysis_unit_type"].fillna("unresolved")
    mag["claim_scope"] = mag["claim_scope"].fillna("not comparable")
    mag["mag_node_id"] = "mag:" + mag["proteome_id"].astype(str)
    mag["report_readiness"] = mag.apply(report_readiness, axis=1)
    return mag


def report_readiness(row: pd.Series) -> str:
    if safe_str(row.get("analysis_unit_type")) != "mag_bin":
        return "blocked_noncomparable_unit"
    if safe_str(row.get("functional_status")) != "complete":
        return "blocked_missing_functional_evidence"
    if safe_float(row.get("kofam_annotated_gene_fraction")) <= 0:
        return "blocked_missing_annotation_coverage"
    if safe_str(row.get("glm_run_id")) == "":
        return "needs_glm_context"
    if safe_str(row.get("qc_tier")) != "pass_review":
        return "molecular_attestation_ready_with_qc_caveat"
    return "molecular_attestation_ready_not_mrv"


def build_claim_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "claim:mag_molecular_attestation",
                "claim_name": "MAG/proteome molecular attestation primitive",
                "claim_status": "allowed_for_completed_mag_bins",
                "allowed_wording": "This MAG carries queryable molecular evidence for functional potential at MAG/proteome grain.",
                "forbidden_wording": "This MAG or sample emits methane.",
                "upgrade_requirement": "Review direct markers, QC, annotation coverage, source context, and candidate-card evidence.",
            },
            {
                "id": "claim:bridge_candidate_attestation",
                "claim_name": "ESM2 + function + gLM bridge candidate attestation",
                "claim_status": "provisional_internal",
                "allowed_wording": "This candidate is prioritized by latent geometry and can be reviewed with functional/context evidence.",
                "forbidden_wording": "This proves source-independent rumen-to-wetland methane transfer.",
                "upgrade_requirement": "Complete bridge-card evidence, source-aware nulls, bootstrap stability, and independent sources.",
            },
            {
                "id": "claim:sample_level_methane_risk",
                "claim_name": "Sample-level methane-risk signal",
                "claim_status": "blocked",
                "allowed_wording": "Not scoreable yet; sample metadata, MAG abundance, environmental context, and validation are required.",
                "forbidden_wording": "This sample is high methane risk based on MAG evidence alone.",
                "upgrade_requirement": "Sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, and flux/process validation.",
            },
            {
                "id": "claim:final_mrv_risk_tier",
                "claim_name": "Final A-E MRV methane-risk tier",
                "claim_status": "blocked",
                "allowed_wording": "A-E tiers are target product vocabulary only.",
                "forbidden_wording": "Final A-E risk tier assigned.",
                "upgrade_requirement": "Externally calibrated sample/project risk model with field/process validation.",
            },
            {
                "id": "claim:carbon_credit_approval",
                "claim_name": "Carbon-credit approval or registry verification",
                "claim_status": "forbidden_from_current_molecular_atlas",
                "allowed_wording": "MethaNet can support screening and monitoring design.",
                "forbidden_wording": "MethaNet approves or verifies carbon credits.",
                "upgrade_requirement": "Methodology integration, third-party validation, project-level uncertainty, and measured greenhouse-gas evidence.",
            },
        ]
    )


def build_gap_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("gap:missing_functional_evidence", "functional_evidence", "Functional warehouse or completed per-MAG run is missing.", "high", "complete or preserve explicit missing status"),
            ("gap:missing_glm_context", "glm_context", "gLM2 contextual feature is missing.", "medium", "run or integrate gLM2 context layer"),
            ("gap:noncomparable_assembly_context", "unit_scope", "Assembly-context unit is not valid MAG-level bridge support.", "high", "quarantine or analyze in assembly-context lane"),
            ("gap:missing_sample_metadata", "sample_metadata", "MAG is not mapped to physical sample/site/project grain.", "high", "add sample/MAG crosswalk with provenance"),
            ("gap:missing_abundance", "abundance", "No MAG/read abundance or coverage weighting is integrated.", "high", "join MAG/sample abundance or marker abundance"),
            ("gap:missing_environment", "environmental_covariates", "No salinity, sulfate, redox, depth, hydrology, or other covariates are integrated.", "high", "join environmental measurement table"),
            ("gap:missing_flux_validation", "flux_validation", "No methane flux/process validation is integrated.", "high", "join chamber, tower, porewater, or incubation validation"),
            ("gap:source_confounding", "source_deconfounding", "Current rumen/wetland POC has source/ecosystem confounding.", "high", "add independent sources and source-aware validation"),
        ],
        columns=["id", "gap_type", "required_evidence", "priority", "next_action"],
    )


class GraphBuilder:
    def __init__(self, artifacts: SourceArtifacts) -> None:
        self.artifacts = artifacts
        self.evidence_rows: list[dict[str, Any]] = []
        self.edge_rows: list[dict[str, Any]] = []
        self.feature_rows: list[dict[str, Any]] = []
        self.edge_i = 0

    def add_edge(
        self,
        src_id: str,
        src_type: str,
        rel_type: str,
        dst_id: str,
        dst_type: str,
        *,
        evidence_atom_id: str = "",
        source_artifact_id: str = "",
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.edge_i += 1
        self.edge_rows.append(
            {
                "edge_id": f"edge:{self.edge_i:08d}",
                "src_id": src_id,
                "src_type": src_type,
                "rel_type": rel_type,
                "dst_id": dst_id,
                "dst_type": dst_type,
                "evidence_atom_id": evidence_atom_id,
                "source_artifact_id": source_artifact_id,
                "properties_json": json.dumps(properties or {}, sort_keys=True),
            }
        )

    def add_evidence(
        self,
        *,
        subject_id: str,
        proteome_id: str,
        mag_id: str,
        predicate: str,
        object_type: str,
        object_id: str,
        evidence_type: str,
        evidence_direction: str,
        strength: float,
        confidence_tier: str,
        source_table: str,
        source_tool: str,
        claim_id: str,
        value_json: dict[str, Any] | None = None,
    ) -> str:
        source_artifact_id = self.artifacts.get(source_table)
        evidence_id = f"evidence:{len(self.evidence_rows) + 1:08d}"
        self.evidence_rows.append(
            {
                "id": evidence_id,
                "subject_type": "MAG",
                "subject_id": subject_id,
                "predicate": predicate,
                "object_type": object_type,
                "object_id": object_id,
                "evidence_type": evidence_type,
                "evidence_direction": evidence_direction,
                "evidence_strength": strength,
                "confidence_tier": confidence_tier,
                "source_table": source_table,
                "source_artifact_id": source_artifact_id,
                "source_tool": source_tool,
                "proteome_id": proteome_id,
                "mag_id": mag_id,
                "claim_boundary_id": claim_id,
                "value_json": json.dumps(value_json or {}, sort_keys=True),
            }
        )
        self.add_edge(subject_id, "MAG", "HAS_EVIDENCE", evidence_id, "EvidenceAtom", evidence_atom_id=evidence_id, source_artifact_id=source_artifact_id)
        self.add_edge(evidence_id, "EvidenceAtom", "GENERATED_BY", source_artifact_id, "Artifact", evidence_atom_id=evidence_id, source_artifact_id=source_artifact_id)
        if claim_id:
            self.add_edge(evidence_id, "EvidenceAtom", "EVIDENCE_SUPPORTS_CLAIM", claim_id, "Claim", evidence_atom_id=evidence_id, source_artifact_id=source_artifact_id)
        return evidence_id

    def add_feature(
        self,
        *,
        mag_node_id: str,
        proteome_id: str,
        feature_type: str,
        feature_name: str,
        score: float,
        status: str,
        source_table: str,
        evidence_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        feature_id = f"feature:{feature_type}:{proteome_id}"
        self.feature_rows.append(
            {
                "id": feature_id,
                "feature_type": feature_type,
                "feature_name": feature_name,
                "score": score,
                "status": status,
                "source_table": source_table,
                "properties_json": json.dumps(properties or {}, sort_keys=True),
            }
        )
        self.add_edge(mag_node_id, "MAG", "HAS_FEATURE", feature_id, "Feature", evidence_atom_id=evidence_id, source_artifact_id=self.artifacts.get(source_table))
        return feature_id


def evidence_direction(score: float, missing: bool = False) -> str:
    if missing:
        return "missing"
    if score > 0:
        return "supports"
    return "unknown"


def build_graph_components(
    mag: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    artifacts: SourceArtifacts,
) -> dict[str, pd.DataFrame]:
    claims = build_claim_nodes()
    gaps = build_gap_nodes()
    builder = GraphBuilder(artifacts)

    claim_mag = "claim:mag_molecular_attestation"
    claim_bridge = "claim:bridge_candidate_attestation"
    claim_sample = "claim:sample_level_methane_risk"
    claim_mrv = "claim:final_mrv_risk_tier"

    claim_gap_edges = {
        claim_bridge: [
            "gap:source_confounding",
            "gap:missing_flux_validation",
        ],
        claim_sample: [
            "gap:missing_sample_metadata",
            "gap:missing_abundance",
            "gap:missing_environment",
            "gap:missing_flux_validation",
        ],
        claim_mrv: [
            "gap:missing_sample_metadata",
            "gap:missing_abundance",
            "gap:missing_environment",
            "gap:missing_flux_validation",
            "gap:source_confounding",
        ],
        "claim:carbon_credit_approval": [
            "gap:missing_environment",
            "gap:missing_flux_validation",
            "gap:source_confounding",
        ],
    }
    for claim_id, gap_ids in claim_gap_edges.items():
        for gap_id in gap_ids:
            builder.add_edge(
                claim_id,
                "Claim",
                "CLAIM_BLOCKED_BY",
                gap_id,
                "ValidationGap",
                source_artifact_id=artifacts.get("unit_scope"),
            )

    for row in mag.to_dict("records"):
        node_id = row["mag_node_id"]
        proteome_id = safe_str(row["proteome_id"])
        mag_id = safe_str(row.get("mag_id") or row.get("mag_id_unit") or proteome_id)
        source_key = f"source:{safe_str(row.get('source'), 'unknown')}:{safe_str(row.get('ecosystem'), 'unknown')}"
        builder.add_edge(
            node_id,
            "MAG",
            "FROM_SOURCE",
            source_key,
            "SourceDomain",
            source_artifact_id=artifacts.get("unit_scope"),
        )

        tax_id = ""
        if safe_str(row.get("gtdb_classification")):
            tax_id = f"taxon:{short_hash(safe_str(row.get('gtdb_classification')))}"
            builder.add_edge(node_id, "MAG", "HAS_TAXONOMY", tax_id, "Taxon", source_artifact_id=artifacts.get("taxonomy"))

        if safe_str(row.get("analysis_unit_type")) != "mag_bin":
            builder.add_edge(
                node_id,
                "MAG",
                "BLOCKED_BY",
                "gap:noncomparable_assembly_context",
                "ValidationGap",
                source_artifact_id=artifacts.get("unit_scope"),
            )
        if safe_str(row.get("functional_status")) != "complete":
            builder.add_edge(
                node_id,
                "MAG",
                "BLOCKED_BY",
                "gap:missing_functional_evidence",
                "ValidationGap",
                source_artifact_id=artifacts.get("run_status"),
            )
        if not safe_str(row.get("glm_run_id")):
            builder.add_edge(
                node_id,
                "MAG",
                "BLOCKED_BY",
                "gap:missing_glm_context",
                "ValidationGap",
                source_artifact_id=artifacts.get("glm_features"),
            )
        for gap_id in [
            "gap:missing_sample_metadata",
            "gap:missing_abundance",
            "gap:missing_environment",
            "gap:missing_flux_validation",
            "gap:source_confounding",
        ]:
            builder.add_edge(
                node_id,
                "MAG",
                "BLOCKED_BY",
                gap_id,
                "ValidationGap",
                source_artifact_id=artifacts.get("unit_scope"),
            )

        complete = safe_str(row.get("functional_status")) == "complete"
        if safe_str(row.get("analysis_unit_type")) == "mag_bin" and complete:
            builder.add_edge(
                node_id,
                "MAG",
                "MAG_SUPPORTS_CLAIM",
                claim_mag,
                "Claim",
                source_artifact_id=artifacts.get("mrv"),
            )
        builder.add_edge(
            node_id,
            "MAG",
            "MAG_BLOCKED_FROM_CLAIM",
            claim_sample,
            "Claim",
            source_artifact_id=artifacts.get("unit_scope"),
        )
        builder.add_edge(
            node_id,
            "MAG",
            "MAG_BLOCKED_FROM_CLAIM",
            claim_mrv,
            "Claim",
            source_artifact_id=artifacts.get("unit_scope"),
        )

        builder.add_evidence(
            subject_id=node_id,
            proteome_id=proteome_id,
            mag_id=mag_id,
            predicate="has_run_status",
            object_type="RunStatus",
            object_id=safe_str(row.get("functional_status")),
            evidence_type="run_status",
            evidence_direction="supports" if complete else "missing",
            strength=1.0 if complete else 0.0,
            confidence_tier="high" if complete else "blocked",
            source_table="run_status",
            source_tool="functional_atlas",
            claim_id=claim_mag,
            value_json={"functional_status": safe_str(row.get("functional_status"))},
        )

        methane_score = safe_float(row.get("methane_evidence_score"))
        methane_ev = builder.add_evidence(
            subject_id=node_id,
            proteome_id=proteome_id,
            mag_id=mag_id,
            predicate="has_methane_functional_evidence",
            object_type="Feature",
            object_id=f"feature:methane:{proteome_id}",
            evidence_type="functional_annotation",
            evidence_direction=evidence_direction(methane_score, missing=not complete),
            strength=methane_score,
            confidence_tier="medium" if methane_score > 0 else "unknown",
            source_table="methane",
            source_tool="functional_atlas",
            claim_id=claim_mag,
            value_json={"methane_evidence_score": methane_score},
        )
        builder.add_feature(
            mag_node_id=node_id,
            proteome_id=proteome_id,
            feature_type="methane",
            feature_name="methane_evidence_score",
            score=methane_score,
            status=evidence_direction(methane_score, missing=not complete),
            source_table="methane",
            evidence_id=methane_ev,
        )

        sulfur_score = safe_float(row.get("sulfur_competition_score"))
        sulfur_ev = builder.add_evidence(
            subject_id=node_id,
            proteome_id=proteome_id,
            mag_id=mag_id,
            predicate="has_sulfur_competition_context",
            object_type="Feature",
            object_id=f"feature:sulfur:{proteome_id}",
            evidence_type="functional_annotation",
            evidence_direction=evidence_direction(sulfur_score, missing=not complete),
            strength=sulfur_score,
            confidence_tier="medium" if sulfur_score > 0 else "unknown",
            source_table="sulfur",
            source_tool="functional_atlas",
            claim_id=claim_mag,
            value_json={"sulfur_competition_score": sulfur_score},
        )
        builder.add_feature(
            mag_node_id=node_id,
            proteome_id=proteome_id,
            feature_type="sulfur",
            feature_name="sulfur_competition_score",
            score=sulfur_score,
            status=evidence_direction(sulfur_score, missing=not complete),
            source_table="sulfur",
            evidence_id=sulfur_ev,
        )

        coverage = safe_float(row.get("kofam_annotated_gene_fraction"))
        coverage_ev = builder.add_evidence(
            subject_id=node_id,
            proteome_id=proteome_id,
            mag_id=mag_id,
            predicate="has_annotation_coverage",
            object_type="Feature",
            object_id=f"feature:annotation_coverage:{proteome_id}",
            evidence_type="coverage",
            evidence_direction="supports" if coverage > 0 else "missing",
            strength=coverage,
            confidence_tier="high" if coverage >= 0.5 else "low",
            source_table="mrv",
            source_tool="functional_atlas",
            claim_id=claim_mag,
            value_json={"kofam_annotated_gene_fraction": coverage},
        )
        builder.add_feature(
            mag_node_id=node_id,
            proteome_id=proteome_id,
            feature_type="annotation_coverage",
            feature_name="kofam_annotated_gene_fraction",
            score=coverage,
            status="supports" if coverage > 0 else "missing",
            source_table="mrv",
            evidence_id=coverage_ev,
        )

        if safe_str(row.get("glm_run_id")):
            glm_strength = 1.0 if safe_bool_text(row.get("all_embeddings_finite")) == "true" else 0.5
            glm_ev = builder.add_evidence(
                subject_id=node_id,
                proteome_id=proteome_id,
                mag_id=mag_id,
                predicate="has_glm2_context",
                object_type="Feature",
                object_id=f"feature:glm_context:{proteome_id}",
                evidence_type="context_window",
                evidence_direction="supports",
                strength=glm_strength,
                confidence_tier="medium",
                source_table="glm_features",
                source_tool="gLM2",
                claim_id=claim_bridge,
                value_json={
                    "glm_run_id": safe_str(row.get("glm_run_id")),
                    "native_window_count": safe_float(row.get("native_window_count")),
                    "context_qc_tier": safe_str(row.get("context_qc_tier")),
                },
            )
            builder.add_feature(
                mag_node_id=node_id,
                proteome_id=proteome_id,
                feature_type="glm_context",
                feature_name="gLM2 MAG-level context",
                score=glm_strength,
                status="supports",
                source_table="glm_features",
                evidence_id=glm_ev,
                properties={"context_qc_tier": safe_str(row.get("context_qc_tier"))},
            )

        if pd.notna(row.get("rank")):
            bridge_strength = safe_float(row.get("mbag_score_provisional"))
            bridge_ev = builder.add_evidence(
                subject_id=node_id,
                proteome_id=proteome_id,
                mag_id=mag_id,
                predicate="near_bridge_in_esm2_space",
                object_type="Feature",
                object_id=f"feature:esm2_bridge:{proteome_id}",
                evidence_type="embedding_similarity",
                evidence_direction="supports",
                strength=bridge_strength,
                confidence_tier="provisional",
                source_table="bridge_cards",
                source_tool="ESM2_Mbag",
                claim_id=claim_bridge,
                value_json={
                    "rank": safe_float(row.get("rank")),
                    "cross_domain_neighbor_fraction": safe_float(row.get("cross_domain_neighbor_fraction")),
                    "ot_partner": safe_str(row.get("ot_partner")),
                },
            )
            builder.add_feature(
                mag_node_id=node_id,
                proteome_id=proteome_id,
                feature_type="esm2_bridge",
                feature_name="provisional_bridge_score",
                score=bridge_strength,
                status="provisional_internal",
                source_table="bridge_cards",
                evidence_id=bridge_ev,
                properties={"rank": safe_float(row.get("rank"))},
            )
            builder.add_edge(node_id, "MAG", "MAG_SUPPORTS_CLAIM", claim_bridge, "Claim", evidence_atom_id=bridge_ev, source_artifact_id=artifacts.get("bridge_cards"))

        builder.add_evidence(
            subject_id=node_id,
            proteome_id=proteome_id,
            mag_id=mag_id,
            predicate="blocked_from_final_mrv_risk",
            object_type="Claim",
            object_id=claim_mrv,
            evidence_type="claim_boundary",
            evidence_direction="weakens",
            strength=1.0,
            confidence_tier="blocked",
            source_table="unit_scope",
            source_tool="claim_boundary",
            claim_id=claim_mrv,
            value_json={
                "blocking_gaps": [
                    "sample_metadata",
                    "abundance",
                    "environmental_covariates",
                    "flux_validation",
                ]
            },
        )

    source_nodes = (
        mag[["source", "ecosystem"]]
        .drop_duplicates()
        .fillna("unknown")
        .assign(
            id=lambda df: "source:" + df["source"].astype(str) + ":" + df["ecosystem"].astype(str),
            source_name=lambda df: df["source"].astype(str),
            ecosystem_name=lambda df: df["ecosystem"].astype(str),
        )[["id", "source_name", "ecosystem_name"]]
    )

    tax_rows = []
    for row in mag.dropna(subset=["gtdb_classification"]).to_dict("records"):
        classification = safe_str(row.get("gtdb_classification"))
        if not classification:
            continue
        tax_rows.append(
            {
                "id": f"taxon:{short_hash(classification)}",
                "gtdb_classification": classification,
                "domain": safe_str(row.get("domain")),
                "phylum": safe_str(row.get("phylum")),
                "tax_class": safe_str(row.get("class")),
                "tax_order": safe_str(row.get("order")),
                "family": safe_str(row.get("family")),
                "genus": safe_str(row.get("genus")),
                "species": safe_str(row.get("species")),
            }
        )
    taxon_nodes = pd.DataFrame(tax_rows).drop_duplicates("id") if tax_rows else pd.DataFrame()

    add_esm2_edges(inputs["esm2_edges"], builder, artifacts)

    mag_nodes = pd.DataFrame(
        {
            "id": mag["mag_node_id"],
            "proteome_id": mag["proteome_id"],
            "mag_id": mag.get("mag_id", mag["proteome_id"]).fillna(mag["proteome_id"]),
            "source": mag["source"].fillna("unknown"),
            "ecosystem": mag["ecosystem"].fillna("unknown"),
            "domain": mag.get("domain", "").fillna(""),
            "analysis_unit_type": mag["analysis_unit_type"].fillna("unresolved"),
            "claim_scope": mag["claim_scope"].fillna("not comparable"),
            "functional_status": mag["functional_status"].fillna("missing_or_not_started"),
            "report_readiness": mag["report_readiness"],
            "qc_tier": mag.get("qc_tier", "").fillna(""),
            "checkm2_completeness": mag.get("checkm2_completeness", 0).fillna(0),
            "checkm2_contamination": mag.get("checkm2_contamination", 0).fillna(0),
            "gunc_pass": mag.get("gunc_pass", "").map(safe_bool_text),
            "bridge_rank": pd.to_numeric(mag.get("rank", pd.Series([None] * len(mag))), errors="coerce"),
            "mbag_score_provisional": pd.to_numeric(
                mag.get("mbag_score_provisional", pd.Series([None] * len(mag))),
                errors="coerce",
            ).fillna(0),
            "methane_evidence_score": pd.to_numeric(
                mag.get("methane_evidence_score", pd.Series([None] * len(mag))),
                errors="coerce",
            ).fillna(0),
            "sulfur_competition_score": pd.to_numeric(
                mag.get("sulfur_competition_score", pd.Series([None] * len(mag))),
                errors="coerce",
            ).fillna(0),
            "kofam_annotated_gene_fraction": pd.to_numeric(
                mag.get("kofam_annotated_gene_fraction", pd.Series([None] * len(mag))),
                errors="coerce",
            ).fillna(0),
            "glm_run_id": mag.get("glm_run_id", "").fillna(""),
        }
    )

    return {
        "MAG": mag_nodes,
        "SourceDomain": source_nodes,
        "Taxon": taxon_nodes,
        "Feature": pd.DataFrame(builder.feature_rows).drop_duplicates("id"),
        "EvidenceAtom": pd.DataFrame(builder.evidence_rows),
        "Claim": claims,
        "ValidationGap": gaps,
        "edges": pd.DataFrame(builder.edge_rows),
    }


def add_esm2_edges(esm2_edges: pd.DataFrame, builder: GraphBuilder, artifacts: SourceArtifacts) -> None:
    if esm2_edges.empty:
        return
    for row in esm2_edges.to_dict("records"):
        src = f"mag:{safe_str(row.get('source_id'))}"
        dst = f"mag:{safe_str(row.get('target_id'))}"
        if not src or not dst:
            continue
        builder.add_edge(
            src,
            "MAG",
            "NEAR_IN_ESM2_SPACE",
            dst,
            "MAG",
            source_artifact_id=artifacts.get("esm2_edges"),
            properties={
                "neighbor_rank": safe_float(row.get("neighbor_rank")),
                "distance": safe_float(row.get("distance")),
                "weight": safe_float(row.get("weight")),
                "cross_domain": safe_bool_text(row.get("cross_domain")),
            },
        )


def write_graph_exports(output_dir: Path, components: dict[str, pd.DataFrame], registry: pd.DataFrame) -> None:
    nodes = []
    for node_type, frame in components.items():
        if node_type == "edges":
            continue
        frame = frame.copy()
        if frame.empty:
            continue
        for row in frame.to_dict("records"):
            node_id = row.get("id") or row.get("artifact_id")
            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "display_name": safe_str(row.get("proteome_id") or row.get("claim_name") or row.get("gap_type") or node_id),
                    "properties_json": json.dumps(row, sort_keys=True, default=str),
                }
            )
    artifact_nodes = registry.rename(columns={"artifact_id": "id"})
    components["Artifact"] = artifact_nodes
    for row in artifact_nodes.to_dict("records"):
        nodes.append(
            {
                "node_id": row["id"],
                "node_type": "Artifact",
                "display_name": row["artifact_key"],
                "properties_json": json.dumps(row, sort_keys=True, default=str),
            }
        )

    graph_nodes = pd.DataFrame(nodes)
    graph_edges = components["edges"].copy()
    graph_nodes.to_parquet(output_dir / "graph_nodes.parquet", index=False)
    graph_edges.to_parquet(output_dir / "graph_edges.parquet", index=False)
    graph_nodes.to_csv(output_dir / "graph_nodes.tsv", sep="\t", index=False)
    graph_edges.to_csv(output_dir / "graph_edges.tsv", sep="\t", index=False)
    components["EvidenceAtom"].to_parquet(output_dir / "evidence_atom.parquet", index=False)
    components["EvidenceAtom"].to_csv(output_dir / "evidence_atom.tsv", sep="\t", index=False)


def clean_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].fillna("").astype(str)
    return out


def write_kuzu(output_dir: Path, components: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    try:
        import kuzu
    except ImportError as exc:
        return False, f"kuzu unavailable: {exc}"

    csv_dir = output_dir / "kuzu_csv"
    db_dir = output_dir / "mmag.kuzu"
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    if db_dir.exists():
        if db_dir.is_dir():
            shutil.rmtree(db_dir)
        else:
            db_dir.unlink()
    csv_dir.mkdir(parents=True, exist_ok=True)

    node_tables = {
        "MAG": "id STRING, proteome_id STRING, mag_id STRING, source STRING, ecosystem STRING, domain STRING, analysis_unit_type STRING, claim_scope STRING, functional_status STRING, report_readiness STRING, qc_tier STRING, checkm2_completeness DOUBLE, checkm2_contamination DOUBLE, gunc_pass STRING, bridge_rank DOUBLE, mbag_score_provisional DOUBLE, methane_evidence_score DOUBLE, sulfur_competition_score DOUBLE, kofam_annotated_gene_fraction DOUBLE, glm_run_id STRING, PRIMARY KEY(id)",
        "SourceDomain": "id STRING, source_name STRING, ecosystem_name STRING, PRIMARY KEY(id)",
        "Taxon": "id STRING, gtdb_classification STRING, domain STRING, phylum STRING, tax_class STRING, tax_order STRING, family STRING, genus STRING, species STRING, PRIMARY KEY(id)",
        "Feature": "id STRING, feature_type STRING, feature_name STRING, score DOUBLE, status STRING, source_table STRING, properties_json STRING, PRIMARY KEY(id)",
        "EvidenceAtom": "id STRING, subject_type STRING, subject_id STRING, predicate STRING, object_type STRING, object_id STRING, evidence_type STRING, evidence_direction STRING, evidence_strength DOUBLE, confidence_tier STRING, source_table STRING, source_artifact_id STRING, source_tool STRING, proteome_id STRING, mag_id STRING, claim_boundary_id STRING, value_json STRING, PRIMARY KEY(id)",
        "Claim": "id STRING, claim_name STRING, claim_status STRING, allowed_wording STRING, forbidden_wording STRING, upgrade_requirement STRING, PRIMARY KEY(id)",
        "ValidationGap": "id STRING, gap_type STRING, required_evidence STRING, priority STRING, next_action STRING, PRIMARY KEY(id)",
        "Artifact": "id STRING, artifact_key STRING, artifact_uri STRING, artifact_class STRING, artifact_format STRING, source_system STRING, source_tool STRING, size_bytes INT64, modified_at STRING, sha256 STRING, license_or_use_caveat STRING, PRIMARY KEY(id)",
    }

    db = kuzu.Database(str(db_dir))
    conn = kuzu.Connection(db)
    for table, ddl in node_tables.items():
        frame = components.get(table, pd.DataFrame())
        if frame.empty:
            frame = pd.DataFrame(columns=[part.split()[0] for part in ddl.split(", ") if not part.startswith("PRIMARY")])
        csv_path = csv_dir / f"{table}.csv"
        clean_csv_frame(frame).to_csv(csv_path, index=False)
        conn.execute(f"CREATE NODE TABLE {table}({ddl})")
        conn.execute(f"COPY {table} FROM '{csv_path}' (HEADER=true)")

    edges = components["edges"].copy()
    rel_specs = {
        "FROM_SOURCE": ("MAG", "SourceDomain"),
        "HAS_TAXONOMY": ("MAG", "Taxon"),
        "HAS_FEATURE": ("MAG", "Feature"),
        "HAS_EVIDENCE": ("MAG", "EvidenceAtom"),
        "GENERATED_BY": ("EvidenceAtom", "Artifact"),
        "EVIDENCE_SUPPORTS_CLAIM": ("EvidenceAtom", "Claim"),
        "MAG_SUPPORTS_CLAIM": ("MAG", "Claim"),
        "MAG_BLOCKED_FROM_CLAIM": ("MAG", "Claim"),
        "CLAIM_BLOCKED_BY": ("Claim", "ValidationGap"),
        "BLOCKED_BY": ("MAG", "ValidationGap"),
        "NEAR_IN_ESM2_SPACE": ("MAG", "MAG"),
    }
    for rel_type, (src_type, dst_type) in rel_specs.items():
        rel = edges[
            (edges["rel_type"] == rel_type)
            & (edges["src_type"] == src_type)
            & (edges["dst_type"] == dst_type)
        ].copy()
        rel_frame = pd.DataFrame(
            {
                "from": rel["src_id"] if not rel.empty else [],
                "to": rel["dst_id"] if not rel.empty else [],
                "edge_id": rel["edge_id"] if not rel.empty else [],
                "evidence_atom_id": rel["evidence_atom_id"] if not rel.empty else [],
                "source_artifact_id": rel["source_artifact_id"] if not rel.empty else [],
                "properties_json": rel["properties_json"] if not rel.empty else [],
            }
        )
        csv_path = csv_dir / f"{rel_type}.csv"
        clean_csv_frame(rel_frame).to_csv(csv_path, index=False)
        conn.execute(
            f"CREATE REL TABLE {rel_type}(FROM {src_type} TO {dst_type}, edge_id STRING, evidence_atom_id STRING, source_artifact_id STRING, properties_json STRING)"
        )
        if not rel_frame.empty:
            conn.execute(f"COPY {rel_type} FROM '{csv_path}' (HEADER=true)")
    return True, "kuzu graph built"


QUERY_LIBRARY = {
    "top_complete_multiview_bridge_candidates": """
MATCH (m:MAG)-[:HAS_FEATURE]->(b:Feature),
      (m)-[:HAS_FEATURE]->(meth:Feature),
      (m)-[:HAS_FEATURE]->(glm:Feature)
WHERE b.feature_type = 'esm2_bridge'
  AND meth.feature_type = 'methane'
  AND glm.feature_type = 'glm_context'
  AND m.functional_status = 'complete'
RETURN m.proteome_id AS proteome_id,
       m.source AS source,
       m.ecosystem AS ecosystem,
       b.score AS provisional_bridge_score,
       meth.score AS methane_score,
       glm.status AS glm_status,
       m.report_readiness AS readiness
ORDER BY provisional_bridge_score DESC
LIMIT 25
""",
    "wetland_methane_plus_sulfur_context": """
MATCH (m:MAG)-[:HAS_FEATURE]->(meth:Feature),
      (m)-[:HAS_FEATURE]->(sulf:Feature)
WHERE m.source = 'mucc'
  AND meth.feature_type = 'methane'
  AND sulf.feature_type = 'sulfur'
  AND meth.score > 0
  AND sulf.score > 0
RETURN m.proteome_id AS proteome_id,
       m.qc_tier AS qc_tier,
       meth.score AS methane_score,
       sulf.score AS sulfur_score,
       m.report_readiness AS readiness
ORDER BY methane_score DESC, sulfur_score DESC
LIMIT 25
""",
    "strong_bridge_but_blocked_or_weak_qc": """
MATCH (m:MAG)-[:HAS_FEATURE]->(b:Feature),
      (m)-[:HAS_FEATURE]->(cov:Feature)
WHERE b.feature_type = 'esm2_bridge'
  AND cov.feature_type = 'annotation_coverage'
  AND (m.functional_status <> 'complete'
       OR m.qc_tier <> 'pass_review'
       OR cov.score < 0.5)
RETURN m.proteome_id AS proteome_id,
       b.score AS bridge_score,
       m.functional_status AS functional_status,
       m.qc_tier AS qc_tier,
       cov.score AS annotation_coverage,
       m.report_readiness AS readiness
ORDER BY bridge_score DESC
LIMIT 30
""",
    "candidate_evidence_path": """
MATCH (m:MAG)-[:HAS_EVIDENCE]->(e:EvidenceAtom)-[:GENERATED_BY]->(a:Artifact)
WHERE m.proteome_id = 'mucc__GCA_002495465.1_ASM249546v1_genomic'
RETURN m.proteome_id AS proteome_id,
       e.predicate AS predicate,
       e.evidence_direction AS direction,
       e.evidence_strength AS strength,
       e.confidence_tier AS confidence,
       a.artifact_key AS artifact
ORDER BY predicate
LIMIT 50
""",
    "source_taxonomy_confounding_patterns": """
MATCH (m:MAG)-[:FROM_SOURCE]->(s:SourceDomain)
OPTIONAL MATCH (m)-[:HAS_TAXONOMY]->(t:Taxon)
RETURN s.source_name AS source,
       s.ecosystem_name AS ecosystem,
       t.family AS family,
       count(m) AS mag_count
ORDER BY mag_count DESC
LIMIT 30
""",
    "report_ready_vs_blocked_counts": """
MATCH (m:MAG)
RETURN m.report_readiness AS report_readiness,
       count(m) AS mag_count
ORDER BY mag_count DESC
""",
    "molecular_attestation_not_final_mrv": """
MATCH (m:MAG)-[:MAG_SUPPORTS_CLAIM]->(allowed:Claim),
      (m)-[:MAG_BLOCKED_FROM_CLAIM]->(blocked:Claim)
WHERE allowed.id = 'claim:mag_molecular_attestation'
  AND blocked.id = 'claim:final_mrv_risk_tier'
RETURN m.proteome_id AS proteome_id,
       m.report_readiness AS readiness,
       allowed.allowed_wording AS allowed_wording,
       blocked.allowed_wording AS blocked_wording
LIMIT 25
""",
    "claim_upgrade_blocker_register": """
MATCH (claim:Claim)-[:CLAIM_BLOCKED_BY]->(gap:ValidationGap)
RETURN claim.claim_name AS claim_name,
       claim.claim_status AS claim_status,
       gap.gap_type AS blocking_gap,
       gap.required_evidence AS required_evidence,
       gap.next_action AS next_action
ORDER BY claim_name, blocking_gap
""",
    "candidate_full_evidence_packet": """
MATCH (m:MAG)-[:HAS_FEATURE]->(f:Feature),
      (m)-[:HAS_EVIDENCE]->(e:EvidenceAtom)-[:GENERATED_BY]->(a:Artifact)
WHERE m.proteome_id = 'rumen__10674_0004_idba_bin.23'
RETURN m.proteome_id AS proteome_id,
       f.feature_type AS feature_type,
       f.score AS feature_score,
       e.predicate AS evidence_predicate,
       e.evidence_direction AS direction,
       e.evidence_strength AS evidence_strength,
       a.artifact_key AS source_artifact
ORDER BY feature_type, evidence_predicate
LIMIT 100
""",
    "cross_domain_neighbor_evidence": """
MATCH (m:MAG)-[:NEAR_IN_ESM2_SPACE]->(n:MAG)
WHERE m.source <> n.source
RETURN m.proteome_id AS query_proteome_id,
       m.source AS query_source,
       n.proteome_id AS neighbor_proteome_id,
       n.source AS neighbor_source,
       m.methane_evidence_score AS query_methane_score,
       n.methane_evidence_score AS neighbor_methane_score,
       m.report_readiness AS query_readiness,
       n.report_readiness AS neighbor_readiness
LIMIT 50
""",
    "artifact_provenance_fanout": """
MATCH (e:EvidenceAtom)-[:GENERATED_BY]->(a:Artifact)
RETURN a.artifact_key AS artifact,
       count(e) AS evidence_atoms,
       collect(DISTINCT e.predicate) AS predicates
ORDER BY evidence_atoms DESC
""",
    "assembly_context_quarantine": """
MATCH (m:MAG)-[:BLOCKED_BY]->(gap:ValidationGap)
WHERE m.analysis_unit_type = 'assembly_context'
  AND gap.id = 'gap:noncomparable_assembly_context'
RETURN count(m) AS quarantined_assembly_context_units
""",
}

QUERY_MIN_ROWS = {
    "top_complete_multiview_bridge_candidates": 1,
    "wetland_methane_plus_sulfur_context": 1,
    "strong_bridge_but_blocked_or_weak_qc": 1,
    "candidate_evidence_path": 1,
    "source_taxonomy_confounding_patterns": 1,
    "report_ready_vs_blocked_counts": 3,
    "molecular_attestation_not_final_mrv": 1,
    "claim_upgrade_blocker_register": 5,
    "candidate_full_evidence_packet": 1,
    "cross_domain_neighbor_evidence": 1,
    "artifact_provenance_fanout": 5,
    "assembly_context_quarantine": 1,
}


def write_query_library(output_dir: Path, kuzu_built: bool) -> pd.DataFrame:
    query_dir = output_dir / "query_results"
    query_dir.mkdir(exist_ok=True)
    library_path = output_dir / "QUERY_LIBRARY.cypher"
    with library_path.open("w") as handle:
        for name, query in QUERY_LIBRARY.items():
            handle.write(f"// {name}\n{query.strip()}\n\n")

    rows = []
    if not kuzu_built:
        for name in QUERY_LIBRARY:
            rows.append({"query_name": name, "status": "not_run_no_kuzu", "rows": 0, "path": ""})
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "query_results_summary.tsv", sep="\t", index=False)
        return summary

    import kuzu

    conn = kuzu.Connection(kuzu.Database(str(output_dir / "mmag.kuzu")))
    for name, query in QUERY_LIBRARY.items():
        result = conn.execute(query)
        frame = result.get_as_df()
        out = query_dir / f"{name}.tsv"
        frame.to_csv(out, sep="\t", index=False)
        rows.append({"query_name": name, "status": "ok", "rows": len(frame), "path": str(out)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "query_results_summary.tsv", sep="\t", index=False)
    return summary


def validate_snapshot(
    components: dict[str, pd.DataFrame],
    inputs: dict[str, pd.DataFrame],
    query_summary: pd.DataFrame,
    kuzu_built: bool,
) -> pd.DataFrame:
    mag = components["MAG"]
    evidence = components["EvidenceAtom"]
    edges = components["edges"]
    claims = components["Claim"]
    validations = []

    def add(gate: str, status: str, observed: Any, expected: Any, detail: str = "") -> None:
        validations.append(
            {
                "gate": gate,
                "status": status,
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    add("cohort_node_count", "pass" if len(mag) == EXPECTED_COHORT_ROWS else "fail", len(mag), EXPECTED_COHORT_ROWS)
    add("unique_proteome_id", "pass" if mag["proteome_id"].is_unique else "fail", mag["proteome_id"].nunique(), len(mag))
    mag_bin_count = int(mag["analysis_unit_type"].eq("mag_bin").sum())
    assembly_count = int(mag["analysis_unit_type"].eq("assembly_context").sum())
    add("mag_bin_count", "pass" if mag_bin_count == EXPECTED_MAG_BIN_ROWS else "fail", mag_bin_count, EXPECTED_MAG_BIN_ROWS)
    add(
        "assembly_context_count",
        "pass" if assembly_count == EXPECTED_ASSEMBLY_CONTEXT_ROWS else "fail",
        assembly_count,
        EXPECTED_ASSEMBLY_CONTEXT_ROWS,
    )
    completed = inputs["dim_mag"]["proteome_id"].astype(str)
    missing_completed = sorted(set(completed) - set(mag["proteome_id"].astype(str)))
    add("no_dropped_completed_mags", "pass" if not missing_completed else "fail", len(missing_completed), 0, ";".join(missing_completed[:5]))

    node_ids = []
    for node_type, frame in components.items():
        if node_type == "edges":
            continue
        id_col = "artifact_id" if node_type == "Artifact" and "artifact_id" in frame.columns else "id"
        if id_col in frame.columns:
            node_ids.extend(frame[id_col].astype(str).tolist())
    add("no_duplicate_primary_graph_entities", "pass" if len(node_ids) == len(set(node_ids)) else "fail", len(node_ids) - len(set(node_ids)), 0)
    node_id_set = set(node_ids)

    missing_src = sorted(set(edges["src_id"].astype(str)) - node_id_set)
    missing_dst = sorted(set(edges["dst_id"].astype(str)) - node_id_set)
    add(
        "all_graph_edge_endpoints_resolve",
        "pass" if not missing_src and not missing_dst else "fail",
        len(missing_src) + len(missing_dst),
        0,
        ";".join((missing_src + missing_dst)[:5]),
    )

    missing_edge_sources = edges[edges["source_artifact_id"].fillna("").astype(str).eq("")]
    add(
        "all_graph_edges_have_source_or_boundary_provenance",
        "pass" if missing_edge_sources.empty else "fail",
        len(missing_edge_sources),
        0,
        ",".join(missing_edge_sources["rel_type"].drop_duplicates().head(5).astype(str)),
    )

    source_artifact_ids = set(edges["source_artifact_id"].dropna().astype(str)) - {""}
    missing_artifact_nodes = sorted(source_artifact_ids - node_id_set)
    add(
        "all_edge_source_artifacts_are_nodes",
        "pass" if not missing_artifact_nodes else "fail",
        len(missing_artifact_nodes),
        0,
        ";".join(missing_artifact_nodes[:5]),
    )

    generated = edges[edges["rel_type"].eq("GENERATED_BY")]
    evidence_with_provenance = set(generated["src_id"].astype(str))
    missing_provenance = sorted(set(evidence["id"].astype(str)) - evidence_with_provenance)
    add("no_evidence_without_source_provenance", "pass" if not missing_provenance else "fail", len(missing_provenance), 0, ";".join(missing_provenance[:5]))

    forbidden_evidence = evidence[
        evidence["predicate"].str.contains("final_mrv_risk_tier_assigned|carbon_credit_approved", case=False, na=False)
    ]
    final_claim = claims[claims["id"].eq("claim:final_mrv_risk_tier")]
    final_blocked = not final_claim.empty and final_claim.iloc[0]["claim_status"] == "blocked"
    add(
        "no_final_mrv_risk_claim_encoded_as_fact",
        "pass" if forbidden_evidence.empty and final_blocked else "fail",
        len(forbidden_evidence),
        0,
        "final_mrv_risk_tier claim must remain blocked",
    )

    final_mrv_support = edges[
        edges["rel_type"].eq("MAG_SUPPORTS_CLAIM")
        & edges["dst_id"].eq("claim:final_mrv_risk_tier")
    ]
    add("no_mag_supports_final_mrv_claim", "pass" if final_mrv_support.empty else "fail", len(final_mrv_support), 0)

    final_mrv_block = edges[
        edges["rel_type"].eq("MAG_BLOCKED_FROM_CLAIM")
        & edges["dst_id"].eq("claim:final_mrv_risk_tier")
    ]
    sample_block = edges[
        edges["rel_type"].eq("MAG_BLOCKED_FROM_CLAIM")
        & edges["dst_id"].eq("claim:sample_level_methane_risk")
    ]
    add("all_mags_blocked_from_final_mrv_claim", "pass" if len(final_mrv_block) == len(mag) else "fail", len(final_mrv_block), len(mag))
    add("all_mags_blocked_from_sample_risk_claim", "pass" if len(sample_block) == len(mag) else "fail", len(sample_block), len(mag))

    assembly_block = edges[
        edges["rel_type"].eq("BLOCKED_BY")
        & edges["dst_id"].eq("gap:noncomparable_assembly_context")
    ]
    add(
        "assembly_context_units_have_quarantine_edges",
        "pass" if len(assembly_block) == EXPECTED_ASSEMBLY_CONTEXT_ROWS else "fail",
        len(assembly_block),
        EXPECTED_ASSEMBLY_CONTEXT_ROWS,
    )

    query_failures = 0 if query_summary.empty else int(query_summary["status"].ne("ok").sum())
    add("kuzu_built", "pass" if kuzu_built else "warn", kuzu_built, True)
    add("canonical_queries_executed", "pass" if query_failures == 0 else "warn", query_failures, 0)
    for query_name, min_rows in QUERY_MIN_ROWS.items():
        observed_rows = 0
        if not query_summary.empty and query_name in set(query_summary["query_name"]):
            observed_rows = int(query_summary.loc[query_summary["query_name"].eq(query_name), "rows"].iloc[0])
        if not kuzu_built:
            add(
                f"query_min_rows:{query_name}",
                "warn",
                "not_run_no_kuzu",
                min_rows,
                "Kuzu is optional for repository smoke validation; install the attestation extra to enforce graph query row gates.",
            )
            continue
        add(
            f"query_min_rows:{query_name}",
            "pass" if observed_rows >= min_rows else "fail",
            observed_rows,
            min_rows,
        )
    return pd.DataFrame(validations)


def write_validation_report(output_dir: Path, validation: pd.DataFrame, query_summary: pd.DataFrame, kuzu_message: str) -> None:
    validation.to_csv(output_dir / "validation_gates.tsv", sep="\t", index=False)
    status_counts = validation["status"].value_counts().to_dict()
    report = [
        "# MethaNet Molecular Attestation MVP Validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Kuzu status: {kuzu_message}",
        "",
        "## Gate Summary",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        report.append(f"| {status} | {count} |")
    report.extend(["", "## Gates", "", "| Gate | Status | Observed | Expected | Detail |", "| --- | --- | ---: | ---: | --- |"])
    for row in validation.to_dict("records"):
        report.append(
            f"| {row['gate']} | {row['status']} | {row['observed']} | {row['expected']} | {safe_str(row.get('detail'))} |"
        )
    report.extend(["", "## Query Results", "", "| Query | Status | Rows |", "| --- | --- | ---: |"])
    for row in query_summary.to_dict("records"):
        report.append(f"| {row['query_name']} | {row['status']} | {row['rows']} |")
    report.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This snapshot encodes MAG/proteome-level molecular attestation primitives only. It explicitly blocks sample-level methane-risk scoring, final A-E MRV risk tiers, measured flux claims, and carbon-credit approval claims until sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, and flux/process validation are available.",
        ]
    )
    (output_dir / "validation_report.md").write_text("\n".join(report) + "\n")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    snapshot_id = args.snapshot_id or datetime.now().strftime("mmag_mvp_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or repo_root / "results/attestation" / snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    registry, artifacts = build_artifact_registry(repo_root, output_dir, inputs["_table_paths"])
    mag = build_mag_table(inputs)
    components = build_graph_components(mag, inputs, artifacts)
    write_graph_exports(output_dir, components, registry)
    kuzu_built = False
    kuzu_message = "skipped"
    if not args.skip_kuzu:
        kuzu_built, kuzu_message = write_kuzu(output_dir, components)
    query_summary = write_query_library(output_dir, kuzu_built)
    validation = validate_snapshot(components, inputs, query_summary, kuzu_built)
    write_validation_report(output_dir, validation, query_summary, kuzu_message)

    failures = validation[validation["status"].eq("fail")]
    print(f"snapshot_id={snapshot_id}")
    print(f"output_dir={output_dir}")
    print(f"kuzu={kuzu_message}")
    print(f"validation_failures={len(failures)}")
    if not failures.empty:
        print(failures.to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
