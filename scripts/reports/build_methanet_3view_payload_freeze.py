#!/usr/bin/env python3
"""Build a dated MethaNet 3-view payload freeze decision.

The freeze contract is intentionally conservative. It lists unit-level ESM2,
gLM2, and functional-annotation availability for registered atlas lanes and
marks whether the requested lanes are actually green. A blocked decision is a
valid monitor artifact, but it is not a release freeze for the next atlas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_atlas_lane_registry as lane_registry


TRUE_VALUES = {"true", "1", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lane-registry", type=Path, default=Path("configs/methanet_atlas_lanes.tsv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--exclusion-tsv",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional TSV of explicitly release-excluded units. Required columns: "
            "lane_id, proteome_id. Recommended columns: exclusion_reason, "
            "exclusion_scope, approved_by, approved_at_utc."
        ),
    )
    parser.add_argument(
        "--exclude-unit",
        action="append",
        default=[],
        metavar="LANE_ID:PROTEOME_ID:REASON",
        help=(
            "Inline release exclusion. The first two colon-delimited fields are "
            "lane_id and proteome_id; the remainder is the reason."
        ),
    )
    parser.add_argument(
        "--require-green-lane",
        action="append",
        default=[],
        help=(
            "Lane that must be fully tri-view complete before freeze_allowed=true. "
            "Can be supplied multiple times; defaults to every registered lane."
        ),
    )
    parser.add_argument(
        "--exit-nonzero-if-blocked",
        action="store_true",
        help="Exit with status 2 when required lanes are not green.",
    )
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def resolve(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def resolve_many(repo_root: Path, value: str | None) -> list[Path]:
    return lane_registry.resolve_many(repo_root, value)


def read_tsv(path: Path | None) -> list[dict[str, str]]:
    return lane_registry.read_tsv(path)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def all_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def build_release_entities(
    repo_root: Path,
    registry_rows: list[dict[str, str]],
    unit_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    dim_atlas_unit = [
        {key: row.get(key, "") for key in [
            "lane_id", "lane_role", "proteome_id", "mag_id", "source", "ecosystem", "domain",
            "functional_status", "release_required", "release_excluded", "claim_scope",
        ]}
        for row in unit_rows
    ]
    fact_view_availability: list[dict[str, Any]] = []
    for row in unit_rows:
        for view, field, protocol in [
            ("esm2", "has_esm2", "proteome_embedding"),
            ("glm2", "has_glm2", row.get("glm2_protocol_class", "")),
            ("functional", "has_functional", row.get("functional_evidence_class", "")),
        ]:
            fact_view_availability.append({
                "lane_id": row["lane_id"],
                "proteome_id": row["proteome_id"],
                "view": view,
                "available": row.get(field, "false"),
                "protocol_or_contract": protocol,
                "missingness_semantics": "registered_explicit_absence_or_gap; never imputed",
            })

    fact_evidence_contract: list[dict[str, Any]] = []
    feature_specs = [
        ("esm2_latent_neighborhood", "embedding metadata row", "registered proteome", "artifact row exists"),
        ("glm2_genomic_context", "registered gLM2 feature row", "registered proteome within protocol class", "artifact row exists"),
        ("methane_screening", "accepted KOfam and present METABOLIC; MCycDB best hits separate", "Prodigal proteins and tool catalogs", "accepted/present events"),
        ("sulfur_associated_screening", "best-ranked SCycDB plus accepted KOfam and present METABOLIC", "Prodigal proteins and tool catalogs", "accepted/present/best-hit events"),
        ("qc_taxonomy", "CheckM2, GUNC, GTDB-Tk", "completed functional bundle", "tool outputs present"),
        ("sample_linkage", "source accessions and explicit mapping tables", "registered sample or context", "declared resolution tier"),
        ("expression_detection", "processed expression detection/occupancy", "MUCC processed sample columns", "nonzero detection; magnitude not authorized"),
    ]
    units_by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in unit_rows:
        units_by_lane.setdefault(str(row["lane_id"]), []).append(row)
    for registry_row in registry_rows:
        lane_id = str(registry_row.get("lane_id") or "")
        lane_units = units_by_lane.get(lane_id, [])
        contract_state = lane_units[0].get("mechanism_equivalence_status", "") if lane_units else ""
        for family, numerator, denominator, threshold in feature_specs:
            if family == "expression_detection" and lane_id != "mucc_v1_owc_wetland":
                availability = "not_applicable"
            else:
                availability = "available_or_partially_available"
            fact_evidence_contract.append({
                "lane_id": lane_id,
                "feature_family": family,
                "numerator_provenance": numerator,
                "denominator": denominator,
                "threshold": threshold,
                "missingness_semantics": "unknown and absent remain distinct; gaps are explicit rows",
                "comparability_state": (
                    contract_state if family in {"methane_screening", "sulfur_associated_screening"}
                    else "protocol_or_source_stratification_required"
                ),
                "availability": availability,
                "allowed_use": "molecular screening, evidence review, and next-measurement design",
                "forbidden_use": "sample risk, process rate, flux attribution, final tier, or crediting conclusion",
                "next_validation_action": "lock versions/checksums and run source-aware semantic, null, stability, and field-validation gates",
            })

    fact_run_status: list[dict[str, Any]] = []
    selected_by_key = {
        (str(row["lane_id"]), str(row["proteome_id"])): str(row.get("selected_run_dir") or "")
        for row in unit_rows
    }
    for registry_row in registry_rows:
        lane_id = str(registry_row.get("lane_id") or "")
        for per_mag_dir in resolve_many(repo_root, registry_row.get("functional_per_mag_dirs")):
            if not per_mag_dir.exists():
                continue
            for proteome_dir in sorted(path for path in per_mag_dir.iterdir() if path.is_dir()):
                for run_dir in sorted(path for path in proteome_dir.iterdir() if path.is_dir()):
                    rel_run = str(run_dir.relative_to(repo_root)) if run_dir.is_relative_to(repo_root) else str(run_dir)
                    status = lane_registry.run_dir_status(run_dir)
                    fact_run_status.append({
                        "lane_id": lane_id,
                        "proteome_id": proteome_dir.name,
                        "run_id": run_dir.name,
                        "run_status": status,
                        "selected_for_release": str(
                            selected_by_key.get((lane_id, proteome_dir.name), "") in {str(run_dir), rel_run}
                            and status == "complete"
                        ).lower(),
                        "run_dir": rel_run,
                        "integrity_status": "failed_or_quarantined" if status == "failed" and (run_dir / "COMPLETE").exists() else "status_validated",
                    })

    fact_source_provenance = [{
        "lane_id": row.get("lane_id", ""),
        "source_lane_manifest": row.get("source_lane_manifest", ""),
        "functional_manifest": row.get("functional_manifest", ""),
        "source_provenance_dir": row.get("source_provenance_dir", ""),
        "source_provenance_checksums": row.get("source_provenance_checksums", ""),
        "claim_scope": row.get("claim_scope", ""),
    } for row in registry_rows]
    fact_metadata_resolution = [{
        "lane_id": row["lane_id"],
        "proteome_id": row["proteome_id"],
        "resolution_tier": row.get("metadata_resolution", "unresolved"),
        "mapping_method": "source_manifest_declared",
        "mapping_confidence": "source_declared_not_forced_to_one_to_one",
        "ambiguity_set_size": "unknown_or_group_level",
        "provenance_pointer": "registered source manifest",
        "claim_scope": "context only unless exact sample-MAG linkage is independently validated",
    } for row in unit_rows]
    fact_validation_gap = [{
        "lane_id": row["lane_id"],
        "proteome_id": row["proteome_id"],
        "gap": row.get("blocking_gap", ""),
        "affected_claim": "mechanism comparison, sample risk, field validation, or calibration",
        "next_action": row.get("next_validation_action", ""),
        "release_excluded": row.get("release_excluded", "false"),
    } for row in unit_rows]
    feature_multiview_mag_level = [{
        key: row.get(key, "") for key in [
            "lane_id", "proteome_id", "mag_id", "has_esm2", "has_glm2", "has_functional",
            "payload_complete", "schema_normalized", "mechanism_comparable", "glm2_protocol_class",
            "qc_taxonomy_available", "metadata_resolution",
        ]
    } for row in unit_rows]
    feature_bridge_evidence = [{
        "lane_id": row["lane_id"],
        "proteome_id": row["proteome_id"],
        "bridge_evidence_status": "not_ranked_in_freeze",
        "direct_evidence_status": row.get("functional_evidence_class", ""),
        "propagated_evidence_status": "not_authorized_without_source_aware_stability_gates",
        "claim_scope": "candidate review priority only",
    } for row in unit_rows]
    feature_sample_linkage_readiness = [{
        "lane_id": row["lane_id"],
        "proteome_id": row["proteome_id"],
        "resolution_tier": row.get("metadata_resolution", "unresolved"),
        "sample_linked": row.get("sample_linked", "false"),
        "abundance_weighted": row.get("abundance_weighted", "false"),
        "environmentally_contextualized": row.get("environmentally_contextualized", "false"),
        "field_validated": row.get("field_validated", "false"),
        "calibrated": row.get("calibrated", "false"),
    } for row in unit_rows]
    candidate_evidence_card = [{
        "lane_id": row["lane_id"],
        "proteome_id": row["proteome_id"],
        "mag_id": row.get("mag_id", ""),
        "source": row.get("source", ""),
        "ecosystem": row.get("ecosystem", ""),
        "taxonomy": row.get("domain", ""),
        "qc_and_annotation_coverage": row.get("qc_taxonomy_available", "false"),
        "esm2_neighborhood_evidence": "available" if truthy(row.get("has_esm2")) else "missing",
        "glm2_context_evidence": row.get("glm2_protocol_class", "") if truthy(row.get("has_glm2")) else "missing",
        "direct_methane_evidence": "screening facts require warehouse query",
        "direct_sulfur_evidence": "screening breadth only; competition not validated",
        "expression_evidence": "detection/occupancy only where MUCC evidence exists",
        "evidence_contract_state": row.get("mechanism_equivalence_status", ""),
        "metadata_resolution": row.get("metadata_resolution", "unresolved"),
        "sample_linkage_state": row.get("sample_linkage_readiness", ""),
        "uncertainty_and_stability_tier": "not_ranked_source_aware_gates_pending",
        "supporting_evidence": "tri-view availability is recorded without imputing missing evidence",
        "contradictory_evidence": "not systematically recomputed in freeze",
        "missing_evidence": row.get("blocking_gap", ""),
        "allowed_claim_wording": row.get("allowed_claim_wording", ""),
        "forbidden_claim_wording": "sample risk, flux, activity magnitude, final tier, transfer, or crediting approval",
        "next_highest_value_measurement": row.get("next_validation_action", ""),
    } for row in unit_rows]
    release_claim_boundary = [
        {"claim": "MAG/proteome molecular screening", "status": "allowed", "wording": "Molecular screening evidence and review priorities", "blocking_gap": "none at payload-complete grain"},
        {"claim": "validated mechanism-comparable atlas", "status": "blocked", "wording": "Do not claim", "blocking_gap": "semantic rebuild, version/database fingerprints, and cross-lane audit"},
        {"claim": "sample methane-risk score", "status": "blocked", "wording": "Do not claim", "blocking_gap": "exact mapping, abundance, environment, uncertainty, and field validation"},
        {"claim": "measured methane flux", "status": "blocked", "wording": "Do not claim from molecular atlas", "blocking_gap": "authoritative paired process observations"},
        {"claim": "expression-derived activity", "status": "blocked", "wording": "Use detection/occupancy only", "blocking_gap": "normalization and exact ecological joins"},
        {"claim": "source-independent transfer", "status": "blocked", "wording": "Do not claim", "blocking_gap": "multi-source validation and leave-source-out generalization"},
        {"claim": "MRV or carbon-credit approval", "status": "blocked", "wording": "Do not claim", "blocking_gap": "calibrated validated end-to-end MRV contract"},
    ]
    return {
        "dim_atlas_unit": dim_atlas_unit,
        "fact_view_availability": fact_view_availability,
        "fact_evidence_contract": fact_evidence_contract,
        "fact_run_status": fact_run_status,
        "fact_source_provenance": fact_source_provenance,
        "fact_metadata_resolution": fact_metadata_resolution,
        "fact_validation_gap": fact_validation_gap,
        "feature_multiview_mag_level": feature_multiview_mag_level,
        "feature_bridge_evidence": feature_bridge_evidence,
        "feature_sample_linkage_readiness": feature_sample_linkage_readiness,
        "candidate_evidence_card": candidate_evidence_card,
        "release_claim_boundary": release_claim_boundary,
    }


def release_exclusion_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("lane_id", "")).strip(), str(row.get("proteome_id", "")).strip())


def read_release_exclusions(
    repo_root: Path,
    paths: list[Path],
    inline_specs: list[str],
) -> dict[tuple[str, str], dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        resolved = resolve(repo_root, str(path)) or path
        rows.extend(read_tsv(resolved))
    generated_at = datetime.now(timezone.utc).isoformat()
    for spec in inline_specs:
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise SystemExit(f"--exclude-unit must be LANE_ID:PROTEOME_ID[:REASON], got: {spec}")
        rows.append(
            {
                "lane_id": parts[0],
                "proteome_id": parts[1],
                "exclusion_reason": parts[2] if len(parts) == 3 else "release-approved exclusion",
                "exclusion_scope": "report_release_denominator",
                "approved_by": "operator",
                "approved_at_utc": generated_at,
            }
        )

    exclusions: dict[tuple[str, str], dict[str, str]] = {}
    for idx, row in enumerate(rows, start=1):
        lane_id = str(row.get("lane_id", "")).strip()
        proteome_id = str(row.get("proteome_id", "")).strip()
        if not lane_id or not proteome_id:
            raise SystemExit(f"Release exclusion row {idx} is missing lane_id or proteome_id")
        key = (lane_id, proteome_id)
        if key in exclusions:
            raise SystemExit(f"Duplicate release exclusion for {lane_id}:{proteome_id}")
        exclusions[key] = {
            "exclusion_reason": row.get("exclusion_reason") or row.get("reason") or "release-approved exclusion",
            "exclusion_scope": row.get("exclusion_scope") or "report_release_denominator",
            "approved_by": row.get("approved_by") or "operator",
            "approved_at_utc": row.get("approved_at_utc") or generated_at,
        }
    return exclusions


def apply_release_exclusions(
    rows: list[dict[str, Any]],
    exclusions: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[tuple[str, str]] = set()
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        key = release_exclusion_key(row)
        seen.add(key)
        exclusion = exclusions.get(key)
        base_excluded = truthy(row.get("release_excluded"))
        excluded = base_excluded or exclusion is not None
        row["release_excluded"] = str(excluded).lower()
        row["release_required"] = str(not excluded).lower()
        row["release_exclusion_reason"] = (
            exclusion.get("exclusion_reason", "") if exclusion else row.get("release_exclusion_reason", "")
        )
        row["release_exclusion_scope"] = (
            exclusion.get("exclusion_scope", "") if exclusion else row.get("release_exclusion_scope", "")
        )
        row["release_exclusion_approved_by"] = (
            exclusion.get("approved_by", "") if exclusion else row.get("release_exclusion_approved_by", "")
        )
        row["release_exclusion_approved_at_utc"] = (
            exclusion.get("approved_at_utc", "") if exclusion else row.get("release_exclusion_approved_at_utc", "")
        )
        if excluded:
            excluded_rows.append(dict(row))

    missing = sorted(key for key in exclusions if key not in seen)
    if missing:
        formatted = ", ".join(f"{lane_id}:{proteome_id}" for lane_id, proteome_id in missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... plus {len(missing) - 10} more"
        raise SystemExit(f"Release exclusions did not match freeze manifest rows: {formatted}{suffix}")
    return rows, excluded_rows


def esm2_ids(paths: list[Path], expected_ids: set[str]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for metadata_path in [
            path / "embedding_metadata.tsv",
            path / "embedding_checkpoints/checkpoint_metadata.tsv",
        ]:
            for row in read_tsv(metadata_path):
                proteome_id = row.get("proteome_id") or row.get("sample")
                if proteome_id:
                    ids.add(str(proteome_id))
    return ids & expected_ids if expected_ids else ids


def glm2_ids(paths: list[Path], expected_ids: set[str]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for candidate in [
            path / "feature_glm_mag_level.tsv",
            path / "features/glm2_smoke_window_embedding_summary.tsv",
            path / "manifests/glm2_smoke_mag_manifest.tsv",
            path / "manifests/glm2_multiwindow_manifest.tsv",
        ]:
            for row in read_tsv(candidate):
                proteome_id = row.get("proteome_id") or row.get("unit_id")
                if proteome_id:
                    ids.add(str(proteome_id))
    return ids & expected_ids if expected_ids else ids


def functional_evidence_contract(
    repo_root: Path,
    registry_row: dict[str, str],
) -> dict[str, str]:
    """Classify functional evidence without conflating completion and comparability.

    A completed annotation run is necessary for a functional view, but it is
    not sufficient for cross-lane mechanism equivalence.  The POC warehouse is
    the only current lane whose exported ``feature_mrv_mag_level`` contract is
    explicitly MAG/bin comparable.  External per-MAG pipelines retain their
    completed raw annotation outputs while a common accepted/present mechanism
    feature rebuild is pending.
    """
    lane_id = str(registry_row.get("lane_id") or "")
    lane_role = str(registry_row.get("lane_role") or "")
    warehouse_tables: set[str] = set()
    for warehouse_dir in resolve_many(repo_root, registry_row.get("functional_warehouse_dir")):
        for row in read_tsv(warehouse_dir / "cohort_table_manifest.tsv"):
            table = row.get("table")
            if table:
                warehouse_tables.add(str(table))
    if lane_role == "calibration_core" or lane_id == "poc_core":
        return {
            "functional_evidence_class": "normalized_screening_warehouse",
            "functional_harmonization_status": (
                "version_database_and_semantic_lock_audit_pending"
                if "feature_mrv_mag_level" in warehouse_tables
                else "semantic_feature_rebuild_required"
            ),
            "mechanism_equivalence_status": "not_yet_mechanism_equivalent",
        }
    if "feature_source_dram_mag_summary" in warehouse_tables:
        return {
            "functional_evidence_class": "source_annotation_scaffold",
            "functional_harmonization_status": (
                "common_screening_axes_harmonized_with_source_scaffold_caveat"
            ),
            "mechanism_equivalence_status": "not_canonical_mechanism_equivalent",
        }
    if "feature_mrv_mag_level" in warehouse_tables:
        return {
            "functional_evidence_class": "canonical_pipeline_normalized_features",
            "functional_harmonization_status": "version_database_and_semantic_lock_audit_pending",
            "mechanism_equivalence_status": "not_yet_mechanism_equivalent",
        }
    return {
        "functional_evidence_class": "annotation_complete_feature_aggregation_pending",
        "functional_harmonization_status": (
            "raw_annotation_outputs_complete_common_mechanism_aggregation_pending"
        ),
        "mechanism_equivalence_status": "not_yet_mechanism_equivalent",
    }


def functional_status_by_id(
    per_mag_dirs: list[Path],
    expected_ids: set[str],
    runnable_ids: set[str],
    warehouse_current: bool,
) -> dict[str, dict[str, str]]:
    if warehouse_current:
        return {
            proteome_id: {
                "functional_status": "complete" if proteome_id in runnable_ids else "non_runnable_gap",
                "functional_status_basis": "warehouse_dim_mag",
                "selected_run_dir": "",
            }
            for proteome_id in expected_ids
        }

    out: dict[str, dict[str, str]] = {}
    for per_mag_dir in per_mag_dirs:
        if not per_mag_dir.exists():
            continue
        for proteome_dir in sorted(path for path in per_mag_dir.iterdir() if path.is_dir()):
            proteome_id = proteome_dir.name
            if expected_ids and proteome_id not in expected_ids:
                continue
            runs = sorted(path for path in proteome_dir.iterdir() if path.is_dir())
            if not runs:
                continue
            complete_runs = [path for path in runs if lane_registry.run_dir_status(path) == "complete"]
            if complete_runs:
                out[proteome_id] = {
                    "functional_status": "complete",
                    "functional_status_basis": "live_per_mag",
                    "selected_run_dir": str(complete_runs[-1]),
                }
                continue
            status = lane_registry.run_dir_status(runs[-1])
            previous = out.get(proteome_id, {})
            if previous.get("functional_status") == "complete":
                continue
            out[proteome_id] = {
                "functional_status": status,
                "functional_status_basis": "live_per_mag",
                "selected_run_dir": str(runs[-1]),
            }
    for proteome_id in expected_ids:
        out.setdefault(
            proteome_id,
            {
                "functional_status": "not_started" if proteome_id in runnable_ids else "non_runnable_gap",
                "functional_status_basis": "live_per_mag",
                "selected_run_dir": "",
            },
        )
    return out


def expected_lane_rows(
    source_rows: list[dict[str, str]],
    functional_rows: list[dict[str, str]],
    denominator_units: int,
) -> list[dict[str, str]]:
    candidates = [
        source_rows,
        [row for row in source_rows if truthy(row.get("mbag_mag_level_include"))],
        functional_rows,
        [row for row in functional_rows if truthy(row.get("functional_run_include"))],
    ]
    for rows in candidates:
        if rows and len(rows) == denominator_units:
            return rows
    raise SystemExit(
        "Could not reconcile registered denominator to source/functional manifests: "
        f"denominator={denominator_units}; source={len(source_rows)}; functional={len(functional_rows)}"
    )


def lane_unit_rows(repo_root: Path, registry_row: dict[str, str], status_row: dict[str, Any]) -> list[dict[str, Any]]:
    lane_id = registry_row.get("lane_id", "")
    source_rows = read_tsv(resolve(repo_root, registry_row.get("source_lane_manifest")))
    functional_rows = read_tsv(resolve(repo_root, registry_row.get("functional_manifest")))
    denominator_units = int(float(registry_row.get("denominator_units") or 0))
    expected_rows = expected_lane_rows(source_rows, functional_rows, denominator_units)
    expected_ids = {row.get("proteome_id", "") for row in expected_rows if row.get("proteome_id")}
    source_by_id = {row.get("proteome_id", ""): row for row in source_rows if row.get("proteome_id")}
    functional_by_id = {row.get("proteome_id", ""): row for row in functional_rows if row.get("proteome_id")}
    runnable_ids = {
        proteome_id
        for proteome_id, row in functional_by_id.items()
        if truthy(row.get("functional_run_include"))
    }
    if not runnable_ids:
        runnable_ids = set(expected_ids)

    esm_ids = esm2_ids(resolve_many(repo_root, registry_row.get("esm2_artifacts_dirs")), expected_ids)
    glm_ids = glm2_ids(resolve_many(repo_root, registry_row.get("glm2_artifacts_dirs")), expected_ids)
    warehouse_current = str(status_row.get("warehouse_current") or "").lower() == "true"
    functional = functional_status_by_id(
        resolve_many(repo_root, registry_row.get("functional_per_mag_dirs")),
        expected_ids,
        runnable_ids,
        warehouse_current,
    )
    evidence_contract = functional_evidence_contract(repo_root, registry_row)

    rows: list[dict[str, Any]] = []
    for row in expected_rows:
        proteome_id = row.get("proteome_id", "")
        source_row = source_by_id.get(proteome_id, {})
        functional_row = functional_by_id.get(proteome_id, {})
        merged_row = {**source_row, **row, **functional_row}
        func = functional.get(proteome_id, {})
        has_esm2 = proteome_id in esm_ids
        has_glm2 = proteome_id in glm_ids
        has_functional = func.get("functional_status") == "complete"
        tri_view_ready = has_esm2 and has_glm2 and has_functional
        row_evidence_contract = dict(evidence_contract)
        if not has_functional:
            row_evidence_contract["functional_harmonization_status"] = (
                "functional_output_incomplete"
            )
            row_evidence_contract["mechanism_equivalence_status"] = (
                "not_applicable_functional_incomplete"
            )
        mechanism_equivalent = (
            row_evidence_contract["mechanism_equivalence_status"]
            == "mechanism_equivalent"
        )
        runnable = proteome_id in runnable_ids
        metadata_resolution = str(
            merged_row.get("metadata_mapping_status") or "unresolved"
        )
        sample_linked = metadata_resolution == "exact_sample_mag_link"
        if lane_id == "mucc_v1_owc_wetland":
            glm2_protocol_class = "multiwindow_10_native_plus_10_shuffled"
        else:
            glm2_protocol_class = "paired_single_native_plus_single_shuffled"
        if not tri_view_ready:
            formal_tri_view_status = "incomplete_tri_view"
        elif mechanism_equivalent:
            formal_tri_view_status = "complete_canonical_mechanism_tri_view"
        elif (
            row_evidence_contract["functional_evidence_class"]
            == "annotation_complete_feature_aggregation_pending"
        ):
            formal_tri_view_status = (
                "complete_annotation_tri_view_harmonization_pending"
            )
        elif row_evidence_contract["functional_evidence_class"] in {
            "normalized_screening_warehouse",
            "canonical_pipeline_normalized_features",
        }:
            formal_tri_view_status = (
                "complete_pipeline_normalized_tri_view_comparability_pending"
            )
        else:
            formal_tri_view_status = "complete_source_scaffold_tri_view"
        rows.append(
            {
                "lane_id": lane_id,
                "lane_role": registry_row.get("lane_role", ""),
                "proteome_id": proteome_id,
                "mag_id": merged_row.get("mag_id") or merged_row.get("mag_id_candidate") or proteome_id,
                "source": merged_row.get("source") or lane_id,
                "ecosystem": merged_row.get("ecosystem") or "",
                "domain": merged_row.get("domain") or "",
                "has_esm2": str(has_esm2).lower(),
                "has_glm2": str(has_glm2).lower(),
                "has_functional": str(has_functional).lower(),
                "tri_view_ready": str(tri_view_ready).lower(),
                **row_evidence_contract,
                "formal_tri_view_status": formal_tri_view_status,
                "mechanism_equivalent_tri_view": str(
                    tri_view_ready and mechanism_equivalent
                ).lower(),
                "functional_status": func.get("functional_status", "not_started"),
                "functional_status_basis": func.get("functional_status_basis", ""),
                "selected_run_dir": func.get("selected_run_dir", ""),
                "glm2_protocol_class": glm2_protocol_class,
                "qc_taxonomy_available": str(has_functional).lower(),
                "metadata_resolution": metadata_resolution,
                "sample_linkage_readiness": "exact" if sample_linked else "ambiguous_or_unresolved",
                "payload_complete": str(tri_view_ready).lower(),
                "schema_normalized": str(
                    has_functional
                    and row_evidence_contract["functional_evidence_class"]
                    in {"normalized_screening_warehouse", "canonical_pipeline_normalized_features", "source_annotation_scaffold"}
                ).lower(),
                "mechanism_comparable": str(tri_view_ready and mechanism_equivalent).lower(),
                "sample_linked": str(sample_linked).lower(),
                "abundance_weighted": "false",
                "environmentally_contextualized": "false",
                "field_validated": "false",
                "calibrated": "false",
                "release_excluded": str(not runnable).lower(),
                "release_exclusion_reason": (
                    merged_row.get("gap_reason")
                    or merged_row.get("match_status")
                    or "registered source unit lacks a runnable tri-view payload"
                ) if not runnable else "",
                "release_exclusion_scope": "payload_release_only" if not runnable else "",
                "release_exclusion_approved_by": "source_manifest_contract" if not runnable else "",
                "release_exclusion_approved_at_utc": "manifest-recorded" if not runnable else "",
                "allowed_claim_wording": "MAG/proteome molecular screening evidence; candidate ranks are review priorities only",
                "blocking_gap": (
                    "runnable payload unavailable" if not runnable else
                    "common semantic/version/database lock and sample/field validation remain pending"
                ),
                "next_validation_action": merged_row.get("recommended_action") or (
                    "recover the source payload" if not runnable else
                    "rebuild semantic features, resolve sample abundance/context, and validate against process evidence"
                ),
                "claim_scope": registry_row.get("claim_scope", ""),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], registry_status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status_by_lane = {row.get("lane_id", ""): row for row in registry_status_rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["lane_id"]), []).append(row)
    out: list[dict[str, Any]] = []
    for lane_id, lane_rows in grouped.items():
        counts = Counter(row["functional_status"] for row in lane_rows)
        tri_view = sum(1 for row in lane_rows if truthy(row.get("tri_view_ready")))
        canonical_tri_view = sum(
            1 for row in lane_rows if truthy(row.get("mechanism_equivalent_tri_view"))
        )
        source_scaffold_tri_view = sum(
            1
            for row in lane_rows
            if row.get("formal_tri_view_status") == "complete_source_scaffold_tri_view"
        )
        annotation_complete_tri_view = sum(
            1
            for row in lane_rows
            if row.get("formal_tri_view_status")
            == "complete_annotation_tri_view_harmonization_pending"
        )
        pipeline_normalized_tri_view = sum(
            1
            for row in lane_rows
            if row.get("formal_tri_view_status")
            == "complete_pipeline_normalized_tri_view_comparability_pending"
        )
        schema_normalized_tri_view = sum(
            1
            for row in lane_rows
            if truthy(row.get("tri_view_ready")) and truthy(row.get("schema_normalized"))
        )
        release_required = sum(1 for row in lane_rows if truthy(row.get("release_required")))
        release_excluded = sum(1 for row in lane_rows if truthy(row.get("release_excluded")))
        release_tri_view = sum(
            1
            for row in lane_rows
            if truthy(row.get("release_required")) and truthy(row.get("tri_view_ready"))
        )
        status = status_by_lane.get(lane_id, {})
        out.append(
            {
                "lane_id": lane_id,
                "expected_units": len(lane_rows),
                "release_required_units": release_required,
                "release_excluded_units": release_excluded,
                "esm2_units": sum(1 for row in lane_rows if truthy(row.get("has_esm2"))),
                "glm2_units": sum(1 for row in lane_rows if truthy(row.get("has_glm2"))),
                "functional_payload_units": sum(
                    1 for row in lane_rows if truthy(row.get("has_functional"))
                ),
                "functional_complete": counts.get("complete", 0),
                "functional_partial": counts.get("partial", 0),
                "functional_failed": counts.get("failed", 0),
                "functional_not_started": counts.get("not_started", 0),
                "functional_non_runnable_gap": counts.get("non_runnable_gap", 0),
                "tri_view_ready_units": tri_view,
                "canonical_mechanism_tri_view_units": canonical_tri_view,
                "pipeline_normalized_tri_view_units": pipeline_normalized_tri_view,
                "schema_normalized_units": sum(
                    1 for row in lane_rows if truthy(row.get("schema_normalized"))
                ),
                "schema_normalized_tri_view_units": schema_normalized_tri_view,
                "annotation_complete_tri_view_units": annotation_complete_tri_view,
                "source_scaffold_tri_view_units": source_scaffold_tri_view,
                "release_tri_view_ready_units": release_tri_view,
                "registry_denominator_units": status.get("denominator_units", ""),
                "registry_manifest_gap_rows": status.get("manifest_gap_rows", ""),
                "warehouse_current": status.get("warehouse_current", ""),
            }
        )
    return out


def markdown(
    stamp: str,
    freeze_allowed: bool,
    required_lanes: list[str],
    summary_rows: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# MethaNet 3-View Payload Freeze Decision",
        "",
        f"Generated UTC: `{stamp}`",
        f"Freeze allowed: `{'true' if freeze_allowed else 'false'}`",
        f"Required green lanes: `{', '.join(required_lanes)}`",
        "",
        "| Lane | Expected | Release-required | Excluded | ESM2 | gLM2 | Functional | Tri-view | Schema-normalized tri-view | Pipeline-normalized, comparability pending | Mechanism-comparable | Annotation complete, aggregation pending | Source scaffold | Release tri-view | Partial | Failed | Not started |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {lane_id} | {expected_units:,} | {release_required_units:,} | "
            "{release_excluded_units:,} | {esm2_units:,} | {glm2_units:,} | "
            "{functional_complete:,} | {tri_view_ready_units:,} | "
            "{schema_normalized_tri_view_units:,} | "
            "{pipeline_normalized_tri_view_units:,} | "
            "{canonical_mechanism_tri_view_units:,} | "
            "{annotation_complete_tri_view_units:,} | "
            "{source_scaffold_tri_view_units:,} | "
            "{release_tri_view_ready_units:,} | {functional_partial:,} | "
            "{functional_failed:,} | {functional_not_started:,} |".format(
                **{key: int(value) if str(value).isdigit() else value for key, value in row.items()}
            )
        )
    lines.extend(["", "## Decision", ""])
    if freeze_allowed:
        if excluded_rows:
            lines.append(
                "All non-excluded required-lane units are green for this release freeze. "
                "Release exclusions are preserved below and in `excluded_units.tsv`."
            )
        else:
            lines.append("All required lanes are green for their registered functional denominators.")
    else:
        lines.append("At least one required lane is not green. Do not use this snapshot as the final report freeze.")
    if blockers:
        lines.extend(["", "## Blocking Units", ""])
        for row in blockers[:25]:
            lines.append(
                f"- `{row['lane_id']}` `{row['proteome_id']}`: "
                f"ESM2={row['has_esm2']}, gLM2={row['has_glm2']}, "
                f"functional={row['functional_status']}"
            )
        if len(blockers) > 25:
            lines.append(f"- ... plus `{len(blockers) - 25}` additional blockers.")
    if excluded_rows:
        lines.extend(["", "## Release Exclusions", ""])
        for row in excluded_rows[:25]:
            lines.append(
                f"- `{row['lane_id']}` `{row['proteome_id']}`: "
                f"{row.get('release_exclusion_reason', 'release-approved exclusion')}"
            )
        if len(excluded_rows) > 25:
            lines.append(f"- ... plus `{len(excluded_rows) - 25}` additional exclusions.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This freeze operates at MAG/proteome grain. It supports molecular "
            "atlas/report rebuild decisions, not final sample-level MRV risk tiers, "
            "measured methane flux, or carbon-credit approval.",
            "",
        ]
    )
    return "\n".join(lines)


def evidence_contract_markdown(
    stamp: str,
    summary_rows: list[dict[str, Any]],
    evidence_contract_rows: list[dict[str, Any]],
) -> str:
    mechanism_total = sum(
        int(row.get("canonical_mechanism_tri_view_units") or 0) for row in summary_rows
    )
    lines = [
        "# MethaNet Evidence-Contract Harmonization",
        "",
        f"Generated UTC: `{stamp}`",
        "",
        "This report distinguishes a validated common table/event schema from a "
        "validated common biological-mechanism contract. Pipeline normalization is "
        "necessary but is not sufficient for cross-lane mechanism equivalence.",
        "",
        "| Lane | Tri-view | Schema-normalized tri-view | Pipeline-normalized, comparability pending | Mechanism-comparable | Source scaffold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {lane_id} | {tri_view_ready_units:,} | "
            "{schema_normalized_tri_view_units:,} | "
            "{pipeline_normalized_tri_view_units:,} | "
            "{canonical_mechanism_tri_view_units:,} | "
            "{source_scaffold_tri_view_units:,} |".format(
                **{
                    key: int(value) if str(value).isdigit() else value
                    for key, value in row.items()
                }
            )
        )
    lines.extend(
        [
            "",
            "## Cross-lane audit",
            "",
            "| Contract element | State | Release interpretation |",
            "| --- | --- | --- |",
            "| Accepted/present event semantics | implemented | KOfam accepted calls, best-ranked MCycDB/SCycDB hits, and METABOLIC presence events remain explicit; raw hit rows are not mechanism numerators. |",
            "| Long-form analytical schema | implemented | Tool-native wide workbooks are normalized and warehouse keys are validated. |",
            "| Deterministic attempt selection | implemented | Failed, partial, corrupt, and superseded attempts remain in the run-status ledger. |",
            "| Code commit and locked configuration equivalence | pending | Do not call POC, MSM, and Futian one mechanism-comparable cohort. |",
            "| Database versions and checksums | pending | Database drift cannot yet be excluded across historical runs. |",
            "| Gene calling, thresholds, ranking, zero/unknown, QC and missingness equivalence | pending cross-lane fingerprint audit | Schema-normalized values remain partitioned by evidence contract. |",
            "| Source-aware nulls, taxonomy controls, bootstrap stability and ablations | pending | Candidate ranks remain review priorities, not biological truth or calibrated risk. |",
            "| MUCC source-scaffold crosswalk | source-specific | Mapped source-scaffold evidence is not numerically pooled with canonical pipeline features. |",
            "",
            "## Decision",
            "",
            f"Mechanism-comparable tri-view units authorized by this audit: **{mechanism_total:,}**.",
            "",
            "Allowed now: payload completeness, schema-normalized evidence review, "
            "source-partitioned molecular screening, and next-measurement design. "
            "Blocked: cross-lane pathway-strength comparisons, sample risk, activity "
            "or flux inference, source-independent transfer, final tiers, and crediting conclusions.",
            "",
            "The machine-readable contract ledger is `entities/fact_evidence_contract.tsv`.",
            "",
            f"Contract rows audited: `{len(evidence_contract_rows):,}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or repo_root / "results/reports" / f"methanet_3view_payload_freeze_{stamp}"
    output_dir = resolve(repo_root, str(output_dir)) or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = resolve(repo_root, str(args.lane_registry)) or args.lane_registry
    registry_rows = read_tsv(registry_path)
    registry_status_rows = [lane_registry.summarize_lane(repo_root, row) for row in registry_rows]
    status_by_lane = {row.get("lane_id", ""): row for row in registry_status_rows}

    unit_rows: list[dict[str, Any]] = []
    for row in registry_rows:
        unit_rows.extend(lane_unit_rows(repo_root, row, status_by_lane.get(row.get("lane_id", ""), {})))

    exclusions = read_release_exclusions(repo_root, args.exclusion_tsv, args.exclude_unit)
    unit_rows, excluded_rows = apply_release_exclusions(unit_rows, exclusions)
    summary_rows = summarize(unit_rows, registry_status_rows)
    required_lanes = list(dict.fromkeys(args.require_green_lane or [row.get("lane_id", "") for row in registry_rows]))
    blockers = [
        row
        for row in unit_rows
        if (
            row.get("lane_id") in required_lanes
            and truthy(row.get("release_required"))
            and not truthy(row.get("tri_view_ready"))
        )
    ]
    freeze_allowed = not blockers

    write_tsv(
        output_dir / "freeze_manifest.tsv",
        unit_rows,
        [
            "lane_id",
            "lane_role",
            "proteome_id",
            "mag_id",
            "source",
            "ecosystem",
            "domain",
            "has_esm2",
            "has_glm2",
            "has_functional",
            "tri_view_ready",
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
            "glm2_protocol_class",
            "qc_taxonomy_available",
            "metadata_resolution",
            "sample_linkage_readiness",
            "payload_complete",
            "schema_normalized",
            "mechanism_comparable",
            "sample_linked",
            "abundance_weighted",
            "environmentally_contextualized",
            "field_validated",
            "calibrated",
            "allowed_claim_wording",
            "blocking_gap",
            "next_validation_action",
            "claim_scope",
        ],
    )
    write_tsv(output_dir / "freeze_summary.tsv", summary_rows, list(summary_rows[0]) if summary_rows else [])
    write_tsv(output_dir / "blocking_units.tsv", blockers, list(unit_rows[0]) if unit_rows else [])
    write_tsv(output_dir / "excluded_units.tsv", excluded_rows, list(unit_rows[0]) if unit_rows else [])
    release_entities = build_release_entities(repo_root, registry_rows, unit_rows)
    entity_dir = output_dir / "entities"
    for entity_name, entity_rows in release_entities.items():
        write_tsv(entity_dir / f"{entity_name}.tsv", entity_rows, all_fields(entity_rows))
    freeze_manifest_path = output_dir / "freeze_manifest.tsv"
    freeze_sha256 = hashlib.sha256(freeze_manifest_path.read_bytes()).hexdigest()
    public_release_ledger = {
        "schema_version": "1.0.0",
        "snapshot_date": datetime.now(timezone.utc).date().isoformat(),
        "freeze_manifest_sha256": freeze_sha256,
        "release_state": "ready" if freeze_allowed else "blocked",
        "indexing_decision": "noindex_controlled_diligence",
        "registered_units": len(unit_rows),
        "esm2_units": sum(1 for row in unit_rows if truthy(row.get("has_esm2"))),
        "glm2_units": sum(1 for row in unit_rows if truthy(row.get("has_glm2"))),
        "functional_payload_units": sum(
            1 for row in unit_rows if truthy(row.get("has_functional"))
        ),
        "release_required_units": sum(1 for row in unit_rows if truthy(row.get("release_required"))),
        "explicit_non_runnable_gaps": len(excluded_rows),
        "tri_view_ready_units": sum(1 for row in unit_rows if truthy(row.get("tri_view_ready"))),
        "schema_normalized_units": sum(
            1 for row in unit_rows if truthy(row.get("schema_normalized"))
        ),
        "schema_normalized_tri_view_units": sum(
            1
            for row in unit_rows
            if truthy(row.get("tri_view_ready")) and truthy(row.get("schema_normalized"))
        ),
        "pipeline_normalized_tri_view_units": sum(
            1
            for row in unit_rows
            if row.get("formal_tri_view_status")
            == "complete_pipeline_normalized_tri_view_comparability_pending"
        ),
        "mechanism_comparable_units": sum(1 for row in unit_rows if truthy(row.get("mechanism_comparable"))),
        "annotation_complete_tri_view_units": sum(
            1
            for row in unit_rows
            if row.get("formal_tri_view_status")
            == "complete_annotation_tri_view_harmonization_pending"
        ),
        "source_scaffold_tri_view_units": sum(
            1
            for row in unit_rows
            if row.get("formal_tri_view_status") == "complete_source_scaffold_tri_view"
        ),
        "sample_linked_units": sum(1 for row in unit_rows if truthy(row.get("sample_linked"))),
        "abundance_weighted_units": sum(1 for row in unit_rows if truthy(row.get("abundance_weighted"))),
        "environmentally_contextualized_units": sum(1 for row in unit_rows if truthy(row.get("environmentally_contextualized"))),
        "field_validated_units": sum(1 for row in unit_rows if truthy(row.get("field_validated"))),
        "calibrated_units": sum(1 for row in unit_rows if truthy(row.get("calibrated"))),
        "blocking_units": len(blockers),
        "lanes": summary_rows,
        "allowed_public_wording": "Molecular screening evidence and review priorities; metadata-rich contexts are not scored samples.",
        "forbidden_public_wording": "Sample risk, measured flux, activity magnitude, final tiers, source-independent transfer, or MRV/crediting approval.",
    }
    (output_dir / "release_ledger.json").write_text(json.dumps(public_release_ledger, indent=2, sort_keys=True) + "\n")
    decision = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_allowed": freeze_allowed,
        "required_green_lanes": required_lanes,
        "unit_rows": len(unit_rows),
        "esm2_units": sum(1 for row in unit_rows if truthy(row.get("has_esm2"))),
        "glm2_units": sum(1 for row in unit_rows if truthy(row.get("has_glm2"))),
        "functional_payload_units": sum(
            1 for row in unit_rows if truthy(row.get("has_functional"))
        ),
        "tri_view_ready_units": sum(1 for row in unit_rows if truthy(row.get("tri_view_ready"))),
        "schema_normalized_units": sum(
            1 for row in unit_rows if truthy(row.get("schema_normalized"))
        ),
        "schema_normalized_tri_view_units": sum(
            1
            for row in unit_rows
            if truthy(row.get("tri_view_ready")) and truthy(row.get("schema_normalized"))
        ),
        "pipeline_normalized_tri_view_units": sum(
            1
            for row in unit_rows
            if row.get("formal_tri_view_status")
            == "complete_pipeline_normalized_tri_view_comparability_pending"
        ),
        "canonical_mechanism_tri_view_units": sum(
            1 for row in unit_rows if truthy(row.get("mechanism_equivalent_tri_view"))
        ),
        "annotation_complete_tri_view_units": sum(
            1
            for row in unit_rows
            if row.get("formal_tri_view_status")
            == "complete_annotation_tri_view_harmonization_pending"
        ),
        "source_scaffold_tri_view_units": sum(
            1
            for row in unit_rows
            if row.get("formal_tri_view_status") == "complete_source_scaffold_tri_view"
        ),
        "release_required_units": sum(1 for row in unit_rows if truthy(row.get("release_required"))),
        "release_tri_view_ready_units": sum(
            1
            for row in unit_rows
            if truthy(row.get("release_required")) and truthy(row.get("tri_view_ready"))
        ),
        "release_excluded_units": len(excluded_rows),
        "blocking_units": len(blockers),
        "outputs": {
            "freeze_manifest": str(output_dir / "freeze_manifest.tsv"),
            "freeze_summary": str(output_dir / "freeze_summary.tsv"),
            "blocking_units": str(output_dir / "blocking_units.tsv"),
            "excluded_units": str(output_dir / "excluded_units.tsv"),
            "decision_md": str(output_dir / "FREEZE_DECISION.md"),
            "release_entities": str(entity_dir),
            "release_ledger": str(output_dir / "release_ledger.json"),
        },
        "release_entity_rows": {name: len(rows) for name, rows in release_entities.items()},
    }
    (output_dir / "freeze_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    (output_dir / "registry_status_snapshot.json").write_text(json.dumps(registry_status_rows, indent=2, sort_keys=True))
    (output_dir / "FREEZE_DECISION.md").write_text(
        markdown(decision["generated_utc"], freeze_allowed, required_lanes, summary_rows, blockers, excluded_rows)
    )
    harmonization_path = output_dir / "EVIDENCE_CONTRACT_HARMONIZATION.md"
    harmonization_path.write_text(
        evidence_contract_markdown(
            decision["generated_utc"],
            summary_rows,
            release_entities["fact_evidence_contract"],
        )
    )
    decision["outputs"]["evidence_contract_harmonization"] = str(harmonization_path)
    (output_dir / "freeze_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    print(json.dumps(decision, indent=2, sort_keys=True))
    if args.exit_nonzero_if_blocked and not freeze_allowed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
