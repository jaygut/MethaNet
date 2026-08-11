#!/usr/bin/env python3
"""Build a conservative cross-lane metadata and validation-readiness package."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
NA_VALUES = {"", "na", "n/a", "nan", "none", "not collected", "unknown"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).read().to_pylist()


def parquet_path(root: Path, table: str) -> Path:
    matches = sorted((root / "parquet" / table).rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"Missing required warehouse table: {table}")
    return matches[0]


def present(value: Any) -> bool:
    return str(value or "").strip().lower() not in NA_VALUES


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "present"}


def split_ids(value: Any) -> list[str]:
    text = str(value or "").replace(";", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def write_table(output_dir: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    tables_dir = output_dir / "tables"
    parquet_dir = output_dir / "parquet"
    tables_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    tsv_path = tables_dir / f"{name}.tsv"
    parquet_file = parquet_dir / f"{name}.parquet"
    frame.to_csv(tsv_path, sep="\t", index=False)
    frame.to_parquet(parquet_file, index=False, compression="zstd")
    return {"table": name, "rows": len(frame), "columns": len(frame.columns), "tsv": str(tsv_path), "parquet": str(parquet_file)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    msm_samples = read_tsv(repo_root / "data/external/msm_china_2025/gigadb_wasabi/metadata_sediment_samples.txt")
    msm_biosamples = read_tsv(repo_root / "data/external/msm_china_2025/metadata/ncbi_biosample_environmental_metadata.tsv")
    msm_manifest = read_tsv(repo_root / "results/functional_metagenomics/msm_china_2025_20260615/manifests/msm_china_2025_functional_mag_manifest.tsv")
    futian_samples = read_tsv(repo_root / "data/external/futian_mangrove_2026_qi/metadata/futian_65_sample_metadata.tsv")
    futian_manifest = read_tsv(repo_root / "data/external/futian_mangrove_2026_qi/manifests/futian_phase1_source_lane_manifest.tsv")
    mucc_root = repo_root / "results/functional_metagenomics/mucc_v1_owc_wetland_20260626/cohort_warehouse"
    mucc_ecology = read_parquet(parquet_path(mucc_root, "feature_mucc_v1_sample_ecological_readiness"))
    mucc_expression = read_parquet(parquet_path(mucc_root, "fact_mag_expression_sample"))
    mucc_chamber = read_parquet(parquet_path(mucc_root, "fact_mucc_v1_essdive_chamber_flux"))
    mucc_porewater = read_parquet(parquet_path(mucc_root, "fact_mucc_v1_essdive_porewater_ch4"))
    mucc_tower = read_parquet(parquet_path(mucc_root, "fact_mucc_v1_essdive_gapfilled_tower_ch4_flux"))

    dim_source_dataset = [
        {"source_dataset_id": "poc_core", "lane_id": "poc_core", "dataset_role": "molecular_reference", "claim_scope": "MAG/proteome identity and molecular screening only"},
        {"source_dataset_id": "msm_china_2025", "lane_id": "msm_china_2025", "dataset_role": "external_mangrove_generalization", "claim_scope": "group-level sample context; published 966 denominator unresolved"},
        {"source_dataset_id": "futian_mangrove_2026_qi", "lane_id": "futian_mangrove_2026_qi", "dataset_role": "environmental_permissiveness_design", "claim_scope": "site-month/depth-set context; no unique MAG-depth assignment"},
        {"source_dataset_id": "mucc_v1_owc_wetland", "lane_id": "mucc_v1_owc_wetland", "dataset_role": "paired_validation_recovery", "claim_scope": "expression detection/occupancy and unlinked flux evidence"},
    ]

    dim_site: list[dict[str, Any]] = []
    seen_sites: set[tuple[str, str]] = set()
    for row in msm_samples:
        key = ("msm_china_2025", row.get("sample_loc", ""))
        if key not in seen_sites:
            seen_sites.add(key)
            dim_site.append({"lane_id": key[0], "site_id": key[1], "site_name": key[1], "latitude": row.get("Latitude", ""), "longitude": row.get("Longitude", ""), "resolution_tier": "exact_sample_context"})
    for row in futian_samples:
        key = ("futian_mangrove_2026_qi", row.get("sample_site", ""))
        if key not in seen_sites:
            seen_sites.add(key)
            dim_site.append({"lane_id": key[0], "site_id": key[1], "site_name": row.get("site_name", ""), "latitude_longitude": row.get("latitude_longitude", ""), "resolution_tier": "exact_site"})
    dim_site.append({"lane_id": "mucc_v1_owc_wetland", "site_id": "US-OWC", "site_name": "Old Woman Creek", "resolution_tier": "site_context"})

    dim_sample: list[dict[str, Any]] = []
    for row in msm_samples:
        dim_sample.append({"lane_id": "msm_china_2025", "sample_id": row.get("sample_id", ""), "site_id": row.get("sample_loc", ""), "collection_date": row.get("collect_date", ""), "depth": row.get("depth", ""), "resolution_tier": "exact_sample_context", "claim_scope": "metadata context; MAG assignment remains group ambiguous"})
    for row in futian_samples:
        dim_sample.append({"lane_id": "futian_mangrove_2026_qi", "sample_id": row.get("sample_name", ""), "site_id": row.get("sample_site", ""), "collection_date": row.get("sampling_month_iso", ""), "depth": row.get("depth_cm", ""), "resolution_tier": "exact_sample_context", "claim_scope": row.get("claim_scope", "")})
    for row in mucc_ecology:
        dim_sample.append({"lane_id": "mucc_v1_owc_wetland", "sample_id": row.get("sample_id", ""), "site_id": row.get("site_or_landcover", ""), "collection_date": row.get("sra_collection_date", ""), "depth": row.get("nominal_depth_interval_cm", ""), "resolution_tier": "exact_expression_sample_link", "claim_scope": row.get("claim_boundary", "")})

    link_sample_mag: list[dict[str, Any]] = []
    msm_sample_ids = {row.get("sample_id", "") for row in msm_samples}
    for row in msm_manifest:
        candidates = split_ids(row.get("source_sample_ids"))
        for sample_id in candidates:
            link_sample_mag.append({
                "lane_id": "msm_china_2025", "sample_id": sample_id, "proteome_id": row.get("proteome_id", ""),
                "mapping_method": "source_group_candidate_set", "mapping_confidence": "ambiguous",
                "resolution_tier": "exact_sample_context_mag_group_ambiguous", "provenance_pointer": "MSM functional manifest source_sample_ids",
                "ambiguity_set_size": len(candidates), "claim_scope": "not a unique MAG-to-sample assignment",
                "sample_present_in_local_metadata": str(sample_id in msm_sample_ids).lower(),
            })
    futian_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in futian_samples:
        futian_by_key[row.get("site_time_key", "")].append(row)
    for row in futian_manifest:
        if not truthy(row.get("functional_run_include")):
            continue
        candidates = futian_by_key.get(row.get("site_time_key", ""), [])
        for sample in candidates:
            link_sample_mag.append({
                "lane_id": "futian_mangrove_2026_qi", "sample_id": sample.get("sample_name", ""), "proteome_id": row.get("proteome_id", ""),
                "mapping_method": "exact_site_month_to_depth_set", "mapping_confidence": "ambiguous_depth",
                "resolution_tier": "exact_site_time_context_depth_ambiguous", "provenance_pointer": "Futian source lane plus 65-sample metadata",
                "ambiguity_set_size": len(candidates), "claim_scope": "site-month context only; no depth-specific MAG attribution",
            })
    for row in mucc_expression:
        link_sample_mag.append({
            "lane_id": "mucc_v1_owc_wetland", "sample_id": row.get("sample_id", ""), "proteome_id": row.get("proteome_id", ""),
            "mapping_method": "source_processed_expression_matrix", "mapping_confidence": "exact_matrix_cell",
            "resolution_tier": "exact_expression_sample_link", "provenance_pointer": "MUCC fact_mag_expression_sample",
            "ambiguity_set_size": 1, "claim_scope": "processed expression detection/occupancy; units pending normalization confirmation",
        })

    fact_sample_environment: list[dict[str, Any]] = []
    for row in msm_samples:
        fact_sample_environment.append({"lane_id": "msm_china_2025", "sample_id": row.get("sample_id", ""), "latitude": row.get("Latitude", ""), "longitude": row.get("Longitude", ""), "depth": row.get("depth", ""), "collection_date": row.get("collect_date", ""), "location": row.get("sample_loc", ""), "mangrove_type": row.get("mangrove type", ""), "resolution_tier": "exact_sample_context"})
    for row in futian_samples:
        fact_sample_environment.append({"lane_id": "futian_mangrove_2026_qi", "sample_id": row.get("sample_name", ""), **{key: row.get(key, "") for key in ["latitude_longitude", "depth_cm", "sampling_month_iso", "ph", "salinity_psu", "toc_mg_g", "ammonium_mg_kg", "nitrate_mg_kg", "tn_mg_g", "tp_mg_g", "ts_mg_g"]}, "resolution_tier": "exact_sample_metadata"})
    for row in mucc_ecology:
        fact_sample_environment.append({"lane_id": "mucc_v1_owc_wetland", "sample_id": row.get("sample_id", ""), "collection_date": row.get("sra_collection_date", ""), "depth": row.get("nominal_depth_interval_cm", ""), "environmental_covariates_status": row.get("environmental_covariates_status", ""), "resolution_tier": row.get("metadata_resolution_tier_y", ""), "claim_scope": "methods context is not an exact environment/flux join"})

    fact_sample_expression_or_abundance = [{
        "lane_id": "mucc_v1_owc_wetland", "sample_id": row.get("sample_id", ""), "proteome_id": row.get("proteome_id", ""),
        "value": row.get("expression_value", ""), "units": row.get("expression_units", ""), "detected": row.get("expression_detected", ""),
        "evidence_type": "processed_expression_detection_or_occupancy", "claim_scope": "not activity magnitude or process rate",
    } for row in mucc_expression]

    fact_flux_or_process_observation: list[dict[str, Any]] = []
    for observation_type, rows, id_field, value_field in [
        ("chamber_methane_flux", mucc_chamber, "flux_observation_id", "methane_flux_nmol_m2_s"),
        ("porewater_methane", mucc_porewater, "porewater_observation_id", "porewater_ch4_mM"),
        ("gapfilled_tower_ch4_flux", mucc_tower, "flux_observation_id", "ch4_flux"),
    ]:
        for row in rows:
            fact_flux_or_process_observation.append({
                "lane_id": "mucc_v1_owc_wetland", "observation_id": row.get(id_field, ""), "observation_type": observation_type,
                "value": row.get(value_field, ""), "source_date": row.get("source_date", row.get("timestamp", "")),
                "site_code": row.get("site_code", "US-OWC"), "sample_join_status": row.get("sample_join_status", "unlinked_no_authoritative_sequence_sample_crosswalk"),
                "claim_scope": "source-staged process evidence; not attributed to a sequencing sample or MAG",
            })

    link_sample_flux_window = [{
        "lane_id": "mucc_v1_owc_wetland", "sample_id": row.get("sample_id", ""), "flux_observation_id": "",
        "mapping_method": "no_mapping", "mapping_confidence": "unresolved", "resolution_tier": "unresolved",
        "provenance_pointer": "MUCC ecological readiness and ESS-DIVE staged facts", "ambiguity_set_size": "unknown",
        "claim_scope": "no authoritative sample/date/depth/environment/flux join",
    } for row in mucc_ecology]

    coverage_specs = {
        "msm_china_2025": (msm_samples, ["sample_id", "Latitude", "Longitude", "depth", "collect_date", "sample_loc", "env_package", "mangrove type"]),
        "futian_mangrove_2026_qi": (futian_samples, ["sample_name", "latitude_longitude", "depth_cm", "sampling_month_iso", "ph", "salinity_psu", "toc_mg_g", "ammonium_mg_kg", "nitrate_mg_kg", "tn_mg_g", "tp_mg_g", "ts_mg_g"]),
        "mucc_v1_owc_wetland": (mucc_ecology, ["sample_id", "sra_collection_date", "nominal_depth_interval_cm", "jgi_sample_identity_status", "environment_measurement_join_status", "flux_window_join_status"]),
    }
    feature_metadata_field_coverage: list[dict[str, Any]] = []
    for lane_id, (rows, fields) in coverage_specs.items():
        for field in fields:
            count = sum(1 for row in rows if present(row.get(field)))
            feature_metadata_field_coverage.append({"lane_id": lane_id, "field": field, "present_rows": count, "denominator_rows": len(rows), "coverage_fraction": count / len(rows) if rows else 0})

    feature_sample_risk_readiness: list[dict[str, Any]] = []
    for row in msm_samples:
        feature_sample_risk_readiness.append({"lane_id": "msm_china_2025", "sample_id": row.get("sample_id", ""), "readiness_label": "metadata_rich_context_mag_group_ambiguous", "exact_mag_mapping": "false", "abundance_available": "false", "flux_join": "false", "next_action": "reconcile sample IDs, published denominator, and read/bin mapping"})
    for row in futian_samples:
        feature_sample_risk_readiness.append({"lane_id": "futian_mangrove_2026_qi", "sample_id": row.get("sample_name", ""), "readiness_label": "metadata_rich_site_month_depth_context", "exact_mag_mapping": "false", "abundance_available": "false", "flux_join": "false", "next_action": "resolve MAG-depth abundance and obtain independent process evidence"})
    for row in mucc_ecology:
        feature_sample_risk_readiness.append({"lane_id": "mucc_v1_owc_wetland", "sample_id": row.get("sample_id", ""), "readiness_label": row.get("readiness_label", ""), "exact_mag_mapping": "true", "abundance_available": "expression_detection_only", "flux_join": "false", "next_action": row.get("next_validation_action", "")})

    exact_mucc = [row for row in mucc_ecology if "exact_source_label" in str(row.get("sra_sample_identity_status")) and "exact_collection_date" in str(row.get("sra_collection_date_status")) and str(row.get("methods_depth_assignment_status", "")).startswith("methods_design_direct")]
    paired_futian_months = {"202101", "202105", "202108", "202111", "202303"}
    full_futian = [
        row for row in futian_samples
        if row.get("sample_site") in {"MF1", "MG1"}
        and row.get("sampling_time_yyyymm") in paired_futian_months
        and all(present(row.get(field)) for field in ["ph", "salinity_psu", "toc_mg_g", "ammonium_mg_kg", "nitrate_mg_kg", "tn_mg_g", "tp_mg_g", "ts_mg_g"])
    ]
    sample_validation_candidate_portfolio: list[dict[str, Any]] = []
    for row in exact_mucc:
        sample_validation_candidate_portfolio.append({"priority": 1, "lane_id": "mucc_v1_owc_wetland", "context_id": row.get("sample_id", ""), "known": "exact sequence identity, date, methods depth, and processed expression", "ambiguous": "authoritative environment/flux join and expression normalization", "information_value": "highest paired-validation recovery opportunity", "next_measurement": "authoritative sample/date/depth-to-environment/flux crosswalk", "claim_unlocked": "field-validation eligibility, not automatic risk calibration"})
    for row in full_futian:
        sample_validation_candidate_portfolio.append({"priority": 2, "lane_id": "futian_mangrove_2026_qi", "context_id": row.get("sample_name", ""), "known": "chemistry-complete mudflat/mangrove depth sample", "ambiguous": "MAG-depth assignment and abundance", "information_value": "paired environmental-permissiveness design", "next_measurement": "depth-resolved read mapping/abundance plus process observation", "claim_unlocked": "sample-context association under explicit uncertainty"})
    for group, count in sorted(Counter(row.get("source_group", "") for row in msm_manifest).items()):
        sample_validation_candidate_portfolio.append({"priority": 3, "lane_id": "msm_china_2025", "context_id": group, "known": f"external source group with {count} local MAG candidates", "ambiguous": "3-20 sample set, ID mismatch, and 966 denominator", "information_value": "external source/study generalization", "next_measurement": "source-ID and raw-read/bin reconciliation", "claim_unlocked": "source-aware generalization tests"})

    metadata_gap_register = [
        {"lane_id": "poc_core", "gap": "reference accessions are not environmental sample resolution", "affected_claim": "blue-carbon sample context", "next_action": "retain as molecular reference; add paired field contexts"},
        {"lane_id": "msm_china_2025", "gap": "zero unique MAG-sample links; sample-ID mismatch; published 966 denominator unresolved", "affected_claim": "sample rollup and external generalization", "next_action": "recover authoritative crosswalk and read/bin provenance"},
        {"lane_id": "futian_mangrove_2026_qi", "gap": "zero depth-resolved MAG links or abundance; no process target", "affected_claim": "MAG-specific environmental attribution", "next_action": "depth-resolved read mapping and paired process measurement"},
        {"lane_id": "mucc_v1_owc_wetland", "gap": "0/133 authoritative ecology/flux joins; processed units and 23 WGS packages unresolved", "affected_claim": "field validation and activity", "next_action": "recover exact crosswalk, normalize expression, and reconcile assays"},
    ]

    expected = {
        "msm_samples": len(msm_samples), "msm_biosample_rows": len(msm_biosamples), "futian_samples": len(futian_samples),
        "mucc_samples": len(mucc_ecology), "mucc_exact_identity_date_depth": len(exact_mucc), "futian_full_chemistry": len(full_futian),
        "exact_sample_flux_links": sum(1 for row in link_sample_flux_window if row["flux_observation_id"]),
    }
    required_counts = {"msm_samples": 82, "msm_biosample_rows": 71, "futian_samples": 65, "mucc_samples": 133, "mucc_exact_identity_date_depth": 89, "futian_full_chemistry": 47, "exact_sample_flux_links": 0}
    mismatches = {key: {"observed": expected[key], "expected": value} for key, value in required_counts.items() if expected[key] != value}
    if mismatches:
        raise SystemExit(f"Metadata readiness count validation failed: {mismatches}")

    tables = {
        "dim_source_dataset": dim_source_dataset,
        "dim_site": dim_site,
        "dim_sample": dim_sample,
        "link_sample_mag": link_sample_mag,
        "fact_sample_environment": fact_sample_environment,
        "fact_sample_expression_or_abundance": fact_sample_expression_or_abundance,
        "fact_flux_or_process_observation": fact_flux_or_process_observation,
        "link_sample_flux_window": link_sample_flux_window,
        "feature_metadata_field_coverage": feature_metadata_field_coverage,
        "feature_sample_risk_readiness": feature_sample_risk_readiness,
        "sample_validation_candidate_portfolio": sample_validation_candidate_portfolio,
        "metadata_gap_register": metadata_gap_register,
    }
    manifest = [write_table(output_dir, name, rows) for name, rows in tables.items()]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "pass",
        "counts": expected,
        "claim_boundary": "Metadata-rich context is not a scored sample; exact sample-MAG and flux joins remain explicit gates.",
        "tables": manifest,
    }
    (output_dir / "metadata_readiness_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
