#!/usr/bin/env python3
"""Validate and stage an authoritative MUCC sequence-to-ecology crosswalk.

This builder is intentionally input-driven: it does not guess a mapping from
sample-label tokens, dates, coordinates, depths, or site names.  It accepts a
publisher- or author-supplied canonical TSV in which every declared link is
explicit.  Complete, validated rows can unlock *ecological-validation
eligibility* in the promoted atlas; they never assign a final MRV risk tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
COMPLETE_EVIDENCE_STATUS = "authoritative_complete"
PARTIAL_EVIDENCE_STATUS = "authoritative_partial"
MISSING_EVIDENCE_STATUS = "authoritative_missing"
VALID_ASSAY_STATUSES = {
    "validated_metatranscriptome",
    "validated_WGS_to_expression_reconciliation",
}
REQUIRED_COLUMNS = [
    "mapping_id",
    "source_sample_column",
    "authoritative_sample_id",
    "collection_datetime",
    "site_id",
    "core_or_plot_id",
    "depth_cm",
    "depth_reference",
    "sequence_assay_type",
    "assay_reconciliation_status",
    "mag_abundance_or_read_coverage_record_id",
    "mag_abundance_or_read_coverage_units",
    "environment_source",
    "environment_record_id",
    "environment_measurement_datetime",
    "environment_measurement_units",
    "flux_source",
    "flux_observation_id",
    "flux_measurement_type",
    "flux_units",
    "flux_window_start_datetime",
    "flux_window_end_datetime",
    "replicate_id",
    "uncertainty_record_id",
    "uncertainty_method",
    "source_evidence_status",
    "missingness_status",
    "source_url",
]
COMPLETE_VALUE_COLUMNS = [
    "authoritative_sample_id",
    "collection_datetime",
    "site_id",
    "core_or_plot_id",
    "depth_cm",
    "depth_reference",
    "sequence_assay_type",
    "assay_reconciliation_status",
    "mag_abundance_or_read_coverage_record_id",
    "mag_abundance_or_read_coverage_units",
    "environment_source",
    "environment_record_id",
    "environment_measurement_datetime",
    "environment_measurement_units",
    "flux_source",
    "flux_observation_id",
    "flux_measurement_type",
    "flux_units",
    "flux_window_start_datetime",
    "flux_window_end_datetime",
    "replicate_id",
    "uncertainty_record_id",
    "uncertainty_method",
    "source_url",
]
CLAIM_BOUNDARY = (
    "An authoritative sequence-to-ecology mapping can establish eligibility for grouped ecological "
    "validation only when its declared metadata, abundance/read coverage, environment, flux window, "
    "assay, and uncertainty fields validate. It does not itself establish a causal mechanism, measured "
    "MAG-level flux effect, final MRV score/A-E tier, carbon-crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical author/publisher TSV meeting the documented ecological-crosswalk contract.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def parse_iso_datetime(value: str) -> datetime | None:
    """Accept only an ISO datetime; date-only values cannot define a flux window."""

    text = str(value).strip()
    if "T" not in text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def valid_nonnegative_number(value: str) -> bool:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _essdive_ids(run_dir: Path) -> tuple[set[str], set[str], set[str]]:
    chamber_path = (
        run_dir / "environmental_metadata/fact_mucc_v1_essdive_chamber_flux.tsv"
    )
    porewater_path = (
        run_dir / "environmental_metadata/fact_mucc_v1_essdive_porewater_ch4.tsv"
    )
    tower_flux_path = (
        run_dir
        / "environmental_metadata/fact_mucc_v1_essdive_gapfilled_tower_ch4_flux.tsv"
    )
    chamber_ids = (
        set(read_tsv(chamber_path)["flux_observation_id"].astype(str))
        if chamber_path.is_file()
        else set()
    )
    porewater_ids = (
        set(read_tsv(porewater_path)["porewater_observation_id"].astype(str))
        if porewater_path.is_file()
        else set()
    )
    tower_flux_ids = (
        set(read_tsv(tower_flux_path)["flux_observation_id"].astype(str))
        if tower_flux_path.is_file()
        else set()
    )
    return chamber_ids, porewater_ids, tower_flux_ids


def _source_is_essdive(value: str) -> bool:
    return str(value).strip().upper().startswith("ESS_DIVE")


def _source_is_essdive_gapfilled_tower(value: str) -> bool:
    normalized = str(value).strip().upper()
    return normalized in {
        "ESS_DIVE_10.15485_2500238",
        "ESS_DIVE_GAPFILLED_TOWER_10.15485_2500238",
    }


def validate_crosswalk(
    crosswalk: pd.DataFrame,
    sample_scaffold: pd.DataFrame,
    chamber_ids: set[str],
    porewater_ids: set[str],
    tower_flux_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return validated link rows and one explicit readiness row for every source sample."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in crosswalk.columns
    ]
    if missing_columns:
        raise ValueError(
            f"authoritative ecological crosswalk lacks required columns: {missing_columns}"
        )
    mapping_ids = crosswalk["mapping_id"].astype(str).str.strip()
    if mapping_ids.eq("").any() or mapping_ids.duplicated().any():
        raise ValueError("mapping_id must be non-empty and unique")
    if (
        sample_scaffold["sample_id"].duplicated().any()
        or sample_scaffold["source_sample_column"].duplicated().any()
    ):
        raise ValueError(
            "source sample scaffold must have unique sample_id and source_sample_column"
        )
    lookup = sample_scaffold.set_index("source_sample_column")["sample_id"].to_dict()
    output_rows: list[dict[str, str]] = []
    valid_status = "validated_authoritative_sample_environment_flux_mapping"
    for _, source in crosswalk.iterrows():
        row = {
            column: str(source.get(column, "")).strip() for column in REQUIRED_COLUMNS
        }
        source_label = row["source_sample_column"]
        row["lane_id"] = LANE_ID
        row["sample_id"] = lookup.get(source_label, "")
        row["mapping_validation_status"] = ""
        row["validation_detail"] = ""
        row["claim_boundary"] = CLAIM_BOUNDARY
        if not row["sample_id"]:
            row["mapping_validation_status"] = "invalid_unknown_source_sample_column"
            row["validation_detail"] = (
                "source_sample_column is absent from the 133-row source roster"
            )
            output_rows.append(row)
            continue
        evidence_status = row["source_evidence_status"]
        if evidence_status not in {
            COMPLETE_EVIDENCE_STATUS,
            PARTIAL_EVIDENCE_STATUS,
            MISSING_EVIDENCE_STATUS,
        }:
            row["mapping_validation_status"] = "invalid_source_evidence_status"
            row["validation_detail"] = (
                "source_evidence_status must be authoritative_complete, authoritative_partial, or authoritative_missing"
            )
            output_rows.append(row)
            continue
        if evidence_status != COMPLETE_EVIDENCE_STATUS:
            row["mapping_validation_status"] = (
                "explicit_authoritative_mapping_missing"
                if evidence_status == MISSING_EVIDENCE_STATUS
                else "explicit_authoritative_mapping_partial"
            )
            row["validation_detail"] = (
                "declared missingness is preserved; this row cannot unlock ecological validation"
            )
            output_rows.append(row)
            continue
        blank = [column for column in COMPLETE_VALUE_COLUMNS if not row[column]]
        if blank:
            row["mapping_validation_status"] = (
                "invalid_complete_mapping_missing_required_values"
            )
            row["validation_detail"] = (
                f"authoritative_complete row has blank values: {';'.join(blank)}"
            )
            output_rows.append(row)
            continue
        timestamps = {
            column: parse_iso_datetime(row[column])
            for column in [
                "collection_datetime",
                "environment_measurement_datetime",
                "flux_window_start_datetime",
                "flux_window_end_datetime",
            ]
        }
        invalid_times = [
            column for column, value in timestamps.items() if value is None
        ]
        if invalid_times:
            row["mapping_validation_status"] = (
                "invalid_complete_mapping_non_ISO_datetime"
            )
            row["validation_detail"] = (
                f"requires ISO datetimes with time component: {';'.join(invalid_times)}"
            )
            output_rows.append(row)
            continue
        if (
            timestamps["flux_window_start_datetime"]
            > timestamps["flux_window_end_datetime"]
        ):
            row["mapping_validation_status"] = "invalid_flux_window_time_order"
            row["validation_detail"] = (
                "flux_window_start_datetime is after flux_window_end_datetime"
            )
            output_rows.append(row)
            continue
        if not valid_nonnegative_number(row["depth_cm"]):
            row["mapping_validation_status"] = "invalid_complete_mapping_depth_cm"
            row["validation_detail"] = "depth_cm must be a finite non-negative number"
            output_rows.append(row)
            continue
        if row["assay_reconciliation_status"] not in VALID_ASSAY_STATUSES:
            row["mapping_validation_status"] = "invalid_assay_reconciliation_status"
            row["validation_detail"] = (
                "assay_reconciliation_status must establish metatranscriptome equivalence or a documented WGS-to-expression reconciliation"
            )
            output_rows.append(row)
            continue
        if _source_is_essdive_gapfilled_tower(row["flux_source"]) and row[
            "flux_observation_id"
        ] not in (tower_flux_ids or set()):
            row["mapping_validation_status"] = (
                "invalid_ESS_DIVE_gapfilled_tower_flux_observation_id"
            )
            row["validation_detail"] = (
                "flux_observation_id is absent from staged ESS-DIVE DOI 10.15485/2500238 "
                "gap-filled tower records"
            )
            output_rows.append(row)
            continue
        if (
            _source_is_essdive(row["flux_source"])
            and not _source_is_essdive_gapfilled_tower(row["flux_source"])
            and row["flux_observation_id"] not in chamber_ids
        ):
            row["mapping_validation_status"] = "invalid_ESS_DIVE_flux_observation_id"
            row["validation_detail"] = (
                "flux_observation_id is absent from staged ESS-DIVE chamber records"
            )
            output_rows.append(row)
            continue
        if (
            _source_is_essdive(row["environment_source"])
            and row["environment_record_id"] not in porewater_ids
        ):
            row["mapping_validation_status"] = "invalid_ESS_DIVE_environment_record_id"
            row["validation_detail"] = (
                "environment_record_id is absent from staged ESS-DIVE porewater records"
            )
            output_rows.append(row)
            continue
        row["mapping_validation_status"] = valid_status
        row["validation_detail"] = (
            "explicit sample/date/depth/assay/coverage/environment/flux/uncertainty mapping validated"
        )
        output_rows.append(row)
    links = pd.DataFrame(output_rows)
    if links["mapping_id"].duplicated().any():
        raise ValueError("validated crosswalk must preserve unique mapping_id values")
    readiness_rows: list[dict[str, str]] = []
    for _, sample in sample_scaffold.iterrows():
        sample_id = str(sample["sample_id"])
        subset = links.loc[links["sample_id"].eq(sample_id)]
        valid = subset["mapping_validation_status"].eq(valid_status)
        readiness_rows.append(
            {
                "lane_id": LANE_ID,
                "sample_id": sample_id,
                "source_sample_column": str(sample["source_sample_column"]),
                "authoritative_mapping_rows": str(len(subset)),
                "validated_authoritative_mapping_rows": str(int(valid.sum())),
                "authoritative_ecology_readiness_status": (
                    "ready_for_grouped_ecological_validation"
                    if valid.any()
                    else "blocked_no_validated_authoritative_mapping"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    readiness = pd.DataFrame(readiness_rows)
    return links, readiness


def source_manifest(input_path: Path, retrieved_utc: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lane_id": LANE_ID,
                "source_id": "authoritative_MUCC_sequence_to_ecology_crosswalk",
                "artifact_role": "authoritative_sample_depth_environment_flux_crosswalk_input",
                "source_url": "retained_in_input_source_url_column",
                "local_path": str(input_path),
                "bytes": str(input_path.stat().st_size),
                "sha256": sha256(input_path),
                "retrieved_utc": retrieved_utc,
                "validation_status": "input_retained_and_schema_validated",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ]
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    input_path = resolve(repo_root, args.input)
    if not input_path.is_file() or not input_path.stat().st_size:
        raise FileNotFoundError(
            f"authoritative ecological crosswalk input not found or empty: {input_path}"
        )
    sample_scaffold = read_tsv(
        run_dir / "environmental_metadata/mucc_v1_sample_columns_scaffold.tsv"
    )
    links, readiness = validate_crosswalk(
        read_tsv(input_path),
        sample_scaffold,
        *_essdive_ids(run_dir),
    )
    environmental_dir = run_dir / "environmental_metadata"
    audit_dir = run_dir / "source_audit/authoritative_ecological_crosswalk"
    link_path = environmental_dir / "link_mucc_v1_sequence_authoritative_ecology.tsv"
    readiness_path = (
        environmental_dir / "feature_mucc_v1_authoritative_ecology_readiness.tsv"
    )
    write_tsv(link_path, links)
    write_tsv(readiness_path, readiness)
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    write_tsv(
        audit_dir / "source_file_manifest.tsv",
        source_manifest(input_path, retrieved_utc),
    )
    summary = {
        "lane_id": LANE_ID,
        "input": str(input_path),
        "link_rows": int(len(links)),
        "source_samples": int(len(readiness)),
        "validated_mapping_rows": int(
            links["mapping_validation_status"]
            .eq("validated_authoritative_sample_environment_flux_mapping")
            .sum()
        ),
        "samples_ready_for_grouped_ecological_validation": int(
            readiness["authoritative_ecology_readiness_status"]
            .eq("ready_for_grouped_ecological_validation")
            .sum()
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (audit_dir / "crosswalk_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
