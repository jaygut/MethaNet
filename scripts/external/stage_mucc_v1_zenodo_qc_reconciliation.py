#!/usr/bin/env python3
"""Reconcile MUCC v1 MAG quality evidence from the record-specific Zenodo payload.

The Borton et al. mSystems paper defines the 2,502 medium/high-quality MAGs
as CheckM completeness >=50% and contamination <10%.  The record-specific
Zenodo DRAM annotation file supplies those fields at gene-row granularity.  We
collapse only direct ``fasta == MAG ID`` evidence to one QC record per archive
MAG, retain any inconsistent repeated values, and keep the six non-qualifying
archive entries explicit.  This establishes a source-QC roster; it does not
establish ecological associations, methane flux, MRV tiers, crediting, or
cross-ecosystem transfer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
PAPER_DOI = "10.1128/msystems.00680-25"
ZENODO_RECORD_DOI = "10.5281/zenodo.8194033"
PUBLISHED_HQMQ_MAGS = 2502
PUBLISHED_ARCHIVE_MAGS = 2508
MIN_COMPLETENESS = 50.0
MAX_CONTAMINATION_EXCLUSIVE = 10.0
CLAIM_BOUNDARY = (
    "Direct source-QC evidence for archive MAGs only; no ecological association, measured methane "
    "flux, final MRV score/A-E tier, carbon-crediting claim, or source-independent transfer claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def clean(value: Any) -> str:
    return str(value or "").strip()


def parse_float(value: Any) -> float | None:
    try:
        return float(clean(value))
    except ValueError:
        return None


def normalized_float(value: Any) -> str:
    number = parse_float(value)
    return f"{number:.12g}" if number is not None else ""


def source_quality_status(completeness: float | None, contamination: float | None) -> str:
    """Classify exact paper-defined medium/high-quality membership from source QC."""
    if completeness is None or contamination is None:
        return "source_qc_value_missing"
    if completeness >= MIN_COMPLETENESS and contamination < MAX_CONTAMINATION_EXCLUSIVE:
        return "meets_published_MQHQ_CheckM_threshold"
    return "does_not_meet_published_MQHQ_CheckM_threshold"


@dataclass
class SourceQC:
    source_annotation_rows: int = 0
    completeness_values: set[str] = field(default_factory=set)
    contamination_values: set[str] = field(default_factory=set)
    taxonomy_values: set[str] = field(default_factory=set)
    direct_source_fasta_values: set[str] = field(default_factory=set)
    crosswalk_source_bin_values: set[str] = field(default_factory=set)

    def add(self, row: dict[str, str], source_fasta: str, direct_catalog_match: bool) -> None:
        self.source_annotation_rows += 1
        if direct_catalog_match:
            self.direct_source_fasta_values.add(source_fasta)
        else:
            self.crosswalk_source_bin_values.add(source_fasta)
        completeness = normalized_float(row.get("bin_completeness"))
        contamination = normalized_float(row.get("bin_contamination"))
        taxonomy = clean(row.get("bin_taxonomy"))
        if completeness:
            self.completeness_values.add(completeness)
        if contamination:
            self.contamination_values.add(contamination)
        if taxonomy:
            self.taxonomy_values.add(taxonomy)


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or "mag_id" not in rows[0]:
        raise ValueError(f"MUCC MAG catalog is missing mag_id rows: {path}")
    ids = [clean(row.get("mag_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"MUCC MAG catalog has duplicate mag_id values: {path}")
    return rows


def read_source_bin_crosswalk(path: Path, catalog_ids: set[str]) -> tuple[dict[str, str], set[str]]:
    """Read only unambiguous source-bin -> catalog-MAG mappings from staged source annotations."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing source-bin crosswalk required for alias QC recovery: {path}")
    candidates: dict[str, set[str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_bin = clean(row.get("source_bin_id"))
            mag_id = clean(row.get("mag_id"))
            if source_bin and mag_id in catalog_ids:
                candidates.setdefault(source_bin, set()).add(mag_id)
    ambiguous = {source_bin for source_bin, mag_ids in candidates.items() if len(mag_ids) != 1}
    return (
        {
            source_bin: next(iter(mag_ids))
            for source_bin, mag_ids in candidates.items()
            if len(mag_ids) == 1
        },
        ambiguous,
    )


def parse_direct_source_qc(
    path: Path,
    catalog_ids: set[str],
    source_bin_to_mag: dict[str, str],
    ambiguous_source_bins: set[str],
) -> tuple[dict[str, SourceQC], Counter[str], Counter[str], int, list[str]]:
    """Stream source DRAM QC, preserving direct and unambiguous alias mappings separately."""
    qc_by_mag: dict[str, SourceQC] = {}
    non_catalog_fasta_rows: Counter[str] = Counter()
    ambiguous_source_bin_rows: Counter[str] = Counter()
    total_rows = 0
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"fasta", "bin_completeness", "bin_contamination", "bin_taxonomy"}
        headers = set(reader.fieldnames or [])
        missing = sorted(expected - headers)
        if missing:
            raise ValueError(f"Source DRAM payload lacks required QC fields: {', '.join(missing)}")
        for row in reader:
            total_rows += 1
            source_fasta = clean(row.get("fasta"))
            direct_catalog_match = source_fasta in catalog_ids
            if direct_catalog_match:
                mag_id = source_fasta
            elif source_fasta in ambiguous_source_bins:
                ambiguous_source_bin_rows[source_fasta] += 1
                continue
            else:
                mag_id = source_bin_to_mag.get(source_fasta, "")
            if not mag_id:
                non_catalog_fasta_rows[source_fasta] += 1
                continue
            qc_by_mag.setdefault(mag_id, SourceQC()).add(row, source_fasta, direct_catalog_match)
    return qc_by_mag, non_catalog_fasta_rows, ambiguous_source_bin_rows, total_rows, sorted(headers)


def one_or_blank(values: set[str]) -> str:
    return next(iter(values)) if len(values) == 1 else ""


def consistency_status(qc: SourceQC | None) -> str:
    if qc is None or not qc.source_annotation_rows:
        return "no_direct_source_annotation_rows"
    if len(qc.completeness_values) > 1 or len(qc.contamination_values) > 1:
        return "conflicting_repeated_source_qc_values_retained_for_review"
    if not qc.completeness_values or not qc.contamination_values:
        return "direct_source_annotation_rows_missing_one_or_more_qc_values"
    return "direct_source_qc_values_consistent_across_annotation_rows"


def mapping_status(qc: SourceQC | None) -> str:
    if qc is None or not qc.source_annotation_rows:
        return "no_direct_or_crosswalk_source_QC_rows"
    if qc.direct_source_fasta_values and qc.crosswalk_source_bin_values:
        return "direct_catalog_fasta_and_source_bin_crosswalk_QC"
    if qc.direct_source_fasta_values:
        return "direct_catalog_fasta_QC"
    return "source_bin_crosswalk_QC"


def output_rows(catalog: list[dict[str, str]], qc_by_mag: dict[str, SourceQC]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for catalog_row in sorted(catalog, key=lambda row: clean(row.get("mag_id"))):
        mag_id = clean(catalog_row.get("mag_id"))
        qc = qc_by_mag.get(mag_id)
        completeness = one_or_blank(qc.completeness_values) if qc else ""
        contamination = one_or_blank(qc.contamination_values) if qc else ""
        completeness_number = parse_float(completeness)
        contamination_number = parse_float(contamination)
        quality_status = source_quality_status(completeness_number, contamination_number)
        qc_consistency = consistency_status(qc)
        source_evidence_status = (
            "direct_Zenodo_DRAM_QC_values_reconciled"
            if qc_consistency == "direct_source_qc_values_consistent_across_annotation_rows"
            else "direct_Zenodo_DRAM_QC_values_require_review"
            if qc
            else "direct_Zenodo_DRAM_QC_values_not_observed"
        )
        rows.append(
            {
                "lane_id": LANE_ID,
                "proteome_id": clean(catalog_row.get("proteome_id")) or f"mucc_v1__{mag_id}",
                "mag_id": mag_id,
                "source_annotation_rows": str(qc.source_annotation_rows if qc else 0),
                "source_qc_mapping_status": mapping_status(qc),
                "source_qc_direct_fasta_values": ";".join(
                    sorted(qc.direct_source_fasta_values) if qc else []
                ),
                "source_qc_crosswalk_bin_values": ";".join(
                    sorted(qc.crosswalk_source_bin_values) if qc else []
                ),
                "bin_completeness": completeness,
                "bin_contamination": contamination,
                "bin_taxonomy": one_or_blank(qc.taxonomy_values) if qc else "",
                "source_qc_value_consistency_status": qc_consistency,
                "source_qc_evidence_status": source_evidence_status,
                "published_mq_hq_membership_status": quality_status,
                "published_quality_definition": "adapted_MIMARKS_CheckM_completeness_ge_50pct_and_contamination_lt_10pct",
                "quality_definition_source": f"Borton_et_al_{PAPER_DOI}_Methods_Figure_S1_and_Table_S3",
                "source_record_doi": ZENODO_RECORD_DOI,
                "source_file": "OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz",
                "allowed_claim_wording": (
                    "Direct Zenodo annotation QC meets the paper-defined MQ/HQ CheckM screen; "
                    "this is MAG quality evidence, not ecological or MRV evidence."
                    if quality_status == "meets_published_MQHQ_CheckM_threshold"
                    else "Direct Zenodo annotation QC does not meet the paper-defined MQ/HQ CheckM screen; "
                    "the archive member is retained explicitly."
                    if quality_status == "does_not_meet_published_MQHQ_CheckM_threshold"
                    else "Direct Zenodo QC values require review before assigning paper-defined MQ/HQ membership."
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def reconciliation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "lane_id": row["lane_id"],
            "proteome_id": row["proteome_id"],
            "mag_id": row["mag_id"],
            "archive_membership_status": "checksum_validated_Zenodo_MAG_archive_member",
            "source_annotation_rows": row["source_annotation_rows"],
            "source_qc_mapping_status": row["source_qc_mapping_status"],
            "bin_completeness": row["bin_completeness"],
            "bin_contamination": row["bin_contamination"],
            "source_qc_value_consistency_status": row["source_qc_value_consistency_status"],
            "published_mq_hq_membership_status": row["published_mq_hq_membership_status"],
            "reconciliation_status": (
                "published_2502_HQMQ_roster_member_reconciled_from_direct_Zenodo_QC"
                if row["published_mq_hq_membership_status"]
                == "meets_published_MQHQ_CheckM_threshold"
                else "archive_member_outside_published_2502_HQMQ_QC_scope_reconciled_from_direct_Zenodo_QC"
                if row["published_mq_hq_membership_status"]
                == "does_not_meet_published_MQHQ_CheckM_threshold"
                else "published_HQMQ_membership_requires_source_QC_review"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for row in rows
    ]


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty QC evidence table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    catalog_path = run_dir / "manifests/mucc_v1_mag_catalog_full.tsv"
    source_bin_crosswalk_path = run_dir / "manifests/mucc_v1_source_bin_crosswalk.tsv"
    source_path = run_dir / "staging/OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing record-specific Zenodo source QC payload: {source_path}")

    catalog = read_catalog(catalog_path)
    catalog_ids = {clean(row.get("mag_id")) for row in catalog}
    source_bin_to_mag, ambiguous_source_bins = read_source_bin_crosswalk(
        source_bin_crosswalk_path,
        catalog_ids,
    )
    (
        qc_by_mag,
        non_catalog_rows,
        ambiguous_source_bin_rows,
        source_rows,
        source_headers,
    ) = parse_direct_source_qc(
        source_path,
        catalog_ids,
        source_bin_to_mag,
        ambiguous_source_bins,
    )
    rows = output_rows(catalog, qc_by_mag)
    reconciled = reconciliation_rows(rows)

    output_path = run_dir / "functional_features/feature_mucc_v1_zenodo_source_qc.tsv"
    reconciliation_path = run_dir / "reconciliation/mucc_v1_zenodo_source_qc_reconciliation.tsv"
    source_manifest_path = run_dir / "source_audit/zenodo_mucc_v1_qc_source_manifest.tsv"
    write_tsv(output_path, rows)
    write_tsv(reconciliation_path, reconciled)

    counts = Counter(row["published_mq_hq_membership_status"] for row in rows)
    consistency_counts = Counter(row["source_qc_value_consistency_status"] for row in rows)
    exact_reconciliation = (
        len(rows) == PUBLISHED_ARCHIVE_MAGS
        and len(qc_by_mag) == PUBLISHED_ARCHIVE_MAGS
        and counts["meets_published_MQHQ_CheckM_threshold"] == PUBLISHED_HQMQ_MAGS
        and counts["does_not_meet_published_MQHQ_CheckM_threshold"]
        == PUBLISHED_ARCHIVE_MAGS - PUBLISHED_HQMQ_MAGS
        and consistency_counts["direct_source_qc_values_consistent_across_annotation_rows"]
        == PUBLISHED_ARCHIVE_MAGS
    )
    reconciliation_status = (
        "exact_published_2502_HQMQ_and_six_archive_scope_difference_reconciled"
        if exact_reconciliation
        else "source_QC_reconciliation_incomplete_or_nonconcordant_review_required"
    )
    source_manifest_rows = [
        {
            "lane_id": LANE_ID,
            "source_record_doi": ZENODO_RECORD_DOI,
            "source_file": str(source_path.relative_to(repo_root)),
            "source_file_sha256": sha256(source_path),
            "source_file_bytes": str(source_path.stat().st_size),
            "source_annotation_rows_streamed": str(source_rows),
            "source_direct_catalog_MAGs": str(len(qc_by_mag)),
            "source_bin_crosswalk_values": str(len(source_bin_to_mag)),
            "ambiguous_source_bin_values": str(len(ambiguous_source_bins)),
            "ambiguous_source_bin_rows": str(sum(ambiguous_source_bin_rows.values())),
            "non_catalog_fasta_values": str(len(non_catalog_rows)),
            "non_catalog_fasta_rows": str(sum(non_catalog_rows.values())),
            "required_source_headers": ";".join(
                ["fasta", "bin_completeness", "bin_contamination", "bin_taxonomy"]
            ),
            "observed_source_headers": ";".join(source_headers),
            "published_quality_definition": "adapted_MIMARKS_CheckM_completeness_ge_50pct_and_contamination_lt_10pct",
            "quality_definition_source": f"Borton_et_al_{PAPER_DOI}_Methods_Figure_S1_and_Table_S3",
            "reconciliation_status": reconciliation_status,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    ]
    write_tsv(source_manifest_path, source_manifest_rows)

    summary = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "lane_id": LANE_ID,
        "source_record_doi": ZENODO_RECORD_DOI,
        "paper_doi": PAPER_DOI,
        "catalog_MAG_rows": len(rows),
        "source_annotation_rows_streamed": source_rows,
        "direct_catalog_MAGs_with_source_QC": len(qc_by_mag),
        "unambiguous_source_bin_crosswalk_values": len(source_bin_to_mag),
        "ambiguous_source_bin_values": len(ambiguous_source_bins),
        "ambiguous_source_bin_rows": sum(ambiguous_source_bin_rows.values()),
        "non_catalog_fasta_values": len(non_catalog_rows),
        "non_catalog_fasta_rows": sum(non_catalog_rows.values()),
        "published_mq_hq_MAGs_reconciled": counts[
            "meets_published_MQHQ_CheckM_threshold"
        ],
        "archive_MAGs_outside_published_mq_hq_scope": counts[
            "does_not_meet_published_MQHQ_CheckM_threshold"
        ],
        "missing_or_conflicting_QC_MAGs": len(rows)
        - consistency_counts["direct_source_qc_values_consistent_across_annotation_rows"],
        "reconciliation_status": reconciliation_status,
        "feature_table": str(output_path.relative_to(repo_root)),
        "reconciliation_table": str(reconciliation_path.relative_to(repo_root)),
        "source_manifest": str(source_manifest_path.relative_to(repo_root)),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    summary_path = run_dir / "reports/mucc_v1_zenodo_source_qc_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
