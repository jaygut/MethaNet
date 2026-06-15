#!/usr/bin/env python3
"""Create denominator and QC reconciliation artifacts for MSM China 2025."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import median


DEFAULT_MANIFEST = Path(
    "data/external/msm_china_2025/manifests/"
    "msm_china_2025_functional_embedding_manifest.tsv"
)
DEFAULT_OUTDIR = Path(
    "results/functional_metagenomics/msm_china_2025_20260615/"
    "qc_reconciliation"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def split_samples(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", value or "") if part.strip()]


def int_or_none(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def path_status(repo_root: Path, row: dict[str, str], column: str) -> str:
    path = repo_root / row[column]
    if path.is_file() and path.stat().st_size > 0:
        return "present"
    return "missing_or_empty"


def build_detail(repo_root: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        protein_count = int_or_none(row.get("protein_count", ""))
        if protein_count is None:
            protein_count_status = "missing"
        elif protein_count < 100:
            protein_count_status = "qc_warning_lt100_proteins"
        elif protein_count < 500:
            protein_count_status = "qc_warning_lt500_proteins"
        elif protein_count < 1000:
            protein_count_status = "qc_review_lt1000_proteins"
        else:
            protein_count_status = "protein_count_plausible"
        source_samples = split_samples(row.get("source_sample_ids", ""))
        out.append({
            "proteome_id": row["proteome_id"],
            "mag_id": row["mag_id"],
            "source_group": row.get("source_group", ""),
            "domain": row.get("domain", ""),
            "phylum": row.get("phylum", ""),
            "class": row.get("class", ""),
            "order": row.get("order", ""),
            "family": row.get("family", ""),
            "genus": row.get("genus", ""),
            "species": row.get("species", ""),
            "protein_count": protein_count if protein_count is not None else "",
            "protein_count_status": protein_count_status,
            "total_bp": row.get("total_bp", ""),
            "contigs": row.get("contigs", ""),
            "n50_bp": row.get("n50_bp", ""),
            "gc_pct": row.get("gc_pct", ""),
            "local_fna_status": path_status(repo_root, row, "local_fna_path"),
            "local_faa_status": path_status(repo_root, row, "local_faa_path"),
            "local_ffn_status": path_status(repo_root, row, "local_ffn_path"),
            "local_gff_status": path_status(repo_root, row, "local_gff_path"),
            "source_sample_count": len(source_samples),
            "mapped_ncbi_biosample_count": len(split_samples(row.get("mapped_ncbi_biosamples", ""))),
            "metadata_mapping_status": row.get("metadata_mapping_status", ""),
            "published_denominator_status": "not_reconciled_to_966_until_qc_metrics_available",
            "archive_denominator_status": row.get("denominator_status", ""),
            "functional_annotation_priority": annotation_priority(row, protein_count_status),
            "claim_scope": row.get("claim_scope", ""),
        })
    return out


def annotation_priority(row: dict[str, str], protein_count_status: str) -> str:
    domain = row.get("domain", "")
    phylum = row.get("phylum", "")
    family = row.get("family", "")
    genus = row.get("genus", "")
    taxon_text = " ".join([domain, phylum, family, genus]).lower()
    if protein_count_status == "qc_warning_lt100_proteins":
        return "qc_triage_before_mechanism_claims"
    if "methan" in taxon_text or domain == "Archaea":
        return "priority_archaea_methane_marker_review"
    if "desulf" in taxon_text or "sulfur" in taxon_text:
        return "priority_sulfur_competition_review"
    return "standard_full_annotation"


def counter_rows(counter: Counter[str], metric: str) -> list[dict[str, object]]:
    return [{"metric": metric, "value": key, "count": count} for key, count in sorted(counter.items())]


def build_summary(rows: list[dict[str, str]], detail: list[dict[str, object]]) -> list[dict[str, object]]:
    protein_counts = [int(d["protein_count"]) for d in detail if d["protein_count"] != ""]
    total_bp = [int_or_none(r.get("total_bp", "")) for r in rows]
    total_bp_int = [v for v in total_bp if v is not None]
    all_paths_ready = all(
        d["local_fna_status"] == d["local_faa_status"] == d["local_ffn_status"] == d["local_gff_status"] == "present"
        for d in detail
    )
    summary: list[dict[str, object]] = [
        {"metric": "paper_reported_medium_high_quality_mags", "value": 966, "interpretation": "published denominator; exact row subset not present in current downloaded metadata"},
        {"metric": "gigadb_archive_fastas", "value": len(rows), "interpretation": "local archive denominator from MAG_file.zip"},
        {"metric": "denominator_gap_archive_minus_paper", "value": len(rows) - 966, "interpretation": "requires CheckM2/contamination reconciliation before external claims"},
        {"metric": "all_payload_paths_ready", "value": str(all_paths_ready).lower(), "interpretation": "FNA/FAA/FFN/GFF path existence in final handoff manifest"},
        {"metric": "unique_proteome_id", "value": len({r["proteome_id"] for r in rows}), "interpretation": "canonical MethaNet key count"},
        {"metric": "unique_mag_id", "value": len({r["mag_id"] for r in rows}), "interpretation": "local MAG key count"},
        {"metric": "median_protein_count", "value": median(protein_counts) if protein_counts else "", "interpretation": "protein prediction sanity check"},
        {"metric": "min_protein_count", "value": min(protein_counts) if protein_counts else "", "interpretation": "low-protein rows need QC review"},
        {"metric": "max_protein_count", "value": max(protein_counts) if protein_counts else "", "interpretation": "large rows should be checked for assembly/bin scope"},
        {"metric": "median_total_bp", "value": median(total_bp_int) if total_bp_int else "", "interpretation": "assembly size sanity check"},
        {"metric": "rows_mapped_to_ncbi_biosample", "value": sum(bool(r.get("mapped_ncbi_biosamples")) for r in rows), "interpretation": "sample metadata mapping coverage"},
    ]
    return summary


def build_status_rows(detail: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in [
        "source_group",
        "domain",
        "protein_count_status",
        "functional_annotation_priority",
        "metadata_mapping_status",
        "archive_denominator_status",
    ]:
        rows.extend(counter_rows(Counter(str(d[column]) for d in detail), column))
    return rows


def build_gap_register() -> list[dict[str, object]]:
    return [
        {
            "gap_id": "MSM_QC_001",
            "gap": "Publication reports 966 medium/high-quality MAGs, but MAG_file.zip contains 1,428 FASTAs.",
            "affected_claim": "MSM added exactly 966 MAGs to MethaNet",
            "current_status": "blocked_for_external_count_claim",
            "next_validation_action": "Run cohort QC with CheckM2/GUNC and identify rows meeting completeness >=50 and contamination <=10, then compare to paper denominator.",
        },
        {
            "gap_id": "MSM_QC_002",
            "gap": "Current downloaded GigaDB small metadata files do not expose per-MAG completeness/contamination fields.",
            "affected_claim": "Downloaded archive rows equal the final published quality-filtered MAG set",
            "current_status": "not_supported",
            "next_validation_action": "Derive local QC metrics with MethaNet stack or recover author supplemental QC table if repository exposes it later.",
        },
        {
            "gap_id": "MSM_QC_003",
            "gap": "MAG-to-sample mapping is present, but abundance/read coverage is not yet joined into MethaNet sample rollups.",
            "affected_claim": "sample-level methane risk or ecosystem methane activity",
            "current_status": "blocked_for_sample_risk",
            "next_validation_action": "Join MAG abundance tables and environmental covariates after MAG-level annotation is complete.",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    outdir = args.outdir if args.outdir.is_absolute() else repo_root / args.outdir

    rows = read_rows(manifest)
    detail = build_detail(repo_root, rows)
    write_tsv(outdir / "msm_china_2025_qc_reconciliation_detail.tsv", detail)
    write_tsv(outdir / "msm_china_2025_qc_reconciliation_summary.tsv", build_summary(rows, detail))
    write_tsv(outdir / "msm_china_2025_qc_reconciliation_counts.tsv", build_status_rows(detail))
    write_tsv(outdir / "msm_china_2025_validation_gap_register.tsv", build_gap_register())

    print(f"wrote\t{outdir}")
    print(f"rows\t{len(rows)}")
    print(f"paper_denominator\t966")
    print(f"archive_denominator\t{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
