#!/usr/bin/env python3
"""Build a MethaNet functional-run manifest for the MSM China 2025 MAG payload."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path(
    "data/external/msm_china_2025/manifests/"
    "msm_china_2025_functional_embedding_manifest.tsv"
)
DEFAULT_OUTPUT = Path(
    "results/functional_metagenomics/msm_china_2025_20260615/manifests/"
    "msm_china_2025_functional_mag_manifest.tsv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def status_from_paths(repo_root: Path, row: dict[str, str]) -> tuple[str, str]:
    fna = repo_root / row["local_fna_path"]
    faa = repo_root / row["local_faa_path"]
    if fna.is_file() and fna.stat().st_size > 0 and faa.is_file() and faa.stat().st_size > 0:
        return "matched", "ready_for_functional_annotation_pending_qc_reconciliation"
    missing = []
    if not fna.is_file() or fna.stat().st_size == 0:
        missing.append("local_fna_path")
    if not faa.is_file() or faa.stat().st_size == 0:
        missing.append("local_faa_path")
    return "missing_payload", "blocked_missing_" + "_".join(missing)


def build_manifest(repo_root: Path, source: Path) -> list[dict[str, str]]:
    rows = read_rows(source)
    out: list[dict[str, str]] = []
    for row in rows:
        match_status, comparability_status = status_from_paths(repo_root, row)
        functional_include = match_status == "matched"
        out.append({
            "proteome_id": row["proteome_id"],
            "mag_id": row["mag_id"],
            "source": "msm_china_2025",
            "ecosystem": "mangrove_sediment",
            "domain": row.get("domain", ""),
            "source_group": row.get("source_group", ""),
            "source_fasta_filename": row.get("source_fasta_filename", ""),
            "mag_fasta": row["local_fna_path"],
            "proteome_faa": row["local_faa_path"],
            "local_ffn_path": row.get("local_ffn_path", ""),
            "local_gff_path": row.get("local_gff_path", ""),
            "protein_count": row.get("protein_count", ""),
            "protein_qc_tier": row.get("protein_qc_tier", ""),
            "match_status": match_status,
            "functional_run_include": "true" if functional_include else "false",
            "analysis_unit_type": "mag_bin",
            "mbag_mag_level_include": "true" if functional_include else "false",
            "claim_scope": "MAG/proteome functional potential after QC and annotation",
            "comparability_status": comparability_status,
            "denominator_status": row.get("denominator_status", ""),
            "metadata_mapping_status": row.get("metadata_mapping_status", ""),
            "mapped_ncbi_biosamples": row.get("mapped_ncbi_biosamples", ""),
            "mapped_ncbi_bioprojects": row.get("mapped_ncbi_bioprojects", ""),
            "source_sample_ids": row.get("source_sample_ids", ""),
            "total_bp": row.get("total_bp", ""),
            "contigs": row.get("contigs", ""),
            "n50_bp": row.get("n50_bp", ""),
            "gc_pct": row.get("gc_pct", ""),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source = args.input if args.input.is_absolute() else repo_root / args.input
    output = args.output if args.output.is_absolute() else repo_root / args.output
    rows = build_manifest(repo_root, source)
    write_rows(output, rows)

    included = sum(row["functional_run_include"] == "true" for row in rows)
    print(f"wrote\t{output}")
    print(f"rows\t{len(rows)}")
    print(f"functional_run_include\t{included}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
