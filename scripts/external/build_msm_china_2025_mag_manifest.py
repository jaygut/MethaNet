#!/usr/bin/env python3
"""Build a resolved MAG FASTA manifest for the MSM China 2025 GigaDB package."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path


DATASET_DIR = Path("data/external/msm_china_2025")
EXTRACTED_ROOT = DATASET_DIR / "extracted" / "MAG_file"
CLASSIFICATION_CSV = DATASET_DIR / "gigadb_wasabi" / "MAGs_gtdbtk_classification.csv"
GENOMES_DIR = DATASET_DIR / "genomes_fna"
MANIFEST_OUT = DATASET_DIR / "manifests" / "msm_china_2025_mag_manifest.resolved.tsv"


def safe_id(name: str) -> str:
    stem = re.sub(r"\.(fa|fna|fasta)$", "", name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def fasta_stats(path: Path) -> dict[str, str]:
    contig_lengths: list[int] = []
    total_bp = 0
    gc = 0
    current = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    contig_lengths.append(current)
                    current = 0
                continue
            seq = line.upper()
            current += len(seq)
            total_bp += len(seq)
            gc += seq.count("G") + seq.count("C")
    if current:
        contig_lengths.append(current)

    contig_lengths.sort(reverse=True)
    half = total_bp / 2
    running = 0
    n50 = 0
    for length in contig_lengths:
        running += length
        if running >= half:
            n50 = length
            break
    return {
        "total_bp": str(total_bp),
        "contigs": str(len(contig_lengths)),
        "n50_bp": str(n50),
        "gc_pct": f"{(gc / total_bp * 100):.4f}" if total_bp else "",
    }


def load_classification() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with CLASSIFICATION_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        next(reader, None)
        for row in reader:
            if not row or not row[0]:
                continue
            rows[row[0]] = {
                "domain": row[1] if len(row) > 1 else "",
                "phylum": row[2] if len(row) > 2 else "",
                "class": row[3] if len(row) > 3 else "",
                "order": row[4] if len(row) > 4 else "",
                "family": row[5] if len(row) > 5 else "",
                "genus": row[6] if len(row) > 6 else "",
                "species": row[7] if len(row) > 7 else "",
                "source_sample_ids": row[8] if len(row) > 8 else "",
            }
    return rows


def ensure_hardlink(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(source, dest)
    except OSError:
        import shutil

        shutil.copy2(source, dest)


def main() -> int:
    classification = load_classification()
    fasta_paths = sorted(EXTRACTED_ROOT.glob("**/*.fa"))
    rows: list[dict[str, str]] = []
    GENOMES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in GENOMES_DIR.glob("*.fna"):
        stale.unlink()

    for fasta_path in fasta_paths:
        source_filename = fasta_path.name
        group_folder = fasta_path.parent.name.replace("groep", "group")
        source_mag_id = f"{group_folder}__{safe_id(source_filename)}"
        local_fna = GENOMES_DIR / f"{source_mag_id}.fna"
        ensure_hardlink(fasta_path, local_fna)
        class_row = classification.get(source_filename, {})
        stats = fasta_stats(local_fna)
        rows.append(
            {
                "external_dataset_id": "msm_china_2025",
                "source_project_id": "MSM_China_2014_2020_Pan_GigaScience_2025",
                "source_group": group_folder,
                "source_fasta_filename": source_filename,
                "mag_id": source_mag_id,
                "proteome_id": f"msm_china_2025__{source_mag_id}",
                "local_fna_path": str(local_fna),
                "local_faa_path": str(DATASET_DIR / "proteomes_faa" / f"{source_mag_id}.faa"),
                "protein_prediction_status": "pending",
                "domain": class_row.get("domain", ""),
                "phylum": class_row.get("phylum", ""),
                "class": class_row.get("class", ""),
                "order": class_row.get("order", ""),
                "family": class_row.get("family", ""),
                "genus": class_row.get("genus", ""),
                "species": class_row.get("species", ""),
                "source_sample_ids": class_row.get("source_sample_ids", ""),
                "source_archive_path": str(fasta_path),
                "archive_md5": "c69a96c13d84ae0fe1a52005bcb644cd",
                "payload_status": "downloaded_fna_ready",
                "analysis_unit_type": "mag_bin_candidate",
                "denominator_status": "archive_1428_reconcile_to_published_966_pending",
                "claim_scope": "MAG/proteome functional potential after QC and protein prediction",
                **stats,
            }
        )

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "external_dataset_id",
        "source_project_id",
        "source_group",
        "source_fasta_filename",
        "mag_id",
        "proteome_id",
        "local_fna_path",
        "local_faa_path",
        "protein_prediction_status",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "source_sample_ids",
        "source_archive_path",
        "archive_md5",
        "payload_status",
        "analysis_unit_type",
        "denominator_status",
        "claim_scope",
        "total_bp",
        "contigs",
        "n50_bp",
        "gc_pct",
    ]
    with MANIFEST_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    missing_classification = sum(1 for row in rows if not row["domain"])
    print(f"fasta_rows\t{len(rows)}")
    print(f"missing_classification\t{missing_classification}")
    print(f"wrote\t{MANIFEST_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
