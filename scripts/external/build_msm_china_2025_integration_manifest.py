#!/usr/bin/env python3
"""Join MSM MAG, proteome, taxonomy, and sample metadata into one manifest."""

from __future__ import annotations

import csv
from pathlib import Path


DATASET_DIR = Path("data/external/msm_china_2025")
MAG_MANIFEST = DATASET_DIR / "manifests" / "msm_china_2025_mag_manifest.resolved.tsv"
PROTEOME_MANIFEST = DATASET_DIR / "manifests" / "msm_china_2025_proteome_manifest.tsv"
NODE_NCBI = DATASET_DIR / "gigadb_wasabi" / "NODE_NCBI.csv"
OUT = DATASET_DIR / "manifests" / "msm_china_2025_functional_embedding_manifest.tsv"


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_node_ncbi() -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = {}
    with NODE_NCBI.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping.setdefault(row["NODE_id"], []).append(row)
    return mapping


def protein_qc_tier(protein_count: int) -> str:
    if protein_count < 100:
        return "qc_warning_lt100_proteins"
    if protein_count < 500:
        return "qc_warning_lt500_proteins"
    if protein_count < 1000:
        return "qc_review_lt1000_proteins"
    return "protein_count_plausible"


def main() -> int:
    mag_rows = {row["proteome_id"]: row for row in load_tsv(MAG_MANIFEST)}
    proteome_rows = {row["proteome_id"]: row for row in load_tsv(PROTEOME_MANIFEST)}
    sample_map = load_node_ncbi()

    rows: list[dict[str, str]] = []
    for proteome_id in sorted(mag_rows):
        mag = mag_rows[proteome_id]
        prot = proteome_rows.get(proteome_id, {})
        source_sample_ids = [
            sample.strip()
            for sample in mag["source_sample_ids"].split(",")
            if sample.strip()
        ]
        mapped_records = [record for sample in source_sample_ids for record in sample_map.get(sample, [])]
        biosamples = sorted({record.get("NCBI_BioSample", "") for record in mapped_records if record.get("NCBI_BioSample")})
        bioprojects = sorted({record.get("NCBI_BioProject", "") for record in mapped_records if record.get("NCBI_BioProject")})
        node_projects = sorted({record.get("NODE_project_id", "") for record in mapped_records if record.get("NODE_project_id")})
        protein_count = int(prot.get("protein_count") or 0)
        faa_ready = Path(prot.get("local_faa_path", "")).exists() and protein_count > 0
        rows.append(
            {
                "external_dataset_id": mag["external_dataset_id"],
                "source_project_id": mag["source_project_id"],
                "proteome_id": proteome_id,
                "mag_id": mag["mag_id"],
                "source_group": mag["source_group"],
                "source_fasta_filename": mag["source_fasta_filename"],
                "local_fna_path": mag["local_fna_path"],
                "local_faa_path": prot.get("local_faa_path", mag["local_faa_path"]),
                "local_ffn_path": prot.get("local_ffn_path", ""),
                "local_gff_path": prot.get("local_gff_path", ""),
                "payload_status": "fna_and_faa_ready" if faa_ready else "fna_ready_faa_missing_or_empty",
                "protein_prediction_status": prot.get("protein_prediction_status", "missing"),
                "protein_count": str(protein_count),
                "protein_qc_tier": protein_qc_tier(protein_count),
                "domain": mag["domain"],
                "phylum": mag["phylum"],
                "class": mag["class"],
                "order": mag["order"],
                "family": mag["family"],
                "genus": mag["genus"],
                "species": mag["species"],
                "source_sample_ids": mag["source_sample_ids"],
                "mapped_node_projects": ";".join(node_projects),
                "mapped_ncbi_bioprojects": ";".join(bioprojects),
                "mapped_ncbi_biosamples": ";".join(biosamples),
                "metadata_mapping_status": "mapped_to_ncbi_biosample" if biosamples else "source_sample_only",
                "total_bp": mag["total_bp"],
                "contigs": mag["contigs"],
                "n50_bp": mag["n50_bp"],
                "gc_pct": mag["gc_pct"],
                "analysis_unit_type": mag["analysis_unit_type"],
                "denominator_status": mag["denominator_status"],
                "claim_scope": mag["claim_scope"],
            }
        )

    fields = list(rows[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows\t{len(rows)}")
    print(f"fna_and_faa_ready\t{sum(row['payload_status'] == 'fna_and_faa_ready' for row in rows)}")
    print(f"mapped_to_ncbi_biosample\t{sum(row['metadata_mapping_status'] == 'mapped_to_ncbi_biosample' for row in rows)}")
    print(f"wrote\t{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
