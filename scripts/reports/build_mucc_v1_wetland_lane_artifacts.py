#!/usr/bin/env python3
"""Build MUCC v1 Old Woman Creek wetland lane artifacts.

This parser turns the staged MUCC v1 payloads into MethaNet-shaped manifests
and feature scaffolds while preserving claim boundaries. It does not assign
MRV risk tiers, flux claims, or source-transfer claims.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LANE_ID = "mucc_v1_owc_wetland"
SOURCE_LANE = "wetland_mucc_v1_owc"
SOURCE_DATASET = "MUCC database v1.0.0"
CONCEPT_DOI = "10.5281/zenodo.8194032"
RECORD_DOI = "10.5281/zenodo.8194033"
PAPER_DOI = "10.1128/msystems.00680-25"

MISSING_VALUES = {"", "NA", "N/A", "na", "nan", "None", "not_assigned"}
TRUE_VALUES = {"true", "1", "yes", "y"}

METHANE_TERMS = [
    "methan",
    "mcr",
    "mtr",
    "mta",
    "mtb",
    "mtt",
    "coenzyme m",
    "coenzyme b",
    "methyl-coenzyme",
    "formylmethanofuran",
    "methanofuran",
    "methanopterin",
]

SULFUR_TERMS = [
    "sulfur",
    "sulphur",
    "sulfate",
    "sulfite",
    "sulfide",
    "thiosulfate",
    "sulfurtransferase",
    "sulfotransferase",
    "dsr",
    "apr",
    "sat",
    "sox",
]

METHYL_SUBSTRATE_TERMS = [
    "methyl",
    "methanol",
    "methanediol",
    "trimethyl",
    "dimethyl",
    "monomethyl",
    "methylamine",
]

SUBSTRATE_TERMS = [
    "acetate",
    "formate",
    "hydrogenase",
    "ferredoxin",
    "dehydrogenase",
    "carbohydrate",
    "cellulose",
    "glycoside",
    "cazy",
    "carbon fixation",
    "one-carbon",
]


@dataclass
class MagAnnotationSummary:
    source_mag_id: str
    gene_annotation_rows: int = 0
    ko_assigned_rows: int = 0
    kegg_id_rows: int = 0
    cazy_rows: int = 0
    pfam_rows: int = 0
    peptidase_rows: int = 0
    vogdb_rows: int = 0
    heme_regulatory_motif_sum: int = 0
    methane_term_rows: int = 0
    sulfur_term_rows: int = 0
    methyl_substrate_term_rows: int = 0
    substrate_term_rows: int = 0
    taxonomy: str = ""
    bin_completeness: str = ""
    bin_contamination: str = ""
    source_bins: Counter[str] = field(default_factory=Counter)


@dataclass
class SourceDramSummary:
    source_mag_id: str
    source_dram_rows: int = 0
    ko_rows: int = 0
    cazy_rows: int = 0
    pfam_rows: int = 0
    peptidase_rows: int = 0
    vogdb_rows: int = 0
    methane_term_rows: int = 0
    sulfur_term_rows: int = 0
    methyl_substrate_term_rows: int = 0
    substrate_term_rows: int = 0


@dataclass
class GeneExpressionSummary:
    source_mag_id: str
    gene_expression_rows: int = 0
    expressed_gene_rows: int = 0
    nonzero_gene_sample_pairs: int = 0
    expression_sum: float = 0.0
    max_gene_expression: float = 0.0
    methane_expressed_gene_rows: int = 0
    sulfur_expressed_gene_rows: int = 0
    methyl_substrate_expressed_gene_rows: int = 0
    substrate_expressed_gene_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626"),
    )
    parser.add_argument("--skip-source-dram", action="store_true")
    parser.add_argument("--skip-gene-expression", action="store_true")
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def ensure_dirs(run_dir: Path) -> None:
    for subdir in [
        "manifests",
        "mag_fastas",
        "functional_features",
        "expression",
        "candidate_cards",
        "cohort_warehouse",
        "reports",
    ]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def clean(value: Any) -> str:
    return str(value or "").strip()


def present(value: Any) -> bool:
    return clean(value) not in MISSING_VALUES


def truthy(value: Any) -> bool:
    return clean(value).lower() in TRUE_VALUES


def text_has(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def safe_int(value: Any) -> int:
    try:
        if not present(value):
            return 0
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        if not present(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def split_taxonomy(taxonomy: str) -> dict[str, str]:
    ranks = {
        "domain": "",
        "phylum": "",
        "class": "",
        "order": "",
        "family": "",
        "genus": "",
        "species": "",
    }
    prefixes = {
        "d__": "domain",
        "p__": "phylum",
        "c__": "class",
        "o__": "order",
        "f__": "family",
        "g__": "genus",
        "s__": "species",
    }
    for part in taxonomy.split(";"):
        part = part.strip()
        for prefix, rank in prefixes.items():
            if part.startswith(prefix):
                ranks[rank] = part
                break
    return ranks


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_mag_expression_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {row["mag_id"]: row for row in rows if row.get("mag_id")}


def read_protein_catalog_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {row["source_mag_id"]: row for row in rows if row.get("source_mag_id")}


def build_mag_catalog(zip_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".fa.gz"):
                continue
            source_mag_id = Path(info.filename).name.removesuffix(".fa.gz")
            rows.append(
                {
                    "lane_id": LANE_ID,
                    "source_mag_id": source_mag_id,
                    "mag_id": source_mag_id,
                    "proteome_id": f"mucc_v1__{source_mag_id}",
                    "archive_member": info.filename,
                    "zip_uncompressed_bytes": info.file_size,
                    "zip_compressed_bytes": info.compress_size,
                    "zip_crc": f"{info.CRC:08x}",
                    "source_mag_fasta_status": "downloaded_validated_in_MAGs.zip",
                    "source_dataset_record_doi": RECORD_DOI,
                }
            )
    return sorted(rows, key=lambda row: row["source_mag_id"])


def extract_mag_fastas(zip_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".fa.gz"):
                continue
            source_mag_id = Path(info.filename).name.removesuffix(".fa.gz")
            target = output_dir / Path(info.filename).name
            if not target.exists() or target.stat().st_size == 0:
                with archive.open(info) as src, target.open("wb") as dst:
                    dst.write(src.read())
            paths[source_mag_id] = str(target)
    return paths


def row_text(row: dict[str, str]) -> str:
    cols = [
        "",
        "KO",
        "kegg_id",
        "kegg_hit",
        "peptidase_id",
        "peptidase_family",
        "peptidase_hit",
        "pfam_hits",
        "cazy_hits",
        "vogdb_description",
        "vogdb_categories",
    ]
    return " ".join(clean(row.get(col)) for col in cols)


def parse_gene_annotations(path: Path) -> tuple[
    dict[str, MagAnnotationSummary],
    dict[str, str],
    dict[str, tuple[str, bool, bool, bool, bool]],
]:
    summaries: dict[str, MagAnnotationSummary] = {}
    bin_to_mag: dict[str, str] = {}
    gene_to_mag_flags: dict[str, tuple[str, bool, bool, bool, bool]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_mag_id = clean(row.get("db_id")) or clean(row.get("fasta"))
            if not source_mag_id:
                continue
            summary = summaries.setdefault(source_mag_id, MagAnnotationSummary(source_mag_id))
            summary.gene_annotation_rows += 1
            bin_id = clean(row.get("bin_id"))
            if bin_id:
                summary.source_bins[bin_id] += 1
                bin_to_mag.setdefault(bin_id, source_mag_id)
            taxonomy = clean(row.get("bin_taxonomy"))
            if taxonomy and not summary.taxonomy:
                summary.taxonomy = taxonomy
            if present(row.get("bin_completeness")) and not summary.bin_completeness:
                summary.bin_completeness = clean(row.get("bin_completeness"))
            if present(row.get("bin_contamination")) and not summary.bin_contamination:
                summary.bin_contamination = clean(row.get("bin_contamination"))

            if present(row.get("KO")):
                summary.ko_assigned_rows += 1
            if present(row.get("kegg_id")):
                summary.kegg_id_rows += 1
            if present(row.get("cazy_hits")):
                summary.cazy_rows += 1
            if present(row.get("pfam_hits")):
                summary.pfam_rows += 1
            if present(row.get("peptidase_id")) or present(row.get("peptidase_family")):
                summary.peptidase_rows += 1
            if present(row.get("vogdb_description")) or present(row.get("vogdb_categories")):
                summary.vogdb_rows += 1
            summary.heme_regulatory_motif_sum += safe_int(row.get("heme_regulatory_motif_count"))

            text = row_text(row)
            methane = text_has(text, METHANE_TERMS)
            sulfur = text_has(text, SULFUR_TERMS)
            methyl = text_has(text, METHYL_SUBSTRATE_TERMS)
            substrate = text_has(text, SUBSTRATE_TERMS)
            summary.methane_term_rows += int(methane)
            summary.sulfur_term_rows += int(sulfur)
            summary.methyl_substrate_term_rows += int(methyl)
            summary.substrate_term_rows += int(substrate)

            gene_id = clean(row.get("")) or clean(row.get("gene_db_id"))
            if gene_id:
                gene_to_mag_flags[gene_id] = (source_mag_id, methane, sulfur, methyl, substrate)
    return summaries, bin_to_mag, gene_to_mag_flags


def parse_source_dram(
    path: Path,
    bin_to_mag: dict[str, str],
    catalog_ids: set[str],
) -> tuple[dict[str, SourceDramSummary], Counter[str]]:
    summaries: dict[str, SourceDramSummary] = {}
    unmapped_bins: Counter[str] = Counter()
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_bin = clean(row.get("fasta"))
            source_mag_id = source_bin if source_bin in catalog_ids else bin_to_mag.get(source_bin, "")
            if not source_mag_id:
                unmapped_bins[source_bin] += 1
                continue
            summary = summaries.setdefault(source_mag_id, SourceDramSummary(source_mag_id))
            summary.source_dram_rows += 1
            if present(row.get("kegg_id")):
                summary.ko_rows += 1
            if present(row.get("cazy_hits")):
                summary.cazy_rows += 1
            if present(row.get("pfam_hits")):
                summary.pfam_rows += 1
            if present(row.get("peptidase_id")) or present(row.get("peptidase_family")):
                summary.peptidase_rows += 1
            if present(row.get("vogdb_description")) or present(row.get("vogdb_categories")):
                summary.vogdb_rows += 1
            text = row_text(row)
            summary.methane_term_rows += int(text_has(text, METHANE_TERMS))
            summary.sulfur_term_rows += int(text_has(text, SULFUR_TERMS))
            summary.methyl_substrate_term_rows += int(text_has(text, METHYL_SUBSTRATE_TERMS))
            summary.substrate_term_rows += int(text_has(text, SUBSTRATE_TERMS))
    return summaries, unmapped_bins


def parse_gene_expression(
    path: Path,
    gene_to_mag_flags: dict[str, tuple[str, bool, bool, bool, bool]],
) -> tuple[dict[str, GeneExpressionSummary], dict[str, list[float]], list[str], int]:
    summaries: dict[str, GeneExpressionSummary] = {}
    mag_sample_sums: dict[str, list[float]] = {}
    unmapped_rows = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        sample_columns = header[1:]
        for row in reader:
            if not row:
                continue
            gene_id = clean(row[0])
            mapped = gene_to_mag_flags.get(gene_id)
            if not mapped:
                unmapped_rows += 1
                continue
            source_mag_id, methane, sulfur, methyl, substrate = mapped
            summary = summaries.setdefault(source_mag_id, GeneExpressionSummary(source_mag_id))
            summary.gene_expression_rows += 1
            sample_sums = mag_sample_sums.setdefault(source_mag_id, [0.0] * len(sample_columns))
            gene_sum = 0.0
            gene_nonzero_pairs = 0
            gene_max = 0.0
            for idx, value in enumerate(row[1:]):
                if value in {"", "0", "0.0", "0.00"}:
                    continue
                expr = safe_float(value)
                if expr <= 0.0:
                    continue
                gene_sum += expr
                gene_nonzero_pairs += 1
                if expr > gene_max:
                    gene_max = expr
                sample_sums[idx] += expr
            if gene_sum > 0.0:
                summary.expressed_gene_rows += 1
                summary.expression_sum += gene_sum
                summary.nonzero_gene_sample_pairs += gene_nonzero_pairs
                if gene_max > summary.max_gene_expression:
                    summary.max_gene_expression = gene_max
                summary.methane_expressed_gene_rows += int(methane)
                summary.sulfur_expressed_gene_rows += int(sulfur)
                summary.methyl_substrate_expressed_gene_rows += int(methyl)
                summary.substrate_expressed_gene_rows += int(substrate)
    return summaries, mag_sample_sums, sample_columns, unmapped_rows


def annotation_summary_rows(
    catalog_ids: list[str],
    summaries: dict[str, MagAnnotationSummary],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_mag_id in catalog_ids:
        summary = summaries.get(source_mag_id, MagAnnotationSummary(source_mag_id))
        tax = split_taxonomy(summary.taxonomy)
        rows.append(
            {
                "lane_id": LANE_ID,
                "source_mag_id": source_mag_id,
                "mag_id": source_mag_id,
                "proteome_id": f"mucc_v1__{source_mag_id}",
                "gene_annotation_rows": summary.gene_annotation_rows,
                "ko_assigned_rows": summary.ko_assigned_rows,
                "kegg_id_rows": summary.kegg_id_rows,
                "cazy_rows": summary.cazy_rows,
                "pfam_rows": summary.pfam_rows,
                "peptidase_rows": summary.peptidase_rows,
                "vogdb_rows": summary.vogdb_rows,
                "heme_regulatory_motif_sum": summary.heme_regulatory_motif_sum,
                "methane_term_rows": summary.methane_term_rows,
                "sulfur_term_rows": summary.sulfur_term_rows,
                "methyl_substrate_term_rows": summary.methyl_substrate_term_rows,
                "substrate_term_rows": summary.substrate_term_rows,
                "bin_taxonomy": summary.taxonomy,
                **tax,
                "bin_completeness": summary.bin_completeness,
                "bin_contamination": summary.bin_contamination,
                "source_bin_count": len(summary.source_bins),
                "source_annotation_status": (
                    "source_gene_annotation_available"
                    if summary.gene_annotation_rows
                    else "not_observed_in_gene_annotation_payload"
                ),
                "allowed_claim_wording": (
                    "Source DRAM-style annotation rows are present for this MAG; "
                    "term counts are feature scaffolds, not final mechanism or MRV claims."
                    if summary.gene_annotation_rows
                    else "This MAG is present in MAGs.zip but absent from the staged gene annotation table."
                ),
            }
        )
    return rows


def source_dram_rows(
    catalog_ids: list[str],
    summaries: dict[str, SourceDramSummary],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_mag_id in catalog_ids:
        summary = summaries.get(source_mag_id, SourceDramSummary(source_mag_id))
        rows.append(
            {
                "lane_id": LANE_ID,
                "source_mag_id": source_mag_id,
                "mag_id": source_mag_id,
                "proteome_id": f"mucc_v1__{source_mag_id}",
                "source_dram_rows": summary.source_dram_rows,
                "ko_rows": summary.ko_rows,
                "cazy_rows": summary.cazy_rows,
                "pfam_rows": summary.pfam_rows,
                "peptidase_rows": summary.peptidase_rows,
                "vogdb_rows": summary.vogdb_rows,
                "methane_term_rows": summary.methane_term_rows,
                "sulfur_term_rows": summary.sulfur_term_rows,
                "methyl_substrate_term_rows": summary.methyl_substrate_term_rows,
                "substrate_term_rows": summary.substrate_term_rows,
                "source_dram_status": (
                    "mapped_source_dram_rows_available"
                    if summary.source_dram_rows
                    else "not_mapped_from_full_source_dram_payload"
                ),
                "claim_scope": "source DRAM feature scaffold only; no final mechanism or MRV claim",
            }
        )
    return rows


def gene_expression_summary_rows(
    catalog_ids: list[str],
    summaries: dict[str, GeneExpressionSummary],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_mag_id in catalog_ids:
        summary = summaries.get(source_mag_id, GeneExpressionSummary(source_mag_id))
        rows.append(
            {
                "lane_id": LANE_ID,
                "source_mag_id": source_mag_id,
                "mag_id": source_mag_id,
                "proteome_id": f"mucc_v1__{source_mag_id}",
                "gene_expression_rows": summary.gene_expression_rows,
                "expressed_gene_rows": summary.expressed_gene_rows,
                "nonzero_gene_sample_pairs": summary.nonzero_gene_sample_pairs,
                "expression_sum": f"{summary.expression_sum:.12g}",
                "max_gene_expression": f"{summary.max_gene_expression:.12g}",
                "methane_expressed_gene_rows": summary.methane_expressed_gene_rows,
                "sulfur_expressed_gene_rows": summary.sulfur_expressed_gene_rows,
                "methyl_substrate_expressed_gene_rows": summary.methyl_substrate_expressed_gene_rows,
                "substrate_expressed_gene_rows": summary.substrate_expressed_gene_rows,
                "gene_expression_support_status": (
                    "gene_expression_rows_available"
                    if summary.gene_expression_rows
                    else "not_observed_in_gene_expression_payload"
                ),
                "allowed_claim_wording": (
                    "Gene-level processed expression supports source-table transcriptional evidence; "
                    "this is not a process-rate, flux, or MRV-risk claim."
                    if summary.gene_expression_rows
                    else "No gene-level expression row was mapped for this MAG in the staged expression table."
                ),
            }
        )
    return rows


def write_mag_sample_gene_expression(
    path: Path,
    mag_sample_sums: dict[str, list[float]],
    sample_columns: list[str],
) -> int:
    row_count = 0
    with gzip.open(path, "wt", newline="") as handle:
        fieldnames = [
            "lane_id",
            "source_mag_id",
            "mag_id",
            "proteome_id",
            "sample_column",
            "gene_expression_sum",
            "gene_expression_support_status",
            "claim_scope",
        ]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for source_mag_id in sorted(mag_sample_sums):
            for sample, value in zip(sample_columns, mag_sample_sums[source_mag_id], strict=True):
                writer.writerow(
                    {
                        "lane_id": LANE_ID,
                        "source_mag_id": source_mag_id,
                        "mag_id": source_mag_id,
                        "proteome_id": f"mucc_v1__{source_mag_id}",
                        "sample_column": sample,
                        "gene_expression_sum": f"{value:.12g}",
                        "gene_expression_support_status": (
                            "processed_gene_expression_sum_by_mag_sample"
                            if value > 0
                            else "zero_or_absent_processed_gene_expression"
                        ),
                        "claim_scope": "processed gene expression support only; no flux or risk tier",
                    }
                )
                row_count += 1
    return row_count


def build_lane_manifest(
    catalog_rows: list[dict[str, Any]],
    local_fna_paths: dict[str, str],
    expression_mag_summary: dict[str, dict[str, str]],
    protein_catalog_summary: dict[str, dict[str, str]],
    annotation_summaries: dict[str, MagAnnotationSummary],
    source_dram_summaries: dict[str, SourceDramSummary],
    gene_expression_summaries: dict[str, GeneExpressionSummary],
) -> list[dict[str, Any]]:
    expression_subset_count = len(expression_mag_summary)
    catalog_count = len(catalog_rows)
    rows: list[dict[str, Any]] = []
    for row in catalog_rows:
        source_mag_id = row["source_mag_id"]
        local_fna_path = local_fna_paths.get(source_mag_id, "")
        expr = expression_mag_summary.get(source_mag_id, {})
        ann = annotation_summaries.get(source_mag_id, MagAnnotationSummary(source_mag_id))
        source_dram = source_dram_summaries.get(source_mag_id, SourceDramSummary(source_mag_id))
        gene_expr = gene_expression_summaries.get(source_mag_id, GeneExpressionSummary(source_mag_id))
        protein = protein_catalog_summary.get(source_mag_id, {})
        protein_count = safe_int(protein.get("protein_count"))
        protein_status = (
            clean(protein.get("protein_catalog_status"))
            or ("source_protein_records_available" if protein_count else "missing_direct_source_protein_records")
        )
        esm2_status = (
            "esm2_input_ready_dry_run_validated"
            if protein_count
            else "blocked_missing_source_protein_records"
        )
        blocking_gap = (
            "Production ESM2/gLM2 embeddings, MethaNet curated functional run, environmental joins, uncertainty, and flux/process validation pending"
            if protein_count
            else "Source protein mapping gap plus production ESM2/gLM2 embeddings, MethaNet curated functional run, environmental joins, uncertainty, and flux/process validation pending"
        )
        next_action = (
            "run production ESM2/gLM2 compute and curated MethaNet feature warehouse for this protein-supported MAG"
            if protein_count
            else "reconcile or regenerate source protein records before adding this MAG to ESM2/gLM2 compute"
        )
        tax = split_taxonomy(ann.taxonomy)
        rows.append(
            {
                "lane_id": LANE_ID,
                "lane": SOURCE_LANE,
                "source_dataset": SOURCE_DATASET,
                "source_dataset_concept_doi": CONCEPT_DOI,
                "source_dataset_record_doi": RECORD_DOI,
                "source_paper_doi": PAPER_DOI,
                "source_mag_id": source_mag_id,
                "mag_id": source_mag_id,
                "proteome_id": row["proteome_id"],
                "source": LANE_ID,
                "analysis_unit_type": "mag_bin",
                "source_project": "Old Woman Creek wetland soil multi-omics",
                "ecosystem": "freshwater_wetland",
                "ecosystem_detail": "temperate_freshwater_wetland_soil",
                "source_group": SOURCE_LANE,
                "source_sample_ids": "OWC_metatranscriptome_sample_columns_pending_environmental_mapping",
                "mag_fasta": local_fna_path,
                "proteome_faa": (
                    f"results/functional_metagenomics/mucc_v1_owc_wetland_20260626/proteomes/mucc_v1__{source_mag_id}.faa"
                    if protein_count
                    else ""
                ),
                "local_fna_path": local_fna_path,
                "local_faa_path": (
                    f"results/functional_metagenomics/mucc_v1_owc_wetland_20260626/proteomes/mucc_v1__{source_mag_id}.faa"
                    if protein_count
                    else ""
                ),
                "local_ffn_path": "",
                "local_gff_path": "",
                "match_status": "matched" if protein_count and local_fna_path else "fna_only",
                "comparability_status": (
                    "ready_for_esm2_and_functional_annotation_pending_gff_context"
                    if protein_count and local_fna_path
                    else "blocked_missing_source_protein_records"
                ),
                "published_quality_denominator": 2502,
                "local_manifest_denominator": catalog_count,
                "processed_mag_expression_subset_denominator": expression_subset_count,
                "source_mag_fasta_status": "downloaded_validated_in_MAGs.zip",
                "source_protein_catalog_status": "downloaded_validated_OWC_HQMQ_DB_genes.faa.gz",
                "source_dram_annotation_status": (
                    "mapped_source_dram_rows_available"
                    if source_dram.source_dram_rows
                    else "full_source_dram_payload_downloaded_not_mapped_for_this_mag"
                ),
                "gene_annotation_status": (
                    "gene_annotation_rows_available"
                    if ann.gene_annotation_rows
                    else "not_observed_in_gene_annotation_payload"
                ),
                "mag_expression_support": "true" if source_mag_id in expression_mag_summary else "false",
                "mag_expression_n_samples": expr.get("n_samples", "0"),
                "mag_expression_nonzero_samples": expr.get("nonzero_samples", "0"),
                "mag_expression_occupancy_fraction": expr.get("occupancy_fraction", "0"),
                "gene_expression_support": "true" if gene_expr.gene_expression_rows else "false",
                "gene_expression_rows": gene_expr.gene_expression_rows,
                "taxonomy_status": "source_taxonomy_available" if ann.taxonomy else "pending_taxonomy_mapping",
                "domain": tax["domain"],
                "phylum": tax["phylum"],
                "class": tax["class"],
                "order": tax["order"],
                "family": tax["family"],
                "genus": tax["genus"],
                "species": tax["species"],
                "bin_completeness": ann.bin_completeness,
                "bin_contamination": ann.bin_contamination,
                "methane_term_rows": ann.methane_term_rows + source_dram.methane_term_rows,
                "sulfur_term_rows": ann.sulfur_term_rows + source_dram.sulfur_term_rows,
                "methyl_substrate_term_rows": ann.methyl_substrate_term_rows
                + source_dram.methyl_substrate_term_rows,
                "substrate_term_rows": ann.substrate_term_rows + source_dram.substrate_term_rows,
                "esm2_status": esm2_status,
                "glm2_status": "pending_context_embedding",
                "functional_run_include": "true",
                "esm2_include": "true" if protein_count else "false",
                "glm2_include": "false",
                "claim_scope": "MAG/proteome wetland reference lane scaffold; no final MRV tiers, flux, crediting, or transfer claims",
                "blocking_gap": blocking_gap,
                "next_action": next_action,
                "protein_count": protein_count,
                "protein_catalog_status": protein_status,
            }
        )
    return rows


def source_ready_rows(lane_manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "external_dataset_id",
        "source_project_id",
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "source_group",
        "source_sample_ids",
        "local_fna_path",
        "proteome_faa",
        "local_faa_path",
        "local_ffn_path",
        "local_gff_path",
        "payload_status",
        "protein_prediction_status",
        "protein_count",
        "match_status",
        "functional_run_include",
        "esm2_include",
        "glm2_include",
        "analysis_unit_type",
        "claim_scope",
        "comparability_status",
        "denominator_status",
        "metadata_mapping_status",
        "download_state",
        "gap_reason",
        "recommended_action",
    ]
    rows: list[dict[str, Any]] = []
    for row in lane_manifest_rows:
        protein_count = safe_int(row.get("protein_count"))
        local_fna_path = clean(row.get("local_fna_path"))
        rows.append(
            {
                "external_dataset_id": LANE_ID,
                "source_project_id": SOURCE_DATASET,
                "proteome_id": row["proteome_id"],
                "mag_id": row["mag_id"],
                "source": row.get("source", LANE_ID),
                "ecosystem": row.get("ecosystem", "freshwater_wetland"),
                "domain": row.get("domain", ""),
                "source_group": row.get("source_group", SOURCE_LANE),
                "source_sample_ids": row.get("source_sample_ids", ""),
                "local_fna_path": local_fna_path,
                "proteome_faa": row.get("proteome_faa", ""),
                "local_faa_path": row.get("local_faa_path", ""),
                "local_ffn_path": "",
                "local_gff_path": "",
                "payload_status": (
                    "fna_and_source_faa_ready"
                    if local_fna_path and protein_count
                    else "fna_ready_source_faa_missing"
                ),
                "protein_prediction_status": (
                    "source_protein_records_available"
                    if protein_count
                    else "blocked_missing_source_protein_records"
                ),
                "protein_count": protein_count,
                "match_status": row.get("match_status", ""),
                "functional_run_include": row.get("functional_run_include", "false"),
                "esm2_include": row.get("esm2_include", "false"),
                "glm2_include": "false",
                "analysis_unit_type": row.get("analysis_unit_type", "mag_bin"),
                "claim_scope": row.get("claim_scope", ""),
                "comparability_status": row.get("comparability_status", ""),
                "denominator_status": "local_manifest_denominator_includes_all_MAGs_zip_entries",
                "metadata_mapping_status": "sample_column_environmental_metadata_pending",
                "download_state": "downloaded_validated_md5",
                "gap_reason": "" if protein_count else "missing_direct_source_protein_records",
                "recommended_action": row.get("next_action", ""),
            }
        )
    return [{field: row.get(field, "") for field in fields} for row in rows]


def build_strategic_candidate_cards(
    lane_manifest_rows: list[dict[str, Any]],
    expression_mag_summary: dict[str, dict[str, str]],
    gene_expression_summaries: dict[str, GeneExpressionSummary],
    max_per_set: int = 10,
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for row in lane_manifest_rows:
        source_mag_id = str(row["source_mag_id"])
        expr = expression_mag_summary.get(source_mag_id, {})
        gene_expr = gene_expression_summaries.get(source_mag_id, GeneExpressionSummary(source_mag_id))
        methane = safe_int(row.get("methane_term_rows"))
        sulfur = safe_int(row.get("sulfur_term_rows"))
        methyl = safe_int(row.get("methyl_substrate_term_rows"))
        substrate = safe_int(row.get("substrate_term_rows"))
        expression_sum = safe_float(expr.get("expression_sum"))
        gene_expression_sum = gene_expr.expression_sum
        occupancy = safe_float(row.get("mag_expression_occupancy_fraction"))
        completeness = safe_float(row.get("bin_completeness"))
        contamination = safe_float(row.get("bin_contamination"))
        protein_count = safe_int(row.get("protein_count"))
        joined.append(
            {
                **row,
                "_expression_sum": expression_sum,
                "_gene_expression_sum": gene_expression_sum,
                "_occupancy": occupancy,
                "_completeness": completeness,
                "_contamination": contamination,
                "_protein_count": protein_count,
                "_methane": methane,
                "_sulfur": sulfur,
                "_methyl": methyl,
                "_substrate": substrate,
                "_gene_expression_rows": gene_expr.gene_expression_rows,
                "_expressed_gene_rows": gene_expr.expressed_gene_rows,
            }
        )

    def top_rows(label: str, rows: list[dict[str, Any]], key) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rank, row in enumerate(sorted(rows, key=key, reverse=True)[:max_per_set], start=1):
            out.append({**row, "_candidate_set": label, "_rank": rank})
        return out

    cards: list[dict[str, Any]] = []
    cards.extend(
        top_rows(
            "archaeal_methane_review",
            [row for row in joined if str(row.get("domain")) == "d__Archaea" and row["_methane"] > 0],
            lambda row: (row["_methane"], row["_gene_expression_sum"], row["_completeness"], -row["_contamination"]),
        )
    )
    cards.extend(
        top_rows(
            "expressed_methyl_substrate_review",
            [row for row in joined if row["_methyl"] > 0 and row["_gene_expression_rows"] > 0],
            lambda row: (row["_methyl"], row["_gene_expression_sum"], row["_occupancy"], row["_completeness"]),
        )
    )
    cards.extend(
        top_rows(
            "sulfur_redox_context_review",
            [row for row in joined if row["_sulfur"] > 0 and row["_gene_expression_rows"] > 0],
            lambda row: (row["_sulfur"], row["_gene_expression_sum"], row["_substrate"], row["_completeness"]),
        )
    )
    cards.extend(
        top_rows(
            "high_expression_wetland_review",
            [row for row in joined if row["_gene_expression_rows"] > 0],
            lambda row: (row["_gene_expression_sum"], row["_occupancy"], row["_methane"] + row["_methyl"]),
        )
    )

    fields = [
        "card_id",
        "candidate_set",
        "set_rank",
        "lane_id",
        "proteome_id",
        "mag_id",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "bin_completeness",
        "bin_contamination",
        "protein_count",
        "mag_expression_occupancy_fraction",
        "mag_expression_sum",
        "gene_expression_rows",
        "expressed_gene_rows",
        "gene_expression_sum",
        "methane_term_rows",
        "sulfur_term_rows",
        "methyl_substrate_term_rows",
        "substrate_term_rows",
        "review_tier",
        "allowed_claim_wording",
        "blocking_gap",
        "next_validation_action",
    ]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in cards:
        key = (str(row["proteome_id"]), str(row["_candidate_set"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "card_id": f"mucc_v1_{row['_candidate_set']}_{row['_rank']:02d}",
                "candidate_set": row["_candidate_set"],
                "set_rank": row["_rank"],
                "lane_id": row["lane_id"],
                "proteome_id": row["proteome_id"],
                "mag_id": row["mag_id"],
                "domain": row.get("domain", ""),
                "phylum": row.get("phylum", ""),
                "class": row.get("class", ""),
                "order": row.get("order", ""),
                "family": row.get("family", ""),
                "genus": row.get("genus", ""),
                "bin_completeness": row.get("bin_completeness", ""),
                "bin_contamination": row.get("bin_contamination", ""),
                "protein_count": row.get("protein_count", ""),
                "mag_expression_occupancy_fraction": row.get("mag_expression_occupancy_fraction", ""),
                "mag_expression_sum": f"{row['_expression_sum']:.12g}",
                "gene_expression_rows": row["_gene_expression_rows"],
                "expressed_gene_rows": row["_expressed_gene_rows"],
                "gene_expression_sum": f"{row['_gene_expression_sum']:.12g}",
                "methane_term_rows": row["_methane"],
                "sulfur_term_rows": row["_sulfur"],
                "methyl_substrate_term_rows": row["_methyl"],
                "substrate_term_rows": row["_substrate"],
                "review_tier": "source-scaffold review",
                "allowed_claim_wording": (
                    "Source-derived annotation and processed expression make this MAG a review candidate; "
                    "this is not a final mechanism, process-rate, flux, risk-tier, crediting, or transfer claim."
                ),
                "blocking_gap": (
                    "curated MethaNet functional calls, completed ESM2/gLM2 embeddings, environmental joins, "
                    "uncertainty propagation, and flux/process validation pending"
                ),
                "next_validation_action": (
                    "inspect source annotation terms, compare against curated MethaNet functional runs and "
                    "embedding neighbors, then promote or reject as a mechanism card"
                ),
            }
        )
    return [{field: row.get(field, "") for field in fields} for row in out]


def update_download_status(repo_root: Path, run_dir: Path) -> None:
    queue_path = run_dir / "downloads/mucc_v1_payload_download_queue.tsv"
    status_rows: list[dict[str, Any]] = []
    if not queue_path.exists():
        return
    with queue_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            target = Path(row["local_target"])
            target = target if target.is_absolute() else repo_root / target
            expected = row["checksum"].replace("md5:", "")
            actual = ""
            local_status = "missing"
            if target.exists() and target.with_suffix(target.suffix + ".md5").exists():
                actual = target.with_suffix(target.suffix + ".md5").read_text().split()[0]
            elif target.exists():
                actual = "present_not_rehashed_here"
            if actual == expected:
                local_status = "downloaded_validated_md5"
            elif target.exists():
                local_status = "downloaded_needs_md5_recheck"
            status_rows.append(
                {
                    **row,
                    "local_status_observed": local_status,
                    "observed_md5": actual,
                    "observed_size_bytes": target.stat().st_size if target.exists() else "",
                }
            )
    fields = list(status_rows[0]) if status_rows else []
    if status_rows:
        write_tsv(run_dir / "downloads/mucc_v1_payload_download_status.tsv", status_rows, fields)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    staging = run_dir / "staging"
    ensure_dirs(run_dir)

    mag_zip = staging / "MAGs.zip"
    catalog_rows = build_mag_catalog(mag_zip)
    local_fna_paths = extract_mag_fastas(mag_zip, run_dir / "mag_fastas")
    catalog_ids = [row["source_mag_id"] for row in catalog_rows]
    expression_mag_summary = read_mag_expression_summary(
        run_dir / "expression/feature_mucc_v1_expression_mag_summary.tsv"
    )
    protein_catalog_summary = read_protein_catalog_summary(
        run_dir / "manifests/mucc_v1_protein_catalog_summary.tsv"
    )

    annotation_summaries, bin_to_mag, gene_to_mag_flags = parse_gene_annotations(
        staging / "owc_metat_table_mags_genes_annotations.csv"
    )

    source_dram_summaries: dict[str, SourceDramSummary] = {}
    unmapped_source_bins: Counter[str] = Counter()
    if not args.skip_source_dram:
        source_dram_summaries, unmapped_source_bins = parse_source_dram(
            staging / "OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz",
            bin_to_mag,
            set(catalog_ids),
        )

    gene_expression_summaries: dict[str, GeneExpressionSummary] = {}
    mag_sample_sums: dict[str, list[float]] = {}
    sample_columns: list[str] = []
    unmapped_expression_rows = 0
    if not args.skip_gene_expression:
        gene_expression_summaries, mag_sample_sums, sample_columns, unmapped_expression_rows = (
            parse_gene_expression(staging / "owc_metat_table_mags_genes.csv", gene_to_mag_flags)
        )

    write_tsv(
        run_dir / "manifests/mucc_v1_mag_catalog_full.tsv",
        catalog_rows,
        list(catalog_rows[0]),
    )

    crosswalk_rows = [
        {
            "lane_id": LANE_ID,
            "source_bin_id": source_bin,
            "source_mag_id": source_mag_id,
            "mag_id": source_mag_id,
            "proteome_id": f"mucc_v1__{source_mag_id}",
            "crosswalk_source": "owc_metat_table_mags_genes_annotations.csv",
        }
        for source_bin, source_mag_id in sorted(bin_to_mag.items())
    ]
    write_tsv(
        run_dir / "manifests/mucc_v1_source_bin_crosswalk.tsv",
        crosswalk_rows,
        list(crosswalk_rows[0]) if crosswalk_rows else ["lane_id"],
    )

    annotation_rows = annotation_summary_rows(catalog_ids, annotation_summaries)
    write_tsv(
        run_dir / "functional_features/feature_mucc_v1_gene_annotation_mag_summary.tsv",
        annotation_rows,
        list(annotation_rows[0]),
    )

    source_dram_summary_rows = source_dram_rows(catalog_ids, source_dram_summaries)
    write_tsv(
        run_dir / "functional_features/feature_mucc_v1_source_dram_mag_summary.tsv",
        source_dram_summary_rows,
        list(source_dram_summary_rows[0]),
    )

    gene_expr_rows = gene_expression_summary_rows(catalog_ids, gene_expression_summaries)
    write_tsv(
        run_dir / "expression/feature_mucc_v1_gene_expression_mag_summary.tsv",
        gene_expr_rows,
        list(gene_expr_rows[0]),
    )

    gene_expr_fact_rows = 0
    if mag_sample_sums and sample_columns:
        gene_expr_fact_rows = write_mag_sample_gene_expression(
            run_dir / "expression/fact_mucc_v1_gene_expression_mag_sample.tsv.gz",
            mag_sample_sums,
            sample_columns,
        )

    lane_manifest_rows = build_lane_manifest(
        catalog_rows,
        local_fna_paths,
        expression_mag_summary,
        protein_catalog_summary,
        annotation_summaries,
        source_dram_summaries,
        gene_expression_summaries,
    )
    write_tsv(
        run_dir / "manifests/mucc_v1_lane_manifest.tsv",
        lane_manifest_rows,
        list(lane_manifest_rows[0]),
    )
    ready_rows = source_ready_rows(lane_manifest_rows)
    write_tsv(
        run_dir / "manifests/mucc_v1_source_ready_manifest.tsv",
        ready_rows,
        list(ready_rows[0]),
    )

    strategic_cards = build_strategic_candidate_cards(
        lane_manifest_rows,
        expression_mag_summary,
        gene_expression_summaries,
    )
    write_tsv(
        run_dir / "candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv",
        strategic_cards,
        list(strategic_cards[0]),
    )

    unmapped_rows = [
        {
            "source_bin_id": source_bin,
            "source_dram_rows_unmapped": count,
            "status": "not_mapped_to_owc_mag_id_from_gene_annotation_crosswalk",
        }
        for source_bin, count in unmapped_source_bins.most_common()
    ]
    if unmapped_rows:
        write_tsv(
            run_dir / "functional_features/source_dram_unmapped_bins.tsv",
            unmapped_rows,
            list(unmapped_rows[0]),
        )

    update_download_status(repo_root, run_dir)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "mag_catalog_rows": len(catalog_rows),
        "mag_expression_subset_rows": len(expression_mag_summary),
        "gene_annotation_mags": sum(1 for row in annotation_rows if row["gene_annotation_rows"]),
        "source_dram_mapped_mags": sum(
            1 for row in source_dram_summary_rows if row["source_dram_rows"]
        ),
        "source_dram_unmapped_bins": len(unmapped_source_bins),
        "gene_expression_mags": sum(1 for row in gene_expr_rows if row["gene_expression_rows"]),
        "gene_expression_mag_sample_rows": gene_expr_fact_rows,
        "gene_expression_unmapped_rows": unmapped_expression_rows,
        "strategic_candidate_cards": len(strategic_cards),
        "sample_columns": len(sample_columns),
        "claim_boundary": (
            "Feature tables are source-derived scaffolds. They do not assign final MRV scores, "
            "A-E tiers, flux claims, carbon-crediting claims, or source-independent transfer."
        ),
    }
    summary_path = run_dir / "reports/mucc_v1_payload_parse_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
