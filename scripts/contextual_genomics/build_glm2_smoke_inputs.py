#!/usr/bin/env python3
"""Build a small MethaNet gLM2 smoke-test payload from FNA/GFF/FAA inputs.

The script creates a dated, reproducible contextual-genomics smoke directory
without launching inference. It keeps MethaNet's MAG/proteome grain explicit:
`proteome_id` is carried through every manifest, window, and span-map record.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_POC_MANIFEST = Path(
    "results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/"
    "poc_662_functional_mag_manifest.mag_bin_only.tsv"
)
DEFAULT_MSM_MANIFEST = Path(
    "results/functional_metagenomics/msm_china_2025_20260615/manifests/"
    "msm_china_2025_functional_mag_manifest.tsv"
)
DEFAULT_BRIDGE_TABLE = Path(
    "results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/"
    "artifacts/bridge_top_candidates.tsv"
)
DEFAULT_RESULTS_DIR = Path("results/contextual_genomics/glm2_smoke_20260615_090023")
FUNCTIONAL_ROOTS = [
    Path("results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean/per_mag"),
    Path("results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag"),
    Path("results/functional_metagenomics/fgx_662_apollo3_20260614/per_mag"),
]
ATTR_RE = re.compile(r"([^=;]+)=([^;]+)")
TRUE_VALUES = {"true", "1", "yes", "y"}


@dataclass(frozen=True)
class CdsFeature:
    contig_id: str
    start: int
    end: int
    strand: str
    gene_id: str
    protein_id: str
    attrs: dict[str, str]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def truthy(value: str | None) -> bool:
    return str(value or "").lower() in TRUE_VALUES


def fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def parse_attrs(value: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in ATTR_RE.finditer(value):
        attrs[match.group(1)] = match.group(2)
    return attrs


def parse_gff(path: Path) -> list[CdsFeature]:
    features: list[CdsFeature] = []
    with open_text(path) as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2].lower() != "cds":
                continue
            attrs = parse_attrs(parts[8])
            gene_id = (
                attrs.get("ID")
                or attrs.get("locus_tag")
                or attrs.get("Name")
                or attrs.get("gene")
                or f"{parts[0]}:{parts[3]}-{parts[4]}:{parts[6]}"
            )
            protein_id = (
                attrs.get("protein_id")
                or attrs.get("Name")
                or attrs.get("ID")
                or attrs.get("locus_tag")
                or gene_id
            )
            features.append(
                CdsFeature(
                    contig_id=parts[0],
                    start=int(parts[3]),
                    end=int(parts[4]),
                    strand=parts[6] if parts[6] in {"+", "-"} else "+",
                    gene_id=gene_id,
                    protein_id=protein_id,
                    attrs=attrs,
                )
            )
    return features


def latest_complete_attempt(repo_root: Path, proteome_id: str) -> Path | None:
    attempts: list[Path] = []
    for root in FUNCTIONAL_ROOTS:
        base = repo_root / root / proteome_id
        if base.exists():
            attempts.extend(path for path in base.iterdir() if path.is_dir())
    complete = [path for path in attempts if (path / "COMPLETE").exists()]
    candidates = complete or attempts
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def first_existing(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path and path.exists() and path.stat().st_size > 0:
            return path
    return None


def resolve_poc_input_paths(repo_root: Path, row: dict[str, str]) -> dict[str, str]:
    proteome_id = row["proteome_id"]
    mag_id = row.get("mag_id", "")
    attempt = latest_complete_attempt(repo_root, proteome_id)
    genes = attempt / "genes" if attempt else None
    stem = Path(row.get("mag_fasta_basename", mag_id)).name
    for suffix in [".gz", ".fasta", ".fa", ".fna"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    fna = first_existing(
        [
            repo_path(repo_root, row.get("mag_fasta")),
            attempt / "staged_fasta" / f"{stem}.fasta" if attempt else None,
        ]
    )
    faa = first_existing(
        [
            genes / f"{stem}.faa" if genes else None,
            genes / f"{mag_id}.faa" if genes else None,
            repo_path(repo_root, row.get("proteome_faa")),
        ]
    )
    gff = first_existing(
        [
            genes / f"{stem}.gff" if genes else None,
            genes / f"{mag_id}.gff" if genes else None,
        ]
    )
    return {
        "source_fna": str(fna or ""),
        "source_faa": str(faa or ""),
        "source_gff": str(gff or ""),
        "source_attempt_dir": str(attempt or ""),
    }


def resolve_msm_input_paths(repo_root: Path, row: dict[str, str]) -> dict[str, str]:
    return {
        "source_fna": str(repo_path(repo_root, row.get("mag_fasta")) or ""),
        "source_faa": str(repo_path(repo_root, row.get("proteome_faa")) or ""),
        "source_gff": str(repo_path(repo_root, row.get("local_gff_path")) or ""),
        "source_attempt_dir": "",
    }


def resolve_source_lane_input_paths(repo_root: Path, row: dict[str, str]) -> dict[str, str]:
    return {
        "source_fna": str(repo_path(repo_root, row.get("mag_fasta") or row.get("local_fna_path")) or ""),
        "source_faa": str(repo_path(repo_root, row.get("proteome_faa") or row.get("local_faa_path")) or ""),
        "source_gff": str(repo_path(repo_root, row.get("local_gff_path")) or ""),
        "source_attempt_dir": "",
    }


def has_usable_inputs(row: dict[str, str]) -> bool:
    return all(
        bool(row.get(field)) and Path(row[field]).is_file()
        for field in ["source_fna", "source_faa", "source_gff"]
    )


def bridge_rank_by_sample(bridge_table: Path) -> dict[str, dict[str, str]]:
    if not bridge_table.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in read_tsv(bridge_table):
        sample = row.get("sample", "")
        if sample:
            out[sample] = row
    return out


def select_smoke_mags(
    repo_root: Path,
    poc_manifest: Path,
    msm_manifest: Path,
    bridge_table: Path,
) -> list[dict[str, str]]:
    bridge = bridge_rank_by_sample(bridge_table)
    poc_rows = read_tsv(poc_manifest)
    poc_by_id = {row["proteome_id"]: row for row in poc_rows}

    selected: list[dict[str, str]] = []

    def add_poc(proteome_id: str, smoke_group: str) -> None:
        row = dict(poc_by_id[proteome_id])
        row.update(resolve_poc_input_paths(repo_root, row))
        if not has_usable_inputs(row):
            return
        bridge_row = bridge.get(proteome_id, {})
        row["smoke_group"] = smoke_group
        row["bridge_rank"] = bridge_row.get("rank", "")
        row["bridge_score"] = bridge_row.get("bridging_score", "")
        selected.append(row)

    for row in sorted(
        bridge.values(), key=lambda item: int(float(item.get("rank") or 999999))
    ):
        pid = row["sample"]
        if pid in poc_by_id and row.get("source") == "rumen":
            add_poc(pid, "rumen_bridge_mag_bin")
        if sum(1 for item in selected if item["smoke_group"] == "rumen_bridge_mag_bin") >= 3:
            break

    for pid in ["mucc__GCA_002495465.1_ASM249546v1_genomic", "mucc__3300001784_29"]:
        if pid in poc_by_id:
            add_poc(pid, "mucc_wetland_mag_bin")

    msm_rows = []
    for row in read_tsv(msm_manifest):
        row = dict(row)
        row.update(resolve_msm_input_paths(repo_root, row))
        if not has_usable_inputs(row):
            continue
        row["sample"] = row["proteome_id"]
        row["mag_fasta_basename"] = Path(row.get("mag_fasta", "")).name
        row["n_proteins_used"] = row.get("protein_count", "")
        row["embedded_final_662"] = "false"
        row["cohort_run_id"] = "msm_china_2025_20260615"
        row["bridge_rank"] = ""
        row["bridge_score"] = ""
        try:
            row["_n50"] = str(float(row.get("n50_bp") or 0))
        except ValueError:
            row["_n50"] = "0"
        try:
            row["_protein_count"] = str(int(float(row.get("protein_count") or 0)))
        except ValueError:
            row["_protein_count"] = "0"
        msm_rows.append(row)

    archaea = [
        row
        for row in msm_rows
        if row.get("domain") == "Archaea" and int(float(row["_protein_count"])) >= 800
    ]
    for row in sorted(
        archaea,
        key=lambda item: (float(item["_n50"]), int(float(item["_protein_count"]))),
        reverse=True,
    )[:2]:
        row["smoke_group"] = "msm_mangrove_archaea_high_n50"
        selected.append(row)

    bacteria = [
        row
        for row in msm_rows
        if row.get("domain") == "Bacteria" and int(float(row["_protein_count"])) >= 1500
    ]
    for row in sorted(
        bacteria,
        key=lambda item: (float(item["_n50"]), int(float(item["_protein_count"]))),
        reverse=True,
    )[:1]:
        row["smoke_group"] = "msm_bacteria_random_control"
        selected.append(row)

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in selected:
        if row["proteome_id"] in seen:
            continue
        seen.add(row["proteome_id"])
        deduped.append(row)
    return deduped[:8]


def select_manifest_mags(
    repo_root: Path,
    manifest_path: Path,
    manifest_kind: str,
    payload_name: str,
    bridge_table: Path,
    include_assembly_context: bool,
    max_mags: int,
) -> list[dict[str, str]]:
    bridge = bridge_rank_by_sample(bridge_table)
    selected: list[dict[str, str]] = []
    rows = read_tsv(manifest_path)

    for source_row in rows:
        row = dict(source_row)
        if manifest_kind == "poc":
            if row.get("analysis_unit_type") == "assembly_context" and not include_assembly_context:
                continue
            row.update(resolve_poc_input_paths(repo_root, row))
        elif manifest_kind == "msm":
            row.update(resolve_msm_input_paths(repo_root, row))
            row["sample"] = row["proteome_id"]
            row["mag_fasta_basename"] = Path(row.get("mag_fasta", "")).name
            row["n_proteins_used"] = row.get("protein_count", "")
            row["embedded_final_662"] = "false"
            row["cohort_run_id"] = row.get("cohort_run_id") or "msm_china_2025_20260615"
        elif manifest_kind == "source_lane":
            row.update(resolve_source_lane_input_paths(repo_root, row))
            row["sample"] = row["proteome_id"]
            row["mag_fasta_basename"] = Path(row.get("mag_fasta") or row.get("local_fna_path", "")).name
            row["n_proteins_used"] = row.get("protein_count", "")
            row["embedded_final_662"] = "false"
            row["cohort_run_id"] = row.get("cohort_run_id") or row.get("source") or payload_name
        else:
            raise ValueError(f"Unknown manifest kind: {manifest_kind}")

        if not has_usable_inputs(row):
            continue

        bridge_row = bridge.get(row["proteome_id"], {})
        row["smoke_group"] = payload_name
        row["bridge_rank"] = bridge_row.get("rank", "")
        row["bridge_score"] = bridge_row.get("bridging_score", "")
        selected.append(row)
        if max_mags and len(selected) >= max_mags:
            break
    return selected


def gene_sequence_lookup(faa_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []

    def flush_record(header: str | None, seq_chunks: list[str]) -> None:
        if header is None:
            return
        seq = "".join(seq_chunks)
        clean = re.sub(r"[^A-Z*]", "", seq.upper()).replace("*", "")
        first_token = header.split()[0]
        aliases = {header, first_token}
        if "|" in first_token:
            aliases.update(part for part in first_token.split("|") if part)
        attrs = parse_attrs(header)
        aliases.update(value for value in attrs.values() if value)
        id_match = re.search(r"\bID=([^;\s]+)", header)
        if id_match:
            aliases.add(id_match.group(1))
        for alias in aliases:
            out[alias] = clean

    with open_text(faa_path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record(name, chunks)
                name = line[1:]
                chunks = []
            else:
                chunks.append(line)
    flush_record(name, chunks)
    return out


def choose_windows(
    row: dict[str, str],
    max_cds_per_window: int,
    max_windows_per_mag: int,
    max_sequence_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fna_records = fasta_records(Path(row["source_fna"]))
    contig_order = {contig_id: idx for idx, (contig_id, _) in enumerate(fna_records)}
    contig_seqs = {contig_id: seq.lower() for contig_id, seq in fna_records}
    cds_features = parse_gff(Path(row["source_gff"]))
    aa_lookup = gene_sequence_lookup(Path(row["source_faa"]))
    by_contig: dict[str, list[CdsFeature]] = defaultdict(list)
    for feature in cds_features:
        aa = (
            aa_lookup.get(feature.protein_id)
            or aa_lookup.get(feature.gene_id)
            or aa_lookup.get(feature.attrs.get("Name", ""))
            or aa_lookup.get(feature.attrs.get("locus_tag", ""))
        )
        if aa:
            by_contig[feature.contig_id].append(feature)

    contigs = sorted(
        by_contig,
        key=lambda contig_id: (-len(by_contig[contig_id]), contig_order.get(contig_id, 10**9)),
    )
    windows: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []

    for contig_id in contigs:
        features = sorted(by_contig[contig_id], key=lambda item: (item.start, item.end))
        if not features:
            continue
        chunks = [features[i : i + max_cds_per_window] for i in range(0, len(features), max_cds_per_window)]
        for chunk in chunks:
            sequence, span_rows = build_glm2_sequence(row, contig_id, chunk, contig_seqs, aa_lookup)
            if len(sequence) > max_sequence_chars:
                sequence, span_rows = build_glm2_sequence(
                    row, contig_id, chunk[: max(2, max_cds_per_window // 2)], contig_seqs, aa_lookup
                )
            if not sequence:
                continue
            window_number = len(windows) + 1
            window_id = f"{row['proteome_id']}__{contig_id}__w{window_number:02d}"
            record = window_record(row, contig_id, chunk, window_id, "contig_cds_order", sequence)
            windows.append(record)
            spans.extend(add_window_to_spans(span_rows, record))
            if len(windows) >= max_windows_per_mag:
                break
        if len(windows) >= max_windows_per_mag:
            break

    if windows:
        first = windows[0]
        first_spans = [span for span in spans if span["window_id"] == first["window_id"]]
        shuffled_features = [span for span in first_spans if span["element_type"] == "CDS"]
        shuffled_sequence, shuffled_spans = shuffled_control_sequence(
            row, first["contig_id"], shuffled_features, aa_lookup
        )
        control_id = f"{first['window_id']}__shuffled_gene_order_control"
        control = {
            **first,
            "window_id": control_id,
            "glm2_sequence_id": control_id,
            "window_type": "shuffled_gene_order_control",
            "sequence_string": shuffled_sequence,
            "control_for_window_id": first["window_id"],
            "window_token_count_estimate": len(shuffled_sequence),
        }
        windows.append(control)
        spans.extend(add_window_to_spans(shuffled_spans, control))

    return windows, spans


def build_glm2_sequence(
    row: dict[str, str],
    contig_id: str,
    features: list[CdsFeature],
    contig_seqs: dict[str, str],
    aa_lookup: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    sequence_parts: list[str] = []
    spans: list[dict[str, Any]] = []
    contig_seq = contig_seqs.get(contig_id, "")
    last_end: int | None = None
    cursor = 0
    for ordinal, feature in enumerate(features, start=1):
        if last_end is not None and feature.start > last_end + 1:
            igs = contig_seq[last_end : feature.start - 1]
            if igs:
                token = f"<+>{igs}"
                sequence_parts.append(token)
                spans.append(
                    {
                        "element_type": "IGS",
                        "gene_ordinal": "",
                        "gene_id": "",
                        "protein_id": "",
                        "strand": "+",
                        "start_bp": last_end + 1,
                        "end_bp": feature.start - 1,
                        "sequence_start": cursor,
                        "sequence_end": cursor + len(token),
                    }
                )
                cursor += len(token)
        aa = (
            aa_lookup.get(feature.protein_id)
            or aa_lookup.get(feature.gene_id)
            or aa_lookup.get(feature.attrs.get("Name", ""))
            or aa_lookup.get(feature.attrs.get("locus_tag", ""))
        )
        if not aa:
            continue
        token = f"<{feature.strand}>{aa}"
        sequence_parts.append(token)
        spans.append(
            {
                "element_type": "CDS",
                "gene_ordinal": ordinal,
                "gene_id": feature.gene_id,
                "protein_id": feature.protein_id,
                "strand": feature.strand,
                "start_bp": feature.start,
                "end_bp": feature.end,
                "sequence_start": cursor,
                "sequence_end": cursor + len(token),
            }
        )
        cursor += len(token)
        last_end = feature.end
    return "".join(sequence_parts), spans


def shuffled_control_sequence(
    row: dict[str, str],
    contig_id: str,
    cds_spans: list[dict[str, Any]],
    aa_lookup: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    rng = random.Random(row["proteome_id"])
    shuffled = list(cds_spans)
    rng.shuffle(shuffled)
    parts: list[str] = []
    out_spans: list[dict[str, Any]] = []
    cursor = 0
    for ordinal, span in enumerate(shuffled, start=1):
        aa = aa_lookup.get(span["protein_id"]) or aa_lookup.get(span["gene_id"]) or ""
        token = f"<{span['strand']}>{aa}"
        parts.append(token)
        out = dict(span)
        out.update(
            {
                "gene_ordinal": ordinal,
                "sequence_start": cursor,
                "sequence_end": cursor + len(token),
                "control_source_gene_id": span["gene_id"],
                "contig_id": contig_id,
            }
        )
        out_spans.append(out)
        cursor += len(token)
    return "".join(parts), out_spans


def window_record(
    row: dict[str, str],
    contig_id: str,
    chunk: list[CdsFeature],
    window_id: str,
    window_type: str,
    sequence: str,
) -> dict[str, Any]:
    return {
        "cohort_run_id": row.get("cohort_run_id", ""),
        "glm_run_id": row.get("glm_run_id", "glm2_smoke_20260615"),
        "model_family": "gLM2",
        "model_name": "tattabio/gLM2_650M",
        "model_version_or_checkpoint": "huggingface_main",
        "model_license": "apache-2.0",
        "proteome_id": row["proteome_id"],
        "mag_id": row.get("mag_id", ""),
        "source": row.get("source", ""),
        "ecosystem": row.get("ecosystem", ""),
        "domain": row.get("domain", ""),
        "smoke_group": row.get("smoke_group", ""),
        "contig_id": contig_id,
        "window_id": window_id,
        "glm2_sequence_id": window_id,
        "window_type": window_type,
        "window_gene_count": len(chunk),
        "window_token_count_estimate": len(sequence),
        "start_gene_ordinal": 1,
        "end_gene_ordinal": len(chunk),
        "center_gene_id": chunk[len(chunk) // 2].gene_id if chunk else "",
        "center_marker_family": "",
        "strand_pattern": "".join(feature.strand for feature in chunk),
        "context_completeness_tier": "smoke_context_window",
        "source_fna": row["source_fna"],
        "source_faa": row["source_faa"],
        "source_gff": row["source_gff"],
        "sequence_string": sequence,
        "control_for_window_id": "",
        "allowed_claim_wording": (
            "gLM2 smoke output is MAG/proteome-level contextual genomic evidence only; "
            "it does not imply measured methane flux, final MRV risk tier, or carbon-credit approval."
        ),
    }


def add_window_to_spans(
    span_rows: list[dict[str, Any]], window: dict[str, Any]
) -> list[dict[str, Any]]:
    out = []
    for span in span_rows:
        out.append(
            {
                "glm_run_id": window["glm_run_id"],
                "proteome_id": window["proteome_id"],
                "mag_id": window["mag_id"],
                "contig_id": window["contig_id"],
                "window_id": window["window_id"],
                "glm2_sequence_id": window["glm2_sequence_id"],
                **span,
            }
        )
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_payload(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    results_dir = repo_root / args.output_dir
    if args.mode == "smoke":
        selected = select_smoke_mags(
            repo_root,
            repo_root / args.poc_manifest,
            repo_root / args.msm_manifest,
            repo_root / args.bridge_table,
        )
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required for --mode manifest")
        selected = select_manifest_mags(
            repo_root=repo_root,
            manifest_path=repo_root / args.manifest,
            manifest_kind=args.manifest_kind,
            payload_name=args.payload_name,
            bridge_table=repo_root / args.bridge_table,
            include_assembly_context=args.include_assembly_context,
            max_mags=args.max_mags,
        )
    if not selected:
        raise SystemExit("No selected MAGs had usable FNA/GFF/FAA inputs.")

    for row in selected:
        row["glm_run_id"] = args.glm_run_id

    all_windows: list[dict[str, Any]] = []
    all_spans: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for row in selected:
        windows, spans = choose_windows(
            row,
            max_cds_per_window=args.max_cds_per_window,
            max_windows_per_mag=args.max_windows_per_mag,
            max_sequence_chars=args.max_sequence_chars,
        )
        all_windows.extend(windows)
        all_spans.extend(spans)
        validation_rows.append(
            {
                "proteome_id": row["proteome_id"],
                "mag_id": row.get("mag_id", ""),
                "smoke_group": row.get("smoke_group", ""),
                "source_fna_exists": Path(row["source_fna"]).exists(),
                "source_faa_exists": Path(row["source_faa"]).exists(),
                "source_gff_exists": Path(row["source_gff"]).exists(),
                "windows_built": len(windows),
                "spans_built": len(spans),
                "status": "pass" if windows and spans else "fail",
            }
        )

    manifest_fields = [
        "proteome_id",
        "mag_id",
        "sample",
        "source",
        "ecosystem",
        "domain",
        "smoke_group",
        "analysis_unit_type",
        "mbag_mag_level_include",
        "claim_scope",
        "bridge_rank",
        "bridge_score",
        "n_proteins_used",
        "protein_count",
        "n50_bp",
        "source_fna",
        "source_faa",
        "source_gff",
        "source_attempt_dir",
    ]
    window_fields = [
        "cohort_run_id",
        "glm_run_id",
        "model_family",
        "model_name",
        "model_version_or_checkpoint",
        "model_license",
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "smoke_group",
        "contig_id",
        "window_id",
        "glm2_sequence_id",
        "window_type",
        "window_gene_count",
        "window_token_count_estimate",
        "start_gene_ordinal",
        "end_gene_ordinal",
        "center_gene_id",
        "center_marker_family",
        "strand_pattern",
        "context_completeness_tier",
        "source_fna",
        "source_faa",
        "source_gff",
        "control_for_window_id",
        "allowed_claim_wording",
    ]
    span_fields = [
        "glm_run_id",
        "proteome_id",
        "mag_id",
        "contig_id",
        "window_id",
        "glm2_sequence_id",
        "element_type",
        "gene_ordinal",
        "gene_id",
        "protein_id",
        "strand",
        "start_bp",
        "end_bp",
        "sequence_start",
        "sequence_end",
        "control_source_gene_id",
    ]
    validation_fields = [
        "proteome_id",
        "mag_id",
        "smoke_group",
        "source_fna_exists",
        "source_faa_exists",
        "source_gff_exists",
        "windows_built",
        "spans_built",
        "status",
    ]

    write_tsv(results_dir / "manifests/glm2_smoke_mag_manifest.tsv", selected, manifest_fields)
    write_tsv(results_dir / "manifests/dim_glm_window.tsv", all_windows, window_fields)
    write_tsv(results_dir / "manifests/glm2_span_map.tsv", all_spans, span_fields)
    write_tsv(results_dir / "validation/prep_validation.tsv", validation_rows, validation_fields)
    write_jsonl(
        results_dir / "prepared_inputs/glm2_sequences.jsonl",
        (
            {
                field: row[field]
                for field in [
                    "glm_run_id",
                    "proteome_id",
                    "mag_id",
                    "contig_id",
                    "window_id",
                    "glm2_sequence_id",
                    "window_type",
                    "control_for_window_id",
                    "sequence_string",
                ]
            }
            for row in all_windows
        ),
    )
    write_jsonl(results_dir / "prepared_inputs/glm2_span_map.jsonl", all_spans)

    report = {
        "results_dir": str(results_dir),
        "glm_run_id": args.glm_run_id,
        "payload_mode": args.mode,
        "payload_name": args.payload_name,
        "mag_count": len(selected),
        "window_count": len(all_windows),
        "span_count": len(all_spans),
        "claim_boundary": (
            "Smoke outputs are contextual genomic feature-layer evidence at MAG/proteome grain. "
            "They complement ESM-2 and functional annotations; they do not replace them or enable "
            "sample/project MRV tiers without abundance, environment, uncertainty, and flux validation."
        ),
    }
    (results_dir / "validation").mkdir(parents=True, exist_ok=True)
    (results_dir / "validation/prep_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--mode", choices=["smoke", "manifest"], default="smoke")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-kind", choices=["poc", "msm", "source_lane"], default="poc")
    parser.add_argument("--payload-name", default="glm2_smoke")
    parser.add_argument("--glm-run-id", default="glm2_smoke_20260615")
    parser.add_argument("--include-assembly-context", action="store_true")
    parser.add_argument("--max-mags", type=int, default=0)
    parser.add_argument("--poc-manifest", type=Path, default=DEFAULT_POC_MANIFEST)
    parser.add_argument("--msm-manifest", type=Path, default=DEFAULT_MSM_MANIFEST)
    parser.add_argument("--bridge-table", type=Path, default=DEFAULT_BRIDGE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-cds-per-window", type=int, default=8)
    parser.add_argument("--max-windows-per-mag", type=int, default=1)
    parser.add_argument("--max-sequence-chars", type=int, default=3800)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build_payload(parse_args()))
