#!/usr/bin/env python3
"""Build a MUCC v1 gLM2-ready manifest from source-lane and Prodigal outputs.

The source-lane manifest preserves direct MUCC evidence. This derivative
manifest is only for contextual gLM2 payload preparation, where FAA/GFF paths
must come from the same Prodigal run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


TRUE_STATUSES = {"predicted", "existing"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def count_fasta_records(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    count = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def resolve(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def file_ready(repo_root: Path, value: str | None) -> bool:
    path = resolve(repo_root, value)
    return bool(path and path.is_file() and path.stat().st_size > 0)


def build(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    source_rows = read_tsv(resolve(repo_root, str(args.source_manifest)) or args.source_manifest)
    prodigal_manifest = resolve(repo_root, str(args.prodigal_manifest)) or args.prodigal_manifest
    prodigal_rows = read_tsv(prodigal_manifest) if prodigal_manifest.exists() else []
    prodigal_by_pid = {row["proteome_id"]: row for row in prodigal_rows}

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for source in source_rows:
        proteome_id = source["proteome_id"]
        mag_id = source.get("mag_id", "")
        prodigal = prodigal_by_pid.get(proteome_id, {})
        if not prodigal and args.allow_file_scan_fallback and mag_id:
            fallback_faa = resolve(repo_root, str(args.prodigal_proteome_dir / f"{mag_id}.faa"))
            fallback_ffn = resolve(repo_root, str(args.prodigal_ffn_dir / f"{mag_id}.ffn"))
            fallback_gff = resolve(repo_root, str(args.prodigal_gff_dir / f"{mag_id}.gff"))
            if fallback_faa and fallback_gff and fallback_faa.exists() and fallback_gff.exists():
                prodigal = {
                    "proteome_id": proteome_id,
                    "mag_id": mag_id,
                    "local_faa_path": str(fallback_faa),
                    "local_ffn_path": str(fallback_ffn or ""),
                    "local_gff_path": str(fallback_gff),
                    "protein_prediction_status": "predicted_file_scan_fallback",
                    "protein_count": str(count_fasta_records(fallback_faa)),
                }
        prediction_status = prodigal.get("protein_prediction_status", "")
        fna = source.get("mag_fasta") or source.get("local_fna_path", "")
        faa = prodigal.get("local_faa_path", "")
        ffn = prodigal.get("local_ffn_path", "")
        gff = prodigal.get("local_gff_path", "")
        ready = (
            (prediction_status in TRUE_STATUSES or prediction_status == "predicted_file_scan_fallback")
            and file_ready(repo_root, fna)
            and file_ready(repo_root, faa)
            and file_ready(repo_root, gff)
        )
        source_faa = source.get("proteome_faa") or source.get("local_faa_path", "")
        row = {
            **source,
            "proteome_faa": faa if ready else "",
            "local_faa_path": faa if ready else "",
            "local_ffn_path": ffn if ready else "",
            "local_gff_path": gff if ready else "",
            "source_faa_path": source_faa,
            "prodigal_faa_path": faa,
            "prodigal_ffn_path": ffn,
            "prodigal_gff_path": gff,
            "protein_prediction_status": prediction_status or "missing_prodigal_row",
            "protein_count": prodigal.get("protein_count", source.get("protein_count", "")),
            "payload_status": "fna_prodigal_faa_gff_ready" if ready else "blocked_context_inputs",
            "comparability_status": (
                "ready_for_glm2_contextual_embedding_with_generated_gene_context"
                if ready
                else "blocked_missing_generated_gene_context"
            ),
            "glm2_include": "true" if ready else "false",
            "glm2_status": "glm2_input_ready" if ready else "blocked_missing_context_inputs",
            "context_derivation": "Prodigal -p meta FAA/FFN/GFF generated from MUCC MAG FASTA",
            "claim_scope": (
                "MAG/proteome wetland contextual-genomics derivative layer; no final MRV "
                "tiers, flux, crediting, or source-independent transfer claims"
            ),
        }
        rows.append(row)
        if not ready:
            missing = []
            if prediction_status not in TRUE_STATUSES:
                if prediction_status == "predicted_file_scan_fallback":
                    pass
                else:
                    missing.append(f"prediction_status={prediction_status or 'missing'}")
            if not file_ready(repo_root, fna):
                missing.append("fna")
            if not file_ready(repo_root, faa):
                missing.append("prodigal_faa")
            if not file_ready(repo_root, gff):
                missing.append("prodigal_gff")
            status_rows.append(
                {
                    "proteome_id": proteome_id,
                    "mag_id": source.get("mag_id", ""),
                    "status": "blocked",
                    "blocking_inputs": ";".join(missing),
                    "recommended_action": "complete or repair Prodigal context generation before gLM2 payload build",
                }
            )

    fieldnames = list(source_rows[0].keys())
    for field in [
        "source_faa_path",
        "prodigal_faa_path",
        "prodigal_ffn_path",
        "prodigal_gff_path",
        "context_derivation",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)
    write_tsv(args.output_manifest, rows, fieldnames)
    write_tsv(
        args.output_gap_register,
        status_rows,
        ["proteome_id", "mag_id", "status", "blocking_inputs", "recommended_action"],
    )
    ready_count = sum(1 for row in rows if row.get("glm2_include") == "true")
    print(
        f"wrote\t{args.output_manifest}\n"
        f"rows\t{len(rows)}\n"
        f"glm2_ready_rows\t{ready_count}\n"
        f"blocked_rows\t{len(rows) - ready_count}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prodigal-manifest", type=Path, required=True)
    parser.add_argument("--allow-file-scan-fallback", action="store_true")
    parser.add_argument(
        "--prodigal-proteome-dir",
        type=Path,
        default=Path(
            "results/functional_metagenomics/mucc_v1_owc_wetland_20260626/prodigal/proteomes_faa"
        ),
    )
    parser.add_argument(
        "--prodigal-ffn-dir",
        type=Path,
        default=Path(
            "results/functional_metagenomics/mucc_v1_owc_wetland_20260626/prodigal/genes_ffn"
        ),
    )
    parser.add_argument(
        "--prodigal-gff-dir",
        type=Path,
        default=Path(
            "results/functional_metagenomics/mucc_v1_owc_wetland_20260626/prodigal/genes_gff"
        ),
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-gap-register", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
