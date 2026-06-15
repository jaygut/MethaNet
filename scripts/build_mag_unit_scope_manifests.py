#!/usr/bin/env python3
"""Build MethaNet MAG/bin versus assembly-context launch manifests.

This script is read-only with respect to production per-MAG output folders. It
uses the 662-row functional manifest as the cohort backbone, enriches rows with
available live run evidence, classifies each proteome by analytical unit, and
writes filtered relaunch manifests for downstream Slurm/Snakemake use.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/"
    "poc_662_functional_mag_manifest.proposed.tsv"
)
DEFAULT_COHORT_DIR = Path("results/functional_metagenomics/fgx_662_apollo3_20260612")
DEFAULT_OUTPUT_DIR = Path("results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255")

ASSEMBLY_RUMEN_RE = re.compile(r"^rumen__10676_\d+_idba$")
TRUE_VALUES = {"true", "1", "yes", "y"}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def _abs_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _parse_metric_tsv(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            metric = row.get("metric")
            value = row.get("value")
            if metric:
                out[metric] = value or ""
    return out


def _latest_attempt(per_mag_dir: Path, proteome_id: str) -> Path | None:
    root = per_mag_dir / proteome_id
    if not root.exists():
        return None
    attempts = [path for path in root.iterdir() if path.is_dir()]
    if not attempts:
        return None
    return sorted(attempts, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def _attempt_status(attempt: Path | None) -> str:
    if attempt is None:
        return "not_started"
    if (attempt / "COMPLETE").exists():
        return "complete"
    if (attempt / "FAILED").exists():
        return "failed"
    if (attempt / "status.tsv").exists():
        return "partial"
    return "attempt_created"


def _read_run_record(attempt: Path | None) -> dict[str, Any]:
    if attempt is None:
        return {}
    path = attempt / "curated/run_record.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _count_faa_headers(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    count = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def _find_predicted_faa(attempt: Path | None, mag_id: str, mag_fasta_basename: str) -> Path | None:
    if attempt is None:
        return None
    stems = [
        Path(mag_fasta_basename).name.removesuffix(".gz").removesuffix(".fasta").removesuffix(".fa"),
        mag_id,
    ]
    candidates: list[Path] = []
    for stem in stems:
        if stem:
            candidates.extend([attempt / "genes" / f"{stem}.faa", attempt / "bakta" / f"{stem}.faa"])
    candidates.extend(sorted((attempt / "genes").glob("*.faa")) if (attempt / "genes").exists() else [])
    candidates.extend(sorted((attempt / "bakta").glob("*.faa")) if (attempt / "bakta").exists() else [])
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _input_uncompressed_bytes(attempt: Path | None, fasta_path: Path | None) -> str:
    if fasta_path and fasta_path.exists() and fasta_path.suffix != ".gz":
        return str(fasta_path.stat().st_size)
    if attempt:
        for root in [attempt / "input", attempt / "staged_fasta"]:
            if root.exists():
                for path in sorted(root.iterdir()):
                    if path.is_file() and path.suffix != ".gz":
                        return str(path.stat().st_size)
    return ""


def _safe_ratio(numerator: int | None, denominator: str | None) -> str:
    if numerator is None:
        return ""
    try:
        denom = float(denominator or 0)
    except ValueError:
        return ""
    if denom <= 0:
        return ""
    return f"{numerator / denom:.6g}"


def _classify(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    proteome_id = str(row.get("proteome_id") or "")
    source = str(row.get("source") or "").lower()
    ecosystem = str(row.get("ecosystem") or "").lower()
    basename = str(row.get("mag_fasta_basename") or "")
    scope_ratio_text = str(row.get("scope_ratio") or "")
    total_bp_text = str(row.get("input_total_bp") or "")
    compressed_bytes_text = str(row.get("input_fasta_compressed_bytes") or "")

    has_bin = "bin." in basename or "bin." in proteome_id
    is_mucc_mag_like = source == "mucc" or ecosystem == "wetland"
    is_no_bin_assembly_name = source == "rumen" and ASSEMBLY_RUMEN_RE.match(proteome_id) and not has_bin

    scope_ratio = float(scope_ratio_text) if scope_ratio_text else 0.0
    total_bp = int(float(total_bp_text)) if total_bp_text else 0
    compressed_bytes = int(float(compressed_bytes_text)) if compressed_bytes_text else 0

    if is_no_bin_assembly_name or (source == "rumen" and not has_bin and (scope_ratio > 2.0 or total_bp > 20_000_000 or compressed_bytes > 10_000_000)):
        return (
            "assembly_context",
            "false",
            "true",
            "assembly/metagenome context",
            "blocked_noncomparable_assembly_context",
            "rumen no-bin 10676 assembly-scale record; preserve separately and exclude from MAG-level MBAG",
            "preserve_quarantine_exclude_from_mag_level_mbag",
        )
    if has_bin or is_mucc_mag_like:
        return (
            "mag_bin",
            "true",
            "false",
            "MAG functional potential",
            "comparable_mag_bin",
            "MAG/bin-like input; eligible for MAG-level functional atlas if QC and table gates pass",
            "include_in_mag_bin_relaunch_or_retain_if_complete",
        )
    return (
        "unresolved",
        "false",
        "false",
        "not comparable",
        "blocked_unresolved_unit_scope",
        "unit scope could not be proven from source and FASTA basename",
        "manual_review_before_any_relaunch",
    )


def build_rows(repo_root: Path, manifest: Path, cohort_dir: Path) -> list[dict[str, Any]]:
    rows = _read_tsv(manifest)
    per_mag_dir = cohort_dir / "per_mag"
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched: dict[str, Any] = dict(row)
        proteome_id = row["proteome_id"]
        attempt = _latest_attempt(per_mag_dir, proteome_id)
        record = _read_run_record(attempt)
        stats = _parse_metric_tsv(attempt / "input_stats.tsv") if attempt else {}
        mag_fasta = _abs_path(repo_root, row.get("mag_fasta"))
        predicted_faa = _find_predicted_faa(attempt, row.get("mag_id", ""), row.get("mag_fasta_basename", ""))
        predicted_proteins = _count_faa_headers(predicted_faa) if predicted_faa else None

        enriched["cohort_run_id"] = cohort_dir.name
        enriched["latest_run_id"] = record.get("run_id") or (attempt.name if attempt else "")
        enriched["latest_run_status"] = record.get("status") or _attempt_status(attempt)
        enriched["latest_run_dir"] = str(attempt) if attempt else ""
        enriched["functional_predicted_proteins"] = predicted_proteins if predicted_proteins is not None else ""
        enriched["scope_ratio"] = _safe_ratio(predicted_proteins, row.get("n_proteins_used"))
        enriched["input_total_bp"] = stats.get("total_bp", "")
        enriched["input_contigs"] = stats.get("contigs", "")
        enriched["input_n50_bp"] = stats.get("n50_bp", "")
        enriched["input_fasta_kind"] = "compressed_gzip" if (mag_fasta and mag_fasta.suffix == ".gz") else "plain_fasta"
        enriched["input_fasta_compressed_bytes"] = (
            str(mag_fasta.stat().st_size) if mag_fasta and mag_fasta.exists() and mag_fasta.suffix == ".gz" else ""
        )
        enriched["input_fasta_uncompressed_bytes"] = _input_uncompressed_bytes(attempt, mag_fasta)

        (
            enriched["analysis_unit_type"],
            enriched["mbag_mag_level_include"],
            enriched["assembly_context_include"],
            enriched["claim_scope"],
            enriched["comparability_status"],
            enriched["comparability_reason"],
            enriched["recommended_action"],
        ) = _classify(enriched)

        out.append(enriched)
    return out


def validate(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    pids = [row.get("proteome_id", "") for row in rows]
    counts = Counter(row.get("analysis_unit_type", "") for row in rows)
    mag_bin_rows = [row for row in rows if row.get("analysis_unit_type") == "mag_bin"]
    remaining_mag_bin_rows = [
        row
        for row in mag_bin_rows
        if str(row.get("latest_run_status") or "").strip().lower() != "complete"
    ]
    assembly_rows = [row for row in rows if row.get("analysis_unit_type") == "assembly_context"]
    unresolved_rows = [row for row in rows if row.get("analysis_unit_type") == "unresolved"]
    no_bin_in_mag = [
        row["proteome_id"]
        for row in mag_bin_rows
        if row.get("source") == "rumen" and ASSEMBLY_RUMEN_RE.match(row["proteome_id"])
    ]

    def add(gate: str, status: str, detail: str) -> None:
        checks.append({"gate": gate, "status": status, "detail": detail})

    add("embedded_backbone_rows", "pass" if len(rows) == 662 else "fail", f"rows={len(rows)} expected=662")
    add("unique_proteome_id", "pass" if len(pids) == len(set(pids)) else "fail", f"unique={len(set(pids))} total={len(pids)}")
    add("one_classification_per_proteome", "pass" if sum(counts.values()) == len(rows) else "fail", dict(counts).__repr__())
    add("mag_bin_denom", "pass" if len(mag_bin_rows) == 625 else "warn", f"mag_bin={len(mag_bin_rows)} expected_current_rule=625")
    add("remaining_mag_bin_denom", "pass" if remaining_mag_bin_rows else "warn", f"remaining_mag_bin={len(remaining_mag_bin_rows)}")
    add("assembly_context_quarantine", "pass" if len(assembly_rows) == 37 else "warn", f"assembly_context={len(assembly_rows)} expected_current_rule=37")
    add("unresolved_units", "pass" if not unresolved_rows else "fail", f"unresolved={len(unresolved_rows)}")
    add("no_no_bin_rumen_in_mag_bin", "pass" if not no_bin_in_mag else "fail", f"examples={no_bin_in_mag[:5]}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    cohort_dir = args.cohort_dir if args.cohort_dir.is_absolute() else repo_root / args.cohort_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    rows = build_rows(repo_root, manifest, cohort_dir)
    checks = validate(rows)

    original_fields = list(rows[0].keys()) if rows else []
    added_fields = [
        "cohort_run_id",
        "latest_run_id",
        "latest_run_status",
        "latest_run_dir",
        "functional_predicted_proteins",
        "scope_ratio",
        "input_total_bp",
        "input_contigs",
        "input_n50_bp",
        "input_fasta_compressed_bytes",
        "input_fasta_uncompressed_bytes",
        "input_fasta_kind",
        "analysis_unit_type",
        "mbag_mag_level_include",
        "assembly_context_include",
        "claim_scope",
        "comparability_status",
        "comparability_reason",
        "recommended_action",
    ]
    fieldnames = []
    for field in original_fields + added_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    scoped_path = output_dir / "poc_662_functional_mag_manifest.with_unit_scope.tsv"
    mag_path = output_dir / "poc_662_functional_mag_manifest.mag_bin_only.tsv"
    remaining_path = output_dir / "poc_662_functional_mag_manifest.mag_bin_remaining.tsv"
    assembly_path = output_dir / "poc_662_functional_mag_manifest.assembly_context.tsv"
    validation_path = output_dir / "poc_662_functional_mag_manifest.unit_scope_validation.tsv"

    _write_tsv(scoped_path, rows, fieldnames)
    _write_tsv(
        mag_path,
        [row for row in rows if row.get("analysis_unit_type") == "mag_bin" and row.get("mbag_mag_level_include") == "true"],
        fieldnames,
    )
    _write_tsv(
        remaining_path,
        [
            row
            for row in rows
            if row.get("analysis_unit_type") == "mag_bin"
            and row.get("mbag_mag_level_include") == "true"
            and str(row.get("latest_run_status") or "").strip().lower() != "complete"
        ],
        fieldnames,
    )
    _write_tsv(
        assembly_path,
        [row for row in rows if row.get("analysis_unit_type") == "assembly_context"],
        fieldnames,
    )
    _write_tsv(validation_path, checks, ["gate", "status", "detail"])

    for check in checks:
        print(f"{check['status'].upper()}\t{check['gate']}\t{check['detail']}")
    print(f"WROTE\t{scoped_path}")
    print(f"WROTE\t{mag_path}")
    print(f"WROTE\t{remaining_path}")
    print(f"WROTE\t{assembly_path}")
    print(f"WROTE\t{validation_path}")

    return 1 if any(check["status"] == "fail" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
