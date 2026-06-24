#!/usr/bin/env python3
"""Predict proteins and gene features for any MethaNet external MAG lane.

Inputs are standard source-lane manifests with `proteome_id`, `mag_id`, and a
FASTA column (`mag_fasta` by default). Outputs are Prodigal FAA/FFN/GFF files
plus a proteome manifest that can be merged back into the source-lane contract.
Gzipped FASTA inputs are staged to temporary plain FASTA files per MAG.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


TRUE_VALUES = {"true", "1", "yes", "y"}


@dataclass(frozen=True)
class Result:
    proteome_id: str
    mag_id: str
    local_fna_path: str
    local_faa_path: str
    local_ffn_path: str
    local_gff_path: str
    protein_prediction_status: str
    protein_count: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--proteome-dir", type=Path, required=True)
    parser.add_argument("--ffn-dir", type=Path, required=True)
    parser.add_argument("--gff-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--fna-col", default="mag_fasta")
    parser.add_argument("--faa-col", default="proteome_faa")
    parser.add_argument("--include-col", default="")
    parser.add_argument(
        "--stratify-cols",
        default="",
        help="Comma-separated columns used to round-robin select --limit rows for smoke tests.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prodigal", default="prodigal")
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_value(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_unique_proteome_ids(rows: list[dict[str, str]], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    missing = 0
    for row in rows:
        proteome_id = str(row.get("proteome_id") or "").strip()
        if not proteome_id:
            missing += 1
        elif proteome_id in seen:
            duplicates.add(proteome_id)
        else:
            seen.add(proteome_id)
    if missing or duplicates:
        parts = []
        if missing:
            parts.append(f"{missing} rows missing proteome_id")
        if duplicates:
            parts.append("duplicate proteome_id values: " + ", ".join(sorted(duplicates)))
        raise SystemExit(f"{label} has invalid proteome_id values: {'; '.join(parts)}")


def write_rows(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "proteome_id",
        "mag_id",
        "local_fna_path",
        "local_faa_path",
        "local_ffn_path",
        "local_gff_path",
        "protein_prediction_status",
        "protein_count",
        "note",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for result in rows:
            writer.writerow(result.__dict__)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def count_fasta_records(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def safe_stem(row: dict[str, str]) -> str:
    return row.get("mag_id") or row["proteome_id"]


def default_faa_path(repo_root: Path, proteome_dir: Path, row: dict[str, str], faa_col: str) -> Path:
    value = row.get(faa_col, "")
    if value:
        return resolve_value(repo_root, value)
    return proteome_dir / f"{safe_stem(row)}.faa"


def stage_fasta(source: Path, scratch_dir: Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if source.suffix != ".gz":
        return source, None
    tmp = tempfile.TemporaryDirectory(dir=str(scratch_dir) if scratch_dir else None)
    staged = Path(tmp.name) / source.name.removesuffix(".gz")
    with gzip.open(source, "rb") as src, staged.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return staged, tmp


def predict_one(
    row: dict[str, str],
    *,
    repo_root: Path,
    prodigal: str,
    fna_col: str,
    faa_col: str,
    proteome_dir: Path,
    ffn_dir: Path,
    gff_dir: Path,
    log_dir: Path,
    scratch_dir: Path | None,
    force: bool,
) -> Result:
    proteome_id = row["proteome_id"]
    mag_id = row.get("mag_id") or proteome_id
    fna_value = row.get(fna_col) or row.get("local_fna_path") or row.get("fasta_path") or ""
    if not fna_value:
        return Result(proteome_id, mag_id, "", "", "", "", "skipped_missing_fna_path", 0, f"Missing {fna_col}.")
    fna = resolve_value(repo_root, fna_value)
    faa = default_faa_path(repo_root, proteome_dir, row, faa_col)
    ffn = ffn_dir / f"{mag_id}.ffn"
    gff = gff_dir / f"{mag_id}.gff"
    out_log = log_dir / f"{mag_id}.out"
    err_log = log_dir / f"{mag_id}.err"

    for directory in [faa.parent, ffn.parent, gff.parent, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if not fna.exists() or fna.stat().st_size == 0:
        return Result(proteome_id, mag_id, str(fna), str(faa), str(ffn), str(gff), "skipped_missing_fna", 0, "Input FASTA missing or empty.")
    if faa.exists() and faa.stat().st_size > 0 and gff.exists() and gff.stat().st_size > 0 and not force:
        return Result(proteome_id, mag_id, str(fna), str(faa), str(ffn), str(gff), "existing", count_fasta_records(faa), "Existing FAA/GFF reused.")

    staged_fna, tmp = stage_fasta(fna, scratch_dir)
    try:
        cmd = [
            prodigal,
            "-i",
            str(staged_fna),
            "-a",
            str(faa),
            "-d",
            str(ffn),
            "-o",
            str(gff),
            "-f",
            "gff",
            "-p",
            "meta",
        ]
        with out_log.open("w") as stdout, err_log.open("w") as stderr:
            proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)
    finally:
        if tmp is not None:
            tmp.cleanup()

    protein_count = count_fasta_records(faa)
    if proc.returncode != 0:
        return Result(proteome_id, mag_id, str(fna), str(faa), str(ffn), str(gff), "failed", protein_count, f"Prodigal exited {proc.returncode}; see {err_log}")
    if protein_count == 0:
        return Result(proteome_id, mag_id, str(fna), str(faa), str(ffn), str(gff), "failed_empty_faa", 0, f"Prodigal produced zero proteins; see {err_log}")
    return Result(proteome_id, mag_id, str(fna), str(faa), str(ffn), str(gff), "predicted", protein_count, "Prodigal -p meta.")


def filter_rows(rows: list[dict[str, str]], include_col: str) -> list[dict[str, str]]:
    if not include_col:
        return rows
    return [row for row in rows if truthy(row.get(include_col))]


def stratified_limit(rows: list[dict[str, str]], limit: int | None, cols: str) -> list[dict[str, str]]:
    if limit is None or limit <= 0 or not cols:
        return rows[: max(0, limit)] if limit is not None else rows
    strat_cols = [col.strip() for col in cols.split(",") if col.strip()]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(col, "") for col in strat_cols)
        groups.setdefault(key, []).append(row)
    selected: list[dict[str, str]] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = resolve(repo_root, args.manifest)
    output_manifest = resolve(repo_root, args.output_manifest)
    proteome_dir = resolve(repo_root, args.proteome_dir)
    ffn_dir = resolve(repo_root, args.ffn_dir)
    gff_dir = resolve(repo_root, args.gff_dir)
    log_dir = resolve(repo_root, args.log_dir)
    scratch_dir = resolve(repo_root, args.scratch_dir) if args.scratch_dir else None
    if scratch_dir:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    prodigal = shutil.which(args.prodigal)
    if not prodigal:
        raise SystemExit(f"Prodigal not found on PATH: {args.prodigal}")

    rows = filter_rows(read_rows(manifest), args.include_col)
    rows = stratified_limit(rows, args.limit, args.stratify_cols)
    if not rows:
        raise SystemExit("No manifest rows selected for protein prediction.")
    require_unique_proteome_ids(rows, "selected protein-prediction rows")

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                predict_one,
                row,
                repo_root=repo_root,
                prodigal=prodigal,
                fna_col=args.fna_col,
                faa_col=args.faa_col,
                proteome_dir=proteome_dir,
                ffn_dir=ffn_dir,
                gff_dir=gff_dir,
                log_dir=log_dir,
                scratch_dir=scratch_dir,
                force=args.force,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result.protein_prediction_status}\t{result.protein_count}\t{result.mag_id}", flush=True)
    results.sort(key=lambda result: (result.proteome_id, result.mag_id))
    write_rows(output_manifest, results)
    print(f"wrote\t{output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
