#!/usr/bin/env python3
"""Predict MSM China 2025 MAG proteomes with Prodigal.

Run inside the MethaNet functional-genomics environment, e.g.:

  source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
  conda activate methanet-fgx
  python scripts/external/predict_msm_china_2025_proteomes.py --workers 16
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DATASET_DIR = Path("data/external/msm_china_2025")
DEFAULT_MANIFEST = DATASET_DIR / "manifests" / "msm_china_2025_mag_manifest.resolved.tsv"
OUT_MANIFEST = DATASET_DIR / "manifests" / "msm_china_2025_proteome_manifest.tsv"
LOG_DIR = DATASET_DIR / "logs" / "prodigal_proteome_prediction"
FFN_DIR = DATASET_DIR / "genes_ffn"
GFF_DIR = DATASET_DIR / "genes_gff"


@dataclass
class Result:
    proteome_id: str
    mag_id: str
    local_fna_path: str
    local_faa_path: str
    local_ffn_path: str
    local_gff_path: str
    status: str
    protein_count: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MSM MAG protein FASTAs with Prodigal.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prodigal", default="prodigal")
    return parser.parse_args()


def count_fasta_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def predict(row: dict[str, str], prodigal: str, force: bool) -> Result:
    fna = Path(row["local_fna_path"])
    faa = Path(row["local_faa_path"])
    ffn = FFN_DIR / f"{row['mag_id']}.ffn"
    gff = GFF_DIR / f"{row['mag_id']}.gff"
    out_log = LOG_DIR / f"{row['mag_id']}.out"
    err_log = LOG_DIR / f"{row['mag_id']}.err"
    faa.parent.mkdir(parents=True, exist_ok=True)
    ffn.parent.mkdir(parents=True, exist_ok=True)
    gff.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if faa.exists() and faa.stat().st_size > 0 and not force:
        return Result(
            row["proteome_id"],
            row["mag_id"],
            str(fna),
            str(faa),
            str(ffn),
            str(gff),
            "existing",
            count_fasta_records(faa),
            "Existing FAA reused.",
        )

    cmd = [
        prodigal,
        "-i",
        str(fna),
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
    protein_count = count_fasta_records(faa)
    if proc.returncode != 0:
        return Result(
            row["proteome_id"],
            row["mag_id"],
            str(fna),
            str(faa),
            str(ffn),
            str(gff),
            "failed",
            protein_count,
            f"Prodigal exited {proc.returncode}; see {err_log}",
        )
    if protein_count == 0:
        return Result(
            row["proteome_id"],
            row["mag_id"],
            str(fna),
            str(faa),
            str(ffn),
            str(gff),
            "failed_empty_faa",
            0,
            f"Prodigal produced zero proteins; see {err_log}",
        )
    return Result(
        row["proteome_id"],
        row["mag_id"],
        str(fna),
        str(faa),
        str(ffn),
        str(gff),
        "predicted",
        protein_count,
        "Prodigal 2.6.3 -p meta.",
    )


def main() -> int:
    args = parse_args()
    prodigal = shutil.which(args.prodigal)
    if not prodigal:
        raise SystemExit(f"Prodigal not found on PATH: {args.prodigal}")

    with Path(args.manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_row = {executor.submit(predict, row, prodigal, args.force): row for row in rows}
        for future in as_completed(future_to_row):
            result = future.result()
            results.append(result)
            print(f"{result.status}\t{result.protein_count}\t{result.mag_id}", flush=True)

    results.sort(key=lambda result: result.mag_id)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
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
    with OUT_MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "proteome_id": result.proteome_id,
                    "mag_id": result.mag_id,
                    "local_fna_path": result.local_fna_path,
                    "local_faa_path": result.local_faa_path,
                    "local_ffn_path": result.local_ffn_path,
                    "local_gff_path": result.local_gff_path,
                    "protein_prediction_status": result.status,
                    "protein_count": result.protein_count,
                    "note": result.note,
                }
            )
    print(f"wrote\t{OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
