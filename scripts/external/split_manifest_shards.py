#!/usr/bin/env python3
"""Split a tabular MethaNet manifest into deterministic shards.

This is intentionally generic: it can split source-lane, functional, ESM2, or
gLM2 manifests while preserving headers and optional boolean include filters.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TRUE_VALUES = {"true", "1", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--include-col", default="")
    parser.add_argument("--id-col", default="proteome_id")
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def require_unique_ids(rows: list[dict[str, str]], id_col: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    missing = 0
    for row in rows:
        value = str(row.get(id_col) or "").strip()
        if not value:
            missing += 1
        elif value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    if missing or duplicates:
        parts = []
        if missing:
            parts.append(f"{missing} rows missing {id_col}")
        if duplicates:
            parts.append(f"duplicate {id_col} values: " + ", ".join(sorted(duplicates)))
        raise SystemExit(f"Selected shard rows have invalid {id_col} values: {'; '.join(parts)}")


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")
    fields, rows = read_tsv(args.input)
    if not fields:
        raise SystemExit(f"Manifest has no header: {args.input}")
    if args.include_col:
        if args.include_col not in fields:
            raise SystemExit(f"Missing include column {args.include_col}: {args.input}")
        rows = [row for row in rows if truthy(row.get(args.include_col))]
    if args.id_col in fields:
        require_unique_ids(rows, args.id_col)
        rows = sorted(rows, key=lambda row: row.get(args.id_col, ""))
    shard_size = max(1, math.ceil(len(rows) / args.shards)) if rows else 1
    manifest_rows = []
    for idx in range(args.shards):
        start = idx * shard_size
        stop = min(len(rows), start + shard_size)
        shard_rows = rows[start:stop]
        shard_path = args.output_dir / f"{args.prefix}.shard_{idx + 1:03d}.tsv"
        write_tsv(shard_path, fields, shard_rows)
        manifest_rows.append(
            {
                "shard": f"{idx + 1:03d}",
                "path": str(shard_path),
                "rows": str(len(shard_rows)),
                "start_index_1based": str(start + 1 if shard_rows else ""),
                "end_index_1based": str(stop if shard_rows else ""),
            }
        )
    write_tsv(args.output_dir / f"{args.prefix}.shard_manifest.tsv", list(manifest_rows[0]), manifest_rows)
    print(f"input_rows\t{len(rows)}")
    print(f"shards\t{args.shards}")
    print(f"wrote\t{args.output_dir / (args.prefix + '.shard_manifest.tsv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
