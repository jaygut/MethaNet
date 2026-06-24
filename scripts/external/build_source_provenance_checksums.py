#!/usr/bin/env python3
"""Build a checksum ledger for external source-provenance files.

Source documents live under data/external/<dataset_id>/source_docs/. This
utility creates a small, deterministic TSV that the lane registry validator can
use to prove source files have not drifted on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any


FIELDS = ["artifact", "path", "size_bytes", "md5", "sha256", "source_url"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument(
        "--artifact-prefix",
        default="",
        help="Optional prefix prepended to artifact names, e.g. msm_china_2025.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Relative file or directory path under source-dir to exclude. May be repeated.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def file_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def read_existing_source_urls(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {
            str(row.get("path") or "").strip(): str(row.get("source_url") or "").strip()
            for row in reader
            if str(row.get("path") or "").strip()
        }


def artifact_name(relative_path: Path, prefix: str) -> str:
    stem = "__".join(relative_path.with_suffix("").parts)
    normalized = "".join(char if char.isalnum() else "_" for char in stem).strip("_")
    if prefix:
        return f"{prefix}__{normalized}"
    return normalized


def collect_rows(
    repo_root: Path,
    source_dir: Path,
    output_tsv: Path,
    artifact_prefix: str = "",
    excludes: list[str] | None = None,
) -> list[dict[str, Any]]:
    excludes = [item.strip("/") for item in (excludes or []) if item.strip()]
    existing_source_urls = read_existing_source_urls(output_tsv)
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == output_tsv.resolve():
            continue
        rel = path.relative_to(source_dir)
        rel_str = str(rel)
        if any(rel_str == exclude or rel_str.startswith(f"{exclude}/") for exclude in excludes):
            continue
        repo_path = relative_to_repo(repo_root, path)
        md5, sha256 = file_hashes(path)
        rows.append(
            {
                "artifact": artifact_name(rel, artifact_prefix),
                "path": repo_path,
                "size_bytes": path.stat().st_size,
                "md5": md5,
                "sha256": sha256,
                "source_url": existing_source_urls.get(repo_path, ""),
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_dir = resolve(repo_root, args.source_dir).resolve()
    output_tsv = resolve(repo_root, args.output_tsv).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source provenance directory missing: {source_dir}")
    rows = collect_rows(repo_root, source_dir, output_tsv, args.artifact_prefix, args.exclude)
    if not rows:
        raise SystemExit(f"No source provenance files found under: {source_dir}")
    write_tsv(output_tsv, rows)
    print(f"Wrote {len(rows)} source provenance checksum rows to {output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
