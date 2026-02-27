#!/usr/bin/env python3
"""Fail-fast validator for Blue Catalyst artifact completeness.

Usage:
  python scripts/validate_blue_catalyst_artifacts.py \
    --artifacts-dir /path/to/run/artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FILES = {
    "genome_embeddings.npz",
    "embedding_metadata.tsv",
    "embedding_projection_clusters.tsv",
    "bridging_genomes_top.tsv",
    "poc_metrics.json",
    "advanced_analytics_summary.json",
    "artifact_manifest.tsv",
    "artifact_manifest.json",
    "artifact_registry.tsv",
}

REQUIRED_TABLE_COLUMNS = {
    "artifact_manifest.tsv": {"file", "relative_path", "category", "description", "size_bytes"},
    "artifact_registry.tsv": {"file", "relative_path", "category", "description"},
}


def _read_tsv_header(path: Path) -> set[str]:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return set(first_line.split("\t"))


def validate(artifacts_dir: Path) -> list[str]:
    errors: list[str] = []

    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return [f"Artifacts directory not found: {artifacts_dir}"]

    existing = {p.name for p in artifacts_dir.glob("*") if p.is_file()}

    missing = sorted(REQUIRED_FILES - existing)
    if missing:
        errors.append("Missing required files: " + ", ".join(missing))

    for name, expected_cols in REQUIRED_TABLE_COLUMNS.items():
        fp = artifacts_dir / name
        if not fp.exists():
            continue
        try:
            cols = _read_tsv_header(fp)
        except Exception as exc:  # pragma: no cover - hard fail path
            errors.append(f"Failed reading header from {name}: {exc}")
            continue
        missing_cols = sorted(expected_cols - cols)
        if missing_cols:
            errors.append(f"{name} missing columns: {', '.join(missing_cols)}")

    manifest_json = artifacts_dir / "artifact_manifest.json"
    if manifest_json.exists():
        try:
            payload = json.loads(manifest_json.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or not payload:
                errors.append("artifact_manifest.json must be a non-empty list")
        except json.JSONDecodeError as exc:
            errors.append(f"artifact_manifest.json is not valid JSON: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Blue Catalyst artifact completeness.")
    parser.add_argument("--artifacts-dir", required=True, type=Path, help="Artifacts directory path")
    args = parser.parse_args()

    errors = validate(args.artifacts_dir)
    if errors:
        print("[FAIL] Blue Catalyst artifact validation failed", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        return 1

    print(f"[OK] Blue Catalyst artifacts validated: {args.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
