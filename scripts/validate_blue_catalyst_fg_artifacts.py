#!/usr/bin/env python3
"""Fail-fast validator for Blue Catalyst FG-batch artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FILES = {
    "embedding_index_frozen.tsv",
    "canonical_id_map.tsv",
    "fg_processing_manifest.tsv",
    "fg_batch_plan.tsv",
    "fg_plan_summary.json",
    "fg_functional_features.tsv",
    "fg_failures.tsv",
    "fg_embedding_join.tsv",
    "id_reconciliation_report.tsv",
    "id_reconciliation_report.json",
    "modeling_feature_matrix.tsv",
    "modeling_feature_manifest.json",
}


REQUIRED_REPORT_KEYS = {
    "n_embeddings",
    "n_functional_profiles",
    "n_joined",
    "n_embedding_only",
    "n_failures",
    "duplicate_functional_ids",
    "join_coverage",
    "min_join_coverage",
}


def validate(artifacts_dir: Path) -> list[str]:
    errors: list[str] = []

    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return [f"Artifacts directory not found: {artifacts_dir}"]

    existing = {p.name for p in artifacts_dir.glob("*") if p.is_file()}
    missing = sorted(REQUIRED_FILES - existing)
    if missing:
        errors.append("Missing required files: " + ", ".join(missing))

    plan_summary_fp = artifacts_dir / "fg_plan_summary.json"
    if plan_summary_fp.exists():
        try:
            payload = json.loads(plan_summary_fp.read_text(encoding="utf-8"))
            if payload.get("n_embeddings", 0) < 1:
                errors.append("fg_plan_summary.json has n_embeddings < 1")
            if payload.get("n_batches", 0) < 1:
                errors.append("fg_plan_summary.json has n_batches < 1")
        except json.JSONDecodeError as exc:
            errors.append(f"fg_plan_summary.json is invalid JSON: {exc}")

    report_fp = artifacts_dir / "id_reconciliation_report.json"
    if report_fp.exists():
        try:
            report = json.loads(report_fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"id_reconciliation_report.json is invalid JSON: {exc}")
        else:
            missing_keys = sorted(REQUIRED_REPORT_KEYS - set(report.keys()))
            if missing_keys:
                errors.append(
                    "id_reconciliation_report.json missing keys: "
                    + ", ".join(missing_keys)
                )
            elif float(report["join_coverage"]) < float(report["min_join_coverage"]):
                errors.append(
                    "join coverage below threshold: "
                    f"{report['join_coverage']} < {report['min_join_coverage']}"
                )

    matrix_fp = artifacts_dir / "modeling_feature_matrix.tsv"
    if matrix_fp.exists():
        rows = matrix_fp.read_text(encoding="utf-8").splitlines()
        if len(rows) < 2:
            errors.append("modeling_feature_matrix.tsv has no data rows")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FG-batch artifact completeness")
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    args = parser.parse_args()

    errors = validate(args.artifacts_dir)
    if errors:
        print("[FAIL] Blue Catalyst FG artifact validation failed", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        return 1

    print(f"[OK] Blue Catalyst FG artifacts validated: {args.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
