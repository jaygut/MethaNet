#!/usr/bin/env python3
"""Build a conflict-preserving, atlas-native MUCC v1 taxonomy projection.

The source annotation lineage remains primary at every rank.  Public KBase
GTDB 214.1 values can fill only ranks missing from the source annotation.  The
projection retains both complete source lineages, both values at each rank,
rank-level provenance, and explicit disagreement counts so a convenient atlas
view never silently turns a taxonomy update into a source correction.

This is MAG/proteome-level context only. It does not establish MAG quality,
community abundance, sample ecology, methane flux, or MRV risk.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
RANKS = (
    ("domain", "d__"),
    ("phylum", "p__"),
    ("class", "c__"),
    ("order", "o__"),
    ("family", "f__"),
    ("genus", "g__"),
    ("species", "s__"),
)
CLAIM_BOUNDARY = (
    "Rank-aware source/KBase taxonomy projection for MAG/proteome-level atlas "
    "context only. Source taxonomy remains primary; KBase fills missing ranks. "
    "The projection does not establish a published MUCC HQ/MQ quality tier, "
    "sample ecology, measured methane flux, final MRV score/A-E tier, "
    "carbon-crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            BASE
            / "source_audit/kbase_public_workspace_147022/"
            "mucc_v1_kbase_public_catalog_reconciliation.tsv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "functional_features/feature_mucc_v1_taxonomy_projection.tsv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=BASE / "reconciliation/mucc_v1_taxonomy_projection_summary.tsv",
    )
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def normalize_lineage(value: str) -> str:
    return ";".join(part.strip() for part in str(value or "").split(";") if part.strip())


def lineage_by_rank(value: str) -> dict[str, str]:
    lineage = {rank: "" for rank, _ in RANKS}
    for part in normalize_lineage(value).split(";"):
        for rank, prefix in RANKS:
            if part.startswith(prefix):
                # GTDB's bare rank placeholders (for example ``g__``) are
                # unknown values, not an observed source rank that should
                # block a more specific KBase fallback.
                lineage[rank] = "" if part == prefix else part
                break
    return lineage


def taxonomy_lineage(values: dict[str, str]) -> str:
    return ";".join(values[rank] for rank, _ in RANKS if values[rank])


def rank_projection(
    source: dict[str, str], kbase: dict[str, str]
) -> tuple[dict[str, str], dict[str, str], int, int]:
    """Use source rank values first, tracking fallback and disagreement explicitly."""
    atlas: dict[str, str] = {}
    provenance: dict[str, str] = {}
    kbase_fallbacks = 0
    disagreements = 0
    for rank, _ in RANKS:
        source_value = source[rank]
        kbase_value = kbase[rank]
        if source_value:
            atlas[rank] = source_value
            provenance[rank] = "source_annotation_primary"
            if kbase_value and kbase_value != source_value:
                disagreements += 1
        elif kbase_value:
            atlas[rank] = kbase_value
            provenance[rank] = "KBase_GTDB_214_1_fallback"
            kbase_fallbacks += 1
        else:
            atlas[rank] = ""
            provenance[rank] = "unavailable"
    return atlas, provenance, kbase_fallbacks, disagreements


def projection_status(
    source_lineage: str,
    kbase_lineage: str,
    kbase_fallbacks: int,
    disagreements: int,
) -> str:
    if not source_lineage and not kbase_lineage:
        return "taxonomy_unavailable_in_source_and_KBase"
    if not source_lineage:
        return "KBase_GTDB_214_1_supplemental_taxonomy_only"
    if not kbase_lineage:
        return "source_annotation_taxonomy_only_KBase_absent"
    if disagreements:
        return "source_primary_with_explicit_KBase_rank_disagreements"
    if kbase_fallbacks:
        return "source_primary_with_KBase_missing_rank_fallback"
    return "source_and_KBase_taxonomy_agree_at_observed_ranks"


def build_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_row in sorted(source_rows, key=lambda row: row["proteome_id"]):
        source_lineage = normalize_lineage(source_row.get("source_bin_taxonomy", ""))
        kbase_lineage = normalize_lineage(source_row.get("kbase_gtdb_lineage", ""))
        source = lineage_by_rank(source_lineage)
        kbase = lineage_by_rank(kbase_lineage)
        atlas, provenance, kbase_fallbacks, disagreements = rank_projection(source, kbase)
        row = {
            "lane_id": source_row.get("lane_id", LANE_ID),
            "proteome_id": source_row["proteome_id"],
            "mag_id": source_row["mag_id"],
            "source_annotation_lineage": source_lineage,
            "kbase_gtdb_214_1_lineage": kbase_lineage,
            "kbase_gtdb_source_version": source_row.get("kbase_gtdb_source_version", ""),
            "kbase_roster_reconciliation_status": source_row.get(
                "kbase_roster_reconciliation_status", ""
            ),
            "source_kbase_taxonomy_reconciliation_status": source_row.get(
                "taxonomy_reconciliation_status", ""
            ),
            "atlas_taxonomy_lineage": taxonomy_lineage(atlas),
            "atlas_taxonomy_projection_status": projection_status(
                source_lineage, kbase_lineage, kbase_fallbacks, disagreements
            ),
            "atlas_taxonomy_policy": (
                "source_annotation_rank_primary;KBase_GTDB_214_1_fills_only_missing_rank"
            ),
            "source_kbase_rank_disagreement_count": str(disagreements),
            "kbase_rank_fallback_count": str(kbase_fallbacks),
            "atlas_taxonomy_available_rank_count": str(sum(bool(atlas[rank]) for rank, _ in RANKS)),
            "atlas_taxonomy_rank_provenance_json": json.dumps(
                provenance, sort_keys=True, separators=(",", ":")
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for rank, _ in RANKS:
            row[f"source_{rank}"] = source[rank]
            row[f"kbase_{rank}"] = kbase[rank]
            row[f"atlas_{rank}"] = atlas[rank]
            row[f"atlas_{rank}_provenance"] = provenance[rank]
        rows.append(row)
    return rows


def summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        raise ValueError("cannot summarize an empty taxonomy projection")
    status = Counter(row["atlas_taxonomy_projection_status"] for row in rows)
    source_kbase = Counter(
        row["source_kbase_taxonomy_reconciliation_status"] for row in rows
    )
    kbase_fallbacks = sum(int(row["kbase_rank_fallback_count"]) for row in rows)
    disagreements = sum(int(row["source_kbase_rank_disagreement_count"]) for row in rows)
    atlas_taxonomy = sum(bool(row["atlas_taxonomy_lineage"]) for row in rows)
    metrics = [
        (
            "projection_rows",
            len(rows),
            "pass",
            "one row per archive MAG/proteome with no MAGs dropped",
        ),
        (
            "unique_proteome_id",
            len({row["proteome_id"] for row in rows}),
            "pass" if len({row["proteome_id"] for row in rows}) == len(rows) else "fail",
            "canonical atlas MAG key",
        ),
        (
            "atlas_taxonomy_available_MAGs",
            atlas_taxonomy,
            "partial" if atlas_taxonomy < len(rows) else "pass",
            "source-primary/KBase-fallback lineage available",
        ),
        (
            "KBase_rank_fallbacks",
            kbase_fallbacks,
            "informational",
            "rank values filled only where source annotation rank is empty",
        ),
        (
            "source_KBase_rank_disagreements",
            disagreements,
            "informational",
            "all disagreements remain explicit; source rank is retained as atlas projection value",
        ),
    ]
    for name, count in sorted(status.items()):
        metrics.append(
            (
                f"projection_status:{name}",
                count,
                "informational",
                "explicit row-level taxonomy projection status",
            )
        )
    for name, count in sorted(source_kbase.items()):
        metrics.append(
            (
                f"source_KBase_status:{name}",
                count,
                "informational",
                "upstream source/KBase taxonomy reconciliation status",
            )
        )
    return [
        {
            "lane_id": LANE_ID,
            "metric": metric,
            "value": str(value),
            "status": status_value,
            "detail": detail,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for metric, value, status_value, detail in metrics
    ]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    input_path = resolve(repo_root, args.input)
    output_path = resolve(repo_root, args.output)
    summary_path = resolve(repo_root, args.summary_output)
    with input_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "proteome_id",
        "mag_id",
        "source_bin_taxonomy",
        "kbase_gtdb_lineage",
        "taxonomy_reconciliation_status",
    }
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError(f"KBase reconciliation input lacks required columns: {input_path}")
    if len({row["proteome_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("KBase reconciliation input has duplicate proteome_id values")
    rows = build_rows(source_rows)
    summaries = summary_rows(rows)
    write_tsv(output_path, rows)
    write_tsv(summary_path, summaries)
    print(
        json.dumps(
            {
                "projection_rows": len(rows),
                "output": str(output_path),
                "summary": str(summary_path),
                "atlas_taxonomy_available_MAGs": sum(
                    bool(row["atlas_taxonomy_lineage"]) for row in rows
                ),
                "source_KBase_rank_disagreements": sum(
                    int(row["source_kbase_rank_disagreement_count"]) for row in rows
                ),
                "KBase_rank_fallbacks": sum(
                    int(row["kbase_rank_fallback_count"]) for row in rows
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
