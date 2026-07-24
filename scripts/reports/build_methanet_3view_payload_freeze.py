#!/usr/bin/env python3
"""Build a dated MethaNet 3-view payload freeze decision.

The freeze contract is intentionally conservative. It lists unit-level ESM2,
gLM2, and functional-annotation availability for registered atlas lanes and
marks whether the requested lanes are actually green. A blocked decision is a
valid monitor artifact, but it is not a release freeze for the next atlas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_atlas_lane_registry as lane_registry


TRUE_VALUES = {"true", "1", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lane-registry", type=Path, default=Path("configs/methanet_atlas_lanes.tsv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--exclusion-tsv",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional TSV of explicitly release-excluded units. Required columns: "
            "lane_id, proteome_id. Recommended columns: exclusion_reason, "
            "exclusion_scope, approved_by, approved_at_utc."
        ),
    )
    parser.add_argument(
        "--exclude-unit",
        action="append",
        default=[],
        metavar="LANE_ID:PROTEOME_ID:REASON",
        help=(
            "Inline release exclusion. The first two colon-delimited fields are "
            "lane_id and proteome_id; the remainder is the reason."
        ),
    )
    parser.add_argument(
        "--require-green-lane",
        action="append",
        default=["msm_china_2025"],
        help=(
            "Lane that must be fully tri-view complete before freeze_allowed=true. "
            "Can be supplied multiple times; defaults to msm_china_2025."
        ),
    )
    parser.add_argument(
        "--exit-nonzero-if-blocked",
        action="store_true",
        help="Exit with status 2 when required lanes are not green.",
    )
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def resolve(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def resolve_many(repo_root: Path, value: str | None) -> list[Path]:
    return lane_registry.resolve_many(repo_root, value)


def read_tsv(path: Path | None) -> list[dict[str, str]]:
    return lane_registry.read_tsv(path)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def release_exclusion_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("lane_id", "")).strip(), str(row.get("proteome_id", "")).strip())


def read_release_exclusions(
    repo_root: Path,
    paths: list[Path],
    inline_specs: list[str],
) -> dict[tuple[str, str], dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        resolved = resolve(repo_root, str(path)) or path
        rows.extend(read_tsv(resolved))
    generated_at = datetime.now(timezone.utc).isoformat()
    for spec in inline_specs:
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise SystemExit(f"--exclude-unit must be LANE_ID:PROTEOME_ID[:REASON], got: {spec}")
        rows.append(
            {
                "lane_id": parts[0],
                "proteome_id": parts[1],
                "exclusion_reason": parts[2] if len(parts) == 3 else "release-approved exclusion",
                "exclusion_scope": "report_release_denominator",
                "approved_by": "operator",
                "approved_at_utc": generated_at,
            }
        )

    exclusions: dict[tuple[str, str], dict[str, str]] = {}
    for idx, row in enumerate(rows, start=1):
        lane_id = str(row.get("lane_id", "")).strip()
        proteome_id = str(row.get("proteome_id", "")).strip()
        if not lane_id or not proteome_id:
            raise SystemExit(f"Release exclusion row {idx} is missing lane_id or proteome_id")
        key = (lane_id, proteome_id)
        if key in exclusions:
            raise SystemExit(f"Duplicate release exclusion for {lane_id}:{proteome_id}")
        exclusions[key] = {
            "exclusion_reason": row.get("exclusion_reason") or row.get("reason") or "release-approved exclusion",
            "exclusion_scope": row.get("exclusion_scope") or "report_release_denominator",
            "approved_by": row.get("approved_by") or "operator",
            "approved_at_utc": row.get("approved_at_utc") or generated_at,
        }
    return exclusions


def apply_release_exclusions(
    rows: list[dict[str, Any]],
    exclusions: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[tuple[str, str]] = set()
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        key = release_exclusion_key(row)
        seen.add(key)
        exclusion = exclusions.get(key)
        excluded = exclusion is not None
        row["release_excluded"] = str(excluded).lower()
        row["release_required"] = str(not excluded).lower()
        row["release_exclusion_reason"] = exclusion.get("exclusion_reason", "") if exclusion else ""
        row["release_exclusion_scope"] = exclusion.get("exclusion_scope", "") if exclusion else ""
        row["release_exclusion_approved_by"] = exclusion.get("approved_by", "") if exclusion else ""
        row["release_exclusion_approved_at_utc"] = exclusion.get("approved_at_utc", "") if exclusion else ""
        if excluded:
            excluded_rows.append(dict(row))

    missing = sorted(key for key in exclusions if key not in seen)
    if missing:
        formatted = ", ".join(f"{lane_id}:{proteome_id}" for lane_id, proteome_id in missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... plus {len(missing) - 10} more"
        raise SystemExit(f"Release exclusions did not match freeze manifest rows: {formatted}{suffix}")
    return rows, excluded_rows


def esm2_ids(paths: list[Path], expected_ids: set[str]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for metadata_path in [
            path / "embedding_metadata.tsv",
            path / "embedding_checkpoints/checkpoint_metadata.tsv",
        ]:
            for row in read_tsv(metadata_path):
                proteome_id = row.get("proteome_id") or row.get("sample")
                if proteome_id:
                    ids.add(str(proteome_id))
    return ids & expected_ids if expected_ids else ids


def glm2_ids(paths: list[Path], expected_ids: set[str]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for candidate in [
            path / "feature_glm_mag_level.tsv",
            path / "features/glm2_smoke_window_embedding_summary.tsv",
            path / "manifests/glm2_smoke_mag_manifest.tsv",
            path / "manifests/glm2_multiwindow_manifest.tsv",
        ]:
            for row in read_tsv(candidate):
                proteome_id = row.get("proteome_id") or row.get("unit_id")
                if proteome_id:
                    ids.add(str(proteome_id))
    return ids & expected_ids if expected_ids else ids


def functional_evidence_contract(
    repo_root: Path,
    registry_row: dict[str, str],
) -> dict[str, str]:
    """Classify functional evidence without equating source scaffolds to curated mechanisms."""
    warehouse_tables: set[str] = set()
    for warehouse_dir in resolve_many(repo_root, registry_row.get("functional_warehouse_dir")):
        for row in read_tsv(warehouse_dir / "cohort_table_manifest.tsv"):
            table = row.get("table")
            if table:
                warehouse_tables.add(str(table))
    if "feature_source_dram_mag_summary" in warehouse_tables:
        return {
            "functional_evidence_class": "source_annotation_scaffold",
            "functional_harmonization_status": (
                "common_screening_axes_harmonized_with_source_scaffold_caveat"
            ),
            "mechanism_equivalence_status": "not_canonical_mechanism_equivalent",
        }
    return {
        "functional_evidence_class": "canonical_curated_mechanism_features",
        "functional_harmonization_status": "canonical_feature_contract",
        "mechanism_equivalence_status": "mechanism_equivalent",
    }


def functional_status_by_id(
    per_mag_dirs: list[Path],
    expected_ids: set[str],
    warehouse_current: bool,
) -> dict[str, dict[str, str]]:
    if warehouse_current:
        return {
            proteome_id: {
                "functional_status": "complete",
                "functional_status_basis": "warehouse_dim_mag",
                "selected_run_dir": "",
            }
            for proteome_id in expected_ids
        }

    out: dict[str, dict[str, str]] = {}
    for per_mag_dir in per_mag_dirs:
        if not per_mag_dir.exists():
            continue
        for proteome_dir in sorted(path for path in per_mag_dir.iterdir() if path.is_dir()):
            proteome_id = proteome_dir.name
            if expected_ids and proteome_id not in expected_ids:
                continue
            runs = sorted(path for path in proteome_dir.iterdir() if path.is_dir())
            if not runs:
                continue
            complete_runs = [path for path in runs if lane_registry.run_dir_status(path) == "complete"]
            if complete_runs:
                out[proteome_id] = {
                    "functional_status": "complete",
                    "functional_status_basis": "live_per_mag",
                    "selected_run_dir": str(complete_runs[-1]),
                }
                continue
            status = lane_registry.run_dir_status(runs[-1])
            previous = out.get(proteome_id, {})
            if previous.get("functional_status") == "complete":
                continue
            out[proteome_id] = {
                "functional_status": status,
                "functional_status_basis": "live_per_mag",
                "selected_run_dir": str(runs[-1]),
            }
    for proteome_id in expected_ids:
        out.setdefault(
            proteome_id,
            {
                "functional_status": "not_started",
                "functional_status_basis": "live_per_mag",
                "selected_run_dir": "",
            },
        )
    return out


def expected_lane_rows(
    source_rows: list[dict[str, str]],
    functional_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if functional_rows:
        rows = functional_rows
    else:
        rows = source_rows
    if rows and "functional_run_include" in rows[0]:
        return [row for row in rows if truthy(row.get("functional_run_include"))]
    return rows


def lane_unit_rows(repo_root: Path, registry_row: dict[str, str], status_row: dict[str, Any]) -> list[dict[str, Any]]:
    lane_id = registry_row.get("lane_id", "")
    source_rows = read_tsv(resolve(repo_root, registry_row.get("source_lane_manifest")))
    functional_rows = read_tsv(resolve(repo_root, registry_row.get("functional_manifest")))
    expected_rows = expected_lane_rows(source_rows, functional_rows)
    expected_ids = {row.get("proteome_id", "") for row in expected_rows if row.get("proteome_id")}
    source_by_id = {row.get("proteome_id", ""): row for row in source_rows if row.get("proteome_id")}

    esm_ids = esm2_ids(resolve_many(repo_root, registry_row.get("esm2_artifacts_dirs")), expected_ids)
    glm_ids = glm2_ids(resolve_many(repo_root, registry_row.get("glm2_artifacts_dirs")), expected_ids)
    warehouse_current = str(status_row.get("warehouse_current") or "").lower() == "true"
    functional = functional_status_by_id(
        resolve_many(repo_root, registry_row.get("functional_per_mag_dirs")),
        expected_ids,
        warehouse_current,
    )
    evidence_contract = functional_evidence_contract(repo_root, registry_row)

    rows: list[dict[str, Any]] = []
    for row in expected_rows:
        proteome_id = row.get("proteome_id", "")
        source_row = source_by_id.get(proteome_id, {})
        func = functional.get(proteome_id, {})
        has_esm2 = proteome_id in esm_ids
        has_glm2 = proteome_id in glm_ids
        has_functional = func.get("functional_status") == "complete"
        tri_view_ready = has_esm2 and has_glm2 and has_functional
        mechanism_equivalent = (
            evidence_contract["mechanism_equivalence_status"] == "mechanism_equivalent"
        )
        if not tri_view_ready:
            formal_tri_view_status = "incomplete_tri_view"
        elif mechanism_equivalent:
            formal_tri_view_status = "complete_canonical_mechanism_tri_view"
        else:
            formal_tri_view_status = "complete_source_scaffold_tri_view"
        rows.append(
            {
                "lane_id": lane_id,
                "lane_role": registry_row.get("lane_role", ""),
                "proteome_id": proteome_id,
                "mag_id": row.get("mag_id") or row.get("mag_id_candidate") or source_row.get("mag_id") or proteome_id,
                "source": row.get("source") or source_row.get("source") or lane_id,
                "ecosystem": row.get("ecosystem") or source_row.get("ecosystem") or "",
                "domain": row.get("domain") or source_row.get("domain") or "",
                "has_esm2": str(has_esm2).lower(),
                "has_glm2": str(has_glm2).lower(),
                "has_functional": str(has_functional).lower(),
                "tri_view_ready": str(tri_view_ready).lower(),
                **evidence_contract,
                "formal_tri_view_status": formal_tri_view_status,
                "mechanism_equivalent_tri_view": str(
                    tri_view_ready and mechanism_equivalent
                ).lower(),
                "functional_status": func.get("functional_status", "not_started"),
                "functional_status_basis": func.get("functional_status_basis", ""),
                "selected_run_dir": func.get("selected_run_dir", ""),
                "claim_scope": registry_row.get("claim_scope", ""),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], registry_status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status_by_lane = {row.get("lane_id", ""): row for row in registry_status_rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["lane_id"]), []).append(row)
    out: list[dict[str, Any]] = []
    for lane_id, lane_rows in grouped.items():
        counts = Counter(row["functional_status"] for row in lane_rows)
        tri_view = sum(1 for row in lane_rows if truthy(row.get("tri_view_ready")))
        canonical_tri_view = sum(
            1 for row in lane_rows if truthy(row.get("mechanism_equivalent_tri_view"))
        )
        source_scaffold_tri_view = sum(
            1
            for row in lane_rows
            if row.get("formal_tri_view_status") == "complete_source_scaffold_tri_view"
        )
        release_required = sum(1 for row in lane_rows if truthy(row.get("release_required")))
        release_excluded = sum(1 for row in lane_rows if truthy(row.get("release_excluded")))
        release_tri_view = sum(
            1
            for row in lane_rows
            if truthy(row.get("release_required")) and truthy(row.get("tri_view_ready"))
        )
        status = status_by_lane.get(lane_id, {})
        out.append(
            {
                "lane_id": lane_id,
                "expected_units": len(lane_rows),
                "release_required_units": release_required,
                "release_excluded_units": release_excluded,
                "esm2_units": sum(1 for row in lane_rows if truthy(row.get("has_esm2"))),
                "glm2_units": sum(1 for row in lane_rows if truthy(row.get("has_glm2"))),
                "functional_complete": counts.get("complete", 0),
                "functional_partial": counts.get("partial", 0),
                "functional_failed": counts.get("failed", 0),
                "functional_not_started": counts.get("not_started", 0),
                "tri_view_ready_units": tri_view,
                "canonical_mechanism_tri_view_units": canonical_tri_view,
                "source_scaffold_tri_view_units": source_scaffold_tri_view,
                "release_tri_view_ready_units": release_tri_view,
                "registry_denominator_units": status.get("denominator_units", ""),
                "registry_manifest_gap_rows": status.get("manifest_gap_rows", ""),
                "warehouse_current": status.get("warehouse_current", ""),
            }
        )
    return out


def markdown(
    stamp: str,
    freeze_allowed: bool,
    required_lanes: list[str],
    summary_rows: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# MethaNet 3-View Payload Freeze Decision",
        "",
        f"Generated UTC: `{stamp}`",
        f"Freeze allowed: `{'true' if freeze_allowed else 'false'}`",
        f"Required green lanes: `{', '.join(required_lanes)}`",
        "",
        "| Lane | Expected | Release-required | Excluded | ESM2 | gLM2 | Functional | Tri-view | Canonical mechanism | Source scaffold | Release tri-view | Partial | Failed | Not started |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {lane_id} | {expected_units:,} | {release_required_units:,} | "
            "{release_excluded_units:,} | {esm2_units:,} | {glm2_units:,} | "
            "{functional_complete:,} | {tri_view_ready_units:,} | "
            "{canonical_mechanism_tri_view_units:,} | {source_scaffold_tri_view_units:,} | "
            "{release_tri_view_ready_units:,} | {functional_partial:,} | "
            "{functional_failed:,} | {functional_not_started:,} |".format(
                **{key: int(value) if str(value).isdigit() else value for key, value in row.items()}
            )
        )
    lines.extend(["", "## Decision", ""])
    if freeze_allowed:
        if excluded_rows:
            lines.append(
                "All non-excluded required-lane units are green for this release freeze. "
                "Release exclusions are preserved below and in `excluded_units.tsv`."
            )
        else:
            lines.append("All required lanes are green for their registered functional denominators.")
    else:
        lines.append("At least one required lane is not green. Do not use this snapshot as the final report freeze.")
    if blockers:
        lines.extend(["", "## Blocking Units", ""])
        for row in blockers[:25]:
            lines.append(
                f"- `{row['lane_id']}` `{row['proteome_id']}`: "
                f"ESM2={row['has_esm2']}, gLM2={row['has_glm2']}, "
                f"functional={row['functional_status']}"
            )
        if len(blockers) > 25:
            lines.append(f"- ... plus `{len(blockers) - 25}` additional blockers.")
    if excluded_rows:
        lines.extend(["", "## Release Exclusions", ""])
        for row in excluded_rows[:25]:
            lines.append(
                f"- `{row['lane_id']}` `{row['proteome_id']}`: "
                f"{row.get('release_exclusion_reason', 'release-approved exclusion')}"
            )
        if len(excluded_rows) > 25:
            lines.append(f"- ... plus `{len(excluded_rows) - 25}` additional exclusions.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This freeze operates at MAG/proteome grain. It supports molecular "
            "atlas/report rebuild decisions, not final sample-level MRV risk tiers, "
            "measured methane flux, or carbon-credit approval.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or repo_root / "results/reports" / f"methanet_3view_payload_freeze_{stamp}"
    output_dir = resolve(repo_root, str(output_dir)) or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = resolve(repo_root, str(args.lane_registry)) or args.lane_registry
    registry_rows = read_tsv(registry_path)
    registry_status_rows = [lane_registry.summarize_lane(repo_root, row) for row in registry_rows]
    status_by_lane = {row.get("lane_id", ""): row for row in registry_status_rows}

    unit_rows: list[dict[str, Any]] = []
    for row in registry_rows:
        unit_rows.extend(lane_unit_rows(repo_root, row, status_by_lane.get(row.get("lane_id", ""), {})))

    exclusions = read_release_exclusions(repo_root, args.exclusion_tsv, args.exclude_unit)
    unit_rows, excluded_rows = apply_release_exclusions(unit_rows, exclusions)
    summary_rows = summarize(unit_rows, registry_status_rows)
    required_lanes = list(dict.fromkeys(args.require_green_lane or []))
    blockers = [
        row
        for row in unit_rows
        if (
            row.get("lane_id") in required_lanes
            and truthy(row.get("release_required"))
            and not truthy(row.get("tri_view_ready"))
        )
    ]
    freeze_allowed = not blockers

    write_tsv(
        output_dir / "freeze_manifest.tsv",
        unit_rows,
        [
            "lane_id",
            "lane_role",
            "proteome_id",
            "mag_id",
            "source",
            "ecosystem",
            "domain",
            "has_esm2",
            "has_glm2",
            "has_functional",
            "tri_view_ready",
            "functional_evidence_class",
            "functional_harmonization_status",
            "mechanism_equivalence_status",
            "formal_tri_view_status",
            "mechanism_equivalent_tri_view",
            "release_required",
            "release_excluded",
            "release_exclusion_reason",
            "release_exclusion_scope",
            "release_exclusion_approved_by",
            "release_exclusion_approved_at_utc",
            "functional_status",
            "functional_status_basis",
            "selected_run_dir",
            "claim_scope",
        ],
    )
    write_tsv(output_dir / "freeze_summary.tsv", summary_rows, list(summary_rows[0]) if summary_rows else [])
    write_tsv(output_dir / "blocking_units.tsv", blockers, list(unit_rows[0]) if unit_rows else [])
    write_tsv(output_dir / "excluded_units.tsv", excluded_rows, list(unit_rows[0]) if unit_rows else [])
    decision = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_allowed": freeze_allowed,
        "required_green_lanes": required_lanes,
        "unit_rows": len(unit_rows),
        "tri_view_ready_units": sum(1 for row in unit_rows if truthy(row.get("tri_view_ready"))),
        "canonical_mechanism_tri_view_units": sum(
            1 for row in unit_rows if truthy(row.get("mechanism_equivalent_tri_view"))
        ),
        "source_scaffold_tri_view_units": sum(
            1
            for row in unit_rows
            if row.get("formal_tri_view_status") == "complete_source_scaffold_tri_view"
        ),
        "release_required_units": sum(1 for row in unit_rows if truthy(row.get("release_required"))),
        "release_tri_view_ready_units": sum(
            1
            for row in unit_rows
            if truthy(row.get("release_required")) and truthy(row.get("tri_view_ready"))
        ),
        "release_excluded_units": len(excluded_rows),
        "blocking_units": len(blockers),
        "outputs": {
            "freeze_manifest": str(output_dir / "freeze_manifest.tsv"),
            "freeze_summary": str(output_dir / "freeze_summary.tsv"),
            "blocking_units": str(output_dir / "blocking_units.tsv"),
            "excluded_units": str(output_dir / "excluded_units.tsv"),
            "decision_md": str(output_dir / "FREEZE_DECISION.md"),
        },
    }
    (output_dir / "freeze_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    (output_dir / "registry_status_snapshot.json").write_text(json.dumps(registry_status_rows, indent=2, sort_keys=True))
    (output_dir / "FREEZE_DECISION.md").write_text(
        markdown(decision["generated_utc"], freeze_allowed, required_lanes, summary_rows, blockers, excluded_rows)
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    if args.exit_nonzero_if_blocked and not freeze_allowed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
