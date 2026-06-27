#!/usr/bin/env python3
"""Materialize the MUCC v1 source-scaffold lane as a queryable warehouse.

This does not create final MethaNet-curated mechanism calls. It packages the
validated MUCC source manifests, expression support, source-derived feature
scaffolds, MRV readiness scaffold, and candidate review cards as partitioned
Parquet tables plus a DuckDB catalog so atlas tooling can query the lane
natively while claim boundaries remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
COHORT_RUN_ID = "mucc_v1_owc_wetland_20260626"
CLAIM_BOUNDARY = (
    "MUCC v1 source-scaffold warehouse only; no final MRV risk score, A-E tier, "
    "measured methane flux, carbon-crediting claim, or source-independent "
    "transfer claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--cohort-run-id", default=COHORT_RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=BASE / "cohort_warehouse")
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def read_tsv_df(pd: Any, path: Path) -> Any:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(path, sep="\t", dtype=str, compression=compression).fillna("")


def read_csv_df(pd: Any, path: Path) -> Any:
    return pd.read_csv(path, dtype=str).fillna("")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_cohort(pd: Any, df: Any, cohort_run_id: str) -> Any:
    if "cohort_run_id" not in df.columns:
        df.insert(0, "cohort_run_id", cohort_run_id)
    else:
        df["cohort_run_id"] = cohort_run_id
    return df


def ensure_claim(df: Any, claim_boundary: str = CLAIM_BOUNDARY) -> Any:
    if "claim_boundary" not in df.columns:
        df["claim_boundary"] = claim_boundary
    return df


def build_dim_mag(pd: Any, run_dir: Path, cohort_run_id: str) -> Any:
    manifest = read_tsv_df(pd, run_dir / "manifests/mucc_v1_source_lane_manifest.tsv")
    readiness = read_tsv_df(
        pd,
        run_dir / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv",
    )
    cols = [
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "source_group",
        "analysis_unit_type",
        "local_fna_path",
        "proteome_faa",
        "local_faa_path",
        "local_ffn_path",
        "local_gff_path",
        "payload_status",
        "protein_prediction_status",
        "protein_count",
        "match_status",
        "functional_run_include",
        "esm2_include",
        "glm2_include",
        "mbag_mag_level_include",
        "comparability_status",
        "denominator_status",
        "metadata_mapping_status",
        "gap_reason",
        "recommended_action",
    ]
    dim = manifest[[col for col in cols if col in manifest.columns]].copy()
    readiness_cols = [
        "proteome_id",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "bin_completeness",
        "bin_contamination",
        "source_qc_label",
        "mrv_readiness_label",
        "review_priority_score",
    ]
    dim = dim.merge(
        readiness[[col for col in readiness_cols if col in readiness.columns]],
        on="proteome_id",
        how="left",
    ).fillna("")
    dim = add_cohort(pd, dim, cohort_run_id)
    return ensure_claim(dim)


def table_specs(run_dir: Path) -> list[tuple[str, Path, str]]:
    return [
        ("source_lane_manifest", run_dir / "manifests/mucc_v1_source_lane_manifest.tsv", "tsv"),
        ("functional_manifest", run_dir / "manifests/mucc_v1_functional_manifest.tsv", "tsv"),
        ("glm2_ready_manifest", run_dir / "manifests/mucc_v1_glm2_ready_manifest.tsv", "tsv"),
        ("glm2_ready_gap_register", run_dir / "manifests/mucc_v1_glm2_ready_gap_register.tsv", "tsv"),
        ("mag_catalog", run_dir / "manifests/mucc_v1_mag_catalog_full.tsv", "tsv"),
        (
            "feature_mrv_readiness_mag_level",
            run_dir / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv",
            "tsv",
        ),
        (
            "feature_source_dram_mag_summary",
            run_dir / "functional_features/feature_mucc_v1_source_dram_mag_summary.tsv",
            "tsv",
        ),
        (
            "feature_gene_annotation_mag_summary",
            run_dir / "functional_features/feature_mucc_v1_gene_annotation_mag_summary.tsv",
            "tsv",
        ),
        (
            "feature_mag_expression_summary",
            run_dir / "expression/feature_mucc_v1_expression_mag_summary.tsv",
            "tsv",
        ),
        (
            "feature_gene_expression_mag_summary",
            run_dir / "expression/feature_mucc_v1_gene_expression_mag_summary.tsv",
            "tsv",
        ),
        (
            "fact_mag_expression_sample",
            run_dir / "expression/fact_mucc_v1_expression_mag_sample.tsv.gz",
            "tsv",
        ),
        (
            "fact_gene_expression_mag_sample",
            run_dir / "expression/fact_mucc_v1_gene_expression_mag_sample.tsv.gz",
            "tsv",
        ),
        (
            "candidate_mrv_readiness_cards",
            run_dir / "candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv",
            "tsv",
        ),
        (
            "candidate_strategic_review_cards",
            run_dir / "candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv",
            "tsv",
        ),
        (
            "validation_gap_register",
            run_dir / "reports/validation_gap_register.tsv",
            "tsv",
        ),
        (
            "claim_boundary_matrix",
            run_dir / "reports/claim_boundary_matrix.tsv",
            "tsv",
        ),
    ]


def write_partitioned_table(df: Any, output_dir: Path, table: str, cohort_run_id: str) -> dict[str, Any]:
    table_dir = output_dir / "parquet" / table / f"cohort_run_id={cohort_run_id}"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "part-00000.parquet"
    df.to_parquet(path, index=False)
    return {
        "table": table,
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "bytes": path.stat().st_size,
    }


def build_duckdb(output_dir: Path, manifest_rows: list[dict[str, Any]]) -> Path:
    import duckdb

    db_path = output_dir / "functional_atlas.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for row in manifest_rows:
        table = row["table"]
        parquet_glob = str(output_dir / "parquet" / table / "*" / "*.parquet").replace("'", "''")
        con.execute(f'CREATE VIEW "{table}" AS SELECT * FROM read_parquet(\'{parquet_glob}\')')
    con.close()
    return db_path


def validate(pd: Any, tables: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(gate: str, status: str, detail: str) -> None:
        checks.append({"gate": gate, "status": status, "detail": detail})

    dim_mag = tables["dim_mag"]
    readiness = tables["feature_mrv_readiness_mag_level"]
    source_manifest = tables["source_lane_manifest"]
    mag_catalog = tables["mag_catalog"]
    mag_expr = tables["feature_mag_expression_summary"]
    gene_expr = tables["feature_gene_expression_mag_summary"]
    source_dram = tables["feature_source_dram_mag_summary"]
    glm2_ready = tables["glm2_ready_manifest"]

    add(
        "source_scaffold_warehouse_materialized",
        "pass",
        f"warehouse tables={len(tables)}; dim_mag rows={len(dim_mag)}; DuckDB catalog is built by this script",
    )
    add(
        "dim_mag_rows",
        "pass" if len(dim_mag) == 2508 else "fail",
        f"dim_mag rows={len(dim_mag)} expected=2508",
    )
    add(
        "mag_catalog_rows",
        "warn" if len(mag_catalog) == 2508 else "fail",
        "MAGs.zip local catalog rows=2508; published/deposit HQ/MQ denominator remains 2502 and requires reconciliation",
    )
    add(
        "source_dram_mapping",
        "pass" if len(source_dram) == 2508 else "fail",
        f"source DRAM summary rows={len(source_dram)} expected=2508",
    )
    add(
        "processed_expression_support",
        "pass" if len(mag_expr) == 1948 and len(gene_expr) == 2508 else "warn",
        f"processed MAG expression rows={len(mag_expr)}; gene-expression summary rows={len(gene_expr)}",
    )
    direct_esm2_ready = (
        source_manifest["esm2_include"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).sum()
        if "esm2_include" in source_manifest.columns
        else 0
    )
    add(
        "direct_source_esm2_input_readiness",
        "warn" if direct_esm2_ready == 2501 else "fail",
        f"direct source-protein ESM2-ready rows={direct_esm2_ready}; seven local FASTA entries remain excluded under current parser",
    )
    glm2_ready_rows = (
        glm2_ready["glm2_include"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).sum()
        if "glm2_include" in glm2_ready.columns
        else 0
    )
    add(
        "prodigal_glm2_input_readiness",
        "pass" if glm2_ready_rows == 2508 else "fail",
        f"Prodigal-derived gLM2-ready rows={glm2_ready_rows} expected=2508",
    )
    add(
        "mrv_readiness_rows",
        "pass" if len(readiness) == 2508 else "fail",
        f"feature_mrv_readiness_mag_level rows={len(readiness)} expected=2508",
    )
    add(
        "proteome_id_unique",
        "pass" if dim_mag["proteome_id"].nunique() == len(dim_mag) else "fail",
        f"unique proteome_id={dim_mag['proteome_id'].nunique()} rows={len(dim_mag)}",
    )
    add(
        "claim_boundary_present",
        "pass" if "claim_boundary" in readiness.columns else "fail",
        "MRV readiness table carries claim_boundary column",
    )
    add(
        "final_claim_lock",
        "pass",
        "Warehouse is source-scaffold only and does not authorize final MRV scores, A-E tiers, measured flux, crediting, or transfer claims.",
    )
    add(
        "wetland_neighbor_bridge_table",
        "blocked",
        "Final MUCC wetland-neighbor bridge table remains blocked until ESM2/gLM2 embedding outputs complete and validate.",
    )
    return checks


def markdown_report(
    output_dir: Path,
    cohort_run_id: str,
    manifest_rows: list[dict[str, Any]],
    checks: list[dict[str, str]],
    duckdb_path: Path,
) -> str:
    status = "pass" if all(row["status"] != "fail" for row in checks) else "fail"
    lines = [
        "# MUCC v1 Source-Scaffold Warehouse",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Cohort run ID: `{cohort_run_id}`",
        f"Validation status: `{status}`",
        "",
        "This warehouse materializes validated MUCC v1 source-scaffold artifacts as "
        "partitioned Parquet tables plus a DuckDB catalog. It is queryable atlas "
        "infrastructure, not a final curated MethaNet mechanism warehouse.",
        "",
        f"DuckDB catalog: `{duckdb_path}`",
        "",
        "## Tables",
        "",
        "| Table | Rows | Columns |",
        "| --- | ---: | ---: |",
    ]
    for row in manifest_rows:
        lines.append(f"| `{row['table']}` | {row['rows']} | {row['columns']} |")
    lines.extend(
        [
            "",
            "## Validation Gates",
            "",
            "| Gate | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for row in checks:
        lines.append(f"| `{row['gate']}` | {row['status']} | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            CLAIM_BOUNDARY,
            "",
            "The warehouse preserves missing, pending, scaffolded, and blocked evidence as "
            "explicit rows. ESM2/gLM2 embeddings and wetland-neighbor bridge tables remain "
            "external completion gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> int:
    import pandas as pd

    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    output_dir = resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, Any] = {"dim_mag": build_dim_mag(pd, run_dir, args.cohort_run_id)}
    for table, path, kind in table_specs(run_dir):
        if not path.exists():
            continue
        df = read_tsv_df(pd, path) if kind == "tsv" else read_csv_df(pd, path)
        df = add_cohort(pd, df, args.cohort_run_id)
        tables[table] = ensure_claim(df)

    manifest_rows = [
        write_partitioned_table(df, output_dir, table, args.cohort_run_id)
        for table, df in sorted(tables.items())
    ]
    write_tsv(output_dir / "cohort_table_manifest.tsv", manifest_rows)
    duckdb_path = build_duckdb(output_dir, manifest_rows)
    checks = validate(pd, tables)
    write_tsv(output_dir / "validation_gates.tsv", checks)
    (output_dir / "DATA_ARCHITECTURE_VALIDATION.md").write_text(
        markdown_report(output_dir, args.cohort_run_id, manifest_rows, checks, duckdb_path)
    )
    summary = {
        "output_dir": str(output_dir.relative_to(repo_root)),
        "cohort_run_id": args.cohort_run_id,
        "tables": len(manifest_rows),
        "duckdb": str(duckdb_path.relative_to(repo_root)),
        "validation_status": "pass" if all(row["status"] != "fail" for row in checks) else "fail",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / "warehouse_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["validation_status"] == "pass" else 1


def main() -> int:
    return build(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
