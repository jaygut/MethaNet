"""Read-only physical source audit for the bounded molecular expansion.

No source is modified. File row ordinals identify source events, not biological
entities. Natural keys are tested independently; no row-count equality is used
as a uniqueness test. Admission remains an adapter-specific decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .model import digest


def read_tsv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def write_tsv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def local_path(repo, value):
    """Explicit supported legacy root rebasing; never search by basename."""
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    elif not path.is_relative_to(repo):
        marker = "/Jay_Proyects/MethaNet/"
        if marker not in str(path):
            raise ValueError(f"Unresolved foreign root: {path}")
        path = repo / str(path).split(marker, 1)[1]
    path = path.resolve()
    if not path.is_relative_to(repo):
        raise ValueError("Source escaped repository")
    return path


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def natural_key(name, columns):
    """Declared candidate keys, not guesses made from observed row counts."""
    common = ["cohort_run_id", "run_id", "proteome_id"]
    scoped = [x for x in common if x in columns]
    exact = {
        "dim_gene": scoped + ["source_tool", "gene_id"],
        "dim_mag": ["proteome_id"],
        "fact_run_status": scoped + ["run_dir"],
        "feature_annotation_coverage": scoped + ["annotation_tool", "source_table"],
        "fact_kofam_hits": scoped + ["gene_id", "ko_id"],
        "fact_mcycdb_hits": scoped + ["gene_id", "subject_id", "hit_rank_bitscore"],
        "fact_scycdb_hits": scoped + ["gene_id", "subject_id", "hit_rank_bitscore"],
        "fact_bakta_features": scoped + ["Sequence Id", "Type", "Start", "Stop", "Strand", "Locus Tag"],
        "fact_dbcan_hits": scoped + ["Gene ID"],
        "fact_metabolic_hmm_hits": scoped + ["function_category", "function_name", "gene_abbreviation", "hmm_file"],
        "fact_metabolic_function_presence": scoped + ["function_category", "function_name", "gene_abbreviation"],
        "fact_metabolic_module_presence": scoped + ["module_id"],
        "fact_metabolic_module_step_presence": scoped + ["module_step_id", "ko_id"],
        "fact_qc_checkm2": scoped,
        "fact_qc_gunc": scoped,
        "fact_taxonomy_gtdbtk": scoped,
        "fact_input_stats": scoped + ["metric"],
        "run_summary_metrics": scoped + ["metric"],
        "fact_tool_timing": scoped + ["step", "start_epoch"],
        "fact_cazy_hits": scoped + ["cazy_family"],
        "fact_merops_hits": scoped + ["merops_peptidase_id"],
        "fact_gene_expression_mag_sample": ["proteome_id", "sample_column"],
        "fact_mag_expression_sample": ["proteome_id", "sample_id"],
    }
    if name in exact:
        return exact[name]
    for key in ("flux_observation_id", "porewater_observation_id", "card_id", "gap_id"):
        if key in columns:
            return [key]
    if name.startswith("link_mucc_v1_sequence_") or name in (
        "feature_mucc_v1_sample_ecological_readiness", "feature_mucc_v1_sample_methods_design_context",
        "fact_mucc_v1_wgcna_secondary_module_eigengenes"):
        return ["sample_id"]
    if name in ("functional_manifest", "source_lane_manifest", "mag_catalog", "glm2_ready_manifest",
                "mucc_v1_mag_reconciliation", "mucc_v1_zenodo_source_qc_reconciliation"):
        return ["proteome_id"]
    if name.startswith("feature_") and "proteome_id" in columns and "sample_id" not in columns:
        return ["proteome_id"]
    if name.endswith("validation_gates") or name.endswith("promotion_gates"):
        return ["gate"]
    return []


def key_stats(con, relation, keys):
    if not keys:
        return {"status": "not_declared", "duplicate_rows": None, "null_or_blank_key_rows": None}
    cols = ",".join(quote(x) for x in keys)
    null = " OR ".join(f"{quote(x)} IS NULL OR trim(CAST({quote(x)} AS VARCHAR))=''" for x in keys)
    n, missing = con.execute(f"SELECT count(*),count(*) FILTER (WHERE {null}) FROM {relation}").fetchone()
    unique = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT {cols} FROM {relation})").fetchone()[0]
    duplicates = n - unique
    return {"status": "pass" if not duplicates and not missing else "not_unique_or_incomplete",
            "duplicate_rows": duplicates, "null_or_blank_key_rows": missing, "distinct_keys": unique}


def table_semantics(name):
    if name == "dim_gene":
        return "Bakta feature; not Prodigal annotation query namespace; caller crosswalk required"
    if name == "fact_kofam_hits":
        return "Raw KO alignment event; only accepted_hit=true supports candidate detection"
    if name in ("fact_mcycdb_hits", "fact_scycdb_hits"):
        return "Alignment event; rank=1 is best-ranked similarity, not validated enzyme function"
    if "expression" in name:
        return "Source-processed RNA feature; not DNA abundance or raw read counts"
    if "flux" in name:
        return "Typed source observation; no molecular sample pairing inferred"
    if "neighbor" in name:
        return "Historical embedding similarity; model/pooling compatibility must be checked before use"
    return "Source table record; biological grain determined by source fields and adapter contract"


def audit(repo: Path, out: Path):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((repo / "configs/atlas_current_release.json").read_text())
    freeze = read_tsv(repo / pointer["freeze_manifest"])
    release_units = {(x["lane_id"], x["proteome_id"]) for x in freeze}
    con = duckdb.connect()
    con.execute("SET memory_limit='768MB'")
    con.execute("SET threads=2")
    # Spill artifacts are local task outputs, never the source warehouses.
    con.execute("SET temp_directory=?", [str(out / "duckdb-spill")])
    con.execute("SET preserve_insertion_order=false")
    results, sources, issues, sequence_candidates = [], [], [], []
    for lane in read_tsv(repo / pointer["lane_registry"]):
        lid = lane["lane_id"]
        base = repo / lane["functional_warehouse_dir"]
        manifest = base / "cohort_table_manifest.tsv"
        tabs = read_tsv(manifest)
        sources.append({"lane_id": lid, "path": str(manifest.relative_to(repo)), "sha256": digest(manifest)})
        magfile = local_path(repo, next(x for x in tabs if x["table"] == "dim_mag")["path"])
        mags = pq.ParquetFile(magfile).read().to_pylist()
        con.execute("CREATE OR REPLACE TEMP TABLE admitted_mag AS SELECT * FROM read_parquet(?, hive_partitioning=false)", [str(magfile)])
        for m in mags:
            for role, field in (("assembly", "mag_fasta" if "mag_fasta" in m else "local_fna_path"),
                                ("protein", "proteome_faa")):
                value = m.get(field)
                if not value:
                    continue
                p = local_path(repo, value)
                sequence_candidates.append({"lane_id": lid, "proteome_id": m["proteome_id"],
                    "run_id": m.get("run_id", "source_DRAM"), "role": role,
                    "path": str(p.relative_to(repo)), "exists": p.is_file(),
                    "bytes": p.stat().st_size if p.is_file() else None,
                    "digest_status": "not_yet_selected_or_hashed", "identity_status": "source_locator_only"})
        for tab in tabs:
            name, path = tab["table"], local_path(repo, tab["path"])
            pf = pq.ParquetFile(path)
            columns = pf.schema_arrow.names
            file_hash = digest(path)
            con.read_parquet(str(path), hive_partitioning=False).create_view("source_table", replace=True)
            keys = natural_key(name, columns)
            if not set(keys).issubset(columns):
                raise ValueError(f"Invalid declared key for {lid}/{name}: {keys}")
            stats = key_stats(con, "source_table", keys)
            fk = None
            if "proteome_id" in columns:
                fk = con.execute("SELECT count(*) FROM source_table s ANTI JOIN admitted_mag m USING (proteome_id)").fetchone()[0]
            run_fk = None
            if "run_id" in columns and "run_id" in mags[0] and name != "fact_run_status":
                run_fk = con.execute("SELECT count(*) FROM source_table s ANTI JOIN admitted_mag m USING (proteome_id,run_id)").fetchone()[0]
            cohort_values = con.execute("SELECT DISTINCT cohort_run_id FROM source_table ORDER BY 1").fetchall() if "cohort_run_id" in columns else []
            unitcols = [x for x in columns if x.endswith("_units") or x.endswith("_unit")]
            units = {x: [r[0] for r in con.execute(f"SELECT DISTINCT {quote(x)} FROM source_table ORDER BY 1 LIMIT 30").fetchall()] for x in unitcols}
            row = {"lane_id": lid, "table": name, "authority": "registered_source_warehouse",
                "rights": "internal_review_only_external_reuse_unresolved",
                "release": pointer["release_id"], "path": str(path.relative_to(repo)),
                "logical_path": f"{lid}/{name}", "sha256": file_hash, "bytes": path.stat().st_size,
                "rows": pf.metadata.num_rows, "manifest_rows": int(tab["rows"]),
                "row_count_match": pf.metadata.num_rows == int(tab["rows"]),
                "columns": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in pf.schema_arrow],
                "grain": table_semantics(name), "proposed_natural_key": keys, "key_audit": stats,
                "physical_source_key": ["file_sha256", "zero_based_row_ordinal"],
                "foreign_keys": {"proteome_id_to_dim_mag_unmatched_rows": fk, "selected_run_to_dim_mag_unmatched_rows": run_fk},
                "cohort_run_values": [x[0] for x in cohort_values],
                "selected_run_semantics": "dim_mag.(proteome_id,run_id) identifies selected run; raw cohort namespace retained; fact_run_status retains other attempts",
                "null_semantics": "Unknown or source-empty; never biological absence",
                "tool_database_versions": "Resolve from selected curated/run_record.json or explicit source release; not inferred from table name",
                "measurement_units": units or "No generic unit field; typed column/source methods required",
                "admission": "catalog_only_pending_adapter_checks"}
            results.append(row)
            if stats["status"] == "not_unique_or_incomplete" or fk or run_fk or not row["row_count_match"]:
                issues.append({"lane_id": lid, "table": name, "key_status": stats["status"],
                    "duplicate_rows": stats["duplicate_rows"], "missing_key_rows": stats["null_or_blank_key_rows"],
                    "outside_dim_mag_rows": fk, "nonselected_run_rows": run_fk,
                    "action": "Retain source event row identity; quarantine any biological join until adapter-specific key and scope pass"})
            write_json(out / "source-key-audit.partial.json", results)
            print(f"AUDIT {lid}/{name}: {pf.metadata.num_rows} rows; key={stats['status']}", flush=True)
    con.close()
    write_json(out / "source-key-audit.json", results)
    write_tsv(out / "source-key-audit.tsv", [{k: v for k, v in r.items() if k != "columns"} for r in results])
    write_tsv(out / "source-issues.tsv", issues)
    write_tsv(out / "sequence-source-inventory.tsv", sequence_candidates)
    sources.extend({"path": r["path"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in results)
    for name in ("lane_registry", "freeze_manifest", "freeze_decision", "release_ledger"):
        p = repo / pointer[name]
        h = digest(p)
        if h != pointer[name + "_sha256"]:
            raise ValueError(f"Protected pointer hash mismatch: {name}")
        sources.append({"path": pointer[name], "sha256": h, "bytes": p.stat().st_size})
    write_json(out / "sources.json", sources)
    receipt = {"status": "audit_complete_not_admission", "tables": len(results),
        "rows": sum(r["rows"] for r in results), "bytes": sum(r["bytes"] for r in results),
        "registered_units": len(release_units), "sequence_locators": len(sequence_candidates),
        "sequence_files_present": sum(x["exists"] for x in sequence_candidates),
        "issues": len(issues), "elapsed_seconds": round(time.monotonic() - started, 3),
        "duckdb_version": duckdb.__version__, "audit_code_sha256": digest(Path(__file__)),
        "sequence_identity_admission": "pending_selected_sequence_digest_and_coordinate_audit",
        "natural_key_policy": "Keys tested over all rows; source ordinal never presented as biological key",
        "outputs": {p.name: digest(p) for p in sorted(out.glob("*")) if p.is_file() and p.name not in ("source-key-audit.partial.json", "audit-receipt.json")}}
    write_json(out / "audit-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.repo.resolve(), args.out.resolve()), indent=2))
