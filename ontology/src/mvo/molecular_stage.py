"""Deterministic, bounded source-event staging. No graph admission by inference.

Full source-event counts and per-release-unit coverage stay in Parquet/TSV;
only the preregistered deterministic slice is selected for molecular expansion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .model import digest
from .molecular_audit import local_path, read_tsv, write_json, write_tsv

MUCC = "mucc_v1_owc_wetland"
KO_PATTERN = re.compile(r"(?<![A-Za-z0-9])K\d{5}(?!\d)")
GH5_PATTERN = re.compile(r"(?<![A-Za-z0-9])GH5(?:_\d+)?(?!\d)")


def stable_id(kind, *parts):
    return kind + "-" + hashlib.sha256(json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def habitat(lane, row):
    if lane == "poc_core":
        return "non_wetland_rumen_control" if row.get("ecosystem") == "rumen" else "wetland_subtype_unresolved"
    if lane == MUCC:
        return "freshwater_wetland"
    if lane == "msm_china_2025":
        return "mangrove"
    # This group prefix is explicitly paired with source sample-site metadata,
    # not used to infer a unique specimen or MAG abundance.
    mag = row["mag_id"]
    return "mudflat" if mag.startswith("MF1_") else "mangrove" if mag.startswith("MG1_") else "unresolved"


def parquet_rows(path):
    return pq.ParquetFile(path).read().to_pylist()


def optional_number(value):
    if value in (None, "", "NA", "nan"):
        return None
    return float(value)


def mucc_record_mapping(row, catalog):
    """Use the explicit source db_id; preserve non-OWC reference aliases.

    A named reference may have an OWC catalog assignment but no local source
    protein. This is a source container crosswalk, not sequence equivalence.
    """
    db_id, fasta_id = row.get("db_id", ""), row.get("fasta", "")
    if db_id not in catalog:
        return None, "non_catalog_db_id"
    if fasta_id in catalog and fasta_id != db_id:
        return None, "conflicting_catalog_identifiers"
    if row.get("gene_db_id") != row.get(""):
        return None, "conflicting_source_gene_identifiers"
    return db_id, "direct_catalog_fasta" if fasta_id == db_id else "explicit_source_db_id_reference_alias"


def add_source(repo, sources, path, role):
    path = local_path(repo, path)
    key = str(path.relative_to(repo))
    if key not in sources:
        sources[key] = {"path": key, "sha256": digest(path), "bytes": path.stat().st_size, "role": role}
    return sources[key]


def source_event(lid, pid, run, gene, accession, family, tool, source, rownum, raw, accepted=True):
    eid = stable_id("event", lid, source["sha256"], rownum, accession, family)
    return {"event_id": eid, "lane_id": lid, "proteome_id": pid, "run_id": run,
        "gene_id": gene, "family_id": family, "accession": accession, "tool": tool,
        "source_path": source["path"], "source_sha256": source["sha256"],
        "source_row_ordinal": rownum, "source_row_convention": "zero_based_data_record_excluding_header",
        "accepted_under_source_rule": bool(accepted), "review_status": "pending_independent_review",
        "tool_version": "not_recorded_in_selected_source", "database_version": "not_recorded_in_selected_source",
        "threshold": raw.get("threshold", "source_call_threshold_unreported"),
        "score": raw.get("score", raw.get("kegg_bitScore")), "evalue": raw.get("evalue", raw.get("kegg_eVal")),
        "identity_status": "source_event_identity_only_pending_sequence_check", "raw": raw}


def stage(repo, audit_dir, out, panel_path):
    started = time.monotonic()
    executing_code_sha256 = digest(Path(__file__))
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((audit_dir / "source-key-audit.json").read_text())
    panel = json.loads(panel_path.read_text())
    families = {f["id"]: f for f in panel["families"]}
    ko_family = {ko: f["id"] for f in families.values() for ko in f["ko_ids"]}
    if len(ko_family) != sum(len(f["ko_ids"]) for f in families.values()):
        raise ValueError("Panel KO mappings must be unambiguous at candidate-family level")
    sources = {}
    add_source(repo, sources, panel_path, "proposed_panel_mapping")
    pointer = json.loads((repo / "configs/atlas_current_release.json").read_text())
    freeze = read_tsv(repo / pointer["freeze_manifest"])
    frozen = {(r["lane_id"], r["proteome_id"]): r for r in freeze}
    lane_denominators = Counter(lid for lid,pid in frozen)
    for name in ("lane_registry", "freeze_manifest", "release_ledger"):
        add_source(repo, sources, repo / pointer[name], "protected_release_input")
    registry = read_tsv(repo / pointer["lane_registry"])
    tables = {(r["lane_id"], r["table"]): r for r in audit}
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='768MB'")
    con.execute("SET temp_directory=?", [str(out / "duckdb-spill")])
    mags, runs, cov, events, quarantine, raw_summaries = {}, {}, {}, [], [], []

    def src(lid, name):
        entry = tables[(lid, name)]
        s = add_source(repo, sources, repo / entry["path"], "warehouse_table")
        if s["sha256"] != entry["sha256"]:
            raise ValueError(f"Source drift after key audit: {lid}/{name}")
        return s

    for lane in registry:
        lid = lane["lane_id"]
        mag_rows = parquet_rows(repo / src(lid, "dim_mag")["path"])
        for m in mag_rows:
            mags[(lid, m["proteome_id"])] = m
        if lid == MUCC:
            continue
        for r in parquet_rows(repo / src(lid, "fact_run_status")["path"]):
            key = (lid, r["proteome_id"], r["run_id"])
            if key in runs:
                raise ValueError("Run identity ambiguous; admission halted")
            runs[key] = r
        for r in parquet_rows(repo / src(lid, "feature_annotation_coverage")["path"]):
            cov[(lid, r["proteome_id"], r["annotation_tool"])] = r
        for name in ("fact_kofam_hits", "fact_dbcan_hits"):
            s = src(lid, name)
            if tables[(lid, name)]["key_audit"]["status"] != "pass":
                raise ValueError(f"Selected primary annotation keys not valid: {lid}/{name}")
            con.read_parquet(str(repo / s["path"]), hive_partitioning=False, file_row_number=True).create_view("hits", replace=True)
            if name == "fact_kofam_hits":
                # Validate the selected run, not the warehouse's union cohort label.
                query = "SELECT * FROM hits WHERE ko_id IN (SELECT unnest(?)) AND accepted_hit=true ORDER BY file_row_number"
                rows = con.execute(query, [sorted(ko_family)]).fetch_arrow_table().to_pylist()
                for r in rows:
                    m = mags.get((lid, r["proteome_id"]))
                    if not m or m["run_id"] != r["run_id"]:
                        quarantine.append({"lane_id": lid, "source": s["path"], "row": r["file_row_number"], "reason": "nonselected_run"})
                        continue
                    e = source_event(lid, r["proteome_id"], r["run_id"], r["gene_id"], r["ko_id"],
                                     ko_family[r["ko_id"]], "KOfam", s, r["file_row_number"], r)
                    events.append(e)
                raw_summaries.extend({"lane_id": lid, "accession": r[0], "accepted": r[1], "rows": r[2], "source_sha256": s["sha256"]}
                    for r in con.execute("SELECT ko_id,accepted_hit,count(*) FROM hits WHERE ko_id IN (SELECT unnest(?)) GROUP BY 1,2 ORDER BY 1,2", [sorted(ko_family)]).fetchall())
            else:
                # Exact family token, including defined GH5 subfamilies; no keyword hit.
                rows = con.execute('SELECT * FROM hits WHERE regexp_matches(coalesce("HMMER",\'\')||\' \'||coalesce("dbCAN_sub",\'\')||\' \'||coalesce("DIAMOND",\'\'), \'(^|[^A-Za-z0-9])GH5(_[0-9]+)?([^0-9]|$)\') ORDER BY file_row_number').fetch_arrow_table().to_pylist()
                for r in rows:
                    m = mags.get((lid, r["proteome_id"]))
                    if not m or m["run_id"] != r["run_id"]:
                        continue
                    events.append(source_event(lid, r["proteome_id"], r["run_id"], r["Gene ID"], "GH5", "gh5_family", "dbCAN", s, r["file_row_number"], r))
        print(f"STAGE {lid}: primary candidate events extracted", flush=True)

    staging = repo / "results/functional_metagenomics/mucc_v1_owc_wetland_20260626/staging"
    ann = add_source(repo, sources, staging / "owc_metat_table_mags_genes_annotations.csv", "source_gene_expression_annotation_crosswalk")
    dram = add_source(repo, sources, staging / "OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz", "full_source_DRAM_annotation")
    historical_path = repo / "docs/white-paper/v17/analyses/marker_expression/marker_candidate_ledger.tsv"
    historical = {r["source_gene_id"]: r for r in read_tsv(historical_path)}
    add_source(repo, sources, historical_path, "historical_automated_marker_screen_not_independent_review")
    add_source(repo, sources, repo / "docs/white-paper/v17/analysis/curation_audit/marker_label_review.tsv", "historical_sequence_and_label_audit")
    catalog = {m["mag_id"]: pid for (lid, pid), m in mags.items() if lid == MUCC}
    bin_map = defaultdict(set)
    csv_seen, expression_gene_map, crosswalk_issues = set(), {}, []
    with (repo / ann["path"]).open(newline="") as stream:
        for i, r in enumerate(csv.DictReader(stream)):
            gene = r[""]
            if gene in csv_seen:
                raise ValueError(f"Duplicate source RNA annotation gene: {gene}")
            csv_seen.add(gene)
            magid, mapping_status = mucc_record_mapping(r, catalog)
            if magid is None:
                crosswalk_issues.append({"row": i, "source_gene_id": gene, "reason": mapping_status})
                continue
            for alias in (r.get("bin_id"), r.get("fasta")):
                if alias:
                    bin_map[alias].add(magid)
            if gene in historical:
                old = historical[gene]
                for ko in sorted(set(KO_PATTERN.findall(r.get("KO", "") + ";" + r.get("kegg_id", "")))):
                    if ko in ko_family:
                        e = source_event(MUCC, catalog[magid], "source_DRAM_expression_crosswalk", gene, ko,
                            ko_family[ko], "source_DRAM_expression_crosswalk", ann, i, r)
                        e["historical_curation_status"] = old["curation_status"]
                        e["historical_included"] = old["included"] == "True"
                        e["source_MAG_mapping"] = mapping_status
                        e["non_independence"] = "Derived from source DRAM; not an independent annotation replicate"
                        events.append(e)
                        expression_gene_map[gene] = {"proteome_id": catalog[magid], "annotation_row": i}
    write_json(out / "mucc-gene-crosswalk-audit.json", {"rows": len(csv_seen),
        "unique_source_gene_ids": len(csv_seen), "source_sha256": ann["sha256"],
        "ambiguous_bin_values": {k: sorted(v) for k, v in bin_map.items() if len(v) != 1},
        "field_disagreement_rows": crosswalk_issues, "historical_genes_joined": len(expression_gene_map)})
    if len(expression_gene_map) != len(historical):
        raise ValueError("Historical marker denominator not recovered; retain audit and stop this adapter")
    mucc_covered = Counter()
    dram_unmapped = Counter()
    dram_total, dram_selected = 0, 0
    # Full payload scanned once. No setdefault crosswalk: a multi-MAG bin is ambiguous.
    with gzip.open(repo / dram["path"], "rt", newline="") as stream:
        for i, r in enumerate(csv.DictReader(stream, delimiter="\t")):
            dram_total += 1
            label = r.get("fasta", "")
            candidates = {label} if label in catalog else bin_map.get(label, set())
            if len(candidates) != 1:
                dram_unmapped[label] += 1
                continue
            magid = next(iter(candidates))
            pid = catalog[magid]
            mucc_covered[pid] += 1
            targets = [(ko_family[k], k) for k in set(KO_PATTERN.findall(r.get("kegg_id", ""))) if k in ko_family]
            if GH5_PATTERN.search(r.get("cazy_hits", "")):
                targets.append(("gh5_family", "GH5"))
            for fid, accession in sorted(targets):
                e = source_event(MUCC, pid, "source_DRAM_20220208", r[""], accession, fid, "source_DRAM", dram, i, r)
                e["source_MAG_mapping"] = "direct_catalog_fasta" if label == magid else "unique_annotation_bin_to_MAG_crosswalk"
                e["gene_namespace"] = "source_DRAM_original"
                events.append(e)
                dram_selected += 1
            if dram_total % 1000000 == 0:
                print(f"STAGE MUCC full DRAM: {dram_total} source rows scanned", flush=True)
    write_json(out / "mucc-full-dram-audit.json", {"source_sha256": dram["sha256"], "rows": dram_total,
        "mapped_MAG_rows": sum(mucc_covered.values()), "mapped_MAGs": len(mucc_covered),
        "candidate_events": dram_selected, "unmapped_or_ambiguous_rows": sum(dram_unmapped.values()),
        "unmapped_or_ambiguous_source_bins": dict(sorted(dram_unmapped.items())),
        "coverage_boundary": "Mapping source MAG container does not prove every gene is represented in local proteome or every legacy locus has an exact contig crosswalk"})

    # Coverage denominator is EVERY registered unit × EVERY panel family.
    event_index = defaultdict(list)
    for e in events:
        if e["tool"] != "source_DRAM_expression_crosswalk":
            event_index[(e["lane_id"], e["proteome_id"], e["family_id"])].append(e)
    coverage, selection, selected_mags = [], [], set()
    strata = defaultdict(set)
    for (lid, pid, fid), hits in event_index.items():
        row = frozen.get((lid, pid))
        if row:
            strata[(lid, habitat(lid, row), fid)].add(pid)
    for (lid, hab, fid), pids in sorted(strata.items()):
        for pid in sorted(pids)[:2]:
            selected_mags.add((lid, pid))
            selection.append({"lane_id": lid, "habitat": hab, "family_id": fid,
                "proteome_id": pid, "rule": "first_two_hit_bearing_MAGs_by_canonical_key_per_lane_habitat_family"})
    for e in events:
        if e["tool"] == "source_DRAM_expression_crosswalk":
            selected_mags.add((e["lane_id"], e["proteome_id"]))
    for (lid, pid), release in sorted(frozen.items()):
        m = mags.get((lid, pid), {})
        for fid, family in families.items():
            hits = event_index.get((lid, pid, fid), [])
            is_assayed = bool(family["ko_ids"] or family.get("dbcan_families"))
            annotation_tool = "dbCAN" if family.get("dbcan_families") else "KOfam"
            coverage_row = cov.get((lid, pid, annotation_tool), {})
            run = runs.get((lid, pid, m.get("run_id")), {})
            if release["release_excluded"] == "true":
                state, reason = "excluded", release["release_exclusion_reason"]
            elif not is_assayed:
                state, reason = "not_assayed", "No admitted subgroup-specific assay mapping in panel v0.1.0"
            elif hits:
                state = "ambiguous" if family["ambiguity"] else "present"
                reason = "Source-derived candidate; independent interpretation review pending"
            elif lid == MUCC:
                state, reason = "missing_input", "No family call among mapped full-DRAM rows; full locus-to-proteome coverage not validated, so biological no-hit not admitted"
            elif run.get("run_status") == "failed":
                state, reason = "failed", "Selected source run failed"
            elif not m or not coverage_row or run.get("run_status") != "complete":
                state, reason = "missing_input", "Completed selected-run assay coverage unavailable"
            else:
                state, reason = "covered_no_accepted_hit", "Completed source assay contains no accepted panel call; incomplete MAG and assay sensitivity remain limits"
            observed = sorted({e["accession"] for e in hits})
            required = family["ko_ids"] or family.get("dbcan_families", [])
            coverage.append({"lane_id": lid, "proteome_id": pid, "family_id": fid,
                "habitat": habitat(lid, release), "panel_version": panel["version"],
                "status": state, "reason": reason, "candidate_event_count": len(hits),
                "observed_components": ";".join(observed), "assayed_panel_components": ";".join(required),
                "component_support": "not_assayed" if not required else "no_accepted_component" if not observed else
                    "all_assayed_alternatives_supported" if family.get("logic") == "any" else
                    "all_assayed_components_supported" if set(required).issubset(observed) else "partial_assayed_components",
                "pathway_complete": False, "independently_reviewed_function": False,
                "completeness": optional_number(m.get("checkm2_completeness", m.get("bin_completeness"))),
                "contamination": optional_number(m.get("checkm2_contamination", m.get("bin_contamination"))),
                "taxonomy": m.get("gtdb_classification", m.get("atlas_taxonomy_lineage")),
                "taxonomy_version": m.get("gtdb_release", "source_DRAM_plus_KBase_214.1_not_R232"),
                "assay_contract": "source_DRAM_mapped_rows_not_pipeline_normalized" if lid == MUCC else "pipeline_KOfam_accepted_or_dbCAN_family",
                "cross_lane_quantitative_comparability": "not_validated",
                "selected_for_molecular_slice": (lid, pid) in selected_mags,
                "release_excluded": release["release_excluded"] == "true",
                "registered_lane_denominator": lane_denominators[lid],
                "source_coverage_rows": mucc_covered.get(pid, 0) if lid == MUCC else coverage_row.get("row_count"),
                "review_status": "pending_independent_review"})
    selected = [e for e in events if (e["lane_id"], e["proteome_id"]) in selected_mags]
    # Ensure genomic identities are not silently collapsed across source namespaces.
    locus_keys = {(e["lane_id"], e["proteome_id"], e["run_id"], e["gene_id"]) for e in selected}
    if len(locus_keys) > panel["selection"]["maximum_selected_loci"]:
        write_json(out / "size-halt.json", {"selected_locus_keys": len(locus_keys), "budget": panel["selection"]["maximum_selected_loci"], "action": "Stop; revise deterministic selection prospectively and record reason"})
        raise ValueError("Preregistered molecular selection exceeds locus budget")
    # Supplemental methods are attached only to a tested same-run query gene.
    selected_query_keys = {(e["lane_id"], e["proteome_id"], e["run_id"], e["gene_id"]) for e in selected if e["lane_id"] != MUCC}
    corroboration = []
    for lane in registry:
        lid = lane["lane_id"]
        if lid == MUCC:
            continue
        keys = [(p, run, gene) for l, p, run, gene in sorted(selected_query_keys) if l == lid]
        con.register("selected_gene_keys", pa.Table.from_pylist([{"proteome_id": p, "run_id": run, "gene_id": gene} for p, run, gene in keys]))
        for name in ("fact_mcycdb_hits", "fact_scycdb_hits", "fact_kofam_hits"):
            s = src(lid, name)
            con.read_parquet(str(repo / s["path"]), hive_partitioning=False, file_row_number=True).create_view("hits", replace=True)
            where = "h.hit_rank_bitscore=1" if name != "fact_kofam_hits" else "h.accepted_hit=false AND h.ko_id IN (SELECT unnest(?))"
            params = [] if name != "fact_kofam_hits" else [sorted(ko_family)]
            query = "SELECT h.* FROM hits h JOIN selected_gene_keys k USING(proteome_id,run_id,gene_id) WHERE " + where + " ORDER BY h.file_row_number"
            for r in con.execute(query, params).fetch_arrow_table().to_pylist():
                if name != "fact_kofam_hits" and (not r.get("gene_id") or not r.get("subject_id")):
                    quarantine.append({"lane_id": lid, "source": s["path"], "row": r["file_row_number"], "reason": "incomplete_best_hit_identity"})
                    continue
                corroboration.append({"lane_id": lid, "proteome_id": r["proteome_id"], "run_id": r["run_id"], "gene_id": r["gene_id"],
                    "tool": r["source_tool"], "source_sha256": s["sha256"], "source_path": s["path"], "source_row_ordinal": r["file_row_number"],
                    "interpretation": "best_ranked_similarity_not_validated_function" if name != "fact_kofam_hits" else "rejected_KO_hit_not_detection", "raw": r})
        # Preserve module/HMM evidence at its actual MAG grain; no splitting a
        # comma-separated tool-native gene list into exact protein identities.
        s = src(lid, "fact_metabolic_hmm_hits")
        con.read_parquet(str(repo / s["path"]), hive_partitioning=False, file_row_number=True).create_view("mhmm", replace=True)
        pids = sorted(p for l, p in selected_mags if l == lid)
        for r in con.execute("SELECT * FROM mhmm WHERE proteome_id IN (SELECT unnest(?)) AND ko_id IN (SELECT unnest(?)) ORDER BY file_row_number", [pids, sorted(ko_family)]).fetch_arrow_table().to_pylist():
            corroboration.append({"lane_id": lid, "proteome_id": r["proteome_id"], "run_id": r["run_id"], "gene_id": "",
                "tool": "METABOLIC", "source_sha256": s["sha256"], "source_path": s["path"], "source_row_ordinal": r["file_row_number"],
                "interpretation": "MAG_HMM_presence_source_semantics_no_exact_gene_join", "raw": r})
    con.close()
    # Durable checkpoint before serialization; no expensive source rescan is
    # required merely to repair a presentation or Parquet typing issue.
    write_json(out / "staging-checkpoint.json", {"coverage": coverage, "selection": selection,
        "raw_summaries": raw_summaries, "selected_events": selected, "corroboration": corroboration,
        "sources": list(sources.values()), "code_sha256": executing_code_sha256, "panel_sha256": digest(panel_path)})
    for name, rows in (("panel-coverage", coverage), ("selection-register", selection), ("raw-hit-counts", raw_summaries)):
        write_tsv(out / f"{name}.tsv", rows)
        pq.write_table(pa.Table.from_pylist(rows), out / f"{name}.parquet", compression="zstd")
    write_json(out / "selected-annotation-events.json", sorted(selected, key=lambda x: x["event_id"]))
    write_json(out / "supplementary-method-events.json", corroboration)
    write_json(out / "quarantine.json", quarantine)
    write_json(out / "selected-mag-context.json", [{"lane_id": lid, "proteome_id": pid, "warehouse": mags[(lid,pid)], "release": frozen[(lid,pid)],
        "selected_run": runs.get((lid,pid,mags[(lid,pid)].get("run_id")), {})} for lid,pid in sorted(selected_mags)])
    write_json(out / "sources.json", list(sources.values()))
    receipt = {"status": "staged_pending_sequence_and_graph_admission", "registered_units": len(frozen),
        "panel_families": len(families), "coverage_rows": len(coverage), "primary_candidate_events": len(events),
        "selected_events": len(selected), "selected_locus_keys": len(locus_keys), "selected_MAGs": len(selected_mags),
        "selected_MAGs_by_lane": dict(Counter(l for l,p in selected_mags)), "supplementary_events": len(corroboration),
        "coverage_statuses": dict(Counter(r["status"] for r in coverage)), "quarantined_rows": len(quarantine),
        "historical_MUCC_marker_genes": len(expression_gene_map), "elapsed_seconds": round(time.monotonic()-started,3),
        "code_sha256": executing_code_sha256, "panel_sha256": digest(panel_path),
        "outputs": {p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "stage-receipt.json"}}
    write_json(out / "stage-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=Path("ontology/mappings/molecular-panel-0.1.0.json"))
    args = parser.parse_args()
    print(json.dumps(stage(args.repo.resolve(), args.audit.resolve(), args.out.resolve(), args.panel.resolve()), indent=2))
