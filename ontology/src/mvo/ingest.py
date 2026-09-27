"""Read-only atlas adapter. Checks release pins, catalog parity and source keys."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rdflib import Graph, Literal
from rdflib.namespace import RDF, RDFS, XSD

from .model import M, C, PROV, ROOT, iri, digest, add, node, graph, foundation, gap, assertion, bind_policy

MAPPING = json.loads((ROOT / "mappings/warehouse.json").read_text())


def rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f, delimiter="\t")


def boolean(value: str) -> bool:
    if value not in ("true", "false"):
        raise ValueError(f"Invalid boolean (no missing-to-false coercion): {value!r}")
    return value == "true"


def resolve(repo: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        # Legacy manifests contain absolute paths. Rebase only the repository suffix.
        if "MethaNet" not in p.parts:
            raise ValueError(f"Absolute source path outside known repository: {p}")
        p = Path(*p.parts[p.parts.index("MethaNet") + 1:])
    target = (repo / p).resolve()
    if not target.is_relative_to(repo.resolve()):
        raise ValueError(f"Source escapes repository: {value}")
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


class Inventory:
    def __init__(self, repo: Path, g: Graph, ctx: dict):
        self.repo, self.g, self.ctx = repo.resolve(), g, ctx
        self.entries = {}

    def file(self, path: Path, expected: str | None = None, cls=M.Artifact):
        rel = path.resolve().relative_to(self.repo).as_posix()
        if rel in self.entries:
            item = self.entries[rel]
            if expected and item["sha256"] != expected:
                raise ValueError(f"Source hash mismatch: {rel}")
            return iri("artifact", item["sha256"], rel)
        checksum = digest(path)
        if expected and checksum != expected:
            raise ValueError(f"Source hash mismatch: {rel}")
        item = dict(path=rel, sha256=checksum, bytes=path.stat().st_size)
        self.entries[rel] = item
        key = node(self.g, iri("artifact", checksum, rel), cls, path.name)
        for p, v in ((M.sha256, checksum), (M.sourceLocator, rel), (M.hasPolicy, self.ctx["policy"])):
            add(self.g, key, p, v)
        return key


def build_atlas(repo: Path, recorded_at: str, metadata: Path | None = None, diagnostics: list[Path] | None = None):
    repo = repo.resolve()
    pointer_path = resolve(repo, "configs/atlas_current_release.json")
    pointer = json.loads(pointer_path.read_text())
    g = graph()
    ctx = foundation(g, pointer["release_id"], recorded_at)
    inventory = Inventory(repo, g, ctx)
    pointer_artifact = inventory.file(pointer_path)
    pins = {}
    for name in ("lane_registry", "freeze_manifest", "freeze_decision", "release_ledger"):
        path = resolve(repo, pointer[name])
        pins[name] = inventory.file(path, pointer[name + "_sha256"])
    ledger = json.loads(resolve(repo, pointer["release_ledger"]).read_text())
    release = node(g, iri("release", pointer["release_id"]), M.DatasetRelease, pointer["release_id"])
    add(g, release, M.sourceLocator, pointer["freeze_manifest"])
    add(g, release, M.hasPolicy, ctx["policy"])
    add(g, release, PROV.wasDerivedFrom, pins["freeze_manifest"])
    add(g, release, PROV.wasDerivedFrom, pointer_artifact)
    records = {}
    counters = Counter()
    lane_counts = Counter()
    lanes = list(rows(resolve(repo, pointer["lane_registry"])))
    if len({r["lane_id"] for r in lanes}) != len(lanes):
        raise ValueError("Duplicate lane registry identity")
    lane_keys = {r["lane_id"]: node(g, iri("lane", r["lane_id"]), M.AtlasLane, r["denominator_label"]) for r in lanes}
    for lane_id, key in lane_keys.items():
        add(g, key, M.laneId, lane_id)
        add(g, key, PROV.wasDerivedFrom, pins["lane_registry"])
    global_gap = gap(g, iri("gap", pointer["release_id"], "unvalidated-ecological-transfer"),
                     "No release-authorized ecological sample pairing, abundance weighting, field validation or calibrated molecular-to-flux transfer.",
                     "Validate exact sample joins, environment, DNA abundance, uncertainty and independent flux/process evidence.")
    embedding_gap = gap(g, iri("gap", pointer["release_id"], "embedding-configuration-compatibility"),
                       "Cross-run extraction/configuration compatibility has not been established by the release; availability is not comparability.",
                       "Reconcile model/layer/pooling/input fingerprints and same-sequence controls before cross-run retrieval.", C.partial)
    for row_no, row in enumerate(rows(resolve(repo, pointer["freeze_manifest"])), 2):
        key = (row["lane_id"], row["proteome_id"])
        if key in records or key[0] not in lane_keys:
            raise ValueError(f"Duplicate or unknown release identity: {key}")
        unit = node(g, iri("molecular-record", *key), M.MolecularRecord)
        records[key] = unit
        for p, v in ((M.laneId, key[0]), (M.proteomeId, key[1]), (M.inLane, lane_keys[key[0]]),
                     (M.inRelease, release), (PROV.wasDerivedFrom, pins["freeze_manifest"]),
                     (M.sourceRow, f"freeze_manifest.tsv:line={row_no};lane_id={key[0]};proteome_id={key[1]}"),
                     (M.blockedBy, global_gap), (M.blockedBy, embedding_gap), (M.sourceIdentifier, row["mag_id"])):
            add(g, unit, p, v)
        for source, target in MAPPING["freeze_boolean_fields"].items():
            value = boolean(row[source])
            add(g, unit, M[target], value)
            counters[source] += value
        for source, target in MAPPING["freeze_literal_fields"].items():
            add(g, unit, M[target], row[source])
        if boolean(row["release_excluded"]):
            if not row["release_exclusion_reason"]:
                raise ValueError(f"Excluded identity without reason: {key}")
            exclusion = gap(g, iri("gap", "release-exclusion", *key), row["release_exclusion_reason"],
                            row["next_validation_action"] or "Resolve source payload gap before a new release.")
            add(g, unit, M.blockedBy, exclusion)
            add(g, exclusion, PROV.wasDerivedFrom, pins["freeze_manifest"])
        lane_counts[key[0]] += 1
    expected = {"has_esm2": "esm2_units", "has_glm2": "glm2_units", "has_functional": "functional_payload_units",
                "tri_view_ready": "tri_view_ready_units", "release_excluded": "explicit_non_runnable_gaps",
                "mechanism_comparable": "mechanism_comparable_units"}
    if len(records) != ledger["registered_units"]:
        raise ValueError("Registered denominator mismatch")
    for field, ledger_key in expected.items():
        if counters[field] != ledger[ledger_key]:
            raise ValueError(f"Release ledger mismatch: {field}")
    for lane in ledger["lanes"]:
        if lane_counts[lane["lane_id"]] != lane["registry_denominator_units"]:
            raise ValueError("Lane denominator mismatch")
        claim = assertion(g, iri("claim", pointer["release_id"], lane["lane_id"], "screening-scope"), M.Claim,
                          lane_keys[lane["lane_id"]], M.functionalPotential,
                          Literal("Molecular screening evidence; not ecological activity, flux or credit eligibility."),
                          pins["release_ledger"], ctx)
        add(g, claim, M.allowedWording, ledger["allowed_public_wording"])
        add(g, claim, M.nextAction, "Complete source-aware comparability and independent sample/process validation.")
        add(g, claim, M.blockedBy, global_gap)
        add(g, claim, M.validFrom, pointer["snapshot_date"] + "T00:00:00Z", XSD.dateTime)
    for path in diagnostics or []:
        source = inventory.file(resolve(repo, str(path)))
        add(g, embedding_gap, PROV.wasDerivedFrom, source)
        add(g, embedding_gap, M.status, "post_release_diagnostic_not_release_promotion")
    table_count, attempt_count, quarantined = 0, 0, []
    for lane in lanes:
        manifest = resolve(repo, lane["functional_warehouse_dir"] + "/cohort_table_manifest.tsv")
        manifest_id = inventory.file(manifest)
        seen_tables = set()
        for table in rows(manifest):
            if table["table"] in seen_tables:
                raise ValueError(f"Duplicate physical table in manifest: {table['table']}")
            seen_tables.add(table["table"])
            path = resolve(repo, table["path"])
            art = inventory.file(path)
            parquet = pq.ParquetFile(path)
            count = parquet.metadata.num_rows
            if count != int(table["rows"]) or path.stat().st_size != int(table["bytes"]):
                raise ValueError(f"Warehouse manifest row/byte mismatch: {path}")
            add(g, art, M.rowCount, count)
            add(g, art, M.inLane, lane_keys[lane["lane_id"]])
            add(g, art, PROV.wasDerivedFrom, manifest_id)
            table_count += 1
            if table["table"] != "fact_run_status":
                continue
            seen_attempts = set()
            for row in parquet.read().to_pylist():
                key = (lane["lane_id"], row["proteome_id"])
                attempt_key = (*key, row["cohort_run_id"], row["run_id"])
                if attempt_key in seen_attempts:
                    raise ValueError(f"Duplicate attempt key: {attempt_key}")
                seen_attempts.add(attempt_key)
                if key not in records:
                    quarantined.append(dict(lane_id=key[0], proteome_id=key[1], run_id=row["run_id"], source=str(art), reason="outside_registered_release"))
                    q = gap(g, iri("gap", "out-of-release-attempt", *attempt_key), "Attempt is outside the frozen registered denominator.", "Review source unit scope; do not silently add it to the release.", C.ambiguous)
                    add(g, q, PROV.wasDerivedFrom, art)
                    continue
                attempt = node(g, iri("attempt", *attempt_key), M.CurationAttempt)
                for p, v in ((M.forRecord, records[key]), (M.cohortRunId, row["cohort_run_id"]),
                             (M.runId, row["run_id"]), (M.status, row["run_status"]), (PROV.used, art),
                             (M.sourceRow, json.dumps(row, sort_keys=True, default=str))):
                    add(g, attempt, p, v)
                attempt_count += 1
    metadata_counts = {}
    unresolved_site_references = []
    if metadata is not None:
        metadata = (repo / metadata).resolve()
        if not metadata.is_relative_to(repo):
            raise ValueError("Metadata path outside repository")
        files = {p.stem: p for p in sorted((metadata / "tables").glob("*.tsv"))}
        if not {"dim_sample", "dim_site", "link_sample_flux_window"} <= files.keys():
            raise ValueError("Incomplete metadata catalog")
        artifacts = {}
        for name, path in files.items():
            artifacts[name] = inventory.file(path)
            metadata_counts[name] = sum(1 for _ in rows(path))
            add(g, artifacts[name], M.rowCount, metadata_counts[name])
        for row in rows(files["dim_site"]):
            key = iri("site", row["lane_id"], row["site_id"])
            if (key, RDF.type, M.Site) in g:
                raise ValueError("Duplicate metadata site")
            node(g, key, M.Site, row["site_name"])
            add(g, key, PROV.wasDerivedFrom, artifacts["dim_site"])
            add(g, key, M.sourceRow, json.dumps(row, sort_keys=True))
            add(g, key, M.spatialSupport, "source context; coordinates retained verbatim, not normalized to GeoSPARQL")
        samples = {}
        for row in rows(files["dim_sample"]):
            key = (row["lane_id"], row["sample_id"])
            if key in samples:
                raise ValueError("Duplicate metadata sequencing sample")
            sample = node(g, iri("sequencing-sample", *key), M.SequencingSample)
            samples[key] = sample
            add(g, sample, M.sourceIdentifier, row["sample_id"])
            add(g, sample, M.laneId, row["lane_id"])
            add(g, sample, PROV.wasDerivedFrom, artifacts["dim_sample"])
            add(g, sample, M.sourceRow, json.dumps(row, sort_keys=True))
            add(g, sample, M.status, row["resolution_tier"])
            add(g, sample, M.blockedBy, global_gap)
            if row["site_id"]:
                site = iri("site", row["lane_id"], row["site_id"])
                if (site, RDF.type, M.Site) not in g:
                    unresolved_site_references.append(dict(lane_id=key[0], sample_id=key[1], site_label=row["site_id"]))
                    unresolved = gap(g, iri("gap", "unresolved-site-reference", *key),
                                     f"Source sample site label {row['site_id']!r} has no matching lane-scoped dim_site identity.",
                                     "Curate an evidence-backed source-label crosswalk; do not infer a site from string similarity.", C.ambiguous)
                    add(g, unresolved, PROV.wasDerivedFrom, artifacts["dim_sample"])
                    add(g, sample, M.blockedBy, unresolved)
                else:
                    add(g, sample, M.hasSite, site)
        for row in rows(files["link_sample_flux_window"]):
            sample = samples[(row["lane_id"], row["sample_id"])]
            missing = gap(g, iri("gap", "sample-flux", row["lane_id"], row["sample_id"]),
                          row["claim_scope"], "Resolve collection date, depth, environment and observation window before pairing.", C.ambiguous)
            add(g, missing, PROV.wasDerivedFrom, artifacts["link_sample_flux_window"])
            add(g, missing, M.sourceRow, json.dumps(row, sort_keys=True))
            add(g, sample, M.blockedBy, missing)
    bind_policy(g, ctx["policy"])
    summary = dict(source_release=pointer["release_id"], registered_records=len(records),
                   lane_records=dict(lane_counts), payload_counts=dict(counters),
                   cataloged_warehouse_tables=table_count, curation_attempts=attempt_count,
                   quarantined_attempts=quarantined, metadata_catalog_rows=metadata_counts,
                   unresolved_site_references=unresolved_site_references,
                   accepted_physical_sample_flux_pairs=0, active_cross_run_similarity_edges=0,
                   verified_credit_decisions=0, projection_scope="release_records_attempts_metadata_catalog_not_full_warehouse_fact_graph")
    return g, summary, sorted(inventory.entries.values(), key=lambda e: e["path"])
