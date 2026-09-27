"""Immutable graph snapshots and provenance receipts; no publication side effects."""
from __future__ import annotations

import json
from pathlib import Path
from rdflib import Dataset
from rdflib.namespace import XSD

from .model import M, ROOT, RELEASE, PROV, iri, node, add, digest, sha_bytes, canonical, bind_policy
from .validation import validate_graph
from .projection import export


def code_inventory():
    paths = []
    for folder in ("schema", "shapes", "vocabulary", "src", "mappings", "contracts", "config", "queries"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".ttl", ".json", ".tsv", ".rq", ".cypher", ".sql"))
    paths.extend(p for p in (ROOT / "release.json", ROOT / "requirements.lock", ROOT / "pyproject.toml") if p.exists())
    return [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(paths)]


def materialize(g, summary, sources, output: Path, recorded_at: str):
    if output.exists():
        raise FileExistsError(f"Immutable output already exists: {output}; choose a new snapshot directory")
    source_code = code_inventory()
    fingerprint = sha_bytes(json.dumps(dict(sources=sources, code=source_code, versions=RELEASE, recorded_at=recorded_at), sort_keys=True).encode())
    snapshot = "mvo-" + fingerprint[:24]
    gr = node(g, iri("graph-release", snapshot), M.GraphRelease)
    add(g, gr, M.sourceLocator, "graph.nt")
    add(g, gr, M.version, RELEASE["ontology_version"])
    add(g, gr, M.recordedAt, recorded_at, XSD.dateTime)
    activity = node(g, iri("build", snapshot), M.Activity)
    add(g, gr, PROV.wasGeneratedBy, activity)
    add(g, activity, M.recordedAt, recorded_at, XSD.dateTime)
    for src in sources:
        art = iri("artifact", src["sha256"], src["path"])
        add(g, activity, PROV.used, art)
    bind_policy(g, iri("policy", "internal-evidence-review-v1"))
    valid, errors, report = validate_graph(g)
    output.mkdir(parents=True, exist_ok=False)
    report.serialize(output / "shacl-report.ttl", format="turtle")
    if not valid:
        (output / "validation-errors.json").write_text(json.dumps(errors, indent=2))
        raise ValueError(f"Graph failed validation; quarantined report: {output}")
    data = canonical(g)
    (output / "graph.nt").write_bytes(data)
    ds = Dataset()
    named = ds.graph(gr)
    for triple in g:
        named.add(triple)
    ds.serialize(output / "graph.trig", format="trig")
    manifest = dict(snapshot=snapshot, named_graph=str(gr), graph_sha256=sha_bytes(data), graph_triples=len(g),
                    recorded_at=recorded_at, versions=RELEASE, summary=summary,
                    sources=sources, code=source_code, validation="SHACL_and_identity_checks_pass",
                    source_rights="unknown_no_external_export", status="local_validated_not_independently_reviewed")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    projection = export(g, output / "neo4j", snapshot)
    crate_files = ["graph.nt", "graph.trig", "manifest.json", "shacl-report.ttl", "neo4j/nodes.jsonl", "neo4j/statements.jsonl", "neo4j/projection.json"]
    entities = [
        {"@id": "ro-crate-metadata.json", "@type": "CreativeWork", "about": {"@id": "./"}, "conformsTo": {"@id": "https://w3id.org/ro/crate/1.2"}},
        {"@id": "./", "@type": "Dataset", "name": snapshot, "description": "Private molecular evidence graph. Not credit verification or ecological validation.",
         "datePublished": recorded_at, "license": {"@id": "#restricted"}, "hasPart": [{"@id": p} for p in crate_files]},
        {"@id": "#restricted", "@type": "CreativeWork", "name": "Internal review only", "description": "Source rights unresolved; this package grants no external redistribution permission."},
    ]
    entities.extend({"@id": p, "@type": "File", "name": p, "contentSize": str((output / p).stat().st_size),
                     "mvo:sha256": digest(output / p)} for p in crate_files)
    crate = {"@context": ["https://w3id.org/ro/crate/1.2/context", {"mvo": str(M)}], "@graph": entities}
    (output / "ro-crate-metadata.json").write_text(json.dumps(crate, indent=2) + "\n")
    (output / "README.md").write_text(f"# {snapshot}\n\nValidated local projection, not an externally certified product.\n\n"
                                     f"Graph: {len(g):,} triples. RDF SHA-256: `{manifest['graph_sha256']}`.\n\n"
                                     "Read manifest.json for exact scope, retained gaps, source checksums and code versions.\n"
                                     "The Neo4j projection is lossless and derived. No source assets were modified.\n")
    return dict(snapshot=snapshot, graph_triples=len(g), graph_sha256=manifest["graph_sha256"], summary=summary, projection=projection)
