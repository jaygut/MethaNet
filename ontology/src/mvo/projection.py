"""Lossless RDF 1.1 -> Neo4j projection. No APOC, RDF-star or arbitrary Cypher."""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF

from .model import M, sha_bytes, canonical, digest


def encode(term):
    if isinstance(term, URIRef):
        return dict(kind="iri", value=str(term))
    if isinstance(term, Literal):
        return dict(kind="literal", value=str(term), datatype=str(term.datatype) if term.datatype else None, language=term.language)
    raise ValueError("Only skolemized IRI resources and RDF literals can be projected")


def decode(value):
    if value["kind"] == "iri":
        return URIRef(value["value"])
    if value["kind"] != "literal":
        raise ValueError("Unknown RDF term encoding")
    return Literal(value["value"], datatype=URIRef(value["datatype"]) if value.get("datatype") else None,
                   lang=value.get("language"), normalize=False)


def export(g: Graph, destination: Path, snapshot: str):
    destination.mkdir(parents=True, exist_ok=False)
    nodes = {str(s) for s in g.subjects()} | {str(o) for o in g.objects() if isinstance(o, URIRef)}
    statements = []
    for s, p, o in g:
        row = dict(snapshot=snapshot, subject=str(s), predicate=str(p), object=encode(o))
        row["id"] = sha_bytes(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        statements.append(row)
    statements.sort(key=lambda r: r["id"])
    with (destination / "nodes.jsonl").open("w") as f:
        for key in sorted(nodes):
            types = sorted(str(t) for t in g.objects(URIRef(key), RDF.type))
            f.write(json.dumps(dict(iri=key, snapshot=snapshot, types=types), ensure_ascii=False) + "\n")
    with (destination / "statements.jsonl").open("w") as f:
        for row in statements:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = dict(snapshot=snapshot, resources=len(nodes), statements=len(statements), rdf_sha256=sha_bytes(canonical(g)),
                  files={p.name: digest(p) for p in destination.glob("*.jsonl")})
    (destination / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
    restored = restore(destination)
    if canonical(restored) != canonical(g):
        raise ValueError("Projection roundtrip failed")
    return result


def read_rows(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def restore(path: Path):
    g = Graph()
    for row in read_rows(path / "statements.jsonl"):
        g.add((URIRef(row["subject"]), URIRef(row["predicate"]), decode(row["object"])))
    return g


def batches(rows, size=1000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def connect(uri: str, user: str, password: str):
    from neo4j import GraphDatabase
    address = urlparse(uri)
    if address.scheme not in ("bolt", "neo4j", "bolt+s", "neo4j+s") or address.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Local-only graph adapter: remote graph writes are not supported")
    if not password:
        raise ValueError("Authentication is required")
    driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10)
    driver.verify_connectivity()
    return driver


def load_neo4j(path: Path, uri: str, user: str, password: str, database="neo4j"):
    started = time.monotonic()
    manifest = json.loads((path / "projection.json").read_text())
    for name, checksum in manifest["files"].items():
        if digest(path / name) != checksum:
            raise ValueError(f"Projection checksum mismatch: {name}")
    snapshot = manifest["snapshot"]
    with connect(uri, user, password) as driver, driver.session(database=database) as session:
        session.run("CREATE CONSTRAINT mvo_resource IF NOT EXISTS FOR (n:MVOResource) REQUIRE (n.snapshot,n.iri) IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT mvo_statement IF NOT EXISTS FOR (n:MVOStatement) REQUIRE (n.snapshot,n.id) IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT mvo_snapshot IF NOT EXISTS FOR (n:MVOSnapshot) REQUIRE n.id IS UNIQUE").consume()
        existing = session.run("MATCH (n:MVOSnapshot {id:$id}) RETURN n.rdf_sha256 AS hash", id=snapshot).single()
        if existing and existing["hash"] != manifest["rdf_sha256"]:
            raise ValueError("Immutable graph snapshot already exists with a different content digest")
        session.run("MERGE (n:MVOSnapshot {id:$id}) SET n.rdf_sha256=$hash,n.status='loading'", id=snapshot, hash=manifest["rdf_sha256"]).consume()
        for batch in batches(read_rows(path / "nodes.jsonl")):
            session.run("UNWIND $rows AS r MERGE (n:MVOResource {snapshot:r.snapshot,iri:r.iri}) SET n.types=r.types", rows=batch).consume()
        for batch in batches(read_rows(path / "statements.jsonl")):
            prepared = []
            for row in batch:
                obj = row["object"]
                prepared.append({**row, "object_json": json.dumps(obj, sort_keys=True), "object_iri": obj["value"] if obj["kind"] == "iri" else None,
                                 "lexical": obj["value"], "kind": obj["kind"]})
            session.run("""UNWIND $rows AS r
              MATCH (s:MVOResource {snapshot:r.snapshot,iri:r.subject})
              MERGE (t:MVOStatement {snapshot:r.snapshot,id:r.id})
              SET t.subject=r.subject,t.predicate=r.predicate,t.object_json=r.object_json,t.lexical=r.lexical,t.kind=r.kind
              MERGE (s)-[:MVO_HAS_STATEMENT]->(t)
              WITH r,s,t WHERE r.object_iri IS NOT NULL
              MATCH (o:MVOResource {snapshot:r.snapshot,iri:r.object_iri})
              MERGE (t)-[:MVO_OBJECT]->(o)
              MERGE (s)-[:MVO_REL {statement_id:r.id,predicate:r.predicate}]->(o)""", rows=prepared).consume()
        count = session.run("MATCH (s:MVOStatement {snapshot:$id}) RETURN count(s) AS n", id=snapshot).single()["n"]
        resource_count = session.run("MATCH (s:MVOResource {snapshot:$id}) RETURN count(s) AS n", id=snapshot).single()["n"]
        if resource_count != manifest["resources"]:
            raise ValueError("Neo4j resource count mismatch; snapshot remains loading")
        if count != manifest["statements"]:
            raise ValueError("Neo4j statement count mismatch; snapshot remains loading")
        # Bound each database transaction; decoding a complete RDF graph while
        # one Bolt result stays open can exceed the isolated runtime's timeout.
        # The independently checked total counts reject extra statements, and
        # every expected ID must be returned exactly once before hashing.
        restored = Graph()
        for batch in batches(read_rows(path / "statements.jsonl")):
            expected_ids = {r["id"] for r in batch}
            recovered = session.run("""UNWIND $ids AS statement_id
              MATCH (s:MVOStatement {snapshot:$snapshot,id:statement_id})
              RETURN s.id AS id,s.subject AS s,s.predicate AS p,s.object_json AS o""",
              snapshot=snapshot, ids=sorted(expected_ids)).data()
            if len(recovered) != len(expected_ids) or {r["id"] for r in recovered} != expected_ids:
                raise ValueError("Neo4j bounded read-back ID mismatch; snapshot remains loading")
            for row in recovered:
                restored.add((URIRef(row["s"]), URIRef(row["p"]), decode(json.loads(row["o"]))))
        actual = sha_bytes(canonical(restored))
        if actual != manifest["rdf_sha256"]:
            raise ValueError("Neo4j semantic roundtrip mismatch; snapshot remains loading")
        session.run("MATCH (n:MVOSnapshot {id:$id}) SET n.status='validated'", id=snapshot).consume()
    return dict(snapshot=snapshot, resources=resource_count, statements=count, rdf_sha256=actual, status="validated", roundtrip="exact",
                verification="all_expected_statement_IDs_in_bounded_transactions_plus_total_counts_and_RDF_hash",
                loader_sha256=digest(Path(__file__)), elapsed_seconds=round(time.monotonic()-started,3))
