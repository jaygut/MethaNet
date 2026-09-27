"""Optional, versioned query projection; canonical RDF remains authoritative.

Each derived literal property and edge retains its canonical statement ID.
The projection is admitted only after its complete property/edge inventory is
recovered from live Neo4j and compared with the canonical projection. No reset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

from .model import M, ID, PROV, sha_bytes, digest
from .projection import connect, read_rows, batches
from .local_runtime import credentials
from .query import allowed
from .molecular_audit import write_json

VERSION="0.1.0"
PRINCIPAL={"tenant":"emergentbiome","projects":["atlas"],"purposes":["internal_review"]}
PREFIXES=((str(M),"mvo_"),(str(PROV),"prov_"),(str(RDFS),"rdfs_"),(str(RDF),"rdf_"))


def key(iri):
    for prefix,short in PREFIXES:
        if iri.startswith(prefix):
            rest=iri[len(prefix):]
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",rest):return short+rest
    return "iri_"+sha_bytes(iri.encode())


def fingerprint(rows):
    return sha_bytes(("\n".join(sorted(json.dumps(r,sort_keys=True,separators=(",",":"),ensure_ascii=True) for r in rows))+"\n").encode())


def prepare(build):
    manifest=json.loads((build/"manifest.json").read_text())
    if digest(build/"graph.nt")!=manifest["graph_sha256"]:raise ValueError("Canonical graph drift")
    g=Graph().parse(build/"graph.nt",format="nt")
    now=datetime.now(timezone.utc)
    nodes={r["iri"]:{**r,"projection_version":VERSION,"authorizedInternal":allowed(g,URIRef(r["iri"]),PRINCIPAL,"internal_review",now),
                     "literal_statement_ids":[]} for r in read_rows(build/"neo4j/nodes.jsonl")}
    # Fixed joins traverse provenance endpoints. Fail closed for mixed-policy
    # local data; a future multi-tenant projection needs per-hop authorization.
    if any(n["iri"].startswith(str(ID)) and not n["authorizedInternal"] for n in nodes.values()):
        raise ValueError("Mixed or unauthorized local policies are not supported by this internal-review domain projection")
    edges=[]
    for r in read_rows(build/"neo4j/statements.jsonl"):
        obj=r["object"];prop=key(r["predicate"])
        if obj["kind"]=="literal":
            nodes[r["subject"]].setdefault(prop,[]).append(obj["value"])
            nodes[r["subject"]]["literal_statement_ids"].append(r["id"])
        else:
            edges.append(dict(snapshot=r["snapshot"],subject=r["subject"],object=obj["value"],
                predicate=r["predicate"],statement_id=r["id"],relation="MVD_"+prop))
    for n in nodes.values():
        for k,v in n.items():
            if isinstance(v,list):v.sort()
    return manifest,list(nodes.values()),edges


def install(build,out):
    started=time.monotonic()
    manifest,nodes,edges=prepare(build)
    sid=manifest["snapshot"];auth=credentials()
    expected_nodes=fingerprint(nodes);expected_edges=fingerprint(edges)
    with connect(**{k:auth[k] for k in ("uri","user","password")}) as driver,driver.session() as session:
        snap=session.run("MATCH (s:MVOSnapshot {id:$sid,status:'validated'}) RETURN s.rdf_sha256 AS sha",sid=sid).single()
        if not snap or snap["sha"]!=manifest["graph_sha256"]:raise ValueError("Canonical Neo4j snapshot is not validated")
        session.run("CREATE CONSTRAINT mvo_domain_resource IF NOT EXISTS FOR (n:MVODResource) REQUIRE (n.snapshot,n.iri) IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT mvo_domain_snapshot IF NOT EXISTS FOR (n:MVODomainSnapshot) REQUIRE (n.id,n.version) IS UNIQUE").consume()
        existing=session.run("MATCH (s:MVODomainSnapshot {id:$sid,version:$version}) RETURN s.node_hash AS n,s.edge_hash AS e",sid=sid,version=VERSION).single()
        if existing and (existing["n"]!=expected_nodes or existing["e"]!=expected_edges):raise ValueError("Immutable domain projection conflict")
        session.run("MERGE (s:MVODomainSnapshot {id:$sid,version:$version}) SET s.status='loading',s.node_hash=$n,s.edge_hash=$e",sid=sid,version=VERSION,n=expected_nodes,e=expected_edges).consume()
        for batch in batches(nodes,500):
            session.run("UNWIND $rows AS r MERGE (n:MVODResource {snapshot:r.snapshot,iri:r.iri}) SET n += r",rows=batch).consume()
        labels=defaultdict(list)
        for n in nodes:
            for cls in n["types"]:
                if cls.startswith(str(M)):
                    label="MVD_"+key(cls)
                    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",label):raise ValueError("Unsafe derived label")
                    labels[label].append(n["iri"])
        for label,ids in sorted(labels.items()):
            for batch in batches(ids):
                session.run("UNWIND $ids AS id MATCH (n:MVODResource {snapshot:$sid,iri:id}) SET n:"+label,ids=batch,sid=sid).consume()
        groups=defaultdict(list)
        for e in edges:groups[e["relation"]].append(e)
        for rel,rows in sorted(groups.items()):
            for batch in batches(rows,500):
                session.run("UNWIND $rows AS r MATCH (s:MVODResource {snapshot:r.snapshot,iri:r.subject}),(o:MVODResource {snapshot:r.snapshot,iri:r.object}) MERGE (s)-[e:"+rel+" {snapshot:r.snapshot,statement_id:r.statement_id}]->(o) SET e.predicate=r.predicate",rows=batch).consume()
        actual_nodes=[dict(r["p"]) for r in session.run("MATCH (n:MVODResource {snapshot:$sid}) RETURN properties(n) AS p",sid=sid)]
        actual_edges=[dict(r) for r in session.run("MATCH (s:MVODResource {snapshot:$sid})-[e]->(o:MVODResource {snapshot:$sid}) RETURN s.snapshot AS snapshot,s.iri AS subject,o.iri AS object,e.predicate AS predicate,e.statement_id AS statement_id,type(e) AS relation",sid=sid)]
        if fingerprint(actual_nodes)!=expected_nodes or fingerprint(actual_edges)!=expected_edges:raise ValueError("Live domain projection differs from canonical derived inventory")
        session.run("MATCH (s:MVODomainSnapshot {id:$sid,version:$version}) SET s.status='validated',s.rdf_sha256=$sha",sid=sid,version=VERSION,sha=manifest["graph_sha256"]).consume()
    result=dict(status="validated",snapshot=sid,projection_version=VERSION,nodes=len(nodes),edges=len(edges),literal_statement_ids=sum(len(n["literal_statement_ids"]) for n in nodes),
                node_hash=expected_nodes,edge_hash=expected_edges,canonical_rdf_sha256=manifest["graph_sha256"],full_live_inventory_parity=True,
                elapsed_seconds=round(time.monotonic()-started,3),canonical_authority="lossless MVOStatement projection; domain arrays are a derived query convenience")
    out.mkdir(parents=True,exist_ok=True);write_json(out/(sid+"-domain-receipt.json"),result)
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--build",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();install(a.build,a.out)
