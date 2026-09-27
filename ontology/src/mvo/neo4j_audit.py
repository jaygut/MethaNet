"""Fixed query parity checks for an explicitly selected local snapshot."""
import json
from pathlib import Path
from rdflib import Graph
from rdflib.namespace import RDF
from .model import M, ROOT, canonical, sha_bytes
from .local_runtime import credentials
from .projection import connect


def audit(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    g = Graph().parse(path / "graph.nt", format="nt")
    if sha_bytes(canonical(g)) != manifest["graph_sha256"]:
        raise ValueError("Source graph was changed after snapshot creation")
    expected = [{"lane": str(r.lane), "registered": int(r.registered)} for r in g.query((ROOT / "queries/release_summary.rq").read_text())]
    auth = credentials()
    with connect(auth["uri"], auth["user"], auth["password"]) as driver, driver.session() as session:
        state = session.run("MATCH (n:MVOSnapshot {id:$id,status:'validated'}) RETURN n.rdf_sha256 AS digest", id=manifest["snapshot"]).single()
        if not state or state["digest"] != manifest["graph_sha256"]:
            raise ValueError("Graph snapshot is not validated or digest differs")
        actual = session.run((ROOT / "queries/release_summary.cypher").read_text(), snapshot=manifest["snapshot"]).data()
        if actual != expected:
            raise ValueError(f"SPARQL/Cypher parity failed: {expected} != {actual}")
        counts = session.run("""MATCH (u:MVOResource {snapshot:$snapshot})-[:MVO_HAS_STATEMENT]->(s:MVOStatement)
          WHERE s.predicate=$predicate AND s.lexical='true' RETURN count(DISTINCT u) AS n""",
                             snapshot=manifest["snapshot"], predicate=str(M.releaseExcluded)).single()["n"]
        excluded = sum(str(g.value(u, M.releaseExcluded)) == "true" for u in g.subjects(RDF.type, M.MolecularRecord))
        if counts != excluded:
            raise ValueError("Excluded-record query parity failed")
    return dict(status="pass", snapshot=manifest["snapshot"], lane_counts=actual, excluded_records=excluded,
                query_parity="SPARQL_vs_Cypher_exact", rdf_sha256=manifest["graph_sha256"])
