"""Canonical RDF primitives. No network I/O and no automatic identity merging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import rdflib
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

ROOT = Path(__file__).resolve().parents[2]
M = Namespace("https://emergentbiome.earth/ontology/mvo/")
C = Namespace(str(M) + "concept/")
ID = Namespace("https://emergentbiome.earth/id/")
PROV = Namespace("http://www.w3.org/ns/prov#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
RELEASE = json.loads((ROOT / "release.json").read_text())
# This isolated CLI preserves RDF lexical forms (e.g. "01"^^xsd:integer).
# Literal normalization would make the purportedly lossless projection lossy.
rdflib.NORMALIZE_LITERALS = False


def iri(kind: str, *parts: str) -> URIRef:
    """Injective escaped composite key; do not normalize source identifiers."""
    if not parts or any(not str(p) for p in parts):
        raise ValueError("Empty identity component")
    return URIRef(str(ID) + quote(kind, safe="") + "/" + "/".join(quote(str(p), safe="") for p in parts))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(graph: Graph) -> bytes:
    """Sorted N-Triples, stable for our deliberately blank-node-free data plane."""
    from rdflib.compare import to_canonical_graph
    g = to_canonical_graph(graph)
    return ("\n".join(sorted(line for line in g.serialize(format="nt").splitlines() if line)) + "\n").encode()


def load_bundle(folder: str) -> Graph:
    graph = Graph()
    for path in sorted((ROOT / folder).glob("*.ttl")):
        graph.parse(path, format="turtle")
    return graph


def graph() -> Graph:
    g = Graph()
    for prefix, ns in (("mvo", M), ("c", C), ("id", ID), ("prov", PROV), ("qudt", QUDT)):
        g.bind(prefix, ns)
    return g


def add(g: Graph, s: URIRef, p: URIRef, value, datatype=None):
    if value is None or value == "":
        return
    term = value if isinstance(value, (URIRef, Literal)) else Literal(value, datatype=datatype)
    g.add((s, p, term))


def node(g: Graph, key: URIRef, cls: URIRef, label: str | None = None) -> URIRef:
    g.add((key, RDF.type, cls))
    if label:
        add(g, key, RDFS.label, label)
    return key


def foundation(g: Graph, release_id: str, recorded_at: str) -> dict:
    """Internal policy is not a claim to upstream licensing or human signoff."""
    policy = node(g, iri("policy", "internal-evidence-review-v1"), M.Policy)
    for p, v in ((M.tenant, "emergentbiome"), (M.projectKey, "atlas"), (M.purpose, "internal_review"),
                 (M.rightsStatus, "unknown"), (M.externalExportAllowed, False)):
        add(g, policy, p, v)
    agent = node(g, iri("agent", "mvo-importer-0.1.0"), M.Agent)
    g.add((agent, RDF.type, PROV.SoftwareAgent))
    method = node(g, iri("method", "release-mapping-0.1.0"), M.Method)
    add(g, method, M.version, RELEASE["mapping_version"])
    scope = node(g, iri("scope", release_id), M.Scope)
    add(g, scope, M.scopeDescription, "Lane-scoped molecular records and source metadata; no ecological pairing, field validation, calibrated risk or credit decision.")
    uncertainty = node(g, iri("uncertainty", "unquantified-molecular-transfer"), M.Uncertainty)
    add(g, uncertainty, M.uncertaintyKind, "not_quantified")
    add(g, uncertainty, M.uncertaintyDescription, "Molecular-to-process transfer uncertainty is not calibrated. Unknown values are not zeros.")
    return dict(policy=policy, agent=agent, method=method, scope=scope, uncertainty=uncertainty, recorded_at=recorded_at)


def assertion(g: Graph, key: URIRef, cls: URIRef, subject: URIRef, predicate: URIRef,
              obj: URIRef | Literal, source: URIRef, ctx: dict, state=C.imported,
              review=C.pending) -> URIRef:
    node(g, key, cls)
    for p, v in ((M.subject, subject), (M.predicate, predicate),
                 (M.object if isinstance(obj, URIRef) else M.literalValue, obj),
                 (M.epistemicState, state), (M.reviewState, review),
                 (PROV.wasDerivedFrom, source), (PROV.wasAttributedTo, ctx["agent"]),
                 (M.usesMethod, ctx["method"]), (M.hasScope, ctx["scope"]),
                 (M.hasUncertainty, ctx["uncertainty"]), (M.hasPolicy, ctx["policy"]),
                 (M.timeStatus, "release_snapshot")):
        add(g, key, p, v)
    add(g, key, M.recordedAt, ctx["recorded_at"], XSD.dateTime)
    return key


def gap(g: Graph, key: URIRef, reason: str, action: str, state=C.missing) -> URIRef:
    node(g, key, M.ValidationGap)
    for p, v in ((M.reason, reason), (M.nextAction, action), (M.evidenceState, state)):
        add(g, key, p, v)
    return key


def bind_policy(g: Graph, policy: URIRef):
    """Default-deny source rights, explicitly propagated to every data resource."""
    subjects = set(g.subjects())
    for s in subjects:
        if not list(g.objects(s, M.hasPolicy)):
            g.add((s, M.hasPolicy, policy))


def write_graph(g: Graph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(g))
