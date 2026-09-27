import csv
import json
from rdflib.namespace import RDF
from rdflib import URIRef, Literal

from mvo.fixtures import synthetic
from mvo.model import M, C, QUDT, ROOT, iri, load_bundle
from mvo.validation import validate_graph
from mvo.query import execute


def test_named_sparql_counts():
    result = list(synthetic().query((ROOT / "queries/release_summary.rq").read_text()))
    assert [(str(r.lane), int(r.registered)) for r in result] == [("a", 2), ("b", 1)]


def test_all_ghg_scopes_present_only_in_synthetic_observations():
    g = synthetic()
    assert set(g.objects(None, M.gas)) == {C.CH4, C.CO2, C.N2O}
    assert len(list(g.subjects(RDF.type, M.BridgeAxiom))) == 3
    assert not list(g.subjects(RDF.type, M.GHGStatement))


def test_exclusion_returns_gap_action():
    values = list(synthetic().query((ROOT / "queries/exclusions.rq").read_text()))
    assert len(values) == 1
    assert "Rebuild" in str(values[0].action)


def test_flux_unit_dimension_guard():
    g = synthetic()
    g.set((iri("synthetic", "quantity", "CH4"), QUDT.unit, URIRef("http://qudt.org/vocab/unit/MOL-PER-M3")))
    assert not validate_graph(g)[0]


def test_standards_mappings_match_declared_axioms():
    mapping = ROOT / "mappings/standards.sssom.tsv"
    prefixes = {}
    lines = mapping.read_text().splitlines()
    for line in lines:
        if line.startswith("#   "):
            prefix, iri_base = line[4:].split(": ", 1)
            prefixes[prefix] = iri_base
    expand = lambda curie: URIRef(prefixes[curie.split(":", 1)[0]] + curie.split(":", 1)[1])
    schema = load_bundle("schema")
    for row in csv.DictReader([s for s in lines if not s.startswith("#")], delimiter="\t"):
        assert (expand(row["subject_id"]), expand(row["predicate_id"]), expand(row["object_id"])) in schema
        assert row["mapping_justification"] and row["object_source_version"]


def test_bounded_rows_and_contextual_evidence():
    g = synthetic()
    owner = dict(tenant="emergentbiome", projects=["atlas"], purposes=["internal_review"])
    result = execute(g, dict(query="source_inventory", purpose="internal_review", limit=1), owner)
    assert len(result["rows"]) == 1 and result["truncated"]
    result = execute(g, dict(query="record_evidence", purpose="internal_review", subject=str(iri("synthetic", "annotation"))), owner)
    assert any(x["predicate"].endswith("wasDerivedFrom") for x in result["rows"][0]["facts"])
