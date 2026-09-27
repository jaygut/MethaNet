import json
from rdflib import Graph, Literal
from rdflib.namespace import RDF, OWL
import pytest

from mvo.documentation import generate, compare
from mvo.release import materialize
from mvo.fixtures import synthetic
from mvo.model import ROOT, M, iri, canonical
from mvo.validation import validate_graph


def test_generated_reference_is_current(tmp_path):
    generate(tmp_path)
    for name in ("context.jsonld", "schema-reference.md"):
        assert (tmp_path / name).read_bytes() == (ROOT / "docs/generated" / name).read_bytes()


def test_diff_detects_term_removal(tmp_path):
    old = Graph()
    old.add((M.Example, RDF.type, OWL.Class))
    a, b = tmp_path / "a.ttl", tmp_path / "b.ttl"
    old.serialize(a, format="turtle")
    Graph().serialize(b, format="turtle")
    result = compare(a, b)
    assert str(M.Example) in result["removed_terms"]
    assert result["disposition"] == "review_required"


def test_immutable_release_and_crate(tmp_path):
    out = tmp_path / "snapshot"
    result = materialize(synthetic(), {"scope": "SYNTHETIC"}, [], out, "2026-09-26T00:00:00Z")
    with pytest.raises(FileExistsError):
        materialize(synthetic(), {}, [], out, "2026-09-26T00:00:00Z")
    crate = json.loads((out / "ro-crate-metadata.json").read_text())
    assert crate["@context"][0] == "https://w3id.org/ro/crate/1.2/context"
    root = next(e for e in crate["@graph"] if e["@id"] == "./")
    assert root["license"]["@id"] == "#restricted"
    assert all((out / p["@id"]).is_file() for p in root["hasPart"])
    from rdflib import Dataset
    dataset = Dataset().parse(out / "graph.trig", format="trig")
    manifest = json.loads((out / "manifest.json").read_text())
    from rdflib import URIRef
    assert canonical(dataset.graph(URIRef(manifest["named_graph"]))) == (out / "graph.nt").read_bytes()


def test_bridge_cannot_claim_independent_validation():
    g = synthetic()
    g.set((iri("synthetic", "bridge", "CH4"), M.validationState, Literal("independently_validated")))
    assert not validate_graph(g)[0]
