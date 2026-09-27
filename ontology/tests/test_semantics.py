import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, OWL, XSD
from owlrl import DeductiveClosure, OWLRL_Semantics
from jsonschema import ValidationError

from mvo.model import M, C, PROV, QUDT, ROOT, iri, canonical, load_bundle, add, node, assertion, foundation
from mvo.fixtures import synthetic
from mvo.validation import validate_graph, verify_schema
from mvo.projection import export, restore, encode, decode, connect
from mvo.query import execute, AccessDenied, BudgetExceeded
from mvo.ingest import boolean, resolve


@pytest.fixture
def g():
    return synthetic()


def test_valid_synthetic_graph(g):
    ok, errors, _ = validate_graph(g)
    assert ok, errors


def test_schema_parse_and_lint(tmp_path):
    result = verify_schema(tmp_path)
    assert result["status"] == "pass", result


def test_owl_entailment_and_non_entailment(g):
    h = g + load_bundle("schema")
    DeductiveClosure(OWLRL_Semantics).expand(h)
    assert (iri("synthetic", "annotation"), RDF.type, M.Evidence) in h
    assert (iri("synthetic", "annotation"), RDF.type, M.Assertion) in h
    assert (iri("synthetic", "record", "0"), M.functionalPotential, C.methanogenesis) not in h
    assert not list(h.subjects(RDF.type, M.VerificationDecision))
    assert (iri("synthetic", "record", "0"), M.molecularSimilarity, iri("synthetic", "record", "1")) not in h


@pytest.mark.parametrize("predicate", [M.subject, M.predicate, M.object, M.hasUncertainty, M.hasPolicy, M.hasScope, M.usesMethod, M.epistemicState, M.reviewState, M.recordedAt, M.timeStatus, PROV.wasDerivedFrom, PROV.wasAttributedTo])
def test_assertion_required_fields(g, predicate):
    g.remove((iri("synthetic", "annotation"), predicate, None))
    assert not validate_graph(g)[0]


def test_literal_and_object_are_exclusive(g):
    add(g, iri("synthetic", "annotation"), M.literalValue, "not both")
    assert not validate_graph(g)[0]


@pytest.mark.parametrize("case", ["duplicate_identity", "missing_exclusion_gap", "false_tri_view", "dangling", "same_as", "type_collision", "no_hit_without_coverage", "unreviewed_acceptance", "accepted_hypothesis", "credit_approval", "flux_wrong_quantity", "reversed_time", "invalid_rights", "incompatible_similarity", "manufactured_physical_link"])
def test_negative_semantics(g, case):
    ann = iri("synthetic", "annotation")
    if case == "duplicate_identity":
        for p, o in list(g.predicate_objects(iri("synthetic", "record", "0"))):
            g.add((iri("synthetic", "duplicate"), p, o))
    elif case == "missing_exclusion_gap":
        g.remove((iri("synthetic", "record", "2"), M.blockedBy, None))
    elif case == "false_tri_view":
        g.set((iri("synthetic", "record", "2"), M.triViewReady, Literal(True)))
    elif case == "dangling":
        add(g, ann, M.supports, iri("synthetic", "nonexistent"))
    elif case == "same_as":
        add(g, ann, OWL.sameAs, iri("synthetic", "record", "0"))
    elif case == "type_collision":
        g.add((iri("synthetic", "physical-sample"), RDF.type, M.SequencingSample))
    elif case == "no_hit_without_coverage":
        g.set((ann, M.evidenceState, C.observedNoHit))
        g.set((ann, M.coverageState, Literal("not_assayed")))
    elif case == "unreviewed_acceptance":
        g.set((ann, M.reviewState, C.accepted))
    elif case == "accepted_hypothesis":
        key = iri("synthetic", "bridge", "CH4")
        g.set((key, M.reviewState, C.accepted))
        add(g, key, M.reviewedBy, iri("agent", "mvo-importer-0.1.0"))
    elif case == "credit_approval":
        key = node(g, iri("synthetic", "decision"), M.VerificationDecision)
        add(g, key, M.decisionDisposition, "approved")
    elif case == "flux_wrong_quantity":
        g.set((iri("synthetic", "quantity", "CH4"), M.quantityKind, Literal("concentration")))
    elif case == "reversed_time":
        add(g, ann, M.validFrom, "2026-09-26T00:00:00Z", XSD.dateTime)
        add(g, ann, M.validUntil, "2026-09-25T00:00:00Z", XSD.dateTime)
    elif case == "invalid_rights":
        g.set((iri("policy", "internal-evidence-review-v1"), M.externalExportAllowed, Literal(True)))
    elif case == "incompatible_similarity":
        sim = iri("synthetic", "similarity")
        g.set((sim, M.reviewState, C.accepted))
        add(g, sim, M.reviewedBy, iri("agent", "mvo-importer-0.1.0"))
    else:
        g.add((ann, RDF.type, M.IdentityAssertion))
        g.set((ann, M.predicate, M.exactPhysicalSample))
        add(g, ann, M.linkResolution, "exact_matrix_cell")
    assert not validate_graph(g)[0], case


def test_roundtrip_literals(g, tmp_path):
    for n, literal in enumerate([Literal("bonjour", lang="fr"), Literal("01", datatype=XSD.integer, normalize=False), Literal("snow\n☃"), Literal("1.250", datatype=XSD.decimal, normalize=False)]):
        assert decode(encode(literal)) == literal
        add(g, iri("synthetic", "literal", str(n)), M.literalValue, literal)
    export(g, tmp_path / "projection", "test")
    assert canonical(restore(tmp_path / "projection")) == canonical(g)
    reparsed = Graph().parse(data=canonical(g), format="nt")
    assert canonical(reparsed) == canonical(g)


def test_composite_identity_no_collisions():
    assert iri("record", "a/b", "c") != iri("record", "a", "b/c")
    assert iri("record", "a", "x") != iri("record", "b", "x")
    with pytest.raises(ValueError):
        iri("record", "")


@pytest.mark.parametrize("raw", ["", "None", "unknown", "TRUE", "0"])
def test_no_missing_boolean_coercion(raw):
    with pytest.raises(ValueError):
        boolean(raw)


def test_path_escape(tmp_path):
    with pytest.raises(ValueError):
        resolve(tmp_path, "../secret")


OWNER = dict(tenant="emergentbiome", projects=["atlas"], purposes=["internal_review"])


def test_tools_scope_and_denominator(g):
    response = execute(g, {"query": "release_summary", "purpose": "internal_review"}, OWNER)
    assert response["rows"] == [dict(registered_records=3, data_complete_tri_views=2, excluded_records=1, mechanism_comparable=0)]


@pytest.mark.parametrize("bad", [{"query": "MATCH (n) DETACH DELETE n", "purpose": "internal_review"}, {"query": "release_summary", "purpose": "internal_review", "limit": 101}, {"query": "record_evidence", "purpose": "internal_review"}, {"query": "release_summary", "purpose": "internal_review", "tenant": "admin"}])
def test_request_contract(g, bad):
    with pytest.raises(ValidationError):
        execute(g, bad, OWNER)


def test_external_export_denied(g):
    with pytest.raises(AccessDenied):
        execute(g, {"query": "release_summary", "purpose": "external_export"}, OWNER)
    export_principal = {**OWNER, "purposes": ["external_export"]}
    with pytest.raises(AccessDenied):
        execute(g, {"query": "release_summary", "purpose": "external_export"}, export_principal)


@pytest.mark.parametrize("bad_principal", [dict(tenant="other", projects=["atlas"], purposes=["internal_review"]), dict(tenant="emergentbiome", projects=["other"], purposes=["internal_review"])])
def test_tenant_project_isolation(g, bad_principal):
    with pytest.raises(AccessDenied):
        execute(g, {"query": "release_summary", "purpose": "internal_review"}, bad_principal)


def test_expired_policy(g):
    add(g, iri("policy", "internal-evidence-review-v1"), M.expiresAt, "2026-01-01T00:00:00Z", XSD.dateTime)
    with pytest.raises(AccessDenied):
        execute(g, {"query": "release_summary", "purpose": "internal_review"}, OWNER)


def test_ambiguous_policy_fails_closed(g):
    add(g, iri("policy", "internal-evidence-review-v1"), M.tenant, "other-tenant")
    with pytest.raises(AccessDenied):
        execute(g, {"query": "release_summary", "purpose": "internal_review"}, OWNER)


def test_validation_preserves_literal_policy(g):
    import rdflib
    before = rdflib.NORMALIZE_LITERALS
    validate_graph(g)
    assert rdflib.NORMALIZE_LITERALS == before


def test_no_remote_graph_writes():
    with pytest.raises(ValueError):
        connect("bolt://example.com:7687", "neo4j", "testpassword")
