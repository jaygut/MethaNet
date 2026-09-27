"""Offline semantic checks; OWL reasoning and closed-world admission are separate."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import rdflib

from pyshacl import validate
from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL

from .model import M, C, ROOT, load_bundle, canonical, digest


def shacl_validate(*args, **kwargs):
    """pySHACL 0.31 resets RDFLib's lexical policy; preserve our caller's setting."""
    normalize = rdflib.NORMALIZE_LITERALS
    try:
        return validate(*args, **kwargs)
    finally:
        rdflib.NORMALIZE_LITERALS = normalize


def validate_graph(data: Graph):
    schema = load_bundle("schema")
    conforms, report, text = shacl_validate(data, shacl_graph=load_bundle("shapes"), ont_graph=schema,
                                       inference="none", advanced=True, meta_shacl=True,
                                       do_owl_imports=False, inplace=False)
    errors = [] if conforms else [text]
    # No silent RDF identities, no dangling local entity links, no invented equivalence.
    for s, p, o in data:
        if isinstance(s, BNode) or isinstance(o, BNode):
            errors.append("Data resources must have stable IRIs (blank nodes prohibited)")
            break
        if p == OWL.sameAs:
            errors.append("owl:sameAs is not admitted; retain IdentityAssertion")
        if isinstance(o, URIRef) and str(o).startswith("https://emergentbiome.earth/id/") and not list(data.predicate_objects(o)):
            errors.append(f"Dangling internal endpoint: {o}")
    identity = set()
    for subject in data.subjects(RDF.type, M.MolecularRecord):
        key = (str(data.value(subject, M.laneId)), str(data.value(subject, M.proteomeId)))
        if key in identity:
            errors.append(f"Duplicate molecular identity {key}")
        identity.add(key)
    locus_identity = set()
    for subject in data.subjects(RDF.type, M.CalledLocus):
        key = tuple(str(data.value(subject, p)) for p in (M.forRecord, M.sourceNamespace, M.callerNamespace, M.geneId))
        if key in locus_identity:
            errors.append(f"Duplicate called-locus source identity {key}")
        locus_identity.add(key)
    # OWL disjointness check is explicit for instance admission; reasoner checks TBox separately.
    for left, right in schema.subject_objects(OWL.disjointWith):
        def instances(cls):
            subclasses = {cls} | set(schema.transitive_subjects(RDFS.subClassOf, cls))
            return {s for c in subclasses for s in data.subjects(RDF.type, c)}
        for subject in instances(left) & instances(right):
            errors.append(f"Disjoint class membership: {subject}: {left} / {right}")
    return not errors, errors, report


def verify_schema(output: Path, robot: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    schema, vocab, shapes = (load_bundle(x) for x in ("schema", "vocabulary", "shapes"))
    errors = []
    for folder in ("schema", "vocabulary", "shapes"):
        g = load_bundle(folder)
        clone = Graph().parse(data=g.serialize(format="turtle"), format="turtle")
        if canonical(g) != canonical(clone):
            errors.append(f"RDF roundtrip failure: {folder}")
    for term in schema.subjects(RDF.type, OWL.Class):
        if str(term).startswith(str(M)) and not (schema.value(term, RDFS.label) and schema.value(term, URIRef("http://www.w3.org/2004/02/skos/core#definition"))):
            errors.append(f"Undocumented local class: {term}")
    declared = set(schema.subjects()) | set(vocab.subjects())
    for _, p, o in schema + shapes:
        for term in (p, o):
            if p != OWL.versionIRI and isinstance(term, URIRef) and str(term).startswith(str(M)) and term not in declared:
                errors.append(f"Undeclared term: {term}")
    # Meta-SHACL checks shapes themselves, with an empty instance graph.
    shacl_validate(Graph(), shacl_graph=shapes, ont_graph=schema, meta_shacl=True, advanced=True)
    merged = output / "mvo.owl.ttl"
    merged.write_text(schema.serialize(format="turtle"))
    result = dict(schema_triples=len(schema), vocabulary_triples=len(vocab), shape_triples=len(shapes),
                  classes=len(set(schema.subjects(RDF.type, OWL.Class))),
                  object_properties=len(set(schema.subjects(RDF.type, OWL.ObjectProperty))),
                  datatype_properties=len(set(schema.subjects(RDF.type, OWL.DatatypeProperty))),
                  errors=errors, robot_status="not_run", reasoner_status="not_run")
    if robot:
        commands = [
            ["validate-profile", "--profile", "RL", "--input", str(merged), "--output", str(output / "owl2rl-profile.txt")],
            ["reason", "--reasoner", "hermit", "--input", str(merged), "--output", str(output / "reasoned.owl"), "--equivalent-classes-allowed", "asserted-only"]]
        for i, command in enumerate(commands):
            run = subprocess.run(["java", "-Xmx1g", "-jar", str(robot), *command], capture_output=True, text=True, timeout=120)
            (output / ("profile.log" if i == 0 else "reasoner.log")).write_text(run.stdout + run.stderr)
            name = "robot_status" if i == 0 else "reasoner_status"
            result[name] = "pass" if run.returncode == 0 else "fail"
            if run.returncode:
                errors.append(f"{name} failed; inspect logs")
        result["robot_sha256"] = digest(robot)
    result["status"] = "pass" if not errors else "fail"
    (output / "schema-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
