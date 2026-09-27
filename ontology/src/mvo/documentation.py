"""Deterministic semantic reference and context generation; conservative RDF diff."""
import json
from pathlib import Path
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL, SKOS

from .model import M, load_bundle, RELEASE, canonical


def generate(output: Path):
    g = load_bundle("schema")
    output.mkdir(parents=True, exist_ok=True)
    context = {"mvo": str(M), "prov": "http://www.w3.org/ns/prov#", "id": "@id", "type": "@type"}
    lines = ["# Generated ontology reference", "", "Generated from canonical Turtle; do not edit independently.", "",
             f"Ontology version: {RELEASE['ontology_version']}. Local development namespace; not yet publicly served.", ""]
    for heading, cls in (("Classes", OWL.Class), ("Object properties", OWL.ObjectProperty), ("Datatype properties", OWL.DatatypeProperty)):
        lines.extend([f"## {heading}", "", "| Term | Meaning | Parent |", "| --- | --- | --- |"])
        for term in sorted(set(g.subjects(RDF.type, cls)), key=str):
            if not str(term).startswith(str(M)):
                continue
            name = str(term)[len(str(M)):]
            label = str(g.value(term, SKOS.definition) or g.value(term, RDFS.label) or "")
            parents = ", ".join(str(p).replace(str(M), "mvo:") for p in g.objects(term, RDFS.subClassOf))
            lines.append(f"| `{name}` | {label.replace('|', '/')} | {parents} |")
            context[name] = {"@id": str(term), "@type": "@id"} if cls == OWL.ObjectProperty else str(term)
        lines.append("")
    (output / "schema-reference.md").write_text("\n".join(lines))
    (output / "context.jsonld").write_text(json.dumps({"@context": context}, indent=2, sort_keys=True) + "\n")
    return dict(output=str(output), generated=["schema-reference.md", "context.jsonld"])


def compare(before: Path, after: Path):
    old, new = Graph().parse(before), Graph().parse(after)
    left, right = set(canonical(old).decode().splitlines()), set(canonical(new).decode().splitlines())
    removed, added = sorted(left - right), sorted(right - left)
    public_types = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty)
    old_terms = {str(s) for cls in public_types for s in old.subjects(RDF.type, cls)}
    new_terms = {str(s) for cls in public_types for s in new.subjects(RDF.type, cls)}
    return dict(added_triples=added, removed_triples=removed, removed_terms=sorted(old_terms-new_terms),
                added_terms=sorted(new_terms-old_terms),
                disposition="review_required" if added or removed else "unchanged",
                warning="Conservative RDF axiom diff, not proof of logical equivalence. Shape tightening and identity changes require migration review.")
