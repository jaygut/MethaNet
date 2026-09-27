"""Bounded, read-only tools. Trusted principal is supplied outside the tool payload."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from collections import Counter
from rdflib import URIRef
from rdflib.namespace import RDF
from jsonschema import Draft202012Validator, FormatChecker

from .model import M, ROOT, RELEASE, canonical, sha_bytes


class AccessDenied(ValueError):
    pass


class BudgetExceeded(ValueError):
    pass


def allowed(g, subject, principal, purpose, now):
    policies = list(g.objects(subject, M.hasPolicy))
    if len(policies) != 1:
        return False
    p = policies[0]
    # Enforce policy cardinality here too: query authorization must not depend
    # on the caller remembering to run SHACL on an arbitrary input graph.
    for pred in (M.tenant, M.projectKey, M.rightsStatus, M.externalExportAllowed):
        if len(list(g.objects(p, pred))) != 1:
            return False
    if len(list(g.objects(p, M.expiresAt))) > 1:
        return False
    one = lambda pred: str(g.value(p, pred))
    if one(M.tenant) != principal["tenant"] or one(M.projectKey) not in principal["projects"]:
        return False
    if purpose not in principal["purposes"] or purpose not in {str(x) for x in g.objects(p, M.purpose)}:
        return False
    expiry = g.value(p, M.expiresAt)
    if expiry:
        try:
            expires = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
            if expires.tzinfo is None or expires <= now:
                return False
        except ValueError:
            return False
    if purpose == "external_export":
        return one(M.rightsStatus) == "licensed" and one(M.externalExportAllowed) == "true" and bool(g.value(p, M.licenseURI))
    return purpose == "internal_review"


def execute(g, request: dict, principal: dict, snapshot_sha256: str | None = None, now=None):
    schema = json.loads((ROOT / "contracts/query-request.schema.json").read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(request)
    if len(g) > 2_000_000:
        raise BudgetExceeded("Graph exceeds local tool budget")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Authorization time must be timezone-aware")
    purpose, name, limit = request["purpose"], request["query"], request.get("limit", 25)
    if purpose not in principal["purposes"]:
        raise AccessDenied("Purpose not authorized")
    start = time.monotonic()
    def check(subject):
        if time.monotonic() - start > 10:
            raise BudgetExceeded("Query execution time exceeded")
        return allowed(g, subject, principal, purpose, now)
    result, truncated = [], False
    if name == "release_summary":
        counts = Counter()
        for subject in g.subjects(RDF.type, M.MolecularRecord):
            if check(subject):
                counts["registered_records"] += 1
                for key, pred in (("data_complete_tri_views", M.triViewReady), ("excluded_records", M.releaseExcluded), ("mechanism_comparable", M.mechanismComparable)):
                    counts[key] += str(g.value(subject, pred)) == "true"
        if not counts:
            raise AccessDenied("No authorized release records")
        result = [dict(counts)]
    else:
        classes = dict(validation_gaps=M.ValidationGap, source_inventory=M.Artifact, sample_readiness=M.SequencingSample)
        if name == "record_evidence":
            subjects = [URIRef(request["subject"])]
        else:
            # Artifact subclasses are explicitly included without broad OWL materialization.
            types = (M.Artifact, M.DatasetRelease, M.GraphRelease, M.Embedding) if name == "source_inventory" else (classes[name],)
            subjects = sorted({s for cls in types for s in g.subjects(RDF.type, cls)}, key=str)
        for subject in subjects:
            if not check(subject):
                if name == "record_evidence":
                    raise AccessDenied("Unrecognized or unauthorized subject")
                continue
            if len(result) == limit:
                truncated = True
                break
            facts = []
            for p, o in sorted(g.predicate_objects(subject), key=lambda x: (str(x[0]), str(x[1]))):
                # Internal resource references obey object-level policy too.
                if isinstance(o, URIRef) and str(o).startswith("https://emergentbiome.earth/id/") and not check(o):
                    continue
                facts.append(dict(predicate=str(p), value=str(o), kind="iri" if isinstance(o, URIRef) else "literal",
                                  datatype=str(getattr(o, "datatype", "") or ""), language=str(getattr(o, "language", "") or "")))
            result.append(dict(iri=str(subject), facts=facts))
    response = dict(query=name, snapshot_sha256=snapshot_sha256 or sha_bytes(canonical(g)),
                    tool_version=RELEASE["tool_version"], rows=result, truncated=truncated,
                    claim_boundary="Molecular evidence and review readiness only; not measured activity, calibrated risk or credit approval.")
    if len(json.dumps(response).encode()) > 65536:
        raise BudgetExceeded("Response exceeds byte budget; narrow the query")
    Draft202012Validator(json.loads((ROOT / "contracts/query-response.schema.json").read_text())).validate(response)
    return response
