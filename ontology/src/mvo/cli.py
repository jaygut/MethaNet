"""Local commands. Network/database writes occur only in explicit neo4j-load."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph
from .model import ROOT, canonical, sha_bytes
from .validation import verify_schema, validate_graph
from .fixtures import synthetic
from .ingest import build_atlas
from .release import materialize
from .projection import export, load_neo4j
from .query import execute


def timestamp(value=None):
    value = value or datetime.now(timezone.utc).isoformat(timespec="seconds")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("recorded-at requires timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--out", type=Path, default=ROOT / "build/schema")
    v.add_argument("--robot", type=Path)
    f = sub.add_parser("fixture")
    f.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", type=Path, default=ROOT.parent)
    b.add_argument("--metadata", type=Path)
    b.add_argument("--diagnostic", type=Path, action="append", default=[])
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--recorded-at")
    for name in ("validate", "query", "project"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--graph", type=Path, required=True)
        if name == "query":
            cmd.add_argument("--request", required=True, help="JSON request, not SPARQL/Cypher")
            cmd.add_argument("--principal", default="local-owner")
        if name == "project":
            cmd.add_argument("--out", type=Path, required=True)
    n = sub.add_parser("neo4j-load")
    n.add_argument("--projection", type=Path, required=True)
    n.add_argument("--uri", default="bolt://127.0.0.1:17687")
    n.add_argument("--user", default="neo4j")
    n.add_argument("--database", default="neo4j")
    n.add_argument("--receipt", type=Path)
    runtime = sub.add_parser("local-neo4j")
    runtime.add_argument("action", choices=["install", "start", "stop", "load"])
    runtime.add_argument("--projection", type=Path)
    runtime.add_argument("--receipt", type=Path)
    document = sub.add_parser("document")
    document.add_argument("--out", type=Path, required=True)
    diff = sub.add_parser("diff")
    diff.add_argument("--before", type=Path, required=True)
    diff.add_argument("--after", type=Path, required=True)
    diff.add_argument("--out", type=Path)
    audit = sub.add_parser("neo4j-audit")
    audit.add_argument("--snapshot-dir", type=Path, required=True)
    audit.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.command == "document":
        from .documentation import generate
        result = generate(a.out)
    elif a.command == "diff":
        from .documentation import compare
        result = compare(a.before, a.after)
        if a.out:
            a.out.write_text(json.dumps(result, indent=2) + "\n")
    elif a.command == "neo4j-audit":
        from .neo4j_audit import audit
        result = audit(a.snapshot_dir)
        if a.out:
            a.out.write_text(json.dumps(result, indent=2) + "\n")
    elif a.command == "local-neo4j":
        from . import local_runtime
        if a.action == "load":
            if not a.projection:
                p.error("local-neo4j load requires --projection")
            result = local_runtime.load(a.projection)
        else:
            result = getattr(local_runtime, a.action)()
        if a.receipt:
            a.receipt.write_text(json.dumps(result, indent=2) + "\n")
    elif a.command == "verify":
        result = verify_schema(a.out, a.robot)
        if result["status"] != "pass":
            print(json.dumps(result, indent=2))
            raise SystemExit(1)
    elif a.command == "fixture":
        result = materialize(synthetic(), {"scope": "SYNTHETIC_TEST_ONLY", "registered_records": 3}, [], a.out, "2026-09-26T00:00:00+00:00")
    elif a.command == "build":
        recorded = timestamp(a.recorded_at)
        g, summary, sources = build_atlas(a.repo, recorded, a.metadata, a.diagnostic)
        result = materialize(g, summary, sources, a.out, recorded)
    elif a.command == "neo4j-load":
        result = load_neo4j(a.projection, a.uri, a.user, os.environ.get("MVO_NEO4J_PASSWORD", ""), a.database)
        if a.receipt:
            a.receipt.write_text(json.dumps(result, indent=2) + "\n")
    else:
        g = Graph().parse(a.graph, format="nt")
        if a.command == "validate":
            ok, errors, _ = validate_graph(g)
            result = dict(conforms=ok, errors=errors)
            if not ok:
                print(json.dumps(result, indent=2))
                raise SystemExit(1)
        elif a.command == "query":
            principals = json.loads((ROOT / "config/access.json").read_text())
            if a.principal not in principals:
                raise ValueError("Unrecognized trusted principal")
            result = execute(g, json.loads(a.request), principals[a.principal], sha_bytes(canonical(g)))
        else:
            result = export(g, a.out, "mvo-" + sha_bytes(canonical(g))[:24])
    if "summary" in result:
        result["summary"] = dict(result["summary"])
        for key in ("quarantined_attempts", "unresolved_site_references"):
            if key in result["summary"]:
                result["summary"][key + "_count"] = len(result["summary"].pop(key))
    print(json.dumps(result, indent=2, sort_keys=True))
