# Governance, extension and release gates

## What can be promoted now

Current MVO 0.2.0 is a local, executable ontology/evidence-graph foundation,
additive to the retained 0.1.0 graph snapshot. Automated checks establish
syntactic validity, local semantic consistency, admission behavior, reproducible
source identity and projection parity for the declared data slice. They do not
establish empirical validity, independent scientific review, registry acceptance
or operational certification. The exact current scope is in the
[molecular handoff](MOLECULAR_HANDOFF_20260926.md).

Required human roles are **proposed responsibilities**, not appointments:
ontology steward, warehouse/source steward, microbial ecology reviewer,
GHG-methodology reviewer, rights/community-governance reviewer and security
owner. The person responsible for a scientific claim must not be replaced by an
LLM's `accepted` status. Repository code review and domain review are distinct.

## Extension recipe

1. Define a decision and competency question; list prohibited interpretations.
2. Inventory source identity, rights, jurisdiction, sample resolution and assay
   grain. Preserve failures and missing evidence. Obtain authorized access.
3. Review existing standard terms before adding local ones. Record exact source
   version and mapping rationale. Do not use equivalence for approximate matches.
4. Add canonical Turtle, labels/definitions, SHACL and synthetic positive/negative
   fixtures. Add a source-specific adapter with explicit keys and null semantics.
5. Run OWL profile/reasoning, SHACL, source-key/hash checks, query regressions,
   roundtrip parity and policy tests. Generate the reference/context again.
6. Review the RDF axiom diff and shape tightening. Version ontology, shapes,
   mappings and tools independently; any identity change requires migration.
7. Build a new immutable graph snapshot. Never edit the August atlas freeze or
   reuse a graph ID for changed content. Mark the prior snapshot superseded only
   after the new one passes and an authorized person approves promotion.

Deprecations retain old IRIs and source evidence. Add replacement annotations
and a migration explanation rather than deleting provenance. Schema removal,
stronger claim scope, key changes and altered measurement meaning require a
major compatibility review even if an implementation edit is small.

## Prioritized next scientific releases

| Priority | Extension | Required evidence before promotion |
| --- | --- | --- |
| P0 | Common embedding configuration | Input-sequence hashes, layer/pooling/model fingerprints, same-DNA controls; re-embed where needed |
| P0 | Annotation event adapters | Accepted KOfam/best-ranked MCyc-SCyc semantics, tool/database versions, coverage, source keys; no raw-hit score equivalence |
| P0 | MUCC specimen/site crosswalk | Evidence-backed M1/M2/M3/habitat label reconciliation, date/depth/replication units; never label-only joins |
| P0 | Typed process data | Tower mapping repair provenance, chamber/tower/porewater distinction, units, times, instrument QC |
| P1 | Prospective molecular/process validation | Co-sampled DNA/RNA, abundance, environment, calibrated flux, independent replication and held-out assessment |
| P1 | CO2/N2O mechanism panels | Reviewed marker specificity and pathway direction/completeness; gas-specific outcome validation |
| P1 | Partner evidence bundle | Project boundary, baseline, protocol/version, uncertainty, tenure/consent, external review responsibilities |
| P2 | Authenticated enterprise service | Service identity, tenant isolation outside prompts, least-privilege graph user, audit logging, backup/restore drills, threat model |

## Operational controls and rollback

Inputs are read-only. Outputs and runtime assets are ignored local artifacts.
Graph snapshots are selected by immutable ID and must have `validated` status;
never query whichever import happens to be newest. Rollback selects the prior
validated snapshot; it does not delete source records or rewrite claims.

Retain `manifest.json`, graph hash, formal-validation logs, SHACL report,
Neo4j receipt, query-parity receipt, dependency/runtime locks and the command.
The source inventory hashes complete file contents, including large Parquet
files; no file is skipped merely because of its size. A successful catalog audit
does not substitute for full PK/FK audits of every source-specific fact table.

Monitor source drift, orphan references, excluded units, quarantined attempts,
SHACL violations, unknown rights, unresolved sample/measurement joins and
configuration incompatibility. These are evidence-health metrics, not methane
risk scores. Add query latency and failure/abstention metrics when a service is
deployed. No autonomous review-state promotion or external registry write path
exists in version 0.2.0.

An emergency response stops the local service and returns to the last validated
snapshot. Source artifacts and prior snapshots remain recoverable. Do not use
`MATCH (n) DETACH DELETE n`, recursive workspace cleanup, or a production graph
to reset a development test.
