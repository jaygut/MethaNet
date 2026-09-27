# MVO verification history

## Current additive local snapshot — 0.2.0

The current validated local evidence snapshot is documented in the detailed
[2026-09-26 handoff](MOLECULAR_HANDOFF_20260926.md). It is additive to, and does
not invalidate, the historical 0.1.0 snapshot below.

| Item | Current result |
| --- | --- |
| Package ontology/shapes/mappings | 0.2.0; canonical lossless Neo4j statement projection remains 0.1.0 |
| Graph ID | `mvo-d2f96653ce0645b1da62a836` |
| Canonical RDF SHA-256 | `3a710cd0504b8b35807a3d437182527e6c38b13a1a26cfef679258896751fb11` |
| Graph size | 735,060 statements; 37,071 resources |
| Preserved original graph | 233,190 statements, unchanged |
| Selected molecular slice | 145 MAGs; 934 primary features/annotations; 863 verified loci; 1,596 processed RNA cells; 519 numeric measurements; 37 missing-value gaps |
| Full release denominator | 7,965 registered records; 7,710 complete tri-views; 255 exclusions; 0 mechanism-comparable units |
| Fixed queries | MQ01–MQ18 executed with declared source-parity checks |
| SQL controls | MQ03/MQ09/MQ18 complete rows and provenance matched exactly; SQL warm medians lower in these three local tests |
| Exact molecular-to-flux pairs | 0 |
| Biological, rights, and production approval | Pending; local implementation verification is not those approvals |

The complete per-query receipts, Neo4j reconstruction and domain projection
audits, warehouse/source audits, build commands, and retained gaps are in the
handoff and local-only `results/reports/mvo_molecular_evidence_20260926/`
bundle. The local build is under the ignored
`ontology/build/molecular-20260926-v2/` directory and is not included in a
clean clone. See the dedicated [SQL/Neo4j comparison](STORE_COMPARISON_20260926.md)
for benchmark scope and limitations.

## Historical end-to-end verification — 0.1.0 (2026-09-26)

Outcome: **a materialized local ontology and evidence-graph foundation**, not
just an architecture proposal. The canonical OWL/SHACL package, real-source
adapter, local query tools and Neo4j projection are implemented and tested.
Independent scientific signoff, field calibration, protocol acceptance and
enterprise service certification are not claimed.

The [machine-readable receipt](verification-20260926.json) records exact results.
The local graph and complete per-file receipts are in
`ontology/build/atlas-20260926-validated/` (intentionally ignored by Git).

## Final snapshot

- Graph ID: `mvo-c709e10af4abe9f3ff904eca`.
- Transaction timestamp: `2026-09-26T14:36:15+00:00`.
- Canonical RDF SHA-256: `5fce9c99a86c6e8fc4fd751bfd6497ff45a2b905d89b75bb7bf732504d6559c5`.
- Source release: `molecular-atlas-20260810`, unchanged.
- 233,190 RDF statements; 14,266 IRI resources in the Neo4j projection.
- 157 complete input files hashed, totaling 4,207,847,977 bytes read; source
  files were not copied into the graph or modified.
- 57 local classes, 47 local object properties and 78 local datatype properties.
  The merged ontology also declares selected external vocabulary terms.

| Real materialization | Count | Interpretation |
| --- | ---: | --- |
| Registered lane/proteome records | 7,965 | Not unique biological genomes across lanes |
| Data-complete tri-views | 7,710 | Availability/schema completeness, not validated common mechanism space |
| Release mechanism-comparable records | 0 | Original claim boundary preserved |
| Explicit release exclusions | 255 | Retained with reasons and next actions |
| In-scope curation attempts | 5,261 | Includes source run-status evidence, not only selected successful attempts |
| Out-of-release attempts | 23 | Quarantined rather than inserted into the release denominator |
| Cataloged physical warehouse tables | 133 | Full-byte hashes and Parquet row/byte parity |
| Sequencing-sample records | 280 | No implicit conversion to physical specimens |
| Source site records | 18 | Source contexts, not proof of exact specimen location |
| Unresolved MUCC sample-site references | 133 | Source site labels have no exact lane-scoped dimension endpoint |
| Unresolved sample-flux windows | 133 | Retained as gaps; no measured molecular-to-flux relation asserted |

The metadata catalog retains pointers to 259,084 processed expression rows,
293,245 sample/MAG link rows and 34,835 generic process/flux rows. Those are
**catalog counts**, not individually admitted graph observations, exact physical
links, independent replicates or validated field pairs. The known tower-mapping
and source-resolution limitations remain outside this graph's claim scope.

## Verification performed

| Check | Result |
| --- | --- |
| Canonical RDF parse/roundtrip, labels, term declarations, SHACL metavalidation | Pass |
| OWL 2 RL profile via ROBOT/OWLAPI | Pass |
| HermiT consistency and unsatisfiable-class checks | Pass |
| HermiT negative control with intentionally contradictory class | Correctly rejected |
| New ontology test suite | 68 passed; 0 failed |
| Existing atlas-foundation tests | 8 passed; 0 failed |
| Existing full local atlas validator | 25,019 checks; 0 errors; original MUCC warning retained |
| Full real graph SHACL, identity, exclusion, endpoint and source parity | Pass |
| RDF → JSONL projection → RDF | Exact, including typed-literal lexical forms and language tags |
| Live authenticated Neo4j import → complete RDF reconstruction | Exact final graph SHA-256 |
| Second import of the same real snapshot | Same counts/hash; idempotent |
| Named SPARQL versus Cypher lane and exclusion queries | Exact agreement |
| Source/code receipt against current package | Match |
| Isolated dependency consistency | `pip check` passed |
| CI configuration | Added and YAML parsed; remote GitHub Actions not run or claimed |

The 68 tests include invalid provenance, duplicate identity, hash/denominator
drift, unreviewed acceptance, unsupported physical-sample links, incompatible
embedding comparison, missing/no-hit confusion, wrong flux units, reversed
time, unknown export rights, tenant/project isolation, expiry, malformed policy,
arbitrary-query rejection, immutable releases, and documentation consistency.
Five dependency deprecation warnings concern RDFLib's internal TriG APIs; they
are visible and do not indicate failed roundtrip or scientific validation.

A regression test protects against pySHACL resetting RDFLib literal
normalization after validation. The wrapper preserves the prior lexical policy;
lossless projection is tested on noncanonical numeric spellings and language
tags, not only on the current release's simple values.

## New findings and retained scientific limits

The import surfaced **133 unresolved MUCC site references** in the generic
metadata layer. Labels such as M1 or habitat labels do not match the lane-scoped
site dimension. Samples remain present with their original source rows and
explicit gaps; no label-based physical location is manufactured.

Post-freeze embedding diagnostics are included as checksum-pinned evidence for a
compatibility gap. There are **zero active cross-run similarity
edges**. This sprint does not re-embed proteins, resolve missing historical
sequence bytes, normalize mixed gLM2 replicate protocols, or make source-scaffold
functional scores interchangeable with pipeline-derived features.

The source atlas validator retains its MUCC non-pass gates. Successful graph
compilation means the graph faithfully represents the available evidence and
its gaps; it does not close those empirical or source-reconciliation gaps.
No exact sample-flux pairs, independent validation bridges, calibrated risk
tiers, accepted GHG statements or credit approvals were created.

## Handoff and maturity

Neo4j Community 5.26.31 was actually run locally with authentication and
loopback-only listeners. The verification server is stopped after testing;
its stored graph remains available. See the [restart/load/query runbook](../README.md).
The final canonical RDF file is 65,335,208 bytes. Runtime data also contains
earlier test snapshots; its size is not a benchmark for one optimized graph.

The package establishes the formal-schema and reconstructable-evidence
foundation described in the Notion guideline. It does not claim all
organizational maturity levels or production service controls. Human ontology,
microbial-ecology, GHG-methodology, rights and security reviews remain explicit
promotion gates. A protocol-facing evidence interface is proposed, not certified.

The public website/report, GitHub Pages aliases, frozen atlas and user-owned
in-progress manuscript/report edits were not changed. No commit, push, registry
submission or external publication was made. Apart from this dedicated package,
the change adds a focused CI workflow and a pointer from the existing knowledge
graph design document.
