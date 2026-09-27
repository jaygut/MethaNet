# SQL and knowledge-graph store comparison

Measurement note: 2026-09-26. This is a bounded local comparison of three
preregistered molecular evidence questions. It is not a general benchmark,
production capacity test, or user study. Read it with the
[MVO 0.2.0 handoff](MOLECULAR_HANDOFF_20260926.md) and the
[preregistered query plan](MOLECULAR_EXPANSION_20260926.md).

## Decision in brief

The implemented asset is the governed semantic model and its traceable
evidence contracts: stable identity, qualified assertions, explicit provenance,
review state, alternatives, missingness, and claim boundaries. OWL/SHACL
semantics, source mappings, fixed query definitions, and admission policy
collectively determine what a query means. Ontology alone does not create
correct answers; trustworthy source mappings and validated inputs are also
necessary.

The warehouse remains the source authority. SQL is the natural interface for
tabular selection, filters, joins, and aggregation. RDF is the canonical
semantic/interchange snapshot. Neo4j is a rebuildable relationship projection
that supports explicit multi-hop navigation and evidence-path inspection. The
systems answer different operational needs; neither changes the evidence or
scientific truth.

In the three matched controls below, SQL returned exact full-row equivalents
with lower warm medians. That observation supports SQL for these workloads. It
does not establish that graph stores are generally slower, that a graph is a
bottleneck in every workload, or that the graph has no product value.

## What was compared

- Same admitted and normalized evidence facts; SQL setup was disclosed
  separately and was not charged to each query.
- The SQL control is in-process DuckDB over a prepared Parquet-backed normalized
  fact projection. It is not evidence about a production-persistent SQL server.
- Three fixed questions selected before viewing their result or speed:
  competing interpretations (MQ03), nearest admissible molecular-to-field path
  (MQ09), and a review packet with its gaps (MQ18).
- Complete output rows, declared grain, and provenance were checked, not just
  displayed subsets or aggregate counts.
- Three warm sequential local timings per query. The SQL setup to materialize
  normalized fact tables was 0.317 s in this run.
- Full receipts, result hashes, plans, and source-parity fields are retained in
  the ignored local review bundle and are not distributed with this document.

| Fixed question | Complete rows | Neo4j warm median | DuckDB SQL warm median | Exact row parity |
| --- | ---: | ---: | ---: | --- |
| MQ03: competing interpretations | 1,032 | 71.64 ms | 18.27 ms | Pass |
| MQ09: nearest admissible partial chain | 1,596 | 111.72 ms | 28.46 ms | Pass |
| MQ18: diligence packet and gaps | 435 | 51.63 ms | 16.05 ms | Pass |

These are local medians for this snapshot, implementation, machine, and query
set. There was no controlled concurrency, cold-cache, memory, storage-footprint,
scale-out, service-latency, or analyst-effort study. Do not use the timings as a
production service-level objective or as a graph-versus-SQL market claim.

## What the results do and do not establish

They establish that the three SQL controls can reproduce the graph's complete
declared answers, including provenance fields and gaps, over the shared
normalized facts. In this test, their warm query medians were lower. They also
show that the ontology-backed evidence model can be queried through more than
one serving representation without changing the selected evidence.

They do not establish independent biological validity, completeness of the
source datasets, truth of an annotation, sample-level ecological pairing,
measured methane flux, prediction quality, calibrated risk, credit eligibility,
or an improvement in review decisions. Query parity is software/data parity at
the declared grain, not scientific validation.

The graph's plausible product contribution is inspectable semantic
relationships: follow an assertion to its source row, method, review state,
alternatives, related evidence, and blocking gap. Whether this improves partner
comprehension or review decisions remains a hypothesis and needs a user study.
The SQL model can also return provenance when those relations are represented
as normalized keys and joins; the relevant comparison is total workflow
reliability and usability, not query syntax alone.

## Recommended architecture

```text
Source files and warehouse tables
            │ verified keys, hashes, row contracts
            ▼
Canonical source facts + source mappings + claim/admission policy
            │
            ├── SQL / Parquet projections for tabular analytics and aggregation
            ├── RDF snapshots for ontology semantics and portable assertions
            └── Neo4j projection for bounded relationship-path inspection
```

Keep warehouse tables and their source manifests authoritative for source
measurements and cohort denominators. Keep RDF snapshots immutable and
rebuildable from pinned inputs. Treat Neo4j as a query-serving projection, not a
second authority. Give fixed query contracts the same filters, evidence scope,
output grain, status handling, provenance, and abstention rules in each store.
Test complete result parity when either projection changes.

Choose the store per operation:

- Use SQL for cohort rollups, grouping, filtering, feature tables, statistical
  analysis, and large scans over columnar data.
- Use graph traversal when a task naturally follows a bounded path across
  source, entity, assertion, alternative, review, and gap relationships.
- Use ontology semantics and validation independently of the serving engine;
  a graph database does not itself enforce the scientific meaning of an edge.
- Export only a rights-reviewed, allowlisted projection to a public experience.
  Local Neo4j authentication and tool policy are not a deployed enterprise
  security boundary.

## Evidence and reproducibility pointers

- [Current molecular handoff](MOLECULAR_HANDOFF_20260926.md): snapshot ID/hash,
  included evidence, explicit gaps, query commands, and recovery.
- [Verification history](VERIFICATION.md): original 0.1.0 release and additive
  0.2.0 evidence snapshot.
- [Expansion preregistration](MOLECULAR_EXPANSION_20260926.md): fixed questions,
  selection rules, and comparison safeguards.

The current molecular package and detailed receipts remain an internal
implementation/review asset. Scientific review and external source/data rights
clearance are separate from code correctness and are still required before
redistributing detailed source records or a data-bearing bundle.
