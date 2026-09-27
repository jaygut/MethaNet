# Molecular Verification Ontology architecture

Decision record: 2026-09-26; documentation reconciled 2026-09-27. Current
package version: 0.2.0, additive to the preserved 0.1.0 snapshot. Scope: a
working local ontology/evidence-graph foundation, not an independently approved
verification platform. Scientific, rights, and commercial release authorization
remain human gates. See the [0.2.0 molecular handoff](MOLECULAR_HANDOFF_20260926.md)
and [verification history](VERIFICATION.md) for the exact current snapshot.

## Product purpose and differentiated value

The first decision supported is **which molecular evidence can be used for
which review question, and which additional measurements would resolve the
blocking uncertainty?** The first deliverable to a project developer or verifier
is an evidence-readiness dossier, not an issued credit or predicted tonnes CO2e.

Differentiation should come from reconstructable sequence-to-claim provenance,
source-aware mechanism evidence, assay/configuration compatibility, and targeted
validation design. A large graph or an embedding neighborhood alone is not a
competitive scientific claim. Customers benefit when the system can explain why
apparently convincing evidence must not yet be used in a credit calculation.

CH4, CO2 and N2O are separate gas concepts. Molecular hypotheses cover methane
production/oxidation, carbon fixation/respiration/decomposition, nitrification,
denitrification and N2O reduction. A marker can have multiple interpretations;
gene detection is neither pathway completeness nor expression, directionality,
rate, net gas balance, additionality or permanence. Sulfate reduction is relevant
context, not an automatic deterministic methane suppression rule.

## Store and authority boundaries

```text
Pinned atlas release + Parquet/TSV + source artifacts
                  │ read-only hashes, keys, row/byte audits
                  ▼
        Typed mapping + explicit quarantine/gap records
                  │ SHACL + identity + claim admission gates
                  ▼
     Immutable RDF snapshot / named graph / source receipts
          │                 │                    │
          ▼                 ▼                    ▼
   Neo4j projection   Read-only local tools   Review evidence bundle
   exact roundtrip    object/purpose policy   human decision boundary

Embeddings / large matrices / raw genes remain warehouse assets.
They are evidence inputs, not alternative authorities for identity or truth.
```

| Concern | Authority | What this package implements |
| --- | --- | --- |
| Frozen cohort and payload availability | Current atlas pointer, freeze and ledger | Exact pins and denominator checks |
| Large gene/annotation/matrix facts | Source warehouse tables | Full-byte artifact hashes and row/byte catalog; no silent raw-hit aggregation |
| Meaning and class relations | `schema/*.ttl` | RDF 1.1 / OWL 2 RL, formal class/property hierarchy and disjointness |
| Controlled categories | `vocabulary/*.ttl` | SKOS schemes; no class/concept identity conflation |
| Admission requirements | `shapes/*.ttl` | SHACL Core and SPARQL constraints plus key/endpoint checks |
| Record-level provenance | Data graph and manifest | Source artifacts, row locators, activities, versions, checksums |
| Graph navigation | Derived Neo4j snapshot | Lossless statement nodes plus traversable IRI relations |
| Rights and action authority | Trusted local principal + object policy | Deny external export with unknown rights; no mutation tools |
| Field validation or certification | Independent measurements and authorized reviewers | Explicit gaps and blocked admission; not supplied by this package |

No source warehouse, atlas freeze, public website, or report alias is changed by
this build. MVO is a downstream evidence projection, not a new atlas release.
Version 0.2.0 adds selected feature, locus, annotation, RNA, context-measurement,
alternative, and gap evidence to the preserved release-record layer; it does
not materialize every fact in the 7,965-record denominator.

## Canonical semantic modules

`core.ttl` defines artifacts, releases, evidence, qualified assertions, uncertainty,
scope, methods, activities, agents, reviews, policy and decisions. `molecular.ttl`
adds lane-scoped records, curation attempts, genes/proteins, annotation assertions,
embedding configurations, similarity/identity assertions and assay records.
`environment.ttl` separates sites, physical specimens, sampling/custody events,
observations, flux, concentrations, quantities and observation windows.
`verification.ttl` adds project, intervention, baseline, carbon pool, reporting
period, GHG statement, protocol requirement, tenure, consent, monitoring,
accounting, bridge hypotheses and external review disposition. `alignment.ttl`
provides conservative SOSA property specialization.

Modules are loaded together as an offline ontology. External terms are declared
and selectively aligned; full remote ontologies are not silently imported.
Consequently, local OWL profile/consistency results do not certify the entirety
of PROV, SOSA, QUDT, GeoSPARQL or a downstream consumer's merged import closure.
There are no existential rules that create observations or authorize decisions.

## Identity, claims and time

Molecular IRIs encode `(lane_id, proteome_id)` with reversible percent-escaping.
The same sequence in two source lanes remains two records. Exact-sequence
evidence can support an `IdentityAssertion`; `owl:sameAs` is not admitted. Attempt
identity adds `cohort_run_id` and `run_id`. Human labels, source MAG identifiers,
BioProjects and site names are never global identity keys.

An assertion is an independently addressable resource carrying subject,
predicate and either an IRI object or typed literal, plus epistemic/review state,
source, agent, method, applicability, uncertainty, rights and transaction time.
Supporting, contradictory, superseding and withdrawn evidence are retained.
Reifying a proposition **does not assert its unqualified triple**. This is how
unvalidated bridge hypotheses coexist with observations safely.

`validFrom/validUntil` describe applicability; `recordedAt/transactionEnd`
describe record history. An unresolved source time remains unresolved. Import
time is never relabeled as collection time. Immutable graph snapshots provide
append-only versions; a future correction uses a new snapshot and supersession
links rather than editing historical source evidence.

## Scientific admission and abstention

The current profile allows screening evidence, imported availability assertions,
review readiness and hypotheses. It rejects unsupported physical-sample
crosswalks, accepted incompatible embedding comparisons, no-hit claims without
covered assays, malformed flux quantities, accepted hypothesis bridges,
independently-validated bridge status, accepted GHG statements, and credit
approval decisions. The latter gates require a future independently reviewed
profile; changing a prompt cannot bypass them.

The data model permits later expression and abundance observations, but the
current real-data adapter catalogs their matrices without turning expression
cells into DNA abundance or sample-to-flux pairs. The generic legacy process
table is cataloged, not admitted as measured flux: its known tower mapping
problem requires typed-source reconciliation first. Zero successful ecological
joins does not mean there are no staged field measurements.

The August ESM-2 availability count does not prove a common vector space.
Optional post-freeze diagnostics are separate evidence for compatibility
review; they are not silently promoted into the August release.
No cross-run similarity edges are generated. gLM2 protocol classes and functional
evidence contracts are preserved separately.

## Neo4j representation and agent boundary

`MVOResource(snapshot, iri)` identifies a resource in a graph snapshot.
`MVOStatement(snapshot, id)` preserves each triple including literal lexical
form, datatype and language. `MVO_HAS_STATEMENT`, `MVO_OBJECT` and `MVO_REL`
enable provenance and domain traversal. Relations retain their full predicate
IRI; no lossy label conversion or implicit semantic promotion occurs. Class IRIs
remain in the resource's `types` property. Human-friendly named Cypher and SPARQL
queries are included. This conservative projection is intentionally more verbose
than a domain-specific denormalized graph; specialized views may be compiled
later with their own parity tests.

Snapshots progress from `loading` to `validated` only after row counts and a full
RDF reconstruction match. Import is idempotent; existing snapshot IDs cannot be
reused for different RDF content. Readers must select `validated` snapshots.
The database is local, authenticated, and loopback-only. Community Edition is
not represented as production multi-tenant authorization.

The executable CLI exposes five named read-only tools. A tool request cannot
select its tenant, add Cypher, mutate review state or issue credits. Trusted
principal configuration lives outside the request. Per-object tenant, project,
purpose, expiry and external-licensing checks apply before results are returned;
limits constrain rows, traversal, graph size, time and bytes. This is an
OS-trusted local interface, **not remote authentication or a deployed MCP/API**.
An enterprise deployment must add authenticated gateway/service identity,
database least-privilege accounts, secret management and security review.

## Explicit architecture decisions

1. RDF 1.1 + explicit assertion resources are the stable interchange baseline.
   RDF 1.2 triple-term support can be added without making it a dependency now.
2. OWL 2 RL defines entailment; SHACL and explicit application guards define
   admission. Passing either is not empirical scientific validation.
3. Canonical Turtle is the semantic authority. JSON-LD context and schema reference
   are generated; they are not separately hand-maintained semantic models.
4. Neo4j is a reversible projection, never the only copy of meaning or provenance.
5. The graph stores evidence relationships, not millions of duplicate vector or
   matrix values. Bounded, separately validated adapters can add focused slices.
6. No ontology term or protocol mapping constitutes external endorsement.
7. New gases/habitats reuse evidence/scoping contracts. They do not inherit
   methane calibration or automatically satisfy another ecosystem's protocol.
8. Deployment-grade controls and independent review are gates, not achievements
   inferred from successful unit tests.
