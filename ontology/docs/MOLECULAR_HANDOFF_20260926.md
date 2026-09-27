# Molecular evidence extension — 2026-09-26 handoff

Status: implemented, source/semantic/query validated local slice. Independent
biological review, external rights clearance and production security approval
are **not** implied. The data-bearing whitepaper, cases, query receipts,
scientific review register and production QA remain in an ignored local review
bundle; they are not part of a clean public clone.

## Immutable snapshot and preservation

`mvo-d2f96653ce0645b1da62a836` lives at
`ontology/build/molecular-20260926-v2` (repository-relative).
Canonical RDF SHA-256:
`3a710cd0504b8b35807a3d437182527e6c38b13a1a26cfef679258896751fb11`.

The graph contains 735,060 statements and 37,071 resources. All 233,190 original
statements and four historical claims are unchanged. Total append-only growth
is 501,870 statements, including 745 final release/provenance statements. The
builder's `summary.new_triples=501125` is the pre-release-metadata subtotal,
not the complete graph difference. No count in this table is an independent
ecological sample count.

| Admitted evidence | Count / interpretation |
| --- | --- |
| Registered records / payload-ready / excluded | 7,965 / 7,710 / 255, unchanged |
| Mechanism-comparable records | 0, unchanged |
| Selected MAG records | 145: MSM 14; Futian 42; MUCC 73; POC 16 |
| POC selected strata | 3 wetland; 13 rumen methods controls |
| Primary source features / annotations | 934 / 934 |
| Verified loci / available protein records | 863 / 864; protein records are not unique sequences |
| Missing sequence/coordinates / unresolved mismatch | 70 / 1 |
| Exact Bakta crosswalks | 327 additional tool-native feature identities |
| Supplementary annotation assertions | 2,243; source method and grain retained |
| Full compact coverage ledger | 143,370 MAG × 18-family rows |
| Selected/negative graph evaluations / aggregates | 2,632 / 234 |
| Processed RNA cells | 1,596, from 12 genes × 133 source columns |
| Context samples / source site labels | 280 / 18; not verified physical specimens/sites |
| Selected MAG-to-sample context assertions | 9,510; exact matrix cells are not ecological presence |
| Numeric source measurements / missing-value gaps | 519 / 37 |
| Evidence packets / validation actions | 145 / 6 |
| Exact current POC–MUCC DNA-content overlaps | 23; not independent cohorts |
| Exact molecular–flux pairs | 0 |
| Active cross-run embedding similarity / credit decisions | 0 / 0 |

The complete class/relation inventory is in
`real-graph/real-graph-validation.json`. Source-feature class counts include
the auxiliary Bakta features; do not substitute that class count for the 934
primary selected features. The on-disk build was about 640 MiB at handoff;
the multi-snapshot Neo4j store was about 3.8 GiB, not the marginal cost of this
snapshot alone. Heap/page cache remain 768/256 MiB.

## Evidence-backed verification

- 119 ontology tests passed; five RDFLib deprecation warnings retained.
- OWL 2 RL profile, reasoner consistency and real-data SHACL passed.
- All 934 original primary rows match; the full DRAM key audit retains
  9,402,083 unique source-scoped keys and no duplicate/null keys.
- All 133 physical warehouse tables were audited: 390,950,494 heterogeneous rows.
  Five source/key issues remain explicit, not globally relabeled as failures.
- Both canonical imports reconstruct the same exact RDF hash. The first
  unbounded read-back timed out; the bounded indexed revision and repeat import
  passed without resetting the database or increasing the transaction timeout.
- The domain projection's full live inventory passed: 37,071 nodes,
  356,094 edges and 378,966 literal statement IDs.
- MQ01–MQ18 all executed, with complete-result source parity at declared columns.
  Original and successor release queries also passed SPARQL/Cypher parity.
- MQ03/MQ09/MQ18 matched SQL controls have identical complete rows. SQL warm
  medians were lower in all three measured local comparisons. No speed or
  analyst-time advantage is claimed.
- Root atlas foundation: 25,019 checks, zero errors; existing MUCC warning
  remains. Lane registry passes with its existing empty-gap-register warning.

## Query the existing validated snapshot

Run from repository root. If the service is stopped, start the authenticated
local runtime with `ontology/.venv/bin/python -m mvo local-neo4j start`; use the
documented retained-console mode if detached processes are cleaned up.

```bash
ontology/.venv/bin/python -m mvo.molecular_query_tool \
  --build ontology/build/molecular-20260926-v2 \
  --request '{"query":"MQ02","purpose":"internal_review","filters":{"lane":"mucc_v1_owc_wetland","minimum_evidence":"sequence_verified"},"limit":3}'

ontology/.venv/bin/python -m mvo.molecular_query_tool \
  --build ontology/build/molecular-20260926-v2 \
  --request '{"query":"MQ18","purpose":"internal_review","filters":{"lane":"mucc_v1_owc_wetland","proteome":"mucc_v1__OWC_0000","review_status":"pending"},"limit":10}'
```

The first example selects 481 sequence-verified rows from 550 MUCC source
annotation rows, displaying three. The second returns three blocking-gap rows
for one nominated packet. Both were executed and saved under `query-results/`.
Limits apply only after full bounded capture and explicit filters; registered
denominators stay unchanged. Unsupported filters, caller principals, arbitrary
query text and external-export purposes are rejected. This CLI uses local OS
trust plus Neo4j authentication; it is not remote user authentication.

## Fixed question pack, filters and scope

Queries are `queries/molecular/MQ01.cypher` through `MQ18.cypher`; the three
SQL controls are beside them. Exact plans, parameters, full result hashes,
source-parity columns, repeated timings and PROFILE DB hits are in the dated
bundle's `query-results/run-01/`. The question/decision preregistration is
`MOLECULAR_EXPANSION_20260926.md`. Result row counts range from 1 to 9,510;
the pack guards a 100,000-row result ceiling and 60-second query timeout.

Available Cypher filters are discovered per fixed query: lane, habitat, family,
site, proteome, process, date prefix and depth label. Inapplicable filters are
errors, not ignored requests. Time/depth filters operate on source labels,
not invented UTC intervals or harmonized depth ranges. The restricted CLI also
supports explicit post-selection by identity tier for MQ02/03/06/09 and by
claim-review state for MQ10/18. These do not modify coverage denominators or
silently declare unreviewed mechanisms accepted. The pack remains internal-only.

## Reload, re-run, rollback and reproduce

```bash
ontology/.venv/bin/python -m mvo local-neo4j load \
  --projection ontology/build/molecular-20260926-v2/neo4j
ontology/.venv/bin/python -m mvo.molecular_domain \
  --build ontology/build/molecular-20260926-v2 --out /tmp/mvo-domain-check
ontology/.venv/bin/python -m mvo.molecular_queries \
  --build ontology/build/molecular-20260926-v2 \
  --out results/reports/mvo_molecular_evidence_20260926/query-results/run-new
ontology/.venv/bin/python -m mvo neo4j-audit \
  --snapshot-dir ontology/build/atlas-20260926-validated
ontology/.venv/bin/python -m mvo local-neo4j stop
```

Use a **new** query output path. No blanket delete/reset is necessary. Rollback
is selecting the still-validated `mvo-c709e10af4abe9f3ff904eca` snapshot, whose
live regression receipt is retained. Stopping Neo4j leaves all snapshots on disk.

Canonical reproduction requires the exact 70-file build-time archive
`graph-pilots/molecular-v2-code.tar.gz`, matching source hashes and the manifest's
`recorded_at`, not the later operational source inventory. The bundle's
`engineering/code-archive-and-operational-revisions.json` verifies every archive
member. Restore these into an **isolated** source tree with the recorded input
layout and pinned environment; never overwrite the working repository or an
existing build. Then use `mvo.molecular_graph --out <new-build> --recorded-at
2026-09-26T20:40:17.955430+00:00`. A new code inventory intentionally yields a
successor snapshot rather than impersonating the frozen one.

`engineering/operational-code.tar.gz` separately binds the corrected loader and
demo tool. This distinction preserves immutable build provenance instead of
rewriting an old manifest to describe code that did not create it.

## Next safe expansion

Review the proposed panel with a qualified microbiologist; recover source
specimen/library/depth identities; obtain paired, unit-resolved field process
measurements; resolve external data/database rights. A held-out site/season
validation and uncertainty protocol must precede a prediction, risk tier or
project-level carbon claim. Salt marsh, seagrass and peatland remain extension
targets, not newly instantiated datasets. The website, August release pointer
and remote repository were not changed by that implementation task.
