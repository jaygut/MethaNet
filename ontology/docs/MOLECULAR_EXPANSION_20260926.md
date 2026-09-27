# Molecular expansion execution register

Status: preregistration and execution design, superseded by the completed local
0.2.0 implementation documented in the [molecular handoff](MOLECULAR_HANDOFF_20260926.md)
and [verification history](VERIFICATION.md). Keep the pre-result question,
selection, and comparison rules below as the design record; use the handoff for
observed outcomes and current artifact pointers.

The authorized specification is `prompts/molecular_evidence_graph_expansion_20260926.md`.
The resumable execution ledger is
`results/reports/mvo_molecular_evidence_20260926/execution-status.json` (repository-relative).
The August atlas pointer and `build/atlas-20260926-validated` are immutable inputs.
No publication, deployment beyond authenticated loopback Neo4j, or external approval is authorized.

## Questions registered before schema changes and molecular result inspection

All queries operate on a specified validated snapshot and retain source contracts.
Unknown, excluded, failed and ambiguous evidence is returned, not silently filtered.

| Question | Decision | Output grain | Required evidence | Acceptable missingness | Prohibited interpretation | Query | Verification |
|---|---|---|---|---|---|---|---|
| CQ-M01: What process candidates are encoded? | Select candidates for review | lane × MAG × panel family | annotation event, accepted-hit semantics, QC, coverage | family not assayed remains explicit | MAG count as independent ecosystem replication | MQ01, MQ02, MQ05 | Source-event and denominator parity |
| CQ-M02: What supports an exact molecular statement? | Admit or quarantine identity | source-scoped locus/protein/event | source digest, row, caller, coordinates or explicit gap, sequence digest when available | absent sequence must remain unavailable | bare gene-ID join across callers | MQ02, MQ10 | Identity/translation/digest tests |
| CQ-M03: Which homolog interpretations compete? | Plan discriminatory validation | annotation interpretation set | accessions, context, alternatives, review state | phylogeny missing blocks specificity | pmo/amo or MCR hit as unique process proof | MQ03, MQ15 | Homolog-negative controls and read-only sensitivity |
| CQ-M04: What does non-detection mean? | Distinguish assay failure from no-hit | release MAG × panel family | run status, input coverage, accepted events, MAG QC | missing/failed/excluded separate | incomplete MAG no-hit as biological absence | MQ01, MQ04 | Full registered denominator recovery |
| CQ-M05: What is the ecological resolution? | Choose admissible comparison | identity edge and biological unit | source label, actual sample/site/depth/date, evidence grain | candidate and unresolved edges retained | assembly/MAG as specimen/site | MQ07, MQ13 | Exact-edge endpoint and fan-out audit |
| CQ-M06: What blocks molecular/flux pairing? | Acquire the missing source record | proposed source-to-observation path | source identity, time/space support, modality | nearest valid partial chain | shared site label as exact field pairing | MQ09, MQ17 | Zero-pair control unless exact source supports pair |
| CQ-M07: What was actually measured? | Select compatible observations | typed observation | quantity, unit, modality, normalization, window, method | unknown units/methods quarantined | RNA as DNA or concentration as flux | MQ06, MQ08, MQ16 | Modality and tower-mapping negative controls |
| CQ-M08: What may be shared? | Internal review versus external excerpt | evidence/claim/policy | rights, source release, purpose, reviewer | unknown rights restrict external use | public source as unrestricted redistribution permission | MQ10, MQ18 | Tenant, purpose, unknown-rights denial |
| CQ-M09: What is the next useful action? | Allocate validation effort | action × blocked path | explicit blocker and affected denominator | unquantified expected benefit explicit | path counts as methane-risk or carbon benefit | MQ11, MQ17 | Transparent counts and no credit inference |
| CQ-M10: What changes the review decision? | Test robustness of candidate cards | within-MAG chain and alternative view | multi-tool evidence, QC, disputed identity/annotation | unavailable alternative remains untested | apparent co-occurrence as community activity | MQ14, MQ15 | Same evidence with disputed edges excluded |
| CQ-M11: What can be compared across sources? | Compare readiness or abstain | source × habitat × region × family | habitat authority, source contract, QC, assay mapping | source-specific evidence allowed | common concept as common quantitative assay | MQ05, MQ13, MQ16 | Source/QC/identity/ascertainment controls |
| CQ-M12: Does a graph improve review? | Choose useful product interface | reproducible evidence packet | matched graph/SQL facts and filters, provenance, timings | SQL may be equally effective | graph intrinsically faster or information-creating | MQ03, MQ09, MQ18; MQ12 stability | Answer parity, measured query plans and effort |

## Preregistered showcase and fair-baseline selection

Registered 2026-09-26 before viewing new molecular query results. The three
showcases are **MQ03 (competing interpretations), MQ09 (partial or complete
molecular-to-field path), and MQ18 (diligence packet)**. They represent a
scientific discrimination decision, an identity/measurement decision, and a
review workflow. Negative or empty findings remain in the report.

Use all admissible rows for aggregate denominators. Display exemplars use the
lexicographically first eligible source-scoped candidate in each wetland lane;
the MUCC homolog example uses the first disputed candidate by stable source
key. If none exists, return a coverage-backed absence/gap. Do not rank examples
by favorable effect, apparent ecological separation, or graph/SQL speed.
Record any required change and its reason before inspecting replacement results.

The SQL baseline uses the same admitted, normalized evidence tables, not an
artificially handicapped raw-file scan. Both interfaces receive identical
filters, source policies, output grains and result limits. Report setup effort
separately from warm query latency and disclose shared preprocessing. Measure
structural query effort (joins/traversals, lines), provenance/counterevidence
parity and execution timings; do not label these measurements a human user study.
No graph advantage is presumed.

## Baseline reproduced before the molecular extension

- 68 ontology tests passed (5 rdflib deprecation warnings).
- OWL 2 RL, HermiT and schema/shape verification passed.
- Atlas-foundation validator: 25,019 checks, zero errors, one warning retaining
  MUCC's 10 non-pass ecological/source-readiness gates.
- This was the 0.1.0 catalog baseline at preregistration time. It does not
  describe the subsequently completed 0.2.0 molecular layer; consult the
  handoff for its 119-test result, 18-query execution, and retained limitations.
