# MethaNet documentation map

Use this page to find the current source of truth by topic. Dated documents are
versioned evidence snapshots; a later implementation note supplements rather
than silently rewrites a prior release freeze.

## Start here

| Need | Canonical guide |
| --- | --- |
| Understand the product, current atlas counts, and claim boundaries | [Project README](../README.md) and [positioning and claims](methanet_positioning_and_claims.md) |
| Find datasets, lanes, warehouses, and generated artifacts | [Current artifact inventory](current_artifact_inventory.md) |
| Understand frozen atlas identity and data authority | [Atlas data foundation](atlas_data_foundation.md) and [release index](releases/README.md) |
| Understand graph scope, identity, provenance, and planned expansion | [Knowledge-graph foundation](knowledge_graph_foundation.md) |
| Compare SQL, RDF, and Neo4j roles and local query evidence | [MVO store comparison](../ontology/docs/STORE_COMPARISON_20260926.md) |
| Read the current MVO graph implementation and local recovery commands | [MVO package guide](../ontology/README.md) and [0.2.0 handoff](../ontology/docs/MOLECULAR_HANDOFF_20260926.md) |
| Review ontology design and formal verification | [MVO architecture](../ontology/docs/ARCHITECTURE.md), [verification history](../ontology/docs/VERIFICATION.md), and [governance gates](../ontology/docs/GOVERNANCE_AND_EXTENSION.md) |
| Understand the end-to-end status as of 2026-09-27 | [Project state recap](project_state_20260927.md) |
| Run local workflows and tests from a clean clone | [Repository guide](repository_guide.md) and [reproducibility](reproducibility.md) |

## Data and implementation contracts

- The August 10, 2026 atlas freeze remains the promoted molecular release.
  MVO 0.2.0 is an additive local graph snapshot, not a successor atlas freeze.
- Use `(lane_id, proteome_id)` as the release-scoped molecular identity.
  Keep sample, specimen, site, measurement, MAG, and claim grains distinct.
- Preserve failed, partial, pending, excluded, and missing evidence explicitly.
- MAG-level molecular potential is not sample-level activity or methane flux.
  No final A–E tiers or credit approvals are supported by the current atlas.
- SQL/Parquet warehouse facts remain source authority. RDF is the canonical
  semantic snapshot; Neo4j is a rebuildable graph-query projection.
- Large source data, warehouse outputs, graph builds, internal report bundles,
  and local credentials are not repository source files and must not be added
  just to make the worktree appear complete.

## Supporting collections

Unpublished manuscript versions and internal research/review work remain in
the local workspace while their author, source-rights, and public-deposition
gates are open. They are deliberately excluded from this public source tree.
The older tracked concept draft under `white-paper/` is prominently marked as
historical and unvalidated; it is not a current results document.

- [`../ai_docs/functional_metagenomics_expansion/`](../ai_docs/functional_metagenomics_expansion/)
  contains internal functional-atlas and MRV planning contracts.
