# DIGEST: EmergentBiome / MethaNet controlled-diligence release

Snapshot date: **2026-08-10**

This digest is the public rendering contract for the August 10 source and
locally built experience. The dated `release_ledger.json` in
`results/reports/methanet_3view_payload_freeze_20260810_end_to_end/` is the
numerical authority. A payload-freeze state of `ready` does not mean the public
deployment is authorized; publication remains separately gated.

<!-- METHANET_RELEASE_LEDGER_BEGIN -->
```json
{
  "registered_units": 7965,
  "esm2_units": 7710,
  "glm2_units": 7717,
  "functional_payload_units": 7710,
  "release_required_units": 7710,
  "explicit_non_runnable_gaps": 255,
  "tri_view_ready_units": 7710,
  "schema_normalized_units": 7710,
  "schema_normalized_tri_view_units": 7710,
  "pipeline_normalized_tri_view_units": 5209,
  "mechanism_comparable_units": 0,
  "annotation_complete_tri_view_units": 0,
  "source_scaffold_tri_view_units": 2501,
  "blocking_units": 0,
  "schema_version": "1.0.0",
  "snapshot_date": "2026-08-10",
  "freeze_manifest_sha256": "7dd870ac3cdf0142b8050dbeec4f310ac18e6c21d1f34c4728e43b18df986cfc",
  "release_state": "ready",
  "indexing_decision": "noindex_controlled_diligence",
  "allowed_public_wording": "Molecular screening evidence and review priorities; metadata-rich contexts are not scored samples.",
  "forbidden_public_wording": "Sample risk, measured flux, activity magnitude, final tiers, source-independent transfer, or MRV/crediting approval."
}
```
<!-- METHANET_RELEASE_LEDGER_END -->

## Verified release snapshot

| Lane | Registered | Release-required | ESM-2 | gLM2 | Functional | Tri-view | Evidence contract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| POC core | 625 | 625 | 625 | 625 | 625 | 625 | pipeline-normalized; cross-lane comparability pending |
| MSM China 2025 | 1,428 | 1,428 | 1,428 | 1,428 | 1,428 | 1,428 | pipeline-normalized; cross-lane comparability pending |
| Futian 2026 | 3,404 | 3,156 | 3,156 | 3,156 | 3,156 | 3,156 | pipeline-normalized; cross-lane comparability pending |
| MUCC v1 Old Woman Creek | 2,508 | 2,501 | 2,501 | 2,508 | 2,501 | 2,501 | source scaffold; non-equivalent |
| **Total** | **7,965** | **7,710** | **7,710** | **7,717** | **7,710** | **7,710** | **0 mechanism-comparable** |

The 255 non-runnable rows are explicit source gaps: 248 Futian rows and seven
MUCC rows. They remain in the registered-unit ledger and are not plotted as
embedding-bearing points.

## Functional and comparison contract

- KOfam numerators use accepted calls, not every hit row.
- MCycDB and SCycDB event counts use the best-ranked hit per gene.
- METABOLIC workbook outputs are normalized into long presence/event tables.
- Failed, partial, corrupt, and superseded attempts remain in run-status facts.
- POC, MSM, and Futian are pipeline-normalized, but code/configuration/database
  fingerprints and source-aware statistical gates are not yet closed.
- MUCC expression supports detection/occupancy review only. It is not activity
  magnitude or process rate, and the source scaffold is not pooled numerically
  with the guarded functional pipeline.

## Metadata readiness

The normalized metadata package contains 280 samples, 293,245 sample/MAG or
context links, 259,084 expression-or-abundance rows, 34,835 process-observation
rows, and 133 sample/flux-window context links. It contains **zero**
authoritative exact sample/environment/process joins.

High-value contexts are:

- MSM: 82 sediment samples and 71 exact BioSample environmental rows; MAGs map
  to source-group ambiguity sets rather than one exact sample.
- Futian: 65 sediment samples across 14 site-time keys; 47 contexts have the
  strongest paired chemistry coverage, but MAGs remain depth-ambiguous.
- MUCC v1: 133 expression columns and 89 best-recovered sample contexts;
  processed expression, chamber, porewater, and tower records are staged but
  lack an authoritative ecological join.

These are metadata-rich validation opportunities, not scored samples.

## Visual evidence contract

| Surface | Evidence mode | Interpretation limit |
| --- | --- | --- |
| Landing manifold | Real report coordinates and source-audited counts | Navigation and hypothesis generation, not transfer or risk proof |
| Projection controls | Diffusion plus available UMAP/PHATE and PCA sensitivity views | Two-dimensional layouts do not rank candidates |
| Evidence cards | Derived evidence records with direct, missing, contradictory, and next-action fields | Review priority, not biological truth |
| Sample/context cards | Real metadata coverage and explicit ambiguity tiers | Context value, not exact sample risk |
| Climate, proxy, and product scenes | Clearly badged sourced anchor, roadmap, or illustrative product shape | No illustrative score is a released prediction |

Interactive figures must expose keyboard-operable controls, labels, legends,
reset actions, mobile-safe dimensions, and a legible static or no-JavaScript
fallback. The generated report and landing page retain `noindex`.

## Claim boundary and publication decision

Allowed now: MAG/proteome molecular screening, evidence-card review, candidate
triage, metadata-readiness assessment, monitoring prioritization, and
next-measurement design.

Blocked now: measured methane flux, expression-derived activity magnitude,
sample/project methane risk, final A-E tiers, source-independent transfer,
carbon-credit approval, registry acceptance, customers, contracts, or revenue.

Public deployment of the August 10 source is blocked until source-aware and
taxonomy-aware nulls, bootstrap neighbor/rank stability, view and QC
ablations, dimensionality/graph sensitivity, multiple-testing control, and the
final local browser/accessibility/public-tree gates are recorded as passing.
The existing public URLs remain historical until then.

## Provenance pointers

- Lane registry: `configs/methanet_atlas_lanes.tsv`
- Freeze: `results/reports/methanet_3view_payload_freeze_20260810_end_to_end/`
- Metadata readiness: `results/reports/methanet_atlas_metadata_readiness_20260810/`
- Report: `results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/`
- Claim contract: `docs/methanet_positioning_and_claims.md`
- Release inventory: `docs/current_artifact_inventory.md`

The July 24 and earlier report directories remain historical snapshots; they
are not rewritten to imply the August 10 evidence contract.
