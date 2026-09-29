# DIGEST: EmergentBiome Molecular Atlas controlled-diligence release

Snapshot date: **2026-08-10**

This digest is the public rendering contract for the August 10 source and the
live, claim-bounded September correction of the landing/report pair. The dated `release_ledger.json` in
`results/reports/methanet_3view_payload_freeze_20260810_end_to_end/` is the
numerical authority. A payload-freeze state of `ready` does not mean the public
deployment is authorized; routine scientific publication and indexing remain
separately gated. The corrected public surfaces retain `noindex`.

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

## Landing narrative and nearest-core contract

The public landing uses **EmergentBiome Molecular Atlas** and
**EmergentBiome evidence graph**. MethaNet/MBAG remain internal or historical
artifact identifiers and do not appear as the landing brand. The current
artifact guides molecular screening, evidence review, and field-validation
design. It does not claim validated methane-flux biomarkers, a measured flux
estimate, or a calibrated site-risk product.

The 29 September geometry release maps **7,710 MAG/proteome records**. It draws **2,226 links**: 2,200 sampled cross-habitat neighbor links and 26 selected one-way nearest-core links. These are exploratory representation similarities.

Outside the 625-record core (518 rumen, 107 wetland), **1,403/2,501 wetland** and **2,823/4,584 mangrove** records have a rumen genome as their closest core match. The nearest-core median cosine is 0.9973; the random-pair median is 0.9942. The corrected dimension-standardized graph contains 486 mutual pairs involving rumen and another habitat. One-way core matches and reciprocal full-atlas neighbors answer different questions; neither validates ecological transfer.

Counts and candidate selection are recomputed from the corrected vectors. The public export includes its geometry audit and contract hash, and publication rejects stale quantities or a mismatched report. UMAP, diffusion, t-SNE and PCA show the same records; changing projection does not change high-dimensional link membership. The 255 registered gap rows remain explicit and unplotted.

The proposal-facing OWC_1885 card retains its source QC and expression-detection facts. Its nearest-core reference and similarity are refreshed by `tools/sync_geometry_release.py` from the corrected `embedding_context_table.tsv`; old cosine values are superseded. Expression detection remains distinct from activity or flux, and exact sample/environment/process pairing remains unresolved.

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
| Landing candidate card | Frozen MUCC v1 OWC_1885 record with explicit evidence states | MAG/proteome review only; expression detection is not process rate |
| Projection controls | UMAP opens by default on the landing page and in the report; diffusion map, t-SNE and PCA remain selectable for the same 7,710 embedding-bearing records. PHATE is not built (`--skip-phate`) | Two-dimensional layouts do not rank candidates or change high-dimensional link membership; 255 registered gap rows have no projection coordinates |
| Evidence cards | Derived evidence records with direct, missing, contradictory, and next-action fields | Review priority, not biological truth |
| Sample/context cards | Real metadata coverage and explicit ambiguity tiers | Context value, not exact sample risk |
| Climate, proxy, and product scenes | Clearly badged sourced anchor, roadmap, or illustrative product shape | No illustrative score is a released prediction; the climate scene's net-balance range carries no values |
| Proposed field study (scene 10, closing) | Schematic of the proposed design, badged "Proposed study": 36 plots × 2 microsites × 2 seasons | 144 planned sample-events are repeated measurements, not collected or independent observations; conditional on funding, site access and permits; kept outside the atlas counts |

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

Routine scientific publication of the August 10 source is blocked until source-aware and
taxonomy-aware nulls, bootstrap neighbor/rank stability, view and QC
ablations, dimensionality/graph sensitivity, multiple-testing control, and the
final local browser/accessibility/public-tree gates are recorded as passing.
The August preflight audit still records `deployment_allowed=false` for that
ungated release. On September 23, 2026, the user requested a scoped correction
of the already-public domain: replace its stale landing/report pair with this
reconciled, claim-bounded `noindex` bundle. That direction authorizes the
correction deployment only; it does not clear the scientific gates, permit
indexing, or expand the claims above. The September 23 local bundle passed its recorded release-parity, report, browser and public-tree checks. The QA record is in
`results/reports/emergentbiome_public_browser_verification_20260923_tsne/`.

The user explicitly authorized pooling reconciliation and correction deployment on September 28. The September 29 release preserves the `noindex` and claim boundaries above. Its current validation receipt is `docs/releases/atlas_embedding_reconciliation_20260929.json`; the case-review payloads retain their original scope.

## Provenance pointers

- Current geometry registry: `configs/methanet_atlas_lanes_20260929.tsv`
- Current embedding contract: `configs/atlas_embedding_contract_20260929.json`
- Current public report: `results/reports/emergentbiome_molecular_atlas_20260929_layer33/` (publisher default)

- Lane registry: `configs/methanet_atlas_lanes.tsv`
- Freeze: `results/reports/methanet_3view_payload_freeze_20260810_end_to_end/`
- Metadata readiness: `results/reports/methanet_atlas_metadata_readiness_20260810/`
- August source report: `results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/`
- Superseded public report, published 28 September 2026 (same 10 August freeze): `results/reports/emergentbiome_molecular_atlas_20260928_consolidated/` (kept locally; the stable `/report/` alias now serves the September 29 correction)
- Superseded 23 September reconciled report: `results/reports/emergentbiome_molecular_atlas_20260923_reconciled/` (kept locally; the pre-correction pointer bytes are archived in `docs/releases/atlas_20260810_pointer_before_geometry_reconciliation.json`)
- Correction deployment receipt: `results/reports/emergentbiome_public_browser_verification_20260923_tsne/deployment_receipt.json`
- Claim contract: `docs/methanet_positioning_and_claims.md`
- Release inventory: `docs/current_artifact_inventory.md`

The July 24 and earlier report directories remain historical snapshots; they
are not rewritten to imply the August 10 evidence contract.
