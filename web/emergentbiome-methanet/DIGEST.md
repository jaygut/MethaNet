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

The 7,710 mapped units are MAG/proteome records with embeddings, not a claim
of 7,710 globally dereplicated genomes. The landing's 2,226 links are
displayed visual edges: 2,200 sampled cross-domain full-atlas kNN links and
26 selected one-way nearest-core candidate links. They are not 2,226
biologically verified transfer bridges.

The frozen embedding-context and candidate-card tables independently support
the nearest-core statement: 2,434 of 2,608 wetland and 4,475 of 4,584
mangrove records have a rumen record as their **single raw-cosine nearest
neighbor within the 625-record POC core** (518 rumen, 107 wetland). This
includes 26 of the 27 selected wetland/mangrove candidate cards. The
rumen-heavy reference composition is part of the interpretation.

The report's zero is a different analysis: **reciprocal cross-domain top-35
neighbors across the full atlas after per-dimension standardization**.
One-way nearest-core assignments and reciprocal full-atlas neighbors are not
interchangeable and neither establishes biological transfer. Source and
method detail live in the frozen report's
`tables/embedding_context_table.tsv`,
`tables/candidate_cards.tsv`, and geometry audit.

The proposal-facing candidate example is MUCC v1 `mucc_v1__OWC_1885`.
Its frozen card records strong source QC (94.89% completeness, 1.14%
contamination), processed expression-marker **detection**, and a one-way
raw-cosine nearest POC rumen record (0.9842857). Expression is not an
activity or flux magnitude; exact sample/date/depth, environmental,
abundance, and methane-process joins remain unresolved. Its next action is
to make those joins before ecological or risk interpretation.

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
| Projection controls | UMAP opens as the navigable landing view; diffusion, t-SNE, and PCA remain selectable for the same 7,710 embedding-bearing records. PHATE is unavailable in the frozen report | Two-dimensional layouts do not rank candidates or change high-dimensional link membership; 255 registered gap rows have no projection coordinates |
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

Routine scientific publication of the August 10 source is blocked until source-aware and
taxonomy-aware nulls, bootstrap neighbor/rank stability, view and QC
ablations, dimensionality/graph sensitivity, multiple-testing control, and the
final local browser/accessibility/public-tree gates are recorded as passing.
The August preflight audit still records `deployment_allowed=false` for that
ungated release. On September 23, 2026, the user requested a scoped correction
of the already-public domain: replace its stale landing/report pair with this
reconciled, claim-bounded `noindex` bundle. That direction authorizes the
correction deployment only; it does not clear the scientific gates, permit
indexing, or expand the claims above. The current local bundle passes 201/201 release
parity checks, 19/19 report gates, browser checks, and no-JavaScript,
reduced-motion, link, Open Graph, and public-tree checks. The QA record is in
`results/reports/emergentbiome_public_browser_verification_20260923_tsne/`.

## Provenance pointers

- Lane registry: `configs/methanet_atlas_lanes.tsv`
- Freeze: `results/reports/methanet_3view_payload_freeze_20260810_end_to_end/`
- Metadata readiness: `results/reports/methanet_atlas_metadata_readiness_20260810/`
- August source report: `results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/`
- Reconciled public report: `results/reports/emergentbiome_molecular_atlas_20260923_reconciled/` and [stable `/report/` alias](https://emergentbiome.earth/report/)
- Correction deployment receipt: `results/reports/emergentbiome_public_browser_verification_20260923_tsne/deployment_receipt.json`
- Claim contract: `docs/methanet_positioning_and_claims.md`
- Release inventory: `docs/current_artifact_inventory.md`

The July 24 and earlier report directories remain historical snapshots; they
are not rewritten to imply the August 10 evidence contract.
