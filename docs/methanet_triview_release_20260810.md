# MethaNet Tri-View Controlled-Diligence Release: 2026-08-10

## Decision recorded on 2026-08-10

The four registered source-lane molecular payloads are release-freeze ready after bounded repair
and integrity-aware warehouse consolidation. The August 10 public source and
local build remain `noindex` and are **not authorized for deployment** until
the source/taxonomy statistical, stability, browser, accessibility, and final
publication gates pass.

The POC source lane contains two interpretation groups, rumen and
wetland/MUCC; this is why public material describes five evidence lanes while
the release registry has four source lanes.

## Subsequent scoped correction, 2026-09-23/24

The already-public [landing page](https://emergentbiome.earth/) and
[technical report](https://emergentbiome.earth/report/) were corrected and
redeployed against this August freeze under `noindex`. The user-directed
correction addressed stale public presentation and aligned the landing and
report; it did not change the frozen counts, authorize indexing, or close the
scientific publication gates below. The reconciled local report is
`results/reports/emergentbiome_molecular_atlas_20260923_reconciled/`; the latest
deployment receipt is under
`results/reports/emergentbiome_public_browser_verification_20260923_tsne/`.
For future release history and the current pointer, see
[`releases/README.md`](releases/README.md) and
`configs/atlas_current_release.json`.

## Reconciled ledger

| State | Units |
| --- | ---: |
| Registered | 7,965 |
| Release-required | 7,710 |
| Explicit source gaps | 255 |
| ESM-2 | 7,710 |
| gLM2 | 7,717 |
| Functional payload | 7,710 |
| Tri-view-ready | 7,710 |
| Schema-normalized tri-view | 7,710 |
| Pipeline-normalized, comparability pending | 5,209 |
| MUCC source-scaffold tri-view | 2,501 |
| Mechanism-comparable | 0 |
| Sample-linked | 0 |
| Field-validated | 0 |
| Calibrated | 0 |

The zeros for sample linkage and field validation are release-authorization
states for molecular-to-sample/process scoring. They do not erase the staged
sample contexts, processed expression, or field observations; accepted exact
sample–flux joins remain zero in the August metadata-readiness summary.

## Tail-unit disposition

- MSM `msm_china_2025__group3_MAGs__mi_bins_mi.1767_sub` was rerun in an
  isolated METABOLIC working directory and selected complete; its original
  partial attempt remains in the run-status ledger.
- Futian `futian_mangrove_2026_qi__MF1_202108_bin_656` was rerun and selected
  complete after the historical shared-workbook move failure.
- Futian `futian_mangrove_2026_qi__MG1_202105_bin_189` was rerun after a
  cross-MAG workbook-identity failure. The contaminated historical attempt is
  retained as failed/quarantined and cannot be selected.

The guarded runner now gives every METABOLIC invocation an attempt-local
working directory and checks workbook structure plus MAG identity before
writing `COMPLETE`.

## Current artifacts

- POC warehouse:
  `results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_semantic_rebuild_20260810/`
- MSM warehouse:
  `results/functional_metagenomics/msm_china_2025_20260615/cohort_warehouse/`
- Futian warehouse:
  `results/functional_metagenomics/futian_mangrove_2026_phase1/cohort_warehouse/`
- Release freeze:
  `results/reports/methanet_3view_payload_freeze_20260810_end_to_end/`
- Metadata readiness:
  `results/reports/methanet_atlas_metadata_readiness_20260810/`
- Technical report:
  `results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/`
- Reconciled public report from the same freeze:
  `results/reports/emergentbiome_molecular_atlas_20260923_reconciled/`
- Locally built public tree:
  `web/emergentbiome-methanet/_site/`

## Scientific change

The release no longer treats raw functional hit rows as pathway strength.
KOfam accepted events, best-ranked MCycDB/SCycDB hits, and METABOLIC presence
events are explicit. POC, MSM, and Futian share a normalized table/event
schema, but no unit is called mechanism-comparable until locked
code/configuration/database fingerprints and source-aware statistical gates
pass. MUCC remains a separate mapped source scaffold; processed expression is
detection/occupancy support only.

## Metadata and product change

The metadata warehouse identifies 280 samples and ranks metadata-rich
validation contexts without forcing one-to-one MAG/sample links. MSM offers
source-group environmental context, Futian offers the strongest repeated
site-time chemistry design, and MUCC offers the strongest near-term
expression/process recovery opportunity. None has an authoritative exact
sample/environment/process join today.

The public narrative now leads with the decision and delivered object:
evidence cards for molecular review, provenance, readiness, and the next
measurement. It no longer presents current atlas rows as calibrated methane
risk, activity, measured flux, final tiers, or crediting evidence.

## Publication blockers retained from the August decision

Before routine scientific publication or indexing, record passing source-aware permutation nulls,
taxonomy-matched controls, leave-source checks where possible, bootstrap
neighbor and candidate stability, view/QC/missingness ablations,
dimensionality and graph sensitivity, repeated-test correction, and explicit
failure regions. Then pass the release-ledger parity, Firefox desktop/tablet/
mobile, keyboard, reduced-motion, static-fallback, Open Graph, outbound-link,
and public-tree privacy gates.
