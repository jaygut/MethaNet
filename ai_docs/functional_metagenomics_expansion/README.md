# MethaNet Functional-Metagenomics Expansion Package

Date: 2026-06-13
Documentation refresh: 2026-06-25
Scope: Blue Catalyst/MethaNet functional expansion from the 662-genome rumen +
wetland/MUCC POC into a broader multi-lane molecular atlas spanning rumen,
wetland/MUCC, mangrove/MSM, and mangrove/Futian MAG/proteome evidence.

## Purpose

This folder turns the v2.0 report roadmap into an operational plan that can be run first on one MAG, then scaled across the full MAG/proteome set on Apolo-3.

The objective is to convert the current claim:

> 662 genomes embedded with zero attrition; the ESM2 latent space separates methane-producing ecosystems while preserving cross-ecosystem bridges.

into a stronger, fundable platform claim:

> MethaNet can rank methane-relevant genomes by combining latent protein-language geometry with independently measured genome quality, taxonomy, methane-cycle mechanism, substrate/electron-transfer function, sulfur competition, and source-aware transfer validation.

## Current Implemented Arc And Artifact State

By the 2026-06-20 documentation refresh, this folder was no longer only a planning package. As of the 2026-06-25 refresh, the same architecture is being extended across the MSM and Futian mangrove lanes:

1. `embedded_662_proteome_id_crosswalk.tsv` remains the 662-proteome ESM2 backbone.
2. Unit-scope manifests split that backbone into 625 MAG/bin-comparable units and 37 assembly-context rumen no-bin units.
3. Apollo-3 per-MAG runs write immutable evidence bundles under `results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag/`.
4. `scripts/curate_functional_mag_run.py` converts raw tool output into per-run curated Parquet and `run_record.json` provenance.
5. `scripts/consolidate_functional_mag_cohort.py` builds the Parquet-first cohort warehouse and optional DuckDB catalog.
6. `scripts/attestation/build_molecular_attestation_mvp.py` builds a local queryable molecular attestation graph over the 662-row denominator.

Latest generated launch-ready warehouse observed locally:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/
```

This warehouse reports:

| Item | Current generated state |
| --- | ---: |
| Run attempts inspected | 683 |
| Selected completed MAG/bin runs | 625 |
| Complete attempts preserved in status | 644 |
| Failed attempts preserved in status | 24 |
| Partial attempts preserved in status | 15 |
| Cohort tables written | 24 |
| Validation gates | 711 pass, 0 warn, 0 fail |
| DuckDB catalog | `functional_atlas.duckdb` present |
| Launch decision | `LAUNCH-READY` for inspected data-format gates |

The current molecular attestation graph snapshot is:

```text
results/attestation/mmag_mvp_20260617/
```

It preserves 662 MAG/proteome nodes, separates 625 MAG-bin units from 37 assembly-context units, links every evidence atom to an artifact, and explicitly blocks sample-level methane risk, final A-E MRV tiers, measured flux, and carbon-credit claims.

## Current Multi-Lane Status

The functional expansion is no longer a single POC denominator. Current local
work should be interpreted as related but distinct evidence lanes:

| Lane | Status at 2026-06-25 documentation refresh | Operational meaning |
| --- | --- | --- |
| Rumen POC | 555 ESM2 proteomes; 518 MAG/bin-comparable units in the completed POC functional/gLM2 layer; 37 assembly-context rumen units quarantined from MAG-bin tables | source reference lane for methane-system molecular neighborhoods and cross-domain bridge hypotheses |
| Wetland/MUCC POC | 107 ESM2 proteomes; 107 completed functional MAG/bin outputs; 107 gLM2 units | target wetland POC lane for MBAG bridge-candidate evidence |
| Mangrove/MSM expansion | 1,428 ESM2 proteome embeddings complete; 1,428 gLM2 units complete; 1,427/1,428 functional MAGs complete at the 2026-06-25 snapshot | broader blue-carbon target expansion; one partial unit remains before final consolidation |
| Mangrove/Futian 2026 expansion | 3,404 dereplicated rMAGs (3,156 ready payload rows + 248 gap rows); 3,156/3,156 ESM2 complete; 3,156/3,156 gLM2 complete; live functional status is 302/312 archaea complete, 2 running/partial, 8 pending/not-started, 0 manifest-scoped failed rows at the 2026-06-25 15:39 UTC refresh; 2,844 bacteria queued | newest blue-carbon mangrove target lane; archaea-only functional coverage so far; no consolidated warehouse yet |

The latest expanded HTML atlas is:

```text
results/reports/mbag_nextgen_molecular_niche_atlas_20260625_release_freeze_145509_bridge_v4/report.html
```

It is backed by:

```text
results/reports/methanet_3view_payload_freeze_20260625_145509/
```

This freeze contains 2,352 release-required tri-view MAG/proteome units
(625 POC core + 1,427 MSM + 300 Futian archaea), explicitly preserves one
release-excluded MSM partial unit, and retains 248 Futian source-lane gap rows
outside the ready-payload denominator. Live production has moved beyond that
freeze to 2,354 tri-view units as of the 2026-06-25 15:39 UTC registry refresh,
but external reports should remain tied to their dated freeze manifests. The
`bridge_v4` render keeps that denominator fixed while improving the molecular
niche-space bridge view with all embedding-bearing units, case-study halos,
nearest-POC evidence links, and diffusion/UMAP/t-SNE/PCA sensitivity views.
Future rebuilds should continue to use `configs/methanet_atlas_lanes.tsv` and a
dated freeze manifest from
`scripts/reports/build_methanet_3view_payload_freeze.py`, not ad hoc folder
discovery.

## Contents

- [pipeline_plan.md](pipeline_plan.md)
  End-to-end execution plan: single-MAG pilot, bridge-candidate pilot, full-cohort parallelization, Apollo 3 deployment, scientific gates, investor-demo outputs.

- [data_aggregation_strategy.md](data_aggregation_strategy.md)
  Canonical 662-row backbone, layer-by-layer integration plan, feature-store design, statistical integration plan, bridge-card requirements, and claim boundaries.

- [run_output_storage_architecture.md](run_output_storage_architecture.md)
  Storage model for per-MAG outputs, Parquet/DuckDB cohort tables, log retention, generated artifact policy, and production gates.

- [cohort_data_architecture_hardening.md](cohort_data_architecture_hardening.md)
  Cohort warehouse table model, validation gates, METABOLIC normalization rules, and active-run safety notes.

- [pipeline_reproducibility_contract.md](pipeline_reproducibility_contract.md)
  Maintainability contract for source-controlled code/docs, generated outputs, metadata resolution, documentation freshness, and pre-commit checks.

- [dataset_expansion_strategy_20260614/](dataset_expansion_strategy_20260614/)
  Strategic external-dataset expansion package for mangrove and methane-rich
  MAG-resolved sources, including the MSM China 2025 staging status and
  accession-gated ingestion plan.

- [tool_database_registry.md](tool_database_registry.md)
  Web-verified tool/database matrix, current release notes, production/default choices, database provisioning policy, and exact manifest fields to capture.

- [output_contracts_and_gates.md](output_contracts_and_gates.md)
  Required tables, cards, matrices, QC gates, mechanism classes, platform-ready outputs, and completion criteria.

- [final_mrv_risk_scoring_roadmap.md](final_mrv_risk_scoring_roadmap.md)
  Strategic maturity ladder from current MBAG/MAG-level molecular screening to sample/project-level MRV risk scoring, including required metadata, abundance, environmental, flux-validation, uncertainty, and claim-boundary gates.

- [three_view_freeze_and_nextgen_report_plan.md](three_view_freeze_and_nextgen_report_plan.md)
  Current registry/freeze workflow for building the next expanded MBAG atlas over POC, MSM, and Futian three-view payloads without losing pending, partial, gap, or provenance rows.

- [`../../docs/current_artifact_inventory.md`](../../docs/current_artifact_inventory.md)
  Current source-controlled inventory of datasets, databases, generated warehouses, metadata outputs, attestation graph artifacts, and the docs that must stay in sync.

- [source_provenance_environmental_metadata_reconciliation.md](source_provenance_environmental_metadata_reconciliation.md)
  Verified paper/data-object provenance for the rumen, wetland/MUCC, and mangrove lanes; current environmental metadata readiness; resolution tiers; and free API routes for accession, site, flux, soil, weather, and modeled-context enrichment.

- [mag_level_atlas_relaunch_prompt.md](mag_level_atlas_relaunch_prompt.md) and [mag_level_atlas_relaunch_recovery_20260614.md](mag_level_atlas_relaunch_recovery_20260614.md)
  Recovery contract and dated operational snapshot for separating the 662-proteome ESM2 backbone from the clean 625-row MAG/bin functional-atlas denominator, with 37 no-bin rumen assembly-context records quarantined from MAG-level MBAG. The current operational relaunch manifest is the 518-row remaining MAG/bin subset because 107 comparable MUCC outputs are already preserved as completed evidence.

- [production_batch_deployment_plan.md](production_batch_deployment_plan.md)
  Apollo-3 resource model, Slurm array strategy, launch checklist, and post-run aggregation steps. Treat operational counts as dated snapshots, not live scheduler state.

- [snakemake_backbone/](snakemake_backbone/)
  A concrete Snakemake production-contract backbone for Apollo 3. It now
  orchestrates the current proven Slurm runner, curator, optional eggNOG v2
  sidecar, and cohort consolidator so future Snakemake runs reproduce the same
  curated per-MAG Parquet/DuckDB artifact contract.

## Source-Controlled Utilities

- `scripts/metadata/recover_environmental_metadata.py` recovers ENA, MUCC/Zenodo, OWC, and NCBI context for the 662-proteome cohort while preserving metadata resolution tiers.
- `scripts/figures/functional_metagenomics/plot_functional_calibration_panel.py` builds a preliminary QC-aware functional figure panel from completed curated per-MAG outputs.

Both utilities write generated artifacts under `results/`, which is ignored by
git. Promote only small reviewed snapshots into this documentation folder.

## Inputs Expected

Minimum inputs for a single-MAG pilot:

- MAG FASTA: `.fa`, `.fna`, or `.fasta`
- MAG ID that can be mapped to the 662-genome POC metadata
- Source/ecosystem/domain metadata
- Existing ESM2 bridge metrics when available: `alpha_transfer_score`, `bridge_entropy`, `opp_neighbor_fraction`, `silhouette`, PCA/UMAP coordinates

Minimum inputs for the cohort run:

- A manifest with one row per MAG/proteome-linked genome
- Canonical POC metadata and bridge scores
- MAG FASTA paths for all genomes that will receive functional characterization
- A pinned tool/database manifest

## Design Principle

The pipeline must not let missingness masquerade as biology. Every bridge candidate should receive:

1. QC/taxonomy/derep status
2. methane-cycle mechanism status
3. broad-function coverage status
4. source-aware validation status
5. platform feature status

Only candidates that pass the relevant gates should be shown as high-confidence methane MRV leads.

## Freshness Note

The files in this folder are contracts, runbooks, and dated snapshots. During
active Apollo-3 runs, live queue state and per-MAG completion counts can change
faster than the documentation. Use the docs for reproducible interpretation and
refresh live operational state separately before launch, requeue, pruning, or
external reporting decisions.

For full system orientation, see:

```text
ai_docs/system_operational_map_20260619/TECHNICAL_BRIEF.md
```

That brief is an ignored/generated documentation artifact under `ai_docs/*`; keep the source-controlled docs in this folder as the stable contract layer.
