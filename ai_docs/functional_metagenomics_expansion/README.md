# MethaNet Functional-Metagenomics Expansion Package

Date: 2026-06-11  
Scope: Blue Catalyst/MethaNet 662-genome POC expansion from ESM2 latent geometry into mechanistic, source-aware methane functional metagenomics.

## Purpose

This folder turns the v2.0 report roadmap into an operational plan that can be run first on one MAG, then scaled across the full MAG/proteome set on Apolo-3.

The objective is to convert the current claim:

> 662 genomes embedded with zero attrition; the ESM2 latent space separates methane-producing ecosystems while preserving cross-ecosystem bridges.

into a stronger, fundable platform claim:

> MethaNet can rank methane-relevant genomes by combining latent protein-language geometry with independently measured genome quality, taxonomy, methane-cycle mechanism, substrate/electron-transfer function, sulfur competition, and source-aware transfer validation.

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

- [tool_database_registry.md](tool_database_registry.md)
  Web-verified tool/database matrix, current release notes, production/default choices, database provisioning policy, and exact manifest fields to capture.

- [output_contracts_and_gates.md](output_contracts_and_gates.md)
  Required tables, cards, matrices, QC gates, mechanism classes, platform-ready outputs, and completion criteria.

- [production_batch_deployment_plan.md](production_batch_deployment_plan.md)
  Apollo-3 resource model, Slurm array strategy, launch checklist, and post-run aggregation steps. Treat operational counts as dated snapshots, not live scheduler state.

- [snakemake_backbone/](snakemake_backbone/)
  A concrete Snakemake backbone for Apollo 3. It is intentionally a scaffold: it defines DAG shape, resources, outputs, and integration hooks, while leaving project-specific parser scripts to be implemented in the production workflow.

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
