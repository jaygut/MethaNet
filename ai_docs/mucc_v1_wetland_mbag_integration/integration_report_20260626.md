# MUCC v1 OWC Wetland Integration Report

Date: 2026-06-26
Status: source-audited, payload-parsed, ESM2-input-validated, normalized source-lane registered, and corrected production ESM2 GPU shards queued
Lane ID: `mucc_v1_owc_wetland`
Cohort run scaffold: `mucc_v1_owc_wetland_20260626`

## Answer To The Goal-Mode Question

Running the revised MUCC v1 prompt under `/goal` mode cannot mathematically assure full warehouse integration by itself. The assurance comes from executable gates: source provenance, checksum validation, denominator reconciliation, lane registry integration, feature generation, embedding completion, environmental metadata joins, and explicit claim locks.

Current result: the asset has moved beyond prompt design. All six data-bearing MUCC v1 payloads are local and md5-validated, MethaNet now has a parsed native wetland lane scaffold, the protein-supported subset has passed the existing ESM2 builder dry-run input contract, a normalized source-lane handoff is registered, and four corrected GPU-requesting production ESM2 shards are queued on Apolo accel. It is still not a completed warehouse/atlas integration.

## What Is Now Grounded

- Paper DOI: `10.1128/msystems.00680-25`
- Zenodo concept DOI: `10.5281/zenodo.8194032`
- Data-bearing Zenodo record: `10.5281/zenodo.8194033`
- Latest Zenodo record observed through the API: `10.5281/zenodo.10622292`
- All six Zenodo `8194033` payload files are downloaded and md5-validated.
- `MAGs.zip` contains 2,508 local OWC FASTA entries, while the published/deposit headline denominator remains 2,502 HQ/MQ MAGs.

## Key Outputs

Generated artifacts are under:

```text
results/functional_metagenomics/mucc_v1_owc_wetland_20260626/
```

Key outputs:

- `downloads/mucc_v1_payload_download_status.tsv`
- `manifests/mucc_v1_mag_catalog_full.tsv`
- `manifests/mucc_v1_lane_manifest.tsv`
- `manifests/mucc_v1_source_lane_manifest.tsv`
- `manifests/mucc_v1_functional_manifest.tsv`
- `manifests/mucc_v1_source_ready_manifest.tsv`
- `manifests/mucc_v1_source_bin_crosswalk.tsv`
- `manifests/mucc_v1_protein_catalog_summary.tsv`
- `manifests/mucc_v1_esm2_input_manifest.tsv`
- `proteomes/`
- `expression/feature_mucc_v1_expression_mag_summary.tsv`
- `expression/fact_mucc_v1_expression_mag_sample.tsv.gz`
- `expression/feature_mucc_v1_gene_expression_mag_summary.tsv`
- `expression/fact_mucc_v1_gene_expression_mag_sample.tsv.gz`
- `functional_features/feature_mucc_v1_gene_annotation_mag_summary.tsv`
- `functional_features/feature_mucc_v1_source_dram_mag_summary.tsv`
- `functional_features/feature_mucc_v1_functional_status.tsv`
- `functional_features/feature_mucc_v1_glm2_status.tsv`
- `functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv`
- `environmental_metadata/feature_sample_risk_readiness_scaffold.tsv`
- `candidate_cards/mucc_v1_expression_candidate_cards_status.tsv`
- `candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv`
- `candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv`
- `reports/validation_gap_register.tsv`
- `reports/claim_boundary_matrix.tsv`
- `reports/mucc_v1_esm2_launch_status.tsv`
- `manifests/mucc_v1_glm2_ready_manifest.partial_file_scan.tsv`
- `manifests/mucc_v1_glm2_ready_gap_register.partial_file_scan.tsv`
- `manifests/mucc_v1_glm2_ready_manifest.tsv`
- `manifests/mucc_v1_glm2_ready_gap_register.tsv`
- `results/contextual_genomics/mucc_v1_owc_wetland_glm2_full_20260626/validation/prep_summary.json`
- `results/contextual_genomics/mucc_v1_owc_wetland_glm2_full_20260626/prepared_inputs/glm2_sequences.jsonl`
- `bridge_reanchoring/legacy_poc_mucc_neighbor_validation/wetland_reference_neighbor_edges.tsv`
- `bridge_reanchoring/legacy_poc_mucc_neighbor_validation/wetland_reference_neighbor_summary.tsv`
- `bridge_reanchoring/consolidator_validation_futian_shards/embedding_metadata.tsv`
- `bridge_reanchoring/consolidator_validation_futian_shards/genome_embeddings.npz`
- `cohort_warehouse/cohort_table_manifest.tsv`
- `cohort_warehouse/functional_atlas.duckdb`
- `reports/mucc_v1_stop_condition_external_compute_blockers_20260626.tsv`
- `reports/MUCC_V1_STOP_CONDITION_20260626.md`
- `reports/mucc_v1_integration_completion_audit.tsv`
- `reports/mucc_v1_integration_completion_audit.json`

The lane is registered in:

```text
configs/methanet_atlas_lanes.tsv
```

## Validated Counts

| Check | Result |
| --- | ---: |
| Published MUCC v1 HQ/MQ MAG headline denominator | 2,502 |
| Local FASTA entries in `MAGs.zip` | 2,508 |
| Processed MAG expression rows | 1,948 |
| Source metatranscriptome/sample columns | 133 |
| Source DRAM mapped MAGs | 2,508 |
| Directly protein-mapped MAGs | 2,501 |
| Per-MAG proteome FASTA files | 2,501 |
| ESM2 dry-run inventory candidates present | 2,501 |
| ESM2 dry-run missing FAA files | 0 |
| Production ESM2 shard jobs submitted | 4 |
| Active production ESM2 GPU shard job IDs | 11354, 11355, 11356, 11357 |
| Superseded no-checkpoint ESM2 launch | 11350, 11351, 11352, 11353 canceled/replaced |
| Prodigal/GFF generation job for gLM2 prep | 11359 complete; final manifest written; log finished 2026-06-25T21:44:55-05:00 |
| Partial gLM2 smoke inference job | 11361 |
| Partial gLM2 prepared smoke payload | 32 MAGs, 64 windows, 658 spans |
| Full-lane gLM2 prepared payload | 2,508 MAGs, 5,016 windows, 51,141 spans |
| Full-lane gLM2 inference job | 11363 queued |
| Gene-expression annotation mapped MAGs | 1,948 |
| Gene-expression MAG x sample rows | 259,084 |
| Expression-supported review/status cards | 25 |
| Strategic source-scaffold review cards | 40 |
| MAG-level MRV readiness scaffold rows | 2,508 |
| MRV feature scaffold rows ready pending embedding outputs | 1,945 |
| MRV readiness review cards | 100 |
| Source-scaffold warehouse tables | 17 |
| Source-scaffold warehouse `dim_mag` rows | 2,508 |
| Source-scaffold warehouse MAG x sample expression fact rows | 259,084 |
| Source-scaffold warehouse gene-expression MAG x sample fact rows | 259,084 |
| Stop-condition blocker rows | 6 |
| Prodigal-derived gLM2-ready final manifest rows | 2,508 |
| Prodigal-derived gLM2 blocked final manifest rows | 0 |
| Prodigal non-empty triplets observed by completion audit | 2,508 FAA / 2,508 GFF / 2,508 FFN / 2,508 complete triplets |
| Legacy POC MUCC neighbor-validation query rows | 107 |
| Legacy POC MUCC neighbor-validation edge rows | 1,605 |
| ESM2 consolidator validation rows over completed Futian shards | 3,156 |
| Requirement-level completion audit | 5 fully complete; 6 partial or incomplete; final goal complete = false |

## Claim Boundary

Allowed wording:

> MUCC v1 is now source-audited and locally staged as a data-bearing Old Woman Creek wetland reference lane scaffold for MethaNet, with source DRAM annotation support for 2,508 local FASTA entries, direct source protein support and corrected production ESM2 GPU jobs queued for 2,501 entries under the current parser, normalized source-lane handoff rows, processed MAG/gene expression support for 1,948 MAGs across 133 source sample columns, a 2,508-row MAG-level MRV readiness feature scaffold for review, and a native source-scaffold warehouse with 17 Parquet-backed tables plus a DuckDB catalog.

Blocked wording:

- Do not claim exact local integration of the published 2,502-MAG HQ/MQ denominator until the six-entry local FASTA discrepancy is reconciled.
- Do not claim all 2,508 local FASTA entries are embedding-ready until the seven source protein mapping gaps are reconciled.
- Do not claim ESM2 embeddings, gLM2 features, or wetland-neighbor tables exist from the dry-run input inventory or submitted jobs.
- Do not claim final MRV risk scores or final A-E methane-risk tiers.
- Do not claim measured methane flux from MAG or expression evidence alone.
- Do not claim carbon-credit approval, registry readiness, or VM0033 compliance from molecular evidence alone.
- Do not claim source-independent wetland-to-mangrove or rumen-to-mangrove transfer.
- Do not promote expression-supported status cards into methane-mechanism candidate cards until curated features and embedding gates pass.
- Do not promote strategic source-scaffold review cards into final bridge/mechanism cards until curated functional runs and embedding-neighbor gates pass.
- Do not treat `feature_mucc_v1_mrv_readiness_mag_level.tsv` or `mucc_v1_mrv_readiness_candidate_cards.tsv` as final MRV scoring outputs; they are source-scaffold review artifacts pending embeddings, curated features, environmental joins, uncertainty propagation, and validation.
- Do not treat `cohort_warehouse/functional_atlas.duckdb` as final MethaNet-curated mechanism evidence; it is native query infrastructure over source-scaffold evidence.
- Do not treat `mucc_v1_glm2_ready_manifest.tsv` or the prepared full-lane gLM2 inputs as completed gLM2 feature coverage; they prove input readiness, not model inference completion.
- Do not treat `legacy_poc_mucc_neighbor_validation` as the MUCC v1 bridge table; it validates the neighbor-builder path on older POC MUCC rows only.
- Do not treat `consolidator_validation_futian_shards` as MUCC evidence; it validates the sharded-embedding consolidation path on already completed Futian artifacts.
- Treat Prodigal context as manifest-backed for input readiness because `mucc_v1_prodigal_proteome_manifest.tsv` is written and validates at 2,508/2,508 rows; the Prodigal wrapper log reports a clean finish at 2026-06-25T21:44:55-05:00.

## Remaining Work Before Full Atlas Value

The active stop-condition ledger is `reports/mucc_v1_stop_condition_external_compute_blockers_20260626.tsv`, with human-readable handoff in `reports/MUCC_V1_STOP_CONDITION_20260626.md`.

1. Reconcile `MAGs.zip` 2,508 local FASTA entries against the 2,502 published HQ/MQ denominator.
2. Reconcile seven local FASTA entries without direct source FAA protein records.
3. Confirm expression units/normalization from methods or source metadata.
4. Build MethaNet-curated feature warehouse tables from the staged source MAG/protein/annotation payloads.
5. Monitor corrected production ESM2 GPU shard jobs `11354`, `11355`, `11356`, and `11357`; once complete, refresh the lane registry and validate `embedding_metadata.tsv`/`genome_embeddings.npz` coverage.
6. Preserve the final Prodigal manifest as the source for generated FAA/GFF/FFN context; rerun only if source FASTA or derivation settings change.
7. Monitor partial gLM2 smoke job `11361`; use it only to validate the MUCC context path, not as full-lane gLM2 coverage.
8. Monitor full-lane gLM2 inference job `11363`; summarize outputs only after `features/glm2_smoke_window_embedding_summary.tsv` and `embeddings/glm2_smoke_window_embeddings.npz` exist and validate.
9. Use the new source-scaffold warehouse and 2,508-row MRV readiness scaffold as the review queue for curated MethaNet feature generation, but promote rows only after embedding-neighbor, gLM2, environmental, uncertainty, and validation gates pass.
10. Join environmental metadata, depth, geochemistry, flux context, and metabolomics with explicit resolution tiers.
11. Rebuild wetland reference neighbor tables with `scripts/reports/build_wetland_reference_neighbor_table.py` after MUCC ESM2 embeddings are complete.
12. Promote the lane from `staged_payload_parsed_esm2_running` only after validation gates pass.

## Current Decision

Conditional no-launch for final warehouse and atlas claims. The prompt is strong enough for goal-mode execution, and the first high-value scaffold now includes payload parsing plus production ESM2 launch, but the business asset becomes defensible only after the remaining denominator, production embedding completion, gLM2, curated feature, environmental, neighbor-table, and validation gates are complete.
