# Functional MAG Cohort Data Architecture Validation

Generated: 2026-06-12T16:19:15.006883+00:00

## Scope

- Cohort run: `fgx_662_apollo3_20260612`
- Run attempts inspected: 49
- Completed curated MAGs selected: 24
- Attempt status counts: {'failed': 24, 'partial': 1, 'complete': 24}
- Output root: `/home/rsg-jcorre38/Jay_Proyects/MethaNet/results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse`
- DuckDB catalog: `/home/rsg-jcorre38/Jay_Proyects/MethaNet/results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse/functional_atlas.duckdb`

## Decision

LAUNCH-READY: data-format gates passed for the inspected calibration outputs.

## Table Model Written

| table | rows | columns | bytes |
|---|---:|---:|---:|
| dim_gene | 47187 | 13 | 530710 |
| dim_mag | 24 | 29 | 20745 |
| fact_bakta_features | 48267 | 15 | 647296 |
| fact_cazy_hits | 48 | 10 | 7933 |
| fact_dbcan_hits | 909 | 12 | 17567 |
| fact_input_stats | 96 | 8 | 6663 |
| fact_kofam_hits | 945792 | 14 | 8463971 |
| fact_mcycdb_hits | 55443 | 21 | 1169891 |
| fact_merops_hits | 410 | 10 | 12092 |
| fact_metabolic_function_presence | 2496 | 11 | 11503 |
| fact_metabolic_hmm_hits | 7560 | 20 | 52257 |
| fact_metabolic_module_presence | 11280 | 13 | 24915 |
| fact_metabolic_module_step_presence | 48024 | 12 | 155484 |
| fact_qc_checkm2 | 24 | 20 | 15281 |
| fact_qc_gunc | 24 | 19 | 14450 |
| fact_run_status | 49 | 12 | 11020 |
| fact_scycdb_hits | 54328 | 21 | 1170976 |
| fact_taxonomy_gtdbtk | 24 | 14 | 10193 |
| fact_tool_timing | 360 | 13 | 15300 |
| feature_annotation_coverage | 264 | 11 | 10262 |
| feature_methane_mechanism | 24 | 9 | 7613 |
| feature_mrv_mag_level | 24 | 15 | 12071 |
| feature_sulfur_competition | 24 | 10 | 8432 |
| run_summary_metrics | 528 | 8 | 9362 |

## Validation Summary

- Gate counts: {'pass': 81}

## Issues

- None.

## Final Architecture Notes

- Per-MAG run folders remain immutable evidence bundles.
- The cohort layer selects the latest completed curated run per `proteome_id`.
- Failed, partial, and superseded attempts are preserved in `fact_run_status`.
- Legacy METABOLIC workbook-derived wide columns are normalized into long analytical tables.
- Cohort tables are written as Parquet-first partitions under `parquet/<table>/cohort_run_id=<id>/`.
- DuckDB is used as a lightweight SQL catalog over the Parquet files when available.
- Absence claims must be filtered or caveated with CheckM2 completeness, contamination, GUNC status, and annotation coverage.
