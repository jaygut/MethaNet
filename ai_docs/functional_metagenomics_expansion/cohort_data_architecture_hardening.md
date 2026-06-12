# Functional MAG Cohort Data Architecture Hardening

## Purpose

This note defines the hardened data architecture for the MethaNet
functional-genomics MAG cohort. It is based on the Apollo-3 calibration tranche
under `results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag/`.

The design goal is a compact, reproducible, Parquet-first functional atlas that
can be queried from Python, R, SQL/DuckDB, dashboards, and future MRV workflows
without re-reading large tool-native output bundles.

## Current Calibration State

The calibration tranche contains multiple attempts per MAG:

- Initial attempts failed at KOfam because the runtime did not expose
  `exec_annotation`.
- Retry attempts are completing successfully.
- Per-MAG closeout writes `curated/run_record.json`,
  `curated/file_manifest.tsv`, `curated/parquet_manifest.tsv`, and Parquet
  shards.
- The cohort consolidation layer must select the latest completed curated run
  per `proteome_id`.
- Failed, partial, and superseded attempts must remain visible in
  `fact_run_status` instead of being deleted from the analytical record.

## Persistence Rule For Active Runs

The per-MAG runner invokes `scripts/curate_functional_mag_run.py` by path at the
end of each successful task. Updating that script during a running calibration
therefore affects tasks that have not yet reached curation. Existing completed
per-MAG shards are not rewritten in place; the cohort consolidator normalizes
their legacy shape into the cohort warehouse.

This gives two layers of protection:

- Future per-MAG outputs are normalized at closeout.
- Already completed per-MAG outputs are normalized at consolidation.

## Cohort Warehouse

The deterministic consolidation command is:

```bash
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
conda activate methanet-fgx

scripts/consolidate_functional_mag_cohort.py \
  --cohort-run-id fgx_662_apollo3_20260612 \
  --cohort-dir results/functional_metagenomics/fgx_662_apollo3_20260612 \
  --expected-complete-count 24 \
  --build-duckdb
```

For the full production cohort, use `--expected-complete-count 662`.

The output layout is:

```text
results/functional_metagenomics/<cohort_run_id>/cohort_warehouse/
  DATA_ARCHITECTURE_VALIDATION.md
  cohort_table_manifest.tsv
  validation_gates.tsv
  functional_atlas.duckdb
  parquet/
    <table>/
      cohort_run_id=<cohort_run_id>/
        part-00000.parquet
```

The per-MAG folders remain immutable evidence bundles. The cohort warehouse is
safe to delete and regenerate from completed per-MAG curated outputs.

## Required Identity Columns

Every cohort table carries:

- `cohort_run_id`
- `run_id`
- `proteome_id`
- `mag_id`
- `source_tool`

Stable table-specific keys are validated where available.

## Core Table Model

### Dimensions

- `dim_mag`: one row per selected completed `proteome_id`; carries MAG
  provenance, input paths, QC summaries, taxonomy, and basic assembly/protein
  counts.
- `dim_gene`: one row per Bakta feature-level gene identifier when available.

### Run And QC Facts

- `fact_run_status`: every run attempt, including failed first attempts,
  partial active runs, and successful selected runs.
- `fact_tool_timing`: per-tool walltime and RSS records.
- `fact_qc_checkm2`: completeness and contamination evidence.
- `fact_qc_gunc`: chimerism/contamination consistency evidence.
- `fact_taxonomy_gtdbtk`: normalized GTDB-Tk taxonomy extracted from
  `run_record.json`.

### Functional Hit Facts

- `fact_kofam_hits`: all KOfam detail hits with `accepted_hit`.
- `fact_mcycdb_hits`: MCycDB DIAMOND hits with `hit_rank_bitscore`.
- `fact_scycdb_hits`: SCycDB DIAMOND hits with `hit_rank_bitscore`.
- `fact_dbcan_hits`: dbCAN overview hits.
- `fact_bakta_features`: Bakta feature table.

### Normalized METABOLIC Facts

Legacy METABOLIC workbook-derived wide columns such as
`X3300004775_9.Module.presence` are not allowed in cohort analytical tables.
They are normalized into:

- `fact_metabolic_hmm_hits`
- `fact_metabolic_function_presence`
- `fact_metabolic_module_presence`
- `fact_metabolic_module_step_presence`
- `fact_cazy_hits`
- `fact_merops_hits`

The target grains are:

- MAG x HMM/gene function
- MAG x function
- MAG x KEGG module
- MAG x KEGG module step
- MAG x CAZy family
- MAG x MEROPS peptidase family

Sparse event tables such as `fact_cazy_hits` and `fact_merops_hits` may have no
rows for a MAG. That is valid biological absence in the event table, provided
`feature_annotation_coverage` records the tool/table coverage row.

### Feature Tables

- `feature_annotation_coverage`: MAG x annotation tool coverage, including row
  counts and gene-level coverage fractions where gene identifiers are available.
- `feature_methane_mechanism`: MAG-level methane mechanism evidence screen.
- `feature_sulfur_competition`: MAG-level sulfur-cycle evidence screen.
- `feature_mrv_mag_level`: compact MRV-facing summary that combines QC,
  annotation coverage, METABOLIC module counts, CAZy/MEROPS breadth, methane
  evidence, and sulfur evidence.

The methane/sulfur feature tables are screening features, not final mechanistic
claims. They must be interpreted with QC and annotation coverage.

## Validation Gates

The consolidation script validates:

- one row per `proteome_id` in `dim_mag`
- required identity columns in every table
- duplicate primary keys for stable key sets
- completed curated MAGs not missing core QC, taxonomy, or functional tables
- KOfam `accepted_hit` presence
- MCycDB/SCycDB `hit_rank_bitscore` presence
- annotation coverage per MAG x tool
- failed/partial attempts preserved in `fact_run_status`
- no tool-native wide MAG columns in normalized METABOLIC long tables

The launch decision is encoded in
`cohort_warehouse/DATA_ARCHITECTURE_VALIDATION.md`.

## Storage Policy

Keep:

- curated per-MAG Parquet shards
- `curated/run_record.json`
- `curated/file_manifest.tsv`
- `curated/parquet_manifest.tsv`
- timing/status/summary TSVs
- selected raw evidence listed in each run record
- cohort-level partitioned Parquet
- cohort-level DuckDB catalog
- validation report and manifests

Avoid:

- duplicated dbCAN compatibility/index directories per MAG
- repeated decompressed staging FASTA after successful extraction
- full logs when compressed logs are sufficient
- keeping failed first-attempt bundles after they have been reviewed and
  summarized in `fact_run_status`

Quarantine or remove failed first-attempt calibration folders only after their
failure mode is confirmed and the status rows are present in the cohort
warehouse.

## Current Launch Decision

For the active 24-MAG calibration tranche, the current decision is:

```text
CONDITIONAL NO-LAUNCH
```

The data model gates pass, but the calibration tranche should not be treated as
complete until all 24 retry tasks have completed and the consolidation command
is rerun with `--expected-complete-count 24`.

For the full 662-MAG production launch, proceed only after:

1. The 24-MAG calibration tranche reports 24 selected completed curated MAGs.
2. `validation_gates.tsv` has no `fail` rows.
3. Any `warn` rows are either resolved or explicitly accepted.
4. The final cohort warehouse contains normalized METABOLIC long tables and no
   tool-native wide MAG columns.
