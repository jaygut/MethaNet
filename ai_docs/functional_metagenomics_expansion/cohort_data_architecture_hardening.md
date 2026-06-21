# Functional MAG Cohort Data Architecture Hardening

Date: 2026-06-13
Documentation refresh: 2026-06-20

## Purpose

This note defines the hardened data architecture for the MethaNet
functional-genomics MAG cohort. It is based on the Apollo-3 calibration tranche
under `results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag/`.

The design goal is a compact, reproducible, Parquet-first functional atlas that
can be queried from Python, R, SQL/DuckDB, dashboards, and future MRV workflows
without re-reading large tool-native output bundles.

This document describes data architecture and validation contracts. It is not a
live scheduler-status report. During active Apollo-3 runs, refresh Slurm and
per-MAG sentinel state separately before making operational decisions.

## Historical Calibration State

The initial calibration tranche contained multiple attempts per MAG:

- Initial attempts failed at KOfam because the runtime did not expose
  `exec_annotation`.
- Retry attempts completed successfully enough to validate the curation and
  cohort-consolidation model.
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

While production jobs are active, avoid editing runner behavior, pruning
per-MAG folders, or regenerating cohort warehouses unless that is the explicit
operational task. Documentation and standalone read-only/reporting utilities are
safe to update.

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

For the current MAG/bin-comparable POC functional-atlas denominator, use the
unit-scoped manifest and expect 625 selected MAG/bin rows. The 662-row backbone
remains authoritative for ESM2/attestation identity, but 37 assembly-context
rumen no-bin records must remain excluded from MAG/bin feature tables unless an
explicit assembly-context evidence lane is requested.

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

## Multi-Lane Warehouse Model

Use separate cohort warehouses for analytically distinct denominators, then
build an explicit multi-view union layer on top. Do not merge lanes by whatever
folders happen to be complete on disk.

| Lane | Recommended warehouse role | Denominator rule |
| --- | --- | --- |
| POC MAG-bin rumen + wetland/MUCC | completed reference warehouse for current MBAG reporting | 625 MAG/bin-comparable units from the 662-row ESM2 backbone |
| POC assembly-context units | preserved evidence/status lane, not MAG-bin feature denominator | 37 rumen no-bin or assembly-context records, explicitly quarantined |
| Mangrove/MSM expansion | target-domain expansion warehouse after active functional tranche completion or dated interim snapshot | 1,428 local candidates as the local processing denominator, with a separate reconciled view for the paper-reported 966 final medium/high-quality MAG denominator |
| Multi-view atlas/report layer | report/query union across ESM2, functional warehouse, gLM2, provenance, and QC | manifest-driven left joins with explicit missingness and lane labels |

For every warehouse, the manifest is authoritative. Completed folders indicate
available evidence, not the cohort denominator. Failed, partial, superseded,
duplicate, and not-yet-started records must remain visible in status tables.

## Current Generated Warehouse Snapshot

The latest launch-ready generated warehouse observed during the documentation
refresh is:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/
```

Snapshot summary from `DATA_ARCHITECTURE_VALIDATION.md`:

| Item | Value |
| --- | ---: |
| Cohort run | `fgx_poc_magbin_union_20260616` |
| Run attempts inspected | 683 |
| Completed curated MAG/bin runs selected | 625 |
| Complete attempts | 644 |
| Failed attempts | 24 |
| Partial attempts | 15 |
| Validation gates | 711 pass |
| Launch decision | `LAUNCH-READY` |

The warehouse contains these table families:

- dimensions: `dim_mag`, `dim_gene`
- operational facts: `fact_run_status`, `fact_tool_timing`, `fact_input_stats`
- QC/taxonomy facts: `fact_qc_checkm2`, `fact_qc_gunc`, `fact_taxonomy_gtdbtk`
- functional facts: `fact_kofam_hits`, `fact_mcycdb_hits`,
  `fact_scycdb_hits`, `fact_dbcan_hits`, `fact_bakta_features`,
  `fact_metabolic_hmm_hits`, `fact_metabolic_function_presence`,
  `fact_metabolic_module_presence`, `fact_metabolic_module_step_presence`,
  `fact_cazy_hits`, `fact_merops_hits`
- feature summaries: `feature_annotation_coverage`,
  `feature_methane_mechanism`, `feature_sulfur_competition`,
  `feature_mrv_mag_level`
- summary metrics: `run_summary_metrics`

The optional DuckDB catalog is present at:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/functional_atlas.duckdb
```

## Mangrove/MSM Expansion Readiness

The mangrove/MSM lane is an active expansion, not yet the final consolidated
warehouse. Snapshot at the 2026-06-20 documentation refresh:

| Item | Value |
| --- | ---: |
| Local mangrove/MSM MAG/proteome candidates | 1,428 |
| ESM2 embeddings | 1,428 / 1,428 complete |
| gLM2 contextual units | 1,428 / 1,428 complete |
| Functional MAGs complete | 1,002 / 1,428 |
| Partial/running MAGs | 3 |
| Failed MAGs | 0 |
| Not yet started | 423 |

When this tranche is consolidated, use the same per-run curated Parquet
contract and validation gates as the POC warehouse. Add two denominator fields
to every mangrove/MSM cohort summary:

- `local_archive_denominator = 1428`
- `published_quality_denominator = 966` when reconciling to the source paper's
  reported final medium/high-quality MAG set.

This prevents three different concepts from being collapsed: local processable
MAG candidates, source-publication quality-filtered MAGs, and the subset that
has completed functional evidence at a dated point in time.

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

## Launch Decision Semantics

Validation reports encode a dated decision for the outputs they inspected.
Historical calibration reports may contain:

```text
CONDITIONAL NO-LAUNCH
```

That means the data model can be structurally sound while the inspected tranche
is incomplete or still carrying accepted warnings. It should not be interpreted
as a permanent production-launch state.

For full-cohort scientific analysis, proceed only after:

1. The intended tranche or cohort reports the expected selected completed
   curated MAG count.
2. `validation_gates.tsv` has no `fail` rows for that dated warehouse.
3. Any `warn` rows are either resolved or explicitly accepted in a report.
4. The cohort warehouse contains normalized METABOLIC long tables and no
   tool-native wide MAG columns.
5. Metadata joins preserve `metadata_resolution` and do not inflate source/site
   provenance into sample-level environmental covariates.
