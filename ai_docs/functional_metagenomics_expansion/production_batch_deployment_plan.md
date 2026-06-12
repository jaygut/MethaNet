# Production Batch Deployment Plan

Date: 2026-06-12

Scope: launch strategy for the 662-MAG MethaNet functional-metagenomics cohort on Apollo-3, after the real-MAG smoke run and storage architecture review.

## Current Production Gate Status

Validation command:

```bash
scripts/validate_functional_mag_production_gates.py \
  --json-out results/functional_metagenomics/production_gate_preflight_20260612.json
```

Current result: all gates pass.

Verified items:

| gate | status |
| --- | --- |
| 662-row functional manifest exists | pass |
| required manifest columns present | pass |
| 662 included MAGs | pass |
| unique `proteome_id` and `mag_id` | pass |
| all included rows have `match_status=matched` | pass |
| all MAG FASTA files exist | pass |
| all proteome FAA files exist | pass |
| shared dbCAN compatibility cache exists | pass |
| runner no longer requires per-run dbCAN pressed indexes | pass |
| closeout writes curated JSON/TSV manifests | pass |
| closeout writes smoke-tested Parquet shards | pass |

The shared dbCAN compatibility cache is now built once at:

```text
/home/rsg-jcorre38/scratch/methanet_db/dbcan_compat_pressed
```

Size:

```text
5.5G
```

This prevents the former per-MAG `dbcan_db_compat/` duplication, which would otherwise waste multiple TB across the full cohort.

## Implemented Pipeline Adaptations

New or adapted scripts:

| script | role |
| --- | --- |
| `scripts/prepare_dbcan_compat_cache_apollo3.sh` | creates lock-protected shared dbCAN pressed-index cache |
| `scripts/curate_functional_mag_run.py` | writes per-run JSON record, file manifest, prune plan, and Parquet shards |
| `scripts/slurm/run_one_mag_functional_smoke_apollo3.sh` | now uses shared dbCAN cache and calls curation closeout |
| `scripts/slurm/run_functional_mag_array_apollo3.sh` | one-MAG-per-array-task worker using the 662-row manifest |
| `scripts/submit_functional_mag_batches_apollo3.sh` | dry-run-by-default `sbatch` command builder |
| `scripts/validate_functional_mag_production_gates.py` | no-submit preflight validator for launch readiness |

The closeout step writes compact Parquet under each MAG run:

```text
curated/parquet/
```

Smoke fixture size:

```text
668K
```

Smoke fixture tables include:

- `fact_tool_timing`
- `fact_qc_checkm2`
- `fact_qc_gunc`
- `fact_kofam_hits`
- `fact_mcycdb_hits`
- `fact_scycdb_hits`
- `fact_dbcan_hits`
- `fact_bakta_features`
- `fact_metabolic_*`

## Resource Model

Smoke run:

| metric | value |
| --- | ---: |
| MAG | `rumen__10674_0002_idba_bin.8` |
| CPUs | 16 |
| memory request | 128G |
| elapsed | 40m 53s |
| dbCAN press now removed from per-run path | yes |

The smoke MAG is small and moderately incomplete, so production walltime must allow substantially larger MAGs. The recommended per-task allocation is:

```text
cpus-per-task = 16
memory = 128G
time = 08:00:00
partition = longjobs
```

Rationale:

- 8 hours gives roughly 8-12x margin over the completed smoke walltime.
- One MAG per array task limits failure blast radius.
- Per-MAG closeout prunes avoidable scratch after successful extraction.
- Failed MAGs keep enough local state for debugging.

## Batch Strategy

Use a SLURM array with capped concurrency:

```text
--array=1-662%24
```

Default full-cohort dry-run command:

```bash
DRY_RUN=1 \
CONCURRENCY=24 \
TIME_LIMIT=08:00:00 \
MEM=128G \
THREADS=16 \
scripts/submit_functional_mag_batches_apollo3.sh
```

This currently resolves to:

```text
sbatch --partition=longjobs --cpus-per-task=16 --mem=128G --time=08:00:00 --array=1-662%24 ...
```

Expected completion envelope:

| assumption | approximate full-cohort elapsed |
| --- | ---: |
| average 1 hour/MAG, concurrency 24 | 28 hours |
| average 1.5 hours/MAG, concurrency 24 | 42 hours |
| average 2 hours/MAG, concurrency 24 | 56 hours |
| pessimistic every MAG reaches 8h limit, concurrency 24 | 224 hours |

Operational recommendation:

1. Run one final no-submit preflight.
2. Launch a calibration tranche of 24 MAGs with `--array=1-24%6` or equivalent override.
3. Review elapsed time, memory, failure modes, and curated Parquet/log size.
4. If clean, launch the full array with `1-662%24`.
5. Requeue only failed task indices after diagnosis.

The submit helper currently prepares the full array command by default. For a calibration tranche, either pass a manually edited `sbatch --array=1-24%6` command or temporarily run the helper output through a checked wrapper. Do not alter the manifest to create a pilot subset unless the subset manifest is explicitly named and archived.

## Launch Checklist

Before submitting:

1. Confirm the production gate validator passes.
2. Confirm `DRY_RUN=1 scripts/submit_functional_mag_batches_apollo3.sh` prints the intended command.
3. Confirm `ARRAY_DRY_RUN=1 TASK_INDEX=1 scripts/slurm/run_functional_mag_array_apollo3.sh` resolves the first manifest row.
4. Confirm `ARRAY_DRY_RUN=1 TASK_INDEX=662 scripts/slurm/run_functional_mag_array_apollo3.sh` resolves the last manifest row.
5. Confirm available scratch is sufficient for concurrent transient outputs.
6. Confirm no user-reviewed smoke evidence still needs preservation before pruning legacy per-run scratch.

Recommended first real submission, after explicit user approval:

```bash
DRY_RUN=0 \
START_INDEX=1 \
END_INDEX=24 \
CONCURRENCY=6 \
TIME_LIMIT=08:00:00 \
MEM=128G \
THREADS=16 \
scripts/submit_functional_mag_batches_apollo3.sh
```

This emits `--array=1-24%6` for calibration.

Recommended full production submission, after calibration passes:

```bash
DRY_RUN=0 \
CONCURRENCY=24 \
TIME_LIMIT=08:00:00 \
MEM=128G \
THREADS=16 \
scripts/submit_functional_mag_batches_apollo3.sh
```

## Post-Run Aggregation

After each tranche:

1. Collect every `per_mag/*/*/curated/run_record.json`.
2. Collect every `per_mag/*/*/curated/parquet_manifest.tsv`.
3. Build a cohort-level manifest of complete, failed, partial, and skipped MAGs.
4. Copy or compact per-run Parquet shards into the cohort warehouse layout from `run_output_storage_architecture.md`.
5. Run the scientific gates from `output_contracts_and_gates.md`: QC/taxonomy coverage, annotation coverage, mechanism class, and absent-pathway caveats.

No full-cohort submission has been made in this pass.
