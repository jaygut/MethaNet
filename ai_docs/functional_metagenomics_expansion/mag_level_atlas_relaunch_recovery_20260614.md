# MAG-Level Atlas Relaunch Recovery Snapshot

Date: 2026-06-14 05:30:36 -05:00

Latest refresh: 2026-06-14 05:56:53 -05:00

Relaunch update: 2026-06-14 06:07:19 -05:00

Launch-health update: 2026-06-14 06:11:01 -05:00

Incident/repair update: 2026-06-14 06:57:51 -05:00

Scope: dated operational snapshot and recovery decision for the MethaNet functional-genomics atlas after identifying assembly-scale rumen records in the original 662-proteome launch manifest.

This document follows:

- `ai_docs/functional_metagenomics_expansion/mag_level_atlas_relaunch_prompt.md`
- `ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md`
- `ai_docs/functional_metagenomics_expansion/pipeline_reproducibility_contract.md`

## Executive Decision

The 662-row ESM2 proteome backbone remains authoritative for embedding geometry.

The MAG-level functional atlas should not use all 662 rows as MAG/bin-comparable units. The clean MAG/bin relaunch denominator is currently:

```text
625 MAG/bin-comparable units
37 assembly-context units
0 unresolved units
```

The 37 assembly-context units are all rumen `10676_*_idba` no-bin records. They should be preserved as assembly/metagenome functional reservoir evidence, but excluded from MAG-level MBAG, MAG mechanism cards, and MAG-level MRV feature tables.

## Live Run-State Snapshot

Point-in-time Slurm state:

```text
JOBID              PARTITION  NAME                    ST  TIME     TIME_LIMIT  NODES  CPUS  MIN_MEMORY  NODELIST(REASON)
8504_[131-662%12]  longjobs   methanet_fgx_array      PD  0:00     1-00:00:00  1      16    128G        (Resources)
8504_130           longjobs   methanet_fgx_array      R   3:19:32  1-00:00:00  1      16    128G        a3-longjobs-0
8504_129           longjobs   methanet_fgx_array      R   3:24:54  1-00:00:00  1      16    128G        a3-longjobs-1
8504_128           longjobs   methanet_fgx_array      R   7:08:10  1-00:00:00  1      16    128G        a3-longjobs-1
8504_127           longjobs   methanet_fgx_array      R   7:19:01  1-00:00:00  1      16    128G        a3-longjobs-0
```

Per-MAG sentinel state in `results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag/`:

```text
COMPLETE sentinels: 126
FAILED sentinels:   0
```

The four active running tasks are no-bin rumen assembly-context records:

```text
rumen__10676_0036_idba
rumen__10676_0032_idba
rumen__10676_0026_idba
rumen__10676_0033_idba
```

These are non-comparable for MAG-level MBAG.

### Refresh At 2026-06-14 05:56:53 -05:00

The live state remained unchanged in sentinel terms:

```text
COMPLETE sentinels: 126
FAILED sentinels:   0
```

Slurm array `8504` was still active:

```text
8504_[131-662%12]  PD  old mixed-scope pending tranche
8504_127           R   rumen__10676_0036_idba  assembly_context
8504_128           R   rumen__10676_0032_idba  assembly_context
8504_129           R   rumen__10676_0026_idba  assembly_context
8504_130           R   rumen__10676_0033_idba  assembly_context
```

The scoped manifest maps task indices 127-130 to:

```text
analysis_unit_type:       assembly_context
mbag_mag_level_include:   false
claim_scope:              assembly/metagenome context
```

The old array is therefore still consuming resources on non-comparable no-bin
rumen assembly-context rows and still has the old mixed-scope pending tranche.

## Unit-Scope Classification

Generated classification artifacts:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_only.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_remaining.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.assembly_context.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.unit_scope_validation.tsv
```

Line counts:

```text
with_unit_scope.tsv:       662 data rows
mag_bin_only.tsv:          625 data rows
mag_bin_remaining.tsv:     518 data rows
assembly_context.tsv:       37 data rows
unit_scope_validation.tsv:   8 validation gates
```

Classification counts:

| Scope | Count |
| --- | ---: |
| `mag_bin` | 625 |
| `assembly_context` | 37 |
| `unresolved` | 0 |

Source by scope:

| Source | Scope | Count |
| --- | --- | ---: |
| `mucc` | `mag_bin` | 107 |
| `rumen` | `mag_bin` | 518 |
| `rumen` | `assembly_context` | 37 |

Current run status by scope:

| Scope | Latest status | Count |
| --- | --- | ---: |
| `mag_bin` | `complete` | 107 |
| `mag_bin` | `not_started` | 517 |
| `mag_bin` | `attempt_created` | 1 |
| `assembly_context` | `complete` | 19 |
| `assembly_context` | `partial` | 4 |
| `assembly_context` | `not_started` | 14 |

## Validation Gates

The unit-scope manifest builder passed:

```text
PASS embedded_backbone_rows rows=662 expected=662
PASS unique_proteome_id unique=662 total=662
PASS one_classification_per_proteome {'mag_bin': 625, 'assembly_context': 37}
PASS mag_bin_denom mag_bin=625 expected_current_rule=625
PASS remaining_mag_bin_denom remaining_mag_bin=518
PASS assembly_context_quarantine assembly_context=37 expected_current_rule=37
PASS unresolved_units unresolved=0
PASS no_no_bin_rumen_in_mag_bin examples=[]
```

The production preflight validator passed on both MAG-bin manifests:

```text
full MAG/bin manifest:      included=625 expected=625
remaining MAG/bin manifest: included=518 expected=518
scope guards:               no assembly_context or unresolved rows included
file checks:                MAG FASTA missing=0; proteome FAA missing=0
dbCAN shared cache ready
curated Parquet fixture tables present
```

## Scoped Cohort Consolidation Dry-Run

After adding unit-scope enforcement to the cohort consolidator, a scoped dry-run warehouse was generated at:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_scope_dryrun_20260614_0540/
```

Command:

```bash
./.venv/bin/python scripts/consolidate_functional_mag_cohort.py \
  --repo-root /home/rsg-jcorre38/Jay_Proyects/MethaNet \
  --cohort-run-id fgx_662_apollo3_20260612 \
  --cohort-dir results/functional_metagenomics/fgx_662_apollo3_20260612 \
  --manifest results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv \
  --expected-complete-count 107 \
  --output-dir results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_scope_dryrun_20260614_0540
```

Decision:

```text
LAUNCH-READY: data-format gates passed for the inspected calibration outputs.
```

Key dry-run validation facts:

| Check | Result |
| --- | ---: |
| validation gates | 193 pass, 0 warn, 0 fail |
| `dim_mag` rows | 107 |
| `dim_mag` non-MAG rows | 0 |
| `feature_mrv_mag_level` rows | 107 |
| `feature_mrv_mag_level` non-MAG rows | 0 |
| `fact_run_status` rows | 157 |
| `fact_run_status` MAG/bin attempts | 134 |
| `fact_run_status` assembly-context attempts | 23 |
| assembly-context complete attempts preserved in status | 19 |
| assembly-context partial attempts preserved in status | 4 |

The consolidator now uses this policy:

```text
Default: load MAG/bin facts only; preserve assembly-context attempts in fact_run_status.
Optional: use --include-assembly-context-facts only for an explicit assembly-context evidence lane.
```

DuckDB build was not verified in `.venv` because `duckdb` is not installed there; the optional catalog builder returned `None`. Parquet write/read validation was verified directly with PyArrow `ParquetFile` reads.

## Dry-Run Relaunch Command

The clean dry-run wrapper now defaults to the remaining MAG/bin manifest, preserving the 107 completed comparable MUCC outputs and avoiding duplicate relaunch work:

```bash
DRY_RUN=1 COHORT_RUN_ID=fgx_magbin_remaining_apollo3_20260614 scripts/submit_functional_mag_batches_apollo3.sh
```

Dry-run result:

```text
Prepared cohort batch command for 518 included MAGs
array=1-518%12
time=08:00:00
mem=64G
cpus=16
manifest=poc_662_functional_mag_manifest.mag_bin_remaining.tsv
ALLOW_ASSEMBLY_CONTEXT=0
DRY_RUN=1: not submitting.
```

The worker dry-run for task 1 resolved to the first remaining rumen MAG/bin unit:

```text
task_index: 1
proteome_id: rumen__10674_0001_idba_bin.10
analysis_unit_type: mag_bin
mbag_mag_level_include: true
claim_scope: MAG functional potential
```

The full filtered MAG-bin manifest starts with 107 MUCC rows, then rumen `idba_bin.*` rows. The remaining MAG/bin relaunch manifest starts after the completed MUCC block and contains only rumen `idba_bin.*` rows. Its final row is:

```text
task_index: 518
proteome_id: rumen__10676_0041_idba_bin.75
analysis_unit_type: mag_bin
```

## Code Guardrails Added

Small source-level guardrails were added so future operators do not accidentally relaunch MAG-level production from the unscoped 662 manifest:

| File | Change |
| --- | --- |
| `scripts/build_mag_unit_scope_manifests.py` | New builder for `with_unit_scope`, `mag_bin_only`, `mag_bin_remaining`, `assembly_context`, and validation TSVs. |
| `scripts/slurm/run_functional_mag_array_apollo3.sh` | Default manifest changed to `mag_bin_only`; worker refuses non-`mag_bin` rows unless `ALLOW_ASSEMBLY_CONTEXT=1`. |
| `scripts/submit_functional_mag_batches_apollo3.sh` | Default manifest changed to `mag_bin_remaining`; default MAG-bin resources set to 16 CPU, 64G, 8h. |
| `scripts/validate_functional_mag_production_gates.py` | Default manifest and expected count changed to 625; added scope gates. |
| `ai_docs/functional_metagenomics_expansion/snakemake_backbone/config.apollo3.yaml` | MAG manifest now points to `mag_bin_only`; production memory reduced to 64G for MAG-bin runs. |
| `src/methanet/mbag/data.py` | MBAG loader now prefers the unit-scope manifest and carries scope columns. |
| `scripts/reports/build_mbag_smoke_report.py` | MBAG report filters completed functional evidence to MAG-bin rows and marks non-comparable completed output as quarantined. |

Additional refresh hardening on 2026-06-14:

| File | Change |
| --- | --- |
| `scripts/slurm/run_functional_mag_array_apollo3.sh` | `DRY_RUN=1` is now accepted as an alias for `ARRAY_DRY_RUN=1`; worker dry-run exits before creating result directories or invoking the one-MAG runner. |
| `scripts/submit_functional_mag_batches_apollo3.sh` | Submit dry-run now exits before creating result/log directories. |
| `scripts/reports/build_mbag_smoke_report.py` | Smoke report validation now uses the scoped denominator: 625 comparable MAG/bin units and 37 assembly-context units. |
| `scripts/slurm/run_one_mag_functional_smoke_apollo3.sh` | Standalone runner SBATCH defaults now match MAG-bin production resources: 16 CPU, 64G, 8h; status notes are sanitized to remain single-line TSV records; future input-stat commands are launched through a small helper script instead of embedding multi-line shell functions in `status.tsv`. |

The dry-run guard was verified with fresh throwaway cohort IDs:

```text
PASS submit dry-run created no result directory
PASS worker dry-run created no result directory
```

Two pre-fix dry-run folders were marked as non-production so they cannot be
mistaken for evidence:

```text
results/functional_metagenomics/fgx_662_apollo3_20260614/README_NOT_PRODUCTION.md
results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614/README_DRY_RUN_ONLY.md
```

## What Was Preserved

No production per-MAG output folders were deleted, rewritten, pruned, or consolidated.

The current run directory remains intact:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/
```

Existing assembly-scale outputs should remain available as evidence bundles, but their analytical lane is now:

```text
assembly/metagenome functional reservoir context
```

not:

```text
MAG-level bridge-candidate functional support
```

## Scoped MBAG Smoke Report

A scoped MBAG smoke DOCX was generated as a derived, non-production report:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/reports/mbag_smoke_full_docx_scoped_20260614_0554_final/mbag_smoke_full_report.docx
```

The report uses these scoped counts:

| Measure | Count |
| --- | ---: |
| Embedded ESM2 backbone | 662 |
| Comparable MAG/bin denominator | 625 |
| Completed comparable MAG/bin outputs | 107 |
| Assembly-context denominator | 37 |
| Completed assembly-context outputs quarantined | 19 |
| Failed functional MAGs | 0 |
| Top latent bridge candidates with completed functional evidence | 1/10 |

The DOCX explicitly labels the current MBAG output as MAG-level molecular
screening, not final sample-level methane-risk scoring or carbon-credit
approval.

## Operational Recommendation

```text
cancel_current_array: yes, pending explicit user authorization
relaunch_mag_bin_only: yes, after current unscoped array is canceled or otherwise confirmed stopped
quarantine_assembly_context: yes, already implemented in scoped manifests
```

Reason: active Slurm array `8504` is still running and queuing the original unscoped task sequence. It is currently spending resources on no-bin rumen assembly-context records and will continue through the old manifest ordering unless canceled or allowed to finish.

Do not submit the MAG-bin relaunch while the unscoped array is still active unless the operator intentionally wants both to run. That would duplicate work and increase I/O pressure.

## Clean Relaunch Submitted

After explicit operator approval, the old mixed-scope Slurm array was canceled:

```text
scancel 8504
```

`sacct` confirmed tasks `8504_127`-`8504_130` and pending tranche
`8504_[131-662%12]` as `CANCELLED+`.

The clean MAG/bin-only remaining array was submitted with a fresh cohort run ID
to avoid mixing real production output into pre-existing dry-run-only folders:

```text
cohort_run_id: fgx_magbin_remaining_apollo3_20260614_clean
job_id:        8611
array:         1-518%12
resources:     16 CPUs, 64G, 08:00:00
manifest:      poc_662_functional_mag_manifest.mag_bin_remaining.tsv
```

Early launch verification:

```text
8611_[9-518%12] pending
8611_1 through 8611_8 running
first 8 tasks map to analysis_unit_type=mag_bin, mbag_mag_level_include=true
first 8 array stderr files were 0 bytes at early check
```

Launch-health update at 2026-06-14 06:11:01 -05:00:

```text
8611_1 through 8611_8 still running
8611_[9-518%12] pending under the %12 throttle/resource availability
new COMPLETE sentinels: 0
new FAILED sentinels:   0
new per-MAG attempt directories: 8
Slurm stderr files with non-zero size: 0 observed
current step for first 8 tasks: dbCAN started after successful Prodigal, KOfam, MCycDB, and SCycDB
run directory size: ~334M
```

The first eight attempts inherited the historical multi-line `input_stats`
status-note format already present in the earlier calibration tranche. This is
not expected to affect cohort analytics because timing and status consolidation
use `timings.tsv`, curated run records, and the final status line. The runner
was hardened immediately after this launch-health check so pending/future tasks
write cleaner one-line status notes.

### Incident And Repair Update At 2026-06-14 06:57:51 -05:00

Tasks `8611_1` through `8611_8` failed with Slurm exit `126` after dbCAN
completed successfully:

```text
run_one_mag_functional_smoke_apollo3.sh: line 291: .../input/<MAG>.fasta: Permission denied
```

The failure pattern is consistent with mutating the live runner script while
those first eight Bash processes were still executing. Their `status.tsv` and
`timings.tsv` show dbCAN `rc=0`; the error appears after dbCAN, and tasks
started after the runner hardening patch crossed that boundary successfully.
This is therefore treated as a live-script mutation incident, not as evidence
that dbCAN, Bakta, CheckM2, GUNC, GTDB-Tk, or METABOLIC are structurally broken.

Immediate repair:

```text
repair_job_id: 8628
array:         1-8%8
scope:         same mag_bin_remaining manifest rows that failed in 8611_1..8611_8
cohort_run_id: fgx_magbin_remaining_apollo3_20260614_clean
policy:        preserve partial failed attempts; write new timestamped attempts
```

The main array had already produced successful completed attempts after the
patch. At the repair snapshot:

```text
complete attempts in clean relaunch: 7
FAILED sentinels in clean relaunch: 0
repair array 8628: pending
main array 8611: advancing into later rows
```

A two-hour, 15-minute monitor loop is running in detached `screen`:

```text
screen session: 73159.methanet_fgx_monitor_8611_8628
monitor script: results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean/monitoring/monitor_relaunch_8611_8628.sh
snapshot log:   results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean/monitoring/monitor_8611_8628_20260614_065733.log
interval:       900 seconds
iterations:     9
```

The monitor records `squeue`, artifact counts, non-empty array stderr files,
and latest per-attempt status lines. Detached `sacct` calls were intentionally
disabled in the monitor after proving fragile in background mode; run `sacct`
manually when detailed Slurm accounting is needed.

Queue-control update at 2026-06-14 07:15:11 -05:00:

```text
intervention: scontrol update JobId=8611 ArrayTaskThrottle=1
reason:       repair array 8628 was pending on Priority and estimated for the next day
effect:       8628 moved to pending on Resources; 8611 pending rows now wait on JobArrayTaskLimit
running jobs: existing 8611 running tasks were not canceled or interrupted
restore:      set 8611 ArrayTaskThrottle back to 12 after repair rows 1-8 complete
```

Launch provenance is recorded in:

```text
results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean/LAUNCH_RECORD.md
```

## Scientific Claim Boundary

Allowed now:

> MethaNet has recovered a clean MAG/bin-level relaunch manifest for 625 comparable MAG/bin units while preserving 37 no-bin rumen records as assembly-context reservoir evidence.

Operationally, the default relaunch manifest is the 518-row remaining MAG/bin subset, because the 107 completed comparable MUCC outputs are preserved as valid MAG/bin evidence.

Allowed for completed comparable units:

> Completed MAG/bin units can support QC-aware MAG-level functional potential and MBAG candidate review after table gates pass.

Not allowed:

> No-bin rumen `10676_*_idba` assembly-context outputs support MAG-level MBAG bridge evidence.

Not allowed:

> Current molecular atlas outputs are final sample-level methane flux, final MRV risk scores, A-E risk tiers, carbon-credit approval, or source-independent rumen-to-wetland transfer proof.

## Next Action

The clean relaunch is now active as Slurm job `8611`. Monitor it with:

```bash
squeue -j 8611 -o "%.18i %.10P %.24j %.2t %.10M %.10l %.5D %.5C %.10m %.20R"
find results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean/per_mag -name COMPLETE | wc -l
find results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean/per_mag -name FAILED | wc -l
```

Do not consolidate or generate partner-facing MBAG claims from either the old
mixed-scope `fgx_662_apollo3_20260612` output or the new relaunch until the
unit-scope quarantine and post-run validation gates are applied.

### Repair Resolution: 2026-06-14 08:56 -05:00

- Repair array `8628` completed all 8 repair tasks (`8628_1` through `8628_8`) with Slurm exit `0:0`.
- All 8 originally failed proteomes now have clean `COMPLETE` sentinels in the clean relaunch directory.
- No non-empty `8628_*.err` files were present at closeout.
- The old failure mode was not reproduced; the repair tasks crossed dbCAN into Bakta/CheckM2/GUNC/GTDB-Tk/METABOLIC and curated successfully.
- Main array `8611` was restored from temporary throttle `%1` to `%12` with `scontrol update JobId=8611 ArrayTaskThrottle=12`.
- Verified scheduler state after restore: `8611_[28-518%12]` pending and `8611_27` running.
- Repair guard marker: `results/functional_metagenomics/fgx_magbin_remaining_apollo3_20260614_clean/monitoring/repair_8628_completed_and_8611_throttle_restored.marker`.
