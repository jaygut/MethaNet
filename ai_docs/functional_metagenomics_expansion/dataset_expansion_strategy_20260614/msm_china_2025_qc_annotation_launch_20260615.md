# MSM China 2025 QC Reconciliation And Functional Annotation Launch

Date: 2026-06-15

Scope: dated operational snapshot for the first MethaNet QC reconciliation and
deep functional annotation run over the Pan et al. 2025 Southeast China
mangrove sediment MAG payload.

This is a MAG/proteome-level functional-potential run. It is not a sample-level
methane-risk score, flux estimate, carbon-crediting claim, or final A-E risk
tier assignment.

## Starting Denominators

| Denominator | Count | Status |
| --- | ---: | --- |
| Paper-reported medium/high-quality MAGs | 966 | Published denominator from GigaScience article. |
| GigaDB `MAG_file.zip` FASTAs | 1,428 | Downloaded, MD5-verified, extracted, and normalized locally. |
| MethaNet Prodigal proteomes | 1,428 | Generated with Prodigal 2.6.3 `-p meta`. |
| MethaNet functional-run candidates | 1,428 | All rows have local FNA and FAA paths and passed preflight. |

Current interpretation:

```text
archive_denominator = 1428
published_quality_denominator = 966
reconciliation_status = pending_CheckM2_GUNC_quality_evidence
```

The repository metadata downloaded so far does not expose per-MAG completeness
and contamination fields. The correct reconciliation path is therefore to
derive local QC metrics with MethaNet's CheckM2/GUNC layer and then mark the
subset satisfying the paper-style quality gate.

## New Artifacts

| Artifact | Path | Purpose |
| --- | --- | --- |
| Functional-run manifest | `results/functional_metagenomics/msm_china_2025_20260615/manifests/msm_china_2025_functional_mag_manifest.tsv` | Adapter from the external handoff manifest into the standard MethaNet functional runner schema. |
| Tranche 1 manifest | `results/functional_metagenomics/msm_china_2025_20260615/manifests/tranches/msm_china_2025_functional_mag_manifest.tranche_001_1000.tsv` | Slurm-safe first 1,000 MAGs. |
| Tranche 2 manifest | `results/functional_metagenomics/msm_china_2025_20260615/manifests/tranches/msm_china_2025_functional_mag_manifest.tranche_1001_1428.tsv` | Slurm-safe remaining 428 MAGs. |
| QC reconciliation detail | `results/functional_metagenomics/msm_china_2025_20260615/qc_reconciliation/msm_china_2025_qc_reconciliation_detail.tsv` | One row per MAG/proteome with payload, protein-count, metadata, and reconciliation status. |
| QC reconciliation summary | `results/functional_metagenomics/msm_china_2025_20260615/qc_reconciliation/msm_china_2025_qc_reconciliation_summary.tsv` | Denominator and payload-readiness summary. |
| QC reconciliation counts | `results/functional_metagenomics/msm_china_2025_20260615/qc_reconciliation/msm_china_2025_qc_reconciliation_counts.tsv` | Source group, domain, protein QC, annotation-priority, and metadata counts. |
| Validation gap register | `results/functional_metagenomics/msm_china_2025_20260615/qc_reconciliation/msm_china_2025_validation_gap_register.tsv` | Claim blockers and next validation action. |
| CheckM2/GUNC reconciliation baseline | `results/functional_metagenomics/msm_china_2025_20260615/qc_reconciliation/msm_china_2025_qc_reconciliation_with_checkm2.tsv` | Merge target for per-MAG QC outputs as jobs complete. |
| CheckM2/GUNC reconciliation summary | `results/functional_metagenomics/msm_china_2025_20260615/qc_reconciliation/msm_china_2025_qc_reconciliation_checkm2_summary.tsv` | Current completion and local quality-gate counts. |
| Functional status summary | `results/functional_metagenomics/msm_china_2025_20260615/status/msm_china_2025_functional_status_summary.tsv` | Live per-MAG sentinel summary. |
| Functional status detail | `results/functional_metagenomics/msm_china_2025_20260615/status/msm_china_2025_functional_status_detail.tsv` | One row per expected MAG/proteome with latest run state. |
| Scheduler status JSON | `results/functional_metagenomics/msm_china_2025_20260615/status/msm_china_2025_scheduler_status.json` | Captured `squeue`/`sacct` output for jobs `8797` and `8798`. |
| Production gate JSON | `results/functional_metagenomics/msm_china_2025_20260615/manifests/production_gate_validation.json` | Full 1,428-row preflight output. |
| Tranche gate JSONs | `results/functional_metagenomics/msm_china_2025_20260615/manifests/tranches/production_gate_validation.tranche_*.json` | Per-tranche preflight outputs. |

New source utilities:

```text
scripts/external/build_msm_china_2025_functional_manifest.py
scripts/external/build_msm_china_2025_qc_reconciliation.py
scripts/external/update_msm_china_2025_qc_from_functional_runs.py
scripts/external/summarize_msm_china_2025_functional_status.py
```

## Preflight Result

The full 1,428-row manifest passed the existing MethaNet production gate
validator:

```text
included_mag_count = 1428 / 1428
unique_proteome_id = 1428 / 1428
unique_mag_id = 1428 / 1428
mag_fasta_files_exist = 1428 / 1428
proteome_faa_files_exist = 1428 / 1428
analysis_unit_type = mag_bin for all included rows
assembly_context rows included = 0
dbcan_shared_cache_ready = pass
curated_parquet_fixture_tables = pass
```

Both Slurm-safe tranche manifests also passed validation:

```text
tranche_001_1000 = 1000 included rows, 0 missing FNA/FAA paths
tranche_1001_1428 = 428 included rows, 0 missing FNA/FAA paths
```

## Annotation Priority Split

| Priority class | Count | Interpretation |
| --- | ---: | --- |
| `priority_archaea_methane_marker_review` | 80 | Archaea and methanogenesis-relevant marker review priority. |
| `priority_sulfur_competition_review` | 231 | Desulfobacterota/sulfur-associated competitors and sulfur-cycle context. |
| `qc_triage_before_mechanism_claims` | 5 | Very low protein count; annotate for status but block strong mechanism claims until QC review. |
| `standard_full_annotation` | 1,112 | Standard full MethaNet annotation. |

These labels are triage labels, not final biological mechanism classes.

## Submitted Functional Annotation Jobs

The full mangrove annotation run was submitted in two Slurm arrays because the
cluster rejected a single 1,428-task array specification.

Initial jobs `8797` and `8798` were canceled before starting after a 4-MAG
`bigmem` pilot exposed an inherited `RESULT_ROOT` export bug in
`scripts/submit_functional_mag_batches_apollo3.sh`. Pilot job `8800` wrote
partial scratch outputs into the cohort root instead of
`per_mag/<proteome_id>/<run_id>/`; those outputs were moved to:

```text
results/functional_metagenomics/msm_china_2025_20260615/quarantine/misdirected_pilot_8800_pre_export_fix/
```

The submit wrapper now exports `RESULT_ROOT=` explicitly so the array worker
derives the correct per-MAG result root from `RESULT_BASE`.

Current fixed submissions:

| Job | Partition | Array | Rows | Concurrency | Status |
| --- | --- | --- | ---: | ---: | --- |
| `8804` | `bigmem` | `1-4%2` | 4 | 2 | Pilot running; tasks 1 and 2 completed and wrote curated Parquet outputs, tasks 3 and 4 running. |
| `8807` | `longjobs` | `1-1000%4` | 1,000 | 4 | Canceled while still pending because `longjobs` remained saturated. Superseded by job `8810`. |
| `8808` | `longjobs` | `1-428%4` | 428 | 4 | Canceled while still pending because `longjobs` remained saturated. Superseded by job `8813`. |
| `8810` | `bigmem` | `1-1000%2` | 1,000 | 2 | Active full tranche 1 run; task 1 completed, tasks 2 and 3 running, later tasks pending by array limit. |
| `8813` | `bigmem` | `1-428%2` | 428 | 2 | Active full tranche 2 run; tasks 1 and 2 running. |

Combined active full-tranche maximum concurrency is four MAG annotations, each
using 16 CPUs, 64G, and an 8-hour walltime limit. The pilot uses at most two
additional `bigmem` slots and exists as the early warning lane for the
corrected output contract.

The jobs were pending at launch time because an existing MethaNet functional
array was already active. Do not cancel or prune that existing run as part of
this MSM launch.

Refresh at 2026-06-15T02:22:49Z:

```text
complete = 0
failed = 0
partial = 6
attempt_created = 0
not_started = 1422
jobs 8804_1 and 8804_2 = RUNNING
jobs 8810_1, 8810_2, 8813_1, and 8813_2 = RUNNING
jobs 8804_[3-4%2], 8810_[3-1000%2], and 8813_[3-428%2] = PENDING by JobArrayTaskLimit
current pilot steps = both pilot MAGs are actively writing METABOLIC HMM outputs after GTDB-Tk completed
full tranche current steps = three first-wave MAGs remain in GUNC and one has advanced to GTDB-Tk after GUNC completed
observed CheckM2 metrics = 6 partial MAGs
observed GUNC metrics = 3 partial MAGs; one GUNC pass=True and two GUNC pass=False
completed local quality-gate candidates = 0 because no run has a COMPLETE sentinel yet
```

Refresh at 2026-06-15T02:26:37Z:

```text
complete = 1
failed = 0
partial = 6
attempt_created = 0
not_started = 1421
curated_manifests_present = 1
jobs 8804_2, 8804_3, 8810_1, 8810_2, 8813_1, and 8813_2 = RUNNING
jobs 8804_4, 8810_[3-1000%2], and 8813_[3-428%2] = PENDING by JobArrayTaskLimit
completed MAG/proteome = msm_china_2025__group6_MAGs__v1_bins_848
completed MAG QC = CheckM2 completeness 84.0, contamination 2.46; GUNC pass=True
completed local quality-gate candidates = 1
observed CheckM2 metrics = 6 runs
observed GUNC metrics = 5 runs; two GUNC pass=True and three GUNC pass=False
partial-stage spread = 2 GTDB-Tk started, 1 GUNC started, 1 KOfam started, 2 METABOLIC started
```

The completed pilot bundle
`msm_china_2025__group6_MAGs__v1_bins_848/fgx_1_msm_china_2025__group6_MAGs__v1_bins_848_20260615_014809`
contains `COMPLETE`, `summary.tsv`, `status.tsv`, `timings.tsv`,
`curated/run_record.json`, `curated/file_manifest.tsv`, and
`curated/parquet_manifest.tsv`. The curated Parquet manifest contains 16
tables: run summary metrics, input stats, tool timing, CheckM2, GUNC, KOfam,
MCycDB, SCycDB, dbCAN, Bakta, METABOLIC HMM hits, METABOLIC function presence,
METABOLIC module presence, METABOLIC module-step presence, CAZy hits, and
MEROPS hits. A schema spot-check in the `methanet-fgx` environment confirmed
that every curated Parquet shard preserves `cohort_run_id`, `run_id`,
`proteome_id`, `mag_id`, and `source_tool`.

Operational note: sentinel searches from the cohort root must descend to at
least `per_mag/<proteome_id>/<run_id>/COMPLETE`; a `maxdepth 3` search from
`per_mag` is too shallow for this layout.

The QC reconciliation updater now separates observed partial-run CheckM2/GUNC
metrics from completed-run reconciliation evidence. At the latest refresh, one
completed pilot MAG is a completed local medium-quality-like candidate for the
published 966 denominator comparison. One additional partial run currently has
observed medium-quality-like CheckM2/GUNC evidence but remains labeled
`pending_terminal_run_before_reconciliation` until the run writes `COMPLETE`
and curated outputs. Four other first-wave MAGs have observed CheckM2 metrics
that fail the local paper-style gate because of low completeness or elevated
contamination. GUNC pass/fail rows from partial runs remain monitoring evidence,
not denominator-resolution evidence.

Refresh at 2026-06-15T02:31:53Z:

```text
complete = 2
failed = 0
partial = 6
attempt_created = 0
not_started = 1420
curated_manifests_present = 2
jobs 8804_3, 8804_4, 8810_1, 8810_2, 8813_1, and 8813_2 = RUNNING
jobs 8810_[3-1000%2] and 8813_[3-428%2] = PENDING by JobArrayTaskLimit
completed MAG/proteomes = msm_china_2025__group6_MAGs__v1_bins_848; msm_china_2025__group3_MAGs__v6_bins_2205
completed local quality-gate candidates = 1
completed QC-failed functional bundles = 1
observed CheckM2 metrics = 7 runs
observed GUNC metrics = 6 runs; two GUNC pass=True and four GUNC pass=False
partial-stage spread = 2 GTDB-Tk started, 1 GUNC started, 2 METABOLIC started, 1 Prodigal started
```

The second completed bundle,
`msm_china_2025__group3_MAGs__v6_bins_2205/fgx_2_msm_china_2025__group3_MAGs__v6_bins_2205_20260615_014809`,
also contains 16 curated Parquet shards with the required identity columns, but
it does not count as a local 966-denominator candidate because its completed QC
evidence is CheckM2 completeness 18.64, contamination 0.5, and GUNC pass=False.
Preserve it as explicit `complete_run_qc_metrics` plus `local_quality_gate_fail`
evidence for downstream QC/functional audit tables.

Refresh at 2026-06-15T02:40:22Z:

```text
complete = 3
failed = 0
partial = 6
attempt_created = 0
not_started = 1419
curated_manifests_present = 3
jobs 8804_3, 8804_4, 8810_2, 8810_3, 8813_1, and 8813_2 = RUNNING
jobs 8810_[4-1000%2] and 8813_[3-428%2] = PENDING by JobArrayTaskLimit
completed MAG/proteomes = msm_china_2025__group1_MAGs__m1_bins_1_bin.128; msm_china_2025__group6_MAGs__v1_bins_848; msm_china_2025__group3_MAGs__v6_bins_2205
completed local quality-gate candidates = 2
completed QC-failed functional bundles = 1
observed CheckM2 metrics = 7 runs
observed GUNC metrics = 7 runs; three GUNC pass=True and four GUNC pass=False
partial-stage spread = 1 CheckM2 started, 2 GTDB-Tk started, 1 KOfam started, 2 METABOLIC started
active MSM array stderr files with content = 0
```

The newly completed tranche-1 bundle,
`msm_china_2025__group1_MAGs__m1_bins_1_bin.128/fgx_1_msm_china_2025__group1_MAGs__m1_bins_1_bin.128_20260615_015618`,
contains 16 curated Parquet shards, preserves the required identity columns,
and carries a large functional payload: 3,667 Prodigal proteins, 103,345 KOfam
rows, 7,052 MCycDB hits, 10,598 SCycDB hits, 148 dbCAN overview rows, 3,469
Bakta feature rows, one CheckM2 row, one GUNC row, one GTDB-Tk summary row, and
15,176 METABOLIC result files. Its completed QC evidence is CheckM2
completeness 97.97, contamination 5.23, and GUNC pass=True. Under the current
local comparison rule it is `local_medium_quality_like`, not high-quality-like,
because contamination is above the 5 percent high-quality cutoff but within the
medium-quality-like `<=10` cutoff. It is therefore the second completed local
candidate for the published 966-denominator comparison.

Refresh at 2026-06-15T02:47:31Z:

```text
complete = 4
failed = 0
partial = 6
attempt_created = 0
not_started = 1418
curated_manifests_present = 4
jobs 8804_3, 8804_4, 8810_2, 8810_3, 8813_2, and 8813_3 = RUNNING
jobs 8810_[4-1000%2] and 8813_[4-428%2] = PENDING by JobArrayTaskLimit
completed MAG/proteomes = msm_china_2025__group1_MAGs__m1_bins_1_bin.128; msm_china_2025__group6_MAGs__v1_bins_848; msm_china_2025__group3_MAGs__v6_bins_2205; msm_china_2025__group3_MAGs__v6_bins_70_sub
completed local quality-gate candidates = 2
completed QC-failed functional bundles = 2
observed CheckM2 metrics = 8 runs
observed GUNC metrics = 7 runs; three GUNC pass=True and four GUNC pass=False
partial-stage spread = 2 dbCAN started, 1 GUNC started, 3 METABOLIC started
active MSM array stderr files with content = 0
```

The newly completed bundle,
`msm_china_2025__group3_MAGs__v6_bins_70_sub/fgx_1_msm_china_2025__group3_MAGs__v6_bins_70_sub_20260615_015623`,
contains 16 curated Parquet shards and preserves the required identity columns.
Its completed functional payload includes 2,637 Prodigal proteins, 55,333 KOfam
rows, 4,693 MCycDB hits, 3,996 SCycDB hits, 116 dbCAN overview rows, 2,407
Bakta feature rows, one CheckM2 row, one GUNC row, one GTDB-Tk summary row, and
13,396 METABOLIC result files. Its completed QC evidence is CheckM2
completeness 62.12, contamination 16.27, and GUNC pass=False, so it remains
`local_quality_gate_fail` and does not count toward the local 966-denominator
candidate tally.

Refresh at 2026-06-15T02:55:21Z:

```text
complete = 5
failed = 0
partial = 6
attempt_created = 0
not_started = 1417
curated_manifests_present = 5
jobs 8804_3, 8804_4, 8810_2, 8810_3, 8813_3, and 8813_4 = RUNNING
jobs 8810_[4-1000%2] and 8813_[5-428%2] = PENDING by JobArrayTaskLimit
completed MAG/proteomes = msm_china_2025__group1_MAGs__m1_bins_1_bin.128; msm_china_2025__group6_MAGs__v1_bins_848; msm_china_2025__group3_MAGs__v6_bins_2205; msm_china_2025__group3_MAGs__v6_bins_70_sub; msm_china_2025__group3_MAGs__v6_bins_751_sub
completed local quality-gate candidates = 2
completed QC-failed functional bundles = 3
observed CheckM2 metrics = 9 runs
observed GUNC metrics = 8 runs; three GUNC pass=True and five GUNC pass=False
partial-stage spread = 1 Bakta started, 1 dbCAN started, 1 GTDB-Tk started, 1 GUNC started, 2 METABOLIC started
active MSM array stderr files with content = 0
```

The newly completed bundle,
`msm_china_2025__group3_MAGs__v6_bins_751_sub/fgx_2_msm_china_2025__group3_MAGs__v6_bins_751_sub_20260615_015622`,
contains 16 curated Parquet shards and preserves the required identity columns.
Its completed functional payload includes 4,012 Prodigal proteins, 101,628
KOfam rows, 4,613 MCycDB hits, 6,140 SCycDB hits, 206 dbCAN overview rows,
3,396 Bakta feature rows, one CheckM2 row, one GUNC row, one GTDB-Tk summary
row, and 13,554 METABOLIC result files. Its completed QC evidence is CheckM2
completeness 56.68, contamination 18.53, and GUNC pass=False, so it remains
`local_quality_gate_fail` and does not count toward the local 966-denominator
candidate tally. The observed local quality-gate total is now 3 only because
`msm_china_2025__group1_MAGs__m1_bins_1_bin.157` has partial
medium-quality-like CheckM2 evidence; it remains
`pending_terminal_run_before_reconciliation` until a `COMPLETE` sentinel and
curated outputs exist.

Refresh at 2026-06-15T02:58:51Z:

```text
complete = 5
failed = 0
partial = 6
attempt_created = 0
not_started = 1417
curated_manifests_present = 5
jobs 8804_3, 8804_4, 8810_2, 8810_3, 8813_3, and 8813_4 = RUNNING
jobs 8810_[4-1000%2] and 8813_[5-428%2] = PENDING by JobArrayTaskLimit
completed MAG/proteomes = msm_china_2025__group1_MAGs__m1_bins_1_bin.128; msm_china_2025__group6_MAGs__v1_bins_848; msm_china_2025__group3_MAGs__v6_bins_2205; msm_china_2025__group3_MAGs__v6_bins_70_sub; msm_china_2025__group3_MAGs__v6_bins_751_sub
completed local quality-gate candidates = 2
completed QC-failed functional bundles = 3
observed CheckM2 metrics = 10 runs
observed GUNC metrics = 8 runs; three GUNC pass=True and five GUNC pass=False
observed local quality-gate candidates = 4
partial local quality-gate candidates = msm_china_2025__group1_MAGs__m1_bins_1_bin.157; msm_china_2025__group3_MAGs__v6_bins_753
partial-stage spread = 1 dbCAN started, 2 GUNC started, 3 METABOLIC started
active MSM array stderr files with content = 0
```

No new `COMPLETE` sentinel appeared in this refresh, but
`msm_china_2025__group3_MAGs__v6_bins_753` now has partial CheckM2 evidence:
completeness 67.23 and contamination 9.24. It is therefore
`local_medium_quality_like` at the partial-run QC stage, but it must remain
`pending_terminal_run_before_reconciliation` and must not count toward the
published 966-denominator candidate tally until the run reaches terminal
`COMPLETE` status and curated outputs validate. All five completed bundles were
revalidated with PyArrow at this snapshot; each contains 16 curated Parquet
shards, a `curated/parquet_manifest.tsv`, and the required identity columns
`cohort_run_id`, `run_id`, `proteome_id`, `mag_id`, and `source_tool`.

Refresh at 2026-06-15T03:04:52Z:

```text
complete = 7
failed = 0
partial = 5
attempt_created = 0
not_started = 1416
curated_manifests_present = 7
jobs 8804_4, 8810_3, 8810_4, 8813_3, and 8813_4 = RUNNING at 2026-06-15T03:05:06Z
jobs 8810_[5-1000%2] and 8813_[5-428%2] = PENDING by JobArrayTaskLimit
new completed MAG/proteomes = msm_china_2025__group3_MAGs__v3_bins_240731_sub; msm_china_2025__group1_MAGs__m1_bins_1_bin.152_sub
completed local quality-gate candidates = 2
completed QC-failed functional bundles = 5
observed CheckM2 metrics = 10 runs
observed GUNC metrics = 9 runs; four GUNC pass=True and five GUNC pass=False
observed local quality-gate candidates = 4
partial local quality-gate candidates = msm_china_2025__group1_MAGs__m1_bins_1_bin.157; msm_china_2025__group3_MAGs__v6_bins_753
partial-stage spread = 1 Bakta started, 1 dbCAN started, 1 GTDB-Tk started, 1 GUNC started, 1 METABOLIC started
active MSM array stderr files with content = 0
```

The newly completed bundle
`msm_china_2025__group3_MAGs__v3_bins_240731_sub/fgx_3_msm_china_2025__group3_MAGs__v3_bins_240731_sub_20260615_022605`
contains 15 curated Parquet shards and preserves the required identity columns.
Its manifest is missing only the sparse `fact_cazy_hits` table. Treat this as a
MAG-level annotation-coverage caveat for CAZy/METABOLIC CAZy evidence, not as a
sample-level absence claim. The input is very small: two contigs, 24,956 bp,
N50 15,543 bp, and 36 Prodigal proteins. Its completed functional payload
includes 96 KOfam rows, 157 raw MCycDB hits with 113 curated MCycDB rows, 84
SCycDB hits, two dbCAN overview rows with one curated dbCAN hit, 37 Bakta
feature rows, one CheckM2 row, one GUNC row, and METABOLIC function/module
tables. Its QC evidence is CheckM2 completeness 37.67, contamination 0.01, and
GUNC pass=True. Because completeness is below 50, it remains
`local_quality_gate_fail` and does not count toward the local 966-denominator
candidate tally.

The newly completed bundle
`msm_china_2025__group1_MAGs__m1_bins_1_bin.152_sub/fgx_2_msm_china_2025__group1_MAGs__m1_bins_1_bin.152_sub_20260615_015618`
contains 16 curated Parquet shards and preserves the required identity columns.
Its completed functional payload includes 6,734 Prodigal proteins, 122,066
KOfam rows, 6,247 raw MCycDB hits with 5,101 curated MCycDB rows, 8,096 raw
SCycDB hits with 7,990 curated SCycDB rows, 286 dbCAN overview rows with 285
curated dbCAN hits, 38 CAZy rows, 41 MEROPS rows, 6,370 Bakta feature rows, one
CheckM2 row, one GUNC row, and METABOLIC function/module tables. Its QC
evidence is CheckM2 completeness 82.31, contamination 25.55, and GUNC
pass=False. It therefore remains `local_quality_gate_fail` despite complete
functional evidence and does not count toward the local 966-denominator
candidate tally.

Use this refresh command to update the status artifacts:

```bash
python scripts/external/summarize_msm_china_2025_functional_status.py
python scripts/external/update_msm_china_2025_qc_from_functional_runs.py
```

## Expected Output Contract

For each completed MAG, the standard runner should write:

```text
results/functional_metagenomics/msm_china_2025_20260615/per_mag/<proteome_id>/<run_id>/
  COMPLETE
  summary.tsv
  status.tsv
  timings.tsv
  genes/
  kofam/
  mcycdb/
  scycdb/
  dbcan/
  bakta/
  checkm2/
  gunc/
  gtdbtk/
  metabolic/
  curated/run_record.json
  curated/file_manifest.tsv
  curated/parquet_manifest.tsv
  curated/parquet/*.parquet
```

## Completion Evidence Needed

The run should not be considered complete until current-state evidence proves:

1. jobs `8797` and `8798` have finished or all intended MAGs have terminal
   `COMPLETE`/`FAILED` status rows;
2. every failed, partial, or timed-out MAG has an explicit status record;
3. CheckM2 and GUNC outputs are available for all completed MAGs;
4. a reconciliation table marks which archive FASTAs satisfy completeness
   `>=50` and contamination `<=10`;
5. the 966-vs-1,428 denominator discrepancy is either resolved or preserved as
   an explicit blocker;
6. a cohort warehouse is consolidated from completed curated outputs;
7. methane, sulfur, CAZy/dbCAN, KOfam, Bakta, GTDB-Tk, METABOLIC, CheckM2, and
   GUNC coverage are summarized before interpreting pathway absences.

## Allowed Wording

Allowed now:

> The MSM China mangrove MAG archive has been locally packaged for MethaNet,
> preflighted, and submitted for MAG-level functional annotation.

Not allowed yet:

> MethaNet has integrated the final 966 published MSM MAGs as validated MRV risk
> evidence.

Upgrade condition:

> After QC completion and cohort consolidation, MethaNet can state which
> mangrove MAG/proteome units carry QC-aware methane, sulfur, substrate, and
> taxonomy evidence suitable for molecular screening and bridge-candidate
> prioritization.
