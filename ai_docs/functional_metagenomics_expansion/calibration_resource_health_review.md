# Functional MAG Calibration Resource Health Review

Generated on Apollo-3 after the 24-MAG calibration tranche for
`fgx_662_apollo3_20260612`.

## Scheduler Health

The MethaNet calibration array job `8479` completed successfully:

- 24/24 retry array tasks completed.
- All retry tasks exited `0:0`.
- No non-empty retry stderr files were detected.
- The non-empty array stderr files belong to superseded job `8455`, which failed
  early at KOfam because `exec_annotation` was not visible in the runtime PATH.
  That failure mode was fixed before retry job `8479`.

At review time there was no active MethaNet functional MAG job in `squeue`.

## Data-Format Health

The cohort consolidation report for the completed calibration tranche is:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse/DATA_ARCHITECTURE_VALIDATION.md
```

It reports:

- 24 completed curated MAGs selected.
- 49 total run attempts preserved in `fact_run_status`.
- 81 validation gates passed.
- 0 validation failures.
- 0 validation warnings.
- Decision: `LAUNCH-READY` for data-format readiness.

This is enough evidence to conclude that the per-MAG closeout, cohort
consolidation, normalized METABOLIC tables, Parquet output, and DuckDB catalog
are functional for the inspected tranche.

## Runtime Evidence

The completed calibration tranche used 16 CPUs, 128G memory, and 8h walltime per
task.

Observed per-MAG elapsed time:

| metric | seconds | approximate |
|---|---:|---:|
| minimum | 2083 | 34.7 min |
| median | 2365.5 | 39.4 min |
| mean | 2397.4 | 40.0 min |
| p95 | 2741 | 45.7 min |
| maximum | 2753 | 45.9 min |

Runtime was dominated by:

| step | share of runtime |
|---|---:|
| METABOLIC | 38.9% |
| GUNC | 30.7% |
| dbCAN | 12.9% |
| GTDB-Tk | 8.3% |
| Bakta | 3.2% |
| KOfam | 2.7% |

Calibration runtime correlated strongly with input size:

- correlation input bp vs elapsed: about 0.87
- correlation protein count vs elapsed: about 0.86

## Why 8h Is Not The Right Production Default

The calibration tranche consisted of wetland/MUCC records around 1.1-3.3 Mbp.
The full manifest includes much larger rumen records. A fast file-size scan of
the 662-row manifest showed:

| field | median | p95 | p99 | max |
|---|---:|---:|---:|---:|
| MAG FASTA bytes | 708,919 | 7,162,879 | 142,587,188 | 333,096,647 |
| proteome FAA bytes | 871,574 | 9,662,148 | 197,600,646 | 473,522,221 |

The largest records are rumen assemblies such as:

- `rumen__10676_0024_idba`
- `rumen__10676_0023_idba`
- `rumen__10676_0009_idba`
- `rumen__10676_0014_idba`

These inputs are far larger than the calibration set. The 24-MAG tranche is
therefore sufficient for pipeline/data-format confidence, but not sufficient to
prove the worst-case runtime bound for the largest rumen records.

## CPU And Memory Assessment

### CPUs

Keep `THREADS=16` for the next production launch.

Rationale:

- METABOLIC, GUNC, dbCAN, GTDB-Tk, KOfam, DIAMOND, Bakta, and CheckM2 all accept
  thread or CPU arguments in the current runner.
- Runtime is dominated by tools that can use parallelism or benefit indirectly
  from available CPU.
- Reducing CPU now would increase walltime risk before large-rumen behavior has
  been measured.

### Memory

Keep `MEM=128G` for the next launch.

Rationale:

- Slurm accounting did not expose `MaxRSS` for the completed array.
- Per-step `/usr/bin/time` logs recorded `NA` for RSS during the calibration
  jobs, so memory right-sizing is not evidence-backed yet.
- The full manifest contains very large rumen inputs and large proteome FAA
  files.
- 128G is conservative but reasonable until a large-rumen tranche has measured
  memory telemetry.

Do not reduce memory based only on the 24-MAG calibration tranche.

## Adjusted Production Defaults

The production submitter now defaults to:

```text
THREADS=16
MEM=128G
TIME_LIMIT=24:00:00
CONCURRENCY=12
PARTITION=longjobs
```

The Slurm array worker header also uses 24h walltime for direct `sbatch` use.

`CONCURRENCY=12` is more conservative than the earlier `%24` plan. It avoids
fully saturating both longjobs nodes at 16 CPUs/task and leaves room for I/O,
memory pressure, and other cluster users.

## Storage Health

Current calibration storage:

- per-MAG run tree: about 1.8G
- cohort warehouse: about 13M
- duplicated per-run `dbcan_db_compat` directories: 0
- `staged_fasta` directories remain in 24 run folders, but these are compact
  enough in calibration and can be pruned after final review.

The shared dbCAN compatibility cache is being used correctly.

## Recommendation

There is enough evidence to submit the remaining batches from a data-format and
pipeline-health perspective.

Recommended full production launch settings:

```bash
DRY_RUN=0 \
CONCURRENCY=12 \
TIME_LIMIT=24:00:00 \
MEM=128G \
THREADS=16 \
scripts/submit_functional_mag_batches_apollo3.sh
```

Recommended operational guardrail:

Run a large-rumen sentinel tranche before or at the beginning of the full
launch if queue strategy allows it. Prioritize one or more of the largest rumen
records to measure true runtime and memory behavior. If they pass comfortably,
resource requests can be reduced in a later tranche. If they approach 24h or
show memory pressure, keep the conservative settings.

No additional jobs were submitted during this review.
