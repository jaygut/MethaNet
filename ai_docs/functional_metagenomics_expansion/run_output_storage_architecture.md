# Functional Run Output Storage Architecture

Date: 2026-06-12

Scope: storage, retention, and aggregation design for the 662-MAG MethaNet functional-metagenomics production run, based on the completed real-MAG smoke run:

```text
results/functional_metagenomics/one_mag_smoke/one_mag_fgx_rumen__10674_0002_idba_bin.8_20260611_231754
```

## Smoke Run Verdict

The smoke run is scientifically sufficient to prove the end-to-end pipeline shape and to start compelling MAG functional analysis, with caveats that should be explicit in the cohort tables.

Observed useful layers:

| layer | smoke evidence |
| --- | --- |
| identity | `proteome_id=rumen__10674_0002_idba_bin.8`, `mag_id=10674_0002_idba_bin.8` |
| assembly stats | 246 contigs, 1,175,663 bp, N50 5,292 bp |
| gene calls | 1,335 Prodigal proteins |
| QC | CheckM2 completeness 55.85%, contamination 0.0%; GUNC pass true |
| taxonomy | GTDB-Tk resolves to Methanobrevibacter sp900314635 |
| KOfam | 20,558 detailed rows, including accepted threshold-marked hits |
| methane/sulfur hits | MCycDB 1,743 rows; SCycDB 1,407 rows |
| CAZy | dbCAN overview has 19 called rows |
| broad annotation | Bakta has 1,260 feature rows plus JSON/GFF3 outputs |
| biogeochemistry | METABOLIC workbook and worksheet outputs exist |
| walltime | SLURM job 8454 completed in 40m 53s on 16 CPUs |

This is enough to run the remaining MAGs if the next production step stores extracted, normalized tables and removes or centralizes re-creatable scratch outputs. The main scientific caveat is that this MAG is medium/low completeness, so any absent pathway should be reported as "not detected under current completeness/coverage", not as definitive absence.

## Scaling Blocker Found

The smoke directory is approximately 5.6 GB. About 5.5 GB is avoidable per-run dbCAN compatibility/index state:

```text
dbcan_db_compat/
```

Keeping that directory for 662 MAGs would waste roughly 3.6 TB on duplicated HMMER pressed indexes. The production runner must do one of the following before the cohort run:

1. Build a single shared, read-only dbCAN compatibility/index directory under `DB_ROOT` and point all jobs to it.
2. If a tool insists on local working copies, delete `dbcan_db_compat/` after successful extraction and preserve only `dbcan/overview.txt` plus compact raw evidence if needed.

This is the only large storage defect found in the smoke output. Other outputs are manageable after selective compression and deletion of duplicate/intermediate files.

## External Format Review

The storage choice should optimize for small files, stable schemas, fast cohort joins, incremental append, and easy downstream query from Python/R/SQL.

Key source-backed observations:

- Apache Parquet is a column-oriented format designed for efficient storage, retrieval, compression, and analytics-tool interoperability: https://parquet.apache.org/
- DuckDB can query Parquet with projection and filter pushdown, so large hit tables can be scanned without reading unnecessary columns or row groups: https://duckdb.org/docs/current/data/parquet/overview
- DuckDB can write Hive-partitioned Parquet folders with `PARTITION_BY`, which matches a multi-run cohort warehouse: https://duckdb.org/docs/lts/data/partitioning/partitioned_writes
- Apache Arrow/PyArrow is the natural Python bridge for writing Parquet while controlling row-group, page, and encoding behavior: https://arrow.apache.org/docs/python/parquet.html
- SQLite is excellent for a single-file application/reporting format with stable schema, transactions, and long-term portability, but it is less ideal than partitioned Parquet for large columnar hit lakes: https://www.sqlite.org/appfileformat.html
- RO-Crate is a lightweight standard for packaging research data with metadata, and is a good provenance model for run manifests: https://www.researchobject.org/ro-crate/
- Zarr is excellent for chunked compressed N-dimensional arrays, but the primary functional outputs here are relational/event tables, not native dense tensors: https://zarr.readthedocs.io/
- AnnData has a useful sparse matrix on-disk model for MAG x feature matrices, but it should be a derived analysis artifact, not the canonical raw annotation store: https://anndata.readthedocs.io/en/latest/fileformat-prose.html

## Recommended Stack

Use a three-layer storage model:

| layer | format | role |
| --- | --- | --- |
| canonical extracted tables | partitioned Parquet | compact, appendable, columnar fact/dimension tables |
| query/catalog layer | DuckDB database or SQL views over Parquet | cohort joins, validation gates, feature summaries |
| per-run provenance | compact JSON plus TSV manifests | run status, tool versions, file paths, checksums, resource use |

Do not use one large JSON file for hit-level annotations. JSON should only hold metadata, run summaries, and card-like objects. Do not use xarray/Zarr as the primary store for this pipeline; use them later only if we derive dense tensor-like arrays. Use AnnData/Zarr or sparse matrix formats later for ML-ready MAG x KO/CAZy/module matrices.

## Proposed Directory Layout

```text
results/functional_metagenomics/{cohort_run_id}/
├── manifests/
│   ├── cohort_identity.parquet
│   ├── run_manifest.json
│   ├── file_manifest.parquet
│   ├── tool_versions.tsv
│   └── tool_db_manifest.tsv
├── warehouse/
│   ├── duckdb/
│   │   └── functional_atlas.duckdb
│   └── parquet/
│       ├── dim_mag/
│       ├── dim_gene/
│       ├── fact_input_stats/
│       ├── fact_tool_timing/
│       ├── fact_qc_checkm2/
│       ├── fact_qc_gunc/
│       ├── fact_taxonomy_gtdbtk/
│       ├── fact_kofam_hits/
│       ├── fact_mcycdb_hits/
│       ├── fact_scycdb_hits/
│       ├── fact_dbcan_hits/
│       ├── fact_bakta_features/
│       ├── fact_metabolic_hmm_hits/
│       ├── fact_metabolic_function_presence/
│       ├── fact_metabolic_module_presence/
│       ├── fact_metabolic_module_step_presence/
│       ├── fact_cazy_hits/
│       ├── fact_merops_hits/
│       ├── feature_annotation_coverage/
│       ├── feature_methane_mechanism/
│       ├── feature_sulfur_competition/
│       └── feature_mrv_mag_level/
├── per_mag/
│   └── {proteome_id}/
│       ├── provenance/run_record.json
│       ├── logs/
│       └── raw_selected/
└── failures/
    └── {proteome_id}/
```

Partition high-volume Parquet facts by at least `tool` or table family, and include `cohort_run_id`, `run_id`, `proteome_id`, and `mag_id` in every table. For append safety, write job-local Parquet shards first, validate them, then atomically move them into the cohort warehouse.

## Canonical Table Grains

| table | grain | notes |
| --- | --- | --- |
| `dim_mag` | one row per `proteome_id` | left-join backbone for all 662 MAGs |
| `dim_gene` | one row per called gene/protein | from Prodigal/Bakta; stable `gene_id` |
| `fact_input_stats` | MAG x statistic | contigs, total bp, N50, GC if available |
| `fact_tool_timing` | MAG x step | walltime, exit code, resource request |
| `fact_qc_checkm2` | MAG | completeness, contamination, model, protein count |
| `fact_qc_gunc` | MAG | pass flag, CSS, RRS, mapped genes |
| `fact_taxonomy_gtdbtk` | MAG | full taxonomy plus ANI/AF and release |
| `fact_kofam_hits` | protein x KO hit | keep all hits with `accepted_hit`; derive accepted-only views |
| `fact_mcycdb_hits` | protein x methane DB hit | deduplicate exact duplicates and keep best-hit ranking |
| `fact_scycdb_hits` | protein x sulfur DB hit | same DIAMOND schema as MCycDB |
| `fact_dbcan_hits` | protein x CAZy call | parsed from overview plus optional HMMER/DIAMOND evidence |
| `fact_bakta_features` | feature | use TSV/GFF3/JSON-derived feature IDs |
| `fact_metabolic_*` | MAG x function/module/step/hit | parse workbook/worksheet outputs into long tables |
| `feature_annotation_coverage` | MAG x tool | protein count, annotated count, coverage fraction |
| `feature_mrv_mag_level` | MAG | final dashboard/model-ready summary |

## Retention Policy

Keep these for successful runs:

- `summary.tsv`, `status.tsv`, `timings.tsv`, `input_stats.tsv`, `submission.tsv`
- CheckM2 `quality_report.tsv`
- GUNC maxCSS report
- GTDB-Tk summary/log/warnings/json
- KOfam detail TSV
- MCycDB and SCycDB DIAMOND TSVs
- dbCAN `overview.txt`
- Bakta `*.tsv`, `*.json`, `*.gff3`, and optionally `*.txt`
- METABOLIC result workbook and worksheet tables after extraction
- compressed selected logs described below

Delete or do not copy these after successful extraction:

- per-run `dbcan_db_compat/` HMMER pressed indexes
- staged/decompressed input FASTA copies when the immutable source path and checksum are recorded
- duplicated FAA files under CheckM2/GUNC/tool scratch
- METABOLIC `intermediate_files/` and `Each_HMM_Amino_Acid_Sequence/` unless failure/debug retention is enabled
- empty stdout/stderr files
- temporary GTDB-Tk/GUNC working directories

For failed runs, preserve a full `failures/{proteome_id}/` bundle with complete logs and enough scratch state to debug. Successful runs should be slim.

## Log Policy

Create a dedicated per-MAG log bundle:

```text
per_mag/{proteome_id}/logs/
├── run.status.tsv
├── timings.tsv
├── resource_usage.tsv
├── tool_versions.tsv
├── error_scan.tsv
├── slurm.out.gz
├── slurm.err.gz
└── by_tool/
    ├── {tool}.stderr.gz
    └── {tool}.stdout.gz
```

Rules:

- Always keep status, timings, SLURM stdout/stderr, tool versions, and error scan.
- Gzip text logs at closeout.
- Keep per-tool stdout/stderr only if non-empty, warning-bearing, or failed; otherwise record a zero-byte/suppressed entry in `file_manifest`.
- Move verbose tool-native logs into `raw_selected/` only when they are scientifically or operationally useful.

## Production Gate Before 662 MAGs

Before launching the cohort:

1. Centralize or purge dbCAN compatibility indexes so they do not duplicate per MAG.
2. Add an extraction/closeout step that writes the per-run JSON record and Parquet shards.
3. Validate every MAG leaves a row in `dim_mag`, even failed or skipped MAGs.
4. Validate every top bridge candidate has QC, taxonomy, annotation coverage, and mechanism-class status.
5. Validate absent pathways are never interpreted without annotation coverage and MAG completeness context.

This design is the safest path to a compact, auditable, analysis-ready functional atlas for the full MethaNet cohort.

Implementation status as of 2026-06-12:

- Shared dbCAN compatibility cache is implemented at `/home/rsg-jcorre38/scratch/methanet_db/dbcan_compat_pressed`.
- The one-MAG runner now uses `DBCAN_COMPAT_DIR` instead of creating per-run pressed indexes.
- The closeout utility writes `curated/run_record.json`, `curated/file_manifest.tsv`, `curated/prune_plan.json`, and smoke-tested Parquet shards.
- The production validator checks the 662-MAG manifest, file existence, script readiness, dbCAN cache readiness, and the smoke Parquet fixture.
- Batch deployment details live in `ai_docs/functional_metagenomics_expansion/production_batch_deployment_plan.md`.
