# Apollo-3 MAG Functional Analytics Operations

Date: 2026-06-13
Documentation refresh: 2026-07-24

This page is the operational path for running MethaNet MAG functional analytics
on Apolo-3 with the databases that are actually installed and validated under:

```bash
export REPO_ROOT=/home/rsg-jcorre38/Jay_Proyects/MethaNet
export DB_ROOT=/home/rsg-jcorre38/scratch/methanet_db
```

## Current Readiness

Run this before launching MAG jobs:

```bash
cd "$REPO_ROOT"
scripts/check_functional_mag_db_readiness_apollo3.sh | column -t -s $'\t'
```

As of `fgx_bakta_light_20260611_231310`, the production-ready stack is:

| Layer | Status | Production use |
| --- | --- | --- |
| CheckM2 | ready | MAG completeness/contamination |
| GUNC ProGenomes3 | ready | chimerism/artifact screening |
| GTDB-Tk R232 | ready | taxonomy |
| KOfam | ready | KO/module evidence |
| MCycDB | ready | curated methane cycling genes |
| SCycDB | ready | curated sulfur cycling genes |
| dbCAN V5 | ready | CAZyme/CGC/substrate evidence |
| METABOLIC-G | ready | biogeochemical distillation fallback for DRAM |
| Bakta 1.12.0 + light DB v6.0 | ready | optional standardized MAG annotation add-on |
| eggNOG v2 | ready, optional | staged and integrity-validated under `$DB_ROOT/eggnog_v2`; keep out of the active Slurm run unless explicitly launching the sidecar |
| DRAM/DRAM2 | gated | use only after fresh official provisioning; not a production blocker |

## Current Created Data And Database Artifacts

The operational path has produced a launch-ready local functional-atlas
warehouse plus a molecular attestation graph. Treat these as generated evidence
snapshots, not live scheduler state.

Functional atlas warehouse:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/
```

Key contents:

| Artifact | Current state |
| --- | ---: |
| `dim_mag` | 625 selected MAG/bin rows |
| `fact_run_status` | 683 run attempts, including failed and partial attempts |
| `feature_mrv_mag_level` | 625 MAG-level feature rows |
| `feature_annotation_coverage` | 7,500 MAG x tool coverage rows |
| `fact_kofam_hits` | 23,845,557 rows |
| `fact_mcycdb_hits` | 1,223,407 rows |
| `fact_scycdb_hits` | 1,525,120 rows |
| `functional_atlas.duckdb` | present |

Attestation snapshot:

```text
results/attestation/mmag_mvp_20260617/
```

Key contents:

| Artifact | Current state |
| --- | ---: |
| `graph_nodes.tsv` / `.parquet` | 662 MAG nodes plus claims, gaps, features, artifacts, evidence atoms |
| `graph_edges.tsv` / `.parquet` | evidence, feature, source, taxonomy, claim, blocker, and ESM2-neighbor relationships |
| `evidence_atom.tsv` / `.parquet` | 3,968 evidence atoms |
| `registry_artifact.tsv` / `.parquet` | 13 source artifacts with hashes/provenance |
| `mmag.kuzu/` | optional embedded graph database when Kuzu is available |
| `EXPERT_AUDIT_REPORT.md` | static and query stress audit; all listed gates pass |

Claim boundary for operators: these artifacts are suitable for MAG/proteome-level
molecular attestation, candidate triage, warehouse analytics, and partner-demo
evidence packets. They are not final sample/project MRV risk scores and do not
claim measured methane flux, final A-E tiers, source-independent transfer, or
carbon-credit approval.

## Mangrove Expansion Snapshot

The current August 10 release has validated cohort warehouses for both
mangrove lanes: MSM contains 1,428/1,428 selected complete functional payloads
and Futian contains 3,156/3,156 release-required complete payloads plus 248
explicit source gaps. Both remain pipeline-normalized screening contracts;
cross-lane mechanism comparability is not yet established.

The table below preserves the July 8 operational shard snapshot for scheduler
provenance only. Refresh the lane registry and Slurm state before operational
decisions; use `docs/current_artifact_inventory.md` for the current release.

| Payload | Current state |
| --- | ---: |
| Local mangrove/MSM MAG/proteome candidates | 1,428 |
| ESM2 embeddings | 1,428 / 1,428 complete |
| gLM2 contextual units | 1,428 / 1,428 complete |
| MSM functional MAGs complete | 1,427 / 1,428 |
| MSM functional MAGs partial/running | 1 |
| MSM manifest-scoped failed MAGs | 0 |
| MSM raw duplicate complete attempts | 4 proteome IDs |
| Futian ready MAG/proteome payload rows | 3,156 |
| Futian ESM2 embeddings | 3,156 / 3,156 complete |
| Futian gLM2 contextual units | 3,156 / 3,156 complete |
| Futian archaea functional MAGs complete | 312 / 312 |
| Futian archaea functional MAGs running/partial | 0 |
| Futian bacteria functional MAGs complete at historical shard snapshot | 15 / 2,844 |
| Futian bacteria pending at historical shard snapshot | 2,829 / 2,844 |
| Futian manifest-scoped failed rows | 0 |
| Futian archaea pending/not-started | 0 |
| Futian bacteria shards | 3 x 948 dependency-free shards; shard 001 has 7 complete MAGs, shards 002 and 003 have 4 complete MAGs each, and remaining rows are pending/not complete |

Scheduler caveat: the selected per-MAG evidence scan has complete curated
bundles for all 312 manifest-scoped archaea, even though Slurm evaluated the
parent archaea array dependency as failed for downstream `afterok` purposes.
Treat curated `status.tsv`/`parquet_manifest.tsv` sentinels as the
payload-readiness source for the archaea tranche and Slurm task state as
operational telemetry. The old dependent bacteria arrays were canceled and
replaced with explicit split-manifest arrays:

```text
11644 -> futian_phase1_functional_bacteria_001_rows_1_948.tsv
11645 -> futian_phase1_functional_bacteria_002_rows_949_1896.tsv
11646 -> futian_phase1_functional_bacteria_003_rows_1897_2844.tsv
```

All split bacteria shards are dependency-free at the manifest level. The first
15 bacteria tasks have complete curated evidence bundles on disk. Earlier
`CG`/`TIMEOUT` scheduler discrepancies were traced by Apolo admins to a Slurm
controller cleanup misconfiguration; the controller has reportedly been fixed,
so new completed jobs should release resources normally. Rows without a
curated `COMPLETE` sentinel remain pending in an operational snapshot. The
historical report freeze associated with this scheduler snapshot is
`methanet_3view_payload_freeze_20260724_scientific_reconciliation`, with 7,484
data-complete tri-views. The current controlled-diligence freeze is
`methanet_3view_payload_freeze_20260810_end_to_end`, with 7,710 data-complete
tri-views and explicit evidence-contract states.

Current sidecar paths:

```text
results/blue_catalyst_poc/runs/msm_china_2025_esm2_20260616_082112/artifacts/
results/contextual_genomics/glm2_msm_magbin_full_20260615_092737/
results/functional_metagenomics/msm_china_2025_20260615/
results/blue_catalyst_poc/runs/futian_mangrove_2026_esm2_phase1_shard*_20260621/
results/contextual_genomics/glm2_futian_phase1_shard*_20260621/
results/functional_metagenomics/futian_mangrove_2026_phase1_archaea/
results/functional_metagenomics/futian_mangrove_2026_phase1_bacteria_001/
results/functional_metagenomics/futian_mangrove_2026_phase1_bacteria_002/
results/functional_metagenomics/futian_mangrove_2026_phase1_bacteria_003/
```

Before rebuilding the expanded MBAG atlas, consolidate each mangrove tranche by
manifest rather than by completed folders alone. Preserve duplicate, partial,
failed, and not-started rows. Keep the MSM 1,428-candidate processing
denominator separate from the paper's 966 final medium/high-quality MAG
denominator, and keep the Futian 3,156-ready payload denominator separate from
the 3,404 phase-1 rMAG denominator and 248-row missing-payload gap register.
Use `scripts/reports/refresh_atlas_lane_registry_status.sh` for the live
readiness snapshot and `scripts/reports/build_methanet_3view_payload_freeze.py`
before any report rebuild or external handoff.

## Source-Backed Tool Decisions

- eggNOG-mapper upstream says v3 is still under heavy testing and production
  should use stable v2; v3 targets eggNOG v7 and is not database-compatible
  with v2. Source: https://github.com/eggnogdb/eggnog-mapper
- dbCAN upstream recommends `run_dbcan database --db_dir db --aws_s3` after the
  2026 online database outage, and the local dbCAN V5 install uses that path.
  Source: https://github.com/bcb-unl/run_dbcan
- DRAM2 is a public beta, Nextflow-based, and requires preformatted databases
  transferred via Globus plus Conda or a container runtime. Source:
  https://dramit.readthedocs.io/en/latest/installation.html
- Bakta supports isolates, MAGs, and plasmids; its recommended database command
  is `bakta_db download --output <path> --type [light|full]`. Source:
  https://github.com/oschwengers/bakta
- anvi'o metabolism is a strong optional pathway-completeness layer when KEGG
  module estimation and custom modules are needed. It reconstructs metabolic
  pathways and estimates pathway completeness from KEGG functions/modules.
  Source: https://anvio.org/help/main/programs/anvi-estimate-metabolism/

## Minimal Production Workflow

Use this stack for actual MAG analytics now:

1. QC and identity:
   CheckM2, GUNC, GTDB-Tk R232, dRep if dereplication is required.
2. Gene prediction:
   Prodigal `-p meta`, one `.faa`, `.ffn`, and `.gff` per MAG.
3. Functional evidence:
   KOfam, MCycDB, SCycDB, dbCAN, METABOLIC-G.
4. Optional broad standardized annotation:
   Bakta light DB for first-pass MAG annotations; use full DB later if the
   cohort needs deeper UniRef-backed annotations and the throughput budget is
   acceptable.
5. Optional broad orthology/EC/COG:
   eggNOG-mapper v2 is staged and validated under `$DB_ROOT/eggnog_v2`, but
   should run as a future sidecar/Snakemake lane rather than disrupting the
   active production Slurm array.

## MAG Manifest

Create a TSV with at least:

```text
mag_id	fasta_path	ecosystem	source
MAG001	/abs/path/MAG001.fa	rumen	project_a
```

Paths must be absolute or resolvable from `$REPO_ROOT`. Use stable MAG IDs
without spaces because downstream file names use `mag_id`.

## Per-MAG Commands

The commands below are intentionally explicit. They are suitable for pilots,
debugging, and translating into SLURM arrays.

```bash
export MAG_ID=MAG001
export FASTA=/abs/path/MAG001.fa
export OUT="$REPO_ROOT/results/functional_metagenomics/manual/$MAG_ID"
export THREADS=16
mkdir -p "$OUT"/{genes,kofam,mcycdb,scycdb,dbcan,bakta}
```

Gene prediction:

```bash
conda activate methanet-fgx
prodigal \
  -i "$FASTA" \
  -a "$OUT/genes/$MAG_ID.faa" \
  -d "$OUT/genes/$MAG_ID.ffn" \
  -o "$OUT/genes/$MAG_ID.gff" \
  -f gff \
  -p meta
```

KOfam:

```bash
exec_annotation \
  --cpu "$THREADS" \
  --profile "$DB_ROOT/kofam/profiles/prokaryote.hal" \
  --ko-list "$DB_ROOT/kofam/ko_list" \
  --format detail-tsv \
  -o "$OUT/kofam/$MAG_ID.kofam.detail.tsv" \
  "$OUT/genes/$MAG_ID.faa"
```

MCycDB:

```bash
diamond blastp \
  -q "$OUT/genes/$MAG_ID.faa" \
  -d "$DB_ROOT/mcycdb/MCycDB_2021.dmnd" \
  -o "$OUT/mcycdb/$MAG_ID.diamond.tsv" \
  -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp scovhsp \
  --evalue 1e-10 \
  --query-cover 70 \
  --subject-cover 50 \
  --threads "$THREADS"
```

SCycDB:

```bash
diamond blastp \
  -q "$OUT/genes/$MAG_ID.faa" \
  -d "$DB_ROOT/scycdb/SCycDB_2020Mar.dmnd" \
  -o "$OUT/scycdb/$MAG_ID.diamond.tsv" \
  -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp scovhsp \
  --evalue 1e-10 \
  --query-cover 70 \
  --subject-cover 50 \
  --threads "$THREADS"
```

dbCAN:

```bash
run_dbcan "$FASTA" prok \
  --db_dir "$DB_ROOT/dbcan" \
  --out_dir "$OUT/dbcan" \
  --dia_cpu "$THREADS" \
  --hmm_cpu "$THREADS"
```

Bakta optional annotation:

```bash
conda activate methanet-bakta
bakta \
  --db "$DB_ROOT/bakta/db-light" \
  --meta \
  --threads "$THREADS" \
  --output "$OUT/bakta" \
  --prefix "$MAG_ID" \
  --force \
  "$FASTA"
```

## Cohort-Level Commands

Stage MAGs into one folder for tools that run by genome directory:

```bash
export STAGED="$REPO_ROOT/results/functional_metagenomics/manual/staged_genomes"
mkdir -p "$STAGED"
awk 'NR > 1 {print $1 "\t" $2}' mag_manifest.tsv |
while IFS=$'\t' read -r mag_id fasta; do
  ln -sf "$(readlink -f "$fasta")" "$STAGED/$mag_id.fa"
done
```

CheckM2:

```bash
conda activate checkm2_py38
checkm2 predict \
  --input "$STAGED" \
  --output-directory "$REPO_ROOT/results/functional_metagenomics/manual/checkm2" \
  --threads 32 \
  --extension fa \
  --database_path "$DB_ROOT/checkm2"
```

GUNC:

```bash
conda activate methanet-gunc3
gunc run \
  -d "$STAGED" \
  -o "$REPO_ROOT/results/functional_metagenomics/manual/gunc" \
  -t 32 \
  -e .fa \
  --db_file "$DB_ROOT/gunc/gunc_db_progenomes3.dmnd"
```

GTDB-Tk:

```bash
conda activate methanet-gtdbtk272
export GTDBTK_DATA_PATH="$DB_ROOT/gtdbtk_r232/release232"
gtdbtk classify_wf \
  --genome_dir "$STAGED" \
  --out_dir "$REPO_ROOT/results/functional_metagenomics/manual/gtdbtk" \
  --extension fa \
  --cpus 64
```

METABOLIC-G:

```bash
conda activate methanet-metabolic
export METABOLIC_DIR="$DB_ROOT/metabolic/METABOLIC"
find "$STAGED" -name '*.fa' | sort > "$REPO_ROOT/results/functional_metagenomics/manual/metabolic_genomes.txt"
perl "$METABOLIC_DIR/METABOLIC-G.pl" \
  -in-gn "$REPO_ROOT/results/functional_metagenomics/manual/metabolic_genomes.txt" \
  -o "$REPO_ROOT/results/functional_metagenomics/manual/metabolic" \
  -t 64
```

## Gated Tools

### eggNOG

Do not direct-download `eggnog.db.gz` from Apolo compute nodes. The server
reports `Accept-Ranges=none`, range resume fails, and direct transfers truncate
near 1.1 GB. As of 2026-06-13 the required files have been staged locally and
validated, so future work should reuse this directory rather than attempting a
fresh compute-node download:

```bash
$DB_ROOT/eggnog_v2/eggnog.db
$DB_ROOT/eggnog_v2/eggnog.taxa.db
$DB_ROOT/eggnog_v2/eggnog_proteins.dmnd
```

Validate:

```bash
conda activate methanet-fgx
test -s "$DB_ROOT/eggnog_v2/eggnog.db"
diamond dbinfo --db "$DB_ROOT/eggnog_v2/eggnog_proteins.dmnd"
emapper.py --data_dir "$DB_ROOT/eggnog_v2" --version
```

### DRAM

Do not use the broken local DRAM env for production. Use METABOLIC-G and Bakta
now. If DRAM is required, provision DRAM2 with Nextflow plus Singularity or a
fresh Conda runtime, then transfer preformatted databases from Globus as
described in the official DRAM2 installation docs.

## Completion Gates

A MAG functional analytics run is complete only when:

- every MAG has QC rows from CheckM2 and GUNC,
- every MAG has GTDB-Tk taxonomy or an explicit unresolved status,
- every MAG has Prodigal proteins,
- KOfam, MCycDB, SCycDB, and dbCAN outputs exist per MAG,
- METABOLIC-G cohort output exists,
- Bakta outputs exist if `run_bakta_optional` was enabled,
- eggNOG outputs exist only if the optional v2 sidecar/Snakemake lane was run,
- bridge candidates have explicit mechanism labels or missing-evidence reasons.
