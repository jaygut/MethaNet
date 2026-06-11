# Apollo-3 Functional MAG Runbook

Date: 2026-06-11

This runbook records the Apolo-3 database setup state for MethaNet and the
next operational steps to turn MAGs into QC, taxonomy, methane/sulfur,
CAZyme, KO/EC, and pathway evidence.

## Current DB_ROOT

```bash
export DB_ROOT=/home/rsg-jcorre38/scratch/methanet_db
export REPO_ROOT=/home/rsg-jcorre38/Jay_Proyects/MethaNet
```

The initial setup job completed successfully as SLURM job `8437`.

Original setup manifest:

```bash
$DB_ROOT/manifests/tool_db_manifest.fgx_db_setup_20260611_160901.tsv
```

Manual repair manifest for the solved MCycDB, SCycDB, dbCAN, and METABOLIC
runtime gates:

```bash
$DB_ROOT/manifests/tool_db_manifest.manual_repair_20260611_222352.tsv
```

## Installed And Gated Resources

| Layer | Status | Path | Notes |
| --- | --- | --- | --- |
| CheckM2 | installed | `$DB_ROOT/checkm2/CheckM2_database/uniref100.KO.1.dmnd` | DIAMOND validation passed. |
| GTDB-Tk R232 | installed, path-sensitive | `$DB_ROOT/gtdbtk_r232/release232` | Set `GTDBTK_DATA_PATH` to `release232`, not the parent directory. |
| GUNC ProGenomes3 | installed | `$DB_ROOT/gunc/gunc_db_progenomes3.dmnd` | DIAMOND validation passed. |
| KOfam | installed | `$DB_ROOT/kofam` | `ko_list` and `profiles/prokaryote.hal` present. |
| eggNOG-mapper v2 data | gated | `$DB_ROOT/eggnog_v2` | Apollo HTTP downloads repeatedly truncate near 1.1 GB; the source advertises `Accept-Ranges=none`, so compute-node resume is not possible from this endpoint. |
| MCycDB | installed | `$DB_ROOT/mcycdb/MCycDB_2021.dmnd` | Repaired from split archive by ordered concatenation, FASTA normalization, and DIAMOND validation. |
| SCycDB | installed | `$DB_ROOT/scycdb/SCycDB_2020Mar.dmnd` | Repaired from split archive by ordered concatenation, FASTA normalization, and DIAMOND validation. |
| dbCAN | installed | `$DB_ROOT/dbcan` | Repaired with `run_dbcan database --db_dir "$DB_ROOT/dbcan" --aws_s3`; DIAMOND and CLI validation passed. |
| METABOLIC | installed, runtime validated | `$DB_ROOT/metabolic/METABOLIC` | `METABOLIC-G.pl -h` passes in `methanet-metabolic`; bundled upstream test FASTAs are absent from this clone. |
| DRAM | gated | `$DB_ROOT/dram` | Existing env has a broken `DRAM-setup.py`; rebuild from official env or use a prebuilt config bundle. |
| MMseqs2 | runtime registered | `$DB_ROOT/mmseqs` | Binary only; no generic database is required yet. |

## Production Stack

Use this as the default MAG analytics stack:

| Purpose | Production choice | Rationale |
| --- | --- | --- |
| MAG quality | CheckM2 | Completeness/contamination for MAGs. |
| Chimerism/artifacts | GUNC ProGenomes3 | Flags taxonomic discordance and contamination-like artifacts. |
| Taxonomy | GTDB-Tk 2.7.x + GTDB R232 | Current GTDB release package installed locally. |
| Gene prediction | Prodigal `-p meta` | Canonical proteins for downstream annotation. |
| KO/modules | KOfamScan | Stable KO HMM layer. |
| Orthology/EC/COG | eggNOG-mapper stable v2.1.15 | Official eggNOG-mapper repo warns v3 is still heavy-testing; keep v3/v7 exploratory. |
| CAZymes/CGCs | dbCAN V5 | Installed from the supported AWS S3 database path exposed by `run_dbcan database`. |
| Methane cycling | MCycDB_2021 | Curated methane pathway specificity. |
| Sulfur cycling | SCycDB_2020Mar | Curated sulfur pathway specificity. |
| Metabolic summary | METABOLIC-G, plus DRAM if repaired | METABOLIC-G handles MAG/SAG/isolate genomes; DRAM remains useful when its DB bundle is validated. |
| Search/clustering | DIAMOND, HMMER, MMseqs2 | Backing engines for custom marker panels and novelty searches. |

## Research-Backed Decisions For Gated Tools

| Gate | Root cause on Apollo-3 | Decision |
| --- | --- | --- |
| eggNOG | Legacy v2 HTTP downloads repeatedly truncated around 1.1 GB on compute-node network paths, and the server reports `Accept-Ranges=none` for the 6.8 GB `eggnog.db.gz`. | Keep stable eggNOG-mapper v2.1.15 as production, but stage data via `download_eggnog_data.py` from a network path that completes full files or from a trusted mirror. Keep v3/v7 isolated as preview because upstream warns v3 is still heavy testing. |
| MCycDB | Split zip parts were present, but `zip -s 0` created truncated archives on Apollo. | Use ordered concatenation of `.z01 ... .zNN + .zip`, extract, normalize FASTA, and require `diamond dbinfo`. This is now repaired locally. |
| SCycDB | Same split-zip issue as MCycDB; partial extraction also created malformed FASTA records. | Use ordered concatenation, extract, normalize FASTA, and require `diamond dbinfo`. This is now repaired locally. |
| dbCAN | The first setup used an unsupported/older database path. | Use the modern `run_dbcan database --db_dir DIR --aws_s3` command with retries/timeouts, or `dbcan_build` if that command is the one exposed by the installed package. This is now repaired locally. |
| METABOLIC | Initial runtime env missed Perl modules and the older `libnsl.so.1` ABI expected by the conda Perl build. | Use `methanet-metabolic` with Perl/R/HMMER/DIAMOND/BLAST/KOfamScan dependencies plus `libnsl`; on Apolo-3, create a compatibility symlink from `libnsl.so.1` to the available `libnsl.so.3` inside the env. This is now repaired locally for CLI/runtime validation. |
| DRAM | Existing `methanet-fgintel` has a broken `DRAM-setup.py`. | Do not make DRAM a production blocker. Use METABOLIC-G for biogeochemical summaries now, repair DRAM1 in a fresh official env if needed, and treat DRAM2 as optional public beta requiring Nextflow/container runtime and Globus-provisioned preformatted DBs. |
| Alternatives | DRAM is brittle and eggNOG data staging may remain slow. | Add optional gapseq for genome-scale metabolic pathway/model reconstruction when the project needs transporters or metabolic network predictions beyond annotation matrices. |

## Repair Rerun

Use a new `RUN_ID` for a clean manifest while reusing installed files:

```bash
cd "$REPO_ROOT"
RUN_ID=fgx_db_repair_$(date -u +%Y%m%d_%H%M%S) \
MIN_FREE_GB=200 \
EGGNOG_MODE=gated \
scripts/submit_functional_metagenomics_db_setup_apollo3.sh
```

For a same-run repair, remove only the gated checkpoints from the old state
directory and rerun the setup script:

```bash
rm -f "$DB_ROOT/.setup_state/fgx_db_setup_20260611_160901/"{eggnog_v2,mcycdb,scycdb,dbcan,dram,metabolic}".done"
```

The safer approach is a new `RUN_ID`; it preserves the original manifest.

## Gate-Specific Fixes

### eggNOG

The Apollo compute-node HTTP path repeatedly returned truncated gzip files from
the legacy v2 data endpoint. The server does not reliably support resumable
range downloads for these large files, so `curl -C -` or `aria2c` is not a full
fix.

Current Apolo-3 DNS note: `eggnogdb.embl.de` did not resolve during the
2026-06-11 retry, while `eggnog5.embl.de` resolved to `194.94.44.170` and
served the expected `emapperdb-5.0.2` files. Keep
`EGGNOG_BASE_URL=http://eggnog5.embl.de/download/emapperdb-5.0.2` unless DNS
behavior changes.

Current Apolo-3 transfer note: retry job `8439` reached the correct host, but
the first `eggnog.db.gz` transfer failed with `curl: (18) transfer closed with
5689944197 bytes remaining to read`. Resume-capable retry job `8441` preserved
the partial file, but an explicit range request returned `416 Requested Range
Not Satisfiable`, and a HEAD check returned `Accept-Ranges: none` for the
6,776,977,123-byte `eggnog.db.gz`. Gate-recording job `8442` completed
successfully and wrote:

```bash
$DB_ROOT/manifests/tool_db_manifest.fgx_eggnog_gate_20260611_230353.tsv
```

This confirms that the remaining eggNOG gate is not disk capacity or script
syntax; it is a non-resumable large-file transfer path from the eggNOG HTTP
server to the cluster. The setup script now refuses this direct transfer by
default instead of wasting scheduler time on repeated truncation. The reliable
path is to stage the three required files from a network that completes large
transfers, or to place them on a local institutional mirror, then validate them
in `$DB_ROOT`.

Recommended path:

1. Create an isolated stable env:

   ```bash
   conda create -y -n methanet-eggnog-v2 -c conda-forge -c bioconda python=3.10 eggnog-mapper=2.1.15
   conda activate methanet-eggnog-v2
   ```

2. Download on a network path that completes full files, or stage from a trusted
   mirror:

   ```bash
   export EGGNOG_BASE_URL=http://eggnog5.embl.de/download/emapperdb-5.0.2
   wget "$EGGNOG_BASE_URL/eggnog.db.gz"
   wget "$EGGNOG_BASE_URL/eggnog.taxa.tar.gz"
   wget "$EGGNOG_BASE_URL/eggnog_proteins.dmnd.gz"
   gunzip eggnog.db.gz
   tar -xzf eggnog.taxa.tar.gz
   gunzip eggnog_proteins.dmnd.gz
   ```

3. Validate before production:

   ```bash
   test -s "$DB_ROOT/eggnog_v2/eggnog.db"
   diamond dbinfo --db "$DB_ROOT/eggnog_v2/eggnog_proteins.dmnd"
   ```

Keep eggNOG v3/v7 in a separate preview path if tested:
`$DB_ROOT/eggnog_v3_preview`.

### MCycDB And SCycDB

The official repos store large FASTA resources as split zip archives. The setup
script now concatenates split archives in lexical order, extracts the FASTA,
normalizes FASTA records, and builds DIAMOND DBs.

Live Apollo-3 validation:

```bash
$DB_ROOT/mcycdb/MCycDB_2021.dmnd        # 923,871 sequences; 327,226,703 letters
$DB_ROOT/scycdb/SCycDB_2020Mar.dmnd     # 911,805 sequences; 306,406,976 letters
```

Repair validation:

```bash
diamond dbinfo --db "$DB_ROOT/mcycdb/MCycDB_2021.dmnd"
diamond dbinfo --db "$DB_ROOT/scycdb/SCycDB_2020Mar.dmnd"
```

Keep the mapping files with the search outputs:

```bash
$DB_ROOT/mcycdb/repo/id2gene.map
$DB_ROOT/scycdb/repo/id2gene.2020Mar.map
```

### dbCAN

The local dbCAN database is installed and validated under `$DB_ROOT/dbcan`.
The repaired command path was:

```bash
conda activate methanet-fgx
run_dbcan database \
  --db_dir "$DB_ROOT/dbcan" \
  --aws_s3 \
  --retries 4 \
  --timeout 120 \
  --log-level INFO
```

Validated Apollo-3 artifacts:

```bash
$DB_ROOT/dbcan/CAZy.dmnd          # 4,098,879 sequences; 2,011,582,247 letters
$DB_ROOT/dbcan/dbCAN.hmm          # 124M
$DB_ROOT/dbcan/dbCAN-sub.hmm      # 4.8G
$DB_ROOT/dbcan/peptidase_db.dmnd  # 1,227,939 sequences; 287,283,028 letters
$DB_ROOT/dbcan/sulfatlas_db.dmnd  # 151,467 sequences; 81,529,053 letters
```

Validation:

```bash
diamond dbinfo --db "$DB_ROOT/dbcan/CAZy.dmnd"
diamond dbinfo --db "$DB_ROOT/dbcan/peptidase_db.dmnd"
diamond dbinfo --db "$DB_ROOT/dbcan/sulfatlas_db.dmnd"
run_dbcan --help
run_dbcan database --help
```

### METABOLIC

METABOLIC is useful for biogeochemical trait summaries, and it is now the
production fallback for DRAM-specific biogeochemical summaries on Apolo-3. The
validated runtime is `METABOLIC_ENV=methanet-metabolic`.

Runtime repair and validation:

```bash
conda create -y -n methanet-metabolic -c conda-forge -c bioconda \
  perl perl-statistics-descriptive perl-parallel-forkmanager perl-list-util perl-getopt-long libnsl \
  r-base r-ggplot2 r-data.table hmmer diamond blast kofamscan
conda activate methanet-metabolic
ln -s libnsl.so.3 "$CONDA_PREFIX/lib/libnsl.so.1"  # only if libnsl.so.1 is missing
cd "$DB_ROOT/metabolic/METABOLIC"
bash run_to_setup.sh
perl -MStatistics::Descriptive -e 'print "Statistics::Descriptive OK\n"'
perl -MParallel::ForkManager -e 'print "Parallel::ForkManager OK\n"'
perl METABOLIC-G.pl -h
```

The upstream `-test true` mode was not used as a completion gate because this
clone does not include `METABOLIC_test_files/Guaymas_Basin_genome_files`.

### DRAM

Do not rely on the existing `methanet-fgintel` DRAM install until the syntax
error in `DRAM-setup.py` is gone. Preferred repair is either:

- a fresh official DRAM environment from the upstream repository/environment
  file, followed by database setup, or
- a pre-provisioned DRAM database/config bundle copied into `$DB_ROOT/dram`.

Validation:

```bash
DRAM.py -h
DRAM-setup.py print_config
```

### Optional gapseq

Use gapseq when the question needs metabolic network reconstruction, transporter
inference, or growth-medium/phenotype hypotheses beyond KO/EC/module matrices.
It is not a replacement for methane/sulfur curated marker databases.

```bash
conda create -y -n methanet-gapseq -c conda-forge -c bioconda gapseq
conda activate methanet-gapseq
gapseq update-sequences -t Bacteria
gapseq update-sequences -t Archaea
gapseq doall proteins.faa.gz
```

## End-To-End MAG Analytics

### 1. Manifest Inputs

Create a MAG manifest with absolute paths:

```bash
sample	genome_fasta	source	ecosystem	project
MAG001	/path/to/MAG001.fa	rumen	rumen	project_a
MAG002	/path/to/MAG002.fa	wetland	wetland	project_b
```

### 2. QC And Taxonomy

Run in batch jobs:

```bash
checkm2 predict \
  --threads 16 \
  --database_path "$DB_ROOT/checkm2/CheckM2_database" \
  --input <genome_dir> \
  --output-directory <out>/checkm2

gunc run \
  --input_dir <genome_dir> \
  --db_file "$DB_ROOT/gunc/gunc_db_progenomes3.dmnd" \
  --out_dir <out>/gunc \
  --threads 16

GTDBTK_DATA_PATH="$DB_ROOT/gtdbtk_r232/release232" \
gtdbtk classify_wf \
  --genome_dir <genome_dir> \
  --out_dir <out>/gtdbtk \
  --cpus 16
```

Output contract:
`mag_qc_integrated.tsv` as described in
`docs/functional_metagenomics_expansion.md`.

### 3. Proteins

```bash
prodigal -p meta -i genome.fa -a proteins.faa -d genes.fna -o prodigal.gff
```

Keep one canonical protein FASTA per MAG; all annotation layers should point to
that same protein set.

### 4. Functional Annotation

KOfam:

```bash
exec_annotation \
  --cpu 16 \
  --profile "$DB_ROOT/kofam/profiles/prokaryote.hal" \
  --ko-list "$DB_ROOT/kofam/ko_list" \
  --format mapper \
  proteins.faa > kofam.tsv
```

eggNOG:

```bash
emapper.py \
  -i proteins.faa \
  --itype proteins \
  --data_dir "$DB_ROOT/eggnog_v2" \
  --cpu 16 \
  -o eggnog
```

MCycDB and SCycDB:

```bash
diamond blastp \
  --query proteins.faa \
  --db "$DB_ROOT/mcycdb/MCycDB_2021.dmnd" \
  --out mcycdb.tsv \
  --outfmt 6 qseqid sseqid pident length evalue bitscore qcovhsp scovhsp \
  --threads 16

diamond blastp \
  --query proteins.faa \
  --db "$DB_ROOT/scycdb/SCycDB_2020Mar.dmnd" \
  --out scycdb.tsv \
  --outfmt 6 qseqid sseqid pident length evalue bitscore qcovhsp scovhsp \
  --threads 16
```

dbCAN:

```bash
run_dbcan proteins.faa protein \
  --db_dir "$DB_ROOT/dbcan" \
  --out_dir dbcan_out \
  --dia_cpu 16 \
  --hmm_cpu 16
```

METABOLIC-G:

```bash
perl "$DB_ROOT/metabolic/METABOLIC/METABOLIC-G.pl" \
  -in-gn <genome_dir> \
  -o <out>/metabolic_g \
  -t 16
```

### 5. Integration Gates

Before claiming mechanism-level readiness:

- Every bridge candidate has CheckM2, GUNC, GTDB-Tk, and derep status.
- Every bridge candidate has one canonical protein set.
- KOfam, eggNOG, dbCAN, MCycDB, and SCycDB coverage is measured per genome.
- Missing annotations are represented as explicit `missing` or `not_run`
  states, not silent nulls.
- Mechanism cards separate methanogenesis, methanotrophy/AOM,
  methylotrophy, sulfur competition, substrate flexibility, and artifact flags.

## Source Notes

- eggNOG-mapper upstream: https://github.com/eggnogdb/eggnog-mapper
- eggNOG v2 data index: https://eggnog5.embl.de/download/emapperdb-5.0.2/
- dbCAN installation docs: https://dbcan.readthedocs.io/en/latest/installation.html
- run_dbCAN upstream: https://github.com/bcb-unl/run_dbcan
- MCycDB upstream: https://github.com/qichao1984/MCycDB
- SCycDB upstream: https://github.com/qichao1984/SCycDB
- METABOLIC upstream: https://github.com/AnantharamanLab/METABOLIC
- DRAM upstream: https://github.com/WrightonLabCSU/DRAM
- DRAM2 docs: https://dramit.readthedocs.io/en/latest/installation.html
