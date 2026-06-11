# Tool and Database Registry

Date checked: 2026-06-11  
Policy: pin exact tool versions, database releases, install paths, and validation commands in `tool_db_manifest.tsv` for every Apollo 3 run.

## Web-Verified Current Guidance

| layer | default tool | database/resource | current guidance | production decision |
| --- | --- | --- | --- | --- |
| MAG quality | CheckM2 | CheckM2 DIAMOND database | CheckM2 uses lineage-independent ML models and emits `quality_report.tsv`; official usage supports folders, lists, gzip, and protein input with `--genes`. Source: https://github.com/chklovski/CheckM2 | Default QC completeness/contamination estimator |
| Chimerism | GUNC | ProGenomes 3 or GTDB r214 reference | GUNC v1.1.1 was latest in GitHub releases as of May 11, 2026; v1.1.0 added ProGenomes 3, GTDB r214, `gunc check`, and custom DB support. Source: https://github.com/grp-bork/gunc/releases | Default chimerism/artifact detector |
| Taxonomy | GTDB-Tk | GTDB-Tk R232 reference package | GTDB-Tk 2.7.2 was current in official docs; 2.7.0+ is compatible with GTDB Release 232. Source: https://ecogenomics.github.io/GTDBTk/announcements.html | Use GTDB-Tk 2.7.2 + R232 for new runs |
| Taxonomy resources | GTDB | Release 11-RS232 | GTDB homepage reports Release 11-RS232 on 2026-04-15. Source: https://gtdb.ecogenomic.org/ | Pin R232; record if Apollo has R226 instead |
| Dereplication | dRep | Mash + ANImf/FastANI | dRep docs recommend species-level representative genomes and note 95% ANI for species representatives; docs discuss 98% ANI as mapping-oriented threshold. Sources: https://drep.readthedocs.io/en/latest/overview.html and https://drep.readthedocs.io/en/stable/choosing_parameters.html | Use 95% ANI for species representatives; optionally 98% for read-mapping catalog |
| Gene calling | Prodigal | none | GTDB-Tk dependencies include Prodigal >=2.6.2; eggNOG-mapper also uses Prodigal for gene prediction when needed. Source: https://ecogenomics.github.io/GTDBTk/installing/index.html | Use Prodigal `-p meta` as canonical ORF set |
| KO annotation | KOfamScan | KOfam `profiles/` and `ko_list` | KOfamScan requires amino-acid FASTA, Ruby >=2.4, HMMER >=3.1, GNU Parallel, and KOfam database from GenomeNet FTP. Source: https://github.com/takaram/kofam_scan | Default KO/module layer |
| Orthology | eggNOG-mapper | eggNOG data | Official repo says v3 targets eggNOG v7 but is under heavy testing; for production, install stable v2 release. Source: https://github.com/eggnogdb/eggnog-mapper | Production: stable v2; optional gated v3 comparison |
| Methane cycling | MCycDB | MCycDB_2021 | MCycDB contains 298 methane cycling gene families across 10 methane metabolism pathways and 610,208 representative sequences. Source: https://github.com/qichao1984/MCycDB | Default methane-specific database |
| Sulfur cycling | SCycDB | SCycDB_2020Mar / mapping files | SCycDB contains 207 sulfur cycling gene families and 585,055 representative sequences; supports DIAMOND/usearch/blast profiling. Source: https://github.com/qichao1984/SCycDB | Default sulfur-competition/coupling layer |
| CAZymes/substrate | run_dbCAN | dbCAN V5 databases | Official current repo moved to `bcb-unl/run_dbcan`; latest shown release v5.2.9 on 2026-05-27; database command supports HTTP/AWS S3 and CGC resources. Source: https://github.com/bcb-unl/run_dbcan | Default CAZyme/CGC/substrate layer |
| Metabolic distillation | DRAM / DRAM2 | KOfam, Pfam, dbCAN, MEROPS, VOGDB, UniRef, optional KEGG | DRAM2 docs describe annotation then distillation and custom databases, but public availability has been evolving. Sources: https://dram2beta.readthedocs.io/en/latest/index.html and https://github.com/WrightonLabCSU/DRAM | Gate DRAM2; keep DRAM-compatible fallback |
| Biogeochemical traits | METABOLIC-G | METABOLIC temp/db resources | METABOLIC-G profiles MAG/SAG/isolate genomes without reads; METABOLIC-C adds read coverage/community metabolism. Source: https://github.com/AnantharamanLab/METABOLIC | Default biogeochemical trait summarizer |
| Search/clustering | MMseqs2, DIAMOND, HMMER | per-tool DBs | Required by eggNOG/dbCAN/DRAM/custom searches; GTDB-Tk docs list HMMER as dependency. | Use for custom marker and novelty layers |

## Production Pinning Rules

1. Do not rely on "latest" at runtime.
2. Download current databases once into `DB_ROOT`.
3. Record exact release, path, validation command, and date.
4. Do not update databases mid-analysis.
5. If a database updates, create a new run ID and rerun affected stages.

## Required `tool_db_manifest.tsv`

Columns:

| column | meaning |
| --- | --- |
| `run_id` | run-specific ID, e.g. `fgx_20260611_r232` |
| `tool` | `gtdbtk`, `checkm2`, `mcycdb`, etc. |
| `binary_version` | output of version/help command |
| `db_name` | exact database/resource |
| `db_release` | e.g. `GTDB R232`, `MCycDB_2021`, `dbCAN v5.2 database` |
| `install_path` | absolute Apollo 3 path |
| `env_var` | `GTDBTK_DATA_PATH`, `EGGNOG_DATA_DIR`, etc. |
| `checksum_or_signature` | md5/sha256, file count, or official archive hash |
| `validated_on` | ISO date |
| `validation_command` | command used to validate |
| `notes` | caveats, mirrors, non-default settings |

## Recommended Apollo 3 Database Layout

```bash
export DB_ROOT="$HOME/scratch/methanet_db"
mkdir -p "$DB_ROOT"/{checkm2,gtdbtk_r232,gunc,eggnog_v2,eggnog_v3_preview,kofam,mcycdb,scycdb,dbcan,dram,metabolic,pfam,tigrfam,mmseqs}
```

## Database Provisioning Commands

### CheckM2

```bash
checkm2 database --download --path "$DB_ROOT/checkm2"
checkm2 testrun --database_path "$DB_ROOT/checkm2"
```

### GUNC

```bash
gunc download_db --out_dir "$DB_ROOT/gunc" --db progenomes_3
gunc check --db_file "$DB_ROOT/gunc/<resolved>.dmnd"
```

If Apollo already has a validated GUNC database, record the exact path and do not redownload.

### GTDB-Tk R232

```bash
# Use the official GTDB-Tk R232 reference package for GTDB-Tk 2.7.x.
export GTDBTK_DATA_PATH="$DB_ROOT/gtdbtk_r232"
gtdbtk check_install
```

For runs forced to R226, mark the taxonomy release explicitly and do not mix R226 and R232 results in the same matrix.

### KOfam

```bash
cd "$DB_ROOT/kofam"
curl -L -O ftp://ftp.genome.jp/pub/db/kofam/ko_list.gz
curl -L -O ftp://ftp.genome.jp/pub/db/kofam/profiles.tar.gz
gunzip -f ko_list.gz
tar -xzf profiles.tar.gz
```

Use `profiles/prokaryote.hal` for prokaryotic MAGs unless a targeted KO list is being tested.

### eggNOG-mapper

Production:

```bash
pip install eggnog-mapper==2.1.15
download_eggnog_data.py --data_dir "$DB_ROOT/eggnog_v2"
```

Preview:

```bash
# Keep v3 isolated. Do not mix v2/v3 outputs without explicit version columns.
pip install eggnog-mapper
download_eggnog_data.py --data_dir "$DB_ROOT/eggnog_v3_preview"
```

### MCycDB

```bash
git clone https://github.com/qichao1984/MCycDB.git "$DB_ROOT/mcycdb/MCycDB"
diamond makedb \
  --in "$DB_ROOT/mcycdb/MCycDB/MCycDB_2021.fa" \
  -d "$DB_ROOT/mcycdb/MCycDB_2021"
```

The exact FASTA name may differ after unzipping the split archives; record the resolved filename.

### SCycDB

```bash
git clone https://github.com/qichao1984/SCycDB.git "$DB_ROOT/scycdb/SCycDB"
diamond makedb \
  --in "$DB_ROOT/scycdb/SCycDB/SCycDB_2020Mar.fa" \
  -d "$DB_ROOT/scycdb/SCycDB_2020Mar"
```

The exact FASTA name may differ after unzipping; record the resolved filename.

### run_dbCAN

```bash
conda install -c conda-forge -c bioconda dbcan
run_dbcan database --db_dir "$DB_ROOT/dbcan" --aws_s3
run_dbcan -h
```

The official docs also mention `dbcan_build`; use whichever command is supported by the installed V5 package and record it.

### DRAM / DRAM2

Recommended approach:

1. Use DRAM stable or the DRAM Nextflow/dev route available on Apollo.
2. Treat DRAM2-specific output as gated until the installed version is validated.
3. Record all included databases because DRAM results are database-composition dependent.

Minimum validation:

```bash
DRAM.py -h || dram2 -h
```

### METABOLIC

Install from the official repository and record:

- METABOLIC version or git commit
- `METABOLIC_temp_and_db` path
- accessory script path
- KEGG/KOfam resource path used internally

Validation:

```bash
perl METABOLIC-G.pl -h
```

## Stability Cautions

- eggNOG v2/v3 are not database-compatible; keep them separate.
- GTDB R226/R232 taxonomy labels may differ materially.
- MCycDB and SCycDB are older but methane/sulfur-specific; their value is specificity, not recency.
- DRAM/DRAM2 database setup is heavy and prone to HPC mirror/certificate issues; use pre-provisioned database bundles where possible.
- run_dbCAN database URLs changed recently; prefer the V5 `database` command or AWS S3 mirror.

