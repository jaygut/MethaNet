# Functional-Metagenomics Expansion Plan

Date: 2026-06-11
Target platform: Apolo-3 SLURM HPC
Primary current cohort: 662 genomes = 555 rumen + 107 wetland

Freshness note, 2026-06-24: this plan describes the original POC pipeline.
It has been superseded in practice by a three-lane expansion (POC core 625,
MSM China 2025 1,428, Futian 2026 3,404). Use `configs/methanet_atlas_lanes.tsv`
and `scripts/reports/summarize_atlas_lane_registry.py` for current operational
counts and per-lane consolidation commands.

## Executive Strategy

The current POC is strong because it produced a complete, finite `662 x 1280` ESM2 embedding matrix and biologically meaningful separation with sparse cross-ecosystem bridge candidates. The next level is not "more embedding"; it is mechanistic interpretation.

The expansion should run in four gates:

1. MAG QC and identity
2. methane mechanism assignment
3. broad function and substrate/electron-transfer context
4. latent-function linkage and source-aware transfer validation

The correct execution mode is:

1. run the entire workflow on one MAG,
2. run it on a small bridge-candidate panel,
3. run the same DAG across the full cohort,
4. generate platform-ready features/cards for the investor demo.

## Stage 0 - Input Normalization

### Required manifest

Create `mag_manifest.tsv` with at least:

| column | required | meaning |
| --- | --- | --- |
| `mag_id` | yes | stable internal ID; must match POC metadata where possible |
| `fasta_path` | yes | absolute MAG FASTA path on Apollo 3 |
| `source` | yes | `rumen`, `mucc`, or later source-project ID |
| `ecosystem` | yes | `rumen` or `wetland` |
| `domain_prior` | yes | current metadata domain, may be `Unknown` |
| `source_analysis_accession` | yes | accession/project provenance |
| `poc_embedding_id` | yes | ID in `embedding_metadata.tsv` or explicit `missing` |
| `is_bridge_candidate` | recommended | true for top bridge/alpha-transfer genomes |
| `alpha_transfer_score` | recommended | from the POC projection table |
| `bridge_entropy` | recommended | from the POC projection table |
| `opp_neighbor_fraction` | recommended | from the POC projection table |

### Pilot manifest

Start with one MAG:

- one high-priority bridge candidate if FASTA exists,
- preferably one rumen archaeal bridge candidate because current strongest bridge signal is rumen Archaea,
- if testing wetland first, explicitly expect taxonomy to be unresolved until GTDB-Tk runs.

Then expand to a bridge panel:

- top 8 alpha-transfer candidates,
- top 12 per ecosystem where FASTA exists,
- 5 non-bridge controls per ecosystem matched by protein count and domain/taxonomy where possible.

## Stage 1 - Single-MAG Pilot

Run the complete DAG on one MAG before scaling.

### 1. Assembly and MAG quality

Tools:

- QUAST for contiguity and GC/length sanity
- CheckM2 for completeness/contamination
- GUNC for chimerism/contamination from discordant gene taxonomy

Outputs:

- `quast/report.tsv`
- `checkm2/quality_report.tsv`
- `gunc/GUNC.*.maxCSS_level.tsv`
- `mag_qc_integrated.tsv`

Decision:

- high quality: completeness >= 90 and contamination <= 5 and GUNC pass
- medium quality: completeness >= 50 and contamination <= 10 and GUNC pass or reviewed warning
- fail: completeness < 50, contamination > 10, or strong GUNC chimerism

Investor-demo behavior:

- Do not hide failed candidates.
- Show them as "geometry-prioritized but QC-blocked"; that is stronger scientifically than overclaiming.

### 2. Taxonomy and derep identity

Tools:

- GTDB-Tk for taxonomy
- dRep for cohort dereplication after more than one MAG is included

Outputs:

- `gtdbtk.bac120.summary.tsv`
- `gtdbtk.ar53.summary.tsv`
- `taxonomy_resolved.tsv`
- `derep_clusters.tsv`

Gate:

- every bridge candidate must have GTDB-Tk taxonomy or an explicit unresolved status.
- every candidate must have derep cluster membership once cohort mode is enabled.

### 3. Gene calling

Use one canonical ORF set per MAG to avoid annotation drift.

Default:

- Prodigal in metagenomic mode for MAGs

Outputs:

- `{mag}.faa`
- `{mag}.ffn`
- `{mag}.gff`
- `{mag}.prodigal.log`

Rule:

- all downstream custom searches should use this canonical `.faa`.
- tools that internally call genes may still run, but integration should reconcile back to canonical gene IDs where possible.

### 4. Methane mechanism layer

This is the key MethaNet scientific layer.

Core evidence streams:

- existing 12-marker panel plus ratio, preserved for continuity
- MCycDB methane-cycle families
- KOfam/KEGG modules for methanogenesis, methanotrophy, methylotrophy, AOM, Hdr/electron bifurcation
- METABOLIC biogeochemical summaries
- manual curated marker rules for mcr/mtr/hdr/methyltransferase/pmo/mmo modules

Minimum marker/pathway groups:

| mechanism | marker/module families |
| --- | --- |
| hydrogenotrophic methanogenesis | `mcrABG`, `mtrABCDEFGH`, `fwd/fmd`, `ftr`, `mch`, `mtd`, `mer`, `hdrABC`, `mvhADG`, `frh`, `eha/ehb` |
| acetoclastic methanogenesis | `mcrABG`, `cdh/acs`, acetate kinase/phosphotransacetylase context |
| methylotrophic methanogenesis | `mtaABC`, `mtmBC`, `mtbBC`, `mttBC`, corrinoid activation proteins such as `ramA` |
| aerobic methane oxidation | `pmoABC`, `mmoXYZBCD`, methanol dehydrogenases `xoxF/mxaF` |
| anaerobic methane oxidation | ANME-like `mcrABG`, reverse methanogenesis context, `hdr`, `dsrAB` coupling where relevant |
| sulfur competition/coupling | `dsrAB`, `aprAB`, `sat`, `sox`, `sqr`, `fcc`, sulfur reductases |
| reducing power/electron transfer | hydrogenases, formate dehydrogenases, ferredoxin/flavodoxin, Hdr/Mvh/Eha/Ehb |

Outputs:

- `methane_marker_panel.tsv`
- `mcycdb_hits.tsv`
- `methane_pathway_completeness.tsv`
- `methane_module_evidence.json`
- `bridge_mechanism_card.json`
- `bridge_mechanism_card.md`

Mechanism classes:

- `methane_relevant_high_confidence`
- `methane_relevant_partial`
- `substrate_flexible`
- `sulfur_associated`
- `unclear_function`
- `likely_artifact_or_qc_blocked`

### 5. Broad functional layer

Tools:

- KOfamScan for KO calls
- eggNOG-mapper stable release for orthology-derived functions
- DRAM/DRAM2 for distillates
- METABOLIC-G for genome-level biogeochemical function
- run_dbCAN for CAZymes, CAZyme gene clusters, and substrate predictions
- SCycDB for sulfur-cycle specificity where sulfur competition matters

Outputs:

- `ko_matrix.tsv`
- `ec_matrix.tsv`
- `module_completeness.tsv`
- `eggnog_annotations.tsv`
- `dram_distillate.tsv`
- `metabolic_traits.tsv`
- `dbcan_overview.tsv`
- `cazy_family_matrix.tsv`
- `cgc_substrate_predictions.tsv`
- `scycdb_hits.tsv`
- `annotation_coverage_qc.tsv`

Coverage gate:

- report per-MAG protein annotation coverage for every tool.
- if bridge candidates rank high only where annotation coverage is low, mark as unresolved.
- compare annotation coverage across rumen/wetland before interpreting pathway absence.

### 6. Mechanism card

Every top bridge candidate gets a one-page card:

- identity: MAG ID, ecosystem, source, GTDB taxonomy, derep cluster
- QC: completeness, contamination, GUNC status, N50/contigs
- latent context: bridge rank, alpha-transfer score, entropy, opposite-neighbor fraction
- methane mechanism: marker/module completeness
- substrate/electron/sulfur context
- broad function summary
- confidence tier and caveats
- next experiment/model action

## Stage 2 - Bridge Panel Pilot

Run the same workflow on:

- top 8 alpha-transfer candidates,
- top 12 per ecosystem where FASTA exists,
- matched non-bridge controls.

Questions to answer:

1. Do high bridge scores correspond to coherent methane mechanism?
2. Are wetland candidates truly unknown taxonomically, or resolved by GTDB-Tk?
3. Are rumen archaeal bridge candidates functionally methane-complete or only latent-space intermediates?
4. Do sulfur-cycle or substrate-use features explain bridge geometry?

Outputs:

- `bridge_candidates_registry.tsv`
- `bridge_mechanism_cards/`
- `bridge_mechanism_summary.tsv`
- `bridge_vs_control_functional_enrichment.tsv`
- `annotation_coverage_qc.tsv`

## Stage 3 - Full-Cohort Parallel Run

Parallelize all per-MAG jobs after the bridge panel passes.

Per-MAG jobs:

- Prodigal
- KOfamScan
- eggNOG-mapper
- MCycDB/SCycDB DIAMOND searches
- run_dbCAN

Batch/cohort jobs:

- CheckM2 can run on a directory of MAGs.
- GUNC can run on a directory of MAGs.
- GTDB-Tk should run as a cohort/batch job because reference loading dominates runtime.
- dRep is inherently cohort-level.
- DRAM/METABOLIC are usually more efficient in batch mode.

Apollo 3 resource guidance:

| stage | partition | CPUs | memory | notes |
| --- | --- | ---: | ---: | --- |
| CheckM2 | bigmem | 32-64 | 0 or 128-256G | use `--lowmem` only if needed |
| GUNC | bigmem | 32-64 | 0 or 128-256G | DIAMOND-heavy |
| GTDB-Tk | bigmem | 64 | 0 | R232 needs large RAM; avoid full-tree unless needed |
| dRep | bigmem | 32-64 | 0 | run after QC for medium/high-quality genomes |
| Prodigal | CPU | 4 per MAG | 8G | embarrassingly parallel |
| KOfamScan | CPU/bigmem | 8-16 per MAG | 16-64G | HMMER-heavy |
| eggNOG-mapper | CPU/bigmem | 16-32 per MAG | 32-128G | DIAMOND/MMseqs-heavy |
| MCycDB/SCycDB | CPU | 8-16 per MAG | 16-64G | DIAMOND-heavy |
| run_dbCAN | CPU/bigmem | 8-16 per MAG | 16-64G | CGC/substrate options add cost |
| METABOLIC-G | bigmem | 32-64 | 0 | batch genomes |
| integration/reporting | CPU | 8-16 | 32-128G | pandas/plotting |

## Stage 4 - Latent-Function Linkage

Join:

- POC embedding metadata
- bridge metrics
- QC/taxonomy/derep
- methane marker/module completeness
- KO/EC/module features
- CAZy/substrate/transporter features
- METABOLIC/DRAM distillates

Analyses:

1. Bridge score vs methane module completeness.
2. Bridge entropy vs substrate/electron-transfer diversity.
3. Rumen downsampling to n=107 repeated at least 100 times.
4. UMAP seed and k-neighbor stability for bridge ranks.
5. Source-aware validation when additional sources are added.
6. Leave-one-source-out benchmark once at least two source projects per ecosystem exist.

Feature benchmarks:

- latent-only
- functional-only
- hybrid latent + functional
- hybrid + QC/taxonomy covariates
- hybrid + source-aware training/evaluation

## Stage 5 - Minimal Investor Platform Artifacts

The platform demo does not need every raw table. It needs faithful, defensible summaries:

1. `mrv_feature_table.parquet`
   - one row per genome
   - ESM latent features/bridge scores
   - QC/taxonomy
   - methane mechanism class
   - substrate/electron/sulfur summary features
   - feature confidence flags

2. `bridge_mechanism_cards.json`
   - card-ready JSON objects for top bridge candidates

3. `functional_similarity_graph.parquet`
   - genome-to-genome edges based on hybrid latent/function distances

4. `platform_dashboard_snapshot.json`
   - counts, gates, top candidates, evidence tiers

5. `investor_demo_readme.md`
   - plain-language explanation of why a candidate is recommended, blocked, or unresolved

Investor-facing rule:

- Every visualization should distinguish "latent-prioritized", "functionally supported", and "MRV-ready". This prevents overclaiming and makes the platform look mature.

## Apollo 3 Execution Sequence

### Preflight

```bash
module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate methanet-fgintel

export METHANET_ROOT=/home/rsg-jcorre38/Jay_Proyects/MethaNet
export DB_ROOT=$HOME/scratch/methanet_db
export XDG_CACHE_HOME=$HOME/.cache
export TMPDIR=$HOME/.cache/tmp
export UV_CACHE_DIR=$HOME/.cache/uv
mkdir -p "$DB_ROOT" "$XDG_CACHE_HOME" "$TMPDIR" "$UV_CACHE_DIR"
```

### Single-MAG Snakemake contract dry run

```bash
conda run -n methanet-fgx snakemake -n \
  -s ai_docs/functional_metagenomics_expansion/snakemake_backbone/Snakefile \
  --configfile ai_docs/functional_metagenomics_expansion/snakemake_backbone/config.apollo3.yaml \
  --config pilot_mag_id='"<MAG_ID_OR_PROTEOME_ID>"' \
  --cores 1 \
  --printshellcmds
```

Quote IDs containing underscores. Without quotes, Snakemake/Python can parse
numeric MAG IDs such as `2162886008_15` as numbers with digit separators.

### Single-MAG Snakemake contract execution

```bash
snakemake \
  -s ai_docs/functional_metagenomics_expansion/snakemake_backbone/Snakefile \
  --configfile ai_docs/functional_metagenomics_expansion/snakemake_backbone/config.apollo3.yaml \
  --config pilot_mag_id='"<MAG_ID_OR_PROTEOME_ID>"' \
  --use-conda \
  --cores 16 \
  --rerun-incomplete
```

### Full cohort with SLURM profile

```bash
snakemake \
  -s ai_docs/functional_metagenomics_expansion/snakemake_backbone/Snakefile \
  --configfile ai_docs/functional_metagenomics_expansion/snakemake_backbone/config.apollo3.yaml \
  --profile ai_docs/functional_metagenomics_expansion/snakemake_backbone/cluster \
  --use-conda \
  --rerun-incomplete \
  --latency-wait 120
```

## Completion Criteria

The expansion is complete when:

1. every included MAG has a curated `run_record.json`, `file_manifest.tsv`,
   `parquet_manifest.tsv`, and normalized Parquet shards,
2. the cohort warehouse has `dim_mag`, `dim_gene`, QC, taxonomy, KOfam, MCycDB,
   SCycDB, dbCAN, Bakta, METABOLIC, optional eggNOG, coverage, methane/sulfur,
   and MRV feature tables,
3. every top bridge candidate has QC/taxonomy status and a methane/sulfur
   mechanism card or explicit missing-evidence reason,
4. annotation coverage is quantified and comparable across ecosystems,
5. pathway/marker features are joined with the ESM2 latent space,
6. source-confounding caveats remain explicit,
7. investor-demo artifacts can be generated from the cohort warehouse without
   manual curation.
