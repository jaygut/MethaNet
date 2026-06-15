# MethaNet Codebase, Cohort, And Dataset Expansion Strategy

Date: 2026-06-14  
Repository: `/home/rsg-jcorre38/Jay_Proyects/MethaNet`  
Prepared for: MethaNet functional atlas / MBAG / blue-carbon MRV expansion  
Status: Strategic internal memo, not a final MRV risk-scoring artifact

## Executive Takeaway

MethaNet is currently treating a 662-proteome ESM2 POC backbone as the canonical
analysis denominator. The codebase now correctly separates this into:

| Analytical lane | Count | Current treatment | Why it exists |
| --- | ---: | --- | --- |
| Wetland/MUCC MAG/bin-comparable units | 107 | Valid MAG-level functional-atlas target | Target-domain blue-carbon/wetland molecular potential; all completed in the inspected run snapshot. |
| Rumen MAG/bin-comparable units | 518 | Primary remaining source-domain MAG-level relaunch target | Methane-rich source-domain functional evidence for MBAG bridge interpretation and transfer hypotheses. |
| Rumen assembly-context units | 37 | Quarantined from MAG-level MBAG; preserved as assembly/metagenome context | These no-bin `10676_*_idba` records are assembly-scale, not MAG/bin-comparable. They are useful reservoir/context evidence but not valid MAG mechanism cards. |

The current defensible product primitive is:

> QC-aware MAG/proteome-level molecular screening and bridge-candidate
> prioritization for methane-risk follow-up.

The current atlas must not be described as final sample-level methane-risk
scoring, carbon-credit approval, measured methane flux, or source-independent
rumen-to-wetland transfer proof.

The most valuable expansion move is to add source-replicated, BioSample-anchored,
MAG-resolved mangrove and methane-rich sediment datasets, then preserve
environment/sample resolution explicitly. The highest-priority external targets
found in web search are:

1. The 966-MAG mangrove sediment MSM catalog from southeastern China.
2. PRJNA1072265, 48 BioSamples from natural and restored Tieshan Bay mangrove
   sediment archaea.
3. PRJNA1139943, methane-rich Hangzhou Bay deep coastal sediment metagenomes
   plus 27 MAG BioSamples.
4. Futian Reserve seven-year mangrove/mudflat metagenomic genome catalog.
5. Broader methane-producing environment controls: Zoige wetland methanogen MAGs,
   rice paddy methane-emission studies, anaerobic digester MAG catalogs,
   methane-rich lakes, peatlands, coalbed methane, and subsurface enrichments.

## Local Codebase Understanding

### What This Repository Is

MethaNet is a Python/Snakemake/HPC-oriented research codebase for connecting
metagenomic molecular evidence to methane-risk intelligence. The repository has
three overlapping layers:

| Layer | Code/artifact locations | Current role |
| --- | --- | --- |
| Transfer-learning package | `src/methanet/`, `workflow/`, `configs/`, `tests/` | Implements feature extraction, embeddings, domain adaptation, risk-tier scaffolding, MBAG graph primitives, and workflow templates. |
| Blue Catalyst ESM2 POC | `results/blue_catalyst_poc/`, `notebooks/`, `scripts/build_blue_catalyst_*` | 662-proteome cross-ecosystem embedding run: 107 wetland/MUCC + 555 rumen PRJEB31266. |
| Functional atlas / MBAG expansion | `scripts/*functional*`, `scripts/build_mag_unit_scope_manifests.py`, `scripts/consolidate_functional_mag_cohort.py`, `ai_docs/functional_metagenomics_expansion/` | Converts per-MAG annotation runs into a Parquet-first functional evidence warehouse for methane/sulfur/substrate/QC candidate interpretation. |

The project metadata in `pyproject.toml` describes the package as
`methanet`, version `1.0.0`, with core dependencies on `numpy`, `pandas`,
`scikit-learn`, and `biopython`. Optional dependency groups cover ML,
bioinformatics annotation, embeddings, domain adaptation, prediction,
Snakemake/DVC/MLflow pipeline orchestration, and ONNX API export.

### Core Package Roles

| Area | Main files | What it does |
| --- | --- | --- |
| MBAG | `src/methanet/mbag/core.py`, `src/methanet/mbag/data.py` | Builds kNN graphs, optimal-transport couplings, leakage audits, reliability weights, and provisional bridge-attestation scoring primitives. |
| Functional features | `src/methanet/functional/quantify.py`, `src/methanet/features.py` | Functional marker and feature extraction scaffolding. |
| Embeddings | `src/methanet/embedding/esm2.py`, `genomeocean.py`, `fusion.py` | Protein/genome embedding generation and fusion. |
| Domain adaptation | `src/methanet/domain_adapt/*`, `workflow/scripts/train_coral.py`, `train_dann.py`, `measure_shift.py` | Transfer-learning and domain-shift controls, not proof of source-independent transfer by themselves. |
| Classification/risk | `src/methanet/classification/*`, `src/methanet/models.py`, `workflow/scripts/train_flux_model.py` | Predictive-model scaffolding and risk-tier vocabulary; final calibrated risk tiers are blocked until sample/flux validation exists. |
| API bridge | `src/api_bridge/*` | Inference/export surface for downstream app/API use. |

### Functional Atlas Scripts That Matter Most

| Script | Role | Strategic importance |
| --- | --- | --- |
| `scripts/build_mag_unit_scope_manifests.py` | Enriches the 662-row manifest with run evidence and classifies each row as `mag_bin`, `assembly_context`, or unresolved. | Prevents assembly-scale rumen records from contaminating MAG-level MBAG. |
| `scripts/consolidate_functional_mag_cohort.py` | Read-only consolidation of per-MAG curated outputs into cohort Parquet tables, preserving failed/partial/non-comparable attempts in status tables. | Makes the functional atlas auditable and queryable without losing failed or quarantined evidence. |
| `scripts/curate_functional_mag_run.py` | Per-MAG curation and Parquet closeout. | Normalizes tool outputs into durable evidence bundles. |
| `scripts/validate_functional_mag_production_gates.py` | Preflight and production gates. | Enforces manifest and DB readiness before launches. |
| `scripts/reports/build_mbag_smoke_report.py` and `build_methanet_atlas_smoke_report.py` | Report/status generation. | Converts operational/functional facts into decision-facing summaries. |

### Cohort Warehouse Model

The intended analytical layer is Parquet-first and regenerable:

```text
results/functional_metagenomics/<cohort_run_id>/cohort_warehouse/
  DATA_ARCHITECTURE_VALIDATION.md
  cohort_table_manifest.tsv
  validation_gates.tsv
  functional_atlas.duckdb
  parquet/<table>/cohort_run_id=<cohort_run_id>/part-00000.parquet
```

The essential invariant is: downstream success never defines the cohort.
Every table must preserve `cohort_run_id`, `run_id`, `proteome_id`, `mag_id`,
and `source_tool` when it belongs to the functional atlas model.

## Current Treated Units

### Full Inventory Source Of Truth

All current treated proteome/MAG/metagenome-context units are identified in:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/
  poc_662_functional_mag_manifest.with_unit_scope.tsv
```

Filtered manifests:

```text
poc_662_functional_mag_manifest.mag_bin_only.tsv       # 625 MAG/bin-comparable units
poc_662_functional_mag_manifest.mag_bin_remaining.tsv  # 518 remaining rumen MAG/bin units
poc_662_functional_mag_manifest.assembly_context.tsv   # 37 quarantined assembly-context units
```

The manifest columns identify each unit by:

```text
proteome_id
sample
source
ecosystem
domain
mag_id
mag_fasta
proteome_faa
source_analysis_accession
analysis_alias
source_filename
n_proteins_used
analysis_unit_type
mbag_mag_level_include
assembly_context_include
claim_scope
comparability_status
recommended_action
```

### Current Denominator

| Source | Ecosystem | Domain in manifest | Count | Interpretation |
| --- | --- | --- | ---: | --- |
| `mucc` | `wetland` | `Unknown` | 107 | Wetland/MUCC MAG/bin-like target-domain units. |
| `rumen` | `rumen` | `Archaea` | 11 | Rumen archaeal MAG/bin units; disproportionately important for methane bridge hypotheses. |
| `rumen` | `rumen` | `Bacteria` | 544 | Rumen bacterial MAG/bin or assembly-context units; useful for broader substrate, sulfur, syntrophy, and source-domain structure. |

By analytical unit:

| `analysis_unit_type` | Count | MBAG MAG-level include? | Claim scope |
| --- | ---: | --- | --- |
| `mag_bin` | 625 | Yes | MAG functional potential |
| `assembly_context` | 37 | No | Assembly/metagenome context |
| `unresolved` | 0 | No | Not applicable |

By run status in the inspected snapshot:

| Scope | Status | Count | Interpretation |
| --- | --- | ---: | --- |
| `mag_bin` | `complete` | 107 | Completed wetland/MUCC calibration/evidence lane. |
| `mag_bin` | `not_started` | 517 | Rumen MAG/bin relaunch backlog. |
| `mag_bin` | `attempt_created` | 1 | Attempt folder exists but not complete. |
| `assembly_context` | `complete` | 19 | Completed but quarantined assembly-context evidence. |
| `assembly_context` | `partial` | 4 | Partial quarantined assembly-context attempts. |
| `assembly_context` | `not_started` | 14 | Not needed for MAG-level relaunch. |

### Why The 662 Units Are Being Treated

| Unit class | Why MethaNet treats it | What it can support | What it cannot support yet |
| --- | --- | --- | --- |
| 662 embedded proteomes | Geometry-aware ESM2 POC backbone; stable denominator for bridge discovery. | Latent cross-ecosystem hypothesis generation, bridge candidate rankings, source/domain diagnostics. | Mechanism proof, sample risk, measured flux, source-independent transfer. |
| 107 wetland/MUCC MAGs/proteomes | Target-domain blue-carbon wetland evidence. | MAG-level functional potential, methane/sulfur/substrate/QC evidence, wetland-side bridge smoke tests. | Project/site risk without sample mapping and abundance/metadata. |
| 518 rumen MAG/bin proteomes | Methane-rich source-domain MAG evidence. | Functional comparison against wetland MAGs, MBAG bridge cards, source-domain methane mechanism priors. | Direct wetland inference without source-aware validation. |
| 37 no-bin rumen assemblies | Large assembly/metagenome records that were embedded only as capped 6,000-protein representations. | Rumen source reservoir context and future community/assembly evidence lane. | MAG-level MBAG, MAG mechanism cards, one-MAG-one-proteome comparisons. |

### The Quarantined 37 Assembly-Context Units

These are all rumen `10676_*_idba` no-bin records. They must be preserved but
excluded from MAG-level MBAG:

| Proteome ID | BioProject-derived accession | Latest status | ESM2 proteins used | Functional predicted proteins when present | Scope ratio when present |
| --- | --- | --- | ---: | ---: | ---: |
| `rumen__10676_0001_idba` | `ERZ1024255` | complete | 6000 | 16226 | 2.70433 |
| `rumen__10676_0002_idba` | `ERZ1024256` | complete | 6000 | 30227 | 5.03783 |
| `rumen__10676_0006_idba` | `ERZ1024258` | complete | 6000 | 85479 | 14.2465 |
| `rumen__10676_0004_idba` | `ERZ1024259` | complete | 6000 | 56946 | 9.491 |
| `rumen__10676_0015_idba` | `ERZ1024260` | complete | 6000 | 49623 | 8.2705 |
| `rumen__10676_0011_idba` | `ERZ1024261` | complete | 6000 | 20810 | 3.46833 |
| `rumen__10676_0010_idba` | `ERZ1024262` | complete | 6000 | 53912 | 8.98533 |
| `rumen__10676_0012_idba` | `ERZ1024263` | complete | 6000 | 31498 | 5.24967 |
| `rumen__10676_0018_idba` | `ERZ1024264` | complete | 6000 | 210885 | 35.1475 |
| `rumen__10676_0017_idba` | `ERZ1024265` | complete | 6000 | 198562 | 33.0937 |
| `rumen__10676_0022_idba` | `ERZ1024266` | complete | 6000 | 11646 | 1.941 |
| `rumen__10676_0021_idba` | `ERZ1024267` | complete | 6000 | 42121 | 7.02017 |
| `rumen__10676_0025_idba` | `ERZ1024268` | complete | 6000 | 113690 | 18.9483 |
| `rumen__10676_0028_idba` | `ERZ1024269` | complete | 6000 | 122628 | 20.438 |
| `rumen__10676_0020_idba` | `ERZ1024270` | complete | 6000 | 123515 | 20.5858 |
| `rumen__10676_0029_idba` | `ERZ1024271` | complete | 6000 | 168413 | 28.0688 |
| `rumen__10676_0035_idba` | `ERZ1024272` | complete | 6000 | 67331 | 11.2218 |
| `rumen__10676_0034_idba` | `ERZ1024273` | complete | 6000 | 64225 | 10.7042 |
| `rumen__10676_0030_idba` | `ERZ1024274` | complete | 6000 | 87234 | 14.539 |
| `rumen__10676_0036_idba` | `ERZ1024275` | partial | 6000 | 175786 | 29.2977 |
| `rumen__10676_0032_idba` | `ERZ1024276` | partial | 6000 | 295582 | 49.2637 |
| `rumen__10676_0026_idba` | `ERZ1024277` | partial | 6000 | 395924 | 65.9873 |
| `rumen__10676_0033_idba` | `ERZ1024278` | partial | 6000 | 369748 | 61.6247 |
| `rumen__10676_0027_idba` | `ERZ1024280` | not_started | 6000 | missing | missing |
| `rumen__10676_0048_idba` | `ERZ1024281` | not_started | 6000 | missing | missing |
| `rumen__10676_0046_idba` | `ERZ1024282` | not_started | 6000 | missing | missing |
| `rumen__10676_0049_idba` | `ERZ1024284` | not_started | 6000 | missing | missing |
| `rumen__10676_0042_idba` | `ERZ1024285` | not_started | 6000 | missing | missing |
| `rumen__10676_0043_idba` | `ERZ1024286` | not_started | 6000 | missing | missing |
| `rumen__10676_0051_idba` | `ERZ1024287` | not_started | 6000 | missing | missing |
| `rumen__10676_0039_idba` | `ERZ1024290` | not_started | 6000 | missing | missing |
| `rumen__10676_0050_idba` | `ERZ1024291` | not_started | 6000 | missing | missing |
| `rumen__10676_0047_idba` | `ERZ1024292` | not_started | 6000 | missing | missing |
| `rumen__10676_0023_idba` | `ERZ1024293` | not_started | 6000 | missing | missing |
| `rumen__10676_0009_idba` | `ERZ1024294` | not_started | 6000 | missing | missing |
| `rumen__10676_0024_idba` | `ERZ1024295` | not_started | 6000 | missing | missing |
| `rumen__10676_0014_idba` | `ERZ1024302` | not_started | 6000 | missing | missing |

The observed scope ratios range up to 65.9873, proving that several no-bin
records are many times larger than the 6,000-protein ESM2 representation used in
the POC. This is the core reason they are blocked from MAG-level claims.

### MAG-Bin Examples And Full List Location

The full 625-row MAG/bin list is:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/
  poc_662_functional_mag_manifest.mag_bin_only.tsv
```

The relaunch backlog is:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/
  poc_662_functional_mag_manifest.mag_bin_remaining.tsv
```

Representative remaining rumen MAG/bin units include:

```text
rumen__10674_0001_idba_bin.10
rumen__10674_0001_idba_bin.100
rumen__10674_0001_idba_bin.102
rumen__10674_0001_idba_bin.11
rumen__10674_0001_idba_bin.20
rumen__10674_0001_idba_bin.23
rumen__10674_0001_idba_bin.24
rumen__10674_0001_idba_bin.26
rumen__10674_0001_idba_bin.37
rumen__10674_0001_idba_bin.4
rumen__10674_0001_idba_bin.45
rumen__10674_0001_idba_bin.49
rumen__10674_0001_idba_bin.52
rumen__10674_0001_idba_bin.54
rumen__10674_0001_idba_bin.56
rumen__10674_0001_idba_bin.6
rumen__10674_0001_idba_bin.66
rumen__10674_0001_idba_bin.72
rumen__10674_0001_idba_bin.73
rumen__10674_0001_idba_bin.75
```

Representative completed MUCC/wetland units include:

```text
mucc__2162886008_15
mucc__3300001784_29
mucc__3300004775_9
mucc__3300005325_23
mucc__3300005326_54
mucc__3300009053_19
mucc__3300009081_10
mucc__3300009153_19
mucc__3300010293_30
mucc__3300013126_82
mucc__3300013126_97
mucc__3300014199_19
mucc__3300014205_53
mucc__3300016725_15
mucc__3300017643_20
mucc__3300017947_8
mucc__3300017959_16
mucc__3300017975_11
mucc__3300018012_6
mucc__3300018019_17
```

## Current Bridge-Candidate Meaning

The ESM2 POC bridge artifact lives at:

```text
results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/
  artifacts/bridge_top_candidates.tsv
```

The top observed bridge candidates include rumen archaeal MAG/bin units such as:

```text
rumen__10674_0004_idba_bin.23
rumen__10674_0002_idba_bin.8
rumen__10674_0004_idba_bin.79
rumen__10674_0001_idba_bin.23
rumen__10674_0005_idba_bin.53
rumen__10674_0008_idba_bin.67
rumen__10674_0006_idba_bin.56
rumen__10674_0005_idba_bin.72
```

and at least one wetland-side candidate:

```text
mucc__GCA_002495465.1_ASM249546v1_genomic
```

The current interpretation is:

| Evidence | Meaning now | Required upgrade |
| --- | --- | --- |
| High bridge score / opposite-ecosystem neighbors | Latent proteome similarity hypothesis. | Functional evidence, QC, taxonomy, source-aware controls, and sample validation. |
| Rumen Archaea among top candidates | Biologically plausible because methanogenesis machinery is conserved. | Direct marker/pathway support and non-confounded wetland comparison sources. |
| Wetland bridge candidate complete | Can be a smoke-test candidate card. | Pair with abundance/sample/environmental evidence before sample-level use. |

## External Web Search: Sources To Enrich The Atlas

Searches were run on 2026-06-14 with queries focused on mangrove sediments,
MAGs, methane-rich environments, NCBI BioSample/BioProject anchors, and
genome-resolved datasets.

### Priority 1: Mangrove MAG Catalog, Southeast China

Source:
[A holistic genome dataset of bacteria and archaea of mangrove sediments](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf081/8232623),
GigaScience 2025; supporting dataset DOI
[10.5524/102702](https://doi.org/10.5524/102702).

Why it matters:

- The paper reports 966 metagenome-assembled genomes from mangrove sediments.
- Samples span six groups of samples and multiple southeastern China mangrove
  reserves from 2014-2020.
- The authors explicitly frame the catalog around microbial diversity and
  biogeochemical functions including CH4, N, and S cycling.
- This is probably the strongest near-term atlas expansion target because it is
  both mangrove-specific and MAG-resolved.

Strategic use:

| Use | MethaNet value |
| --- | --- |
| Add new target-domain MAGs | Break the current wetland = MUCC source confounding. |
| Mine archaeal and sulfur/substrate guilds | Improve bridge candidates beyond Methanoregula-like wetland examples. |
| Add China multi-reserve geography | Support site/source-aware validation and source-deconfounding. |
| Compare to existing MUCC MAGs | Identify shared/novel wetland methane/sulfur/substrate mechanisms. |

Acquisition plan:

1. Download supporting data from GigaDB DOI `10.5524/102702`.
2. Extract MAG FASTA, predicted proteins if supplied, CheckM/CheckM2 quality,
   taxonomy, sample/reserve/depth/habitat metadata.
3. Build a `source_project_id = MSM_China_2014_2020` data-source layer.
4. Assign BioSample or sample IDs where available; if BioSamples are absent from
   the data package, preserve source sample IDs and mark metadata resolution as
   `publication_or_repository_sample`.
5. Run the same unit-scope gate before ESM2/function integration.

Verified ingestion update, 2026-06-15:

- Dedicated handoff:
  [msm_china_2025_ingestion_status.md](msm_china_2025_ingestion_status.md).
- Local staging package:
  `data/external/msm_china_2025/`.
- The article/repository trail strongly supports direct MAG/proteome ingestion:
  the MAGs are deposited in eLMSG under accession range
  `LMSG_G000027425.1-LMSG_G000028852.1`, and eLMSG documents genome bundle
  downloads containing `.fna`, `.faa`, `.gff`, and annotation sidecars.
- A candidate range manifest has been created with 1,428 accession candidates:
  `data/external/msm_china_2025/manifests/elmsg_accession_range_candidates.tsv`.
- A record-aware candidate manifest has also been created:
  `data/external/msm_china_2025/manifests/elmsg_accession_record_candidates.tsv`.
  The inferred `MSG099710-MSG101137` record range is based on an indexed eLMSG
  AMD page where `LMSG_G000004334.1` maps to `/elmsg/record/MSG076619` and the
  eLMSG record IDs increment with the genome accession numeric suffix.
- DataCite exposed additional NCBI BioProject anchors beyond the article's
  umbrella `PRJNA1150796`: `PRJNA1136686`, `PRJNA1159532`, `PRJNA1268148`,
  and `PRJNA1268163`. These yielded 71 SRA run rows and 38 exact BioSample
  records in the local environmental metadata table:
  `data/external/msm_china_2025/metadata/ncbi_biosample_environmental_metadata.tsv`.
  All 71 rows currently carry collection date, depth, coordinates, and
  environmental context fields, making them valuable sample-readiness metadata
  even though they are not MAG/proteome payloads.
- No actual `.fna` or `.faa` payload has been confirmed locally yet. Direct
  programmatic GigaDB/eLMSG requests from this runtime returned empty replies,
  redirect loops, or BMDC maintenance HTML shells.
- Therefore the current state is `source_verified` and
  `payload_route_blocked`, not `atlas_ready`.
- Do not claim that MSM has added 966 MAGs to MethaNet until the repository
  package is fetched, the 966 published final MAG denominator is reconciled
  against the 1,428-accession numeric range, and both genome/proteome FASTAs pass
  payload validation.
- A reusable helper now exists at
  `scripts/external/fetch_msm_china_2025_elmsg.py` to retry eLMSG accession
  and record-based bundle downloads, reject HTML shells, extract `.fna`/`.faa`,
  and write a functional/embedding readiness manifest.
- A second helper now exists at
  `scripts/external/fetch_msm_china_2025_biosamples.py` to refresh the
  NCBI-linked SRA/BioSample environmental metadata layer.

Resolved payload update, 2026-06-15:

- GigaDB's public Wasabi object route solved the payload problem:
  `https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/102001_103000/102702/`.
- `MAG_file.zip` was downloaded from that route, matched the official GigaDB
  MD5 `c69a96c13d84ae0fe1a52005bcb644cd`, and extracted to 1,428 MAG FASTA
  files.
- The local normalized MAG FASTAs live under
  `data/external/msm_china_2025/genomes_fna/`.
- Prodigal 2.6.3 from the existing `methanet-fgx` environment was run in
  metagenomic mode (`-p meta`) to generate 1,428 protein FASTAs under
  `data/external/msm_china_2025/proteomes_faa/`, plus matching `.ffn` and
  `.gff` files.
- The final handoff table for functional annotation and ESM2 embedding is
  `data/external/msm_china_2025/manifests/msm_china_2025_functional_embedding_manifest.tsv`.
  It contains 1,428 `fna_and_faa_ready` rows with local FNA/FAA paths,
  Prodigal protein counts, GTDB taxonomy, source sample IDs, and BioSample
  mappings.
- Current interpretation: the MSM package is now technically ready for
  MethaNet MAG-level functional annotation and ESM2 embedding without
  reassembly. Scientific interpretation still requires QC and denominator
  reconciliation because the downloadable archive contains 1,428 FASTAs while
  the publication reports 966 medium/high-quality MAGs.

QC and annotation launch update, 2026-06-15:

- A MethaNet-compatible functional-run manifest was generated at
  `results/functional_metagenomics/msm_china_2025_20260615/manifests/msm_china_2025_functional_mag_manifest.tsv`.
- The 966-vs-1,428 reconciliation package was generated under
  `results/functional_metagenomics/msm_china_2025_20260615/qc_reconciliation/`.
  It preserves the published 966-MAG denominator as a claim-boundary gate and
  treats all 1,428 archive FASTAs as local candidates pending CheckM2/GUNC
  quality evidence.
- The existing MethaNet production gate validator passed on the full 1,428-row
  manifest and on two Slurm-safe tranche manifests with no missing FNA/FAA
  paths and no assembly-context rows.
- Deep functional annotation was submitted as Slurm jobs `8797` and `8798`
  over two validated tranches: 1,000 MAGs and 428 MAGs, respectively. The run
  uses the standard MethaNet stack: Prodigal, KOfam, MCycDB, SCycDB, dbCAN,
  Bakta, CheckM2, GUNC, GTDB-Tk, and METABOLIC-G.
- The launch note is
  `ai_docs/functional_metagenomics_expansion/dataset_expansion_strategy_20260614/msm_china_2025_qc_annotation_launch_20260615.md`.

### Priority 2: Tieshan Bay Natural And Restored Mangrove Sediment Archaea

Source:
[NCBI BioProject PRJNA1072265](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1072265/).

NCBI describes this as "Natural and artificially restored mangrove sediment
archaea in Tieshan Bay, China" with 48 SRA experiments and 48 BioSamples.

BioSample anchors found through NCBI linked BioSample pages:

```text
SAMN39740054
SAMN39740055
SAMN39740056
SAMN39740057
SAMN39740058
SAMN39740059
SAMN39740060
SAMN39740061
SAMN39740062
SAMN39740063
SAMN39740064
SAMN39740065
SAMN39740066
SAMN39740067
SAMN39740068
SAMN39740069
SAMN39740070
SAMN39740071
SAMN39740072
SAMN39740073
SAMN39740074
SAMN39740075
SAMN39740076
SAMN39740077
SAMN39740078
SAMN39740079
SAMN39740080
SAMN39740081
SAMN39740082
SAMN39740083
SAMN39740084
SAMN39740085
SAMN39740086
SAMN39740087
SAMN39740088
SAMN39740089
SAMN39740090
SAMN39740091
SAMN39740092
SAMN39740093
SAMN39740094
SAMN39740095
SAMN39740096
SAMN39740097
SAMN39740098
SAMN39740099
SAMN39740100
SAMN39740101
```

First-page sample names observed from NCBI include `SPTR043`, `SPTR042`,
`SPTR041`, `SPTR033`, `SPTR032`, `SPTR031`, `SPTR023`, `SPTR022`,
`SPTR021`, `SPTR013`, `SPTR012`, `SPTR011`, and corresponding `SPRG*`
records.

Why it matters:

- Archaea-focused mangrove sediment data are directly relevant to methane
  production and anaerobic methane oxidation hypotheses.
- Natural versus restored contrast is strategically important for blue-carbon
  MRV because restoration status may modify methane risk.
- BioSample anchoring is strong enough to support a sample-readiness layer.

Strategic use:

| Use | MethaNet value |
| --- | --- |
| Restoration contrast | Begin separating natural/restored site effects from ecosystem effects. |
| Archaea focus | Enrich methanogen and ANME-like candidate search space. |
| Sample-level accession anchors | Improves future abundance/environmental joins. |

Acquisition plan:

1. Pull SRA runs linked to each BioSample.
2. Decide whether to assemble/bin internally or locate existing MAGs if a paper
   or repository supplies them.
3. Build sample metadata with restoration status, site, date, sediment layer,
   and environmental package fields.
4. Mark as `raw_metagenome_pending_mag_reconstruction` until MAGs exist.

### Priority 3: Methane-Rich Deep Coastal Sediments, Hangzhou Bay

Sources:

- [Microbial communities and metagenomes in methane-rich deep coastal sediments](https://www.nature.com/articles/s41597-024-03889-7)
- NCBI BioProject: [PRJNA1139943](https://identifiers.org/ncbi/bioproject:PRJNA1139943)

The article reference section reports two metagenomes and 27
metagenome-assembled genomes from a methane-rich coastal sediment core, plus
related SRA and Figshare geochemical resources.

BioSample anchors found:

```text
SAMN42955229
SAMN42955230
SAMN42955231
SAMN42955232
SAMN42955233
SAMN42955234
SAMN42955235
SAMN42955236
SAMN42955237
SAMN42955238
SAMN42955239
SAMN42955240
SAMN42955241
SAMN42955242
SAMN42955243
SAMN42955244
SAMN42955245
SAMN42955246
SAMN42955247
SAMN42955248
SAMN42955249
SAMN42955250
SAMN42955251
SAMN42955252
SAMN42955253
SAMN42955254
SAMN42955255
SAMN42797935
SAMN42797936
```

Interpretation:

- `SAMN42955229`-`SAMN42955255` appear to be the 27 MAG BioSamples.
- `SAMN42797935` and `SAMN42797936` likely correspond to the two metagenome
  sample records and should be verified before ingestion.

Why it matters:

- This is not mangrove-specific, but it is coastal, sedimentary, methane-rich,
  MAG-resolved, and paired with chemical indicators.
- It can become a high-value methane-positive coastal control lane.
- It should be especially useful for methane/sulfur/redox feature calibration
  and for distinguishing wetland blue-carbon candidates from generic coastal
  methane-rich sediment organisms.

Acquisition plan:

1. Pull GenBank assemblies for PRJNA1139943.
2. Split MAG BioSamples from raw metagenome BioSamples in metadata.
3. Load Figshare chemical indicators as `fact_environmental_measurement` with
   source-level caveats.
4. Run GTDB-Tk/CheckM2/GUNC/KOfam/MCycDB/SCycDB/dbCAN/METABOLIC parity with
   the current atlas.

### Priority 4: Futian Reserve Seven-Year Mangrove/Mudflat Catalog

Source:
[A seven-year metagenomic genome catalogue of mangrove and mudflat sediments from the Futian Reserve, China](https://www.nature.com/articles/s41597-026-07291-3).

Why it matters:

- The paper reports a long-term 2017-2023 sediment sampling frame from paired
  mangrove forest and adjacent mudflat habitats.
- It includes 65 sediment samples across multiple depths.
- It is unusually valuable for MRV because time, depth, and paired habitat
  structure are the exact axes MethaNet needs for sample risk readiness.

Strategic use:

| Use | MethaNet value |
| --- | --- |
| Temporal replication | Helps move beyond one-off MAG catalogs. |
| Mangrove vs mudflat pairing | Lets MethaNet compare blue-carbon vegetation context and sediment state. |
| Depth structure | Supports methane-permissiveness modeling because anoxia, sulfate, and substrate conditions vary with depth. |

Acquisition plan:

1. Locate accession table and supporting data from the article/data-availability
   section.
2. Prioritize MAGs with sample, year, depth, habitat, and environmental metadata.
3. Mark the source as target-domain but separate `mangrove_forest` and `mudflat`
   habitat classes.

### Priority 5: 2,965 Microbial Genomes From Mangrove Sediments

Source:
[Reconstruction of 2,965 Microbial Genomes from Mangrove Sediments](https://www.nature.com/articles/s41597-025-06438-y).

Why it matters:

- Large MAG count.
- Mangrove-specific.
- Potentially complementary to the 966-MAG MSM catalog and Futian long-term
  catalog.

Acquisition plan:

1. Verify whether the 2,965 genomes overlap with the MSM/GigaScience catalog.
2. Build a dereplication layer before adding all genomes to MethaNet.
3. Treat this as a target-domain diversity expansion, not a flux-validation
   source unless environmental/flux metadata exist.

### Priority 6: Brazilian Mangrove Metagenomes

Source:
[The Microbiome of Brazilian Mangrove Sediments as Revealed by Metagenomics](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0038600).

Why it matters:

- It is not MAG-resolved in the modern sense, but it adds geography outside
  China and outside the current MUCC source.
- The study explicitly reports methane, nitrogen, and sulfur pathway signals in
  mangrove sediments.
- It can be used as a historical community/metagenome context lane and as a
  source-discovery seed for newer Brazilian mangrove sequencing.

Strategic use:

| Use | MethaNet value |
| --- | --- |
| Non-China mangrove context | Helps avoid a China-only target-domain expansion. |
| Methane/sulfur functional signal | Useful for marker-panel validation and literature framing. |
| Search seed | Names sites, labs, and sample labels for follow-up SRA/MG-RAST discovery. |

Claim boundary:

This should not be treated as MAG-level evidence unless the raw reads are
reassembled/binned or a later MAG resource is found.

## Other Methane-Producing Or Methane-Cycling Environment Targets

These are not direct mangrove target-domain replacements. They are valuable as
controls, priors, or mechanistic enrichment lanes.

| Environment | Example source found | Why useful | Ingestion priority |
| --- | --- | --- | --- |
| Zoige wetland / Tibetan Plateau | [Two MAGs of hydrogen-dependent methanogens from Zoige wetland](https://journals.asm.org/doi/10.1128/mra.00021-21) | Wetland methanogens with explicit methanogenic metabolism. | High as methane-positive wetland control. |
| Rice paddies | [Coexistence patterns of soil methanogens tied to methane emissions](https://pmc.ncbi.nlm.nih.gov/articles/PMC7825242/) | Paired methane-emission gradient; useful for abundance/interaction features. | Medium-high if raw reads/MAGs and flux metadata are accessible. |
| Anaerobic digesters | [1,600-MAG anaerobic digestion microbiome catalog](https://www.biorxiv.org/content/10.1101/680553v1.full-text), [GigaDB anaerobic digester MAG dataset](https://gigadb.org/dataset/100842) | Strong methane-production positive controls and syntrophy/substrate pathways. | Medium; not ecological target-domain but excellent marker control. |
| Methane-rich lake chemoclines | [Echo Lake MAGs](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812302/) | Methane oxidation and sulfur cycling near steep methane/oxygen gradients. | Medium for oxidation/sulfur sink controls. |
| Peatland/permafrost | [Stordalen Mire MAG resource](https://emerge-db.asc.ohio-state.edu/datasources/162) | Classic wetland methane environment with thaw gradients. | Medium-high if accessions become stable. |
| Coalbed methane | NETL coalbed methane metagenome search hits | Methanogenic hydrocarbon/subsurface systems. | Medium-low for blue carbon, high for methanogenesis breadth. |
| Deep biosphere enrichments | Methanogenic MAG from Eger Rift subsurface enrichment search hit | Clear methanogenesis genes, but enrichment culture/source mismatch. | Low-medium; useful as mechanistic edge case. |

## Recommended Expansion Data Model

Add external sources through a separate staging contract before merging into the
current 662 atlas:

```text
external_dataset_id
source_project_id
source_project_url
source_publication_url
biosample_accession
bioproject_accession
sra_run_accession
assembly_accession
sample_name
sample_or_mag
habitat_class
environment_material
geo_location
collection_date
depth
restoration_status
mag_id
proteome_id
faa_path
fna_path
completeness
contamination
taxonomy
metadata_resolution
metadata_caveat
ingestion_status
recommended_methanet_lane
```

Recommended lanes:

| Lane | Meaning |
| --- | --- |
| `target_mangrove_mag` | Mangrove MAG with acceptable MAG FASTA/proteome and sample metadata. |
| `target_mangrove_raw_metagenome` | BioSample/SRA record requiring assembly/binning. |
| `target_coastal_methane_positive_control` | Coastal sediment methane-rich MAG/metagenome outside mangrove. |
| `source_methane_positive_control` | Rumen/digester/rice/peat/subsurface methane-rich source control. |
| `oxidation_sink_control` | Methane oxidation or ANME-rich sample/MAG lane. |
| `literature_context_only` | Useful paper/context but no usable sequence unit yet. |

## Prioritized Work Plan

### Phase 1: Source Acquisition Register

Create:

```text
results/functional_metagenomics/external_dataset_discovery_20260614/
  external_source_register.tsv
  external_biosample_candidates.tsv
  external_ingestion_decisions.md
```

Minimum rows:

| Rank | Source | Immediate action |
| ---: | --- | --- |
| 1 | MSM 966-MAG mangrove catalog | Download GigaDB package and parse MAG/sample metadata. |
| 2 | PRJNA1072265 Tieshan Bay | Pull BioSample/SRA metadata for all 48 samples; determine if MAG reconstruction is needed. |
| 3 | PRJNA1139943 methane-rich coastal sediment | Pull GenBank assemblies and separate 27 MAGs from two metagenomes. |
| 4 | Futian seven-year catalog | Locate data availability and accession table; preserve year/depth/habitat. |
| 5 | 2,965 mangrove MAG catalog | Check overlap/dereplicate against MSM and Futian. |

### Phase 2: External Unit-Scope Gate

Before running ESM2 or functional annotation, classify every external row as:

```text
mag_bin
raw_metagenome
assembly_context
metagenome_context
unresolved
```

Never infer that a BioSample is one MAG. BioSample is the sample accession; MAGs
and proteomes require separate genome/protein identifiers.

### Phase 3: MAG-Level Feature Parity

For each MAG/bin lane, run the same feature contracts as the current atlas:

```text
CheckM2
GUNC
GTDB-Tk
Bakta/Prodigal
KOfam
MCycDB
SCycDB
dbCAN
METABOLIC
CAZy
MEROPS
annotation coverage
feature_methane_mechanism
feature_sulfur_competition
feature_mrv_mag_level
```

### Phase 4: Source-Deconfounding Design

Target minimum design before stronger transfer claims:

| Ecosystem / source class | Minimum source count | Why |
| --- | ---: | --- |
| Rumen | 2+ independent source projects | Current rumen = PRJEB31266 is source-confounded. |
| Mangrove/wetland | 3+ independent source projects | Needed to separate wetland signal from MUCC/MSM/Futian/source effects. |
| Methane-positive coastal controls | 1-2 | Anchor methane-rich coastal functional states. |
| Methane sink/oxidation controls | 1-2 | Avoid production-only bias and support net-risk interpretation. |

### Phase 5: Sample Risk Readiness Layer

Once BioSamples and MAGs are linked:

```text
dim_external_project
dim_external_site
dim_external_sample
link_sample_mag
fact_external_mag_qc
fact_external_environmental_metadata
fact_external_abundance_or_coverage
feature_sample_molecular_capacity
sample_risk_readiness_table
```

Readiness states:

```text
scoreable
monitor_more
needs_metadata
needs_abundance
needs_flux_validation
blocked_noncomparable
```

## Strategic Claim Boundary Matrix

| Claim | Allowed wording now | Evidence status | Blocking gap | Next validation action |
| --- | --- | --- | --- | --- |
| Current 662 cohort is cleanly identified | "The current ESM2 POC backbone has 662 identified proteome units with local MAG/proteome matching." | Supported by local manifests. | None for identity; live status can still move. | Preserve `proteome_id` and left joins. |
| Current MAG-level denominator is 625 | "After unit-scope classification, 625 units are MAG/bin-comparable for MAG-level functional atlas work." | Supported by unit-scope manifest. | Rumen MAG functional completion is still pending for most units. | Complete/relaunch remaining 518 rumen MAG/bin units. |
| 37 rumen no-bin records are not MAG-comparable | "The 37 no-bin `10676_*_idba` records are assembly/metagenome context, not MAG-level MBAG evidence." | Strongly supported by scope ratios and input size. | None; must maintain quarantine. | Preserve in status/assembly-context lane only. |
| Expanded mangrove atlas can deconfound target domain | "Additional mangrove MAG sources can reduce current target-domain source confounding." | Directional until ingested and dereplicated. | Need source metadata, BioSamples, MAG FASTAs, QC. | Ingest MSM, PRJNA1072265, Futian, and 2,965-MAG catalogs. |
| MethaNet can assign final A-E risk tiers | Not allowed. Use: "A-E tiers are target product vocabulary pending calibrated sample/project evidence." | Blocked. | Missing abundance, environmental covariates, repeated observations, flux/process validation. | Build sample risk readiness and validation datasets. |

## Web Search Source List

Primary sources used or triaged:

- [A holistic genome dataset of bacteria and archaea of mangrove sediments](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf081/8232623)
- [Supporting data DOI 10.5524/102702](https://doi.org/10.5524/102702)
- [NCBI BioProject PRJNA1072265, Mangrove sediment archaea](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1072265/)
- [Microbial communities and metagenomes in methane-rich deep coastal sediments](https://www.nature.com/articles/s41597-024-03889-7)
- [NCBI BioProject PRJNA1139943](https://identifiers.org/ncbi/bioproject:PRJNA1139943)
- [A seven-year metagenomic genome catalogue of mangrove and mudflat sediments from the Futian Reserve, China](https://www.nature.com/articles/s41597-026-07291-3)
- [Reconstruction of 2,965 Microbial Genomes from Mangrove Sediments](https://www.nature.com/articles/s41597-025-06438-y)
- [The Microbiome of Brazilian Mangrove Sediments as Revealed by Metagenomics](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0038600)
- [Two MAGs of hydrogen-dependent methanogens from Zoige wetland](https://journals.asm.org/doi/10.1128/mra.00021-21)
- [Coexistence patterns of soil methanogens tied to methane emissions in rice paddies](https://pmc.ncbi.nlm.nih.gov/articles/PMC7825242/)
- [Anaerobic digestion microbiome 1,600-MAG catalog](https://www.biorxiv.org/content/10.1101/680553v1.full-text)
- [GigaDB anaerobic digestion MAG dataset](https://gigadb.org/dataset/100842)
- [Stordalen Mire MAG resource](https://emerge-db.asc.ohio-state.edu/datasources/162)

## Final Recommendation

Do not expand the atlas by simply appending every available metagenome. Expand
it as a source-aware evidence system:

1. Preserve the current 625 MAG/bin versus 37 assembly-context split.
2. Finish the 518 remaining rumen MAG/bin relaunch so MBAG has a complete
   source-domain functional layer.
3. In parallel, ingest external mangrove datasets through a strict
   BioSample/MAG/proteome staging table.
4. Prioritize MSM 966 MAGs, PRJNA1072265, PRJNA1139943, Futian, and the 2,965
   mangrove MAG catalog because they directly attack the current source
   confounding problem.
5. Use rice paddy, Zoige wetland, anaerobic digester, methane-rich lake,
   peatland, and subsurface datasets as controls and mechanism priors, not as
   blue-carbon substitutes.
6. Build a sample risk readiness layer before assigning any sample/project
   methane-risk tier.

The strategic goal is an atlas that can say:

> "This MAG or sample has methane-relevant molecular potential, supported by
> specific markers, QC, taxonomy, source context, and metadata resolution, and
> this is the exact validation evidence needed before stronger MRV claims are
> allowed."

That is the path from the current strong research artifact to a defensible
MethaNet molecular intelligence product.
