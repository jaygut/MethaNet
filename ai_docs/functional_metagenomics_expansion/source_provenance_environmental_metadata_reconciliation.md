# Source Provenance And Environmental Metadata Reconciliation

Date: 2026-06-18

Scope: source-of-truth provenance and environmental metadata strategy for the current MethaNet rumen, wetland/MUCC, and mangrove MAG/proteome lanes.

This document is a documentation and reconciliation layer. It does not replace production outputs under `results/`; it records what can be claimed from the current local evidence, what remains estimate-tier context, and which free APIs can be used to enrich the metadata deterministically.

## Executive Summary

The three current source domains are traceable to defensible papers or data objects:

| MethaNet lane | Primary citation | Primary data object | Local resolution now | Claim status |
| --- | --- | --- | --- | --- |
| Rumen POC MAGs | Stewart et al. 2019, *Nature Biotechnology*, `10.1038/s41587-019-0202-3` | ENA `PRJEB31266`; protein resource `10.7488/ds/2470` | 555/555 exact `ERZ...` analysis-accession matches | Conclusive MAG/proteome provenance; environmental context is mostly cohort-level cattle rumen |
| Wetland/MUCC POC MAGs | Bechtold et al. 2025, *Nature Communications*, `10.1038/s41467-025-56133-0` | MUCC v2.0.0 Zenodo `10.5281/zenodo.14532347` | 107 Methanoregula MAG/proteome units: 20 exact NCBI assembly/BioSample, 23 OWC bin plus site/project, 64 source-bucket rows | Conclusive paper/dataset provenance; mixed per-MAG sample metadata resolution |
| Mangrove/MSM MAGs | Pan et al. 2025, *GigaScience*, `10.1093/gigascience/giaf081` | GigaDB `10.5524/102702`; NCBI `PRJNA1150796`; NODE/eLMSG accessions | 82 local sediment-sample metadata rows, 71 exact BioSample rows, 1428 local MAG candidates | Conclusive source provenance; per-MAG sample mapping and 966-vs-1428 denominator reconciliation still required |

The practical metadata rule is:

> Use exact accession metadata where present, use paper/site/project metadata as context, and label modeled environmental covariates as estimates. Do not let estimated covariates masquerade as measured sample metadata.

## Local Sources Checked

Authoritative local files:

- `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv`
- `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/README.md`
- `results/functional_metagenomics/environmental_metadata_recovery_20260612/METADATA_RECOVERY_REPORT.md`
- `results/functional_metagenomics/environmental_metadata_recovery_20260612/cohort_662_environmental_metadata_crosswalk.tsv`
- `results/functional_metagenomics/environmental_metadata_recovery_20260612/rumen_proteome_environmental_metadata.tsv`
- `results/functional_metagenomics/environmental_metadata_recovery_20260612/mucc_proteome_environmental_metadata.tsv`
- `results/functional_metagenomics/environmental_metadata_recovery_20260612/source_bioproject_summaries.tsv`
- `results/functional_metagenomics/environmental_metadata_recovery_20260612/mucc_source_bucket_context.tsv`
- `data/external/msm_china_2025/metadata/source_register.tsv`
- `data/external/msm_china_2025/metadata/ncbi_biosample_environmental_metadata.tsv`
- `data/external/msm_china_2025/gigadb_wasabi/metadata_sediment_samples.txt`

Important local caveat: `data/external/msm_china_2025/gigadb_wasabi/Supplementary_Tables.xlsx` is not currently a valid XLSX file. It contains an XML `AccessDenied` response and must not be treated as evidence until reacquired.

## Verified Origins

### Rumen

Primary paper:

- Stewart, R. D. et al. "Compendium of 4,941 rumen metagenome-assembled genomes for rumen microbiome biology and enzyme discovery." *Nature Biotechnology* 37, 953-961 (2019). DOI: `10.1038/s41587-019-0202-3`.

External verification:

- The paper reports the rumen MAG resource and states that raw reads and assemblies are deposited under ENA `PRJEB31266`.
- ENA Portal API query for `PRJEB31266` returns fields including `analysis_accession`, `sample_accession`, `scientific_name`, `country`, `collection_date`, `host`, `environment_feature`, `environment_material`, and `submitted_ftp`.

Local reconciliation:

- Current POC uses 555 rumen MAG/proteome units.
- All 555 have exact `source_analysis_accession` values that join to ENA `analysis_accession`.
- Local recovered fields support:
  - `source_domain`: rumen
  - `ecosystem`: rumen / cattle digestive system
  - `country`: United Kingdom
  - `environment_feature`: stomach
  - `environment_material`: bodily fluid
  - `collection_date`: not collected
  - `host_context`: cattle / Bos taurus at project-paper level, but host is not consistently populated for every per-analysis row.

Interpretation boundary:

- This is exact MAG/proteome provenance.
- It is not exact animal-level metadata for every MAG.
- The rumen lane is useful as a data-rich methane-domain reference, not as blue-carbon sample context.

### Wetland / MUCC

Primary paper:

- Bechtold, E. K. et al. "Metabolic interactions underpinning high methane fluxes across terrestrial freshwater wetlands." *Nature Communications* 16, 944 (2025). DOI: `10.1038/s41467-025-56133-0`.

Primary data object:

- MUCC v2.0.0 Zenodo dataset, DOI `10.5281/zenodo.14532347`.

External verification:

- The paper describes the MUCC v2.0.0 database across nine terrestrial freshwater wetlands and reports a genomic analysis of 107 dereplicated MUCC-derived and public GTDB/JGI Methanoregula MAGs.
- The Zenodo record exposes the relevant Methanoregula files:
  - `Methanoregula_MAGs_DB.zip`
  - `Methanoregula_MAGs_list.txt`
  - `Methanoregula_metabolism_summary.xlsx`
  - `Methanoregula_physiology.txt`
  - `classification_w_outgroup.txt`

Local reconciliation:

Current POC wetland/MUCC set:

| Bucket | Rows | Current interpretation |
| --- | ---: | --- |
| JGI | 56 | Exact source bucket from Methanoregula list; compact public files do not expose per-MAG sample metadata |
| OWC | 23 | Old Woman Creek; exact bin match plus site/project context |
| GTDB | 20 | GTDB/NCBI reference MAGs; exact NCBI assembly/BioSample when resolvable |
| STM | 7 | Stordalen Mire; source/site/project-level context |
| PPR | 1 | Prairie Pothole Region; source/site/project-level context |

Resolution tiers:

| Resolution | Rows | Meaning |
| --- | ---: | --- |
| `exact_ncbi_assembly_biosample` | 20 | Exact assembly/BioSample metadata is available for GTDB/NCBI-derived records |
| `exact_owc_bin_plus_site_project` | 23 | Bin identity is exact for OWC; environmental context is site/project/design level |
| `exact_mucc_source_bucket` | 64 | MAG source is exact within MUCC/public Methanoregula files, but per-sample metadata is not exposed in compact local files |

Paper-level environmental context:

- Nine freshwater wetlands were used: OWC, PPR P7, PPR P8, LA2, TWI, JLA, STM-fen, STM-bog, and SPRUCE.
- Wetland types include marsh, swamp, fen, and bog.
- Seven sites are in the United States and two are in northern Sweden.
- The paper integrates 16S, metagenomes, metatranscriptomes, and annual methane flux data.
- Methanoregula dominance and activity are explicitly interpreted in relation to methane flux.

Interpretation boundary:

- Wetland source provenance is strong at the paper/dataset/MAG-list level.
- Per-sample environmental metadata is not uniformly available for all 107 Methanoregula MAGs.
- Source-bucket rows must not be treated as sample-resolved records until a MAG-to-BioSample or MAG-to-sample mapping table is recovered.

### Mangrove / MSM China

Primary paper:

- Pan, S. et al. "A holistic genome dataset of bacteria and archaea of mangrove sediments." *GigaScience* 14, giaf081 (2025). DOI: `10.1093/gigascience/giaf081`.

Primary data object:

- GigaDB supporting dataset, DOI `10.5524/102702`.

External verification:

- The paper reports metagenome sequencing of mangrove sediment microbial communities across Southeast China from 2014 to 2020.
- The paper reports 966 MAGs passing completeness and contamination thresholds.
- The paper data availability section lists NODE accessions, eLMSG MAG accessions, NCBI `PRJNA1150796`, and GigaDB.
- DataCite API resolves `10.5524/102702` as the GigaScience Database supporting dataset for the Pan et al. article.

Local reconciliation:

Local source register:

- 1428 local archive MAG candidates.
- 966 paper-reported final MAGs.
- 644 metagenomes.
- 82 local GigaDB sediment-sample metadata rows.
- 71 NCBI BioSample environmental metadata rows.

Usable local sample metadata from `metadata_sediment_samples.txt`:

| Field | Current coverage |
| --- | --- |
| `sample_id` | 82/82 |
| `group` | 82/82 across six groups |
| `Latitude`, `Longitude` | 82/82 |
| `depth` | 82/82 |
| `collect_date` | 82/82 |
| `sample_loc` | 82/82 |
| `env_package` | 82/82, sediment |
| `mangrove type` | 21/82 non-empty |

Observed local distributions:

- Years: 2014 through 2020.
- Most represented year: 2019.
- Depths include 0-2 cm, 5 cm, 6-8 cm, 10 cm, 10-20 cm, 20 cm, and related ranges.
- Site labels include Ximendao, Aojiang River estuary, Fujian, Hanjiang River estuary, Zhangjiang River estuary, Shenzhen/Futian Mangrove Nature Reserve, Hong Kong/Mai Po, Beibu Gulf, Leizhou, Danzhou, and Dongzhaigang.
- Vegetation context is available for a subset: `Kandelia candel`, `Kandelia obovata`, `Aegiceras corniculatum`, and `Sonneratia apetala`.

Interpretation boundary:

- The mangrove source is conclusive.
- The sample-level sediment metadata is strong for local samples.
- Per-MAG sample assignment must still be verified before rolling MAG features up to site/sample MRV features.
- External claims must reconcile the 1428 local archive candidates against the published 966 final medium/high-quality MAGs.

## Recommended Normalized Metadata Schema

Use this schema for the future consolidated environmental metadata table:

```text
metadata_unit_id
metadata_unit_type              # mag, proteome, sample, site, project, modeled_point
proteome_id
mag_id
sample_id
source_domain                   # rumen, wetland_mucc, mangrove_msm
source_paper_doi
source_dataset_doi
primary_accession
primary_accession_type          # ENA analysis, NCBI BioSample, NCBI BioProject, Zenodo file, GigaDB sample
provenance_resolution_tier      # exact_analysis_accession, exact_biosample, exact_bin_site_project, exact_source_bucket, paper_level, modeled_estimate
ecosystem
habitat_type
site_label
source_bucket
country_or_region
latitude
longitude
coordinate_resolution_tier
collection_date
collection_date_resolution_tier
depth_value
depth_unit
depth_resolution_tier
sample_material
host_or_vegetation
wetland_type
salinity_context
tidal_context
temperature_context
methane_flux_context
sulfate_context
ph_context
soil_carbon_context
covariate_source
covariate_source_url
covariate_is_measured
covariate_is_modeled
claim_allowed
blocking_gaps
```

## Free API Routes For Metadata Enrichment

| API/source | What it can provide | Recommended use | Evidence tier |
| --- | --- | --- | --- |
| ENA Portal API | ENA study, analysis, sample, run, FTP, host, country, collection date, environmental fields | Rumen `PRJEB31266` and any ENA-backed MAG/sample accessions | Exact accession metadata |
| NCBI E-utilities | PubMed, BioProject, BioSample, Assembly, SRA metadata | BioSample/Assembly resolution for MUCC GTDB rows and mangrove sample metadata | Exact accession metadata |
| NCBI Datasets API | Assembly/genome and BioSample linked metadata | Bulk assembly/BioSample reconciliation where accessions are known | Exact accession metadata |
| EBI BioSamples API | Rich BioSample attributes from ENA/INSDC | Cross-check ENA/NCBI sample attributes and normalize MIxS fields | Exact accession metadata |
| Zenodo REST API | MUCC record metadata and file list/download links | Verify MUCC version and reacquire Methanoregula files | Dataset-level provenance |
| DataCite REST API | DOI metadata and related identifiers | Verify GigaDB DOI, publisher, creators, related BioProjects | Dataset-level provenance |
| GigaDB dataset pages / file endpoints | Mangrove supporting files where publicly accessible | Reacquire invalid `Supplementary_Tables.xlsx`; verify file checksums | Dataset-level provenance |
| AmeriFlux API / `amerifluxr` | Site metadata, flux products, variables, date coverage | OWC, LA2, TWI methane/meteorology context where paper cites AmeriFlux | Measured site-level flux/context |
| FLUXNET-CH4 | Methane flux and meteorological products for wetland towers | Site-level methane validation and benchmark covariates | Measured site-level flux/context |
| USGS data releases | PPR greenhouse gas fluxes and environmental data | Prairie Pothole site covariates and flux validation | Measured site/project context |
| Open-Meteo Historical API | Historical/reanalysis temperature, precipitation, radiation and related weather variables by coordinate/date | Fill climate covariates for mangrove samples and wetland sites when measured metadata is absent | Modeled estimate |
| SoilGrids REST API | pH, SOC, nitrogen, texture, bulk density, CEC and uncertainty at 250 m resolution | Soil/sediment context for coordinate-bearing samples; useful for mangrove samples | Modeled estimate |
| WorldClim / CHELSA / TerraClimate | Long-term climate normals and hydroclimate context | Site-level ecological context where daily weather is not needed | Modeled estimate |
| NOAA CO-OPS / tidal APIs | Water level and tidal context for coastal sites where stations are nearby | Mangrove hydrologic/tidal context, with station-distance caveats | Observed/model-assisted site estimate |
| Global Mangrove Watch / remote-sensing layers | Mangrove extent and change context | Habitat confirmation and surrounding mangrove context | Modeled/remote-sensing estimate |

Example verified API probes from this pass:

- ENA query for `PRJEB31266` returned `analysis_accession`, `sample_accession`, `scientific_name`, `country`, `collection_date`, `host`, `environment_feature`, `environment_material`, and `submitted_ftp`.
- Zenodo API for `14532347` returned the MUCC v2.0.0 title, DOI, and Methanoregula file list.
- DataCite API for `10.5524/102702` returned the GigaScience Database title, publisher, and year.
- SoilGrids REST API returned pH, SOC, and nitrogen layers for a mangrove coordinate.
- Open-Meteo Historical API returned daily mean temperature and precipitation for a mangrove coordinate/date.

## Current Reconciled Metadata Readiness

| Lane | Ready now | Estimate-ready | Still blocked |
| --- | --- | --- | --- |
| Rumen | Exact ENA analysis accession, country, digestive-system context, source paper/data DOI | Broad cattle-rumen context from paper; no need for climate/soil estimates for MRV target context | Animal-level metadata and exact cattle/sample mapping beyond ENA analysis context |
| Wetland/MUCC | Paper/dataset provenance, source bucket, OWC bin/site context, some NCBI assembly/BioSample rows | Site-level wetland type, methane flux context, AmeriFlux/USGS/site-source covariates | Uniform MAG-to-sample BioSample mapping for JGI/PPR/STM/source-bucket rows |
| Mangrove/MSM | Paper/dataset provenance, 82 local sediment sample rows, 71 exact BioSample rows, coordinates/dates/depths for local sample metadata | SoilGrids/Open-Meteo/tidal/remote-sensing context by coordinate/date | MAG-to-sample assignment and 966-vs-1428 denominator reconciliation |

## Recommended Next Consolidation Actions

1. Build `dim_source_dataset` with one row per source paper/data object:
   - Stewart 2019 / `PRJEB31266`
   - Bechtold 2025 / MUCC v2.0.0 / Zenodo `14532347`
   - Pan 2025 / GigaDB `102702`

2. Build `dim_environmental_context` with the schema above and preserve resolution tiers.

3. Reacquire the invalid mangrove `Supplementary_Tables.xlsx` from GigaDB before using any Excel-only metadata.

4. For wetland/MUCC, prioritize recovering the missing MAG-to-BioSample or MAG-to-sample metadata for JGI, PPR, and STM rows. Until then, these rows remain source-bucket/site-level context.

5. For mangrove/MSM, reconcile local MAG IDs to sample IDs before reporting sample-level MAG functional rollups.

6. Treat Open-Meteo, SoilGrids, tidal, and remote-sensing covariates as `modeled_estimate` rows with source URL, query date, coordinate, date range, and uncertainty when available.

7. Do not assign final MRV methane-risk tiers from this metadata alone. Use it as context for MAG-level functional features, sample rollup readiness, and monitoring-priority hypotheses.

## Allowed Wording

Allowed:

- "The rumen POC MAGs trace to Stewart et al. 2019 and ENA PRJEB31266 through exact analysis accessions."
- "The wetland/MUCC POC MAGs trace to the Bechtold et al. 2025 MUCC v2.0.0 Methanoregula analysis and Zenodo data package, with mixed per-MAG sample metadata resolution."
- "The mangrove/MSM payload traces to Pan et al. 2025 and GigaDB 10.5524/102702, with local sediment sample metadata available and per-MAG sample rollup still pending."
- "Modeled environmental covariates from SoilGrids/Open-Meteo can support ecological context and uncertainty-aware feature engineering."

Not allowed:

- "Every wetland/MUCC MAG has exact sample-level environmental metadata."
- "The mangrove MAG features are already sample-level MRV features."
- "Modeled climate/soil covariates are measured field metadata."
- "These provenance and environmental metadata alone support final A-E risk scoring or carbon-credit approval."
