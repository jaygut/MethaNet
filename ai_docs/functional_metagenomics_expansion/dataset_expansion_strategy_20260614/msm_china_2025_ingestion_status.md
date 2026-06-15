# MSM China 2025 Mangrove MAG Catalog Ingestion Status

Date: 2026-06-15

Scope: verified ingestion status and strategic integration plan for Pan et al.
2025, "A holistic genome dataset of bacteria and archaea of mangrove
sediments," GigaScience, dataset DOI `10.5524/102702`.

This note is part of the MethaNet functional-metagenomics expansion package. It
does not replace the broader dataset strategy:

```text
ai_docs/functional_metagenomics_expansion/dataset_expansion_strategy_20260614/methanet_codebase_dataset_inventory_and_expansion_strategy.md
```

## Executive Position

The Pan et al. 2025 MSM catalog is the highest-priority external expansion
target identified so far for the MethaNet wetland/mangrove atlas.

Why:

- It is explicitly mangrove sediment, not generic wetland or freshwater
  sediment.
- It is MAG-resolved, not only amplicon, metagenome, or marker-gene evidence.
- The paper reports methane, sulfur, nitrogen, carbon, and substrate-relevant
  annotation layers.
- The repository accession range points to eLMSG genome records where direct
  genome FASTA and predicted protein FASTA should be available.
- It adds a multi-reserve China mangrove source axis that can help break the
  current target-domain source confounding in the MethaNet 662-proteome POC.

The current local status is therefore:

```text
strategic_fit = excellent
format_readiness = direct_MAG_FASTA_ready_and_Prodigal_proteome_ready
actual_payload_fetched = yes_from_GigaDB_Wasabi
atlas_inclusion_status = ready_for_MethaNet_QC_functional_annotation_and_ESM2_embedding
```

Do not treat the 1,428 candidate eLMSG accession rows as confirmed MethaNet
MAGs. The paper reports 966 final MSM MAGs, while the published eLMSG accession
numeric range spans 1,428 identifiers. This discrepancy must be reconciled
against repository metadata before any atlas inclusion, dereplication,
embedding, annotation, or scientific claim.

2026-06-15 execution update:

- The broken GigaDB landing page was bypassed through the public GigaDB Wasabi
  object route:
  `https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/102001_103000/102702/`.
- `MAG_file.zip` was downloaded, official MD5
  `c69a96c13d84ae0fe1a52005bcb644cd` was verified, and 1,428 `.fa` MAG FASTAs
  were extracted.
- The FASTAs were normalized into unique local paths under
  `data/external/msm_china_2025/genomes_fna/`.
- Prodigal 2.6.3 in the existing `methanet-fgx` environment was run with
  `-p meta`, producing 1,428 `.faa` proteomes, 1,428 `.ffn` files, and 1,428
  `.gff` files.
- The final current handoff manifest is
  `data/external/msm_china_2025/manifests/msm_china_2025_functional_embedding_manifest.tsv`.

## Primary Sources Verified

| Source | Verified fact | MethaNet implication |
| --- | --- | --- |
| GigaScience article | Pan et al. 2025 is a GigaScience data article for mangrove sediment bacteria and archaea. | Citable source for scope, methods, and caveats. |
| Dataset DOI `10.5524/102702` | Supporting data are registered through GigaDB/DOI metadata; DataCite reports 109.06 GB and CC0 licensing. | Primary package target for sample/MAG metadata and supplements. |
| GigaDB Wasabi object listing | Dataset objects are reachable under `s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/102001_103000/102702/`. | Working payload route for direct MAG FASTA acquisition. |
| eLMSG help page | eLMSG genome downloads are described as free tar.gz bundles containing `.fna`, `.faa`, `.gff`, `.ko`, `.cog`, `.pfam`, `.ec`, `.cyc`, and other files when available. | If eLMSG is reachable, the catalog should support direct genome/proteome ingestion without expensive reassembly. |
| Article Data Availability | Raw sequences are reported in NODE, the MAGs in eLMSG, and raw sequences also in NCBI under BioProject `PRJNA1150796`. | eLMSG is the primary MAG/proteome route; NCBI/NODE are secondary mirrors for raw reads and sample metadata. |
| DataCite related identifiers | The DOI references `PRJNA1150796`, `PRJNA1136686`, `PRJNA1159532`, `PRJNA1268148`, `PRJNA1268163`, nine NODE projects, and eLMSG. | NCBI-linked projects expose rich run/BioSample metadata even while MAG payload access remains blocked. |

## Biological And Dataset Content

Published catalog summary:

| Layer | Published content |
| --- | --- |
| Habitat | Mangrove sediments. |
| Geography | Six mangrove natural reserves across southeastern China. |
| Provinces | Fujian, Guangxi, Guangdong, Zhejiang, Hainan. |
| Time span | 2014-2020. |
| Samples/metagenomes | The paper reports hundreds of metagenomes and six sample groups; specific sections report 644 metagenomes. |
| MAG count | 966 medium/high-quality MAGs reported in the article. |
| Quality threshold | Completeness at least 50 percent and contamination at most 10 percent for the reported medium/high-quality catalog. |
| MAG taxonomy | Bacteria and archaea, including 8 archaeal phyla and 50 bacterial phyla in the phylogenomic set. |
| Methane relevance | Article reports methane-cycle genes including methanogenesis and methane oxidation markers. |
| Sulfur relevance | Article reports sulfate reduction and sulfur oxidation genes, important for mangrove methane suppression or competition context. |

Important internal consistency gates:

- The article reports 966 MAGs, but the eLMSG accession range
  `LMSG_G000027425.1-LMSG_G000028852.1` spans 1,428 numeric accessions.
- The article text includes count differences across "MAGs," "MSMs,"
  phylogenomic subsets, and table totals. Treat these as publication/repository
  reconciliation tasks, not as license to choose a convenient denominator.
- The article contains high-quality MAG count wording that may vary by section.
  Resolve final counts from repository metadata and QC tables, not only prose.

## Why This Can Avoid Expensive Reassembly

MethaNet does not need raw-read reassembly. GigaDB exposes `MAG_file.zip`, a
compressed archive containing the MAG FASTA sequences. Protein FASTAs were then
generated locally with the same Prodigal mode used by the MethaNet functional
annotation runner.

The desired minimal direct-ingestion files are:

| File type | Need | Expected source |
| --- | --- | --- |
| MAG nucleotide FASTA `.fna` | Functional annotation, QC, taxonomy, dereplication. | Downloaded from GigaDB `MAG_file.zip` and normalized locally. |
| Predicted protein FASTA `.faa` | ESM2 proteome embedding and protein-level annotation. | Generated locally with Prodigal 2.6.3 `-p meta`. |
| GFF `.gff` | Gene coordinates and crosswalk from nucleotide to proteins. | Generated locally with Prodigal 2.6.3 `-p meta`. |
| FFN `.ffn` | Nucleotide CDS sequences. | Generated locally with Prodigal 2.6.3 `-p meta`. |
| KO/COG/Pfam/EC/Cyc annotations | Optional imported evidence layer; still rerun MethaNet standard tools for comparability. | GigaDB functional archive and/or MethaNet standard annotation stack. |
| Sample/reserve/depth metadata | Sample linkage and source-aware controls. | GigaDB `metadata_sediment_samples.txt`, `NODE_NCBI.csv`, and NCBI BioSample reports. |
| Quality/taxonomy table | Gate A and denominator reconciliation. | GigaDB `MAGs_gtdbtk_classification.csv` plus future MethaNet QC reruns. |

Because both `.fna` and `.faa` now exist for each local archive MAG candidate,
the appropriate route is:

1. Register each source FASTA as
   `mag_id = <source_group>__<source_fasta_stem>`.
2. Create canonical MethaNet
   `proteome_id = msm_china_2025__<source_group>__<source_fasta_stem>`.
3. Run FASTA sanity checks on `.fna` and `.faa`.
4. Embed the `.faa` proteins with the standard ESM2 path.
5. Run the same functional annotation and QC stack used for current MAG/bin
   units.
6. Preserve imported eLMSG annotations as external evidence, not as a substitute
   for MethaNet-standard comparability runs.

## Local Staging Package

Local root:

```text
data/external/msm_china_2025/
```

Current layout:

| Path | Purpose | Status |
| --- | --- | --- |
| `source_docs/` | Cached article/repository/API HTML or JSON source records. | Populated with DataCite JSON, NODE shell pages, eLMSG help, and download-attempt captures. |
| `gigadb_wasabi/` | Working GigaDB object-route files, readme, checksums, file-size manifest, taxonomy, metadata, and small support files. | Populated. |
| `metadata/source_register.tsv` | Source registry for the MSM dataset. | Created. |
| `metadata/site_context_from_article.tsv` | Human-curated site/reserve context from article text. | Created. |
| `metadata/sample_group_summary_from_article.tsv` | Six-group summary extracted from article/table text. | Created. |
| `metadata/ncbi_sra_biosample_manifest.tsv` | Run-level SRA/BioSample manifest for DataCite-linked NCBI projects. | Created with 71 run rows. |
| `metadata/ncbi_biosample_environmental_metadata.tsv` | Flattened BioSample environmental metadata table. | Created with 71 run rows and 38 unique BioSamples; all rows currently have collection date, depth, coordinates, ENVO/environmental fields, and exact BioSample provenance. |
| `metadata/biosample_reports/` | Cached NCBI BioSample text reports. | Created with 38 reports. |
| `manifests/elmsg_accession_range_candidates.tsv` | Candidate accession range from the published eLMSG range. | Created with 1,428 candidates plus header. |
| `manifests/elmsg_accession_record_candidates.tsv` | Candidate accession range plus inferred eLMSG record URLs. | Created with 1,428 candidates; inferred `MSG099710-MSG101137` from an indexed eLMSG AMD page where `LMSG_G000004334.1` maps to `MSG076619` and record IDs increment with the genome accession numeric suffix. |
| `manifests/msm_china_2025_mag_manifest.resolved.tsv` | Resolved archive MAG FASTA manifest with taxonomy, source samples, assembly stats, and local paths. | Created with 1,428 rows. |
| `manifests/msm_china_2025_proteome_manifest.tsv` | Prodigal proteome manifest with FAA/FFN/GFF paths and protein counts. | Created with 1,428 rows. |
| `manifests/msm_china_2025_functional_embedding_manifest.tsv` | Final handoff manifest for MethaNet functional annotation and ESM2 embedding. | Created with 1,428 `fna_and_faa_ready` rows. |
| `manifests/functional_embedding_ready_manifest.tsv` | eLMSG accession-readiness output produced by the helper script. | Retained as an eLMSG dry-run status manifest; GigaDB Wasabi is now the working payload route. |
| `raw_downloads/` | Downloaded source archives. | `MAG_file.zip` downloaded and MD5 verified. |
| `genomes_fna/` | Normalized nucleotide MAG FASTAs. | Created with 1,428 `.fna` files. |
| `proteomes_faa/` | Predicted protein FASTAs for ESM2 and annotation. | Created with 1,428 `.faa` files. |
| `genes_ffn/` | Predicted nucleotide CDS FASTAs. | Created with 1,428 `.ffn` files. |
| `genes_gff/` | Prodigal GFF files. | Created with 1,428 `.gff` files. |
| `annotations/` | Expected extracted eLMSG annotation sidecars. | Not required for initial integration; MethaNet will rerun comparable annotations. |
| `logs/download_attempts.tsv` | Transport and repository access audit log. | Created. |

Current validated counts:

| Check | Result |
| --- | --- |
| Candidate eLMSG rows | 1,428 candidates plus header. |
| Candidate eLMSG record URLs | 1,428 inferred record URLs plus header. |
| GigaDB archive FASTA rows | 1,428. |
| Final functional/embedding manifest rows | 1,428. |
| NCBI run rows from DataCite-linked projects | 71. |
| Unique NCBI BioSamples resolved | 38. |
| BioSample rows with collection date, depth, coordinates, and environmental context | 71. |
| Confirmed `.fna` files | 1,428. |
| Confirmed `.faa` files | 1,428. |
| Confirmed `.ffn` files | 1,428. |
| Confirmed `.gff` files | 1,428. |
| Confirmed ready MAG/proteome pairs | 1,428, subject to QC and 966-vs-1,428 denominator reconciliation. |

## Fetch Helper

Script:

```text
scripts/external/fetch_msm_china_2025_elmsg.py
```

Purpose:

- read `elmsg_accession_range_candidates.tsv`;
- try one or more accession or eLMSG record URL templates;
- reject HTML/maintenance pages;
- validate tar/gzip payloads;
- extract expected `.fna`, `.faa`, `.gff`, `.ko`, `.cog`, `.pfam`, `.ec`,
  `.cyc`, and related files;
- write `manifests/functional_embedding_ready_manifest.tsv`;
- preserve each accession as a status row whether successful, incomplete, or
  blocked.

Dry-run command:

```bash
python scripts/external/fetch_msm_china_2025_elmsg.py \
  --manifest data/external/msm_china_2025/manifests/elmsg_accession_record_candidates.tsv \
  --dry-run --limit 5
```

Template override pattern when the working eLMSG route is known:

```bash
python scripts/external/fetch_msm_china_2025_elmsg.py \
  --url-template 'https://example.org/download/{accession}.tar.gz'
```

The helper should be run first with `--limit 1` or `--limit 5`. Only scale to
the full accession range after at least one accession yields a real tar/gzip
bundle containing both `.fna` and `.faa`.

NCBI metadata helper:

```text
scripts/external/fetch_msm_china_2025_biosamples.py
```

Purpose:

- fetch SRA run-info for the DataCite-linked NCBI projects that currently
  expose SRA rows;
- cache BioSample reports;
- flatten collection date, depth, environmental context, geography, coordinates,
  sequencing run, and download path into
  `metadata/ncbi_biosample_environmental_metadata.tsv`;
- preserve this as sample/read metadata, not as MAG/proteome payload evidence.

## Repository Access Status

As of the ingestion pass:

| Resource | Result | Interpretation |
| --- | --- | --- |
| GigaDB dataset page | DOI/web index confirms the dataset, but direct `curl` from this runtime returned empty replies or redirect loops. | Retain DOI and retry later; use browser/manual download if necessary. |
| GigaDB Wasabi object route | `readme_102702.txt`, object listing, checksums, MAG archive, metadata, and taxonomy support files were downloaded successfully. | This is the working payload route; use it for reproducible acquisition. |
| eLMSG direct route probes | Known route guesses returned BMDC maintenance HTML shells rather than data bundles. | Do not ingest these files; retry after maintenance or discover the active API route. |
| eLMSG record route inference | Indexed eLMSG AMD page shows `LMSG_G000004334.1 -> /elmsg/record/MSG076619` and `LMSG_G000011340.1 -> /elmsg/record/MSG083625`, supporting a derived MSM record range `MSG099710-MSG101137`. | Use record-aware URLs first when eLMSG returns from maintenance; current probes still return BMDC maintenance shells. |
| eLMSG help page | Saved and confirms expected free genome bundle formats. | Strong evidence that direct MAG/proteome ingestion should be possible once route works. |
| NCBI BioProject `PRJNA1150796` | BioProject exists, but linked SRA/Assembly/BioSample records did not resolve through EUtils or run-info in this run. | Keep as the article-declared umbrella/raw mirror; not currently a usable payload route. |
| DataCite-linked NCBI projects | `PRJNA1136686`, `PRJNA1159532`, `PRJNA1268148`, and `PRJNA1268163` expose 71 SRA run rows and 38 BioSamples through NCBI run-info/BioSample reports. | Use for sample/environmental metadata enrichment and possible future raw-read checks; these are not MAG/proteome bundles. |
| NODE project pages | Accession pages saved as short shell HTML pages. | Mine JS/API endpoints only if raw-read metadata becomes necessary. |

This means the eLMSG route is still blocked, but the required MethaNet payload
has been acquired through GigaDB Wasabi.

## Canonical MethaNet Identity Plan

Use `proteome_id` as the canonical cohort key.

Proposed identifiers:

| Field | Value pattern | Meaning |
| --- | --- | --- |
| `source_project_id` | `MSM_China_2014_2020` | Source project/study. |
| `external_dataset_id` | `msm_china_2025` | Local dataset slug. |
| `source_accession` | source FASTA filename, e.g. `m1_bins_1_bin.128.fa` | GigaDB archive MAG filename. |
| `mag_id` | `group1_MAGs__m1_bins_1_bin.128` | Locally unique MAG/genome identifier. |
| `proteome_id` | `msm_china_2025__group1_MAGs__m1_bins_1_bin.128` | Canonical MethaNet proteome key. |
| `sample_id` | `OES...` source sample IDs and BioSample where mapped | Physical sample/source sample key. |
| `biosample_id` | `SAMN...` or equivalent when verified | Public BioSample anchor. |
| `metadata_resolution_tier` | exact sample, site, publication, repository, inferred, missing | Provenance of environmental context. |

No `sample_id` or `biosample_id` should be invented from the eLMSG genome
accession. If BioSamples are absent or not yet resolved, use source sample IDs
and mark `biosample_id = missing`.

## Integration Gates Before Atlas Inclusion

### Gate 1: Payload

Required for each included MAG:

- real `.fna` exists;
- real `.faa` exists;
- neither file is HTML or a repository shell;
- FASTA headers are parseable;
- protein count is plausible for a MAG;
- nucleotide assembly size is plausible for a MAG, not an assembly-scale sample.

### Gate 2: Denominator Reconciliation

Required before claiming a count:

- resolve why accession range spans 1,428 IDs while article reports 966 MAGs;
- identify which accessions are final MSM MAGs;
- identify which accessions are failed, withdrawn, redundant, non-MAG,
  intermediate, or sidecar records if applicable;
- preserve excluded records in a status table.

### Gate 3: Unit Scope

Each row must be classified as:

```text
mag_bin
assembly_context
embedding_subset
unresolved
```

Only `mag_bin` rows can enter MAG-level MBAG functional interpretation.

### Gate 4: QC And Taxonomy

Minimum required fields:

- completeness;
- contamination;
- genome size;
- contig count;
- N50;
- GC percentage;
- domain/phylum/class/order/family/genus/species where available;
- GTDB release or source taxonomy method;
- quality status.

Imported QC can seed the table, but MethaNet should rerun or normalize QC where
possible for comparability with current wetland and rumen cohorts.

### Gate 5: Sample/Environmental Metadata

Minimum useful fields:

- reserve/site;
- province/country;
- latitude/longitude or coordinate resolution tier;
- date/year;
- sediment depth;
- sample group;
- habitat type;
- restoration/natural status when available;
- salinity, sulfate, redox, pH, temperature, organic carbon, vegetation, or
  explicit missingness.

This metadata supports sample-risk readiness only. It does not make a MAG-level
functional annotation into a sample-level methane risk estimate.

### Gate 6: MethaNet Functional Comparability

For included MAGs, run the MethaNet standard annotation stack:

- Prodigal/Bakta as applicable;
- CheckM2/GUNC/QC layer;
- GTDB-Tk or comparable taxonomy normalization;
- KOfam/KEGG;
- MCycDB methane marker panel;
- SCycDB sulfur marker panel;
- dbCAN/CAZy;
- METABOLIC or curated biogeochemical trait extraction;
- annotation coverage metrics.

Imported eLMSG annotations are useful as cross-checks and provenance evidence,
not final comparability evidence by themselves.

## Strategic Value For MethaNet

| Current MethaNet limitation | MSM contribution |
| --- | --- |
| Wetland/MUCC source confounding | Adds independent mangrove reserve source axis. |
| Limited target-domain geographic diversity | Adds China coastal mangrove samples across five provinces. |
| Need methane/sulfur/substrate mechanism diversity | Article reports CH4, S, N, CAZy, KEGG, eggNOG, and related layers. |
| Need partner-facing mangrove credibility | Directly aligned to blue carbon/mangrove sediment domain. |
| Need source-aware validation | Six groups/reserves can become source/project strata after metadata resolution. |

## Claim Boundary Matrix

| Candidate claim | Allowed wording now | Evidence status | Blocking gap | Next action |
| --- | --- | --- | --- | --- |
| MSM is a high-priority MethaNet expansion target. | "The Pan et al. MSM mangrove sediment MAG catalog is a high-priority candidate expansion source." | Supported by article and repository metadata. | None for strategic prioritization. | Keep as Priority 1. |
| MSM can be directly embedded with ESM2. | "The GigaDB MAG archive has been locally verified and Prodigal `-p meta` proteomes have been generated for all 1,428 archive FASTAs." | Locally supported for proteome payload readiness. | ESM2 embedding run still pending. | Run ESM2 over `msm_china_2025_functional_embedding_manifest.tsv` after QC triage priorities are registered. |
| MSM can avoid reassembly. | "The local MethaNet payload uses downloaded MAG FASTAs and generated proteomes, so raw-read reassembly is not needed for the initial atlas expansion." | Supported by local GigaDB archive extraction and MD5 verification. | None for initial MAG/proteome ingestion. | Preserve raw-read sources for future abundance/coverage, not MAG reconstruction. |
| MSM adds 966 MAGs to MethaNet. | Not allowed yet. Use "published 966-MAG catalog, with a local 1,428-FASTA archive undergoing QC reconciliation." | Publication supported; local archive denominator verified. | Downloaded archive contains 1,428 FASTAs and small metadata files do not expose per-MAG completeness/contamination. | Use MethaNet CheckM2/GUNC outputs from the launched functional annotation run to identify rows meeting the published quality definition. |
| MSM supports sample-level methane risk scoring. | Not allowed. Use "MAG-level functional potential and sample-risk-readiness context." | MAG-level potential likely; sample scoring not supported. | Abundance, environmental covariates, uncertainty, and flux/process validation absent. | Build sample linkage and readiness layer after MAG ingestion. |
| MSM supports carbon-credit claims. | Not allowed. Use "screening and monitoring-design evidence only." | Molecular atlas only. | No registry validation or measured flux. | Preserve MRV claim boundaries. |

## Concrete Next Actions

1. Monitor the launched MethaNet functional annotation arrays `8797` and
   `8798`, which process the 1,428 archive MAG/proteome rows in two validated
   Slurm tranches.
2. Use the first completed CheckM2/GUNC results to resolve the published 966
   medium/high-quality MAG denominator from the 1,428 archive FASTAs.
3. After enough MAGs complete, consolidate the functional warehouse and build
   methane/sulfur/substrate candidate tables for the 80 archaeal priority MAGs
   and 231 sulfur-competition priority MAGs.
4. Run ESM2 proteome embedding after functional QC status exists, or run it in
   parallel with clear labels that embedding is not a QC substitute.
5. Dereplicate against existing MUCC/wetland MAGs and other proposed mangrove
   expansion sets.
6. Add MSM as a separate source stratum in downstream MBAG/source-aware
    validation.

## Decision

Proceed, but only as a staged ingestion:

```text
stage_0 = source_verified
stage_1 = payload_route_resolved_through_GigaDB_Wasabi
stage_2 = MAG_FASTA_payload_downloaded_and_MD5_verified
stage_3 = Prodigal_proteomes_generated
stage_4 = final_966_vs_1428_denominator_QC_reconciliation_pending
stage_5 = MethaNet_functional_annotation_arrays_submitted_jobs_8797_8798
stage_6 = ESM2_embedding_pending
stage_7 = atlas_inclusion_pending
```

The scientifically correct current status is completed payload acquisition and
proteome preparation plus launched MAG-level functional annotation, with
QC/denominator reconciliation still required before external claims about the
published 966-MAG subset.

## References

- GigaScience article:
  `https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf081/8232623`
- Supporting dataset DOI:
  `https://doi.org/10.5524/102702`
- eLMSG help page:
  `https://www.biosino.org/elmsg/help`
- NCBI BioProject:
  `https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1150796`
