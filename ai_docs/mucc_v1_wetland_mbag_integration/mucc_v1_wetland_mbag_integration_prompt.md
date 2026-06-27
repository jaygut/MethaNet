# MethaNet Agent Task Prompt: MUCC v1 Wetland MAG And Metatranscriptome Integration

Date: 2026-06-26
Status: copy-ready operating prompt; 2026-06-26 local execution checkpoint appended
Scope: future MethaNet work to ingest, audit, interpret, and integrate the MUCC v1 Old Woman Creek wetland MAG and metatranscriptome catalog as a native wetland reference lane for MBAG.

## Why This Prompt Exists

This prompt is a business-aligned revision of the attached MUCC v1 ingest prompt. The original prompt correctly recognized the scientific opportunity: MUCC v1 provides 2,502 high- and medium-quality wetland MAGs from Old Woman Creek (OWC), paired with processed metatranscriptomic expression across 133 samples and depth-resolved environmental context.

This revision makes the MethaNet objective explicit:

> Convert MUCC v1 from a public wetland genome catalog into a claim-safe, activity-aware, depth-resolved MethaNet reference lane for methane permanence-risk screening, bridge-candidate review, sample-risk readiness, and partner-facing molecular intelligence.

This is not a prompt to produce final MRV risk scores. It is a prompt to build the evidence layer that can make future MRV scoring more defensible.

## Current Local Execution Checkpoint

As of 2026-06-26, this prompt has already produced a high-value staging layer under:

```text
results/functional_metagenomics/mucc_v1_owc_wetland_20260626/
```

Current grounded state:

- All six Zenodo `8194033` payloads are downloaded and md5-validated.
- `MAGs.zip` parses to 2,508 local OWC FASTA entries, while the paper/deposit headline denominator remains 2,502 HQ/MQ MAGs.
- Source DRAM rows map to all 2,508 local FASTA entries.
- Processed MAG/gene expression support covers 1,948 MAGs across 133 source sample columns.
- Source protein FASTA maps directly to 2,501 local FASTA entries, has been split into 2,501 per-MAG proteome FASTA files, passes the existing ESM2 builder dry-run input inventory with zero missing FAA files, and is registered through normalized source-lane and functional manifests.
- Corrected GPU-requesting Apolo accel production ESM2 shards are queued as jobs `11354`, `11355`, `11356`, and `11357`; a superseded no-checkpoint launch `11350`-`11353` was canceled/replaced after the submitter was patched to pass `--gres=gpu:1`.
- Prodigal/GFF generation for MUCC gLM2 context has produced a final manifest with 2,508/2,508 non-empty FAA/GFF/FFN triplets. Slurm still reports longjobs job `11359` as RUNNING, so monitor that operational state, but use the validated manifest as input-readiness evidence.
- A partial gLM2 smoke payload has been prepared for 32 MAGs and queued as job `11361`; this validates the context path only and must not be counted as full-lane gLM2 coverage.
- A full-lane gLM2 payload has been prepared from the final Prodigal manifest for 2,508 MAGs, 5,016 windows, and 51,141 spans, with full-lane inference queued as job `11363`.
- A MAG-level MRV readiness scaffold exists at `functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv` with 2,508 rows; 1,945 rows are source-scaffold MRV feature candidates pending embedding outputs, and `candidate_cards/mucc_v1_mrv_readiness_candidate_cards.tsv` contains 100 review cards.
- A native source-scaffold warehouse exists at `cohort_warehouse/` with 17 Parquet-backed tables, a 2,508-row `dim_mag`, two 259,084-row expression fact tables, and `functional_atlas.duckdb` for queryable atlas integration.
- The active stop-condition ledger is `reports/mucc_v1_stop_condition_external_compute_blockers_20260626.tsv`, with human-readable handoff in `reports/MUCC_V1_STOP_CONDITION_20260626.md`.
- Reproducible strategic source-scaffold review cards are generated at `candidate_cards/mucc_v1_strategic_review_candidate_cards.tsv`; these are candidate-review queues, not final bridge or mechanism claims.
- Seven local FASTA entries currently lack direct source protein records: `OWC_0091`, `OWC_0093`, `OWC_0095`, `OWC_0097`, `OWC_0098`, `OWC_0099`, and `OWC_0105`.

Therefore, future runs should not restart from source discovery unless provenance changes. They should continue from ESM2 shard monitoring, gLM2 inference monitoring, Slurm state reconciliation for `11359`, denominator reconciliation, MethaNet-curated mechanism feature generation, environmental joins, and wetland-neighbor rebuild.

## Required Local Grounding

Before executing this prompt, read these repository contracts:

```text
AGENTS.md
ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md
ai_docs/functional_metagenomics_expansion/data_aggregation_strategy.md
ai_docs/functional_metagenomics_expansion/source_provenance_environmental_metadata_reconciliation.md
ai_docs/functional_metagenomics_expansion/cohort_data_architecture_hardening.md
ai_docs/functional_metagenomics_expansion/output_contracts_and_gates.md
ai_docs/functional_metagenomics_expansion/pipeline_reproducibility_contract.md
ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/methanet_embedding_functional_transfer_framework.md
```

Use the MethaNet functional-atlas semantic layer when answering interpretation questions. Keep `proteome_id` as the canonical MethaNet cohort key unless a source-specific primary key is required and explicitly crosswalked.

## Mission

You are operating inside:

```text
/home/rsg-jcorre38/Jay_Proyects/MethaNet
```

Your mission is to integrate MUCC v1 as a new native wetland reference lane in MethaNet/MBAG, while preserving the exact evidence grain of every claim:

- MAG/proteome-level functional potential.
- MAG/gene/sample-level transcriptional support from processed expression tables.
- Sample/site/depth/month environmental context.
- Candidate bridge evidence between wetland references and mangrove target lanes.
- Sample risk readiness, not final risk tiers.

Treat MUCC v1 as a strategic bridge between the current rumen reference/control lane and the mangrove target lanes. It should reduce the ecological distance of MethaNet bridge interpretation by adding a freshwater wetland reference with in situ activity evidence, but it must not erase habitat-specific uncertainty.

## Strategic Business Objectives

MethaNet's goal is not generic wetland microbiome annotation. The goal is scalable, biologically grounded methane risk intelligence for blue carbon verification, screening, monitoring design, and partner diligence where direct flux measurements are expensive, sparse, and retrospective.

MUCC v1 should strengthen five business/product primitives:

1. **Wetland reference re-anchoring**
   - Add a native, activity-supported wetland lane so mangrove candidates are not interpreted only against a rumen methane-domain control.
   - Keep rumen as a reference/control and positive methane-domain comparator, not as a blue-carbon proxy.

2. **Activity-aware molecular attestation**
   - Upgrade selected evidence from "encoded potential" to "encoded and expressed in wetland soil" when processed metatranscriptome tables support it.
   - Separate expression evidence from process-rate or flux claims.

3. **Depth-resolved environmental permissiveness**
   - Treat centimeter-scale depth as a first-class axis for sample risk readiness.
   - Capture Fe(II), CEC, pH, acetate, formate, methanol, methylated substrates, porewater CH4/CO2, and flux context with resolution tiers.

4. **Candidate and mechanism prioritization**
   - Build candidate cards for methanogen, methanotroph, sulfur, fermenter, iron-reducer, methylotrophic, and WGCNA-module-linked MAGs.
   - Prioritize mechanisms that competitors often under-model, especially methylotrophic substrates and community-level redox interactions.

5. **Partner-facing intelligence**
   - Produce reusable artifacts: feature tables, wetland neighbor cards, sample readiness tables, validation gap registers, claim-boundary matrices, and report-ready summaries.
   - Every business-facing artifact must include allowed wording, evidence status, blocking gaps, and next validation action.

## Non-Negotiable Claim Boundaries

Do not violate these locks:

- Do not describe MUCC v1 integration as final MethaNet MRV risk scoring.
- Do not assign final A-E methane-risk tiers.
- Do not claim measured methane flux from MAG or expression evidence alone.
- Do not claim carbon-credit approval, registry readiness, or VM0033 compliance from molecular evidence alone.
- Do not claim wetland-to-mangrove or rumen-to-mangrove source-independent transfer.
- Do not treat one temperate freshwater marsh as a universal mangrove, saltmarsh, or blue-carbon analog.
- Do not treat modeled or site-level environmental covariates as exact sample-level measurements.
- Do not silently drop failed, pending, partial, low-quality, ambiguous, unlinked, or excluded evidence.

Allowed near-term wording:

> MUCC v1 can provide a native wetland molecular reference lane with genome-resolved activity evidence, enabling MethaNet to build better methane-risk feature primitives, bridge candidate cards, and sample readiness labels for future blue carbon validation.

Blocked wording:

> MUCC v1 proves MethaNet can score mangrove methane permanence risk or certify carbon credits from molecular data alone.

## Source Identity And Verification Targets

Treat the following as the starting source map. Verify each landing page and record access timestamp, URL, DOI/accession, file inventory, license, checksum or checksum availability, and any mismatch before consuming data.

### Primary Paper

```text
Borton, Oliverio, Narrowe, Villa, Rinke, Hoyt, Liu, et al.
"Mapping the soil microbiome functions shaping wetland methane emissions."
mSystems. Published online 2026-06-01.
DOI: 10.1128/msystems.00680-25
URL: https://journals.asm.org/doi/10.1128/msystems.00680-25
```

Key source facts to verify against the paper:

- OWC is a high methane-emitting freshwater wetland in the AmeriFlux network.
- The paper reports a catalog of 2,502 high- and medium-quality MAGs.
- The study integrates genome-resolved metatranscriptomes from 133 soil metatranscriptomes.
- Depth explains microbial community and functional expression patterns more strongly than land cover or time.
- WGCNA modules include surface/deep methane-predictive modules.

### Primary Data Deposit

```text
Zenodo concept DOI: 10.5281/zenodo.8194032
Zenodo data-bearing record DOI: 10.5281/zenodo.8194033
Zenodo latest record DOI observed via API redirect: 10.5281/zenodo.10622292
Record title: Multi-omics for Understanding Climate Change (MUCC) database v1.0.0
Data-bearing record URL: https://zenodo.org/records/8194033
Latest record URL: https://zenodo.org/records/10622292
```

Important source-audit note:

- The paper data availability section cites the concept DOI `10.5281/zenodo.8194032`.
- Zenodo's versions API identifies record `8194033`, DOI `10.5281/zenodo.8194033`, as the data-bearing MUCC v1.0.0 payload source for `MAGs.zip`, source proteins, DRAM annotations, and processed metatranscriptome tables.
- Zenodo currently resolves the concept DOI to latest record `10622292`, DOI `10.5281/zenodo.10622292`; this latest record is useful for latest metadata/tree context but may expose an incomplete file list through the API.
- Record the concept DOI, data-bearing record DOI, and latest record DOI. Use `8194033` for payload staging unless a future source audit shows that the data files have moved or the version relationship has changed.

The Zenodo record description lists:

```text
MAGs.zip
OWC_HQMQ_DB_genes.faa.gz
OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz
owc_metat_table_mags.csv
owc_metat_table_mags_genes.csv
owc_metat_table_mags_genes_annotations.csv
gtdbtk.ar53.decorated.tree
```

If the API file list or UI inventory appears incomplete relative to the description, log the discrepancy and pause before assuming the data are unavailable.

### Additional Source Objects

Verify but do not consume raw reads unless explicitly approved:

```text
KBase MUCC v1 collection: https://narrative.kbase.us/narrative/147022
NCBI raw reads and accessions: paper Table S1
Supplemental tables: msystems.00680-25-s0002.xlsx
Supplemental methods/figures: msystems.00680-25-s0001.pdf
AmeriFlux: Site ID US-OWC
ESS-DIVE: porewater CH4/CO2, chamber fluxes, nutrient/carbon sequestration and soil accretion context where applicable
MassIVE LC-MS metabolomics: MSV000093935
NERR CDMO: OWC meteorological, hydrological, and water quality context
```

Stop and request human review if any DOI, accession, or landing page resolves to a materially different dataset than described above.

## Dataset Scope

Target lane name:

```text
lane_id: mucc_v1_owc_wetland
lane: wetland_mucc_v1_owc
source_dataset: MUCC v1.0.0
source_project: Old Woman Creek wetland soil multi-omics
source_paper_doi: 10.1128/msystems.00680-25
source_dataset_concept_doi: 10.5281/zenodo.8194032
source_dataset_record_doi: 10.5281/zenodo.8194033
source_dataset_latest_record_doi: 10.5281/zenodo.10622292
```

Do not confuse this lane with the existing MethaNet POC wetland/MUCC Methanoregula lane documented in the repo, which is based on a different MUCC/Methanoregula framing. If an ID overlaps an existing local POC unit, preserve both contexts with explicit `lane_id`, `source_dataset_version`, and `source_record`.

Primary denominator:

```text
2,502 high- and medium-quality MUCC v1 MAGs
```

Intermediate counts that must be reconciled:

```text
17,333 draft genomes
3,217 dereplicated representatives
2,502 high- and medium-quality released MAGs
```

Expected key biological subsets:

```text
all_mags: 2,502
archaeal_mags: verify from paper/supplement
methanogen_mags: verify expected 85 HQ/MQ methanogen MAGs
most_wanted_lineages: Methanoregula, Methanothrix, Methanomethylicus, Methanospirillaceae UBA9949 or source-paper equivalent labels
```

## Hard Scope Constraints

1. Use the manifest as the denominator.
   - The manifest defines the cohort, not successful downloads or successful annotations.
   - Preserve skipped, missing, malformed, excluded, unlinked, and QC-blocked units as explicit status rows.

2. Use processed metatranscriptome abundance tables.
   - Use `owc_metat_table_mags.csv`, `owc_metat_table_mags_genes.csv`, and `owc_metat_table_mags_genes_annotations.csv` or their verified equivalent.
   - Do not download raw RNA reads.
   - Do not build a new RNA QC, rRNA removal, mapping, or quantification pipeline without explicit human approval.
   - If units, normalization, or MAG/gene joins are ambiguous, pause and create a metadata gap record.

3. Prefer author gene calls and annotations where fit for purpose.
   - Reuse the source DRAM gene calls and annotations for first-pass mapping.
   - Run MethaNet annotators only where needed to align to MBAG schema or produce comparable cross-lane features.
   - Document divergence between source annotations and MethaNet annotations.

4. Keep grains separate.
   - MAG/proteome features are not sample-level ecology.
   - Gene expression is not flux.
   - Site/depth context is not exact MAG-to-sample mapping unless proven.
   - WGCNA modules are feature primitives, not risk scores.

5. Keep generated data out of git.
   - Put large downloads, caches, run outputs, and intermediate ledgers under ignored `results/`, `data/external/`, or another approved generated-output area.
   - Promote only reviewed, small, dated documentation or templates to `ai_docs/`.

## Recommended Output Locations

Use a dated run ID for execution outputs:

```text
results/functional_metagenomics/mucc_v1_owc_wetland_<YYYYMMDD>/
```

Suggested layout:

```text
results/functional_metagenomics/mucc_v1_owc_wetland_<YYYYMMDD>/
  manifests/
  source_audit/
  raw_inventory/
  staging/
  per_mag/
  expression/
  environmental_metadata/
  cohort_warehouse/
  bridge_reanchoring/
  reports/
```

Recommended reviewed documentation outputs:

```text
ai_docs/mucc_v1_wetland_mbag_integration/
  mucc_v1_wetland_mbag_integration_prompt.md
  integration_report_<YYYYMMDD>.md                 # only after a reviewed execution snapshot exists
  claim_boundary_matrix_<YYYYMMDD>.md              # optional reviewed snapshot
```

## Work Plan

### Step 0: Preflight And Existing-System Alignment

Before downloading or running analysis:

1. Refresh local MethaNet lane status from `docs/current_artifact_inventory.md` and `configs/methanet_atlas_lanes.tsv` if present.
2. Identify whether any MUCC v1 files already exist locally.
3. Confirm available storage before large Zenodo downloads.
4. Confirm whether the task is source audit only, full ingest, or full MBAG integration.
5. Create a planned `cohort_run_id` and `lane_id`.

Minimum preflight output:

```text
preflight_status.tsv
lane_registration_draft.tsv
storage_budget_check.tsv
```

### Step A: Source Provenance Ledger

Create a source provenance ledger with one row per artifact or source object.

Required fields:

```text
source_artifact_id
source_type
source_name
source_url
doi_or_accession
record_id
concept_id
version
landing_page_title
expected_dataset_identity
observed_dataset_identity
license
file_name
file_size_bytes
checksum
checksum_type
download_or_access_status
access_timestamp_utc
resolution_tier
provenance_status
blocking_gap
next_action
```

Resolution tiers:

```text
exact_dataset_record
exact_file_record
exact_kbase_collection
exact_raw_read_accession
exact_sample_accession
site_level_context
paper_level_context
source_bucket
pending
blocked_mismatch
```

Acceptance criteria:

- Zenodo concept DOI and record DOI are reconciled.
- File inventory is recorded even if some large files are not yet downloaded.
- MAG count source is documented.
- Processed metatranscriptome tables are identified and units are recorded.

### Step B: MUCC v1 MAG Catalog Manifest

Build the authoritative lane manifest before downstream processing.

Required fields:

```text
lane_id
source_dataset_version
source_record_doi
source_file
source_file_checksum
source_mag_id
mag_id
proteome_id
gene_call_source
proteome_faa_path
mag_fna_path
quality_tier_source
source_completeness
source_contamination
source_gtdb_taxonomy
methanogen_flag_source
most_wanted_lineage_flag
published_denominator
manifest_include
manifest_status
exclusion_reason
claim_scope
```

Rules:

- Generate stable MethaNet `proteome_id` values for MUCC v1 if the source lacks them.
- Preserve source MAG IDs exactly in `source_mag_id`.
- Do not overwrite existing POC `proteome_id` values unless an explicit crosswalk proves identity.
- If a MAG is excluded, keep it in the manifest with `manifest_include=false` and an exclusion reason.

Acceptance criteria:

- Exactly 2,502 HQ/MQ MAG rows are either included or the deviation is reconciled.
- 3,217 and 17,333 source counts are documented as upstream denominators, not current lane denominators.
- Low-quality or non-HQ/MQ rows are visible as excluded or out-of-scope rows if encountered.

### Step C: MAG QC, Taxonomy, And Novelty Review

Ingest source QC and taxonomy, then run or stage MethaNet-compatible re-QC only as needed.

Minimum evidence:

```text
source CheckM/CheckM2 or catalog QC fields
source GTDB-Tk taxonomy
MethaNet CheckM2 re-QC status
MethaNet GUNC status
MethaNet GTDB-Tk status and release
novel lineage flags from source Table S3 or equivalent
```

Required output tables:

```text
dim_mucc_v1_mag
fact_mucc_v1_qc_source
fact_qc_checkm2
fact_qc_gunc
fact_taxonomy_gtdbtk
feature_mucc_v1_novelty
```

Interpretation rules:

- Caveat absence calls by completeness, contamination, GUNC, and annotation coverage.
- Do not present novelty as MethaNet discovery; attribute it to the MUCC source analysis unless MethaNet re-analysis independently supports it.

### Step D: Proteome, ESM2, And gLM2 Compatibility

Create a MethaNet-compatible proteome layer:

1. Prefer `OWC_HQMQ_DB_genes.faa.gz` or verified source gene calls.
2. Crosswalk proteins to MAG IDs.
3. Split source proteins into per-MAG proteome FASTA files and build a manifest compatible with `scripts/embedding/build_manifest_esm2_genome_embeddings.py`.
4. Run the ESM2 builder dry-run before production compute; record candidate count, missing FAA count, protein caps, and exact command.
5. Compute or register ESM2 embeddings with the same model/settings used for other MethaNet lanes.
6. Compute or register gLM2 contextual genome evidence with the same comparable lane settings.
7. Preserve protein counts and any mismatch between source gene calls and MethaNet gene calls.

Required outputs:

```text
dim_mucc_v1_proteome
fact_gene_calling_summary
mucc_v1_esm2_input_manifest
embedding_input_inventory
esm2_feature_store_status
glm2_context_status
latent_geometry_features
```

Do not use bespoke per-lane embedding settings unless the deviation is approved and recorded.

### Step E: Functional Pillars And MethaNet Feature Mapping

Map MUCC v1 annotations and MethaNet re-annotations into MBAG pillars.

Pillar groups:

```text
methane_machinery
methane_oxidation
methylotrophic_substrates
hydrogenotrophic_route
acetoclastic_route
electron_transfer
sulfur_competition
iron_reduction_context
fermentation_and_syntrophy
carbon_substrate_breadth
CAZy_MEROPS_substrate_processing
broad_KO_METABOLIC_traits
annotation_coverage
QC_taxonomy
```

Treat methylotrophy as first-class:

```text
methylamines
methanol
betaine
carnitine
methylated nitrogen compounds
methylated sulfur compounds
methylated oxygen compounds
methyltransferase modules
```

Required outputs:

```text
fact_mcycdb_hits
fact_scycdb_hits
fact_kofam_hits
fact_dbcan_hits
fact_bakta_features
fact_metabolic_hmm_hits
fact_metabolic_function_presence
fact_metabolic_module_presence
fact_cazy_hits
fact_merops_hits
feature_annotation_coverage
feature_methane_mechanism
feature_sulfur_competition
feature_methylotrophic_substrate
feature_mrv_mag_level
```

If using source DRAM annotations for first-pass mapping, record `source_tool=source_dram_mucc_v1` and do not merge them invisibly with MethaNet-produced facts.

### Step F: Activity Layer From Processed Metatranscriptomes

This is the main new capability.

Use processed expression tables to derive transcriptional support:

```text
owc_metat_table_mags.csv
owc_metat_table_mags_genes.csv
owc_metat_table_mags_genes_annotations.csv
```

Required grains:

```text
MAG x sample
gene x sample
MAG x functional_pillar x sample
MAG x functional_pillar
sample x functional_pillar
```

Suggested expression features:

```text
expression_detected
expression_units
expression_normalization
max_expression
mean_expression
median_expression
transcriptional_occupancy
sample_prevalence
depth_stratum_prevalence
encoded_not_expressed_flag
expressed_without_methanet_annotation_flag
source_annotation_support
```

Required outputs:

```text
fact_mucc_v1_expression_mag_sample
fact_mucc_v1_expression_gene_sample
feature_transcriptional_support_mag_pillar_sample
feature_transcriptional_support_mag_pillar
feature_activity_supported_mrv_mag_level
```

Interpretation rules:

- "Encoded" means source or MethaNet genome/protein annotation supports the function.
- "Expressed" means processed metatranscriptome evidence supports transcription in a sample or depth context.
- "Activity-supported" does not mean measured process rate.
- If expression units are unclear, label the feature `blocked_expression_units` and do not score expression strength.

### Step G: Depth, Geochemistry, Metabolomics, And Flux Context

Build a sample linkage scaffold that preserves resolution tiers.

Required entities:

```text
dim_site
dim_sampling_event
dim_sample
link_sample_mag
fact_environmental_measurement
fact_metabolite_measurement
fact_flux_measurement
link_sample_flux_window
feature_environmental_permissiveness
feature_sample_risk_readiness
```

First-class covariates:

```text
depth_cm
land_cover_patch
sampling_date
porewater_CH4
porewater_CO2
chamber_CH4_flux
AmeriFlux_CH4_or_meteorology_context
Fe_II
CEC
pH
acetate
formate
methanol
methylamines_or_related_metabolites
sulfate_or_sulfur_context
dissolved_oxygen_or_redox_proxy
temperature
water_level_or_flooding_context
```

Resolution tiers:

```text
exact_sample
sample_depth_context
site_depth_month_context
site_month_context
site_level_context
paper_level_context
modeled_estimate
pending
not_available
```

Rules:

- Depth must be preserved in every rollup where available.
- Unlinked MAGs are `not_scoreable_sample_linkage`, not dropped.
- Flux context can support validation planning and time-window matching; it does not convert MAG evidence into measured flux.

### Step H: Co-Expression Module And Predictive Neighborhood Templates

Import WGCNA/co-expression modules as feature primitives, not risk scores.

Minimum module fields:

```text
module_id
module_label_source
source_table_or_figure
member_mag_id
member_gene_id_optional
depth_association
CH4_association_direction
CH4_association_statistic
CH4_association_p_value
module_size
functional_enrichment
claim_scope
```

Expected templates to verify:

```text
surface turquoise module: CH4-negative association
deep brown module: CH4-positive association
```

Required outputs:

```text
wgcna_module_templates
feature_methane_predictive_molecular_neighborhood
candidate_module_membership_cards
```

Interpretation rule:

> A co-expression module is a methane-predictive molecular neighborhood observed in OWC. It is not a portable MethaNet risk score until source-aware validation and sample-level calibration exist.

### Step I: Wetland Reference Re-Anchoring For MBAG

Add MUCC v1 as a native wetland reference lane for MBAG bridge analysis.

Recompute or extend:

```text
ESM2 kNN neighborhoods
wetland-neighbor lookup for mangrove/MSM candidates
wetland-neighbor lookup for mangrove/Futian candidates
rumen-vs-wetland nearest-neighbor comparison
functional similarity graph
activity-supported similarity graph
bridge attestation graph sidecar
```

Required mangrove candidate outputs:

```text
mangrove_candidate_id
lane_id
nearest_mucc_v1_wetland_neighbor
nearest_poc_wetland_neighbor
nearest_rumen_reference_neighbor
latent_distance_to_mucc_v1
functional_similarity_to_mucc_v1
activity_supported_mechanism_overlap
shared_methane_mechanisms
shared_methylotrophic_substrate_features
shared_sulfur_or_redox_context
claim_status
next_validation_action
```

Rules:

- Report wetland and rumen neighbors side-by-side.
- Do not synthesize a cross-domain emission or transfer claim.
- Treat MUCC v1 as narrowing environmental distance, not eliminating target-domain validation requirements for mangroves.

### Step J: Product Artifacts And Reports

Produce reusable MethaNet intelligence artifacts.

Minimum required deliverables:

```text
source_provenance_ledger
mucc_v1_mag_catalog
mucc_v1_lane_manifest
dim_mucc_v1_mag
feature_mrv_mag_level
feature_activity_supported_mrv_mag_level
feature_transcriptional_support_mag_pillar
sample_linkage_scaffold
feature_sample_risk_readiness
wgcna_module_templates
wetland_reference_neighbor_table
bridge_mechanism_cards
validation_gap_register
claim_boundary_matrix
INTEGRATION_REPORT.md
```

Business-facing report sections:

1. What signal MUCC v1 adds to MethaNet.
2. Which product primitive it strengthens.
3. Which candidate lineages and modules deserve follow-up.
4. Which samples/depths are readiness candidates versus blocked.
5. What can be said now.
6. What remains blocked.
7. What next data acquisition or validation most increases commercial credibility.

## Candidate Card Contract

Produce candidate cards for:

- most-wanted methanogen genera and families;
- highly expressed methane-cycling MAGs;
- methylotrophic substrate candidates;
- sulfur/iron/redox competitors or modifiers;
- WGCNA module hubs;
- nearest MUCC v1 wetland neighbors for mangrove candidates;
- high-novelty lineages with strong expression or methane-neighborhood evidence.

Required fields:

```text
candidate_id
candidate_type
source_mag_id
proteome_id
lane_id
taxonomy
quality_tier
checkm2_completeness
checkm2_contamination
gunc_status
annotation_coverage_tier
encoded_methane_mechanism
expressed_methane_mechanism
methylotrophic_substrate_evidence
sulfur_or_redox_context
depth_association
module_membership
nearest_mangrove_or_reference_neighbors
evidence_status
confidence_tier
allowed_claim_wording
blocking_gaps
next_validation_action
```

Card labels:

```text
wetland_reference_anchor
activity_supported_candidate
depth_specific_candidate
methylotrophic_differentiator
module_hub_candidate
mangrove_neighbor_reference
needs_expression_units_review
needs_sample_linkage
blocked_qc
not_scoreable
```

## Sample Risk Readiness Contract

Do not assign final risk tiers. Produce readiness labels.

Required fields:

```text
sample_id
site_id
depth_cm
sampling_date
linked_mag_count
linked_methanogen_mag_count
linked_activity_supported_mag_count
molecular_methane_potential
molecular_methane_activity_support
methylotrophic_substrate_support
methane_oxidation_support
sulfur_competition_strength
iron_reduction_context
environmental_permissiveness_status
flux_validation_status
metadata_resolution_tier
abundance_or_expression_resolution_tier
uncertainty_tier
readiness_label
blocking_gaps
next_action
allowed_claim_wording
```

Readiness labels:

```text
scoreable_provisional_internal
monitor_more
needs_metadata
needs_abundance_or_coverage
needs_expression_unit_review
needs_environment
needs_flux_validation
blocked_qc
not_scoreable
```

## Acceptance Criteria

The integration is not complete until:

- The source provenance ledger records all target deposits and access timestamps.
- Zenodo concept DOI and record DOI are reconciled.
- The 2,502 HQ/MQ MAG denominator is confirmed or deviations are logged.
- The MAG manifest preserves excluded, missing, failed, ambiguous, and unlinked rows.
- Processed metatranscriptome tables are used; raw RNA reads are not reprocessed.
- Expression units and normalization are recorded before deriving expression-strength features.
- Depth is represented in the sample linkage and readiness tables.
- Methylotrophic substrate evidence exists as a first-class feature group.
- MUCC v1 is registered as its own lane and not merged with prior MUCC/Methanoregula POC records without a crosswalk.
- Mangrove candidates can carry wetland and rumen nearest-neighbor context without transfer claims.
- Candidate cards include allowed wording, evidence status, blocking gaps, and next validation action.
- The final report separates MAG potential, expression support, environmental context, and flux validation.

## Stop Conditions

Stop and ask for human review if:

- A DOI/accession resolves to a different dataset than expected.
- The Zenodo/KBase inventory cannot be reconciled with the paper.
- The 2,502 HQ/MQ MAG denominator cannot be reconstructed or explained.
- Processed metatranscriptome tables are missing, malformed, or ambiguous in units/normalization.
- A requested step would require raw-read metatranscriptome processing.
- MAG-to-sample linkage requires inference beyond documented source metadata.
- A downstream report would imply final A-E risk tiers, measured flux, carbon-credit approval, registry readiness, or source-independent transfer.
- Storage, compute, or licensing constraints make full ingest unsafe.

## Final Communication Contract

When reporting results from this prompt, always separate:

```text
What was verified
What was downloaded or staged
What was not downloaded
What denominator was used
What evidence is MAG-level potential
What evidence is expression/activity support
What evidence is sample/depth context
What evidence is flux/process validation
What can be claimed now
What remains blocked
What next action most increases MethaNet's business and scientific leverage
```

Use precise language:

- Say "MUCC v1 wetland reference lane" for this OWC catalog.
- Say "MAG-level functional potential" for genome/proteome annotations.
- Say "processed metatranscriptome support" for expression evidence.
- Say "sample risk readiness" for readiness labels.
- Say "not final MRV risk scoring" unless the required sample, abundance, environmental, uncertainty, and flux validation layers are present.

## One-Line Operating Guardrail

MUCC v1 should make MethaNet more credible by adding native wetland, depth-resolved, expression-supported evidence; it should never be used to skip the sample-level validation required for defensible blue carbon methane-risk claims.
