# MUCC v1 wetland catalog: integrated-atlas playbook

## Decision and current state

Integrate the OWC wetland catalog as a **promoted molecular-reference lane**.
It combines 2,502 reported high/medium-quality MAGs, 133 metatranscriptomes,
705 field samples, depth-resolved ecology, and in-situ methane measurements in
one methane-emitting wetland system. It should improve wetland-specific
reference coverage and hypothesis generation; it must not be treated as a
completed ecological-MRV dataset until exact sample, depth, environmental,
abundance, and flux records are reconciled.

The local lane is molecular_reference_promoted_ecological_validation_pending.
The checksum-validated Zenodo MAG archive roster has been recovered and matches
all 2,508 local entries by archive member and ZIP CRC. The paper and Zenodo
description report 2,502 HQ/MQ MAGs. The record-specific Zenodo DRAM payload
now resolves that denominator: it supplies consistent CheckM completeness and
contamination values for all 2,508 archive MAGs, including two MAGs recovered
through an unambiguous source-bin crosswalk. Applying the paper's adapted
MIMARKS screen (completeness >=50%; contamination <10%) produces exactly 2,502
HQ/MQ MAGs; the remaining six are retained with explicit out-of-scope status.
The article's Table S1-S13 spreadsheet is named and confirmed by open full
text. Its publisher listing reports 95.67 KB, which matches the 97,962-byte
payload retrieved through Europe PMC, but that payload is malformed (no
readable ZIP central directory). This size match is diagnostic evidence only,
not proof of byte identity or a usable workbook. In contrast, the intact Supplemental
Information PDF confirms the 2018 metatranscriptome cohort design: 109 field
samples, five 5-cm depth strata in August, and three strata in July and
September. The published raw labels use `D6` for 18 July/September samples
where the methods describe the deep stratum as `D5`; that discrepancy remains
explicitly unreconciled pending Table S4. Although the published KBase
narrative is browser-login gated, its public Workspace API exposes a
2,494-member GenomeSet with exact MAG-ID matches inside the Zenodo archive;
14 archive MAGs are absent. KBase provides GTDB 214.1 lineage metadata, adding
555 taxonomy-only rows, but it exposes no completeness, contamination, CheckM,
quality-tier, or N50 field. Its membership must therefore not be used to infer
the paper's 2,502 HQ/MQ roster. The six-record archive-versus-headline gap and
the eight-record headline-versus-KBase difference remain explicit quality-scope
facts, not silently removed MAGs.

For reproducibility, the six downloaded source payloads are pinned to the
record-specific [Zenodo release 8194033](https://doi.org/10.5281/zenodo.8194033),
not to the paper's concept DOI `10.5281/zenodo.8194032` alone. The live concept
record currently resolves to a newer one-file release; it is useful for
discovery but cannot reproduce the six original payloads without the pinned
record and retained checksums.

| Evidence layer | Materialized state | Claim limit |
| --- | --- | --- |
| Source MAG catalog and QC scope | 2,508 checksum-validated members from record-specific Zenodo release 8194033; direct Zenodo CheckM values are repeated-value-consistent for every MAG; 2,502 meet the paper's HQ/MQ screen and six do not | Retain the record-specific release and checksums because the concept DOI now points to a newer one-file record; per-MAG HQ/MQ scope is resolved, while NCBI accession mapping remains unresolved |
| Public KBase catalog | 2,494 exact MAG-ID matches; 14 Zenodo-only MAGs; 555 supplemental GTDB 214.1 lineage rows; 1,299 source/KBase taxonomy differences retained | Identity and supplementary taxonomy only; neither membership nor absence is a quality-tier assignment |
| Rank-aware atlas taxonomy | 2,503 MAGs with a source-primary/KBase-missing-rank-fallback lineage; 5 unavailable in both sources; 1,173 rank disagreements retained | MAG/proteome context only; use source/KBase provenance before any taxon-stratified interpretation |
| ESM2 | 2,501 finite 1,280-dimension embeddings | Molecular similarity only |
| gLM2 | 2,508 multi-window context summaries | Genome-context stability only; not independent functional validation |
| Processed expression | 1,948 MAGs across 133 samples | Processed relative expression support, not abundance or flux |
| NCBI BioProject crosswalk | 107/133 expression labels linked to authoritative project-title suffixes; 26 preserved unmatched | Sequence-project context only; no run/BioSample, collection date, depth-in-cm, or flux link |
| NCBI SRA package crosswalk | 130/133 entity-validated expression-label matches to SRA/BioSample/experiment/run packages in SRP456134; 107 exact collection dates, 23 year-only dates, and 3 unresolved labels | Package identity and date metadata only. No package reports usable depth; it does not establish environmental, abundance/read-coverage, or flux-window joins. Declared WGS packages require assay reconciliation before pooling with RNA-Seq expression. |
| JGI Sample QC crosswalk | 107/133 BioProject-linked labels have a unique exact JGI Sample ID, receipt timestamp, and QC result; 26 labels remain unmatched | Operational identity/receipt/QC evidence only; receipt time is not collection date, and measured depth, environment, and flux links remain unresolved |
| JGI Data Portal catalog crosswalk | 107/133 exact source-label-specific JGI catalog record pairs (80 direct labels; 27 controlled `July_` to `Jul_` aliases); 8,633 indexed assets, 8,619 marked `PURGED`; 26 labels remain unresolved | Independent catalog/processing and source-record geolocation evidence only. The public indexed assets require authenticated JGI download and do not report collection time, depth, chemistry, or flux links. |
| Supplemental-methods design context | 109 2018 cohort rows: 91 direct raw-depth-code contexts, 18 D6/D5 reconciliation-pending; 24 legacy label-only rows | Methods-derived context only; it is not a Table S4 row, measured field metadata, or an ecological/flux join |
| ESS-DIVE field observations | 275 chamber records; 5,280 porewater CH4 records (1,563 valid concentrations) | Source-staged site/date/depth evidence only; no sequence-sample link is assumed |
| ESS-DIVE gap-filled tower flux context | 29,280 checksum-validated half-hourly CH4/CO2 tower observations across 2015, 2016, 2020, 2021, and 2022 (all CH4 values present) | Site/time context only. Its 2015-2016 temporal overlap with MUCC does not establish a sequencing-sample, plot, depth, or flux-window join. |
| Source-aware neighbors | 2,501 embedded MAGs | Reference anchors, not ecological transfer proof |
| FlashWeave network | 694 MAG-to-MAG, 61 metadata-involving edges from 133 samples/300 MAGs | Exploratory conditional associations; stability and exact ecological covariates remain blocked |
| Secondary WGCNA modules | R 4.4.3/WGCNA 1.74 source-method-aligned comparator: 1,948 MAGs / 133 retained samples, three non-grey modules (1,141 turquoise; 479 blue; 267 brown) and 61 unassigned grey MAGs; fixed source-reported power 7 yields SFT R-squared 0.6071 | Descriptive coexpression comparison only; it retains the unidentified source outlier and cannot reproduce source trait correlations or establish ecology/flux |
| Conditional-network explorer | 694 association contexts and 300 node contexts; 126 edges meet the stability-and-taxonomy navigation filter; 388 edges expose a source/KBase taxonomy disagreement at one or both endpoints | Query/filter surface only; neither stability, taxonomy, nor marker terms establish interaction, ecology, or methane flux |

The queryable warehouse is
results/functional_metagenomics/mucc_v1_owc_wetland_20260626/cohort_warehouse/functional_atlas.duckdb.
It has 61 Parquet-backed tables and a machine-readable promotion-gate report.

## Integration architecture

~~~mermaid
flowchart LR
  A[Published OWC source payloads] --> B[Source manifests and MAG catalog]
  B --> C[Molecular evidence: ESM2 and gLM2]
  B --> D[Processed expression: MAG x sample]
  C --> E[Source-stratified reference neighbors]
  D --> F[FlashWeave input contract]
  B --> G[Exact source sample, depth, environmental, and flux records]
  E --> H[Promoted molecular-reference cards]
  G --> I[Ecological validation tables]
  F --> J[Exploratory conditional association network]
  H --> K[Integrated atlas]
  I --> K
  J --> K
~~~

The canonical molecular key is proteome_id; mag_id remains the source-facing
identifier. Sample evidence stays at sample/metagenome grain. A MAG-level
functional potential, processed transcriptional support, or embedding neighbor
must never be upcast to a sample-level ecological statement.

## Implemented contracts and artifacts

Run the supported builders in this order:

~~~bash
./.venv/bin/python scripts/external/stage_essdive_owc_flux.py
./.venv/bin/python scripts/external/stage_essdive_owc_gapfilled_tower_flux.py
./.venv/bin/python scripts/external/stage_mucc_v1_kbase_public_catalog.py
./.venv/bin/python scripts/reports/build_mucc_v1_taxonomy_projection.py
./.venv/bin/python scripts/external/stage_mucc_v1_zenodo_qc_reconciliation.py
./.venv/bin/python scripts/reports/build_mucc_v1_source_recovery_ledger.py
./.venv/bin/python scripts/external/stage_mucc_v1_ncbi_bioproject_crosswalk.py
./.venv/bin/python scripts/external/stage_mucc_v1_ncbi_sra_sample_crosswalk.py --batch-size 20
./.venv/bin/python scripts/external/stage_mucc_v1_jgi_sample_crosswalk.py --jgi-workers 2
./.venv/bin/python scripts/external/stage_mucc_v1_jgi_data_portal_catalog.py --workers 3
# Run only with an author/publisher-supplied canonical crosswalk; never generate this input from labels.
./.venv/bin/python scripts/external/stage_mucc_v1_authoritative_ecological_crosswalk.py --input <authoritative_crosswalk.tsv>
./.venv/bin/python scripts/reports/build_mucc_v1_mrv_readiness_features.py
./.venv/bin/python scripts/reports/build_mucc_v1_network_analysis_inputs.py
# Run network_analysis/run_flashweave.jl with Julia and FlashWeave before the next command.
./.venv/bin/python scripts/reports/summarize_mucc_v1_flashweave_network.py
./.venv/bin/python scripts/reports/run_mucc_v1_flashweave_stability.py --iterations 20
./.venv/bin/python scripts/reports/build_mucc_v1_flashweave_atlas_explorer.py
# Use a compatible isolated R + WGCNA runtime, then validate its outputs.
<Rscript-with-WGCNA> results/functional_metagenomics/mucc_v1_owc_wetland_20260626/network_analysis/run_wgcna_secondary.R
./.venv/bin/python scripts/reports/summarize_mucc_v1_wgcna_secondary.py
./.venv/bin/python scripts/reports/promote_mucc_v1_integrated_atlas.py
./.venv/bin/python scripts/reports/build_mucc_v1_source_scaffold_warehouse.py
./.venv/bin/python scripts/reports/audit_mucc_v1_integration_completion.py
./.venv/bin/python scripts/reports/build_mucc_v1_wetland_dashboard.py
~~~

The promotion builder writes durable evidence without overwriting source facts:

- functional_features/feature_mucc_v1_embedding_status.tsv
- functional_features/feature_mucc_v1_glm2_context.parquet
- functional_features/feature_mucc_v1_zenodo_source_qc.tsv
- bridge_reanchoring/integrated_atlas/wetland_reference_neighbor_{edges,summary}.tsv
- environmental_metadata/feature_mucc_v1_sample_ecological_readiness.tsv
- environmental_metadata/feature_mucc_v1_sample_methods_design_context.tsv
- environmental_metadata/link_mucc_v1_sequence_bioproject_sample.tsv
- environmental_metadata/link_mucc_v1_sequence_sra_sample.tsv
- environmental_metadata/link_mucc_v1_sequence_jgi_sample.tsv
- environmental_metadata/link_mucc_v1_sequence_jgi_data_portal.tsv
- environmental_metadata/link_mucc_v1_sequence_authoritative_ecology.tsv (only after strict source-input validation)
- environmental_metadata/feature_mucc_v1_authoritative_ecology_readiness.tsv (only after strict source-input validation)
- environmental_metadata/fact_mucc_v1_essdive_{chamber_flux,porewater_ch4}.tsv
- environmental_metadata/fact_mucc_v1_essdive_gapfilled_tower_ch4_flux.tsv
- reconciliation/mucc_v1_{denominator,mag}_reconciliation.tsv
- reconciliation/mucc_v1_zenodo_source_qc_reconciliation.tsv
- reconciliation/mucc_v1_methods_sample_design_reconciliation.tsv
- source_audit/mucc_v1_source_metadata_recovery_ledger.tsv
- source_audit/zenodo_mucc_v1_qc_source_manifest.tsv
- source_audit/kbase_public_workspace_147022/mucc_v1_kbase_public_catalog_reconciliation.tsv
- functional_features/feature_mucc_v1_taxonomy_projection.tsv
- reconciliation/mucc_v1_taxonomy_projection_summary.tsv
- network_analysis/feature_mucc_v1_flashweave_node_atlas_context.tsv
- network_analysis/fact_mucc_v1_flashweave_edge_atlas_context.tsv
- network_analysis/mucc_v1_flashweave_atlas_explorer_summary.tsv
- reports/mucc_v1_integrated_atlas_promotion_gates.tsv
- candidate_cards/mucc_v1_promoted_molecular_reference_cards.tsv

Rows remain explicit when evidence is missing or non-comparable. The promotion
state is not a mechanism call, MRV score, or methane-risk tier.

## Reconciliation protocol

1. Retain the checksum-validated Zenodo archive roster (all 2,508 entries,
   member paths, and CRCs) from record-specific release 8194033 as the source
   membership crosswalk. Do not refresh from the mutable concept DOI alone.
   Use the direct Zenodo DRAM QC reconciliation, which applies the published
   CheckM completeness >=50% and contamination <10% screen to all 2,508 MAGs,
   to distinguish the 2,502 paper-defined HQ/MQ rows from six explicit
   archive-scope exceptions. The staged Europe PMC Table S1-S13 copy remains
   `malformed_no_central_directory`, so use a parseable Table S3 or
   author-provided roster only when its additional accession/bin-stat fields
   are needed.
2. Retain the public KBase Workspace reconciliation as a separate identity and
   supplementary-taxonomy layer: 2,494 exact MAG-ID matches and 14
   Zenodo-only MAGs. Preserve source and KBase GTDB values side by side; do not
   silently overwrite the 1,299 differences or use KBase membership to assign
   HQ/MQ quality.
3. Normalize archive member, local mag_id, proteome_id, and NCBI assembly/bin
   identifiers into a crosswalk. Never use row order as a mapping.
4. Use `published_mq_hq_membership_status` and
   `source_qc_value_consistency_status` on every local MAG. Retain the six
   non-qualifying archive rows rather than silently dropping them; filter them
   only for analyses that require the paper-defined 2,502-MAG denominator.
5. Retain the SRA package accession and reported collection-date precision for
   every entity-validated expression label. Then reconcile each analysis-ready
   sample to site/plot, measured depth in cm, assay provenance, and the
   corresponding field-measurement window. Use a link table when a flux record
   covers a time interval rather than a sequencing instant.
6. Build dim_sample, fact_environmental_measurement, fact_flux_measurement,
   and link_sample_flux_window only where the source provides a key or a
   documented deterministic mapping. Otherwise retain
   blocked_missing_authoritative_mapping.

The archive-membership and published HQ/MQ quality-scope reconciliations are
complete. The ecological gate is resolved only after steps 5 and 6; it cannot
be inferred from the 133-column expression matrix.

The NCBI title crosswalk provides sequence-project context for 107/133 labels
(80 exact suffixes and 27 deterministic `July`→`Jul` aliases). It advances
identity provenance but intentionally does not establish a field-observation
window.

The direct NCBI SRA crosswalk is the stronger package-level tier: it resolves
130/133 expression labels to entity-validated SRA/BioSample/experiment/run
packages in `SRP456134` and preserves all three unresolved labels as rows. The
validation requires an exact source label (or deterministic `July`→`Jul`
alias) in the SRA package title and Old Woman Creek context; it deliberately
rejects the entity-conflicted BioProject-XML `biosample_id` encountered during
recovery. Of the 130 packages, 107 provide an exact collection date and 23
provide year-only precision. None reports a usable depth value or a field
chemistry, porewater, or chamber-flux window, so date evidence alone does not
unblock ecological validation.

The SRA metadata declares 107 RNA-Seq packages and 23 WGS packages. This is a
source-metadata compatibility flag, not a correction to the published assay
interpretation: the WGS-declared packages must be reconciled to the processed
expression pipeline before they are pooled with RNA-Seq samples in any
expression, network, or ecological comparison.

The additional ESS-DIVE DOI `10.15485/2500238` release supplies 29,280
checksum-validated, half-hourly gap-filled eddy-covariance CH4/CO2 tower
observations for US-OWC, spanning 2015-2016 and 2020-2022. It is now a
queryable `fact_mucc_v1_essdive_gapfilled_tower_ch4_flux` source-context
table, with source timestamps deliberately retained as unzoned rather than
converted to UTC. The source's 2015-2016 coverage is temporally relevant to
the MUCC period, but it carries no sequencing-sample accession, plot, or depth
key. It therefore remains unlinked until a publisher/author mapping supplies
both an exact sample correspondence and a documented tower-context rule.

### Minimum authoritative recovery package for ecological promotion

Do not infer this package from label patterns, calendar proximity, or shared
site names. A publisher-original Table S4 (or an author-provided CSV/TSV
export with the same semantics) must provide, for every sequence sample where
available: the source sample label and accession; collection date and time;
site/transect/plot and core; measured depth with unit and depth reference;
assay and library provenance; paired geochemistry/metabolite record IDs and
units; and the spatial-temporal key linking a sample to porewater or chamber
CH4 observations. A separate assay reconciliation note is required for each
of the 23 SRA packages declared as `WGS` before its processed expression value
can be pooled with declared `RNA-Seq` samples. The source should also state
replicate and missing-value semantics. Until this package exists, retain every
sequence-to-environment/flux relationship as
`blocked_missing_authoritative_mapping`.

For a send-ready specification of the corrected Table S4-equivalent package,
see the [authoritative-crosswalk data-recovery handoff](mucc_v1_authoritative_crosswalk_data_request.md).

The JGI crosswalk is a narrower sample-identity tier. Each of the 107
BioProject-linked labels resolves to a final JGI delivery portal and a unique
exact JGI Sample QC record, including operational sample-receipt and QC
timestamps. Receipt time must never be promoted to collection datetime. The
crosswalk does not contain measured depth, environmental measurements, or a
methane-flux window, so it cannot unblock ecological validation by itself.
An entity-conflicted non-OWC `biosample_id` surfaced in a BioProject XML spot
check and is explicitly excluded. Only the package-level SRA XML records that
pass exact label and Old Woman Creek context validation are staged; neither
source is a substitute for an authoritative OWC field-metadata roster.

The independently queried public JGI Data Portal provides a complementary,
source-label-specific provenance tier. It resolves 107 of 133 expression labels
to exactly one JGI analysis record (with an IMG taxon identifier) and one
expression record under proposal `504205`; 80 labels match directly and 27 use
the controlled `July_`/`Jul_` alias documented in the catalog. Across those
pairs, the index reports 8,633 assets, of which 8,619 are marked `PURGED`; an
attempted file download returns an authentication requirement even when the
catalog calls the data unrestricted. Treat this as catalog and processing
provenance plus the catalog's source-record coordinate only—not as sequence-file
availability, collection time, depth, chemistry, or a field-observation join.
The 26 unmatched labels remain explicit, including the legacy 2014/2015 subset.

The recovered Supplemental Information PDF adds a bounded methods-design tier.
It establishes that the 109 labels matching the 2018 expression design occupy
July/August/September cohorts, with direct 5-cm depth-code contexts for 91
rows. For the remaining 18 July/September `D6` labels, the 20-25 cm interval
is a cohort-design slot inferred from the published three-depth design, while
the raw `D6` versus published `D5` code mismatch is retained as a reconciliation
blocker. The 24 2014/2015 labels remain label-only. None of these contexts is
an exact collection date, field chemistry, porewater, chamber-flux, abundance,
or read-coverage join; Table S4 remains required for that promotion.

### Machine-enforced authoritative crosswalk contract

`scripts/external/stage_mucc_v1_authoritative_ecological_crosswalk.py` is the
only promotion route for a recovered publisher/author ecology crosswalk. Its
input must be a tab-separated file with one non-empty, unique `mapping_id` per
declared relationship. It retains incomplete and missing evidence as rows, and
it never generates links from label parsing or calendar/site proximity.

| Required evidence group | Required canonical fields |
| --- | --- |
| Sequence identity and location | `source_sample_column`, `authoritative_sample_id`, `collection_datetime` (full ISO time), `site_id`, `core_or_plot_id`, `depth_cm`, `depth_reference` |
| Assay and coverage | `sequence_assay_type`, `assay_reconciliation_status`, `mag_abundance_or_read_coverage_record_id`, `mag_abundance_or_read_coverage_units` |
| Environmental record | `environment_source`, `environment_record_id`, `environment_measurement_datetime`, `environment_measurement_units` |
| Flux window | `flux_source`, `flux_observation_id`, `flux_measurement_type`, `flux_units`, `flux_window_start_datetime`, `flux_window_end_datetime` |
| Replication and provenance | `replicate_id`, `uncertainty_record_id`, `uncertainty_method`, `source_evidence_status`, `missingness_status`, `source_url` |

Only `source_evidence_status=authoritative_complete` rows can be eligible, and
only when timestamps are ordered, depth is finite and non-negative, the assay
is validated metatranscriptomic or explicitly reconciled WGS-to-expression,
and any ESS-DIVE record identifiers exist in the staged source tables. Valid
rows are labelled
`eligible_for_grouped_ecological_validation_not_final_MRV`; they do not prove
a MAG-level flux effect, causal mechanism, final risk tier, or crediting claim.

## Ecological validation plan

Each analysis-ready sample needs a source accession, collection date/time,
site/plot, depth in cm, assay, MAG abundance or read coverage, documented
expression units, environmental measurements with time windows, a flux/process
measurement, and uncertainty/replicate metadata. Do not substitute depth-context
labels for measured depth or relative expression for read coverage.

After the contract is complete, use site/plot, depth stratum, season/month, and
year as grouped leakage-resistant folds. Compare environmental-only,
molecular-only, and combined models. Report held-out calibration, intervals,
directional stability, missingness by group, and sensitivity to prevalence and
compositional transform. Use permuted-label and context-matched non-marker
controls. A final A-E risk tier remains out of scope until these prerequisites
are met.

## Network analysis: FlashWeave primary, WGCNA secondary

FlashWeave is the primary exploratory network method because it estimates
conditional/direct associations and can condition on environmental and technical
metadata. WGCNA is retained only as a secondary descriptive coexpression-module
analysis; it does not replace conditional association inference.

| Setting | Current value | Rationale |
| --- | --- | --- |
| Samples | 133 | All processed metatranscriptome columns; still a small network cohort |
| FlashWeave MAG features | 300 | At least 10% prevalence, then log1p-CPM variability while retaining taxonomy-screened methanogen candidates |
| Conditioning metadata | Month, site/land-cover, depth-context code | Available scaffold covariates, not substitutes for measured depth or chemistry |
| FlashWeave mode | sensitive=true, heterogeneous=false, max_k=3, FDR enabled | One wetland and 133 samples do not justify multi-habitat FlashWeaveHE |
| WGCNA MAG features | 1,948 | All MAGs with non-zero processed expression, matching the paper's reported recruited-transcript numerator |
| WGCNA | Secondary source-method-aligned R comparator | Signed-hybrid, fixed power 7, minimum module size 50, merge height 0.3; the publication reports 132 samples after an unidentified outlier, so the available 133-row run is not an exact reproduction |

Run the generated script in a Julia environment with FlashWeave installed:

~~~bash
julia results/functional_metagenomics/mucc_v1_owc_wetland_20260626/network_analysis/run_flashweave.jl
~~~

The initial controlled run completed with FlashWeave 0.19.2: 694 MAG-to-MAG
edges, 61 metadata-involving edges, 300 connected MAG nodes, and FDR enabled
during inference. The raw result is split into MAG-to-MAG and metadata-edge
tables so conditioning-variable links are not silently mixed into the MAG
network. Per-edge q-values are not emitted by the FlashWeave edgelist format.
The stability runner performs 20 deterministic 80% scaffold-stratified
subsamples, records each exact sample selection, and attaches each baseline
edge's selection frequency. This is an exploratory reproducibility check, not
an ecological validation substitute.

The secondary WGCNA comparator has been executed with R 4.4.3 and WGCNA 1.74
on all 1,948 non-zero-expression MAGs and all 133 available samples. It aligns
to the paper's reported signed-hybrid network, fixed power 7, minimum module
size 50, and merge height 0.3. The source reports 132 samples after screening
one outlier, but does not identify that row in the accessible material; the
comparator therefore retains all 133. Its power-7 scale-free fit is 0.6071 and
it finds three non-grey descriptive modules (1,141 turquoise, 479 blue, and
267 brown) plus 61 grey MAGs. That differs from the source's reported module
structure and is intentionally recorded as a controlled, partial
method-comparison—not a reproduction, trait-correlation result, ecological
interpretation, or replacement for FlashWeave conditional associations.

Before an edge is shown in the atlas, require FDR output, bootstrap or grouped
leave-out stability, prevalence sensitivity, and a null/permutation comparison.
Each edge must retain method, transform, covariates, stability statistic, and
claim boundary. Interpret edges as conditional associations in processed data,
not interactions, causal effects, or methane-flux mechanisms.

## Ocean-M-inspired visualization and analytics

Ocean-M motivates source-aware filters, prevalence-gated networks, interactive
edge selection, and companion taxonomic/co-occurrence tables. The implemented
explorer materializes rank-aware source-primary/KBase-fallback taxonomy, source
annotation marker counts, processed-expression prevalence, and the existing
selection-stability value at both MAG endpoints. It exposes 126 associations
with both endpoint lineages and at least 0.70 selection frequency, while
retaining 388 associations with explicit source/KBase taxonomy-conflict exposure
for provenance-aware filtering. These are navigation fields for exploratory
conditional associations only; no field is an ecological, causal, or flux
claim. Build only from validated warehouse tables:

Ocean-M applies a prevalence filter of at least 5% within its environmental
categories. The local MUCC explorer instead retains features at 10% prevalence
across its single 133-sample wetland cohort, then caps at 300 by log1p-CPM
variability while retaining taxonomically screened methanogens. This is a
small-cohort adaptation, not a claim of equivalence to Ocean-M's multi-biome
global interaction network.

The implemented starting surface is a portable-dashboard **artifact** at
`reports/mucc_v1_wetland_atlas_dashboard_artifact.json`. It is generated from
the DuckDB warehouse, is explicitly marked `partial`, and includes warehouse
coverage, the molecular-reference review queue, promotion gates, reconciliation
checks, and all remaining ecological blockers. It deliberately shows zero exact
sequence-to-environment/flux joins rather than filling the gap with an inferred
mapping. Package it with the Data Analytics portable reader only in an
environment with a supported Node runtime:

~~~bash
node <data-analytics-plugin>/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input results/functional_metagenomics/mucc_v1_owc_wetland_20260626/reports/mucc_v1_wetland_atlas_dashboard_artifact.json \
  --output results/functional_metagenomics/mucc_v1_owc_wetland_20260626/reports/mucc_v1_wetland_atlas_dashboard.html
~~~

1. **Lane status strip** — source version, local/published denominator,
   molecular coverage, ecological-link coverage, and persistent claim boundary.
2. **MAG reference card** — taxonomy, source QC, marker terms, processed
   expression, ESM2/gLM2 status, source-stratified neighbors, and blocking gaps.
3. **Depth-aware sample explorer** — sample × MAG values with exact depth,
   units, mapping confidence, and unavailable values visibly distinguished.
4. **Conditional-network explorer** — now materialized as node and edge context
   tables, with filters for taxon, marker class, processed-expression prevalence,
   source, and stability. Depth becomes selectable only after an authoritative
   sample crosswalk is recovered; selected nodes reveal evidence, never an
   ecological conclusion.
5. **Reconciliation dashboard** — denominator categories, sample-link coverage,
   flux-alignment quality, held-out results, and blocked rows.

## Sources and stop conditions

- [Borton et al., *mSystems*: wetland methane catalog](https://journals.asm.org/doi/10.1128/msystems.00680-25)
- [PMC open-access record and supplemental assets](https://pmc.ncbi.nlm.nih.gov/articles/PMC13289110/)
- [JGI Genome Portal](https://genome.jgi.doe.gov/) public project/Sample QC records
- [FlashWeave methods paper](https://doi.org/10.1016/j.cels.2019.08.002) and [FlashWeave.jl](https://github.com/meringlab/FlashWeave.jl)
- [Ocean-M database, *Nucleic Acids Research*](https://academic.oup.com/nar/article/54/D1/D813/8307366)
- [AmeriFlux US-OWC site record](https://ameriflux.lbl.gov/sites/siteinfo/US-OWC)
- [ESS-DIVE](https://www.ess-dive.lbl.gov/)

Do not mark this lane ecologically validated, assign methane-risk tiers, make a
crediting claim, or describe a reference neighbor as source-independent transfer
evidence until the MAG roster is reconciled, exact sample/depth mappings and
environment/flux alignments are authoritative, abundance/read coverage and
uncertainty exist, and grouped held-out validation is complete.
