# Data Aggregation Strategy

Date: 2026-06-12
Documentation refresh: 2026-06-25

Scope: integrative analysis for the 662-genome MethaNet POC cohort, combining
geometry-aware ESM2 protein embeddings with MAG-level functional genomics from
the Apollo-3 annotation stack. The same design is now extended as a multi-lane
union across rumen, wetland/MUCC, mangrove/MSM, and mangrove/Futian molecular
evidence.

## Purpose

The 662-genome POC established a strong latent separation between rumen and wetland proteomes and identified cross-ecosystem bridge candidates. The next scientific step is to explain that latent geometry with independent genome evidence: MAG quality, taxonomy, methane and sulfur pathway functions, substrate-processing capacity, broad metabolic traits, and annotation coverage.

This strategy defines how to aggregate those layers into one coherent, auditable analysis table without letting source leakage, duplicated IDs, or annotation missingness masquerade as biology.

## Authoritative Cohort Backbone

The cohort backbone is the 662-row crosswalk:

```text
ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv
```

The local provenance audit confirmed full file coverage:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/
```

Key audit results:

| item | result |
| --- | ---: |
| final embedded proteomes | 662 |
| rumen proteomes | 555 |
| wetland/MUCC proteomes | 107 |
| local MAG FASTA matches | 662 / 662 |
| local proteome FAA matches | 662 / 662 |
| unmatched final proteomes | 0 |

Canonical identifier:

```text
proteome_id
```

`proteome_id` is equivalent to the original POC sample ID and must be carried through every table. For MAG functional processing, use the proposed manifest:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.proposed.tsv
```

Recommended primary matching keys:

| data layer | preferred key | backup keys | notes |
| --- | --- | --- | --- |
| ESM2 geometry artifacts | `proteome_id` | `sample` | canonical POC ID |
| proteome FAA | `proteome_faa_stem` or `proteome_id` | `proteome_faa_basename` | exact local match for all 662 |
| MUCC MAG FASTA | `mag_id_candidate` | `proteome_faa_stem`, basename without `mucc__` | local preferred path is `data/assemblies` |
| rumen MAG FASTA | `filename`, then `mag_id_candidate` | `analysis_alias`, basename | local preferred path is `data/blue_catalyst_poc/rumen/raw` |
| PRJEB31266 provenance | `source_analysis_accession` | `analysis_alias`, `filename` | provenance key, not local filename key |

The excluded coassembly must remain excluded from final integrated tables unless explicitly included as a negative-control record:

```text
mucc__PPR_1022_P7D_M_E_concat_coassembly_mesocosms_megahit_bin.197
```

## Multi-Lane Extension

The original 662-row POC remains the source-audited backbone for rumen +
wetland/MUCC bridge-candidate interpretation. The expanded MBAG atlas now also
needs explicit target-domain lanes for mangrove/MSM and mangrove/Futian MAGs.

| Lane | Backbone rule | Current molecular layers | Special caveat |
| --- | --- | --- | --- |
| POC rumen + wetland/MUCC | 662-row ESM2 backbone, with 625 MAG/bin-comparable units and 37 assembly-context units | ESM2, functional warehouse, gLM2, metadata, attestation graph | source and ecosystem are still confounded; assembly-context units are not MAG-bin feature rows |
| Mangrove/MSM expansion | 1,428 local candidate manifest under `data/external/msm_china_2025/` | ESM2 1,428/1,428, gLM2 1,428/1,428, functional tranche 1,427/1,428 complete at the 2026-06-25 snapshot | local 1,428 candidates must be reconciled to the paper-reported 966 final medium/high-quality MAG denominator before sample-level rollups |
| Mangrove/Futian 2026 expansion | 3,404 phase-1 rMAG source-lane manifest under `data/external/futian_mangrove_2026_qi/`, with 3,156 ready payload rows and 248 gap rows | ESM2 3,156/3,156, gLM2 3,156/3,156, functional annotation active: 302/312 archaea complete at the 2026-06-25 15:39 UTC live refresh while the current external report freeze remains fixed at 300/312, and bacteria shards queued | site/month/habitat metadata are strong, but depth-resolved MAG-to-sample assignment, abundance/read coverage, and flux/process validation remain pending |
| Multi-view atlas union | explicit left join across lane-specific backbones; current release freezes should be generated from `scripts/reports/build_methanet_3view_payload_freeze.py` | ESM2 + functional + gLM2 + QC/taxonomy + provenance + report features | never infer sample-level risk without MAG-to-sample mapping, abundance/read coverage, environmental covariates, and validation |

Recommended union identity fields:

```text
atlas_unit_id
proteome_id
lane_id                      # poc_core, msm_china_2025, futian_mangrove_2026_qi
lane                         # poc_rumen, poc_wetland_mucc, mangrove_msm, mangrove_futian
source_project
source_dataset
mag_id
unit_scope                   # mag_bin, assembly_context, candidate_mag
local_archive_denominator
published_quality_denominator
ready_payload_denominator
gap_payload_denominator
esm2_status
functional_status
glm2_status
metadata_resolution
sample_rollup_status
claim_status
```

The operational uniqueness key for the expanded atlas is:

```text
lane_id + proteome_id
```

All high-level reports should expose `lane`, `unit_scope`, `functional_status`,
and `sample_rollup_status` so that a completed MAG-bin evidence object cannot be
mistaken for a scoreable environmental sample.

## Aggregation Principle

Build the integrated dataset as a set of left joins onto the 662-row cohort backbone. No downstream layer is allowed to define the cohort. Functional failures and missing annotations should become explicit status fields, not dropped rows.

The invariant is:

```text
one row per proteome_id in the master MAG-level table
```

Feature matrices may be wide or sparse, but all must preserve:

```text
proteome_id
mag_id
source
ecosystem
domain
mag_fasta
proteome_faa
```

## Data Layers

### Layer 0 - Identity And File Provenance

Source tables:

- `embedded_662_proteome_id_crosswalk.tsv`
- `poc_662_functional_mag_manifest.proposed.tsv`
- original POC metadata: `embedding_metadata.tsv`, `embedding_projection_clusters.tsv`, bridge ranking artifacts

Required output:

```text
cohort_identity.tsv
```

Minimum columns:

```text
proteome_id
sample
mag_id
source
ecosystem
domain
mag_fasta
mag_fasta_basename
proteome_faa
proteome_faa_basename
source_analysis_accession
analysis_alias
source_filename
n_proteins_used
embedded_final_662
match_key
match_status
```

Gate:

- exactly 662 rows
- no duplicate `proteome_id`
- no duplicate `mag_fasta` unless explicitly documented as replicate or copied source path
- final excluded coassembly absent

### Layer 1 - Latent Geometry

Purpose: preserve the POC signal as the hypothesis-generating layer.

Inputs:

- ESM2 embedding matrix: `genome_embeddings.npz`
- embedding metadata: `embedding_metadata.tsv`
- embedding projection/cluster outputs
- bridge candidate rankings and alpha-transfer metrics

Required output:

```text
latent_geometry_features.tsv
```

Recommended columns:

```text
proteome_id
embedding_run_id
embedding_index
esm2_dim
pc1
pc2
umap1
umap2
tsne1
tsne2
cluster_id
cluster_label
bridge_rank
alpha_transfer_score
bridge_entropy
opp_neighbor_fraction
knn_k
trajectory_axis_score
latent_qc_status
```

Rules:

- Keep raw 1280-dimensional embeddings in `.npz`/Parquet, not in human-readable TSVs.
- Store low-dimensional coordinates and bridge scores in a compact feature table.
- Do not use source labels to create final biological conclusions without the source-aware controls in Layer 7.

### Layer 2 - MAG QC And Taxonomy

Purpose: prevent bridge candidates from being artifacts of low-quality or chimeric MAGs.

Tools:

- CheckM2
- GUNC ProGenomes3
- GTDB-Tk R232
- optional dRep for dereplication when scaling beyond the 662 set

Required outputs:

```text
mag_qc_integrated.tsv
taxonomy_resolved.tsv
derep_clusters.tsv
```

Recommended MAG-level features:

```text
proteome_id
mag_id
total_bp
contigs
n50
gc_pct
completeness
contamination
checkm2_model
gunc_pass
gunc_css
gunc_rss
gunc_contamination_portion
gtdb_domain
gtdb_phylum
gtdb_class
gtdb_order
gtdb_family
gtdb_genus
gtdb_species
taxonomy_status
qc_status
```

Gate:

- every top bridge candidate has QC and taxonomy status
- high-contamination or GUNC-failing genomes are down-weighted or blocked from high-confidence mechanism claims

### Layer 3 - Gene Calling And Protein Accounting

Purpose: make every annotation layer comparable by normalizing to a canonical protein set.

Tool:

- Prodigal `-p meta`

Required output:

```text
gene_calling_summary.tsv
```

Recommended columns:

```text
proteome_id
mag_id
predicted_proteins
predicted_cds
total_aa
median_aa_length
proteome_faa_proteins_used
protein_count_delta
protein_count_status
gene_calling_status
```

Rule:

- If the POC proteome FAA and new Prodigal FAA differ, record the difference. Do not silently replace one with the other.
- Functional annotations should state whether they used the POC proteome FAA or newly predicted proteins.

### Layer 4 - Methane And Sulfur Mechanism Evidence

Purpose: determine whether latent bridge candidates have mechanistic methane relevance.

Tools/databases:

- existing 12-marker HMM panel
- MCycDB 2021
- SCycDB 2020Mar
- KOfam pathway support
- METABOLIC-G trait summaries

Required outputs:

```text
methane_marker_panel.tsv
mcycdb_hits.tsv
mcycdb_pathway_completeness.tsv
scycdb_hits.tsv
sulfur_competition_features.tsv
methane_pathway_completeness.tsv
electron_transfer_features.tsv
```

Core mechanism feature groups:

| group | expected evidence |
| --- | --- |
| methanogenesis core | `mcrA`, `mcrB`, `mcrG`, MCR complex status |
| hydrogenotrophy | hydrogenase, formate dehydrogenase, CO2 reduction modules |
| methylotrophy | `mtaB`, `mttB`, `mtbA`, methyltransferase modules |
| aceticlastic route | acetate module evidence where detectable |
| methane oxidation | `pmoA`, `mmoX`, AOM-associated markers |
| electron transfer | HdrABC, Mvh, Eha/Ehb, ferredoxin context |
| sulfur competition | `dsrAB`, `aprAB`, `sat`, `sox`, `sqr`, SCycDB pathway coverage |

Aggregation rules:

- Keep raw hit tables at gene-hit grain.
- Collapse to MAG-level features only after applying explicit thresholds.
- Record both count and confidence:

```text
hit_count
best_bitscore
best_evalue
best_identity
best_coverage
module_completeness
call_status
```

Mechanism classes:

```text
methane_relevant_high_confidence
methane_relevant_partial
substrate_flexible
sulfur_associated
unclear_function
likely_artifact_or_qc_blocked
```

Gate:

- no top bridge candidate can remain unclassified
- absent evidence must be separated from unannotated evidence

### Layer 5 - Broad Functional Ecology

Purpose: test whether bridge candidates are explained by broader ecology rather than narrow marker hits.

Tools/databases:

- KOfamScan
- dbCAN V5
- Bakta
- METABOLIC-G
- eggNOG v2 as an optional validated sidecar when broad orthology/EC/COG
  coverage is needed
- DRAM/DRAM2 only after repair/provisioning; not a blocker

Required outputs:

```text
ko_matrix.tsv
module_completeness.tsv
dbcan_overview.tsv
cazy_family_matrix.tsv
cgc_substrate_predictions.tsv
bakta_annotation_summary.tsv
metabolic_traits.tsv
annotation_coverage_qc.tsv
```

Functional themes to aggregate:

| theme | source |
| --- | --- |
| KO/module capacity | KOfam, METABOLIC-G |
| CAZyme/substrate degradation | dbCAN |
| standardized gene products | Bakta |
| carbon, sulfur, nitrogen, energy traits | METABOLIC-G |
| broad orthology/EC/COG | eggNOG v2 optional sidecar after DB validation |

Gate:

- compute annotation coverage per MAG and per source group
- compare coverage across rumen, wetland, bridge, and non-bridge groups
- bridge interpretations must be blocked or caveated when low annotation coverage could explain missing pathways

### Layer 6 - Integrated Feature Store

Purpose: provide one analysis-ready table plus sparse matrices for modeling and discovery.

Recommended output directory:

```text
results/functional_metagenomics/{run_id}/cohort/integrated/
```

Core artifacts:

```text
mrv_feature_table.parquet
mrv_feature_table.tsv
latent_function_joined_features.parquet
feature_dictionary.tsv
feature_provenance.tsv
annotation_coverage_qc.tsv
```

Recommended `mrv_feature_table` grain:

```text
one row per proteome_id
```

Recommended column groups:

```text
id.*
file.*
source.*
latent.*
qc.*
taxonomy.*
gene_calling.*
methane_marker.*
mcycdb.*
scycdb.*
kofam.*
dbcan.*
bakta.*
metabolic.*
coverage.*
mechanism.*
gate.*
```

Feature dictionary fields:

```text
feature_name
feature_group
grain
source_tool
source_database
database_release
transform
normalization
missing_value_meaning
allowed_values
used_in_model
notes
```

Feature provenance fields:

```text
artifact
source_file
command_or_rule
tool_version
database_path
database_release
run_id
created_at
checksum_if_available
```

Normalization rules:

- marker and family counts should have raw count plus normalized values:

```text
raw_count
per_1k_proteins
binary_presence
confidence_status
```

- module/pathway features should use:

```text
completeness_fraction
observed_steps
required_steps
limiting_steps
confidence
```

- continuous latent features should be preserved unscaled in the feature store and scaled only inside modeling workflows.

## Statistical Integration Plan

### Analysis 1 - Mechanism Enrichment Among Bridge Candidates

Question: are high alpha-transfer bridge candidates enriched for methane-cycle, methylotrophy, electron-transfer, or sulfur-competition mechanisms?

Inputs:

- bridge rank and alpha-transfer score
- mechanism classes
- methane/sulfur module completeness
- QC and annotation coverage gates

Methods:

- rank-biserial or Spearman association between bridge score and mechanism features
- Fisher/exact tests for top-N bridge enrichment
- permutation test preserving source and domain labels

Required controls:

- compare against source-matched non-bridge genomes
- exclude or separately analyze QC-blocked genomes
- include annotation coverage as a covariate or stratification factor

### Analysis 2 - Latent-Function Alignment

Question: which functional features explain the ESM2 latent geometry?

Inputs:

- PC/UMAP/trajectory coordinates
- full functional feature matrix
- source/ecosystem labels

Methods:

- sparse regularized regression from functional features to trajectory axis
- random forest or gradient boosting feature ranking with source-aware splits
- Mantel/Procrustes-style comparison between latent distance and functional distance
- neighborhood enrichment: functional similarity among k-nearest latent neighbors

Outputs:

```text
bridge_function_associations.tsv
latent_function_alignment_metrics.tsv
functional_similarity_graph.parquet
```

### Analysis 3 - Hybrid Candidate Ranking

Question: which genomes are both latent bridges and mechanistically plausible methane MRV candidates?

Candidate score components:

| component | example |
| --- | --- |
| latent priority | normalized alpha-transfer score, bridge entropy, opposite-neighbor fraction |
| mechanism support | MCycDB/KOfam methane module support, MCR/Hdr/methylotrophy status |
| ecological context | sulfur competition, CAZyme substrate capacity, METABOLIC traits |
| QC confidence | CheckM2, GUNC, taxonomy resolution |
| coverage confidence | annotation rates across tools |

Recommended score:

```text
hybrid_priority_score =
  latent_priority_z
  + mechanism_support_z
  + substrate_flexibility_z
  - qc_penalty
  - annotation_missingness_penalty
```

This score is for prioritization, not final causal inference.

### Analysis 4 - Source-Aware Robustness

Question: does the bridge signal survive obvious source/project confounding?

Current caveat:

- all rumen genomes come from PRJEB31266
- wetland/MUCC records currently lack source accessions

Immediate controls possible now:

- downsample rumen to 107 to match wetland count
- compare bridge ranks across repeated downsampling
- include domain and assembly/protein-count covariates
- run sensitivity by removing coassembly-like or QC-warning records

Controls required before stronger transfer claims:

- add at least one more source project per ecosystem
- two-factor PERMANOVA with source and ecosystem terms
- leave-one-source-out validation

## Bridge Mechanism Cards

Each top candidate should receive a compact card that joins latent, QC, taxonomy, and function.

Required card fields:

```text
proteome_id
mag_id
source
ecosystem
domain
bridge_rank
alpha_transfer_score
bridge_entropy
opp_neighbor_fraction
qc_status
taxonomy_status
mechanism_class
mcr_status
hdr_status
methylotrophy_status
methanotrophy_status
sulfur_competition_status
substrate_flexibility
annotation_coverage_tier
confidence_tier
supporting_features
missing_features
blocking_caveats
recommended_next_action
```

These cards should be the primary human-review artifact before any platform/demo ranking is shown externally.

## Execution Order

1. Freeze the 662-row manifest from the crosswalk audit.
2. Run the one-MAG smoke runner until the newest hardened run completes cleanly.
3. Convert the one-MAG runner to a Slurm array over the 662-row manifest.
4. Write parser/integration scripts for each tool output.
5. Build `cohort_identity.tsv`, `mag_qc_integrated.tsv`, `taxonomy_resolved.tsv`, and `gene_calling_summary.tsv`.
6. Build methane/sulfur mechanism outputs from marker HMMs, MCycDB, SCycDB, KOfam, and METABOLIC-G.
7. Build broad functional matrices from KOfam, dbCAN, Bakta, METABOLIC-G, and optional eggNOG.
8. Build `mrv_feature_table.parquet` and `feature_dictionary.tsv`.
9. Build bridge mechanism cards and run manual review.
10. Run latent-function association and hybrid ranking analyses.
11. Run source-aware sensitivity checks before making any transfer-learning claim beyond the current POC caveat.

## Completion Criteria

The aggregation layer is complete when:

- `mrv_feature_table.parquet` has exactly 662 rows and no duplicate `proteome_id`
- every row has `match_status=matched`
- every row has QC, taxonomy, gene-calling, and annotation coverage status
- every top bridge candidate has a mechanism class and confidence tier
- every missing functional layer has an explicit missingness reason
- feature provenance records tool versions and database paths
- latent-only, functional-only, and hybrid feature tables can be generated reproducibly from the same run directory

## Claim Boundaries

Allowed near-term claim:

> The 662-genome MethaNet POC can now be expanded into a source-audited functional atlas that links ESM2 latent bridge candidates to independent MAG quality, taxonomy, methane-cycle, sulfur-cycle, CAZyme, and metabolic-trait evidence.

Not yet allowed:

> MethaNet has proven source-independent methane MRV transfer from rumen to wetland ecosystems.

That stronger claim requires additional source-balanced data and the Phase D source-aware validation gates.
