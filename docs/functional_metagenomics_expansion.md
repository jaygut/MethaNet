# Functional Metagenomics Expansion Contracts

The 662-genome POC proves that ESM2 latent space separates rumen and wetland
ecosystems while retaining bridge candidates. It does not yet prove the bridge
genomes are mechanistically methane-relevant, source-balanced, or robust to
annotation missingness. This page defines the gated work needed to move from
latent geometry to source-aware methane MRV features.

All heavy tools are disabled by default in `configs/pipeline.yaml` under
`functional_metagenomics`. A gate can be marked complete only when its required
tables exist, join to every bridge candidate, and pass the stated checks.

## Phase A: MAG QC And Identity Layer

Purpose: prevent bridge rankings from being driven by low-quality assemblies,
contamination, unresolved taxonomy, or duplicated genomes.

Required outputs:

| Output | Required columns | Source tools |
|--------|------------------|--------------|
| `mag_qc_integrated.tsv` | `sample`, `completeness`, `contamination`, `gunc_css`, `gunc_rss`, `taxonomy`, `gtdb_domain`, `gtdb_phylum`, `gtdb_genus`, `drep_cluster`, `qc_status`, `taxonomy_status`, `derep_status` | CheckM2, GUNC, GTDB-Tk, dRep |

Gate:

- Every bridge candidate in `bridging_genomes_top.tsv` has QC, taxonomy, and
  dereplication status.
- Wetland `Unknown` taxonomy is resolved through GTDB-Tk or explicitly marked
  `unresolved`.
- Bridge candidates with high contamination, low completeness, high chimerism,
  or duplicate representatives are excluded or down-weighted in downstream
  mechanism cards.

## Phase B: Methane Mechanism Layer

Purpose: explain whether each bridge is methane-relevant and through which
biochemical route.

Required outputs:

| Output | Required columns |
|--------|------------------|
| `methane_marker_panel.tsv` | `sample`, 12 current marker columns, expanded methane/sulfur marker columns, normalized abundances, hit counts |
| `methanogenesis_operon_completeness.tsv` | `sample`, `mcr_complete`, `hdr_complete`, `mtr_complete`, `methylotrophic_complete`, `hydrogenotrophic_complete`, `acetoclastic_complete`, `aom_complete` |
| `mcycdb_family_hits.tsv` | `sample`, `family_id`, `family_name`, `hit_count`, `coverage`, `best_evalue`, `best_bitscore` |
| `mcycdb_pathway_completeness.tsv` | `sample`, `pathway`, `module`, `completeness`, `limiting_markers`, `confidence` |
| `bridge_mechanism_cards.tsv` | `sample`, `bridge_rank`, `mechanism_class`, `supporting_features`, `missing_features`, `artifact_flags` |

Mechanism classes:

- `methane_relevant`
- `substrate_flexible`
- `sulfur_associated`
- `unclear`
- `likely_artifact`

Gate:

- Top bridge candidates can be classified into one mechanism class.
- MCR, Hdr/electron bifurcation, methylotrophy, methanotrophy, anaerobic
  oxidation of methane, and sulfur competition evidence is recorded separately.
- `unclear` candidates carry explicit missing-evidence reasons rather than
  silent nulls.

## Phase C: Broad Functional Layer

Purpose: make sure bridge rankings are not an artifact of narrow methane-marker
coverage and expose broader substrate/ecology signals.

Required outputs:

| Output | Required columns |
|--------|------------------|
| `ko_matrix.tsv` | `sample` plus KO count/completeness columns from KOfamScan |
| `ec_matrix.tsv` | `sample` plus EC abundance/completeness columns from eggNOG-mapper |
| `module_matrix.tsv` | `sample`, KEGG/MetaCyc module completeness features |
| `annotation_coverage.tsv` | `sample`, `n_proteins`, `n_annotated`, `annotation_rate`, `ko_rate`, `ec_rate`, `cazyme_rate`, `transporter_rate` |
| DRAM2/METABOLIC distillates | `sample`, pathway/process summaries and confidence fields |
| dbCAN/CAZyme summaries | `sample`, CAZyme families, substrate classes, counts |

Gate:

- Annotation coverage is measured per genome and per source.
- Bridge, rumen, and wetland groups have comparable enough annotation coverage
  that missingness cannot silently explain bridge rankings.
- CAZyme, transporter, substrate, sulfur, nitrogen, and carbon fixation summaries
  are available for bridge mechanism cards.

## Phase D: Latent-Function Linkage And Deconfounding

Purpose: test whether latent-space bridges are stable biological candidates
rather than source/project artifacts.

Required outputs:

| Output | Required columns |
|--------|------------------|
| `latent_function_stability.tsv` | `sample`, `k`, `umap_seed`, `downsample_seed`, `bridge_score`, `rank`, `rank_stability` |
| `source_aware_validation.tsv` | `feature_set`, `validation_scheme`, `held_out_source`, `metric`, `value`, `ci_low`, `ci_high` |
| PERMANOVA report | `term`, `pseudo_f`, `p_value`, `r2`, `permutations` for ecosystem and source terms |

Required analyses:

- Bridge score stability across k values, UMAP seeds, and rumen downsampling to
  n=107.
- Source-balanced cohorts with at least two source projects per ecosystem.
- Two-factor PERMANOVA with ecosystem and source terms.
- Latent-only, functional-only, and hybrid feature benchmarks under
  leave-one-source-out validation.

Gate:

- Ecosystem signal remains after source-aware controls.
- Bridge rankings remain stable under downsampling and projection perturbation.
- Hybrid features outperform or explain latent-only rankings without relying on
  source leakage.

## Immediate Missing Work

The next engineering milestone is Phase A plus a minimal Phase B bridge-card
prototype. That gives every current bridge candidate a QC/taxonomy/derep status
and a first mechanistic label before adding broad KO/EC/module matrices.

The project should not claim mechanistic methane MRV readiness until Phases A-D
pass. Until then, the precise claim remains: the current 662-genome ESM2 POC
shows strong latent ecosystem structure with cross-ecosystem bridge candidates,
under a known source-confounding caveat.
