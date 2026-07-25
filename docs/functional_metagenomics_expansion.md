# MBAG Molecular Attestation And Functional Expansion Contracts

The 662-genome POC established a source-confounded ESM-2 separation and a set
of bridge hypotheses. The current MBAG release extends that starting point into
a governed molecular-attestation warehouse. This page defines the gates that
connect representation context to functional mechanism evidence, monitoring
readiness, and future calibrated methane-risk features.

Documentation refresh: 2026-07-24.

Use [`methanet_positioning_and_claims.md`](methanet_positioning_and_claims.md)
for the shared narrative and claim contract used by the repository, landing
page, and public report.

The early gates below remain the conceptual contract, but much of the first
implementation arc now exists as generated local artifacts:

- 662-row ESM2/crosswalk backbone:
  `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv`
- unit-scope split:
  625 MAG/bin-comparable units and 37 assembly-context rumen units;
- functional-atlas warehouse:
  `results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/`
- molecular attestation graph:
  `results/attestation/mmag_mvp_20260617/`

The current implemented product primitive is MAG/proteome molecular attestation
through evidence cards, candidate review, monitoring prioritization, and
validation-study design. Sample and project MRV scoring becomes eligible after
sample mapping, abundance or read coverage, environmental covariates,
uncertainty propagation, and flux or process validation.

The July 24 scientific-reconciliation release contains 7,965 registered units,
7,710 ESM-2 embeddings, 7,717 gLM2 payloads, and 7,484 data-complete
tri-views. The molecular atlas has five evidence lanes:

| Lane | Current state | Interpretation |
| --- | --- | --- |
| Rumen POC | 555 ESM2 proteomes; 518 MAG/bin-comparable multi-view units plus 37 assembly-context units | source reference lane for methane-system bridge hypotheses |
| Wetland/MUCC POC | 107 ESM2/function/gLM2 MAG-bin units | target wetland POC lane |
| Mangrove/MSM expansion | 1,428/1,428 ESM2 embeddings and 1,428/1,428 gLM2 units complete; 1,427/1,428 functional MAGs complete | broader blue-carbon target expansion, not yet a final sample-level MRV layer |
| Mangrove/Futian expansion | 3,156/3,156 ESM-2 embeddings and 3,156/3,156 gLM2 units; 2,931 annotation-complete functional payloads | largest target-domain expansion; common mechanism-feature aggregation remains pending |
| MUCC v1 Old Woman Creek | 2,508 registered MAGs; 2,501 ESM-2; 2,508 gLM2 and source-functional payloads; 2,501 data-complete source-scaffold tri-views | wetland reference and expression-detection lane under a distinct non-equivalent functional contract |

The evidence contract separates 625 mechanism-comparable POC tri-views, 4,358
annotation-complete mangrove tri-views awaiting common feature aggregation, and
2,501 MUCC v1 source-scaffold tri-views. Data completeness records payload
availability. Quantitative mechanism comparability requires the common
accepted/present feature contract.

For dated counts, paths, and report freshness, use
`docs/current_artifact_inventory.md`.
For live lane status and report-freeze decisions, use
`scripts/reports/refresh_atlas_lane_registry_status.sh` followed by
`scripts/reports/build_methanet_3view_payload_freeze.py`.

All heavy tools are disabled by default in `configs/pipeline.yaml` under
`functional_metagenomics`. A gate can be marked complete only when its required
tables exist, join to every bridge candidate, and pass the stated checks.

Operational setup, Apollo-3 database paths, repair commands, and end-to-end MAG
annotation commands are recorded in
`docs/apollo3_functional_mag_runbook.md`.
For executable Apollo-3 run commands, readiness checks, and gated-tool decisions,
use `docs/apollo3_mag_functional_analytics_ops.md`.

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

The next engineering milestone is no longer just Phase A plus a minimal Phase B
prototype. Those concepts now exist as a curated warehouse and attestation MVP
for the current POC denominator. The highest-priority remaining work is:

1. Promote gene/marker/pathway-level evidence from the warehouse into richer
   bridge mechanism cards and graph nodes.
2. Finish or deliberately snapshot the mangrove/MSM and Futian functional
   tranches, then build manifest-driven expansion warehouses that preserve
   complete, partial, failed, duplicate, and not-started status rows.
3. Reconcile local mangrove/MSM 1,428 processable candidates with the
   paper-reported 966 final medium/high-quality MAG denominator.
4. Preserve Futian's 3,156-ready payload denominator, 3,404 phase-1 rMAG
   denominator, and 248-row missing-payload gap register as separate status
   concepts.
5. Reconcile sample/MAG links and metadata resolution for sample-level rollups.
6. Add abundance/read-coverage tables so MAG potential can become
   community-capacity estimates.
7. Join environmental covariates and measured/process methane validation.
8. Add source-replicated cohorts and source-aware controls before claiming
   transfer beyond the current confounded rumen/MUCC design.

The project should not claim final methane MRV readiness until Phases A-D plus
the sample, abundance, environment, uncertainty, and validation layers pass. The
current claim is: MethaNet has a queryable MAG/proteome-level molecular
attestation and functional-atlas substrate for bridge-candidate prioritization,
under explicit source-confounding and sample-level MRV caveats.
