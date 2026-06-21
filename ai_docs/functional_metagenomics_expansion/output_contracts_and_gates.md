# Output Contracts and Scientific Gates

Date: 2026-06-11
Documentation refresh: 2026-06-20

## Run Directory Contract

All outputs should live under:

```text
results/functional_metagenomics/{run_id}/
```

Recommended layout:

```text
results/functional_metagenomics/{run_id}/
├── manifests/
│   ├── mag_manifest.resolved.tsv
│   ├── tool_db_manifest.tsv
│   └── run_config.locked.yaml
├── per_mag/{mag_id}/
│   ├── qc/
│   ├── taxonomy/
│   ├── genes/
│   ├── annotation/
│   └── cards/
├── cohort/
│   ├── qc_taxonomy/
│   ├── functional_matrices/
│   ├── bridge_cards/
│   ├── latent_function/
│   └── platform_demo/
├── figures/
└── logs/
```

## Implemented Apollo-3 Layout

The implemented production stack now uses a more explicit per-attempt layout
than the early recommended sketch above. The current evidence bundles live
under:

```text
results/functional_metagenomics/{cohort_run_id}/per_mag/{proteome_id}/{run_id}/
```

Each successful run closes out through
`scripts/curate_functional_mag_run.py` and writes:

```text
curated/run_record.json
curated/file_manifest.tsv
curated/parquet_manifest.tsv
curated/parquet/<logical_table>.parquet
status.tsv
timings.tsv
summary_metrics.tsv
```

The cohort warehouse then writes:

```text
results/functional_metagenomics/{cohort_run_id}/cohort_warehouse*/DATA_ARCHITECTURE_VALIDATION.md
results/functional_metagenomics/{cohort_run_id}/cohort_warehouse*/cohort_table_manifest.tsv
results/functional_metagenomics/{cohort_run_id}/cohort_warehouse*/validation_gates.tsv
results/functional_metagenomics/{cohort_run_id}/cohort_warehouse*/functional_atlas.duckdb
results/functional_metagenomics/{cohort_run_id}/cohort_warehouse*/parquet/<table>/cohort_run_id=<id>/part-00000.parquet
```

Current launch-ready generated snapshot:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/
```

This snapshot contains 625 selected MAG/bin rows in `dim_mag`, preserves 683
attempts in `fact_run_status`, and provides 24 table/model artifacts including
the DuckDB catalog. The 37 assembly-context units remain part of the 662-row
identity/attestation denominator but are excluded from MAG-level feature tables
by default.

## Multi-Lane Output Gates

The same output contract now applies across rumen, wetland/MUCC, and
mangrove/MSM lanes, with lane-specific denominators:

| Gate | Applies to | Pass condition |
| --- | --- | --- |
| POC MAG-bin gate | 518 rumen MAG-bin units + 107 wetland/MUCC MAG-bin units | 625/625 units present in ESM2, functional warehouse, gLM2, QC/taxonomy, and annotation-coverage tables |
| POC assembly-context gate | 37 rumen no-bin or assembly-context units | explicitly present in identity/status/attestation views and excluded from MAG-bin feature tables unless a separate assembly-context analysis is requested |
| Mangrove/MSM expansion gate | 1,428 local mangrove/MSM candidates | ESM2 and gLM2 complete; functional tranche consolidated by manifest with complete, failed, partial, duplicate, and not-started status rows |
| Published-denominator reconciliation gate | mangrove/MSM source-paper comparison | local 1,428-candidate processing denominator reconciled against the paper-reported 966 final medium/high-quality MAG denominator |
| Sample MRV gate | any lane used for sample/project claims | MAG-to-sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, and flux/process validation joined with explicit resolution tiers |

Reports must name the lane and denominator for every count. A figure or table
that mixes lanes must expose the join status for ESM2, functional annotation,
gLM2 context, metadata, QC, and sample-readiness separately.

## Phase A - MAG QC and Identity

### Required outputs

| artifact | grain | required columns |
| --- | --- | --- |
| `mag_qc_integrated.tsv` | MAG | `mag_id`, `fasta_path`, `total_bp`, `contigs`, `n50`, `gc_pct`, `completeness`, `contamination`, `checkm2_model`, `gunc_pass`, `css`, `contamination_portion`, `qc_status` |
| `taxonomy_resolved.tsv` | MAG | `mag_id`, `domain`, `phylum`, `class`, `order`, `family`, `genus`, `species`, `classification_method`, `gtdb_release`, `taxonomy_status` |
| `derep_clusters.tsv` | MAG | `mag_id`, `derep_cluster`, `representative_mag_id`, `ani_threshold`, `cluster_method`, `derep_status` |
| `bridge_candidates_registry.tsv` | bridge candidate | `mag_id`, `ecosystem`, `source`, `alpha_transfer_score`, `bridge_entropy`, `opp_neighbor_fraction`, `qc_status`, `taxonomy_status`, `derep_status` |

### Gate A

Every bridge candidate must be one of:

- `pass_high_quality`
- `pass_medium_quality`
- `qc_warning_review`
- `qc_fail_blocked`

No bridge candidate can remain unclassified for QC/taxonomy/derep status.

## Phase B - Methane Mechanism Layer

### Required outputs

| artifact | grain | required columns |
| --- | --- | --- |
| `methane_marker_panel.tsv` | MAG x marker | `mag_id`, `marker`, `copy_count`, `best_hit_gene`, `source_db`, `bitscore`, `evalue`, `coverage`, `identity`, `call_status` |
| `mcycdb_hits.tsv` | gene hit | `mag_id`, `gene_id`, `mcyc_family`, `pathway`, `bitscore`, `evalue`, `identity`, `query_coverage`, `subject_coverage`, `hit_status` |
| `methane_pathway_completeness.tsv` | MAG x pathway/module | `mag_id`, `module`, `required_steps`, `observed_steps`, `completeness`, `limiting_steps`, `confidence` |
| `electron_transfer_features.tsv` | MAG | `mag_id`, `hydrogenase_count`, `formate_dehydrogenase_count`, `hdr_complex_status`, `mvh_status`, `eha_ehb_status`, `ferredoxin_context_score` |
| `sulfur_competition_features.tsv` | MAG | `mag_id`, `dsrAB_status`, `aprAB_status`, `sat_status`, `sox_status`, `sqr_status`, `scycdb_coverage` |
| `bridge_mechanism_cards.json` | bridge candidate | card object described below |

### Mechanism card JSON contract

```json
{
  "mag_id": "string",
  "ecosystem": "rumen|wetland",
  "source": "string",
  "latent": {
    "alpha_transfer_score": 0.0,
    "bridge_entropy": 0.0,
    "opp_neighbor_fraction": 0.0,
    "bridge_rank": 0
  },
  "qc": {
    "completeness": 0.0,
    "contamination": 0.0,
    "gunc_pass": true,
    "qc_status": "pass_medium_quality"
  },
  "taxonomy": {
    "gtdb_release": "R232",
    "domain": "d__Archaea",
    "species": "string",
    "taxonomy_status": "resolved"
  },
  "methane_mechanism": {
    "class": "methane_relevant_partial",
    "mcr_status": "complete|partial|absent",
    "mtr_status": "complete|partial|absent",
    "methylotrophy_status": "complete|partial|absent",
    "methanotrophy_status": "complete|partial|absent",
    "aom_status": "complete|partial|absent",
    "evidence_summary": ["string"]
  },
  "substrate_electron_sulfur": {
    "substrate_flexibility": "low|medium|high|unknown",
    "electron_transfer_score": 0.0,
    "sulfur_competition_score": 0.0
  },
  "confidence": {
    "tier": "high|medium|low|blocked",
    "blocking_reasons": ["string"],
    "next_actions": ["string"]
  }
}
```

### Gate B

Every top bridge candidate must be classified as:

- `methane_relevant_high_confidence`
- `methane_relevant_partial`
- `substrate_flexible`
- `sulfur_associated`
- `unclear_function`
- `likely_artifact_or_qc_blocked`

## Phase C - Broad Functional Layer

### Required outputs

| artifact | grain | purpose |
| --- | --- | --- |
| `ko_matrix.tsv` | MAG x KO | KOfam/eggNOG KO presence or copy count |
| `ec_matrix.tsv` | MAG x EC | enzyme feature layer |
| `module_completeness.tsv` | MAG x module | KEGG/curated module completeness |
| `fact_eggnog_annotations` | gene | optional eggNOG-mapper orthology, COG category, GO/EC/KO/Pfam where available |
| `dram_distillate.tsv` | MAG x trait | genome metabolism distillation |
| `metabolic_traits.tsv` | MAG x trait | METABOLIC-G biogeochemical profile |
| `dbcan_overview.tsv` | gene/MAG | CAZyme overview |
| `cazy_family_matrix.tsv` | MAG x CAZy family | substrate-processing capacity |
| `cgc_substrate_predictions.tsv` | CGC | CAZyme gene cluster substrate predictions |
| `transport_substrate_features.tsv` | MAG | transporter/substrate summary |
| `annotation_coverage_qc.tsv` | MAG x tool | coverage and missingness |

### Annotation coverage contract

`annotation_coverage_qc.tsv` must include:

| column | meaning |
| --- | --- |
| `mag_id` | genome |
| `tool` | KOfam, optional eggNOG, MCycDB, dbCAN, etc. |
| `protein_count` | canonical protein count |
| `annotated_proteins` | proteins with accepted calls |
| `coverage_fraction` | annotated / protein count |
| `low_coverage_flag` | true/false |
| `notes` | reason for low coverage |

### Gate C

Bridge rankings cannot be interpreted mechanistically until:

- annotation coverage is measured for every top candidate,
- bridge and non-bridge coverage are compared,
- ecosystem-level missingness is checked,
- absent pathways are distinguished from unannotated/unresolved pathways.

## Phase D - Latent-Function Linkage and Source Controls

### Required outputs

| artifact | grain | purpose |
| --- | --- | --- |
| `latent_function_joined_features.parquet` | MAG | full feature table |
| `bridge_function_associations.tsv` | feature | association with bridge score/entropy |
| `downsampling_sensitivity.tsv` | iteration | rumen n=107 downsampling metrics |
| `knn_seed_sensitivity.tsv` | parameter set | k/seed stability |
| `source_aware_validation.tsv` | model/fold | leave-one-source-out once possible |
| `hybrid_feature_benchmark.tsv` | model | latent-only vs functional-only vs hybrid |

### Gate D

The current 662-genome POC can support mechanism-enriched prioritization, but not final source-independent transfer claims. For source-independent claims:

- each ecosystem needs at least two source projects,
- two-factor PERMANOVA must include ecosystem and source terms,
- leave-one-source-out must be run,
- bridge rankings must remain stable after source-aware controls.

## Investor Platform Output Contract

The minimal demo should consume only reviewed artifacts:

| artifact | purpose |
| --- | --- |
| `mrv_feature_table.parquet` | primary platform table for model/demo |
| `bridge_mechanism_cards.json` | card UI data |
| `functional_similarity_graph.parquet` | hybrid network visualization |
| `platform_dashboard_snapshot.json` | summary counts and evidence tiers |
| `candidate_recommendation_table.tsv` | ranked candidates with confidence and caveats |

### Candidate recommendation fields

| column | meaning |
| --- | --- |
| `rank` | platform rank |
| `mag_id` | candidate |
| `ecosystem` | origin ecosystem |
| `latent_priority` | normalized latent/bridge score |
| `mechanism_class` | methane/sulfur/substrate class |
| `qc_tier` | QC confidence |
| `taxonomy_status` | resolved/unresolved |
| `functional_coverage_tier` | high/medium/low |
| `mrv_readiness_tier` | `ready_for_followup`, `promising_needs_qc`, `promising_needs_function`, `blocked` |
| `why_it_matters` | short explanation |
| `blocking_caveats` | semicolon-separated caveats |

## Scientific Guardrails

1. Never interpret a missing pathway without checking annotation coverage.
2. Never call a geometry-only bridge mechanistic.
3. Never combine GTDB releases without release labels.
4. Never compare eggNOG v2 and v3 outputs without version columns.
5. Never hide QC-failed bridge candidates; mark them blocked.
6. Never claim source-independent transfer from the current single-source-per-ecosystem design.
