# MethaNet Current Artifact Inventory

Documentation refresh: 2026-08-10

This page summarizes the datasets, databases, generated warehouses, and graph
artifacts that currently define the MethaNet operational arc. The shared
product language and claim matrix live in
[`methanet_positioning_and_claims.md`](methanet_positioning_and_claims.md). This
inventory complements
the deeper system map in
`ai_docs/system_operational_map_20260619/TECHNICAL_BRIEF.md`.

Counts below are dated operational snapshots, not live scheduler state. During
active runs, refresh Slurm state and per-MAG completion sentinels before launch,
requeue, pruning, report rebuild, or external-sharing decisions.

## Claim Boundary

Current artifacts support MAG/proteome molecular attestation, functional-atlas
analytics, evidence-card review, bridge-candidate triage, monitoring
prioritization, and validation-study design. Calibrated sample and project risk
requires exact sample linkage, abundance or read coverage, environmental
covariates, uncertainty propagation, and paired field or process validation.
Final A to E tiers and carbon-market determinations follow independent
calibration and methodology review.

## Current Controlled-Diligence Evidence Contract

The August 10 source and locally built report use the following reconciled
ledger. The public URLs remain on the historical deployment until the source,
taxonomy, stability, accessibility, and publication gates pass:

| Measure | Release count | Interpretation |
| --- | ---: | --- |
| Registered MAG/proteome units | 7,965 | Warehouse registry, including explicit gaps and incomplete states |
| ESM-2-bearing units | 7,710 | Proteome-neighborhood navigation |
| gLM2 payloads | 7,717 | Genomic context, stratified by protocol |
| Data-complete tri-views | 7,710 | ESM-2, gLM2, and functional payload present |
| Schema-normalized tri-views | 7,710 | Long-form table and event semantics reconciled |
| Pipeline-normalized tri-views | 5,209 | POC, MSM, and Futian; comparability audit pending |
| Mechanism-comparable tri-views | 0 | No cross-lane quantitative comparison authorized |
| Source-scaffold tri-views | 2,501 | MUCC v1 DRAM, gene, and expression evidence |

The 7,710 total records payload availability. Accepted KOfam calls,
best-ranked MCycDB/SCycDB hits, and METABOLIC presence events are explicit, but
common code/configuration/database fingerprints and source-aware statistical
gates remain pending. MUCC v1 retains its source-scaffold evidence class.

## Canonical Backbones

| Artifact | Path | Grain | Use |
| --- | --- | --- | --- |
| 662 proteome crosswalk | `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv` | one `proteome_id` | Canonical identity backbone for ESM2, functional atlas, metadata, and attestation |
| Unit-scope manifest | `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv` | one `proteome_id` | Splits 625 MAG/bin-comparable units from 37 assembly-context units |
| MAG-bin-only manifest | `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_only.tsv` | one MAG/bin unit | Functional-atlas denominator for MAG-level analytics |
| Remaining MAG-bin manifest | `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_remaining.tsv` | one MAG/bin unit | Relaunch/submission manifest after preserved completed evidence |
| Assembly-context manifest | `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.assembly_context.tsv` | one assembly-context unit | Preserved evidence lane; excluded from MAG-level feature tables by default |

Canonical join key: `proteome_id`, unless a source-specific table explicitly
requires another key.

## Multi-Lane Molecular Payload Snapshot

MethaNet now has five analytically distinct evidence lanes. The rumen and
wetland POC lanes form the `poc_core` pipeline-normalized contract. MSM China
2025 and Futian 2026 use the same normalized event schema, with cross-lane
comparability still pending. MUCC v1 Old Woman Creek forms a wetland
source-scaffold contract with processed expression evidence.

The authoritative live-state source is the lane registry
`configs/methanet_atlas_lanes.tsv`, summarized by
`scripts/reports/summarize_atlas_lane_registry.py` (manifest-driven,
sentinel-based, dedup-aware). Counts below are reconciled against that
summarizer.

Release snapshot: 2026-08-10. Counts are tied to
`results/reports/methanet_3view_payload_freeze_20260810_end_to_end/`
and the corresponding MBAG report bundle. Refresh the lane registry before a
new operational or external release.

| Lane | Current molecular payload | Current use | Claim boundary |
| --- | --- | --- | --- |
| Rumen POC | 555 ESM2 proteomes in the 662-row backbone; 518 MAG/bin-comparable units in the POC functional/gLM2 denominator; 37 no-bin or assembly-context rumen units quarantined from MAG-bin feature tables | source reference lane for methane-system molecular neighborhoods and bridge hypotheses | source and ecosystem remain confounded; rumen evidence is not direct wetland risk evidence |
| Wetland/MUCC POC | 107 ESM2 proteomes; 107 MAG/bin-comparable functional outputs; 107 gLM2 contextual units | target-domain wetland POC lane for bridge-candidate validation and molecular-attestation cards | MAG-level potential only; sample abundance, environmental covariates, and flux validation are incomplete |
| Mangrove/MSM expansion | 1,428 local MAG/proteome candidates; 1,428/1,428 ESM2, gLM2, and functional payloads complete; 4 superseded complete attempts, 2 failed attempts, and 4 partial attempts remain in `fact_run_status` | target-domain expansion lane for broader blue-carbon molecular niche-space and future sample-level readiness | local 1,428-candidate denominator must be reconciled with the paper-reported 966 final medium/high-quality MAGs before final ecological denominators or sample MRV rollups |
| Mangrove/Futian 2026 expansion | 3,404 phase-1 dereplicated rMAGs at 99% ANI (3,156 ready payload rows + 248 explicit missing-payload gap rows); 3,156/3,156 ESM-2, gLM2, and functional payloads complete | target-domain expansion for molecular neighborhood and candidate review | code/configuration/database fingerprint equivalence, depth-resolved MAG-to-sample mapping, abundance, and field validation remain pending |
| MUCC v1 Old Woman Creek wetland | 2,508 checksum-validated archive MAGs; 2,501 ESM-2; 2,508 gLM2; 2,508 source-functional payloads; 2,501 data-complete source-scaffold tri-views | wetland molecular-reference screening, expression-detection review, and field-validation planning | source-scaffold mechanism features remain distinct; exact sample, depth, environmental, abundance, and flux joins are 0/133 |

All 625 POC units carry ESM-2, functional warehouse rows, and gLM2 context. The
current release contains 7,710 data-complete tri-views: 5,209
pipeline-normalized POC/MSM/Futian rows whose cross-lane comparability remains
pending, plus 2,501 MUCC v1 source-scaffold rows.

Scheduler state changes faster than this release inventory. Use curated
per-MAG sentinels and manifests for payload readiness, then use Slurm state as
operational telemetry. Run
`scripts/reports/refresh_atlas_lane_registry_status.sh` before launch, requeue,
or new release decisions.

## External Tool And Database Layer

Default database root on Apolo-3:

```bash
DB_ROOT=/home/rsg-jcorre38/scratch/methanet_db
```

| Tool/database | Local role | Production status |
| --- | --- | --- |
| CheckM2 | MAG completeness/contamination | ready |
| GUNC ProGenomes3 | chimerism/artifact screen | ready |
| GTDB-Tk R232 | taxonomy | ready |
| Prodigal | gene/protein prediction | ready |
| KOfam | KO/HMM functional evidence | ready |
| MCycDB 2021 | methane-cycle DIAMOND evidence | ready |
| SCycDB 2020Mar | sulfur-cycle DIAMOND evidence | ready |
| dbCAN V5 | CAZyme/CGC/substrate evidence | ready |
| METABOLIC-G | biogeochemical module/function summaries | ready |
| Bakta light DB v6.0 | standardized MAG annotation | ready, optional |
| eggNOG-mapper v2 data | broad orthology/EC/COG sidecar | staged, optional |
| DRAM/DRAM2 | metabolic distillation alternative | gated, not production blocker |

The authoritative runbook for this layer is
`docs/apollo3_functional_mag_runbook.md`.

## Per-MAG Evidence Bundles

Implemented layout:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag/{proteome_id}/{run_id}/
```

Each completed run should contain:

| Artifact | Purpose |
| --- | --- |
| `status.tsv` | per-step status, including failure step when applicable |
| `timings.tsv` | per-step runtime telemetry |
| `summary_metrics.tsv` | compact run metrics |
| `curated/run_record.json` | provenance, inputs, hashes, QC/taxonomy summaries, output registry |
| `curated/file_manifest.tsv` | raw and curated file provenance |
| `curated/parquet_manifest.tsv` | logical table to Parquet shard registry |
| `curated/parquet/*.parquet` | normalized per-run evidence shards |

Per-MAG folders are immutable evidence bundles. Failed and partial attempts are
preserved through `fact_run_status` instead of being dropped from downstream
analytics.

## Functional Atlas Warehouse

The `poc_core` lane is the **only consolidated, validated warehouse** at this
snapshot. Neither mangrove lane (MSM, Futian) has been consolidated yet; their
evidence currently lives only as per-MAG bundles (see the lane sections below),
and the lane summarizer gates both as "not ready to consolidate" until their
functional tranches finish. The per-lane `consolidate_functional_mag_cohort.py`
commands are emitted by `scripts/reports/summarize_atlas_lane_registry.py`.

Latest launch-ready generated warehouse observed locally:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/
```

Summary:

| Item | Value |
| --- | ---: |
| Cohort run ID | `fgx_poc_magbin_union_20260616` |
| Run attempts inspected | 683 |
| Selected completed MAG/bin runs | 625 |
| Complete attempts | 644 |
| Failed attempts | 24 |
| Partial attempts | 15 |
| Validation gates | 711 pass |
| DuckDB catalog | `functional_atlas.duckdb` |

Table families:

| Table family | Tables |
| --- | --- |
| Dimensions | `dim_mag`, `dim_gene` |
| Operational facts | `fact_run_status`, `fact_tool_timing`, `fact_input_stats`, `run_summary_metrics` |
| QC/taxonomy | `fact_qc_checkm2`, `fact_qc_gunc`, `fact_taxonomy_gtdbtk` |
| Functional hits | `fact_kofam_hits`, `fact_mcycdb_hits`, `fact_scycdb_hits`, `fact_dbcan_hits`, `fact_bakta_features` |
| METABOLIC/CAZy/MEROPS | `fact_metabolic_hmm_hits`, `fact_metabolic_function_presence`, `fact_metabolic_module_presence`, `fact_metabolic_module_step_presence`, `fact_cazy_hits`, `fact_merops_hits` |
| Feature summaries | `feature_annotation_coverage`, `feature_methane_mechanism`, `feature_sulfur_competition`, `feature_mrv_mag_level` |

High-volume tables in the latest snapshot:

| Table | Rows |
| --- | ---: |
| `fact_kofam_hits` | 23,845,557 |
| `fact_scycdb_hits` | 1,525,120 |
| `fact_bakta_features` | 1,226,606 |
| `fact_mcycdb_hits` | 1,223,407 |
| `dim_gene` | 1,202,529 |
| `fact_metabolic_module_step_presence` | 1,250,625 |

## Mangrove/MSM Functional Expansion Payload

Current active functional run directory:

```text
results/functional_metagenomics/msm_china_2025_20260615/
```

Rebuilt warehouse status for the local 1,428-candidate archive (2026-08-10):

| Item | Value |
| --- | ---: |
| Selected complete functional MAGs | 1,428 / 1,428 |
| Completion fraction | 100% |
| Preserved partial attempts | 4 |
| Preserved failed attempts | 2 |
| Not yet started | 0 |
| Proteome IDs with duplicate complete attempts | 4 |
| Raw complete sentinels on disk | 1,432 |
| Raw failed attempt sentinels preserved on disk | 2 |
| Selected completed per-run manifests | 1,428 |
| Domain denominator | 80 Archaea + 1,348 Bacteria |

The authoritative count is the 1,428-row `dim_mag` table in
`results/functional_metagenomics/msm_china_2025_20260615/cohort_warehouse/`.
It deterministically selects one complete run per `proteome_id`; older
complete, failed, and partial attempts remain in `fact_run_status`.

Aggregate rows across the 1,431 completed mangrove/MSM run manifests on disk
(summed from per-run `curated/parquet_manifest.tsv`; not yet a consolidated
warehouse):

| Table | Rows |
| --- | ---: |
| `fact_kofam_hits` | 95,830,531 |
| `fact_scycdb_hits` | 6,885,892 |
| `fact_mcycdb_hits` | 4,841,274 |
| `fact_bakta_features` | 3,990,092 |
| `fact_metabolic_module_step_presence` | 2,863,431 |
| `fact_metabolic_module_presence` | 672,570 |
| `fact_metabolic_hmm_hits` | 450,765 |
| `fact_dbcan_hits` | 199,984 |
| `fact_metabolic_function_presence` | 148,824 |
| `fact_merops_hits` | 31,708 |
| `fact_cazy_hits` | 19,567 |
| `fact_qc_checkm2` | 1,431 |
| `fact_qc_gunc` | 1,431 |

QC snapshot across the 1,431 completed mangrove/MSM run attempts (MIMAG-style buckets:
HQ = completeness >90% and contamination <5%; MQ = completeness >=50% and
contamination <=10%; otherwise LQ/QC-gated):

| QC field | Value |
| --- | ---: |
| High-quality-like MAGs | 106 |
| Medium-quality-like MAGs | 634 |
| Lower-quality or QC-gated MAGs | 691 |
| Median completeness | 73.27% |
| Median contamination | 5.52% |
| GUNC pass | 842 |
| GUNC fail | 589 |

The rebuilt Parquet/DuckDB warehouse passes 1,515 architecture gates. Its
feature tables use accepted KOfam events, best-ranked MCycDB/SCycDB hits, and
METABOLIC presence events. The 966 published-MAG denominator remains separate
from the 1,428 local archive denominator.

## Futian Mangrove 2026 Functional Expansion Payload

Newest mangrove source lane (`futian_mangrove_2026_qi`), phase-1 dereplicated
rMAGs at 99% ANI. Active per-MAG functional directories are split by domain and
bacteria shard:

```text
results/functional_metagenomics/futian_mangrove_2026_phase1_archaea/
results/functional_metagenomics/futian_mangrove_2026_phase1_bacteria_001/
results/functional_metagenomics/futian_mangrove_2026_phase1_bacteria_002/
results/functional_metagenomics/futian_mangrove_2026_phase1_bacteria_003/
```

Controlled-diligence release status (2026-08-10):

| Item | Value |
| --- | ---: |
| Phase-1 dereplicated rMAGs | 3,404 |
| Ready payload rows (functional include) | 3,156 |
| Explicit missing-payload gap rows | 248 |
| Functional denominator split | 312 Archaea + 2,844 Bacteria |
| Selected complete functional payloads | 3,156 / 3,156 |
| Archaea functional complete | 312 / 312 |
| Archaea running/partial | 0 |
| Archaea failed (manifest-scoped) | 0 |
| Archaea pending/not-started | 0 |
| Bacteria selected complete | 2,844 / 2,844 |
| Ready rows without functional payload | 0 |
| ESM2 embeddings | 3,156 / 3,156 complete across four shards |
| gLM2 contextual units | 3,156 / 3,156 complete |

The rebuilt warehouse includes every ready Futian functional payload admitted
by the dated freeze and preserves 248 source gaps as explicit exclusions. The
common long-form event schema does not establish cross-lane mechanism
equivalence; locked code/configuration/database fingerprints and source-aware
statistical gates remain the main scientific blockers. Slurm timeout, failed,
corrupt, and partial attempts remain status records when a validated curated
evidence bundle supersedes them.
The earlier 312-archaea aggregate remains a dated operational diagnostic:

| Table | Rows |
| --- | ---: |
| `fact_kofam_hits` | 8,428,544 |
| `fact_metabolic_module_step_presence` | 624,312 |
| `fact_scycdb_hits` | 552,492 |
| `fact_mcycdb_hits` | 511,204 |
| `fact_bakta_features` | 468,909 |
| `fact_metabolic_module_presence` | 146,640 |
| `fact_metabolic_hmm_hits` | 98,280 |
| `fact_metabolic_function_presence` | 32,448 |
| `fact_dbcan_hits` | 11,491 |
| `fact_merops_hits` | 2,550 |
| `fact_cazy_hits` | 1,322 |
| `fact_qc_checkm2` | 312 |
| `fact_qc_gunc` | 312 |

An early 12-bacteria operational tranche added the following raw per-run
Parquet rows before lane-level consolidation:

| Table | Rows |
| --- | ---: |
| `fact_kofam_hits` | 728,673 |
| `fact_scycdb_hits` | 51,874 |
| `fact_bakta_features` | 32,952 |
| `fact_mcycdb_hits` | 33,642 |
| `fact_metabolic_module_step_presence` | 24,012 |
| `fact_metabolic_module_presence` | 5,640 |
| `fact_metabolic_hmm_hits` | 3,780 |
| `fact_dbcan_hits` | 2,305 |
| `fact_metabolic_function_presence` | 1,248 |
| `fact_merops_hits` | 257 |
| `fact_cazy_hits` | 261 |
| `fact_qc_checkm2` | 12 |
| `fact_qc_gunc` | 12 |

QC snapshot across the 312 completed Futian archaea runs:

| QC field | Value |
| --- | ---: |
| High-quality-like MAGs | 33 |
| Medium-quality-like MAGs | 278 |
| Lower-quality or QC-gated MAGs | 1 |
| Median completeness | 73.75% |
| Median contamination | 0.875% |
| GUNC pass | 298 |
| GUNC fail | 14 |

The archaea tranche is comparatively clean (near-zero median contamination and
298/312 GUNC pass), but it is a small, taxonomically biased slice of the lane;
do not generalize lane-level QC until the bacteria shards complete.
Source provenance and checksums live under
`data/external/futian_mangrove_2026_qi/source_docs/`; the 248-row gap register is
`data/external/futian_mangrove_2026_qi/manifests/futian_phase1_download_gap_register.tsv`.

## Metadata And Provenance Layer

Generated metadata recovery snapshot:

```text
results/functional_metagenomics/environmental_metadata_recovery_20260612/
```

Key outputs:

| Artifact | Use |
| --- | --- |
| `METADATA_RECOVERY_REPORT.md` | human-readable provenance and caveat report |
| `cohort_662_environmental_metadata_crosswalk.tsv` | crosswalk over current 662-row cohort |
| `rumen_proteome_environmental_metadata.tsv` | exact ENA analysis-accession metadata where available |
| `mucc_proteome_environmental_metadata.tsv` | MUCC source-bucket/BioSample/site context |
| `source_bioproject_summaries.tsv` | project-level source provenance |
| `metadata_recovery_validation.tsv` | metadata recovery checks |

Metadata resolution is mixed. Exact accession metadata, site/project context,
source-bucket context, and modeled estimates must remain distinct in reports and
downstream scoring.

## External MSM China 2025 Lane

Local source directory:

```text
data/external/msm_china_2025/
```

Important generated/used files:

| Artifact | Use |
| --- | --- |
| `metadata/source_register.tsv` | source-object inventory |
| `metadata/ncbi_biosample_environmental_metadata.tsv` | exact BioSample metadata where resolved |
| `gigadb_wasabi/metadata_sediment_samples.txt` | local sediment-sample metadata |
| `manifests/msm_china_2025_functional_embedding_manifest.tsv` | integration manifest for embedding/functional work |
| `results/functional_metagenomics/msm_china_2025_20260615/manifests/` | functional-run manifests when generated |

Open issue: local MAG candidates, paper-reported 966 final MAGs, and sample/MAG
mapping still need reconciliation before sample-level MRV rollups.

Current molecular sidecars for this lane:

| Artifact | Status | Path |
| --- | --- | --- |
| ESM2 proteome embeddings | 1,428 / 1,428 complete with `facebook/esm2_t33_650M_UR50D`; 0 missing FAA, 0 pending | `results/blue_catalyst_poc/runs/msm_china_2025_esm2_20260616_082112/artifacts/` |
| gLM2 contextual windows/spans | 1,428 / 1,428 complete; 2,856 windows and 28,536 spans | `results/contextual_genomics/glm2_msm_magbin_full_20260615_092737/` |
| Functional annotation | 1,428 / 1,428 selected in the validated 2026-08-10 cohort warehouse | `results/functional_metagenomics/msm_china_2025_20260615/cohort_warehouse/` |
| Sample/environment metadata | 82 local GigaDB sediment-sample metadata rows and 71 exact BioSample environmental rows | `data/external/msm_china_2025/metadata/` and `data/external/msm_china_2025/gigadb_wasabi/` |

Sample-level claims remain blocked until the MAG-to-sample mapping, MAG/read
abundance or coverage, environmental covariates, and validation outcomes are
joined with explicit resolution tiers.

## Latest Partner-Facing Report Artifacts

Current generated controlled-diligence atlas:

```text
results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/report.html
```

That report is backed by the dated freeze:

```text
results/reports/methanet_3view_payload_freeze_20260810_end_to_end/
```

It registers 7,965 MAG/proteome units and exposes 7,710 ESM-2 embeddings, 7,717
gLM2 payloads, and 7,710 data-complete tri-views. The tri-views comprise 5,209
pipeline-normalized POC/MSM/Futian rows with cross-lane comparability pending
and 2,501 MUCC v1 source-scaffold rows. The 255 source gaps remain explicit
release exclusions. Treat older published reports as historical snapshots.

## Molecular Attestation Graph

Foundational graph MVP:

```text
results/attestation/mmag_mvp_20260617/
```

Key outputs:

| Artifact | Use |
| --- | --- |
| `registry_artifact.tsv` / `.parquet` | source artifact registry with hashes/provenance |
| `evidence_atom.tsv` / `.parquet` | typed evidence facts linked to artifacts |
| `graph_nodes.tsv` / `.parquet` | MAG, source, taxon, feature, claim, gap, artifact, evidence nodes |
| `graph_edges.tsv` / `.parquet` | evidence, feature, claim, blocker, source, taxonomy, and ESM2-neighbor relationships |
| `mmag.kuzu/` | optional embedded graph database |
| `QUERY_LIBRARY.cypher` | canonical graph queries |
| `validation_report.md` | builder validation |
| `EXPERT_AUDIT_REPORT.md` | static and multi-hop query audit |

Audit inventory:

| Graph element | Count |
| --- | ---: |
| MAG nodes | 662 |
| Evidence atoms | 3,968 |
| Feature nodes | 2,644 |
| Taxon nodes | 397 |
| Artifact nodes | 13 |
| ValidationGap nodes | 8 |
| Claim nodes | 5 |
| SourceDomain nodes | 2 |
| `NEAR_IN_ESM2_SPACE` edges | 9,930 |

Readiness distribution:

| Readiness | MAGs |
| --- | ---: |
| `molecular_attestation_ready_not_mrv` | 437 |
| `molecular_attestation_ready_with_qc_caveat` | 188 |
| `blocked_noncomparable_unit` | 37 |

The July 24 release projects the same evidence-governance principle across the
full 7,965-unit warehouse. It adds release-level evidence contracts, candidate
cards, source provenance, protocol classes, authorized claim wording,
validation gaps, and next actions. The 662-node MVP remains the foundational
queryable graph artifact, while the current report is the warehouse-wide MBAG
decision surface.

## Core Docs To Keep In Sync

| Document | Role |
| --- | --- |
| `README.md` | public-facing project overview and current artifact arc |
| `docs/methanet_positioning_and_claims.md` | canonical product narrative, terminology, release counts, and claim matrix |
| `docs/apollo3_functional_mag_runbook.md` | Apolo-3 database/tool setup |
| `docs/apollo3_mag_functional_analytics_ops.md` | executable operations and current generated artifacts |
| `docs/functional_metagenomics_expansion.md` | phase gates and current implementation status |
| `ai_docs/functional_metagenomics_expansion/README.md` | functional expansion package index and implemented arc |
| `ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md` | MRV maturity ladder and claim boundaries |
| `ai_docs/functional_metagenomics_expansion/cohort_data_architecture_hardening.md` | warehouse schemas, validation gates, storage policy |
| `ai_docs/functional_metagenomics_expansion/output_contracts_and_gates.md` | required outputs and implemented layouts |
| `ai_docs/functional_metagenomics_expansion/source_provenance_environmental_metadata_reconciliation.md` | metadata provenance and environmental context |
| `configs/methanet_atlas_lanes.tsv` | atlas lane registry; authoritative source of truth for multi-view denominators and per-lane artifact locations |
| `scripts/reports/refresh_atlas_lane_registry_status.sh` | regenerates the dated lane status snapshot (`results/reports/atlas_lane_registry_status_*`) used to reconcile this inventory |
