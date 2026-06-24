# MethaNet Current Artifact Inventory

Documentation refresh: 2026-06-22

This page summarizes the datasets, databases, generated warehouses, and graph
artifacts that currently define the MethaNet operational arc. It complements
the deeper system map in
`ai_docs/system_operational_map_20260619/TECHNICAL_BRIEF.md`.

Counts below are dated operational snapshots, not live scheduler state. During
active runs, refresh Slurm state and per-MAG completion sentinels before launch,
requeue, pruning, report rebuild, or external-sharing decisions.

## Claim Boundary

Current artifacts support MAG/proteome-level molecular evidence, functional
atlas analytics, bridge-candidate triage, and molecular attestation. They do
not support final sample/project MRV risk scores, final A-E methane-risk tiers,
measured methane flux claims, source-independent transfer proof, or carbon-credit
approval.

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

MethaNet now has four analytically distinct molecular lanes: the rumen and
wetland POC lanes (fused as the closed `poc_core` calibration lane) plus two
mangrove expansion lanes (MSM China 2025 and the new Futian 2026 lane). Keep
their denominators and claim boundaries separate until a deliberate multi-view
union is rebuilt.

The authoritative live-state source is the lane registry
`configs/methanet_atlas_lanes.tsv`, summarized by
`scripts/reports/summarize_atlas_lane_registry.py` (manifest-driven,
sentinel-based, dedup-aware). Counts below are reconciled against that
summarizer.

Snapshot time: 2026-06-22 ~22:05 America/Bogota, reconciled against lane status
refresh `results/reports/atlas_lane_registry_status_20260623_030548.{md,json,tsv}`
(regenerate with `scripts/reports/refresh_atlas_lane_registry_status.sh`).

| Lane | Current molecular payload | Current use | Claim boundary |
| --- | --- | --- | --- |
| Rumen POC | 555 ESM2 proteomes in the 662-row backbone; 518 MAG/bin-comparable units in the POC functional/gLM2 denominator; 37 no-bin or assembly-context rumen units quarantined from MAG-bin feature tables | source reference lane for methane-system molecular neighborhoods and bridge hypotheses | source and ecosystem remain confounded; rumen evidence is not direct wetland risk evidence |
| Wetland/MUCC POC | 107 ESM2 proteomes; 107 MAG/bin-comparable functional outputs; 107 gLM2 contextual units | target-domain wetland POC lane for bridge-candidate validation and molecular-attestation cards | MAG-level potential only; sample abundance, environmental covariates, and flux validation are incomplete |
| Mangrove/MSM expansion | 1,428 local MAG/proteome candidates; 1,428 ESM2 embeddings complete; 1,428 gLM2 units complete; 1,225 functional MAGs complete (manifest-scoped), 5 partial/running, 198 not yet started, 0 failed | target-domain expansion lane for broader blue-carbon molecular niche-space and future sample-level readiness | local 1,428-candidate denominator must be reconciled with the paper-reported 966 final medium/high-quality MAGs before final ecological denominators or sample MRV rollups |
| Mangrove/Futian 2026 expansion (new) | 3,404 phase-1 dereplicated rMAGs at 99% ANI (3,156 ready payload rows + 248 explicit missing-payload gap rows); 1,425/3,156 ESM2 complete (in progress); 3,156 gLM2 complete; 114 functional MAGs complete (archaea shard only), 2 partial, 0 failed, 3,040 not yet started | newest blue-carbon mangrove target lane; broadens molecular niche-space and future sample-level readiness | local 3,156-ready denominator; archaea-only functional coverage so far; no consolidated warehouse yet |

The POC rumen + wetland MAG-bin layer is already coherent for MBAG reporting:
625/625 MAG-bin units have ESM2, functional warehouse rows, and gLM2 context.
Current tri-view-ready units across all lanes total **1,964** (625 POC core +
1,225 MSM + 114 Futian). Both mangrove lanes have complete or in-progress ESM2
and gLM2 coverage but their functional annotation tranches are still moving and
neither has a consolidated warehouse yet; rebuild the expanded atlas only after
a deliberate interim snapshot or lane completion.

Live functional jobs on Apolo-3 at snapshot time:

- MSM China 2025 functional array (`8810`): finishing the remaining ~200 MAGs;
  self-completing.
- Futian functional arrays: archaea (`10557`, 1-312) running; three bacteria
  shards (`10560`/`10561`/`10562`, 948 each) PENDING in a strict sequential
  `afterok` dependency chain (archaea -> 001 -> 002 -> 003).
- Futian ESM2 embedders (`10572`-`10575`, sharded) running; gLM2 already
  complete.

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

Snapshot status for the local 1,428-candidate archive (2026-06-22 ~22:05 Bogota):

| Item | Value |
| --- | ---: |
| Complete functional MAGs (manifest-scoped, authoritative) | 1,225 / 1,428 |
| Completion fraction | 85.78% |
| Partial/running MAGs | 5 |
| Failed MAGs | 0 |
| Not yet started | 198 |
| Duplicate complete attempts observed | 1 |
| Completed per-MAG runs scanned for tables below | 1,220 |
| Domain: Archaea complete (scan) | 67 / 80 |
| Domain: Bacteria complete (scan) | 1,153 / 1,348 |

The authoritative manifest-scoped complete count (1,225, from the lane status
refresh) leads the per-MAG run-record deep-scan (1,220 runs) used for the
aggregate-row and QC tables below: that deep-scan was taken ~2 h earlier in the
same evening window and MSM jobs are actively completing (array `8810`), so it
trails the live count by a few MAGs. The small persistent gap also reflects
duplicate/near-duplicate completed attempts preserved on disk — the same
attempts-vs-unique distinction the POC warehouse uses (683 attempts -> 625
selected). The aggregate-row and QC magnitudes are stable at this scale; treat
them as representative until a manifest-denominator consolidation is run.

Aggregate rows across the 1,220 completed mangrove/MSM runs on disk (summed from
per-run `curated/parquet_manifest.tsv`; not yet a consolidated warehouse):

| Table | Rows |
| --- | ---: |
| `fact_kofam_hits` | 83,479,724 |
| `fact_scycdb_hits` | 6,033,385 |
| `fact_mcycdb_hits` | 4,219,114 |
| `fact_bakta_features` | 3,454,183 |
| `fact_metabolic_module_step_presence` | 2,441,220 |
| `fact_metabolic_module_presence` | 573,400 |
| `fact_metabolic_hmm_hits` | 384,300 |
| `fact_dbcan_hits` | 171,356 |
| `fact_metabolic_function_presence` | 126,880 |
| `fact_merops_hits` | 27,598 |
| `fact_cazy_hits` | 16,705 |
| `fact_qc_checkm2` | 1,220 |
| `fact_qc_gunc` | 1,220 |

QC snapshot across the 1,220 completed mangrove/MSM runs (MIMAG-style buckets:
HQ = completeness >90% and contamination <5%; MQ = completeness >=50% and
contamination <=10%; otherwise LQ/QC-gated):

| QC field | Value |
| --- | ---: |
| High-quality-like MAGs | 100 |
| Medium-quality-like MAGs | 572 |
| Lower-quality or QC-gated MAGs | 548 |
| Median completeness | 74.15% |
| Median contamination | 5.29% |
| GUNC pass | 735 |
| GUNC fail | 485 |

This is already large enough for expanded molecular niche-space exploration, but
not yet a final mangrove functional warehouse. The next consolidation must use
the manifest denominator, preserve partial/failed/duplicate attempts explicitly,
and keep the 966 published-MAG denominator separate from the 1,428 local archive
denominator.

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

Snapshot status (2026-06-22 ~22:05 Bogota):

| Item | Value |
| --- | ---: |
| Phase-1 dereplicated rMAGs | 3,404 |
| Ready payload rows (functional include) | 3,156 |
| Explicit missing-payload gap rows | 248 |
| Functional denominator split | 312 Archaea + 2,844 Bacteria |
| Complete functional MAGs (manifest-scoped, authoritative) | 114 / 3,156 |
| Completed per-MAG runs scanned for tables below | 112 |
| Partial MAGs | 2 |
| Failed MAGs | 0 |
| Not yet started | 3,040 |
| ESM2 embeddings | 1,425 / 3,156 (in progress) |
| gLM2 contextual units | 3,156 / 3,156 complete |

Functional coverage is archaea-only so far (the archaea array runs first; the
three bacteria shards are queued behind it in a strict sequential `afterok`
dependency chain), so the bacterial bulk is the long pole. Aggregate rows across
the 112 completed Futian runs on disk (summed from per-run parquet manifests):

| Table | Rows |
| --- | ---: |
| `fact_kofam_hits` | 3,252,053 |
| `fact_metabolic_module_step_presence` | 224,112 |
| `fact_scycdb_hits` | 214,830 |
| `fact_mcycdb_hits` | 198,031 |
| `fact_bakta_features` | 178,001 |
| `fact_metabolic_module_presence` | 52,640 |
| `fact_metabolic_hmm_hits` | 35,280 |
| `fact_metabolic_function_presence` | 11,648 |
| `fact_dbcan_hits` | 4,228 |
| `fact_merops_hits` | 956 |
| `fact_cazy_hits` | 455 |
| `fact_qc_checkm2` | 112 |
| `fact_qc_gunc` | 112 |

QC snapshot across the 112 completed Futian archaea runs:

| QC field | Value |
| --- | ---: |
| High-quality-like MAGs | 20 |
| Medium-quality-like MAGs | 92 |
| Lower-quality or QC-gated MAGs | 0 |
| Median completeness | 81.13% |
| Median contamination | 0.94% |
| GUNC pass | 110 |
| GUNC fail | 2 |

The archaea tranche is high-quality (median completeness 81%, near-zero
contamination, 110/112 GUNC pass), but it is a small, taxonomically biased slice
of the lane; do not generalize lane-level QC until the bacteria shards complete.
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
| Functional annotation | 1,225 / 1,428 complete at the 2026-06-22 snapshot | `results/functional_metagenomics/msm_china_2025_20260615/` |
| Sample/environment metadata | 82 local GigaDB sediment-sample metadata rows and 71 exact BioSample environmental rows | `data/external/msm_china_2025/metadata/` and `data/external/msm_china_2025/gigadb_wasabi/` |

Sample-level claims remain blocked until the MAG-to-sample mapping, MAG/read
abundance or coverage, environmental covariates, and validation outcomes are
joined with explicit resolution tiers.

## Latest Partner-Facing Report Artifacts

Latest generated expanded HTML atlas:

```text
results/reports/mbag_nextgen_molecular_niche_atlas_20260619_113355/report.html
```

That report is a strong partner-facing narrative artifact, but it was generated
before the latest mangrove/MSM functional completion snapshot and predates the
Futian lane entirely. It currently records 848 mangrove functional units and
1,473 tri-view units. With the 2026-06-22 payload snapshot, an interim rebuild
would be expected to expose roughly 1,964 MAG/proteome-level tri-view units
(625 POC core + 1,225 MSM + 114 Futian), subject to the report builder's
manifest and QC filters. Treat the published/static report as a dated artifact
until rebuilt.

## Molecular Attestation Graph

Current graph snapshot:

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

## Core Docs To Keep In Sync

| Document | Role |
| --- | --- |
| `README.md` | public-facing project overview and current artifact arc |
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
