# MethaNet Final MRV Risk Scoring Roadmap

Date: 2026-06-13
Documentation refresh: 2026-08-10

Scope: strategic roadmap for moving MethaNet from the current MBAG/functional-atlas molecular screening layer toward defensible sample/project-level methane permanence risk scoring for blue carbon MRV.

The canonical product language and release claim matrix live in
`docs/methanet_positioning_and_claims.md`.

## Agent Loading Contract

Any agent working on MethaNet MRV, MBAG, bridge candidates, functional atlas interpretation, blue carbon risk tiers, carbon-crediting language, or sample/metagenome rollups should read this file before proposing analyses, claims, dashboards, model outputs, or partner-facing reports.

This roadmap should be used together with:

- `ai_docs/functional_metagenomics_expansion/data_aggregation_strategy.md`
- `ai_docs/functional_metagenomics_expansion/cohort_data_architecture_hardening.md`
- `ai_docs/functional_metagenomics_expansion/output_contracts_and_gates.md`
- `ai_docs/functional_metagenomics_expansion/pipeline_reproducibility_contract.md`
- `ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/methanet_embedding_functional_transfer_framework.md`
- `ai_docs/functional_metagenomics_expansion/glm_contextual_genomics_strategy_20260615/README.md`

## Executive Summary

MethaNet currently has a strong foundation for molecular methane-risk screening:

- a 662-proteome ESM2 geometry-aware rumen + wetland/MUCC POC;
- a clean proteome/MAG crosswalk;
- local MAG/proteome file matching;
- an Apollo-3 functional-genomics production stack for QC, taxonomy, KOfam, MCycDB, SCycDB, dbCAN, Bakta, CheckM2, GUNC, METABOLIC, CAZy, MEROPS, timing, and status records;
- a completed POC MAG-bin multi-view layer with 625/625 units carrying ESM2, functional annotation, and gLM2 contextual evidence;
- a mangrove/MSM expansion lane with 1,428/1,428 ESM2, gLM2, and validated functional payloads complete;
- a Futian 2026 mangrove/mudflat expansion lane with 3,156/3,156 ready MAG/proteome units complete in ESM-2, gLM2, and the validated functional warehouse, plus 248 explicit source gaps;
- a MUCC v1 Old Woman Creek wetland lane with 2,508 registered MAGs, 2,501 ESM-2 embeddings, 2,508 gLM2 and source-functional payloads, 2,501 data-complete source-scaffold tri-views, and processed expression detection;
- an MBAG design that integrates latent proteome geometry with functional evidence, QC, source leakage, graph structure, and uncertainty.

The current product primitive is:

> MAG/proteome molecular attestation, evidence-card review, bridge-candidate prioritization, and monitoring-readiness design.

Final MRV risk scoring requires a harder target:

> Sample/project-level, uncertainty-calibrated methane permanence risk, supported by molecular evidence, abundance, environmental context, repeated observations, and external validation against methane or net GHG measurements.

The next product-grade layer is the **MethaNet Sample Risk Readiness Layer**,
which marks each sample or project unit as `scoreable`, `monitor_more`,
`needs_metadata`, `needs_abundance`, `needs_flux_validation`, or `blocked`.

## Current State

This is the current defensible state, based on local MethaNet docs, generated functional-atlas warehouses, and the molecular attestation graph snapshot. Live production counts may move during Apollo-3 runs, so operational status must be refreshed before any external report.

| Layer | Current status | What it supports now | What it does not support yet |
| --- | --- | --- | --- |
| ESM-2 proteome geometry | 7,710 release units across POC, MSM, Futian, and MUCC v1 | hypothesis generation, molecular niche-space mapping, and bridge prioritization | source-independent transfer proof or activity |
| Proteome/MAG crosswalk | authoritative 662-row POC mapping plus unit-scope manifests; MSM and Futian source-lane manifests under `data/external/` | stable joins within each lane across embeddings, MAGs, functional outputs, gLM2, attestation, and metadata | sample/project-level rollup by itself |
| Functional atlas | 5,209 pipeline-normalized POC/MSM/Futian units with cross-lane comparability pending; 2,508 MUCC v1 source-functional payloads with 2,501 data-complete source-scaffold tri-views; 0 mechanism-comparable units authorized | MAG-level mechanism evidence, QC/taxonomy, annotation coverage, expression detection, run-status audit, and MRV feature primitives | direct activity, abundance-weighted sample capacity, methane flux, or cross-lane pathway strength |
| gLM2 context | 7,717 release payloads; 5,209 single-window units and 2,508 MUCC multiwindow units | independent genomic-context sidecar for molecular neighborhood interpretation | direct pathway evidence; cross-protocol numerical comparison |
| MBAG / attestation | 7,965 registered units, 7,710 data-complete tri-views, explicit evidence contracts, candidate cards, provenance, claim wording, 255 source gaps, and next actions; foundational queryable MVP remains the 662-node POC graph | molecular diligence, bridge review, evidence provenance, missingness visibility, monitoring prioritization, and claim-safe querying | calibrated sample/project risk model |
| Historical smoke reports | dated snapshots such as 121/662 complete, 625 POC complete, and expanded HTML atlas snapshots | demonstration of report logic and validation gaps at the time they were generated | current production denominator unless refreshed |
| Sample metadata | incomplete/mixed resolution across rumen, wetland/MUCC, MSM, and Futian; MSM has BioSample-linked context, Futian has 65 exact sample metadata rows but MAGs resolve mainly to site/month rather than depth-resolved samples | provenance context where available | sample-level ecological inference |
| Abundance/coverage | not yet integrated as final rollup layer | future weighting design | community-level capacity estimates |
| Environmental covariates | not yet joined | future methane permissiveness model | site-specific risk scoring |
| Flux/process validation | planned/targeted, not integrated | validation roadmap | calibrated A-E risk tiers |

Current generated evidence snapshots that define the implemented state:

| Artifact | Path | Current use |
| --- | --- | --- |
| 662 proteome crosswalk | `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv` | canonical ESM2/cohort backbone |
| Unit-scope manifest | `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv` | 625 MAG/bin + 37 assembly-context split |
| Functional atlas warehouse | `results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/` | 24 Parquet tables plus DuckDB over 625 selected MAG/bin runs |
| Metadata recovery | `results/functional_metagenomics/environmental_metadata_recovery_20260612/` | exact/contextual source provenance and environmental metadata tiers |
| Molecular attestation graph | `results/attestation/mmag_mvp_20260617/` | graph nodes/edges/evidence/artifacts/query/audit snapshot |
| Mangrove/MSM ESM2 | `results/blue_catalyst_poc/runs/msm_china_2025_esm2_20260616_082112/artifacts/` | 1,428/1,428 local mangrove/MSM proteome embeddings |
| Mangrove/MSM gLM2 | `results/contextual_genomics/glm2_msm_magbin_full_20260615_092737/` | 1,428/1,428 contextual genome units |
| Mangrove/MSM functional warehouse | `results/functional_metagenomics/msm_china_2025_20260615/cohort_warehouse/` | 1,428/1,428 selected complete; failed, partial, and superseded attempts preserved |
| Mangrove/Futian ESM2 | `results/blue_catalyst_poc/runs/futian_mangrove_2026_esm2_phase1_shard*_20260621/` | 3,156/3,156 ready Futian proteome embeddings |
| Mangrove/Futian gLM2 | `results/contextual_genomics/glm2_futian_phase1_shard*_20260621/` | 3,156/3,156 contextual genome units |
| Mangrove/Futian functional warehouse | `results/functional_metagenomics/futian_mangrove_2026_phase1/cohort_warehouse/` | 3,156/3,156 selected complete under normalized accepted/best/present event semantics; 248 source gaps remain explicit |
| MUCC v1 wetland lane | `results/functional_metagenomics/mucc_v1_owc_wetland_20260626/` | 2,508 registered MAGs with source-functional, expression, network, and staged field-validation evidence |
| Freeze-backed expanded atlas | `results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/report.html` | controlled-diligence molecular-attestation report over 7,965 registered units and 7,710 data-complete tri-views; publication gates remain pending |
| Current payload inventory | `docs/current_artifact_inventory.md` | dated cross-lane status, live-count caveats, and report freshness notes |
| Three-view freeze plan | `ai_docs/functional_metagenomics_expansion/three_view_freeze_and_nextgen_report_plan.md` | registry-driven rule for freezing POC + MSM + Futian report payloads without losing pending, partial, failed, duplicate, or gap rows |

## Definition Of Final MRV Risk Scoring

Final MRV risk scoring should not mean "a nice molecular score." It should mean an auditable estimate of methane permanence risk at the decision grain used by a project, monitoring plan, buyer, insurer, validator, or registry-facing workflow.

Recommended formal target:

> Given a blue carbon sample, site, or project unit, estimate the probability that methane emissions materially reduce expected net climate benefit during the relevant monitoring period, conditioned on molecular, ecological, environmental, temporal, and validation evidence.

Recommended output shape:

```text
risk_unit_id
risk_unit_type                  # sample, site, project_polygon, monitoring_period
assessment_date
methane_risk_score_distribution # not only a point estimate
methane_risk_tier               # A-E only when calibrated
confidence_tier
dominant_evidence_drivers
dominant_uncertainty_drivers
recommended_monitoring_action
claim_status
allowed_claim_wording
validation_status
audit_packet_uri
```

The score should be a distribution or interval before it is a single class. If the evidence is insufficient, the correct output is `not_scoreable`, not a forced tier.

## Non-Negotiable Claim Boundaries

| Claim | Current status | Allowed now | Required for upgrade |
| --- | --- | --- | --- |
| MAGs encode methane/sulfur/substrate functional potential | supported for completed MAGs | "completed MAGs carry QC-aware functional evidence" | full cohort consolidation and validation gates |
| Bridge candidates are biologically plausible | provisional | "bridge candidates are hypotheses with direct evidence where available" | full top-candidate functional completion, stable ranks, source-aware nulls |
| MBAG score is calibrated | not yet | "MBAG gives provisional internal prioritization" | external labels, source-replicated cohorts, sample/flux validation |
| Sample methane risk can be assigned | not yet | "sample risk requires abundance, environmental context, and validation" | sample mapping, MAG/read abundance, environmental covariates, uncertainty propagation |
| Final A-E risk tiers can be assigned | not yet | "A-E tiers are target product vocabulary" | calibrated sample/project model with holdout validation |
| Carbon credits can be approved from MethaNet output | not allowed | "MethaNet can support screening and monitoring design" | formal methodology integration, third-party validation, registry-aligned evidence |

## Maturity Ladder

### Level 0: MAG Molecular Screening

Status: active/current.

Core evidence:

- ESM2 proteome embeddings;
- MAG functional annotations;
- QC/taxonomy;
- annotation coverage;
- source leakage warnings;
- MBAG candidate cards.

Allowed output:

- bridge candidate cards;
- MAG-level mechanism confidence;
- molecular screening feature table;
- monitoring-priority hypotheses.

Blocked output:

- sample risk;
- measured flux;
- A-E final risk tier;
- crediting or registry claims.

Exit criteria:

- complete selected functional runs for all MAG/bin-comparable target units;
- regenerate cohort warehouse from curated per-MAG evidence bundles;
- pass identity, schema, QC, coverage, and no-wide-METABOLIC validation gates;
- preserve failed, partial, superseded, and assembly-context evidence as explicit rows;
- produce candidate cards or graph evidence paths with direct and missing evidence separated.

Current status for Level 0: substantially implemented for the 625
MAG/bin-comparable POC functional-atlas denominator in
`cohort_warehouse_poc_magbin_union_20260616_075022`, with an attestation MVP in
`results/attestation/mmag_mvp_20260617`. The mangrove/MSM expansion is moving
Level 0 from a two-source POC toward a broader blue-carbon molecular niche atlas:
ESM-2 and gLM2 are complete for both MSM (1,428 units) and Futian (3,156 ready
units). MSM contributes 1,428 and Futian contributes 3,156
pipeline-normalized tri-views. Their code/configuration/database fingerprints
and source-aware statistical comparability gates remain pending. MUCC v1 adds 2,501 data-complete source-scaffold
tri-views under a distinct functional contract. This still does not advance the
system to Level 1 because sample
identity, abundance, environmental covariates, and validation outcomes remain
incomplete.

### Level 1: Sample Identity And Metadata Resolution

Goal: map molecular units into ecological decision units.

Required new entities:

| Table | Grain | Required fields |
| --- | --- | --- |
| `dim_project` | one project or source study | project_id, registry_context, geography, habitat, baseline/project status |
| `dim_site` | one field site or polygon | site_id, project_id, coordinates/polygon, habitat, hydrologic setting |
| `dim_sample` | one physical sample | sample_id, site_id, date, depth, compartment, method, source accession |
| `link_sample_mag` | sample x MAG | sample_id, proteome_id, mag_id, evidence_source, mapping_confidence |
| `fact_metadata_resolution` | sample/MAG/source | resolution_tier, missing_fields, provenance_uri |

Key constraints:

- `proteome_id` can look sample-like, but current functional outputs are MAG/proteome units.
- Do not roll up to sample level until the sample/MAG link is explicit and provenance-ranked.
- Every environmental context field must have a resolution tier: exact sample, site, project, publication, inferred, or missing.

Exit criteria:

- every MAG used in sample-level analysis maps to a sample or is explicitly marked unlinked;
- every sample has habitat, source, collection date or date tier, and provenance;
- unresolved samples are preserved as `not_scoreable_metadata`.

### Level 2: Abundance And Community Capacity

Goal: convert genome potential into sample/community potential.

Required evidence:

- MAG read coverage per sample;
- relative abundance per MAG;
- ideally absolute abundance from qPCR, ddPCR, spike-ins, or calibrated read counts;
- marker abundance for methanogenesis, methane oxidation, sulfur competition, and substrate routes;
- unbinned read or contig-level marker evidence to avoid MAG-only blind spots.

Required tables:

| Table | Grain | Required fields |
| --- | --- | --- |
| `fact_mag_abundance` | sample x MAG | coverage, relative_abundance, absolute_abundance_optional, method, uncertainty |
| `fact_marker_abundance` | sample x marker/family | marker_id, normalized_abundance, copy_number_model, evidence_source |
| `fact_unbinned_function_evidence` | sample x function | function_id, read_support, contig_support, confidence |
| `feature_sample_molecular_capacity` | sample | methanogenesis_capacity, oxidation_capacity, sulfur_competition_capacity, substrate_capacity, uncertainty |

Important principle:

> An abundant weak methanogen may matter more than a rare complete methanogen; a complete but nearly absent MAG should not dominate sample risk.

Exit criteria:

- sample-level features are abundance-weighted;
- unweighted MAG counts are retained only as supporting diagnostics;
- absent features are distinguished from low-coverage unknowns;
- sample molecular capacity can be recomputed deterministically from raw abundance and functional facts.

### Level 3: Environmental Methane Permissiveness

Goal: model whether the site conditions allow encoded methane potential to become methane emissions.

Minimum environmental covariates:

- salinity;
- sulfate;
- redox potential or proxy;
- oxygen;
- pH;
- temperature;
- water table, tidal inundation, or hydroperiod;
- organic carbon quantity and quality;
- acetate, methanol, methylamines, or methylated sulfur substrates when available;
- vegetation type;
- sediment depth;
- season;
- restoration/intervention status.

Required tables:

| Table | Grain | Required fields |
| --- | --- | --- |
| `fact_environmental_measurement` | sample/site/time x variable | variable, value, unit, method, timestamp, uncertainty |
| `feature_environmental_permissiveness` | sample/site/time | methanogenesis_permissiveness, oxidation_permissiveness, sulfate_competition_context, substrate_context |
| `fact_intervention_context` | project/site/time | restoration_action, hydrologic_change, vegetation_change, management_status |

Interpretation rules:

- Sulfate-rich, oxidized, or strongly tidally flushed settings may suppress methane expression even when genes exist.
- Organic substrate pulses, freshening, anoxia, or hydrologic changes may increase methane risk.
- Environmental permissiveness modifies molecular potential; it does not replace molecular evidence.

Exit criteria:

- each scored sample has environmental covariates or an explicit missingness penalty;
- variables are unit-normalized;
- environmental context is time-aligned to molecular and flux observations;
- environmental uncertainty is propagated into risk confidence.

### Level 4: Flux And Process Validation

Goal: connect molecular/environmental features to measured methane or net GHG outcomes.

Required outcome evidence:

- chamber methane flux;
- eddy covariance or flux tower data where available;
- incubation assays;
- porewater methane;
- ebullition evidence where relevant;
- CO2 and ideally N2O for net GHG context;
- repeated measurements across seasons, tides, hydrologic states, and restoration phases.

Required tables:

| Table | Grain | Required fields |
| --- | --- | --- |
| `fact_flux_measurement` | site/sample/time x gas | gas, flux_value, unit, method, timestamp, uncertainty, detection_limit |
| `fact_process_assay` | sample x assay | assay_type, condition, response_value, unit, uncertainty |
| `link_sample_flux_window` | molecular sample x flux window | time_delta, spatial_delta, match_confidence |

Validation design:

- train on one site/project, test on held-out sites/projects;
- hold out seasons;
- hold out habitat classes where possible;
- compare molecular-only, environment-only, and combined models;
- evaluate calibration, not only rank performance.

Exit criteria:

- paired molecular/environmental/flux records exist for target-domain wetland samples;
- performance is reported on held-out sites or projects;
- uncertainty intervals show calibrated coverage;
- failure cases are documented.

### Level 5: Probabilistic MRV Risk Model

Goal: estimate methane permanence risk with uncertainty.

Recommended model family:

- Bayesian hierarchical model or probabilistic ensemble;
- site/project/habitat random effects;
- explicit measurement error;
- source-aware domain adaptation;
- conservative missingness penalties;
- MBAG as an interpretable molecular prior, not the entire risk model.

Conceptual model:

```text
MAG functions
+ MAG/read abundance
+ unbinned marker evidence
+ ESM2/MBAG bridge evidence
+ sulfur/substrate modifiers
+ site geochemistry
+ habitat/project metadata
+ temporal context
-> methane risk distribution
-> risk tier, confidence, and monitoring action
```

Recommended outputs:

| Output | Grain | Meaning |
| --- | --- | --- |
| `model_mrv_risk_prediction` | sample/site/time | methane risk distribution, tier, uncertainty |
| `risk_driver_decomposition` | prediction x feature group | molecular, abundance, environment, temporal, validation contributions |
| `risk_counterfactual_sensitivity` | prediction x scenario | how risk changes under hydrology, sulfate, vegetation, or sampling uncertainty |
| `risk_validation_status` | model/version | calibration, holdout metrics, blocked claims |

Risk target options:

- probability methane flux exceeds a material threshold;
- expected CH4 CO2e penalty;
- probability methane offsets a material share of CO2 sequestration benefit;
- monitoring intensity class;
- permanence risk modifier;
- A-E risk tier.

Recommended first production target:

> Probability that methane emissions materially reduce expected net climate benefit under the project monitoring period.

Exit criteria:

- risk outputs are calibrated against external target-domain data;
- A-E tiers map to defined quantitative thresholds or decision policies;
- uncertainty is visible and conservative;
- model cards and audit packets are generated for every model version.

### Level 6: MRV Product And Audit Integration

Goal: make MethaNet usable in high-integrity MRV workflows without overstating authority.

MRV-aligned evidence package:

| Artifact | Required content |
| --- | --- |
| `risk_evidence_packet` | input data versions, feature tables, model version, validation status, uncertainty, allowed claim |
| `monitoring_recommendation` | where/when to measure methane next and why |
| `validation_gap_register` | missing evidence blocking stronger claims |
| `project_screening_summary` | risk drivers, confidence, next actions |
| `third_party_review_bundle` | provenance, reproducibility instructions, model card, data dictionary |

External alignment:

- ICVCM Core Carbon Principles emphasize transparency, robust quantification, permanence, and independent validation.
- Verra VCS requirements emphasize project validation, monitoring, verification, robust quantification, durability, traceability, and non-double-counting.
- VM0033 is the relevant tidal wetland/seagrass restoration context, but MethaNet outputs should remain a molecular intelligence layer unless formally integrated into an approved methodology workflow.

Exit criteria:

- MethaNet risk output is reproducible from immutable inputs;
- every risk score has an uncertainty interval and validation status;
- every claim maps to allowed wording;
- third-party reviewers can inspect the evidence chain;
- registry-facing language is reviewed separately from scientific/product language.

## MethaNet Sample Risk Readiness Layer

This is the recommended next product primitive.

It should be built before final MRV scoring because it creates immediate business value while preserving scientific integrity.

Required output:

```text
sample_id
site_id
project_id
assessment_date
linked_mag_count
linked_mag_abundance_coverage
molecular_methane_potential
methane_oxidation_potential
sulfur_competition_strength
substrate_permissiveness
environmental_permissiveness_status
flux_validation_status
metadata_resolution_tier
uncertainty_tier
readiness_label
blocking_gaps
next_action
allowed_claim_wording
```

Recommended readiness labels:

| Label | Meaning | Action |
| --- | --- | --- |
| `scoreable_provisional` | enough molecular, abundance, context, and validation support for provisional sample risk | generate risk distribution with caveats |
| `monitor_more` | evidence suggests possible methane risk but validation is sparse | target field measurements |
| `needs_metadata` | molecular evidence exists but sample/site context is insufficient | recover metadata |
| `needs_abundance` | MAG evidence exists but sample weighting is missing | map reads or add abundance assay |
| `needs_environment` | molecular potential exists but site permissiveness is unknown | collect/geocode environmental covariates |
| `needs_flux_validation` | features are complete but outcome calibration is absent | pair with chamber/flux/incubation data |
| `blocked_qc` | molecular evidence is unreliable | repair or exclude affected MAG/sample |
| `not_scoreable` | required evidence is absent | do not assign risk tier |

## Data Architecture Additions Needed

The current MAG-level atlas should remain Parquet-first and DuckDB-queryable. The final MRV layer adds sample/project grains.

Minimum new schema:

```text
dim_project
dim_site
dim_sample
link_sample_mag
fact_mag_abundance
fact_marker_abundance
fact_unbinned_function_evidence
fact_environmental_measurement
fact_intervention_context
fact_flux_measurement
fact_process_assay
link_sample_flux_window
feature_sample_molecular_capacity
feature_environmental_permissiveness
feature_sample_risk_readiness
model_mrv_risk_prediction
risk_driver_decomposition
risk_validation_status
risk_evidence_packet_manifest
```

All tables should carry:

- `assessment_run_id`
- `cohort_run_id` where molecular atlas evidence is used
- `sample_id`, `site_id`, or `project_id` at the appropriate grain
- `proteome_id` and `mag_id` when MAG evidence contributes
- `source_tool` or `evidence_source`
- provenance URI/path
- version or timestamp
- uncertainty fields where applicable

## Statistical Validation Gates

Final MRV scoring requires these gates, not just high model accuracy.

| Gate | Required test | Failure interpretation |
| --- | --- | --- |
| Identity gate | no duplicated IDs; all joins preserve denominators | score not reproducible |
| Source leakage gate | source/project labels not driving risk claims | model may learn provenance artifacts |
| Site holdout gate | leave-site/project-out validation | model may not transfer |
| Season holdout gate | temporal holdout | model may fail under hydroseasonality |
| Habitat holdout gate | mangrove/saltmarsh/seagrass transfer checks | habitat-specific overfit |
| Molecular ablation | compare molecular-only vs environment-only vs combined | molecular layer may add no value |
| Abundance ablation | weighted vs unweighted MAG evidence | MAG count artifacts may dominate |
| Flux calibration | calibration curve and interval coverage | point estimates may be misleading |
| Conservative missingness | missing evidence returns wider intervals or `not_scoreable` | false precision risk |
| External validation | independent cohort or partner dataset | not final MRV-ready |

## Product Roadmap

### Immediate: After Functional Production Completes

Deliverables:

- regenerate cohort warehouse;
- produce full MBAG candidate cards;
- produce `feature_mrv_mag_level`;
- produce validation gap register;
- update infographic/report language to avoid final-risk implication.

Decision unlocked:

> Which MAGs and mechanisms deserve sample-level follow-up?

### Near Term: Sample Risk Readiness

Deliverables:

- sample/MAG metadata resolver;
- sample-level abundance/read-coverage table;
- sample risk readiness table;
- monitoring-priority report for any sample set with enough metadata.

Decision unlocked:

> Which samples/sites are ready for molecular risk screening, and what data blocks the rest?

### Mid Term: Paired Validation Dataset

Deliverables:

- paired molecular, environmental, and flux/process dataset;
- validation split design;
- baseline predictive models;
- uncertainty and calibration audit.

Decision unlocked:

> Does MethaNet molecular evidence improve methane-risk prediction beyond environmental covariates alone?

### Later: Calibrated MRV Risk Model

Deliverables:

- probabilistic sample/site risk model;
- A-E tier mapping with uncertainty;
- project-level aggregation;
- model card and third-party review packet.

Decision unlocked:

> Can MethaNet support verifier-facing methane permanence risk assessment as a validated molecular intelligence layer?

### Registry-Facing Future

Deliverables:

- methodology-aligned evidence package;
- independent validation;
- external technical review;
- formal positioning relative to VM0033 or other applicable methods.

Decision unlocked:

> Can MethaNet be incorporated into formal MRV workflows as a recognized supporting evidence layer?

## Practical Next Actions

1. Finish and consolidate the 662-MAG functional atlas.
2. Generate full MBAG candidate cards and update the evidence-tier dashboard.
3. Build the `feature_sample_risk_readiness` schema before building any final risk tier model.
4. Identify the authoritative wetland/MUCC sample metadata source and assign metadata resolution tiers.
5. Define the MAG/sample abundance workflow: read mapping, marker read counts, qPCR/ddPCR, or public coverage recovery.
6. Inventory candidate public or partner datasets with paired methane flux/process measurements.
7. Draft a validation study design with leave-site/project/season holdouts.
8. Reframe external communications around "molecular attestation for screening and monitoring design" until final validation exists.

## References And External Alignment

- ICVCM Core Carbon Principles: https://icvcm.org/core-carbon-principles/
- ICVCM Assessment Framework: https://icvcm.org/assessment-framework/
- Verra VM0033 Methodology for Tidal Wetland and Seagrass Restoration, v2.1: https://verra.org/methodologies/vm0033-methodology-for-tidal-wetland-and-seagrass-restoration-v2-1/
- Verra VCS Program Details: https://verra.org/programs/verified-carbon-standard/vcs-program-details/

## One-Line Guardrail

Until sample mapping, abundance, environmental covariates, uncertainty propagation, and flux/process validation are in place, MethaNet should report **molecular risk features and monitoring priorities**, not final MRV risk scores.
