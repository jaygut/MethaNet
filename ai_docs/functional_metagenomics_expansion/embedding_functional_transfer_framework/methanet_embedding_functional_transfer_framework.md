# MethaNet Embedding-Functional Transfer Framework

Date: 2026-06-13

Scope: design framework for integrating the 662-proteome geometry-aware ESM2 POC with the Apollo-3 functional-genomics atlas in support of MethaNet bridge-candidate interpretation and methane permanence-risk MRV product development.

Status note, 2026-07-24: this document remains the conceptual MBAG framework,
while its original operational counts are historical. The current
scientific-reconciliation release registers 7,965 units, 7,710 ESM-2
embeddings, 7,717 gLM2 payloads, and 7,484 data-complete tri-views. It keeps
625 mechanism-comparable POC rows, 4,358 annotation-complete mangrove rows
awaiting common feature aggregation, and 2,501 MUCC v1 source-scaffold rows
separate. Use `docs/methanet_positioning_and_claims.md` and
`docs/current_artifact_inventory.md` for current language, counts, and release
status.

This is a strategy/design artifact. It does not submit jobs, alter production outputs, or regenerate cohort warehouses.

Hard-audit revision note, 2026-06-13: this document was reviewed under a severe scientific, statistical, graph-ML, MRV, and business-claim standard. The revisions below deliberately separate hypothesis generation from validation, MAG-level potential from sample/metagenome inference, and molecular screening from carbon-crediting claims.

## Executive Summary

MethaNet now has the right ingredients for a defensible "bridge candidate" research and product-intelligence layer: a 662-proteome ESM2 geometry run, a clean MAG/proteome crosswalk, complete local MAG FASTA/proteome matching, and a production functional-genomics stack emitting curated per-MAG Parquet evidence across QC, taxonomy, KOfam, MCycDB, SCycDB, dbCAN, Bakta, CheckM2, GUNC, METABOLIC, CAZy, MEROPS, and timing/status records.

The strongest scientific claim today is not "we have proven source-independent methane MRV transfer." The stronger and correct claim is:

> MethaNet has a source-audited functional atlas architecture that can test whether ESM2 latent bridge candidates between rumen methane systems and blue-carbon wetland MAGs are supported by separately computed methane, sulfur, substrate, taxonomy, QC, and annotation-coverage evidence.

The immediate opportunity is to convert bridge candidates into **MethaNet Bridge Attestation Cards**: one candidate per card, with latent-geometry evidence, functional mechanism evidence, QC/coverage penalties, source/domain caveats, and an explicit confidence tier. This directly supports MethaNet's business objective: a molecular attestation layer that helps blue carbon developers, technical validators, scientific partners, insurers, and buyers decide where follow-up methane-risk measurement is most justified. It does not yet support stand-alone credit issuance, registry approval, or field-validated permanence claims.

The recommended core algorithm is the **MethaNet Bridge Attestation Graph (MBAG)**. MBAG fuses multiple MAG similarity views, estimates auditable rumen-wetland couplings with optimal transport, propagates weak mechanism evidence over reliability-aware graphs, and reports a provisional Bridge Attestation Score with uncertainty. It is deliberately not a black-box classifier. It is a graph-based evidence system built to be inspected, ablated, challenged, and packaged into partner-facing molecular-attestation artifacts. The score becomes "calibrated" only after source-replicated cohorts and external sample/flux validation exist.

Historical local status at framework generation time, superseded by the
2026-06-20 inventory in `docs/current_artifact_inventory.md`:

| Layer | Local status | Interpretation |
| --- | ---: | --- |
| Embedded final POC cohort | 662 proteomes | 555 rumen + 107 wetland/MUCC |
| Local MAG FASTA match | 662/662 | functional run denominator is clean |
| Local proteome FAA match | 662/662 | embedding/function joins can use `proteome_id` |
| Functional production snapshot | 121 complete, 0 failed | historical smoke state: 107 wetland/MUCC + 14 rumen complete as of the last local status pass |
| Top ESM2 bridge rumen candidates | mostly not functionally complete yet | full bridge-card validation should wait for those MAGs |
| One top wetland bridge candidate | functionally complete | can serve as a wetland-side smoke-test card |
| Cohort warehouse | valid calibration snapshot, stale versus live run | regenerate after intended tranche/full run completion |

## Hard-Audit Position

The framework is promising only if it refuses three attractive but invalid shortcuts:

1. A visually separated ESM2 manifold is not mechanism proof.
2. A MAG-level methane/sulfur/substrate marker profile is not a sample-level methane flux estimate.
3. A molecular risk screen is not a carbon-crediting decision.

The defensible near-term claim is narrower and stronger: MethaNet can build a source-audited, QC-aware evidence system that prioritizes bridge candidates and identifies which methane-risk measurements would most increase confidence in blue carbon projects.

### Reviewer-Level Corrections Applied

| Risk found in prior framing | Correction applied in this revision |
| --- | --- |
| "Calibrated" bridge score implied calibration before external labels exist | Renamed current score as provisional; calibration requires source-replicated cohorts and field/sample validation |
| OT/domain alignment could be read as solving source confounding | Reframed OT as an auditable coupling hypothesis, not deconfounding proof |
| Source-aware nulls were too weak because source and ecosystem are currently confounded | Added explicit leakage audits, negative controls, and a future-source requirement before source-independent claims |
| Multipanel figure could become visually persuasive without being evidence-defensible | Rebuilt it as a next-generation intelligence report with evidence ledger, uncertainty, missingness, and claim locks |
| MRV language could sound credit-decisional | Restricted current use to screening, monitoring design, validation prioritization, and partner/investor proof points |
| Graph propagation could be misread as creating mechanism evidence | Clarified that propagated evidence is secondary support and cannot substitute for direct marker calls |

## Local Source Grounding

### Authoritative Local Sources

The framework uses these local sources as the source of truth:

| Source | Role |
| --- | --- |
| `README.md` | MethaNet value proposition, ESM2 POC headline metrics, bridge-candidate framing |
| `docs/functional_metagenomics_expansion.md` | phase gates from latent geometry to MRV-ready functional evidence |
| `ai_docs/functional_metagenomics_expansion/data_aggregation_strategy.md` | cohort backbone, layer model, join strategy, claim boundaries |
| `ai_docs/functional_metagenomics_expansion/cohort_data_architecture_hardening.md` | Parquet-first functional atlas table model and validation gates |
| `ai_docs/functional_metagenomics_expansion/pipeline_reproducibility_contract.md` | non-negotiable invariants and allowed/not-allowed claims |
| `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv` | authoritative embedded proteome list |
| `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/README.md` | local MAG FASTA/proteome coverage audit |
| `results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts/` | 662-genome ESM2 embedding, projection, and bridge artifacts |
| `results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag/` | live per-MAG curated functional evidence bundles |
| `results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse/` | calibration-valid cohort Parquet/DuckDB architecture snapshot |
| `results/functional_metagenomics/environmental_metadata_recovery_20260612/` | metadata resolution and caveat layer |
| `results/functional_metagenomics/fgx_662_apollo3_20260612/reports/methanet_atlas_smoke_20260613/` | preliminary smoke report from 113 complete MAGs |

### Local Evidence Summary

The 662-genome ESM2 POC reported:

- 662 embedded proteomes: 107 wetland/MUCC and 555 rumen.
- ESM2-650M, layer 33, mean-pooled genome/proteome vectors with 1280 dimensions.
- Zero attrition and zero non-finite embeddings.
- PERMANOVA ecosystem signal around R2 ~0.202 to 0.206 with p=0.001.
- Silhouette around 0.398 to 0.410 depending on artifact/report source.
- Wetland-vs-rumen classifier separability near AUC 1.0, but with a major caveat: source and ecosystem are confounded.
- Bridge artifacts including `bridge_top_candidates.tsv`, `bridging_genomes_top.tsv`, `bridge_knn_neighborhoods.tsv`, and projection coordinates.

The top ESM2 bridge candidates are dominated by rumen Archaea, which is biologically plausible for methane transfer hypotheses because conserved methanogenesis machinery can be shared across otherwise distant ecological contexts. The local bridge table also includes wetland candidates such as `mucc__GCA_002495465.1_ASM249546v1_genomic`.

Functional production has already shown the pipeline can generate high-volume,
long-form analytical evidence. The original 121-MAG smoke state demonstrated
the curation contract; later POC and mangrove/MSM payload counts are tracked in
`docs/current_artifact_inventory.md`. At the smoke-test stage, per-MAG manifests
contained rows across:

- `fact_kofam_hits`
- `fact_mcycdb_hits`
- `fact_scycdb_hits`
- `fact_dbcan_hits`
- `fact_bakta_features`
- `fact_qc_checkm2`
- `fact_qc_gunc`
- normalized METABOLIC tables
- `fact_cazy_hits`
- `fact_merops_hits`
- `fact_tool_timing`

The early smoke report at 113 completed MAGs already observed all completed MAGs had GTDB-Tk taxonomy in run records, CheckM2/GUNC evidence, MCycDB and SCycDB best-hit availability, dbCAN hits, and substantial KOfam evidence. It correctly warned that rumen completion was pilot-scale and that sample/metagenome claims require abundance, coverage, environmental metadata, and field validation.

### Critical Local Caveats

1. `proteome_id` is the canonical key. Do not let downstream success define the cohort.
2. Current POC source design has source/ecosystem confounding: rumen = PRJEB31266 and wetland = MUCC.
3. Functional production is active; live per-MAG sentinels are fresher than the warehouse.
4. Most top ESM2 rumen bridge candidates were not complete in functional production at the time this framework was generated.
5. MAG-level functional potential is not sample/metagenome-level methane flux.
6. Missing functions must be interpreted with CheckM2 completeness, contamination, GUNC, and annotation coverage.
7. Environmental metadata resolution is mixed: rumen analysis accessions are strong; many MUCC rows remain source/site/project-level rather than sample-level.

## Business And Product Relevance

MethaNet's product opportunity is not another annotation report. The product primitive is **molecular attestation for methane permanence risk** in blue carbon systems.

The bridge framework supports four product surfaces:

| Product surface | What the framework produces | Business use |
| --- | --- | --- |
| Bridge Attestation Cards | Candidate-level evidence cards with latent geometry, functional support, QC, uncertainty, and caveats | Partner demos, scientific review, investor/grant proof points |
| MRV Feature Table | MAG-level methane/sulfur/substrate/QC features with confidence tiers | API/dashboard primitive for project screening |
| Risk Readiness Register | What can/cannot be claimed for each candidate/sample | Avoids overclaiming; aligns with verifier expectations |
| Validation Roadmap | Exact evidence needed to upgrade claims | Guides field data, abundance mapping, and source expansion |

This is relevant to carbon markets because high-integrity credits require transparent quantification, monitoring, additionality/permanence reasoning, and uncertainty treatment. ICVCM's Core Carbon Principles emphasize real, verifiable climate impact, and Verra VM0033 provides procedures to quantify tidal wetland/seagrass restoration greenhouse-gas benefits. MethaNet can become an upstream molecular intelligence layer: not replacing flux accounting, registry methods, or third-party validation, but prioritizing where methane risk may be elevated, which samples need more measurement, and which molecular evidence supports or weakens confidence in permanence-risk hypotheses.

Current business-valid language:

```text
MethaNet provides a molecular evidence layer for methane-risk screening and monitoring design in blue carbon systems.
```

Not yet valid:

```text
MethaNet certifies carbon credits, proves methane permanence, or replaces flux measurements.
```

## Literature-Grounded Rationale

Citation rule: final external reports should cite primary papers, official tool documentation, or registry/methodology documents for each method or claim. Blog posts, package READMEs, and review articles can guide implementation but should not be the sole support for scientific or MRV-facing claims.

### Protein Language Model Embeddings

ESM2 and related protein language models learn representations that encode protein structure and function from sequence. The ESM repository documents ESM2, ESMFold, and the ESM Metagenomic Atlas; Lin et al. used ESMFold/ESM2-style representations to scale structural prediction to hundreds of millions of metagenomic proteins.

MethaNet implication:

- ESM2 proteome embeddings are a plausible hypothesis layer for cross-ecosystem biological similarity.
- They should not be treated as mechanistic proof. They should seed candidate discovery, then be tested against separately computed functional annotation and QC evidence. These evidence channels are orthogonal analyses of related sequence data, not statistically independent observations.
- Proteome-level mean pooling can blur gene-level mechanism; bridge candidates must therefore retain gene/family/module evidence and not rely only on one aggregate vector.

Key sources:

- Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science, 2023: https://www.science.org/doi/10.1126/science.ade2574
- Meta FAIR ESM repository: https://github.com/facebookresearch/ESM
- ESM Metagenomic Atlas: https://esmatlas.com/about
- Protein language model review/application examples: https://academic.oup.com/bib/article/25/3/bbae177/7665115

### Manifold Learning And Biological Geometry

UMAP, PHATE, and diffusion maps are useful complementary views of high-dimensional biological manifolds:

- UMAP is scalable and preserves neighborhood structure for exploratory visualization and graph construction.
- PHATE is designed for biological high-dimensional data and emphasizes local-to-global progression structure.
- Diffusion maps provide a principled graph diffusion view of data geometry and are natural for transition/neighborhood reasoning.

MethaNet implication:

- Use UMAP/PHATE/diffusion maps as visualization and stability diagnostics, not as the primary statistical proof.
- Bridge status should be based on high-dimensional embedding distances, graph neighborhoods, and functional evidence, not just a 2D plot.
- Treat every 2D panel as a communication layer. Statistical decisions should use the original 1280-dimensional embeddings or controlled feature spaces, not UMAP coordinates.

Key sources:

- UMAP: https://joss.theoj.org/papers/10.21105/joss.00861, https://arxiv.org/abs/1802.03426, and docs https://umap-learn.readthedocs.io/
- PHATE: https://pubmed.ncbi.nlm.nih.gov/31796933/ and project page https://krishnaswamylab.org/projects/phate
- Diffusion maps: https://www.pnas.org/doi/10.1073/pnas.0500334102

### Multi-View Integration

Similarity Network Fusion (SNF) integrates multiple data-type similarity networks through cross-diffusion. This is directly relevant because MethaNet has distinct evidence views:

- ESM2 latent geometry
- methane-cycle functions
- sulfur-cycle functions
- CAZy/substrate traits
- broad KO/METABOLIC modules
- QC/taxonomy/context

MethaNet implication:

- Do not concatenate everything blindly.
- Build view-specific MAG similarity networks, validate each view, then fuse them with reliability weights and ablation tests.
- SNF-like fusion is appropriate only after each view has a documented feature dictionary, sparsity/coverage audit, scaling rule, and missingness policy.

Key sources:

- Similarity Network Fusion, Nature Methods 2014: https://pubmed.ncbi.nlm.nih.gov/24464287/
- SNF overview and software: https://compbio.cs.toronto.edu/SNF/
- Recent multi-omics integration reviews: https://academic.oup.com/bib/article/26/4/bbaf355/8220754 and https://www.nature.com/articles/s41416-024-02706-7

### Domain Adaptation, Manifold Alignment, And Optimal Transport

Rumen-to-wetland transfer is a domain-adaptation problem. The core risk is source leakage: the model learns project/source artifacts rather than transferable methane biology.

Relevant methods:

- Domain-adversarial training aligns representations so source/target domains become less distinguishable while preserving task signal.
- CORAL aligns second-order statistics across domains.
- Optimal transport learns a coupling between source and target distributions.
- Manifold alignment can preserve local geometry while projecting domains into a shared space.

MethaNet implication:

- Optimal transport is especially attractive for bridge discovery because it yields explicit cross-domain couplings that can be visualized and audited.
- Domain-adversarial objectives are useful later, when there are enough sources per ecosystem and labels/weak labels to support neural adaptation.
- In the current cohort, OT can rank plausible cross-domain pairings but cannot prove that transfer is source-independent because source and ecosystem are aliased.

Key sources:

- Domain-Adversarial Training of Neural Networks: https://www.jmlr.org/papers/v17/15-239.html
- Deep CORAL: https://arxiv.org/abs/1607.01719
- Optimal Transport for Domain Adaptation: https://dl.acm.org/doi/10.1109/TPAMI.2016.2615921 and https://arxiv.org/abs/1507.00504
- Manifold alignment without correspondence: https://www.ijcai.org/Proceedings/09/Papers/214.pdf

### Graph-Based Evidence Propagation

Graph semi-supervised learning and label propagation provide a transparent way to diffuse evidence over a similarity graph. For MethaNet, the "labels" are not final methane-risk classes yet. They are weak mechanistic priors: methane mechanism support, sulfur competition support, substrate flexibility, QC pass, and source/domain flags.

MethaNet implication:

- Graph propagation should smooth mechanistic evidence across reliable neighborhoods but remain constrained by QC and annotation coverage.
- It should produce uncertainty, not just a winner-take-all label.
- Propagated labels are secondary evidence. Direct MCycDB/SCycDB/KOfam/METABOLIC marker calls remain the primary evidence for candidate cards.

Key sources:

- Zhu, Ghahramani, and Lafferty, Gaussian fields and harmonic functions: https://aaai.org/papers/icml03-118-semi-supervised-learning-using-gaussian-fields-and-harmonic-functions/
- Zhou et al., learning with local and global consistency, cited in graph SSL resources: https://mlg.eng.cam.ac.uk/zoubin/talks/lect3ssl.pdf

### Functional Annotation Evidence

MethaNet's current tool stack is well matched to bridge interpretation:

- MCycDB gives curated methane-cycle specificity.
- SCycDB captures sulfur-cycle competition and alternative electron acceptor context.
- dbCAN captures CAZy and substrate-use capacity.
- METABOLIC provides biogeochemical trait summaries for genomes.
- KOfam/KOfamScan provides broad KO evidence with adaptive thresholds.
- CheckM2, GUNC, and GTDB-Tk provide quality and taxonomy controls.

Key sources:

- MCycDB: https://pubmed.ncbi.nlm.nih.gov/35080120/ and https://github.com/qichao1984/MCycDB
- SCycDB: https://onlinelibrary.wiley.com/doi/abs/10.1111/1755-0998.13306 and https://github.com/qichao1984/SCycDB
- dbCAN3/run_dbcan docs: https://dbcan.readthedocs.io/
- METABOLIC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8851854/ and https://github.com/AnantharamanLab/METABOLIC
- KOfamKOALA: https://academic.oup.com/bioinformatics/article/36/7/2251/5631907
- GTDB-Tk: https://academic.oup.com/bioinformatics/article/36/6/1925/5626182
- CheckM2: https://pubmed.ncbi.nlm.nih.gov/37500759/
- GUNC: https://link.springer.com/article/10.1186/s13059-021-02393-0

### MRV And Carbon-Market Framing

The MRV opportunity is to turn molecular data into decision-useful risk intelligence, not to bypass validated GHG accounting. ICVCM and VM0033 both reinforce the need for transparent evidence and quantification. Molecular attestation can strengthen project screening, monitoring design, uncertainty accounting, and targeted validation.

Key sources:

- ICVCM Core Carbon Principles: https://icvcm.org/core-carbon-principles/
- ICVCM Assessment Framework: https://icvcm.org/assessment-framework/
- Verra VM0033: https://verra.org/methodologies/vm0033-methodology-for-tidal-wetland-and-seagrass-restoration-v2-1/
- IPCC Wetlands Supplement: https://www.ipcc-nggip.iges.or.jp/public/wetlands/index.html

## Next-Generation MethaNet Intelligence Report Blueprint

The report should be a category-defining narrative artifact, not just a figure. It must let a technical reviewer audit the evidence chain while letting a partner or investor understand why the work matters. Its central promise is:

```text
MethaNet turns protein-embedding bridge hypotheses into QC-aware, functionally interpretable, uncertainty-labeled molecular evidence for methane-risk screening.
```

Its central restraint is:

```text
This is MAG-level molecular potential and prioritization intelligence, not sample-level methane flux or carbon-credit certification.
```

### Report Architecture

| Report layer | Audience job | Required evidence | Failure mode to avoid |
| --- | --- | --- | --- |
| Executive decision page | explain what is usable now | cohort denominator, completed subset, top evidence, top blockers | imply final MRV validation |
| Scientific audit ledger | let reviewers trace every claim | source tables, join keys, QC, coverage, status, missingness | hide failed/pending rows |
| Latent geometry page | show why ESM2 generated bridge hypotheses | high-dimensional bridge metrics plus 2D visualization | treat UMAP as proof |
| Functional mechanism page | show separately computed methane/sulfur/substrate support | MCycDB, SCycDB, KOfam, METABOLIC, dbCAN, CAZy, MEROPS | equate all hits with accepted/best evidence |
| Graph transfer page | show how MBAG links domains | kNN graphs, OT couplings, graph confidence, leakage tests | portray OT as causal transfer |
| Candidate cards | make review decisions concrete | per-candidate direct evidence, propagated evidence, QC, caveats, next action | rank candidates without missing-evidence labels |
| Validation dashboard | show what has passed and what is pending | nulls, bootstraps, ablations, source controls, FDR | bury uncertainty |
| MRV/product translation | map science to product primitives | MRV feature table, sample-rollup requirements, monitoring decisions | imply credit-decisional authority |

### Narrative Flow

1. **Evidence base:** "The 662-proteome cohort is cleanly crosswalked and all MAG/proteome files are locally matchable."
2. **Hypothesis signal:** "ESM2 geometry reveals strong source/ecosystem structure and cross-domain bridge candidates."
3. **Severe caveat:** "Because current source and ecosystem labels are confounded, bridge candidates are hypotheses, not source-independent transfer proof."
4. **Functional test:** "The functional atlas tests whether bridge candidates carry separately computed methane/sulfur/substrate/QC support."
5. **Graph synthesis:** "MBAG integrates latent geometry and functional evidence while penalizing QC, missingness, and leakage."
6. **Candidate decisions:** "Each candidate receives an evidence card: pursue, wait for functional completion, block for QC, or use as negative/control."
7. **MRV translation:** "MAG-level evidence becomes a feature primitive for future sample-level risk screening once abundance, metadata, and flux validation are joined."

### Multipanel Hero Figure: "From Latent Proteome Geometry To Molecular Attestation"

This figure should be the visual centerpiece of the report, but every panel must point back to an auditable table.

**Panel 1: Cohort Ledger**

- 662 horizontal ticks, grouped by rumen and wetland/MUCC.
- Layers per tick: embedded, MAG FASTA matched, proteome FAA matched, functional complete/pending/failed.
- Required callouts: `proteome_id` is canonical; current completion snapshot is dated; warehouse may lag live per-MAG outputs.
- Evidence table: crosswalk, local FASTA audit, run status.

**Panel 2: Latent Geometry With Claim Lock**

- ESM2 UMAP/PHATE/diffusion projection as a visual map only.
- Points colored by source/ecosystem, shapes by high-level taxonomy where available.
- Top bridge candidates ringed, with incomplete candidates shown as hollow rings.
- A visible lock label: "Hypothesis-generating; statistical decisions use high-dimensional features."

**Panel 3: High-Dimensional Bridge Neighborhoods**

- kNN graph built in controlled feature space, not 2D coordinates.
- Edge encoding:
  - gray = within-domain neighbor;
  - amber = cross-domain neighbor;
  - blue = OT coupling hypothesis;
  - dashed = low coverage or pending functional evidence.
- Node encoding:
  - fill = ecosystem/source;
  - border = QC tier;
  - halo wedges = direct methane, sulfur, substrate, and coverage support.

**Panel 4: Functional Mechanism Matrix**

- Rows: top bridge candidates plus matched negative/control MAGs.
- Columns grouped into direct evidence, context evidence, and reliability:
  - MCycDB best-hit methane families;
  - accepted KOfam methane/sulfur KOs;
  - METABOLIC methane/sulfur/modules;
  - SCycDB sulfur competition;
  - dbCAN/CAZy/substrate and MEROPS;
  - CheckM2/GUNC/taxonomy;
  - annotation coverage.
- Cell states:
  - present direct evidence;
  - absent with sufficient coverage;
  - unknown due to low coverage/pending run;
  - blocked by QC.

**Panel 5: MBAG Evidence Decomposition**

- For each candidate, show a stacked but uncertainty-aware decomposition:
  - latent bridge component;
  - OT coupling component;
  - direct function concordance;
  - propagated-neighborhood support;
  - QC/coverage penalty;
  - source-leakage penalty.
- Use intervals or faint bands for bootstrap uncertainty.
- Avoid a single dominant "score gauge" until calibration exists.

**Panel 6: Candidate Attestation Cards**

Each card should fit on one partner-facing page and one machine-readable row.

Required card fields:

```text
proteome_id
mag_id
source/ecosystem
taxonomy
latent_bridge_rank
direct_methane_evidence
direct_sulfur_evidence
substrate/CAZy/MEROPS evidence
METABOLIC module/function support
QC tier
annotation coverage tier
source_leakage_risk
evidence tier
blocking caveats
next validation action
allowed claim wording
```

Card labels:

- `pursue_now`: direct mechanism evidence, good QC, stable bridge support.
- `wait_for_completion`: high latent rank, functional evidence pending.
- `needs_review`: plausible but low coverage, uncertain taxonomy, or mixed evidence.
- `blocked`: QC/source leakage/missingness prevents interpretation.
- `negative_control`: useful comparator, not bridge-like.

**Panel 7: Validation Dashboard**

Show pass/warn/fail states, not only pretty metrics:

- denominator and duplicate-ID gate;
- source leakage classifier/AUC or equivalent;
- embeddings-only vs functions-only vs fused MBAG ablation;
- bootstrap rank stability;
- source-balanced downsampling;
- taxonomy-aware nulls;
- PERMANOVA plus dispersion check;
- FDR-controlled enrichment tests;
- pending external validation.

**Panel 8: MRV Translation Map**

The final panel maps evidence to action without overclaiming:

```text
MAG molecular evidence -> sample rollup requirement -> field measurement priority -> monitoring design recommendation
```

Allowed product outputs:

- molecular risk-feature table;
- bridge candidate review queue;
- monitoring-priority map once samples are joined;
- validation gap register;
- partner/investor proof-of-capability report.

Disallowed output labels at this stage:

- certified methane permanence;
- carbon-credit approval;
- measured methane flux;
- source-independent transfer proof.

### Design Rules For Trustworthy Visual Communication

- Use visual intensity for evidence strength only when the evidence is direct and coverage-qualified.
- Use gray or hatch patterns for pending/missing evidence; do not let missingness appear as biological absence.
- Always show the denominator: 662 total, completed subset, pending subset, failed subset.
- Keep source/ecosystem confounding visible in every cross-domain panel.
- Put "MAG-level potential, not sample-level flux" in the report footer.
- Do not use causal arrows from genes to credits. Use "supports screening" or "prioritizes measurement" language.
- Scientific version exposes full feature names and QC metrics; partner version compresses them into evidence groups but retains caveats.

## Proposed Core Algorithm: MethaNet Bridge Attestation Graph

### Objective

Rank MAGs by the degree to which their ESM2 latent bridge status is supported by separately computed functional, taxonomic, QC, and annotation-coverage evidence, while explicitly estimating source/domain leakage and uncertainty. MBAG is a prioritization and explanation model. It is not a causal model of methane emissions, not a field-calibrated MRV model, and not yet a supervised transfer-learning model.

### Inputs

| Symbol | Table/artifact | Grain |
| --- | --- | --- |
| `E` | ESM2 1280-dimensional proteome embeddings | MAG/proteome |
| `P` | embedding projection and bridge features | MAG/proteome |
| `F_methane` | MCycDB, accepted KOfam methane KOs, METABOLIC methane functions/modules | MAG x feature |
| `F_sulfur` | SCycDB, KOfam sulfur KOs, METABOLIC sulfur functions/modules | MAG x feature |
| `F_substrate` | dbCAN/CAZy, METABOLIC CAZy, MEROPS, substrate modules | MAG x feature |
| `F_broad` | KOfam, METABOLIC broad modules/functions, Bakta features | MAG x feature |
| `Q` | CheckM2, GUNC, GTDB-Tk, annotation coverage | MAG |
| `D` | ecosystem/source/domain labels | MAG |
| `M` | environmental metadata resolution | MAG/sample provenance |

### Reliability And Missingness Model

Each MAG receives a reliability profile, not just one opaque quality score:

```text
R_i = {
  q_completeness_i,
  q_contamination_i,
  q_gunc_i,
  q_taxonomy_i,
  q_coverage_i_by_tool,
  run_status_i,
  source_metadata_resolution_i
}
```

For visualization and ranking, a bounded summary weight may be used:

```text
r_i = min(q_completeness_i, q_gunc_i, q_coverage_required_i) * q_contamination_i * q_taxonomy_i
```

The `min` term intentionally prevents a high average score from hiding a missing required evidence layer. Suggested smoke-test transforms are:

```text
q_completeness_i = clip((completeness_i - 50) / 40, 0, 1)
q_contamination_i = exp(-max(contamination_i - 5, 0) / 5)
q_gunc_i = 1.0 if GUNC pass else 0.6 or lower
q_taxonomy_i = 1.0 resolved genus/family, 0.8 higher-rank only, 0.5 unresolved
q_coverage_required_i = minimum coverage tier across the tools required for the candidate's claimed mechanism
```

These transforms are engineering priors for smoke testing, not final scientific thresholds. They must be stress-tested and replaced or calibrated when external validation labels exist. Absence claims are allowed only when the relevant tool-specific coverage is sufficient; a low-coverage zero is `unknown`, not `absent`.

### View-Specific Similarity Graphs

For each evidence view `v`, construct a kNN similarity graph:

```text
K_v(i,j) = exp(-d_v(i,j)^2 / (sigma_v_i sigma_v_j))
W_v(i,j) = K_v(i,j) * sqrt(r_i r_j)
```

Keep both `K_v` and `W_v`. `K_v` records biological/feature proximity before reliability penalties; `W_v` is used for confidence-aware ranking and propagation. This prevents QC penalties from erasing the ability to diagnose whether a low-quality MAG is biologically near a bridge neighborhood.

Recommended distances:

| View | Distance |
| --- | --- |
| ESM2 | cosine or Euclidean in PCA-whitened high-dimensional ESM2 space |
| methane | weighted Jaccard/cosine over MCycDB best hits, accepted KOs, METABOLIC methane features |
| sulfur | weighted Jaccard/cosine over SCycDB best hits and sulfur modules |
| substrate | weighted Jaccard/cosine over CAZy/dbCAN/MEROPS/substrate traits |
| broad function | cosine over transformed KO/module features |
| taxonomy | tree distance or rank-aware categorical penalty |
| QC/coverage | penalty graph, not attraction graph |

Then fuse reliability-aware graphs, while preserving single-view outputs for ablation:

```text
W_fused = SNF({W_ESM2, W_methane, W_sulfur, W_substrate, W_broad}, weights=a_v)
```

Initial smoke-test view weights:

```text
a_ESM2 = 0.30
a_methane = 0.25
a_sulfur = 0.15
a_substrate = 0.10
a_broad = 0.10
```

The remaining weight can be reserved for leakage/context penalties rather than attraction edges. Taxonomy and metadata should usually constrain or annotate the graph, not force attraction between MAGs unless the analysis question explicitly calls for taxonomy-aware grouping. These priors are for report prototyping only. They should be re-estimated by ablation, bootstrap stability, and later supervised/external calibration. No biological conclusion should depend on one unvalidated weight vector.

### Source-Aware Optimal Transport Alignment

Define source domain `S` as rumen and target domain `T` as wetland/MUCC, with the current caveat that source and ecosystem are confounded. In the current cohort, the transport plan is a candidate-pairing hypothesis, not evidence that a learned mechanism transfers independent of source.

Cost between rumen MAG `i` and wetland MAG `j`:

```text
C_ij =
  beta_E * d_E(i,j)
+ beta_M * d_methane(i,j)
+ beta_S * d_sulfur(i,j)
+ beta_C * d_substrate(i,j)
+ beta_B * d_broad(i,j)
+ beta_T * d_taxonomy(i,j)
+ beta_Q * penalty_Q(i,j)
+ beta_missing * missingness_penalty(i,j)
```

Learn an entropic transport plan:

```text
Gamma* = argmin_Gamma <Gamma, C> + epsilon * KL(Gamma || u v^T)
subject to Gamma 1 = u, Gamma^T 1 = v
```

Optional biological constraints:

- penalize pairings where both MAGs lack sufficient annotation coverage;
- down-weight pairings with contradictory taxonomy only when taxonomy is relevant to the claim;
- up-weight pairings where methane/sulfur mechanism classes agree;
- report transport uncertainty across bootstraps.
- compare against transport plans from feature-shuffled, taxonomy-preserving, and source-balanced nulls.

### Graph-Regularized Mechanism Propagation

Let `Y` be weak mechanistic evidence labels:

```text
Y_i = [methane_relevant, methylotrophic, hydrogenotrophic, acetoclastic,
       AOM_related, methanotrophy_related, sulfur_competition,
       substrate_flexible, unclear]
```

Solve graph-smoothed evidence:

```text
Z* = argmin_Z ||R (Z - Y)||_F^2 + lambda * Tr(Z^T L_fused Z)
```

Where:

- `R` is a diagonal reliability matrix from QC/coverage;
- `L_fused` is the graph Laplacian of `W_fused`;
- `Z*` is propagated mechanism support.

This does not invent mechanisms. It smooths weak evidence over trusted neighborhoods and reports uncertainty when support is thin. Candidate cards must separate:

- direct evidence: actual marker/module/hit calls in the candidate;
- neighborhood evidence: support propagated from graph neighbors;
- missing evidence: absent, low coverage, or pending runs.

### Bridge Attestation Score

For each candidate `i`, compute a provisional score:

```text
BAS_i =
  w1 * z(cross_domain_knn_i)
+ w2 * z(OT_support_i)
+ w3 * z(fused_graph_betweenness_i)
+ w4 * z(functional_concordance_i)
+ w5 * z(mechanism_support_i)
+ w6 * z(candidate_specificity_i)
- w7 * qc_penalty_i
- w8 * annotation_missingness_i
- w9 * source_leakage_penalty_i
```

Where:

- `cross_domain_knn_i`: fraction or entropy of opposite-domain neighbors in high-dimensional ESM2 space;
- `OT_support_i`: total transport mass connecting `i` to opposite-domain MAGs;
- `fused_graph_betweenness_i`: bridge centrality in the fused graph;
- `functional_concordance_i`: similarity between latent-neighbor functions and candidate functions;
- `mechanism_support_i`: methane/sulfur/substrate evidence strength after graph regularization;
- `candidate_specificity_i`: evidence that the candidate is not merely reflecting features ubiquitous across all MAGs or all MAGs in one source;
- penalties enforce QC, missingness, and source-confounding humility.

The score must be reported with a status:

| Status | Meaning |
| --- | --- |
| `provisional_internal` | available now; useful for review and smoke reports |
| `source_audited` | passed source-balanced/null/leakage checks within current limits |
| `externally_calibrated` | calibrated against independent sources and sample/flux validation |

Only `externally_calibrated` scores may be used for registry-facing quantitative risk claims.

### Confidence Tiers

These are evidence tiers for MAG-level candidate review. They are not methane-risk tiers and do not imply source-independent transfer.

| Tier | Meaning | Required evidence |
| --- | --- | --- |
| High-evidence bridge | latent bridge plus direct functional mechanism support and good QC | complete functional outputs, QC pass, high annotation coverage, stable rank, source-aware/current-cohort nulls passed |
| Moderate-evidence bridge | latent bridge plus partial functional support | good QC, at least methane or sulfur/substrate support, rank stable in bootstraps |
| Hypothesis-only bridge | latent bridge but insufficient functional evidence | pending functional run or low annotation coverage |
| Artifact-risk bridge | bridge score driven by low quality, contamination, source leakage, or missingness | blocked until repaired or independently validated |
| Negative/control | not bridge-like under fused evidence | useful as visualization/statistical control |

Rename final report tiers as evidence tiers rather than risk tiers unless sample-level validation exists.

### Pseudocode

```python
backbone = load_662_crosswalk()
emb = load_esm2_embeddings()
geometry = load_bridge_projection_tables()
facts = load_functional_tables_or_live_per_mag_outputs()
qc = load_checkm2_gunc_taxonomy_coverage()

X = left_join(backbone, emb, geometry, qc)
F = build_feature_views(facts, accepted_kofam=True, best_hit_rank=True)
R = compute_reliability(qc, coverage=F.coverage)

graphs = {}
for view in ["esm2", "methane", "sulfur", "substrate", "broad"]:
    graphs[view] = build_knn_graph(X_or_F=view, reliability=R)

W = similarity_network_fusion(graphs, view_weights)
Gamma = source_aware_optimal_transport(W, X.domain, cost_views=graphs, reliability=R)
Z = graph_regularized_mechanism_propagation(W, weak_mechanism_labels(F), R)
leakage = estimate_source_leakage(X, graphs, W)

bridge_scores = compute_bas(
    geometry=geometry,
    graph=W,
    transport=Gamma,
    propagated_mechanisms=Z,
    qc=qc,
    coverage=F.coverage,
    source_labels=X.source,
    source_leakage=leakage,
)

validate_with_nulls_and_bootstraps(bridge_scores, graphs, Gamma, Z)
write_bridge_attestation_cards(bridge_scores, X, F, qc, validation)
```

## Statistical Validation Plan

### Validation Gate 1: Identity And Denominator

- Exactly 662 `proteome_id` rows in the analysis backbone.
- No duplicate `proteome_id`.
- All feature tables left-joined to backbone.
- Failed/pending functional rows preserved as status, not dropped.

### Validation Gate 2: Source-Aware Null Models

Current source design limits deconfounding because source and ecosystem are currently aliased. Current-cohort tests can detect obvious artifacts and leakage, but they cannot prove source-independent transfer. Use two levels:

**Current-cohort nulls**

- Permute functional feature labels within source/ecosystem blocks to test whether candidate mechanisms exceed source-preserved random expectation.
- Degree-preserving graph rewiring of cross-domain edges.
- Randomize functional feature labels within source while preserving feature prevalence.
- Rumen downsampling to 107 to match wetland/MUCC count.
- Taxonomy-aware nulls that preserve domain/taxonomic rank distributions.
- Protein-count and genome-quality matched negative controls.
- Shuffled-embedding controls that preserve source labels but destroy candidate-specific geometry.
- Source classifier leakage audit: if source/ecosystem can be recovered too easily from the fused representation, bridge scores require stronger caveats.

**Future source-deconfounded nulls**

- Add at least two independent sources per ecosystem.
- PERMANOVA or distance-based models with source and ecosystem terms, plus dispersion checks.
- Leave-one-source-out validation.
- Train on one rumen source and test transfer to held-out rumen plus wetland sources.
- Require candidate ranks to remain stable when any one source project is removed.

### Validation Gate 3: Bootstrap Rank Stability

For each candidate:

- bootstrap MAGs within ecosystem/source;
- bootstrap functional feature groups;
- rerun kNN graph with multiple `k`;
- rerun UMAP/PHATE seeds for visualization only;
- recompute MBAG score;
- report rank median, 90% interval, and selection frequency.
- stratify bootstraps so wetland/MUCC and rumen imbalance does not dominate uncertainty estimates.

Stable candidates should retain high ranks across:

- ESM2-only graph;
- functional-only graph;
- fused graph;
- QC-penalized graph;
- source-balanced downsampling.

### Validation Gate 4: Ablation

Compare:

| Model | Purpose |
| --- | --- |
| ESM2-only | tests latent geometry alone |
| methane/sulfur-only | tests targeted mechanism evidence |
| broad-function-only | tests whether general function explains bridges |
| QC/taxonomy-only | detects artifact-driven bridge rankings |
| source-label-only or leakage probe | detects whether rankings are reducible to source/project artifacts |
| fused MBAG | tests whether integration improves stability and interpretability |

Expected strong result:

- fused MBAG improves bridge-card interpretability and rank stability;
- it does not merely reproduce source labels;
- top candidates remain plausible after QC and annotation-coverage penalties.
- candidate-specific direct evidence remains visible and is not replaced by propagated neighborhood support.

### Validation Gate 5: QC And Coverage Sensitivity

Stress tests:

- remove MAGs with completeness <50%, <70%, <90%;
- compare contamination thresholds <=5% and <=10%;
- remove GUNC-failing MAGs;
- require minimum annotation coverage by tool;
- test absence calls only on high-coverage MAGs.

Bridge claims should degrade gracefully, not flip chaotically.

### Validation Gate 6: Statistical Testing And FDR

- Use PERMANOVA for multivariate feature shifts, but include source terms as soon as source replication exists.
- Pair PERMANOVA with dispersion checks because distance-based multivariate tests can confound location and dispersion.
- Use MMD or energy distance for embedding/function distribution shift.
- Use Benjamini-Hochberg FDR for pathway/gene family enrichment tests.
- Report effect sizes and confidence intervals, not only p-values.
- Treat feature-enrichment tests as descriptive until source replication exists.

Key statistical sources:

- PERMANOVA: https://onlinelibrary.wiley.com/doi/10.1111/j.1442-9993.2001.01070.pp.x
- vegan `adonis2`: https://www.rdocumentation.org/packages/vegan/versions/2.6-4/topics/adonis
- PERMANOVA/dispersion caveat: https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210x.2011.00127.x
- Benjamini-Hochberg FDR: https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x
- Bootstrap methods: Efron and Tibshirani, "An Introduction to the Bootstrap"

### Validation Gate 7: MRV Claim Gate

No MRV-facing output may be labeled as methane permanence risk unless it has:

- sample/metagenome mapping;
- abundance or read-coverage weighting;
- environmental covariates relevant to methane production/oxidation and sulfate competition;
- uncertainty propagation from MAG QC, annotation coverage, abundance, and source metadata;
- external field or incubation measurement for calibration or at least validation.

Before that point, use `molecular_screening_feature`, `monitoring_priority`, or `validation_gap` language.

## Current Artifact And Data Requirements

### Required To Build The First MBAG Smoke Test

| Requirement | Current source | Status |
| --- | --- | --- |
| 662-row backbone | crosswalk TSV | ready |
| MAG FASTA/proteome paths | functional MAG manifest | ready |
| ESM2 embeddings | `genome_embeddings.npz` | ready |
| ESM2 projections/bridge ranks | bridge/projection TSVs | ready |
| Functional evidence | per-MAG curated Parquet manifests | partially ready |
| QC/taxonomy | per-MAG run records and Parquet | partially ready |
| Annotation coverage | per-MAG/cohort coverage tables | calibration-ready; live warehouse stale |
| Metadata resolution | environmental metadata recovery | partially ready, caveated |

Smoke-test scope limit: if top latent bridge candidates are not functionally complete, the smoke report should still show their latent rank and missing evidence, but it must not assign them a mechanism-supported bridge tier.

### Required For Strong Bridge Claims

- Functional completion for top ESM2 rumen bridge candidates.
- Regenerated cohort warehouse after the desired tranche/full run completes.
- Normalized feature matrices for methane, sulfur, substrate, broad function, QC, taxonomy, and annotation coverage.
- Additional wetland and rumen sources to break source/ecosystem confounding.
- Sample/metagenome abundance or read-coverage weights for blue carbon sample interpretation.
- Environmental covariates: salinity, redox/sulfate, depth, vegetation, temperature, hydrology, carbon stock, methane flux where available.
- Source leakage audit showing bridge ranks are not reducible to project/source labels.
- Negative and positive controls, including taxonomy/QC-matched non-bridge candidates.

### Required For MRV/Product Claims

- Sample-level mapping from MAGs to metagenomes/samples.
- Abundance or coverage weighting.
- Validation against flux/chamber/tower/incubation measurements.
- Project-level uncertainty model.
- External claim wording aligned to registry and buyer expectations.
- Separation of screening outputs from credit-decisional outputs in product language and UI.

## Implementation Roadmap

### Phase 0: Bridge Smoke Test

Goal: produce a minimal MBAG report from available completed MAGs.

Tasks:

1. Build `cohort_identity` from the 662 crosswalk and functional manifest.
2. Load ESM2 bridge candidates and projections.
3. Join completed functional outputs for available candidates.
4. Produce bridge-card placeholders for incomplete top candidates.
5. Produce one complete wetland candidate card if evidence is present.
6. Report "hypothesis-only" for top rumen candidates without functional completion.
7. Include source-leakage, QC, coverage, and missingness panels even if they are mostly warnings.

Output:

```text
bridge_attestation_cards_smoke.tsv
bridge_attestation_smoke_report.md/html
validation_gap_register.tsv
```

### Phase 1: Full Tranche Functional Join

Goal: regenerate the cohort warehouse after the active Slurm production target completes.

Tasks:

1. Run the consolidator with the correct expected complete count after user approval.
2. Validate `validation_gates.tsv`.
3. Build normalized feature matrices from cohort Parquet.
4. Recompute MBAG on completed tranche.
5. Freeze a dated evidence snapshot so figures and candidate cards are reproducible.

Output:

```text
mbag_feature_matrix.parquet
mbag_graph_edges.parquet
mbag_bridge_scores.parquet
bridge_mechanism_cards.parquet
next_generation_intelligence_report.html
```

### Phase 2: Full 662 Bridge Atlas

Goal: produce a complete MAG-level bridge atlas.

Tasks:

1. All 662 proteomes have selected complete functional runs or explicit failure statuses.
2. All top bridge candidates have cards.
3. Ablation and bootstrap validation are complete.
4. Figure panels A-F are generated.

Output:

```text
MethaNet Functional-Geometry Bridge Atlas
```

### Phase 3: Source-Deconfounded Transfer Study

Goal: separate ecosystem biology from source/project artifacts.

Tasks:

1. Add independent rumen and wetland/mangrove/saltmarsh/seagrass sources.
2. Run source-aware PERMANOVA and leave-source-out tests.
3. Train/test MBAG with source holdouts.
4. Upgrade or downgrade claims.

Output:

```text
source_aware_validation.tsv
claim_boundary_matrix_v2.md
```

### Phase 4: Sample/Metagenome MRV Layer

Goal: move from MAG potential to sample-level molecular risk evidence.

Tasks:

1. Map MAGs to samples/metagenomes.
2. Add abundance/read-coverage weights.
3. Add environmental covariates and flux measurements where available.
4. Build preliminary sample risk-readiness table.
5. Define monitoring recommendations and uncertainty bands.

Output:

```text
sample_risk_readiness_table.parquet
mrv_feature_dictionary.md
partner_demo_dashboard
```

## Claim-Boundary Matrix

| Claim | Current status | Allowed wording | Required upgrade evidence |
| --- | --- | --- | --- |
| ESM2 proteome geometry separates rumen and wetland/MUCC domains | supported by 662 POC | "The 662-proteome POC shows strong latent ecosystem/source structure with cross-domain bridge candidates." | source-replicated cohorts to separate ecosystem from project/source |
| Bridge candidates are biologically meaningful methane candidates | provisional | "Top bridge candidates are hypotheses requiring direct, separately computed functional support." | full functional completion for top candidates, QC pass, direct methane/sulfur/substrate support, stable ranks, source leakage audit |
| Functional annotations support bridge mechanisms | partially testable | "Completed MAGs already produce the functional evidence needed for bridge-card construction; top-rank rumen cards remain pending until their runs complete." | regenerated warehouse and cards for all top candidates |
| MethaNet can assign MAG-level methane mechanism confidence | emerging | "MethaNet can produce preliminary MAG-level mechanism evidence tiers using QC-aware functional annotations." | validation gates, coverage thresholds, ablations, direct evidence review, expert review |
| MethaNet can assign sample/metagenome methane risk | not yet | "Sample-level risk requires abundance/coverage, environmental metadata, and validation." | sample mapping, abundance/read coverage, geochemistry, flux validation, uncertainty propagation |
| MethaNet has proven source-independent rumen-to-wetland MRV transfer | not allowed | "Current work is building and validating a source-audited transfer framework." | multiple independent sources per ecosystem, leave-source-out validation, external flux data |
| MethaNet supports carbon-crediting decisions directly | not yet | "MethaNet can support upstream screening, monitoring design, and validation prioritization for methane permanence risk." | registry-aligned validation, project-level uncertainty, field measurements, accepted methodology integration |
| Molecular attestation is a credible product primitive | supported as design direction | "The combined embedding/function/QC atlas can become a partner-facing molecular attestation layer for screening and follow-up design." | complete bridge cards, dashboard/API schema, validation roadmap |
| MBAG score is calibrated | not yet | "MBAG currently produces provisional internal prioritization scores." | external labels, independent sources, source-holdout validation, field/sample calibration |
| Graph propagation confirms missing mechanisms | not allowed | "Graph propagation provides neighborhood context only." | direct gene/module evidence in candidate plus coverage-qualified absence |

## Recommended Next Decision

The best immediate action is to build a lightweight MBAG smoke-test report once the next useful tranche is complete, without waiting for the entire production run if top bridge candidates become available earlier. The smoke test should not rank final MRV risk. It should demonstrate:

1. the join from ESM2 bridge candidate to MAG FASTA/proteome/function rows;
2. one or more complete bridge cards;
3. explicit pending/missing evidence for incomplete top candidates;
4. the first version of the multipanel figure skeleton;
5. validation gates that show what remains provisional.

This gives MethaNet a credible, visually compelling, scientifically cautious bridge between the ESM2 POC and the functional atlas. It is exactly the kind of artifact that can mature into partner-facing molecular attestation without stepping beyond what the evidence can defend.
