# MethaNet Positioning And Claim Contract

Documentation refresh: 2026-09-27

This document is the shared narrative and claim contract for the MethaNet
repository, the [EmergentBiome landing page](https://emergentbiome.earth/), and
the [public MBAG report](https://emergentbiome.earth/report/). Detailed run
status and artifact paths remain in
[`current_artifact_inventory.md`](current_artifact_inventory.md).

## Canonical Position

Ecosphere Blue, Inc. develops **EmergentBiome**, the molecular-evidence
platform, whose public face is the **EmergentBiome Molecular Atlas**. MethaNet
is the platform's first application, methane screening and field-validation
planning for blue carbon, and the name of this repository. The public page
calls the reviewable evidence layer the **EmergentBiome evidence graph**; MBAG
remains the internal architecture and historical artifact identifier. The landing
page's August atlas and `/report/` use the 2026-08-10 release. Its separately
reviewed September three-case explorer uses a curated public projection of
selected MVO 0.2.0 evidence; it is not a live connection to the local graph
service and does not update the atlas denominator.

MethaNet supports blue-carbon methane diligence.
It turns microbiome sequencing into an evidence graph that links each MAG or
proteome to three molecular views, reliability guardrails, source provenance,
claim eligibility, and the next measurement that would improve a decision.

The current product layer supports:

- MAG/proteome molecular screening;
- bridge-candidate and evidence-card review;
- project-data diligence;
- monitoring and validation-study design;
- prioritization of sample identity, abundance, environmental, and field-data
  gaps.

The future calibrated layer adds sample and project methane-risk distributions
after exact sample linkage, abundance or read coverage, environmental
covariates, uncertainty propagation, and field or process validation are
available.

## MBAG Knowledge Graph

The MethaNet Bridge Attestation Graph, or MBAG, is the internal evidence-graph
architecture. The historical persistent Kuzu MVP covers the 662-record
rumen/wetland POC. MVO 0.2.0 is a separate additive OWL/SHACL-governed RDF
snapshot with a rebuildable local Neo4j projection; it preserves the original
233,190 statements and adds selected molecular evidence for 145 MAGs. The
August release still registers 7,965 `(lane_id, proteome_id)` rows and remains
the atlas authority. MVO does not materialize all molecular facts for those
7,965 records or create a sample-resolved methane-flux graph. The live landing
page's three-case network is a separately reviewed, allowlisted presentation
projection, not a public connection to the internal graph service. See the
[knowledge-graph foundation](knowledge_graph_foundation.md), [MVO handoff](../ontology/docs/MOLECULAR_HANDOFF_20260926.md),
and [SQL/Neo4j comparison](../ontology/docs/STORE_COMPARISON_20260926.md) for
current implementation and claim boundaries. The review path links:

```text
MAG or proteome record
  + ESM-2 proteome neighborhood
  + gLM2 genomic context
  + functional and expression evidence
  + QC, taxonomy, and provenance
  + evidence-contract and protocol status
  + sample-linkage and validation gaps
  -> evidence card
  -> next measurement or review action
```

Each relationship retains its evidence class. Representation proximity,
functional annotation, expression detection, and field evidence therefore
remain distinguishable in every review path.

## Proposed Path From Atlas To Flux Models

EmergentBiome connects molecular evidence to methane-flux prediction through a
shared semantic and provenance layer. The ontology defines gene, pathway,
sample, site, and measurement meaning. The evidence graph links those entities
with their source, method, QC, status, and identity confidence. The model uses
versioned features from this atlas; model outputs return to evidence cards with
their feature lineage, uncertainty, and next measurement.

The first proposed predictor is an interpretable benchmark for measured net
methane flux. It compares environmental covariates alone, the published
`mcrA` to `pmoA` abundance-ratio benchmark, and a combined model with coverage-qualified
methane, sulfur, and substrate gene-set features plus salinity, sulfate,
hydrology, and temperature. Regularized regression is a transparent candidate
learner. The final feature definitions and learner should follow the paired
sample design and preregistered evaluation.

A complementary transfer learning module uses gene-level ESM-2 and gLM2
representations around methane-pathway genes. It prioritizes bridge genomes
and transfers functional hypotheses from characterized references to
underrepresented wetland genomes. Transfer-derived features then enter the
same methane flux evaluation with matched environmental context. The proposed
tests compare environment-only, marker-only, interpretable hybrid, and
transfer-augmented inputs. Use grouped plot folds, public-source holdouts where
data allow, and the prelabel dry-season holdout at the target site. Several
tens of independent sites are needed to evaluate geographic transfer.

These are proposed model stages. Training on field outcomes requires exact
sample identity, abundance or read coverage, environmental covariates, and
paired methane-flux measurements. The current MVO 0.2.0 graph contains selected
molecular evidence for 145 MAGs. The broader atlas registers 7,965 lane-scoped
records. The current graph admits zero exact molecular-to-flux pairs, so the
field predictor remains a research objective.

## Current Release Snapshot

The August 10, 2026 controlled-diligence release contains:

| Measure | Current release | Interpretation |
| --- | ---: | --- |
| Registered MAG/proteome units | 7,965 | Full warehouse registry, including explicit gap and incomplete states |
| ESM-2-bearing units | 7,710 | Proteome-neighborhood navigation |
| gLM2 payloads | 7,717 | Genomic-context evidence, compared within protocol class |
| Data-complete tri-views | 7,710 | ESM-2, gLM2, and a functional payload are all present |
| Schema-normalized tri-views | 7,710 | Long-form tables and accepted/best/present event semantics are explicit |
| Pipeline-normalized tri-views | 5,209 | POC, MSM, and Futian; cross-lane equivalence remains pending |
| Mechanism-comparable tri-views | 0 | No cross-lane quantitative mechanism comparison is authorized |
| MUCC v1 source-scaffold tri-views | 2,501 | DRAM, gene, and processed expression evidence under a distinct contract |

The 7,710 tri-view total is an availability statement. Pipeline normalization
does not establish biological-mechanism equivalence. These states are useful
for navigation, diligence, candidate review, and harmonization planning while
retaining their distinct evidence contracts.

The August 10 evidence now powers a live `noindex` controlled-diligence page
and reconciled report following a scoped September correction. That deployment
did not authorize routine scientific publication or indexing.
Source-aware and taxonomy-aware nulls, bootstrap and graph sensitivity,
view ablations, multiple-testing control, and remaining scientific review are
publication gates, not claims already earned. Browser/accessibility checks for
the scoped correction are recorded separately from those scientific gates.

## Tri-View Interpretation

| View | What it contributes | Current boundary |
| --- | --- | --- |
| ESM-2 | Proteome-level molecular neighborhoods and candidate discovery | Similarity supports navigation and hypothesis generation |
| gLM2 | Native and shuffled genomic-context evidence | Numerical comparison stays within protocol class |
| Functional mechanism | Methane, sulfur, substrate, expression, and broader annotation evidence | Cross-lane quantitative ranking requires one accepted/present feature contract |
| QC and provenance | Completeness, contamination, taxonomy, source, artifact, and missingness controls | Weak, partial, and blocked evidence remains visible |

## Decision Outputs

The current system can produce:

1. **Candidate evidence cards** with molecular evidence, provenance,
   comparability state, authorized claim wording, blocking gaps, and next
   action.
2. **Monitoring-readiness decisions** that identify the highest-value missing
   measurement for a sample, site, or study.
3. **Validation portfolios** that direct field investment toward contexts with
   high information value.
4. **An evidence ledger** that preserves the molecular-to-measurement path
   needed for later calibrated MRV models.

## Claim Matrix

| Topic | Authorized wording now | Upgrade requirement |
| --- | --- | --- |
| Molecular potential | “Completed MAGs carry QC-aware functional evidence.” | Common cross-lane mechanism-feature contract |
| ESM-2 neighborhoods | “Embedding neighborhoods identify review candidates.” | Source-balanced, taxonomy-aware validation for transfer claims |
| MUCC expression | “Processed expression detection provides occupancy support where present.” | Normalization, exact sample linkage, and activity validation |
| Monitoring | “MethaNet prioritizes where to measure next. That prioritization will be tested against field outcomes before it is used to waive any measurement.” | Partner workflow validation and outcome testing |
| Sample methane risk | “Sample risk remains a calibrated target.” | Abundance, environment, uncertainty, and paired field evidence |
| A to E tiers | “A to E tiers are target product vocabulary.” | Quantitative thresholds and holdout calibration |
| Carbon-market use | “MethaNet can support upstream diligence and monitoring design.” | Methodology integration, independent review, and registry-aligned validation |

## Terminology

Use these terms consistently:

- **Ecosphere Blue, Inc.** for the company, **EmergentBiome** for the
  platform, and **MethaNet** for its first application and this repository.
- **Molecular attestation** for the current product category.
- **MBAG evidence graph** or **molecular-attestation knowledge graph** for the
  internal architecture; **EmergentBiome evidence graph** for the public
  proposal page. Distinguish the historical 662-record Kuzu MVP, the selected
  MVO 0.2.0 graph slice, the 7,965-row August release, and the allowlisted
  three-case landing presentation when describing graph scope.
- **Data-complete tri-view** when all three payloads exist.
- **Mechanism-comparable tri-view** only for rows that share the validated
  accepted/present feature contract.
- **Evidence card** for the reviewable product object.
- **Monitoring readiness** or **validation readiness** for present decision
  support.
- **Calibrated methane-risk intelligence** for the future sample and project
  layer.

Avoid language that presents current molecular outputs as measured methane
flux, final sample or project risk, calibrated A to E tiers, source-independent
rumen-to-wetland transfer, registry approval, or carbon-credit determination.
The release ledger's zero `sample_linked_units` and `field_validated_units`
refer to release-authorized molecular-to-sample/process joins for scoring;
sample contexts, processed expression, and staged field observations exist
under separate typed evidence contracts.

## Narrative Sequence

Repository and external materials should follow one coherent sequence:

1. Methane can erode blue-carbon climate value.
2. Direct monitoring is expensive and spatially sparse, so it must be targeted.
3. The EmergentBiome Molecular Atlas presents sequencing-derived evidence and
   explicit gaps through a traceable review surface.
4. The internal MBAG architecture separates representation context, mechanism evidence, and reliability
   guardrails.
5. Evidence cards guide molecular diligence and the next measurement.
6. Paired abundance, environmental, uncertainty, and field evidence unlock
   calibrated methane-risk intelligence.

This sequence keeps the present capability commercially useful and
scientifically bounded while showing a credible route into MRV.
