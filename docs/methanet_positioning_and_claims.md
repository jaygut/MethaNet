# MethaNet Positioning And Claim Contract

Documentation refresh: 2026-09-24

This document is the shared narrative and claim contract for the MethaNet
repository, the [EmergentBiome landing page](https://emergentbiome.earth/), and
the [public MBAG report](https://emergentbiome.earth/report/). Detailed run
status and artifact paths remain in
[`current_artifact_inventory.md`](current_artifact_inventory.md).

## Canonical Position

MethaNet is the internal molecular-attestation platform and repository behind
the public **EmergentBiome Molecular Atlas**. The public page calls its
reviewable evidence layer the **EmergentBiome evidence graph**; MBAG remains
the internal architecture and historical artifact identifier. Both names use
the same bounded August 10 molecular evidence contract.

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
architecture. Its existing persistent, queryable MVP covers the 662-record
rumen/wetland POC. The August release adds 7,965 registered
`(lane_id, proteome_id)` rows with evidence-contract, candidate-card, and
validation-gap projections; those rows are not yet a materialized
sample-resolved knowledge graph. See
[`knowledge_graph_foundation.md`](knowledge_graph_foundation.md) for the
current/future boundary. The review path links:

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
| Monitoring | “MethaNet prioritizes the next measurement or validation action.” | Partner workflow validation |
| Sample methane risk | “Sample risk remains a calibrated target.” | Abundance, environment, uncertainty, and paired field evidence |
| A to E tiers | “A to E tiers are target product vocabulary.” | Quantitative thresholds and holdout calibration |
| Carbon-market use | “MethaNet can support upstream diligence and monitoring design.” | Methodology integration, independent review, and registry-aligned validation |

## Terminology

Use these terms consistently:

- **Molecular attestation** for the current product category.
- **MBAG evidence graph** or **molecular-attestation knowledge graph** for the
  internal architecture; **EmergentBiome evidence graph** for the public
  proposal page. Name the 662-record persistent MVP and 7,965-row release
  projection separately when describing implemented graph scope.
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
2. Direct monitoring is expensive and spatially sparse.
3. The EmergentBiome Molecular Atlas presents sequencing-derived evidence and
   explicit gaps through a traceable review surface.
4. The internal MBAG architecture separates representation context, mechanism evidence, and reliability
   guardrails.
5. Evidence cards guide molecular diligence and the next measurement.
6. Paired abundance, environmental, uncertainty, and field evidence unlock
   calibrated methane-risk intelligence.

This sequence keeps the present capability commercially useful and
scientifically bounded while showing a credible route into MRV.
