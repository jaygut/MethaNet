# EmergentBiome / MethaNet Public Site

This directory builds the
[EmergentBiome landing page](https://emergentbiome.earth/) and publishes the
stable [MBAG report alias](https://emergentbiome.earth/report/). MBAG means
MethaNet Bridge Attestation Graph.

The shared position is:

> MethaNet is a molecular-attestation system for blue-carbon methane
> diligence. The MethaNet Bridge Attestation Graph, or MBAG, turns sequencing
> into a knowledge graph for candidate review, monitoring design, validation
> planning, and future calibrated methane-risk intelligence.

The repository-wide narrative and claim rules live in
[`../../docs/methanet_positioning_and_claims.md`](../../docs/methanet_positioning_and_claims.md).
[`DIGEST.md`](DIGEST.md) records the verified page numbers, evidence sources,
and real-versus-illustrative visual contract.

## Public Reading Path

The landing page and report perform different jobs while telling the same
story:

| Surface | Primary audience | Role |
| --- | --- | --- |
| Landing page | Blue-carbon developers, verifiers, raters, buyers, partners, and funders | Explain the methane blind spot, present molecular attestation as the current product, and show the route to calibrated MRV |
| MBAG report | Scientific, technical, and diligence reviewers | Expose the tri-view evidence contract, comparability boundaries, candidate cards, source provenance, and validation agenda |

Both surfaces use the July 24, 2026 scientific-reconciliation release:

| Measure | Current release |
| --- | ---: |
| Registered MAG/proteome units | 7,965 |
| ESM-2-bearing units | 7,710 |
| gLM2 payloads | 7,717 |
| Data-complete tri-views | 7,484 |
| Mechanism-comparable POC tri-views | 625 |
| Harmonization-pending mangrove tri-views | 4,358 |
| MUCC v1 source-scaffold tri-views | 2,501 |

Data-complete and mechanism-comparable describe different evidence states.
This distinction remains visible in page copy, report tables, candidate cards,
and claim boundaries.

The landing page defines MBAG and monitoring, reporting, and verification in
the hero. Its closing evidence-language key also defines MAG, ESM-2, gLM2,
tri-view, MUCC v1, and VM0033. Public narrative copy expands or removes other
shorthand when the abbreviation would add friction.

## Landing Page Story

The page uses a title sequence, nine scroll-driven scenes, and a closing ask:

| Scene | Reader takeaway | Evidence mode |
| --- | --- | --- |
| Hero | Sequencing becomes a traceable molecular evidence graph | Real ESM-2 coordinates |
| 1. The Stakes | Methane can erode blue-carbon climate value | Sourced climate facts with illustrative motion |
| 2. The Blind Spot | Direct monitoring is costly and spatially sparse | Illustrative measurement gap |
| 3. What You Get | Evidence cards identify a candidate, its confidence, and the next measurement | Illustrative product shape grounded in the real evidence contract |
| 4. Versus the Cheap Method | Single environmental proxies miss microbial pathway structure | Illustrative teaching plot with sourced anchors |
| 5. The Evidence | The warehouse carries 7,484 data-complete tri-views under three explicit functional contracts | Real counts and coordinates |
| 6. One Engine, Many Maps | Each gas or mechanism lens requires its own harmonization and validation gate | Real atlas geometry with bounded lens states |
| 7. Platform And Moat | MBAG preserves evidence, provenance, claim scope, and validation actions | Real graph schema |
| 8. The Honest Ladder | Molecular attestation is available now; calibrated MRV follows paired validation | Real MRV roadmap |
| 9. Path And Ask | Field partnerships convert the evidence graph into a calibrated risk system | Real roadmap and partnership target |

## Claim Boundary

Current authorization covers MAG/proteome molecular screening, candidate
triage, evidence-card review, monitoring prioritization, and validation-study
design.

Calibrated sample and project methane-risk estimates require exact sample
linkage, abundance or read coverage, environmental covariates, uncertainty
propagation, and paired field or process validation. A to E risk tiers remain
target product vocabulary until those gates pass. Carbon-credit determinations
require methodology integration and independent review.

## Local Preview

The landing page loads its local visualization payload over HTTP:

```bash
cd web/emergentbiome-methanet
python3 -m http.server 8848
```

Open `http://localhost:8848/`.

## Refresh Workflow

`config.js` is the landing-page source of truth for headline numbers, snapshot
dates, public copy, the maturity ladder, claim boundaries, and milestones.

1. Reconcile the new release against `DIGEST.md` and the dated report freeze.
2. Update the relevant values and copy in `config.js`.
3. Refresh the landing visualization with `tools/export_atlas.py`.
4. Build the public tree with `tools/publish_site.sh build`.
5. Verify the landing page, the stable `/report/` alias, claim-boundary text,
   and the absence of public raw report bundles.

## File Map

```text
index.html              public semantic scaffold and metadata
styles.css              responsive visual and accessibility system
config.js               verified numbers, public copy, claims, and milestones
main.js                 page orchestration, copy injection, and accessibility
scenes/                 seeded visual scenes
data/atlas.json         local landing visualization feed
tools/export_atlas.py   deterministic atlas exporter
tools/verify_page.py    browser verification helper
tools/publish_site.sh   public-tree builder and GitHub Pages publisher
DIGEST.md               evidence reconciliation and visual honesty contract
```

The public `/report/` alias contains the reader-facing report and its visual
assets. Internal audit tables, raw report data bundles, and source JSON remain
outside that stable public path.
