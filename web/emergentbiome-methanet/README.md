# EmergentBiome Molecular Atlas site

This directory builds the
[EmergentBiome Molecular Atlas landing page](https://emergentbiome.earth/) and
assembles the dated technical [report alias](https://emergentbiome.earth/report/).
The internal repository, report-builder, and some historical artifacts retain
MethaNet/MBAG identifiers for traceability. Public landing copy uses
**EmergentBiome evidence graph**.

Landing source changes on `main` are published to `gh-pages` by the repository's
GitHub Actions workflow. The workflow updates landing assets and preserves the
`/report/` tree, frozen atlas, dated archive and custom-domain binding. The live
three-case public explorer is at [Explore the evidence](https://emergentbiome.earth/#scene-network).
The three-reviewer package, deployment receipt and browser checks are retained
locally under `results/reports/mvo_landing_release_20260927/`; the current exact
publication revision is recorded in the `gh-pages` branch history. The 28 September
narrative revision and the regenerated report were published together as `gh-pages`
revision `30d8715`; [`design-qa.md`](design-qa.md) records the changes and checks, and
the local QA package is `results/reports/landing_narrative_review_20260928/`.

The shared position is:

> The EmergentBiome Molecular Atlas organizes MAG/proteome representations,
> pathway evidence, provenance, and validation gaps to guide candidate review
> and the next field measurement. It does not infer measured methane flux or
> calibrated sample risk from the current molecular release.

The repository-wide narrative and claim rules live in
[`../../docs/methanet_positioning_and_claims.md`](../../docs/methanet_positioning_and_claims.md).
[`DIGEST.md`](DIGEST.md) records the verified page numbers, evidence sources,
and real-versus-illustrative visual contract.

## Public Reading Path

The landing page carries the complete proposal-facing evidence narrative.
The linked technical report is a dated deep dive and must be checked for
release parity before its public alias is promoted.

| Surface | Primary audience | Role |
| --- | --- | --- |
| Landing page | Proposal reviewers, blue-carbon developers, verifiers, partners, and funders | Explain the methane measurement gap, show one real evidence card and the frozen atlas, and distinguish current screening from field validation |
| Technical report | Scientific and diligence reviewers | Expose the tri-view evidence contract, comparability boundaries, candidate cards, source provenance, and validation agenda |

The source and locally built surfaces use the August 10, 2026 controlled-diligence release. Routine scientific publication and indexing remain blocked until the publication gates pass. A September 23 user-directed correction of the already-public landing and stale report is scoped to the reconciled, claim-bounded `noindex` bundle:

| Measure | Current release |
| --- | ---: |
| Registered MAG/proteome units | 7,965 |
| ESM-2-bearing units | 7,710 |
| gLM2 payloads | 7,717 |
| Data-complete tri-views | 7,710 |
| Schema-normalized tri-views | 7,710 |
| Pipeline-normalized tri-views, comparability pending | 5,209 |
| Cross-lane mechanism-comparable tri-views | 0 |
| MUCC v1 source-scaffold tri-views | 2,501 |

Data-complete and mechanism-comparable describe different evidence states.
This distinction remains visible in page copy, report tables, candidate cards,
and claim boundaries.

The hero defines the molecular atlas and EmergentBiome evidence graph. The
closing evidence-language key defines monitoring, reporting, and verification,
MAG, ESM-2, gLM2, tri-view, MUCC v1, and VM0033.

The August 10 visual export contains 7,710 embedding-bearing MAG/proteome
records and 2,226 **displayed map links**. Its 26 selected one-way
nearest-core candidate links are distinct from the 2,200 sampled full-atlas
cross-domain kNN links. In the frozen tables, 2,434 of 2,608 wetland and
4,475 of 4,584 mangrove records have a rumen record as their raw-cosine
nearest neighbor in the 625-record POC core; 26 of 27 selected wetland or
mangrove candidate cards share that property. The reference core is
rumen-heavy (518 rumen, 107 wetland), and these assignments do not prove
transfer. The report's zero concerns reciprocal cross-domain top-35 pairs
after per-dimension standardization, a separate statistic.
The map opens in UMAP for visual navigation because the diffusion projection
compresses most target records into a narrow band. Diffusion, t-SNE, and PCA remain
selectable. t-SNE is computed for the same 7,710 embedding-bearing records as the
other views; the 255 registered gap rows have no projection coordinates. All link
membership is computed in the high-dimensional ESM-2
space, independently of the displayed 2D projection.

## Landing Page Story

The page uses a title sequence, ten scenes, and a closing ask:

| Scene | Reader takeaway | Evidence mode |
| --- | --- | --- |
| Hero | States the problem (wetland methane can cancel part of the carbon benefit) and the offer (DNA turned into checkable evidence, plus the field measurement that settles what DNA cannot) | Decorative seeded particle field |
| 1. The Climate Question | Methane can erode a wetland's climate benefit, most where water is fresh, brackish or cut off from the tide | Sourced facts (IPCC AR6; Poffenbarger et al. 2011; Kroeger et al. 2017) with illustrative motion; the net-balance range is labeled illustrative and carries no values |
| 2. The Measurement Gap | VM0033 allows a default methane value only above 18 ppt salinity; the atlas has zero genome-to-flux pairs, which is not zero emissions | Real zero-pairing anchor with an illustrative field; names what a usable DNA-to-flux pair records (specimen, depth, time window, chamber footprint) |
| 3. The Molecular Atlas | 7,710 genome records mapped by their proteins; nearness is a lead that the four views test | Real counts, UMAP coordinates and selected links; the map is fitted to the space the reading panel and copy card leave free |
| 4. The Evidence Card | One real genome's card separates what is recorded, what is unresolved and the next measurement | Real candidate record with a schematic canvas |
| 5. Beyond Salinity | Salinity sets the baseline; genome evidence flags exceptions worth measuring | Explicitly illustrative teaching plot, sourced (Poffenbarger et al. 2011; Krause and Treude 2021) |
| 6. Evidence Scope | Methane screening runs through two annotation routes that are compared separately; other gas lenses are planned | Real atlas geometry colored by route (5,209 shared pipeline, 2,501 source annotations; 0 compared across routes) |
| 7. Molecular evidence, practical decisions | Check what a gene can mean, pin down its sample and depth, choose the test that settles it | Illustrative mangrove workflow with three real evidence-case dialogs |
| 8. Explore the Evidence | The ontology keeps easily conflated evidence apart and shows why a claim is on hold; three real review profiles | Separately scoped September projection; six rules drawn from the MVO competency questions |
| 9. Validation Path | Molecular review works today; calibrated risk needs five more rungs of paired evidence | Real MRV roadmap |
| 10. Partnership Path | The proposed study pairs each sediment metagenome with chamber flux, chemistry and hydrology | Schematic of the proposed design from `EB.study`; not collected data or a site map |

The closing ask describes a proposed two-season study of 144 metagenome sample-events: 3 restoration stages × 4 salinity positions × 3 replicate plots = 36 plots, two microsites per plot, revisited in a wet and a dry campaign. Revisits are repeated measurements, not independent replicates; plots are the replicate unit. Funding, site access and permits remain conditions. The planned study is separate from the frozen atlas denominator. Environmental and reviewed marker baselines are compared on the same held-out records, with each model frozen before its flux outcomes are unblinded: the first campaign tests a new site, and the second tests seasonal transfer at the same points, not independent-site validation.

Chronology stays visible without release-management detail. The hero and header name both dated scopes (the August 10 atlas and the September 26 case reviews), scene 07 marks where the case evidence enters, and the claim bar's date chip follows the scene in view: atlas data, case evidence, or the proposed study.

## Three-case public projection

The September extension uses `molecular-application-cases-public-v1.json` and
`molecular-evidence-network-public-v1.json`. These files retain selected numerical
source facts, gene identities, uncertainty and project-authored review status.
Dataset creators, versioned DOI links, CC BY 4.0 attribution and modification
notices accompany the cases. The advanced view is labeled **Selected source
facts**. It contains an allowlisted subset of the internal graph.

Raw annotation-database text, KO assignment tables, raw sequence payloads,
internal policy objects, private paths and the original internal case JSON files
are excluded from every publication path. The canonical graph and internal
query-purpose authorization remain unchanged. `noindex` is retained as an
indexing choice; it provides no access control.

`tools/assemble_landing.py` creates a new landing-only staging directory, requires
a hash-bound publication-review receipt and rejects extra data files. It never
changes the report alias. The independent three-reviewer package and release
receipts are stored internally under
`results/reports/mvo_landing_release_20260927/`. Reviews are automated internal
reviews, with scientific-content, funding-narrative and UX scopes separately
documented. They do not constitute independent biological or legal certification.

Both GitHub Actions deployment and `tools/publish_site.sh deploy [--push]`
require `publication-review.json` to approve all three release-review lanes and
the exact SHA-256 hashes of both public case-data files. Any change to either
projection invalidates that approval until the review receipt is refreshed.
The receipt is source-side release control and is deliberately excluded from
the published runtime. `tools/publish_site.sh build` remains available for
local QA; a local build is not publication approval.

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

The dated `release_ledger.json` is the numerical authority. `config.js` is its
public rendering projection for headline numbers, snapshot dates, copy, the
maturity ladder, claim boundaries, and milestones.

1. Reconcile the new release against `DIGEST.md` and the dated report freeze.
2. Update the relevant values and copy in `config.js`.
3. Refresh the landing visualization with `tools/export_atlas.py`.
4. Generate the technical report into
   `results/reports/emergentbiome_molecular_atlas_20260928_consolidated/`.
5. Build the public tree with `tools/publish_site.sh build`.
6. Verify the landing page, the stable `/report/` alias, claim-boundary text,
   and the absence of public raw report bundles.

For the current August 10 freeze, the report build command from the repository
root is:

```bash
MPLCONFIGDIR=/tmp/methanet_mpl NUMBA_CACHE_DIR=/tmp/methanet_numba \
.venv/bin/python scripts/reports/build_mbag_nextgen_molecular_niche_atlas.py \
  --lane-registry configs/methanet_atlas_lanes.tsv \
  --freeze-manifest results/reports/methanet_3view_payload_freeze_20260810_end_to_end/freeze_manifest.tsv \
  --skip-phate --output-dir results/reports/emergentbiome_molecular_atlas_20260928_consolidated
```

`--skip-phate` keeps the published projection buttons to UMAP (the default),
diffusion map, t-SNE and PCA; an environment with PHATE installed would add a
fifth button and fail the browser check.

That report bundle is ignored by Git and depends on the local frozen lane
warehouses and source manifests. Regenerate or restore it before running the
site builder on a clean checkout.

Run `tools/validate_release_parity.py` against the ledger, freeze, report,
`DIGEST.md`, `config.js`, and `data/atlas.json`. Run
`tools/verify_page_firefox.py` against the built tree for desktop, tablet,
mobile, keyboard, overflow, noindex, and static-fallback checks. It also drives
the scene-07 case dialog, its handoff (case and keyboard focus) into the scene-08
explorer, selected source facts, immersive mode and recovery from a data outage.

## File Map

```text
index.html              public semantic scaffold and metadata
styles.css              responsive visual and accessibility system
config.js               verified numbers, public copy, claims, and the proposed-study design
main.js                 page orchestration, copy injection, and accessibility
scenes/                 seeded visual scenes
data/atlas.json         local landing visualization feed
data/*-public-v1.json   allowlisted, attributed three-case presentation
tools/assemble_landing.py reviewed landing-only staging and exposure guard
tools/export_atlas.py   deterministic atlas exporter
tools/verify_page_firefox.py Firefox browser and accessibility verifier
tools/validate_release_parity.py cross-artifact release-ledger parity gate
tools/publish_site.sh   public-tree builder and GitHub Pages publisher
DIGEST.md               evidence reconciliation and visual honesty contract
```

The public `/report/` alias contains the reader-facing report and its visual
assets. Internal audit tables, raw report data bundles, and source JSON remain
outside that stable public path.
