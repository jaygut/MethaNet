# EmergentBiome Molecular Atlas site

This directory builds the
[EmergentBiome Molecular Atlas landing page](https://emergentbiome.earth/) and
assembles the dated technical [report alias](https://emergentbiome.earth/report/).
The internal repository, report-builder, and some historical artifacts retain
MethaNet/MBAG identifiers for traceability. Public landing copy uses
**EmergentBiome evidence graph**.

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

The page uses a title sequence, nine scroll-driven scenes, and a closing ask:

| Scene | Reader takeaway | Evidence mode |
| --- | --- | --- |
| Hero | Introduces the molecular atlas and scoped evidence graph | Decorative seeded particle field |
| 1. The Climate Question | Methane can erode blue-carbon climate value | Sourced climate facts with illustrative motion |
| 2. The Measurement Gap | Direct monitoring is costly and spatially sparse | Illustrative measurement gap |
| 3. The Evidence Card | One frozen candidate exposes recorded evidence, unresolved joins, and the next measurement | Real candidate record with a schematic canvas |
| 4. Complementary Evidence | Proxies and molecular evidence answer complementary field-design questions | Explicitly illustrative teaching plot |
| 5. Explore the Atlas | Inspect 7,710 frozen records and distinguish nearest-core matches from reciprocal-pair sensitivity | Real counts, coordinates, and selected links |
| 6. Evidence Scope | Each gas or mechanism lens requires its own harmonization and validation gate | Real atlas geometry with bounded lens states |
| 7. The Evidence Trail | The current POC graph preserves evidence, provenance, claim scope, and validation gaps | Real 662-record graph schema |
| 8. Validation Path | Molecular attestation is available now; calibrated MRV follows paired validation | Real MRV roadmap |
| 9. Partnership Path | A field cohort could connect molecular screening with paired outcomes | Roadmap and partnership target |

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
4. Generate the reconciled technical report into
   `results/reports/emergentbiome_molecular_atlas_20260923_reconciled/`.
5. Build the public tree with `tools/publish_site.sh build`.
6. Verify the landing page, the stable `/report/` alias, claim-boundary text,
   and the absence of public raw report bundles.

For the current August 10 freeze, the report build command from the repository
root is:

```bash
MPLCONFIGDIR=/tmp/methanet_mpl_20260923 NUMBA_CACHE_DIR=/tmp/methanet_numba_20260923 \
.venv/bin/python scripts/reports/build_mbag_nextgen_molecular_niche_atlas.py \
  --lane-registry configs/methanet_atlas_lanes.tsv \
  --freeze-manifest results/reports/methanet_3view_payload_freeze_20260810_end_to_end/freeze_manifest.tsv \
  --output-dir results/reports/emergentbiome_molecular_atlas_20260923_reconciled
```

That report bundle is ignored by Git and depends on the local frozen lane
warehouses and source manifests. Regenerate or restore it before running the
site builder on a clean checkout.

Run `tools/validate_release_parity.py` against the ledger, freeze, report,
`DIGEST.md`, `config.js`, and `data/atlas.json`. Run
`tools/verify_page_firefox.py` against the built tree for desktop, tablet,
mobile, keyboard, overflow, noindex, and static-fallback checks.

## File Map

```text
index.html              public semantic scaffold and metadata
styles.css              responsive visual and accessibility system
config.js               verified numbers, public copy, claims, and milestones
main.js                 page orchestration, copy injection, and accessibility
scenes/                 seeded visual scenes
data/atlas.json         local landing visualization feed
tools/export_atlas.py   deterministic atlas exporter
tools/verify_page_firefox.py Firefox browser and accessibility verifier
tools/validate_release_parity.py cross-artifact release-ledger parity gate
tools/publish_site.sh   public-tree builder and GitHub Pages publisher
DIGEST.md               evidence reconciliation and visual honesty contract
```

The public `/report/` alias contains the reader-facing report and its visual
assets. Internal audit tables, raw report data bundles, and source JSON remain
outside that stable public path.
