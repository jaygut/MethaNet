# MethaNet project state recap

Status snapshot of 16 November 2026. This recap updates the
[27 September recap](project_state_20260927.md), which stays as a dated
snapshot. It does not promote a new data release, and it does not authorise
distribution of internal reports, manuscripts or source data.

## Executive summary

The molecular release is unchanged. The August 10, 2026 freeze still defines
7,965 lane-scoped `(lane_id, proteome_id)` records, of which 7,710 carry the
complete protein, genomic-context and functional tri-view and 255 remain
explicit exclusions. Since September the work has corrected how the atlas
geometry is read, tightened claims and provenance, brought the corrected report
builder and landing copy onto `main`, deployed the corrected landing copy, and
designed the field evidence that the atlas lacks.

No sample in the atlas is yet paired with a methane measurement taken at the
same place, depth and time. That missing pair is the main gap between the
molecular atlas and any validated methane model.

## What changed since 27 September

| Date | Change | Where it is recorded |
| --- | --- | --- |
| 29 Sep 2026 | The 662 pilot proteomes were re-embedded with final-layer pooling, so the reference core and the other lanes share one embedding configuration | [Reconciliation receipt](releases/atlas_embedding_reconciliation_20260929.md) |
| 21 Oct 2026 | Taxonomy-controlled geometry retest. The pilot's ecosystem separation is a domain and order separation. Genome embedding neighbours share a phylum 77% of the time against 10% for random pairs, so the links support navigation. The embeddings carry a source signature, and no environment signal could be separated from study, processing or amino-acid composition effects with the current lanes | [README](../README.md), [positioning and claims](methanet_positioning_and_claims.md) |
| 21 Oct 2026 | Claim corrections in the core documents, report-builder fixes (lane-warehouse screening scores, GTDB release coverage), and a 21 October report bundle as the publisher default | Branch commits listed below |
| 21 Oct 2026 | MVO shapes 0.2.1 require a named human reviewer (`prov:Person`) for accepted assertions | [Ontology package](../ontology/README.md) |
| 22 Oct 2026 | Repository history normalised to one commit identity, with a commit guard (`scripts/check_provenance.py`, `.githooks/`) that blocks retired identities and machine details | [Repository guide](repository_guide.md) |
| 21 Oct and 1 Nov 2026 | Unpublished manuscript packages (V18 to V21) and their builders, the October design-lab sprint and private drafts kept out of the public tree by ignore rules | `.gitignore` |
| 16 Nov 2026 | Field sampling blueprint for paired campaigns, reviewed end to end, maintained as an internal working document | Section below |

The 7,965 count is read as lane-scoped records. A cross-lane identity check is
open for the next release, and the August freeze is not rewritten.

## Warehouse and atlas

The four lanes and their contracts are unchanged.

| Lane | Registered | Tri-view complete | Contract |
| --- | ---: | ---: | --- |
| POC reference core | 625 | 625 | Pipeline-normalised warehouse |
| MSM China mangrove | 1,428 | 1,428 | Pipeline-normalised warehouse |
| Futian mangrove | 3,404 | 3,156 | Pipeline-normalised warehouse, 248 source gaps |
| MUCC v1 Old Woman Creek | 2,508 | 2,501 | 61-table source scaffold with expression, chamber, porewater and tower records |

The sample layer of the August metadata-readiness build holds 280 staged sample
rows (133 Old Woman Creek, 82 China coast and 65 Futian) and 293,245 candidate
genome-to-sample links, which are exact only for Old Woman Creek expression
cells. It holds no release-authorised exact sample join and no exact
molecular-to-flux pair.

The Old Woman Creek process records (275 chamber fluxes, 5,280 porewater
methane concentrations, 29,280 gap-filled tower fluxes) are not linked to any
sequenced sample. They also mix two chamber methods with no shared year. MVO
0.2.0 remains an additive local projection over 145 selected genomes, and SQL
over the lane warehouses remains the source of truth.

## Claim boundaries

These hold for every document, report, deck and page.

- MAG and proteome functional potential is distinct from sample-level
  ecological interpretation.
- Current MBAG outputs are not final MRV risk scores.
- Final A to E methane-risk tiers wait for sample mapping, abundance or read
  coverage, environmental covariates, uncertainty propagation, calibration, and
  flux or process validation.
- The molecular atlas alone does not support claims of carbon-credit approval,
  measured methane flux, or source-independent rumen-to-wetland transfer.
- Genome embeddings serve navigation. Environment enters a model through
  measured sample context.
- A molecular screen can help decide where to measure. It cannot clear a stratum
  or site until a multi-site study bounds how often it misses, so field methane
  stays measured and molecular results stay supplementary.
- Failed, pending, partial and missing evidence stay as explicit status rows.

## Field evidence program

The next layer of the platform is field data that pairs molecular, chemical and
flux records by design. A field sampling blueprint dated 16 November 2026
defines that layer. It sets out the event core, conditional and anchor modules,
a field data contract that extends design-lab schema 0.1.0, a rights passport
per specimen, and a model data plan with site-level holdouts.

Six reviews covered it, in biogeochemistry, molecular methods and biodiversity,
machine learning, strategy and rights, an evidence audit, and design. The
blueprint is an internal working document and is not part of this public
repository.

The next deliverables follow in order.

1. Field schema 0.2, written from the blueprint's field data contract and mapped
   to MIxS 7.0.1, the Darwin Core Data Package and the Coastal Carbon Network
   templates.
2. A training-data contract covering units, label definitions, decision-time
   inputs, splits, leakage controls, licence lineage and documentation.
3. A preregistered validation design fixed before any outcome is seen.
4. A method rehearsal at partner sites, followed by a multi-site survey that
   measures the variance components the final design needs.

## Branch and deployment state

- `main` on GitHub is the rewritten history from 22 October 2026 plus ten
  correction commits and this documentation update. The corrections carry the
  21 October claim, report and landing corrections, the ignore rules for
  unpublished manuscripts, and the geometry-retest wording. They were rebased
  onto the rewritten `main` as the branch `fix/strategic-review-20261021`, and
  `main` was fast-forwarded to that branch on 16 November 2026.
- Several of these commits change `web/emergentbiome-methanet/`, so the push to
  `main` deployed the landing page through
  `.github/workflows/deploy-landing.yml` on 16 November 2026, after a locally
  verified preview. The workflow refreshes the landing files only. It changed
  the landing copy and scene scripts and left `/report/` and the landing data
  as they were.
- The live landing now carries the 21 October wording. The report and the
  landing data still come from the 29 September layer-33 bundle (last deployed
  13 October 2026) against the August freeze, and the site stays under
  `noindex`. The 21 October report bundle is not deployed yet. Publishing it is
  a separate step that needs its own approved preview.

## Verification recorded for this recap

- The unit suite on the working branch gave 264 passed and 1 skipped, with 95
  subtests passed (`.venv/bin/python -m pytest tests/unit -q`).
- The commit guard (`scripts/check_provenance.py range`) passed on the ten branch
  commits, and `tree HEAD` passed on the full tree.
- The rebased branch tree is byte-identical to the pre-rebase branch tip.
- Before the push to `main`, the deploy workflow's steps were run locally
  against the published site. The assembled landing loaded at desktop and phone
  widths with no console errors, and the files the workflow then published
  match the previewed ones. GitHub reported the Pages deployment as successful.
  A browser check of the live page is still pending.

These checks cover code contracts and repository hygiene. They do not provide
independent biological review, clear publication rights, or close the
sample-level validation gaps.
