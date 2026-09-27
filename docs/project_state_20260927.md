# MethaNet project state recap

Status snapshot: 2026-09-27. This is a navigational recap of the warehouse,
atlas, evidence graph, public experience, and open gates. It does not promote a
new data release or authorize distribution of internal reports or source data.

## Executive summary

MethaNet has moved from a 662-record proof-of-concept graph to a governed
multi-lane molecular warehouse plus a formal, ontology-backed evidence graph.
The data warehouse and August 10 release remain the source authority. MVO 0.2.0
adds a selected local molecular evidence projection and validated query
contracts; it does not replace the frozen release or make its entire 7,965-row
denominator a fully materialized sample-resolved graph. SQL reproduced all
three matched fixed-query outputs exactly and had lower warm medians in this
small local test. The graph's proposed value is explicit relationship/provenance
navigation; its decision-utility advantage remains unmeasured.

The EmergentBiome landing/report experience was separately corrected and
deployed to GitHub Pages. That website deployment does not deploy the local MVO
graph, change the warehouse, or authorize publication of internal MVO review
materials.

## System map

| Layer | Current role and status | Authority boundary |
| --- | --- | --- |
| Source datasets and lane warehouses | Four different contracts: three pipeline-normalized 24-table warehouses and the MUCC v1 61-table source-scaffold warehouse | Dataset-specific source files/manifests remain authoritative; data are not bundled into a clean clone |
| August 10 atlas release | 7,965 registered `(lane_id, proteome_id)` units; 7,710 complete molecular tri-views; 255 explicit exclusions; 0 mechanism-comparable units | Frozen denominator and status contract; not sample-resolved emissions evidence |
| Historical attestation graph | 662 POC molecular units with the older Kuzu/TSV/Parquet graph artifacts | Separate historical MVP, not the 7,965-unit release or MVO graph |
| MVO 0.1.0 | Formal OWL/SHACL core and immutable release-record/provenance/readiness graph | Original local validated snapshot; retained as rollback/historical evidence |
| MVO 0.2.0 | 735,060 RDF statements, 37,071 resources; additive selected molecular evidence over the 0.1.0 graph | Local validated implementation; 145 selected MAG records, not a full expansion across all 7,965 records |
| SQL/DuckDB controls | Three matched fixed-query controls over shared normalized facts | Exact result parity in those cases; lower measured warm medians, not a universal speed claim |
| Website and report | EmergentBiome landing and `/report/` on GitHub Pages, corrected from the declared August freeze | Presentation layer; separate from the current local ontology graph and scientific release promotion |

## Warehouse and atlas

The Atlas has four release lanes: POC core (625), MSM China mangrove (1,428),
Futian mangrove (3,156 release-ready plus 248 source gaps), and MUCC v1 Old
Woman Creek (2,501 release-complete source-scaffold units from a 2,508 MAG
roster). The August ledger totals 7,965 lane-scoped registered records, of
which 7,710 meet its tri-view availability contract. Those totals combine
different evidence contracts and do not authorize cross-lane quantitative
mechanism comparison.

The three pipeline warehouses share a normalized event schema. MUCC keeps a
distinct source-scaffold contract. Its RNA, sample context, chamber, porewater,
and tower records preserve their original measurement grains. The known
tower-mapping issue and unresolved sample/site identities remain explicit.
MVO's 133 sample/site references and 133 sample/flux windows remain unresolved;
the latest selected snapshot admits zero exact molecular-to-flux pairs.
The MVO source audit covered 133 physical warehouse tables and 390,950,494
heterogeneous rows; this is an audit inventory count, not a deduplicated
biological-observation denominator.

See [atlas foundation](atlas_data_foundation.md), [artifact inventory](current_artifact_inventory.md),
and the [release ledger](releases/atlas_20260810_release_ledger.json).

## Ontology, graph, and SQL

MVO separates source artifacts, lane-scoped molecular records, evidence
assertions, review states, contextual measurements, hypotheses, rights, and
decisions. OWL/SHACL and executable admission guards encode the current local
semantic and data-quality profile. The builder preserves the original 0.1.0
statements and adds selected 0.2.0 evidence. Neo4j is a reconstructable
projection; the canonical RDF snapshot and warehouse provenance are retained.

The selected 0.2.0 slice includes 145 MAGs; 934 primary feature/annotation
records; 863 verified loci and 864 protein records; 2,243 supplementary
annotation assertions; 1,596 processed RNA cells; 519 numeric measurements and
37 missing-value gaps; 9,510 source-context assertions; 18 executed fixed
queries; and explicit review/validation packets. These counts have different
units and must not be conflated with samples, abundance, independent ecological
replicates, or observations of methane flux. The full registered coverage
ledger remains 7,965 × 18 panel-family rows. No cross-run embedding similarity
edges, calibrated risk tiers, or credit decisions were created.

Three preregistered graph-vs-SQL controls produced exact full-row parity:

| Query | Rows | Neo4j warm median | SQL warm median |
| --- | ---: | ---: | ---: |
| MQ03 competing interpretations | 1,032 | 71.64 ms | 18.27 ms |
| MQ09 molecular-to-field path | 1,596 | 111.72 ms | 28.46 ms |
| MQ18 diligence packet | 435 | 51.63 ms | 16.05 ms |

Only three warm sequential local timings were run. There is no concurrency,
cold-start, production-scale, resource-use, or analyst-time comparison. Keep SQL
for tabular analytics and aggregation; use graph traversal where explicit
multi-hop provenance and relationships aid inspection. The semantic contracts,
source mappings, and data quality together determine query meaning.

Read [MVO architecture](../ontology/docs/ARCHITECTURE.md), [handoff](../ontology/docs/MOLECULAR_HANDOFF_20260926.md),
[verification](../ontology/docs/VERIFICATION.md), and [store comparison](../ontology/docs/STORE_COMPARISON_20260926.md).

## Research, manuscript, and supporting engineering work

The local workspace also contains unpublished wetland-methane manuscript
packages, an MVO technical whitepaper/demo, source-data workbooks, validation
receipts, and reproducibility artifacts. Their package-level records identify
author declarations, source permissions, licenses, or public-deposition steps
that remain open. Proposed deposit directories are drafts, not proof of an
authorized public deposit. The internal MVO report bundle also contains source
locators and rights-sensitive evidence and is not a partner-cleared artifact.

Supporting implementation includes a separate tower/process mapping repair
derivation, atlas metadata-readiness and lane-registry validation updates,
public-projection allowlisting for the website explorer, and browser/release
verification scripts. The tower repair is a new derived output; the August
warehouse was not silently rewritten. The generated evidence receipts and
large working datasets live in ignored local `results/` paths and are not
required in the source-control release.

## Website and deployment

The landing page now presents molecular evidence as a path from a blue-carbon
question to mechanisms, organisms, evidence review, uncertainty, and a next
field observation. The interactive evidence-network scene uses three bounded
case profiles and review status; its allowlisted public projection contains 492
presentation nodes, 459 source objects, and 4,725 selected source facts. It is
not a live, unrestricted graph-browser connection or a sample-level emissions
model. Three internal science/funding/experience review lanes and live
interaction/page checks closed for this scoped landing release; they are not
external peer review, biological certification, legal advice, or accessibility
certification. The website deployment is independent of the MVO local Neo4j
service.

The latest landing-only `gh-pages` publication is commit
[`ec963456`](https://github.com/jaygut/MethaNet/commit/ec963456aa6b8cd63d016c7b889a39e4a0f1729e).
It preserves the `/report/` alias and published August atlas. The report alias
should not be described as serving MVO 0.2.0 unless a separately reviewed
release explicitly changes that wiring. See the current [landing source and
runbook](../web/emergentbiome-methanet/README.md) and [public release
record](methanet_triview_release_20260810.md).

## What is still open

- Independent microbial-ecology review of marker specificity and interpretation.
- External redistribution/license/source approvals for detailed data-bearing
  evidence and all manuscript contributors.
- Exact sample/specimen/site/depth/time resolution and validated field-process
  measurements paired to molecular records.
- Abundance/read coverage, compatible functional assays, environmental
  covariates, uncertainty propagation, and held-out site/season validation.
- Any predictive claim about incremental model value, methane flux, calibrated
  sample/project risk, carbon-credit eligibility, or intervention outcomes.
- A user study measuring whether graph exploration improves scientific review
  decisions or partner comprehension compared with a clear SQL/report workflow.
- Production security, service identity, multi-tenant authorization, backup,
  operational SLOs, and externally reviewed API/MCP boundaries.

The manuscript package records say that required author and rights clearances
remain outstanding. A public Git push is publication. This recap therefore
does not imply clearance to publish manuscript drafts, internal review bundles,
or underlying data.

## Verification recorded for this recap

- Root workspace test suite: **221 passed, 2 skipped**; 45 existing deprecation
  warnings were emitted. A fresh worktree with only Git-tracked files passed
  **219 tests with 4 skips**; the two additional skips reflect checks whose
  inputs are deliberately retained as local-only data.
- MVO package test suite: **119 passed**; five RDFLib deprecation warnings
  remain visible. The same 119 tests passed from the fresh worktree.
- Atlas release contract validator: **352 checks, zero errors, zero warnings**
  in tracked-contract-only mode. This mode does not scan the large local data
  payloads.
- Seven landing/atlas focused test modules: **29 passed** (included in the root
  suite total, not additional tests).
- Local Markdown links resolved in the reviewed documentation set; `git diff
  --check` passed.

These checks validate code contracts and source organization. They do not clear
publication rights, provide independent biological review, or close the listed
sample-level validation gaps.
