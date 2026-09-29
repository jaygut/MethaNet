# Embedding correction: 29 September 2026

The current geometry is recomputed under layer-33 pooling. The historical QA notes below describe superseded geometry wherever they discuss coordinates, nearest-reference matches, similarities, or neighbor counts.

- All 662 pilot records passed retained-input, protein-count, configuration and historical-reproduction checks.
- The rebuilt report passed 20/20 gates. The local browser audit passed desktop, tablet, mobile, atlas controls, evidence cases and report checks.
- PDF and editable PowerPoint contain 19 slides. Layout checks found no overflow; a separate native PowerPoint render retained all text and seven reference links.
- Current release evidence: `docs/releases/atlas_embedding_reconciliation_20260929.json` and `results/reports/atlas_embedding_release_20260929/`.
- The user explicitly authorized this correction and deployment. Existing public-case scope and `noindex` remain in force.

---

# Narrative and report consolidation: 28 September 2026

final result: passed locally and on the live site; published 28 September 2026 after
the owner approved the local preview.

Second revision on top of the 27 September coherence pass: source `main` at `d9c6665`
plus uncommitted landing, report-builder and tool edits. The two reviewed public case
files are byte-identical to the hashes in `publication-review.json`; `atlas.json`,
`CNAME`, `noindex` and the public-data allowlist are unchanged. The regenerated report
is a new local bundle, `results/reports/emergentbiome_molecular_atlas_20260928_consolidated/`,
built from the same 10 August freeze; the 23 September bundle is untouched. The
local evidence package, gitignored, is `results/reports/landing_narrative_review_20260928/`.

## Landing narrative

- The hero now states the problem and the offer in two sentences: wetland microbes can
  release methane that cancels part of the carbon benefit, and EmergentBiome turns
  their DNA into checkable evidence and names the field measurement that settles what
  DNA cannot. What is available now and what is proposed are separated.
- Scene order follows the argument: problem (01), measurement gap (02), the data asset
  (03 atlas), what each genome's evidence looks like (04 card), why genomes add
  information beyond salinity (05), what is screened today (06), the three cases (07),
  the ontology (08), the validation ladder (09) and the proposed study (10).
- External facts carry source lines: IPCC AR6 warming potential, Poffenbarger et al.
  2011, Kroeger et al. 2017, Verra VM0033 (default methane value only above 18 ppt,
  checked against the methodology text) and Krause and Treude 2021.
- Plain words replace jargon on the landing (genome record, protein fingerprint,
  annotation route); technical terms stay in the glossary and the report. The hero's
  MAG chip moved to the glossary.
- Scene 03 explains what a point is, what the fingerprint is and why nearness is only a
  lead. Its four views have new explanatory text, including why raw cosine matches are
  weak and what the stricter mutual-neighbor test shows. The body now shows on phones.
- Scene 06 was highlighting nothing (its filter matched no record). It now colors the
  map by annotation route (5,209 shared pipeline, 2,501 Old Woman Creek source
  annotations; 0 compared across routes).
- Scene 08 explains why the ontology matters (an evidence-readiness dossier for
  developers and verifiers, not a credit or tonnage estimate) and lists six rules drawn
  from the MVO competency questions (CQ08, CQ10, CQ11, CQ13, CQ14, CQ19). The note
  states the verified status: 145 genome records, 18 fixed review queries checked
  against the source tables, no independent scientific review yet.

## Corrections of fact

- Nearest-reference denominators included the 107 core wetland genomes, which match
  themselves. Outside the core the counts are 2,434 of 2,501 wetland and 4,475 of
  4,584 mangrove records, and all 26 selected candidates outside the core. These
  matches are weak: the median closest-match cosine outside the core (0.983) is below
  the median of random atlas pairs (0.994). Landing, report and DIGEST now say so.
- The report showed expression as absent for every candidate. The source flag is the
  string "true", which the numeric coercion turned into 0. Old Woman Creek cards now
  show expression; other sources read "not assayed", not "absent".
- 5,209 records showed the status "evidence contract unresolved"; they now read
  "shared-pipeline screening; cross-route comparison pending".
- Futian records were labeled "Mangrove/MSM" in tooltips, cards and the sample chart,
  and carried a stale caveat about an unfinished bacterial annotation.
- The report called the diffusion map the primary view while the landing opened on
  UMAP. Both now open on UMAP; the report says why the diffusion view is dominated by
  the reference-core separation. Diffusion coordinates are now seeded and
  sign-normalized for reproducibility.

## Report consolidation

- Internal history is gone from the reader-facing text: prior ledgers, retired counts,
  "legacy" and "former" rankings, and instructions to report authors. One plain
  sentence keeps the quarantined combined index as a disclosed limitation.
- Raw status codes are replaced by labels in tables, tooltips and cards. Tables use one
  lane vocabulary shared with the landing; source papers link to their DOIs.
- Figures: the evidence-graph diagram has a legend and a stacked phone layout; the map
  draws all 2,226 links with a full legend and states which links are drawn; the matrix
  uses available and not-available cells with unique row codes (P, M, O); the wheel has
  a ring legend and scales on phones; the evidence-contract chart shows annotation
  routes and axis titles; the sample chart has readable labels, a gap row named as a
  gap, and correct lane colors. The unlabeled AI-style graphical abstract, whose check
  marks could read as completed validation, is replaced by the four gated layers with
  their current status.
- Lane colors in the report mirror the landing's four source hues.

## Verification of the final build

Headless Firefox 128.10 ESR with geckodriver 0.34.0 (driver advisory noted). Widths
below Firefox's 450px window minimum used full-page zoom; each record gives the
actual `clientWidth`.

- Interaction suite, all passing: 1,264 of 1,264 checks across 1440×900 (reduced and
  normal motion), 390×844, 320×568, 820×1180 and 900×500 (normal motion). The atlas
  projection check now expects the new announcement ("t-SNE view. The links do not
  change.").
- `tools/verify_page_firefox.py` on the final build: pass, no failures. Its report
  checks now expect the UMAP-first buttons (UMAP, Diffusion map, t-SNE, PCA) and 7,710
  plotted points in each projection; visible report values match the release ledger.
- Release parity 202/202. Publication-review validation passes; both reviewed case
  files keep their approved hashes. The report builder's 19 gates pass and its ledger
  reconciliation completes.
- Landing journey at 1440×900, 1280×720, 820×1180, 900×500, 390×844 and 320×568: no
  horizontal overflow, no console messages. On laptop-height screens the reading cards
  are tighter and more opaque, and Scenes 02, 05, 06 and 09 lay their canvases out
  around the card.
- 200% text-only zoom at 1440 and 390px and 200% full zoom on a 1280×720 window: no
  problems found.
- Keyboard Tab walk, first lap: 80 stops at desktop and 74 at 390px, the same pattern as
  the 27 September build (the four explorer edges show a stroke highlight rather than
  an outline; at desktop the large review-profile node and the tall inspector remain
  partly visible).
- Unit tests: 235 passed, 1 skipped. No em-dashes in shipped landing or report text.

## Publication

The owner approved the local preview on 28 September 2026. `tools/publish_site.sh deploy
--push` passed the hash-bound publication gate, rebuilt the site from this source and
the `20260928_consolidated` report bundle, and pushed `gh-pages` revision
`30d8715` (from `458954b`). That commit changes only landing files and `report/`;
the reviewed case data, `atlas.json`, `CNAME` (`emergentbiome.earth`), `noindex` and the
dated June snapshot are unchanged, and the retired graphical abstract is no longer
shipped. Cache-busted and plain requests to the live landing, `config.js` and
`/report/` returned the new content within a minute, and
`tools/verify_page_firefox.py --base https://emergentbiome.earth` passed with no
failures, no horizontal overflow and visible release values matching the ledger.

`configs/atlas_current_release.json` still names the 23 September bundle as
`reconciled_public_report_bundle`. It is deliberately unchanged: the ontology build
records that file's checksum as a pinned input, so editing it would invalidate
existing ontology snapshots. The published report is the 28 September bundle.

## Remaining limits

The report's evidence tables scroll sideways on phones, and the matrix and two charts
keep their text size by scrolling rather than shrinking. The report page has no
favicon. Safari, Chromium, physical devices, screen-reader speech and formal WCAG
certification were not tested.

---


# Landing coherence revision: 27 September 2026

final result: passed locally; published on 28 September 2026 together with the
narrative and report consolidation above.

Whole-journey review of the assembled landing: source `main` at `8c58cd0` plus
uncommitted landing edits. Both reviewed public case-data files are byte-identical
to the hashes in `publication-review.json`; the data, `/report/` alias, atlas
payload and CNAME are unchanged. The local evidence package, gitignored, is
`results/reports/landing_coherence_review_20260927/`.

## Confirmed problems fixed

1. Canvas labels drawn through `EBDraw.label` requested "JetBrains Mono", which is
   not vendored, so they rendered in a fallback serif. They now use IBM Plex Mono;
   static reduced-motion frames redraw once fonts load.
2. Scene 02 drew "validation target 80 to 100 sites". The repository has no source
   for it (added 27 June alongside `pairedFluxNow: 23`), and its unit conflicts with
   the proposed design. It is replaced by what a defensible molecule-to-flux pair
   records: specimen ID, sample depth, time window and chamber footprint. No count
   is substituted.
3. Scene 10 showed "AGENTIC PIPELINE < 4 days", which has no source and was flagged in
   the funding review, and a milestone strip that rendered "NOW · NOW". It now draws
   the proposed design from `EB.study`: 3 restoration stages × 4 salinity positions
   × 3 replicate plots, two microsites per plot, revisited in wet and dry seasons.
   That is 144 planned sample-events of repeated measurement, with plots as the
   replicate unit. It is badged "Proposed study" and marked schematic and
   conditional. The numbers match the submitted proposal's research plan (PDF
   SHA-256 `d681c153…`) and whitepaper section 7.
4. The rail and skip link named scene 07 "The Evidence Trail" while its heading
   reads "Molecular evidence, practical decisions", and its config copy still
   described the retired 662-record graph scene. Names are aligned, the stale
   readout is removed, and `tests/unit/test_landing_scene_navigation.py` guards the match.
5. The dialog-to-explorer handoff dropped keyboard focus to `<body>`. Focus now lands
   on the matching case button without scrolling.
6. The fixed scene rail, 39px of buttons, covered right-aligned cards, workflow
   labels, the explorer's inspector edge and closing actions between 721 and 1440px,
   by up to 37px at tablet width. A `--rail-gutter` token keeps content clear.
7. The fixed header and claim bar could hide keyboard-focused controls. Focusable
   elements and keyboard-scrollable panels now carry scroll margins. The closing
   ghost button had no focus ring; every closing action now shares one.
8. At 320×568 the hero centered its content under the fixed header and claim bar,
   and the scroll hint overlapped the term chips. Short and narrow heroes now grow
   with their content.
9. With 200% text on a 390px layout the page was 431px wide, because of the closing
   grid's min-content width and non-wrapping kickers. It now reflows at 390px.
10. Scene 03 drew its pending FIELD stage behind the candidate card. Stages are now
    placed from the card's measured position.
11. Scene 01 printed unitless values such as "-23 net". A labeled illustrative range,
    "range without measured site flux", replaces them.
12. At the close, the primary button scrolled back to the atlas under an external-link
    arrow, while the ask was the least prominent action. "Discuss field validation"
    is now primary, the in-page return uses ↑, and the glossary gains
    "Sample-event", the proposal's unit, which completes its grid.
13. The dialog's depth table hid the "Missing" salinity column off-screen at 390 and
    320px. It now fits.

## Coherence changes

- Chronology: the hero, header, scene-07 marker, dialog header and factsheet
  ("Atlas release counts") name the dated scopes. The claim-bar chip follows the
  scene in view: atlas data, case evidence or proposed study.
- Evaluation wording: each model is frozen before its flux outcomes are unblinded.
  The first campaign tests a new site; the second tests seasonal transfer at the same
  points, not independent-site validation. Comparisons with environmental covariates
  and a reviewed marker baseline use the same held-out records. The data are
  released even if molecular features add little.
- The hero defines MAG at first use.

## Verification of the final build

Headless Firefox 128.10 ESR with geckodriver 0.34.0 (driver advisory noted).
Widths below Firefox's 450px window minimum used full-page zoom; each record gives
the actual `clientWidth`.

- Interaction suite of 210 to 212 checks per configuration, all passing: 1440×900
  in reduced and normal motion, 390×844, 320×568, 820×1180, and 900×500 in normal
  motion. It covers all three dialog cases, the depth table and attribution; each
  handoff, with case, scroll and focus; all three explorer cases and four
  dimensions; nodes, edges, source facts and breadcrumbs; later pages, including
  page 3 of a 133-value RNA series; search, no-results, Escape and reset; Records
  and Network views; zoom, fit and pan; immersive mode with Escape, focus, scroll,
  selection and five cycles; the About panel; keyboard traversal; outages with retry
  for both data files; all four atlas projections; the report alias; both contact
  addresses and the organization link; and the chronology chip. After the final
  table change, dialog and handoff checks were re-run at 1440, 390 and 320px: 65/65 each.
- `tools/verify_page_firefox.py`, now also driving the dialog, handoff, source facts,
  immersive mode and outage recovery: pass. The same checker on the pre-revision
  build fails only on handoff focus.
- 200% text-only zoom at 1440 and 390px, and 200% full zoom on a 1280×720 window
  (640×360 CSS): no horizontal overflow; explorer reading cards do not overlap;
  resetting restores the standard layout.
- Keyboard Tab walk: every stop on the first lap has a visible indicator (80
  stops at desktop, 73 at 390px). No stop is hidden at 390px. At desktop, the large
  review-profile node and the tall inspector panel remain partly visible.
- Offscreen p5 scenes stop drawing while visible ones animate. No page console
  errors across all runs. Of 7,372 requests, the only failures were four
  `/favicon.ico` 404s from the unchanged report page.
- Release parity 202/202, including the one extra `report_manifest.summary.present`
  gate. Publication-review validation passes. Unit tests: 235 passed, 1 skipped.

## Remaining limits

With 200% text on a 390px layout, the dialog's four-column depth table scrolls
inside its own container, a data table that needs two-dimensional layout. Safari,
Chromium, physical devices, screen-reader speech and formal WCAG certification were
not tested. The report page has no favicon and was left unchanged.

---

# Reviewed landing release: 27 September 2026

final result: passed

Scoped visual and interaction acceptance for the three-case public presentation.
This section supersedes the local-only release status in the archived reviews
below. Deployment and live verification are recorded separately in
`results/reports/mvo_landing_release_20260927/`.

## Current comparison and inspected surfaces

- Target: preserve the existing landing and focus-network design while closing
  independently identified science, funding and experience findings.
- Before/after: `reviews/experience-qa/comparison-full.png` and
  `comparison-detail.png`, beneath the release package. Same interpret case,
  1440 × 1000 CSS viewport, DPR 1, headless Firefox. Both captures are the current
  public data projection; full views use equal proportional scaling and detail
  crops use matching framing. Both combined artifacts were visually inspected.
- Fresh assembled implementation: `assembled-page-qa/landing_desktop.png` and
  the network screenshots in `assembled-network-qa-final/`. Individual full-size
  normal and enlarged-text captures are in `reviews/experience-qa/iteration-2/`.
- Display/body/monospace fonts, teal/amber/grey/violet status palette, center
  profile, restrained glow, illustration quality and scene spacing are retained.
  Status meanings remain available as text. No decorative data marks were added.
- Narrative remains application → evidence interrogation → validation → funding
  ask. The closing copy now identifies the conditional two-season study and its
  comparison baselines. The 144 planned metagenomes represent repeated sample
  events at 72 microsites within 36 plots, not independent replicates.

## Findings closed

1. Source mode restores the original review page and selected RNA cell. Graph
   redraw preserves its own keyboard focus without stealing focus from search.
2. Source fact selection has an explicit pressed state and live announcement;
   the review-method panel places focus on its heading and returns it correctly.
3. Native 200% text-only enlargement switches to measured reading-column nodes,
   wraps status/case controls and releases header/claim-bar overlays into flow.
   Resetting text size restores the circular immersive graph and fixed chrome.
4. Public source links identify creators, deposited versions, licenses and the
   derived scope. Historical MethaNet screening is attributed correctly. OWC QC
   is source-reported; the mixed wetland group is labeled Wetland references.
5. Both new stylesheets are present in the release. Only explicitly allowlisted
   public data files are packaged; internal annotation exports remain excluded.
6. The source-local report 404 is resolved in the assembled release, where the
   existing report opens and all report/atlas parity checks pass unchanged.

## Verification and limitations

The final corrected-state experience receipt records 65 passing assertions,
with 7 supplementary observer-lifecycle checks. The preceding iteration-2
receipt records 63 passing assertions at the same runtime hashes.
The unit suites pass 23 tests; release parity passes 201 gates. Complete receipts
and preserved initial failures are in the three-reviewer package. The final
network fault test uses a controlled outage until Retry is observed; the earlier
one-shot test could auto-recover before the test runner observed the error UI.

Normal motion and reduced motion were exercised. Actual inspected CSS widths
include 1440, 1366, 1280, 900, 820, 650 and 450 across the review and regression
audits. Native 200% text-only zoom was tested separately from viewport zoom.
Firefox's environment clamps narrow windows at 450px; narrower physical devices,
Safari/Chromium and screen-reader speech output remain untested. This is scoped
acceptance, without a claim of formal WCAG certification or biological validation.

## Archived prototype review

# Source-linked evidence network: scene 08 design QA

final result: passed

Local visual and interaction acceptance, 26 September 2026. No actionable P0/P1/P2 findings remain within the inspected scope. Biological validation, external-use clearance and public deployment remain separate approvals.

## Comparison and viewing conditions

- Established visual language: scene 07 at `results/reports/mvo_landing_editorial_20260926/qa/browser/landing-desktop-final.png`.
- Implementation: <http://127.0.0.1:8858/#scene-network>.
- Combined, inspected comparison: `results/reports/mvo_network_explorer_20260926/qa/design-comparison.png`, generated from the adjacent HTML comparison file.
- Both source screenshots are 1440 CSS pixels wide, DPR 1, headless Firefox and reduced motion. Scene 07 is 1024px high; the new network capture is 1100px high. In the combined comparison both screenshots are proportionally scaled to the same width. Browser chrome is excluded.
- The target is coherent integration into the existing landing design. The new network has different content and geometry from the preceding illustrative scene. It is an analytical interaction, with real source data and native interface elements.
- Full-resolution inspection: `network-desktop-final.png`, `network-source-assertions.png`, `network-immersive.png`, `network-450-overview.png` and `network-450-genes.png`. Additional screenshots cover alternatives, source chemistry, RNA and tablet layout.

## Findings and fixes

1. [P2, fixed] Screen-reader-only labels and live announcements were initially visible inside the scene. Added a scoped visually hidden utility. The final desktop and narrow captures show clean controls while retaining accessible text.
2. [P2, fixed] Long canonical-object labels crowded adjacent source nodes. Increased per-row spacing and bounded secondary labels; exact values remain available in the inspector. The recaptured source-assertions view has distinct, readable cards.
3. [P2, fixed] The scroll rail could identify the following scene while the long explorer still occupied the viewport. Active-scene selection now uses the section spanning the viewport midpoint. The browser check confirms scene 08 remains selected.
4. [P2, fixed] Selecting a leaf on a later RNA page could reset the neighborhood to page one. Leaf selection now retains its containing page; the browser check follows a second-page cell through to its canonical RNA record.
5. [Test-harness correction] The original two-digit rail assertion inspected rendered text while its label was intentionally opacity-hidden. Inspecting the element's text content confirms the existing value `10`. This was a QA selector correction; the final rail numbering and scene behavior are tested independently.

## Design and interaction acceptance

- **Narrative:** application example, evidence interrogation, then field-validation pathway. All three existing case dialogs hand off to their matching network case. The preceding illustration remains intact.
- **Typography and color:** existing display, body and monospace families; near-black canvas, teal source evidence, amber review, grey unresolved context and violet next action. Text duplicates color meaning.
- **Layout:** a bounded focus neighborhood with a persistent inspector on desktop, stacked details and a jump link on narrow screens. The review profile remains visible above the graph. Native immersive dialog removes surrounding-page distraction.
- **Data meaning:** qualitative review status at the center; explicit navigation relationships; a separate canonical assertion mode. Geometry conveys reading structure. Source units, missingness, alternative explanations and review restrictions remain inspectable.
- **Interaction:** case switch, theme expansion, source selection, pagination, search and empty state, network/records views, reset, breadcrumb, zoom/fit, pointer pan and immersive close. Source-loading error recovery was tested with one deliberately injected HTTP 503.
- **Accessibility:** semantic controls, keyboard focus retention, Escape dismissal, focus return, visible focus, status labels and reduced-motion support. This is functional QA, without a claim of formal accessibility certification.
- **Preservation:** atlas data and original rendering assets remain intact; the dated evidence extension retains its own source scope. Public report links and archived bundles are unchanged.

## Evidence and residual scope

The latest `qa/browser-checks.json` records 60 passing network assertions. Existing-landing regression checks are retained under `qa/landing-regression/`. Source/preservation results are in `results/reports/mvo_network_explorer_20260926/validation.json`; unit contracts cover source identity, record grain, missingness, status and canonical references.

Inspected actual CSS viewports: 1440 × 1100, 900 × 1000 and 450 × 950. The user-approved Firefox/Selenium workflow completed despite an installed-driver compatibility advisory. Chromium, Safari, physical-device widths below 450px, enlarged-text layouts, assistive technology and independent scientific/partner review remain open. No public release was performed.

## Archived scene 07 acceptance

The acceptance below records the earlier application-scene revision. Its source bundle and QA evidence remain preserved.

# Application-led scene 07 — design QA

final result: passed

No actionable P0/P1/P2 findings remain in the inspected scope. This is local visual/interaction acceptance, not scientific validation or publication approval.

## Comparison target and evidence

- Source visual truth: `results/reports/mvo_blue_carbon_applications_20260926/concepts/selected-concept-2-revised.png` (repository-relative).
- Implementation: <http://127.0.0.1:8858/#scene-platform>.
- Screenshot: `results/reports/mvo_blue_carbon_applications_20260926/qa/browser/landing-desktop-final.png`.
- Combined comparison: `results/reports/mvo_blue_carbon_applications_20260926/qa/reference-vs-implementation.png`.
- Desktop CSS viewport and screenshot: 1440 × 1024, 1 CSS pixel per screenshot pixel. Source pixels: 1487 × 1058. The source is normalized proportionally to 1440px wide; its normalized height is 1025px. There is no browser chrome or device frame in either comparison.
- State: scene 07 visible, dark theme, dialog closed, reduced motion. Original live-site chrome is intentionally retained.
- Additional implementation inspection: open dialog for each case; 900 × 900 and 450 × 844 actual narrow viewports; companion at 1440 × 1024, 768 × 1024 and 450 × 844. Firefox's minimum width prevented a true 390px screenshot; filenames retain the request, while receipts report actual dimensions.
- Focused evidence: individual full-resolution desktop headline/workflow capture; `landing-case-locate.png` for numerical table/labels and `landing-390-dialog.png` for narrow-screen wrapping and controls. The combined full-view at original resolution makes the principal hierarchy, labels and artwork legible; these additional views inspect details independently rather than claiming a pixel-identical modal reference.

## Findings and comparison history

1. [P2, fixed] The first dialog capture was aligned to the top-left because the global reset removed native dialog margins. Added scoped `margin:auto` and a theme-compatible scrollbar. Post-fix `landing-case-locate.png` shows a centered, bounded dialog; the browser assertion checks horizontal centering, Escape dismissal and focus restoration.
2. [P2, fixed] The first companion render lacked chart selectors and an authored report frame. Replaced the unsupported `controls` prop with `headerControls`; set the public authored-report geometry. Post-fix desktop/narrow captures and selector-switching tests pass. `explorer-initial-checks.json` retains the original failure receipt.
3. [P2, fixed] The first implementation lacked the reference's label leaders, weakening association between labels and artwork. Added restrained desktop leader rules to the three labels, disabled for the separate narrow-screen composition. Re-captured and compared against the reference; labels now identify the specimen, molecular inset and chamber without implying a measured data link.

## Required fidelity surfaces

- **Typography:** existing landing display/body/monospace families retained; live HTML rather than rasterized text. Strong headline hierarchy, orange emphasis, readable body and secondary labels. Deliberate wrapping adapts to the original site chrome and explicit evidence-cutoff note.
- **Spacing/layout:** retained left editorial field/right illustration split, ordered workflow labels and bottom available-now/next-test strip. Dialog centering fixed; no document-level horizontal overflow in tested states. The compact existing-site button shape is an intentional design-system choice rather than the reference's pill shape.
- **Colors/tokens:** near-black background, turquoise actionable/evidence labels, orange headline emphasis and amber unresolved pairing. No success-green risk or credit badge.
- **Image quality:** generated art follows the selected mangrove/root/core/chamber direction; no custom CSS/SVG substitute for the scene. Labels are separate HTML. The conceptual illustration is explicitly labeled and is not presented as a real sampled site.
- **Copy/content:** headline speaks to the application. Real examples retain their actual freshwater/mudflat settings. Present evidence review and future predictive testing are separate. Original August atlas metrics remain in the unchanged site chrome; September examples are labeled separately.
- **Interaction/accessibility:** semantic buttons/selects/dialog, labels/alt text, visible keyboard focus, Escape dismissal, focus return, three actual case states and source-table inspection. No JavaScript or console errors recorded. No formal WCAG or assistive-technology certification is asserted.

## Implementation checklist

- [x] Refined selected concept 2, not a replacement visual direction.
- [x] Integrate into existing scene/navigation rather than a new disconnected site.
- [x] Separate illustrative artwork from source-derived quantitative figures.
- [x] Exercise evidence dialog, existing atlas candidate/projection controls and companion source inspectors.
- [x] Verify desktop/tablet/narrow layouts and retain original release/alias context.
- [x] Recompare after visual fixes.
- [x] Keep public deployment pending.

## Residual test gaps / follow-up polish

True 390px physical-device rendering, Safari/Chromium compatibility, screen-reader behavior, full network-failure injection and enlarged-text accessibility deserve later testing. The present checks used user-approved Firefox/Selenium because the in-app browser and agent-browser were unavailable. The geckodriver version emitted a compatibility advisory but all recorded workflows completed. No claim of broad cross-browser production certification is made.
