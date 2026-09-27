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
