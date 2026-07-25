# DIGEST — verified ground truth for the EmergentBiome / MethaNet landing page

This file is the reconciliation between the build spec's *expected* numbers and the
**live repository snapshot**. Everything the page asserts traces back to here, and
here traces back to the repo artifacts cited in each row. Read this before editing
`config.js`.

- **Snapshot date:** 2026-07-24
- **Authoritative report snapshot:** `2026-07-24 scientific reconciliation release`
- **Payload-freeze audit:** 7,709 release-required rows; 7,484 data-complete tri-views;
  one incomplete MSM unit preserved as a release exclusion
- **Primary sources digested:** `README.md`, `AGENTS.md`,
  `docs/methanet_positioning_and_claims.md`, `docs/current_artifact_inventory.md`,
  `ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md`,
  `ai_docs/.../queryable_molecular_attestation_moat/methanet_queryable_molecular_attestation_moat.md`,
  `configs/methanet_atlas_lanes.tsv`,
  `results/reports/methanet_3view_payload_freeze_20260724_scientific_reconciliation/freeze_summary.tsv`,
  `results/reports/mbag_nextgen_molecular_niche_atlas_20260724_scientific_reconciliation/report_bundle_manifest.json`,
  `results/functional_metagenomics/mucc_v1_owc_wetland_20260626/reports/atlas_lane_registry_summary.md`,
  `results/functional_metagenomics/mucc_v1_owc_wetland_20260626/reports/mucc_v1_integration_completion_audit.json`.

### Public terminology contract

- **MBAG** means MethaNet Bridge Attestation Graph. It is the knowledge graph
  that connects each molecular evidence unit to representations, mechanisms,
  provenance, quality gates, claim scope, and validation actions.
- **MAG** means metagenome-assembled genome.
- **ESM-2** names the protein-language-model view. **gLM2** names the
  genome-context language-model view.
- **Tri-view** means ESM-2, gLM2, and a functional payload are present.
  Mechanism comparability remains a separate evidence state.
- **MRV** means monitoring, reporting, and verification. Calibrated MRV remains
  downstream of sample linkage, abundance, environment, uncertainty, and
  field or process validation.
- **MUCC v1** identifies the Old Woman Creek wetland source warehouse.
  **VM0033** identifies Verra's tidal-wetland and seagrass methodology context.

---

## 1. Headline numbers (verified) — and corrections to the spec's expected values

| Quantity | Spec expected | **Verified live** | Notes / source |
| --- | --- | --- | --- |
| Calibration core (rumen + wetland POC) | 625 MAGs | **625** ✓ | Consolidated DuckDB warehouse, 625/625 tri-view, 711 validation gates pass. `cohort_warehouse_poc_magbin_union_20260616_075022` |
| Mangrove / MSM China 2025 candidates | 1,428 | **1,428** ✓ | ESM2 1,428/1,428; gLM2 1,428/1,428; functional **1,427/1,428** (99.93%, 1 partial) at snapshot |
| Mangrove / Futian 2026 (Qi et al.) rMAGs | 3,404 | **3,404** ✓ | Phase-1 dereplicated at 99% ANI. 3,156 ready payload + 248 explicit gap rows |
| Futian ESM2 / gLM2 | complete | **3,156 / 3,156 both** ✓ | Across four shards |
| Futian functional | in progress | **312/312 archaea + 2,619/2,844 bacteria complete** | 2,931/3,156 ready payload rows have annotation-complete outputs; common mechanism-feature aggregation remains pending |
| Data-complete tri-view units | 2,317 | **7,484** | 625 mechanism-comparable POC + 4,358 annotation-complete/harmonization-pending mangrove + 2,501 MUCC source-scaffold |
| ESM-2 molecular space | ~5,200 | **7,710 embedding-bearing units** | 625 POC + 1,428 MSM + 3,156 Futian + 2,501 MUCC v1 |
| MUCC v1 OWC wetland warehouse | not in original page | **2,508 wetland genomes** ✓ | 2,501 ESM-2; 2,508 multiwindow gLM2; 2,508 source-functional scaffold; 2,501 data-complete tri-view |
| Foundation models | ESM2 + gLM2 | **ESM2 (650M, `esm2_t33_650M_UR50D`) + gLM2 (650M)** ✓ | |
| Speed | <4 days vs 6–12 mo | **<4 days (calibration run)** ✓ | Raw sequences → consolidated reports for the calibration core |
| Paired CH₄-flux samples | 23 → 80–100 | **23 targeted → 80–100 target** ✓ | README validation strategy |
| Benchmark | mcrA/pmoA R²=0.879 (Lee 2014, freshwater) | **as a literature target, not yet measured here** ✓ | Honest boundary: to be validated in saline systems |
| Field validation site | Cispatá Bay | **Cispatá Bay, Colombian Caribbean mangrove** ✓ | Conservation International context; VM0033 methodology family |

**The single most important update is the three-state functional contract.** `config.js` and the
published report use 7,484 only as the data-complete tri-view total. The 625-unit POC core is the
current mechanism-comparable set. MSM/Futian contribute 4,358 real annotation-complete tri-views
whose report-level mechanism features must be rebuilt from one common accepted/present contract.
MUCC contributes 2,501 data-complete source-scaffold tri-views plus processed expression detection;
its DRAM/gene/expression evidence is not numerically equated with either other contract.

### ESM2 POC geometry (662-genome cohort, real statistics)
- Cohort: **662 genomes** = 555 rumen (PRJEB31266) + 107 wetland/MUCC; embedding 662 × 1,280 (ESM2-650M, layer 33, mean-pooled).
- **PERMANOVA R² = 0.202 (p = 0.001)** — ecosystem explains 20.2% of embedding variance.
- **Silhouette = 0.398** [95% CI 0.364–0.439].
- **CV classifier AUC = 1.000**, balanced accuracy 0.999 (5-fold, PCA-50).
- **Cohen's d = 3.63** on the trajectory axis.
- **14 bridge genomes** with ≥1 opposite-ecosystem k-NN neighbor (of 662). All top 11 alpha-transfer candidates are **rumen Archaea** (conserved methanogenesis machinery: mcrA, HdrABC). Top bridge `bin.8` (Archaea).
- **P0 caveat (stated honestly on-page):** source and ecosystem are perfectly confounded (all rumen = PRJEB31266, all wetland = MUCC). Deconfounding is the top next step.

### Molecular attestation graph (MMAG MVP, `results/attestation/mmag_mvp_20260617/`)
- **662 MAG nodes · 3,968 evidence atoms · 2,644 feature nodes · 397 taxa · 13 artifacts · 8 validation-gap nodes · 5 claim nodes · 2 source domains.**
- **9,930 `NEAR_IN_ESM2_SPACE` edges.**
- Readiness: 437 `molecular_attestation_ready_not_mrv`, 188 `…_with_qc_caveat`, 37 `blocked_noncomparable_unit`.

---

## 2. Ecosystem lanes (all four registered lanes represented)

| Lane | Role | Units (embedding-bearing) | Status |
| --- | --- | --- | --- |
| `rumen` (POC) | source reference (methane-system neighborhoods) | 518 MAG/bin in POC denominator (555 ESM2 in 662 backbone) | complete, source-confounded |
| `wetland` / MUCC (POC) | target-domain wetland | 107 | complete (MAG-level) |
| `mangrove_msm` (MSM China 2025) | blue-carbon target expansion | 1,428 | ESM2+gLM2 complete; functional 1,427/1,428 |
| `mangrove_futian` (Futian 2026, Qi et al.) | larger time/depth/habitat mangrove target | 3,156 ready (+248 gap) | ESM2+gLM2 complete; annotation-complete 2,931/3,156; common feature aggregation pending |
| `mucc_v1_owc_wetland` | Old Woman Creek wetland molecular-reference lane | 2,501 embedded (+7 non-embedded archive rows) | ESM2 2,501/2,508; gLM2 2,508/2,508; source-functional 2,508/2,508; explicitly not canonical mechanism-equivalent |

---

## 3. Bridge-taxa result (what the hero encodes)

- **POC scale:** 14 cross-ecosystem bridge genomes; top alpha-transfer candidates all rumen Archaea — methanogen machinery is the conserved corridor.
- **Expanded public export:** **2,226 cross-domain k-NN/case-study links** among **930 linked nodes** and **36 case-study candidates** in the 7,710-point ESM-2 map.
- These edges are **provisional bridge hypotheses**, not proof of rumen→wetland transfer (source confounding stands).

---

## 4. Non-negotiable claim boundaries (must be visible, not buried)

1. Current results = **molecular screening + monitoring prioritization**, at MAG/proteome grain.
2. **A–E risk tiers are TARGET product vocabulary — explicitly "not yet calibrated."**
3. From the molecular atlas alone, MethaNet does **not** claim: measured methane flux · final
   sample/project MRV risk scores · final A–E tiers · source-independent rumen→wetland transfer
   proof · carbon-credit approval.
4. Maturity ladder: **Level 0 (molecular screening) = complete/lit; Level 1 (sample identity &
   metadata) = in progress** (sample/site/season metadata recovered for the mangrove lanes — 147
   sediment samples + 71 environmental rows — but full genome-to-sample mapping is not done).
   Calibrated probabilistic MRV risk (Level 5) still requires abundance → environmental covariates
   → flux/process validation → calibration. We do not claim anything beyond rung 1, and say so.

Maturity ladder (roadmap Levels 0–6; page shows rungs 0–5 with Level 6 as the registry horizon):
`0 MAG molecular screening (LIT NOW)` → `1 sample identity/metadata` → `2 abundance/community
capacity` → `3 environmental permissiveness` → `4 flux/process validation` → `5 calibrated
probabilistic MRV risk (TARGET)` → `(6 MRV product & audit / registry integration)`.

---

## 5. Phase 1 data export — which option was used

**Option (1): DIFFUSION MAP coordinates of the proteome embeddings — REAL.**
The hero uses the **diffusion map** as its 2D backbone (the report's primary niche-space view),
because the bridge structure reads most clearly there: the gold "nearest-reference" links fan from
the mangrove line back to specific rumen reference genomes.

**Evidence-reconciled recompute (2026-07-24):** the atlas and report were regenerated end-to-end from all
four registered lanes. Pipeline: `build_methanet_3view_payload_freeze.py` →
`methanet_3view_payload_freeze_20260724_scientific_reconciliation` →
`build_mbag_nextgen_molecular_niche_atlas.py` →
`mbag_nextgen_molecular_niche_atlas_20260724_scientific_reconciliation`. The payload audit preserves
one partial MSM row as an explicit release exclusion and retains all other incomplete/source-gap rows.

`tools/export_atlas.py` reads the recomputed niche projection
`results/reports/mbag_nextgen_molecular_niche_atlas_20260724_scientific_reconciliation/assets/data/niche.json`
and writes `data/atlas.json`:

- **7,710 real embedding-bearing points** (625 POC + 1,428 MSM + 3,156 Futian + 2,501 MUCC v1).
  The 255 non-embedded source-gap rows are acknowledged in `meta.excluded_gap_rows`.
- Primary `x,y` = **diffusion map**, scaled per-axis by a **linear min-max** (0.3/99.7 clip) so the
  full 7,710-point graph remains legible. UMAP (`hx,hy`) and PCA (`px,py`) are exported as real
  projection-sensitivity views.
- Per point: ecosystem, domain, `br` (documented bridge node), `cs` (case-study nearest-reference),
  `fc` (functional-contract state), `ma` (POC-only internal attestation score), `mz` (POC-only
  curated methane-marker density), and `nps` (mangrove nearest-POC similarity). `ma`/`mz` are null
  outside the comparable POC contract; null is not interpreted as biological absence.
- **2,226 bridge edges** as `{s, t, w=cosine similarity, cd=cross-domain, cs=nearest-reference, rk}`.
  In the hero: `cs` links render **gold** ("bridge genome → nearest reference"), the rest render faint
  **teal** ("cross-ecosystem neighbor") — matching the report's legend.
- **Deterministic / seeded:** no RNG, no subsampling — re-running on the pinned source is byte-identical.

The displayed coordinates are deterministic transforms of the report's real diffusion map.
Centroids and edge counts are stored directly in `data/atlas.json`; visual geometry is navigation,
not proof of ecological transfer or methane flux.

**Plain-language note:** the *landing page copy* is deliberately jargon-light for investors — it says
"proteome embeddings / genomic-context embeddings / genomes / molecular map" and foregrounds the
business meaning: candidate triage, monitoring prioritization, and the next validation rung. The
linked *report* remains the technical deep-dive.

---

## 6. Real vs stylized data per scene (for scientific review before any investor sees it)

Scene order re-sequenced 2026-07-01 (decision-first; see `ai_docs/prompts/landing_page_value_prop_tightening_prompt.md`).
The old abstract "Insight" manifold scene was dropped; the Survey was promoted into a concrete
"What You Get" output artifact and a new "Versus the Cheap Method" scene was added, both before the
Atlas (now recast as "The Evidence"). External facts live in `config.js` `ext` with sources; see the
grounding dossier. The hero backdrop (`scenes/hero.js`) still uses the real diffusion-map manifold.

| Scene | Visual | Data basis |
| --- | --- | --- |
| Hero | diffusion-map manifold backdrop; decision-first copy | **REAL coords** (`atlas.json`); copy is a business statement |
| 1 — The Stakes | net-climate-balance line crossing net-zero into warming | **Illustrative** concept; CH₄ ≈80× CO₂ over 20yr + ≈12yr lifetime real (IPCC AR6, labeled); blue-carbon record ≈$29/tCO2e (Platts DBC) |
| 2 — The Blind Spot | grey unknown field; scan finds nothing; 0-of-80–100 coverage row | **Illustrative** scale; honest present state **≈ 0** sites paired; VM0033 no-default-below-18-ppt is real (source in `ext`) |
| 3 — What You Get | ranked site-triage list (signal, confidence, evidence link, measure-here) + two-column what-it-is/is-not | **Illustrative product shape** (exact output object; per-site scores are a mock; ranking follows the real science: risk concentrates in brackish/freshened/restored) |
| 4 — Versus the Cheap Method | salinity-vs-methane field; exceptions the community flags; method-resolution legend | **Illustrative teaching plot**; the molecular anchor is real: mcrA vs measured flux Spearman r > 0.7 (Baker et al. 2022, ISME J) |
| 5 — The Evidence (atlas) | particles; counters show 7,484 data-complete tri-views under three evidence contracts | **Real counts and coordinates**; motion stylized |
| 6 — One Engine, Many Maps | one atlas re-read by a clickable gas "lens"; methane lights only the 625-unit comparable POC core, while other lanes and N₂O/sulfur remain explicitly unweighted | **REAL coords** (diffusion view of `atlas.json`); POC methane density is within-contract screening only; no measured/validated flux |
| 7 — Platform & Moat | attestation graph; one claim's evidence chain | **Real schema + counts** (MMAG graph); future slots = honest empty optionality |
| 8 — Honest Ladder | 6-rung maturity gauge; rung 0 lit, reframed as the product sold today | **Real** (MRV roadmap Levels 0–5) |
| 9 — Path & Ask | Cispatá Bay pin; buyer set; <4-day pipeline | **Real** geography + roadmap + speed; the ask is a business statement |

**One-paragraph honesty summary:** The atlas-backed scenes (Hero, 5 The Evidence, 6 One Engine) are
driven by real coordinates in `data/atlas.json` and verified repository counts (7,484 data-complete tri-view
units, 7,710 mapped in ESM-2 space, 2,508 MUCC wetland source-warehouse genomes); scenes 7–9 use the real molecular-attestation
graph schema and the published MRV maturity ladder. Scenes 1, 2, 3, and 4 are deliberately illustrative
and badged so: the climate concept, the measurement gap (honest present state **≈ 0** sites with paired
methane-flux + molecular data), the exact *shape* of the product output (per-site scores mocked, the
ranking following real biogeochemistry), and a teaching plot of the salinity-vs-community argument. Every
external fact carries a source in `config.js` `ext`; every A–E risk claim is marked TARGET/not-yet-calibrated;
a persistent claim-boundary bar and Scene 8 state the molecular-screening boundary with the 2026-07-24 snapshot.
