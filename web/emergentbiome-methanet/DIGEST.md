# DIGEST — verified ground truth for the EmergentBiome / MethaNet landing page

This file is the reconciliation between the build spec's *expected* numbers and the
**live repository snapshot**. Everything the page asserts traces back to here, and
here traces back to the repo artifacts cited in each row. Read this before editing
`config.js`.

- **Snapshot date:** 2026-06-27
- **Authoritative live snapshot:** `2026-06-27 16:37 UTC` (lane registry / per-MAG sentinels)
- **Report-freeze snapshot:** `2026-06-25 14:55:09 UTC` (the dated freeze the external HTML atlas is built from)
- **Primary sources digested:** `README.md`, `AGENTS.md`, `docs/current_artifact_inventory.md`,
  `ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md`,
  `ai_docs/.../queryable_molecular_attestation_moat/methanet_queryable_molecular_attestation_moat.md`,
  `configs/methanet_atlas_lanes.tsv`,
  `results/reports/atlas_lane_registry_status_20260625_145509.md`,
  `results/reports/methanet_3view_payload_freeze_20260625_145509/freeze_summary.tsv`,
  `results/functional_metagenomics/mucc_v1_owc_wetland_20260626/reports/atlas_lane_registry_summary.md`,
  `results/functional_metagenomics/mucc_v1_owc_wetland_20260626/reports/mucc_v1_integration_completion_audit.json`.

---

## 1. Headline numbers (verified) — and corrections to the spec's expected values

| Quantity | Spec expected | **Verified live** | Notes / source |
| --- | --- | --- | --- |
| Calibration core (rumen + wetland POC) | 625 MAGs | **625** ✓ | Consolidated DuckDB warehouse, 625/625 tri-view, 711 validation gates pass. `cohort_warehouse_poc_magbin_union_20260616_075022` |
| Mangrove / MSM China 2025 candidates | 1,428 | **1,428** ✓ | ESM2 1,428/1,428; gLM2 1,428/1,428; functional **1,427/1,428** (99.93%, 1 partial) at snapshot |
| Mangrove / Futian 2026 (Qi et al.) rMAGs | 3,404 | **3,404** ✓ | Phase-1 dereplicated at 99% ANI. 3,156 ready payload + 248 explicit gap rows |
| Futian ESM2 / gLM2 | complete | **3,156 / 3,156 both** ✓ | Across four shards |
| Futian functional (archaea-first) | in progress | **312/312 archaea complete**; bacteria **0/2,844 queued** | IN PROGRESS. Registry summary generated 2026-06-27 |
| Atlas units mapped | 2,317 | **2,364 live** ⚠️ **UPDATED** | 625 POC + 1,427 MSM + **312** Futian |
| Scaling target | ~5,200 | **5,209 embedding-bearing units** ✓ | 625 + 1,428 + 3,156. (5,457 plotted incl. 248 gap rows) |
| MUCC v1 OWC wetland warehouse | not in original page | **2,508 wetland genomes** ✓ | Queryable source-scaffold warehouse; 1,948 genomes with expression support across 133 samples; not yet plotted in the manifold |
| Foundation models | ESM2 + gLM2 | **ESM2 (650M, `esm2_t33_650M_UR50D`) + gLM2 (650M)** ✓ | |
| Speed | <4 days vs 6–12 mo | **<4 days (calibration run)** ✓ | Raw sequences → consolidated reports for the calibration core |
| Paired CH₄-flux samples | 23 → 80–100 | **23 targeted → 80–100 target** ✓ | README validation strategy |
| Benchmark | mcrA/pmoA R²=0.879 (Lee 2014, freshwater) | **as a literature target, not yet measured here** ✓ | Honest boundary: to be validated in saline systems |
| Field validation site | Cispatá Bay | **Cispatá Bay, Colombian Caribbean mangrove** ✓ | Conservation International context; VM0033 methodology family |

**The single most important update: the landing page now separates what is mapped from what is
newly queryable.** `config.js` uses 2,364 as the live tri-view atlas count and calls out the new
2,508-genome MUCC wetland warehouse as the next source-scaffold expansion, without claiming those
genomes are already plotted in the manifold.

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

## 2. Ecosystem lanes (four embedding-bearing lanes + one staged)

| Lane | Role | Units (embedding-bearing) | Status |
| --- | --- | --- | --- |
| `rumen` (POC) | source reference (methane-system neighborhoods) | 518 MAG/bin in POC denominator (555 ESM2 in 662 backbone) | complete, source-confounded |
| `wetland` / MUCC (POC) | target-domain wetland | 107 | complete (MAG-level) |
| `mangrove_msm` (MSM China 2025) | blue-carbon target expansion | 1,428 | ESM2+gLM2 complete; functional 1,427/1,428 |
| `mangrove_futian` (Futian 2026, Qi et al.) | larger time/depth/habitat mangrove target | 3,156 ready (+248 gap) | ESM2+gLM2 complete; archaea functional 312/312 complete at the live registry snapshot; **bacteria queued** |

> A fifth lane exists in the registry — `mucc_v1_owc_wetland` (MUCC v1 OWC, 2,508 local FASTA
> entries versus the published/deposit 2,502 HQ/MQ headline denominator). It now has a queryable
> source-scaffold warehouse and expression support, but **no completed MUCC embeddings yet**. It is
> **not** plotted in the atlas and not counted in the 5,209 embedding-bearing units. Mentioned only
> as honest pipeline expansion until ESM2/gLM2 outputs complete.

---

## 3. Bridge-taxa result (what the hero encodes)

- **POC scale:** 14 cross-ecosystem bridge genomes; top alpha-transfer candidates all rumen Archaea — methanogen machinery is the conserved corridor.
- **Expanded scale (atlas):** **372 documented cross-domain k-NN bridge edges** among **136 bridge nodes**; 356 cross-domain k-NN + 16 case-study nearest-POC links; **26 case-study candidates** (16 mangrove, 9 rumen, 1 wetland). Mangrove→rumen nearest-POC cosine similarities reach **0.976–1.000** (median high-0.98) — tight cross-ecosystem proximity in ESM2 space.
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

**Interim recompute (2026-06-29):** the Atlas and report were regenerated end-to-end from the
current warehouse (Futian archaea 312/312) so the landing, `atlas.json`, and `report.html` all read
the same **2,364**. Pipeline: `build_methanet_3view_payload_freeze.py` (3-lane registry, 1 MSM
exclusion) → freeze `methanet_3view_payload_freeze_20260629_195927` (2,364) →
`build_mbag_nextgen_molecular_niche_atlas.py` → report
`mbag_nextgen_molecular_niche_atlas_20260629_interim_2364`. The diffusion geometry is **byte-identical**
to the prior export (same embeddings → deterministic diffusion map; max per-point L1 drift 0.0); only
the newly-functional archaea gained functional attributes and the tri-view count rose 2,352 → 2,364.
The final Atlas will be re-run once the full warehouse (incl. Futian bacteria) completes.

`tools/export_atlas.py` reads the recomputed niche projection
`results/reports/mbag_nextgen_molecular_niche_atlas_20260629_interim_2364/assets/data/niche.json`
and writes `data/atlas.json`:

- **5,209 real embedding-bearing points** (625 POC + 1,428 MSM + 3,156 Futian; 248 non-embedded
  Futian gap rows excluded). The 248 gap rows are acknowledged in `meta.excluded_gap_rows`.
- Primary `x,y` = **diffusion map**, scaled per-axis by a **linear min-max** (0.3/99.7 clip) so the
  report's fan structure is faithfully reproduced: rumen and wetland references form fans on the
  **left**, the mangrove expansion forms the **line on the right**. PHATE (`hx,hy`) and PCA (`px,py`)
  are exported full-cohort as real projection-sensitivity toggles.
- Per point: ecosystem, domain, `br` (documented bridge node), `cs` (case-study nearest-reference),
  `ma` (attestation index), `mz` (methane-marker density), `nps` (mangrove nearest-POC similarity).
- **372 bridge edges** as `{s, t, w=cosine similarity, cd=cross-domain, cs=nearest-reference, rk}`.
  In the hero: `cs` links render **gold** ("bridge genome → nearest reference"), the rest render faint
  **teal** ("cross-ecosystem neighbor") — matching the report's legend.
- **Deterministic / seeded:** no RNG, no subsampling — re-running on the pinned source is byte-identical.

Real diffusion basins (display coords): rumen ≈ (−0.87, +0.72), wetland ≈ (−0.84, −0.77),
mangrove ≈ (+0.40, +0.45). The POC calibration core fans out on the left; the mangrove expansion
is the line on the right; the 372 real bridge links span between them.

**Plain-language note:** the *landing page copy* is deliberately jargon-light for investors — it says
"proteome embeddings / genomic-context embeddings / genomes / molecular map" and foregrounds the
business meaning: candidate triage, monitoring prioritization, and the next validation rung. The
linked *report* remains the technical deep-dive.

---

## 6. Real vs stylized data per scene (for scientific review before any investor sees it)

| Scene | Visual | Data basis |
| --- | --- | --- |
| 1 — The Stakes | CO₂ settling / CH₄ eroding a net-benefit bar | **Illustrative** (concept; CH₄ GWP≈30× CO₂/100yr is real, labeled) |
| 2 — The Blind Spot | grey unknown-genome field; scan finds nothing | **Illustrative** scale; the honest present state shown is **≈ 0** sites with paired flux + molecular data today (no fabricated "23") |
| 3 — The Insight (HERO) | diffusion-map manifold; gold bridge links | **REAL coordinates** (`atlas.json`, diffusion map; PHATE/PCA also exported) + **real 372 bridge links** |
| 4 — The Atlas | particles pour in; counters 625→2,364; MUCC wetland warehouse callout | **Real counts** (mapped lanes plus MUCC warehouse); particle motion stylized |
| 5 — Platform & Moat | attestation graph; one claim's evidence chain | **Real schema + counts** (MMAG graph nodes/edges/claims/gaps); future slots = honest empty optionality |
| 6 — Honest Ladder | 6-rung maturity gauge; rung 0 lit | **Real** (MRV roadmap Levels 0–5; rung 0 = current) |
| 7 — Path & Ask | Cispatá Bay pin; milestones; <4-day pipeline | **Real** geography + roadmap milestones + pipeline speed; the ask is a business statement |

**One-paragraph honesty summary:** The hero (Scene 3) is driven entirely by real diffusion-map
coordinates of the proteome embeddings and the real 372 bridge links exported in `data/atlas.json`;
nothing in it is invented. Scenes 4–7 are built from verified repository counts (2,364 fully-mapped
genomes, 5,209 mapped, 2,508 wetland-warehouse genomes), the real molecular-attestation graph schema,
and the published MRV maturity ladder, with generative motion used to *present* (not fabricate) those
quantities. Scenes 1–2 are deliberately illustrative and badged so: the climate concept, and the
measurement gap whose honest present state is **≈ 0** sites with paired methane-flux + molecular data
(the earlier "23" was a curation target, not a measurement, and has been removed). The 80–100 paired
samples figure appears only as a stated **target**. Every A–E risk claim is marked
TARGET/not-yet-calibrated, and a persistent footer plus Scene 6 state the molecular-screening claim
boundary with the 2026-06-27 snapshot date.
