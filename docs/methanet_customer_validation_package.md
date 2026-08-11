# MethaNet Customer Validation and Traction Package

> **Status: 2026-08-10 controlled-diligence validation design; not customer-validated.**
>
> This package preserves customer-discovery questions and proposed pilot
> designs. The August 10 release has 7,965 registered units, 7,710
> data-complete and schema-normalized tri-views, 5,209 pipeline-normalized
> tri-views with cross-lane comparability pending, 2,501 MUCC source-scaffold
> tri-views, and **zero** mechanism-comparable, sample-linked, field-validated,
> or calibrated units. Detailed predicted-versus-observed demonstrations below
> are unexecuted designs, not current results. In particular, processed
> expression is detection/occupancy support, not validated activity magnitude,
> and the current metadata warehouse has zero authoritative exact
> sample/environment/process joins. Current
> product language, release counts, and claim eligibility are governed by
> [`methanet_positioning_and_claims.md`](methanet_positioning_and_claims.md),
> [`current_artifact_inventory.md`](current_artifact_inventory.md), and the
> [public MBAG report](https://emergentbiome.earth/report/).

## Executive Framing

What MethaNet can show today is a source-audited, four-lane molecular evidence ledger, validated functional warehouses, explicit evidence contracts, metadata-rich context portfolios, and reviewable evidence-card records with blockers. The prior POC geometry result remains a source-confounded structure-recovery diagnostic, not a methane metric. MethaNet does not yet predict measured methane flux, produce a calibrated sample/project risk score, assign final A-E tiers, prove source-independent transfer, or support registry or carbon-credit certification. Functional annotation is mechanism-relevant potential; processed expression is detection/occupancy support, not validated activity or flux. The next validation target is an exact molecular/sample/abundance/environment/process join followed by source-aware held-out analysis.

Three capability tiers recur throughout and are kept distinct everywhere:

- **(A) Methane flux prediction (continuous):** predicting an actual methane emission rate. Aspirational and currently blocked. Requires a paired molecular, environmental, and measured-flux calibration layer that does not yet exist inside MethaNet.
- **(B) Methane-risk classification (ranking / binary):** flagging and ranking sites where a material methane liability is more plausible. The credible near-term target. Validatable against proxies and held-out labels once a flux or activity label is joined.
- **(C) Microbial fingerprint discovery:** identifying which MAGs and functions co-occur with methane-relevant contexts, delivered as reviewable candidate cards with explicit blockers. Supported today as hypothesis generation.

---

## 1. Core validation question

### The single customer-legible question

A carbon-project developer, credit buyer, insurer, or MRV actor does not begin by asking for a flux number. They ask a decision question about where their existing accounting is exposed:

> **"Which of my sites are most likely to be hidden methane emitters that conventional above-ground carbon accounting is missing, and how confident are you in that ranking?"**

This is a screening-and-prioritization question, not a measurement question. It is answerable with molecular evidence plus a defensible label, and it maps directly to a management action: where to send the next field crew, chamber, or flux tower.

### Candidate framings, most-honest-today to most-aspirational

| Rank | Framing | Capability tier | Honesty status today |
|---|---|---|---|
| 1 | "Which MAGs and functions co-occur with methane-relevant contexts, and what evidence and blockers attach to each candidate?" | (C) Fingerprint discovery | Supported now. Delivered as reviewable candidate cards. No calibration required. |
| 2 | "Can we flag sites where conventional above-ground carbon accounting is likely missing a material methane liability, and rank sites by that risk?" | (B) Risk classification | Near-term. Validatable against activity proxies and held-out labels once a flux/activity label is joined. No calibrated tiers exist yet. |
| 3 | "What is the net methane-emitter probability for this specific sample, with a calibrated risk tier?" | (B) Risk classification, sample-level | Planned. Requires the Sample Risk Readiness primitive (specified, not built) plus abundance mapping and environmental covariates. |
| 4 | "What is the predicted methane flux at this site in physical units, validated against measured flux?" | (A) Flux prediction | Aspirational and blocked. Requires the paired flux calibration layer that does not exist in MethaNet today. |

### Recommended primary question for the early demo

**Framing 1** is the recommended primary question for the current demo:

> **"Which MAGs and functions carry methane-relevant molecular evidence, what contradicts or limits that evidence, and what should be measured next?"**

This is fingerprint discovery and monitoring design (tier C), not a site-risk or flux-prediction claim. Site ranking becomes eligible only after exact sample linkage, abundance, environment, uncertainty, and paired field/process validation are joined and tested.

### Why exact-flux prediction is deferred, not attempted now

The literal "predicted methane flux vs observed methane flux, with precision and AUC" demonstration cannot be produced today with real flux numbers. This is a hard, verified boundary:

- There is **no paired molecular / environmental / flux calibration layer** anywhere in the current MethaNet artifacts. A repository-wide search for measured flux units (nmol CH4, mg CH4, mmol CH4 m-2, umol CH4, g CH4) returns zero data hits.
- Functional annotation in MethaNet is **mechanism-relevant potential**, not expression, activity, or flux. A MAG carrying methanogenesis genes represents molecular capacity, not a measured emission.
- The source and ecosystem are currently **confounded** (rumen from a single project, wetland from a single source), so even a strong molecular signal cannot yet be claimed as source-independent.

Continuous flux prediction is therefore treated as a defined future milestone gated on the flux-join and holdout-validation work described in Section 2, not as a present capability. Attempting a flux demo now would require inventing numbers, which would violate the claim boundary this package is built on.

---

## 2. Predicted vs observed framework

This section defines, precisely, what a prediction is, what an observation is, which target and units apply, which prediction mode is honest today, the minimum sample size for an early demo, and, most importantly, the split between what predicted-vs-observed can be shown right now versus what requires a dataset join.

### Definitions

- **Prediction:** a value the system emits before seeing the outcome variable. Depending on mode, this is a class label (emitter likely / not), a rank position, a probability, or a continuous rate. For the demo, the prediction is a per-site or per-sample **methane-risk tier plus probability**, derived from molecular fingerprints and a methanogenesis-vs-methanotrophy marker balance.
- **Observation:** the outcome variable the prediction is scored against. This is the honesty pivot. Three tiers of observation exist, in ascending order of what they license:
  1. **Held-out label** (available now): the true ecosystem or domain label of a sample, hidden from the unsupervised embedding and recovered from geometry.
  2. **Expression detection** (available now, one lane): processed metatranscriptomic detection/occupancy support, not validated activity magnitude or measured flux.
  3. **Measured methane flux** (not in repo): annual or chamber flux in physical units. Exists in source publications and flux networks, not yet joined into MethaNet.

### Candidate target variables and units

| Target | Units | Mode | Honest today? |
|---|---|---|---|
| Ecosystem / domain label | categorical | classification | Yes. Real result now (structure recovery). |
| Methanogen-activity score (mcrABG-bearing expression minus pmoA/mmoX expression) | expression-derived index | ranking | Yes, on the OWC/MUCC wetland lane only. Activity proxy, not flux. |
| Net-emitter class (high vs low methane) | binary | classification | After flux/label join. |
| Methane-risk rank across sites | ordinal | ranking | After flux/label join. |
| Methane flux | mg CH4 m-2 d-1 or g CH4 m-2 yr-1 | continuous | Aspirational. Blocked until paired flux layer exists. |

### Which mode is honest today

- **Classification** is honest today only for the **label-recovery** target (ecosystem/domain), which is structure recovery, not methane.
- **Ranking** is not authorized today as methane activity or risk; processed OWC expression may support detection/occupancy review only until normalization and exact ecological joins are resolved.
- **Continuous** flux prediction is **not** honest today under any framing.
- Methane-risk classification and ranking (tier B) become honest **after** a flux or emitter label is joined and a held-out split is run.

### Minimum viable sample size for an early demo

- The existing structure-recovery result runs on **40 samples** (5 clusters excluding noise, 10 percent noise fraction). This is adequate to demonstrate that the latent space carries biological structure, not to demonstrate methane skill.
- For a near-term risk-classification demo (tier B), the wetland side is adequate: the Old Woman Creek (OWC) scaffold provides **2,508 wetland MAG rows (2,501 source-protein supported), of which 1,948 are expression-supported across 133 depth- and month-resolved metatranscriptome samples**, giving a genuine held-out split by month or depth. The multi-wetland MUCC source study (Bechtold et al. 2025, Nature Communications 16:944) spans nine freshwater wetlands (seven in the United States, two in northern Sweden; marsh, swamp, fen, and bog) and integrates 16S, metagenomes, metatranscriptomes, and annual methane flux, providing site-level high-vs-low methane contrast usable as classification labels once that flux is joined. The measured flux lives in that external publication and in AmeriFlux/FLUXNET-CH4, not yet ingested into MethaNet.
- A defensible minimum for a first classification readout is roughly a **dozen or more sites** spanning both high- and low-methane contexts with a leave-one-site-out or leave-one-month-out split. Below that, precision and recall estimates are too unstable to report.

### What predicted-vs-observed CAN be shown right now

Two real, reproducible predicted-vs-observed demonstrations exist today, neither of which involves measured flux:

1. **Held-out ecosystem/domain structure recovery.** Unsupervised ESM2 proteome-embedding geometry recovers held-out ecosystem and domain labels at about 99 percent purity: `cluster_purity_ecosystem = 0.990`, `cluster_purity_domain = 0.990`, `silhouette_non_noise = 0.433`, 5 clusters, `noise_fraction = 0.10`, over 40 samples (`results/blue_catalyst_poc/poc_metrics.json`). This is a legitimate predicted-label-vs-observed-label result. It demonstrates that the latent space carries biologically meaningful structure. **It is structure recovery, not methane accuracy, and the 99 percent number must never be presented as a methane metric.**

2. **Candidate-card fingerprint discovery.** Which MAGs and functions co-occur with methane-relevant contexts, delivered as reviewable candidate cards with mechanism class (methane_relevant, substrate_flexible, sulfur_associated, unclear, likely_artifact) and explicit blockers: 100 MRV-readiness candidate cards on the OWC/MUCC lane and 26 candidate cards in the frozen atlas. This is supported today as hypothesis generation (tier C).

The 1,948-MAG by 133-sample OWC expression matrix can support a detection/occupancy sensitivity analysis after normalization is confirmed. It cannot currently support a predicted-activity, methane-risk, or measured-flux result because exact ecological joins remain unresolved.

### What requires the dataset join to show real flux

True methane predicted-vs-observed, with measured flux, requires joining a paired flux dataset that is not currently in the repository:

- The most credible route is the **OWC / MUCC wetland lane**, the only MethaNet lane with a built-in activity proxy. Its published source study (Bechtold et al. 2025, Nature Communications 16:944, DOI 10.1038/s41467-025-56133-0; data MUCC v2.0.0, Zenodo 10.5281/zenodo.14532347) reports annual methane flux for these wetlands and integrates 16S, metagenomes, metatranscriptomes, and flux. That measured flux is **not yet joined into MethaNet**.
- The coastal/mangrove lanes (MSM China 2025; Futian 2026) are molecularly rich but **flux-blocked** and lack per-MAG-to-sample abundance mapping.

**True flux predicted-vs-observed is a planned milestone, not a current result.** The canonical unlock protocol is: (1) fetch published wetland flux (Bechtold et al. 2025) for OWC and the other MUCC wetlands; (2) build a `dim_environmental_context` table with a measured `methane_flux_context` column joined to OWC site/core/depth samples; (3) convert site-level flux into a net-emitter classification label; (4) run leave-one-site-out and leave-one-month-out validation, evaluating calibration, not rank alone. Only after a paired molecular-plus-environmental-plus-measured-flux table is assembled and holdout-validated does the literal continuous-flux predicted-vs-observed demo (tier A) become licensed. This is the single canonical protocol referenced by later sections.

---

## 3. Precision and confidence metrics

Metrics are specified per prediction mode, then split by audience. The audience split matters: customers and scientists care about different numbers, and some numbers are computable today while others require the flux join.

### Metrics by prediction mode

**Classification (tier B target, and the label-recovery result today)**

- Accuracy, precision, recall, F1 per class.
- ROC-AUC and PR-AUC. PR-AUC is emphasized because hidden-emitter sites are the rare, high-cost positive class, and PR-AUC is more informative than ROC-AUC under class imbalance.
- Calibration curve (reliability diagram) plus a calibration error summary. A stated 70 percent risk should be right about 70 percent of the time.
- Confidence intervals on every headline metric, via bootstrap over held-out sites/samples.

**Ranking (tier B, and the activity-proxy result today)**

- Spearman rank correlation between predicted risk and the observed label/proxy.
- Site-ranking agreement (concordance of the predicted site order vs observed order).
- Top-k precision: of the top k sites flagged, how many are true high-methane sites. This is the metric closest to the customer's real question: where do I send the crew first?

**Continuous (tier A, aspirational)**

- Pearson correlation between predicted and observed flux.
- RMSE and MAE in physical units.
- Uncertainty bands (prediction intervals) with coverage checks.

### Which metrics matter to customers vs scientists

| Metric | Primary audience | Why |
|---|---|---|
| Precision / recall on "is this site a hidden emitter" | Customer | A false negative is an unaccounted liability; a false positive is a wasted field visit. Both have direct cost. |
| Calibration curve / calibration error | Customer | A risk score is only actionable if its stated confidence is trustworthy. |
| Top-k precision, site-ranking agreement | Customer | Budgets are finite; customers act on the ranked short list, not the full table. |
| ROC-AUC, PR-AUC | Scientist | Threshold-independent discrimination; standard for method comparison. |
| Spearman / Pearson correlation | Scientist | Strength of the predicted-vs-observed relationship. |
| Ablations (molecular-only vs environment-only vs combined) | Scientist | Isolates where signal comes from. |
| Null models / permutation baselines | Scientist | Shows the result beats chance and is not label leakage. |

Customers should be led with precision, recall, calibration, and top-k ranking agreement. Scientists and technical validators should additionally see AUC, correlation, ablations, and null-model comparisons.

### What can be computed today vs after the join

**Computable today (no new data):**

- For the **label-recovery** result: cluster purity, silhouette, and noise fraction are already reported (`cluster_purity_ecosystem = 0.990`, `cluster_purity_domain = 0.990`, `silhouette_non_noise = 0.433`). These are **clustering-quality and structure-recovery** metrics. They are not methane precision/recall/AUC and must be labeled as such.
- For the **activity-proxy ranking** on OWC: Spearman, site/sample-ranking agreement, and top-k precision of predicted methanogen-activity vs observed metatranscriptomic activity, on a held-out month/depth split, using only repo data. Reported against an **activity proxy**, not flux.

**Requires the flux/label join:**

- All **methane** classification metrics (precision, recall, F1, ROC-AUC, PR-AUC, calibration on "is this a net emitter") require an observed emitter label derived from measured flux.
- All **continuous flux** metrics (Pearson, RMSE, MAE, prediction-interval coverage) require the paired measured-flux table.
- Calibration of any methane-risk score requires held-out flux outcomes; until then, MethaNet outputs are **provisional internal prioritization**, not calibrated risk. Per the roadmap, "MBAG score is calibrated" is explicitly **not yet**.

---

## 4. Minimum viable demo

This is a minimum credible demonstration: small, honest, and explicit about which cells are real today versus which need the paired dataset. It is a tier-B risk-classification demo built on the OWC/MUCC wetland lane, the only lane with an activity proxy and a flux-characterized source study.

### Structure of the demo

A small table of sites/samples (a handful to a dozen, drawn from the 1,948 expression-supported MAGs across 133 depth- and month-resolved OWC samples and the multi-wetland MUCC frame). For each row:

- Predicted methane-risk score: a tier plus a probability.
- Observed measurement or defensible proxy.
- Confidence level / uncertainty band.
- Key microbial and functional drivers, named as real mechanism classes.
- Plain-language interpretation.
- Recommended MRV / management action.

### Illustrative rows

The values below are **illustrative of the demo format**, not published results. The mechanism classes and the activity-proxy construction are real; the risk tiers, probabilities, and any specific methanogen identities are placeholders. Specific methanogen MAG identities and taxonomy are to be pulled from the OWC warehouse and are not asserted here beyond the verified expression-supported counts.

| Site / sample | Predicted risk (tier + prob.) | Observed / proxy | Confidence | Key drivers (mechanism class) | Plain-language read | Recommended action |
|---|---|---|---|---|---|---|
| OWC, high-water-table core, mid-season | Elevated risk (illustrative p ~ 0.8) | Activity proxy: high methanogenesis-marker expression; measured flux pending join | Moderate; wide band (single lane, no flux calibration) | Methanogenesis (mcrA/mcrB/mcrG) expression; low methanotrophy (pmoA/mmoX); weak sulfate reduction (dsrAB/aprAB) | Active methanogenesis markers with little oxidative or competitive brake. Flux not yet joined; would be tested against measured flux after the paired-flux join. | Prioritize for field flux measurement; treat as candidate hidden emitter in MRV design. |
| OWC, oxic surface / shallow depth | Lower risk (illustrative p ~ 0.3) | Activity proxy: methanotroph markers present relative to methanogens | Moderate; wide band | Methanotrophy (pmoA/mmoX) present; methanogenesis markers lower | Oxidative capacity present near the surface; a share of methane may be consumed before escape. | Lower field-visit priority; monitor. |
| MUCC low-methane reference wetland | Low risk (illustrative) | Site-level low-methane contrast (label pending flux join) | Low confidence until join | Sulfate reduction (dsrAB/aprAB) competing with methanogenesis; low mcrA activity | Sulfate competition suppresses methanogenesis. | Use as negative reference; low priority. |

### What is real today vs what needs the paired dataset

**Real today (repo-only):**

- The mechanism-class drivers: methanogenesis (mcrA/mcrB/mcrG), methanotrophy (pmoA/mmoX), and sulfate-reduction competition (dsrAB/aprAB), computable from existing extracts (37,297 accepted KOfam hits, 11,094 MCycDB best hits, 13,272 SCycDB best hits), reported as molecular potential.
- The OWC activity proxy: 1,948 expression-supported MAGs across 133 depth/month-resolved samples, shown as an activity signal on the wetland lane.
- The candidate-driver cards with explicit blockers (100 MRV-readiness cards on OWC/MUCC; 26 in the frozen atlas).
- A held-out split by month or depth for the activity-proxy ranking.

**Needs the paired dataset (planned, not present):**

- The **observed measured-flux** column: requires fetching published wetland flux (Bechtold et al. 2025) and joining it as `dim_environmental_context.methane_flux_context`.
- **Calibrated risk tiers and probabilities:** the demo shows provisional prioritization only. There are **no final A-E methane-risk tiers today**; A-E is target product vocabulary, allowed only after a calibrated sample/project model with holdout validation exists. Where evidence is insufficient, the correct output is `not_scoreable`, never a forced tier.
- **Sample-level** rollups for MSM and Futian: require MAG-to-sample abundance/read-coverage mapping, which no lane yet has.
- The **Sample Risk Readiness** primitive (schema fully specified in `final_mrv_risk_scoring_roadmap.md`; readiness labels include `scoreable_provisional`, `needs_flux_validation`, `not_scoreable`) is the next primitive to build before any calibrated risk score.

### Honest bottom line for the demo

MethaNet can show today, with real numbers, that its latent space recovers held-out ecosystem/domain structure (about 99 percent purity, structure recovery, not methane) and that it discovers reviewable microbial methane-fingerprints with explicit blockers. On the OWC wetland lane it can show predicted methanogen-activity vs observed transcriptomic activity on a held-out split, as an activity proxy. The step from that proxy to a true predicted-methane-vs-observed-methane result, and from provisional prioritization to a calibrated risk tier, is a defined next milestone that requires joining the named paired flux datasets (OWC/MUCC, plus the mangrove lanes for breadth) and running the holdout protocol from Section 2. It is not claimed as done.

---

## 5. Customer translation

Three buyers engage with MethaNet, each for a different reason and each paying for a different reason. The honest split: today's assets earn meetings and, for the right partner, pilots. Paying at scale waits on one milestone we have not yet cleared: a validated methane-risk classification tied to held-out labels and, eventually, observed flux.

### (a) Mangrove / blue-carbon project developer

- **What they care about.** Avoiding a costly surprise: a site that looks like a sink but leaks methane can sink a credit issuance, a buyer relationship, and a reputation. They want to know where to point limited field budget before they commit to a location or a monitoring plan.
- **The output they need.** A per-site or per-sample screening read: where methane-emission potential is more plausible, where it is not, and where the evidence is too thin to say. Plain flags with the reasoning attached, not a black-box grade.
- **What makes them take a meeting.** We can show, on real data, that protein-embedding geometry recovers held-out ecosystem and domain labels at about 99 percent cluster purity on the 40-sample rumen-plus-wetland POC (structure recovery, not a methane metric), and that we can hand them candidate cards naming which MAGs and functions co-occur with methane-relevant contexts, each card carrying its own blockers. This is fingerprint discovery: it tells them where to look, not what the flux is.
- **What makes them PAY.** A screen that has been checked against outcomes: methane-risk classification (net-emitter likely or not, ranked) validated on held-out sites against a defensible proxy, and then against observed flux once the paired dataset is joined. Until that milestone lands, we do not ask a developer to pay for risk tiers. We are honest that today's product is triage and prioritization, and paid engagement before validation is a co-development pilot, not a risk-scoring subscription.

### (b) Carbon-credit buyer / investor / insurer

- **What they care about.** Downside. They are pricing the chance that a credit they buy, back, or underwrite turns out not to represent real, durable removal. Methane leakage is a permanence risk that most diligence never looks at, because it is invisible at the accounting layer.
- **The output they need.** An independent, upstream molecular signal they can put into diligence: a flag that says "this project's biology carries elevated methane-emission potential, prioritize verification here," with the evidence and the uncertainty both legible.
- **What makes them take a meeting.** We are looking at a risk axis their current process ignores, we have a real molecular backbone across blue-carbon systems (662 POC proteomes; 625 MAG/bin-comparable units after quarantining 37 non-comparable units, plus the MSM and Futian coastal lanes in preparation), and we can produce reviewable candidate cards with explicit confidence tiers and blockers. The pitch is a new diligence input, positioned upstream of registry accounting, not a replacement for it.
- **What makes them PAY.** Calibration. A buyer or insurer pays for a signal that has been shown to track observed methane behavior: risk classification validated on an independent holdout, ideally with calibrated risk versus observed flux. That is exactly the milestone we have not cleared. So the honest offer today is a paid pilot to build and test that signal on their portfolio, with the explicit statement that MethaNet is not registry-approved, not credit-decisional, and does not certify or replace flux accounting.

### (c) Sequencing / MRV / technical partner

- **What they care about.** Whether the method is real, reproducible, and defensible, and whether it turns their sequencing throughput into a differentiated, repeatable product rather than a one-off analysis.
- **The output they need.** A documented, auditable pipeline: reproducible embeddings and features, a warehouse they can query, an attestation layer that exposes both evidence and gaps, and a clear protocol for the validation that is still pending.
- **What makes them take a meeting.** The engineering is already built and gate-checked: a consolidated MAG/proteome warehouse over 625 MAG/bin-comparable units (after quarantining 37 non-comparable units) with 711 validation gates passing, a 7,699-node attestation graph with explicit Claim and ValidationGap node types (437 MAGs machine-labeled molecular_attestation_ready_not_mrv, with 188 carrying a QC caveat and 37 blocked as noncomparable), and a frozen partner-facing atlas that separates ready payload from gap rows honestly. A technical partner can see that the boundaries are enforced in the data, not just in the slide.
- **What makes them PAY.** A joint path to the validated milestone: the named public paired datasets (OWC/MUCC wetland expression scaffold, MSM China 2025, Futian mangrove 2026), the exact holdout protocol (train one site, test held-out sites and seasons; evaluate calibration, not just rank), and the "one engine, many maps" reuse story where the next high-GWP gas is a lens swap on the same sequencing input, not a new instrument. They pay to co-own the pipeline and the validation, not to license a finished score.

---

## 6. Validation thresholds for traction

Four escalating bars. Each names the concrete evidence that clears it. We state plainly which one MethaNet stands on today.

### Bar 1: Enough to talk

The minimum to earn a serious first meeting.

- Reproducible structure recovery on real data: ESM2 proteome embeddings recover held-out ecosystem and domain labels at about 99 percent cluster purity (cluster_purity_ecosystem = 0.990, cluster_purity_domain = 0.990, silhouette_non_noise = 0.433, 5 clusters, 10 percent noise) on the 40-sample POC. Presented as label recovery, not flux.
- Fingerprint-discovery candidate cards: reviewable cards naming which MAGs and functions co-occur with methane-relevant contexts (26 in the frozen atlas, 100 MRV-readiness cards on OWC/MUCC), each with a mechanism class and explicit blockers.
- A credible protocol on paper: the named paired datasets and the holdout design that would turn fingerprints into validated risk classification.

### Bar 2: Enough to pilot

Enough for a partner to commit budget to a scoped co-development.

- A held-out methane-risk classification result on a proxy: a per-sample methanogen-activity target built from the OWC expression scaffold (2,508 wetland MAG rows, 2,501 source-protein supported, of which 1,948 are expression-supported across 133 depth- and month-resolved samples) using the already-extracted mcrABG-vs-pmoA marker balance, evaluated on a real held-out split (leave-one-month-out or leave-one-depth-out) with reported precision and ROC-AUC.
- Sample size stated honestly: 133 OWC samples give a genuine within-site split; multi-site classification labels wait on the flux join.
- The source-confounding caveat carried explicitly, with leave-one-source-out named as the next control.

This bar is a classification-against-a-proxy result, not observed flux. It proves the method can rank, on held-out data, before anyone claims it predicts.

### Bar 3: Enough to sell

Enough to charge for a risk-classification product, not just a pilot.

- Calibrated methane-risk versus observed flux on an independent holdout: the published multi-wetland methane flux (Bechtold et al. 2025) fetched and joined, site-level high-versus-low methane labels assigned, and leave-one-site-out validation across the MUCC wetlands reported as precision, ROC-AUC, and a calibration check, not rank alone.
- Every count and denominator reconciled and carried with its gap (for example MSM 1,428 local versus 966 published).
- The claim held to risk classification (distinction B). Continuous flux prediction remains labeled aspirational.

### Bar 4: Enough to scale

Enough to sell across biomes and align to registry-grade diligence.

- Multi-biome external validation: the classifier holding up across wetland and mangrove lanes (MSM and Futian) once MAG-to-sample abundance is recovered and functional coverage is complete, with source-deconfounding cleared (>=2 source projects per ecosystem, two-factor PERMANOVA, leave-one-source-out).
- Registry-aligned evidence: the sample_risk_readiness table built and populated, uncertainty propagated, and outputs mapped to ICVCM Core Carbon Principles and Verra VM0033 as an upstream diligence layer, never as a certification or a flux replacement.
- Continuous-flux calibration (distinction A) attempted only here, gated on a paired molecular-environmental-measured-flux table.

### Where MethaNet is today

**MethaNet is firmly at Bar 1, "Enough to talk," and holds the specific assets needed to move a willing partner to Bar 2, "Enough to pilot," using only data already in the repository.** The structure-recovery result, the fingerprint-discovery cards, the gate-passed 625-unit warehouse, and the OWC expression scaffold are real today. The step to Bar 2 (a held-out risk-classification result against the expression proxy) has not yet been executed but requires no new data. Bars 3 and 4 require external joins we have named but not yet made: measured flux, reconciled denominators, abundance mapping, and source deconfounding. On the MRV maturity ladder this is Level 0 (MAG molecular screening) substantially implemented, with Levels 1 through 6 planned.

---

## 7. Narrative reframing

**Positioning line:** MethaNet turns blue-carbon sequencing into a molecular
attestation graph that helps developers, buyers, raters, and verifiers inspect
methane-pathway evidence and decide where to measure next.

**Narrative (120-160 words):**

Methane can reduce the climate value of blue-carbon projects, while direct
monitoring remains expensive and spatially sparse. MethaNet organizes
sequencing evidence into MBAG, a molecular-attestation knowledge graph. Each
candidate record carries ESM-2 neighborhood context, gLM2 genomic context,
functional evidence, QC, taxonomy, provenance, claim eligibility, and the next
validation action. The August 10 release registers 7,965 MAG/proteome units and
7,710 data-complete tri-views. Of these, 5,209 use a pipeline-normalized
screening contract and 2,501 use the distinct MUCC source-scaffold contract;
zero are yet certified mechanism-comparable across lanes. This structure
supports molecular diligence, candidate review, and monitoring design today.
Exact sample linkage, abundance, environmental
covariates, uncertainty, and paired field evidence provide the route to
calibrated methane-risk intelligence.

**Before / after:**

| Dimension | Old atlas framing | New risk-detection framing |
|---|---|---|
| Headline | "We built a sophisticated microbiome atlas of blue-carbon ecosystems." | "We detect hidden methane-risk signals before they become liabilities." |
| Value to buyer | A rich molecular map to explore. | A decision: where methane risk is more plausible and where to verify first. |
| Hero metric | A large molecular atlas presented as proof by scale alone. | 7,965 registered units, 7,710 data-complete tri-views, 5,209 pipeline-normalized screening tri-views, 2,501 source-scaffold tri-views, and zero cross-lane mechanism-comparable units pending validation. |
| What it protects | Scientific completeness. | Money, ecological integrity, and reputation against invisible methane leakage. |
| Honesty stance | Atlas is impressive, therefore trustworthy. | Screening now, risk classification next, calibrated flux aspirational; blockers named in the data. |
| Product shape | One large static atlas. | One engine, many maps: methane live, other high-GWP gases a lens swap on the same sequencing input. |

---

## 8. Outputs

### 8a. One-Page Validation Memo

**MethaNet: What we can show today, and the path to predicted-vs-observed methane validation**

**What MethaNet answers.** Given metagenome-derived microbial genomes (MAGs) and proteomes from a wetland or coastal site, MethaNet screens the molecular evidence for methane-relevant capacity: which organisms and functions are present, where methane permanence risk is more plausible, and where field flux measurement should be prioritized. It is an upstream molecular screening layer for methane-risk prioritization and monitoring design, not a replacement for flux measurement.

**What we can show today (real numbers).**
- **Structure recovery, honest predicted-vs-observed.** On a 40-sample proof-of-concept, unsupervised ESM2 proteome embeddings recover held-out ecosystem and domain labels at about 99 percent cluster purity (ecosystem 0.990, domain 0.990; silhouette 0.433; 5 clusters; 10 percent noise). This shows the latent space carries real biological structure. It is label recovery, not methane prediction.
- **Molecular backbone.** 662 POC proteomes (555 rumen reference, 107 wetland/MUCC); 625 MAG/bin-comparable units after quarantining 37 non-comparable units.
- **Consolidated warehouse.** A gate-passed functional atlas over the 625 units, DuckDB-queryable, 711 validation gates passing, with methane-mechanism, sulfur-competition, and MAG-level MRV feature tables.
- **Wetland activity scaffold.** MUCC v1 (Old Woman Creek): 2,508 wetland MAG rows (2,501 source-protein supported), of which 1,948 are expression-supported across 133 depth- and month-resolved metatranscriptome samples (259,084 MAG-by-sample expression rows), plus 100 MRV-readiness candidate cards.
- **Candidate discovery.** Reviewable candidate cards naming which MAGs and functions co-occur with methane-relevant contexts, each with an explicit mechanism class and its blockers (26 in the frozen atlas; 100 on OWC/MUCC).

**What we are NOT claiming yet.** No measured methane flux is predicted today. No calibrated sample or project risk score. No final A-E risk tiers. No source-independent transfer proof (the POC is single-source-per-ecosystem, so rumen-to-wetland neighborhoods are nominated hypotheses, not proven transfer). No carbon-credit certification, and no registry approval. Functional annotation is mechanism-relevant potential, not expression, activity, or flux.

**The predicted-vs-observed milestone.** The literal predicted-flux-vs-observed-flux demo requires a paired molecular-plus-flux calibration layer that MethaNet does not yet contain. The honest route has three distinct targets: (A) continuous methane-flux prediction, aspirational and gated on that layer; (B) methane-risk classification (net-emitter likely or not, risk ranking), the credible near-term validation target against held-out labels and proxies; (C) microbial fingerprint discovery, supported today. Our next milestone is target B: join the OWC expression scaffold to published multi-wetland flux (Bechtold et al. 2025) and run leave-one-site-out and leave-one-month-out validation reporting precision and ROC-AUC; the full protocol is in the validation framework (Section 2).

**Timeline (indicative).** Near term: wetland activity-proxy predicted-vs-observed on data in hand. Following that: fetch and join measured wetland flux for the first genuine observed-methane point, then site-level classification with holdout validation. Continuous flux calibration (target A) follows the paired-flux join and is the longest-lead item.

**The ask.** A scoped pilot to (1) fund the paired-flux join and holdout validation for the wetland lane, and (2) identify partner sites with paired sequencing and flux or chamber measurements so risk classification can be validated on partner ground truth.

---

### 8b. Predicted-vs-Observed Table Template

**Schema template. The rows below are illustrative / schema only, not results. No measured methane flux is predicted by MethaNet today; this template defines the join we are building toward for target (B), methane-risk classification.**

| site_id | predicted_risk_tier | predicted_score | uncertainty_band | observed_measurement_or_proxy | agreement_flag | top_microbial_drivers | recommended_action |
|---|---|---|---|---|---|---|---|
| ILLUSTRATIVE-OWC-01 | higher-risk (provisional) | 0.71 | 0.58 to 0.83 | metatranscriptomic methanogen-activity proxy (expression), flux not yet joined | n/a (proxy only, no agreement claim) | methanogenesis markers (mcrABG) expressed | prioritize for field flux measurement |
| ILLUSTRATIVE-OWC-02 | lower-risk (provisional) | 0.24 | 0.12 to 0.39 | low methanogen expression, elevated sulfur-competition markers | n/a (proxy only, no agreement claim) | dsrAB/aprAB sulfur reducers | monitor, defer flux measurement |
| ILLUSTRATIVE-SITE-03 | not_scoreable | n/a | n/a | no MAG-to-sample abundance mapping | n/a | insufficient sample linkage | resolve sample metadata and abundance first |
| ILLUSTRATIVE-SITE-04 | needs_flux_validation | 0.55 | 0.40 to 0.69 | measured annual CH4 flux (external, to be joined from source publication/AmeriFlux) | pending join | mixed methanogen/methanotroph, pmoA present | join flux, then holdout-validate |

Column notes:
- **predicted_risk_tier**: uses readiness/provisional vocabulary only (higher-risk provisional, lower-risk provisional, not_scoreable, needs_flux_validation, needs_metadata, needs_abundance, needs_environment). No A-E tiers exist today.
- **predicted_score**: provisional internal prioritization, not a calibrated probability, until holdout validation exists.
- **observed_measurement_or_proxy**: distinguishes expression/activity proxy (available now for OWC) from measured flux (external, not yet joined).
- **agreement_flag**: only meaningful once observed values are joined; proxy-only rows carry no agreement claim.
- **top_microbial_drivers**: drawn from the attestation graph evidence atoms with mechanism class.

---

### 8c. Five-Slide Customer Demo Outline

**Slide 1: The hidden methane liability**
- Blue-carbon and freshwater wetland projects can be net methane emitters; that liability is often invisible until field flux measurement, which is late and expensive.
- Methane potential is a microbial process: methanogens produce it, methanotrophs and sulfur competitors offset it.
- Sequencing already happens on many of these sites; the molecular signal is being collected but not yet read for methane risk.
- The question a developer, insurer, or buyer needs answered early: where is methane permanence risk more plausible, and where should we spend on flux measurement first?
- *Visual:* map/schematic of a wetland project with an unmeasured methane-risk gradient.

**Slide 2: What we detect (one engine, one live map)**
- From MAGs and proteomes, MethaNet reads methane-mechanism markers (mcrABG methanogenesis vs pmoA/mmoX oxidation), sulfur-competition markers, substrate capacity, taxonomy, and QC.
- Output is a molecular screening map: capacity signal, never a measured flux rate. The field measurement stays the assay.
- Three distinct claims kept separate: (A) continuous flux prediction is aspirational, (B) methane-risk classification is the near-term target, (C) fingerprint discovery is supported today.
- Same engine, same sequencing data; other high-GWP gases are a lens swap, honestly empty candidate slots today.
- *Visual:* the "one engine, many maps" gene-lens schematic; methane live, other gases labeled candidate/not built.

**Slide 3: Evidence we have today**
- 40-sample POC: proteome embeddings recover held-out ecosystem/domain labels at about 99 percent purity (0.990/0.990; silhouette 0.433). Real predicted-vs-observed for labels, not for flux.
- 662 POC proteomes, 625 gate-passed MAG/bin units, 711 validation gates passing, DuckDB-queryable functional atlas.
- MUCC v1 Old Woman Creek: 2,508 wetland MAGs (2,501 source-protein supported), 1,948 expression-supported across 133 samples; methanogenesis markers are transcriptionally active on the wetland lane.
- Reviewable candidate cards with mechanism class and explicit blockers; an attestation graph (7,699 nodes) that makes both the evidence and the validation gaps auditable.
- *Visual:* the frozen molecular niche-space atlas with a labeled candidate card and its blockers.

**Slide 4: The predicted-vs-observed validation plan**
- Today there is no paired flux dataset inside MethaNet; we do not claim a flux demo we cannot run.
- Step 1 (data in hand): per-sample methanogen-vs-methanotroph activity proxy on the OWC expression scaffold, with a held-out month/depth split.
- Step 2: join measured methane flux (Bechtold et al. 2025 multi-wetland study; AmeriFlux/FLUXNET-CH4) for the first genuine observed-methane point.
- Step 3: define net-emitter classification labels, run leave-one-site-out and leave-one-month-out validation, report precision, ROC-AUC, and calibration for target (B). Full protocol in the validation framework (Section 2).
- *Visual:* the predicted-vs-observed table template (schema-only) plus the maturity ladder showing MethaNet at Level 0 with the path to Level 4/5.

**Slide 5: The ask / pilot**
- Fund the paired-flux join and holdout validation for the wetland lane to convert proxy validation into observed-methane classification validation.
- Contribute or identify partner sites with paired sequencing and flux or chamber measurements so classification is validated on partner ground truth.
- Scope: MethaNet stays an upstream screening and monitoring-design layer, aligned to prioritizing field measurement, not replacing it or certifying credits.
- Clear success metric: precision and ROC-AUC on held-out sites for methane-risk classification, delivered with its uncertainty and blockers.
- *Visual:* one-slide pilot scope box (deliverables, timeline, success metric, explicit non-claims).

---

### 8d. Follow-Up Message to the Partner

Subject: Predicted-vs-observed validation, and the precision demo we are preparing

Hello,

Thank you for the clear ask on predicted-vs-observed validation and precision metrics. We understand exactly what you need: not just that the molecular signal separates cleanly, but that it predicts something you can check against a measured outcome, with precision reported honestly.

Here is where we are. The genuine predicted-vs-observed result we can show today is structure recovery: on our proof-of-concept, proteome embeddings recover held-out ecosystem and domain labels at about 99 percent purity. That is real, but it is label recovery, not methane flux. We will not present it as a flux result.

The precision-focused demo we are preparing targets methane-risk classification. We are joining our Old Woman Creek wetland expression scaffold to a methanogen-versus-methanotroph marker balance, holding out sites and months, and reporting precision and ROC-AUC against held-out labels. The step that turns this into observed methane is joining measured flux from the published multi-wetland dataset, which we have identified and scoped.

We would rather show you a validated classification result with its uncertainty than an overstated flux number. We expect the first activity-proxy version on data in hand shortly, with the observed-flux join following. Happy to walk through the plan whenever suits you.

Best regards,
The MethaNet team

---

### 8e. Pre-Customer Analysis Checklist

Minimum analyses before approaching customers, grouped by how far each carries the conversation. Items are drawn from the roadmap backlog.

**(i) Can show today**
- [x] Structure-recovery predicted-vs-observed: ESM2 embeddings recover held-out ecosystem/domain labels at about 99 percent purity on the 40-sample POC (label recovery, not flux).
- [x] Gate-passed functional atlas over 625 MAG/bin units (711 gates passing), queryable, with methane-mechanism and sulfur-competition feature tables.
- [x] Methane-proxy per MAG/sample from existing extracts: mcrABG methanogenesis vs pmoA/mmoX oxidation and dsrAB/aprAB sulfur-competition markers (molecular potential, not flux).
- [x] OWC expression activity view: 1,948 expression-supported MAGs across 133 samples, methanogenesis markers transcriptionally active on the wetland lane.
- [x] Candidate-card fingerprint discovery with mechanism class and explicit blockers; attestation graph exposing evidence and validation gaps.

**(ii) Required for pilot (target B: methane-risk classification)**
- [ ] Matched controls: define lower- vs higher-methane contrasts so classification labels are not confounded by site or habitat class.
- [ ] Source-preserving nulls: build null models that respect source/project structure so recovered signal is not a source artifact.
- [ ] Ablations: molecular-only vs environment-only vs combined feature sets, to show what each layer contributes.
- [ ] Bootstrap robustness: resampling and kNN-seed/downsampling stability for the classification metrics.
- [ ] Holdout validation: leave-one-site-out and leave-one-month-out across the MUCC wetlands, reporting precision and ROC-AUC.
- [ ] Calibration: evaluate calibration of predicted risk, not only rank ordering.
- [ ] Sample-linkage and abundance: build sample identity, MAG-to-sample mapping, and abundance/coverage so scoring is sample-level, not MAG-level; unresolved rows preserved as not_scoreable.
- [ ] Environmental permissiveness join: salinity/sulfate/redox/pH/temperature covariates with explicit resolution tiers.

**(iii) Required to sell (target A: calibrated flux, plus transfer claims)**
- [ ] Paired-flux join: assemble molecular + environmental + measured-flux table (OWC/MUCC flux from source publication and AmeriFlux; coastal chamber/porewater CH4 where available). This is the single blocking gap for any observed-methane claim.
- [ ] Continuous-flux holdout validation: train on one site/project, test on held-out sites/seasons/habitat classes; report calibration and error, not just rank.
- [ ] Source-deconfounding for transfer: >=2 independent source projects per ecosystem, two-factor PERMANOVA (ecosystem + source), leave-one-source-out validation, before any source-independent transfer claim.
- [ ] Denominator reconciliation: resolve MSM 1,428-local vs 966-published and finish the Futian bacterial functional shards so mangrove lanes become consolidated warehouses eligible for sample-level rollups.
- [ ] Sample risk-readiness table: build the specified feature_sample_risk_readiness primitive (readiness_label output) before any A-E tier vocabulary is used externally.

---

Every quantitative figure above traces to a verified MethaNet artifact. The (A) flux / (B) risk-classification / (C) fingerprint-discovery distinction is preserved in each artifact, and no capability is stated in present tense unless it exists today.

---

## Source datasets (provenance verified in-repo)

These are the underlying cohorts. MethaNet holds their molecular data; measured methane flux from the wetland study is not yet joined in (see Section 2).

- **Rumen POC reference:** Stewart et al. 2019, Nature Biotechnology, DOI 10.1038/s41587-019-0202-3 (ENA PRJEB31266).
- **Wetland / MUCC (OWC and eight other freshwater wetlands, with annual methane flux):** Bechtold et al. 2025, Nature Communications 16:944, DOI 10.1038/s41467-025-56133-0; data MUCC v2.0.0, Zenodo 10.5281/zenodo.14532347.
- **Mangrove / MSM China:** Pan et al. 2025, GigaScience, DOI 10.1093/gigascience/giaf081 (GigaDB 10.5524/102702).
- **Mangrove / Futian 2026:** Qi et al. 2026, Scientific Data, DOI 10.1038/s41597-026-07291-3.
- **Measured-flux sources for the planned join:** AmeriFlux and FLUXNET-CH4 (site-level wetland methane flux), USGS Prairie Pothole flux releases.
