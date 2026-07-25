/* =====================================================================
   EmergentBiome / MethaNet - SINGLE SOURCE OF TRUTH
   All numbers, the snapshot date, brand tokens, and scene copy live here.
   Every figure is verified in DIGEST.md against the live repo snapshot.
   External facts live in `ext` with a source; see the grounding dossier.
   To update the page after a new run, edit THIS file only.

   THE STANDING BAR (every scene must let a first-time viewer answer, in one
   sentence each, after a single scroll):
     1. What does it do?            (the decision it produces, not the method)
     2. Who is it for?              (the named buyer and their job-to-be-done)
     3. What exactly do I get?      (the concrete output object, shown)
     4. Why beat the cheap method?  (vs metabarcoding, qPCR, the salinity proxy)
     5. What does it NOT claim?     (the honesty that makes it credible)
   No em-dashes in any public copy. Decision-first headlines. Body <= 25 words.
   ===================================================================== */
window.EB = (function () {
  "use strict";

  /* ---- brand tokens: "Deep Field" ---- */
  const color = {
    bgBase: "#06090D",
    bgPanel: "#0C1218",
    bgElevated: "#121A22",
    hairline: "#1B2730",
    textPrimary: "#EAF2F2",
    textMuted: "#8A9BA5",
    // platform (EmergentBiome) - emergence / verified glow
    emergence: "#2FE3C2",
    // application (MethaNet) - methane / heat / risk gradient
    methaneA: "#FF8A4C",
    methaneB: "#FF5C3A",
    attested: "#86F0E0",
    // ecosystems (distinct, AA-mindful on near-black)
    rumen: "#FF6F91",
    wetland: "#9DE24A",
    mangroveMsm: "#38D0FF",
    mangroveFutian: "#6E7BFF",
  };

  const ecosystems = [
    { key: "rumen",           code: 0, label: "Rumen",           sub: "source reference",      color: color.rumen,          count: 518 },
    { key: "wetland",         code: 1, label: "Wetland / MUCC",  sub: "POC + MUCC v1 reference", color: color.wetland,      count: 2608 },
    { key: "mangrove_msm",    code: 2, label: "Mangrove · MSM",  sub: "China 2025 expansion",  color: color.mangroveMsm,    count: 1428 },
    { key: "mangrove_futian", code: 3, label: "Mangrove · Futian", sub: "2026 expansion (Qi et al.)", color: color.mangroveFutian, count: 3156 },
  ];

  /* ---- verified headline numbers (see DIGEST.md §1) ---- */
  const num = {
    snapshot: "2026-07-24",
    snapshotLiveUTC: "2026-07-24 20:33 UTC",
    snapshotFreezeUTC: "2026-07-24 20:31 UTC",

    calibrationCore: 625,            // rumen + wetland POC, consolidated warehouse
    msmCandidates: 1428,
    msmFunctionalComplete: 1427,
    futianRMAGs: 3404,               // phase-1 dereplicated genomes
    futianReady: 3156,               // ready payload rows (embeddings complete)
    futianGapRows: 248,
    futianArchaeaTotal: 312,
    futianArchaeaComplete: 312,      // 312/312 archaea functional complete
    futianArchaeaCompleteLive: 312,
    futianBacteriaTotal: 2844,
    futianBacteriaComplete: 2619,    // 2,619/2,844 complete in the reconciled release
    futianBacteriaPending: 225,

    triViewReady: 7484,              // data-complete; mechanism comparability is tracked separately
    triViewReadyLive: 7484,
    reportFreezeTriView: 7484,
    mechanismComparableTriView: 625,
    annotationCompletePendingTriView: 4358,
    sourceScaffoldTriView: 2501,
    embeddingBearingUnits: 7710,     // registered ESM-2-bearing units
    plottedNodes: 7965,              // all registered rows, including explicit gaps

    muccWarehouseGenomes: 2508,
    muccTriViewReady: 2501,
    muccExpressionMags: 1948,
    muccExpressionSamples: 133,
    muccCandidateCards: 100,
    warehouseReach: 7965,            // all registered atlas rows across four lanes

    // sample-metadata layer (the start of rung 1: linking genomes to real samples/sites)
    msmSedimentSamples: 82,          // local sediment-sample rows
    msmBiosampleRows: 71,            // exact BioSample environmental rows
    futianSedimentSamples: 65,       // exact sediment-sample rows
    futianSiteTimeKeys: 14,
    futianSites: 2,
    futianMonths: 8,

    bridgeEdges: 2226,               // exported cross-domain k-NN and case-study links
    bridgeNodes: 930,
    caseStudies: 36,
    pocBridgeGenomes: 14,            // POC 662 cohort

    // POC geometry (662-genome cohort)
    pocCohort: 662,
    pocRumen: 555,
    pocWetland: 107,
    embedDim: 1280,
    permanovaR2: 0.202,
    permanovaP: 0.001,
    silhouette: 0.398,
    classifierAUC: 1.0,
    cohensD: 3.63,

    // attestation graph (MMAG MVP)
    magNodes: 662,
    evidenceAtoms: 3968,
    featureNodes: 2644,
    taxonNodes: 397,
    artifactNodes: 13,
    validationGapNodes: 8,
    claimNodes: 5,
    sourceDomains: 2,
    nearEsm2Edges: 9930,

    // program
    pairedFluxNow: 23,
    pairedFluxTargetLo: 80,
    pairedFluxTargetHi: 100,
    pipelineDays: 4,
    analystMonthsLo: 6,
    analystMonthsHi: 12,
    methaneGWP: 30,                  // ~30x CO2 over 100yr (round GWP-100)
    methaneGWP20: 80,                // ~80x CO2 over 20yr (biogenic GWP-20, IPCC AR6, the front-loaded story)
    methaneLifetimeYears: 12,        // perturbation lifetime ~11.8yr (IPCC AR6). NOT a half-life.
    benchmarkR2: 0.879,              // marker-ratio literature target (Lee et al. 2014, freshwater)
  };

  /* ---- external grounded facts (each traces to the dossier; sources + dates inline) ----
     ai_docs/prompts/landing_page_grounding_dossier.md. Flags: SAFE to print. ---- */
  const ext = {
    // Methane climate metrics: IPCC AR6 WG1 Ch.7 Table 7.15, biogenic (non-fossil).
    methaneGWP20: 80,                // GWP-20 ~80.8 non-fossil
    methaneLifetimeYears: 12,        // perturbation lifetime ~11.8 yr (adjustment time)
    methaneERF: 0.54,                // W/m2, second-largest anthropogenic forcing after CO2
    methaneSource: "IPCC AR6 WG1",
    // Blue-carbon premium: Ecosystem Marketplace 2024; S&P Global Platts DBC (carboncredits.com 2025).
    dbcRecordPrice: 29,              // $/tCO2e record, 28 Aug 2025 (Platts DBC blue-carbon benchmark)
    blueCreditsCumulativeM: 7,       // ~7M credits issued cumulatively
    blueActiveProjects: 10,          // ~10 projects actively issuing (supply-constrained)
    premiumSource: "Platts DBC, Aug 2025",
    // Methane wedge: VM0033 v2.1; Frontiers Env Sci 2024; par.nsf.gov review.
    vm0033SalinityPpt: 18,           // below 18 ppt salinity: NO default CH4 factor permitted
    ch4HotspotX: 400,                // methane hotspots ~400x background (Robison et al. 2021)
    wedgeSource: "VM0033 v2.1",
    // Flight to quality: BeZero / Sylvera (State of Carbon Credits 2025).
    integrityRetireFromPct: 10,      // high-integrity retirements 10% share (2022)
    integrityRetireToPct: 22,        // -> 22% share (early 2025); volume to value
    // Science backbone: Baker et al. 2022 ISME J (salt ponds); ISME J 2022 methylotrophy.
    mcraFluxSpearman: "0.7",         // methanogen marker mcrA vs measured CH4 flux, Spearman r > 0.7
    scienceSource: "Baker et al. 2022, ISME J",
    // Competitive white space: structured primary-source review (2026-07-01).
    whiteSpaceAsOf: "July 2026",     // no company found doing molecular methane-risk attestation for blue carbon
  };

  /* ---- hero copy (decision-first; the first sentence a stranger reads is their decision) ---- */
  const hero = {
    eyebrow: "Methane-risk screening for blue carbon",
    sub:
      "Carbon credits price the carbon a wetland stores and ignore the methane it can leak. " +
      "MethaNet reads the sediment community from a single sample and flags which sites carry the pathway " +
      "signature of a hidden methane source, so you know where to measure and which credits to scrutinize before you spend.",
  };

  /* ---- model views (kept jargon-free for the public page) ---- */
  const stack = {
    foundationModels: ["Proteome embeddings", "Genomic-context embeddings"],
    views: ["Proteome embedding", "Genomic context", "Functional annotation"],
  };

  /* ---- non-negotiable claim boundaries (visible, not buried) ---- */
  const claims = {
    footer:
      "Current results are genome-level molecular screening, candidate triage, and monitoring prioritization. " +
      "Measured methane flux, final risk scores, A–E tiers, and carbon-credit decisions require paired abundance, environmental, uncertainty, and field-validation evidence. " +
      "A–E risk tiers remain target product vocabulary while calibration is completed. " +
      "Snapshot " + num.snapshot + ".",
    short: "Molecular screening, candidate triage, and monitoring prioritization. Calibrated MRV follows paired validation.",
    boundaries: [
      "Molecular screening, candidate triage, and monitoring prioritization, at the genome level.",
      "A–E risk tiers are target product vocabulary. Calibration requires paired validation.",
      "Measured flux, final MRV scores, and carbon-credit decisions require evidence beyond the molecular map.",
      "Reference-to-target signals stay provisional until source-balanced validation exists.",
    ],
  };

  /* ---- maturity ladder (MRV roadmap Levels 0–5, + 6 horizon) ---- */
  const ladder = [
    { rung: 0, title: "Molecular screening", state: "lit",
      unlock: "Proteome embeddings, genomic context, functional annotation, quality control, and the attestation graph. Done now." },
    { rung: 1, title: "Sample identity & metadata", state: "progress",
      unlock: "Underway: sample, site, and season metadata recovered for the mangrove lanes (147 sediment samples, 71 environmental rows). Full genome-to-sample mapping is next." },
    { rung: 2, title: "Abundance & community capacity", state: "dim",
      unlock: "Read coverage and relative abundance. Weight genome potential by who is actually there." },
    { rung: 3, title: "Environmental permissiveness", state: "dim",
      unlock: "Salinity, sulfate, redox, temperature, hydroperiod: the site conditions that let methane express, or suppress it." },
    { rung: 4, title: "Flux & process validation", state: "dim",
      unlock: "Chamber and eddy-covariance methane flux, incubations. Paired molecular and measured GHG." },
    { rung: 5, title: "Calibrated probabilistic MRV risk", state: "target",
      unlock: "Holdout-validated risk distribution; A–E tiers mapped to thresholds with uncertainty." },
  ];
  const ladderHorizon = "6 · MRV product & audit / registry integration";

  /* ---- attestation evidence chain (Scene 5): one claim, traced ---- */
  const attestation = {
    claimText: "This genome carries molecular evidence of a methane pathway, consistent with a methane-relevant review hypothesis.",
    forbidden: "“This genome emits methane.”",
    chain: [
      { stage: "Genome",            detail: "One genome, quality-controlled and taxonomically placed", node: "genome" },
      { stage: "Pathway markers",   detail: "Methane pathway marker evidence across producing and consuming guilds is screened and linked", node: "pathway markers" },
      { stage: "Embedding neighbors", detail: "Nearest neighbors in molecular space land on known methanogens", node: "embedding neighbor" },
      { stage: "Quality gate",      detail: "Completeness, contamination, and annotation-coverage checks", node: "validation gate" },
      { stage: "Claim boundary",    detail: "Allowed wording, blocked wording, and the path to upgrade it", node: "claim" },
    ],
    futureSlots: ["", "", ""], // honest empty optionality: no non-methane application built yet
  };

  /* ---- Scene 7 milestones ---- */
  const timeline = [
    { phase: "Now",   label: "Molecular atlas", detail: "7,710 genomes in ESM-2 space; 7,484 data-complete tri-views across three explicit evidence contracts." },
    { phase: "Field", label: "Cispatá Bay validation", detail: "Colombian Caribbean mangrove · paired methane-flux sampling." },
    { phase: "Pair",  label: "Paired data", detail: "Target: pair molecular evidence with field methane flux across seasons and habitats." },
    { phase: "Model", label: "Calibrated methane risk", detail: "Holdout-validated risk distribution; A–E tiers earn their thresholds." },
    { phase: "MRV",   label: "Registry integration", detail: "Reproducible evidence packets aligned to VM0033 / ICVCM review." },
  ];

  /* ---- 9 scenes: kicker, headline, copy (<=25 words), data honesty. See THE STANDING BAR up top. ---- */
  const scenes = [
    {
      id: "stakes", n: 1, label: "01 · The Stakes",
      kicker: "Blue carbon",
      headline: "A carbon sink that can quietly run in reverse.",
      copy: "Methane traps roughly 80x CO₂ over 20 years. In brackish, freshened, or restored systems it can shift to net methane and erode the paid-for benefit.",
      data: "illustrative",
    },
    {
      id: "blindspot", n: 2, label: "02 · The Blind Spot",
      kicker: "The measurement gap",
      headline: "Methane risk is unmeasurable at scale.",
      copy: "Direct methane monitoring is costly and patchy. Below 18 ppt salinity the rules give no default, so the sediment process stays uncounted.",
      data: "illustrative",
    },
    {
      id: "surveyor", n: 3, label: "03 · What You Get",
      kicker: "What you get",
      headline: "A ranked list of where methane risk hides, and where to measure first.",
      copy: "For each site, a methane-capacity evidence profile with its confidence, provenance, and next measurement. It directs field attention to the highest-information contexts.",
      data: "illustrative",
    },
    {
      id: "cheap", n: 4, label: "04 · Versus the Cheap Method",
      kicker: "Versus the cheap method",
      headline: "One gene tells you little. We read the whole pathway balance.",
      copy: "Salinity sets the baseline. The balance of methane-making and methane-eating guilds sets the exception, exactly where a project's methane assumption is most likely wrong.",
      data: "illustrative",
    },
    {
      id: "atlas", n: 5, label: "05 · The Evidence",
      kicker: "The evidence behind the ranking",
      headline: "Every decision traces through a source-audited evidence graph.",
      copy: "The warehouse maps 7,710 genomes and 7,484 data-complete tri-views. MBAG carries the 625-unit mechanism-comparable POC core as a distinct evidence state.",
      data: "real-coords",
    },
    {
      id: "engine", n: 6, label: "06 · One Engine, Many Maps",
      kicker: "One engine, many maps",
      headline: "One evidence engine, with every gas gated separately.",
      copy: "Methane has a 625-unit comparable POC screen. Mangrove, MUCC, and additional gas lenses follow dedicated harmonization and validation paths.",
      data: "real-coords",
    },
    {
      id: "platform", n: 7, label: "07 · The Platform & the Moat",
      kicker: "EmergentBiome substrate",
      headline: "Every claim traces to its evidence.",
      copy: "Embeddings, an attestation graph, and an agentic pipeline. The atlas grows with every project sequenced, and every new pathway inherits the same evidence trail.",
      data: "real-schema",
    },
    {
      id: "ladder", n: 8, label: "08 · The Honest Ladder",
      kicker: "MRV maturity",
      headline: "Molecular screening is the current product layer.",
      copy: "Rung 0 delivers screening, triage, and monitoring prioritization for buyers and raters today. Rungs 1 to 5 add the paired evidence needed for calibrated MRV.",
      data: "real-ladder",
    },
    {
      id: "path", n: 9, label: "09 · The Path & the Ask",
      kicker: "Who it is for, and the ask",
      headline: "Field validation, then paired data, then calibrated risk.",
      copy: "For blue-carbon developers, verifiers, raters, and buyers doing diligence. Cispatá Bay pairs the sediment community screen with measured flux, the step toward calibrated methane risk.",
      data: "real-path",
    },
  ];

  /* ---- outbound links (single place to update the published report path) ---- */
  const links = {
    report: "report/",                 // stable alias on GitHub Pages → current freeze
    reportName: "MethaNet Molecular Attestation Graph (full scientific report)",
    reportDate: "2026-07-24",
    siteUrl: "https://emergentbiome.earth/",
    contactEmail: "jg@graphoflife.com",
  };

  /* ---- brand lockup strings ---- */
  const brand = {
    platform: "EmergentBiome",
    application: "MethaNet",
    lockup: "EmergentBiome / MethaNet",
    tagline: "molecular methane intelligence for blue carbon",
    platformDef: "molecular attestation graph + evidence pipeline",
    applicationDef: "the first application on a substrate architected to generalize beyond methane",
  };

  return {
    color, ecosystems, num, ext, hero, stack, claims, ladder, ladderHorizon,
    attestation, timeline, scenes, brand, links,
    // global seed for all reproducible sketches
    seed: 0xE13B10,
  };
})();
