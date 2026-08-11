/* =====================================================================
   EmergentBiome / MethaNet - PUBLIC RENDERING CONFIGURATION
   The dated release_ledger.json is the numerical source of truth. This file
   projects those verified counts into scene copy and is checked by
   tools/validate_release_parity.py before publication.
   External facts live in `ext` with a source; see the grounding dossier.
   To update the page after a new run, regenerate the ledger and then update
   this projection plus DIGEST.md; parity validation must pass.

   THE STANDING BAR (every scene must let a first-time viewer answer, in one
   sentence each, after a single scroll):
     1. What does it do?            (the decision it produces, not the method)
     2. Who is it for?              (the named buyer and their job-to-be-done)
     3. What exactly do I get?      (the concrete output object, shown)
     4. Why add it to proxy screening? (vs using metabarcoding, qPCR, or salinity alone)
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
    { key: "wetland",         code: 1, label: "Old Woman Creek", sub: "wetland reference lane", color: color.wetland,      count: 2608 },
    { key: "mangrove_msm",    code: 2, label: "Mangrove · MSM",  sub: "China 2025 expansion",  color: color.mangroveMsm,    count: 1428 },
    { key: "mangrove_futian", code: 3, label: "Mangrove · Futian", sub: "2026 expansion (Qi et al.)", color: color.mangroveFutian, count: 3156 },
  ];

  /* ---- verified headline numbers (see DIGEST.md §1) ---- */
  const num = {
    snapshot: "2026-08-10",
    snapshotLiveUTC: "2026-08-10 controlled-diligence audit",
    snapshotFreezeUTC: "2026-08-10 release-ledger freeze",

    calibrationCore: 625,            // rumen + wetland POC, consolidated warehouse
    msmCandidates: 1428,
    msmFunctionalComplete: 1428,
    futianRMAGs: 3404,               // phase-1 dereplicated genomes
    futianReady: 3156,               // ready payload rows (embeddings complete)
    futianGapRows: 248,
    futianArchaeaTotal: 312,
    futianArchaeaComplete: 312,      // 312/312 archaea functional complete
    futianArchaeaCompleteLive: 312,
    futianBacteriaTotal: 2844,
    futianBacteriaComplete: 2844,
    futianBacteriaPending: 0,

    triViewReady: 7710,              // data-complete; mechanism comparability is tracked separately
    triViewReadyLive: 7710,
    reportFreezeTriView: 7710,
    schemaNormalizedTriView: 7710,
    pipelineNormalizedTriView: 5209,
    mechanismComparableTriView: 0,
    annotationCompletePendingTriView: 0,
    sourceScaffoldTriView: 2501,
    embeddingBearingUnits: 7710,     // registered ESM-2-bearing units
    esm2Units: 7710,
    glm2Units: 7717,
    functionalPayloadUnits: 7710,
    sampleLinkedUnits: 0,
    fieldValidatedUnits: 0,
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
    pairedFluxNow: 0,                // authoritative exact sample + environment + process joins
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
    eyebrow: "Molecular attestation for blue-carbon methane diligence",
    sub:
      "Carbon credits price the carbon a wetland stores while methane can erode the net climate benefit. " +
      "MethaNet turns blue-carbon sequencing into an evidence graph that shows pathway evidence, provenance, and the next measurement to prioritize.",
  };

  /* ---- model views (kept jargon-free for the public page) ---- */
  const stack = {
    foundationModels: ["Proteome embeddings", "Genomic-context embeddings"],
    views: ["Proteome embedding", "Genomic context", "Functional annotation"],
  };

  /* ---- public terminology: visible in the hero and closing glossary ---- */
  const terminology = [
    {
      term: "MBAG",
      full: "MethaNet Bridge Attestation Graph",
      detail: "The knowledge graph linking each molecular evidence unit to representations, mechanism features, provenance, quality gates, claim scope, and next validation actions.",
      hero: true,
    },
    {
      term: "MRV",
      full: "Monitoring, reporting, and verification",
      detail: "The calibrated application layer. It requires sample linkage, abundance, environmental context, uncertainty, and field or process validation.",
      hero: true,
    },
    {
      term: "MAG",
      full: "Metagenome-assembled genome",
      detail: "A reconstructed microbial genome used as one molecular evidence unit. Sample-level conclusions require community mapping and weighting.",
    },
    {
      term: "ESM-2",
      full: "Protein language model view",
      detail: "A protein-sequence representation used to map molecular neighborhoods and nominate candidates for review.",
    },
    {
      term: "gLM2",
      full: "Genome-context language model view",
      detail: "Gene-order and neighborhood context. Numerical comparisons remain within the applicable protocol class.",
    },
    {
      term: "Tri-view",
      full: "Three coordinated evidence views",
      detail: "ESM-2, gLM2, and a functional payload. MBAG tracks payload completeness and mechanism comparability as separate states.",
    },
    {
      term: "MUCC v1",
      full: "Old Woman Creek wetland reference lane",
      detail: "A genome and metatranscriptome source warehouse. Its current functional evidence uses a source-scaffold contract.",
    },
    {
      term: "VM0033",
      full: "Verra Methodology for Tidal Wetland and Seagrass Restoration",
      detail: "A project-methodology context for blue-carbon restoration and MethaNet validation planning.",
    },
  ];

  /* ---- non-negotiable claim boundaries (visible, not buried) ---- */
  const claims = {
    footer:
      "Current results are molecular screening, candidate triage, evidence-card review, and monitoring prioritization at metagenome-assembled genome or proteome grain. " +
      "Measured methane flux, final risk scores, A–E tiers, and carbon-credit decisions require paired abundance, environmental, uncertainty, and field-validation evidence. " +
      "A–E risk tiers remain target product vocabulary while calibration is completed. " +
      "Snapshot " + num.snapshot + ".",
    short: "Molecular attestation, candidate triage, and monitoring prioritization. Calibrated MRV follows paired validation.",
    boundaries: [
      "Molecular attestation, candidate triage, and monitoring prioritization at metagenome-assembled genome or proteome grain.",
      "A–E risk tiers are target product vocabulary. Calibration requires paired validation.",
      "Measured flux, final MRV scores, and carbon-credit decisions require evidence beyond the molecular map.",
      "Reference-to-target signals stay provisional until source-balanced validation exists.",
    ],
  };

  /* ---- maturity ladder (MRV roadmap Levels 0–5, + 6 horizon) ---- */
  const ladder = [
    { rung: 0, title: "Molecular attestation", state: "lit",
      unlock: "Proteome embeddings, genomic context, functional annotation, quality control, and MethaNet Bridge Attestation Graph evidence cards. Done now." },
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
    claimText: "This metagenome-assembled genome or proteome carries molecular evidence of a methane pathway, consistent with a methane-relevant review hypothesis.",
    forbidden: "“This genome emits methane.”",
    chain: [
      { stage: "Genome or proteome", detail: "One molecular evidence unit, quality-controlled and taxonomically placed", node: "genome" },
      { stage: "Pathway markers",   detail: "Methane pathway marker evidence across producing and consuming guilds is screened and linked", node: "pathway markers" },
      { stage: "Embedding neighbors", detail: "Nearest neighbors in molecular space land on known methanogens", node: "embedding neighbor" },
      { stage: "Quality gate",      detail: "Completeness, contamination, and annotation-coverage checks", node: "validation gate" },
      { stage: "Claim boundary",    detail: "Allowed wording, blocked wording, and the path to upgrade it", node: "claim" },
    ],
    futureSlots: ["", "", ""], // honest empty optionality: no non-methane application built yet
  };

  /* ---- Scene 7 milestones ---- */
  const timeline = [
    { phase: "Now",   label: "MBAG knowledge graph", detail: "7,710 genomes in protein-representation space; 7,710 data-complete tri-views across pipeline-normalized and source-scaffold contracts." },
    { phase: "Field", label: "Partner validation cohort", detail: "Paired molecular, environmental, abundance, and methane-process measurements." },
    { phase: "Pair",  label: "Paired data", detail: "Target: pair molecular evidence with field methane flux across seasons and habitats." },
    { phase: "Model", label: "Calibrated methane risk", detail: "Holdout-validated risk distribution; A–E tiers earn their thresholds." },
    { phase: "Audit", label: "Methodology integration", detail: "Reproducible evidence packets aligned to relevant methodology and integrity review." },
  ];

  /* ---- 9 scenes: kicker, headline, copy (<=25 words), data honesty. See THE STANDING BAR up top. ---- */
  const scenes = [
    {
      id: "stakes", n: 1, label: "01 · The Stakes",
      kicker: "Blue carbon",
      headline: "Methane can erode a wetland's net climate benefit.",
      copy: "Methane traps roughly 80x CO₂ over 20 years. In brackish, freshened, or restored systems it can shift to net methane and erode the paid-for benefit.",
      data: "mixed-real-and-illustrative",
    },
    {
      id: "blindspot", n: 2, label: "02 · The Blind Spot",
      kicker: "The measurement gap",
      headline: "Project-scale methane evidence is still sparse.",
      copy: "Direct methane monitoring is costly and patchy. Below 18 ppt salinity the rules give no default, so the sediment process stays uncounted.",
      data: "mixed-real-and-illustrative",
    },
    {
      id: "surveyor", n: 3, label: "03 · What You Get",
      kicker: "What you get",
      headline: "Evidence cards that show where to measure first.",
      copy: "The MethaNet Bridge Attestation Graph (MBAG) returns molecular evidence, confidence, provenance, and the next measurement needed to calibrate risk.",
      data: "illustrative",
    },
    {
      id: "cheap", n: 4, label: "04 · Beyond Proxy-Only Screening",
      kicker: "Beyond proxy-only screening",
      headline: "Proxy measures and pathway evidence answer different questions.",
      copy: "Salinity and targeted markers remain useful. Multi-view pathway, substrate, sulfur, quality, and provenance evidence can show where more measurement adds value.",
      data: "illustrative",
    },
    {
      id: "atlas", n: 5, label: "05 · The Evidence",
      kicker: "The evidence behind the ranking",
      headline: "Every decision traces through a source-audited evidence graph.",
      copy: "The ledger records 7,710 data-complete tri-views. It keeps 5,209 pipeline-normalized and 2,501 source-scaffold units separate; cross-lane mechanism comparability is pending.",
      data: "real-coords",
    },
    {
      id: "engine", n: 6, label: "06 · One Engine, Many Maps",
      kicker: "One engine, many maps",
      headline: "One evidence engine, with every gas gated separately.",
      copy: "Methane screening is available under partitioned evidence contracts. Cross-lane comparison waits for locked fingerprints, source-aware statistics, and paired field validation.",
      data: "real-coords",
    },
    {
      id: "platform", n: 7, label: "07 · The Platform & the Moat",
      kicker: "EmergentBiome substrate",
      headline: "Every claim traces to its evidence.",
      copy: "MBAG joins molecular representations, mechanism evidence, provenance, and validation gaps. The atlas grows with every project sequenced while preserving each evidence trail.",
      data: "real-schema",
    },
    {
      id: "ladder", n: 8, label: "08 · The Honest Ladder",
      kicker: "Evidence maturity",
      headline: "Molecular evidence review is the available layer.",
      copy: "Rung 0 delivers screening, triage, and monitoring prioritization. Rungs 1 to 5 add the paired evidence required for calibrated monitoring, reporting, and verification.",
      data: "real-ladder",
    },
    {
      id: "path", n: 9, label: "09 · The Path & the Ask",
      kicker: "Who it is for, and the ask",
      headline: "Field validation, then paired data, then calibrated risk.",
      copy: "For blue-carbon developers, verifiers, raters, and buyers doing diligence. A partner cohort can pair molecular evidence with abundance, environmental context, and methane-process measurements.",
      data: "roadmap",
    },
  ];

  /* ---- outbound links (single place to update the published report path) ---- */
  const links = {
    report: "report/",                 // stable alias on GitHub Pages → current freeze
    reportName: "MethaNet Bridge Attestation Graph (full scientific report)",
    reportDate: "2026-08-10",
    siteUrl: "https://emergentbiome.earth/",
    contactEmail: "jg@graphoflife.com",
  };

  /* ---- brand lockup strings ---- */
  const brand = {
    platform: "EmergentBiome",
    application: "MethaNet",
    lockup: "EmergentBiome / MethaNet",
    tagline: "molecular attestation for blue-carbon methane diligence",
    platformDef: "molecular-attestation knowledge graph + evidence pipeline",
    applicationDef: "the first application on a substrate architected to generalize beyond methane",
  };

  return {
    color, ecosystems, num, ext, hero, stack, terminology, claims, ladder, ladderHorizon,
    attestation, timeline, scenes, brand, links,
    // global seed for all reproducible sketches
    seed: 0xE13B10,
  };
})();
