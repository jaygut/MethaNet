/* =====================================================================
   EmergentBiome Molecular Atlas - PUBLIC RENDERING CONFIGURATION
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
    // atlas - emergence / verified glow
    emergence: "#2FE3C2",
    // methane-pathway evidence accent
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

    bridgeEdges: 2226,               // displayed map links: 2,200 sampled cross-domain k-NN + 26 highlighted candidate links
    bridgeNodes: 930,
    caseStudies: 36,
    pocBridgeGenomes: 14,            // POC 662 cohort
    nearestCoreWetland: 2434,        // wetland records whose raw-cosine nearest POC-core match is rumen
    nearestCoreMangrove: 4475,       // mangrove records whose raw-cosine nearest POC-core match is rumen
    nearestCoreCandidates: 26,       // of 27 selected wetland/mangrove candidate cards
    candidateCards: 27,
    sampledNeighborLinks: 2200,
    highlightedCandidateLinks: 26,
    standardizedRumenReciprocalPairs: 0, // full-atlas, dimension-standardized reciprocal top-35 pairs

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

  /* ---- hero copy: published molecular evidence, with validation status visible ---- */
  const hero = {
    eyebrow: "Frozen August 10, 2026 evidence release",
    sub:
      "A molecular evidence atlas for methane-pathway screening in blue-carbon systems. " +
      "Explore 7,710 data-complete MAG/proteome records, candidate reference links, and the measurements needed before flux or risk can be estimated.",
  };

  /* ---- model views (kept jargon-free for the public page) ---- */
  const stack = {
    foundationModels: ["Proteome embeddings", "Genomic-context embeddings"],
    views: ["Proteome embedding", "Genomic context", "Functional annotation"],
  };

  /* ---- public terminology: visible in the hero and closing glossary ---- */
  const terminology = [
    {
      term: "Atlas",
      full: "EmergentBiome Molecular Atlas",
      detail: "A frozen release view of 7,710 data-complete MAG/proteome records across distinct evidence contracts. Completeness does not establish cross-lane mechanism comparability.",
      hero: true,
    },
    {
      term: "Graph",
      full: "EmergentBiome evidence graph",
      detail: "The queryable proof-of-concept graph covers 662 records. It connects molecular evidence to provenance, quality, claim scope, and validation gaps; atlas-wide persistence is planned.",
      hero: true,
    },
    {
      term: "MRV",
      full: "Monitoring, reporting, and verification",
      detail: "A future calibrated application requiring exact sample linkage, abundance, environmental context, uncertainty, and process or field validation.",
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
      detail: "ESM-2, gLM2, and a functional payload. The atlas tracks payload completeness and mechanism comparability as separate states.",
    },
    {
      term: "MUCC v1",
      full: "Old Woman Creek wetland reference lane",
      detail: "A genome and metatranscriptome source warehouse. Its current functional evidence uses a source-scaffold contract.",
    },
    {
      term: "VM0033",
      full: "Verra Methodology for Tidal Wetland and Seagrass Restoration",
      detail: "A project-methodology context for blue-carbon restoration and future field-validation planning.",
    },
  ];

  /* ---- non-negotiable claim boundaries (visible, not buried) ---- */
  const claims = {
    footer:
      "Current results are molecular screening, candidate triage, evidence-card review, and monitoring prioritization at MAG/proteome grain. " +
      "Measured methane flux, final risk scores, A–E tiers, and carbon-credit decisions require paired abundance, environmental, uncertainty, and field-validation evidence. " +
      "A–E risk tiers remain target product vocabulary while calibration is completed. " +
      "Snapshot " + num.snapshot + ".",
    short: "MAG/proteome screening and candidate review only. Flux and calibrated risk require paired field validation.",
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
      unlock: "Proteome embeddings, genomic context, functional annotation, quality control, atlas records, and a scoped POC evidence graph. Available now." },
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
      { stage: "Embedding neighbors", detail: "Nearest neighbors nominate reference comparisons; functional identity is checked separately", node: "embedding neighbor" },
      { stage: "Quality gate",      detail: "Completeness, contamination, and annotation-coverage checks", node: "validation gate" },
      { stage: "Claim boundary",    detail: "Allowed wording, blocked wording, and the path to upgrade it", node: "claim" },
    ],
    futureSlots: ["", "", ""], // honest empty optionality: no non-methane application built yet
  };

  /* ---- real MUCC candidate example, sourced to the frozen candidate_cards.tsv ---- */
  const candidateExample = {
    id: "mucc_v1__OWC_1885",
    views: {
      recorded: {
        title: "Recorded molecular evidence",
        points: [
          "CheckM2: 94.89% completeness and 1.14% contamination, reconciled to the MAG archive and source QC.",
          "Processed expression detects marker terms. Detection is neither activity magnitude nor methane-process rate.",
          "Raw-cosine nearest match in the 625-record POC core is rumen (0.9842857). This is a one-way reference match.",
        ],
      },
      pending: {
        title: "What remains unresolved",
        points: [
          "No authoritative exact sample, collection date, and depth link for this MAG.",
          "No joined abundance/read coverage, matched environment, or methane-flux/process observation.",
          "The source-scaffold functional contract is not mechanism-equivalent to the pipeline-normalized lanes.",
        ],
      },
      next: {
        title: "Next validation action",
        points: [
          "Resolve the MAG to an exact physical sample with date and depth.",
          "Pair abundance and environment with a compatible methane-process measurement.",
          "Then review marker identity and test source-aware stability before ecological or risk inference.",
        ],
      },
    },
  };

  /* ---- Scene 7 milestones ---- */
  const timeline = [
    { phase: "Now",   label: "Molecular atlas + POC graph", detail: "7,710 data-complete MAG/proteome records in the atlas; a separate 662-record queryable POC evidence graph." },
    { phase: "Field", label: "Partner validation cohort", detail: "Paired molecular, environmental, abundance, and methane-process measurements." },
    { phase: "Pair",  label: "Paired data", detail: "Target: pair molecular evidence with field methane flux across seasons and habitats." },
    { phase: "Model", label: "Calibrated methane risk", detail: "Holdout-validated risk distribution; A–E tiers earn their thresholds." },
    { phase: "Audit", label: "Methodology review", detail: "Reproducible evidence packets tested against applicable methodology and integrity requirements." },
  ];

  /* ---- 9 scenes: kicker, headline, copy (<=25 words), data honesty. See THE STANDING BAR up top. ---- */
  const scenes = [
    {
      id: "stakes", n: 1, label: "01 · The Climate Question",
      kicker: "Blue carbon",
      headline: "Methane can narrow a wetland's net climate benefit.",
      copy: "Methane has roughly 80 times CO₂'s warming impact over 20 years. Its contribution depends on habitat, conditions, and measured exchange.",
      data: "mixed-real-and-illustrative",
    },
    {
      id: "blindspot", n: 2, label: "02 · The Measurement Gap",
      kicker: "The measurement gap",
      headline: "Project-scale methane evidence is still sparse.",
      copy: "Field methane measurements are uneven. Molecular screening can help select what to measure next while site-specific flux remains a field question.",
      data: "mixed-real-and-illustrative",
    },
    {
      id: "surveyor", n: 3, label: "03 · The Evidence Card",
      kicker: "What the atlas gives you",
      headline: "Inspect one candidate's evidence trail.",
      copy: "This frozen wetland MAG card joins source, QC, expression detection, a one-way reference match, unresolved joins, and the next measurement.",
      data: "real-schema",
    },
    {
      id: "cheap", n: 4, label: "04 · Complementary Evidence",
      kicker: "Context plus biology",
      headline: "Salinity, markers, and genomes each add context.",
      copy: "Salinity frames field conditions. Marker and genome evidence nominate review hypotheses; none alone yields methane flux or calibrated site risk.",
      data: "illustrative",
    },
    {
      id: "atlas", n: 5, label: "05 · Explore the Atlas",
      kicker: "A map with explicit scope",
      headline: "Explore the links, then test what survives.",
      copy: "Switch between atlas coverage, nearest-core candidates, sampled neighbor links, and standardized sensitivity. Each view asks a different question of the same frozen release.",
      data: "real-coords",
    },
    {
      id: "engine", n: 6, label: "06 · Evidence Scope",
      kicker: "Scope before extension",
      headline: "Methane screening is the present use.",
      copy: "Other pathway lenses are research options. Each needs its own evidence contract, independent validation, and paired field outcomes before process claims.",
      data: "real-coords",
    },
    {
      id: "platform", n: 7, label: "07 · The Evidence Trail",
      kicker: "An auditable starting point",
      headline: "The released graph and atlas have different coverage.",
      copy: "A queryable 662-record POC evidence graph anchors the broader 7,710-record atlas. Extending graph persistence across the atlas is a planned step.",
      data: "real-schema",
    },
    {
      id: "ladder", n: 8, label: "08 · Validation Path",
      kicker: "Evidence maturity",
      headline: "Today's layer supports molecular review.",
      copy: "Screening and candidate triage are available now. Exact sample links, abundance, environment, uncertainty, and field outcomes are needed for calibrated risk.",
      data: "real-ladder",
    },
    {
      id: "path", n: 9, label: "09 · Partnership Path",
      kicker: "The next decision",
      headline: "Pair molecular evidence with field outcomes.",
      copy: "For blue-carbon teams planning monitoring and diligence: a partner cohort can connect exact samples, abundance, environment, and methane-process measurements.",
      data: "roadmap",
    },
  ];

  /* ---- outbound links (single place to update the published report path) ---- */
  const links = {
    report: "report/",                 // deployed alias is an older build; label as historical until resynchronized
    reportName: "Dated technical archive",
    reportDate: "2026-08-10",
    siteUrl: "https://emergentbiome.earth/",
    contactEmail: "jg@graphoflife.com",
  };

  /* ---- brand lockup strings ---- */
  const brand = {
    platform: "EmergentBiome",
    application: "Molecular Atlas",
    lockup: "EmergentBiome Molecular Atlas",
    tagline: "molecular evidence for blue-carbon methane diligence",
    platformDef: "frozen molecular atlas plus scoped proof-of-concept evidence graph",
    applicationDef: "methane-pathway screening and validation planning",
  };

  return {
    color, ecosystems, num, ext, hero, stack, terminology, claims, ladder, ladderHorizon,
    attestation, candidateExample, timeline, scenes, brand, links,
    // global seed for all reproducible sketches
    seed: 0xE13B10,
  };
})();
