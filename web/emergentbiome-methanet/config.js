/* =====================================================================
   EmergentBiome / MethaNet - SINGLE SOURCE OF TRUTH
   All numbers, the snapshot date, brand tokens, and scene copy live here.
   Every figure is verified in DIGEST.md against the live repo snapshot.
   To update the page after a new run, edit THIS file only.
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
    { key: "wetland",         code: 1, label: "Wetland / MUCC",  sub: "target domain",         color: color.wetland,        count: 107 },
    { key: "mangrove_msm",    code: 2, label: "Mangrove · MSM",  sub: "China 2025 expansion",  color: color.mangroveMsm,    count: 1428 },
    { key: "mangrove_futian", code: 3, label: "Mangrove · Futian", sub: "2026 expansion (Qi et al.)", color: color.mangroveFutian, count: 3156 },
  ];

  /* ---- verified headline numbers (see DIGEST.md §1) ---- */
  const num = {
    snapshot: "2026-06-27",
    snapshotLiveUTC: "2026-06-27 16:37 UTC",
    snapshotFreezeUTC: "2026-06-25 14:55 UTC",

    calibrationCore: 625,            // rumen + wetland POC, consolidated warehouse
    msmCandidates: 1428,
    msmFunctionalComplete: 1427,
    futianRMAGs: 3404,               // phase-1 dereplicated genomes
    futianReady: 3156,               // ready payload rows (embeddings complete)
    futianGapRows: 248,
    futianArchaeaTotal: 312,
    futianArchaeaComplete: 312,      // 312/312 archaea functional complete (2026-06-27 registry)
    futianArchaeaCompleteLive: 312,
    futianBacteriaQueued: 2844,      // 0/2,844 bacteria complete; queued behind archaea chain

    triViewReady: 2364,              // live registry-backed (625 + 1427 + 312), 2026-06-27
    triViewReadyLive: 2364,
    reportFreezeTriView: 2364,       // report recomputed in sync with live (312 Futian archaea)
    embeddingBearingUnits: 5209,     // 625 + 1428 + 3156  (scaling target ~5,200)
    plottedNodes: 5457,              // incl. 248 non-embedded gap rows

    muccWarehouseGenomes: 2508,
    muccExpressionMags: 1948,
    muccExpressionSamples: 133,
    muccCandidateCards: 100,
    warehouseReach: 7717,            // 5,209 mapped + 2,508 MUCC wetland warehouse genomes

    // sample-metadata layer (the start of rung 1: linking genomes to real samples/sites)
    msmSedimentSamples: 82,          // local sediment-sample rows
    msmBiosampleRows: 71,            // exact BioSample environmental rows
    futianSedimentSamples: 65,       // exact sediment-sample rows
    futianSiteTimeKeys: 14,
    futianSites: 2,
    futianMonths: 8,

    bridgeEdges: 372,                // documented cross-domain k-NN bridge edges
    bridgeNodes: 136,
    caseStudies: 26,
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
    methaneGWP: 30,                  // ~30x CO2 over 100yr
    benchmarkR2: 0.879,              // marker-ratio literature target (Lee et al. 2014, freshwater)
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
      "No measured methane flux, final risk scores, A–E tiers, or carbon-credit approval are claimed " +
      "from the molecular map alone. A–E risk tiers are target product vocabulary, not yet calibrated. " +
      "Snapshot " + num.snapshot + ".",
    short: "Molecular screening and candidate triage, not calibrated MRV. A–E tiers are target vocabulary.",
    boundaries: [
      "Molecular screening, candidate triage, and monitoring prioritization, at the genome level.",
      "A–E risk tiers are TARGET product vocabulary, explicitly not yet calibrated.",
      "No measured flux, final MRV scores, or carbon-credit approval from the map alone.",
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
    claimText: "This genome carries molecular evidence consistent with a methane-relevant review hypothesis.",
    forbidden: "“This genome emits methane.”",
    chain: [
      { stage: "Genome",            detail: "One genome, quality-controlled and taxonomically placed", node: "genome" },
      { stage: "Markers",           detail: "Methane-related marker evidence is screened and linked", node: "marker genes" },
      { stage: "Embedding neighbors", detail: "Nearest neighbors in molecular space land on known methanogens", node: "embedding neighbor" },
      { stage: "Quality gate",      detail: "Completeness, contamination, and annotation-coverage checks", node: "validation gate" },
      { stage: "Claim boundary",    detail: "Allowed wording, blocked wording, and the path to upgrade it", node: "claim" },
    ],
    futureSlots: ["", "", ""], // honest empty optionality: no non-methane application built yet
  };

  /* ---- Scene 7 milestones ---- */
  const timeline = [
    { phase: "Now",   label: "Molecular atlas", detail: "5,209 genomes mapped; 2,508 wetland genomes queryable as the next warehouse-backed expansion." },
    { phase: "Field", label: "Cispatá Bay validation", detail: "Colombian Caribbean mangrove · paired methane-flux sampling." },
    { phase: "Pair",  label: "Paired data", detail: "Target: pair molecular evidence with field methane flux across seasons and habitats." },
    { phase: "Model", label: "Calibrated methane risk", detail: "Holdout-validated risk distribution; A–E tiers earn their thresholds." },
    { phase: "MRV",   label: "Registry integration", detail: "Reproducible evidence packets aligned to VM0033 / ICVCM review." },
  ];

  /* ---- 7 scenes: kicker, headline, copy (<=25 words), data honesty ---- */
  const scenes = [
    {
      id: "stakes", n: 1, label: "01 · The Stakes",
      kicker: "Blue carbon",
      headline: "A carbon sink that can quietly run in reverse.",
      copy: "Coastal wetlands lock away carbon. But methane, at roughly 30× the warming power of CO₂, can quietly erode that net benefit.",
      data: "illustrative",
    },
    {
      id: "blindspot", n: 2, label: "02 · The Blind Spot",
      kicker: "The measurement gap",
      headline: "Methane risk is unmeasurable at scale.",
      copy: "The sediment microbiome is climate dark matter. Direct methane measurements are rare and costly, so most of the risk surface has never been observed.",
      data: "illustrative",
    },
    {
      id: "insight", n: 3, label: "03 · The Insight",
      kicker: "The molecular map",
      headline: "The map turns bridge candidates into reviewable hypotheses.",
      copy: "We place every genome in a learned map of its proteins. Reference communities separate into basins, and bridge candidates appear: genomes whose molecular signatures make cross-ecosystem methane hypotheses reviewable.",
      data: "real-coords",
    },
    {
      id: "surveyor", n: 4, label: "04 · The Survey",
      kicker: "Where to measure first",
      headline: "We map where to drill, not what's in the ground.",
      copy: "A mineral survey reads rock chemistry to rank where the first costly core goes. MethaNet ranks where methane deserves its first field measurement.",
      data: "real-coords",
    },
    {
      id: "atlas", n: 5, label: "05 · The Atlas",
      kicker: "A source-audited atlas",
      headline: "2,364 tri-view genomes. 2,508 more in the wetland warehouse.",
      copy: "The live atlas spans rumen, wetland, and two mangrove systems. MUCC adds 2,508 source-audited wetland genomes as the warehouse-backed scaffold for the next manifold refresh.",
      data: "real-counts",
    },
    {
      id: "engine", n: 6, label: "06 · One Engine, Many Maps",
      kicker: "One engine, many maps",
      headline: "One engine that keeps printing new risk maps.",
      copy: "The engine that ranks methane risk can be pointed at the genes behind nitrous oxide, the gas today's blue-carbon rules still let projects count as zero.",
      data: "real-coords",
    },
    {
      id: "platform", n: 7, label: "07 · The Platform & the Moat",
      kicker: "EmergentBiome substrate",
      headline: "Every claim traces to its evidence.",
      copy: "Embeddings, an attestation graph, and an agentic pipeline. The breadth you just saw runs on one substrate, where every new gas inherits the same evidence trail.",
      data: "real-schema",
    },
    {
      id: "ladder", n: 8, label: "08 · The Honest Ladder",
      kicker: "MRV maturity",
      headline: "We are at molecular screening, and we say so.",
      copy: "Calibrated A–E methane risk requires sample mapping, abundance, environment, and field flux. Rung 0 is lit. The rest is the roadmap.",
      data: "real-ladder",
    },
    {
      id: "path", n: 9, label: "09 · The Path & the Ask",
      kicker: "From sequences to signal",
      headline: "Field validation → paired data → calibrated risk.",
      copy: "An agentic pipeline turns raw sequences into reproducible reports in under four days. Cispatá Bay is where the molecular map meets the mud.",
      data: "real-path",
    },
  ];

  /* ---- outbound links (single place to update the published report path) ---- */
  const links = {
    report: "report/",                 // stable alias on GitHub Pages → current freeze
    reportName: "MethaNet Molecular Niche Atlas (full scientific report)",
    reportDate: "2026-06-29",
    siteUrl: "https://emergentbiome.earth/",
    contactEmail: "jg@graphoflife.com",
  };

  /* ---- brand lockup strings ---- */
  const brand = {
    platform: "EmergentBiome",
    application: "MethaNet",
    lockup: "EmergentBiome / MethaNet",
    tagline: "methane-risk intelligence for blue carbon",
    platformDef: "embeddings + attestation graph + agentic pipeline",
    applicationDef: "the first application on a substrate architected to generalize beyond methane",
  };

  return {
    color, ecosystems, num, stack, claims, ladder, ladderHorizon,
    attestation, timeline, scenes, brand, links,
    // global seed for all reproducible sketches
    seed: 0xE13B10,
  };
})();
