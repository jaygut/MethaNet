/* =====================================================================
   EmergentBiome / MethaNet - SINGLE SOURCE OF TRUTH
   All numbers, the snapshot date, brand tokens, and scene copy live here.
   Every figure is verified in DIGEST.md against the live repo snapshot
   (2026-06-25). To update the page after a new run, edit THIS file only.
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
    snapshot: "2026-06-25",
    snapshotLiveUTC: "2026-06-25 15:39 UTC",
    snapshotFreezeUTC: "2026-06-25 14:55 UTC",

    calibrationCore: 625,            // rumen + wetland POC, consolidated warehouse
    msmCandidates: 1428,
    msmFunctionalComplete: 1427,
    futianRMAGs: 3404,               // phase-1 dereplicated at 99% ANI
    futianReady: 3156,               // ready payload rows (ESM2+gLM2 complete)
    futianGapRows: 248,
    futianArchaeaTotal: 312,
    futianArchaeaComplete: 300,      // report freeze (live: 302)
    futianArchaeaCompleteLive: 302,
    futianBacteriaQueued: 2844,

    triViewReady: 2352,              // FROZEN, report-backed (625 + 1427 + 300)
    triViewReadyLive: 2354,          // live (625 + 1427 + 302)
    embeddingBearingUnits: 5209,     // 625 + 1428 + 3156  (scaling target ~5,200)
    plottedNodes: 5457,              // incl. 248 non-embedded gap rows

    bridgeEdges: 372,                // documented cross-domain k-NN bridge edges
    bridgeNodes: 136,
    caseStudies: 26,
    pocBridgeGenomes: 14,            // POC 662 cohort

    // ESM2 POC geometry (662 cohort)
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
    benchmarkR2: 0.879,              // mcrA/pmoA literature target (Lee et al. 2014, freshwater)
  };

  /* ---- foundation-model + tool stack ---- */
  const stack = {
    foundationModels: ["ESM2 (650M protein LM)", "gLM2 (genomic-context LM)"],
    functional: ["KOfam", "MCycDB", "SCycDB", "dbCAN", "METABOLIC"],
    qcTax: ["CheckM2", "GUNC", "GTDB-Tk", "Bakta", "Prodigal"],
  };

  /* ---- non-negotiable claim boundaries (visible, not buried) ---- */
  const claims = {
    footer:
      "Current results are MAG/proteome-level molecular screening and monitoring prioritization. " +
      "No measured methane flux, final risk scores, A–E tiers, or carbon-credit approval are claimed " +
      "from the molecular atlas alone. A–E risk tiers are target product vocabulary, not yet calibrated. " +
      "Snapshot " + num.snapshot + ".",
    short: "Molecular screening, not calibrated MRV. A–E tiers are target vocabulary, not yet calibrated.",
    boundaries: [
      "Molecular screening and monitoring prioritization, at MAG/proteome grain.",
      "A–E risk tiers are TARGET product vocabulary, explicitly not yet calibrated.",
      "No measured flux, final MRV scores, or carbon-credit approval from the atlas alone.",
      "Rumen↔wetland source confounding stands until source-balanced validation exists.",
    ],
  };

  /* ---- maturity ladder (MRV roadmap Levels 0–5, + 6 horizon) ---- */
  const ladder = [
    { rung: 0, title: "MAG molecular screening", state: "lit",
      unlock: "ESM2, gLM2, functional annotation, QC/taxonomy, and the attestation graph. Done now." },
    { rung: 1, title: "Sample identity & metadata", state: "next",
      unlock: "Map every MAG to a physical sample, site, and project with provenance tiers." },
    { rung: 2, title: "Abundance & community capacity", state: "dim",
      unlock: "Read coverage and relative abundance. Weight genome potential by who is actually there." },
    { rung: 3, title: "Environmental permissiveness", state: "dim",
      unlock: "Salinity, sulfate, redox, temperature, hydroperiod. Does the site let methane express?" },
    { rung: 4, title: "Flux & process validation", state: "dim",
      unlock: "Chamber and eddy-covariance methane flux, incubations. Paired molecular and measured GHG." },
    { rung: 5, title: "Calibrated probabilistic MRV risk", state: "target",
      unlock: "Holdout-validated risk distribution; A–E tiers mapped to thresholds with uncertainty." },
  ];
  const ladderHorizon = "6 · MRV product & audit / registry integration";

  /* ---- attestation evidence chain (Scene 5): one claim, traced ---- */
  const attestation = {
    claimText: "This MAG carries molecular evidence consistent with methane-related functional potential.",
    forbidden: "“This MAG emits methane.”",
    chain: [
      { stage: "Genome",            detail: "MAG / proteome unit · QC + GTDB-Tk taxonomy", node: "MAG" },
      { stage: "Markers",           detail: "mcrA · HdrABC · MCycDB / KOfam direct hits",   node: "MarkerFamily" },
      { stage: "Embedding neighbors", detail: "k-NN in ESM2 space → rumen methanogens",      node: "NEAR_IN_ESM2_SPACE" },
      { stage: "QC gate",           detail: "CheckM2 completeness · GUNC · annotation coverage", node: "ValidationGate" },
      { stage: "Claim boundary",    detail: "allowed wording · blocked wording · upgrade path", node: "Claim" },
    ],
    futureSlots: ["", "", ""], // honest empty optionality: no non-methane application built yet
  };

  /* ---- Scene 7 milestones ---- */
  const timeline = [
    { phase: "Now",   label: "Molecular atlas", detail: "5,209 embedding-bearing units; source-audited; queryable attestation." },
    { phase: "Field", label: "Cispatá Bay validation", detail: "Colombian Caribbean mangrove · paired CH₄-flux sampling." },
    { phase: "Pair",  label: "Paired data", detail: "23 → 80–100 paired molecular + flux records across seasons & habitats." },
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
      copy: "The sediment microbiome is dark matter. Across the field, only a handful of sites carry paired methane-flux data. Today: 23.",
      data: "real-number",
    },
    {
      id: "insight", n: 3, label: "03 · The Insight",
      kicker: "EmergentBiome representation",
      headline: "Foundation models reveal the bridge taxa.",
      copy: "ESM2 proteome geometry resolves the rumen and wetland basins out of noise, then lights the bridge taxa that carry methane knowledge across ecosystems.",
      data: "real-coords",
    },
    {
      id: "atlas", n: 4, label: "04 · The Atlas",
      kicker: "Source-audited functional atlas",
      headline: "2,352 tri-view units, and growing.",
      copy: "Mangrove genomes pour into the manifold. Each unit stacks three views: ESM2 embedding, gLM2 context, and functional annotation.",
      data: "real-counts",
    },
    {
      id: "platform", n: 5, label: "05 · The Platform & the Moat",
      kicker: "EmergentBiome substrate",
      headline: "Every claim traces to its evidence.",
      copy: "Embeddings, an attestation graph, and an agentic pipeline. MethaNet is the first application on a substrate we are building to generalize beyond methane.",
      data: "real-schema",
    },
    {
      id: "ladder", n: 6, label: "06 · The Honest Ladder",
      kicker: "MRV maturity",
      headline: "We are at molecular screening, and we say so.",
      copy: "Calibrated A–E methane risk requires sample mapping, abundance, environment, and field flux. Rung 0 is lit. The rest is the roadmap.",
      data: "real-ladder",
    },
    {
      id: "path", n: 7, label: "07 · The Path & the Ask",
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
    reportDate: "2026-06-25",
    siteUrl: "https://jaygut.github.io/MethaNet/",
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
