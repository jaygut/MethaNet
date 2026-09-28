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
   Voice: plain words first, technical terms in the glossary and report. One
   caveat per point, placed where it changes the reading. Every external fact
   carries a source line. No em-dashes in any public copy. Decision-first
   headlines. Bodies up to about 60 words; scenes that explain the atlas may
   run to about 80.
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
    { key: "wetland",         code: 1, label: "Wetland references", sub: "POC wetland + MUCC v1", color: color.wetland,      count: 2608 },
    { key: "mangrove_msm",    code: 2, label: "Mangrove · MSM",  sub: "China 2025 expansion",  color: color.mangroveMsm,    count: 1428 },
    { key: "mangrove_futian", code: 3, label: "Mangrove · Futian", sub: "2026 expansion (Qi et al.)", color: color.mangroveFutian, count: 3156 },
  ];

  /* ---- verified headline numbers (see DIGEST.md §1) ---- */
  const num = {
    snapshot: "2026-08-10",
    snapshotLiveUTC: "2026-08-10 controlled-diligence audit",
    snapshotFreezeUTC: "2026-08-10 release-ledger freeze",
    // Separately reviewed three-case extension (scenes 07-08); its own date and
    // scope, never merged into the atlas counts. Matches the public case files.
    caseEvidenceDate: "2026-09-26",

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
    pocBridgeGenomes: 14,            // POC 662 cohort
    nearestCoreWetland: 2434,        // wetland records outside the core whose raw-cosine nearest POC-core match is rumen
    wetlandOutsideCore: 2501,        // wetland records outside the 625-record core; the 107 core wetland records match themselves
    nearestCoreMangrove: 4475,       // of 4,584 mangrove records (none is in the core)
    nearestCoreMedianCosine: 0.983,  // median raw cosine to the nearest core member, 7,085 records outside the core
    randomPairMedianCosine: 0.994,   // median raw cosine of random atlas pairs (embedding geometry audit)
    nearestCoreCandidates: 26,       // of 27 selected wetland/mangrove candidate cards; the 27th is a core record matching itself
    candidateCards: 27,
    sampledNeighborLinks: 2200,
    crossHabitatNeighborEdges: 57193, // directed raw-cosine top-35 edges that cross habitats (embedding geometry audit)
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
    eyebrow: "Methane evidence for blue carbon, from microbial DNA",
    // Problem, then value proposition: the two sentences a visitor must keep.
    lead:
      "Microbes in wetland sediments can release methane that cancels part of the climate benefit of the carbon those wetlands store. " +
      "EmergentBiome turns their DNA into evidence you can check, and shows which field measurement would settle what DNA alone cannot.",
    sub:
      "Available now: 7,710 genome records from mangroves, wetlands and a rumen reference set, and three worked evidence cases. " +
      "Next: a proposed field study to test whether molecular data improves methane prediction.",
    // Two dated scopes: the frozen August atlas and the September case reviews.
    release: "Atlas released 10 August 2026 · Case reviews added 26 September 2026",
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
      chip: "7,710 genome records mapped by their proteins",
      detail: "A frozen map of 7,710 genome records, each with protein, genomic-context and functional evidence. Complete data is not the same as comparable data: comparing sources needs a shared annotation standard.",
      hero: true,
    },
    {
      term: "Graph",
      full: "EmergentBiome evidence graph",
      chip: "each claim tied to its source and status",
      detail: "The evidence model behind the atlas and the cases: a 662-record proof-of-concept graph, and a formal ontology slice with selected evidence for 145 genome records. The explorer shows three curated cases from it.",
      hero: true,
    },
    {
      term: "MAG",
      full: "Metagenome-assembled genome",
      detail: "A microbial genome reconstructed from the mixed DNA of a sediment or gut sample. It shows what an organism could do; judging its role in a sample needs abundance data.",
    },
    {
      term: "MRV",
      full: "Monitoring, reporting and verification",
      detail: "How climate projects demonstrate their outcomes. A calibrated methane-risk layer for MRV needs exact sample links, abundance, environmental context, uncertainty and field validation.",
    },
    {
      term: "ESM-2",
      full: "Protein language model",
      detail: "A model that turns each protein sequence into numbers. Combined across a genome's proteins, it places genomes with similar protein sets near each other on the atlas.",
    },
    {
      term: "gLM2",
      full: "Genomic-context language model",
      detail: "A model that reads genes in their neighborhood order on the genome. Its scores are compared only within the same analysis protocol.",
    },
    {
      term: "Tri-view",
      full: "Three evidence views of one genome",
      detail: "Protein embedding, genomic context and functional annotation. The atlas records whether all three exist separately from whether they can be compared across sources.",
    },
    {
      term: "MUCC v1",
      full: "Old Woman Creek wetland data, Ohio",
      detail: "Genomes and RNA data from a freshwater coastal wetland on Lake Erie, annotated by the source's own pipeline and kept separate from the shared pipeline.",
    },
    {
      term: "VM0033",
      full: "Verra methodology for tidal wetland restoration",
      detail: "The carbon-crediting method for tidal wetland and seagrass restoration. It allows a default methane value only where salinity is above 18 ppt.",
    },
    {
      term: "Sample-event",
      full: "One planned collection in the proposed study",
      detail: "A sediment metagenome from one microsite in one campaign, paired with chamber methane flux, chemistry and hydrology. Revisits to a microsite are repeated measurements, not independent replicates.",
    },
  ];

  /* ---- non-negotiable claim boundaries (visible, not buried) ---- */
  const claims = {
    footer:
      "Current results are molecular screening, candidate triage, evidence-card review, and monitoring prioritization at MAG/proteome grain. " +
      "Measured methane flux, final risk scores, A–E tiers, and carbon-credit decisions require paired abundance, environmental, uncertainty, and field-validation evidence. " +
      "A–E risk tiers remain target product vocabulary while calibration is completed. " +
      "Snapshot " + num.snapshot + ".",
    short: "Genome-level screening and review only. Methane flux and calibrated site risk need paired field validation.",
    boundaries: [
      "Genome-level molecular screening, candidate review and measurement planning.",
      "A–E risk tiers are target product vocabulary. Calibration requires paired validation.",
      "Measured flux, final MRV scores, and carbon-credit decisions require evidence beyond the molecular map.",
      "Reference-to-target signals stay provisional until source-balanced validation exists.",
    ],
  };

  /* ---- evidence-maturity ladder (MRV roadmap levels 0–5, + 6 horizon) ---- */
  const ladder = [
    { rung: 0, title: "Molecular review", state: "lit",
      unlock: "Protein fingerprints, genome context, functional annotation, quality control, atlas records and a proof-of-concept evidence graph. Available now." },
    { rung: 1, title: "Exact sample links", state: "progress",
      unlock: "Underway: sample and site context recovered for part of the mangrove records. Exact genome-to-sample links are next." },
    { rung: 2, title: "Abundance", state: "dim",
      unlock: "Read coverage and relative abundance, so genome potential is weighted by which microbes are actually present." },
    { rung: 3, title: "Site conditions", state: "dim",
      unlock: "Salinity, sulfate, redox, temperature and hydroperiod: the conditions that allow or suppress methane production." },
    { rung: 4, title: "Measured flux", state: "dim",
      unlock: "Chamber or eddy-covariance methane flux and incubations, paired with the molecular evidence." },
    { rung: 5, title: "Calibrated risk", state: "target",
      unlock: "A risk model tested on held-out sites, with A–E tiers mapped to thresholds and uncertainty." },
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
        title: "What is recorded",
        points: [
          "Genome quality reported by the source: 94.89% complete, 1.14% contamination, checked against the archived genome.",
          "Marker transcripts were detected in the source's processed RNA data. Detection shows presence, not how active the pathway is.",
          "Closest genome in the 625-genome reference core: a rumen genome, raw cosine 0.9843. A one-way lead for review.",
        ],
      },
      pending: {
        title: "What is still unresolved",
        points: [
          "No exact sample, collection date or depth is linked to this genome yet.",
          "No abundance, matched environmental data or methane-process measurement is joined to it.",
          "Its annotations come from the source's own pipeline, which is not yet comparable with the shared pipeline.",
        ],
      },
      next: {
        title: "The next measurement",
        points: [
          "Link the genome to its physical sample, with date and depth.",
          "Measure abundance and environment alongside a compatible methane-process measurement.",
          "Then review the marker genes and test whether the signal holds across sources before any ecological reading.",
        ],
      },
    },
  };

  /* ---- Proposed field study (scene 10 and the closing ask) ----
     A prospective design, not collected data, and conditional on funding, site
     access and permits. Source: the submitted proposal's research plan and
     docs/research/mvo_application_revision/whitepaper.md section 7. It stays
     outside the frozen atlas counts above. */
  const study = {
    stages: ["Recently reconnected", "Transitional", "Mature or reference"],
    salinityPositions: 4,
    plotsPerCombination: 3,          // replicate plots per stage and salinity position
    plots: 36,
    micrositesPerPlot: 2,            // two tidal heights per plot
    microsites: 72,
    campaigns: ["Wet season", "Dry season"],
    sampleEvents: 144,               // repeated visits, not 144 independent replicates
  };

  /* ---- 10 scenes: source-backed atlas and a separately scoped case explorer. ---- */
  const scenes = [
    {
      id: "stakes", n: 1, label: "01 · The Climate Question",
      kicker: "Blue carbon",
      headline: "Methane can erode a wetland's climate benefit.",
      copy: "Wetland sediments store carbon, but their microbes can also make methane, which warms about 80 times more than CO₂ over 20 years, tonne for tonne. " +
        "Where the water is fresh or brackish, or cut off from the tide, those emissions can offset much of the benefit of the carbon the site stores.",
      source: "Sources: IPCC AR6 WG1 (warming potential); Poffenbarger et al. 2011, Wetlands; Kroeger et al. 2017, Scientific Reports.",
      data: "mixed-real-and-illustrative",
    },
    {
      id: "blindspot", n: 2, label: "02 · The Measurement Gap",
      kicker: "Verification today",
      headline: "Methane evidence is thinnest where it matters most.",
      copy: "Under Verra's VM0033 methodology, only tidal wetlands above 18 ppt salinity may use a default methane value; fresher sites must estimate it from field data, published values, proxies or models. " +
        "Field measurements are costly, cover little ground, and are rarely paired with the sediment DNA. " +
        "In this atlas, no wetland or mangrove genome yet has a methane measurement from the same sample: zero pairs, not zero emissions.",
      source: "Source: Verra VM0033, default factor from Poffenbarger et al. 2011.",
      data: "mixed-real-and-illustrative",
    },
    {
      id: "atlas", n: 3, label: "03 · The Molecular Atlas",
      kicker: "The data asset",
      headline: "7,710 genome records, mapped by their proteins.",
      copy: "Each point is a genome rebuilt from environmental DNA, drawn from three settings: the rumen, a well-studied methane system used as a reference; " +
        "freshwater wetlands, mostly Old Woman Creek in Ohio; and mangrove sediments along the coast of China. " +
        "A protein language model condenses each genome's proteins into one fingerprint, so genomes with similar protein sets sit close together. " +
        "Nearness is a lead, not shared function: the map's views test which resemblances hold up.",
      data: "real-coords",
    },
    {
      id: "surveyor", n: 4, label: "04 · The Evidence Card",
      kicker: "What you get for each genome",
      headline: "Every genome comes with its evidence, and its gaps.",
      copy: "Here is one wetland genome from the atlas. Its card separates what is recorded from what is still unknown, and names the measurement that would move it forward. " +
        "This one is 95% complete and its marker transcripts were detected, yet no sample, depth or methane measurement is linked to it.",
      data: "real-schema",
    },
    {
      id: "cheap", n: 5, label: "05 · Beyond Salinity",
      kicker: "Why genomes add information",
      headline: "Salinity sets the baseline. Microbes set the exceptions.",
      copy: "Saltier wetlands usually emit less methane, because sulfate-reducing microbes outcompete most methane makers. " +
        "Some methanogens use substrates that sulfate reducers ignore, and keep producing methane in salty sediments. " +
        "A salinity reading cannot see them. Genome evidence can reveal the microbes involved, which tells a field team where a measurement is most informative.",
      source: "Sources: Poffenbarger et al. 2011, Wetlands; Krause and Treude 2021, Geochimica et Cosmochimica Acta.",
      data: "illustrative",
    },
    {
      id: "engine", n: 6, label: "06 · Evidence Scope",
      kicker: "What is screened today",
      headline: "Screened for methane. Compared like with like.",
      copy: "Every genome on the map was screened for methane-cycle genes through one of two annotation routes: a shared pipeline for 5,209 records, and the source's own annotations for 2,501 Old Woman Creek records. " +
        "Until the two are harmonized, methane evidence is compared within a route, never ranked across the map. " +
        "Nitrous oxide and sulfur lenses are planned; each will need its own validation.",
      data: "real-coords",
    },
    {
      // Scenes 07 and 08 render their own HTML in index.html. These entries
      // name them in the rail and skip links; keep each label equal to the
      // scene's visible kicker.
      id: "platform", n: 7, label: "07 · Molecular Evidence, Practical Decisions",
      kicker: "Three September evidence cases",
      headline: "Connect the molecule to the measurement.",
      copy: "Check what a gene can mean. Pin down the sample and depth it came from. Choose the test that settles it.",
      data: "illustrative",
    },
    {
      id: "network", n: 8, label: "08 · Explore the Evidence",
      kicker: "The evidence model",
      headline: "Every interpretation has an evidence trail.",
      copy: "A formal evidence model keeps genes, samples, measurements and hypotheses distinct, so each case shows its sources, its open questions and the reason a claim is on hold.",
      data: "real-schema",
    },
    {
      id: "ladder", n: 9, label: "09 · Validation Path",
      kicker: "Evidence maturity",
      headline: "Molecular review works today. Calibrated risk is five rungs up.",
      copy: "Each rung adds evidence the one below cannot supply: exact sample links, community abundance, environmental conditions, measured flux, " +
        "and finally a model tested on sites it has never seen. Until then, A–E risk tiers remain a target.",
      data: "real-ladder",
    },
    {
      id: "path", n: 10, label: "10 · Partnership Path",
      kicker: "Proposed field study",
      headline: "Pair molecular evidence with field outcomes.",
      copy: "A proposed two-season study would pair each sediment metagenome with chamber methane flux, chemistry and hydrology at two microsites in each of 36 plots.",
      data: "proposed",
    },
  ];

  /* ---- outbound links (single place to update the published report path) ---- */
  const links = {
    report: "report/",                 // stable alias published with the landing bundle
    reportName: "EmergentBiome Molecular Atlas technical report",
    reportDate: "2026-09-28",
    siteUrl: "https://emergentbiome.earth/",
    contactEmails: ["jay@ecosphereblue.earth", "aphilosof@ecosphereblue.earth"],
    organizationUrl: "https://www.ecosphereblue.earth/",
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
    attestation, candidateExample, study, scenes, brand, links,
    // global seed for all reproducible sketches
    seed: 0xE13B10,
  };
})();
