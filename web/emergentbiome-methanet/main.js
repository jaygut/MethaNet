/* =====================================================================
   main.js - orchestration
   - injects all copy/numbers from config.js (single source of truth)
   - loads data/atlas.json (real diffusion-map coordinates)
   - one p5 instance per scene (instance mode), lazily created
   - drives scroll progress, scene activation, reduced-motion fallback,
     and offscreen pause/throttle for 60fps.
   ===================================================================== */
(function () {
  "use strict";
  const EB = window.EB;
  const D = window.EBDraw;
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.EBScenes = window.EBScenes || {};

  // scene order incl. hero intro
  const ORDER = ["hero", "stakes", "blindspot", "surveyor", "cheap", "atlas", "engine", "platform", "ladder", "path"];

  // ---------- copy + chrome injection ----------
  function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }

  const READOUTS = {
    stakes: [["~" + EB.num.methaneGWP20 + "×", "CH₄ vs CO₂ over 20 years", true], ["$" + EB.ext.dbcRecordPrice + "/t", "blue-carbon record price, Aug 2025"]],
    blindspot: [["≈ 0", "sites with paired flux + molecular data today", true], ["< " + EB.ext.vm0033SalinityPpt + " ppt", "salinity: no default methane factor"]],
    surveyor: [["evidence profile", "per site with a next measurement", true], ["validation path", "paired field evidence calibrates flux inference", false]],
    cheap: [["r > " + EB.ext.mcraFluxSpearman, "methanogen marker vs measured flux", true], ["salinity", "contextual covariate for field design", false]],
    atlas: [[D.fmt(EB.num.triViewReady), "data-complete tri-view units", true], [D.fmt(EB.num.mechanismComparableTriView), "mechanism-comparable POC"], [D.fmt(EB.num.annotationCompletePendingTriView), "annotation-complete, harmonization pending"], [D.fmt(EB.num.sourceScaffoldTriView), "MUCC source-scaffold tri-view"]],
    engine: [["273×", "N₂O vs CO₂ over 100 years", true], [D.fmt(EB.num.mechanismComparableTriView), "POC units in the comparable methane screen", false], ["0", "field assays for new gases", false]],
    platform: [[D.fmt(EB.num.evidenceAtoms), "evidence atoms", true], [D.fmt(EB.num.nearEsm2Edges), "embedding-neighbor links"]],
    ladder: [["rung 0", "product today · screening + triage", true], ["rungs 1-5", "path to calibrated MRV"]],
    path: [["< " + EB.num.pipelineDays + " days", "raw sequences to report", true], ["Cispatá Bay", "field validation, next rung"]],
  };

  function injectCopy() {
    EB.scenes.forEach((s) => {
      const host = document.querySelector('[data-copy="' + s.id + '"]');
      if (!host) return;
      const accent = document.getElementById("scene-" + s.id).dataset.accent || "--emergence";
      host.style.setProperty("--accent", "var(" + accent + ")");
      const badge = s.data === "illustrative"
        ? '<span class="badge illus">Illustrative</span>'
        : '<span class="badge real">Real data</span>';
      let readoutHtml = "";
      const rs = READOUTS[s.id] || [];
      if (rs.length) {
        readoutHtml = '<div class="copy__readout">' + rs.map((r) =>
          '<span class="readout"><span class="readout__v ' + (r[2] ? "accent" : "") + '">' + r[0] + '</span><span class="readout__k">' + r[1] + '</span></span>'
        ).join("") + "</div>";
      }
      host.innerHTML =
        '<div class="copy__kicker"><span class="smallcaps">' + s.label + "</span>" + badge + "</div>" +
        '<div class="copy__num smallcaps" style="margin-bottom:8px;color:var(--text-muted)">' + s.kicker + "</div>" +
        '<h2 class="copy__headline">' + s.headline + "</h2>" +
        '<p class="copy__body">' + s.copy + "</p>" +
        readoutHtml;
    });
  }

  function injectChrome() {
    // hero (decision-first copy from the single source of truth)
    document.getElementById("heroEyebrow").textContent = EB.hero.eyebrow;
    document.getElementById("heroSub").textContent = EB.hero.sub;
    // header meta
    document.getElementById("headerMeta").innerHTML =
      '<span class="dot">●</span> live snapshot ' + EB.num.snapshot +
      ' · ' + D.fmt(EB.num.embeddingBearingUnits) + ' genomes mapped · ' + EB.num.bridgeEdges + ' bridges';
    // rail
    const rail = document.getElementById("rail");
    EB.scenes.forEach((s) => {
      const item = el('<button class="rail__item" data-target="scene-' + s.id + '" aria-label="' + s.label + '"><span class="rail__num mono">0' + s.n + '</span><span class="rail__dot"></span></button>');
      item.addEventListener("click", () => document.getElementById("scene-" + s.id).scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" }));
      rail.appendChild(item);
    });
    // claim strip
    document.getElementById("claimText").textContent = EB.claims.short;
    document.getElementById("claimDate").textContent = "snapshot " + EB.num.snapshot;
    // ask + factsheet
    document.getElementById("fsDate").textContent = EB.num.snapshot;
    document.getElementById("askBody").innerHTML =
      "MethaNet gives blue-carbon developers, verifiers, raters, and buyers a traceable molecular evidence layer for methane-sensitive diligence. " +
      "The molecular-attestation graph connects <b>" + D.fmt(EB.num.embeddingBearingUnits) + "</b> embedded genomes to functional evidence, provenance, and the next validation action. " +
      "Field validation at Cispatá Bay will add the paired evidence required for calibrated methane MRV.";
    const points = [
      "Field validation at Cispatá Bay (Colombian Caribbean mangrove, VM0033 context)",
      "Pair molecular evidence with field methane flux (target " + EB.num.pairedFluxTargetLo + "–" + EB.num.pairedFluxTargetHi + " samples)",
      "Calibrate the probabilistic risk model to earn the A–E tiers under holdout validation",
      "Harden the attestation graph into a partner-facing, registry-aligned evidence product",
    ];
    document.getElementById("askPoints").innerHTML = points.map((p) => "<li>" + p + "</li>").join("");
    // factsheet rows
    const F = [
      ["Calibration core (rumen + wetland)", D.fmt(EB.num.calibrationCore) + " genomes"],
      ["Mangrove · MSM China 2025", D.fmt(EB.num.msmCandidates) + " genomes (" + D.fmt(EB.num.msmFunctionalComplete) + " annotated)"],
      ["Mangrove · Futian 2026 (Qi et al.)", D.fmt(EB.num.futianRMAGs) + " genomes <span class='in-progress'>(" + D.fmt(EB.num.futianBacteriaComplete) + "/" + D.fmt(EB.num.futianBacteriaTotal) + " bacteria annotated)</span>"],
      ["Data-complete tri-view units", D.fmt(EB.num.triViewReady) + " <span class='in-progress'>(payload completeness; common mechanism comparability is tracked separately)</span>"],
      ["Mechanism-comparable POC tri-view", D.fmt(EB.num.mechanismComparableTriView)],
      ["Annotation-complete, harmonization pending", D.fmt(EB.num.annotationCompletePendingTriView)],
      ["MUCC v1 source-scaffold tri-view", D.fmt(EB.num.sourceScaffoldTriView) + " <span class='in-progress'>(separate mechanism contract)</span>"],
      ["Genomes mapped (molecular space)", D.fmt(EB.num.embeddingBearingUnits)],
      ["MUCC v1 wetland source warehouse", D.fmt(EB.num.muccWarehouseGenomes) + " genomes"],
      ["Total evidence reach", D.fmt(EB.num.warehouseReach) + " genomes"],
      ["Sample metadata recovered", (EB.num.msmSedimentSamples + EB.num.futianSedimentSamples) + " sediment samples · " + EB.num.msmBiosampleRows + " environmental rows"],
      ["Model views per genome", "proteome + genomic-context embeddings, plus function"],
      ["Cross-ecosystem bridge links", D.fmt(EB.num.bridgeEdges)],
      ["Pipeline speed", "&lt;" + EB.num.pipelineDays + " days, vs months of manual analysis"],
      ["Field site", "Cispatá Bay, Colombia"],
    ];
    document.getElementById("factsheet").innerHTML = F.map((r) =>
      '<div class="factsheet__row"><span class="factsheet__k">' + r[0] + '</span><span class="factsheet__v">' + r[1] + "</span></div>"
    ).join("");
    document.getElementById("contact").innerHTML =
      EB.claims.boundaries[0] + " &nbsp;·&nbsp; A–E tiers remain a calibration target. &nbsp;·&nbsp; " +
      'Graph of Life &nbsp;·&nbsp; <a href="mailto:' + EB.links.contactEmail + '">' + EB.links.contactEmail + "</a>";

    // report CTAs - single source of truth is EB.links.report
    const rep = EB.links.report;
    const setHref = (id, href) => { const el = document.getElementById(id); if (el) el.href = href; };
    setHref("headerReportCta", rep);
    setHref("reportCta", rep);
    setHref("contactCta", "mailto:" + EB.links.contactEmail);
    const note = document.getElementById("reportNote");
    if (note) note.innerHTML =
      "The report presents the molecular-attestation knowledge graph, bridge evidence, and sample-readiness ladder behind these numbers. Recomputed " +
      EB.links.reportDate + ", in sync with the live atlas at " + D.fmt(EB.num.triViewReady) + " data-complete tri-view units." +
      ' <a href="' + rep + '" target="_blank" rel="noopener">Open the interactive report ↗</a>';
  }

  // ---------- scene lifecycle ----------
  const scenes = {}; // id -> { ctx, instance, holder, section, inited }

  function makeCtx(id, holder, section, data) {
    return {
      id, holder, section, data, reduced: REDUCED,
      progress: 0, active: false,
      get W() { return holder.clientWidth; },
      get H() { return holder.clientHeight; },
    };
  }

  function initScene(rec) {
    if (rec.inited) return;
    rec.inited = true;
    const factory = window.EBScenes[rec.ctx.id];
    if (!factory) { console.warn("no scene factory for", rec.ctx.id); return; }
    rec.instance = new window.p5((p) => factory(p, rec.ctx), rec.holder);
    if (REDUCED && rec.instance && rec.instance.noLoop) {
      // render a representative static frame; redraw only on scroll (throttled)
      setTimeout(() => { try { rec.instance.noLoop(); rec.instance.redraw(); } catch (e) {} }, 30);
    }
  }

  function setActive(rec, on) {
    rec.ctx.active = on;
    if (!rec.instance) return;
    try {
      if (on && !REDUCED) rec.instance.loop();
      else rec.instance.noLoop();
    } catch (e) {}
  }

  // progress for a sticky scene = how far the tall section has scrolled through the pin
  function computeProgress(section) {
    const vh = window.innerHeight;
    const rect = section.getBoundingClientRect();
    const total = rect.height - vh; // scrollable distance while pinned
    if (total <= 0) return D.clamp(rect.top <= 0 ? 1 : 0);
    return D.clamp(-rect.top / total);
  }

  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      ORDER.forEach((id) => {
        const rec = scenes[id];
        if (!rec) return;
        const pr = computeProgress(rec.section);
        rec.ctx.progress = pr;
        if (REDUCED && rec.inited && rec.ctx.active && rec.instance) {
          try { rec.instance.redraw(); } catch (e) {}
        }
      });
      updateRail();
    });
  }

  function updateRail() {
    let activeId = null;
    for (const id of ORDER) {
      const rec = scenes[id];
      if (rec && rec.ctx.active && id !== "hero") activeId = id;
    }
    document.querySelectorAll(".rail__item").forEach((it) => {
      it.setAttribute("aria-current", it.dataset.target === "scene-" + activeId ? "true" : "false");
    });
  }

  function boot(data) {
    ORDER.forEach((id) => {
      const section = document.getElementById("scene-" + id);
      const holder = document.getElementById("canvas-" + id);
      if (!section || !holder) return;
      const ctx = makeCtx(id, holder, section, data);
      scenes[id] = { ctx, holder, section, instance: null, inited: false };
    });

    // lazy init + activation via IntersectionObserver on the stage/section
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        const id = e.target.dataset.scene;
        const rec = scenes[id];
        if (!rec) return;
        if (e.isIntersecting) { initScene(rec); setActive(rec, true); }
        else setActive(rec, false);
      });
    }, { rootMargin: "10% 0px 10% 0px", threshold: 0.01 });

    ORDER.forEach((id) => {
      const rec = scenes[id];
      if (rec) io.observe(rec.section);
    });

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", () => { ORDER.forEach((id) => { const r = scenes[id]; if (r && r.instance && r.instance.windowResized) try { r.instance.windowResized(); } catch (e) {} }); onScroll(); });
    document.addEventListener("visibilitychange", () => {
      const hidden = document.hidden;
      ORDER.forEach((id) => { const r = scenes[id]; if (r && r.instance) { try { hidden ? r.instance.noLoop() : (r.ctx.active && !REDUCED && r.instance.loop()); } catch (e) {} } });
    });
    onScroll();
  }

  // ---------- start ----------
  function start() {
    injectCopy();
    injectChrome();
    fetch("data/atlas.json")
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then((atlas) => boot({ atlas }))
      .catch((err) => {
        console.warn("atlas.json not loaded (serve over http, not file://):", err);
        boot({ atlas: null }); // scenes that need atlas show a graceful note
      });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
