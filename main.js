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
    stakes: [["~" + EB.num.methaneGWP20 + "×", "CH₄ vs CO₂ over 20 years", true], ["field flux", "needed for site-specific methane balance"]],
    blindspot: [["0 accepted", "exact molecular + environment + process joins", true], ["next step", "pair samples with compatible field outcomes"]],
    surveyor: [["1 real MAG", "source-audited review example", true], ["0 matched flux", "for this candidate record"]],
    cheap: [["salinity", "field context, not a process rate", true], ["genome evidence", "molecular screening, not site risk"]],
    atlas: [[D.fmt(EB.num.triViewReady), "data-complete MAG/proteome records", true], [D.fmt(EB.num.bridgeEdges), "displayed map links"], [D.fmt(EB.num.highlightedCandidateLinks), "highlighted nearest-core candidate links"], ["0", "standardized reciprocal rumen top-35 pairs"]],
    engine: [["methane", "current pathway-screening focus", true], ["other lenses", "research options requiring separate validation"], [D.fmt(EB.num.mechanismComparableTriView), "cross-lane mechanism-comparable units"]],
    platform: [[D.fmt(EB.num.magNodes), "queryable POC graph records", true], [D.fmt(EB.num.evidenceAtoms), "POC evidence atoms"]],
    ladder: [["rung 0", "available now · evidence review + triage", true], ["rungs 1-5", "path to calibrated MRV"]],
    path: [["partner cohort", "exact sample + field process pairing", true], ["held-out test", "required before calibrated risk"]],
  };

  function injectCopy() {
    EB.scenes.forEach((s) => {
      const host = document.querySelector('[data-copy="' + s.id + '"]');
      if (!host) return;
      const accent = document.getElementById("scene-" + s.id).dataset.accent || "--emergence";
      host.style.setProperty("--accent", "var(" + accent + ")");
      const badge = s.data === "illustrative"
        ? '<span class="badge illus">Illustrative</span>'
        : s.data === "mixed-real-and-illustrative"
          ? '<span class="badge illus">Real anchor + illustrative view</span>'
          : s.data === "roadmap"
            ? '<span class="badge illus">Roadmap</span>'
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
    document.getElementById("heroDefs").innerHTML = EB.terminology.filter((t) => t.hero).map((t) =>
      '<span class="hero__def"><b>' + t.term + '</b> · ' + t.full + "</span>"
    ).join("");
    // header meta
    document.getElementById("headerMeta").innerHTML =
      '<span class="dot">●</span> frozen ' + EB.num.snapshot +
      ' · ' + D.fmt(EB.num.embeddingBearingUnits) + ' MAG/proteome records · ' + D.fmt(EB.num.bridgeEdges) + ' displayed links';
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
      "The EmergentBiome Molecular Atlas gives blue-carbon developers, verifiers, and research partners a source-audited way to review methane-pathway hypotheses and choose measurements. " +
      "Its frozen release has <b>" + D.fmt(EB.num.triViewReady) + "</b> data-complete MAG/proteome records across separate evidence contracts. " +
      "A <b>" + D.fmt(EB.num.magNodes) + "</b>-record queryable POC evidence graph demonstrates traceable claim review; the full-atlas graph and calibrated risk model require further work.";
    const points = [
      "Resolve exact physical sample, date, depth, and MAG links while preserving unresolved records",
      "Pair abundance, environmental context, and compatible methane-process measurements",
      "Test whether molecular evidence improves an environment-only baseline at held-out sites and seasons",
      "Publish uncertainty-calibrated risk only after those external validation gates pass",
    ];
    document.getElementById("askPoints").innerHTML = points.map((p) => "<li>" + p + "</li>").join("");
    // factsheet rows
    const F = [
      ["Registered release records", D.fmt(EB.num.warehouseReach) + " <span class='in-progress'>(including 255 explicit source gaps)</span>"],
      ["Data-complete MAG/proteome tri-views", D.fmt(EB.num.triViewReady) + " <span class='in-progress'>(payload completeness, not mechanism equivalence)</span>"],
      ["Pipeline-normalized tri-views", D.fmt(EB.num.pipelineNormalizedTriView) + " <span class='in-progress'>(cross-lane comparison pending)</span>"],
      ["Old Woman Creek source-scaffold tri-views", D.fmt(EB.num.sourceScaffoldTriView) + " <span class='in-progress'>(distinct contract)</span>"],
      ["Cross-lane mechanism-comparable units", D.fmt(EB.num.mechanismComparableTriView)],
      ["Queryable POC evidence graph", D.fmt(EB.num.magNodes) + " records <span class='in-progress'>(atlas-wide extension planned)</span>"],
      ["Displayed atlas map links", D.fmt(EB.num.bridgeEdges) + " <span class='in-progress'>(2,200 sampled neighbors + 26 candidate links)</span>"],
      ["Exact molecular + environment + process joins", "0 accepted <span class='in-progress'>(metadata-rich contexts are not scored samples)</span>"],
      ["Field-validation need", "Exact linked samples and matched process observations"],
    ];
    document.getElementById("factsheet").innerHTML = F.map((r) =>
      '<div class="factsheet__row"><span class="factsheet__k">' + r[0] + '</span><span class="factsheet__v">' + r[1] + "</span></div>"
    ).join("");
    const mangroveRecords = EB.ecosystems[2].count + EB.ecosystems[3].count;
    const evidenceCards = [
      {
        metric: D.fmt(EB.num.nearestCoreWetland) + " wetland · " + D.fmt(EB.num.nearestCoreMangrove) + " mangrove",
        title: "One-way nearest-core matches",
        detail: "Among " + D.fmt(EB.ecosystems[1].count) + " wetland and " + D.fmt(mangroveRecords) + " mangrove records, these have a rumen raw-cosine nearest neighbor within the 625-record POC core. They nominate review hypotheses, not transfer.",
      },
      {
        metric: EB.num.nearestCoreCandidates + " of " + EB.num.candidateCards,
        title: "Selected candidate records",
        detail: "Selected wetland/mangrove cards with a rumen nearest-core match. The map draws 26 highlighted candidate links plus 2,200 sampled cross-domain neighbor links, not every nearest-core match.",
      },
      {
        metric: String(EB.num.standardizedRumenReciprocalPairs),
        title: "Standardized reciprocal rumen pairs",
        detail: "Across the full atlas, no rumen–wetland or rumen–mangrove pair is reciprocal within each other's top 35 after per-dimension standardization. This is a different comparison from the one-way nearest-core matches.",
      },
    ];
    const evidenceGrid = document.getElementById("evidenceSummaryGrid");
    evidenceCards.forEach((card) => {
      const item = document.createElement("article");
      item.className = "evidence-summary__card";
      const metric = document.createElement("p"); metric.className = "evidence-summary__metric"; metric.textContent = card.metric;
      const title = document.createElement("h4"); title.textContent = card.title;
      const detail = document.createElement("p"); detail.textContent = card.detail;
      item.append(metric, title, detail);
      evidenceGrid.appendChild(item);
    });
    document.getElementById("terminologyList").innerHTML = EB.terminology.map((t) =>
      '<div class="terminology__item"><dt><span class="terminology__term">' + t.term + '</span><span class="terminology__full">' + t.full + '</span></dt><dd>' + t.detail + "</dd></div>"
    ).join("");
    document.getElementById("contact").innerHTML =
      EB.claims.boundaries[0] + " &nbsp;·&nbsp; A–E tiers remain a calibration target. &nbsp;·&nbsp; " +
      'Graph of Life &nbsp;·&nbsp; <a href="mailto:' + EB.links.contactEmail + '">' + EB.links.contactEmail + "</a>";

    // Primary journey stays on-page; the bundled report expands the same freeze.
    const rep = EB.links.report;
    const setHref = (id, href) => { const el = document.getElementById(id); if (el) el.href = href; };
    setHref("headerAtlasCta", "#scene-atlas");
    setHref("heroAtlasCta", "#scene-atlas");
    setHref("atlasCta", "#scene-atlas");
    setHref("headerReportCta", rep);
    setHref("reportCta", rep);
    setHref("contactCta", "mailto:" + EB.links.contactEmail);
    document.querySelectorAll("[data-engine-lens]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-engine-lens]").forEach((candidate) => {
          candidate.setAttribute("aria-pressed", String(candidate === button));
        });
        document.dispatchEvent(new CustomEvent("emergentbiome:engine-lens", {
          detail: { index: Number(button.dataset.engineLens) },
        }));
      });
    });
    const note = document.getElementById("reportNote");
    if (note) note.innerHTML =
      'The <a href="' + rep + '" target="_blank" rel="noopener">reconciled technical report ↗</a> expands the same frozen ' + EB.num.snapshot +
      " release with methods, tables, and evidence limits.";

    const cardPanel = document.getElementById("candidateEvidencePanel");
    const cardButtons = Array.from(document.querySelectorAll("[data-card-view]"));
    function showCandidateView(key) {
      const view = EB.candidateExample.views[key];
      if (!view || !cardPanel) return;
      cardButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.cardView === key)));
      const heading = document.createElement("h4"); heading.textContent = view.title;
      const list = document.createElement("ul");
      view.points.forEach((point) => { const li = document.createElement("li"); li.textContent = point; list.appendChild(li); });
      cardPanel.setAttribute("aria-label", view.title);
      cardPanel.replaceChildren(heading, list);
    }
    cardButtons.forEach((button) => button.addEventListener("click", () => showCandidateView(button.dataset.cardView)));
    showCandidateView("recorded");
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
