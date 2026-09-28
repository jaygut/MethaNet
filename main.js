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
  const ORDER = ["hero", "stakes", "blindspot", "atlas", "surveyor", "cheap", "engine", "platform", "network", "ladder", "path"];

  // ---------- copy + chrome injection ----------
  function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }

  const READOUTS = {
    stakes: [["~" + EB.num.methaneGWP20 + "×", "CH₄ vs CO₂ over 20 years", true], ["field flux", "needed for a site's methane balance"]],
    blindspot: [["0 verified pairs", "wetland/mangrove genome-to-flux pairs in this atlas", true], ["no default", "VM0033 methane value below 18 ppt"]],
    atlas: [[D.fmt(EB.num.triViewReady), "genome records on the map", true], [D.fmt(EB.num.sampledNeighborLinks), "neighbor links drawn"], [D.fmt(EB.num.highlightedCandidateLinks), "selected nearest-reference links"], [String(EB.num.standardizedRumenReciprocalPairs), "rumen matches that stay mutual after standardizing"]],
    surveyor: [["1 real genome", "from the August atlas", true], ["0 matched flux", "measurements for this record"]],
    cheap: [["salinity", "sets the baseline", true], ["genome evidence", "flags exceptions worth measuring"]],
    engine: [[D.fmt(EB.num.triViewReady), "records screened for methane-cycle genes", true], ["2", "annotation routes, compared separately"], [D.fmt(EB.num.mechanismComparableTriView), "records with a cross-route methane score"]],
    ladder: [["rung 0", "available now: review and triage", true], ["rungs 1–5", "the path to calibrated risk"]],
    path: [[EB.study.sampleEvents + " planned", "sample-events, not independent replicates", true], [EB.study.plots + " plots", "the replicate unit in analysis"]],
  };

  // The claim-bar chip names the dated scope of the scene in view.
  const SCOPE = {
    atlas: "atlas data · " + EB.num.snapshot,
    cases: "case evidence · " + EB.num.caseEvidenceDate,
    study: "proposed study · not yet collected",
    both: "atlas " + EB.num.snapshot + " · cases " + EB.num.caseEvidenceDate,
  };
  const SCENE_SCOPE = { platform: "cases", network: "cases", path: "study" };

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
            : s.data === "proposed"
              ? '<span class="badge illus">Proposed study</span>'
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
        readoutHtml +
        (s.source ? '<p class="copy__source">' + s.source + "</p>" : "");
    });
  }

  function injectChrome() {
    // hero (decision-first copy from the single source of truth)
    // Keep each dated scope phrase on one line when the eyebrow wraps.
    document.getElementById("heroEyebrow").innerHTML = EB.hero.eyebrow.split(" · ")
      .map((part) => '<span class="hero__eyebrow-part">' + part + "</span>").join(" · ");
    document.getElementById("heroLead").textContent = EB.hero.lead;
    document.getElementById("heroSub").textContent = EB.hero.sub;
    document.getElementById("heroRelease").textContent = EB.hero.release;
    // Hero terms carry a plain definition, so the opening explains its own words.
    document.getElementById("heroDefs").innerHTML = EB.terminology.filter((t) => t.hero).map((t) =>
      '<span class="hero__def"><b>' + t.term + '</b> · ' + (t.chip || t.full) + "</span>"
    ).join("");
    // header meta: both dated scopes, so later September cases never read as
    // part of the frozen August atlas counts
    document.getElementById("headerMeta").innerHTML =
      '<span class="dot">●</span> atlas ' + EB.num.snapshot +
      ' · ' + D.fmt(EB.num.embeddingBearingUnits) + ' genome records · case reviews ' + EB.num.caseEvidenceDate;
    // rail
    const rail = document.getElementById("rail");
    EB.scenes.forEach((s) => {
      const item = el('<button class="rail__item" data-target="scene-' + s.id + '" aria-label="' + s.label + '"><span class="rail__num mono">' + String(s.n).padStart(2, "0") + '</span><span class="rail__dot"></span></button>');
      item.addEventListener("click", () => document.getElementById("scene-" + s.id).scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" }));
      rail.appendChild(item);
    });
    // claim strip
    document.getElementById("claimText").textContent = EB.claims.short;
    document.getElementById("claimDate").textContent = SCOPE.both;
    // ask + factsheet
    document.getElementById("fsDate").textContent = EB.num.snapshot;
    document.getElementById("askBody").innerHTML =
      "EmergentBiome gives blue-carbon developers, verifiers and researchers a source-audited way to review methane-pathway evidence and decide what to measure. " +
      "We are seeking research funding and field partners for a proposed two-season mangrove study. Fieldwork depends on funding, site access and permits.";
    const S = EB.study;
    const points = [
      "Pair " + S.sampleEvents + " planned sediment metagenomes with chamber methane flux, chemistry and hydrology across three restoration stages, " +
        "four salinity positions and two seasons",
      "Test whether molecular features predict net methane flux better than environmental covariates and a reviewed methane-marker baseline, on the same held-out records",
      "Freeze each model before its flux outcomes are unblinded: the first campaign tests a new site; the second tests seasonal transfer at the same points, not independent-site validation",
      "Release the paired data and reproducible workflows under the agreed access and sharing terms, including if molecular features add little",
    ];
    document.getElementById("askPoints").innerHTML = points.map((p) => "<li>" + p + "</li>").join("");
    document.getElementById("askStudyNote").textContent =
      S.sampleEvents + " sample-events = " + S.plots + " plots × " + S.micrositesPerPlot + " microsites × " + S.campaigns.length + " seasons. " +
      "Revisits are repeated measurements, not independent replicates; the " + S.plotsPerCombination +
      " plots at each stage and salinity position are the replicate unit.";
    // factsheet rows
    const F = [
      ["Genome records registered", D.fmt(EB.num.warehouseReach) + " <span class='in-progress'>(255 documented source gaps)</span>"],
      ["With all three evidence views", D.fmt(EB.num.triViewReady)],
      ["Annotated through the shared pipeline", D.fmt(EB.num.pipelineNormalizedTriView)],
      ["Annotated by the Old Woman Creek source", D.fmt(EB.num.sourceScaffoldTriView)],
      ["With a cross-source methane score", D.fmt(EB.num.mechanismComparableTriView) + " <span class='in-progress'>(awaits harmonization)</span>"],
      ["Proof-of-concept evidence graph", D.fmt(EB.num.magNodes) + " records <span class='in-progress'>(atlas-wide extension planned)</span>"],
      ["Links drawn on the map", D.fmt(EB.num.bridgeEdges) + " <span class='in-progress'>(" + D.fmt(EB.num.sampledNeighborLinks) + " neighbors + " + EB.num.highlightedCandidateLinks + " nearest-reference)</span>"],
      ["Exact genome + environment + flux joins", "0 <span class='in-progress'>(context exists for some samples; no flux yet)</span>"],
      ["What field validation needs", "Exact sample links and matched process measurements"],
    ];
    document.getElementById("factsheet").innerHTML = F.map((r) =>
      '<div class="factsheet__row"><span class="factsheet__k">' + r[0] + '</span><span class="factsheet__v">' + r[1] + "</span></div>"
    ).join("");
    const mangroveRecords = EB.ecosystems[2].count + EB.ecosystems[3].count;
    const evidenceCards = [
      {
        metric: D.fmt(EB.num.nearestCoreWetland) + " wetland · " + D.fmt(EB.num.nearestCoreMangrove) + " mangrove",
        title: "Point first to a rumen genome",
        detail: "Of the " + D.fmt(EB.num.wetlandOutsideCore) + " wetland and " + D.fmt(mangroveRecords) + " mangrove records outside the 625-genome reference core, these have a rumen genome as their closest core match. " +
          "The core is mostly rumen (518 genomes), and the matches are weak: their median similarity (cosine " + EB.num.nearestCoreMedianCosine + ") is below that of two random atlas genomes (" + EB.num.randomPairMedianCosine + "). Each is a lead for review, not evidence of shared function.",
      },
      {
        metric: EB.num.nearestCoreCandidates + " of " + EB.num.candidateCards,
        title: "Selected candidate links",
        detail: "Of " + EB.num.candidateCards + " selected wetland and mangrove candidates, " + EB.num.nearestCoreCandidates + " point to a rumen genome; the other sits inside the reference core and matches itself. The map draws these " + EB.num.highlightedCandidateLinks + " gold links and " + D.fmt(EB.num.sampledNeighborLinks) + " sampled neighbor links, not every match.",
      },
      {
        metric: String(EB.num.standardizedRumenReciprocalPairs),
        title: "Mutual rumen neighbors",
        detail: "After standardizing the representation, no rumen genome and wetland or mangrove genome appear in each other's 35 closest neighbors. This stricter test is separate from the one-way matches.",
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
      '<span class="contact__people"><a href="' + EB.links.organizationUrl + '">Ecosphere Blue</a> &nbsp;·&nbsp; ' +
      EB.links.contactEmails.map((email) => '<a href="mailto:' + email + '">' + email + "</a>").join(" &nbsp;·&nbsp; ") + "</span>" +
      '<span class="contact__boundary">' + EB.claims.boundaries[0] + " A–E tiers remain a calibration target.</span>";

    // Primary journey stays on-page; the bundled report expands the same freeze.
    const rep = EB.links.report;
    const setHref = (id, href) => { const el = document.getElementById(id); if (el) el.href = href; };
    setHref("headerAtlasCta", "#scene-atlas");
    setHref("heroAtlasCta", "#scene-atlas");
    setHref("casesCta", "#scene-network");
    setHref("headerReportCta", rep);
    setHref("reportCta", rep);
    setHref("contactCta", "mailto:" + EB.links.contactEmails.join(","));
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
      'The <a href="' + rep + '" target="_blank" rel="noopener">technical report ↗</a> covers the same frozen ' + EB.num.snapshot +
      " atlas in full, with methods, tables and evidence limits.";

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
    if (rec.section.dataset.nativeScene) return;
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
      if (!rec) continue;
      const bounds = rec.section.getBoundingClientRect();
      if (bounds.top <= innerHeight * 0.5 && bounds.bottom > innerHeight * 0.5) {
        activeId = id === "hero" ? null : id;
        break;
      }
    }
    document.querySelectorAll(".rail__item").forEach((it) => {
      it.setAttribute("aria-current", it.dataset.target === "scene-" + activeId ? "true" : "false");
    });
    // Hero and closing span both dated scopes; atlas scenes use the August release.
    const scope = activeId ? SCOPE[SCENE_SCOPE[activeId] || "atlas"] : SCOPE.both;
    const chip = document.getElementById("claimDate");
    if (chip && chip.textContent !== scope) chip.textContent = scope;
  }

  function boot(data) {
    ORDER.forEach((id) => {
      const section = document.getElementById("scene-" + id);
      const holder = document.getElementById("canvas-" + id) || (section && section.dataset.nativeScene ? section : null);
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
      updateRail();
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
    // Canvas text uses the vendored faces only after they load. Static
    // (reduced-motion) frames drawn earlier are redrawn once, not looped.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => {
        ORDER.forEach((id) => { const r = scenes[id]; if (r && r.instance) { try { r.instance.redraw(); } catch (e) {} } });
      });
    }
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
