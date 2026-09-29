/* Molecular payload 2026-08-10; geometry reconciled 2026-09-29. Points are MAG/proteome records, not samples.
   Gold: selected one-way nearest-core links. Teal: a separate capped export
   of 2,200 cross-domain kNN links. 2D projection never changes link membership. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.atlas = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, GOLD = "#FFC75A";
    const ECOC = {}; EB.ecosystems.forEach((e) => { ECOC[e.code] = e.color; });
    const VIEWS = ["overview", "candidates", "neighbors", "sensitivity"];
    const PROJECTIONS = ["diffusion", "umap", "tsne", "pca"];
    // Every count below comes from the same hash-bound export as the points.
    const META = (ctx.data && ctx.data.atlas && ctx.data.atlas.meta) || {};
    const AUDIT = META.nearest_core_audit || {};
    const GEOMETRY = META.geometry_audit || {};
    const FROZEN = {
      snapshot: META.snapshot, points: META.n_points,
      candidates: ((ctx.data && ctx.data.atlas && ctx.data.atlas.bridges) || []).filter(b => b.cs).length,
      neighbors: ((ctx.data && ctx.data.atlas && ctx.data.atlas.bridges) || []).filter(b => !b.cs).length,
      wetlandRumen: AUDIT.wetland_outside_core_nearest_rumen_units,
      wetlandTotal: AUDIT.wetland_outside_core_units,
      mangroveRumen: AUDIT.mangrove_nearest_rumen_units,
      mangroveTotal: AUDIT.mangrove_embedding_units,
      cardsRumen: AUDIT.target_candidate_nearest_rumen_cards,
      cardsTotal: AUDIT.target_candidate_cards,
    };
    const pairCount = (mode, key) => fmt(((GEOMETRY[mode] || {})[key]) || 0);
    let rng, pts = [], byEco = [[], [], [], []], candidates = [], neighbors = [];
    let region = {}, cent = {}, ready = false, view = "overview";
    // UMAP is the initial navigation view. All projections remain inspectable.
    let projection = "umap", selectedCandidate = -1;
    let panelBody, panelAnnounce;

    function fmt(n) { return D.fmt(n); }
    function validFreeze() {
      const meta = ctx.data && ctx.data.atlas && ctx.data.atlas.meta;
      return !!meta && meta.embedding_configuration?.status === "verified" && meta.snapshot === FROZEN.snapshot &&
        pts.length === FROZEN.points && candidates.length === FROZEN.candidates &&
        neighbors.length === FROZEN.neighbors;
    }
    function projectCoordinates() {
      for (const pt of pts) {
        const xy = pt.coords[projection];
        pt.sx = region.cx + xy[0] * region.s;
        pt.sy = region.cy - xy[1] * region.s;
      }
      cent = {};
      EB.ecosystems.forEach((eco) => {
        const group = byEco[eco.code];
        if (!group || !group.length) return;
        cent[eco.key] = {
          x: group.reduce((sum, q) => sum + q.sx, 0) / group.length,
          y: group.reduce((sum, q) => sum + q.sy, 0) / group.length,
        };
      });
    }
    // Stage-relative box of an overlay element, or null when it is hidden.
    function rel(el) {
      if (!el || !ctx.holder) return null;
      const base = ctx.holder.getBoundingClientRect(), r = el.getBoundingClientRect();
      if (!r.width || !r.height) return null;
      return { l: r.left - base.left, t: r.top - base.top, r: r.right - base.left, b: r.bottom - base.top };
    }
    // Fit the map into the space the reading panel, view buttons and copy card
    // leave free, so no record or link end sits behind text. Coordinates are
    // normalized to about ±1 in every projection.
    function fitRegion(w, h) {
      const panel = rel(document.getElementById("atlasEvidencePanel"));
      const copy = rel(ctx.section && ctx.section.querySelector(".copy"));
      const controls = rel(document.getElementById("atlasViewControls"));
      const box = { l: 18, r: w - 18, t: (controls ? controls.b : h * 0.12) + 18, b: h - 54 };
      if (w > 800) {
        if (panel && panel.l < w * 0.3) box.l = panel.r + 30;
        if (copy && copy.l > w * 0.45) box.r = copy.l - 30;
      } else {
        if (panel) box.t = Math.max(box.t, panel.b + 14);
        if (copy) box.b = Math.min(box.b, copy.t - 14);
      }
      const bw = box.r - box.l, bh = box.b - box.t;
      if (bw < 150 || bh < 120) return null;
      return { cx: (box.l + box.r) / 2, cy: Math.min((box.t + box.b) / 2, h * 0.5), s: Math.min(bw, bh) / 1.94 };
    }
    function project() {
      const w = ctx.W, h = ctx.H;
      region = fitRegion(w, h) || {
        cx: w <= 800 ? w * 0.5 : w * 0.43,
        cy: h * 0.42,
        s: w <= 800 ? Math.min(w * 0.42, h * 0.28) : Math.min(w * 0.17, h * 0.29),
      };
      const atlas = ctx.data && ctx.data.atlas;
      if (!atlas || !Array.isArray(atlas.points) || !Array.isArray(atlas.bridges)) {
        ready = false; return;
      }
      rng = window.EBRandom.RNG("evidence");
      pts = atlas.points.map((pt) => {
        const a = rng.range(0, 6.2831), r = rng.range(0, Math.min(w, h) * 0.5);
        return {
          id: pt.id, e: pt.e, mz: pt.mz || 0,
          coords: { diffusion: [pt.x, pt.y], umap: [pt.hx, pt.hy], tsne: [pt.tx, pt.ty], pca: [pt.px, pt.py] },
          nsx: region.cx + Math.cos(a) * r, nsy: region.cy + Math.sin(a) * r,
        };
      });
      byEco = [[], [], [], []];
      pts.forEach((pt) => { if (byEco[pt.e]) byEco[pt.e].push(pt); });
      candidates = atlas.bridges.filter((b) => b.cs).sort((a, b) => b.w - a.w);
      neighbors = atlas.bridges.filter((b) => !b.cs).sort((a, b) => b.w - a.w);
      projectCoordinates();
      ready = true;
    }
    function requestRender() { if (p && p.redraw) p.redraw(); }
    function keyboardGroup(container, selector) {
      container.addEventListener("keydown", (event) => {
        if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"].includes(event.key)) return;
        const buttons = Array.from(container.querySelectorAll(selector));
        const index = buttons.indexOf(document.activeElement);
        if (index < 0) return;
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 :
          (index + (event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
        buttons[next].focus(); buttons[next].click();
      });
    }
    function initControls() {
      const controls = document.getElementById("atlasViewControls");
      const panel = document.getElementById("atlasEvidencePanel");
      if (!controls || !panel) return;
      if (!ready) {
        controls.querySelectorAll("button").forEach((button) => { button.disabled = true; });
        panel.textContent = "The atlas data could not be loaded. Open this page through a web server to explore the map.";
        return;
      }
      const viewButtons = Array.from(controls.querySelectorAll("[data-atlas-view]"));
      viewButtons.forEach((button) => {
        button.addEventListener("click", () => {
          if (!VIEWS.includes(button.dataset.atlasView)) return;
          view = button.dataset.atlasView;
          selectedCandidate = -1;
          viewButtons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
          renderPanel(); panel.scrollTop = 0; requestRender();
        });
      });
      keyboardGroup(controls, "[data-atlas-view]");
      panel.innerHTML =
        '<div class="atlas__panel-meta"><span>READ THE MAP</span><span class="atlas__panel-snapshot">GEOMETRY · 29 SEP 2026</span></div>' +
        '<div class="atlas__panel-body" id="atlasPanelBody"></div>' +
        '<div class="atlas__projection"><span class="atlas__projection-label">2D PROJECTION</span>' +
        '<div class="atlas__projection-buttons" role="group" aria-label="Atlas projection">' +
        '<button type="button" data-atlas-projection="umap" aria-pressed="true">UMAP</button>' +
        '<button type="button" data-atlas-projection="diffusion" aria-pressed="false">Diffusion</button>' +
        '<button type="button" data-atlas-projection="tsne" aria-pressed="false">t-SNE</button>' +
        '<button type="button" data-atlas-projection="pca" aria-pressed="false">PCA</button></div>' +
        '<p>Four ways to flatten the same map, each with its own distortions. UMAP opens by default because it keeps local neighborhoods readable. Links are computed in the full representation, so switching views moves points but never changes which links exist.</p></div>' +
        '<span id="atlasPanelAnnounce" class="vh" aria-live="polite"></span>';
      panelBody = panel.querySelector("#atlasPanelBody");
      panelAnnounce = panel.querySelector("#atlasPanelAnnounce");
      const projectionButtons = Array.from(panel.querySelectorAll("[data-atlas-projection]"));
      projectionButtons.forEach((button) => {
        button.addEventListener("click", () => {
          if (!PROJECTIONS.includes(button.dataset.atlasProjection)) return;
          projection = button.dataset.atlasProjection;
          projectionButtons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
          projectCoordinates();
          panelAnnounce.textContent = button.textContent + " view. The links do not change.";
          requestRender();
        });
      });
      keyboardGroup(panel.querySelector(".atlas__projection-buttons"), "[data-atlas-projection]");
      renderPanel();
    }
    function renderPanel() {
      if (!panelBody) return;
      if (view === "overview") {
        const wetland = byEco[1].length, mangrove = byEco[2].length + byEco[3].length;
        panelBody.innerHTML =
          '<p class="atlas__eyebrow">01 / What you see</p>' +
          '<h3>' + fmt(pts.length) + ' genome records</h3>' +
          '<p>Each point is one genome record, colored by where it came from. Position reflects protein content, compressed into two dimensions for display; it is not a measure of methane activity.</p>' +
          '<dl class="atlas__stat-grid"><div><dt>Rumen reference</dt><dd>' + fmt(byEco[0].length) + '</dd></div>' +
          '<div><dt>Wetland</dt><dd>' + fmt(wetland) + '</dd></div>' +
          '<div><dt>Mangrove</dt><dd>' + fmt(mangrove) + '</dd></div></dl>' +
          '<ul class="atlas__legend" aria-label="Atlas point colors">' +
          '<li><span class="atlas__swatch atlas__swatch--rumen" aria-hidden="true"></span>Rumen reference · ' + fmt(byEco[0].length) + '</li>' +
          '<li><span class="atlas__swatch atlas__swatch--wetland" aria-hidden="true"></span>Wetland · ' + fmt(byEco[1].length) + '</li>' +
          '<li><span class="atlas__swatch atlas__swatch--msm" aria-hidden="true"></span>Mangrove, China coast · ' + fmt(byEco[2].length) + '</li>' +
          '<li><span class="atlas__swatch atlas__swatch--futian" aria-hidden="true"></span>Mangrove, Futian, Shenzhen · ' + fmt(byEco[3].length) + '</li></ul>' +
          '<p class="atlas__fineprint">Scroll to reveal the map. The overview previews 240 of the ' + fmt(neighbors.length) + ' neighbor links; choose Neighbor links to see them all.</p>';
      } else if (view === "candidates") {
        const counts = validFreeze()
          ? '<dl class="atlas__stat-grid"><div><dt>Wetland → rumen</dt><dd>' + fmt(FROZEN.wetlandRumen) + ' / ' + fmt(FROZEN.wetlandTotal) + '</dd></div>' +
            '<div><dt>Mangrove → rumen</dt><dd>' + fmt(FROZEN.mangroveRumen) + ' / ' + fmt(FROZEN.mangroveTotal) + '</dd></div>' +
            '<div><dt>Selected cards</dt><dd>' + fmt(FROZEN.cardsRumen) + ' / ' + fmt(FROZEN.cardsTotal) + '</dd></div></dl>'
          : '<p class="atlas__fineprint">The frozen nearest-reference counts are unavailable for this export.</p>';
        panelBody.innerHTML =
          '<p class="atlas__eyebrow">02 / Nearest reference</p>' +
          '<h3>' + fmt(candidates.length) + ' highlighted links</h3>' +
          '<p>Gold lines join selected wetland and mangrove genomes to their closest genome in the 625-genome reference core (518 rumen, 107 wetland). The counts below describe the corrected final-layer representation. Reference composition and shared ancestry can influence the closest match.</p>' +
          counts +
          '<p class="atlas__fineprint">Median nearest-core cosine: ' + Number(AUDIT.outside_core_nearest_similarity_median).toFixed(4) + '. Random-pair median: ' + Number(GEOMETRY.random_pair_similarity_median).toFixed(4) + '. High raw cosine alone does not establish shared function. Only the ' + fmt(candidates.length) + ' selected links are drawn; core self-matches are excluded.</p>' +
          '<label class="atlas__select-label" for="atlasCandidateSelect">Inspect a highlighted link</label>' +
          '<select id="atlasCandidateSelect" class="atlas__candidate-select"><option value="-1">All ' + fmt(candidates.length) + ' highlighted links</option></select>' +
          '<p class="atlas__candidate-readout" id="atlasCandidateReadout" role="status" aria-live="polite" aria-atomic="true">Choose a link to see both record IDs and their raw cosine similarity.</p>';
        const select = panelBody.querySelector("#atlasCandidateSelect");
        candidates.forEach((link, index) => {
          const source = pts[link.s];
          const option = document.createElement("option");
          option.value = String(index);
          option.textContent = (index + 1) + ". " + (source.e === 1 ? "Wetland" : "Mangrove") + " · " + source.id.split("__").pop();
          select.appendChild(option);
        });
        select.addEventListener("change", () => {
          selectedCandidate = Number(select.value);
          updateCandidateReadout(); requestRender();
        });
      } else if (view === "neighbors") {
        panelBody.innerHTML =
          '<p class="atlas__eyebrow">03 / Neighbor links</p>' +
          '<h3>' + fmt(neighbors.length) + ' cross-habitat links</h3>' +
          '<p>Teal lines show the ' + fmt(neighbors.length) + ' most similar of the ' + fmt(EB.num.crossHabitatNeighborEdges) + ' links that cross habitats in the full neighbor graph, which is computed in the complete representation. They are separate from the ' + fmt(candidates.length) + ' gold reference links.</p>' +
          '<p class="atlas__fineprint">A neighbor link means similarity in this protein representation. It is not gene exchange, a shared pathway or a methane measurement.</p>';
      } else {
        // Top-35 reciprocal counts: frozen audit/scientific_audit.json,
        // embedding geometry sensitivity block (raw vs dimension_zscore).
        panelBody.innerHTML =
          '<p class="atlas__eyebrow">04 / Stricter test</p>' +
          '<h3>Which resemblances are mutual?</h3>' +
          '<p>This test counts pairs in which each genome is among the other’s 35 closest neighbors across the whole atlas. Standardizing each dimension first removes the shared offset that makes all genomes look alike.</p>' +
          '<dl class="atlas__sensitivity-grid">' +
          [['Rumen ↔ wetland','rumen↔wetland'],['Rumen ↔ mangrove','mangrove↔rumen'],['Mangrove ↔ wetland','mangrove↔wetland']].map(([label,key]) =>
            '<div><dt>' + label + '</dt><dd><span>Raw</span><b>' + pairCount('raw_reciprocal_pair_counts',key) + '</b><span>Standardized</span><b>' + pairCount('dimension_zscore_reciprocal_pair_counts',key) + '</b></dd></div>').join('') + '</dl>' +
          '<p class="atlas__fineprint">These are geometry sensitivity counts after pooling reconciliation. Mutual neighbors remain hypotheses for review; phylogeny, source effects and functional evidence must be tested independently. This test draws no edges of its own.</p>';

      }
      if (panelAnnounce) {
        const labels = { overview: "Overview of the atlas", candidates: "Nearest-reference links", neighbors: "Cross-habitat neighbor links", sensitivity: "Stricter mutual-neighbor test" };
        panelAnnounce.textContent = labels[view] + " selected.";
      }
    }
    function updateCandidateReadout() {
      const readout = panelBody && panelBody.querySelector("#atlasCandidateReadout");
      if (!readout) return;
      if (selectedCandidate < 0 || !candidates[selectedCandidate]) {
        readout.textContent = "Choose a link to see both record IDs and their raw cosine similarity.";
        return;
      }
      const link = candidates[selectedCandidate];
      const source = pts[link.s], reference = pts[link.t];
      readout.textContent = "Candidate: " + source.id + ". Closest core reference: " + reference.id +
        ". Raw cosine similarity: " + Number(link.w).toFixed(4) + ". A lead for evidence review, not validated transfer.";
    }
    p.setup = function () {
      p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1));
      project(); initControls();
      project(); // refit once the reading panel has content
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { project(); requestRender(); });
      if (ctx.reduced) p.noLoop();
    };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); project(); requestRender(); };
    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.22, 110);
      if (!ready) { drawNoData(w, h); return; }
      const manual = view !== "overview";
      const resolve = manual ? 1 : D.easeInOut(D.window01(t, 0.0, 0.30));
      const clarify = manual ? 1 : D.window01(t, 0.22, 0.50);
      const neighborAlpha = view === "neighbors" ? 1 : view === "overview" ? D.window01(t, 0.46, 0.70) : 0;
      const candidateAlpha = view === "candidates" ? 1 : view === "overview" ? D.window01(t, 0.60, 0.85) : 0;
      const dc = p.drawingContext;
      for (const pt of pts) {
        pt.x = D.lerp(pt.nsx, pt.sx, resolve);
        pt.y = D.lerp(pt.nsy, pt.sy, resolve);
      }
      if (resolve > 0.15 && cent.mangrove_msm) {
        const mc = cent.mangrove_msm;
        const glow = dc.createRadialGradient(mc.x, mc.y, 0, mc.x, mc.y, region.s);
        glow.addColorStop(0, D.rgba(EB.color.mangroveMsm, 0.09 * resolve));
        glow.addColorStop(0.55, D.rgba(EB.color.mangroveFutian, 0.035 * resolve));
        glow.addColorStop(1, "rgba(0,0,0,0)");
        dc.save(); dc.fillStyle = glow; dc.fillRect(0, 0, w, h); dc.restore();
      }
      dc.save();
      for (let ecosystem = 0; ecosystem < 4; ecosystem++) {
        const group = byEco[ecosystem]; if (!group.length) continue;
        const poc = ecosystem < 2;
        dc.fillStyle = D.lerpHex("#586771", ECOC[ecosystem], clarify);
        dc.globalAlpha = (poc ? 0.56 + 0.38 * clarify : 0.38 + 0.42 * clarify) *
          (view === "sensitivity" ? 0.8 : 1);
        for (const pt of group) {
          const size = (poc ? 2.4 : 1.4) + pt.mz * 1.8 * clarify;
          dc.fillRect(pt.x - size * 0.5, pt.y - size * 0.5, size, size);
        }
      }
      dc.restore();
      // Draw the full 2,200-link export only on request; overview previews 240.
      const neighborLimit = view === "neighbors" ? neighbors.length : Math.round(neighborAlpha * Math.min(240, neighbors.length));
      if (neighborLimit > 0) {
        dc.save(); dc.globalCompositeOperation = "lighter";
        dc.strokeStyle = D.rgba(EB.color.emergence, view === "neighbors" ? 0.095 : 0.13 * neighborAlpha);
        dc.lineWidth = view === "neighbors" ? 0.9 : 1; dc.beginPath();
        for (let i = 0; i < neighborLimit; i++) {
          const link = neighbors[i], a = pts[link.s], b = pts[link.t];
          if (!a || !b) continue;
          dc.moveTo(a.x, a.y); dc.lineTo(b.x, b.y);
        }
        dc.stroke(); dc.restore();
      }
      const candidateLimit = view === "candidates" ? candidates.length : Math.round(candidateAlpha * candidates.length);
      if (candidateLimit > 0) {
        p.push(); p.blendMode(p.ADD);
        for (let i = 0; i < candidateLimit; i++) {
          if (selectedCandidate >= 0 && view === "candidates" && i !== selectedCandidate) continue;
          const link = candidates[i], source = pts[link.s], core = pts[link.t];
          if (!source || !core) continue;
          const selected = selectedCandidate === i && view === "candidates";
          p.stroke(D.rgba(GOLD, selected ? 0.95 : 0.63 * candidateAlpha));
          p.strokeWeight(selected ? 2.7 : 1.35);
          p.line(source.x, source.y, core.x, core.y);
          p.noFill(); p.strokeWeight(selected ? 2 : 1);
          p.circle(source.x, source.y, selected ? 18 : 11);
          if (selected) {
            p.stroke(D.rgba(EB.color.attested, 0.9)); p.circle(core.x, core.y, 15);
            D.glow(p, source.x, source.y, 4, GOLD, 0.8);
            D.glow(p, core.x, core.y, 3, EB.color.attested, 0.75);
          }
        }
        p.pop();
      }
      D.vignette(p, w, h, EB.color.bgBase, 0.42);
    };
    function drawNoData(w, h) {
      p.push(); p.fill(EB.color.textMuted); p.textAlign(p.CENTER, p.CENTER);
      p.textFont("IBM Plex Mono"); p.textSize(13);
      p.text("Atlas data unavailable", w / 2, h / 2); p.pop();
    }
  };
})();
