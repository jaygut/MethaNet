/* Source-audited 2026-08-10 atlas. Points are MAG/proteome records, not samples.
   Gold: 26 selected one-way nearest-core links. Teal: a separate capped export
   of 2,200 cross-domain kNN links. 2D projection never changes link membership. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.atlas = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, GOLD = "#FFC75A";
    const ECOC = {}; EB.ecosystems.forEach((e) => { ECOC[e.code] = e.color; });
    const VIEWS = ["overview", "candidates", "neighbors", "sensitivity"];
    const PROJECTIONS = ["diffusion", "umap", "tsne", "pca"];
    // Frozen raw-cosine nearest-core results from
    // results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/
    // tables/embedding_context_table.tsv and tables/candidate_cards.tsv.
    // These are not counts of the selected links in this smaller visual export.
    const FROZEN = {
      snapshot: "2026-08-10", points: 7710, candidates: 26, neighbors: 2200,
      wetlandRumen: 2434, wetlandTotal: 2608,
      mangroveRumen: 4475, mangroveTotal: 4584, cardsRumen: 26, cardsTotal: 27,
    };
    let rng, pts = [], byEco = [[], [], [], []], candidates = [], neighbors = [];
    let region = {}, cent = {}, ready = false, view = "overview";
    // UMAP is the initial navigation layout because the diffusion view compresses
    // the target cohorts into a near-vertical band. Both remain inspectable.
    let projection = "umap", selectedCandidate = -1;
    let panelBody, panelAnnounce;

    function fmt(n) { return D.fmt(n); }
    function validFreeze() {
      const meta = ctx.data && ctx.data.atlas && ctx.data.atlas.meta;
      return !!meta && meta.snapshot === FROZEN.snapshot &&
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
    function project() {
      const w = ctx.W, h = ctx.H;
      region = {
        cx: w <= 800 ? w * 0.5 : w * 0.43,
        cy: w <= 800 ? h * 0.42 : h * 0.42,
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
        panel.textContent = "The atlas data could not be loaded. Open this page through a web server to explore the evidence view.";
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
        '<div class="atlas__panel-meta"><span>EXPLORE THE EVIDENCE</span><span class="atlas__panel-snapshot">FREEZE · 10 AUG 2026</span></div>' +
        '<div class="atlas__panel-body" id="atlasPanelBody"></div>' +
        '<div class="atlas__projection"><span class="atlas__projection-label">2D PROJECTION</span>' +
        '<div class="atlas__projection-buttons" role="group" aria-label="Atlas projection">' +
        '<button type="button" data-atlas-projection="diffusion" aria-pressed="false">Diffusion</button>' +
        '<button type="button" data-atlas-projection="umap" aria-pressed="true">UMAP</button>' +
        '<button type="button" data-atlas-projection="tsne" aria-pressed="false">t-SNE</button>' +
        '<button type="button" data-atlas-projection="pca" aria-pressed="false">PCA</button></div>' +
        '<p>UMAP opens for visual navigation; compare diffusion, t-SNE, and PCA. These 2D layouts are display views; link membership stays in high-dimensional ESM-2 cosine space.</p></div>' +
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
          panelAnnounce.textContent = projection.toUpperCase() + " projection. Link membership is unchanged.";
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
          '<p class="atlas__eyebrow">01 / Cohort</p>' +
          '<h3>' + fmt(pts.length) + ' MAG / proteome records</h3>' +
          '<p>Each point is a molecular record, colored by source. The 2D layout is exploratory, not a measure of ecological transfer or methane flux.</p>' +
          '<dl class="atlas__stat-grid"><div><dt>Rumen core</dt><dd>' + fmt(byEco[0].length) + '</dd></div>' +
          '<div><dt>Wetland</dt><dd>' + fmt(wetland) + '</dd></div>' +
          '<div><dt>Mangrove</dt><dd>' + fmt(mangrove) + '</dd></div></dl>' +
          '<ul class="atlas__legend" aria-label="Atlas point colors">' +
          '<li><span class="atlas__swatch atlas__swatch--rumen" aria-hidden="true"></span>Rumen · ' + fmt(byEco[0].length) + '</li>' +
          '<li><span class="atlas__swatch atlas__swatch--wetland" aria-hidden="true"></span>Wetland · ' + fmt(byEco[1].length) + '</li>' +
          '<li><span class="atlas__swatch atlas__swatch--msm" aria-hidden="true"></span>Mangrove MSM · ' + fmt(byEco[2].length) + '</li>' +
          '<li><span class="atlas__swatch atlas__swatch--futian" aria-hidden="true"></span>Mangrove Futian · ' + fmt(byEco[3].length) + '</li></ul>' +
          '<p class="atlas__fineprint">Scroll reveals the map. The overview previews up to 240 teal links; use Neighbor sample to see all ' + fmt(neighbors.length) + ' exported links.</p>';
      } else if (view === "candidates") {
        const counts = validFreeze()
          ? '<dl class="atlas__stat-grid"><div><dt>Wetland → rumen core</dt><dd>' + fmt(FROZEN.wetlandRumen) + ' / ' + fmt(FROZEN.wetlandTotal) + '</dd></div>' +
            '<div><dt>Mangrove → rumen core</dt><dd>' + fmt(FROZEN.mangroveRumen) + ' / ' + fmt(FROZEN.mangroveTotal) + '</dd></div>' +
            '<div><dt>Selected cards</dt><dd>' + fmt(FROZEN.cardsRumen) + ' / ' + fmt(FROZEN.cardsTotal) + '</dd></div></dl>'
          : '<p class="atlas__fineprint">Frozen nearest-core denominators are unavailable for this visual export.</p>';
        panelBody.innerHTML =
          '<p class="atlas__eyebrow">02 / One-way nearest core</p>' +
          '<h3>' + fmt(candidates.length) + ' highlighted candidate links</h3>' +
          '<p>Gold lines join selected wetland or mangrove records to their raw-cosine nearest rumen reference record in the 625-record POC core. These are one-way reference matches.</p>' +
          counts +
          '<p class="atlas__fineprint">The large denominators count records in a separate nearest-core table; only ' + fmt(candidates.length) + ' selected nearest-core links are drawn in gold. The 27th card is a wetland core record whose nearest core member is itself. Raw-cosine ESM-2 space is anisotropic; these links nominate review, not validated transfer.</p>' +
          '<label class="atlas__select-label" for="atlasCandidateSelect">Inspect a highlighted link</label>' +
          '<select id="atlasCandidateSelect" class="atlas__candidate-select"><option value="-1">All ' + fmt(candidates.length) + ' highlighted links</option></select>' +
          '<p class="atlas__candidate-readout" id="atlasCandidateReadout" role="status" aria-live="polite" aria-atomic="true">Choose a candidate to see its proteome IDs and raw cosine similarity.</p>';
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
          '<p class="atlas__eyebrow">03 / Capped graph sample</p>' +
          '<h3>' + fmt(neighbors.length) + ' cross-domain kNN links</h3>' +
          '<p>Teal lines show every link in a capped visualization export from the high-dimensional ESM-2 cosine neighbor graph. They are distinct from the ' + fmt(candidates.length) + ' nearest-core candidate links.</p>' +
          '<p class="atlas__fineprint">A neighbor relation is molecular geometry. This export is not all graph edges, a proof of source-independent transfer, a pathway assay, or a methane-flux measurement.</p>';
      } else {
        // Top-35 reciprocal counts: frozen audit/scientific_audit.json,
        // embedding geometry sensitivity block (raw vs dimension_zscore).
        panelBody.innerHTML =
          '<p class="atlas__eyebrow">04 / Full-atlas sensitivity</p>' +
          '<h3>What “falls to zero” means</h3>' +
          '<p>A reciprocal pair requires each record to place the other in its top 35 high-dimensional neighbors. This full-atlas comparison uses a different reference set and preprocessing from one-way nearest matches to the 625-record core.</p>' +
          '<dl class="atlas__sensitivity-grid"><div><dt>Rumen ↔ wetland</dt><dd><span>Raw</span><b>1</b><span>Standardized</span><b>0</b></dd></div>' +
          '<div><dt>Rumen ↔ mangrove</dt><dd><span>Raw</span><b>0</b><span>Standardized</span><b>0</b></dd></div>' +
          '<div><dt>Mangrove ↔ wetland</dt><dd><span>Raw</span><b>15,728</b><span>Standardized</span><b>15,064</b></dd></div></dl>' +
          '<p class="atlas__fineprint">“Standardized” means per-dimension z-scoring of the full embedding before neighbor search. The zero does not erase the one-way nearest-core links. This map does not draw reciprocal-pair edges.</p>';
      }
      if (panelAnnounce) {
        const labels = { overview: "Atlas cohort view", candidates: "One-way nearest-core candidate view", neighbors: "Cross-domain neighbor sample view", sensitivity: "Full-atlas reciprocal-neighbor sensitivity view" };
        panelAnnounce.textContent = labels[view] + " selected.";
      }
    }
    function updateCandidateReadout() {
      const readout = panelBody && panelBody.querySelector("#atlasCandidateReadout");
      if (!readout) return;
      if (selectedCandidate < 0 || !candidates[selectedCandidate]) {
        readout.textContent = "Choose a candidate to see its proteome IDs and raw cosine similarity.";
        return;
      }
      const link = candidates[selectedCandidate];
      const source = pts[link.s], reference = pts[link.t];
      readout.textContent = "Candidate: " + source.id + ". Nearest rumen core reference: " + reference.id +
        ". Raw cosine similarity: " + Number(link.w).toFixed(4) + ". This is a candidate for evidence review, not validated transfer.";
    }
    p.setup = function () {
      p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1));
      project(); initControls(); if (ctx.reduced) p.noLoop();
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
