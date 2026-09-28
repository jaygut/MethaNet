/* SCENE — EVIDENCE SCOPE.
   Every record on the map was screened for methane-cycle genes through one of
   two annotation routes: the shared pipeline (functional contract 4) or the
   source's own annotations (contract 3, Old Woman Creek). The methane lens
   colors records by route, because methane evidence is compared within a route
   and never ranked across the map. The other lenses are planned directions on
   the same UMAP layout; no gas-specific result exists for them. Accessible DOM
   controls own the selected lens. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.engine = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, N = EB.num;
    const COOL_N2O = "#7C8CC4", COOL_S = "#A9B6CE", NEUTRAL = "#52606C";
    const SHARED = EB.color.methaneA, SOURCE = "#F6D8B8";
    const LENSES = [
      { name: "METHANE", sub: "screened now · two annotation routes", col: EB.color.methaneA },
      { name: "NITROUS OXIDE", sub: "planned · needs its own validation", col: COOL_N2O },
      { name: "SULFUR / DMS", sub: "planned · needs its own validation", col: COOL_S },
    ];
    let pts = [], shared = [], source = [], region = {}, ready = false;
    let manualSel = 0;

    // Stage-relative box of an overlay element, or null when it is hidden.
    function rel(el) {
      if (!el || !ctx.holder) return null;
      const base = ctx.holder.getBoundingClientRect(), r = el.getBoundingClientRect();
      if (!r.width || !r.height) return null;
      return { l: r.left - base.left, t: r.top - base.top, r: r.right - base.left, b: r.bottom - base.top };
    }
    // Fit the cloud into the space the copy card, lens buttons and lens plates
    // leave free, so no record sits behind text.
    function fitRegion(w, h) {
      const copy = rel(ctx.section && ctx.section.querySelector(".copy"));
      const controls = rel(document.getElementById("engineLensControls"));
      const box = { l: 16, r: w - 16, t: (controls ? controls.b : h * 0.1) + 16, b: h - 50 };
      if (w >= 640) {
        box.r = w * 0.72 - 30;
        if (copy && copy.r < w * 0.6) box.l = copy.r + 30;
      } else if (copy) {
        box.b = copy.t - 50; // room for the two-line legend under the cloud
      }
      const bw = box.r - box.l, bh = box.b - box.t;
      if (bw < 140 || bh < 100) return null;
      return { cx: (box.l + box.r) / 2, cy: Math.min((box.t + box.b) / 2, h * 0.5), s: Math.min(bw, bh) / 1.94 };
    }

    function project() {
      const w = ctx.W, h = ctx.H;
      region = fitRegion(w, h) || (w < 640
        ? { cx: w * 0.5, cy: h * 0.27, s: Math.min(w * 0.39, h * 0.21) }
        : { cx: w * 0.53, cy: h * 0.42, s: Math.min(w * 0.23, h * 0.29) });
      const atlas = ctx.data && ctx.data.atlas;
      if (!atlas) { ready = false; return; }
      // Same frozen records and UMAP coordinates as the atlas's opening view.
      pts = atlas.points.map((pt) => ({
        x: region.cx + (pt.hx || 0) * region.s,
        y: region.cy - (pt.hy || 0) * region.s,
        route: pt.fc === 4 ? 1 : pt.fc === 3 ? 2 : 0,
      }));
      // Center the cloud's bounding box on the region.
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const q of pts) { x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y); }
      const dx = (x0 + x1) / 2 - region.cx, dy = (y0 + y1) / 2 - region.cy;
      for (const q of pts) { q.x -= dx; q.y -= dy; }
      region.cloudBottom = y1 - dy;
      shared = pts.filter((q) => q.route === 1);
      source = pts.filter((q) => q.route === 2);
      ready = true;
    }

    function setManualLens(index) {
      if (!Number.isInteger(index) || index < 0 || index >= LENSES.length) return;
      manualSel = index;
      if (ctx.reduced && p.redraw) p.redraw();
    }
    p.setup = function () {
      p.createCanvas(ctx.W, ctx.H);
      p.pixelDensity(Math.min(2, window.devicePixelRatio || 1));
      project();
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { project(); if (p.redraw) p.redraw(); });
      document.addEventListener("emergentbiome:engine-lens", (event) => setManualLens(event.detail && event.detail.index));
      if (ctx.reduced) p.noLoop();
    };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); project(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.24, 100);
      if (!ready) { drawNoData(w, h); return; }

      const settle = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.0, 0.20));
      const methaneW = (manualSel === 0 ? 1 : 0) * settle;
      const sulfW = manualSel === 2 ? 1 : 0;
      const coolW = (manualSel === 0 ? 0 : 1) * settle;

      drawCloud(methaneW, coolW, sulfW, settle);
      if (w >= 640) {
        drawPlates(w, h, t);
        drawLegend(w, h, settle, methaneW);
        drawHonestyBand(w, h, settle);
      } else if (methaneW > 0.05) {
        drawMobileLegend(w, methaneW);
      }
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function dots(list, color, alpha, size) {
      const dc = p.drawingContext, half = size / 2;
      dc.fillStyle = color; dc.globalAlpha = alpha;
      for (let i = 0; i < list.length; i++) { const q = list[i]; dc.fillRect(q.x - half, q.y - half, size, size); }
    }

    function drawCloud(methaneW, coolW, sulfW, settle) {
      const dc = p.drawingContext;
      dc.save();
      // Every record stays visible; the active lens is a layer on top.
      dots(pts, NEUTRAL, 0.42 * settle, 2.6);
      if (methaneW > 0.01) {
        dots(shared, SHARED, 0.62 * methaneW, 2.8);
        dots(source, SOURCE, 0.7 * methaneW, 2.8);
      }
      // Planned lenses: the same points in one uniform wash, no per-record weighting.
      if (coolW > 0.01) dots(pts, D.lerpHex(COOL_N2O, COOL_S, sulfW), 0.34 * coolW, 2.4);
      dc.globalAlpha = 1; dc.restore();

      if (coolW > 0.05) {
        const halfW = region.s * 1.05, halfH = region.s * 0.72, cool = D.lerpHex(COOL_N2O, COOL_S, sulfW);
        p.push(); p.drawingContext.setLineDash([5, 5]);
        p.noFill(); p.stroke(D.rgba(cool, 0.4 * coolW)); p.strokeWeight(1.2);
        p.rect(region.cx - halfW, region.cy - halfH, halfW * 2, halfH * 2, 8);
        p.drawingContext.setLineDash([]); p.pop();
        D.label(p, "planned lens · no results yet", region.cx, region.cy + halfH + 14, D.rgba(cool, 0.8 * coolW), 8.5, [p.CENTER, p.TOP]);
      }
    }

    function drawPlates(w, h, t) {
      const x = w * 0.72, y0 = h * 0.30, pw = Math.min(w * 0.255, 250), ph = 50, sp = 62;
      const thr = [0.06, 0.30, 0.58];
      for (let i = 0; i < 3; i++) {
        const a = ctx.reduced ? 1 : D.window01(t, thr[i], thr[i] + 0.12);
        if (a < 0.01) continue;
        const y = y0 + i * sp, L = LENSES[i];
        p.push(); p.rectMode(p.CORNER);
        // Solid frame for the screen that exists, dashed for the planned lenses.
        p.noFill();
        if (i === 0) { p.stroke(D.rgba(L.col, 0.7 * a)); p.strokeWeight(1.3); }
        else { p.drawingContext.setLineDash([4, 4]); p.stroke(D.rgba(L.col, 0.5 * a)); p.strokeWeight(1); }
        p.rect(x, y, pw, ph, 4); p.drawingContext.setLineDash([]);
        const sw = 26, sx = x + 10, sy = y + (ph - sw) / 2;
        if (i === 0) {
          // Two halves: one per annotation route.
          p.noStroke();
          p.fill(D.rgba(SHARED, 0.85 * a)); p.rect(sx, sy, sw / 2, sw, 3, 0, 0, 3);
          p.fill(D.rgba(SOURCE, 0.85 * a)); p.rect(sx + sw / 2, sy, sw / 2, sw, 0, 3, 3, 0);
        } else {
          p.noFill(); p.drawingContext.setLineDash([3, 3]); p.stroke(D.rgba(L.col, 0.5 * a)); p.strokeWeight(1);
          p.rect(sx, sy, sw, sw, 3); p.drawingContext.setLineDash([]);
        }
        p.noStroke(); p.textAlign(p.LEFT, p.BOTTOM);
        p.fill(D.rgba(i === 0 ? EB.color.textPrimary : EB.color.textMuted, a)); p.textFont("Bricolage Grotesque"); p.textStyle(p.BOLD); p.textSize(10.5);
        p.text("LENS 0" + (i + 1) + " · " + L.name, sx + sw + 10, y + ph / 2); p.textStyle(p.NORMAL);
        p.textAlign(p.LEFT, p.TOP); p.textFont("IBM Plex Mono"); p.textSize(8);
        p.fill(D.rgba(i === 0 ? EB.color.methaneA : EB.color.textMuted, 0.85 * a));
        p.text(L.sub, sx + sw + 10, y + ph / 2 + 2);
        p.pop();
      }
    }

    function legendRow(x, y, color, value, text, alpha, size) {
      p.noStroke(); p.fill(D.rgba(color, 0.9 * alpha)); p.rect(x, y + 2, 9, 9, 2);
      p.fill(D.rgba(EB.color.textPrimary, alpha)); p.textFont("IBM Plex Mono"); p.textSize(size); p.textAlign(p.LEFT, p.TOP);
      p.text(value, x + 16, y);
      p.fill(D.rgba(EB.color.textMuted, 0.95 * alpha));
      p.text(text, x + 16 + p.textWidth(value + " "), y);
    }

    // Route legend under the lens plates, where the methane lens is described.
    function drawLegend(w, h, settle, methaneW) {
      const x = w * 0.72 + 10, y = h * 0.30 + 2 * 62 + 50 + 18;
      p.push();
      const a = Math.max(0.35, methaneW) * settle;
      legendRow(x, y, SHARED, D.fmt(N.pipelineNormalizedTriView), "shared pipeline", a, 10);
      legendRow(x, y + 17, SOURCE, D.fmt(N.sourceScaffoldTriView), "source annotations", a, 10);
      p.fill(D.rgba(EB.color.textMuted, 0.9 * settle)); p.textFont("IBM Plex Mono"); p.textSize(9); p.textAlign(p.LEFT, p.TOP);
      p.text(String(N.mechanismComparableTriView) + " ranked across both routes", x + 16, y + 36);
      p.pop();
    }

    function drawMobileLegend(w, a) {
      const y = region.cloudBottom + 12, x = 20;
      p.push();
      legendRow(x, y, SHARED, D.fmt(N.pipelineNormalizedTriView), "shared pipeline", a, 9.5);
      legendRow(x, y + 16, SOURCE, D.fmt(N.sourceScaffoldTriView), "source annotations", a, 9.5);
      p.pop();
    }

    function drawHonestyBand(w, h, settle) {
      p.push(); p.textAlign(p.CENTER, p.BOTTOM); p.textFont("IBM Plex Mono"); p.textSize(9.5);
      p.fill(D.rgba(EB.color.textMuted, 0.85 * settle));
      p.text("Methane evidence is compared within a route until the two are harmonized. No flux rate is inferred.", w / 2, h - 16);
      p.pop();
    }

    function drawNoData(w, h) {
      p.push(); p.fill(EB.color.textMuted); p.textAlign(p.CENTER, p.CENTER); p.textFont("IBM Plex Mono"); p.textSize(13);
      p.text("Serve over http:// to load the atlas data (data/atlas.json)", w / 2, h / 2);
      p.pop();
    }
  };
})();
