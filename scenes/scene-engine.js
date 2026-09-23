/* SCENE — EVIDENCE SCOPE.
   The current molecular atlas supports methane-pathway screening. The other
   controls show proposed research directions using the same UMAP navigation
   layout, without implying gas-specific features or flux calibration already
   exist. Accessible DOM controls own the selected view. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.engine = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const ECOC = {}; EB.ecosystems.forEach((e) => (ECOC[e.code] = e.color));
    const COOL_N2O = "#7C8CC4", COOL_S = "#A9B6CE", NEUTRAL = "#52606C";
    const LENSES = [
      { name: "METHANE", sub: "screening events; comparability pending", col: EB.color.methaneA },
      { name: "NITROUS OXIDE", sub: "candidate, not built", col: COOL_N2O },
      { name: "SULFUR · DMS", sub: "not built", col: COOL_S },
    ];
    let rng, pts = [], tierLo = [], tierMid = [], tierHi = [], region = {}, ready = false;
    let manualSel = 0;

    function project() {
      const w = ctx.W, h = ctx.H;
      region = w < 640
        ? { cx: w * 0.5, cy: h * 0.27, s: Math.min(w * 0.39, h * 0.21) }
        : { cx: w * 0.53, cy: h * 0.42, s: Math.min(w * 0.23, h * 0.29) };
      const atlas = ctx.data && ctx.data.atlas;
      if (!atlas) { ready = false; return; }
      rng = window.EBRandom.RNG("engine");
      // Same frozen records and UMAP navigation coordinates as the atlas's
      // initial view. This does not generate any gas-specific measurement.
      pts = atlas.points.map((pt) => ({
        x: region.cx + (pt.hx || 0) * region.s,
        y: region.cy - (pt.hy || 0) * region.s,
        mz: pt.mz == null ? 0 : pt.mz,
        eligible: pt.fc === 1,
        ph: rng.range(0, 6.2831),
      }));
      // recenter the fan centroid on (region.cx, region.cy) so the lens bar sits over it and the
      // beam connects to it; store the cloud top for the beam endpoint.
      let mx = 0, my = 0; for (const q of pts) { mx += q.x; my += q.y; }
      mx = mx / pts.length - region.cx; my = my / pts.length - region.cy;
      let top = Infinity; for (const q of pts) { q.x -= mx; q.y -= my; if (q.y < top) top = q.y; }
      region.cloudTop = top;
      tierLo = []; tierMid = []; tierHi = [];
      for (const q of pts) {
        if (!q.eligible || q.mz <= 0.001) continue;
        if (q.mz < 0.45) tierLo.push(q); else if (q.mz < 0.72) tierMid.push(q); else tierHi.push(q);
      }
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
      document.addEventListener("emergentbiome:engine-lens", (event) => setManualLens(event.detail && event.detail.index));
      if (ctx.reduced) p.noLoop();
    };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); project(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.24, 100);
      if (!ready) { drawNoData(w, h); return; }

      const settle = D.easeInOut(D.window01(t, 0.0, 0.20));
      const sel = manualSel;
      const methaneW = D.clamp(sel <= 1 ? 1 - sel : 0, 0, 1) * settle;
      const n2oW = D.clamp(sel <= 1 ? sel : 2 - sel, 0, 1);
      const sulfW = D.clamp(sel >= 1 ? sel - 1 : 0, 0, 1);
      const coolW = D.clamp(n2oW + sulfW, 0, 1) * settle;
      const tm = ctx.reduced ? 0.4 : p.frameCount * 0.02;

      drawCloud(methaneW, coolW, n2oW, sulfW, settle, tm);
      drawPlates(w, h, t);
      if (w >= 640) {
        drawHud(w, h, settle);
        drawHonestyBand(w, h, settle);
      }
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawCloud(methaneW, coolW, n2oW, sulfW, settle, tm) {
      const dc = p.drawingContext;
      dc.save();
      // always-present structure: 7,710 MAG/proteome records,
      // with the active gas as a highlight on top
      dc.fillStyle = NEUTRAL; dc.globalAlpha = 0.42 * settle;
      for (let i = 0; i < pts.length; i++) { const q = pts[i]; dc.fillRect(q.x - 1.3, q.y - 1.3, 2.6, 2.6); }
      // LENS 01 methane: curated POC-only density (mz), bright + structured,
      // with non-comparable lanes left in the neutral atlas layer.
      if (methaneW > 0.01) {
        dc.fillStyle = EB.color.methaneA; dc.globalAlpha = 0.55 * methaneW;
        for (let i = 0; i < tierLo.length; i++) { const q = tierLo[i]; dc.fillRect(q.x - 1.4, q.y - 1.4, 2.8, 2.8); }
        dc.fillStyle = D.lerpHex(EB.color.methaneA, EB.color.methaneB, 0.5); dc.globalAlpha = 0.7 * methaneW;
        for (let i = 0; i < tierMid.length; i++) { const q = tierMid[i]; dc.fillRect(q.x - 1.6, q.y - 1.6, 3.2, 3.2); }
        dc.fillStyle = EB.color.methaneB; dc.globalAlpha = 0.92 * methaneW;
        for (let i = 0; i < tierHi.length; i++) { const q = tierHi[i]; dc.fillRect(q.x - 1.9, q.y - 1.9, 3.8, 3.8); }
      }
      // LENS 02/03 cool wash: SAME points, uniform low alpha, NO per-genome weighting
      if (coolW > 0.01) {
        dc.fillStyle = D.lerpHex(COOL_N2O, COOL_S, sulfW); dc.globalAlpha = 0.34 * coolW;
        for (let i = 0; i < pts.length; i++) { const q = pts[i]; dc.fillRect(q.x - 1.2, q.y - 1.2, 2.4, 2.4); }
      }
      dc.globalAlpha = 1; dc.restore();

      // "aimed but unfilled" tag when a non-methane lens is active: a dashed bracket hugging
      // the thin cloud (not a giant circle), so it reads as the empty plate the lens points at
      if (coolW > 0.05) {
        const halfW = region.s * 1.05, halfH = region.s * 0.72, cool = D.lerpHex(COOL_N2O, COOL_S, sulfW);
        p.push(); p.drawingContext.setLineDash([5, 5]);
        p.noFill(); p.stroke(D.rgba(cool, 0.4 * coolW)); p.strokeWeight(1.2);
        p.rect(region.cx - halfW, region.cy - halfH, halfW * 2, halfH * 2, 8);
        p.drawingContext.setLineDash([]); p.pop();
        D.label(p, "aimed, not computed", region.cx, region.cy + halfH + 14, D.rgba(cool, 0.75 * coolW), 8.5, [p.CENTER, p.TOP]);
      }
    }

    function drawPlates(w, h, t) {
      if (w < 640) return;   // mobile: the lens bar + copy-card readout carry the maps story
      const x = w * 0.72, y0 = h * 0.30, pw = Math.min(w * 0.255, 250), ph = 50, sp = 62;
      const thr = [0.06, 0.30, 0.58];
      for (let i = 0; i < 3; i++) {
        const a = ctx.reduced ? 1 : D.window01(t, thr[i], thr[i] + 0.12);
        if (a < 0.01) continue;
        const y = y0 + i * sp, L = LENSES[i];
        const xx = x;
        p.push(); p.rectMode(p.CORNER);
        // card frame: solid for the bounded methane screen, dashed for the candidates
        p.noFill();
        if (i === 0) { p.stroke(D.rgba(L.col, 0.7 * a)); p.strokeWeight(1.3); }
        else { p.drawingContext.setLineDash([4, 4]); p.stroke(D.rgba(L.col, 0.5 * a)); p.strokeWeight(1); }
        p.rect(xx, y, pw, ph, 4); p.drawingContext.setLineDash([]);
        // Swatches indicate evidence scope, never a released per-record score.
        const sw = 26, sx = xx + 10, sy = y + (ph - sw) / 2;
        if (i === 0) {
          p.noFill(); p.stroke(D.rgba(EB.color.methaneA, 0.85 * a)); p.strokeWeight(1.4);
          p.rect(sx, sy, sw, sw, 3);
          p.noStroke(); p.fill(D.rgba(EB.color.methaneA, 0.9 * a)); p.circle(sx + sw / 2, sy + sw / 2, 5);
        } else {
          p.noFill(); p.drawingContext.setLineDash([3, 3]); p.stroke(D.rgba(L.col, 0.5 * a)); p.strokeWeight(1);
          p.rect(sx, sy, sw, sw, 3); p.drawingContext.setLineDash([]);
        }
        // title + sub
        p.noStroke(); p.textAlign(p.LEFT, p.BOTTOM);
        p.fill(D.rgba(i === 0 ? EB.color.textPrimary : EB.color.textMuted, a)); p.textFont("Bricolage Grotesque"); p.textStyle(p.BOLD); p.textSize(10.5);
        p.text((i === 0 ? "MAP 01 · " : i === 1 ? "LENS 02 · " : "SLOT 03 · ") + L.name, sx + sw + 10, y + ph / 2); p.textStyle(p.NORMAL);
        p.textAlign(p.LEFT, p.TOP); p.textFont("IBM Plex Mono"); p.textSize(8);
        p.fill(D.rgba(i === 0 ? EB.color.methaneA : EB.color.textMuted, 0.85 * a));
        p.text(i === 0 ? "screening only · comparability pending" : "research concept · not built", sx + sw + 10, y + ph / 2 + 2);
        p.pop();
      }
    }

    function drawHud(w, h, settle) {
      const x = Math.max(16, w * 0.04), y = h * 0.13;
      D.label(p, "CURRENT EVIDENCE · FUTURE OPTIONS", x, y, D.rgba(EB.color.emergence, 0.95 * settle), 11);
      D.label(p, "same records · separate validation for every pathway", x, y + 15, D.rgba(EB.color.textMuted, 0.9 * settle), 9.5);
      // One pipeline-normalized screening lane / two candidate lenses.
      p.push(); p.textAlign(p.LEFT, p.TOP); p.textFont("IBM Plex Mono");
      p.fill(D.rgba(EB.color.methaneA, settle)); p.textSize(12); p.text("normalized screening events available", x, y + 34);
      p.fill(D.rgba(EB.color.textMuted, 0.9 * settle)); p.textSize(10); p.text("2 other lens concepts · not built", x, y + 50);
      p.pop();
    }

    function drawHonestyBand(w, h, settle) {
      p.push(); p.textAlign(p.CENTER, p.BOTTOM); p.textFont("IBM Plex Mono"); p.textSize(9.5);
      p.fill(D.rgba(EB.color.textMuted, 0.85 * settle));
      p.text("Cross-lane mechanism weights remain off until comparability gates pass. No flux rate is inferred.", w / 2, h - 16);
      p.pop();
    }

    function drawNoData(w, h) {
      p.push(); p.fill(EB.color.textMuted); p.textAlign(p.CENTER, p.CENTER); p.textFont("IBM Plex Mono"); p.textSize(13);
      p.text("Serve over http:// to load the live atlas (data/atlas.json)", w / 2, h / 2);
      p.pop();
    }
  };
})();
