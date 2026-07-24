/* SCENE - ONE ENGINE, MANY MAPS  ·  REAL DATA + HONESTLY-EMPTY SLOTS
   The platform payoff. The SAME genome atlas, read through a swappable "gene lens."
   LENS 01 = methane: only the 625-unit POC core lights up on its curated,
   within-contract methane-marker density (the mz field). The other atlas lanes remain
   visible but unweighted because their functional payloads are not yet quantitatively
   harmonized. This is mechanism screening, never measured flux. LENS 02 = nitrous
   oxide: the lens advances and the SAME points dim to a uniform cool wash - no new
   coordinates, no clusters, no per-genome weights - an empty plate the lens is aimed
   at, honestly labelled "candidate, not built, 0 paired flux." LENS 03 = sulfur/DMS:
   a faint unfilled outline only. The maps tray accumulates: one bounded screen live,
   the rest architecturally ready but uncalibrated. Why it is fundable: same engine,
   same sequencing data, the next high-GWP gas is a lens swap, not a new instrument.
   HONESTY: atlas.json carries ZERO denitrification/sulfur fields, so the non-methane
   lenses draw NO per-genome signal - dimming the real cloud, never inventing one. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.engine = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const ECOC = {}; EB.ecosystems.forEach((e) => (ECOC[e.code] = e.color));
    const COOL_N2O = "#7C8CC4", COOL_S = "#A9B6CE", NEUTRAL = "#52606C";
    const LENSES = [
      { name: "METHANE", sub: "POC mechanism screen", col: EB.color.methaneA },
      { name: "NITROUS OXIDE", sub: "candidate, not built", col: COOL_N2O },
      { name: "SULFUR · DMS", sub: "not built", col: COOL_S },
    ];
    let rng, pts = [], tierLo = [], tierMid = [], tierHi = [], region = {}, ready = false;
    let manualSel = null, hoverSel = -1;   // interactivity: click a lens to lock it; hover to preview

    function project() {
      const w = ctx.W, h = ctx.H;
      region = w < 640
        ? { cx: w * 0.5, cy: h * 0.5, s: Math.min(w, h) * 0.36 }
        : { cx: w * 0.55, cy: h * 0.45, s: Math.min(w, h) * 0.30 };
      const atlas = ctx.data && ctx.data.atlas;
      if (!atlas) { ready = false; return; }
      rng = window.EBRandom.RNG("engine");
      // Diffusion view: the SAME projection as the Evidence atlas, so this is literally the same
      // map re-read by a gas lens. One connected fan (no stray outlier clusters, unlike PHATE/PCA here).
      pts = atlas.points.map((pt) => ({
        x: region.cx + (pt.x || 0) * region.s,
        y: region.cy - (pt.y || 0) * region.s,
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

    // lens-tab geometry, shared by the renderer and the click/hover hit-test
    function tabGeom() {
      const w = ctx.W, h = ctx.H, narrow = w < 640;
      const y = h * (narrow ? 0.25 : 0.12);
      const segW = narrow ? (w - 36) / 3 - 8 : Math.min(w * 0.11, 132), gap = narrow ? 8 : 10;
      const totalW = 3 * segW + 2 * gap, x0 = narrow ? 18 : region.cx - totalW / 2;
      return { x0, y, segW, gap, bh: 26 };
    }
    function tabAt(mx, my) {
      const g = tabGeom();
      for (let i = 0; i < 3; i++) { const x = g.x0 + i * (g.segW + g.gap); if (mx >= x && mx <= x + g.segW && my >= g.y - 4 && my <= g.y + g.bh + 4) return i; }
      return -1;
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); project(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); project(); };

    // interactivity: click a gene lens to lock the map to that gas; hover to preview
    p.mousePressed = function () {
      const i = tabAt(p.mouseX, p.mouseY);
      if (i >= 0) { manualSel = i; if (ctx.reduced && p.redraw) p.redraw(); return false; }
    };
    p.mouseMoved = function () {
      hoverSel = tabAt(p.mouseX, p.mouseY);
      if (p.cursor) p.cursor(hoverSel >= 0 ? "pointer" : "default");
    };

    // lens selector 0=methane, 1=n2o, 2=sulfur; returns to methane for the closing tray
    function lensSel(t) {
      if (ctx.reduced) return 0;
      if (t < 0.28) return 0;
      if (t < 0.40) return D.easeInOut(D.window01(t, 0.28, 0.40));
      if (t < 0.56) return 1;
      if (t < 0.68) return 1 + D.easeInOut(D.window01(t, 0.56, 0.68));
      if (t < 0.82) return 2;
      if (t < 0.92) return 2 - 2 * D.easeInOut(D.window01(t, 0.82, 0.92));
      return 0;
    }

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.24, 100);
      if (!ready) { drawNoData(w, h); return; }

      const settle = D.easeInOut(D.window01(t, 0.0, 0.20));
      const sel = manualSel != null ? manualSel : lensSel(t);   // click locks the lens, else scroll drives it
      const methaneW = D.clamp(sel <= 1 ? 1 - sel : 0, 0, 1) * settle;
      const n2oW = D.clamp(sel <= 1 ? sel : 2 - sel, 0, 1);
      const sulfW = D.clamp(sel >= 1 ? sel - 1 : 0, 0, 1);
      const coolW = D.clamp(n2oW + sulfW, 0, 1) * settle;
      const tm = ctx.reduced ? 0.4 : p.frameCount * 0.02;

      drawCloud(methaneW, coolW, n2oW, sulfW, settle, tm);
      drawLensBar(w, h, sel, settle, tm);
      drawPlates(w, h, t);
      drawHud(w, h, settle);
      if (w >= 640 && (D.window01(t, 0.80, 0.92) > 0.01 || ctx.reduced)) drawFactCard(w, h, ctx.reduced ? 1 : D.window01(t, 0.80, 0.95));
      drawHonestyBand(w, h, settle);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawCloud(methaneW, coolW, n2oW, sulfW, settle, tm) {
      const dc = p.drawingContext;
      dc.save();
      // always-present structure so the whole shared atlas fan reads (all 7,710 genomes),
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

    function drawLensBar(w, h, sel, settle, tm) {
      const narrow = w < 640;
      const labels = narrow ? ["CH₄", "N₂O", "DMS"] : LENSES.map((l) => l.name);
      const g = tabGeom(), x0 = g.x0, y = g.y, segW = g.segW, gap = g.gap;
      const hint = (!narrow && manualSel == null && settle > 0.5) ? "GENE LENS · click to swap" : "GENE LENS";
      D.label(p, hint, x0, y - 16, D.rgba(EB.color.emergence, 0.85 * settle), 9.5);
      for (let i = 0; i < 3; i++) {
        const x = x0 + i * (segW + gap);
        const prox = D.clamp(1 - Math.abs(sel - i), 0, 1);     // 1 when this lens is active
        const active = prox > 0.5;
        const hovered = hoverSel === i && !active;
        p.push(); p.rectMode(p.CORNER);
        p.noFill(); p.stroke(D.rgba(active ? EB.color.emergence : (hovered ? EB.color.attested : EB.color.textMuted), (active ? 0.85 : hovered ? 0.7 : 0.35) * settle));
        p.strokeWeight(active ? 1.5 : 1);
        if (!active) p.drawingContext.setLineDash([3, 3]);
        p.rect(x, y, segW, g.bh, 4); p.drawingContext.setLineDash([]);
        p.noStroke(); p.fill(D.rgba(active ? EB.color.textPrimary : (hovered ? EB.color.textPrimary : EB.color.textMuted), (active ? 1 : hovered ? 0.85 : 0.55) * settle));
        p.textAlign(p.CENTER, p.CENTER); p.textFont("IBM Plex Mono"); p.textSize(narrow ? 9 : (i === 2 ? 8.5 : 9.5));
        p.text(labels[i], x + segW / 2, y + 13);
        p.pop();
      }
      // beam from the active lens down into the shared atlas (the engine reading it)
      if (!ctx.reduced && settle > 0.4) {
        const ax = x0 + sel * (segW + gap) + segW / 2;
        p.push(); p.blendMode(p.ADD); p.strokeWeight(1.2);
        p.stroke(D.rgba(EB.color.emergence, 0.30 * settle));
        p.line(ax, y + g.bh + 2, region.cx, region.cy - region.s * 0.45);
        p.pop();
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
        // mini-swatch: filled gradient for methane, empty/aimed for the rest
        const sw = 26, sx = xx + 10, sy = y + (ph - sw) / 2;
        if (i === 0) {
          const g = p.drawingContext.createLinearGradient(sx, 0, sx + sw, 0);
          g.addColorStop(0, EB.color.methaneA); g.addColorStop(1, EB.color.methaneB);
          p.drawingContext.fillStyle = g; p.drawingContext.globalAlpha = a; p.noStroke();
          p.rect(sx, sy, sw, sw, 3); p.drawingContext.globalAlpha = 1;
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
        p.text(i === 0 ? "625-unit POC screen · 0 flux" : i === 1 ? "candidate · 0 paired flux" : "not built", sx + sw + 10, y + ph / 2 + 2);
        p.pop();
      }
    }

    function drawHud(w, h, settle) {
      const x = Math.max(16, w * 0.04), y = h * 0.13;
      D.label(p, "ONE ENGINE · MANY MAPS", x, y, D.rgba(EB.color.emergence, 0.95 * settle), 11);
      D.label(p, "same atlas, swap the gene lens", x, y + 15, D.rgba(EB.color.textMuted, 0.9 * settle), 9.5);
      // One mechanism-comparable core / two candidate lenses.
      p.push(); p.textAlign(p.LEFT, p.TOP); p.textFont("IBM Plex Mono");
      p.fill(D.rgba(EB.color.methaneA, settle)); p.textSize(12); p.text("625-unit POC screen live", x, y + 34);
      p.fill(D.rgba(EB.color.textMuted, 0.9 * settle)); p.textSize(10); p.text("2 lenses ready, uncalibrated", x, y + 50);
      p.pop();
    }

    function drawFactCard(w, h, a) {
      const cw = Math.min(w * 0.38, 384), x = D.clamp(w * 0.54, 16, w - cw - 16), y = h * 0.60;
      p.push(); p.rectMode(p.CORNER);
      p.fill(D.rgba(EB.color.bgPanel, 0.8 * a)); p.stroke(D.rgba(COOL_N2O, 0.5 * a)); p.strokeWeight(1);
      p.rect(x, y, cw, 50, 4); p.noStroke();
      D.label(p, "WHY THE SECOND GAS MATTERS", x + 14, y + 16, D.rgba(COOL_N2O, a), 8.5);
      p.fill(D.rgba(EB.color.textPrimary, a)); p.textFont("Hanken Grotesk"); p.textSize(11.5); p.textAlign(p.LEFT, p.TOP);
      p.text("Nitrous oxide traps 273× the heat of CO₂ over 100 years.", x + 14, y + 24);
      p.fill(D.rgba(EB.color.textMuted, a)); p.textSize(10);
      p.text("Today's blue-carbon rules can still count it as zero.", x + 14, y + 38);
      p.pop();
    }

    function drawHonestyBand(w, h, settle) {
      p.push(); p.textAlign(p.CENTER, p.BOTTOM); p.textFont("IBM Plex Mono"); p.textSize(9.5);
      p.fill(D.rgba(EB.color.textMuted, 0.85 * settle));
      p.text("POC mechanism screen only; other lanes await harmonization. no flux rate is inferred.", w / 2, h - 16);
      p.pop();
    }

    function drawNoData(w, h) {
      p.push(); p.fill(EB.color.textMuted); p.textAlign(p.CENTER, p.CENTER); p.textFont("IBM Plex Mono"); p.textSize(13);
      p.text("Serve over http:// to load the live atlas (data/atlas.json)", w / 2, h / 2);
      p.pop();
    }
  };
})();
