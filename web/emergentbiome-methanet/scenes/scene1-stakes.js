/* SCENE 1 - THE STAKES (illustrative; the one hard number, CH4 ~30x GWP, is real).
   Cross-section of a coastal sediment: CO2 particles settle and build a stored-carbon
   layer; CH4 bubbles rise from vents and, weighted by ~30x GWP, erode a "net climate
   benefit" meter. Methane emission rate rises with scroll - the sink runs in reverse. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.stakes = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const GWP = EB.num.methaneGWP;
    const SINK = EB.color.wetland, M_A = EB.color.methaneA, M_B = EB.color.methaneB;
    let rng, co2 = [], ch4 = [], vents = [], sedY = 0, settled = 0;

    function layout() {
      rng = window.EBRandom.RNG("stakes");
      sedY = ctx.H * 0.72;
      co2 = []; ch4 = []; settled = 0;
      const N = Math.round(D.clamp(ctx.W / 9, 60, 150));
      for (let i = 0; i < N; i++) co2.push(newCO2(true));
      vents = [];
      const vN = Math.max(4, Math.round(ctx.W / 220));
      for (let i = 0; i < vN; i++) vents.push({ x: (i + 0.5) / vN * ctx.W + rng.gauss(0, 18), pile: 0 });
    }
    function newCO2(seedY) {
      return { x: rng.range(0, ctx.W), y: seedY ? rng.range(0, sedY) : -10,
        vy: rng.range(0.25, 0.7), sz: rng.range(1.2, 2.6), drift: rng.range(-0.2, 0.2), ph: rng.range(0, 6.28) };
    }
    function spawnCH4() {
      const v = rng.pick(vents);
      ch4.push({ x: v.x + rng.gauss(0, 6), y: sedY - 2, vy: rng.range(0.8, 1.7),
        sz: rng.range(2.2, 5.5), wob: rng.range(0.6, 1.6), ph: rng.range(0, 6.28), pop: 0 });
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear();
      // water column gradient (deep field)
      const g = p.drawingContext.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, "#070C11"); g.addColorStop(0.7, "#081016"); g.addColorStop(1, "#0A0F0C");
      p.drawingContext.fillStyle = g; p.drawingContext.fillRect(0, 0, w, h);

      D.instrumentGrid(p, w, h, EB.color.hairline, 0.25, 90);

      // sediment band
      p.noStroke();
      p.fill("#0C1208"); p.rect(0, sedY, w, h - sedY);
      // stored-carbon layer thickness grows as CO2 settles (illustrative sink)
      const storeH = D.clamp(settled / 900) * (h - sedY) * 0.6;
      p.fill(D.rgba(SINK, 0.18)); p.rect(0, sedY, w, storeH);
      p.stroke(D.rgba(SINK, 0.5)); p.strokeWeight(1); p.line(0, sedY, w, sedY);
      p.noStroke();

      // CO2 settling
      const sink = ctx.reduced ? 0 : 1;
      for (const c of co2) {
        c.y += c.vy * sink; c.x += Math.sin(c.ph += 0.02) * c.drift;
        if (c.y >= sedY - c.sz) { settled += 0.6; c.x = rng.range(0, w); c.y = -8; c.vy = rng.range(0.25, 0.7); }
        p.fill(D.rgba("#9FB4C0", 0.5)); p.circle(c.x, c.y, c.sz);
      }

      // CH4 emission rate rises strongly with scroll
      const rate = (0.12 + t * 1.5) * (ctx.reduced ? 0 : 1);
      let acc = (p._acc || 0) + rate; while (acc >= 1) { spawnCH4(); acc -= 1; } p._acc = acc;
      if (ctx.reduced) { while (ch4.length < Math.round(8 + t * 40)) spawnCH4(); }
      if (ch4.length > 520) ch4.splice(0, ch4.length - 520); // hard cap (safety margin)

      p.push(); p.blendMode(p.ADD);
      for (let i = ch4.length - 1; i >= 0; i--) {
        const b = ch4[i];
        if (!ctx.reduced) { b.y -= b.vy; b.x += Math.sin(b.ph += 0.06) * b.wob; }
        else { b.y = sedY - (1 - (i / ch4.length)) * (sedY - ctx.H * 0.18); }
        const col = D.lerpHex(M_A, M_B, D.clamp((sedY - b.y) / sedY));
        if (b.y < ctx.H * 0.18 || b.pop) {
          b.pop += 1;
          D.glow(p, b.x, b.y, b.sz * (1 + b.pop * 0.4), col, Math.max(0, 0.7 - b.pop * 0.12));
          if (b.pop > 6) ch4.splice(i, 1);
        } else {
          p.noFill(); p.stroke(D.rgba(col, 0.7)); p.strokeWeight(1.2); p.circle(b.x, b.y, b.sz * 2);
          D.glow(p, b.x, b.y, b.sz * 0.7, col, 0.5);
        }
      }
      p.pop();

      drawBalance(w, h, t);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawBalance(w, h, t) {
      // a single "net climate balance" line rides down through a NET ZERO threshold:
      // stored carbon holds it up; methane (x GWP) pushes it down, into net warming.
      const mx = D.clamp(w * 0.06, 24, 80);
      const net = 1 - D.clamp(0.08 + t * 0.9);        // existing erosion model, unchanged
      const topY = h * 0.20, botY = sedY - h * 0.06;
      const frac = 1 - net, zeroFrac = 0.45;
      const zeroY = p.lerp(topY, botY, zeroFrac);
      const lineY = p.lerp(topY, botY, frac);
      const bal = Math.round((zeroFrac - frac) * 180);  // + above zero, - below; 0 at the crossing
      const lineCol = D.lerpHex(M_B, SINK, D.clamp(0.5 + bal / 110, 0, 1));
      // faint directional washes so the empty column reads as the contest
      p.push(); p.noStroke();
      p.fill(D.rgba(M_B, 0.05 * frac)); p.rect(0, topY, w, lineY - topY);
      p.fill(D.rgba(SINK, 0.05 * net)); p.rect(0, lineY, w, botY - lineY);
      p.pop();
      // static NET ZERO threshold + zone tags (pinned right edge, clear of the bottom-left card)
      p.push(); p.stroke(D.rgba(EB.color.hairline, 0.85)); p.strokeWeight(1);
      p.drawingContext.setLineDash([2, 5]); p.line(0, zeroY, w, zeroY);
      p.drawingContext.setLineDash([]); p.pop();
      D.label(p, "NET ZERO", w - mx - 4, zeroY - 6, EB.color.textMuted, 10, [p.RIGHT, p.BOTTOM]);
      D.label(p, "COOLING", w - mx - 4, topY + 14, SINK, 11, [p.RIGHT, p.TOP]);
      D.label(p, "WARMING", w - mx - 4, botY - 6, M_B, 11, [p.RIGHT, p.BOTTOM]);
      // the travelling balance line
      p.push(); p.stroke(lineCol); p.strokeWeight(1.6);
      if (!ctx.reduced) { p.drawingContext.setLineDash([6, 6]); p.drawingContext.lineDashOffset = -(p.frameCount * 0.5); }
      p.line(0, lineY, w, lineY);
      p.drawingContext.setLineDash([]); p.drawingContext.lineDashOffset = 0; p.pop();
      // riding readout (center-right, clear of the bottom-left card)
      const cxr = w * 0.56;
      p.push(); p.noStroke(); p.fill(lineCol); p.circle(cxr - 14, lineY, 6); p.pop();
      D.label(p, (bal > 0 ? "+" : "") + bal + " net", cxr, lineY - 6, lineCol, 14, [p.LEFT, p.BOTTOM]);
      // title + GWP legend (the only hard number, kept verbatim)
      D.label(p, "NET CLIMATE BALANCE", mx, topY - 14, EB.color.textMuted, 11);
      D.label(p, "STORED CARBON ▲   CH₄ ×" + GWP + " GWP ▼", mx, topY + 2, D.rgba(EB.color.textMuted, 0.9), 10.5);
    }
  };
})();
