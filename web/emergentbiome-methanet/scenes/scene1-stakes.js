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

      drawMeter(w, h, t);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawMeter(w, h, t) {
      // net climate benefit meter: stored sink minus methane (x GWP) escaping
      const mx = D.clamp(w * 0.06, 24, 80), my = EB.num ? 96 : 96;
      const barW = Math.min(w * 0.42, 460), barH = 12;
      const erosion = D.clamp(0.08 + t * 0.9); // fraction of benefit eroded by CH4*GWP
      const net = 1 - erosion;
      p.push();
      D.label(p, "NET CLIMATE BENEFIT", mx, my - 12, EB.color.textMuted, 11);
      // track
      p.noStroke(); p.fill(D.rgba(EB.color.hairline, 0.9)); p.rect(mx, my, barW, barH, 2);
      // remaining benefit (green) shrinking
      const col = D.lerpHex(M_B, SINK, net);
      p.fill(col); p.rect(mx, my, barW * net, barH, 2);
      // eroded portion (methane hatch)
      p.fill(D.rgba(M_B, 0.25)); p.rect(mx + barW * net, my, barW * erosion, barH, 2);
      // ticks
      p.stroke(D.rgba(EB.color.textMuted, 0.5)); p.strokeWeight(1);
      for (let i = 0; i <= 4; i++) { const x = mx + (barW / 4) * i; p.line(x, my + barH + 2, x, my + barH + 6); }
      p.noStroke();
      D.label(p, "STORED CARBON ▲   CH₄ ESCAPING ×" + GWP + " GWP ▼", mx, my + barH + 22, D.rgba(EB.color.textMuted, 0.9), 10.5);
      D.label(p, Math.round(net * 100) + "%", mx + barW + 12, my + barH - 1, col, 14, [p.LEFT, p.BOTTOM]);
      p.pop();
    }
  };
})();
