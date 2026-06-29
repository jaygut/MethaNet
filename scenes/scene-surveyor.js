/* SCENE - THE RISK SURVEYOR  ·  REAL DATA  ·  the value-prop clincher.
   The real molecular map, reframed as a geological survey. PROVEN DISTRICTS (rumen +
   wetland reference, data-rich) glow charted on the left. THE FRONTIER (the mangrove
   expansion, unmeasured) spreads dim on the right. A survey scan sweeps left to right,
   RANKING thousands of frontier genomes by molecular fingerprint and lighting a gold
   "measure here" flag wherever one matches a proven reference (real nearest-reference
   links). On the far right, the ASSAY GATE - field methane flux - stays dark: we rank
   where to drill, we never report grade. Honest by construction: a survey is not an
   assay. Uses real diffusion coords + the documented 372 links. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.surveyor = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const ECOC = {}; EB.ecosystems.forEach((e) => (ECOC[e.code] = e.color));
    const GOLD = "#FFC75A";
    let rng, pts = [], proven = [], frontier = [], gold = [], region = {}, cent = {}, gate = {}, ready = false;

    function project() {
      const w = ctx.W, h = ctx.H;
      region = { cx: w * 0.44, cy: h * 0.46, s: Math.min(w, h) * 0.39 };
      const atlas = ctx.data && ctx.data.atlas;
      if (!atlas) { ready = false; return; }
      rng = window.EBRandom.RNG("surveyor");
      pts = atlas.points.map((pt) => ({
        x: region.cx + pt.x * region.s, y: region.cy - pt.y * region.s,
        e: pt.e, br: pt.br, cs: pt.cs, ma: pt.ma || 0, nps: pt.nps || 0,
        pr: D.clamp(((pt.nps || 0) - 0.97) * 30, 0, 1),   // survey priority (nearest-reference similarity)
        col: ECOC[pt.e] || "#889", ph: rng.range(0, 6.2831),
      }));
      proven = pts.filter((q) => q.e < 2);
      frontier = pts.filter((q) => q.e >= 2);
      gold = atlas.bridges.filter((b) => b.cs).map((b) => {
        let a = pts[b.s], c = pts[b.t];
        const flag = a.e >= 2 ? a : c, ref = a.e >= 2 ? c : a;
        return { flag, ref, w: b.w };
      }).filter((g) => g.flag && g.ref).sort((a, b) => b.w - a.w);
      cent = {};
      EB.ecosystems.forEach((eco) => {
        const g = pts.filter((q) => q.e === eco.code);
        if (g.length) cent[eco.key] = { x: g.reduce((s, q) => s + q.x, 0) / g.length, y: g.reduce((s, q) => s + q.y, 0) / g.length };
      });
      gate = { x: w * 0.82, cy: region.cy, gh: region.s * 1.15 };
      ready = true;
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); project(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); project(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.28, 90);
      if (!ready) { drawNoData(w, h); return; }

      const settle = D.easeInOut(D.window01(t, 0.0, 0.22));
      const scan = D.window01(t, 0.16, 0.74);
      const closeOut = D.window01(t, 0.7, 1.0);
      const tm = ctx.reduced ? 0.5 : p.frameCount * 0.02;
      const x0 = region.cx - region.s * 1.05;
      const scanX = ctx.reduced ? gate.x : D.lerp(x0, gate.x - 20, D.easeInOut(scan));
      const dc = p.drawingContext;

      // --- proven-district charted halo + survey contours (rumen + wetland) ---
      if (settle > 0.05) {
        for (const key of ["rumen", "wetland"]) {
          const c = cent[key]; if (!c) continue;
          const col = ECOC[key === "rumen" ? 0 : 1];
          const g = dc.createRadialGradient(c.x, c.y, 0, c.x, c.y, region.s * 0.42);
          g.addColorStop(0, D.rgba(col, 0.13 * settle)); g.addColorStop(1, "rgba(0,0,0,0)");
          dc.save(); dc.fillStyle = g; dc.fillRect(0, 0, w, h); dc.restore();
          p.push(); p.noFill();
          for (let r = 1; r <= 3; r++) { p.stroke(D.rgba(col, 0.18 * settle / r)); p.strokeWeight(1); p.circle(c.x, c.y, region.s * 0.22 * r); }
          p.pop();
        }
      }

      // --- frontier + proven points (batched by style; mz/priority encoded as size) ---
      let ranked = 0;
      for (let i = 0; i < frontier.length; i++) { if (frontier[i].x <= scanX || ctx.reduced) ranked++; }
      dc.save();
      // base frontier: dim, unmeasured (one style)
      dc.fillStyle = "#4C5A66"; dc.globalAlpha = 0.30 + 0.18 * settle;
      for (let i = 0; i < frontier.length; i++) { const q = frontier[i]; if (q.br) continue; dc.fillRect(q.x - 1.2, q.y - 1.2, 2.4, 2.4); }
      // high-priority frontier the scan has ranked: gold (one style; size by priority)
      dc.fillStyle = GOLD; dc.globalAlpha = 0.6;
      for (let i = 0; i < frontier.length; i++) {
        const q = frontier[i]; if (q.br || q.pr < 0.2) continue; if (q.x > scanX && !ctx.reduced) continue;
        const s = 2.2 + q.pr * 1.8; dc.fillRect(q.x - s / 2, q.y - s / 2, s, s);
      }
      // proven: bright, charted (one style)
      dc.fillStyle = "#C7D6E0"; dc.globalAlpha = 0.55 + 0.4 * settle;
      for (let i = 0; i < proven.length; i++) { const q = proven[i]; if (q.br) continue; dc.fillRect(q.x - 1.5, q.y - 1.5, 3, 3); }
      dc.globalAlpha = 1; dc.restore();

      // --- the survey scan line + reticle ---
      if (settle > 0.4 && scan > 0.001 && scan < 0.999 && !ctx.reduced) {
        p.push(); p.blendMode(p.ADD);
        p.stroke(D.rgba(EB.color.attested, 0.28)); p.strokeWeight(1.5); p.line(scanX, h * 0.13, scanX, h * 0.9);
        p.noStroke(); p.fill(D.rgba(EB.color.attested, 0.045)); p.rect(x0, 0, scanX - x0, h);
        p.pop();
        p.push(); p.noFill(); p.stroke(D.rgba(EB.color.attested, 0.7)); p.strokeWeight(1);
        p.line(scanX - 7, region.cy, scanX + 7, region.cy); p.line(scanX, region.cy - 7, scanX, region.cy + 7); p.circle(scanX, region.cy, 16);
        p.pop();
        D.label(p, "surveying fingerprints", scanX - 10, h * 0.9 + 13, D.rgba(EB.color.attested, 0.65), 9, [p.RIGHT, p.TOP]);
      }

      // --- gold "measure here" links + flags for the top documented matches ---
      const litGold = [];
      for (const g of gold) {
        if (g.flag.x > scanX && !ctx.reduced) continue;
        litGold.push(g);
        p.push(); p.blendMode(p.ADD);
        p.stroke(D.rgba(GOLD, 0.55)); p.strokeWeight(1.2); p.line(g.flag.x, g.flag.y, g.ref.x, g.ref.y);
        D.glow(p, g.flag.x, g.flag.y, 2.2, GOLD, ctx.reduced ? 0.8 : 0.5 + 0.4 * Math.sin(tm * 2 + g.flag.ph));
        p.pop();
        drawFlag(g.flag.x, g.flag.y);
      }

      drawAssayGate(closeOut, tm, litGold);
      drawRegionLabels(settle);
      drawHud(w, h, ranked, litGold.length, settle);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawFlag(x, y) {
      p.push();
      p.stroke(D.rgba(GOLD, 0.9)); p.strokeWeight(1.1); p.noFill();
      p.line(x - 7, y, x + 7, y); p.line(x, y - 7, x, y + 7); p.circle(x, y, 12);
      p.pop();
    }

    function drawAssayGate(a, tm, litGold) {
      const x = gate.x, cy = gate.cy, gh = gate.gh, gw = 18;
      // dashed gold "measure these first" arrows from the top flags toward the gate wall
      if (a > 0.05) {
        p.push(); p.drawingContext.setLineDash([4, 4]);
        for (let i = 0; i < Math.min(3, litGold.length); i++) {
          const f = litGold[i].flag;
          p.stroke(D.rgba(GOLD, 0.4 * a)); p.strokeWeight(1); p.line(f.x, f.y, x - gw, D.lerp(cy - 34, cy + 34, i / 2));
        }
        p.drawingContext.setLineDash([]); p.pop();
      }
      // the dark assay gate - a wall we have not crossed. Never lights (field flux unmeasured).
      p.push();
      p.drawingContext.setLineDash([6, 5]);
      p.noFill(); p.stroke(D.rgba(EB.color.methaneA, 0.4 + 0.35 * a)); p.strokeWeight(1.5);
      p.rectMode(p.CENTER); p.rect(x, cy, gw, gh, 4);
      p.drawingContext.setLineDash([]);
      // labels to the LEFT of the gate (clear of the right edge and the rail)
      p.noStroke(); p.textAlign(p.RIGHT, p.BOTTOM);
      p.fill(D.rgba(EB.color.methaneA, 0.7 + 0.3 * a)); p.textFont("Space Grotesk"); p.textStyle(p.BOLD); p.textSize(13);
      p.text("FIELD FLUX", x - gw, cy - 8); p.textStyle(p.NORMAL);
      p.textAlign(p.RIGHT, p.TOP); p.textFont("JetBrains Mono");
      p.fill(D.rgba(EB.color.textMuted, 0.7 + 0.25 * a)); p.textSize(9);
      p.text("the core assay", x - gw, cy + 1);
      // "UNMEASURED" pill, left of the gate
      const pw = 92, ph = 17, px = x - gw - pw, py = cy + 20;
      p.noFill(); p.stroke(D.rgba(EB.color.methaneB, 0.45 + 0.35 * a)); p.strokeWeight(1); p.rect(px + pw / 2, py + ph / 2, pw, ph, 8);
      p.noStroke(); p.fill(D.rgba(EB.color.methaneB, 0.75 + 0.25 * a)); p.textAlign(p.CENTER, p.CENTER); p.textSize(8.5);
      p.text("UNMEASURED", px + pw / 2, py + ph / 2 + 0.5);
      p.pop();
    }

    function drawRegionLabels(a) {
      if (a < 0.3) return;
      p.push(); p.textFont("Space Grotesk");
      const defs = [
        { key: "rumen", name: "PROVEN DISTRICTS", sub: "rumen + wetland reference", dy: -region.s * 0.5, col: ECOC[0] },
        { key: "mangrove_msm", name: "THE FRONTIER", sub: "mangrove, unmeasured", dy: -26, col: ECOC[2] },
      ];
      for (const d of defs) {
        const c = cent[d.key]; if (!c) continue;
        const lx = Math.min(Math.max(c.x, 82), ctx.W - 82); // keep labels inside the frame on narrow screens
        p.noStroke(); p.textAlign(p.CENTER, p.CENTER);
        p.fill(D.rgba(d.col, 0.95 * a)); p.textStyle(p.BOLD); p.textSize(13);
        p.text(d.name, lx, c.y + d.dy); p.textStyle(p.NORMAL);
        p.fill(D.rgba(EB.color.textMuted, 0.85 * a)); p.textFont("JetBrains Mono"); p.textSize(9);
        p.text(d.sub, lx, c.y + d.dy + 14); p.textFont("Space Grotesk");
      }
      p.pop();
    }

    function drawHud(w, h, ranked, flags, settle) {
      const narrow = w < 640;
      const x = Math.max(16, w * 0.04), y = h * (narrow ? 0.12 : 0.15);
      p.push();
      D.label(p, narrow ? "MOLECULAR SURVEY" : "MOLECULAR SURVEY · THE DRILL-WHERE MAP", x, y, EB.color.textMuted, 10);
      p.fill(EB.color.textPrimary); p.noStroke(); p.textFont("JetBrains Mono"); p.textSize(narrow ? 22 : 26); p.textAlign(p.LEFT, p.TOP);
      p.text(D.fmt(ranked), x, y + 12);
      D.label(p, "frontier genomes ranked by fingerprint", x, y + (narrow ? 40 : 46), D.rgba(EB.color.textPrimary, 0.8), narrow ? 9 : 10);
      p.fill(GOLD); p.noStroke(); p.textFont("JetBrains Mono"); p.textSize(narrow ? 11 : 13); p.textAlign(p.LEFT, p.TOP);
      p.text(D.fmt(flags) + " measure-here matches", x, y + (narrow ? 54 : 62));
      p.pop();
      // persistent honesty readout (top-right on desktop; tucked under the survey count on mobile)
      if (narrow) {
        D.label(p, "RANKING, NOT ASSAY", x, y + 74, D.rgba(EB.color.attested, 0.9), 10);
      } else {
        D.label(p, "RANKING, NOT ASSAY", w - 18, h * 0.15, D.rgba(EB.color.attested, 0.9), 11, [p.RIGHT, p.TOP]);
        D.label(p, "we rank where to measure; the field flux is the assay", w - 18, h * 0.15 + 15, D.rgba(EB.color.textMuted, 0.8), 9, [p.RIGHT, p.TOP]);
      }
    }

    function drawNoData(w, h) {
      p.push(); p.fill(EB.color.textMuted); p.textAlign(p.CENTER, p.CENTER); p.textFont("JetBrains Mono"); p.textSize(13);
      p.text("Serve over http:// to load the live atlas (data/atlas.json)", w / 2, h / 2);
      p.pop();
    }
  };
})();
