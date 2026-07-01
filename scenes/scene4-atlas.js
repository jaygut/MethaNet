/* SCENE 5 - THE EVIDENCE  ·  REAL DATA  ·  the source-audited atlas the ranking stands on.
   The stunning projection: every point is a real genome at its true diffusion-map coordinate
   (data/atlas.json). Rumen and wetland references fan out on the left, the mangrove expansion
   forms the line on the right, and the biomes separate in molecular space. On scroll the cloud
   resolves from noise, then the BRIDGE links light up: gold links join a mangrove genome to its
   nearest reference genome, faint teal links are the wider cross-ecosystem neighborhood, and
   knowledge flows along the gold bridges. This is what every ranked call in "What You Get"
   traces back to: real genomes, projected across ecosystems, source-audited. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.atlas = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const ECOC = {}; EB.ecosystems.forEach((e) => (ECOC[e.code] = e.color));
    const GOLD = "#FFC75A"; // bridge-to-nearest-reference links (the bridge genomes)
    let rng, pts = [], byEco = [[], [], [], []], gold = [], teal = [], pulses = [], region = {}, cent = {}, ready = false;
    let bridgeNodes = [], csMangrove = [];

    function project() {
      const w = ctx.W, h = ctx.H;
      const s = Math.min(w, h) * 0.40;
      region = { cx: w * 0.48, cy: h * 0.45, s };
      const atlas = ctx.data && ctx.data.atlas;
      if (!atlas) { ready = false; return; }
      rng = window.EBRandom.RNG("evidence");
      pts = atlas.points.map((pt) => {
        const sx = region.cx + pt.x * s, sy = region.cy - pt.y * s;
        const a = rng.range(0, 6.2831), r = rng.range(0, Math.min(w, h) * 0.5);
        return { sx, sy, nsx: region.cx + Math.cos(a) * r, nsy: region.cy + Math.sin(a) * r,
          e: pt.e, br: pt.br, cs: pt.cs, ma: pt.ma || 0, mz: pt.mz || 0,
          col: ECOC[pt.e] || "#889", ph: rng.range(0, 6.2831) };
      });
      byEco = [[], [], [], []];
      pts.forEach((pt) => byEco[pt.e].push(pt));
      bridgeNodes = pts.filter((pt) => pt.br);
      csMangrove = pts.filter((pt) => pt.cs && pt.e >= 2);
      gold = atlas.bridges.filter((b) => b.cs).sort((a, b) => b.w - a.w);
      teal = atlas.bridges.filter((b) => !b.cs).sort((a, b) => b.w - a.w);
      cent = {};
      EB.ecosystems.forEach((eco) => {
        const g = byEco[eco.code];
        if (g.length) cent[eco.key] = { x: g.reduce((a, q) => a + q.sx, 0) / g.length, y: g.reduce((a, q) => a + q.sy, 0) / g.length };
      });
      pulses = [];
      ready = true;
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); project(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); project(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.22, 110);
      if (!ready) { drawNoData(w, h); return; }

      const resolve = D.easeInOut(D.window01(t, 0.0, 0.30));
      const clarify = D.window01(t, 0.22, 0.5);
      const tealA = D.window01(t, 0.46, 0.70);
      const goldA = D.window01(t, 0.60, 0.85);
      const flow = D.window01(t, 0.74, 1.0);
      const dc = p.drawingContext;

      for (const pt of pts) { pt.x = D.lerp(pt.nsx, pt.sx, resolve); pt.y = D.lerp(pt.nsy, pt.sy, resolve); }

      if (resolve > 0.15 && cent.mangrove_msm) {
        const mc = cent.mangrove_msm;
        const gx = dc.createRadialGradient(mc.x, mc.y, 0, mc.x, mc.y, region.s * 1.0);
        gx.addColorStop(0, D.rgba(EB.color.mangroveMsm, 0.11 * resolve));
        gx.addColorStop(0.55, D.rgba(EB.color.mangroveFutian, 0.04 * resolve));
        gx.addColorStop(1, "rgba(0,0,0,0)");
        dc.save(); dc.fillStyle = gx; dc.fillRect(0, 0, w, h); dc.restore();
      }

      // ---- bulk points: batched fillRect by ecosystem ----
      dc.save();
      for (let e = 0; e < 4; e++) {
        const grp = byEco[e]; if (!grp.length) continue;
        const poc = e < 2;
        dc.fillStyle = D.lerpHex("#586771", ECOC[e], clarify);
        dc.globalAlpha = poc ? 0.62 + 0.36 * clarify : 0.42 + 0.42 * clarify;
        for (let i = 0; i < grp.length; i++) {
          const pt = grp[i]; if (pt.br) continue;
          const sz = (poc ? 2.4 : 1.4) + pt.mz * 1.8 * clarify;
          dc.fillRect(pt.x - sz * 0.5, pt.y - sz * 0.5, sz, sz);
        }
      }
      dc.globalAlpha = 1; dc.restore();

      // ---- teal cross-ecosystem neighborhood links ----
      const nTeal = Math.round(tealA * Math.min(teal.length, 240));
      if (nTeal > 0) {
        p.push(); p.blendMode(p.ADD); p.strokeWeight(1);
        p.stroke(D.rgba(EB.color.emergence, 0.07 + 0.05 * tealA));
        for (let i = 0; i < nTeal; i++) { const b = teal[i], a = pts[b.s], c = pts[b.t]; if (!a || !c) continue; p.line(a.x, a.y, c.x, c.y); }
        p.pop();
      }

      // ---- gold bridge links: each joins a bridge genome to its nearest reference ----
      const nGold = Math.round(goldA * gold.length);
      if (nGold > 0) {
        p.push(); p.blendMode(p.ADD);
        for (let i = 0; i < nGold; i++) {
          const b = gold[i], a = pts[b.s], c = pts[b.t]; if (!a || !c) continue;
          p.stroke(D.rgba(GOLD, 0.30 + 0.45 * goldA)); p.strokeWeight(1.4);
          p.line(a.x, a.y, c.x, c.y);
        }
        p.pop();
      }

      // ---- knowledge flows along the gold bridges (reference -> bridge genome) ----
      if (flow > 0.02 && !ctx.reduced && nGold > 0) {
        if (p.frameCount % 4 === 0 && pulses.length < 48) pulses.push({ b: gold[rng.int(0, nGold - 1)], u: 0, sp: rng.range(0.01, 0.022) });
        p.push(); p.blendMode(p.ADD); p.noStroke();
        for (let i = pulses.length - 1; i >= 0; i--) {
          const pu = pulses[i]; pu.u += pu.sp; if (pu.u >= 1) { pulses.splice(i, 1); continue; }
          let a = pts[pu.b.s], c = pts[pu.b.t]; if (a.e > c.e) { const tmp = a; a = c; c = tmp; }
          const x = D.lerp(a.x, c.x, pu.u), y = D.lerp(a.y, c.y, pu.u);
          D.glow(p, x, y, 2.2, D.lerpHex(EB.color.attested, GOLD, pu.u), 0.85 * (1 - pu.u));
        }
        p.pop();
      } else if (ctx.reduced && flow > 0.3) {
        p.push(); p.blendMode(p.ADD); p.noStroke();
        for (let i = 0; i < nGold; i++) { const b = gold[i]; let a = pts[b.s], c = pts[b.t]; if (a.e > c.e) { const t2 = a; a = c; c = t2; } D.glow(p, D.lerp(a.x, c.x, 0.55), D.lerp(a.y, c.y, 0.55), 2.2, GOLD, 0.7); }
        p.pop();
      }

      // ---- bridge-genome nodes ----
      const nodeA = 0.4 + 0.6 * Math.max(goldA, tealA);
      p.push(); p.blendMode(p.ADD); p.noStroke();
      for (const pt of bridgeNodes) {
        if (pt.cs) { D.glow(p, pt.x, pt.y, 2.4 + pt.ma * 1.6, GOLD, 0.65 * nodeA); }
        else { p.fill(D.rgba(pt.col, (0.35 + pt.ma * 0.5) * nodeA)); p.circle(pt.x, pt.y, 2.6 + pt.ma * 1.4); }
      }
      p.pop();
      if (goldA > 0.2) {
        p.noFill(); p.strokeWeight(1.2);
        let labeled = 0;
        for (const pt of csMangrove) {
          p.stroke(D.rgba(GOLD, 0.65 * goldA)); p.circle(pt.x, pt.y, 13);
          if (labeled < 4 && goldA > 0.6) { p.noStroke(); p.fill(D.rgba(GOLD, goldA)); p.textFont("JetBrains Mono"); p.textSize(8.5); p.textAlign(p.LEFT, p.CENTER); p.text("bridge", pt.x + 9, pt.y); p.noFill(); labeled++; }
        }
      }

      drawBasinLabels(clarify, resolve);
      drawLegend(w, h, clarify, nGold, tealA);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawBasinLabels(clarify, resolve) {
      if (clarify < 0.25) return;
      const a = clarify * resolve;
      const defs = [
        { key: "rumen", name: "RUMEN", sub: "reference · methanogens", dx: 16, dy: -10, ah: p.LEFT },
        { key: "wetland", name: "WETLAND", sub: "target reference", dx: 16, dy: 10, ah: p.LEFT },
        { key: "mangrove_msm", name: "MANGROVE EXPANSION", sub: "the genomes we screen", dx: 0, dy: -24, ah: p.CENTER },
      ];
      p.push();
      for (const d of defs) {
        const s = cent[d.key]; if (!s) continue;
        const col = ECOC[EB.ecosystems.find((e) => e.key === d.key).code];
        const lx = s.x + d.dx, ly = s.y + d.dy;
        p.noStroke(); p.textAlign(d.ah, p.CENTER);
        p.fill(D.rgba(col, 0.95 * a)); p.textFont("Space Grotesk"); p.textStyle(p.BOLD); p.textSize(13);
        p.text(d.name, lx, ly); p.textStyle(p.NORMAL);
        p.fill(D.rgba(EB.color.textMuted, 0.85 * a)); p.textFont("JetBrains Mono"); p.textSize(9);
        p.text(d.sub, lx, ly + 13);
      }
      p.pop();
    }

    function drawLegend(w, h, clarify, nGold, tealA) {
      const x = Math.max(16, w * 0.04), y = h * 0.15;
      p.push();
      D.label(p, "SOURCE-AUDITED ATLAS · DIFFUSION VIEW", x, y, EB.color.textMuted, 10);
      D.label(p, "real coordinates · " + D.fmt(pts.length) + " genomes mapped", x, y + 15, D.rgba(EB.color.attested, 0.9), 10);
      let i = 0; const ly = y + 36;
      for (const e of EB.ecosystems) {
        const yy = ly + i * 15;
        p.noStroke(); p.fill(D.rgba(e.color, 0.5 + 0.5 * clarify)); p.circle(x + 4, yy - 3, 6);
        D.label(p, e.label + "  ·  " + D.fmt(e.count), x + 14, yy, D.rgba(EB.color.textPrimary, 0.85), 10);
        i++;
      }
      let yy = ly + i * 15 + 6;
      p.stroke(D.rgba(GOLD, 0.9)); p.strokeWeight(1.6); p.line(x, yy - 3, x + 11, yy - 3); p.noStroke();
      D.label(p, "bridge genome → nearest reference", x + 16, yy, D.rgba(EB.color.textPrimary, 0.9), 10);
      yy += 15;
      p.stroke(D.rgba(EB.color.emergence, 0.6)); p.strokeWeight(1.2); p.line(x, yy - 3, x + 11, yy - 3); p.noStroke();
      D.label(p, "cross-ecosystem neighbor", x + 16, yy, D.rgba(EB.color.textMuted, 0.9), 10);
      p.pop();
    }

    function drawNoData(w, h) {
      p.push(); p.fill(EB.color.textMuted); p.textAlign(p.CENTER, p.CENTER); p.textFont("JetBrains Mono"); p.textSize(13);
      p.text("Serve over http:// to load the live atlas (data/atlas.json)", w / 2, h / 2);
      p.pop();
    }
  };
})();
