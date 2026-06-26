/* SCENE 3 - THE INSIGHT  ·  HERO  ·  REAL DATA
   The EmergentBiome representation. Every point is a real ESM2 (650M) proteome
   embedding placed at its true diffusion-map coordinate (data/atlas.json). On
   scroll: points resolve from noise into the rumen + wetland basins and the
   mangrove cloud; the 372 documented cross-ecosystem bridge edges illuminate
   (strongest first); then knowledge "diffuses" as pulses travel the bridges from
   the rumen source toward the mangrove target - transfer learning, made visible. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.insight = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const ECOC = {}; EB.ecosystems.forEach((e) => (ECOC[e.code] = e.color));
    let rng, pts = [], byEco = [[], [], [], []], bridges = [], pulses = [], region = {}, ready = false;

    function project() {
      const w = ctx.W, h = ctx.H;
      const s = Math.min(w, h) * 0.56;
      region = { cx: w * 0.52, cy: h * 0.45, s };
      const atlas = ctx.data && ctx.data.atlas;
      if (!atlas) { ready = false; return; }
      rng = window.EBRandom.RNG("insight");
      pts = atlas.points.map((pt, i) => {
        const sx = region.cx + pt.x * s;
        const sy = region.cy - pt.y * s;
        // seeded noise origin (dark-matter cloud before resolution)
        const a = rng.range(0, 6.2831), r = rng.range(0, Math.min(w, h) * 0.5);
        return { sx, sy, nsx: region.cx + Math.cos(a) * r, nsy: region.cy + Math.sin(a) * r,
          e: pt.e, br: pt.br, cs: pt.cs, ma: pt.ma || 0, mz: pt.mz || 0,
          col: ECOC[pt.e] || "#889", ph: rng.range(0, 6.2831) };
      });
      byEco = [[], [], [], []];
      pts.forEach((pt) => byEco[pt.e].push(pt));
      bridges = atlas.bridges.slice().sort((a, b) => b.w - a.w);
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

      const resolve = D.easeInOut(D.window01(t, 0.0, 0.30));   // noise -> real coords
      const clarify = D.window01(t, 0.22, 0.5);                 // colors/labels in
      const illum = D.window01(t, 0.48, 0.80);                  // bridge edges light up
      const diffuse = D.window01(t, 0.70, 1.0);                 // knowledge diffuses
      const tm = ctx.reduced ? 0.4 : p.frameCount * 0.02;

      // live screen pos for each point (resolution from noise)
      for (const pt of pts) {
        pt.x = D.lerp(pt.nsx, pt.sx, resolve);
        pt.y = D.lerp(pt.nsy, pt.sy, resolve);
      }

      // ---- atmospheric nebula behind the densest region (cloud emerging from black) ----
      const dc = p.drawingContext;
      if (resolve > 0.15 && ctx.data && ctx.data.atlas) {
        const m = ctx.data.atlas.meta.centroids.mangrove_msm;
        const mc = { x: region.cx + m[0] * region.s, y: region.cy - m[1] * region.s };
        const gx = dc.createRadialGradient(mc.x, mc.y, 0, mc.x, mc.y, region.s * 1.1);
        gx.addColorStop(0, D.rgba(EB.color.mangroveMsm, 0.13 * resolve));
        gx.addColorStop(0.5, D.rgba(EB.color.mangroveFutian, 0.05 * resolve));
        gx.addColorStop(1, "rgba(0,0,0,0)");
        dc.save(); dc.fillStyle = gx; dc.fillRect(0, 0, w, h); dc.restore();
      }

      // ---- bulk points: batched fillRect by ecosystem (fast; mz encoded as size) ----
      // 4 style changes instead of thousands of circle() calls -> 60fps with 5,209 pts.
      dc.save();
      for (let e = 0; e < 4; e++) {
        const grp = byEco[e]; if (!grp.length) continue;
        const poc = e < 2;
        dc.fillStyle = D.lerpHex("#586771", ECOC[e], clarify);
        dc.globalAlpha = poc ? 0.6 + 0.38 * clarify : 0.42 + 0.42 * clarify;
        for (let i = 0; i < grp.length; i++) {
          const pt = grp[i]; if (pt.br) continue;
          const s = (poc ? 2.4 : 1.4) + pt.mz * 1.8 * clarify;
          dc.fillRect(pt.x - s * 0.5, pt.y - s * 0.5, s, s);
        }
      }
      dc.globalAlpha = 1; dc.restore();

      // ---- bridge edges (illuminate strongest-first) ----
      const nEdge = Math.round(illum * bridges.length);
      if (nEdge > 0) {
        p.push(); p.blendMode(p.ADD); p.strokeWeight(1);
        for (let i = 0; i < nEdge; i++) {
          const b = bridges[i], a = pts[b.s], c = pts[b.t];
          if (!a || !c) continue;
          const al = (0.16 + (b.w - 0.97) * 7) * (0.4 + 0.6 * illum);
          p.stroke(D.rgba(EB.color.emergence, D.clamp(al, 0, 0.7)));
          p.line(a.x, a.y, c.x, c.y);
        }
        p.pop();
      }

      // ---- knowledge diffusion pulses along bridges ----
      if (diffuse > 0.02 && !ctx.reduced) {
        if (p.frameCount % 2 === 0 && pulses.length < 140 && nEdge > 0) {
          const b = bridges[rng.int(0, nEdge - 1)];
          pulses.push({ b, u: 0, sp: rng.range(0.012, 0.03) });
        }
        p.push(); p.blendMode(p.ADD); p.noStroke();
        for (let i = pulses.length - 1; i >= 0; i--) {
          const pu = pulses[i]; pu.u += pu.sp;
          if (pu.u >= 1) { pulses.splice(i, 1); continue; }
          // travel from rumen/source endpoint -> mangrove/target endpoint
          let a = pts[pu.b.s], c = pts[pu.b.t];
          if (a.e > c.e) { const tmp = a; a = c; c = tmp; } // lower eco code = more source-like
          const x = D.lerp(a.x, c.x, pu.u), y = D.lerp(a.y, c.y, pu.u);
          const col = D.lerpHex(EB.color.emergence, EB.color.methaneA, pu.u);
          D.glow(p, x, y, 2.2, col, 0.8 * (1 - pu.u));
        }
        p.pop();
      } else if (ctx.reduced && diffuse > 0.3) {
        // static representative pulses at edge midpoints
        p.push(); p.blendMode(p.ADD); p.noStroke();
        for (let i = 0; i < Math.min(nEdge, 80); i += 2) {
          const b = bridges[i]; let a = pts[b.s], c = pts[b.t]; if (a.e > c.e) { const tmp = a; a = c; c = tmp; }
          const x = D.lerp(a.x, c.x, 0.5), y = D.lerp(a.y, c.y, 0.5);
          D.glow(p, x, y, 2, D.lerpHex(EB.color.emergence, EB.color.methaneA, 0.5), 0.6);
        }
        p.pop();
      }

      // ---- bridge nodes (glow by attestation index) + case-study rings ----
      p.push(); p.blendMode(p.ADD); p.noStroke();
      for (const pt of pts) {
        if (!pt.br) continue;
        const inten = (0.35 + pt.ma * 0.8) * (0.4 + 0.6 * illum);
        D.glow(p, pt.x, pt.y, 2.0 + pt.ma * 1.6, pt.col, inten);
      }
      p.pop();
      if (clarify > 0.3) {
        p.noFill(); p.strokeWeight(1);
        for (const pt of pts) { if (pt.cs) { p.stroke(D.rgba(EB.color.attested, 0.5 * clarify)); p.circle(pt.x, pt.y, 11); } }
      }

      drawBasinLabels(clarify, resolve);
      drawLegend(w, h, clarify, nEdge, diffuse);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function basinScreen(key) {
      const atlas = ctx.data && ctx.data.atlas;
      const c = atlas && atlas.meta.centroids[key];
      if (!c) return { x: region.cx, y: region.cy };
      return { x: region.cx + c[0] * region.s, y: region.cy - c[1] * region.s };
    }
    function drawBasinLabels(clarify, resolve) {
      if (clarify < 0.25) return;
      const a = clarify * resolve;
      // rumen + wetland are adjacent on the left in PHATE; place their labels in the
      // gap toward the mangrove cloud, stacked apart. mangrove label to the right.
      const defs = [
        { key: "rumen", name: "RUMEN", sub: "source reference", dx: 16, dy: 30, ah: p.LEFT },
        { key: "wetland", name: "WETLAND / MUCC", sub: "target domain", dx: 16, dy: -30, ah: p.LEFT },
        { key: "mangrove_msm", name: "MANGROVE", sub: "MSM + Futian expansion", dx: region.s * 0.34, dy: 4, ah: p.LEFT },
      ];
      p.push();
      for (const d of defs) {
        const s = basinScreen(d.key);
        const col = ECOC[EB.ecosystems.find((e) => e.key === d.key).code];
        const lx = s.x + d.dx, ly = s.y + d.dy;
        // leader line from basin to label
        p.stroke(D.rgba(col, 0.35 * a)); p.strokeWeight(1); p.line(s.x, s.y, lx - 4, ly);
        p.noStroke(); p.textAlign(d.ah, p.CENTER);
        p.fill(D.rgba(col, 0.95 * a)); p.textFont("Space Grotesk"); p.textStyle(p.BOLD); p.textSize(13);
        p.text(d.name, lx, ly); p.textStyle(p.NORMAL);
        p.fill(D.rgba(EB.color.textMuted, 0.85 * a)); p.textFont("JetBrains Mono"); p.textSize(9);
        p.text(d.sub, lx, ly + 13);
      }
      p.pop();
    }

    function drawLegend(w, h, clarify, nEdge, diffuse) {
      const x = clamp16(w * 0.04), y = h * 0.18;
      p.push();
      D.label(p, "ESM2 (650M) PROTEOME MANIFOLD · PHATE PROJECTION", x, y, EB.color.textMuted, 10);
      D.label(p, "real coordinates · " + D.fmt(pts.length) + " units", x, y + 15, D.rgba(EB.color.attested, 0.9), 10);
      const ly = y + 36; let i = 0;
      for (const e of EB.ecosystems) {
        const yy = ly + i * 16;
        p.noStroke(); p.fill(D.rgba(e.color, 0.5 + 0.5 * clarify)); p.circle(x + 4, yy - 3, 6);
        D.label(p, e.label + "  ·  " + D.fmt(e.count), x + 14, yy, D.rgba(EB.color.textPrimary, 0.85), 10);
        i++;
      }
      if (nEdge > 0) {
        const yy = ly + i * 16 + 4;
        p.stroke(D.rgba(EB.color.emergence, 0.8)); p.strokeWeight(1.5); p.line(x, yy - 3, x + 9, yy - 3); p.noStroke();
        D.label(p, D.fmt(nEdge) + " / " + D.fmt(bridges.length) + " bridge edges" + (diffuse > 0.1 ? " · diffusing" : ""), x + 14, yy, D.rgba(EB.color.textPrimary, 0.85), 10);
      }
      p.pop();
    }
    function clamp16(v) { return Math.max(16, Math.min(v, ctx.W * 0.3)); }

    function drawNoData(w, h) {
      p.push(); p.fill(EB.color.textMuted); p.textAlign(p.CENTER, p.CENTER); p.textFont("JetBrains Mono"); p.textSize(13);
      p.text("Serve over http:// to load the live atlas (data/atlas.json)", w / 2, h / 2);
      p.pop();
    }
  };
})();
