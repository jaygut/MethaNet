/* SCENE 4 - THE ATLAS  ·  REAL COUNTS
   Mangrove MAG particles (MSM cyan + Futian indigo) pour into the manifold while a
   mono counter ticks 625 -> 2,352 tri-view-ready units. A lane bar fills POC (locked)
   + MSM + Futian(in-progress). A three-plane motif (ESM2 / gLM2 / functional) slides
   together into one multi-view MAG. Every number is the verified live snapshot. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.atlas = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, N = EB.num;
    const C_MSM = EB.color.mangroveMsm, C_FUT = EB.color.mangroveFutian, C_POC = EB.color.emergence;
    let rng, stream = [], cloud = [];

    function layout() {
      rng = window.EBRandom.RNG("atlas");
      stream = []; cloud = [];
      // pre-seed the POC core already present in the manifold
      const w = ctx.W, h = ctx.H, cx = w * 0.5, cy = h * 0.56, s = Math.min(w, h) * 0.3;
      for (let i = 0; i < 160; i++) {
        const a = rng.range(0, 6.28), r = Math.sqrt(rng.next()) * s * 0.7;
        cloud.push({ x: cx + Math.cos(a) * r * 0.7, y: cy + Math.sin(a) * r, c: C_POC, a: 0.5 });
      }
    }
    function spawn(side) {
      const w = ctx.W, h = ctx.H, cx = w * 0.5, cy = h * 0.56, s = Math.min(w, h) * 0.3;
      const a = rng.range(0, 6.28), r = Math.sqrt(rng.next()) * s;
      stream.push({
        x: side < 0 ? rng.range(0, w * 0.3) : rng.range(w * 0.7, w),
        y: -rng.range(0, h * 0.4),
        tx: cx + Math.cos(a) * r, ty: cy + Math.sin(a) * r * 0.9,
        c: side < 0 ? C_MSM : C_FUT, u: 0, sp: rng.range(0.012, 0.03), sz: rng.range(1.2, 2.4),
      });
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.22, 110);

      // emission proportional to progress; cap cloud size for perf
      if (!ctx.reduced) { const rate = 0.4 + t * 2.4; let acc = (p._a || 0) + rate; while (acc >= 1) { spawn(rng.next() < 0.45 ? -1 : 1); acc -= 1; } p._a = acc; }
      else { const want = 160 + Math.round(t * 700); while (cloud.length < want) { const a = rng.range(0, 6.28), r = Math.sqrt(rng.next()) * Math.min(w, h) * 0.3; cloud.push({ x: w * 0.5 + Math.cos(a) * r * 0.7, y: h * 0.56 + Math.sin(a) * r, c: rng.next() < 0.5 ? C_MSM : C_FUT, a: 0.5 }); } }
      if (cloud.length > 1300) cloud.splice(0, cloud.length - 1300);

      // advance streams into the cloud
      p.push(); p.blendMode(p.ADD); p.noStroke();
      for (let i = stream.length - 1; i >= 0; i--) {
        const s = stream[i]; s.u += s.sp;
        const e = D.easeInOut(D.clamp(s.u));
        s.x = D.lerp(s.x, s.tx, 0.04 + e * 0.04); s.y = D.lerp(s.y, s.ty, 0.04 + e * 0.06);
        D.glow(p, s.x, s.y, s.sz, s.c, 0.6);
        if (s.u >= 1) { cloud.push({ x: s.tx, y: s.ty, c: s.c, a: 0.5 }); stream.splice(i, 1); }
      }
      p.pop();

      // accumulated cloud (additive bloom underlay + crisp core)
      p.push(); p.blendMode(p.ADD); p.noStroke();
      for (const c of cloud) { p.fill(D.rgba(c.c, 0.06 * (0.5 + 0.5 * t))); p.circle(c.x, c.y, 4.5); }
      p.pop();
      p.noStroke();
      for (const c of cloud) { p.fill(D.rgba(c.c, c.a * (0.55 + 0.5 * t))); p.circle(c.x, c.y, 2); }

      drawCounter(w, h, t);
      drawLaneBar(w, h, t);
      drawTriView(w, h, t);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawCounter(w, h, t) {
      const v = Math.round(D.lerp(N.calibrationCore, N.triViewReady, D.easeInOut(t)));
      p.push();
      p.textAlign(p.CENTER, p.CENTER);
      D.label(p, "TRI-VIEW-READY UNITS", w * 0.5, h * 0.15, EB.color.textMuted, 11, [p.CENTER, p.CENTER]);
      p.fill(EB.color.textPrimary); p.noStroke(); p.textFont("JetBrains Mono"); p.textSize(Math.min(72, w * 0.1));
      p.text(D.fmt(v), w * 0.5, h * 0.15 + 48);
      p.fill(D.rgba(EB.color.attested, 0.85)); p.textSize(11);
      p.text("625 calibration core  →  scaling toward " + D.fmt(N.embeddingBearingUnits) + " embedding-bearing units", w * 0.5, h * 0.15 + 86);
      p.pop();
    }

    function drawLaneBar(w, h, t) {
      const bw = Math.min(w * 0.6, 540), x = (w - bw) / 2, y = h * 0.7, bh = 14;
      const total = N.triViewReady;
      const segs = [
        ["POC core", N.calibrationCore, C_POC, false],
        ["MSM", N.msmFunctionalComplete, C_MSM, false],
        ["Futian archaea", N.futianArchaeaComplete, C_FUT, true],
      ];
      const grow = D.easeInOut(t);
      p.push(); p.noStroke();
      p.fill(D.rgba(EB.color.hairline, 0.9)); p.rect(x, y, bw, bh, 3);
      let cx = x;
      for (const [label, val, col, prog] of segs) {
        const segW = (val / total) * bw * grow;
        p.fill(prog ? D.rgba(col, 0.6) : col); p.rect(cx, y, segW, bh, 2);
        cx += segW;
      }
      // legend chips
      let lx = x;
      for (const [label, val, col, prog] of segs) {
        p.fill(col); p.circle(lx + 4, y - 12, 7);
        D.label(p, label + " " + D.fmt(val) + (prog ? " ⟳" : ""), lx + 13, y - 8, D.rgba(EB.color.textPrimary, 0.85), 10);
        lx += p.textWidth(label + " " + D.fmt(val)) + 60;
      }
      p.pop();
    }

    function drawTriView(w, h, t) {
      // three planes (ESM2 / gLM2 / functional) slide together into one multi-view MAG
      const conv = D.easeInOut(D.window01(t, 0.35, 0.9));
      const cx = w * 0.84, cy = h * 0.5, pw = Math.min(w * 0.14, 150), ph = 40;
      if (w < 760) return; // hide motif on narrow screens (copy + counter carry it)
      const views = [
        ["ESM2", EB.color.emergence, -1],
        ["gLM2", C_FUT, 0],
        ["functional", EB.color.methaneA, 1],
      ];
      p.push();
      D.label(p, "ONE MULTI-VIEW MAG", cx, cy - 64, EB.color.textMuted, 10, [p.CENTER, p.BOTTOM]);
      for (let i = 0; i < 3; i++) {
        const [name, col, off] = views[i];
        const spread = (1 - conv) * 46;
        const yy = cy + off * (14 + spread);
        p.push();
        p.noStroke(); p.fill(D.rgba(col, 0.12 + 0.06 * conv));
        p.stroke(D.rgba(col, 0.8)); p.strokeWeight(1);
        p.rectMode(p.CENTER); p.rect(cx, yy, pw, ph, 4);
        p.noStroke(); p.fill(D.rgba(col, 0.95)); p.textAlign(p.CENTER, p.CENTER); p.textFont("JetBrains Mono"); p.textSize(11);
        p.text(name, cx, yy);
        p.pop();
      }
      // bracket
      p.stroke(D.rgba(EB.color.attested, 0.5 + 0.4 * conv)); p.strokeWeight(1); p.noFill();
      p.line(cx + pw / 2 + 8, cy - 58, cx + pw / 2 + 16, cy - 58);
      p.line(cx + pw / 2 + 16, cy - 58, cx + pw / 2 + 16, cy + 58);
      p.line(cx + pw / 2 + 8, cy + 58, cx + pw / 2 + 16, cy + 58);
      p.pop();
    }
  };
})();
