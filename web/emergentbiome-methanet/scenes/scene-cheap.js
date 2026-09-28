/* SCENE — BEYOND SALINITY.
   A schematic for a field-design point: salinity sets the expected baseline,
   and genome evidence can flag the exceptions worth measuring. The scatter and
   highlighted cases are illustrative, not observations or risk predictions. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.cheap = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const M_A = EB.color.methaneA, M_B = EB.color.methaneB, EMG = EB.color.emergence, MUT = EB.color.textMuted;
    let rng, pts = [], exc = [], R = {};
    const METHODS = [
      { name: "single-gene test", sees: "one chosen marker gene", kind: "dot" },
      { name: "community census", sees: "which microbes are present", kind: "names" },
      { name: "salinity reading", sees: "one site condition", kind: "flat" },
      { name: "EmergentBiome atlas", sees: "whole-genome pathways, with sources", kind: "guilds", hi: true },
    ];

    // Stage-relative box of the copy card, or null when it is hidden.
    function copyBox() {
      const el = ctx.section && ctx.section.querySelector(".copy");
      if (!el || !ctx.holder) return null;
      const r = el.getBoundingClientRect(), base = ctx.holder.getBoundingClientRect();
      return r.height ? { l: r.left - base.left, t: r.top - base.top, r: r.right - base.left } : null;
    }

    function layout() {
      const w = ctx.W, h = ctx.H, narrow = w < 720;
      R.narrow = narrow;
      // scatter field: right side on desktop; on mobile, above the copy card
      R.px0 = narrow ? w * 0.12 : w * 0.46;
      R.px1 = narrow ? w * 0.94 : w * 0.93;
      R.py0 = narrow ? h * 0.13 : h * 0.20;
      R.py1 = narrow ? h * 0.62 : h * 0.66;
      const copy = copyBox();
      if (narrow && copy) R.py1 = Math.max(R.py0 + 90, Math.min(R.py1, copy.t - 36));
      if (!narrow && copy && copy.l < w * 0.2) R.px0 = Math.max(R.px0, copy.r + 40);
      // The method list sits above a left-hand card; drop it if the card reaches it.
      R.methodsFit = !narrow && !(copy && copy.l < w * 0.3 && copy.t < h * 0.16 + 4 * h * 0.062 + 12);
      rng = window.EBRandom.RNG("cheap");
      pts = [];
      for (let i = 0; i < 22; i++) {
        const sal = rng.range(0.05, 0.98);
        const base = 1 - sal;                       // salinity sets the baseline (down-right)
        const meth = D.clamp(base + rng.gauss(0, 0.08), 0.02, 1);
        pts.push({ sal, meth });
      }
      // synthetic cases used solely to show where paired measurement would help
      exc = [
        { sal: 0.82, meth: 0.72 }, { sal: 0.90, meth: 0.63 }, { sal: 0.74, meth: 0.80 },
      ];
    }

    p.setup = function () {
      p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout();
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { layout(); if (p.redraw) p.redraw(); });
      if (ctx.reduced) p.noLoop();
    };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    function X(s) { return D.lerp(R.px0, R.px1, s); }
    function Y(m) { return D.lerp(R.py1, R.py0, m); }   // methane up

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.22, 100);
      const tm = ctx.reduced ? 0.5 : p.frameCount * 0.02;

      const axesA = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.0, 0.2));
      const baseA = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.15, 0.45));
      const excA = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.5, 0.82));

      drawMethods(w, h, t);
      drawScatter(w, h, axesA, baseA, excA, tm);

      // teaching takeaway; the visible plot is explicitly illustrative
      if (!R.narrow) {
        D.label(p, "salinity sets the baseline; genome evidence flags where to measure", R.px0, R.py1 + h * 0.075, D.rgba(EB.color.textPrimary, 0.9 * axesA), 10.5);
        D.label(p, "schematic · not site data, no flux or risk inferred", R.px0, R.py1 + h * 0.105, D.rgba(MUT, 0.85 * axesA), 9);
      }
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawScatter(w, h, axesA, baseA, excA, tm) {
      // axes
      p.push();
      p.stroke(D.rgba(MUT, 0.5 * axesA)); p.strokeWeight(1);
      p.line(R.px0, R.py0, R.px0, R.py1); p.line(R.px0, R.py1, R.px1, R.py1);
      p.pop();
      D.label(p, "METHANE (ILLUSTRATIVE)", R.px0 - 4, R.py0 - 12, D.rgba(MUT, 0.8 * axesA), 8.5);
      D.label(p, "SALINITY →", R.px1, R.py1 + 14, D.rgba(MUT, 0.8 * axesA), 8.5, [p.RIGHT, p.TOP]);
      // stylized baseline trend for the teaching plot
      if (baseA > 0.01) {
        p.push(); p.drawingContext.setLineDash([5, 5]);
        p.stroke(D.rgba(MUT, 0.55 * baseA)); p.strokeWeight(1.4);
        p.line(X(0.02), Y(0.98), X(0.98), Y(0.06));
        p.drawingContext.setLineDash([]); p.pop();
        D.label(p, "expected from salinity", X(0.5) + 6, Y(0.5) - 8, D.rgba(MUT, 0.7 * baseA), 8.5);
      }
      // on-trend points (what salinity gets right)
      p.push(); p.noStroke();
      for (let i = 0; i < pts.length; i++) {
        const q = pts[i]; const a = ctx.reduced ? 1 : D.easeInOut(D.window01(ctx.progress, 0.15 + (q.sal * 0.2), 0.5 + q.sal * 0.2));
        p.fill(D.rgba(MUT, 0.5 * a)); p.circle(X(q.sal), Y(q.meth), 4.5);
      }
      p.pop();
      // stylized cases to nominate for field measurements
      if (excA > 0.01) {
        for (const e of exc) {
          const x = X(e.sal), y = Y(e.meth);
          p.push(); p.blendMode(p.ADD);
          D.glow(p, x, y, 3, M_B, (ctx.reduced ? 0.9 : 0.6 + 0.4 * Math.sin(tm * 2 + e.sal * 6)) * excA);
          p.pop();
          p.push(); p.noFill(); p.stroke(D.rgba(M_A, 0.85 * excA)); p.strokeWeight(1.2); p.circle(x, y, 15); p.pop();
        }
        // annotations desktop-only (the small mobile plot cannot hold them without overlap)
        if (!R.narrow) {
          D.label(p, "salty, yet methane-active (hypothetical)", X(exc[2].sal) - 12, Y(exc[2].meth) - 14, D.rgba(M_A, excA), 9, [p.RIGHT, p.BOTTOM]);
          D.label(p, "measure here first", X(exc[1].sal), Y(exc[1].meth) + 16, D.rgba(M_A, 0.9 * excA), 8.5, [p.CENTER, p.TOP]);
        }
      }
    }

    function drawMethods(w, h, t) {
      if (R.narrow || !R.methodsFit) return;   // no room: the copy card carries the point
      const x = w * 0.06, y0 = h * 0.16, rh = h * 0.062;
      D.label(p, "WHAT EACH METHOD SEES", x, y0 - h * 0.03, D.rgba(MUT, 0.95), 10);
      for (let i = 0; i < METHODS.length; i++) {
        const m = METHODS[i], y = y0 + i * rh;
        const a = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.05 + i * 0.08, 0.3 + i * 0.08));
        if (a < 0.01) continue;
        const col = m.hi ? EMG : MUT;
        // resolution glyph
        p.push(); p.noStroke();
        if (m.kind === "dot") { p.fill(D.rgba(col, 0.7 * a)); p.circle(x + 6, y, 6); }
        else if (m.kind === "names") { p.fill(D.rgba(col, 0.6 * a)); for (let k = 0; k < 3; k++) p.rect(x, y - 5 + k * 5, 14, 2.4, 1); }
        else if (m.kind === "flat") { p.stroke(D.rgba(col, 0.7 * a)); p.strokeWeight(2); p.line(x, y, x + 16, y); p.noStroke(); }
        else { const cs = [M_A, EB.color.attested, EMG]; for (let k = 0; k < 3; k++) { p.fill(D.rgba(cs[k], 0.85 * a)); p.rect(x + k * 6, y - 6 + (2 - k) * 2, 4, 12 - (2 - k) * 2, 1); } }
        // label
        p.fill(D.rgba(m.hi ? EB.color.textPrimary : MUT, (m.hi ? 1 : 0.85) * a)); p.textFont("Bricolage Grotesque"); p.textStyle(m.hi ? p.BOLD : p.NORMAL); p.textSize(12); p.textAlign(p.LEFT, p.BOTTOM);
        p.text(m.name, x + 26, y + 1); p.textStyle(p.NORMAL);
        p.fill(D.rgba(m.hi ? EMG : MUT, 0.8 * a)); p.textFont("IBM Plex Mono"); p.textSize(8.5); p.textAlign(p.LEFT, p.TOP);
        p.text("sees: " + m.sees, x + 26, y + 3);
        p.pop();
      }
    }
  };
})();
