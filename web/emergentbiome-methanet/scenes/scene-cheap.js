/* SCENE - VERSUS THE CHEAP METHOD  ·  why molecular beats the cheap baselines.
   The thesis, made literal: salinity sets the baseline (methane falls as salinity rises,
   the first-order control the registries encode), but the microbial community sets the
   exception. A few sites run high on methane even at high salinity, because methylotrophic
   methanogens use substrates sulfate-reducers ignore. A salinity number alone misclassifies
   those; a community readout catches them. Left: what each cheap method can see. Right: the
   salinity-vs-methane field, with the exceptions the community flags. The molecular signal
   is real (methanogen marker mcrA tracks measured flux at r > 0.7 in field sediments); the
   scatter here is an illustrative teaching plot, and the product ranks, it does not report flux. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.cheap = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const M_A = EB.color.methaneA, M_B = EB.color.methaneB, EMG = EB.color.emergence, MUT = EB.color.textMuted;
    let rng, pts = [], exc = [], R = {};
    const METHODS = [
      { name: "single marker test", sees: "one gene", kind: "dot" },
      { name: "metabarcoding", sees: "who is present", kind: "names" },
      { name: "the salinity rule", sees: "a site average", kind: "flat" },
      { name: "MethaNet", sees: "the whole community, resolved", kind: "guilds", hi: true },
    ];

    function layout() {
      const w = ctx.W, h = ctx.H, narrow = w < 720;
      R.narrow = narrow;
      // scatter field (right side on desktop, full-width lower on mobile)
      R.px0 = narrow ? w * 0.12 : w * 0.46;
      R.px1 = narrow ? w * 0.94 : w * 0.93;
      R.py0 = narrow ? h * 0.30 : h * 0.20;
      R.py1 = narrow ? h * 0.62 : h * 0.66;
      rng = window.EBRandom.RNG("cheap");
      pts = [];
      for (let i = 0; i < 22; i++) {
        const sal = rng.range(0.05, 0.98);
        const base = 1 - sal;                       // salinity sets the baseline (down-right)
        const meth = D.clamp(base + rng.gauss(0, 0.08), 0.02, 1);
        pts.push({ sal, meth });
      }
      // the exceptions: high methane at high salinity (community catches what salinity misses)
      exc = [
        { sal: 0.82, meth: 0.72 }, { sal: 0.90, meth: 0.63 }, { sal: 0.74, meth: 0.80 },
      ];
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
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

      // thesis + the real molecular anchor (desktop; on mobile the copy card + readout carry it)
      if (!R.narrow) {
        D.label(p, "salinity sets the baseline. the community sets the exception.", R.px0, R.py1 + h * 0.075, D.rgba(EB.color.textPrimary, 0.9 * axesA), 10.5);
        D.label(p, "methanogen marker vs measured flux: r > " + EB.ext.mcraFluxSpearman + " in field sediments", R.px0, R.py1 + h * 0.105, D.rgba(MUT, 0.85 * axesA), 9);
      }
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawScatter(w, h, axesA, baseA, excA, tm) {
      // axes
      p.push();
      p.stroke(D.rgba(MUT, 0.5 * axesA)); p.strokeWeight(1);
      p.line(R.px0, R.py0, R.px0, R.py1); p.line(R.px0, R.py1, R.px1, R.py1);
      p.pop();
      D.label(p, "METHANE RISK", R.px0 - 4, R.py0 - 12, D.rgba(MUT, 0.8 * axesA), 8.5);
      D.label(p, "SALINITY →", R.px1, R.py1 + 14, D.rgba(MUT, 0.8 * axesA), 8.5, [p.RIGHT, p.TOP]);
      // baseline trend (salinity -> methane), the cheap proxy's whole worldview
      if (baseA > 0.01) {
        p.push(); p.drawingContext.setLineDash([5, 5]);
        p.stroke(D.rgba(MUT, 0.55 * baseA)); p.strokeWeight(1.4);
        p.line(X(0.02), Y(0.98), X(0.98), Y(0.06));
        p.drawingContext.setLineDash([]); p.pop();
        D.label(p, "salinity baseline", X(0.5) + 6, Y(0.5) - 8, D.rgba(MUT, 0.7 * baseA), 8.5);
      }
      // on-trend points (what salinity gets right)
      p.push(); p.noStroke();
      for (let i = 0; i < pts.length; i++) {
        const q = pts[i]; const a = ctx.reduced ? 1 : D.easeInOut(D.window01(ctx.progress, 0.15 + (q.sal * 0.2), 0.5 + q.sal * 0.2));
        p.fill(D.rgba(MUT, 0.5 * a)); p.circle(X(q.sal), Y(q.meth), 4.5);
      }
      p.pop();
      // the exceptions: high methane at high salinity, flagged by the community
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
          D.label(p, "high methane at high salinity", X(exc[2].sal) - 12, Y(exc[2].meth) - 14, D.rgba(M_A, excA), 9, [p.RIGHT, p.BOTTOM]);
          D.label(p, "the community flags what salinity misses", X(exc[2].sal) - 12, Y(exc[1].meth) + 20, D.rgba(M_A, 0.9 * excA), 8.5, [p.RIGHT, p.TOP]);
        }
      }
    }

    function drawMethods(w, h, t) {
      if (R.narrow) return;   // mobile: the copy card + scatter carry it; skip the legend to stay uncluttered
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
