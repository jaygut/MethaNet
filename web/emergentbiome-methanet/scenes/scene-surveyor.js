/* SCENE — EVIDENCE CARD
   The semantic card in index.html carries a frozen, source-backed MUCC v1
   candidate. This canvas is visual wayfinding only: molecular evidence is
   recorded, while the exact sample-to-process joins needed for field inference
   remain open. No invented site scores, confidence intervals, or flux values. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.surveyor = function (p, ctx) {
    const EB = window.EB;
    const D = window.EBDraw;
    const stages = [
      { title: "SOURCE", note: "archive + QC", color: EB.color.wetland },
      { title: "MOLECULAR", note: "markers + context", color: EB.color.attested },
      { title: "FIELD", note: "exact paired join", color: EB.color.methaneA },
    ];
    let dots = [];
    let anchors = [];
    let narrow = false;

    function layout() {
      const w = ctx.W, h = ctx.H;
      narrow = w < 720;
      const rng = window.EBRandom.RNG("evidence-card");
      anchors = stages.map((s, i) => ({
        ...s,
        x: narrow ? w * 0.18 : w * (0.17 + i * 0.27),
        y: narrow ? h * (0.23 + i * 0.16) : h * (0.37 + (i % 2) * 0.08),
      }));
      dots = [];
      const total = narrow ? 54 : 96;
      for (let i = 0; i < total; i++) {
        const lane = i % 3;
        const a = anchors[lane];
        dots.push({
          lane,
          x: a.x + rng.gauss(0, narrow ? w * 0.10 : w * 0.065),
          y: a.y + rng.gauss(0, h * 0.07),
          r: rng.range(0.7, 2.1),
          phase: rng.range(0, Math.PI * 2),
        });
      }
    }

    p.setup = function () {
      p.createCanvas(ctx.W, ctx.H);
      p.pixelDensity(Math.min(2, window.devicePixelRatio || 1));
      layout();
      if (ctx.reduced) p.noLoop();
    };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H;
      const t = ctx.reduced ? 1 : ctx.progress;
      const tm = ctx.reduced ? 0 : p.frameCount * 0.018;
      p.clear();
      p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.16, 110);

      p.push();
      p.noStroke();
      for (const dot of dots) {
        const appear = D.easeInOut(D.window01(t, 0.02 + dot.lane * 0.12, 0.38 + dot.lane * 0.12));
        if (appear < 0.01) continue;
        const col = stages[dot.lane].color;
        const drift = ctx.reduced ? 0 : Math.sin(tm + dot.phase) * 2.2;
        p.fill(D.rgba(col, (dot.lane === 2 ? 0.25 : 0.44) * appear));
        p.circle(dot.x + drift, dot.y - drift * 0.4, dot.r * 2);
      }
      p.pop();

      for (let i = 1; i < anchors.length; i++) {
        const from = anchors[i - 1], to = anchors[i];
        const appear = D.easeInOut(D.window01(t, 0.22 + i * 0.15, 0.58 + i * 0.15));
        if (appear < 0.01) continue;
        p.push();
        p.stroke(D.rgba(i === 2 ? EB.color.methaneA : EB.color.attested, (i === 2 ? 0.55 : 0.75) * appear));
        p.strokeWeight(1.6);
        if (i === 2) p.drawingContext.setLineDash([5, 6]);
        p.line(from.x, from.y, D.lerp(from.x, to.x, appear), D.lerp(from.y, to.y, appear));
        p.drawingContext.setLineDash([]);
        p.pop();
      }

      anchors.forEach((a, i) => {
        const appear = D.easeInOut(D.window01(t, 0.12 + i * 0.16, 0.46 + i * 0.16));
        if (appear < 0.01) return;
        const pulse = ctx.reduced ? 1 : 0.80 + Math.sin(tm * 1.5 + i) * 0.2;
        p.push();
        p.stroke(D.rgba(a.color, 0.92 * appear));
        p.strokeWeight(1.4);
        p.fill(i === 2 ? D.rgba(EB.color.bgBase, appear) : D.rgba(a.color, 0.16 * appear));
        p.circle(a.x, a.y, i === 2 ? 26 : 22 + pulse * 2);
        p.noStroke();
        p.fill(D.rgba(a.color, appear));
        p.textFont("IBM Plex Mono");
        p.textSize(narrow ? 9 : 10);
        p.textAlign(p.LEFT, p.BOTTOM);
        p.text(a.title + (i === 2 ? " · PENDING" : " · RECORDED"), a.x + 20, a.y - 3);
        p.fill(D.rgba(EB.color.textMuted, 0.86 * appear));
        p.textSize(narrow ? 8 : 9);
        p.textAlign(p.LEFT, p.TOP);
        p.text(a.note, a.x + 20, a.y + 4);
        p.pop();
      });

      D.vignette(p, w, h, EB.color.bgBase, 0.48);
    };
  };
})();
