/* SCENE 2 - THE BLIND SPOT (real number: 23 paired-flux samples).
   A vast field of grey "unknown genome" nodes - the sediment microbiome as dark
   matter. Exactly 23 nodes light up (paired CH4-flux data) as a measurement reticle
   sweeps the field, dramatizing how little of the risk surface is actually measured. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.blindspot = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const N_MEAS = EB.num.pairedFluxNow; // 23
    const LIT = EB.color.attested;
    let rng, field = [], meas = [];

    function layout() {
      rng = window.EBRandom.RNG("blindspot");
      const w = ctx.W, h = ctx.H;
      const total = Math.round(D.clamp((w * h) / 620, 700, 2400));
      // soft multi-cluster scatter (a few unknown-community blobs)
      const blobs = [];
      const bn = 6;
      for (let i = 0; i < bn; i++) blobs.push({ x: rng.range(w * 0.12, w * 0.88), y: rng.range(h * 0.18, h * 0.86), s: rng.range(h * 0.12, h * 0.26) });
      field = [];
      for (let i = 0; i < total; i++) {
        const b = rng.pick(blobs);
        field.push({ x: b.x + rng.gauss(0, b.s), y: b.y + rng.gauss(0, b.s),
          sz: rng.range(0.7, 1.9), tw: rng.range(0, 6.28) });
      }
      // 23 measured nodes scattered across (rare, spread out)
      meas = [];
      for (let i = 0; i < N_MEAS; i++) {
        const b = rng.pick(blobs);
        meas.push({ x: b.x + rng.gauss(0, b.s * 1.1), y: b.y + rng.gauss(0, b.s * 1.1), ph: rng.range(0, 6.28) });
      }
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.3, 100);

      const tm = ctx.reduced ? 0 : p.frameCount * 0.02;
      // grey unknown field - batched fillRect (single style; gentle global shimmer)
      const dc = p.drawingContext;
      dc.save(); dc.fillStyle = "#7C8B96"; dc.globalAlpha = 0.26 + (ctx.reduced ? 0 : 0.05 * Math.sin(tm));
      for (let i = 0; i < field.length; i++) { const f = field[i]; dc.fillRect(f.x - f.sz * 0.5, f.y - f.sz * 0.5, f.sz, f.sz); }
      dc.globalAlpha = 1; dc.restore();

      // how many measured nodes are lit (scroll reveals them one by one)
      const nLit = ctx.reduced ? N_MEAS : Math.round(D.easeInOut(t) * N_MEAS);

      // sweeping measurement reticle line
      if (!ctx.reduced && t > 0.02 && t < 0.99) {
        const sx = (D.easeInOut(t)) * w;
        p.stroke(D.rgba(LIT, 0.18)); p.strokeWeight(1); p.line(sx, 0, sx, h);
        p.noStroke(); p.fill(D.rgba(LIT, 0.05)); p.rect(sx - 26, 0, 26, h);
      }

      // measured nodes
      p.push(); p.blendMode(p.ADD);
      for (let i = 0; i < meas.length; i++) {
        const m = meas[i];
        if (i < nLit) {
          const pulse = 0.7 + 0.3 * Math.sin(tm * 2 + m.ph);
          D.glow(p, m.x, m.y, 3.2, LIT, pulse);
        }
      }
      p.pop();
      // reticle rings on lit nodes
      p.noFill(); p.strokeWeight(1);
      for (let i = 0; i < nLit; i++) {
        const m = meas[i];
        p.stroke(D.rgba(LIT, 0.5)); p.circle(m.x, m.y, 14);
        p.stroke(D.rgba(LIT, 0.25)); p.circle(m.x, m.y, 22);
      }

      // counter readout
      p.push();
      D.label(p, "MEASURED · PAIRED CH₄-FLUX", w * 0.5, h * 0.16, EB.color.textMuted, 11, [p.CENTER, p.CENTER]);
      p.fill(LIT); p.noStroke(); p.textFont("JetBrains Mono"); p.textAlign(p.CENTER, p.CENTER);
      p.textSize(Math.min(64, w * 0.09));
      p.text(nLit + " / " + D.fmt(field.length), w * 0.5, h * 0.16 + 44);
      D.label(p, "unknown genome units in this view (illustrative scale)", w * 0.5, h * 0.16 + 80, D.rgba(EB.color.textMuted, 0.8), 10.5, [p.CENTER, p.CENTER]);
      p.pop();

      D.vignette(p, w, h, EB.color.bgBase, 0.6);
    };
  };
})();
