/* SCENE 2 - THE BLIND SPOT (illustrative scale; the honest current state is ~0).
   A vast field of grey "unknown genome" nodes: the sediment microbiome as climate
   dark matter. A measurement scan sweeps the field and finds essentially nothing -
   today, almost no site carries paired methane-flux and molecular data. The field
   size is illustrative and labeled; the "~0 measured" is the real present state. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.blindspot = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const LIT = EB.color.attested;
    const TGT_LO = EB.num.pairedFluxTargetLo, TGT_HI = EB.num.pairedFluxTargetHi, SLOTS = TGT_HI; // 80, 100
    let rng, field = [];

    function layout() {
      rng = window.EBRandom.RNG("blindspot");
      const w = ctx.W, h = ctx.H;
      const total = Math.round(D.clamp((w * h) / 620, 700, 2400));
      const blobs = [];
      for (let i = 0; i < 6; i++) blobs.push({ x: rng.range(w * 0.12, w * 0.88), y: rng.range(h * 0.2, h * 0.88), s: rng.range(h * 0.12, h * 0.26) });
      field = [];
      for (let i = 0; i < total; i++) {
        const b = rng.pick(blobs);
        field.push({ x: b.x + rng.gauss(0, b.s), y: b.y + rng.gauss(0, b.s), sz: rng.range(0.7, 1.9), tw: rng.range(0, 6.28) });
      }
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.3, 100);
      const tm = ctx.reduced ? 0 : p.frameCount * 0.02;
      const sweep = ctx.reduced ? 1 : D.easeInOut(D.clamp(t / 0.92));
      const sx = sweep * w;

      // grey unknown field; genomes already passed by the scan dim slightly (probed, nothing found)
      const dc = p.drawingContext;
      dc.save();
      for (let i = 0; i < field.length; i++) {
        const f = field[i];
        const probed = f.x < sx;
        dc.fillStyle = probed ? "#5C6975" : "#7C8B96";
        dc.globalAlpha = (probed ? 0.18 : 0.28) + (ctx.reduced ? 0 : 0.04 * Math.sin(tm + f.tw));
        dc.fillRect(f.x - f.sz * 0.5, f.y - f.sz * 0.5, f.sz, f.sz);
      }
      dc.globalAlpha = 1; dc.restore();

      // sweeping measurement scan - it finds nothing to light up
      if (!ctx.reduced && t > 0.02 && t < 0.99) {
        p.push(); p.blendMode(p.ADD);
        p.stroke(D.rgba(LIT, 0.22)); p.strokeWeight(1.5); p.line(sx, h * 0.1, sx, h * 0.92);
        p.noStroke(); p.fill(D.rgba(LIT, 0.04)); p.rect(sx - 30, 0, 30, h);
        p.pop();
        D.label(p, "scanning for paired flux data…", sx - 8, h * 0.92 + 14, D.rgba(LIT, 0.55), 9, [p.RIGHT, p.TOP]);
      }

      // validation-coverage row: the 80-100 target band, 0 of them lit (measured) today.
      // The scan rakes across it and lights nothing - the same fact as "≈ 0", made spatial.
      {
        const rowY = h * 0.46, x0 = w * 0.18, x1 = w * 0.82;
        const dc2 = p.drawingContext; dc2.save();
        for (let i = 0; i < SLOTS; i++) {
          const sxSlot = p.lerp(x0, x1, i / (SLOTS - 1));
          const reached = ctx.reduced ? true : sxSlot < sx;          // probed by the scan, still empty
          if (reached) { dc2.globalAlpha = 0.30; dc2.fillStyle = D.rgba(LIT, 0.16); }
          else { dc2.globalAlpha = 0.22; dc2.fillStyle = "#3A4650"; }
          dc2.fillRect(sxSlot - 1, rowY - 7, 2, 14);
        }
        dc2.globalAlpha = 1; dc2.restore();
        p.push(); p.stroke(D.rgba(EB.color.textMuted, 0.28)); p.strokeWeight(1);
        p.line(x0, rowY + 9, x1, rowY + 9);
        const x80 = p.lerp(x0, x1, (TGT_LO - 1) / (SLOTS - 1));
        p.line(x80, rowY + 6, x80, rowY + 12); p.line(x1, rowY + 6, x1, rowY + 12);
        p.pop();
        D.label(p, "validation target  " + TGT_LO + " to " + TGT_HI + " sites", (x0 + x1) / 2, rowY + 22, EB.color.textMuted, 9.5, [p.CENTER, p.TOP]);
        D.label(p, "0 lit", x0 - 12, rowY, D.rgba(LIT, 0.7), 11, [p.RIGHT, p.CENTER]);
      }

      // counter readout - exact current-atlas join state, not a global site count
      p.push();
      D.label(p, "PAIRED METHANE-FLUX MEASUREMENTS", w * 0.5, h * 0.15, EB.color.textMuted, 11, [p.CENTER, p.CENTER]);
      p.fill(LIT); p.noStroke(); p.textFont("IBM Plex Mono"); p.textAlign(p.CENTER, p.CENTER);
      p.textSize(Math.min(72, w * 0.1));
      p.text("0 exact", w * 0.5, h * 0.15 + 48);
      D.label(p, "authoritative sample + environment + process joins in this atlas", w * 0.5, h * 0.15 + 88, D.rgba(EB.color.textPrimary, 0.7), 11, [p.CENTER, p.CENTER]);
      D.label(p, "grey: ~" + D.fmt(field.length) + " unknown genomes in this view (illustrative scale)", w * 0.5, h * 0.15 + 106, D.rgba(EB.color.textMuted, 0.7), 9.5, [p.CENTER, p.CENTER]);
      p.pop();

      D.vignette(p, w, h, EB.color.bgBase, 0.6);
    };
  };
})();
