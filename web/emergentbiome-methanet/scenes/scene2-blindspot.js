/* SCENE 2 - THE MEASUREMENT GAP. The field is illustrative. The zero refers only
   to verified wetland/mangrove genome-to-flux pairs in the August atlas, not to
   methane emissions or the wider literature. The requirement row names what a
   usable DNA-to-flux pair must record; it is not a sampling target. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.blindspot = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const LIT = EB.color.attested;
    const NEEDS = [["specimen ID", "specimen"], ["sample depth", "depth"], ["time window", "time"], ["chamber footprint", "footprint"]];
    let rng, field = [], row = null, headX = 0;

    // Reading-card box relative to this scene's stage, or null when hidden.
    function rel(el) {
      if (!el) return null;
      const base = ctx.holder.getBoundingClientRect(), r = el.getBoundingClientRect();
      if (!r.width || !r.height) return null;
      return { l: r.left - base.left, t: r.top - base.top, r: r.right - base.left, b: r.bottom - base.top };
    }

    // Place the requirement row below the headline number and clear of the card.
    function placeRow(w, h) {
      const copy = rel(ctx.section.querySelector(".copy"));
      const y = Math.max(h * 0.42, h * 0.15 + 176);
      let x0 = w * 0.16, x1 = w * 0.84;
      if (copy && y + 34 > copy.t && copy.l > w * 0.3) x1 = Math.min(x1, copy.l - 36);
      if (copy && y + 34 > copy.t && copy.l <= w * 0.3) return null; // stacked card covers the row
      return x1 - x0 < 260 ? null : { y, x0, x1, short: (x1 - x0) / NEEDS.length < 120 };
    }

    function layout() {
      rng = window.EBRandom.RNG("blindspot");
      const w = ctx.W, h = ctx.H;
      row = placeRow(w, h);
      // Keep the headline number clear of a tall reading card on the right.
      const copy = rel(ctx.section.querySelector(".copy"));
      headX = copy && copy.l > w * 0.35 && copy.t < h * 0.15 + 140 ? Math.max(170, copy.l / 2) : w * 0.5;
      const total = Math.round(D.clamp((w * h) / 620, 700, 2400));
      const blobs = [];
      for (let i = 0; i < 6; i++) blobs.push({ x: rng.range(w * 0.12, w * 0.88), y: rng.range(h * 0.2, h * 0.88), s: rng.range(h * 0.12, h * 0.26) });
      field = [];
      for (let i = 0; i < total; i++) {
        const b = rng.pick(blobs);
        field.push({ x: b.x + rng.gauss(0, b.s), y: b.y + rng.gauss(0, b.s), sz: rng.range(0.7, 1.9), tw: rng.range(0, 6.28) });
      }
    }

    p.setup = function () {
      p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout();
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { layout(); if (p.redraw) p.redraw(); });
      if (ctx.reduced) p.noLoop();
    };
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
        D.label(p, "searching for matched flux measurements…", sx - 8, h * 0.92 + 14, D.rgba(LIT, 0.55), 9, [p.RIGHT, p.TOP]);
      }

      // What a defensible pair must record. The scan probes each slot; all stay
      // empty because no exact molecule-to-flux join is admitted in this release.
      if (row) {
        const { y, x0, x1, short } = row, step = (x1 - x0) / NEEDS.length;
        D.label(p, "A USABLE DNA-TO-FLUX PAIR RECORDS", (x0 + x1) / 2, y - 30, EB.color.textMuted, 9.5, [p.CENTER, p.CENTER]);
        for (let i = 0; i < NEEDS.length; i++) {
          const cx = x0 + step * (i + 0.5), bw = Math.min(step - 14, 118), bh = 22;
          const probed = ctx.reduced ? true : cx < sx;
          p.push(); p.noFill(); p.drawingContext.setLineDash([3, 4]);
          p.stroke(probed ? D.rgba(LIT, 0.5) : D.rgba(EB.color.textMuted, 0.32)); p.strokeWeight(1);
          p.rect(cx - bw / 2, y - bh / 2, bw, bh, 4);
          p.drawingContext.setLineDash([]); p.pop();
          if (probed && bw >= 64) D.label(p, "not linked", cx, y, D.rgba(LIT, 0.55), 8.5, [p.CENTER, p.CENTER]);
          D.label(p, NEEDS[i][short ? 1 : 0], cx, y + bh / 2 + 9, D.rgba(EB.color.textPrimary, 0.72), short ? 8.5 : 9.5, [p.CENTER, p.TOP]);
        }
      }

      // Current-release linkage gap, explicitly scoped to this atlas.
      p.push();
      D.label(p, "AUGUST 2026 ATLAS · PUBLIC SOURCES", headX, h * 0.15, EB.color.textMuted, 11, [p.CENTER, p.CENTER]);
      p.fill(LIT); p.noStroke(); p.textFont("IBM Plex Mono"); p.textAlign(p.CENTER, p.CENTER);
      p.textSize(Math.min(72, w * 0.1, headX * 0.36));
      p.text("0 pairs", headX, h * 0.15 + 48);
      D.label(p, "wetland or mangrove genomes with a methane flux", headX, h * 0.15 + 88, D.rgba(EB.color.textPrimary, 0.7), 11, [p.CENTER, p.CENTER]);
      D.label(p, "measured on the same sample", headX, h * 0.15 + 105, D.rgba(EB.color.textPrimary, 0.7), 11, [p.CENTER, p.CENTER]);
      D.label(p, "grey dots are illustrative, not atlas records", headX, h * 0.15 + 126, D.rgba(EB.color.textMuted, 0.7), 9.5, [p.CENTER, p.CENTER]);
      p.pop();

      D.vignette(p, w, h, EB.color.bgBase, 0.6);
    };
  };
})();
