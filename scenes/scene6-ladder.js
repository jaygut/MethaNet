/* SCENE 6 - THE HONEST LADDER  ·  REAL ROADMAP
   The MRV maturity ladder (roadmap Levels 0-5, + Level 6 horizon). An ascending
   staircase: rung 0 (molecular screening) is LIT NOW; rungs 1-4 are the climb;
   rung 5 (calibrated probabilistic MRV / A-E tiers) is the dimmed TARGET. Each rung
   reveals exactly what unlocks it. "YOU ARE HERE" never lies: only rung 0 is lit. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.ladder = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, L = EB.ladder;
    let rungs = [];

    function layout() {
      const w = ctx.W, h = ctx.H;
      const x0 = w * 0.16, x1 = w * 0.80, y0 = h * 0.82, y1 = h * 0.20;
      rungs = L.map((r, i) => ({ ...r, x: D.lerp(x0, x1, i / (L.length - 1)), y: D.lerp(y0, y1, i / (L.length - 1)) }));
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.2, 110);
      const tm = ctx.reduced ? 0 : p.frameCount * 0.02;

      // heading
      D.label(p, "MRV MATURITY LADDER", w * 0.08, h * 0.12, EB.color.textMuted, 11);
      p.push(); p.fill(D.rgba(EB.color.textPrimary, 0.9)); p.textFont("JetBrains Mono"); p.textSize(10);
      p.text("LIT NOW: rung 0 · molecular screening      TARGET: rung 5 · calibrated A–E risk (not yet calibrated)", w * 0.08, h * 0.12 + 16);
      p.pop();

      const reveal = D.easeInOut(t);
      const nShown = ctx.reduced ? rungs.length : D.clamp(reveal * (rungs.length + 0.4)) * rungs.length / rungs.length;
      const shown = ctx.reduced ? rungs.length : Math.ceil(reveal * rungs.length + 0.001);

      // connecting climb path (dashed ahead of rung 0)
      p.push();
      for (let i = 1; i < rungs.length; i++) {
        const a = rungs[i - 1], b = rungs[i];
        const seg = D.clamp((reveal * rungs.length) - (i - 1));
        if (seg <= 0) continue;
        const bx = D.lerp(a.x, b.x, seg), by = D.lerp(a.y, b.y, seg);
        if (i === 1) { p.stroke(D.rgba(EB.color.emergence, 0.55)); p.strokeWeight(1.5); }
        else { p.drawingContext.setLineDash([5, 5]); p.stroke(D.rgba(EB.color.textMuted, 0.35)); p.strokeWeight(1); }
        p.line(a.x, a.y, bx, by);
        p.drawingContext.setLineDash([]);
      }
      p.pop();

      for (let i = 0; i < rungs.length; i++) {
        if (i >= shown && !ctx.reduced) continue;
        const r = rungs[i];
        const a = ctx.reduced ? 1 : D.clamp((reveal * rungs.length) - i + 0.4);
        drawRung(r, i, a, tm);
      }

      // "YOU ARE HERE" marker pinned at rung 0
      const r0 = rungs[0];
      p.push(); p.blendMode(p.ADD); p.noStroke();
      D.glow(p, r0.x, r0.y, 6, EB.color.emergence, ctx.reduced ? 1 : 0.7 + 0.3 * Math.sin(tm * 2));
      p.pop();

      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawRung(r, i, a, tm) {
      if (a <= 0) return;
      const lit = r.state === "lit", target = r.state === "target";
      const col = lit ? EB.color.emergence : target ? EB.color.methaneA : EB.color.textMuted;
      // platform tick
      p.push();
      p.stroke(D.rgba(col, 0.5 * a)); p.strokeWeight(1);
      p.line(r.x - 16, r.y, r.x + 16, r.y);
      // node
      p.noStroke();
      if (lit) { p.push(); p.blendMode(p.ADD); D.glow(p, r.x, r.y, 4, col, a); p.pop(); }
      p.fill(lit ? col : D.rgba(EB.color.bgBase, a)); p.stroke(D.rgba(col, (lit ? 1 : 0.7) * a)); p.strokeWeight(target ? 1.6 : 1.2);
      if (target) p.drawingContext.setLineDash([4, 3]);
      p.circle(r.x, r.y, target ? 18 : 13);
      p.drawingContext.setLineDash([]);
      // rung number inside
      p.noStroke(); p.fill(lit ? D.rgba("#06090D", a) : D.rgba(col, a)); p.textAlign(p.CENTER, p.CENTER); p.textFont("JetBrains Mono"); p.textSize(8.5);
      p.text(r.rung, r.x, r.y + 0.5);

      // label block (alternate above/below to avoid overlap)
      const above = i % 2 === 1;
      const ly = above ? r.y - 26 : r.y + 22;
      p.textAlign(p.LEFT, above ? p.BOTTOM : p.TOP);
      const lx = r.x + 22;
      // state chip
      const chip = lit ? "LIT · NOW" : target ? "TARGET" : i === 1 ? "NEXT" : "ROADMAP";
      p.fill(D.rgba(col, a)); p.textFont("JetBrains Mono"); p.textSize(8);
      p.text(chip, lx, above ? ly - 30 : ly);
      p.fill(D.rgba(EB.color.textPrimary, a)); p.textFont("Space Grotesk"); p.textStyle(p.BOLD); p.textSize(13);
      p.text(r.title, lx, above ? ly - 16 : ly + 12); p.textStyle(p.NORMAL);
      p.fill(D.rgba(EB.color.textMuted, 0.9 * a)); p.textFont("JetBrains Mono"); p.textSize(8.6); p.textAlign(p.LEFT, p.TOP);
      wrapTextLeft(r.unlock, lx, above ? ly - 2 : ly + 28, Math.min(ctx.W * 0.27, 280), 11);
      p.pop();
    }

    function wrapTextLeft(txt, x, y, maxw, lh) {
      const words = txt.split(" "); let line = "", yy = y;
      for (const wd of words) { const test = line + wd + " "; if (p.textWidth(test) > maxw && line) { p.text(line.trim(), x, yy); line = wd + " "; yy += lh; } else line = test; }
      if (line) p.text(line.trim(), x, yy);
    }
  };
})();
