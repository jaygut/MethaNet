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

      if (w < 720) {
        drawMobileLadder(w, h, tm);
        D.vignette(p, w, h, EB.color.bgBase, 0.5);
        return;
      }

      // heading
      D.label(p, "METHANE EVIDENCE MATURITY", w * 0.08, h * 0.12, EB.color.textMuted, 11);
      p.push(); p.fill(D.rgba(EB.color.textPrimary, 0.9)); p.textFont("IBM Plex Mono"); p.textSize(10);
      p.text("LIT NOW: rung 0 · molecular screening      TARGET: rung 5 · calibrated A–E risk (not yet calibrated)", w * 0.08, h * 0.12 + 16);
      p.pop();

      const reveal = D.easeInOut(t);
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
      const lit = r.state === "lit", target = r.state === "target", prog = r.state === "progress";
      const col = (lit || prog) ? EB.color.emergence : target ? EB.color.methaneA : EB.color.textMuted;
      // platform tick
      p.push();
      p.stroke(D.rgba(col, 0.5 * a)); p.strokeWeight(1);
      p.line(r.x - 16, r.y, r.x + 16, r.y);
      // node
      p.noStroke();
      if (lit) { p.push(); p.blendMode(p.ADD); D.glow(p, r.x, r.y, 4, col, a); p.pop(); }
      else if (prog) { p.push(); p.blendMode(p.ADD); D.glow(p, r.x, r.y, 3, col, 0.45 * a); p.pop(); }
      p.fill(lit ? col : prog ? D.rgba(col, 0.35 * a) : D.rgba(EB.color.bgBase, a));
      p.stroke(D.rgba(col, (lit ? 1 : prog ? 0.9 : 0.7) * a)); p.strokeWeight(target ? 1.6 : 1.2);
      if (target) p.drawingContext.setLineDash([4, 3]);
      p.circle(r.x, r.y, target ? 18 : 13);
      p.drawingContext.setLineDash([]);
      // rung number inside
      p.noStroke(); p.fill(lit ? D.rgba("#06090D", a) : D.rgba(col, a)); p.textAlign(p.CENTER, p.CENTER); p.textFont("IBM Plex Mono"); p.textSize(8.5);
      p.text(r.rung, r.x, r.y + 0.5);

      // Keep the graphic scannable; the narrative and roadmap carry the details.
      const above = i % 2 === 1;
      const ly = above ? r.y - 26 : r.y + 22;
      p.textAlign(p.LEFT, above ? p.BOTTOM : p.TOP);
      const lx = r.x + 22;
      // state chip
      const chip = lit ? "LIT · NOW" : prog ? "IN PROGRESS" : target ? "TARGET" : "ROADMAP";
      p.fill(D.rgba(col, a)); p.textFont("IBM Plex Mono"); p.textSize(8);
      p.text(chip, lx, above ? ly - 13 : ly);
      const shortTitles = ["Molecular attestation", "Sample identity", "Abundance", "Environment", "Flux validation", "Calibrated risk"];
      p.fill(D.rgba(EB.color.textPrimary, a)); p.textFont("Bricolage Grotesque"); p.textStyle(p.BOLD); p.textSize(12);
      p.text(shortTitles[i], lx, above ? ly : ly + 14); p.textStyle(p.NORMAL);
      p.pop();
    }

    function drawMobileLadder(w, h, tm) {
      const left = 20, right = w - 20;
      p.push();
      p.noStroke(); p.textAlign(p.LEFT, p.TOP);
      p.fill(EB.color.textMuted); p.textFont("IBM Plex Mono"); p.textSize(10);
      p.text("VALIDATION PATH  /  0–5", left, 91);

      p.fill(D.rgba(EB.color.emergence, 0.08));
      p.stroke(D.rgba(EB.color.emergence, 0.55)); p.strokeWeight(1);
      p.rect(left, 119, right - left, 67, 5);
      p.noStroke(); p.fill(EB.color.emergence); p.textSize(9);
      p.text("00  AVAILABLE NOW", left + 14, 132);
      p.fill(EB.color.textPrimary); p.textFont("Bricolage Grotesque"); p.textStyle(p.BOLD); p.textSize(17);
      p.text("Molecular screening", left + 14, 149);
      p.textStyle(p.NORMAL);

      const labels = ["SAMPLE", "ABUND.", "SITE", "FLUX", "RISK"];
      const x0 = left + 21, x1 = right - 21, y = 224;
      p.stroke(D.rgba(EB.color.textMuted, 0.42)); p.strokeWeight(1);
      p.drawingContext.setLineDash([4, 5]); p.line(x0, y, x1, y); p.drawingContext.setLineDash([]);
      for (let i = 0; i < 5; i++) {
        const x = D.lerp(x0, x1, i / 4);
        const col = i === 0 ? EB.color.emergence : i === 4 ? EB.color.methaneA : EB.color.textMuted;
        p.fill(EB.color.bgBase); p.stroke(D.rgba(col, i === 0 ? 0.95 : 0.66)); p.strokeWeight(1.2);
        p.circle(x, y, i === 4 ? 19 : 16);
        p.noStroke(); p.fill(col); p.textFont("IBM Plex Mono"); p.textSize(8); p.textAlign(p.CENTER, p.CENTER);
        p.text(String(i + 1), x, y + 0.5);
        p.fill(D.rgba(col, 0.9)); p.textSize(7.4); p.textAlign(p.CENTER, p.TOP);
        p.text(labels[i], x, y + 15);
      }
      p.fill(EB.color.textMuted); p.textFont("IBM Plex Mono"); p.textSize(8.5); p.textAlign(p.LEFT, p.TOP);
      p.text("PAIRED EVIDENCE NEEDED", left, 276);
      p.textAlign(p.RIGHT, p.TOP); p.text("TARGET · NOT CALIBRATED", right, 276);
      p.pop();
    }
  };
})();
