/* SCENE 7 - THE PATH & THE ASK  ·  VALIDATION PATH
   An illustrative partner-cohort pin, the calibration-core pipeline benchmark,
   and the milestone timeline: molecular atlas -> field
   validation -> paired data -> calibrated risk -> registry integration. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.path = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, N = EB.num, TL = EB.timeline;
    let rng, tokens = [], coast = [], pin = {}, stages = [], miles = [];

    function layout() {
      rng = window.EBRandom.RNG("path");
      const w = ctx.W, h = ctx.H;
      // stylized coastline curve; geography and pin are illustrative
      coast = [];
      const cy = h * 0.30;
      for (let i = 0; i <= 40; i++) {
        const x = (i / 40) * w;
        const y = cy + Math.sin(i * 0.35) * h * 0.05 + Math.sin(i * 0.11 + 1) * h * 0.03;
        coast.push({ x, y });
      }
      pin = { x: w * 0.34, y: cy + h * 0.02 };
      // pipeline stage gates
      const px0 = w * 0.08, px1 = w * 0.92, py = h * 0.47;
      const names = ["quality", "annotation", "embeddings", "attestation"];
      stages = names.map((nm, i) => ({ nm, x: D.lerp(px0 + 90, px1 - 90, i / (names.length - 1)), y: py }));
      tokens = [];
      // milestones (right band - clears the bottom-left copy card)
      const mx0 = w * 0.42, mx1 = w * 0.95, my = h * 0.76;
      miles = TL.map((m, i) => ({ ...m, x: D.lerp(mx0, mx1, i / (TL.length - 1)), y: my, lit: i === 0 }));
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.18, 110);
      const tm = ctx.reduced ? 0 : p.frameCount * 0.02;

      drawCoast(w, h, tm);
      drawPipeline(w, h, t, tm);
      drawMilestones(w, h, t, tm);
      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawCoast(w, h, tm) {
      // land fill below coast
      p.push();
      p.noStroke(); p.fill(D.rgba("#0B130E", 0.8));
      p.beginShape(); p.vertex(0, h * 0.5); for (const c of coast) p.vertex(c.x, c.y); p.vertex(w, h * 0.5); p.endShape(p.CLOSE);
      p.stroke(D.rgba(EB.color.wetland, 0.4)); p.strokeWeight(1.2); p.noFill();
      p.beginShape(); for (const c of coast) p.vertex(c.x, c.y); p.endShape();
      p.pop();
      // sonar pin
      p.push(); p.blendMode(p.ADD); p.noFill();
      const ph = ctx.reduced ? 0.5 : (p.frameCount % 120) / 120;
      for (let k = 0; k < 2; k++) { const pr = ((ph + k * 0.5) % 1); p.stroke(D.rgba(EB.color.methaneA, (1 - pr) * 0.5)); p.strokeWeight(1.2); p.circle(pin.x, pin.y, pr * 70); }
      D.glow(p, pin.x, pin.y, 4, EB.color.methaneA, 0.9);
      p.pop();
      // pin label
      p.push(); p.noStroke();
      p.stroke(D.rgba(EB.color.methaneA, 0.6)); p.strokeWeight(1); p.line(pin.x, pin.y, pin.x, pin.y - 34); p.noStroke();
      p.fill(EB.color.textPrimary); p.textFont("Bricolage Grotesque"); p.textStyle(p.BOLD); p.textSize(14); p.textAlign(p.LEFT, p.BOTTOM);
      p.text("Partner validation cohort", pin.x + 8, pin.y - 34); p.textStyle(p.NORMAL);
      p.fill(D.rgba(EB.color.textMuted, 0.95)); p.textFont("IBM Plex Mono"); p.textSize(9); p.textAlign(p.LEFT, p.TOP);
      p.text("illustrative location · paired evidence required", pin.x + 8, pin.y - 32);
      p.pop();
    }

    function drawPipeline(w, h, t, tm) {
      const px0 = w * 0.08, px1 = w * 0.92, py = stages[0].y;
      // rail (with additive glow underlay)
      p.push(); p.blendMode(p.ADD); p.stroke(D.rgba(EB.color.emergence, 0.18)); p.strokeWeight(5); p.line(px0, py, px1, py); p.pop();
      p.push(); p.stroke(D.rgba(EB.color.emergence, 0.55)); p.strokeWeight(2); p.line(px0, py, px1, py); p.pop();

      // title (left) + speed readout (centered) - kept on separate rows, no overlap
      D.label(p, "AGENTIC PIPELINE", px0, py - 46, EB.color.textMuted, 11);
      p.push(); p.textAlign(p.CENTER, p.BASELINE);
      p.fill(EB.color.attested); p.noStroke(); p.textFont("IBM Plex Mono"); p.textSize(16);
      p.text("< " + N.pipelineDays + " days", w * 0.5, py - 38);
      p.fill(D.rgba(EB.color.textMuted, 0.85)); p.textSize(10);
      p.text("calibration-core benchmark to a reproducible report", w * 0.5, py - 22);
      p.pop();

      // input / output labels above the rail ends (clear of stage labels below)
      p.push(); p.noStroke(); p.textFont("IBM Plex Mono"); p.textSize(10);
      p.fill(D.rgba(EB.color.mangroveMsm, 0.95)); p.textAlign(p.LEFT, p.BOTTOM); p.text("raw sequences →", px0, py - 10);
      p.fill(D.rgba(EB.color.attested, 0.95)); p.textAlign(p.RIGHT, p.BOTTOM); p.text("→ report", px1, py - 10);
      p.pop();

      // stage gates (boxes on rail, labels below)
      for (let i = 0; i < stages.length; i++) {
        const s = stages[i];
        p.push();
        p.fill(D.rgba(EB.color.bgElevated, 0.95)); p.stroke(D.rgba(EB.color.emergence, 0.7)); p.strokeWeight(1.2);
        p.rectMode(p.CENTER); p.rect(s.x, s.y, 16, 34, 3);
        p.noStroke(); p.fill(D.rgba(EB.color.textPrimary, 0.95)); p.textFont("IBM Plex Mono"); p.textSize(9); p.textAlign(p.CENTER, p.TOP);
        p.text(s.nm, s.x, s.y + 24);
        p.pop();
      }
      // flowing tokens
      if (!ctx.reduced) { if (p.frameCount % 9 === 0 && tokens.length < 40) tokens.push({ u: 0, sp: rng.next() }); }
      else { if (tokens.length === 0) for (let k = 0; k < 8; k++) tokens.push({ u: k / 8, sp: 0.5 }); }
      p.push(); p.blendMode(p.ADD); p.noStroke();
      for (let i = tokens.length - 1; i >= 0; i--) {
        const tk = tokens[i]; if (!ctx.reduced) tk.u += 0.004 + tk.sp * 0.004;
        if (tk.u >= 1) { tokens.splice(i, 1); continue; }
        const x = D.lerp(px0, px1, tk.u);
        const col = D.lerpHex(EB.color.mangroveMsm, EB.color.attested, tk.u);
        D.glow(p, x, py, 2, col, 0.8);
      }
      p.pop();
    }

    function drawMilestones(w, h, t, tm) {
      const reveal = D.easeInOut(t);
      D.label(p, "THE PATH TO CALIBRATED METHANE RISK", miles[0].x, miles[0].y - 40, EB.color.textMuted, 11);
      // connecting line drawn with progress
      for (let i = 1; i < miles.length; i++) {
        const a = miles[i - 1], b = miles[i];
        const seg = D.clamp(reveal * miles.length - (i - 1));
        if (seg <= 0) continue;
        p.push(); p.drawingContext.setLineDash([5, 4]); p.stroke(D.rgba(EB.color.emergence, 0.35)); p.strokeWeight(1);
        p.line(a.x, a.y, D.lerp(a.x, b.x, seg), D.lerp(a.y, b.y, seg)); p.drawingContext.setLineDash([]); p.pop();
      }
      for (let i = 0; i < miles.length; i++) {
        const m = miles[i];
        const a = ctx.reduced ? 1 : D.clamp(reveal * miles.length - i + 0.3);
        if (a <= 0) continue;
        const col = m.lit ? EB.color.emergence : EB.color.textMuted;
        if (m.lit) { p.push(); p.blendMode(p.ADD); D.glow(p, m.x, m.y, 4, col, (ctx.reduced ? 1 : 0.7 + 0.3 * Math.sin(tm * 2)) * a); p.pop(); }
        p.noStroke(); p.fill(m.lit ? col : D.rgba(EB.color.bgBase, a)); p.stroke(D.rgba(col, a)); p.strokeWeight(1.2); p.circle(m.x, m.y, 11);
        p.noStroke(); p.fill(D.rgba(EB.color.textMuted, a)); p.textFont("IBM Plex Mono"); p.textSize(8); p.textAlign(p.CENTER, p.BOTTOM);
        p.text(m.phase.toUpperCase() + (m.lit ? " · NOW" : ""), m.x, m.y - 12);
        p.fill(D.rgba(EB.color.textPrimary, a)); p.textFont("Bricolage Grotesque"); p.textSize(11); p.textAlign(p.CENTER, p.TOP);
        p.text(m.label, m.x, m.y + 10);
      }
    }
  };
})();
