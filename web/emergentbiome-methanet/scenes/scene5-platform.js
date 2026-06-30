/* SCENE 5 - THE PLATFORM & THE MOAT  ·  REAL SCHEMA
   Reveals EmergentBiome as the substrate (embeddings + attestation graph + agentic
   pipeline) with MethaNet lit as the first application and honestly-empty future
   slots. Then a real molecular-attestation graph lights ONE claim and traces its
   evidence chain: Genome -> Markers -> embedding neighbors -> quality gate -> Claim boundary,
   ending in the real allowed-vs-forbidden wording. Counts are the live MMAG snapshot. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.platform = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, N = EB.num, AT = EB.attestation;
    let rng, gnodes = [], gedges = [], chain = [];

    function layout() {
      rng = window.EBRandom.RNG("platform");
      const w = ctx.W, h = ctx.H;
      // background evidence-graph scatter (texture for the real graph)
      gnodes = [];
      const gn = Math.round(D.clamp((w * h) / 5200, 90, 220));
      for (let i = 0; i < gn; i++) gnodes.push({ x: rng.range(0, w), y: rng.range(h * 0.42, h), ph: rng.range(0, 6.28), sz: rng.range(0.8, 2) });
      gedges = [];
      for (let i = 0; i < gnodes.length; i++) {
        for (let k = 0; k < 2; k++) {
          const j = rng.int(0, gnodes.length - 1);
          if (j !== i) gedges.push([i, j]);
        }
      }
      // evidence chain node positions (left -> right band)
      const x0 = w * 0.10, x1 = w * 0.90, cy = h * 0.56;
      chain = AT.chain.map((c, i) => ({
        ...c, x: D.lerp(x0, x1, i / (AT.chain.length - 1)),
        y: cy + Math.sin(i * 1.3) * h * 0.045,
      }));
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.2, 110);

      const subA = D.window01(t, 0.0, 0.22);
      const appA = D.window01(t, 0.18, 0.40);
      const chainStart = 0.40, chainEnd = 0.96;
      const tm = ctx.reduced ? 0 : p.frameCount * 0.02;

      drawPlatform(w, h, subA, appA);
      drawReadouts(w, h, appA);

      // background evidence graph (fades in with chain)
      const gA = D.window01(t, 0.42, 0.7);
      if (gA > 0) {
        p.push(); p.strokeWeight(1);
        for (const [i, j] of gedges) { const a = gnodes[i], b = gnodes[j]; p.stroke(D.rgba(EB.color.emergence, 0.05 * gA)); p.line(a.x, a.y, b.x, b.y); }
        p.noStroke();
        for (const n of gnodes) { const tw = 0.2 + 0.15 * Math.sin(tm + n.ph); p.fill(D.rgba(EB.color.attested, tw * gA)); p.circle(n.x, n.y, n.sz); }
        p.pop();
      }

      // evidence chain
      D.label(p, "ONE CLAIM, TRACED TO ITS EVIDENCE", w * 0.10, h * 0.44, EB.color.textMuted, 11);
      const nStages = chain.length;
      for (let i = 0; i < nStages; i++) {
        const a = D.window01(t, D.lerp(chainStart, chainEnd, i / nStages), D.lerp(chainStart, chainEnd, (i + 0.8) / nStages));
        // beam from previous node
        if (i > 0) {
          const prev = chain[i - 1], cur = chain[i];
          const beam = D.window01(t, D.lerp(chainStart, chainEnd, (i - 0.3) / nStages), D.lerp(chainStart, chainEnd, (i + 0.5) / nStages));
          p.push(); p.blendMode(p.ADD); p.strokeWeight(1.5);
          p.stroke(D.rgba(EB.color.emergence, 0.5 * beam)); p.line(prev.x, prev.y, D.lerp(prev.x, cur.x, beam), D.lerp(prev.y, cur.y, beam));
          // travelling spark
          if (!ctx.reduced && beam > 0 && beam < 1) { const sx = D.lerp(prev.x, cur.x, beam), sy = D.lerp(prev.y, cur.y, beam); D.glow(p, sx, sy, 2.4, EB.color.attested, 0.9); }
          p.pop();
        }
        drawChainNode(chain[i], i, a, tm);
      }

      // claim boundary card at the final node
      const cardA = D.window01(t, 0.84, 1.0);
      if (cardA > 0.01) drawClaimCard(chain[nStages - 1], w, h, cardA);

      D.vignette(p, w, h, EB.color.bgBase, 0.45);
    };

    function drawChainNode(n, i, a, tm) {
      if (a <= 0) return;
      const last = i === chain.length - 1;
      const col = last ? EB.color.attested : EB.color.emergence;
      p.push(); p.blendMode(p.ADD); p.noStroke();
      const pulse = ctx.reduced ? 1 : (0.7 + 0.3 * Math.sin(tm * 2 + i));
      D.glow(p, n.x, n.y, 4 + (last ? 2 : 0), col, a * pulse);
      p.pop();
      p.push();
      p.noFill(); p.stroke(D.rgba(col, 0.6 * a)); p.strokeWeight(1.2); p.circle(n.x, n.y, last ? 30 : 22);
      // labels
      p.noStroke(); p.fill(D.rgba(EB.color.textPrimary, a)); p.textAlign(p.CENTER, p.BOTTOM); p.textFont("Space Grotesk"); p.textSize(13);
      p.text(n.stage, n.x, n.y - 20);
      p.fill(D.rgba(EB.color.textMuted, 0.9 * a)); p.textFont("JetBrains Mono"); p.textSize(8.6); p.textAlign(p.CENTER, p.TOP);
      wrapText(n.detail, n.x, n.y + 18, 150, 11);
      p.fill(D.rgba(col, 0.7 * a)); p.textSize(8);
      p.text(n.node, n.x, n.y + 18 - 12);
      p.pop();
    }

    function drawClaimCard(n, w, h, a) {
      const cw = Math.min(w * 0.42, 420), x = D.clamp(n.x - cw / 2, 16, w - cw - 16), y = n.y + 56;
      p.push();
      p.fill(D.rgba(EB.color.bgPanel, 0.85 * a)); p.stroke(D.rgba(EB.color.attested, 0.5 * a)); p.strokeWeight(1);
      p.rect(x, y, cw, 92, 4);
      p.noStroke();
      D.label(p, "ALLOWED WORDING", x + 14, y + 18, D.rgba(EB.color.attested, a), 8.5);
      p.fill(D.rgba(EB.color.textPrimary, a)); p.textFont("Inter"); p.textSize(11); p.textAlign(p.LEFT, p.TOP);
      wrapTextLeft("“" + AT.claimText + "”", x + 14, y + 28, cw - 28, 14);
      D.label(p, "FORBIDDEN", x + 14, y + 74, D.rgba(EB.color.methaneB, a), 8.5);
      p.fill(D.rgba(EB.color.methaneB, 0.85 * a)); p.textSize(10.5);
      p.text(AT.forbidden, x + 78, y + 70);
      // strike line on forbidden
      p.stroke(D.rgba(EB.color.methaneB, 0.7 * a)); p.strokeWeight(1); p.line(x + 78, y + 75, x + 78 + p.textWidth(AT.forbidden), y + 75);
      p.pop();
    }

    function drawPlatform(w, h, subA, appA) {
      // application slots (top): MethaNet lit + honest empty future slots
      const slotW = Math.min(w * 0.16, 168), gap = 16, n = 3;
      const totalW = n * slotW + (n - 1) * gap;
      const sx = (w - totalW) / 2, sy = h * 0.13, sh = 40;
      D.label(p, "APPLICATIONS", w / 2, sy - 10, D.rgba(EB.color.textMuted, appA), 10, [p.CENTER, p.BOTTOM]);
      for (let i = 0; i < n; i++) {
        const x = sx + i * (slotW + gap);
        p.push(); p.rectMode(p.CORNER);
        if (i === 0) {
          const g = p.drawingContext.createLinearGradient(x, 0, x + slotW, 0);
          g.addColorStop(0, EB.color.methaneA); g.addColorStop(1, EB.color.methaneB);
          p.drawingContext.fillStyle = g; p.drawingContext.globalAlpha = appA;
          p.noStroke(); p.rect(x, sy, slotW, sh, 4); p.drawingContext.globalAlpha = 1;
          p.fill(D.rgba("#0A0F12", appA)); p.textAlign(p.CENTER, p.CENTER); p.textFont("Space Grotesk"); p.textStyle(p.BOLD); p.textSize(13);
          p.text("MethaNet", x + slotW / 2, sy + sh / 2); p.textStyle(p.NORMAL);
          D.label(p, "methane · live", x + slotW / 2, sy + sh + 12, D.rgba(EB.color.methaneA, appA), 8.5, [p.CENTER, p.CENTER]);
        } else if (i === 1) {
          // N2O: the second lens from the prior scene, an honest candidate slot (not a live product)
          p.noFill(); p.drawingContext.setLineDash([4, 3]);
          p.stroke(D.rgba("#7C8CC4", 0.6 * appA)); p.strokeWeight(1.1); p.rect(x, sy, slotW, sh, 4);
          p.drawingContext.setLineDash([]);
          p.noStroke(); p.fill(D.rgba("#A9B6D8", 0.9 * appA)); p.textAlign(p.CENTER, p.CENTER); p.textFont("Space Grotesk"); p.textStyle(p.BOLD); p.textSize(12);
          p.text("N₂O", x + slotW / 2, sy + sh / 2); p.textStyle(p.NORMAL);
          D.label(p, "candidate · uncalibrated", x + slotW / 2, sy + sh + 12, D.rgba("#7C8CC4", 0.7 * appA), 8.5, [p.CENTER, p.CENTER]);
        } else {
          p.noFill(); p.drawingContext.setLineDash([4, 4]);
          p.stroke(D.rgba(EB.color.textMuted, 0.4 * appA)); p.strokeWeight(1); p.rect(x, sy, slotW, sh, 4);
          p.drawingContext.setLineDash([]);
          p.noStroke(); p.fill(D.rgba(EB.color.textMuted, 0.5 * appA)); p.textAlign(p.CENTER, p.CENTER); p.textFont("JetBrains Mono"); p.textSize(16);
          p.text("+", x + slotW / 2, sy + sh / 2);
          D.label(p, "future", x + slotW / 2, sy + sh + 12, D.rgba(EB.color.textMuted, 0.5 * appA), 8.5, [p.CENTER, p.CENTER]);
        }
        p.pop();
      }
      // substrate: three pillars under the apps
      const subY = sy + sh + 30, subH = 26;
      const cells = ["EMBEDDINGS", "ATTESTATION GRAPH", "AGENTIC PIPELINE"];
      const subX = (w - totalW) / 2, subW = totalW;
      p.push();
      p.fill(D.rgba(EB.color.bgElevated, 0.8 * subA)); p.stroke(D.rgba(EB.color.emergence, 0.5 * subA)); p.strokeWeight(1);
      p.rect(subX, subY, subW, subH, 4);
      p.noStroke();
      for (let i = 0; i < 3; i++) {
        const cx = subX + (i + 0.5) * (subW / 3);
        if (i > 0) { p.stroke(D.rgba(EB.color.emergence, 0.25 * subA)); p.line(subX + i * (subW / 3), subY + 4, subX + i * (subW / 3), subY + subH - 4); p.noStroke(); }
        p.fill(D.rgba(EB.color.emergence, 0.95 * subA)); p.textAlign(p.CENTER, p.CENTER); p.textFont("JetBrains Mono"); p.textSize(8.6);
        p.text(cells[i], cx, subY + subH / 2);
      }
      D.label(p, "EmergentBiome substrate · every gas inherits the same evidence trail", w / 2, subY + subH + 14, D.rgba(EB.color.emergence, 0.8 * subA), 9.5, [p.CENTER, p.CENTER]);
      // connector from MethaNet down to substrate
      p.stroke(D.rgba(EB.color.methaneA, 0.4 * subA)); p.strokeWeight(1);
      p.line(sx + slotW / 2, sy + sh + 18, sx + slotW / 2, subY);
      p.pop();
    }

    function drawReadouts(w, h, a) {
      const items = [
        [D.fmt(N.evidenceAtoms), "evidence atoms"],
        [D.fmt(N.nearEsm2Edges), "embedding-neighbor links"],
        [N.claimNodes, "claims"],
        [N.validationGapNodes, "validation gaps"],
      ];
      const x = w - 18; let y = h * 0.40;
      p.push(); p.textAlign(p.RIGHT, p.CENTER);
      for (const [v, k] of items) {
        p.fill(D.rgba(EB.color.textPrimary, a)); p.textFont("JetBrains Mono"); p.textSize(13); p.text(v, x, y);
        p.fill(D.rgba(EB.color.textMuted, a)); p.textSize(8.5); p.text(k.toUpperCase(), x, y + 12);
        y += 30;
      }
      p.pop();
    }

    function wrapText(txt, cx, y, maxw, lh) {
      const words = txt.split(" "); let line = "", yy = y;
      for (const wd of words) { const test = line + wd + " "; if (p.textWidth(test) > maxw && line) { p.text(line.trim(), cx, yy); line = wd + " "; yy += lh; } else line = test; }
      if (line) p.text(line.trim(), cx, yy);
    }
    function wrapTextLeft(txt, x, y, maxw, lh) {
      const words = txt.split(" "); let line = "", yy = y;
      for (const wd of words) { const test = line + wd + " "; if (p.textWidth(test) > maxw && line) { p.text(line.trim(), x, yy); line = wd + " "; yy += lh; } else line = test; }
      if (line) p.text(line.trim(), x, yy);
    }
  };
})();
