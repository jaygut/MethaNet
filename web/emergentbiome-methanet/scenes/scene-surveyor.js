/* SCENE - WHAT YOU GET  ·  the concrete output object, shown (not described).
   The product is a ranked triage list: for each site, a molecular methane-risk SIGNAL
   (relative, unitless), a CONFIDENCE band, an EVIDENCE link back to the atlas, and a
   "measure here next" call on the top sites. Then the two-column truth so the "what are
   the units?" question is pre-empted on the page: what we give, and what we do not give
   (yet). Illustrative product shape: the SHAPE is exact, the per-site scores are a mock,
   and the ranking follows the real science (risk concentrates in brackish / freshened /
   restored systems where the salinity proxy is least reliable). Screening, not a flux. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.surveyor = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    const M_A = EB.color.methaneA, M_B = EB.color.methaneB, LINK = EB.color.attested, MUT = EB.color.textMuted;
    // ranked sites: sorted by molecular methane-risk signal. Order follows the science
    // (brackish / freshened / restored rank high; saline ranks low). sig + confidence are illustrative.
    const SITES = [
      { name: "Brackish delta", note: "freshened", sig: 0.92, conf: 0.07 },
      { name: "Restored impoundment", note: "rewetted", sig: 0.85, conf: 0.10 },
      { name: "Freshened basin", note: "low salinity", sig: 0.77, conf: 0.09 },
      { name: "Tidal creek margin", note: "mixed", sig: 0.60, conf: 0.12 },
      { name: "Peat-fringe wetland", note: "freshwater", sig: 0.54, conf: 0.11 },
      { name: "Mangrove fringe", note: "mesohaline", sig: 0.37, conf: 0.10 },
      { name: "Saline flat", note: "polyhaline", sig: 0.18, conf: 0.08 },
    ];
    let G = {};

    function layout() {
      const w = ctx.W, h = ctx.H, narrow = w < 720;
      G.narrow = narrow;
      G.x0 = narrow ? w * 0.06 : w * 0.40;
      G.x1 = narrow ? w * 0.94 : w * 0.95;
      G.rankX = G.x0;
      G.labelX = G.x0 + (narrow ? 30 : 34);
      G.barX = narrow ? w * 0.42 : w * 0.575;
      G.barMax = (narrow ? w * 0.34 : w * 0.165);
      G.flagX = narrow ? w * 0.90 : w * 0.755;
      G.evX = narrow ? w * 0.90 : w * 0.865;
      G.top = h * (narrow ? 0.15 : 0.185);
      G.rowH = h * (narrow ? 0.058 : 0.062);
      G.twoColY = G.top + SITES.length * G.rowH + h * 0.03;
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.22, 100);
      const tm = ctx.reduced ? 0.5 : p.frameCount * 0.02;

      // header
      D.label(p, "METHANE-RISK SCREENING · SITE PRIORITIES", G.x0, G.top - h * 0.045, D.rgba(MUT, 0.95), 11);
      D.label(p, "SITE", G.labelX, G.top - h * 0.016, D.rgba(MUT, 0.6), 8.5);
      D.label(p, "MOLECULAR METHANE-RISK SIGNAL", G.barX, G.top - h * 0.016, D.rgba(MUT, 0.6), 8.5);

      let measured = 0;
      for (let i = 0; i < SITES.length; i++) {
        const rowA = ctx.reduced ? 1 : D.easeInOut(D.window01(t, i * 0.045, i * 0.045 + 0.28));
        if (rowA < 0.01) continue;
        const flagOn = i < 2 && (ctx.reduced ? true : D.window01(t, 0.32 + i * 0.06, 0.42 + i * 0.06) > 0.5);
        if (flagOn) measured++;
        drawRow(SITES[i], i, rowA, flagOn, tm);
      }

      // two-column truth: what it is / is not (the "not a flux number" boundary lives here + in the claim bar)
      const colA = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.55, 0.8));
      if (colA > 0.01) drawTwoColumn(w, h, colA);

      D.vignette(p, w, h, EB.color.bgBase, 0.5);
    };

    function drawRow(s, i, a, flagOn, tm) {
      const y = G.top + i * G.rowH + G.rowH * 0.5;
      p.push();
      // rank
      p.fill(D.rgba(MUT, 0.7 * a)); p.noStroke(); p.textFont("JetBrains Mono"); p.textSize(10); p.textAlign(p.LEFT, p.CENTER);
      p.text((i + 1 < 10 ? "0" : "") + (i + 1), G.rankX, y);
      // label + note
      p.fill(D.rgba(EB.color.textPrimary, a)); p.textFont("Space Grotesk"); p.textSize(13); p.textAlign(p.LEFT, p.CENTER);
      p.text(s.name, G.labelX, y - 5);
      p.fill(D.rgba(MUT, 0.8 * a)); p.textFont("JetBrains Mono"); p.textSize(8.5);
      p.text(s.note, G.labelX, y + 8);
      // signal bar (methane gradient by signal), grows with row reveal
      const bx = G.barX, bw = G.barMax * s.sig * a, bh = 7;
      const col = D.lerpHex(M_A, M_B, s.sig);
      p.noStroke(); p.fill(D.rgba(EB.color.hairline, 0.9 * a)); p.rect(bx, y - bh / 2, G.barMax, bh, 2);
      p.fill(D.rgba(col, 0.95 * a)); p.rect(bx, y - bh / 2, bw, bh, 2);
      // confidence band: faint range around the bar end
      const cw = G.barMax * s.conf;
      p.fill(D.rgba(col, 0.22 * a)); p.rect(bx + bw - cw, y - bh / 2 - 2, cw * 2, bh + 4, 2);
      p.stroke(D.rgba(col, 0.6 * a)); p.strokeWeight(1);
      p.line(bx + bw - cw, y - bh / 2 - 3, bx + bw - cw, y + bh / 2 + 3);
      p.line(bx + bw + cw, y - bh / 2 - 3, bx + bw + cw, y + bh / 2 + 3);
      p.noStroke();
      // evidence link glyph (traces to the atlas) - desktop only; mobile stays uncluttered
      if (!G.narrow) {
        drawEvidence(G.evX, y, a);
        D.label(p, "evidence", G.evX + 12, y + 0.5, D.rgba(LINK, 0.7 * a), 8, [p.LEFT, p.CENTER]);
      }
      // measure-here-next call on the top sites
      if (flagOn) {
        p.push();
        if (G.narrow) {
          p.noStroke(); p.fill(M_A); p.circle(G.flagX, y, 8);
        } else {
          p.stroke(D.rgba(M_A, 0.9)); p.strokeWeight(1); p.noFill(); p.rect(G.flagX, y - 9, 116, 18, 3);
          p.noStroke(); p.fill(M_A); p.textFont("JetBrains Mono"); p.textSize(8.5); p.textAlign(p.LEFT, p.CENTER);
          p.text("MEASURE HERE NEXT", G.flagX + 7, y + 0.5);
        }
        p.pop();
      }
      p.pop();
    }

    function drawEvidence(x, y, a) {
      // small chain-link glyph
      p.push(); p.noFill(); p.stroke(D.rgba(LINK, 0.85 * a)); p.strokeWeight(1.2);
      p.ellipse(x - 3, y, 7, 5); p.ellipse(x + 3, y, 7, 5);
      p.pop();
    }

    function drawTwoColumn(w, h, a) {
      if (G.narrow) return;   // mobile: the copy card body + readout + claim bar carry the "what it is / is not" boundary
      const sx0 = w * 0.44;   // clear of the bottom-left copy card
      const y = G.twoColY, colW = (G.x1 - sx0 - (G.narrow ? 0 : w * 0.02)) / (G.narrow ? 1 : 2);
      const give = {
        x: sx0, title: "WE GIVE YOU", col: LINK,
        rows: ["a ranked methane-risk signal per site", "where to spend the first flux measurement", "which credits deserve scrutiny", "an evidence trail for every call"],
      };
      const dont = {
        x: G.narrow ? sx0 : sx0 + colW + w * 0.02, title: "WE DO NOT GIVE YOU (YET)", col: MUT,
        rows: ["a methane flux number", "a final A to E risk tier", "a credit adjustment", "those need field validation, which is the roadmap"],
      };
      const cols = G.narrow ? [give] : [give, dont];   // on mobile, show the give column (the boundary lives in the claim bar)
      for (const c of cols) {
        let yy = y;
        p.push();
        p.noFill(); p.stroke(D.rgba(c.col, 0.35 * a)); p.strokeWeight(1);
        p.rect(c.x, yy, colW, h * 0.005 + 22 + c.rows.length * 16, 4);
        p.pop();
        D.label(p, c.title, c.x + 12, yy + 15, D.rgba(c.col, a), 9.5);
        yy += 30;
        for (const r of c.rows) {
          p.push(); p.noStroke();
          p.fill(D.rgba(c.col, 0.8 * a)); p.circle(c.x + 15, yy - 3, 3);
          p.fill(D.rgba(EB.color.textPrimary, 0.82 * a)); p.textFont("Inter"); p.textSize(10.5); p.textAlign(p.LEFT, p.TOP);
          p.text(r, c.x + 24, yy - 9);
          p.pop();
          yy += 16;
        }
      }
    }
  };
})();
