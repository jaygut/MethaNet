/* SCENE 10 - PARTNERSHIP PATH · PROPOSED FIELD STUDY
   Schematic of the proposed first paired dataset (EB.study): 3 restoration
   stages x 4 salinity positions x 3 replicate plots = 36 plots, two microsites
   per plot at different tidal heights, and wet- and dry-season campaigns that
   revisit the same microsites: 144 planned sample-events, not independent
   replicates. Prospective and conditional on funding, site access and permits.
   Nothing drawn here is collected data, and positions are not a site map.
   The layout stays clear of the DOM reading card and the scene rail. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.path = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw, S = EB.study;
    const WET = EB.color.emergence, DRY = EB.color.methaneA, MUT = EB.color.textMuted, TXT = EB.color.textPrimary;
    const MONO = "IBM Plex Mono", DISPLAY = "Bricolage Grotesque";
    let G = null, key = "";

    // Box of a DOM element relative to this scene's stage, or null when hidden.
    function rel(el) {
      if (!el) return null;
      const base = ctx.holder.getBoundingClientRect(), r = el.getBoundingClientRect();
      if (!r.width || !r.height) return null;
      return { l: r.left - base.left, t: r.top - base.top, r: r.right - base.left, b: r.bottom - base.top };
    }
    function cssPx(name, fallback) {
      const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
      return Number.isFinite(v) ? v : fallback;
    }

    // Free region: beside the reading card when wide enough, otherwise above it.
    function layout() {
      const w = ctx.W, h = ctx.H;
      const copy = rel(ctx.section.querySelector(".copy"));
      const rail = rel(document.querySelector(".rail"));
      const next = [w, h, copy && Math.round(copy.t), copy && Math.round(copy.r), rail && Math.round(rail.l)].join("|");
      if (next === key && G) return;
      key = next;
      const top = cssPx("--header-h", 56) + 26, bottom = h - cssPx("--footer-h", 30) - 22;
      const narrow = w < 720, right = rail ? rail.l - 14 : w - 16;
      const left = narrow ? 16 : Math.max(28, w * 0.05);
      const region = !narrow && copy && right - (copy.r + 48) >= 440
        ? { x: copy.r + 48, y: top, w: right - copy.r - 48, h: bottom - top }
        : { x: left, y: top, w: right - left, h: (copy ? copy.t : h * 0.55) - 20 - top };
      G = geometry(region);
    }

    function geometry(R) {
      const g = { R, plots: [], titles: [], axis: null, mode: "summary" };
      if (R.w < 200 || R.h < 50) { g.mode = "none"; return g; }
      const wide = R.w / R.h > 2.2, compact = R.w < 460 || R.h < 330;
      g.compact = compact;
      g.headH = compact ? 32 : 50;
      g.footH = compact ? 58 : 96;
      if (wide) {
        // Stages as rows; each row holds 4 salinity groups of 3 plots.
        const labelW = compact ? 92 : 140, groupGap = compact ? 12 : 22, gx = compact ? 3 : 7, rowGap = compact ? 6 : 12;
        const gridMax = R.h - g.headH - g.footH - 14;
        const gw = (R.w - labelW - 3 * groupGap - 8 * gx) / 12;
        const gh = Math.min(gw * 0.62, (gridMax - 2 * rowGap) / 3, 44);
        if (gh < 14 || gw < 22) return g;
        const gridH = 14 + 3 * gh + 2 * rowGap;
        g.y0 = R.y + Math.max(0, (R.h - g.headH - gridH - g.footH) * 0.4);
        const top = g.y0 + g.headH + 14;
        for (let s = 0; s < 3; s++) {
          const y = top + s * (gh + rowGap);
          g.titles.push({ s, x: R.x, y: y + gh / 2, w: labelW - 12, middle: true });
          for (let k = 0; k < 4; k++) for (let j = 0; j < 3; j++) {
            g.plots.push({ s, k, j, x: R.x + labelW + k * (3 * gw + 2 * gx + groupGap) + j * (gw + gx), y });
          }
        }
        g.axis = { kind: "row", x: R.x + labelW, y: top - 10 };
        g.footY = top + 3 * gh + 2 * rowGap + (compact ? 12 : 20);
        Object.assign(g, { mode: "rows", gw, gh });
      } else {
        // Stages as columns; each column holds 4 salinity rows of 3 plots.
        const axisW = compact ? 16 : 40, colGap = compact ? 10 : 24, gx = compact ? 4 : 10, gy = compact ? 6 : 12;
        const titleH = compact ? 28 : 36;
        const colW = (R.w - axisW - 2 * colGap) / 3, gw = (colW - 2 * gx) / 3;
        const gh = Math.min((R.h - g.headH - g.footH - titleH - 3 * gy) / 4, gw * 0.66, 46);
        if (gh < 14 || gw < 18) return g;
        const gridH = 4 * gh + 3 * gy;
        g.y0 = R.y + Math.max(0, (R.h - g.headH - titleH - gridH - g.footH) * 0.35);
        const top = g.y0 + g.headH + titleH, left = R.x + axisW;
        for (let s = 0; s < 3; s++) {
          const cx = left + s * (colW + colGap);
          g.titles.push({ s, x: cx, y: g.y0 + g.headH, w: colW });
          for (let k = 0; k < 4; k++) for (let j = 0; j < 3; j++) g.plots.push({ s, k, j, x: cx + j * (gw + gx), y: top + k * (gh + gy) });
        }
        g.axis = { kind: "col", x: R.x + (compact ? 5 : 16), y0: top, y1: top + gridH, gh, gy };
        g.footY = top + gridH + (compact ? 14 : 22);
        Object.assign(g, { mode: "columns", gw, gh });
      }
      g.r = Math.max(2, Math.min(g.gw / 6.5, g.gh / 3.3, 7));
      return g;
    }

    // ---------- text helpers ----------
    function setText(size, font, bold) {
      p.textFont(font || MONO); p.textSize(size); p.textStyle(bold ? p.BOLD : p.NORMAL);
    }
    function wrap(str, maxW, size, font, bold) {
      setText(size, font, bold);
      const lines = []; let line = "";
      for (const word of str.split(" ")) {
        const test = line ? line + " " + word : word;
        if (line && p.textWidth(test) > maxW) { lines.push(line); line = word; } else line = test;
      }
      if (line) lines.push(line);
      return lines;
    }
    function write(str, x, y, size, col, opts = {}) {
      p.push(); p.noStroke(); p.fill(col); setText(size, opts.font, opts.bold);
      p.textAlign(opts.align || p.LEFT, opts.valign || p.TOP); p.text(str, x, y); p.pop();
    }
    function writeLines(lines, x, y, size, col, lead, opts) {
      lines.forEach((ln, i) => write(ln, x, y + i * lead, size, col, opts));
      return y + lines.length * lead;
    }

    p.setup = function () { p.createCanvas(ctx.W, ctx.H); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); key = ""; layout(); if (ctx.reduced) p.noLoop(); };
    p.windowResized = function () { p.resizeCanvas(ctx.W, ctx.H); key = ""; layout(); };

    p.draw = function () {
      const w = ctx.W, h = ctx.H, t = ctx.reduced ? 1 : ctx.progress;
      p.clear(); p.background(EB.color.bgBase);
      D.instrumentGrid(p, w, h, EB.color.hairline, 0.16, 110);
      layout();
      if (G.mode === "rows" || G.mode === "columns") drawDesign(G, t);
      else if (G.mode === "summary") drawSummary(G.R, t);
      D.vignette(p, w, h, EB.color.bgBase, 0.3);
    };

    function drawSummary(R, t) {
      const a = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.05, 0.4));
      write("PROPOSED FIELD STUDY · SCHEMATIC", R.x, R.y, 9, D.rgba(WET, 0.95 * a));
      const lines = wrap(S.plots + " plots × " + S.micrositesPerPlot + " microsites × " + S.campaigns.length +
        " seasons = " + S.sampleEvents + " planned sample-events", R.w, 10.5);
      const y = writeLines(lines, R.x, R.y + 16, 10.5, D.rgba(TXT, 0.9 * a), 15);
      if (y + 14 < R.y + R.h) write("repeated visits, not independent replicates", R.x, y + 3, 8.5, D.rgba(MUT, 0.9 * a));
    }

    function drawDesign(g, t) {
      const R = g.R, c = g.compact;
      const reveal = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.02, 0.3));
      const wet = ctx.reduced ? 1 : D.window01(t, 0.22, 0.55);
      const dry = ctx.reduced ? 1 : D.window01(t, 0.5, 0.8);
      const tally = ctx.reduced ? 1 : D.easeInOut(D.window01(t, 0.7, 0.9));

      // heading: what this is, and what it is not
      write("PROPOSED FIRST PAIRED DATASET", R.x, g.y0, c ? 9.5 : 11, D.rgba(WET, 0.95));
      write(c ? "schematic · conditional on funding and access" : "schematic, not a site map · conditional on funding, site access and permits",
        R.x, g.y0 + (c ? 14 : 18), c ? 8 : 9.5, D.rgba(MUT, 0.95));

      // restoration-stage titles
      for (const title of g.titles) {
        const size = c ? 9.5 : 12, lines = wrap(S.stages[title.s], title.w, size, DISPLAY, true).slice(0, 2);
        const lead = size * 1.2, y = title.middle ? title.y - (lines.length * lead) / 2 : title.y;
        writeLines(lines, title.x, y, size, D.rgba(TXT, 0.92 * D.clamp(reveal * 3 - title.s)), lead, { font: DISPLAY, bold: true });
      }

      // salinity positions
      if (g.axis.kind === "col") {
        const A = g.axis;
        p.push(); p.strokeWeight(2);
        for (let k = 0; k < 4; k++) {
          const yc = A.y0 + k * (A.gh + A.gy) + A.gh / 2;
          p.stroke(D.rgba(EB.color.mangroveMsm, (0.25 + k * 0.2) * reveal)); p.line(A.x, yc - A.gh / 2, A.x, yc + A.gh / 2);
        }
        p.pop();
        if (!c) {
          p.push(); p.translate(A.x - 9, (A.y0 + A.y1) / 2); p.rotate(-p.HALF_PI);
          write("4 SALINITY POSITIONS", 0, 0, 8, D.rgba(MUT, 0.9 * reveal), { align: p.CENTER, valign: p.CENTER });
          p.pop();
        }
      } else {
        write("4 salinity positions per stage, 3 plots each", g.axis.x, g.axis.y, c ? 8 : 9, D.rgba(MUT, 0.9 * reveal), { valign: p.CENTER });
      }

      // 36 plots, two microsites each: filled dot = wet-season event,
      // ring = dry-season revisit of the same microsite
      const rr = g.r;
      for (const q of g.plots) {
        const n = q.s * 12 + q.k * 3 + q.j;
        const a = D.clamp(reveal * 3 - q.s), wa = D.clamp(wet * 36 - n), da = D.clamp(dry * 36 - n);
        if (a <= 0) continue;
        p.push();
        p.noFill(); p.stroke(D.rgba(MUT, 0.45 * a)); p.strokeWeight(1);
        p.rect(q.x, q.y, g.gw, g.gh, Math.min(5, g.gh * 0.2));
        const sites = [[q.x + g.gw * 0.33, q.y + g.gh * 0.6], [q.x + g.gw * 0.67, q.y + g.gh * 0.4]];
        for (const [mx, my] of sites) {
          p.noStroke(); p.fill(D.rgba(MUT, 0.4 * a)); p.circle(mx, my, rr * 1.1);
          // the gap between dot and ring keeps "same point, visited again" legible
          if (wa > 0) { p.fill(D.rgba(WET, 0.85 * wa)); p.circle(mx, my, rr * 1.45); }
          if (da > 0) { p.noFill(); p.stroke(D.rgba(DRY, 0.9 * da)); p.strokeWeight(1.1); p.circle(mx, my, rr * 2 + 5); }
        }
        p.pop();
      }

      // legend, tally and the replicate note
      let y = g.footY;
      const lx = R.x;
      p.push(); p.noStroke(); p.fill(D.rgba(WET, 0.85)); p.circle(lx + 5, y + 6, 7); p.pop();
      write(c ? "wet-season event" : "wet-season sample-event", lx + 16, y, c ? 8.5 : 9.5, D.rgba(TXT, 0.85));
      setText(c ? 8.5 : 9.5);
      const lx2 = lx + 16 + p.textWidth(c ? "wet-season event" : "wet-season sample-event") + 20;
      p.push(); p.noFill(); p.stroke(D.rgba(DRY, 0.95)); p.strokeWeight(1.3); p.circle(lx2 + 5, y + 6, 11); p.pop();
      write(c ? "dry-season revisit" : "dry-season revisit, same microsite", lx2 + 16, y, c ? 8.5 : 9.5, D.rgba(TXT, 0.85));
      y += c ? 16 : 20;
      if (!c) {
        y = writeLines(wrap("Each event pairs a sediment metagenome with chamber CH₄ flux, chemistry and hydrology.", R.w, 9), lx, y, 9, D.rgba(MUT, 0.95), 13) + 6;
      }
      const size = c ? 10.5 : 13;
      const parts = [[S.plots + " plots × " + S.micrositesPerPlot + " microsites × " + S.campaigns.length + " seasons = ", TXT], [String(S.sampleEvents), DRY], [" planned sample-events", TXT]];
      setText(size);
      const full = parts.map((x) => x[0]).join("");
      if (p.textWidth(full) <= R.w) {
        let x = lx;
        for (const [str, col] of parts) { write(str, x, y, size, D.rgba(col, 0.95 * tally)); setText(size); x += p.textWidth(str); }
        y += size + 8;
      } else {
        y = writeLines(wrap(full, R.w, size), lx, y, size, D.rgba(TXT, 0.95 * tally), size + 4) + 4;
      }
      writeLines(wrap(c ? "Repeated visits, not independent replicates."
        : "Revisits are repeated measurements, not independent replicates; plots are the replicate unit.", R.w, c ? 8.5 : 9.5),
        lx, y, c ? 8.5 : 9.5, D.rgba(MUT, 0.95 * tally), c ? 12 : 14);
    }
  };
})();
