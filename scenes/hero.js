/* HERO intro - "molecular dark matter → verifiable structure".
   A seeded particle field where scattered particles slowly condense toward a few
   luminous nodes and link up: the emergent glyph, writ large. Decorative brand
   intro (behind the title), not a data claim. */
(function () {
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.hero = function (p, ctx) {
    const EB = window.EB, D = window.EBDraw;
    let rng, particles = [], anchors = [];
    const ACC = EB.color.emergence, ACC2 = EB.color.attested;

    function layout() {
      rng = window.EBRandom.RNG("hero-" + Math.round(ctx.W) + "x" + Math.round(ctx.H));
      const w = ctx.W, h = ctx.H;
      const cx = w * 0.5, cy = h * 0.5;
      // a few condensation anchors arranged like the glyph
      anchors = [];
      const k = 5;
      for (let i = 0; i < k; i++) {
        const a = (i / k) * Math.PI * 2 + 0.3;
        const r = Math.min(w, h) * (i === 0 ? 0 : 0.16);
        anchors.push({ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r });
      }
      const N = Math.round(D.clamp((w * h) / 9000, 120, 280));
      particles = [];
      for (let i = 0; i < N; i++) {
        const anchor = anchors[rng.int(0, anchors.length - 1)];
        particles.push({
          x: rng.range(0, w), y: rng.range(0, h),
          ax: anchor.x + rng.gauss(0, 26), ay: anchor.y + rng.gauss(0, 26),
          ph: rng.range(0, Math.PI * 2), sp: rng.range(0.3, 1),
          sz: rng.range(0.6, 2.0), bright: rng.next() < 0.18,
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
      p.clear();
      p.background(EB.color.bgBase);
      // cohesion grows with time (condensation), capped
      const tt = ctx.reduced ? 1 : D.clamp(p.frameCount / 260);
      const t = p.frameCount * 0.006;

      // hairline links between condensed neighbors
      p.push();
      p.strokeWeight(1);
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        const px = D.lerp(a.x, a.ax + Math.cos(a.ph + t * a.sp) * 8, tt);
        const py = D.lerp(a.y, a.ay + Math.sin(a.ph + t * a.sp) * 8, tt);
        a._px = px; a._py = py;
      }
      for (let i = 0; i < particles.length; i += 2) {
        const a = particles[i];
        for (let j = i + 1; j < Math.min(i + 7, particles.length); j++) {
          const b = particles[j];
          const dx = a._px - b._px, dy = a._py - b._py;
          const d2 = dx * dx + dy * dy;
          if (d2 < 4200) {
            const al = (1 - d2 / 4200) * 0.16 * tt;
            p.stroke(D.rgba(ACC, al));
            p.line(a._px, a._py, b._px, b._py);
          }
        }
      }
      p.pop();

      // particles
      p.push();
      p.blendMode(p.ADD);
      p.noStroke();
      for (const a of particles) {
        if (a.bright) {
          D.glow(p, a._px, a._py, a.sz * 2.4, ACC2, 0.5 + 0.5 * tt);
        } else {
          p.fill(D.rgba(ACC, 0.5));
          p.circle(a._px, a._py, a.sz);
        }
      }
      p.pop();

      D.vignette(p, w, h, EB.color.bgBase, 0.7);
    };
  };
})();
