/* Shared draw + math helpers used by every p5 scene (instance mode).
   Keeps the visual language consistent: same easing, same bloom, same
   instrument-grade grid/ticks across all seven scenes. */
(function () {
  "use strict";

  // ---- math ----
  const clamp = (v, lo = 0, hi = 1) => (v < lo ? lo : v > hi ? hi : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (t) => { t = clamp(t); return t * t * (3 - 2 * t); };
  const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
  const easeIn = (t) => Math.pow(clamp(t), 3);
  const easeInOut = (t) => { t = clamp(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  // map a sub-window [a,b] of progress to [0,1]
  const window01 = (t, a, b) => clamp((t - a) / (b - a));

  // ---- color ----
  function hexToRgb(hex) {
    let h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgba(hex, a) { const [r, g, b] = hexToRgb(hex); return `rgba(${r},${g},${b},${a})`; }
  function lerpHex(a, b, t) {
    const A = hexToRgb(a), B = hexToRgb(b);
    return `rgb(${Math.round(lerp(A[0], B[0], t))},${Math.round(lerp(A[1], B[1], t))},${Math.round(lerp(A[2], B[2], t))})`;
  }
  function lerpRgbaHex(a, b, t, alpha) {
    const A = hexToRgb(a), B = hexToRgb(b);
    return `rgba(${Math.round(lerp(A[0], B[0], t))},${Math.round(lerp(A[1], B[1], t))},${Math.round(lerp(A[2], B[2], t))},${alpha})`;
  }

  // ---- drawing primitives (additive bloom, instrument frame) ----
  // Cheap glow: 3 stacked translucent discs in ADD blend. Call inside an ADD block.
  function glow(p, x, y, r, hex, intensity = 1) {
    const [rr, gg, bb] = hexToRgb(hex);
    p.noStroke();
    p.fill(rr, gg, bb, 26 * intensity); p.circle(x, y, r * 3.2);
    p.fill(rr, gg, bb, 40 * intensity); p.circle(x, y, r * 1.8);
    p.fill(rr, gg, bb, 150 * intensity); p.circle(x, y, r);
  }

  // faint instrument grid with edge ticks; alpha scales the whole thing
  function instrumentGrid(p, w, h, hexLine, alpha = 1, step = 80) {
    p.push();
    p.strokeWeight(1);
    p.stroke(rgba(hexLine, 0.5 * alpha));
    for (let x = step; x < w; x += step) p.line(x, 0, x, h);
    for (let y = step; y < h; y += step) p.line(0, y, w, y);
    // edge ticks
    p.stroke(rgba(hexLine, 0.9 * alpha));
    const t = 6;
    for (let x = step; x < w; x += step) { p.line(x, 0, x, t); p.line(x, h, x, h - t); }
    for (let y = step; y < h; y += step) { p.line(0, y, t, y); p.line(w, y, w - t, y); }
    p.pop();
  }

  // corner brackets - instrument framing
  function cornerBrackets(p, w, h, hex, alpha = 0.6, pad = 26, len = 22) {
    p.push();
    p.noFill();
    p.stroke(rgba(hex, alpha));
    p.strokeWeight(1.25);
    const c = [[pad, pad, 1, 1], [w - pad, pad, -1, 1], [pad, h - pad, 1, -1], [w - pad, h - pad, -1, -1]];
    for (const [x, y, sx, sy] of c) { p.line(x, y, x + sx * len, y); p.line(x, y, x, y + sy * len); }
    p.pop();
  }

  // radial vignette toward near-black edges
  function vignette(p, w, h, bgHex, strength = 0.55) {
    const ctx = p.drawingContext;
    const g = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.32, w / 2, h / 2, Math.max(w, h) * 0.72);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, rgba(bgHex, strength));
    ctx.save(); ctx.fillStyle = g; ctx.fillRect(0, 0, w, h); ctx.restore();
  }

  // device-pixel-aware text helper for small-caps section labels drawn on canvas
  function label(p, txt, x, y, hex, size = 11, align) {
    p.push();
    p.fill(hex); p.noStroke();
    p.textSize(size);
    p.textFont("JetBrains Mono");
    if (align) p.textAlign(align[0], align[1]);
    p.text(txt, x, y);
    p.pop();
  }

  // mono number formatting with thousands separators
  function fmt(n) { return Math.round(n).toLocaleString("en-US"); }

  window.EBDraw = {
    clamp, lerp, smoothstep, easeOut, easeIn, easeInOut, window01,
    hexToRgb, rgba, lerpHex, lerpRgbaHex,
    glow, instrumentGrid, cornerBrackets, vignette, label, fmt,
  };
})();
