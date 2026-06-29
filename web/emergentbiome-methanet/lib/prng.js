/* Seeded, reproducible randomness. Every sketch derives its own stream from a
   fixed seed so visuals are identical on every load and across machines. */
(function () {
  "use strict";

  // xmur3 string hash -> 32-bit seed
  function xmur3(str) {
    let h = 1779033703 ^ str.length;
    for (let i = 0; i < str.length; i++) {
      h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
      h = (h << 13) | (h >>> 19);
    }
    return function () {
      h = Math.imul(h ^ (h >>> 16), 2246822507);
      h = Math.imul(h ^ (h >>> 13), 3266489909);
      h ^= h >>> 16;
      return h >>> 0;
    };
  }

  // mulberry32 PRNG
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Reproducible RNG with convenience helpers. Seed by number or string label.
  function RNG(seed) {
    const base = typeof seed === "string" ? xmur3(seed)() : (seed >>> 0);
    const rand = mulberry32(base);
    let spare = null;
    return {
      next: rand,
      range: (lo, hi) => lo + (hi - lo) * rand(),
      int: (lo, hi) => Math.floor(lo + (hi - lo + 1) * rand()),
      pick: (arr) => arr[Math.floor(rand() * arr.length)],
      // Box–Muller gaussian
      gauss: function (mean = 0, sd = 1) {
        if (spare !== null) { const v = spare; spare = null; return mean + sd * v; }
        let u = 0, v = 0, s = 0;
        do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
        const m = Math.sqrt(-2 * Math.log(s) / s);
        spare = v * m;
        return mean + sd * (u * m);
      },
    };
  }

  window.EBRandom = { RNG, xmur3, mulberry32 };
})();
