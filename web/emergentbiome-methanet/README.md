# EmergentBiome / MethaNet — investor-demo scrollytelling landing page

A self-contained, fully-offline scrollytelling artifact for **EmergentBiome** (the
molecular-intelligence platform) featuring **MethaNet** (its flagship methane-risk
application for blue carbon). The visual backbone is seeded, reproducible algorithmic
art (p5.js, instance mode); **every scene's generative visual encodes a real part of
the science**, not decoration.

> **Read first:** [`DIGEST.md`](DIGEST.md) — the verified ground truth (numbers, snapshot
> dates, claim boundaries) every pixel traces back to. **Honesty is a feature of this page.**

---

## What it is

Seven scroll-driven scenes plus a title intro and a closing ask:

| # | Scene | What the visual encodes | Data |
| --- | --- | --- | --- |
| — | Hero intro | particles condensing into a node — "molecular dark matter → verifiable structure" | brand |
| 1 | The Stakes | CO₂ settling vs CH₄ (×30 GWP) eroding a net-benefit meter | illustrative (×30 real) |
| 2 | The Blind Spot | a vast grey unknown-genome field; present paired-flux + molecular coverage reads as approximately zero at scale | illustrative gap |
| 3 | The Insight **(HERO)** | the **real diffusion-map manifold** of ESM2 embeddings; 372 real bridge edges illuminate reviewable hypotheses | **real coordinates** |
| 4 | The Atlas | mangrove particles pour in; counter 625 → 2,360; 2,508 MUCC wetland genomes enter the warehouse scaffold | real counts |
| 5 | Platform & Moat | EmergentBiome substrate + MethaNet lit; the attestation graph traces one claim | real schema |
| 6 | The Honest Ladder | 6-rung MRV maturity gauge; rung 0 lit, rung 5 the dimmed target | real roadmap |
| 7 | The Path & Ask | Cispatá Bay pin; agentic pipeline (<4 days); milestone timeline | real path |

A persistent footer and Scene 6 state the molecular-screening claim boundary with the
snapshot date, on every screen.

## How to serve

The page loads `data/atlas.json` via `fetch`, so it must be served over HTTP (not
opened as a `file://` URL). No build step, no network, no CDN.

```bash
cd web/emergentbiome-methanet
python3 -m http.server 8848
# open http://localhost:8848/index.html
```

Any static file server works (`npx serve`, `caddy file-server`, nginx, etc.). All
assets — p5.js and the three webfont families — are bundled under `vendor/` and load
locally; the page works with the network cable unplugged.

## How to update the numbers

**`config.js` is the single source of truth.** All headline numbers, the snapshot date,
brand tokens, scene copy, the maturity ladder, the attestation chain, and the milestone
timeline live there. To refresh after a new MethaNet run:

1. Re-verify against the repo (see `DIGEST.md` for which artifact backs each number).
2. Edit the relevant fields in `config.js` (e.g. `num.futianArchaeaComplete`,
   `num.triViewReady`). Nothing else needs to change — copy and chrome are injected
   from `config.js` at load.
3. To refresh the hero's atlas data, re-run the seeded export (reads the latest
   freeze-backed niche projection, writes `data/atlas.json`):

   ```bash
   python3 tools/export_atlas.py          # deterministic; same source → identical output
   python3 tools/export_atlas.py --check  # summarize without writing
   ```

   Point `SOURCE_NICHE` in `tools/export_atlas.py` at a newer report freeze to advance
   the manifold.

To re-verify the page end-to-end (console errors, offline-safety, per-scene
screenshots), run `python3 tools/verify_page.py` with a server running (needs Playwright
+ Chromium).

## File map

```
index.html              semantic scaffold (header lockup, 7 sticky stages, closing, claim strip)
styles.css              "Deep Field" brand system (tokens, scrolly layout, responsive, reduced-motion)
config.js               SINGLE SOURCE OF TRUTH — numbers, snapshot date, brand tokens, copy
main.js                 orchestration: copy injection, scroll progress, scene lifecycle, a11y
lib/prng.js             seeded PRNG (mulberry32 / xmur3) — reproducible randomness
lib/scene-base.js       shared draw + easing + colour helpers (bloom, grid, vignette)
scenes/hero.js          title-intro field
scenes/scene1-stakes.js … scene7-path.js   one p5 instance-mode sketch per scene
data/atlas.json         Phase-1 export: 5,209 real diffusion-map points + 372 bridge edges
tools/export_atlas.py   seeded, reproducible atlas exporter (reads repo niche.json)
tools/verify_page.py    headless verification (console / offline / screenshots)
vendor/                 p5.js + Space Grotesk / Inter / JetBrains Mono (OFL), all local
DIGEST.md               verified ground truth + real-vs-stylized-per-scene summary
```

## Claim-boundary note (non-negotiable)

This page presents **MAG/proteome-level molecular screening and monitoring
prioritization**. It does **not** claim measured methane flux, final sample/project MRV
risk scores, final A–E risk tiers, source-independent rumen→wetland transfer, or
carbon-credit approval from the molecular atlas alone. **A–E risk tiers are target
product vocabulary, explicitly not yet calibrated.** Scene 6 (the honest ladder) and the
persistent footer state this plainly, with the `2026-06-26` snapshot date. The platform's
"beyond methane" framing is stated as design intent (the future-application slots are
honestly empty), not a demonstrated capability.

## Tech & accessibility

- Vanilla HTML/CSS/JS + p5.js **instance mode**; one sketch module per scene; no build step.
- **Seeded** randomness throughout (reproducible across loads/machines).
- 60fps target: only the on-screen scene loops; offscreen sketches are paused, and the
  tab pauses all sketches when hidden.
- Fully responsive (desktop hero → graceful mobile).
- `prefers-reduced-motion`: each scene renders a single static representative frame
  (no animation), still scroll-linked.
- Near-black AA-contrast palette; keyboard/scroll accessible; rail navigation buttons.
- Verified: 0 console errors, 0 page errors, **0 external network requests**.
