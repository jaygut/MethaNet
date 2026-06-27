#!/usr/bin/env python3
"""Headless verification: console errors, offline-safety, per-scene screenshots."""
import sys, os, time
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8848/index.html"
OUT = "/tmp/eb_shots"
os.makedirs(OUT, exist_ok=True)
SCENES = ["hero", "stakes", "blindspot", "insight", "atlas", "platform", "ladder", "path"]

console_msgs, page_errors, requests = [], [], []

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    page.on("console", lambda m: console_msgs.append((m.type, m.text)))
    page.on("pageerror", lambda e: page_errors.append(str(e)))
    page.on("request", lambda r: requests.append(r.url))

    page.goto(BASE, wait_until="networkidle", timeout=30000)
    time.sleep(1.5)

    # how many canvases got created + atlas loaded?
    info = page.evaluate("""() => ({
        canvases: document.querySelectorAll('canvas').length,
        atlasPts: (window.__dbg && window.__dbg.pts) || null,
        rail: document.querySelectorAll('.rail__item').length,
        copyFilled: document.querySelector('[data-copy=\\"insight\\"]').children.length,
        claim: document.getElementById('claimText').textContent.slice(0,40),
        fs: document.querySelectorAll('.factsheet__row').length,
    })""")
    print("PAGE INFO:", info)

    def scroll_to_scene(sid, frac=0.5):
        page.evaluate(f"""(sid) => {{
            const el = document.getElementById('scene-'+sid);
            const r = el.getBoundingClientRect();
            const top = window.scrollY + r.top;
            const dist = Math.max(0, el.offsetHeight - window.innerHeight);
            window.scrollTo(0, top + dist*{frac});
        }}""", sid)

    for sid in SCENES:
        scroll_to_scene(sid, 0.55)
        time.sleep(1.4)  # let the scene animate to mid-progress
        page.screenshot(path=f"{OUT}/{sid}.png")
        print("shot:", sid)

    # closing/ask
    page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(0.8)
    page.screenshot(path=f"{OUT}/ask.png")
    print("shot: ask")

    browser.close()

print("\n==== CONSOLE (errors/warnings) ====")
errs = [m for m in console_msgs if m[0] in ("error", "warning")]
for t, txt in errs[:40]:
    print(f"[{t}] {txt[:200]}")
print(f"... total console error/warn: {len(errs)}")

print("\n==== PAGE ERRORS ====")
for e in page_errors[:40]:
    print("ERR:", e[:200])
print(f"total page errors: {len(page_errors)}")

print("\n==== OFFLINE SAFETY: external requests ====")
ext = [u for u in requests if not (u.startswith("http://127.0.0.1") or u.startswith("http://localhost") or u.startswith("data:") or u.startswith("blob:"))]
for u in ext:
    print("EXTERNAL:", u)
print(f"total requests: {len(requests)}, external: {len(ext)}")
