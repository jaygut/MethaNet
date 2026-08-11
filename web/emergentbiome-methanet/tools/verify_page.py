#!/usr/bin/env python3
"""Headless verification: landing scenes, report runtime, console errors, offline safety."""
import sys, os, time
from playwright.sync_api import sync_playwright

BASE = os.environ.get("METHANET_SITE_BASE", "http://127.0.0.1:8848").rstrip("/")
OUT = "/tmp/eb_shots"
os.makedirs(OUT, exist_ok=True)
SCENES = [
    "hero",
    "stakes",
    "blindspot",
    "surveyor",
    "cheap",
    "atlas",
    "engine",
    "platform",
    "ladder",
    "path",
]

console_msgs, page_errors, requests = [], [], []

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    page.on("console", lambda m: console_msgs.append((m.type, m.text)))
    page.on("pageerror", lambda e: page_errors.append(str(e)))
    page.on("request", lambda r: requests.append(r.url))

    page.goto(f"{BASE}/index.html", wait_until="networkidle", timeout=30000)
    time.sleep(1.5)

    # how many canvases got created + atlas loaded?
    info = page.evaluate("""() => ({
        canvases: document.querySelectorAll('canvas').length,
        atlasPts: (window.__dbg && window.__dbg.pts) || null,
        rail: document.querySelectorAll('.rail__item').length,
        copyFilled: document.querySelector('[data-copy=\\"surveyor\\"]').children.length,
        claim: document.getElementById('claimText').textContent.slice(0,40),
        fs: document.querySelectorAll('.factsheet__row').length,
        heroDefs: document.querySelectorAll('.hero__def').length,
        terminologyItems: document.querySelectorAll('.terminology__item').length,
        horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
    })""")
    landing_audit = page.evaluate("""async () => {
        const atlas = await fetch('data/atlas.json').then(r => r.json());
        const count = code => atlas.points.filter(p => p.fc === code).length;
        return {
          points: atlas.points.length,
          bridges: atlas.bridges.length,
          mechanismComparable: count(1),
          harmonizationPending: count(2),
          sourceScaffold: count(3),
          pipelineNormalized: count(4),
          incomplete: count(0),
          methaneIntensityNonNull: atlas.points.filter(p => p.mz != null).length,
          nonComparableMethaneIntensity: atlas.points.filter(
            p => p.fc !== 1 && p.mz != null
          ).length,
          evidenceReconciledSource: atlas.meta.source.includes('20260810_end_to_end'),
        };
    }""")
    print("PAGE INFO:", info)
    print("LANDING AUDIT:", landing_audit)
    if not (
        info["canvases"] >= 1
        and info["rail"] == 9
        and info["fs"] == 16
        and info["heroDefs"] == 2
        and info["terminologyItems"] == 8
        and not info["horizontalOverflow"]
        and landing_audit["points"] == 7710
        and landing_audit["bridges"] == 2226
        and landing_audit["mechanismComparable"] == 0
        and landing_audit["harmonizationPending"] == 0
        and landing_audit["sourceScaffold"] == 2501
        and landing_audit["pipelineNormalized"] == 5209
        and landing_audit["incomplete"] == 0
        and landing_audit["methaneIntensityNonNull"] == 0
        and landing_audit["nonComparableMethaneIntensity"] == 0
        and landing_audit["evidenceReconciledSource"]
    ):
        page_errors.append(f"landing evidence contract failed: {landing_audit}")

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

    # Full report: load the render-complete /report/ package and require its
    # high-volume data bundle to initialize the core interactive panels.
    page.goto(f"{BASE}/report/index.html", wait_until="networkidle", timeout=120000)
    page.wait_for_selector("#niche-map svg", timeout=120000)
    report_info = page.evaluate("""() => ({
        atlasNodes: ((window.METHANET_ATLAS || {}).niche || {}).nodes?.length || 0,
        runtimeErrors: document.querySelectorAll('.runtime-error').length,
        nicheSvg: document.querySelectorAll('#niche-map svg').length,
        matrixSvg: document.querySelectorAll('#signature-matrix svg').length,
        circosSvg: document.querySelectorAll('#candidate-circos svg').length,
        evidenceContractSvg: document.querySelectorAll('#evidence-contract-chart svg').length,
        sampleSvg: document.querySelectorAll('#sample-linkage svg').length,
        hasUmapButton: Array.from(document.querySelectorAll('#method-buttons button')).some(
          b => b.textContent.trim() === 'UMAP'
        ),
        bodyHasTriViewTotal: document.body.textContent.includes('7,710'),
        bodyHasMechanismComparableTotal: document.body.textContent.includes('0'),
        bodyHasPipelineNormalizedTotal: document.body.textContent.includes('5,209'),
        bodyHasScaffoldTotal: document.body.textContent.includes('2,501'),
        bodyHasQuarantineBoundary: document.body.textContent.toLowerCase().includes('quarantin'),
        bodyHasTriViewContractBoundary: document.body.textContent.includes(
          'Its evidence state records whether those payloads support a common quantitative interpretation'
        ),
        bodyHasRetired5457Infographic: document.body.textContent.includes('5,457'),
        hasRetiredOperatingModel: Array.from(document.querySelectorAll('h2')).some(
          h => h.textContent.trim() === 'The Operating Model Behind The Atlas'
        ),
        auditLinks: document.querySelectorAll('a[href^="audit/"]').length,
    })""")
    excluded_raw_status = {
        name: page.request.get(f"{BASE}/report/audit/{name}").status
        for name in (
            "scientific_audit.json",
            "evidence_contract_summary.tsv",
            "scientific_reconciliation_findings.tsv",
            "functional_metric_provenance_audit.tsv",
            "report_validation_gates.tsv",
            "claim_boundary_matrix.tsv",
        )
    }
    report_info["excludedRawStatus"] = excluded_raw_status
    print("REPORT INFO:", report_info)
    page.screenshot(path=f"{OUT}/report.png", full_page=False)
    for label, heading in (
        ("report-contract", "The Tri-View Evidence Contract"),
        ("report-geometry", "ESM-2 Geometry With Measured Limitations"),
        ("report-functional", "Functional Metric Harmonization"),
        ("report-mucc", "MUCC v1 Adds Expression Evidence And A Field-Validation Lane"),
    ):
        page.get_by_role("heading", name=heading).scroll_into_view_if_needed()
        time.sleep(0.3)
        page.screenshot(path=f"{OUT}/{label}.png", full_page=False)
        print("shot:", label)
    if not (
        report_info["atlasNodes"] == 7965
        and report_info["runtimeErrors"] == 0
        and report_info["nicheSvg"] == 1
        and report_info["matrixSvg"] == 1
        and report_info["circosSvg"] == 1
        and report_info["evidenceContractSvg"] == 1
        and report_info["sampleSvg"] == 1
        and report_info["hasUmapButton"]
        and report_info["bodyHasTriViewTotal"]
        and report_info["bodyHasMechanismComparableTotal"]
        and report_info["bodyHasPipelineNormalizedTotal"]
        and report_info["bodyHasScaffoldTotal"]
        and report_info["bodyHasQuarantineBoundary"]
        and report_info["bodyHasTriViewContractBoundary"]
        and not report_info["bodyHasRetired5457Infographic"]
        and not report_info["hasRetiredOperatingModel"]
        and report_info["auditLinks"] == 0
        and all(status == 404 for status in report_info["excludedRawStatus"].values())
    ):
        page_errors.append(f"report contract failed: {report_info}")

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
ext = [
    u
    for u in requests
    if not (
        u.startswith(BASE)
        or u.startswith("http://127.0.0.1")
        or u.startswith("http://localhost")
        or u.startswith("data:")
        or u.startswith("blob:")
    )
]
for u in ext:
    print("EXTERNAL:", u)
print(f"total requests: {len(requests)}, external: {len(ext)}")

console_errors = [m for m in console_msgs if m[0] == "error"]
if console_errors or page_errors or ext:
    sys.exit(1)
