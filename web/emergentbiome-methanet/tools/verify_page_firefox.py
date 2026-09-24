#!/usr/bin/env python3
"""Firefox/Selenium publication audit for the landing page and stable report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


LEDGER_KEYS = (
    "registered_units",
    "esm2_units",
    "glm2_units",
    "functional_payload_units",
    "release_required_units",
    "explicit_non_runnable_gaps",
    "tri_view_ready_units",
    "schema_normalized_units",
    "schema_normalized_tri_view_units",
    "pipeline_normalized_tri_view_units",
    "mechanism_comparable_units",
    "annotation_complete_tri_view_units",
    "source_scaffold_tri_view_units",
    "blocking_units",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8848")
    parser.add_argument("--release-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def audit_viewport(driver, url: str, width: int, height: int, screenshot: Path) -> dict:
    driver.set_window_size(width, height)
    driver.get(url)
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.ID, "claimText")))
    time.sleep(1)
    result = driver.execute_script(
        """
        return {
          innerWidth: window.innerWidth,
          scrollWidth: document.documentElement.scrollWidth,
          scrollHeight: document.documentElement.scrollHeight,
          noindex: !!document.querySelector('meta[name="robots"][content*="noindex"]'),
          canvases: document.querySelectorAll('canvas').length,
          lensButtons: document.querySelectorAll('[data-engine-lens]').length,
          pressedLensButtons: document.querySelectorAll('[data-engine-lens][aria-pressed="true"]').length,
          claimText: document.getElementById('claimText').textContent.trim(),
          claimHeight: document.querySelector('.claimbar').getBoundingClientRect().height,
          claimTextHeight: document.querySelector('.claimbar__text').getBoundingClientRect().height,
          claimTextClipped: (() => { const e = document.querySelector('.claimbar__text'); return e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1; })(),
          headerBrandClipped: (() => { const e = document.querySelector('.lockup__brand'); return e.scrollWidth > e.clientWidth + 1; })(),
          reportHref: document.getElementById('reportCta').getAttribute('href'),
        };
        """
    )
    driver.save_screenshot(str(screenshot))
    result["horizontalOverflow"] = result["scrollWidth"] > result["innerWidth"] + 1
    return result


def audit_landing_controls(driver, url: str, width: int, height: int) -> dict:
    """Exercise the proposal-facing evidence controls through the rendered UI."""
    driver.set_window_size(width, height)
    driver.get(url)
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.ID, "claimText")))

    def scroll_to_scene(scene_id: str) -> None:
        driver.execute_script(
            """
            const el = document.getElementById(arguments[0]);
            const top = window.scrollY + el.getBoundingClientRect().top;
            const travel = Math.max(0, el.offsetHeight - window.innerHeight);
            window.scrollTo(0, top + travel * 0.6);
            """,
            scene_id,
        )
        time.sleep(0.9)

    scroll_to_scene("scene-surveyor")
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-card-view='pending']"))
    )
    driver.find_element(By.CSS_SELECTOR, "[data-card-view='pending']").click()
    pending = driver.find_element(By.ID, "candidateEvidencePanel").text
    driver.find_element(By.CSS_SELECTOR, "[data-card-view='next']").click()
    next_action = driver.find_element(By.ID, "candidateEvidencePanel").text

    scroll_to_scene("scene-atlas")
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.ID, "atlasPanelBody"))
    )
    view_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-atlas-view]")
    projection_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-atlas-projection]")
    initial_umap = driver.find_element(
        By.CSS_SELECTOR, "[data-atlas-projection='umap']"
    ).get_attribute("aria-pressed")
    driver.find_element(By.CSS_SELECTOR, "[data-atlas-view='candidates']").click()
    select = Select(driver.find_element(By.ID, "atlasCandidateSelect"))
    candidate_options = len(select.options)
    select.select_by_value("0")
    candidate_readout = driver.find_element(By.ID, "atlasCandidateReadout").text
    driver.find_element(By.CSS_SELECTOR, "[data-atlas-view='sensitivity']").click()
    sensitivity = driver.find_element(By.ID, "atlasPanelBody").text
    driver.find_element(By.CSS_SELECTOR, "[data-atlas-projection='diffusion']").click()
    diffusion_pressed = driver.find_element(
        By.CSS_SELECTOR, "[data-atlas-projection='diffusion']"
    ).get_attribute("aria-pressed")
    driver.find_element(By.CSS_SELECTOR, "[data-atlas-projection='tsne']").click()
    tsne_pressed = driver.find_element(
        By.CSS_SELECTOR, "[data-atlas-projection='tsne']"
    ).get_attribute("aria-pressed")
    tsne_announcement = driver.find_element(By.ID, "atlasPanelAnnounce").get_attribute("textContent")

    scroll_to_scene("scene-engine")
    driver.find_element(By.CSS_SELECTOR, "[data-engine-lens='1']").click()
    engine_pressed = driver.find_element(
        By.CSS_SELECTOR, "[data-engine-lens='1']"
    ).get_attribute("aria-pressed")
    original_window = driver.current_window_handle
    driver.find_element(By.ID, "headerReportCta").click()
    WebDriverWait(driver, 30).until(lambda d: len(d.window_handles) == 2)
    driver.switch_to.window(next(handle for handle in driver.window_handles if handle != original_window))
    WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#niche-map svg")))
    report_link_opens = driver.current_url.rstrip("/").endswith("/report")
    driver.close()
    driver.switch_to.window(original_window)
    return {
        "viewButtons": len(view_buttons),
        "projectionButtons": len(projection_buttons),
        "candidateOptions": candidate_options,
        "initialUmapPressed": initial_umap == "true",
        "diffusionPressed": diffusion_pressed == "true",
        "tsnePressed": tsne_pressed == "true",
        "tsneAnnounced": "TSNE projection" in tsne_announcement,
        "reportLinkOpens": report_link_opens,
        "enginePressed": engine_pressed == "true",
        "pendingShowsExactJoinGap": "exact sample" in pending.lower(),
        "nextShowsFieldPairing": "methane-process measurement" in next_action.lower(),
        "candidateShowsOneWaySimilarity": (
            "Raw cosine similarity" in candidate_readout
            and "not validated transfer" in candidate_readout
        ),
        "sensitivityShowsBothMethods": (
            "15,728" in sensitivity
            and "15,064" in sensitivity
            and "Rumen ↔ wetland" in sensitivity
            and "Standardized" in sensitivity
        ),
        "horizontalOverflow": driver.execute_script(
            "return document.documentElement.scrollWidth > window.innerWidth + 1"
        ),
    }


def audit_report(driver, url: str, ledger: dict, width: int, height: int, screenshot: Path) -> dict:
    driver.set_window_size(width, height)
    driver.get(url)
    WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#niche-map svg")))
    time.sleep(1)
    result = driver.execute_script(
        """
        const fallbacks = [...document.querySelectorAll('details.fallback img')];
        const svgs = [...document.querySelectorAll('.viz svg')];
        return {
          innerWidth: window.innerWidth,
          scrollWidth: document.documentElement.scrollWidth,
          noindex: !!document.querySelector('meta[name="robots"][content*="noindex"]'),
          runtimeErrors: document.querySelectorAll('.runtime-error').length,
          svgCount: svgs.length,
          labelledSvgs: svgs.filter(svg => svg.getAttribute('role') === 'img' && svg.getAttribute('aria-label')).length,
          fallbacks: fallbacks.length,
          loadedFallbacks: fallbacks.filter(img => img.complete && img.naturalWidth > 0).length,
          keyboardCells: document.querySelectorAll('rect.cell[tabindex="0"][role="button"]').length,
          keyboardCandidates: document.querySelectorAll('circle[tabindex="0"][role="button"]').length,
          liveCandidateCard: document.getElementById('candidate-card')?.getAttribute('aria-live') || '',
          bodyHasAbsoluteRepoPath: document.body.textContent.includes('/home/'),
        };
        """
    )
    visible = driver.execute_script(
        """
        const keys = arguments[0];
        return Object.fromEntries(keys.map(key => {
          const node = document.querySelector(`[data-release-key="${key}"]`);
          return [key, node ? node.textContent.trim().replaceAll(',', '') : null];
        }));
        """,
        [*LEDGER_KEYS, "snapshot_date"],
    )
    expected = {key: str(ledger[key]) for key in LEDGER_KEYS}
    expected["snapshot_date"] = str(ledger["snapshot_date"])
    result["visibleReleaseValues"] = visible
    result["visibleReleaseExpected"] = expected
    result["visibleReleaseParity"] = visible == expected
    result["horizontalOverflow"] = result["scrollWidth"] > result["innerWidth"] + 1
    driver.save_screenshot(str(screenshot))
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(args.release_ledger.read_text())
    options = Options()
    options.add_argument("-headless")
    driver = webdriver.Firefox(options=options)
    failures: list[str] = []
    try:
        desktop = audit_viewport(
            driver,
            f"{args.base.rstrip('/')}/index.html",
            1440,
            900,
            args.output_dir / "landing_desktop.png",
        )
        mobile = audit_viewport(
            driver,
            f"{args.base.rstrip('/')}/index.html",
            450,
            844,
            args.output_dir / "landing_mobile.png",
        )
        compact_tablet = audit_viewport(
            driver,
            f"{args.base.rstrip('/')}/index.html",
            900,
            600,
            args.output_dir / "landing_compact_tablet.png",
        )
        landscape_phone = audit_viewport(
            driver,
            f"{args.base.rstrip('/')}/index.html",
            650,
            450,
            args.output_dir / "landing_landscape_phone.png",
        )
        landing_controls = {
            "desktop": audit_landing_controls(
                driver, f"{args.base.rstrip('/')}/index.html", 1440, 900
            ),
            "mobile": audit_landing_controls(
                driver, f"{args.base.rstrip('/')}/index.html", 450, 844
            ),
        }
        report = audit_report(
            driver,
            f"{args.base.rstrip('/')}/report/",
            ledger,
            1440,
            900,
            args.output_dir / "report_desktop.png",
        )
        report_tablet = audit_report(
            driver,
            f"{args.base.rstrip('/')}/report/",
            ledger,
            820,
            1180,
            args.output_dir / "report_tablet.png",
        )
        report_mobile = audit_report(
            driver,
            f"{args.base.rstrip('/')}/report/",
            ledger,
            450,
            844,
            args.output_dir / "report_mobile.png",
        )
    finally:
        driver.quit()

    for label, view in (
        ("landing.desktop", desktop),
        ("landing.mobile", mobile),
        ("landing.compact_tablet", compact_tablet),
        ("landing.landscape_phone", landscape_phone),
    ):
        if view["horizontalOverflow"]:
            failures.append(f"{label}: horizontal overflow")
        if not view["noindex"]:
            failures.append(f"{label}: noindex missing")
        if view["lensButtons"] != 3 or view["pressedLensButtons"] != 1:
            failures.append(f"{label}: accessible lens controls invalid")
        if not view["claimText"] or view["claimTextHeight"] > view["claimHeight"] + 1 or view["claimTextClipped"]:
            failures.append(f"{label}: claim boundary clipped or empty")
        if view["headerBrandClipped"]:
            failures.append(f"{label}: header brand clipped")
    for label, controls in landing_controls.items():
        required = {
            "viewButtons": 4,
            "projectionButtons": 4,
            "candidateOptions": 27,
            "initialUmapPressed": True,
            "diffusionPressed": True,
            "tsnePressed": True,
            "tsneAnnounced": True,
            "reportLinkOpens": True,
            "enginePressed": True,
            "pendingShowsExactJoinGap": True,
            "nextShowsFieldPairing": True,
            "candidateShowsOneWaySimilarity": True,
            "sensitivityShowsBothMethods": True,
            "horizontalOverflow": False,
        }
        for key, expected in required.items():
            if controls[key] != expected:
                failures.append(f"landing.{label}: {key} = {controls[key]!r}, expected {expected!r}")

    for label, view in (
        ("report.desktop", report),
        ("report.tablet", report_tablet),
        ("report.mobile", report_mobile),
    ):
        if view["horizontalOverflow"]:
            failures.append(f"{label}: horizontal overflow")
        if not view["noindex"]:
            failures.append(f"{label}: noindex missing")
        if view["runtimeErrors"]:
            failures.append(f"{label}: runtime error panel present")
    if report["svgCount"] < 5 or report["labelledSvgs"] != report["svgCount"]:
        failures.append("report.desktop: interactive SVGs missing accessible labels")
    if report["fallbacks"] < 3 or report["fallbacks"] != report["loadedFallbacks"]:
        failures.append("report.desktop: static fallbacks missing or unloaded")
    if not report["keyboardCells"] or not report["keyboardCandidates"]:
        failures.append("report.desktop: keyboard-operable data marks missing")
    if report["liveCandidateCard"] != "polite":
        failures.append("report.desktop: candidate card live-region missing")
    if report["bodyHasAbsoluteRepoPath"]:
        failures.append("report.desktop: absolute repository path exposed")
    if not report["visibleReleaseParity"]:
        failures.append("report.desktop: visible release values drift from ledger")

    result = {
        "base": args.base,
        "release_snapshot": ledger["snapshot_date"],
        "landing_desktop": desktop,
        "landing_mobile": mobile,
        "landing_compact_tablet": compact_tablet,
        "landing_landscape_phone": landscape_phone,
        "landing_controls": landing_controls,
        "report_desktop": report,
        "report_tablet": report_tablet,
        "report_mobile": report_mobile,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    (args.output_dir / "browser_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
