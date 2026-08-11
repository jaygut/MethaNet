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
from selenium.webdriver.support.ui import WebDriverWait


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
          reportHref: document.getElementById('reportCta').getAttribute('href'),
        };
        """
    )
    driver.save_screenshot(str(screenshot))
    result["horizontalOverflow"] = result["scrollWidth"] > result["innerWidth"] + 1
    return result


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

    for label, view in (("landing.desktop", desktop), ("landing.mobile", mobile)):
        if view["horizontalOverflow"]:
            failures.append(f"{label}: horizontal overflow")
        if not view["noindex"]:
            failures.append(f"{label}: noindex missing")
        if view["lensButtons"] != 3 or view["pressedLensButtons"] != 1:
            failures.append(f"{label}: accessible lens controls invalid")
        if not view["claimText"] or view["claimTextHeight"] > view["claimHeight"] + 1:
            failures.append(f"{label}: claim boundary clipped or empty")

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
