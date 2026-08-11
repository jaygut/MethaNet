#!/usr/bin/env python3
"""Build a professional MBAG-2 framework report and infographic.

The output is deterministic and intentionally text-first. It is meant to
replace disposable generated infographic drafts with an editable, auditable
artifact that keeps MethaNet's evidence grains and claim boundaries visible.
"""

from __future__ import annotations

import csv
import html
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results/reports/mbag2_methane_risk_intelligence_framework_20260630"
FREEZE_SUMMARY = REPO_ROOT / "results/reports/methanet_3view_payload_freeze_20260629_195927/freeze_summary.tsv"


PALETTE = {
    "ink": "#112432",
    "muted": "#5b6b74",
    "line": "#c9d4d8",
    "bg": "#f7faf9",
    "card": "#ffffff",
    "teal": "#087f7b",
    "green": "#598a3d",
    "gold": "#c68418",
    "blue": "#2367a2",
    "red": "#b43b33",
    "purple": "#66538d",
    "gray": "#6d7880",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def lane_rows() -> list[dict[str, str]]:
    rows = read_tsv(FREEZE_SUMMARY)
    if rows:
        return rows
    return [
        {
            "lane_id": "poc_core",
            "expected_units": "625",
            "esm2_units": "625",
            "glm2_units": "625",
            "functional_complete": "625",
            "functional_not_started": "0",
            "tri_view_ready_units": "625",
            "warehouse_current": "True",
        }
    ]


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def text_lines(text: str, width: int = 28) -> list[str]:
    out: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            out.append("")
        else:
            out.extend(wrap(raw, width=width, break_long_words=False))
    return out


def svg_text(x: float, y: float, text: str, *, size: int = 20, weight: int = 500, fill: str | None = None, anchor: str = "start") -> str:
    fill = fill or PALETTE["ink"]
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>'


def multiline(x: float, y: float, text: str, *, width: int = 30, size: int = 18, line_h: int = 24, weight: int = 500, fill: str | None = None) -> str:
    fill = fill or PALETTE["ink"]
    parts = []
    for i, line in enumerate(text_lines(text, width)):
        parts.append(svg_text(x, y + i * line_h, line, size=size, weight=weight, fill=fill))
    return "\n".join(parts)


def rounded_rect(x: float, y: float, w: float, h: float, *, fill: str, stroke: str | None = None, sw: float = 1.5, r: float = 10, opacity: float = 1.0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>'


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#315765", width: float = 3) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"/>'


def small_badge(x: float, y: float, label: str, color: str) -> str:
    return (
        rounded_rect(x, y, 112, 30, fill=color, r=15)
        + svg_text(x + 56, y + 21, label, size=15, weight=700, fill="#fff", anchor="middle")
    )


def source_card(x: float, y: float, title: str, meta: str, color: str, ready: str) -> str:
    return "\n".join(
        [
            rounded_rect(x, y, 282, 106, fill="#fff", stroke=color, sw=2.2, r=12),
            f'<circle cx="{x + 36}" cy="{y + 38}" r="20" fill="{color}" opacity="0.95"/>',
            svg_text(x + 70, y + 34, title, size=22, weight=800),
            multiline(x + 70, y + 60, meta, width=28, size=13, line_h=17, weight=600, fill=PALETTE["muted"]),
            rounded_rect(x + 176, y + 70, 86, 25, fill=color, r=13),
            svg_text(x + 219, y + 88, ready, size=12, weight=800, fill="#fff", anchor="middle"),
        ]
    )


def build_svg(rows: list[dict[str, str]]) -> str:
    lane_lookup = {row["lane_id"]: row for row in rows}
    tri_ready = sum(int(float(row.get("tri_view_ready_units") or 0)) for row in rows if row.get("lane_id") != "mucc_v1_owc_wetland")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    p = PALETTE
    defs = """
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,4 L0,8 Z" fill="#315765"/>
    </marker>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#17323a" flood-opacity="0.12"/>
    </filter>
  </defs>
"""
    lanes = "\n".join(
        [
            source_card(54, 168, "POC core", "625 MAG/bin tri-view; warehouse current", p["teal"], "625 ready"),
            source_card(54, 294, "MSM China", "1,427 release-ready; 1 partial preserved", p["gold"], "1,427 ready"),
            source_card(54, 420, "Futian 2026", "312 archaea ready; bacteria pending", p["green"], "312 ready"),
            source_card(54, 546, "MUCC v1 OWC", "2,508 scaffolded; ESM2/gLM2 active", p["blue"], "staged"),
        ]
    )

    evidence_blocks = [
        ("Potential", "MAG functions: MCycDB, SCycDB, KOfam, METABOLIC, CAZy, MEROPS", p["teal"]),
        ("Capacity", "Sample weights: MAG coverage, marker abundance, unbinned reads", p["gold"]),
        ("Permissiveness", "Environment: salinity, sulfate, redox, hydrology, substrate, season", p["green"]),
        ("Activity", "RNA/protein/metabolite evidence; porewater CH4 and process assays", p["blue"]),
        ("Outcome", "Flux/process validation: chamber, eddy covariance, incubation", p["red"]),
    ]
    evidence_svg = []
    for i, (title, body, color) in enumerate(evidence_blocks):
        y = 190 + i * 86
        evidence_svg.extend(
            [
                rounded_rect(424, y, 330, 64, fill="#fff", stroke=color, sw=1.8, r=10),
                svg_text(444, y + 24, title, size=18, weight=800, fill=color),
                multiline(444, y + 44, body, width=46, size=12, line_h=15, weight=550, fill=p["ink"]),
            ]
        )

    model_blocks = [
        ("Evidence graph", "MAG, gene, sample, site, source, artifact, claim and blocker nodes."),
        ("Multi-view encoder", "Separate ESM2/gLM2, methane, sulfur, substrate, broad function, QC and taxonomy views."),
        ("Bridge layer", "SNF-style fusion, OT couplings, direct-vs-propagated mechanism support."),
        ("Safety layer", "Source leakage probes, OOD splits, calibration, conformal intervals, abstention."),
    ]
    model_svg = []
    for i, (title, body) in enumerate(model_blocks):
        y = 186 + i * 105
        model_svg.extend(
            [
                rounded_rect(840, y, 360, 78, fill="#fff", stroke=p["line"], r=10),
                svg_text(862, y + 28, title, size=20, weight=800, fill=p["purple"] if i == 1 else p["ink"]),
                multiline(862, y + 52, body, width=42, size=13, line_h=17, weight=550, fill=p["muted"]),
            ]
        )

    gates = [
        ("G0", "claim context"),
        ("G1", "identity + sample link"),
        ("G2", "molecular potential"),
        ("G3", "abundance/capacity"),
        ("G4", "environment context"),
        ("G5", "activity evidence"),
        ("G6", "flux/process validation"),
        ("G7", "calibrated MRV risk"),
    ]
    gate_svg = []
    for i, (g, label) in enumerate(gates):
        y = 204 + i * 52
        fill = p["teal"] if i <= 2 else p["gray"] if i <= 4 else p["red"]
        gate_svg.extend(
            [
                f'<circle cx="1296" cy="{y}" r="18" fill="{fill}"/>',
                svg_text(1296, y + 6, g, size=13, weight=800, fill="#fff", anchor="middle"),
                svg_text(1326, y + 6, label, size=15, weight=650, fill=p["ink"]),
            ]
        )

    outputs = [
        ("Candidate cards", "ranked hypotheses with direct evidence, missingness and next action"),
        ("Readiness register", "scoreable_provisional, monitor_more, needs_metadata, needs_abundance"),
        ("Monitoring priority", "where flux chambers, geochemistry or read mapping most increase value"),
        ("Audit packet", "provenance, validation gaps, claim locks and source-aware tests"),
    ]
    output_svg = []
    for i, (title, body) in enumerate(outputs):
        y = 196 + i * 104
        output_svg.extend(
            [
                rounded_rect(1528, y, 320, 78, fill="#fff", stroke=p["line"], r=10),
                svg_text(1550, y + 29, title, size=20, weight=800, fill=p["blue"]),
                multiline(1550, y + 53, body, width=38, size=13, line_h=17, weight=550, fill=p["muted"]),
            ]
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
{defs}
  <rect width="1920" height="1080" fill="{p["bg"]}"/>
  <rect x="32" y="34" width="1856" height="1010" rx="22" fill="#fff" stroke="{p["line"]}" filter="url(#softShadow)"/>
  {svg_text(72, 88, "MBAG-2: Source-Audited Molecular Methane-Risk Intelligence", size=42, weight=850)}
  {svg_text(72, 124, "A professional redesign of MethaNet's bridge interpretation layer: from MAG/proteome evidence to defensible methane-risk readiness.", size=20, weight=520, fill=p["muted"])}
  {svg_text(1848, 88, f"Generated {now}", size=14, weight=600, fill=p["muted"], anchor="end")}
  {rounded_rect(72, 828, 424, 146, fill="#f3faf8", stroke=p["teal"], r=14)}
  {svg_text(96, 864, "Current live/frozen evidence base", size=22, weight=800, fill=p["teal"])}
  {multiline(96, 900, f"{tri_ready:,} tri-view-ready MAG/proteome units in the current June 29 release-ready atlas: POC 625 + MSM 1,427 + Futian archaea 312. Futian bacteria and MUCC v1 remain staged/pending.", width=54, size=16, line_h=23, fill=p["ink"])}
  {lanes}
  {arrow(346, 221, 395, 221)}
  {arrow(346, 347, 395, 347)}
  {arrow(346, 473, 395, 473)}
  {arrow(346, 599, 395, 599)}
  {rounded_rect(400, 140, 382, 535, fill="#f9fbfb", stroke=p["line"], r=16)}
  {svg_text(424, 172, "Evidence hierarchy", size=27, weight=850)}
  {"".join(evidence_svg)}
  {arrow(786, 412, 830, 412)}
  {rounded_rect(820, 140, 410, 535, fill="#fbfbfd", stroke=p["line"], r=16)}
  {svg_text(844, 172, "MBAG-2 engine", size=27, weight=850)}
  {"".join(model_svg)}
  {arrow(1234, 412, 1270, 412)}
  {rounded_rect(1260, 140, 228, 535, fill="#fffdfa", stroke="#d7c8a3", r=16)}
  {svg_text(1284, 172, "Claim gates", size=27, weight=850)}
  {"".join(gate_svg)}
  {rounded_rect(1284, 612, 168, 36, fill="#fff1ef", stroke=p["red"], r=18)}
  {svg_text(1368, 636, "NO final MRV tiers yet", size=15, weight=800, fill=p["red"], anchor="middle")}
  {arrow(1492, 412, 1518, 412)}
  {rounded_rect(1508, 140, 368, 535, fill="#f8fbff", stroke=p["line"], r=16)}
  {svg_text(1532, 172, "Decision products now", size=27, weight=850)}
  {"".join(output_svg)}
  {rounded_rect(548, 828, 650, 146, fill="#fffaf2", stroke=p["gold"], r=14)}
  {svg_text(572, 864, "Core design upgrade", size=22, weight=800, fill=p["gold"])}
  {multiline(572, 900, "Separate five evidence states: genomic potential, abundance-weighted capacity, environmental permissiveness, activity proxies, and validated outcomes. MBAG-2 can score readiness before it scores methane risk.", width=78, size=16, line_h=23, fill=p["ink"])}
  {rounded_rect(1238, 828, 608, 146, fill="#fff5f5", stroke=p["red"], r=14)}
  {svg_text(1262, 864, "Scientific lock", size=22, weight=800, fill=p["red"])}
  {multiline(1262, 900, "Molecular evidence can prioritize monitoring and explain bridge candidates. Final A-E risk tiers require sample mapping, MAG/read abundance, environmental covariates, uncertainty propagation, and flux/process validation.", width=72, size=16, line_h=23, fill=p["ink"])}
</svg>'''


def write_report(out_dir: Path, svg_path: Path, png_path: Path) -> tuple[Path, Path]:
    md_path = out_dir / "MBAG2_FRAMEWORK_REPORT.md"
    html_path = out_dir / "MBAG2_FRAMEWORK_REPORT.html"
    generated = datetime.now(timezone.utc).isoformat()
    md = f"""# MBAG-2: Source-Audited Molecular Methane-Risk Intelligence

Generated: `{generated}`

Main visual:

- SVG: `{svg_path}`
- PNG: `{png_path}`

## Executive Decision

The original MBAG concept should be upgraded from a bridge-ranking model into a
source-audited methane-risk intelligence layer. The near-term output is not a
final risk tier. It is a defensible readiness and monitoring-priority system that
explains which MAGs, samples, sites, and evidence gaps deserve action.

## MBAG-2 Architecture

1. Evidence graph layer: MAG/proteome, gene/function, sample, site, project,
   source study, source artifact, evidence atom, claim, and blocker nodes.
2. Multi-view molecular encoder: separate views for ESM2/gLM2 geometry,
   methane-cycle functions, sulfur/substrate competition, broad function,
   QC/taxonomy, missingness, abundance, and environment.
3. Bridge layer: reliability-weighted graph fusion, optimal-transport coupling,
   direct-vs-propagated mechanism support, and source-leakage diagnostics.
4. Risk-readiness layer: sample/project outputs such as `needs_abundance`,
   `needs_environment`, `needs_flux_validation`, `monitor_more`, and
   `scoreable_provisional`.
5. Future calibrated MRV layer: only after sample mapping, abundance, covariates,
   uncertainty propagation, and flux/process validation exist.

## Evidence Hierarchy

| Layer | Meaning | Unlocks |
| --- | --- | --- |
| Potential | QC-aware MAG functional genes and modules | candidate mechanism cards |
| Capacity | abundance/read-weighted MAG and marker support | sample molecular capacity |
| Permissiveness | salinity, sulfate, redox, hydrology, substrate, season | methane-risk context |
| Activity | transcript/protein/metabolite/process proxies | near-term process evidence |
| Outcome | chamber, eddy covariance, incubation, or paired flux/process labels | calibrated risk distributions |

## Validation Gates

| Gate | Required evidence | Failure behavior |
| --- | --- | --- |
| G0 claim context | method, project boundary, monitoring period | keep as screening only |
| G1 identity | sample/site/project/MAG links and provenance tier | `needs_metadata` |
| G2 molecular potential | QC-aware markers, pathway completeness, coverage | candidate card only |
| G3 community capacity | MAG/read coverage and marker abundance | `needs_abundance` |
| G4 environmental permissiveness | salinity, sulfate, redox, hydrology, carbon, season | `needs_environment` |
| G5 activity evidence | transcript/protein/geochemistry/process assays | `monitor_more` |
| G6 outcome validation | flux/process labels with holdout design | provisional distribution |
| G7 calibrated MRV | OOD validation, calibration, uncertainty intervals | final tiers only here |

## Statistical Discipline

- Use one graph/view per evidence family before fusion.
- Treat optimal transport as an auditable bridge hypothesis, not proof of transfer.
- Keep propagated mechanism support separate from direct marker evidence.
- Report source leakage, leave-source/site/season/habitat-out validation, and
  negative controls before risk language.
- Use calibrated probabilities and conformal intervals only after outcome labels.
- Abstain aggressively: `not_scoreable` is a valid product output.

## Current Evidence Base

- POC core: 625/625 tri-view MAG/bin units with a current warehouse.
- MSM China 2025: 1,427 release-ready tri-view units, one partial preserved.
- Futian 2026: 312 archaea tri-view units; 2,844 bacteria pending.
- MUCC v1 OWC: staged source scaffold and active ESM2/gLM2 work; not tri-view
  ready for MBAG-2 risk intelligence yet.

## Recommended Semantic-Layer Expansion

- `dim_marker`: marker family, pathway role, database, directionality caveats,
  and threshold provenance.
- `fact_marker_hit`: gene/protein/MAG-level marker support with coverage,
  e-value, identity, bitscore, and accepted/rejected status.
- `feature_pathway_completeness`: methane, sulfur, fermentation, CAZy substrate,
  and electron-acceptor pathway completeness by proteome.
- `fact_mag_abundance`: sample-level MAG/read coverage, normalized abundance,
  uncertainty, and mapping method.
- `fact_environment_context`: salinity, sulfate, redox, water table/inundation,
  temperature, organic carbon, vegetation, and season.
- `fact_activity_omics`: transcript, protein, metabolite, porewater chemistry,
  and process-assay evidence.
- `fact_flux_process_validation`: chamber, eddy-covariance, incubation, or
  paired flux/process labels with monitoring period and uncertainty.
- `feature_sample_risk_readiness`: gate status, abstention reason, evidence
  sufficiency, monitoring priority, and next validation action.

## Research Anchors

- Carbon-credit integrity and market-readiness boundaries: ICVCM Core Carbon
  Principles and Assessment Framework, Verra VM0033, and the IPCC Wetlands
  Supplement.
- Outcome evidence targets: FLUXNET-CH4 and site/project chamber or
  eddy-covariance measurements.
- Multi-view evidence fusion: Similarity Network Fusion and MOGONET-style
  view-specific encoders before fusion.
- Bridge discipline: optimal transport and domain-adversarial diagnostics as
  hypothesis generators, not proof of transfer.
- Risk quantification: source-aware OOD splits, group robustness, probability
  calibration, conformal intervals, and abstention.
- Functional evidence: MCycDB, SCycDB, METABOLIC, dbCAN3/CAZy, KOfam,
  GTDB-Tk, CheckM2, GUNC, and explicit marker-direction caveats.

## Claim Boundary

Allowed now: molecular screening, source-lane readiness, bridge-candidate
prioritization, candidate cards, validation gaps, and monitoring-priority
hypotheses.

Blocked now: measured methane flux claims, final A-E risk tiers,
source-independent transfer proof, VM0033/ICVCM approval, or carbon-crediting
claims.
"""
    md_path.write_text(md)
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>MBAG-2 Framework Report</title>
<style>
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #f7faf9; color: #112432; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 72px; }}
.hero {{ background: white; border: 1px solid #c9d4d8; border-radius: 18px; padding: 18px; box-shadow: 0 12px 34px rgba(17,36,50,.08); }}
img {{ width: 100%; height: auto; display: block; border-radius: 10px; }}
h1 {{ font-size: 34px; margin: 34px 0 8px; }}
h2 {{ margin-top: 32px; color: #087f7b; }}
p, li {{ font-size: 17px; line-height: 1.55; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; background: white; }}
th, td {{ border: 1px solid #d6e0e2; padding: 10px 12px; text-align: left; vertical-align: top; }}
th {{ background: #eff7f6; }}
code {{ background: #eef4f5; padding: 1px 5px; border-radius: 4px; }}
</style>
<main>
<section class="hero"><img src="{png_path.name}" alt="MBAG-2 framework infographic"></section>
{markdown_to_html(md)}
</main>
</html>
"""
    )
    return md_path, html_path


def markdown_to_html(md: str) -> str:
    """Tiny markdown renderer for this controlled report."""

    lines = md.splitlines()
    html_lines: list[str] = []
    in_ul = False
    in_table = False
    table_rows: list[str] = []

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        html_lines.append("<table>")
        for idx, row in enumerate(table_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if idx == 1 and all(set(cell) <= {"-", ":", " "} for cell in cells):
                continue
            tag = "th" if idx == 0 else "td"
            html_lines.append("<tr>" + "".join(f"<{tag}>{esc(cell)}</{tag}>" for cell in cells) + "</tr>")
        html_lines.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        if line.startswith("|"):
            close_ul()
            in_table = True
            table_rows.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            close_ul()
            html_lines.append(f"<h1>{esc(line[2:])}</h1>")
        elif line.startswith("## "):
            close_ul()
            html_lines.append(f"<h2>{esc(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{esc(line[2:])}</li>")
        elif line.strip():
            close_ul()
            html_lines.append(f"<p>{esc(line)}</p>")
        else:
            close_ul()
    flush_table()
    close_ul()
    return "\n".join(html_lines)


def main() -> int:
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "mbag2_framework_infographic.svg"
    png_path = out_dir / "mbag2_framework_infographic.png"
    svg_path.write_text(build_svg(lane_rows()))
    subprocess.run(["rsvg-convert", str(svg_path), "-w", "1920", "-h", "1080", "-o", str(png_path)], check=True)
    md_path, html_path = write_report(out_dir, svg_path, png_path)
    print(svg_path)
    print(png_path)
    print(md_path)
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
