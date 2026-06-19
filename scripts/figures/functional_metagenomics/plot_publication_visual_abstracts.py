#!/usr/bin/env python3
"""Render publication-style MethaNet visual abstracts.

These figures distill the existing investor/poster infographics into cleaner
visual abstracts suitable for manuscript, preprint, or partner-facing science
communication. The content deliberately preserves MethaNet claim boundaries:
current outputs support MAG/proteome molecular screening and candidate
attestation, not final sample-level MRV scoring or carbon-credit approval.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


FIG_W, FIG_H = 12.0, 8.0

LIGHT = {
    "bg": "#f7faf9",
    "ink": "#102033",
    "muted": "#52616d",
    "rule": "#d6e3e3",
    "panel": "#ffffff",
    "teal": "#007c78",
    "teal2": "#1db7b4",
    "blue": "#2b6cb0",
    "green": "#57a55a",
    "gold": "#d7a629",
    "orange": "#d87c2a",
    "red": "#c84c45",
    "purple": "#7657b8",
    "shadow": "#c7d5d4",
}

DARK = {
    "bg": "#06191d",
    "ink": "#f7fbfb",
    "muted": "#bdd0d3",
    "rule": "#16454d",
    "panel": "#09282e",
    "panel2": "#0d343b",
    "teal": "#41dfe7",
    "teal2": "#2b9ebb",
    "blue": "#68a4ff",
    "green": "#8fcf6a",
    "gold": "#ffd766",
    "orange": "#f0a72e",
    "red": "#f17062",
    "purple": "#b091ff",
    "shadow": "#000000",
}


def register_font() -> None:
    for font_path in (
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


def setup_ax(bg: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    return fig, ax


def save_all(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"{stem}.{ext}",
            dpi=300,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


def add_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    color: str = "#102033",
    weight: str = "regular",
    ha: str = "left",
    va: str = "top",
    width: int | None = None,
    linespacing: float = 1.18,
    style: str = "normal",
):
    if width:
        text = "\n".join(textwrap.wrap(text, width=width, break_long_words=False))
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
        fontstyle=style,
    )


def box(ax, x, y, w, h, *, fc, ec, lw=1.2, r=0.12, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color, lw=1.8, ms=14, curve=0.0, alpha=1.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
        alpha=alpha,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arr)
    return arr


def soft_arrow(ax, start, end, *, color, curve=0.0, lw=1.6):
    arrow(ax, start, end, color=color, lw=lw + 2.4, curve=curve, alpha=0.12, ms=18)
    arrow(ax, start, end, color=color, lw=lw, curve=curve, alpha=0.95, ms=15)


def node(ax, x, y, label, sub, color, *, dark=False, r=0.34):
    ax.add_patch(Circle((x, y), r + 0.08, facecolor=color, edgecolor="none", alpha=0.12))
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="#ffffff", linewidth=1.2))
    label_text = label.replace("Source file", "Source\nfile")
    label_size = 7.0 if len(label) > 8 else 8.2
    add_text(ax, x, y + 0.08, label_text, size=label_size, color="#ffffff", weight="bold", ha="center", va="center")
    add_text(ax, x, y - 0.13, sub, size=6.2, color="#ffffff", ha="center", va="center")


def mini_network(ax, x, y, scale, color, dark=False):
    pts = [
        (0.0, 0.0),
        (0.42, 0.22),
        (0.72, -0.05),
        (1.02, 0.3),
        (1.25, -0.22),
        (0.47, -0.34),
    ]
    lines = [(0, 1), (1, 2), (2, 3), (2, 4), (1, 5), (5, 2)]
    for a, b in lines:
        ax.plot(
            [x + pts[a][0] * scale, x + pts[b][0] * scale],
            [y + pts[a][1] * scale, y + pts[b][1] * scale],
            color=color,
            linewidth=1.0,
            alpha=0.8,
        )
    for px, py in pts:
        ax.add_patch(Circle((x + px * scale, y + py * scale), 0.055 * scale, fc=color, ec="#ffffff", lw=0.4))


def draw_header(ax, palette, title, subtitle, *, dark=False):
    add_text(ax, 0.45, 7.55, "MethaNet", size=22, color=palette["ink"], weight="bold")
    ax.plot([2.18, 2.18], [7.17, 7.72], color=palette["rule"], linewidth=1.0)
    add_text(ax, 2.48, 7.58, title, size=18.5, color=palette["ink"], weight="bold")
    add_text(ax, 2.48, 7.22, subtitle, size=10.6, color=palette["teal"], weight="bold", width=98)
    mini_network(ax, 10.55, 7.43, 0.48, palette["teal"], dark=dark)


def draw_claim_boundary(ax, palette, y, *, dark=False):
    fc = "#fdf7e6" if not dark else "#1e2a21"
    box(ax, 0.55, y, 10.9, 0.54, fc=fc, ec=palette["gold"], lw=1.0, r=0.12)
    add_text(ax, 0.78, y + 0.35, "Claim boundary", size=9.5, color=palette["gold"], weight="bold", va="center")
    add_text(
        ax,
        2.58,
        y + 0.35,
        "MAG/proteome molecular screening now; sample-level risk needs abundance, environmental context, uncertainty, and flux/process validation.",
        size=8.6,
        color=palette["ink"] if not dark else palette["muted"],
        va="center",
        width=108,
    )


def visual_abstract_layer(out_dir: Path) -> None:
    fig, ax = setup_ax(LIGHT["bg"])
    p = LIGHT
    draw_header(
        ax,
        p,
        "Molecular attestation layer",
        "A graph that connects candidates to evidence, provenance, claim status, and validation paths",
    )

    add_text(ax, 0.65, 6.93, "Multi-source molecular signals", size=12.4, color=p["ink"], weight="bold")
    inputs = [
        ("ESM-2 geometry", "662 proteomes; bridge-neighbor signal", p["blue"]),
        ("gLM2 context", "gene order and neighborhood features", p["purple"]),
        ("Functional atlas", "methane, sulfur, substrate annotations", p["green"]),
        ("QC + provenance", "taxonomy, completeness, files, checksums", p["gold"]),
    ]
    for i, (title, detail, color) in enumerate(inputs):
        y = 5.88 - i * 0.83
        box(ax, 0.55, y, 2.55, 0.58, fc=p["panel"], ec=color, lw=1.25, r=0.08)
        add_text(ax, 0.76, y + 0.39, title, size=10.5, color=color, weight="bold", va="center")
        add_text(ax, 0.76, y + 0.16, detail, size=8.2, color=p["muted"], va="center", width=38)
        soft_arrow(ax, (3.1, y + 0.29), (4.52, 4.62), color=color, curve=0.18 - i * 0.1, lw=1.2)

    box(ax, 4.15, 2.12, 3.55, 4.42, fc="#ffffff", ec=p["rule"], lw=1.1, r=0.14)
    add_text(ax, 4.45, 6.16, "Attestation graph", size=14.5, color=p["ink"], weight="bold")
    add_text(ax, 4.45, 5.87, "Evidence as connected objects", size=9.5, color=p["muted"])
    coords = {
        "Candidate": (5.05, 4.83, p["green"], "proteome_id"),
        "Feature": (6.03, 5.55, p["teal2"], "signal"),
        "Evidence": (6.92, 4.7, p["gold"], "atom"),
        "Source file": (5.08, 3.55, p["blue"], "artifact"),
        "Claim": (6.12, 3.68, p["orange"], "allowed"),
        "Blocker": (6.98, 3.35, p["red"], "gap"),
        "Upgrade": (6.95, 5.55, p["purple"], "validation"),
    }
    edges = [
        ("Candidate", "Feature"),
        ("Feature", "Evidence"),
        ("Evidence", "Claim"),
        ("Source file", "Evidence"),
        ("Claim", "Blocker"),
        ("Blocker", "Upgrade"),
        ("Upgrade", "Evidence"),
        ("Candidate", "Source file"),
    ]
    for a, b in edges:
        xa, ya = coords[a][0], coords[a][1]
        xb, yb = coords[b][0], coords[b][1]
        ax.plot([xa, xb], [ya, yb], color="#9eb8ba", linewidth=1.2, alpha=0.78)
    for label, (x, y, color, sub) in coords.items():
        node(ax, x, y, label, sub, color, r=0.33)

    box(ax, 4.45, 2.46, 2.95, 0.54, fc="#f5faf9", ec=p["teal"], lw=0.9, r=0.08)
    add_text(ax, 5.93, 2.84, "candidate -> evidence -> source file", size=7.4, color=p["teal"], ha="center", va="center")
    add_text(ax, 5.93, 2.66, "claim -> gap -> validation", size=7.4, color=p["teal"], ha="center", va="center")

    add_text(ax, 8.2, 6.74, "Claim-aware outputs", size=13, color=p["ink"], weight="bold")
    outputs = [
        ("Why it matters", "candidate + function + context", p["green"]),
        ("Evidence supports it", "geometry + annotation + QC", p["teal2"]),
        ("File produced it", "manifest path + checksum", p["gold"]),
        ("Claim allowed", "attestation, not final MRV", p["orange"]),
        ("Claim blocked", "abundance / environment / validation missing", p["red"]),
        ("Upgrade action", "sample mapping + coverage + field data", p["purple"]),
    ]
    for i, (title, detail, color) in enumerate(outputs):
        y = 6.04 - i * 0.61
        box(ax, 8.0, y, 3.48, 0.42, fc=p["panel"], ec=color, lw=1.1, r=0.08)
        add_text(ax, 8.18, y + 0.27, title, size=8.0, color=color, weight="bold", va="center", width=18)
        add_text(ax, 9.72, y + 0.27, detail, size=7.2, color=p["muted"], va="center", width=28)
    soft_arrow(ax, (7.72, 4.35), (8.03, 4.35), color=p["teal"], lw=1.5)

    draw_claim_boundary(ax, p, 1.16)
    save_all(fig, out_dir, "molecular_attestation_layer_visual_abstract_light")


def visual_abstract_graph(out_dir: Path) -> None:
    fig, ax = setup_ax(DARK["bg"])
    p = DARK
    draw_header(
        ax,
        p,
        "Graph explains candidates",
        "From candidate rank to evidence, source file, allowed claim, blocked claim, and validation upgrade",
        dark=True,
    )

    box(ax, 0.6, 5.0, 3.0, 1.55, fc=p["panel"], ec=p["rule"], lw=1.0, r=0.12)
    add_text(ax, 0.88, 6.27, "Monolithic table", size=13, color=p["ink"], weight="bold")
    add_text(ax, 0.88, 6.02, "Ranks candidates, but hides the why.", size=9, color=p["muted"])
    rows = [("#1", "rumen...bin.23", "0.97"), ("#2", "mucc...5465.1", "0.94"), ("#3", "wetland...041", "0.91")]
    for i, row in enumerate(rows):
        y = 5.68 - i * 0.28
        ax.plot([0.85, 3.35], [y - 0.09, y - 0.09], color=p["rule"], lw=0.7)
        add_text(ax, 0.88, y, row[0], size=8.4, color=p["ink"], va="center")
        add_text(ax, 1.38, y, row[1], size=8.1, color=p["muted"], va="center")
        add_text(ax, 3.05, y, row[2], size=8.1, color=p["ink"], va="center")
    add_text(ax, 0.88, 5.07, "Prioritization only", size=8.2, color=p["red"], weight="bold")

    for yy in [5.82, 5.55, 5.28]:
        soft_arrow(ax, (3.78, yy), (4.47, 4.84), color=p["teal"], curve=0.07, lw=1.0)

    add_text(ax, 4.55, 6.48, "Attestation graph", size=15.5, color=p["ink"], weight="bold")
    add_text(ax, 4.55, 6.14, "Auditable chain of molecular evidence", size=9.8, color=p["muted"])
    card_data = [
        ("Candidate", "proteome_id retained\nas cohort key", p["green"]),
        ("Evidence atom", "protein family match\nconfidence 0.96", p["teal"]),
        ("Source file", "manifest + path\nchecksum lineage", p["orange"]),
        ("Claim allowed", "molecular attestation\nnot final MRV", p["gold"]),
        ("Claim blocked", "no abundance\nno flux validation", p["red"]),
        ("Upgrade", "sample mapping\ncoverage + field data", p["purple"]),
    ]
    x0 = 4.28
    for i, (title, detail, color) in enumerate(card_data):
        x = x0 + i * 1.23
        box(ax, x, 4.25, 1.05, 1.24, fc=p["panel2"], ec=color, lw=1.1, r=0.09)
        add_text(ax, x + 0.52, 5.27, title, size=8.6, color=color, weight="bold", ha="center", va="center")
        add_text(ax, x + 0.52, 4.75, detail, size=7.2, color=p["ink"], ha="center", va="center", width=17)
        if i < len(card_data) - 1:
            soft_arrow(ax, (x + 1.05, 4.85), (x + 1.22, 4.85), color=p["teal"], lw=1.0)

    box(ax, 0.65, 2.35, 10.65, 1.25, fc="#071f24", ec=p["gold"], lw=1.0, r=0.12)
    add_text(
        ax,
        1.0,
        3.27,
        "A monolithic table can rank candidates; the graph explains why a candidate matters, what evidence supports it, which file produced that evidence, what claim is allowed, what claim is blocked, and what validation would upgrade it.",
        size=12.8,
        color=p["ink"],
        weight="bold",
        width=86,
    )

    box(ax, 0.65, 1.15, 10.65, 0.78, fc="#13251b", ec=p["gold"], lw=1.0, r=0.12)
    add_text(ax, 0.95, 1.64, "Claim boundary", size=10.5, color=p["gold"], weight="bold", va="center")
    add_text(
        ax,
        2.6,
        1.64,
        "Molecular attestation and MRV feature readiness; not final flux, risk tier, or credit approval.",
        size=9.0,
        color=p["muted"],
        va="center",
    )
    save_all(fig, out_dir, "attestation_graph_visual_abstract_dark")


def visual_abstract_workflow(out_dir: Path) -> None:
    fig, ax = setup_ax(LIGHT["bg"])
    p = LIGHT
    draw_header(
        ax,
        p,
        "Agentic workflow to molecular intelligence",
        "A reproducible sprint converts raw molecular assets into claim-aware evidence products",
    )
    stages = [
        ("1", "Frame", "bridge hypothesis\nclaim boundaries", p["blue"]),
        ("2", "Compute", "isolated runtimes\nversioned tools", p["teal"]),
        ("3", "Assemble", "validated DBs\nMAG/proteome keys", p["green"]),
        ("4", "Annotate", "QC + taxonomy\nmethane/sulfur/substrate", p["orange"]),
        ("5", "Attest", "evidence graph\nclaims + provenance", p["purple"]),
        ("6", "Deploy", "candidate cards\nfeature tables", p["gold"]),
    ]
    y = 4.68
    for i, (num, title, detail, color) in enumerate(stages):
        x = 0.7 + i * 1.78
        box(ax, x, y, 1.38, 1.38, fc=p["panel"], ec=color, lw=1.2, r=0.1)
        ax.add_patch(Circle((x + 0.23, y + 1.1), 0.17, fc=color, ec="#ffffff", lw=1.0))
        add_text(ax, x + 0.23, y + 1.1, num, size=9.5, color="#ffffff", weight="bold", ha="center", va="center")
        add_text(ax, x + 0.18, y + 0.83, title, size=11.2, color=p["ink"], weight="bold")
        add_text(ax, x + 0.18, y + 0.5, detail, size=7.2, color=p["muted"], width=17)
        if i < len(stages) - 1:
            soft_arrow(ax, (x + 1.39, y + 0.61), (x + 1.73, y + 0.61), color=p["teal"], lw=1.0)

    box(ax, 0.9, 2.48, 4.82, 1.32, fc="#eef8f4", ec=p["green"], lw=1.1, r=0.12)
    add_text(ax, 1.2, 3.48, "What changed", size=13, color=p["ink"], weight="bold")
    add_text(
        ax,
        1.2,
        3.12,
        "The output is no longer a static candidate list. It is a molecular evidence fabric with explicit provenance, missingness, claim status, and validation paths.",
        size=9.8,
        color=p["muted"],
        width=58,
    )
    box(ax, 6.23, 2.48, 4.82, 1.32, fc="#fff7e6", ec=p["gold"], lw=1.1, r=0.12)
    add_text(ax, 6.55, 3.48, "Why it matters", size=13, color=p["ink"], weight="bold")
    add_text(
        ax,
        6.55,
        3.12,
        "Partners can see which biological signals are ready for follow-up, which claims are blocked, and which data would most increase confidence.",
        size=9.8,
        color=p["muted"],
        width=56,
    )

    draw_claim_boundary(ax, p, 1.16)
    save_all(fig, out_dir, "agentic_workflow_visual_abstract_light")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/"
            "infographics/methanet_publication_visual_abstracts_20260617"
        ),
        help="Directory for PNG/PDF/SVG visual abstract outputs.",
    )
    args = parser.parse_args()
    register_font()
    visual_abstract_layer(args.out_dir)
    visual_abstract_graph(args.out_dir)
    visual_abstract_workflow(args.out_dir)


if __name__ == "__main__":
    main()
