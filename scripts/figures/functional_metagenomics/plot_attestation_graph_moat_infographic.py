#!/usr/bin/env python3
"""Render the MethaNet queryable attestation graph moat infographic."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


W, H = 1536, 1024

PALETTE = {
    "bg0": (5, 24, 28),
    "bg1": (4, 48, 52),
    "cyan": (69, 222, 238),
    "cyan2": (41, 151, 186),
    "green": (123, 184, 92),
    "green2": (45, 129, 72),
    "orange": (245, 174, 45),
    "gold": (246, 210, 103),
    "red": (225, 100, 80),
    "purple": (140, 96, 230),
    "ink": (246, 250, 247),
    "muted": (192, 214, 211),
    "panel": (9, 42, 48),
    "panel2": (17, 62, 68),
    "white_panel": (240, 244, 241),
    "black_text": (24, 35, 43),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in [
        Path("/usr/share/fonts/dejavu-sans-fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ]:
        p = root / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


FONTS = {
    "brand": font(45, True),
    "title": font(31, True),
    "subtitle": font(18),
    "section": font(18, True),
    "body": font(15),
    "small": font(12),
    "tiny": font(10),
    "num": font(25, True),
    "quote": font(20, True),
}


def rgba(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (*rgb, alpha)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if text_size(draw, trial, fnt)[0] <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    width: int,
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    line_gap: int = 4,
    anchor: str = "la",
) -> int:
    x, y = xy
    lines = wrap(draw, text, fnt, width)
    line_h = text_size(draw, "Ag", fnt)[1] + line_gap
    if anchor == "mm":
        y = y - (len(lines) * line_h) // 2
    for i, line in enumerate(lines):
        if anchor == "mm":
            draw.text((x, y + i * line_h), line, font=fnt, fill=fill, anchor="ma")
        else:
            draw.text((x, y + i * line_h), line, font=fnt, fill=fill)
    return y + len(lines) * line_h


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    radius: int = 12,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def glow_line(
    base: Image.Image,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int = 4,
    glow: int = 10,
) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.line(points, fill=rgba(color, 60), width=width + glow, joint="curve")
    overlay = overlay.filter(ImageFilter.GaussianBlur(glow // 2))
    base.alpha_composite(overlay)
    ImageDraw.Draw(base).line(points, fill=rgba(color, 225), width=width, joint="curve")


def arrow(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color, width: int = 3) -> None:
    draw.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 11
    pts = [
        b,
        (int(b[0] - size * math.cos(ang - 0.55)), int(b[1] - size * math.sin(ang - 0.55))),
        (int(b[0] - size * math.cos(ang + 0.55)), int(b[1] - size * math.sin(ang + 0.55))),
    ]
    draw.polygon(pts, fill=color)


def make_background() -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    px = img.load()
    for y in range(H):
        t = y / H
        for x in range(W):
            cx = (x - W * 0.62) / W
            cy = (y - H * 0.35) / H
            radial = max(0.0, 1.0 - math.sqrt(cx * cx + cy * cy) * 2.2)
            base = blend(PALETTE["bg0"], PALETTE["bg1"], min(1, t * 0.7 + radial * 0.45))
            px[x, y] = (*base, 255)
    d = ImageDraw.Draw(img, "RGBA")
    for x in range(0, W, 48):
        d.line([(x, 0), (x, H)], fill=(94, 184, 190, 18), width=1)
    for y in range(0, H, 48):
        d.line([(0, y), (W, y)], fill=(94, 184, 190, 15), width=1)
    for i in range(22):
        x = 1180 + i * 18
        y = 22 + (i % 5) * 25
        r = 4 + (i % 3) * 2
        d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(PALETTE["cyan"], 80), outline=rgba(PALETTE["cyan"], 160))
        if i > 0:
            d.line([(x - 18, y - ((i - 1) % 5) * 25 + (i % 5) * 0), (x, y)], fill=rgba(PALETTE["cyan"], 75), width=1)
    return img


def draw_wetland(draw: ImageDraw.ImageDraw) -> None:
    horizon = 123
    draw.polygon([(0, 68), (0, 178), (295, 178), (240, 96), (150, 120), (76, 82)], fill=(20, 73, 47, 155))
    draw.polygon([(0, 102), (0, 182), (286, 182), (212, 130), (136, 155), (48, 122)], fill=(37, 99, 54, 120))
    draw.rectangle((0, horizon, 305, 178), fill=(20, 92, 104, 110))
    for i in range(42):
        x = (i * 19) % 306
        h = 18 + (i * 7) % 38
        color = rgba(PALETTE["green"], 125 if i % 2 else 90)
        draw.line([(x, 178), (x + 8, 178 - h)], fill=color, width=2)
    draw.line([(0, horizon), (305, horizon)], fill=rgba(PALETTE["cyan"], 70), width=2)


def card_header(draw, box, label, title, color) -> None:
    x0, y0, x1, _ = box
    draw.ellipse((x0 + 16, y0 + 16, x0 + 50, y0 + 50), fill=rgba(color, 210), outline=rgba((255, 255, 255), 90), width=1)
    draw.text((x0 + 33, y0 + 33), label, font=FONTS["section"], fill=PALETTE["ink"], anchor="mm")
    draw_wrapped(draw, (x0 + 62, y0 + 16), title, FONTS["section"], x1 - x0 - 80, PALETTE["ink"], line_gap=2)


def source_card(draw, x, y, title, metric, note, color) -> tuple[int, int]:
    box = (x, y, x + 198, y + 92)
    rounded(draw, box, rgba(PALETTE["panel"], 210), rgba(color, 165), radius=10, width=1)
    draw.text((x + 16, y + 14), title.upper(), font=FONTS["small"], fill=color)
    draw.text((x + 16, y + 36), metric, font=FONTS["section"], fill=PALETTE["ink"])
    draw_wrapped(draw, (x + 16, y + 62), note, FONTS["tiny"], 160, PALETTE["muted"], line_gap=1)
    return x + 198, y + 46


def node(draw, center, label, sub, color, r=35) -> None:
    x, y = center
    draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(color, 220), outline=rgba((255, 255, 255), 145), width=2)
    draw.text((x, y - 5), label, font=FONTS["small"], fill=PALETTE["ink"], anchor="mm")
    draw.text((x, y + 13), sub, font=FONTS["tiny"], fill=PALETTE["ink"], anchor="mm")


def comparison_table(draw, box, title, rows, accent, flat: bool) -> None:
    x0, y0, x1, y1 = box
    fill = rgba(PALETTE["white_panel"], 238) if flat else rgba((9, 42, 48), 230)
    outline = rgba(accent, 210)
    rounded(draw, box, fill, outline, radius=11, width=2)
    title_fill = PALETTE["black_text"] if flat else PALETTE["ink"]
    body_fill = (56, 68, 75) if flat else PALETTE["muted"]
    draw.text((x0 + 18, y0 + 16), title, font=FONTS["section"], fill=title_fill)
    top = y0 + 52
    if flat:
        headers = ["Rank", "proteome_id", "Score"]
        xs = [x0 + 20, x0 + 90, x1 - 86]
        for xx, h in zip(xs, headers):
            draw.text((xx, top), h, font=FONTS["tiny"], fill=(71, 83, 88))
        y = top + 25
        for rank, pid, score in rows:
            draw.rounded_rectangle((x0 + 14, y - 8, x1 - 14, y + 22), radius=5, fill=(255, 255, 255, 215))
            draw.text((x0 + 26, y), rank, font=FONTS["small"], fill=(33, 57, 65))
            draw.text((x0 + 90, y), pid, font=FONTS["tiny"], fill=(33, 57, 65))
            draw.text((x1 - 80, y), score, font=FONTS["small"], fill=(33, 57, 65))
            y += 35
        draw_wrapped(draw, (x0 + 18, y1 - 54), "Useful for prioritization, but provenance, claim status, and validation path are hidden.", FONTS["small"], x1 - x0 - 38, (58, 69, 75), line_gap=2)
    else:
        y = top + 4
        for title2, detail, col in rows:
            rounded(draw, (x0 + 16, y, x1 - 16, y + 33), rgba(col, 42), rgba(col, 135), radius=7, width=1)
            draw.text((x0 + 30, y + 8), title2, font=FONTS["small"], fill=col)
            draw.text((x0 + 182, y + 8), detail, font=FONTS["small"], fill=PALETTE["ink"])
            y += 37


def draw_infographic(out: Path) -> None:
    img = make_background()
    draw = ImageDraw.Draw(img, "RGBA")
    draw_wetland(draw)

    draw.text((28, 24), "MethaNet", font=FONTS["brand"], fill=PALETTE["ink"])
    draw.line([(264, 24), (264, 74)], fill=rgba(PALETTE["muted"], 145), width=1)
    draw.text((296, 20), "Queryable Attestation Graph Moat", font=FONTS["title"], fill=PALETTE["ink"])
    draw.text((296, 55), "From Multi-Source Signals to Explainable Claims", font=font(24, True), fill=PALETTE["green"])
    draw.text(
        (296, 88),
        "Companion to the agentic workflow moat: the evidence graph turns candidate ranking into auditable molecular intelligence.",
        font=font(15),
        fill=PALETTE["ink"],
    )
    rounded(
        draw,
        (346, 107, 1189, 143),
        rgba((0, 14, 20), 60),
        rgba(PALETTE["cyan"], 185),
        radius=4,
        width=1,
    )
    draw.text(
        (768, 125),
        "MISSION: explain why a proteome_id matters, where the evidence came from, and what validation upgrades the claim",
        font=FONTS["subtitle"],
        fill=PALETTE["cyan"],
        anchor="mm",
    )

    # Main three-stage band.
    band_y0, band_y1 = 154, 626
    rounded(draw, (14, band_y0, 1522, band_y1), rgba((2, 26, 31), 154), rgba(PALETTE["cyan"], 94), radius=12, width=1)
    left_box = (30, 174, 414, 594)
    mid_box = (436, 174, 1072, 594)
    right_box = (1094, 174, 1502, 594)
    for box, label, title, color in [
        (left_box, "1", "MULTI-SOURCE MOLECULAR INPUTS", PALETTE["green"]),
        (mid_box, "2", "ATTESTATION GRAPH: EVIDENCE AS CONNECTED OBJECTS", PALETTE["cyan"]),
        (right_box, "3", "CLAIM-AWARE OUTPUTS", PALETTE["orange"]),
    ]:
        rounded(draw, box, rgba(PALETTE["panel"], 182), rgba(color, 138), radius=10, width=1)
        card_header(draw, box, label, title, color)

    sources = [
        ("ESM2 geometry", "662 proteomes", "near-neighbor bridge signal", PALETTE["cyan"]),
        ("MAG atlas", "625 MAG bins", "functions, taxonomy, QC gates", PALETTE["green"]),
        ("gLM2 context", "648 contexts", "genomic neighborhood evidence", PALETTE["purple"]),
        ("Provenance", "13 artifacts", "file-level evidence lineage", PALETTE["orange"]),
    ]
    source_points = []
    y = 238
    for title, metric, note, color in sources:
        source_points.append(source_card(draw, 54, y, title, metric, note, color))
        y += 82

    graph_nodes = [
        ((548, 290), "MAG", "662", PALETTE["green"]),
        ((692, 230), "Feature", "2,644", PALETTE["cyan"]),
        ((850, 298), "Evidence", "3,968", PALETTE["gold"]),
        ((990, 236), "Artifact", "13", PALETTE["orange"]),
        ((706, 426), "Claim", "5 states", PALETTE["purple"]),
        ((940, 455), "Gap", "8", PALETTE["red"]),
    ]
    edges = [
        ((548, 290), (692, 230), PALETTE["green"]),
        ((692, 230), (850, 298), PALETTE["cyan"]),
        ((850, 298), (990, 236), PALETTE["orange"]),
        ((850, 298), (706, 426), PALETTE["gold"]),
        ((706, 426), (940, 455), PALETTE["red"]),
        ((548, 290), (706, 426), PALETTE["green"]),
        ((692, 230), (940, 455), PALETTE["cyan2"]),
    ]
    for a, b, col in edges:
        glow_line(img, [a, b], col, width=3, glow=12)
    draw = ImageDraw.Draw(img, "RGBA")
    for c, label, sub, color in graph_nodes:
        node(draw, c, label, sub, color)
    for sx, sy in source_points:
        arrow(draw, (sx + 8, sy), (506, 302), rgba(PALETTE["cyan"], 175), width=2)

    rounded(draw, (472, 514, 1036, 574), rgba((0, 16, 20), 95), rgba(PALETTE["cyan"], 130), radius=8, width=1)
    draw.text((754, 535), "proteome_id -> feature -> evidence atom -> source file -> claim -> validation gap", font=FONTS["body"], fill=PALETTE["ink"], anchor="mm")
    draw.text((754, 558), "Graph questions survive filtering, review, export, and partner diligence.", font=FONTS["small"], fill=PALETTE["muted"], anchor="mm")

    output_rows = [
        ("WHY IT MATTERS", "bridge candidate + function + context", PALETTE["green"]),
        ("EVIDENCE SUPPORTS", "near geometry + annotation + QC", PALETTE["cyan"]),
        ("FILE PRODUCED IT", "manifest path + checksum lineage", PALETTE["orange"]),
        ("CLAIM ALLOWED", "molecular attestation, not final MRV", PALETTE["gold"]),
        ("CLAIM BLOCKED", "missing abundance, env, flux validation", PALETTE["red"]),
        ("UPGRADE ACTION", "sample mapping + coverage + field validation", PALETTE["purple"]),
    ]
    y = 226
    for label, detail, col in output_rows:
        row_fill = blend(PALETTE["panel"], col, 0.22)
        rounded(draw, (1120, y, 1478, y + 44), rgba(row_fill, 235), rgba(col, 165), radius=8, width=1)
        draw.text((1136, y + 8), label, font=FONTS["tiny"], fill=col)
        draw_wrapped(draw, (1260, y + 7), detail, FONTS["tiny"], 202, PALETTE["ink"], line_gap=1)
        y += 55

    # Comparison strip.
    comparison_table(
        draw,
        (18, 654, 489, 894),
        "MONOLITHIC TABLE: RANKS CANDIDATES",
        [
            ("#1", "rumen__10674_0004_idba_bin.23", "0.97"),
            ("#2", "mucc__GCA_002495465.1", "0.94"),
            ("#3", "wetland__candidate_MAG_041", "0.91"),
        ],
        PALETTE["muted"],
        True,
    )
    comparison_table(
        draw,
        (516, 654, 1518, 894),
        "ATTESTATION GRAPH: EXPLAINS CANDIDATES",
        [
            ("Candidate", "proteome_id retained as the cohort key", PALETTE["green"]),
            ("Evidence", "3,968 atoms connect function, context, QC, and geometry", PALETTE["cyan"]),
            ("Provenance", "every evidence atom points back to the producing artifact", PALETTE["orange"]),
            ("Claim status", "allowed wording is separated from blocked MRV claims", PALETTE["gold"]),
            ("Upgrade path", "validation gaps say what would improve confidence", PALETTE["purple"]),
        ],
        PALETTE["cyan"],
        False,
    )
    arrow(draw, (492, 774), (512, 774), rgba(PALETTE["orange"], 230), width=5)

    # Claim boundary and headline point.
    rounded(draw, (14, 915, 1522, 1009), rgba((0, 22, 27), 160), rgba(PALETTE["orange"], 170), radius=10, width=1)
    quote = (
        "A monolithic table can rank candidates; this graph explains why a candidate matters, "
        "what evidence supports it, which file produced it, what claim is allowed, what claim is blocked, "
        "and what validation would upgrade it."
    )
    draw_wrapped(draw, (58, 938), quote, FONTS["quote"], 1075, PALETTE["ink"], line_gap=5)
    rounded(draw, (1188, 934, 1496, 991), rgba(PALETTE["red"], 34), rgba(PALETTE["red"], 145), radius=8, width=1)
    draw.text((1210, 946), "CLAIM BOUNDARY", font=FONTS["small"], fill=PALETTE["red"])
    draw_wrapped(draw, (1210, 966), "MAG/proteome molecular attestation and MRV feature readiness; not final A-E risk tiers, flux, or credit approval.", FONTS["tiny"], 264, PALETTE["muted"], line_gap=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/"
            "infographics/methanet_attestation_graph_moat_20260617/"
            "methanet_attestation_graph_moat_v1.png"
        ),
    )
    parser.add_argument("--copy-to", type=Path, default=None)
    args = parser.parse_args()

    draw_infographic(args.out)
    if args.copy_to:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, args.copy_to)
    print(args.out)
    if args.copy_to:
        print(args.copy_to)


if __name__ == "__main__":
    main()
