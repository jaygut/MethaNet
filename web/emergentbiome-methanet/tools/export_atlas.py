#!/usr/bin/env python3
"""
EmergentBiome / MethaNet — Phase 1 atlas export (seeded, reproducible).

Reads the freeze-backed molecular niche-space projection produced by the MethaNet
report builder and emits a lean, page-ready `data/atlas.json` that drives the
hero scene (Scene 3) and the atlas scene (Scene 4).

PREFERENCE ORDER (per build spec) — option used: **(1) DIFFUSION MAP / PHATE**.
The source `niche.json` carries REAL low-dimensional coordinates of the ESM2 (650M)
proteome embeddings under five methods, all from the same cosine kNN affinity graph.
The hero's 2D backbone (x,y) is **PHATE** — a diffusion-based method built for 2D
manifold visualization, which preserves the mangrove expansion's within-lane
structure that the raw diffusion eigenmap collapses into a line. The repo's primary
diffusion-map coordinates (dx,dy) and PCA (px,py) are also exported, full-cohort, as
real projection-sensitivity toggles. UMAP and t-SNE are POC-only (625) at the source
and therefore not used as the multi-lane backbone.

Coordinates are REAL. The only transform is a per-axis standardize + symmetric
scale + soft clip so the anisotropic eigenvector components render legibly; this is
a monotonic, structure-preserving rescale (point neighborhoods and bridge topology
are preserved). It is NOT a procedural stylization.

Deterministic: no RNG is used (no subsampling, no jitter). Re-running on the same
source yields byte-identical output. SOURCE_NICHE pins the exact freeze.

Usage:
    python3 tools/export_atlas.py            # from web/emergentbiome-methanet/
    python3 tools/export_atlas.py --check    # print summary, do not write
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from collections import Counter, OrderedDict

# --- pinned source (relative to repo root) -------------------------------------
REPO_ROOT_FROM_HERE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOURCE_NICHE = os.path.join(
    REPO_ROOT_FROM_HERE,
    "results/reports/mbag_nextgen_molecular_niche_atlas_20260625_release_freeze_145509_bridge_v4",
    "assets/data/niche.json",
)
OUT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "atlas.json"))
SNAPSHOT = "2026-06-25"

ECO_FROM_PREFIX = {
    "rumen": "rumen",
    "mucc": "wetland",
    "msm_china_2025": "mangrove_msm",
    "futian_mangrove_2026_qi": "mangrove_futian",
}
ECO_CODE = {"rumen": 0, "wetland": 1, "mangrove_msm": 2, "mangrove_futian": 3}


def prefix(pid: str) -> str:
    return pid.split("__", 1)[0]


def eco_of(pid: str) -> str:
    return ECO_FROM_PREFIX.get(prefix(pid), "unknown")


def domain_code(dom):
    if not dom:
        return "?"
    d = str(dom).lower()
    if "archae" in d:
        return "A"
    if "bacter" in d:
        return "B"
    return "?"


def standardize_scale(values, gain=0.62):
    """Z-score then tanh squash into (-1, 1).

    tanh is monotonic and structure-preserving: it keeps point ordering and local
    neighborhoods of the real diffusion components while gently compressing the few
    far outliers, so the dense core spreads legibly with no hard clip wall.
    """
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var) or 1.0
    return [math.tanh((v - mean) / sd * gain) for v in values]


def minmax_scale(values, lo_pct=0.3, hi_pct=99.7):
    """Linear per-axis min-max to [-1, 1] with mild percentile clipping.

    Faithful to the report's diffusion-map view: a uniform rescale of the raw
    eigenvector values preserves the fan structure (rumen upper, wetland lower,
    mangrove line) instead of saturating the POC fan the way a global tanh does.
    """
    s = sorted(values)
    n = len(s)
    lo = s[max(0, min(n - 1, int(lo_pct / 100 * n)))]
    hi = s[max(0, min(n - 1, int(hi_pct / 100 * n)))]
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        t = (v - lo) / span * 2 - 1
        out.append(-1.0 if t < -1 else (1.0 if t > 1 else t))
    return out


def norm01_percentile(values):
    """Robust 0..1 via 2nd/98th percentile clamp (for visual encoding of densities)."""
    s = sorted(values)
    n = len(s)
    lo = s[max(0, int(0.02 * n))]
    hi = s[min(n - 1, int(0.98 * n))]
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        t = (v - lo) / span
        out.append(0.0 if t < 0 else (1.0 if t > 1 else t))
    return out


def r(x, nd=4):
    if x is None:
        return None
    return round(float(x), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="summarize only; do not write")
    args = ap.parse_args()

    if not os.path.exists(SOURCE_NICHE):
        sys.exit(f"ERROR: source niche.json not found:\n  {SOURCE_NICHE}")

    with open(SOURCE_NICHE) as fh:
        doc = json.load(fh)

    raw_nodes = doc["nodes"]
    raw_links = doc["links"]

    # keep only embedding-bearing nodes (have a primary diffusion coordinate)
    nodes = [n for n in raw_nodes if n.get("diffusion_1") is not None and n.get("diffusion_2") is not None]
    gap_rows = len(raw_nodes) - len(nodes)

    # --- per-projection display coords (REAL, structure-preserving) ---
    # Primary = diffusion map via linear min-max (faithful to the report's fan view).
    # PHATE / PCA kept as secondary projection-sensitivity toggles (standardize+tanh).
    proj = {}
    proj["d"] = (minmax_scale([n["diffusion_1"] for n in nodes]),
                 minmax_scale([n["diffusion_2"] for n in nodes]))
    for key, (a, b) in {"p": ("pca_1", "pca_2"), "h": ("phate_1", "phate_2")}.items():
        proj[key] = (standardize_scale([n[a] for n in nodes]),
                     standardize_scale([n[b] for n in nodes]))

    # methane-marker density (per 1k aa) -> robust 0..1 for honest visual encoding
    mz = norm01_percentile([n.get("methane_marker_density_per_1k") or 0.0 for n in nodes])

    # bridge-link endpoints (the documented cross-ecosystem kNN bridge genomes)
    bridge_ids = set()
    for l in raw_links:
        bridge_ids.add(l["source"])
        bridge_ids.add(l["target"])

    id_to_idx = {}
    points = []
    eco_counts = Counter()
    for i, n in enumerate(nodes):
        pid = n["proteome_id"]
        eco = eco_of(pid)
        eco_counts[eco] += 1
        id_to_idx[pid] = i
        dx, dy = proj["d"][0][i], proj["d"][1][i]
        px, py = proj["p"][0][i], proj["p"][1][i]
        hx, hy = proj["h"][0][i], proj["h"][1][i]
        points.append(OrderedDict([
            ("id", pid),
            ("e", ECO_CODE[eco]),
            ("x", r(dx)), ("y", r(dy)),          # primary display = DIFFUSION MAP (min-max)
            ("hx", r(hx)), ("hy", r(hy)),         # PHATE (secondary toggle)
            ("px", r(px)), ("py", r(py)),         # PCA (linear sanity-check)
            ("d", domain_code(n.get("domain"))),
            ("br", 1 if pid in bridge_ids else 0),
            ("cs", 1 if n.get("is_case_study") else 0),
            ("ma", r(n.get("molecular_attestation_index") or 0.0, 3)),
            ("mz", r(mz[i], 3)),
            ("nps", r(n.get("nearest_poc_similarity"), 3) if eco.startswith("mangrove") else None),
        ]))

    # bridges as integer-indexed edges
    bridges = []
    dropped = 0
    for l in raw_links:
        s = id_to_idx.get(l["source"])
        t = id_to_idx.get(l["target"])
        if s is None or t is None:
            dropped += 1
            continue
        bridges.append(OrderedDict([
            ("s", s), ("t", t),
            ("w", r(l["similarity"], 4)),
            ("cd", 1 if l.get("cross_domain") else 0),
            ("cs", 1 if l.get("evidence_type") == "case_study_nearest_poc" else 0),
            ("rk", int(l.get("rank") or 0)),
        ]))

    # per-ecosystem centroids on the primary (diffusion) display coords
    cent = {}
    for eco in ECO_CODE:
        pts = [p for p in points if p["e"] == ECO_CODE[eco]]
        if pts:
            cent[eco] = [round(sum(p["x"] for p in pts) / len(pts), 4),
                         round(sum(p["y"] for p in pts) / len(pts), 4)]

    out = OrderedDict([
        ("meta", OrderedDict([
            ("artifact", "EmergentBiome/MethaNet atlas — Phase 1 data export"),
            ("source", os.path.relpath(SOURCE_NICHE, REPO_ROOT_FROM_HERE)),
            ("option_used", "1 — DIFFUSION MAP 2D coordinates of the proteome embeddings (REAL); PHATE + PCA also exported as toggles"),
            ("primary_projection", "diffusion"),
            ("secondary_projections", ["phate", "pca"]),
            ("coord_transform", "diffusion: per-axis linear min-max (0.3/99.7 clip), faithful to the report fan; phate/pca: standardize + tanh"),
            ("projection_note", "The primary hero map is the diffusion map (built from the proteome-embedding cosine kNN affinity graph): rumen and wetland references form fans on the left, the mangrove expansion forms the line on the right, and the bridge links span between them. PHATE and PCA are exported as projection-sensitivity toggles."),
            ("snapshot", SNAPSHOT),
            ("n_points", len(points)),
            ("n_bridges", len(bridges)),
            ("n_case_study", sum(p["cs"] for p in points)),
            ("n_bridge_nodes", sum(p["br"] for p in points)),
            ("excluded_gap_rows", gap_rows),
            ("ecosystem_codes", ECO_CODE),
            ("ecosystem_counts", dict(eco_counts)),
            ("centroids", cent),
        ])),
        ("points", points),
        ("bridges", bridges),
    ])

    summary = (
        f"points={len(points)}  bridges={len(bridges)}  bridge_nodes={sum(p['br'] for p in points)}  "
        f"case_study={sum(p['cs'] for p in points)}  gap_excluded={gap_rows}  dropped_edges={dropped}\n"
        f"ecosystems={dict(eco_counts)}\n"
        f"centroids={cent}"
    )
    print(summary)

    if args.check:
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    sz = os.path.getsize(OUT_PATH)
    print(f"wrote {OUT_PATH}  ({sz/1024:.0f} KB)")


if __name__ == "__main__":
    main()
