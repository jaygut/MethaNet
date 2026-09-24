#!/usr/bin/env python3
"""
EmergentBiome / MethaNet — Phase 1 atlas export (seeded, reproducible).

Reads the freeze-backed molecular niche-space projection produced by the MethaNet
report builder and emits a lean, page-ready `data/atlas.json` that drives the
hero scene (Scene 3) and the atlas scene (Scene 4).

Primary view: diffusion map. The source `niche.json` carries computed
low-dimensional coordinates of the ESM-2 (650M) proteome embeddings. The
methods have distinct algorithms; t-SNE is a visual comparison, not a graph
or candidate-ranking substrate.
The hero's 2D backbone (x,y) is the report's primary diffusion map. The exporter
also preserves PCA, t-SNE, and the best available nonlinear sensitivity projection
(PHATE when present, otherwise UMAP) for the same embedding-bearing records.

Coordinates come from the report. Display transforms make them render legibly.
Some per-axis transforms change visual distances. Bridge and neighbor membership
comes from the original high-dimensional ESM-2 cosine analysis. It is NOT a procedural
stylization.

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
    "results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end",
    "assets/data/niche.json",
)
OUT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "atlas.json"))
SNAPSHOT = "2026-08-10"

ECO_FROM_PREFIX = {
    "rumen": "rumen",
    "mucc": "wetland",
    "mucc_v1": "wetland",
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
    """Z-score then tanh squash into (-1, 1) for display only."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var) or 1.0
    return [math.tanh((v - mean) / sd * gain) for v in values]


def uniform_scale_pair(xs, ys, pad=0.92):
    """Center two axes with one scale factor, preserving their 2D distances."""
    x_mid = (min(xs) + max(xs)) / 2
    y_mid = (min(ys) + max(ys)) / 2
    half_span = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1.0
    scale = pad / half_span
    return ([(x - x_mid) * scale for x in xs],
            [(y - y_mid) * scale for y in ys])


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
    if not values:
        # A valid freeze may contain no mechanism-comparable unit. In that
        # state every public methane-intensity value remains null.
        return []
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
    ap.add_argument("--niche", default=SOURCE_NICHE, help="path to a niche.json (defaults to the pinned current Atlas)")
    ap.add_argument("--snapshot", default=SNAPSHOT, help="snapshot date stamped into meta")
    ap.add_argument("--release-ledger", default=None, help="release_ledger.json used to validate exported counts")
    args = ap.parse_args()
    src = args.niche
    snap = args.snapshot

    if not os.path.exists(src):
        sys.exit(f"ERROR: source niche.json not found:\n  {src}")

    with open(src) as fh:
        doc = json.load(fh)

    raw_nodes = doc["nodes"]
    raw_links = doc["links"]

    # keep only embedding-bearing nodes (have a primary diffusion coordinate)
    nodes = [n for n in raw_nodes if n.get("diffusion_1") is not None and n.get("diffusion_2") is not None]
    gap_rows = len(raw_nodes) - len(nodes)
    if len({n["proteome_id"] for n in nodes}) != len(nodes):
        sys.exit("ERROR: duplicate proteome_id among embedding-bearing records")

    # --- per-projection display coords (real; visual distances may change) ---
    # Primary = diffusion map via linear min-max (faithful to the report's fan view).
    # Other computed projections are display-only sensitivity toggles.
    proj = {}
    proj["d"] = (minmax_scale([n["diffusion_1"] for n in nodes]),
                 minmax_scale([n["diffusion_2"] for n in nodes]))
    nonlinear_name = next(
        (
            name
            for name in ("phate", "umap")
            if all(
                n.get(f"{name}_1") is not None
                and n.get(f"{name}_2") is not None
                for n in nodes
            )
        ),
        "pca",
    )
    if any(
        not all(
            n.get(axis) is not None and math.isfinite(float(n[axis]))
            for axis in ("tsne_1", "tsne_2")
        )
        for n in nodes
    ):
        sys.exit("ERROR: t-SNE coordinates are missing or non-finite for an embedding-bearing record")
    for key, (a, b) in {
        "p": ("pca_1", "pca_2"),
        "h": (f"{nonlinear_name}_1", f"{nonlinear_name}_2"),
    }.items():
        proj[key] = (standardize_scale([n[a] for n in nodes]),
                     standardize_scale([n[b] for n in nodes]))
    proj["t"] = uniform_scale_pair(
        [n["tsne_1"] for n in nodes], [n["tsne_2"] for n in nodes]
    )

    # Public methane intensity is restricted to the curated POC mechanism
    # contract. MSM/Futian raw hit-row aggregates and MUCC source terms are
    # intentionally null after the scientific reconciliation; treating those
    # lanes as zero would falsely imply comparable biological absence.
    eligible_mz_nodes = [
        n
        for n in nodes
        if n.get("rate_metric_status") == "comparable_curated_feature_density"
        and n.get("methane_marker_density_per_1k") is not None
    ]
    eligible_mz_scaled = norm01_percentile(
        [float(n["methane_marker_density_per_1k"]) for n in eligible_mz_nodes]
    )
    mz_by_id = {
        n["proteome_id"]: value
        for n, value in zip(eligible_mz_nodes, eligible_mz_scaled, strict=True)
    }
    legacy_contract_code = {
        "canonical_mechanism_comparable": 1,
        "annotation_complete_harmonization_pending": 2,
        "source_scaffold_non_equivalent": 3,
        "pipeline_normalized_comparability_pending": 4,
    }
    formal_contract_code = {
        "complete_canonical_mechanism_tri_view": 1,
        "complete_annotation_tri_view_harmonization_pending": 2,
        "complete_source_scaffold_tri_view": 3,
        "complete_pipeline_normalized_tri_view_comparability_pending": 4,
    }

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
        tx, ty = proj["t"][0][i], proj["t"][1][i]
        points.append(OrderedDict([
            ("id", pid),
            ("e", ECO_CODE[eco]),
            ("x", r(dx)), ("y", r(dy)),          # primary display = DIFFUSION MAP (min-max)
            ("hx", r(hx)), ("hy", r(hy)),         # PHATE or UMAP
            ("tx", r(tx)), ("ty", r(ty)),         # t-SNE
            ("px", r(px)), ("py", r(py)),         # PCA (linear sanity-check)
            ("d", domain_code(n.get("domain"))),
            ("br", 1 if pid in bridge_ids else 0),
            ("cs", 1 if n.get("is_case_study") else 0),
            ("ma", r(n.get("molecular_attestation_index"), 3)),
            ("mz", r(mz_by_id.get(pid), 3)),
            (
                "fc",
                formal_contract_code.get(
                    n.get("formal_tri_view_status"),
                    legacy_contract_code.get(n.get("functional_comparability_tier"), 0),
                ),
            ),
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

    release_ledger = None
    if args.release_ledger:
        with open(args.release_ledger) as fh:
            release_ledger = json.load(fh)
        observed_contracts = Counter(point["fc"] for point in points)
        expected = {
            "snapshot_date": snap,
            "registered_units": len(points) + gap_rows,
            "release_required_units": len(points),
            "explicit_non_runnable_gaps": gap_rows,
            "tri_view_ready_units": sum(observed_contracts[code] for code in (1, 2, 3, 4)),
            "mechanism_comparable_units": observed_contracts[1],
            "annotation_complete_tri_view_units": observed_contracts[2],
            "source_scaffold_tri_view_units": observed_contracts[3],
            "pipeline_normalized_tri_view_units": observed_contracts[4],
            "blocking_units": observed_contracts[0],
        }
        drift = {
            key: {"export": value, "ledger": release_ledger.get(key)}
            for key, value in expected.items()
            if release_ledger.get(key) != value
        }
        if drift:
            sys.exit("ERROR: atlas/release-ledger drift: " + json.dumps(drift, sort_keys=True))

    out = OrderedDict([
        ("meta", OrderedDict([
            ("artifact", "EmergentBiome Molecular Atlas — evidence-reconciled public export"),
            ("source", os.path.relpath(src, REPO_ROOT_FROM_HERE)),
            ("option_used", f"1 — DIFFUSION MAP 2D coordinates of the proteome embeddings (REAL); {nonlinear_name.upper()}, t-SNE and PCA also exported as sensitivity views"),
            ("primary_projection", "diffusion"),
            ("secondary_projections", [nonlinear_name, "tsne", "pca"]),
            ("coord_transform", f"diffusion: per-axis linear min-max (0.3/99.7 clip); {nonlinear_name}/pca: standardize + tanh for display only; tsne: shared affine scale preserving 2D distances"),
            ("projection_note", f"The primary hero map is the diffusion map built from the proteome-embedding cosine kNN affinity graph. {nonlinear_name.upper()}, t-SNE and PCA are retained as projection-sensitivity views. None determines high-dimensional link membership."),
            ("snapshot", snap),
            ("n_points", len(points)),
            ("n_bridges", len(bridges)),
            ("n_case_study", sum(p["cs"] for p in points)),
            ("n_bridge_nodes", sum(p["br"] for p in points)),
            ("excluded_gap_rows", gap_rows),
            ("release_state", release_ledger.get("release_state") if release_ledger else "unvalidated"),
            ("release_ledger_schema_version", release_ledger.get("schema_version") if release_ledger else None),
            ("freeze_manifest_sha256", release_ledger.get("freeze_manifest_sha256") if release_ledger else None),
            ("schema_normalized_tri_view_units", release_ledger.get("schema_normalized_tri_view_units") if release_ledger else None),
            (
                "methane_intensity_scope",
                "non-null only for rows passing the release mechanism-comparability gate; null otherwise",
            ),
            (
                "functional_contract_codes",
                {
                    "0": "incomplete or unclassified",
                    "1": "mechanism-comparable under the release contract",
                    "2": "annotation-complete, normalized aggregation pending",
                    "3": "source scaffold, non-equivalent",
                    "4": "pipeline-normalized screening, cross-lane comparability pending",
                },
            ),
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
