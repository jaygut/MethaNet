#!/usr/bin/env python3
"""Build a run-aware embedding input audit for Blue Catalyst embedding scale-up.

Produces:
- TSV inventory with per-sample/file flags
- JSON denominator summary (global cache vs run-canonical)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def count_fasta_headers(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(">"):
                n += 1
    return n


def classify_sample(sample: str, faa_path: Path) -> tuple[bool, bool]:
    token = f"{sample} {faa_path.name} {faa_path}"
    lower = token.lower()
    is_bin_like = "bin." in lower or "_bin" in lower or ".bin" in lower
    is_coassembly_like = "coassembly" in lower
    return is_bin_like, is_coassembly_like


def load_run_canonical(manifest_path: Path | None) -> pd.DataFrame:
    if manifest_path is None or not manifest_path.exists():
        return pd.DataFrame(columns=["sample", "source", "proteome_faa"])

    df = pd.read_csv(manifest_path, sep="\t")
    required = {"sample", "proteome_faa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Run manifest missing required columns: {', '.join(sorted(missing))}"
        )

    if "source" not in df.columns:
        df["source"] = df["sample"].astype(str).map(
            lambda s: "mucc"
            if s.startswith("mucc__")
            else "rumen"
            if s.startswith("rumen__")
            else "unknown"
        )

    return df[["sample", "source", "proteome_faa"]].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit embedding input inventory.")
    parser.add_argument("--proteomes-dir", required=True, type=Path)
    parser.add_argument("--source-subset", required=True, type=Path)
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    proteomes_dir = args.proteomes_dir
    if not proteomes_dir.exists():
        raise FileNotFoundError(f"Proteomes dir not found: {proteomes_dir}")
    if not args.source_subset.exists():
        raise FileNotFoundError(f"Source subset not found: {args.source_subset}")

    run_df = load_run_canonical(args.run_manifest)

    global_rows: list[dict[str, object]] = []
    for faa in sorted(proteomes_dir.glob("*.faa")):
        sample = faa.stem
        source = (
            "mucc"
            if sample.startswith("mucc__")
            else "rumen"
            if sample.startswith("rumen__")
            else "cache"
        )
        is_bin_like, is_coassembly_like = classify_sample(sample, faa)
        n_headers = count_fasta_headers(faa)
        global_rows.append(
            {
                "sample": sample,
                "source": source,
                "proteome_faa": str(faa),
                "exists": True,
                "size_bytes": int(faa.stat().st_size),
                "n_headers": int(n_headers),
                "is_bin_like": bool(is_bin_like),
                "is_coassembly_like": bool(is_coassembly_like),
                "in_run_canonical": False,
            }
        )

    inv_df = pd.DataFrame(global_rows)
    if inv_df.empty:
        inv_df = pd.DataFrame(
            columns=[
                "sample",
                "source",
                "proteome_faa",
                "exists",
                "size_bytes",
                "n_headers",
                "is_bin_like",
                "is_coassembly_like",
                "in_run_canonical",
            ]
        )

    run_samples = set(run_df["sample"].astype(str)) if not run_df.empty else set()
    if run_samples:
        inv_df["in_run_canonical"] = inv_df["sample"].astype(str).isin(run_samples)

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    inv_df.sort_values(["in_run_canonical", "source", "sample"], ascending=[False, True, True]).to_csv(
        args.out_tsv, sep="\t", index=False
    )

    run_subset_rows = max(sum(1 for _ in args.source_subset.open("r", encoding="utf-8")) - 1, 0)

    summary = {
        "global_cache_count": int(len(inv_df)),
        "run_canonical_count": int(len(run_df)),
        "run_subset_rows": int(run_subset_rows),
        "run_manifest_path": str(args.run_manifest) if args.run_manifest else "",
        "run_manifest_present": bool(args.run_manifest and args.run_manifest.exists()),
        "run_canonical_present_count": int(
            run_df["proteome_faa"].map(lambda p: Path(str(p)).exists()).sum()
        )
        if not run_df.empty
        else 0,
        "run_canonical_missing_count": int(
            (~run_df["proteome_faa"].map(lambda p: Path(str(p)).exists())).sum()
        )
        if not run_df.empty
        else 0,
        "global_coassembly_like_count": int(inv_df["is_coassembly_like"].sum()) if not inv_df.empty else 0,
        "global_bin_like_count": int(inv_df["is_bin_like"].sum()) if not inv_df.empty else 0,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Wrote inventory: {args.out_tsv}")
    print(f"[OK] Wrote summary: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
