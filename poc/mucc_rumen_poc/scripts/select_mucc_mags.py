#!/usr/bin/env python3
"""Select MUCC MAG IDs by site and taxonomy from metadata TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PREFER = [
    "Methano",
    "Methylomonas",
    "Methylocystis",
    "Methylobacter",
    "Methylosinus",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="MUCC MAG metadata TSV")
    parser.add_argument("--out-ids", required=True)
    parser.add_argument("--sites", default="OWC,PPR,JLA,LA2,STORDALEN,SPRUCE")
    parser.add_argument("--per-site", type=int, default=40)
    parser.add_argument("--min-completeness", type=float, default=90.0)
    parser.add_argument("--max-contamination", type=float, default=5.0)
    parser.add_argument("--id-col", default="mag_id")
    parser.add_argument("--site-col", default="site")
    parser.add_argument("--taxonomy-col", default="taxonomy")
    parser.add_argument("--completeness-col", default="completeness")
    parser.add_argument("--contamination-col", default="contamination")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata, sep="\t")

    for col in [args.id_col, args.site_col, args.taxonomy_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")

    df = df[df[args.completeness_col] >= args.min_completeness]
    df = df[df[args.contamination_col] <= args.max_contamination]

    pattern = "|".join(DEFAULT_PREFER)
    df = df.copy()
    df["_prefer"] = df[args.taxonomy_col].str.contains(
        pattern, case=False, na=False
    )

    selected = []
    for site in args.sites.split(","):
        pool = df[df[args.site_col] == site]
        pool = pool.sort_values(
            by=["_prefer", args.completeness_col], ascending=[False, False]
        )
        take = pool.head(args.per_site)
        selected.extend(take[args.id_col].tolist())

    out_path = Path(args.out_ids)
    out_path.write_text("\n".join(selected) + "\n")
    print(f"Wrote {len(selected)} IDs to {out_path}")


if __name__ == "__main__":
    main()
