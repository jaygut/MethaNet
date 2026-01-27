#!/usr/bin/env python3
"""Select rumen MAG accessions from an ENA assembly listing TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assemblies", required=True, help="ENA assembly TSV")
    parser.add_argument("--out-ids", required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--accession-col", default="accession")
    parser.add_argument("--name-col", default="assembly_name")
    parser.add_argument("--desc-col", default="description")
    parser.add_argument("--keyword", default="Methano")
    args = parser.parse_args()

    df = pd.read_csv(args.assemblies, sep="\t")
    for col in [args.accession_col, args.name_col, args.desc_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")

    keyword = args.keyword
    mask = (
        df[args.name_col].str.contains(keyword, case=False, na=False)
        | df[args.desc_col].str.contains(keyword, case=False, na=False)
    )
    hits = df[mask].head(args.count)

    out_path = Path(args.out_ids)
    out_path.write_text("\n".join(hits[args.accession_col].tolist()) + "\n")
    print(f"Wrote {len(hits)} IDs to {out_path}")


if __name__ == "__main__":
    main()
