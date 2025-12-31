import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp


def load_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("No numeric columns found in features file.")
    return numeric


def main() -> None:
    parser = argparse.ArgumentParser(description="Select transferable features.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    source = load_numeric(Path(args.source))
    target = load_numeric(Path(args.target))

    shared = sorted(set(source.columns).intersection(target.columns))
    if not shared:
        raise ValueError("No shared numeric columns between source and target.")

    rows = []
    for name in shared:
        stat = ks_2samp(source[name].values, target[name].values).statistic
        rows.append({"feature": name, "transferability": 1.0 - stat})

    ranked = pd.DataFrame(rows).sort_values("transferability", ascending=False)
    if args.top_k > 0:
        ranked = ranked.head(args.top_k)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
