import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power


def load_numeric(path: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    df = pd.read_parquet(path)
    numeric = df.select_dtypes(include=["number"])
    return df, list(numeric.columns), numeric


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CORAL alignment.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-transform", required=True)
    parser.add_argument("--output-source", required=True)
    parser.add_argument("--output-target", required=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    src_df, src_cols, src_numeric = load_numeric(Path(args.source))
    tgt_df, tgt_cols, tgt_numeric = load_numeric(Path(args.target))

    shared = sorted(set(src_cols).intersection(tgt_cols))
    if not shared:
        raise ValueError("No shared numeric columns between source and target.")

    src_vals = src_numeric[shared].to_numpy()
    tgt_vals = tgt_numeric[shared].to_numpy()

    src_mean = src_vals.mean(axis=0)
    tgt_mean = tgt_vals.mean(axis=0)

    src_centered = src_vals - src_mean
    tgt_centered = tgt_vals - tgt_mean

    cov_src = np.cov(src_centered, rowvar=False) + np.eye(src_vals.shape[1]) * args.eps
    cov_tgt = np.cov(tgt_centered, rowvar=False) + np.eye(tgt_vals.shape[1]) * args.eps

    cov_src_inv_sqrt = fractional_matrix_power(cov_src, -0.5)
    cov_tgt_sqrt = fractional_matrix_power(cov_tgt, 0.5)
    transform = cov_src_inv_sqrt @ cov_tgt_sqrt

    aligned_src = src_centered @ transform + tgt_mean

    transform = np.real_if_close(transform)
    aligned_src = np.real_if_close(aligned_src)

    src_out = src_df.copy()
    src_out.loc[:, shared] = aligned_src
    tgt_out = tgt_df.copy()
    tgt_out.loc[:, shared] = tgt_vals

    out_transform = Path(args.output_transform)
    out_transform.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_transform, transform)

    out_source = Path(args.output_source)
    out_source.parent.mkdir(parents=True, exist_ok=True)
    src_out.to_parquet(out_source, index=False)

    out_target = Path(args.output_target)
    out_target.parent.mkdir(parents=True, exist_ok=True)
    tgt_out.to_parquet(out_target, index=False)


if __name__ == "__main__":
    main()
