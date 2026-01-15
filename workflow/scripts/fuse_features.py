import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from methanet.embedding.fusion import FeatureFusion, FusionConfig


def read_functional(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    row = df.iloc[0]
    return np.array(
        [
            row["mcrA"],
            row["pmoA"],
            row["dsrA"],
            row["nifH"],
            row["cbbL"],
            row["mcrA_pmoA_ratio"],
        ],
        dtype=float,
    )


def load_embedding(path: Path, expected_dim: int) -> np.ndarray:
    if not path.exists():
        return np.zeros(expected_dim, dtype=float)
    arr = np.load(path)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"Unexpected embedding dim {arr.shape[0]} for {path} (expected {expected_dim})"
        )
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse functional and embedding features.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--functional-dir", required=True)
    parser.add_argument("--esm2-dir", required=True)
    parser.add_argument("--genome-dir", required=True)
    parser.add_argument("--output-all", required=True)
    parser.add_argument("--output-source", required=True)
    parser.add_argument("--output-target", required=True)
    parser.add_argument("--source-domain", default="rumen")
    parser.add_argument("--target-domain", default="coastal")
    parser.add_argument("--genome-backend", default="dnabert2")
    parser.add_argument("--genome-dim", type=int, required=True)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata, sep="\t", comment="#")
    if "sample_id" not in meta.columns:
        raise ValueError("metadata file must include sample_id column")

    fusion = FeatureFusion(
        FusionConfig(include_metadata=False, genome_dim=args.genome_dim)
    )

    rows = []
    for _, row in meta.iterrows():
        sample_id = row["sample_id"]
        functional_path = Path(args.functional_dir) / f"{sample_id}.tsv"
        functional = read_functional(functional_path) if functional_path.exists() else None

        esm2_path = Path(args.esm2_dir) / f"{sample_id}_esm2.npy"
        genome_path = Path(args.genome_dir) / f"{sample_id}_{args.genome_backend}.npy"

        esm2 = load_embedding(esm2_path, fusion.ESM2_DIM)
        genome = load_embedding(genome_path, args.genome_dim)

        fused = fusion.fuse(
            sample_id=sample_id,
            functional=functional,
            esm2_embedding=esm2,
            genomeocean_embedding=genome,
            metadata=None,
        )

        feature_row = {f"f_{i}": float(val) for i, val in enumerate(fused.features)}
        payload = row.to_dict()
        payload.update(feature_row)
        rows.append(payload)

    fused_df = pd.DataFrame(rows)
    out_all = Path(args.output_all)
    out_all.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_parquet(out_all, index=False)

    source_df = fused_df[fused_df["domain"] == args.source_domain]
    target_df = fused_df[fused_df["domain"] == args.target_domain]

    out_source = Path(args.output_source)
    out_source.parent.mkdir(parents=True, exist_ok=True)
    source_df.to_parquet(out_source, index=False)

    out_target = Path(args.output_target)
    out_target.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_parquet(out_target, index=False)


if __name__ == "__main__":
    main()
