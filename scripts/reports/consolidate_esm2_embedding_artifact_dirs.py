#!/usr/bin/env python3
"""Consolidate sharded ESM2 artifact directories into one artifact contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_artifact_dir(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta_path = path / "embedding_metadata.tsv"
    npz_path = path / "genome_embeddings.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"Missing embedding artifacts under {path}")
    meta = pd.read_csv(meta_path, sep="\t")
    matrix = np.load(npz_path, allow_pickle=False)["embeddings"].astype(np.float32)
    if len(meta) != matrix.shape[0]:
        raise ValueError(
            f"Metadata/embedding row mismatch under {path}: {len(meta)} vs {matrix.shape[0]}"
        )
    if "proteome_id" not in meta.columns:
        if "sample" not in meta.columns:
            raise ValueError(f"{meta_path} lacks proteome_id/sample")
        meta["proteome_id"] = meta["sample"].astype(str)
    if "sample" not in meta.columns:
        meta["sample"] = meta["proteome_id"].astype(str)
    meta["source_artifact_dir"] = str(path)
    return meta, matrix


def build(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    frames: list[pd.DataFrame] = []
    matrices: list[np.ndarray] = []
    shard_rows: list[dict[str, Any]] = []
    for shard_idx, raw_dir in enumerate(args.artifact_dir, start=1):
        artifact_dir = raw_dir.resolve()
        meta, matrix = load_artifact_dir(artifact_dir)
        meta["consolidated_shard_order"] = shard_idx
        frames.append(meta)
        matrices.append(matrix)
        shard_rows.append(
            {
                "shard_order": shard_idx,
                "artifact_dir": str(artifact_dir),
                "metadata_rows": int(len(meta)),
                "embedding_rows": int(matrix.shape[0]),
                "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else "",
                "status": "pass",
            }
        )

    meta = pd.concat(frames, ignore_index=True, sort=False)
    matrix = np.vstack(matrices).astype(np.float32)
    if meta["proteome_id"].astype(str).duplicated().any():
        duplicates = (
            meta.loc[meta["proteome_id"].astype(str).duplicated(), "proteome_id"]
            .astype(str)
            .head(20)
            .tolist()
        )
        raise ValueError(f"Duplicate proteome_id values across shards: {duplicates}")
    if len(meta) != matrix.shape[0]:
        raise ValueError(f"Consolidated row mismatch: metadata={len(meta)} embeddings={matrix.shape[0]}")
    if not np.isfinite(matrix).all():
        raise ValueError("Consolidated embedding matrix contains non-finite values")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "genome_embeddings.npz",
        embeddings=matrix,
        sample=meta["sample"].astype(str).to_numpy(),
        proteome_id=meta["proteome_id"].astype(str).to_numpy(),
        mag_id=meta.get("mag_id", pd.Series([""] * len(meta))).astype(str).to_numpy(),
        source=meta.get("source", pd.Series(["unknown"] * len(meta))).astype(str).to_numpy(),
        ecosystem=meta.get("ecosystem", pd.Series(["unknown"] * len(meta))).astype(str).to_numpy(),
        domain=meta.get("domain", pd.Series(["Unknown"] * len(meta))).astype(str).to_numpy(),
        source_group=meta.get("source_group", pd.Series([""] * len(meta))).astype(str).to_numpy(),
        n_proteins_used=meta.get("n_proteins_used", pd.Series([0] * len(meta))).fillna(0).astype(int).to_numpy(),
    )
    meta.to_csv(output_dir / "embedding_metadata.tsv", sep="\t", index=False)
    write_tsv(
        output_dir / "consolidated_shard_manifest.tsv",
        shard_rows,
        ["shard_order", "artifact_dir", "metadata_rows", "embedding_rows", "embedding_dim", "status"],
    )
    summary = {
        "artifact_dirs": [str(path.resolve()) for path in args.artifact_dir],
        "output_dir": str(output_dir),
        "metadata_rows": int(len(meta)),
        "embedding_shape": list(matrix.shape),
        "finite_embeddings": bool(np.isfinite(matrix).all()),
        "duplicate_proteome_ids": 0,
        "claim_boundary": (
            "Consolidated ESM2 embeddings are MAG/proteome-level molecular similarity evidence only; "
            "they do not imply measured methane flux, final MRV risk tiers, carbon-credit approval, "
            "or source-independent ecological transfer."
        ),
    }
    (output_dir / "embedding_stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
