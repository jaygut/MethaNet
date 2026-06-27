#!/usr/bin/env python3
"""Build source-aware ESM2 neighbor tables for a wetland reference lane."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


CLAIM_WORDING = (
    "ESM2 nearest-neighbor evidence is MAG/proteome-level molecular similarity for "
    "review prioritization only; it is not measured methane flux, a final MRV risk "
    "score, a carbon-crediting claim, or proof of ecological transfer."
)


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
        raise FileNotFoundError(f"artifact dir missing embedding outputs: {path}")
    meta = pd.read_csv(meta_path, sep="\t").copy()
    if "proteome_id" not in meta.columns:
        if "sample" not in meta.columns:
            raise ValueError(f"{meta_path} lacks both proteome_id and sample columns")
        meta["proteome_id"] = meta["sample"].astype(str)
    if "sample" not in meta.columns:
        meta["sample"] = meta["proteome_id"].astype(str)
    for col in ["source", "ecosystem", "domain", "mag_id", "source_group"]:
        if col not in meta.columns:
            meta[col] = ""
    matrix = np.load(npz_path, allow_pickle=False)["embeddings"]
    if matrix.shape[0] != len(meta):
        raise ValueError(
            f"{path} row mismatch: metadata={len(meta)} embeddings={matrix.shape[0]}"
        )
    meta["artifact_dir"] = str(path)
    return meta, matrix.astype(np.float32, copy=False)


def build(args: argparse.Namespace) -> int:
    artifact_dirs = [path.resolve() for path in args.artifact_dir]
    metas = []
    matrices = []
    for artifact_dir in artifact_dirs:
        meta, matrix = load_artifact_dir(artifact_dir)
        metas.append(meta)
        matrices.append(matrix)
    meta = pd.concat(metas, ignore_index=True)
    matrix = np.vstack(matrices)

    dedupe_key = meta["proteome_id"].astype(str) + "\t" + meta["source"].astype(str)
    keep = ~dedupe_key.duplicated()
    meta = meta.loc[keep].reset_index(drop=True)
    matrix = matrix[keep.to_numpy()]
    if len(meta) < 2:
        raise SystemExit("Need at least two embedded proteomes to build neighbors.")

    query_mask = meta["source"].astype(str).isin(args.query_source)
    if args.query_prefix:
        query_mask = query_mask | meta["proteome_id"].astype(str).str.startswith(
            tuple(args.query_prefix)
        )
    query_idx = np.flatnonzero(query_mask.to_numpy())
    if len(query_idx) == 0:
        raise SystemExit(
            "No query rows matched --query-source/--query-prefix; wait for MUCC ESM2 artifacts or adjust filters."
        )

    x = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    n_neighbors = min(args.k + 1, len(meta))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=args.metric)
    nn.fit(x)
    distances, indices = nn.kneighbors(x[query_idx])

    edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for local_i, source_i in enumerate(query_idx):
        source = meta.iloc[source_i]
        observed = 0
        cross_source = 0
        cross_ecosystem = 0
        nearest_by_source: dict[str, tuple[str, float]] = {}
        for rank, target_i in enumerate(indices[local_i], start=0):
            if target_i == source_i:
                continue
            observed += 1
            target = meta.iloc[target_i]
            distance = float(distances[local_i][rank])
            similarity = 1.0 - distance if args.metric == "cosine" else float(np.exp(-distance))
            target_source = str(target.get("source", ""))
            if target_source != str(source.get("source", "")):
                cross_source += 1
            if str(target.get("ecosystem", "")) != str(source.get("ecosystem", "")):
                cross_ecosystem += 1
            nearest_by_source.setdefault(target_source, (str(target["proteome_id"]), similarity))
            edge_rows.append(
                {
                    "query_proteome_id": source["proteome_id"],
                    "query_mag_id": source.get("mag_id", ""),
                    "query_source": source.get("source", ""),
                    "query_ecosystem": source.get("ecosystem", ""),
                    "query_domain": source.get("domain", ""),
                    "neighbor_rank": observed,
                    "neighbor_proteome_id": target["proteome_id"],
                    "neighbor_mag_id": target.get("mag_id", ""),
                    "neighbor_source": target.get("source", ""),
                    "neighbor_ecosystem": target.get("ecosystem", ""),
                    "neighbor_domain": target.get("domain", ""),
                    "distance": distance,
                    "similarity": similarity,
                    "cross_source_neighbor": target_source != str(source.get("source", "")),
                    "cross_ecosystem_neighbor": str(target.get("ecosystem", ""))
                    != str(source.get("ecosystem", "")),
                    "query_artifact_dir": source.get("artifact_dir", ""),
                    "neighbor_artifact_dir": target.get("artifact_dir", ""),
                    "allowed_claim_wording": CLAIM_WORDING,
                }
            )
        summary = {
            "proteome_id": source["proteome_id"],
            "mag_id": source.get("mag_id", ""),
            "source": source.get("source", ""),
            "ecosystem": source.get("ecosystem", ""),
            "domain": source.get("domain", ""),
            "knn_k": observed,
            "cross_source_neighbor_count": cross_source,
            "cross_source_neighbor_fraction": cross_source / observed if observed else 0.0,
            "cross_ecosystem_neighbor_count": cross_ecosystem,
            "cross_ecosystem_neighbor_fraction": cross_ecosystem / observed if observed else 0.0,
            "allowed_claim_wording": CLAIM_WORDING,
        }
        for target_source, (target_id, similarity) in sorted(nearest_by_source.items()):
            safe = target_source.replace(" ", "_").replace("/", "_") or "unknown"
            summary[f"nearest_{safe}_proteome_id"] = target_id
            summary[f"nearest_{safe}_similarity"] = similarity
        summary_rows.append(summary)

    output_dir = args.output_dir.resolve()
    edge_fields = [
        "query_proteome_id",
        "query_mag_id",
        "query_source",
        "query_ecosystem",
        "query_domain",
        "neighbor_rank",
        "neighbor_proteome_id",
        "neighbor_mag_id",
        "neighbor_source",
        "neighbor_ecosystem",
        "neighbor_domain",
        "distance",
        "similarity",
        "cross_source_neighbor",
        "cross_ecosystem_neighbor",
        "query_artifact_dir",
        "neighbor_artifact_dir",
        "allowed_claim_wording",
    ]
    summary_fields = sorted({key for row in summary_rows for key in row})
    leading = [
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "knn_k",
        "cross_source_neighbor_count",
        "cross_source_neighbor_fraction",
        "cross_ecosystem_neighbor_count",
        "cross_ecosystem_neighbor_fraction",
        "allowed_claim_wording",
    ]
    summary_fields = leading + [field for field in summary_fields if field not in leading]
    write_tsv(output_dir / "wetland_reference_neighbor_edges.tsv", edge_rows, edge_fields)
    write_tsv(output_dir / "wetland_reference_neighbor_summary.tsv", summary_rows, summary_fields)
    report = {
        "artifact_dirs": [str(path) for path in artifact_dirs],
        "embedded_rows": int(len(meta)),
        "query_rows": int(len(query_idx)),
        "k": int(args.k),
        "metric": args.metric,
        "outputs": {
            "edges": str(output_dir / "wetland_reference_neighbor_edges.tsv"),
            "summary": str(output_dir / "wetland_reference_neighbor_summary.tsv"),
        },
        "claim_boundary": CLAIM_WORDING,
    }
    (output_dir / "wetland_reference_neighbor_build_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-source", action="append", default=["mucc_v1_owc_wetland"])
    parser.add_argument("--query-prefix", action="append", default=["mucc_v1__"])
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--metric", default="cosine")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
