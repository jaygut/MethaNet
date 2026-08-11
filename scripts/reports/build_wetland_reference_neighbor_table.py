#!/usr/bin/env python3
"""Build source-aware ESM2 neighbor tables for a wetland reference lane.

The global k-nearest-neighbour view is useful for local geometry, but it can
hide cross-source candidates when one large source dominates the neighbourhood.
This builder therefore also emits source-stratified nearest neighbours.  That
keeps wetland, rumen, and mangrove references visible side by side without
turning a similarity result into a transfer or process-rate claim.
"""

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
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{path} has a non-finite or non-matrix embedding payload")
    meta["artifact_dir"] = str(path)
    return meta, matrix.astype(np.float32, copy=False)


def top_similarity_indices(scores: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Return descending top-k column positions and their cosine similarities."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    positions = np.argpartition(scores, -n_neighbors, axis=1)[:, -n_neighbors:]
    values = np.take_along_axis(scores, positions, axis=1)
    order = np.argsort(-values, axis=1)
    return (
        np.take_along_axis(positions, order, axis=1),
        np.take_along_axis(values, order, axis=1),
    )


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

    if not np.isfinite(matrix).all():
        raise SystemExit("Refusing to construct neighbours from non-finite embeddings.")
    target_sources = args.target_source or sorted(meta["source"].astype(str).unique())
    x = matrix
    n_global = min(args.k, len(meta) - 1)
    if n_global < 1:
        raise SystemExit("Need at least two embedded proteomes to build neighbours.")
    source_result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if args.metric == "cosine":
        norms = np.linalg.norm(x, axis=1)
        if (norms == 0).any():
            raise ValueError("Refusing to construct cosine neighbours from zero-norm embeddings.")
        x = x / norms[:, None]
        global_scores = x[query_idx] @ x.T
        global_scores[np.arange(len(query_idx)), query_idx] = -np.inf
        indices, similarities = top_similarity_indices(global_scores, n_global)
        distances = 1.0 - similarities
        for target_source in target_sources:
            candidate_idx = np.flatnonzero(meta["source"].astype(str).eq(target_source).to_numpy())
            if len(candidate_idx) == 0:
                continue
            scores = x[query_idx] @ x[candidate_idx].T
            if target_source in set(meta.iloc[query_idx]["source"].astype(str)):
                candidate_positions = {global_i: local_i for local_i, global_i in enumerate(candidate_idx)}
                for query_row, global_i in enumerate(query_idx):
                    local_i = candidate_positions.get(int(global_i))
                    if local_i is not None:
                        scores[query_row, local_i] = -np.inf
            count = min(args.per_source_k, len(candidate_idx) - (1 if target_source == str(meta.iloc[query_idx[0]].get("source", "")) else 0))
            if count > 0:
                local_indices, source_similarities = top_similarity_indices(scores, count)
                source_result[target_source] = (
                    candidate_idx,
                    local_indices,
                    1.0 - source_similarities,
                )
    else:
        nn = NearestNeighbors(n_neighbors=n_global + 1, metric=args.metric, n_jobs=-1)
        nn.fit(x)
        raw_distances, raw_indices = nn.kneighbors(x[query_idx])
        indices = np.empty((len(query_idx), n_global), dtype=int)
        distances = np.empty((len(query_idx), n_global), dtype=float)
        for row, global_i in enumerate(query_idx):
            keep_positions = [i for i, target_i in enumerate(raw_indices[row]) if target_i != global_i][:n_global]
            indices[row] = raw_indices[row, keep_positions]
            distances[row] = raw_distances[row, keep_positions]
        for target_source in target_sources:
            candidate_idx = np.flatnonzero(meta["source"].astype(str).eq(target_source).to_numpy())
            query_exclusion = target_source == str(meta.iloc[query_idx[0]].get("source", ""))
            if query_exclusion:
                candidate_idx = candidate_idx[~np.isin(candidate_idx, query_idx)]
            count = min(args.per_source_k, len(candidate_idx))
            if count < 1:
                continue
            source_nn = NearestNeighbors(n_neighbors=count, metric=args.metric, n_jobs=-1)
            source_nn.fit(x[candidate_idx])
            source_distances, local_indices = source_nn.kneighbors(x[query_idx])
            source_result[target_source] = (candidate_idx, local_indices, source_distances)

    edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for local_i, source_i in enumerate(query_idx):
        source = meta.iloc[source_i]
        observed = 0
        cross_source = 0
        cross_ecosystem = 0
        nearest_by_source: dict[str, tuple[str, float]] = {}
        for observed, (target_i, distance) in enumerate(
            zip(indices[local_i], distances[local_i]), start=1
        ):
            target = meta.iloc[target_i]
            distance = float(distance)
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
                    "neighbor_scope": "global_knn",
                    "target_source_rank": "",
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

        # Build a comparison set for every requested source. A source-specific
        # search prevents the 2,501-member MUCC v1 lane from crowding out the
        # smaller rumen reference and external mangrove lanes.
        for target_source, (candidate_idx, local_indices, source_distances) in source_result.items():
            for source_rank, (distance, local_target_i) in enumerate(
                zip(source_distances[local_i], local_indices[local_i]), start=1
            ):
                    target_i = int(candidate_idx[int(local_target_i)])
                    target = meta.iloc[target_i]
                    similarity = (
                        1.0 - float(distance)
                        if args.metric == "cosine"
                        else float(np.exp(-float(distance)))
                    )
                    target_source_value = str(target.get("source", ""))
                    if source_rank == 1:
                        nearest_by_source[target_source_value] = (
                            str(target["proteome_id"]),
                            similarity,
                        )
                    edge_rows.append(
                        {
                            "query_proteome_id": source["proteome_id"],
                            "query_mag_id": source.get("mag_id", ""),
                            "query_source": source.get("source", ""),
                            "query_ecosystem": source.get("ecosystem", ""),
                            "query_domain": source.get("domain", ""),
                            "neighbor_scope": "source_stratified_knn",
                            "target_source_rank": source_rank,
                            "neighbor_rank": "",
                            "neighbor_proteome_id": target["proteome_id"],
                            "neighbor_mag_id": target.get("mag_id", ""),
                            "neighbor_source": target_source_value,
                            "neighbor_ecosystem": target.get("ecosystem", ""),
                            "neighbor_domain": target.get("domain", ""),
                            "distance": float(distance),
                            "similarity": similarity,
                            "cross_source_neighbor": target_source_value
                            != str(source.get("source", "")),
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
        "neighbor_scope",
        "target_source_rank",
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
        "per_source_k": int(args.per_source_k),
        "target_sources": target_sources,
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
    parser.add_argument(
        "--per-source-k",
        type=int,
        default=5,
        help="Additional nearest neighbours to retain for each source lane (0 disables).",
    )
    parser.add_argument(
        "--target-source",
        action="append",
        default=[],
        help="Restrict source-stratified neighbours to this source; repeatable.",
    )
    parser.add_argument("--metric", default="cosine")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
