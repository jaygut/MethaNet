#!/usr/bin/env python3
"""Blue Catalyst functional-genomics batch pipeline helpers.

This script intentionally separates the workflow into three phases:
1) plan: build frozen embedding index + canonical IDs + batch manifests
2) process-batch: quantify one batch of MAG proteomes via HMM markers
3) merge: reconcile batch outputs against frozen embeddings and emit model table
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from methanet.functional.quantify import FunctionalQuantifier

REQUIRED_EMBED_COLS = {"sample", "source", "ecosystem", "domain", "proteome_faa"}
MARKER_COLS = [
    "mcrA",
    "mcrB",
    "mcrG",
    "pmoA",
    "mmoX",
    "dsrA",
    "dsrB",
    "nifH",
    "cbbL",
    "mtaB",
    "mttB",
    "mtbA",
    "mcrA_pmoA_ratio",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_id(sample: Any) -> str:
    return str(sample).strip()


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def run_plan(args: argparse.Namespace) -> int:
    embedding_metadata_fp = Path(args.embedding_metadata)
    embedding_npz_fp = Path(args.embedding_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_df = _read_tsv(embedding_metadata_fp)
    missing_cols = sorted(REQUIRED_EMBED_COLS - set(meta_df.columns))
    if missing_cols:
        raise SystemExit(
            "embedding_metadata.tsv missing required columns: "
            + ", ".join(missing_cols)
        )

    embeddings = np.load(embedding_npz_fp)
    if "embeddings" not in embeddings:
        raise SystemExit("embedding npz does not contain 'embeddings' key")
    emb_arr = embeddings["embeddings"]

    if emb_arr.ndim != 2:
        raise SystemExit("embeddings array must be 2D")
    if emb_arr.shape[0] != meta_df.shape[0]:
        raise SystemExit(
            f"embedding row mismatch: npz rows={emb_arr.shape[0]} vs metadata rows={meta_df.shape[0]}"
        )

    meta_df = meta_df.copy()
    meta_df["canonical_genome_id"] = meta_df["sample"].map(_canonical_id)
    duplicated = meta_df["canonical_genome_id"].duplicated(keep=False)
    if duplicated.any():
        dup_ids = sorted(meta_df.loc[duplicated, "canonical_genome_id"].unique().tolist())
        raise SystemExit("duplicate canonical_genome_id values in embedding metadata: " + ", ".join(dup_ids[:10]))

    meta_df["embedding_index"] = np.arange(meta_df.shape[0], dtype=int)
    meta_df["embedding_run_id"] = args.embedding_run_id
    meta_df["embedding_npz_path"] = str(embedding_npz_fp)
    meta_df["embedding_metadata_path"] = str(embedding_metadata_fp)

    if args.hash_proteomes:
        hashes: list[str] = []
        for fp in meta_df["proteome_faa"].astype(str):
            p = Path(fp)
            hashes.append(_sha256(p) if p.exists() else "")
        meta_df["proteome_sha256"] = hashes
    else:
        meta_df["proteome_sha256"] = ""

    embedding_index_cols = [
        "canonical_genome_id",
        "sample",
        "source",
        "ecosystem",
        "domain",
        "proteome_faa",
        "proteome_sha256",
        "embedding_index",
        "embedding_run_id",
        "embedding_npz_path",
        "embedding_metadata_path",
    ]
    if "n_proteins_used" in meta_df.columns:
        embedding_index_cols.insert(6, "n_proteins_used")

    embedding_index_df = meta_df[embedding_index_cols].copy()
    _write_tsv(embedding_index_df, out_dir / "embedding_index_frozen.tsv")

    canonical_map_df = meta_df[["sample", "canonical_genome_id", "source", "ecosystem", "domain"]].copy()
    _write_tsv(canonical_map_df, out_dir / "canonical_id_map.tsv")

    manifest_df = embedding_index_df.copy()
    manifest_df["batch_id"] = (manifest_df["embedding_index"] // max(1, args.batch_size)).astype(int)
    manifest_df["batch_name"] = manifest_df["batch_id"].map(lambda x: f"batch_{x:04d}")
    manifest_df["batch_row_index"] = manifest_df.groupby("batch_id").cumcount()
    _write_tsv(manifest_df, out_dir / "fg_processing_manifest.tsv")

    batches_dir = out_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = []
    for batch_id, batch_df in manifest_df.groupby("batch_id", sort=True):
        batch_name = f"batch_{int(batch_id):04d}"
        batch_manifest = batches_dir / f"{batch_name}.tsv"
        _write_tsv(batch_df, batch_manifest)
        batch_rows.append(
            {
                "batch_id": int(batch_id),
                "batch_name": batch_name,
                "n_mags": int(batch_df.shape[0]),
                "batch_manifest": str(batch_manifest),
            }
        )

    batch_plan_df = pd.DataFrame(batch_rows).sort_values("batch_id").reset_index(drop=True)
    _write_tsv(batch_plan_df, out_dir / "fg_batch_plan.tsv")

    summary = {
        "embedding_run_id": args.embedding_run_id,
        "output_dir": str(out_dir),
        "n_embeddings": int(embedding_index_df.shape[0]),
        "batch_size": int(args.batch_size),
        "n_batches": int(batch_plan_df.shape[0]),
        "hash_proteomes": bool(args.hash_proteomes),
    }
    (out_dir / "fg_plan_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Plan complete: {out_dir}")
    print(f"[OK] n_embeddings={summary['n_embeddings']} n_batches={summary['n_batches']}")
    return 0


def run_process_batch(args: argparse.Namespace) -> int:
    batch_manifest_fp = Path(args.batch_manifest)
    output_features_fp = Path(args.output_features)
    output_failures_fp = Path(args.output_failures)

    batch_df = _read_tsv(batch_manifest_fp)
    if batch_df.empty:
        raise SystemExit(f"batch manifest is empty: {batch_manifest_fp}")

    required_cols = {"canonical_genome_id", "sample", "proteome_faa", "source", "ecosystem", "domain"}
    missing_cols = sorted(required_cols - set(batch_df.columns))
    if missing_cols:
        raise SystemExit("batch manifest missing required columns: " + ", ".join(missing_cols))

    quantifier = FunctionalQuantifier(
        hmm_dir=Path(args.hmm_dir),
        evalue_threshold=args.evalue_threshold,
        score_threshold=args.score_threshold,
        threads=args.threads,
    )

    feature_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for row in batch_df.itertuples(index=False):
        canonical_id = str(row.canonical_genome_id)
        sample = str(row.sample)
        proteome_fp = Path(str(row.proteome_faa))
        if not proteome_fp.exists():
            failure_rows.append(
                {
                    "canonical_genome_id": canonical_id,
                    "sample": sample,
                    "proteome_faa": str(proteome_fp),
                    "error_type": "missing_proteome",
                    "error_message": "proteome file not found",
                }
            )
            continue

        try:
            profile = quantifier.quantify_mag(proteome_fp, sample_id=sample)
            row_payload = profile.to_dict()
            row_payload.update(
                {
                    "canonical_genome_id": canonical_id,
                    "sample": sample,
                    "source": str(row.source),
                    "ecosystem": str(row.ecosystem),
                    "domain": str(row.domain),
                    "proteome_faa": str(proteome_fp),
                    "proteome_sha256": str(getattr(row, "proteome_sha256", "")),
                    "batch_id": int(getattr(row, "batch_id", -1)),
                    "batch_name": str(getattr(row, "batch_name", "")),
                }
            )
            feature_rows.append(row_payload)
        except Exception as exc:  # pragma: no cover - external tool/runtime behavior
            failure_rows.append(
                {
                    "canonical_genome_id": canonical_id,
                    "sample": sample,
                    "proteome_faa": str(proteome_fp),
                    "error_type": "quantification_error",
                    "error_message": str(exc),
                }
            )

    features_df = pd.DataFrame(feature_rows)
    if features_df.empty:
        features_df = pd.DataFrame(
            columns=[
                "canonical_genome_id",
                "sample",
                "source",
                "ecosystem",
                "domain",
                "proteome_faa",
                "proteome_sha256",
                "batch_id",
                "batch_name",
                "methanogenic_potential",
                *MARKER_COLS,
            ]
        )

    failures_df = pd.DataFrame(failure_rows)
    if failures_df.empty:
        failures_df = pd.DataFrame(
            columns=[
                "canonical_genome_id",
                "sample",
                "proteome_faa",
                "error_type",
                "error_message",
            ]
        )

    _write_tsv(features_df, output_features_fp)
    _write_tsv(failures_df, output_failures_fp)

    print(
        "[OK] Batch processed"
        f" manifest={batch_manifest_fp}"
        f" features={features_df.shape[0]} failures={failures_df.shape[0]}"
    )
    return 0


def run_merge(args: argparse.Namespace) -> int:
    fg_plan_dir = Path(args.fg_plan_dir)
    batch_results_dir = Path(args.batch_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_index_fp = fg_plan_dir / "embedding_index_frozen.tsv"
    batch_plan_fp = fg_plan_dir / "fg_batch_plan.tsv"
    if not embedding_index_fp.exists():
        raise SystemExit(f"missing embedding index: {embedding_index_fp}")
    if not batch_plan_fp.exists():
        raise SystemExit(f"missing batch plan: {batch_plan_fp}")

    embedding_df = _read_tsv(embedding_index_fp)
    batch_plan_df = _read_tsv(batch_plan_fp)

    feature_frames = []
    failure_frames = []
    for row in batch_plan_df.itertuples(index=False):
        batch_name = str(row.batch_name)
        feature_fp = batch_results_dir / batch_name / "fg_features.tsv"
        failure_fp = batch_results_dir / batch_name / "fg_failures.tsv"
        if feature_fp.exists():
            feature_frames.append(_read_tsv(feature_fp))
        if failure_fp.exists():
            failure_frames.append(_read_tsv(failure_fp))

    if not feature_frames:
        raise SystemExit("no batch feature outputs were found; cannot merge")

    feature_frames = [frame for frame in feature_frames if not frame.empty]
    failure_frames = [frame for frame in failure_frames if not frame.empty]
    if not feature_frames:
        raise SystemExit("all batch feature outputs were empty; cannot merge")

    features_df = pd.concat(feature_frames, ignore_index=True)
    failures_df = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(
            columns=[
                "canonical_genome_id",
                "sample",
                "proteome_faa",
                "error_type",
                "error_message",
            ]
        )
    )

    duplicate_mask = features_df["canonical_genome_id"].duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        features_df = features_df.loc[~duplicate_mask].copy()

    merge_df = embedding_df.merge(
        features_df,
        on="canonical_genome_id",
        how="left",
        suffixes=("_embedding", "_functional"),
    )
    merge_df["has_functional_profile"] = merge_df["mcrA"].notna()

    n_embeddings = int(embedding_df.shape[0])
    n_functional = int(features_df.shape[0])
    n_joined = int(merge_df["has_functional_profile"].sum())
    n_embedding_only = int((~merge_df["has_functional_profile"]).sum())
    n_failures = int(failures_df.shape[0])
    join_coverage = (n_joined / n_embeddings) if n_embeddings else 0.0

    report = {
        "n_embeddings": n_embeddings,
        "n_functional_profiles": n_functional,
        "n_joined": n_joined,
        "n_embedding_only": n_embedding_only,
        "n_failures": n_failures,
        "duplicate_functional_ids": duplicate_count,
        "join_coverage": join_coverage,
        "min_join_coverage": float(args.min_join_coverage),
    }

    _write_tsv(features_df, output_dir / "fg_functional_features.tsv")
    _write_tsv(failures_df, output_dir / "fg_failures.tsv")
    _write_tsv(merge_df, output_dir / "fg_embedding_join.tsv")
    _write_tsv(pd.DataFrame([report]), output_dir / "id_reconciliation_report.tsv")
    (output_dir / "id_reconciliation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    matrix_cols = [
        "canonical_genome_id",
        "source_embedding",
        "ecosystem_embedding",
        "domain_embedding",
        "embedding_index",
        *MARKER_COLS,
        "methanogenic_potential",
    ]
    matrix_cols = [c for c in matrix_cols if c in merge_df.columns]
    matrix_df = merge_df.loc[merge_df["has_functional_profile"], matrix_cols].copy()
    _write_tsv(matrix_df, output_dir / "modeling_feature_matrix.tsv")

    artifact_manifest = {
        "fg_plan_dir": str(fg_plan_dir),
        "batch_results_dir": str(batch_results_dir),
        "outputs": [
            "fg_functional_features.tsv",
            "fg_failures.tsv",
            "fg_embedding_join.tsv",
            "id_reconciliation_report.tsv",
            "id_reconciliation_report.json",
            "modeling_feature_matrix.tsv",
        ],
    }
    (output_dir / "modeling_feature_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2),
        encoding="utf-8",
    )

    if join_coverage < float(args.min_join_coverage):
        raise SystemExit(
            f"join coverage below threshold: {join_coverage:.4f} < {float(args.min_join_coverage):.4f}"
        )

    print(
        "[OK] Merge complete "
        f"n_embeddings={n_embeddings} n_functional={n_functional} "
        f"coverage={join_coverage:.3f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blue Catalyst FG batch planning/processing/merge pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Build frozen embedding index and batch manifests")
    plan.add_argument("--embedding-metadata", required=True)
    plan.add_argument("--embedding-npz", required=True)
    plan.add_argument("--embedding-run-id", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--batch-size", type=int, default=25)
    plan.add_argument("--hash-proteomes", action="store_true")
    plan.set_defaults(func=run_plan)

    proc = subparsers.add_parser("process-batch", help="Quantify one FG batch")
    proc.add_argument("--batch-manifest", required=True)
    proc.add_argument("--hmm-dir", required=True)
    proc.add_argument("--output-features", required=True)
    proc.add_argument("--output-failures", required=True)
    proc.add_argument("--threads", type=int, default=4)
    proc.add_argument("--evalue-threshold", type=float, default=1e-10)
    proc.add_argument("--score-threshold", type=float, default=50.0)
    proc.set_defaults(func=run_process_batch)

    merge = subparsers.add_parser("merge", help="Merge batch results and reconcile with embeddings")
    merge.add_argument("--fg-plan-dir", required=True)
    merge.add_argument("--batch-results-dir", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--min-join-coverage", type=float, default=0.95)
    merge.set_defaults(func=run_merge)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
