#!/usr/bin/env python3
"""Build ESM2 genome/proteome embeddings from a manifest of protein FASTA files.

This is the manifest-driven production equivalent of the Blue Catalyst POC
notebook embedding block. It preserves the POC artifact contract while avoiding
notebook/data-acquisition side effects:

- one row per manifest proteome/MAG;
- ESM2 protein embeddings from `facebook/esm2_t33_650M_UR50D` by default;
- sequence truncation at 1022 tokens by default;
- up to 6000 valid proteins embedded per proteome by default;
- genome/proteome embedding = mean of protein embeddings;
- checkpointed batches plus final `genome_embeddings.npz` and
  `embedding_metadata.tsv`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import SeqIO

from methanet.embedding.esm2 import EmbeddingConfig, ESM2Embedder


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-col", default="proteome_id")
    parser.add_argument("--faa-col", default="proteome_faa")
    parser.add_argument("--mag-id-col", default="mag_id")
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--ecosystem-col", default="ecosystem")
    parser.add_argument("--domain-col", default="domain")
    parser.add_argument("--source-group-col", default="source_group")
    parser.add_argument("--protein-count-col", default="protein_count")
    parser.add_argument(
        "--include-col",
        default="",
        help="Optional boolean manifest column used to filter rows before embedding.",
    )
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--max-proteins-per-proteome", type=int, default=6000)
    parser.add_argument("--min-aa-len", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 on CUDA. Default false to match the POC notebook.")
    parser.add_argument("--limit", type=int, help="Optional first-N manifest rows for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Write inventory/stats and exit before loading ESM2.")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def normalize_aa(seq: str) -> str:
    seq = seq.upper().replace("*", "").replace("-", "")
    return "".join(ch if ch in VALID_AA else "X" for ch in seq)


def maybe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    repo_root = args.repo_root.resolve()
    manifest_path = resolve_path(repo_root, args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path, sep="\t")
    for col in [args.id_col, args.faa_col]:
        if col not in df.columns:
            raise ValueError(f"Manifest missing required column `{col}`: {manifest_path}")

    if args.include_col:
        if args.include_col not in df.columns:
            raise ValueError(f"Manifest missing include column `{args.include_col}`: {manifest_path}")
        include = df[args.include_col].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        df = df[include].copy()

    if args.limit is not None:
        df = df.head(max(0, int(args.limit))).copy()

    out = df.copy()
    out["proteome_id"] = out[args.id_col].astype(str)
    out["sample"] = out["proteome_id"]
    out["proteome_faa_resolved"] = out[args.faa_col].map(lambda p: str(resolve_path(repo_root, p)))
    out["proteome_faa_exists"] = out["proteome_faa_resolved"].map(lambda p: Path(p).exists())
    out["proteome_faa_size_bytes"] = out["proteome_faa_resolved"].map(
        lambda p: Path(p).stat().st_size if Path(p).exists() else 0
    )

    if args.protein_count_col in out.columns:
        out["n_proteins_available"] = out[args.protein_count_col].map(maybe_int)
    else:
        out["n_proteins_available"] = None

    out["protein_cap_applies_from_manifest"] = out["n_proteins_available"].fillna(0).astype(int) > int(
        args.max_proteins_per_proteome
    )
    return out


def metadata_from_row(row: Any, args: argparse.Namespace) -> dict[str, Any]:
    def get_col(col: str, default: str = "") -> str:
        if col in row.index and not pd.isna(row[col]):
            return str(row[col])
        return default

    proteome_id = str(row["proteome_id"])
    return {
        "sample": proteome_id,
        "proteome_id": proteome_id,
        "mag_id": get_col(args.mag_id_col, proteome_id),
        "source": get_col(args.source_col, "unknown"),
        "ecosystem": get_col(args.ecosystem_col, "unknown"),
        "domain": get_col(args.domain_col, "Unknown"),
        "source_group": get_col(args.source_group_col, ""),
        "source_analysis_accession": get_col("source_analysis_accession", ""),
        "proteome_faa": str(row["proteome_faa_resolved"]),
        "n_proteins_available": maybe_int(row.get("n_proteins_available")),
    }


def load_sequences(path: Path, *, min_aa_len: int, max_proteins: int) -> tuple[list[str], list[str], int, bool]:
    seqs: list[str] = []
    ids: list[str] = []
    total_valid_seen = 0
    cap_applied = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            seq = normalize_aa(str(rec.seq))
            if len(seq) < min_aa_len:
                continue
            total_valid_seen += 1
            if len(seqs) < max_proteins:
                seqs.append(seq)
                ids.append(str(rec.id))
            else:
                cap_applied = True
    return seqs, ids, total_valid_seen, cap_applied


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def existing_batch_ids(checkpoint_dir: Path) -> list[int]:
    ids: list[int] = []
    for fp in checkpoint_dir.glob("embedding_batch_*.npz"):
        match = re.match(r"embedding_batch_(\d+)\.npz$", fp.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def aggregate_checkpoints(checkpoint_dir: Path, checkpoint_manifest_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    batch_files = sorted(checkpoint_dir.glob("embedding_batch_*.npz"))
    if not batch_files:
        raise RuntimeError(f"No checkpoint batches found in {checkpoint_dir}")

    embeddings: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for batch_npz in batch_files:
        batch_tsv = checkpoint_dir / f"{batch_npz.stem}.tsv"
        if not batch_tsv.exists():
            raise RuntimeError(f"Missing checkpoint TSV for {batch_npz}")
        bundle = np.load(batch_npz, allow_pickle=True)
        batch_emb = bundle["embeddings"].astype(np.float32)
        batch_meta = pd.read_csv(batch_tsv, sep="\t")
        if len(batch_meta) != batch_emb.shape[0]:
            raise RuntimeError(f"Checkpoint row mismatch for {batch_npz}")
        embeddings.append(batch_emb)
        metadata.append(batch_meta)
        manifest_rows.append(
            {
                "batch": batch_npz.stem,
                "rows": int(len(batch_meta)),
                "npz_bytes": int(batch_npz.stat().st_size),
                "tsv_bytes": int(batch_tsv.stat().st_size),
            }
        )

    meta = pd.concat(metadata, ignore_index=True)
    emb = np.vstack(embeddings).astype(np.float32)
    if {"checkpoint_batch", "checkpoint_order"}.issubset(meta.columns):
        order = meta.sort_values(["checkpoint_batch", "checkpoint_order"]).index.to_numpy()
        meta = meta.loc[order].reset_index(drop=True)
        emb = emb[order]
    if meta["sample"].astype(str).duplicated().any():
        dups = meta.loc[meta["sample"].astype(str).duplicated(), "sample"].astype(str).head(10).tolist()
        raise RuntimeError(f"Duplicate checkpoint samples found: {dups}")
    pd.DataFrame(manifest_rows).to_csv(checkpoint_manifest_path, sep="\t", index=False)
    return emb, meta


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve_path(repo_root, args.output_dir)
    checkpoint_dir = output_dir / "embedding_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args)
    inventory_path = output_dir / "embedding_input_inventory.tsv"
    manifest.to_csv(inventory_path, sep="\t", index=False)

    checkpoint_meta_path = checkpoint_dir / "checkpoint_metadata.tsv"
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.tsv"
    partial_stats_path = checkpoint_dir / "embedding_stats_partial.json"

    existing_meta = pd.DataFrame()
    if checkpoint_meta_path.exists():
        existing_meta = pd.read_csv(checkpoint_meta_path, sep="\t")
        if "sample" not in existing_meta.columns:
            raise RuntimeError(f"Existing checkpoint metadata has no sample column: {checkpoint_meta_path}")
        if existing_meta["sample"].astype(str).duplicated().any():
            raise RuntimeError(f"Existing checkpoint metadata contains duplicate samples: {checkpoint_meta_path}")

    processed = set(existing_meta["sample"].astype(str)) if not existing_meta.empty else set()
    pending = manifest[~manifest["sample"].astype(str).isin(processed)].copy()
    batch_ids = existing_batch_ids(checkpoint_dir)

    stats: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(resolve_path(repo_root, args.manifest)),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "max_proteins_per_proteome": int(args.max_proteins_per_proteome),
        "min_aa_len": int(args.min_aa_len),
        "checkpoint_every": int(args.checkpoint_every),
        "device": args.device,
        "fp16": bool(args.fp16),
        "candidate_total": int(len(manifest)),
        "proteome_faa_present": int(manifest["proteome_faa_exists"].sum()),
        "proteome_faa_missing": int((~manifest["proteome_faa_exists"]).sum()),
        "manifest_cap_applies_count": int(manifest["protein_cap_applies_from_manifest"].sum()),
        "pending_initial": int(len(pending)),
        "resumed_preexisting": int(len(processed)),
        "batches_existing": int(len(batch_ids)),
        "embedded_new_this_run": 0,
        "missing_faa": 0,
        "no_valid": 0,
        "empty_embedding": 0,
        "non_finite": 0,
        "capped_by_sequence_scan": 0,
        "dry_run": bool(args.dry_run),
    }
    write_json(partial_stats_path, stats)

    if args.dry_run:
        write_json(output_dir / "embedding_stats.json", stats)
        print(f"[DRY-RUN] Wrote inventory: {inventory_path}")
        print(f"[DRY-RUN] Wrote stats: {output_dir / 'embedding_stats.json'}")
        return 0

    if stats["proteome_faa_missing"]:
        missing = manifest.loc[~manifest["proteome_faa_exists"], ["sample", "proteome_faa_resolved"]].head(20)
        raise RuntimeError(f"{stats['proteome_faa_missing']} FAA files are missing. First rows:\n{missing}")

    emb_cfg = EmbeddingConfig(
        model_name=args.model_name,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        device=args.device,
        fp16=bool(args.fp16),
        cache_dir=args.cache_dir,
    )
    embedder = ESM2Embedder(emb_cfg)

    state: dict[str, Any] = {
        "next_batch_id": (max(batch_ids) + 1) if batch_ids else 1,
        "batch_embeddings": [],
        "batch_rows": [],
    }

    def flush() -> None:
        if not state["batch_embeddings"]:
            return
        batch_id = int(state["next_batch_id"])
        batch_emb = np.vstack(state["batch_embeddings"]).astype(np.float32)
        batch_df = pd.DataFrame(state["batch_rows"])
        batch_df["checkpoint_batch"] = batch_id
        batch_df["checkpoint_order"] = np.arange(len(batch_df), dtype=int)

        npz_path = checkpoint_dir / f"embedding_batch_{batch_id:05d}.npz"
        tsv_path = checkpoint_dir / f"embedding_batch_{batch_id:05d}.tsv"
        np.savez_compressed(
            npz_path,
            embeddings=batch_emb,
            sample=batch_df["sample"].astype(str).values,
            proteome_id=batch_df["proteome_id"].astype(str).values,
        )
        batch_df.to_csv(tsv_path, sep="\t", index=False)
        batch_df.to_csv(
            checkpoint_meta_path,
            sep="\t",
            index=False,
            mode="a",
            header=not checkpoint_meta_path.exists(),
        )
        state["next_batch_id"] = batch_id + 1
        state["batch_embeddings"] = []
        state["batch_rows"] = []
        stats["embedded_new_this_run"] = int(stats["embedded_new_this_run"])
        write_json(partial_stats_path, stats)
        print(f"[CHECKPOINT] batch={batch_id:05d} rows={len(batch_df)} total_new={stats['embedded_new_this_run']}", flush=True)

    for idx, row in enumerate(pending.itertuples(index=False), start=1):
        row_series = pending.iloc[idx - 1]
        meta = metadata_from_row(row_series, args)
        faa = Path(str(row_series["proteome_faa_resolved"]))
        if not faa.exists() or faa.stat().st_size == 0:
            stats["missing_faa"] += 1
            continue

        seqs, ids, total_valid_seen, cap_applied = load_sequences(
            faa,
            min_aa_len=int(args.min_aa_len),
            max_proteins=int(args.max_proteins_per_proteome),
        )
        if cap_applied:
            stats["capped_by_sequence_scan"] += 1
        if not seqs:
            stats["no_valid"] += 1
            continue

        protein_embeddings = embedder.embed_proteins(seqs, ids)
        if not protein_embeddings:
            stats["empty_embedding"] += 1
            continue
        genome_embedding = embedder.embed_genome(protein_embeddings, aggregation="mean").astype(np.float32)
        if not np.isfinite(genome_embedding).all():
            stats["non_finite"] += 1
            continue

        meta.update(
            {
                "n_proteins_used": int(len(seqs)),
                "n_valid_proteins_seen": int(total_valid_seen),
                "protein_cap_applied": bool(cap_applied),
            }
        )
        state["batch_embeddings"].append(genome_embedding)
        state["batch_rows"].append(meta)
        stats["embedded_new_this_run"] += 1

        if idx == 1 or idx % 10 == 0:
            print(
                f"[PROGRESS] pending_idx={idx}/{len(pending)} sample={meta['sample']} "
                f"proteins_used={len(seqs)} cap={cap_applied}",
                flush=True,
            )
        if len(state["batch_rows"]) >= int(args.checkpoint_every):
            flush()

    flush()

    emb, meta = aggregate_checkpoints(checkpoint_dir, checkpoint_manifest_path)
    if len(meta) != emb.shape[0]:
        raise RuntimeError(f"Final aggregation mismatch: metadata={len(meta)} embeddings={emb.shape[0]}")

    stats["embedded_total_with_resume"] = int(emb.shape[0])
    stats["pending_remaining"] = int(max(0, len(manifest) - emb.shape[0]))
    stats["completed_utc"] = datetime.now(timezone.utc).isoformat()

    out_npz = output_dir / "genome_embeddings.npz"
    np.savez_compressed(
        out_npz,
        embeddings=emb,
        sample=meta["sample"].astype(str).values,
        proteome_id=meta["proteome_id"].astype(str).values,
        mag_id=meta.get("mag_id", pd.Series([""] * len(meta))).astype(str).values,
        source=meta.get("source", pd.Series(["unknown"] * len(meta))).astype(str).values,
        ecosystem=meta.get("ecosystem", pd.Series(["unknown"] * len(meta))).astype(str).values,
        domain=meta.get("domain", pd.Series(["Unknown"] * len(meta))).astype(str).values,
        source_group=meta.get("source_group", pd.Series([""] * len(meta))).astype(str).values,
        source_analysis_accession=meta.get("source_analysis_accession", pd.Series([""] * len(meta))).astype(str).values,
        n_proteins_used=meta["n_proteins_used"].astype(int).values,
        n_valid_proteins_seen=meta["n_valid_proteins_seen"].astype(int).values,
        protein_cap_applied=meta["protein_cap_applied"].astype(bool).values,
    )
    meta.to_csv(output_dir / "embedding_metadata.tsv", sep="\t", index=False)
    write_json(output_dir / "embedding_stats.json", stats)

    print(f"[DONE] embeddings={emb.shape} metadata_rows={len(meta)}")
    print(f"[DONE] wrote {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
