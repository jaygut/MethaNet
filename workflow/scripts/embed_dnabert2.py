import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def read_fasta(path: Path) -> list[str]:
    sequences = []
    current = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            sequences.append("".join(current))
    return sequences


def batch_iter(items: list[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _write_metrics(output_path: Path, metrics: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DNABERT-2 embeddings.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "length_weighted"],
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional path to write per-sample embedding metrics JSON.",
    )
    args = parser.parse_args()

    sequences = read_fasta(Path(args.input))

    lengths_bp = np.array([len(seq) for seq in sequences], dtype=np.int64)
    token_lengths = np.maximum(lengths_bp - args.kmer_size + 1, 0)
    effective_token_lengths = token_lengths + 2
    observed_token_lengths = np.minimum(effective_token_lengths, args.max_length)
    truncated_flags = effective_token_lengths > args.max_length
    truncated_tokens = np.maximum(effective_token_lengths - args.max_length, 0)

    metrics_path = None
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)

    if not sequences:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        embedding = np.full(config.hidden_size, np.nan, dtype=np.float32)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, embedding)

        if metrics_path is not None:
            metrics = {
                "n_contigs": 0,
                "total_bp": 0,
                "mean_bp": None,
                "median_bp": None,
                "p90_bp": None,
                "max_bp": None,
                "kmer_size": int(args.kmer_size),
                "max_length": int(args.max_length),
                "n_truncated": 0,
                "frac_truncated": None,
                "mean_truncated_tokens": None,
                "pooling": args.pooling,
            }
            _write_metrics(metrics_path, metrics)
        return

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)
    model.eval()
    embedding_dim = model.config.hidden_size

    sum_embedding = None
    count = 0
    weight_sum = 0.0
    with torch.no_grad():
        for batch in batch_iter(sequences, args.batch_size):
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            pooled = outputs.last_hidden_state.mean(dim=1)
            pooled_np = pooled.cpu().numpy()

            if args.pooling == "length_weighted":
                batch_lengths = np.array([len(seq) for seq in batch], dtype=np.int64)
                batch_token_lengths = np.maximum(batch_lengths - args.kmer_size + 1, 0)
                batch_effective = batch_token_lengths + 2
                batch_observed = np.minimum(batch_effective, args.max_length)
                weights = batch_observed.astype(np.float64)
                if sum_embedding is None:
                    sum_embedding = (pooled_np * weights[:, None]).sum(axis=0)
                else:
                    sum_embedding += (pooled_np * weights[:, None]).sum(axis=0)
                weight_sum += float(weights.sum())
            else:
                if sum_embedding is None:
                    sum_embedding = pooled_np.sum(axis=0)
                else:
                    sum_embedding += pooled_np.sum(axis=0)
            count += pooled_np.shape[0]

    if (args.pooling == "length_weighted" and weight_sum <= 0) or (
        args.pooling != "length_weighted" and count == 0
    ):
        embedding = np.full(embedding_dim, np.nan, dtype=np.float32)
    else:
        denom = weight_sum if args.pooling == "length_weighted" else float(count)
        embedding = (sum_embedding / denom).astype(np.float32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embedding)

    if metrics_path is not None:
        n_contigs = int(len(sequences))
        n_truncated = int(truncated_flags.sum())
        metrics = {
            "n_contigs": n_contigs,
            "total_bp": int(lengths_bp.sum()) if n_contigs else 0,
            "mean_bp": float(lengths_bp.mean()) if n_contigs else None,
            "median_bp": float(np.median(lengths_bp)) if n_contigs else None,
            "p90_bp": float(np.percentile(lengths_bp, 90)) if n_contigs else None,
            "max_bp": int(lengths_bp.max()) if n_contigs else None,
            "kmer_size": int(args.kmer_size),
            "max_length": int(args.max_length),
            "n_truncated": n_truncated,
            "frac_truncated": (n_truncated / n_contigs) if n_contigs else None,
            "mean_truncated_tokens": (
                float(truncated_tokens[truncated_flags].mean()) if n_truncated else 0.0
            ),
            "mean_observed_tokens": float(observed_token_lengths.mean())
            if n_contigs
            else None,
            "pooling": args.pooling,
        }
        _write_metrics(metrics_path, metrics)


if __name__ == "__main__":
    main()
