import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DNABERT-2 embeddings.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    sequences = read_fasta(Path(args.input))
    if not sequences:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        embedding = np.zeros(config.hidden_size, dtype=np.float32)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, embedding)
        return

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)
    model.eval()
    embedding_dim = model.config.hidden_size

    sum_embedding = None
    count = 0
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
            if sum_embedding is None:
                sum_embedding = pooled_np.sum(axis=0)
            else:
                sum_embedding += pooled_np.sum(axis=0)
            count += pooled_np.shape[0]

    if count == 0:
        embedding = np.zeros(embedding_dim, dtype=np.float32)
    else:
        embedding = (sum_embedding / count).astype(np.float32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embedding)


if __name__ == "__main__":
    main()
