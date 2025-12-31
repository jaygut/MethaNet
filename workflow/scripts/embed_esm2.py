import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


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
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    sequences = read_fasta(Path(args.input))
    if not sequences:
        raise ValueError("No sequences found in input FASTA.")

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for batch in batch_iter(sequences, args.batch_size):
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_special_tokens_mask=True,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            attention = tokens["attention_mask"]
            special = tokens.get("special_tokens_mask")
            if special is None:
                mask = attention
            else:
                mask = attention * (1 - special)
            mask = mask.unsqueeze(-1)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())

    array = np.vstack(embeddings)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)


if __name__ == "__main__":
    main()
