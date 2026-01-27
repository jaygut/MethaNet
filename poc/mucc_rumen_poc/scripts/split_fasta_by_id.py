#!/usr/bin/env python3
"""Split a multi-FASTA file into per-ID FASTA files."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_fasta(path: Path):
    header = None
    seq_parts = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, "".join(seq_parts)
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header:
            yield header, "".join(seq_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--ids", required=True, help="File with one ID per line")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ids = {line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for seq_id, seq in parse_fasta(Path(args.fasta)):
        if seq_id not in ids:
            continue
        out_path = out_dir / f"{seq_id}.fasta"
        out_path.write_text(f">{seq_id}\n{seq}\n")
        written += 1

    print(f"Wrote {written} FASTA files to {out_dir}")


if __name__ == "__main__":
    main()
