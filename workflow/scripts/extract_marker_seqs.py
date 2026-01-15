import argparse
from pathlib import Path


def read_fasta(path: Path):
    seq_id = None
    seq_chunks = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_chunks)
                seq_id = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_id is not None:
            yield seq_id, "".join(seq_chunks)


def collect_targets(paths, evalue_threshold: float) -> set[str]:
    targets = set()
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 7:
                    continue
                try:
                    evalue = float(fields[6])
                except ValueError:
                    continue
                if evalue > evalue_threshold:
                    continue
                targets.add(fields[0])
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract marker sequences from ORFs.")
    parser.add_argument("--proteins", required=True)
    parser.add_argument("--hits", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evalue-threshold", type=float, default=1e-10)
    args = parser.parse_args()

    targets = collect_targets(args.hits, args.evalue_threshold)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as handle:
        for seq_id, seq in read_fasta(Path(args.proteins)):
            if seq_id in targets:
                handle.write(f">{seq_id}\n{seq}\n")


if __name__ == "__main__":
    main()
