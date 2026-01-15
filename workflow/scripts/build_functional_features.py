import argparse
import csv
from pathlib import Path
import math


def count_hits(path: Path, evalue_threshold: float) -> int:
    targets = set()
    with path.open() as handle:
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
    return len(targets)


def count_proteins(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build functional marker features.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--proteins", required=True)
    parser.add_argument("--mcrA", required=True)
    parser.add_argument("--pmoA", required=True)
    parser.add_argument("--dsrA", required=True)
    parser.add_argument("--nifH", required=True)
    parser.add_argument("--cbbL", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evalue-threshold", type=float, default=1e-10)
    args = parser.parse_args()

    total_proteins = count_proteins(Path(args.proteins))
    normalization_factor = total_proteins / 1000 if total_proteins > 0 else 1

    counts = {
        "mcrA": count_hits(Path(args.mcrA), args.evalue_threshold),
        "pmoA": count_hits(Path(args.pmoA), args.evalue_threshold),
        "dsrA": count_hits(Path(args.dsrA), args.evalue_threshold),
        "nifH": count_hits(Path(args.nifH), args.evalue_threshold),
        "cbbL": count_hits(Path(args.cbbL), args.evalue_threshold),
    }

    normalized = {k: v / normalization_factor for k, v in counts.items()}
    pseudocount = 1e-6
    ratio = math.log2((normalized["mcrA"] + pseudocount) / (normalized["pmoA"] + pseudocount))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "mcrA",
                "pmoA",
                "dsrA",
                "nifH",
                "cbbL",
                "mcrA_pmoA_ratio",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": args.sample_id,
                "mcrA": normalized["mcrA"],
                "pmoA": normalized["pmoA"],
                "dsrA": normalized["dsrA"],
                "nifH": normalized["nifH"],
                "cbbL": normalized["cbbL"],
                "mcrA_pmoA_ratio": ratio,
            }
        )


if __name__ == "__main__":
    main()
