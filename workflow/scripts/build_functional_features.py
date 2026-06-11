import argparse
import csv
import math
from pathlib import Path

from methanet.schema import FUNCTIONAL_MARKERS, FUNCTIONAL_RATIO_FEATURE


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
    parser.add_argument("--output", required=True)
    parser.add_argument("--evalue-threshold", type=float, default=1e-10)

    # Marker inputs
    parser.add_argument("--mcrA", required=True)
    parser.add_argument("--mcrB", required=False)
    parser.add_argument("--mcrG", required=False)
    parser.add_argument("--pmoA", required=True)
    parser.add_argument("--mmoX", required=False)
    parser.add_argument("--dsrA", required=True)
    parser.add_argument("--dsrB", required=False)
    parser.add_argument("--nifH", required=True)
    parser.add_argument("--cbbL", required=True)
    parser.add_argument("--mtaB", required=False)
    parser.add_argument("--mttB", required=False)
    parser.add_argument("--mtbA", required=False)

    args = parser.parse_args()

    total_proteins = count_proteins(Path(args.proteins))
    normalization_factor = total_proteins / 1000 if total_proteins > 0 else 1

    markers = list(FUNCTIONAL_MARKERS)

    counts = {}
    for marker in markers:
        # Get path from args if present
        path_str = getattr(args, marker, None)
        if path_str:
            counts[marker] = count_hits(Path(path_str), args.evalue_threshold)
        else:
            counts[marker] = 0

    normalized = {k: v / normalization_factor for k, v in counts.items()}

    # Robust ratio calculation
    pseudocount = 1e-6
    # Use primary markers for the main ratio, consistent with FunctionalProfile
    ratio = math.log2(
        (normalized["mcrA"] + pseudocount)
        / (normalized["pmoA"] + pseudocount)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as handle:
        fieldnames = ["sample_id"] + markers + [FUNCTIONAL_RATIO_FEATURE]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()

        row = {"sample_id": args.sample_id, FUNCTIONAL_RATIO_FEATURE: ratio}
        row.update(normalized)

        writer.writerow(row)


if __name__ == "__main__":
    main()
