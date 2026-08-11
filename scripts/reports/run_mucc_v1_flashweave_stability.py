#!/usr/bin/env python3
"""Quantify MUCC v1 FlashWeave edge-selection stability on scaffolded samples.

This runs deterministic, scaffold-stratified 80% subsamples of the already
defined 133-sample/300-MAG FlashWeave input. It tests edge reproducibility in
the processed expression data; it does not turn scaffolded depth/site labels
into exact ecological covariates or flux validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
ANALYSIS_ID = "flashweave_scaffold_stratified_subsample_stability"
CLAIM_BOUNDARY = (
    "FlashWeave stability from source-processed relative MAG expression and scaffolded "
    "sample labels is exploratory association evidence only. It does not establish an "
    "interaction, exact depth/environment/flux relationship, measured methane flux, final "
    "MRV score/A-E tier, crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--stability-threshold", type=float, default=0.7)
    parser.add_argument(
        "--reconcile-existing",
        action="store_true",
        help=(
            "Validate an existing stability table against the current baseline and repair only "
            "its merged columns in the edge table; do not run new subsamples."
        ),
    )
    parser.add_argument(
        "--julia-bin",
        type=Path,
        default=Path("tmp/mucc_flashweave_julia_generic/bin/julia"),
    )
    parser.add_argument(
        "--julia-project",
        type=Path,
        default=Path("tmp/mucc_flashweave_julia_project"),
    )
    parser.add_argument(
        "--julia-depot",
        type=Path,
        default=Path("tmp/mucc_flashweave_julia_depot"),
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def read_raw_edges(path: Path) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.strip().split("\t")
            if not raw.strip() or raw.startswith("#"):
                continue
            if len(fields) != 3:
                raise ValueError(f"unexpected FlashWeave edge line {line_number}: {raw!r}")
            source, target, weight = fields
            rows.append((source, target, float(weight)))
    return rows


def mag_edge_pairs(path: Path, mag_ids: set[str]) -> set[tuple[str, str]]:
    return {
        tuple(sorted((source, target)))
        for source, target, _ in read_raw_edges(path)
        if source in mag_ids and target in mag_ids
    }


def stratified_sample_indices(
    metadata: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    strata = metadata[["month_label", "site_or_landcover", "depth_context_code"]].astype(str)
    group_keys = strata.agg("|".join, axis=1)
    selected: list[int] = []
    for _, indices in group_keys.groupby(group_keys, sort=True).groups.items():
        values = np.array(list(indices), dtype=int)
        take = max(1, int(round(len(values) * fraction)))
        take = min(take, len(values))
        selected.extend(rng.choice(values, size=take, replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def update_gate(path: Path, status: str, detail: str) -> None:
    gates = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    row = pd.DataFrame([{"gate": "edge_stability", "status": status, "detail": detail}])
    gates = gates.loc[gates["gate"].ne("edge_stability")]
    write_tsv(path, pd.concat([gates, row], ignore_index=True))


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 2:
        raise ValueError("--iterations must be at least 2")
    if not 0 < args.sample_fraction < 1:
        raise ValueError("--sample-fraction must be in (0, 1)")
    if not 0 < args.stability_threshold <= 1:
        raise ValueError("--stability-threshold must be in (0, 1]")


def edge_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["edge_key"] = result.apply(
        lambda row: "|".join(sorted((row["source_proteome_id"], row["target_proteome_id"]))),
        axis=1,
    )
    return result


def merge_stability_into_edge_table(
    edge_table_path: Path,
    stability: pd.DataFrame,
    baseline_edges: set[tuple[str, str]],
) -> None:
    required = {
        "source_proteome_id",
        "target_proteome_id",
        "selection_count",
        "iterations",
        "selection_frequency",
        "stability_class",
    }
    missing = required - set(stability.columns)
    if missing:
        raise ValueError(f"stability table lacks required columns: {sorted(missing)}")
    stability = edge_key_frame(stability)
    if stability["edge_key"].duplicated().any() or set(stability["edge_key"]) != {
        "|".join(pair) for pair in baseline_edges
    }:
        raise ValueError("stability table does not cover the exact baseline MAG edge set")
    edge_table = edge_key_frame(
        pd.read_csv(edge_table_path, sep="\t", dtype=str, keep_default_na=False)
    )
    if edge_table["edge_key"].duplicated().any() or set(edge_table["edge_key"]) != {
        "|".join(pair) for pair in baseline_edges
    }:
        raise ValueError("FlashWeave edge table does not cover the exact baseline MAG edge set")
    stability_fields = [
        "selection_count",
        "iterations",
        "selection_frequency",
        "stability_class",
    ]
    stale_columns = [
        column
        for field in stability_fields
        for column in (field, f"{field}_x", f"{field}_y")
        if column in edge_table.columns
    ]
    edge_table = edge_table.drop(columns=stale_columns)
    edge_table = edge_table.merge(
        stability[["edge_key", *stability_fields]],
        on="edge_key",
        how="left",
        validate="one_to_one",
    )
    if edge_table[stability_fields].isna().any().any():
        raise ValueError("stability merge left one or more baseline edges without a result")
    write_tsv(edge_table_path, edge_table.drop(columns="edge_key"))


def validate_existing_stability(
    stability: pd.DataFrame,
    baseline_weight: dict[tuple[str, str], float],
) -> None:
    required = {
        "baseline_association_weight",
        "selection_count",
        "iterations",
        "selection_frequency",
    }
    if missing := required - set(stability.columns):
        raise ValueError(f"existing stability table lacks required columns: {sorted(missing)}")
    expected = {"|".join(pair) for pair in baseline_weight}
    keyed = edge_key_frame(stability)
    if keyed["edge_key"].duplicated().any() or set(keyed["edge_key"]) != expected:
        raise ValueError("existing stability table does not match the current baseline edge set")
    weights = pd.to_numeric(keyed["baseline_association_weight"], errors="coerce")
    expected_weights = keyed["edge_key"].map(
        {"|".join(pair): value for pair, value in baseline_weight.items()}
    )
    if weights.isna().any() or not np.allclose(weights, expected_weights.astype(float), atol=1e-12):
        raise ValueError("existing stability table has baseline weights incompatible with the current run")
    iterations = pd.to_numeric(keyed["iterations"], errors="coerce")
    selection_count = pd.to_numeric(keyed["selection_count"], errors="coerce")
    frequency = pd.to_numeric(keyed["selection_frequency"], errors="coerce")
    if (
        iterations.isna().any()
        or selection_count.isna().any()
        or frequency.isna().any()
        or (iterations < 2).any()
        or (selection_count < 0).any()
        or (selection_count > iterations).any()
        or (frequency < 0).any()
        or (frequency > 1).any()
        or not np.allclose(frequency, selection_count / iterations, atol=1e-12)
    ):
        raise ValueError("existing stability table has invalid selection counts or frequencies")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    julia_bin = resolve(repo_root, args.julia_bin)
    julia_project = resolve(repo_root, args.julia_project)
    julia_depot = resolve(repo_root, args.julia_depot)
    if not julia_bin.is_file() or not julia_project.is_dir() or not julia_depot.is_dir():
        raise FileNotFoundError("isolated Julia, FlashWeave project, or Julia depot is unavailable")
    network_dir = run_dir / "network_analysis"
    expression = pd.read_csv(network_dir / "flashweave_mag_expression.tsv", sep="\t")
    metadata = pd.read_csv(network_dir / "flashweave_conditioning_metadata.tsv", sep="\t", dtype=str)
    nodes = pd.read_csv(network_dir / "network_node_manifest.tsv", sep="\t", dtype=str)
    baseline_path = network_dir / "flashweave_direct_associations.edgelist"
    gate_path = network_dir / "mucc_v1_flashweave_network_validation_gates.tsv"
    required_metadata = ["sample_id", "month_label", "site_or_landcover", "depth_context_code"]
    if expression["sample_id"].tolist() != metadata["sample_id"].tolist():
        raise ValueError("expression and conditioning metadata sample order differs")
    if set(required_metadata) - set(metadata.columns):
        raise ValueError("conditioning metadata lacks required scaffold-stratification columns")
    mag_ids = set(nodes["proteome_id"].astype(str))
    baseline_edges = mag_edge_pairs(baseline_path, mag_ids)
    if not baseline_edges:
        raise ValueError("baseline FlashWeave MAG edge list is empty")
    baseline_weight = {
        tuple(sorted((source, target))): weight
        for source, target, weight in read_raw_edges(baseline_path)
        if source in mag_ids and target in mag_ids
    }
    stability_path = network_dir / "fact_mucc_v1_flashweave_edge_stability.tsv"
    edge_table_path = network_dir / "fact_mucc_v1_flashweave_edges.tsv"
    if args.reconcile_existing:
        stability = pd.read_csv(stability_path, sep="\t", dtype=str, keep_default_na=False)
        validate_existing_stability(stability, baseline_weight)
        merge_stability_into_edge_table(edge_table_path, stability, baseline_edges)
        iterations = int(pd.to_numeric(stability["iterations"]).iloc[0])
        stable_count = int(
            pd.to_numeric(stability["selection_frequency"]).ge(args.stability_threshold).sum()
        )
        gate_status = "pass" if iterations >= 20 else "limited"
        update_gate(
            gate_path,
            gate_status,
            (
                f"reconciled {iterations} validated scaffold-stratified subsamples; "
                f"{stable_count}/{len(stability)} baseline edges have selection frequency >= "
                f"{args.stability_threshold:.2f}"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "reconciled_existing_validated_stability",
                    "iterations": iterations,
                    "baseline_mag_mag_edges": len(stability),
                    "edges_at_or_above_stability_threshold": stable_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    stability_dir = network_dir / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)
    runner = repo_root / "scripts/reports/mucc_v1_flashweave_stability_iteration.jl"
    rng = np.random.default_rng(args.seed)
    observed_counts: Counter[tuple[str, str]] = Counter()
    run_rows: list[dict[str, object]] = []
    sampled_rows: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment["JULIA_DEPOT_PATH"] = str(julia_depot)
    for iteration in range(1, args.iterations + 1):
        indices = stratified_sample_indices(metadata, args.sample_fraction, rng)
        iteration_expression = stability_dir / f"expression_{iteration:03d}.tsv"
        iteration_metadata = stability_dir / f"metadata_{iteration:03d}.tsv"
        iteration_edges = stability_dir / f"edges_{iteration:03d}.edgelist"
        write_tsv(iteration_expression, expression.iloc[indices].reset_index(drop=True))
        write_tsv(iteration_metadata, metadata.iloc[indices].reset_index(drop=True))
        command = [
            str(julia_bin),
            f"--project={julia_project}",
            str(runner),
            str(iteration_expression),
            str(iteration_metadata),
            str(iteration_edges),
        ]
        completed = subprocess.run(
            command,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"FlashWeave stability iteration {iteration} failed: {completed.stderr[-2000:]}"
            )
        pairs = mag_edge_pairs(iteration_edges, mag_ids)
        observed_counts.update(pairs)
        run_rows.append(
            {
                "lane_id": LANE_ID,
                "analysis_id": ANALYSIS_ID,
                "iteration": iteration,
                "sample_count": len(indices),
                "mag_mag_edge_count": len(pairs),
                "status": "completed",
                "input_expression_tsv": str(iteration_expression),
                "input_metadata_tsv": str(iteration_metadata),
                "output_edgelist": str(iteration_edges),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        sampled_rows.extend(
            {
                "iteration": iteration,
                "sample_id": sample_id,
                "stratum": "|".join(metadata.iloc[index][required_metadata[1:]].astype(str)),
            }
            for index, sample_id in zip(indices, metadata.iloc[indices]["sample_id"], strict=True)
        )
    stability = pd.DataFrame(
        [
            {
                "lane_id": LANE_ID,
                "analysis_id": ANALYSIS_ID,
                "source_proteome_id": source,
                "target_proteome_id": target,
                "baseline_association_weight": baseline_weight[(source, target)],
                "selection_count": observed_counts[(source, target)],
                "iterations": args.iterations,
                "selection_frequency": observed_counts[(source, target)] / args.iterations,
                "stability_class": (
                    "stable_at_or_above_threshold"
                    if observed_counts[(source, target)] / args.iterations >= args.stability_threshold
                    else "below_stability_threshold"
                ),
                "sampling_design": (
                    f"scaffold_stratified_{args.sample_fraction:.0%}_subsample_without_replacement"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
            for source, target in sorted(baseline_edges)
        ]
    )
    write_tsv(stability_path, stability)
    write_tsv(stability_dir / "stability_run_manifest.tsv", pd.DataFrame(run_rows))
    write_tsv(stability_dir / "sampled_ids.tsv", pd.DataFrame(sampled_rows))
    merge_stability_into_edge_table(edge_table_path, stability, baseline_edges)
    stable_count = int(stability["selection_frequency"].ge(args.stability_threshold).sum())
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "analysis_id": ANALYSIS_ID,
        "iterations": args.iterations,
        "sample_fraction": args.sample_fraction,
        "seed": args.seed,
        "baseline_mag_mag_edges": len(stability),
        "edges_at_or_above_stability_threshold": stable_count,
        "stability_threshold": args.stability_threshold,
        "median_selection_frequency": float(stability["selection_frequency"].median()),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (network_dir / "mucc_v1_flashweave_stability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    gate_status = "pass" if args.iterations >= 20 else "limited"
    update_gate(
        gate_path,
        gate_status,
        (
            f"{args.iterations} deterministic scaffold-stratified {args.sample_fraction:.0%} subsamples; "
            f"{stable_count}/{len(stability)} baseline edges have selection frequency >= {args.stability_threshold:.2f}; "
            "at least 20 iterations are required for a passing stability gate"
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
