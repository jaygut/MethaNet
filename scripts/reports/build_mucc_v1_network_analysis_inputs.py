#!/usr/bin/env python3
"""Prepare claim-safe MUCC v1 inputs for FlashWeave and complementary WGCNA.

FlashWeave is the primary association method because it can condition on the
available sample descriptors and infer sparse direct associations. WGCNA is
retained as a secondary, descriptive module analysis for comparison with the
paper's module-level results. Neither output is a causal interaction network or
a methane-flux model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
CLAIM_BOUNDARY = (
    "FlashWeave/WGCNA inputs use source-processed relative MAG expression. Any resulting "
    "edge or module is exploratory transcriptional association evidence, not a causal "
    "interaction, measured methane flux, final MRV score/A-E tier, crediting claim, or "
    "source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--max-features",
        type=int,
        default=300,
        help="Feature limit for the primary FlashWeave screen.",
    )
    parser.add_argument(
        "--wgcna-max-features",
        type=int,
        default=0,
        help=(
            "Feature limit for secondary WGCNA; 0 retains every MAG with non-zero "
            "processed expression."
        ),
    )
    parser.add_argument("--min-prevalence", type=float, default=0.1)
    parser.add_argument("--flashweave-max-k", type=int, default=3)
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def write_flashweave_runner(path: Path, max_k: int) -> None:
    path.write_text(
        "# Requires Julia >=1.6 and FlashWeave.\n"
        "# Run from this directory: julia -p 4 run_flashweave.jl\n"
        "using FlashWeave\n\n"
        "data_path = joinpath(@__DIR__, \"flashweave_mag_expression.tsv\")\n"
        "metadata_path = joinpath(@__DIR__, \"flashweave_conditioning_metadata.tsv\")\n"
        "output_path = joinpath(@__DIR__, \"flashweave_direct_associations.edgelist\")\n\n"
        "results = learn_network(\n"
        "    data_path,\n"
        "    metadata_path,\n"
        "    sensitive=true,\n"
        "    heterogeneous=false,\n"
        "    FDR=true,\n"
        f"    max_k={max_k},\n"
        "    normalize=true,\n"
        "    track_rejections=true,\n"
        "    verbose=true,\n"
        ")\n"
        "save_network(output_path, results, detailed=true)\n"
    )


def write_wgcna_runner(path: Path) -> None:
    path.write_text(
        "# Secondary, source-method-aligned descriptive module analysis; FlashWeave remains primary.\n"
        "# Requires R package WGCNA. Run from this directory: Rscript run_wgcna_secondary.R\n"
        "# The paper reports signed-hybrid WGCNA, power 7, minimum module size 50,\n"
        "# and merge height 0.3 on 132 samples after one outlier was removed. The public\n"
        "# materials do not identify that outlier, so this controlled comparator retains all 133.\n"
        "suppressPackageStartupMessages(library(WGCNA))\n"
        "options(stringsAsFactors = FALSE)\n"
        "allowWGCNAThreads()\n\n"
        "expr <- read.delim(\"wgcna_secondary_expression.tsv\", row.names = 1, check.names = FALSE)\n"
        "# Samples are rows; MAGs are columns. Transform only for module discovery.\n"
        "datExpr <- log1p(as.matrix(expr) * 1e6)\n"
        "gsg <- goodSamplesGenes(datExpr, verbose = 3)\n"
        "if (!gsg$allOK) datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]\n"
        "softPower <- 7\n"
        "softPowerSelection <- \"source_reported_fixed_power_7\"\n"
        "# Record the achieved fit at the source-reported power; do not use it to reselect power.\n"
        "sft <- pickSoftThreshold(datExpr, powerVector = softPower, networkType = \"signed hybrid\", verbose = 3)\n"
        "net <- blockwiseModules(datExpr, power = softPower, networkType = \"signed hybrid\", TOMType = \"signed\",\n"
        "    minModuleSize = 50, mergeCutHeight = 0.3, numericLabels = FALSE, verbose = 3)\n"
        "write.table(data.frame(proteome_id = names(net$colors), module = net$colors),\n"
        "    \"wgcna_secondary_modules.tsv\", sep = \"\\t\", quote = FALSE, row.names = FALSE)\n"
        "write.table(data.frame(sample_id = rownames(net$MEs), net$MEs),\n"
        "    \"wgcna_secondary_module_eigengenes.tsv\", sep = \"\\t\", quote = FALSE, row.names = FALSE)\n"
        "fit_row <- sft$fitIndices[match(softPower, sft$fitIndices$Power), , drop = FALSE]\n"
        "module_counts <- table(net$colors)\n"
        "write.table(data.frame(\n"
        "    r_version = as.character(getRversion()),\n"
        "    wgcna_version = as.character(packageVersion(\"WGCNA\")),\n"
        "    samples = nrow(datExpr), mag_features = ncol(datExpr),\n"
        "    soft_power = softPower, selected_sft_r_sq = fit_row$SFT.R.sq,\n"
        "    soft_power_selection = softPowerSelection,\n"
        "    candidate_powers_at_or_above_r_sq_0_8 = \"not_used_source_reported_power_fixed\",\n"
        "    module_count_including_grey = length(module_counts),\n"
        "    non_grey_module_count = sum(names(module_counts) != \"grey\"),\n"
        "    unassigned_grey_mag_count = ifelse(\"grey\" %in% names(module_counts), module_counts[[\"grey\"]], 0),\n"
        "    transform = \"log1p_CPM_from_processed_relative_expression\",\n"
        "    input_matrix = \"wgcna_secondary_expression.tsv\",\n"
        "    network_type = \"signed hybrid\", min_module_size = 50, merge_cut_height = 0.3,\n"
        "    source_reported_wgcna_samples = 132,\n"
        "    source_outlier_reconciliation_status = \"blocked_source_outlier_sample_id_not_available\",\n"
        "    source_parameter_alignment_status = \"aligned_power_7_signed_hybrid_min_module_50_merge_height_0.3\"),\n"
        "    \"wgcna_secondary_run_metadata.tsv\", sep = \"\\t\", quote = FALSE, row.names = FALSE)\n"
    )


def metadata_frame(samples: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    requested = ["month_label", "site_or_landcover", "depth_context_code"]
    included: list[str] = []
    excluded: list[str] = []
    result = pd.DataFrame({"sample_id": samples["sample_id"].astype(str)})
    depth_code = samples.get("depth_code", pd.Series("", index=samples.index)).astype(str).str.strip()
    depth_class = (
        samples.get("depth_class_inferred", pd.Series("", index=samples.index))
        .astype(str)
        .str.strip()
    )
    working = samples.copy()
    working["depth_context_code"] = np.where(
        depth_code.ne(""), "coded_" + depth_code, "class_" + depth_class
    )
    for column in requested:
        values = working.get(column, pd.Series("", index=samples.index)).astype(str).str.strip()
        if values.eq("").any():
            excluded.append(column)
            continue
        result[column] = values
        included.append(column)
    return result, included, excluded


def main() -> int:
    args = parse_args()
    if not 0 < args.min_prevalence <= 1:
        raise SystemExit("--min-prevalence must be in (0, 1]")
    if args.max_features < 25:
        raise SystemExit("--max-features must be at least 25")
    if args.wgcna_max_features < 0 or 0 < args.wgcna_max_features < 25:
        raise SystemExit("--wgcna-max-features must be 0 or at least 25")
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    out_dir = run_dir / "network_analysis"
    fact = pd.read_csv(
        run_dir / "expression/fact_mucc_v1_expression_mag_sample.tsv.gz",
        sep="\t",
        dtype={"sample_id": str, "proteome_id": str, "mag_id": str},
    )
    samples = pd.read_csv(
        run_dir / "environmental_metadata/mucc_v1_sample_columns_scaffold.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    readiness = pd.read_csv(
        run_dir / "functional_features/feature_mucc_v1_mrv_readiness_mag_level.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    sample_order = samples["sample_id"].astype(str).tolist()
    expected_samples = set(sample_order)
    if set(fact["sample_id"].astype(str)) != expected_samples:
        raise ValueError("expression facts and sample scaffold do not have the same sample IDs")
    matrix = fact.pivot(index="sample_id", columns="proteome_id", values="expression_value")
    matrix = matrix.reindex(sample_order).fillna(0.0).astype(float)
    if not np.isfinite(matrix.to_numpy()).all() or (matrix.to_numpy() < 0).any():
        raise ValueError("expression matrix has non-finite or negative values")
    closed = matrix.sum(axis=1)
    if not np.allclose(closed.to_numpy(), 1.0, atol=1e-6):
        raise ValueError("processed expression matrix is not sample-wise compositional")
    prevalence = matrix.gt(0).mean(axis=0)
    log_variance = np.log1p(matrix * 1_000_000.0).var(axis=0)
    taxonomy = readiness.set_index("proteome_id")
    candidate_ids = prevalence.loc[prevalence >= args.min_prevalence].index
    methanogen_screen = taxonomy.reindex(candidate_ids)["class"].fillna("").str.contains(
        "Methano", case=False, regex=False
    )
    protected = candidate_ids[methanogen_screen.to_numpy()].tolist()
    ranked = (
        pd.DataFrame(
            {
                "proteome_id": candidate_ids.to_numpy(),
                "prevalence": prevalence[candidate_ids].to_numpy(),
                "log_variance": log_variance[candidate_ids].to_numpy(),
            }
        )
        .sort_values(["log_variance", "prevalence", "proteome_id"], ascending=[False, False, True])
        ["proteome_id"]
        .tolist()
    )
    selected = protected + [item for item in ranked if item not in set(protected)]
    selected = selected[: args.max_features]
    if len(selected) < 25:
        raise ValueError("prevalence filter leaves too few MAGs for a network analysis")
    selected_matrix = matrix[selected].copy()
    selected_matrix.insert(0, "sample_id", selected_matrix.index)
    wgcna_ranked = (
        pd.DataFrame(
            {
                "proteome_id": prevalence.loc[prevalence.gt(0)].index.to_numpy(),
                "prevalence": prevalence.loc[prevalence.gt(0)].to_numpy(),
                "log_variance": log_variance.loc[prevalence.gt(0)].to_numpy(),
            }
        )
        .sort_values(["log_variance", "prevalence", "proteome_id"], ascending=[False, False, True])
        ["proteome_id"]
        .tolist()
    )
    wgcna_selected = (
        wgcna_ranked
        if args.wgcna_max_features == 0
        else wgcna_ranked[: args.wgcna_max_features]
    )
    if len(wgcna_selected) < 25:
        raise ValueError("secondary WGCNA selection leaves too few MAGs for module discovery")
    wgcna_matrix = matrix[wgcna_selected].copy()
    wgcna_matrix.insert(0, "sample_id", wgcna_matrix.index)
    metadata, conditioning_columns, omitted_metadata = metadata_frame(samples)
    if selected_matrix["sample_id"].tolist() != metadata["sample_id"].tolist():
        raise ValueError("FlashWeave data and metadata row order differs")
    node = pd.DataFrame(
        {
            "proteome_id": selected,
            "prevalence_fraction": prevalence[selected].to_numpy(),
            "log1p_cpm_variance": log_variance[selected].to_numpy(),
            "selection_reason": [
                "taxonomic_methanogen_screen" if item in set(protected) else "high_log_variance"
                for item in selected
            ],
        }
    ).merge(readiness, on="proteome_id", how="left")
    if "lane_id" in node.columns:
        node["lane_id"] = LANE_ID
    else:
        node.insert(0, "lane_id", LANE_ID)
    node["claim_boundary"] = CLAIM_BOUNDARY
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "flashweave_mag_expression.tsv", selected_matrix)
    write_tsv(out_dir / "wgcna_secondary_expression.tsv", wgcna_matrix)
    write_tsv(out_dir / "flashweave_conditioning_metadata.tsv", metadata)
    write_tsv(out_dir / "network_node_manifest.tsv", node)
    write_flashweave_runner(out_dir / "run_flashweave.jl", args.flashweave_max_k)
    write_wgcna_runner(out_dir / "run_wgcna_secondary.R")
    wgcna_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "input_matrix": "wgcna_secondary_expression.tsv",
        "samples": int(len(samples)),
        "available_expression_MAGs": int(matrix.shape[1]),
        "selected_MAG_features": int(len(wgcna_selected)),
        "selection": (
            "all_nonzero_processed_expression_MAGs"
            if args.wgcna_max_features == 0
            else "highest_log1p_CPM_variance_nonzero_processed_expression_MAGs"
        ),
        "max_features": args.wgcna_max_features,
        "source_method_alignment": {
            "publication": "https://journals.asm.org/doi/10.1128/msystems.00680-25",
            "source_reported_samples_after_outlier_screening": 132,
            "current_samples_retained": int(len(samples)),
            "outlier_reconciliation_status": "blocked_source_outlier_sample_id_not_available",
            "network_type": "signed_hybrid",
            "soft_power": 7,
            "min_module_size": 50,
            "merge_cut_height": 0.3,
            "comparability_status": "source_method_aligned_partial_not_exact_reproduction",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (out_dir / "wgcna_secondary_input_manifest.json").write_text(
        json.dumps(wgcna_manifest, indent=2, sort_keys=True) + "\n"
    )

    status = pd.DataFrame(
        [
            {
                "lane_id": LANE_ID,
                "analysis_id": "flashweave_direct_association",
                "role": "primary_ecological_association_screen",
                "status": "ready_to_run_flashweave",
                "execution_state": "runtime_available" if shutil.which("julia") else "runtime_not_available",
                "samples": len(samples),
                "MAG_features": len(selected),
                "conditioning_metadata": ";".join(conditioning_columns),
                "omitted_metadata": ";".join(omitted_metadata),
                "method_parameters": f"sensitive=true; heterogeneous=false; FDR=true; max_k={args.flashweave_max_k}; normalize=true",
                "why_not_flashweaveHE": "single-study 133-sample lane; HE is intended for multi-habitat/protocol data with ideally thousands of samples",
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "lane_id": LANE_ID,
                "analysis_id": "wgcna_secondary_module_discovery",
                "role": "secondary_source_method_aligned_module_comparison",
                "status": "ready_to_run_secondary_wgcna",
                "execution_state": "runtime_available" if shutil.which("Rscript") else "runtime_not_available",
                "samples": len(samples),
                "MAG_features": len(wgcna_selected),
                "conditioning_metadata": "none_in_module_fit; correlate eigengenes only after exact covariates resolve",
                "omitted_metadata": "environmental and flux measurements are not joined",
                "method_parameters": (
                    "source-method-aligned signed-hybrid network; source-reported fixed "
                    "softPower=7; minModuleSize=50; mergeCutHeight=0.3; all 133 samples "
                    "retained because the source's excluded outlier ID is not public"
                ),
                "why_not_flashweaveHE": "not_applicable",
                "claim_boundary": CLAIM_BOUNDARY,
            },
        ]
    )
    write_tsv(out_dir / "network_analysis_status.tsv", status)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "status": "ready_to_run_flashweave",
        "samples": int(len(samples)),
        "available_expression_MAGs": int(matrix.shape[1]),
        "selected_MAG_features": int(len(selected)),
        "selected_taxonomic_methanogen_screen_MAGs": int(len([x for x in selected if x in set(protected)])),
        "min_prevalence": args.min_prevalence,
        "conditioning_metadata": conditioning_columns,
        "omitted_metadata": omitted_metadata,
        "flashweave": {
            "mode": "FlashWeave-S",
            "heterogeneous": False,
            "sensitive": True,
            "FDR": True,
            "max_k": args.flashweave_max_k,
            "runner": "run_flashweave.jl",
        },
        "wgcna": {
            "role": "secondary source-method-aligned module comparison",
            "runner": "run_wgcna_secondary.R",
            "input_matrix": "wgcna_secondary_expression.tsv",
            "selected_MAG_features": int(len(wgcna_selected)),
            "selection": wgcna_manifest["selection"],
            "source_method_alignment": wgcna_manifest["source_method_alignment"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (out_dir / "flashweave_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
