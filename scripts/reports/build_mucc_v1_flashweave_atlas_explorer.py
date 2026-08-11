#!/usr/bin/env python3
"""Materialize an Ocean-M-inspired, claim-safe MUCC FlashWeave explorer layer.

The explorer joins conditional MAG association edges to rank-aware taxonomy,
functional marker summaries, expression prevalence, and selection stability.
It supports taxon/marker/stability filtering in the atlas without relabeling an
association as an interaction, ecological mechanism, or flux effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
CLAIM_BOUNDARY = (
    "Ocean-M-inspired taxonomy/marker/stability explorer for FlashWeave conditional "
    "associations in processed MUCC expression data. It is exploratory MAG-level "
    "association context only, not a causal interaction, ecological mechanism, exact "
    "depth/environment/flux relationship, measured methane flux, final MRV score/A-E "
    "tier, carbon-crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def node_context(nodes: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    if nodes["proteome_id"].duplicated().any() or taxonomy["proteome_id"].duplicated().any():
        raise ValueError("node and taxonomy inputs must each have unique proteome_id values")
    taxonomy_columns = [
        "proteome_id",
        "atlas_taxonomy_lineage",
        "atlas_taxonomy_projection_status",
        "atlas_taxonomy_policy",
        "source_kbase_rank_disagreement_count",
        "kbase_rank_fallback_count",
        "atlas_domain",
        "atlas_phylum",
        "atlas_class",
        "atlas_order",
        "atlas_family",
        "atlas_genus",
        "atlas_species",
        "atlas_phylum_provenance",
        "atlas_class_provenance",
        "atlas_genus_provenance",
    ]
    result = nodes.merge(
        taxonomy[[column for column in taxonomy_columns if column in taxonomy.columns]],
        on="proteome_id",
        how="left",
        validate="one_to_one",
    ).fillna("")
    result["atlas_taxonomy_available"] = result["atlas_taxonomy_lineage"].ne("").map(
        {True: "true", False: "false"}
    )
    result["atlas_taxonomy_filter_status"] = result.apply(
        lambda row: (
            "taxonomy_available_source_primary_or_KBase_rank_fallback"
            if row["atlas_taxonomy_lineage"]
            else "taxonomy_unavailable_in_source_and_KBase"
        ),
        axis=1,
    )
    result["claim_boundary"] = CLAIM_BOUNDARY
    return result


def edge_context(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if edges[["source_proteome_id", "target_proteome_id"]].duplicated().any():
        raise ValueError("FlashWeave edge input must have unique directed source/target pairs")
    edge_columns = [
        "lane_id",
        "analysis_id",
        "source_proteome_id",
        "target_proteome_id",
        "association_weight",
        "association_sign",
        "absolute_association_weight",
        "edge_directionality",
        "fdr_controlled_during_inference",
        "edge_level_q_value",
        "selection_count",
        "iterations",
        "selection_frequency",
        "stability_class",
    ]
    missing_edge_columns = [column for column in edge_columns if column not in edges.columns]
    if missing_edge_columns:
        raise ValueError(f"FlashWeave edge input is missing columns: {missing_edge_columns}")
    # The base edge table already carries older endpoint descriptors.  Start with
    # the association/stability contract only, then add one authoritative atlas
    # context per endpoint.  This prevents pandas merge suffixes from creating
    # ambiguous duplicate taxonomy, MAG ID, or stability fields downstream.
    edges = edges[edge_columns].copy()
    node_columns = [
        "proteome_id",
        "mag_id",
        "atlas_taxonomy_lineage",
        "atlas_taxonomy_projection_status",
        "atlas_domain",
        "atlas_phylum",
        "atlas_class",
        "atlas_order",
        "atlas_family",
        "atlas_genus",
        "atlas_taxonomy_available",
        "source_kbase_rank_disagreement_count",
        "kbase_rank_fallback_count",
        "marker_breadth_count",
        "methane_term_rows",
        "sulfur_term_rows",
        "methyl_substrate_term_rows",
        "substrate_term_rows",
        "processed_mag_expression_occupancy_fraction",
        "prevalence_fraction",
        "source_qc_label",
        "mrv_readiness_label",
    ]
    source = nodes[[column for column in node_columns if column in nodes.columns]].rename(
        columns=lambda column: (
            "source_proteome_id" if column == "proteome_id" else f"source_{column}"
        )
    )
    target = nodes[[column for column in node_columns if column in nodes.columns]].rename(
        columns=lambda column: (
            "target_proteome_id" if column == "proteome_id" else f"target_{column}"
        )
    )
    result = edges.merge(source, on="source_proteome_id", how="left", validate="many_to_one")
    result = result.merge(target, on="target_proteome_id", how="left", validate="many_to_one")
    if result["source_atlas_taxonomy_available"].isna().any() or result[
        "target_atlas_taxonomy_available"
    ].isna().any():
        raise ValueError("every FlashWeave edge endpoint must map to a node-context row")
    stable = result["stability_class"].eq("stable_at_or_above_threshold")
    both_taxonomy = result["source_atlas_taxonomy_available"].eq("true") & result[
        "target_atlas_taxonomy_available"
    ].eq("true")
    result["both_endpoint_taxonomy_available"] = both_taxonomy.map(
        {True: "true", False: "false"}
    )
    result["network_explorer_visibility_status"] = "conditional_association_visible"
    result.loc[stable & both_taxonomy, "network_explorer_visibility_status"] = (
        "stability_and_taxonomy_filter_eligible"
    )
    result.loc[stable & ~both_taxonomy, "network_explorer_visibility_status"] = (
        "stability_eligible_taxonomy_incomplete"
    )
    source_conflict = pd.to_numeric(
        result["source_source_kbase_rank_disagreement_count"], errors="coerce"
    ).fillna(0)
    target_conflict = pd.to_numeric(
        result["target_source_kbase_rank_disagreement_count"], errors="coerce"
    ).fillna(0)
    result["endpoint_taxonomy_conflict_exposure"] = (
        (source_conflict.gt(0) | target_conflict.gt(0)).map({True: "true", False: "false"})
    )
    result["claim_boundary"] = CLAIM_BOUNDARY
    return result


def explorer_summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    stable = edges["stability_class"].eq("stable_at_or_above_threshold")
    taxonomy = edges["both_endpoint_taxonomy_available"].eq("true")
    conflict = edges["endpoint_taxonomy_conflict_exposure"].eq("true")
    values = [
        (
            "FlashWeave_node_rows",
            len(nodes),
            "exploratory_context_materialized",
            "selected FlashWeave node features with rank-aware taxonomy context",
        ),
        (
            "FlashWeave_node_taxonomy_available",
            int(nodes["atlas_taxonomy_available"].eq("true").sum()),
            "exploratory_context_materialized",
            "source-primary/KBase-fallback taxonomy availability for selected nodes",
        ),
        (
            "FlashWeave_edge_rows",
            len(edges),
            "exploratory_context_materialized",
            "conditional MAG-to-MAG association edges",
        ),
        (
            "FlashWeave_stability_eligible_edges",
            int(stable.sum()),
            "exploratory_stability_filter",
            "selection frequency at or above the preconfigured threshold",
        ),
        (
            "FlashWeave_stability_and_taxonomy_filter_eligible_edges",
            int((stable & taxonomy).sum()),
            "exploratory_stability_and_taxonomy_filter",
            "stable conditional associations with atlas taxonomy at both endpoints",
        ),
        (
            "FlashWeave_edges_with_taxonomy_conflict_exposure",
            int(conflict.sum()),
            "explicit_taxonomy_provenance",
            "at least one endpoint has source/KBase rank disagreement; retain rank provenance",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "lane_id": LANE_ID,
                "metric": metric,
                "value": str(value),
                "status": status,
                "detail": detail,
                "claim_boundary": CLAIM_BOUNDARY,
            }
            for metric, value, status, detail in values
        ]
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    network_dir = run_dir / "network_analysis"
    nodes = read_tsv(network_dir / "feature_mucc_v1_flashweave_node_summary.tsv")
    edges = read_tsv(network_dir / "fact_mucc_v1_flashweave_edges.tsv")
    taxonomy = read_tsv(run_dir / "functional_features/feature_mucc_v1_taxonomy_projection.tsv")
    if len(nodes) != 300 or len(edges) != 694:
        raise ValueError(
            f"expected current FlashWeave explorer denominator nodes=300 edges=694; got {len(nodes)}/{len(edges)}"
        )
    node_table = node_context(nodes, taxonomy)
    edge_table = edge_context(edges, node_table)
    summary = explorer_summary(node_table, edge_table)
    write_tsv(network_dir / "feature_mucc_v1_flashweave_node_atlas_context.tsv", node_table)
    write_tsv(network_dir / "fact_mucc_v1_flashweave_edge_atlas_context.tsv", edge_table)
    write_tsv(network_dir / "mucc_v1_flashweave_atlas_explorer_summary.tsv", summary)
    print(
        json.dumps(
            {
                "node_rows": len(node_table),
                "edge_rows": len(edge_table),
                "stability_and_taxonomy_filter_eligible_edges": int(
                    edge_table["network_explorer_visibility_status"].eq(
                        "stability_and_taxonomy_filter_eligible"
                    ).sum()
                ),
                "output_dir": str(network_dir),
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
