#!/usr/bin/env python3
"""Promote a completed MUCC v1 FlashWeave run into claim-safe atlas tables.

The resulting edge table contains conditional associations inferred from
source-processed relative MAG expression. It is deliberately not a causal
interaction network, ecological mechanism map, methane-flux model, or MRV score.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
ANALYSIS_ID = "flashweave_direct_association"
CLAIM_BOUNDARY = (
    "FlashWeave conditional associations from source-processed relative MAG expression are "
    "exploratory transcriptional association evidence only; they do not establish a causal "
    "interaction, measured methane flux, final MRV score/A-E tier, crediting claim, or "
    "source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def read_edgelist(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t")
            if len(fields) != 3:
                raise ValueError(f"unexpected edge-list row {line_number}: expected 3 tab-separated fields")
            source, target, weight = fields
            try:
                association_weight = float(weight)
            except ValueError as exc:
                raise ValueError(f"invalid edge weight on row {line_number}: {weight!r}") from exc
            if not np.isfinite(association_weight):
                raise ValueError(f"non-finite edge weight on row {line_number}")
            if source == target:
                raise ValueError(f"self-edge on row {line_number}: {source}")
            rows.append(
                {
                    "source_proteome_id": source,
                    "target_proteome_id": target,
                    "association_weight": association_weight,
                }
            )
    edges = pd.DataFrame(rows)
    if edges.empty:
        raise ValueError("FlashWeave edge list contains no associations")
    pairs = edges.apply(
        lambda row: tuple(sorted((str(row["source_proteome_id"]), str(row["target_proteome_id"])))),
        axis=1,
    )
    if pairs.duplicated().any():
        raise ValueError("FlashWeave edge list contains duplicate undirected pairs")
    return edges


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def metadata_from_node_manifest(node_manifest: pd.DataFrame, suffix: str) -> pd.DataFrame:
    fields = [
        "proteome_id",
        "mag_id",
        "domain",
        "phylum",
        "class",
        "mrv_readiness_label",
    ]
    metadata = node_manifest[[field for field in fields if field in node_manifest.columns]].copy()
    metadata = metadata.rename(columns={"proteome_id": f"{suffix}_proteome_id"})
    return metadata.rename(
        columns={field: f"{suffix}_{field}" for field in metadata.columns if field != f"{suffix}_proteome_id"}
    )


def write_execution_status(
    path: Path,
    mag_edge_count: int,
    total_edge_count: int,
    node_count: int,
    components: int,
) -> None:
    existing = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    existing = existing.loc[existing["analysis_id"] != ANALYSIS_ID].copy()
    completed = {
        "lane_id": LANE_ID,
        "analysis_id": ANALYSIS_ID,
        "role": "primary_ecological_association_screen",
        "status": "completed_exploratory_conditional_associations",
        "execution_state": "completed_with_isolated_julia_runtime",
        "samples": "133",
        "MAG_features": str(node_count),
        "conditioning_metadata": "month_label;site_or_landcover;depth_context_code",
        "omitted_metadata": "exact environmental and flux measurements are not joined",
        "method_parameters": "FlashWeave-S; sensitive=true; heterogeneous=false; FDR=true; max_k=3; normalize=true; FlashWeave=0.19.2",
        "why_not_flashweaveHE": "single-study 133-sample lane; HE is intended for multi-habitat/protocol data with ideally thousands of samples",
        "claim_boundary": CLAIM_BOUNDARY,
        "edge_count": str(mag_edge_count),
        "total_flashweave_edges_including_metadata": str(total_edge_count),
        "connected_components": str(components),
    }
    columns = list(existing.columns)
    for key in completed:
        if key not in columns:
            columns.append(key)
    output = pd.concat([existing.reindex(columns=columns), pd.DataFrame([completed]).reindex(columns=columns)])
    write_tsv(path, output)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    network_dir = run_dir / "network_analysis"
    edge_path = network_dir / "flashweave_direct_associations.edgelist"
    node_manifest_path = network_dir / "network_node_manifest.tsv"
    input_manifest_path = network_dir / "flashweave_input_manifest.json"
    status_path = network_dir / "network_analysis_status.tsv"
    if not edge_path.is_file():
        raise FileNotFoundError(f"FlashWeave edge list is missing: {edge_path}")
    if not node_manifest_path.is_file() or not input_manifest_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("FlashWeave input artifacts are incomplete; rebuild network inputs first")

    raw_edges = read_edgelist(edge_path)
    node_manifest = pd.read_csv(node_manifest_path, sep="\t", dtype=str, keep_default_na=False)
    input_manifest = json.loads(input_manifest_path.read_text())
    manifest_ids = set(node_manifest["proteome_id"].astype(str))
    conditioning_path = network_dir / "flashweave_conditioning_metadata.tsv"
    conditioning = pd.read_csv(conditioning_path, sep="\t", dtype=str, keep_default_na=False)
    conditioning_columns = [column for column in conditioning.columns if column != "sample_id"]
    metadata_ids = {
        f"{column}_{value}"
        for column in conditioning_columns
        for value in conditioning[column].drop_duplicates()
        if value
    }
    edge_ids = set(raw_edges["source_proteome_id"]).union(raw_edges["target_proteome_id"])
    unknown_ids = sorted(edge_ids - manifest_ids - metadata_ids)
    if unknown_ids:
        raise ValueError(f"FlashWeave emitted nodes outside the selected MAG manifest: {unknown_ids[:5]}")
    if input_manifest.get("status") != "ready_to_run_flashweave":
        raise ValueError("FlashWeave input manifest does not represent a ready-to-run input contract")

    source_is_mag = raw_edges["source_proteome_id"].isin(manifest_ids)
    target_is_mag = raw_edges["target_proteome_id"].isin(manifest_ids)
    edges = raw_edges.loc[source_is_mag & target_is_mag].copy()
    metadata_edges = raw_edges.loc[~(source_is_mag & target_is_mag)].copy()
    if edges.empty:
        raise ValueError("FlashWeave returned no MAG-to-MAG associations")

    graph = nx.Graph()
    graph.add_nodes_from(node_manifest["proteome_id"].astype(str))
    graph.add_weighted_edges_from(
        edges[["source_proteome_id", "target_proteome_id", "association_weight"]].itertuples(
            index=False, name=None
        )
    )
    components = sorted(nx.connected_components(graph), key=lambda nodes: (-len(nodes), sorted(nodes)[0]))
    component_by_node = {
        node: (index + 1, len(component))
        for index, component in enumerate(components)
        for node in component
    }
    node_table = node_manifest.copy()
    node_table["analysis_id"] = ANALYSIS_ID
    node_table["flashweave_degree"] = node_table["proteome_id"].map(dict(graph.degree()))
    node_table["flashweave_absolute_weighted_degree"] = node_table["proteome_id"].map(
        {
            node: sum(abs(float(attrs["weight"])) for _, _, attrs in graph.edges(node, data=True))
            for node in graph.nodes
        }
    )
    node_table["flashweave_component_id"] = node_table["proteome_id"].map(
        lambda node: component_by_node[str(node)][0]
    )
    node_table["flashweave_component_size"] = node_table["proteome_id"].map(
        lambda node: component_by_node[str(node)][1]
    )
    node_table["is_isolated"] = node_table["flashweave_degree"].eq(0).map({True: "true", False: "false"})
    node_table["claim_boundary"] = CLAIM_BOUNDARY

    edge_table = edges.copy()
    edge_table.insert(0, "lane_id", LANE_ID)
    edge_table.insert(1, "analysis_id", ANALYSIS_ID)
    edge_table["association_sign"] = np.where(edge_table["association_weight"] > 0, "positive", "negative")
    edge_table["absolute_association_weight"] = edge_table["association_weight"].abs()
    edge_table["edge_directionality"] = "undirected"
    edge_table["fdr_controlled_during_inference"] = "true"
    edge_table["edge_level_q_value"] = "not_emitted_by_flashweave_edgelist"
    edge_table["claim_boundary"] = CLAIM_BOUNDARY
    edge_table = edge_table.merge(
        metadata_from_node_manifest(node_manifest, "source"), on="source_proteome_id", how="left"
    ).merge(metadata_from_node_manifest(node_manifest, "target"), on="target_proteome_id", how="left")

    metadata_edges.insert(0, "lane_id", LANE_ID)
    metadata_edges.insert(1, "analysis_id", ANALYSIS_ID)
    metadata_edges["edge_class"] = np.where(
        metadata_edges["source_proteome_id"].isin(manifest_ids)
        | metadata_edges["target_proteome_id"].isin(manifest_ids),
        "mag_to_conditioning_metadata",
        "conditioning_metadata_to_metadata",
    )
    metadata_edges["association_sign"] = np.where(
        metadata_edges["association_weight"] > 0, "positive", "negative"
    )
    metadata_edges["absolute_association_weight"] = metadata_edges["association_weight"].abs()
    metadata_edges["edge_directionality"] = "undirected"
    metadata_edges["fdr_controlled_during_inference"] = "true"
    metadata_edges["edge_level_q_value"] = "not_emitted_by_flashweave_edgelist"
    metadata_edges["claim_boundary"] = CLAIM_BOUNDARY

    gate_rows = pd.DataFrame(
        [
            {
                "gate": "flashweave_edge_file",
                "status": "pass",
                "detail": (
                    f"parsed {len(raw_edges)} total edges: {len(edge_table)} MAG-to-MAG and "
                    f"{len(metadata_edges)} metadata-involving associations"
                ),
            },
            {
                "gate": "network_node_contract",
                "status": "pass" if len(node_table) == int(input_manifest["selected_MAG_features"]) else "fail",
                "detail": f"nodes={len(node_table)}; selected input features={input_manifest['selected_MAG_features']}",
            },
            {
                "gate": "conditioning_metadata",
                "status": "pass" if len(input_manifest.get("conditioning_metadata", [])) >= 1 else "fail",
                "detail": ";".join(input_manifest.get("conditioning_metadata", [])),
            },
            {
                "gate": "fdr_controlled_inference",
                "status": "pass" if input_manifest.get("flashweave", {}).get("FDR") is True else "fail",
                "detail": "FDR=true during FlashWeave inference; per-edge q-values are not emitted by this edgelist format",
            },
            {
                "gate": "edge_stability",
                "status": "blocked",
                "detail": "bootstrap or grouped leave-out edge stability has not been run",
            },
            {
                "gate": "exact_ecological_covariates_and_flux",
                "status": "blocked",
                "detail": "exact depth, environmental, abundance/read-coverage, and flux/process joins remain unresolved",
            },
            {
                "gate": "causal_or_mrv_interpretation",
                "status": "blocked",
                "detail": "network supports exploratory conditional association only",
            },
        ]
    )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "analysis_id": ANALYSIS_ID,
        "samples": int(input_manifest["samples"]),
        "nodes": int(len(node_table)),
        "mag_mag_edges": int(len(edge_table)),
        "metadata_involving_edges": int(len(metadata_edges)),
        "total_edges": int(len(raw_edges)),
        "connected_components": int(len(components)),
        "largest_component_nodes": int(len(components[0])),
        "isolated_nodes": int((node_table["flashweave_degree"] == 0).sum()),
        "positive_edges": int((edge_table["association_weight"] > 0).sum()),
        "negative_edges": int((edge_table["association_weight"] < 0).sum()),
        "fdr_controlled_during_inference": True,
        "edge_level_q_values_available": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_tsv(network_dir / "fact_mucc_v1_flashweave_edges.tsv", edge_table)
    write_tsv(network_dir / "fact_mucc_v1_flashweave_metadata_edges.tsv", metadata_edges)
    write_tsv(network_dir / "feature_mucc_v1_flashweave_node_summary.tsv", node_table)
    write_tsv(network_dir / "mucc_v1_flashweave_network_validation_gates.tsv", gate_rows)
    write_execution_status(
        status_path,
        len(edge_table),
        len(raw_edges),
        len(node_table),
        len(components),
    )
    (network_dir / "mucc_v1_flashweave_network_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
