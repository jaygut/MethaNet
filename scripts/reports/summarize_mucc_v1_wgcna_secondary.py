#!/usr/bin/env python3
"""Validate and materialize the secondary MUCC v1 WGCNA module analysis.

WGCNA is deliberately secondary to the completed FlashWeave conditional
association analysis. This script validates that its modules and eigengenes
cover the exact, independently materialized WGCNA expression matrix, records
runtime metadata, and updates only the WGCNA status row. It does not infer
ecological covariates or turn modules into interaction, flux, or MRV claims.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
CLAIM_BOUNDARY = (
    "Secondary WGCNA modules use source-processed relative MAG expression. They are descriptive "
    "coexpression summaries only and do not establish a causal interaction, measured methane flux, "
    "final MRV score/A-E tier, carbon-crediting claim, or source-independent transfer result."
)
WGCNA_ANALYSIS_ID = "wgcna_secondary_module_discovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def finite_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def validate_outputs(
    expression: pd.DataFrame,
    modules: pd.DataFrame,
    eigengenes: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    expected_samples = expression["sample_id"].astype(str).tolist()
    expected_proteomes = expression.columns[1:].astype(str).tolist()
    if list(modules.columns) != ["proteome_id", "module"]:
        raise ValueError("WGCNA modules must contain exactly proteome_id and module columns")
    if modules["proteome_id"].eq("").any() or modules["module"].eq("").any():
        raise ValueError("WGCNA modules contain an empty proteome_id or module")
    if modules["proteome_id"].duplicated().any():
        raise ValueError("WGCNA modules contain duplicate proteome_id values")
    if set(modules["proteome_id"]) != set(expected_proteomes):
        raise ValueError("WGCNA modules do not cover the exact WGCNA proteome feature set")
    if "sample_id" not in eigengenes.columns or eigengenes["sample_id"].duplicated().any():
        raise ValueError("WGCNA eigengenes require a unique sample_id column")
    if set(eigengenes["sample_id"]) != set(expected_samples):
        raise ValueError("WGCNA eigengenes do not cover the exact WGCNA sample set")
    eigengene_columns = [column for column in eigengenes.columns if column != "sample_id"]
    if not eigengene_columns or any(not column.startswith("ME") for column in eigengene_columns):
        raise ValueError("WGCNA eigengenes must contain one or more ME* columns")
    for column in eigengene_columns:
        values = pd.to_numeric(eigengenes[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"WGCNA eigengene column is not finite: {column}")
    required_metadata = {
        "r_version",
        "wgcna_version",
        "samples",
        "mag_features",
        "soft_power",
        "selected_sft_r_sq",
        "module_count_including_grey",
        "non_grey_module_count",
        "unassigned_grey_mag_count",
    }
    if len(metadata) != 1 or not required_metadata.issubset(metadata.columns):
        raise ValueError("WGCNA runtime metadata must contain one complete summary row")
    metadata_row = metadata.iloc[0]
    if int(finite_float(metadata_row["samples"], "WGCNA metadata samples")) != len(expected_samples):
        raise ValueError("WGCNA metadata sample count disagrees with the input matrix")
    if int(finite_float(metadata_row["mag_features"], "WGCNA metadata mag_features")) != len(
        expected_proteomes
    ):
        raise ValueError("WGCNA metadata feature count disagrees with the input matrix")
    if int(finite_float(metadata_row["soft_power"], "WGCNA metadata soft_power")) <= 0:
        raise ValueError("WGCNA soft power must be positive")
    finite_float(metadata_row["selected_sft_r_sq"], "WGCNA selected SFT R-squared")
    counts = modules.groupby("module", as_index=False).size().rename(columns={"size": "mag_count"})
    counts = counts.sort_values(["module"], key=lambda values: values.eq("grey"), kind="stable")
    non_grey = int(counts["module"].ne("grey").sum())
    grey = int(counts.loc[counts["module"].eq("grey"), "mag_count"].sum())
    if int(finite_float(metadata_row["module_count_including_grey"], "WGCNA module count")) != len(
        counts
    ):
        raise ValueError("WGCNA metadata module count disagrees with module assignments")
    if int(finite_float(metadata_row["non_grey_module_count"], "WGCNA non-grey module count")) != non_grey:
        raise ValueError("WGCNA metadata non-grey module count disagrees with module assignments")
    if int(finite_float(metadata_row["unassigned_grey_mag_count"], "WGCNA grey MAG count")) != grey:
        raise ValueError("WGCNA metadata grey MAG count disagrees with module assignments")
    run = {
        "r_version": str(metadata_row["r_version"]),
        "wgcna_version": str(metadata_row["wgcna_version"]),
        "soft_power": int(finite_float(metadata_row["soft_power"], "WGCNA metadata soft_power")),
        "selected_sft_r_sq": finite_float(
            metadata_row["selected_sft_r_sq"], "WGCNA selected SFT R-squared"
        ),
        "module_count_including_grey": len(counts),
        "non_grey_module_count": non_grey,
        "unassigned_grey_mag_count": grey,
        "soft_power_selection": str(metadata_row.get("soft_power_selection", "")),
        "input_matrix": str(metadata_row.get("input_matrix", "wgcna_secondary_expression.tsv")),
        "samples": len(expected_samples),
        "mag_features": len(expected_proteomes),
        "network_type": str(metadata_row.get("network_type", "")),
        "min_module_size": str(metadata_row.get("min_module_size", "")),
        "merge_cut_height": str(metadata_row.get("merge_cut_height", "")),
        "source_reported_wgcna_samples": str(
            metadata_row.get("source_reported_wgcna_samples", "")
        ),
        "source_outlier_reconciliation_status": str(
            metadata_row.get("source_outlier_reconciliation_status", "")
        ),
        "source_parameter_alignment_status": str(
            metadata_row.get("source_parameter_alignment_status", "")
        ),
    }
    return modules, counts, run


def update_network_status(status_path: Path, run: dict[str, float | int | str]) -> None:
    status = read_tsv(status_path)
    matched = status["analysis_id"].eq(WGCNA_ANALYSIS_ID)
    if int(matched.sum()) != 1:
        raise ValueError("network analysis status must contain one WGCNA secondary row")
    extra_columns = [
        "module_count_including_grey",
        "non_grey_module_count",
        "unassigned_grey_mag_count",
        "runtime_versions",
    ]
    for column in extra_columns:
        if column not in status.columns:
            status[column] = ""
    status.loc[matched, "status"] = "completed_secondary_descriptive_modules"
    status.loc[matched, "execution_state"] = "completed_isolated_conda_runtime"
    status.loc[matched, "samples"] = str(run["samples"])
    status.loc[matched, "MAG_features"] = str(run["mag_features"])
    status.loc[matched, "method_parameters"] = (
        f"{run['network_type'] or 'unspecified'} network; log1p CPM from processed relative expression; "
        f"softPower={run['soft_power']}; selected_SFT_R_sq={run['selected_sft_r_sq']:.6f}; "
        f"selection={run['soft_power_selection']}; "
        f"minModuleSize={run['min_module_size'] or 'unspecified'}; "
        f"mergeCutHeight={run['merge_cut_height'] or 'unspecified'}; "
        f"source_alignment={run['source_parameter_alignment_status'] or 'not_recorded'}; "
        f"outlier_status={run['source_outlier_reconciliation_status'] or 'not_recorded'}"
    )
    status.loc[matched, "module_count_including_grey"] = str(run["module_count_including_grey"])
    status.loc[matched, "non_grey_module_count"] = str(run["non_grey_module_count"])
    status.loc[matched, "unassigned_grey_mag_count"] = str(run["unassigned_grey_mag_count"])
    status.loc[matched, "runtime_versions"] = (
        f"R={run['r_version']}; WGCNA={run['wgcna_version']}"
    )
    write_tsv(status_path, status)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    network_dir = run_dir / "network_analysis"
    expression = read_tsv(network_dir / "wgcna_secondary_expression.tsv")
    modules, module_summary, run = validate_outputs(
        expression,
        read_tsv(network_dir / "wgcna_secondary_modules.tsv"),
        read_tsv(network_dir / "wgcna_secondary_module_eigengenes.tsv"),
        read_tsv(network_dir / "wgcna_secondary_run_metadata.tsv"),
    )
    membership = modules.copy()
    membership.insert(0, "lane_id", LANE_ID)
    membership["module_assignment_status"] = "completed_secondary_descriptive_module"
    membership["method_role"] = "secondary_source_method_aligned_to_FlashWeave_conditional_associations"
    membership["claim_boundary"] = CLAIM_BOUNDARY
    module_summary.insert(0, "lane_id", LANE_ID)
    module_summary["module_assignment_status"] = module_summary["module"].map(
        lambda value: "unassigned_grey" if value == "grey" else "detected_descriptive_module"
    )
    module_summary["method_role"] = "secondary_source_method_aligned_to_FlashWeave_conditional_associations"
    module_summary["claim_boundary"] = CLAIM_BOUNDARY
    eigengenes = read_tsv(network_dir / "wgcna_secondary_module_eigengenes.tsv")
    eigengenes.insert(0, "lane_id", LANE_ID)
    eigengenes["ecological_covariate_correlation_status"] = (
        "blocked_until_exact_authoritative_sample_environment_flux_crosswalk"
    )
    eigengenes["claim_boundary"] = CLAIM_BOUNDARY
    write_tsv(network_dir / "feature_mucc_v1_wgcna_secondary_module_membership.tsv", membership)
    write_tsv(network_dir / "feature_mucc_v1_wgcna_secondary_module_summary.tsv", module_summary)
    write_tsv(network_dir / "fact_mucc_v1_wgcna_secondary_module_eigengenes.tsv", eigengenes)
    update_network_status(network_dir / "network_analysis_status.tsv", run)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lane_id": LANE_ID,
        "status": "completed_secondary_descriptive_modules",
        "method_role": "secondary_source_method_aligned_to_FlashWeave_conditional_associations",
        **run,
        "publication": "https://journals.asm.org/doi/10.1128/msystems.00680-25",
        "source_cohort_comparison": (
            "source reports 132 samples after one outlier screen; current 133-sample comparator "
            "retains all rows because the excluded sample identifier is unavailable"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (network_dir / "mucc_v1_wgcna_secondary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
