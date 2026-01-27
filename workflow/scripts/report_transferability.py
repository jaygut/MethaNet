"""Generate transferability and ablation diagnostics for the pilot POC.

This script is designed to be high-leverage for the pilot:
- It quantifies domain shift by feature group.
- It measures cross-domain predictive transfer when target labels exist.
- It surfaces feature-group missingness and extraction stability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import pairwise_distances


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    slices: tuple[slice, ...]

    def select(self, values: np.ndarray) -> np.ndarray:
        parts = [values[:, s] for s in self.slices]
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)


@dataclass
class GroupData:
    source: np.ndarray
    target: np.ndarray
    source_mask: np.ndarray
    target_mask: np.ndarray


def _feature_columns(df: pd.DataFrame) -> list[str]:
    f_cols = [col for col in df.columns if col.startswith("f_")]
    if not f_cols:
        raise ValueError("No fused feature columns found (expected f_0, f_1, ...).")
    return sorted(f_cols, key=lambda c: int(c.split("_")[1]))


def _infer_genome_dim(total_dim: int, functional_dim: int, esm2_dim: int) -> int:
    genome_dim = total_dim - functional_dim - esm2_dim
    if genome_dim <= 0:
        raise ValueError(
            "Could not infer genome_dim; provide --genome-dim explicitly."
        )
    return genome_dim


def _build_groups(
    total_dim: int,
    functional_dim: int,
    esm2_dim: int,
    genome_dim: int,
) -> list[FeatureGroup]:
    functional_slice = slice(0, functional_dim)
    esm2_slice = slice(functional_dim, functional_dim + esm2_dim)
    genome_slice = slice(
        functional_dim + esm2_dim,
        functional_dim + esm2_dim + genome_dim,
    )

    genome_end = genome_slice.stop or 0
    if genome_end > total_dim:
        raise ValueError(
            "Group dimensions exceed total fused feature dimension. "
            f"total={total_dim}, requested_end={genome_end}"
        )

    groups = [
        FeatureGroup("functional", (functional_slice,)),
        FeatureGroup("esm2", (esm2_slice,)),
        FeatureGroup("genome", (genome_slice,)),
        FeatureGroup("functional_esm2", (functional_slice, esm2_slice)),
        FeatureGroup("functional_genome", (functional_slice, genome_slice)),
        FeatureGroup("esm2_genome", (esm2_slice, genome_slice)),
        FeatureGroup("all", (functional_slice, esm2_slice, genome_slice)),
    ]
    return groups


def _subsample(values: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(values) <= max_rows:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=max_rows, replace=False)
    return values[idx]


def _median_gamma(source: np.ndarray, target: np.ndarray) -> float:
    combined = np.vstack([source, target])
    combined = _subsample(combined, max_rows=512, seed=42)
    distances = pairwise_distances(combined, metric="euclidean")
    median = float(np.median(distances))
    if median <= 0:
        return 1.0
    return 1.0 / (2 * median**2)


def _rbf_mmd(source: np.ndarray, target: np.ndarray, gamma: float) -> float:
    source = _subsample(source, max_rows=512, seed=123)
    target = _subsample(target, max_rows=512, seed=456)
    ss = np.exp(-gamma * pairwise_distances(source, source, metric="sqeuclidean"))
    tt = np.exp(-gamma * pairwise_distances(target, target, metric="sqeuclidean"))
    st = np.exp(-gamma * pairwise_distances(source, target, metric="sqeuclidean"))
    return float(ss.mean() + tt.mean() - 2 * st.mean())


def _a_distance(source: np.ndarray, target: np.ndarray) -> float:
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv).mean()
    error = 1.0 - acc
    return float(2 * (1 - 2 * error))


def _domain_auc(source: np.ndarray, target: np.ndarray) -> float:
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(aucs))


def _finite_row_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).all(axis=1)


def _extract_group_data(
    source_values: np.ndarray,
    target_values: np.ndarray,
    group: FeatureGroup,
) -> GroupData:
    source_group = group.select(source_values)
    target_group = group.select(target_values)

    source_mask = _finite_row_mask(source_group)
    target_mask = _finite_row_mask(target_group)

    return GroupData(
        source=source_group[source_mask],
        target=target_group[target_mask],
        source_mask=source_mask,
        target_mask=target_mask,
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    pearson_r = None
    if len(y_true) > 1:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": pearson_r,
    }


def _train_models(random_state: int) -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transferability and ablation diagnostics.")
    parser.add_argument("--features", required=True, help="Path to fused features parquet.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-domain", default="rumen")
    parser.add_argument("--target-domain", default="coastal")
    parser.add_argument("--target-col", default="measured_flux")
    parser.add_argument("--functional-dim", type=int, default=77)
    parser.add_argument("--esm2-dim", type=int, default=1280)
    parser.add_argument("--genome-dim", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--min-labeled-per-domain",
        type=int,
        default=5,
        help="Minimum labeled samples required in both domains for transfer metrics.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    if "domain" not in df.columns:
        raise ValueError("Features parquet must include a 'domain' column.")

    f_cols = _feature_columns(df)
    total_dim = len(f_cols)

    genome_dim = args.genome_dim
    if genome_dim is None:
        genome_dim = _infer_genome_dim(total_dim, args.functional_dim, args.esm2_dim)

    groups = _build_groups(
        total_dim=total_dim,
        functional_dim=args.functional_dim,
        esm2_dim=args.esm2_dim,
        genome_dim=genome_dim,
    )

    source_df = df[df["domain"] == args.source_domain].copy()
    target_df = df[df["domain"] == args.target_domain].copy()

    if source_df.empty or target_df.empty:
        raise ValueError(
            "Source/target domain filtering produced empty dataframes. "
            f"source={len(source_df)}, target={len(target_df)}"
        )

    source_values = source_df[f_cols].to_numpy(dtype=float)
    target_values = target_df[f_cols].to_numpy(dtype=float)

    domain_rows: list[dict] = []
    missing_rows: list[dict] = []

    for group in groups:
        group_data = _extract_group_data(source_values, target_values, group)
        if len(group_data.source) < 3 or len(group_data.target) < 3:
            domain_rows.append(
                {
                    "group": group.name,
                    "n_source": int(len(group_data.source)),
                    "n_target": int(len(group_data.target)),
                    "domain_auc": None,
                    "mmd_rbf": None,
                    "a_distance": None,
                    "gamma": None,
                }
            )
        else:
            gamma = _median_gamma(group_data.source, group_data.target)
            domain_rows.append(
                {
                    "group": group.name,
                    "n_source": int(len(group_data.source)),
                    "n_target": int(len(group_data.target)),
                    "domain_auc": _domain_auc(group_data.source, group_data.target),
                    "mmd_rbf": _rbf_mmd(group_data.source, group_data.target, gamma),
                    "a_distance": _a_distance(group_data.source, group_data.target),
                    "gamma": gamma,
                }
            )

        missing_rows.append(
            {
                "group": group.name,
                "source_rows_total": int(len(group_data.source_mask)),
                "source_rows_finite": int(group_data.source_mask.sum()),
                "source_finite_rate": float(group_data.source_mask.mean()),
                "target_rows_total": int(len(group_data.target_mask)),
                "target_rows_finite": int(group_data.target_mask.sum()),
                "target_finite_rate": float(group_data.target_mask.mean()),
            }
        )

    domain_df = pd.DataFrame(domain_rows)
    missing_df = pd.DataFrame(missing_rows)
    domain_df.to_csv(out_dir / "domain_shift_by_group.csv", index=False)
    missing_df.to_csv(out_dir / "feature_finiteness_by_group.csv", index=False)

    transfer_rows: list[dict] = []
    transfer_summary: dict[str, object] = {}

    if args.target_col in df.columns:
        labeled_mask = np.isfinite(df[args.target_col].to_numpy(dtype=float))
        labeled_df = df[labeled_mask].copy()
        labeled_source = labeled_df[labeled_df["domain"] == args.source_domain]
        labeled_target = labeled_df[labeled_df["domain"] == args.target_domain]

        transfer_summary["n_labeled_source"] = int(len(labeled_source))
        transfer_summary["n_labeled_target"] = int(len(labeled_target))

        if (
            len(labeled_source) >= args.min_labeled_per_domain
            and len(labeled_target) >= args.min_labeled_per_domain
        ):
            models = _train_models(args.random_state)
            for group in groups:
                labeled_source_values = labeled_source[f_cols].to_numpy(dtype=float)
                labeled_target_values = labeled_target[f_cols].to_numpy(dtype=float)
                X_source_all = group.select(labeled_source_values)
                X_target_all = group.select(labeled_target_values)
                y_source_all = labeled_source[args.target_col].to_numpy(dtype=float)
                y_target_all = labeled_target[args.target_col].to_numpy(dtype=float)

                source_mask = _finite_row_mask(X_source_all) & np.isfinite(y_source_all)
                target_mask = _finite_row_mask(X_target_all) & np.isfinite(y_target_all)

                X_source = X_source_all[source_mask]
                y_source = y_source_all[source_mask]
                X_target = X_target_all[target_mask]
                y_target = y_target_all[target_mask]

                if len(X_source) < 3 or len(X_target) < 3:
                    for model_name in models:
                        transfer_rows.append(
                            {
                                "group": group.name,
                                "model": model_name,
                                "n_source": int(len(X_source)),
                                "n_target": int(len(X_target)),
                                "r2": None,
                                "rmse": None,
                                "mae": None,
                                "pearson_r": None,
                            }
                        )
                    continue

                for model_name, model in models.items():
                    model.fit(X_source, y_source)
                    preds = model.predict(X_target)
                    metrics = _regression_metrics(y_target, preds)
                    transfer_rows.append(
                        {
                            "group": group.name,
                            "model": model_name,
                            "n_source": int(len(X_source)),
                            "n_target": int(len(X_target)),
                            **metrics,
                        }
                    )
        else:
            transfer_summary["skipped_reason"] = (
                "Not enough labeled samples per domain for cross-domain transfer metrics."
            )
    else:
        transfer_summary["skipped_reason"] = "Target column not found in features file."

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(out_dir / "cross_domain_transfer_by_group.csv", index=False)

    summary_payload = {
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "target_col": args.target_col,
        "total_fused_dim": total_dim,
        "functional_dim": args.functional_dim,
        "esm2_dim": args.esm2_dim,
        "genome_dim": genome_dim,
        "n_source": int(len(source_df)),
        "n_target": int(len(target_df)),
        "transfer": transfer_summary,
    }
    (out_dir / "transferability_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )


if __name__ == "__main__":
    main()
