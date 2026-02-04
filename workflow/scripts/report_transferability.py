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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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


def _a_distance(source: np.ndarray, target: np.ndarray) -> float | None:
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    clf = LogisticRegression(max_iter=1000)
    min_class = min(len(source), len(target))
    n_splits = min(5, min_class)
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv).mean()
    error = 1.0 - acc
    return float(2 * (1 - 2 * error))


def _domain_auc(source: np.ndarray, target: np.ndarray) -> float | None:
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    clf = LogisticRegression(max_iter=1000)
    min_class = min(len(source), len(target))
    n_splits = min(5, min_class)
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
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


def _load_metrics_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _dnabert2_metrics_summary(
    df: pd.DataFrame,
    metrics_dir: Path,
    suffix: str,
) -> dict[str, object]:
    rows = []
    missing = 0
    for sample_id, domain in zip(df["sample_id"].astype(str), df["domain"].astype(str)):
        metrics_path = metrics_dir / f"{sample_id}{suffix}"
        if not metrics_path.exists():
            missing += 1
            continue
        payload = _load_metrics_json(metrics_path)
        payload["sample_id"] = sample_id
        payload["domain"] = domain
        rows.append(payload)

    summary: dict[str, object] = {
        "metrics_dir": str(metrics_dir),
        "suffix": suffix,
        "n_samples": int(len(df)),
        "n_with_metrics": int(len(rows)),
        "n_missing_metrics": int(missing),
        "by_domain": {},
    }
    if not rows:
        return summary

    metrics_df = pd.DataFrame(rows)
    if "frac_truncated" not in metrics_df.columns:
        return summary

    by_domain = {}
    for domain, sub in metrics_df.groupby("domain"):
        frac = pd.to_numeric(sub["frac_truncated"], errors="coerce").dropna()
        if "mean_truncated_tokens" in sub.columns:
            trunc_series = sub["mean_truncated_tokens"]
        else:
            trunc_series = pd.Series(dtype=float)
        mean_trunc_tokens = pd.to_numeric(trunc_series, errors="coerce").dropna()
        by_domain[str(domain)] = {
            "n_samples": int(len(sub)),
            "mean_frac_truncated": float(frac.mean()) if len(frac) else None,
            "p90_frac_truncated": float(frac.quantile(0.9)) if len(frac) else None,
            "mean_truncated_tokens": float(mean_trunc_tokens.mean())
            if len(mean_trunc_tokens)
            else None,
        }

    summary["by_domain"] = by_domain
    return summary


def _bootstrap_delta_mae(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float | None]:
    if len(y_true) < 2:
        return {
            "delta_mae": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mae_a = mean_absolute_error(y_true[idx], preds_a[idx])
        mae_b = mean_absolute_error(y_true[idx], preds_b[idx])
        deltas[i] = mae_a - mae_b
    return {
        "delta_mae": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transferability and ablation diagnostics.")
    parser.add_argument("--features", required=True, help="Path to fused features parquet.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-domain", default="rumen")
    parser.add_argument("--target-domain", default="coastal")
    parser.add_argument("--target-col", default="flux_value")
    parser.add_argument("--functional-dim", type=int, default=13)
    parser.add_argument("--esm2-dim", type=int, default=1280)
    parser.add_argument("--genome-dim", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--gating-min-finite-rate", type=float, default=0.9)
    parser.add_argument("--gating-max-a-distance", type=float, default=1.2)
    parser.add_argument(
        "--dnabert2-metrics-dir",
        default=None,
        help="Optional directory containing per-sample *_dnabert2_metrics.json files.",
    )
    parser.add_argument("--dnabert2-metrics-suffix", default="_dnabert2_metrics.json")
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
    if "sample_id" not in df.columns:
        raise ValueError("Features parquet must include a 'sample_id' column.")

    if args.target_col not in df.columns:
        fallback_targets = ["flux_value", "measured_flux"]
        if args.target_col in fallback_targets:
            fallback_targets.remove(args.target_col)
        resolved = next((col for col in fallback_targets if col in df.columns), None)
        if resolved is None:
            raise ValueError(
                "Target column not found in features parquet. "
                f"Requested='{args.target_col}'. Available fallbacks={fallback_targets}."
            )
        args.target_col = resolved

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

    domain_by_group = {row["group"]: row for row in domain_rows}
    missing_by_group = {row["group"]: row for row in missing_rows}

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

    gating_payload: dict[str, object] = {
        "recommended_features": "functional",
        "allow_embeddings": False,
        "confidence": "low",
        "reasons": [],
        "delta_mae": None,
        "delta_mae_ci_lower": None,
        "delta_mae_ci_upper": None,
    }

    all_missing = missing_by_group.get("all")
    all_shift = domain_by_group.get("all")
    if all_missing is not None:
        gating_payload["all_source_finite_rate"] = all_missing.get("source_finite_rate")
        gating_payload["all_target_finite_rate"] = all_missing.get("target_finite_rate")
        if (
            all_missing.get("source_finite_rate") is not None
            and all_missing.get("source_finite_rate") < args.gating_min_finite_rate
        ):
            gating_payload["reasons"].append("low_source_finite_rate")
        if (
            all_missing.get("target_finite_rate") is not None
            and all_missing.get("target_finite_rate") < args.gating_min_finite_rate
        ):
            gating_payload["reasons"].append("low_target_finite_rate")

    if all_shift is not None:
        gating_payload["all_a_distance"] = all_shift.get("a_distance")
        if (
            all_shift.get("a_distance") is not None
            and all_shift.get("a_distance") > args.gating_max_a_distance
        ):
            gating_payload["reasons"].append("high_a_distance")

    if args.target_col in df.columns:
        labeled_mask = np.isfinite(df[args.target_col].to_numpy(dtype=float))
        labeled_df = df[labeled_mask].copy()
        labeled_source = labeled_df[labeled_df["domain"] == args.source_domain]
        labeled_target = labeled_df[labeled_df["domain"] == args.target_domain]

        if (
            len(labeled_source) >= args.min_labeled_per_domain
            and len(labeled_target) >= args.min_labeled_per_domain
        ):
            group_map = {g.name: g for g in groups}
            group_all = group_map.get("all")
            group_func = group_map.get("functional")

            if group_all is not None and group_func is not None:
                models = _train_models(args.random_state)
                model = models["linear"]

                src_vals = labeled_source[f_cols].to_numpy(dtype=float)
                tgt_vals = labeled_target[f_cols].to_numpy(dtype=float)
                y_src_all = labeled_source[args.target_col].to_numpy(dtype=float)
                y_tgt_all = labeled_target[args.target_col].to_numpy(dtype=float)

                Xs_all = group_all.select(src_vals)
                Xt_all = group_all.select(tgt_vals)
                src_mask_all = _finite_row_mask(Xs_all) & np.isfinite(y_src_all)
                tgt_mask_all = _finite_row_mask(Xt_all) & np.isfinite(y_tgt_all)

                Xs_func = group_func.select(src_vals)
                Xt_func = group_func.select(tgt_vals)
                src_mask_func = _finite_row_mask(Xs_func) & np.isfinite(y_src_all)

                if int(tgt_mask_all.sum()) < 3:
                    gating_payload["reasons"].append("too_few_target_samples_for_all")
                elif int(src_mask_all.sum()) < 3 or int(src_mask_func.sum()) < 3:
                    gating_payload["reasons"].append("too_few_source_samples")
                else:
                    model.fit(Xs_all[src_mask_all], y_src_all[src_mask_all])
                    preds_all = model.predict(Xt_all[tgt_mask_all])
                    y_tgt = y_tgt_all[tgt_mask_all]

                    model.fit(Xs_func[src_mask_func], y_src_all[src_mask_func])
                    preds_func = model.predict(Xt_func[tgt_mask_all])

                    delta = _bootstrap_delta_mae(
                        y_tgt,
                        preds_all,
                        preds_func,
                        n_boot=args.bootstrap_samples,
                        seed=args.random_state,
                    )
                    gating_payload["delta_mae"] = delta["delta_mae"]
                    gating_payload["delta_mae_ci_lower"] = delta["ci_lower"]
                    gating_payload["delta_mae_ci_upper"] = delta["ci_upper"]

                    if delta["ci_upper"] is not None and delta["ci_upper"] < 0:
                        if not gating_payload["reasons"]:
                            gating_payload["recommended_features"] = "all"
                            gating_payload["allow_embeddings"] = True
                            gating_payload["confidence"] = "high"
                        else:
                            gating_payload["confidence"] = "medium"
                    else:
                        gating_payload["reasons"].append("no_clear_mae_improvement")
        else:
            gating_payload["reasons"].append("insufficient_labeled_per_domain")
    else:
        gating_payload["reasons"].append("target_column_missing")

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

    dnabert2_payload = None
    if args.dnabert2_metrics_dir:
        dnabert2_payload = _dnabert2_metrics_summary(
            df,
            metrics_dir=Path(args.dnabert2_metrics_dir),
            suffix=args.dnabert2_metrics_suffix,
        )

    report_payload = {
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "target_col": args.target_col,
        "feature_dims": {
            "total_fused": total_dim,
            "functional": args.functional_dim,
            "esm2": args.esm2_dim,
            "genome": genome_dim,
        },
        "n_samples": {
            "source": int(len(source_df)),
            "target": int(len(target_df)),
        },
        "missingness_by_group": missing_rows,
        "domain_shift_by_group": domain_rows,
        "cross_domain_transfer": transfer_rows,
        "gating": gating_payload,
        "dnabert2_truncation": dnabert2_payload,
    }

    report_path = Path(args.report_out) if args.report_out else out_dir / "poc_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
    main()
