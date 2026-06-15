"""Core MBAG graph, transport, leakage, and scoring primitives.

These functions are intentionally small and deterministic. They operate on
already-loaded arrays/data frames, which keeps them easy to unit test and lets
the data-loading/reporting layer enforce MethaNet-specific artifact contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


EPS = 1e-12


@dataclass(frozen=True)
class GraphResult:
    """Directed kNN graph plus node-level cross-domain metrics."""

    edges: pd.DataFrame
    node_metrics: pd.DataFrame


@dataclass(frozen=True)
class TransportResult:
    """Entropic optimal-transport coupling between two domains."""

    couplings: pd.DataFrame
    node_metrics: pd.DataFrame
    cost_summary: dict[str, float]


@dataclass(frozen=True)
class LeakageAudit:
    """Source/domain leakage audit from a representation."""

    status: str
    label: str
    n_samples: int
    n_classes: int
    balanced_accuracy: float | None
    roc_auc: float | None
    message: str


def _as_1d_float(values: Iterable[float] | np.ndarray, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array")
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr


def _safe_zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    std = vals.std(ddof=0)
    if not np.isfinite(std) or std < EPS:
        return pd.Series(np.zeros(len(vals)), index=values.index)
    return (vals - vals.mean()) / std


def compute_reliability_weight(
    completeness: float | None,
    contamination: float | None,
    gunc_pass: bool | None,
    coverage_required: float | None,
    taxonomy_status: str | None = None,
) -> float:
    """Compute a bounded smoke-test reliability weight.

    This follows the audited framework: the minimum required layer should limit
    confidence, while contamination and taxonomy act as multiplicative
    penalties. The output is an engineering prior, not a calibrated QC score.
    """

    comp = 0.0 if completeness is None or not np.isfinite(completeness) else float(completeness)
    contam = 0.0 if contamination is None or not np.isfinite(contamination) else float(contamination)
    cov = 0.0 if coverage_required is None or not np.isfinite(coverage_required) else float(coverage_required)

    q_completeness = np.clip((comp - 50.0) / 40.0, 0.0, 1.0)
    q_contamination = float(np.exp(-max(contam - 5.0, 0.0) / 5.0))
    q_gunc = 1.0 if gunc_pass is True else 0.6 if gunc_pass is False else 0.5

    text = (taxonomy_status or "").lower()
    if any(token in text for token in ("genus", "species", "family", "resolved")):
        q_taxonomy = 1.0
    elif any(token in text for token in ("order", "class", "phylum", "domain")):
        q_taxonomy = 0.8
    elif text:
        q_taxonomy = 0.6
    else:
        q_taxonomy = 0.5

    required_floor = min(float(q_completeness), float(q_gunc), float(np.clip(cov, 0.0, 1.0)))
    weight = required_floor * q_contamination * q_taxonomy
    return float(np.clip(weight, 0.0, 1.0))


def build_knn_graph(
    ids: Iterable[str],
    matrix: np.ndarray,
    domains: Iterable[str],
    *,
    k: int = 10,
    metric: str = "cosine",
    reliability: Iterable[float] | None = None,
) -> GraphResult:
    """Build a directed kNN graph and cross-domain neighborhood metrics."""

    ids_arr = np.asarray(list(ids), dtype=object)
    domains_arr = np.asarray(list(domains), dtype=object)
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if len(ids_arr) != X.shape[0] or len(domains_arr) != X.shape[0]:
        raise ValueError("ids, domains, and matrix row count must match")
    if X.shape[0] < 2:
        raise ValueError("At least two rows are required to build a kNN graph")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rel = np.ones(X.shape[0], dtype=float)
    if reliability is not None:
        rel = np.clip(_as_1d_float(np.asarray(list(reliability), dtype=float)), 0.0, 1.0)
        if len(rel) != X.shape[0]:
            raise ValueError("reliability length must match matrix row count")

    n_neighbors = min(max(int(k), 1) + 1, X.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    positive = distances[distances > EPS]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, EPS)

    rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for i, node_id in enumerate(ids_arr):
        cross_count = 0
        cross_weight = 0.0
        total_weight = 0.0
        observed_neighbors = 0
        for rank, (j, dist) in enumerate(zip(indices[i], distances[i]), start=0):
            if j == i:
                continue
            observed_neighbors += 1
            raw_similarity = float(np.exp(-float(dist) ** 2 / sigma**2))
            weight = raw_similarity * float(np.sqrt(rel[i] * rel[j]))
            cross = bool(domains_arr[i] != domains_arr[j])
            cross_count += int(cross)
            cross_weight += weight if cross else 0.0
            total_weight += weight
            rows.append(
                {
                    "source_id": str(node_id),
                    "target_id": str(ids_arr[j]),
                    "source_domain": str(domains_arr[i]),
                    "target_domain": str(domains_arr[j]),
                    "neighbor_rank": observed_neighbors,
                    "distance": float(dist),
                    "raw_similarity": raw_similarity,
                    "weight": weight,
                    "cross_domain": cross,
                }
            )
        denom = max(observed_neighbors, 1)
        metrics.append(
            {
                "proteome_id": str(node_id),
                "domain_label": str(domains_arr[i]),
                "knn_k": observed_neighbors,
                "cross_domain_neighbor_count": cross_count,
                "cross_domain_neighbor_fraction": cross_count / denom,
                "cross_domain_weight": cross_weight,
                "total_neighbor_weight": total_weight,
                "cross_domain_weight_fraction": cross_weight / total_weight if total_weight > EPS else 0.0,
            }
        )

    return GraphResult(edges=pd.DataFrame(rows), node_metrics=pd.DataFrame(metrics))


def sinkhorn_transport(
    source_ids: Iterable[str],
    target_ids: Iterable[str],
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    metric: str = "cosine",
    epsilon: float = 0.05,
    max_iter: int = 300,
    tol: float = 1e-8,
    top_per_source: int = 5,
) -> TransportResult:
    """Compute an entropic transport plan and sparse top couplings.

    The returned per-node support uses the strongest coupling(s), not total row
    mass, because the marginal constraints make total row/column mass constant.
    """

    source_ids_arr = np.asarray(list(source_ids), dtype=object)
    target_ids_arr = np.asarray(list(target_ids), dtype=object)
    Xs = np.nan_to_num(np.asarray(source_matrix, dtype=float), nan=0.0)
    Xt = np.nan_to_num(np.asarray(target_matrix, dtype=float), nan=0.0)
    if Xs.ndim != 2 or Xt.ndim != 2:
        raise ValueError("source_matrix and target_matrix must be two-dimensional")
    if len(source_ids_arr) != Xs.shape[0] or len(target_ids_arr) != Xt.shape[0]:
        raise ValueError("ids and matrix row counts must match")
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError("source and target matrices must have the same column count")
    if Xs.shape[0] == 0 or Xt.shape[0] == 0:
        raise ValueError("source and target matrices must both contain rows")

    C = pairwise_distances(Xs, Xt, metric=metric)
    finite = C[np.isfinite(C)]
    scale = float(np.median(finite[finite > EPS])) if finite.size and np.any(finite > EPS) else 1.0
    C = C / max(scale, EPS)
    eps = max(float(epsilon), EPS)
    K = np.exp(-C / eps)
    K = np.maximum(K, EPS)

    u = np.ones(Xs.shape[0], dtype=float) / Xs.shape[0]
    v = np.ones(Xt.shape[0], dtype=float) / Xt.shape[0]
    a = np.ones_like(u)
    b = np.ones_like(v)
    for _ in range(max_iter):
        a_prev = a.copy()
        a = u / np.maximum(K @ b, EPS)
        b = v / np.maximum(K.T @ a, EPS)
        if np.max(np.abs(a - a_prev)) < tol:
            break
    gamma = (a[:, None] * K) * b[None, :]

    top_k = min(max(int(top_per_source), 1), Xt.shape[0])
    coupling_rows: list[dict[str, object]] = []
    source_metrics: list[dict[str, object]] = []
    target_best = np.zeros(Xt.shape[0], dtype=float)
    for i, source_id in enumerate(source_ids_arr):
        order = np.argsort(gamma[i])[::-1][:top_k]
        top_mass = gamma[i, order]
        source_metrics.append(
            {
                "proteome_id": str(source_id),
                "ot_domain": "source",
                "ot_best_coupling": float(top_mass[0]),
                "ot_top_mean_coupling": float(np.mean(top_mass)),
                "ot_partner": str(target_ids_arr[order[0]]),
                "ot_partner_cost": float(C[i, order[0]]),
            }
        )
        for rank, j in enumerate(order, start=1):
            target_best[j] = max(target_best[j], gamma[i, j])
            coupling_rows.append(
                {
                    "source_id": str(source_id),
                    "target_id": str(target_ids_arr[j]),
                    "coupling_rank": rank,
                    "coupling": float(gamma[i, j]),
                    "cost": float(C[i, j]),
                }
            )

    for j, target_id in enumerate(target_ids_arr):
        order = np.argsort(gamma[:, j])[::-1]
        best_i = int(order[0])
        source_metrics.append(
            {
                "proteome_id": str(target_id),
                "ot_domain": "target",
                "ot_best_coupling": float(gamma[best_i, j]),
                "ot_top_mean_coupling": float(np.mean(gamma[order[: min(top_k, len(order))], j])),
                "ot_partner": str(source_ids_arr[best_i]),
                "ot_partner_cost": float(C[best_i, j]),
            }
        )

    return TransportResult(
        couplings=pd.DataFrame(coupling_rows),
        node_metrics=pd.DataFrame(source_metrics),
        cost_summary={
            "cost_min": float(np.min(C)),
            "cost_median": float(np.median(C)),
            "cost_max": float(np.max(C)),
            "epsilon": eps,
            "row_mass_target": float(u[0]),
            "column_mass_target": float(v[0]),
        },
    )


def source_leakage_audit(
    matrix: np.ndarray,
    labels: Iterable[str],
    *,
    label_name: str = "source",
    random_state: int = 42,
) -> LeakageAudit:
    """Estimate how recoverable a source/domain label is from a feature space."""

    X = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    y_raw = np.asarray(list(labels), dtype=object)
    mask = pd.Series(y_raw).notna().to_numpy()
    X = X[mask]
    y_raw = y_raw[mask]
    if X.shape[0] < 4 or len(np.unique(y_raw)) < 2:
        return LeakageAudit("not_applicable", label_name, int(X.shape[0]), int(len(np.unique(y_raw))), None, None, "Need at least two classes and four samples")

    counts = pd.Series(y_raw).value_counts()
    min_class = int(counts.min())
    if min_class < 2:
        return LeakageAudit("not_applicable", label_name, int(X.shape[0]), int(len(counts)), None, None, "Smallest class has fewer than two samples")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    balanced = float(balanced_accuracy_score(y, pred))

    auc: float | None = None
    if len(np.unique(y)) == 2:
        try:
            proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, proba))
        except ValueError:
            auc = None

    status = "warn_high_leakage" if balanced >= 0.8 else "pass_low_to_moderate_leakage"
    message = (
        "Source/domain labels are highly recoverable; bridge claims need strong caveats"
        if status == "warn_high_leakage"
        else "Source/domain labels are not trivially recovered under this audit"
    )
    return LeakageAudit(status, label_name, int(X.shape[0]), int(len(counts)), balanced, auc, message)


def provisional_bridge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute an auditable provisional MBAG score from named components."""

    required = [
        "cross_domain_neighbor_fraction",
        "ot_best_coupling",
        "mixing_coeff",
        "functional_concordance",
        "mechanism_support",
        "candidate_specificity",
        "qc_penalty",
        "annotation_missingness",
        "source_leakage_penalty",
    ]
    out = df.copy()
    for column in required:
        if column not in out.columns:
            out[column] = 0.0
    positive = [
        "cross_domain_neighbor_fraction",
        "ot_best_coupling",
        "mixing_coeff",
        "functional_concordance",
        "mechanism_support",
        "candidate_specificity",
    ]
    negative = ["qc_penalty", "annotation_missingness", "source_leakage_penalty"]
    score = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    for column in positive:
        score = score + _safe_zscore(out[column])
    for column in negative:
        score = score - _safe_zscore(out[column])
    out["mbag_score_provisional"] = score
    out["mbag_score_status"] = "provisional_internal"
    return out
