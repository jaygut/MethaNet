import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import pairwise_distances


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("No numeric columns found in features file.")
    return numeric


def rbf_mmd(source: np.ndarray, target: np.ndarray, gamma: float) -> float:
    ss = np.exp(-gamma * pairwise_distances(source, source, metric="sqeuclidean"))
    tt = np.exp(-gamma * pairwise_distances(target, target, metric="sqeuclidean"))
    st = np.exp(-gamma * pairwise_distances(source, target, metric="sqeuclidean"))
    return float(ss.mean() + tt.mean() - 2 * st.mean())


def median_heuristic_gamma(source: np.ndarray, target: np.ndarray) -> float:
    combined = np.vstack([source, target])
    if combined.shape[0] > 256:
        rng = np.random.default_rng(42)
        idx = rng.choice(combined.shape[0], size=256, replace=False)
        combined = combined[idx]
    distances = pairwise_distances(combined, metric="euclidean")
    median = np.median(distances)
    if median <= 0:
        return 1.0
    return 1.0 / (2 * median**2)


def a_distance(source: np.ndarray, target: np.ndarray) -> float:
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv).mean()
    error = 1.0 - acc
    return 2 * (1 - 2 * error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure domain shift metrics.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = load_features(Path(args.source))
    target = load_features(Path(args.target))

    shared = sorted(set(source.columns).intersection(target.columns))
    if not shared:
        raise ValueError("No shared numeric columns between source and target.")

    source_vals = source[shared].to_numpy()
    target_vals = target[shared].to_numpy()

    gamma = median_heuristic_gamma(source_vals, target_vals)
    metrics = {
        "mmd_rbf": rbf_mmd(source_vals, target_vals, gamma),
        "a_distance": a_distance(source_vals, target_vals),
        "n_source": int(source_vals.shape[0]),
        "n_target": int(target_vals.shape[0]),
        "n_features": int(source_vals.shape[1]),
        "gamma": gamma,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
