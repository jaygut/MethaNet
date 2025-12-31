import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone


def load_data(path: Path, target: str):
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in features file.")
    numeric = df.select_dtypes(include=["number"]).copy()
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' is not numeric.")
    y = numeric.pop(target).to_numpy()
    X = numeric.to_numpy()
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap prediction intervals.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    X, y = load_data(Path(args.features), args.target)
    payload = joblib.load(args.model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    rng = np.random.default_rng(42)
    preds = []
    n = len(X)
    for _ in range(args.n):
        idx = rng.choice(n, n, replace=True)
        model_fold = clone(model)
        model_fold.fit(X[idx], y[idx])
        preds.append(model_fold.predict(X))

    preds = np.asarray(preds)
    lower = np.percentile(preds, (1 - args.ci) / 2 * 100, axis=0)
    upper = np.percentile(preds, (1 + args.ci) / 2 * 100, axis=0)
    mean_pred = preds.mean(axis=0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mean": mean_pred, "lower": lower, "upper": upper}).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
