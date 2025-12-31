import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


def load_data(path: Path, target: str):
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in features file.")
    numeric = df.select_dtypes(include=["number"]).copy()
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' is not numeric.")
    y = numeric.pop(target).to_numpy()
    X = numeric.to_numpy()
    feature_names = list(numeric.columns)
    return X, y, feature_names


def loocv_predictions(model, X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        model_fold = clone(model)
        model_fold.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = model_fold.predict(X[test_idx])[0]
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a flux prediction model.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["loocv", "direct"], default="loocv")
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    X, y, _ = load_data(Path(args.features), args.target)
    payload = joblib.load(args.model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    if args.method == "loocv":
        preds = loocv_predictions(model, X, y)
    else:
        preds = model.predict(X)

    metrics = {
        "r2": float(r2_score(y, preds)) if len(y) > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "pearson_r": float(np.corrcoef(y, preds)[0, 1]) if len(y) > 1 else None,
        "method": args.method,
        "n_samples": int(len(y)),
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    preds_path = Path(args.predictions_out)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y, "y_pred": preds}).to_csv(preds_path, index=False)


if __name__ == "__main__":
    main()
