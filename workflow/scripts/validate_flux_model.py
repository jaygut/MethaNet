import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from flux_utils import load_labeled_flux_data
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


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

    payload = joblib.load(args.model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    trained_features = payload.get("feature_names") if isinstance(payload, dict) else None
    trained_target = payload.get("target") if isinstance(payload, dict) else None
    target_name = trained_target or args.target

    flux_data = load_labeled_flux_data(
        Path(args.features),
        target_name,
        feature_names=trained_features,
    )
    X = flux_data.X
    y = flux_data.y

    method = args.method
    if method == "loocv" and len(y) < 2:
        method = "direct"

    if method == "loocv":
        preds = loocv_predictions(model, X, y)
    else:
        preds = model.predict(X)

    pearson_r = None
    if len(y) > 1:
        pearson_r = float(np.corrcoef(y, preds)[0, 1])
    metrics = {
        "r2": float(r2_score(y, preds)) if len(y) > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "pearson_r": pearson_r,
        "method": method,
        "n_samples": int(len(y)),
        "data_stats": flux_data.stats,
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    preds_path = Path(args.predictions_out)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(
        {
            "sample_id": flux_data.sample_ids,
            "y_true": y,
            "y_pred": preds,
        }
    ).to_csv(preds_path, index=False)


if __name__ == "__main__":
    main()
