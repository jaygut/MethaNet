import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def build_model(name: str, random_state: int):
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=random_state)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is not installed.") from exc
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is not installed.") from exc
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model type: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flux prediction model.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    X, y, feature_names = load_data(Path(args.features), args.target)
    model = build_model(args.model, args.random_state)
    model.fit(X, y)

    preds = model.predict(X)
    metrics = {
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "pearson_r": float(np.corrcoef(y, preds)[0, 1]) if len(y) > 1 else None,
        "model": args.model,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
    }

    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names, "target": args.target}, model_path)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
