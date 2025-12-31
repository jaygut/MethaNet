import argparse
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap

matplotlib.use("Agg")


def load_data(path: Path, target: str):
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in features file.")
    numeric = df.select_dtypes(include=["number"]).copy()
    if target in numeric.columns:
        numeric = numeric.drop(columns=[target])
    if numeric.empty:
        raise ValueError("No numeric feature columns available for SHAP.")
    return numeric


def build_explainer(model, X):
    if hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model)
    return shap.Explainer(model, X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP summary plot.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    X = load_data(Path(args.features), args.target)
    payload = joblib.load(args.model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    explainer = build_explainer(model, X)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)


if __name__ == "__main__":
    main()
