import argparse
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap

matplotlib.use("Agg")

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

    payload = joblib.load(args.model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    trained_features = payload.get("feature_names") if isinstance(payload, dict) else None
    trained_target = payload.get("target") if isinstance(payload, dict) else None
    target_name = trained_target or args.target

    from flux_utils import load_labeled_flux_data

    flux_data = load_labeled_flux_data(
        Path(args.features),
        target_name,
        feature_names=trained_features,
    )
    X = pd.DataFrame(flux_data.X, columns=flux_data.feature_names)

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
