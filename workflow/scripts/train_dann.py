import argparse
import json
from itertools import cycle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def load_features(path: Path, label_column: str | None = None):
    df = pd.read_parquet(path)
    numeric = df.select_dtypes(include=["number"]).copy()
    if label_column and label_column in numeric.columns:
        y = numeric.pop(label_column).to_numpy()
    else:
        y = None
    if numeric.empty:
        raise ValueError("No numeric features found.")
    return numeric, y


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DANN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.domain_classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, alpha: float):
        features = self.feature_extractor(x)
        reg_out = self.regressor(features)
        reversed_features = GradientReversal.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return reg_out, domain_logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple DANN model.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--label-column", default="measured_flux")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    source_df, source_y = load_features(Path(args.source), args.label_column)
    if source_y is None:
        raise ValueError(f"Label column '{args.label_column}' not found in source.")
    target_df, _ = load_features(Path(args.target), None)

    shared = sorted(set(source_df.columns).intersection(target_df.columns))
    if not shared:
        raise ValueError("No shared numeric columns between source and target.")

    source_df = source_df[shared]
    target_df = target_df[shared]

    scaler = StandardScaler()
    source_x = scaler.fit_transform(source_df.to_numpy())
    target_x = scaler.transform(target_df.to_numpy())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DANN(input_dim=source_x.shape[1], hidden_dim=args.hidden_dim).to(device)

    src_tensor = torch.tensor(source_x, dtype=torch.float32)
    tgt_tensor = torch.tensor(target_x, dtype=torch.float32)
    y_tensor = torch.tensor(source_y, dtype=torch.float32).view(-1, 1)

    src_dataset = torch.utils.data.TensorDataset(src_tensor, y_tensor)
    tgt_dataset = torch.utils.data.TensorDataset(tgt_tensor)

    src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True)
    tgt_loader = torch.utils.data.DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reg_loss_fn = nn.MSELoss()
    dom_loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(args.epochs):
        for (src_batch, y_batch), (tgt_batch,) in zip(src_loader, cycle(tgt_loader)):
            src_batch = src_batch.to(device)
            y_batch = y_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            combined = torch.cat([src_batch, tgt_batch], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(len(src_batch), dtype=torch.long),
                    torch.ones(len(tgt_batch), dtype=torch.long),
                ]
            ).to(device)

            reg_out, domain_logits = model(combined, args.alpha)
            reg_loss = reg_loss_fn(reg_out[: len(src_batch)], y_batch)
            dom_loss = dom_loss_fn(domain_logits, domain_labels)
            loss = reg_loss + dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        src_pred, _ = model(src_tensor.to(device), args.alpha)
        reg_mse = reg_loss_fn(src_pred.cpu(), y_tensor).item()
        domain_logits = model(torch.cat([src_tensor, tgt_tensor]).to(device), args.alpha)[1]
        domain_labels = torch.cat(
            [
                torch.zeros(len(src_tensor), dtype=torch.long),
                torch.ones(len(tgt_tensor), dtype=torch.long),
            ]
        )
        domain_pred = domain_logits.cpu().argmax(dim=1)
        domain_acc = (domain_pred == domain_labels).float().mean().item()

    metrics = {
        "regression_mse": reg_mse,
        "domain_accuracy": domain_acc,
        "epochs": args.epochs,
        "input_dim": int(source_x.shape[1]),
        "hidden_dim": args.hidden_dim,
    }

    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(source_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_names": shared,
        },
        model_path,
    )

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
