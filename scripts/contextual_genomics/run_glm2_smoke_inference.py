#!/usr/bin/env python3
"""Run gLM2 smoke inference and write MethaNet validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"unavailable: {exc}"


def memory_snapshot() -> dict[str, str]:
    return {
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "torch_cuda_allocated": str(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
        "torch_cuda_reserved": str(torch.cuda.memory_reserved() if torch.cuda.is_available() else 0),
    }


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.to(last_hidden_state.device).unsqueeze(-1)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def main(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir).resolve()
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise SystemExit("CUDA was required but is not visible in this process.")
    dtype = torch.bfloat16 if device == "cuda" and args.dtype == "bfloat16" else torch.float32

    records = read_jsonl(results_dir / "prepared_inputs/glm2_sequences.jsonl")
    if args.max_records:
        records = records[: args.max_records]

    env_record = {
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "hf_home": os.environ.get("HF_HOME", ""),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", ""),
        "record_count": len(records),
    }
    (log_dir / "glm2_runtime_environment.json").write_text(json.dumps(env_record, indent=2) + "\n")
    (log_dir / "nvidia_smi_before.txt").write_text(memory_snapshot()["nvidia_smi"])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.model_revision, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        sequence = record["sequence_string"]
        encodings = tokenizer(
            [sequence],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = getattr(encodings, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = output.last_hidden_state
            pooled = mean_pool(hidden, attention_mask).float().cpu().numpy()[0]
        finite = bool(np.isfinite(pooled).all())
        embeddings.append(pooled)
        summary_rows.append(
            {
                "glm_run_id": record["glm_run_id"],
                "proteome_id": record["proteome_id"],
                "mag_id": record["mag_id"],
                "contig_id": record["contig_id"],
                "window_id": record["window_id"],
                "glm2_sequence_id": record["glm2_sequence_id"],
                "window_type": record["window_type"],
                "control_for_window_id": record.get("control_for_window_id", ""),
                "input_chars": len(sequence),
                "token_count": int(input_ids.shape[1]),
                "embedding_dim": int(pooled.shape[0]),
                "embedding_finite": finite,
                "embedding_mean": float(np.mean(pooled)),
                "embedding_std": float(np.std(pooled)),
                "embedding_min": float(np.min(pooled)),
                "embedding_max": float(np.max(pooled)),
                "device": device,
                "model_name": args.model_name,
                "model_revision": args.model_revision,
            }
        )
        validation_rows.append(
            {
                "check_name": f"finite_embedding_{idx:03d}",
                "status": "pass" if finite else "fail",
                "window_id": record["window_id"],
                "details": f"shape=({pooled.shape[0]}) tokens={int(input_ids.shape[1])}",
            }
        )

    matrix = np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)
    out_npz = results_dir / "embeddings/glm2_smoke_window_embeddings.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        embeddings=matrix,
        window_ids=np.array([row["window_id"] for row in summary_rows]),
        proteome_ids=np.array([row["proteome_id"] for row in summary_rows]),
    )

    validation_rows.extend(
        [
            {
                "check_name": "non_empty_embedding_matrix",
                "status": "pass" if matrix.size > 0 else "fail",
                "window_id": "",
                "details": str(matrix.shape),
            },
            {
                "check_name": "shuffled_gene_order_control_present",
                "status": "pass" if any(row["window_type"] == "shuffled_gene_order_control" for row in summary_rows) else "fail",
                "window_id": "",
                "details": "",
            },
            {
                "check_name": "all_embeddings_finite",
                "status": "pass" if bool(np.isfinite(matrix).all()) else "fail",
                "window_id": "",
                "details": "",
            },
        ]
    )

    fields = [
        "glm_run_id",
        "proteome_id",
        "mag_id",
        "contig_id",
        "window_id",
        "glm2_sequence_id",
        "window_type",
        "control_for_window_id",
        "input_chars",
        "token_count",
        "embedding_dim",
        "embedding_finite",
        "embedding_mean",
        "embedding_std",
        "embedding_min",
        "embedding_max",
        "device",
        "model_name",
        "model_revision",
    ]
    write_tsv(results_dir / "features/glm2_smoke_window_embedding_summary.tsv", summary_rows, fields)
    write_tsv(
        results_dir / "validation/glm2_inference_validation.tsv",
        validation_rows,
        ["check_name", "status", "window_id", "details"],
    )
    (log_dir / "nvidia_smi_after.txt").write_text(memory_snapshot()["nvidia_smi"])
    elapsed = time.time() - started
    report = {
        "status": "pass" if all(row["status"] == "pass" for row in validation_rows) else "fail",
        "elapsed_seconds": elapsed,
        "embedding_matrix_shape": list(matrix.shape),
        "outputs": {
            "embeddings_npz": str(out_npz),
            "summary_tsv": str(results_dir / "features/glm2_smoke_window_embedding_summary.tsv"),
            "validation_tsv": str(results_dir / "validation/glm2_inference_validation.tsv"),
        },
        "claim_boundary": (
            "This is a gLM2 contextual genomic smoke test at MAG/proteome grain. "
            "Finite embeddings support model/load/input compatibility only, not final MRV scoring."
        ),
    }
    (results_dir / "validation/glm2_smoke_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-name", default="tattabio/gLM2_650M")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
