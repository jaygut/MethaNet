#!/usr/bin/env python3
"""Build a consolidated MethaNet gLM2 integration index from payload outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def payload_summary(payload_dir: Path) -> dict[str, Any]:
    prep = json.loads((payload_dir / "validation/prep_summary.json").read_text())
    infer_path = payload_dir / "validation/glm2_smoke_validation_report.json"
    infer = json.loads(infer_path.read_text()) if infer_path.exists() else {}
    runtime_path = payload_dir / "logs/glm2_runtime_environment.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    hf_path = payload_dir / "logs/hf_model_info_tattabio_glm2_650m.json"
    hf = json.loads(hf_path.read_text()) if hf_path.exists() else {}
    return {
        "payload_dir": str(payload_dir),
        "payload_name": prep.get("payload_name", ""),
        "glm_run_id": prep.get("glm_run_id", ""),
        "payload_mode": prep.get("payload_mode", ""),
        "mag_count": prep.get("mag_count", ""),
        "window_count": prep.get("window_count", ""),
        "span_count": prep.get("span_count", ""),
        "inference_status": infer.get("status", "missing"),
        "embedding_matrix_shape": "x".join(map(str, infer.get("embedding_matrix_shape", []))),
        "elapsed_seconds": infer.get("elapsed_seconds", ""),
        "model_name": runtime.get("model_name", "tattabio/gLM2_650M"),
        "model_revision": runtime.get("model_revision", ""),
        "cuda_device_name": runtime.get("cuda_device_name", ""),
        "hf_model_sha": hf.get("sha", ""),
        "claim_boundary": prep.get("claim_boundary", ""),
    }


def validate_payload_joins(payload_dir: Path) -> list[dict[str, Any]]:
    mags = {
        row["proteome_id"]: row
        for row in read_tsv(payload_dir / "manifests/glm2_smoke_mag_manifest.tsv")
    }
    windows = {
        row["window_id"]: row
        for row in read_tsv(payload_dir / "manifests/dim_glm_window.tsv")
    }
    summary = read_tsv(payload_dir / "features/glm2_smoke_window_embedding_summary.tsv")
    rows = []
    for row in summary:
        window = windows.get(row["window_id"])
        mag = mags.get(row["proteome_id"])
        status = (
            "pass"
            if window
            and mag
            and window["proteome_id"] == row["proteome_id"]
            and window["mag_id"] == row["mag_id"]
            else "fail"
        )
        rows.append(
            {
                "payload_dir": str(payload_dir),
                "glm_run_id": row.get("glm_run_id", ""),
                "window_id": row["window_id"],
                "proteome_id": row["proteome_id"],
                "mag_id": row["mag_id"],
                "window_manifest_join": bool(window),
                "mag_manifest_join": bool(mag),
                "status": status,
            }
        )
    return rows


def mag_features(payload_dir: Path) -> list[dict[str, Any]]:
    manifest = {
        row["proteome_id"]: row
        for row in read_tsv(payload_dir / "manifests/glm2_smoke_mag_manifest.tsv")
    }
    summary = read_tsv(payload_dir / "features/glm2_smoke_window_embedding_summary.tsv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        grouped[(row["proteome_id"], row["mag_id"])].append(row)

    out = []
    for (proteome_id, mag_id), rows in grouped.items():
        native = [row for row in rows if row["window_type"] != "shuffled_gene_order_control"]
        shuffled = [row for row in rows if row["window_type"] == "shuffled_gene_order_control"]
        manifest_row = manifest.get(proteome_id, {})
        native_std = [float(row["embedding_std"]) for row in native]
        shuffled_std = [float(row["embedding_std"]) for row in shuffled]
        token_counts = [int(float(row["token_count"])) for row in rows]
        out.append(
            {
                "glm_run_id": rows[0].get("glm_run_id", ""),
                "model_family": "gLM2",
                "model_name": rows[0].get("model_name", ""),
                "model_revision": rows[0].get("model_revision", ""),
                "proteome_id": proteome_id,
                "mag_id": mag_id,
                "source": manifest_row.get("source", ""),
                "ecosystem": manifest_row.get("ecosystem", ""),
                "domain": manifest_row.get("domain", ""),
                "analysis_unit_type": manifest_row.get("analysis_unit_type", ""),
                "claim_scope": manifest_row.get("claim_scope", ""),
                "payload_name": manifest_row.get("smoke_group", ""),
                "native_window_count": len(native),
                "shuffled_control_count": len(shuffled),
                "all_embeddings_finite": all(row["embedding_finite"] == "True" for row in rows),
                "embedding_dim": rows[0].get("embedding_dim", ""),
                "native_embedding_std_mean": sum(native_std) / len(native_std) if native_std else "",
                "shuffled_embedding_std_mean": sum(shuffled_std) / len(shuffled_std) if shuffled_std else "",
                "max_token_count": max(token_counts) if token_counts else "",
                "context_qc_tier": "breadth_first_one_window_per_mag",
                "allowed_claim_wording": (
                    "gLM2 feature is MAG/proteome-level contextual genomic evidence; "
                    "it complements ESM-2 and functional annotations and is not a final MRV risk score."
                ),
            }
        )
    return out


def build(args: argparse.Namespace) -> int:
    payload_dirs = [Path(path).resolve() for path in args.payload_dir]
    output_dir = Path(args.output_dir).resolve()

    payload_rows = [payload_summary(path) for path in payload_dirs]
    all_windows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_join_rows: list[dict[str, Any]] = []
    all_mag_features: list[dict[str, Any]] = []
    embedding_arrays = []
    embedding_window_ids: list[str] = []
    embedding_proteome_ids: list[str] = []
    embedding_mag_ids: list[str] = []

    for payload_dir in payload_dirs:
        windows = read_tsv(payload_dir / "manifests/dim_glm_window.tsv")
        summary = read_tsv(payload_dir / "features/glm2_smoke_window_embedding_summary.tsv")
        join_rows = validate_payload_joins(payload_dir)
        features = mag_features(payload_dir)

        for row in windows:
            row["payload_dir"] = str(payload_dir)
        for row in summary:
            row["payload_dir"] = str(payload_dir)
        all_windows.extend(windows)
        all_summary.extend(summary)
        all_join_rows.extend(join_rows)
        all_mag_features.extend(features)

        npz_path = payload_dir / "embeddings/glm2_smoke_window_embeddings.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=False)
            embedding_arrays.append(data["embeddings"])
            embedding_window_ids.extend([str(x) for x in data["window_ids"]])
            embedding_proteome_ids.extend([str(x) for x in data["proteome_ids"]])
            summary_by_window = {row["window_id"]: row for row in summary}
            embedding_mag_ids.extend(
                [summary_by_window.get(str(window_id), {}).get("mag_id", "") for window_id in data["window_ids"]]
            )

        write_tsv(
            payload_dir / "validation/id_join_validation.tsv",
            join_rows,
            [
                "payload_dir",
                "glm_run_id",
                "window_id",
                "proteome_id",
                "mag_id",
                "window_manifest_join",
                "mag_manifest_join",
                "status",
            ],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_fields = [
        "payload_dir",
        "payload_name",
        "glm_run_id",
        "payload_mode",
        "mag_count",
        "window_count",
        "span_count",
        "inference_status",
        "embedding_matrix_shape",
        "elapsed_seconds",
        "model_name",
        "model_revision",
        "cuda_device_name",
        "hf_model_sha",
        "claim_boundary",
    ]
    mag_fields = [
        "glm_run_id",
        "model_family",
        "model_name",
        "model_revision",
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "domain",
        "analysis_unit_type",
        "claim_scope",
        "payload_name",
        "native_window_count",
        "shuffled_control_count",
        "all_embeddings_finite",
        "embedding_dim",
        "native_embedding_std_mean",
        "shuffled_embedding_std_mean",
        "max_token_count",
        "context_qc_tier",
        "allowed_claim_wording",
    ]

    write_tsv(output_dir / "payload_index.tsv", payload_rows, payload_fields)
    write_tsv(output_dir / "dim_glm_window.tsv", all_windows, list(all_windows[0].keys()))
    write_tsv(output_dir / "fact_glm_window_embedding_summary.tsv", all_summary, list(all_summary[0].keys()))
    write_tsv(output_dir / "feature_glm_mag_level.tsv", all_mag_features, mag_fields)
    write_tsv(
        output_dir / "validation/id_join_validation.tsv",
        all_join_rows,
        [
            "payload_dir",
            "glm_run_id",
            "window_id",
            "proteome_id",
            "mag_id",
            "window_manifest_join",
            "mag_manifest_join",
            "status",
        ],
    )

    write_parquet(output_dir / "parquet/payload_index.parquet", payload_rows)
    write_parquet(output_dir / "parquet/dim_glm_window.parquet", all_windows)
    write_parquet(output_dir / "parquet/fact_glm_window_embedding_summary.parquet", all_summary)
    write_parquet(output_dir / "parquet/feature_glm_mag_level.parquet", all_mag_features)

    if embedding_arrays:
        matrix = np.vstack(embedding_arrays)
        np.savez_compressed(
            output_dir / "fact_glm_window_embedding_matrix.npz",
            embeddings=matrix,
            window_ids=np.array(embedding_window_ids),
            proteome_ids=np.array(embedding_proteome_ids),
            mag_ids=np.array(embedding_mag_ids),
        )
    else:
        matrix = np.empty((0, 0), dtype=np.float32)

    validation = {
        "payload_count": len(payload_dirs),
        "payloads": payload_rows,
        "window_rows": len(all_windows),
        "embedding_summary_rows": len(all_summary),
        "mag_feature_rows": len(all_mag_features),
        "embedding_matrix_shape": list(matrix.shape),
        "join_failures": sum(1 for row in all_join_rows if row["status"] != "pass"),
        "claim_boundary": (
            "Combined gLM2 outputs are MAG/proteome or quarantined assembly-context feature evidence. "
            "They are not sample-level MRV risk scores and do not replace ESM-2 or functional annotations."
        ),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    return 0 if validation["join_failures"] == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-dir", action="append", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
