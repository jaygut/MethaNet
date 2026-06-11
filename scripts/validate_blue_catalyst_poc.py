"""Validate and summarize the canonical Blue Catalyst POC artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ARTIFACTS_DIR = Path(
    "results/blue_catalyst_poc/runs/"
    "apolo_full_20260228_080644_embed_20260305_061952/artifacts"
)
DEFAULT_ANALYTICS_SUMMARY = Path(
    "results/blue_catalyst_poc/interim_snapshots/"
    "apolo_full_20260228_080644_embed_20260305_061952_notebook_interim_"
    "20260306_055012/analytics/analytics_summary.json"
)


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("Unknown").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _records(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t").head(n)
    preferred = [
        "sample",
        "source",
        "ecosystem",
        "domain",
        "bridging_score",
        "mixing_coeff",
    ]
    columns = [column for column in preferred if column in df.columns]
    if not columns:
        columns = list(df.columns[: min(6, len(df.columns))])
    return df[columns].where(pd.notna(df[columns]), None).to_dict(orient="records")


def _load_embeddings(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        if "embeddings" in payload.files:
            return np.asarray(payload["embeddings"])
        for key in payload.files:
            array = np.asarray(payload[key])
            if array.ndim == 2:
                return array
    raise ValueError(f"No 2D embedding matrix found in {path}")


def _source_confounding_note(analytics: dict[str, Any]) -> tuple[bool | None, str]:
    assessment = analytics.get("confounding_assessment", {})
    source_confounded = assessment.get("source_confounded")
    if source_confounded is True:
        return (
            True,
            "Ecosystem and source are perfectly confounded in the current POC: "
            "rumen genomes come from PRJEB31266 and wetland genomes from MUCC.",
        )
    if source_confounded is False:
        return False, "Analytics summary reports source-aware cohort balance."
    return None, "Source confounding was not reported in the analytics summary."


def build_report(
    artifacts_dir: Path,
    analytics_summary: Path,
    top_n: int = 10,
    expected_total: int = 662,
    expected_rumen: int = 555,
    expected_wetland: int = 107,
    expected_dim: int = 1280,
) -> dict[str, Any]:
    """Build a source-aware validation report from canonical POC artifacts."""
    embeddings_path = artifacts_dir / "genome_embeddings.npz"
    metadata_path = artifacts_dir / "embedding_metadata.tsv"
    stats_path = artifacts_dir / "embedding_stats.json"
    source_counts_path = artifacts_dir / "sample_source_counts.tsv"
    bridge_path = artifacts_dir / "bridging_genomes_top.tsv"

    embeddings = _load_embeddings(embeddings_path)
    metadata = pd.read_csv(metadata_path, sep="\t")
    stats = _json_load(stats_path)
    analytics = _json_load(analytics_summary)

    final_ecosystem_counts = _value_counts(metadata, "ecosystem")
    source_confounded, source_confounding_note = _source_confounding_note(analytics)

    pre_final_counts: list[dict[str, Any]] = []
    if source_counts_path.exists():
        source_counts = pd.read_csv(source_counts_path, sep="\t")
        pre_final_counts = source_counts.where(
            pd.notna(source_counts), None
        ).to_dict(orient="records")

    finite_mask = np.isfinite(embeddings).all(axis=1)
    validation_errors = []
    if embeddings.shape != (expected_total, expected_dim):
        validation_errors.append(
            "expected embedding shape "
            f"({expected_total}, {expected_dim}), observed {embeddings.shape}"
        )
    if int(finite_mask.sum()) != expected_total:
        validation_errors.append(
            f"expected {expected_total} finite vectors, observed {int(finite_mask.sum())}"
        )
    if final_ecosystem_counts.get("rumen") != expected_rumen:
        validation_errors.append(
            f"expected {expected_rumen} rumen genomes, "
            f"observed {final_ecosystem_counts.get('rumen')}"
        )
    if final_ecosystem_counts.get("wetland") != expected_wetland:
        validation_errors.append(
            f"expected {expected_wetland} wetland genomes, "
            f"observed {final_ecosystem_counts.get('wetland')}"
        )

    report = {
        "status": "ok" if not validation_errors else "failed",
        "validation_errors": validation_errors,
        "artifacts_dir": str(artifacts_dir),
        "analytics_summary": str(analytics_summary),
        "final_embedding_matrix_shape": list(embeddings.shape),
        "finite_vector_count": int(finite_mask.sum()),
        "non_finite_vector_count": int((~finite_mask).sum()),
        "final_counts": {
            "source": _value_counts(metadata, "source"),
            "ecosystem": final_ecosystem_counts,
            "domain": _value_counts(metadata, "domain"),
        },
        "pre_final_source_counts": pre_final_counts,
        "embedding_attrition": {
            "candidate_total": stats.get("candidate_total"),
            "embedded": stats.get("embedded"),
            "no_valid": stats.get("no_valid"),
            "empty_emb": stats.get("empty_emb"),
            "non_finite": stats.get("non_finite"),
            "excluded_coassembly": stats.get("excluded_coassembly"),
            "pending_remaining": stats.get("pending_remaining"),
        },
        "top_bridge_candidates": _records(bridge_path, top_n),
        "source_confounded": source_confounded,
        "source_confounding_caveat": source_confounding_note,
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    shape = report["final_embedding_matrix_shape"]
    counts = report["final_counts"]["ecosystem"]
    lines = [
        "# Blue Catalyst POC Denominator and QC Report",
        "",
        f"- Status: {report['status']}",
        f"- Final embedding matrix: {shape[0]} x {shape[1]}",
        f"- Finite vectors: {report['finite_vector_count']}",
        f"- Final ecosystem counts: {counts}",
        f"- Source confounded: {report['source_confounded']}",
        f"- Caveat: {report['source_confounding_caveat']}",
        "",
        "## Top Bridge Candidates",
        "",
    ]
    for row in report["top_bridge_candidates"]:
        lines.append(
            "- {sample} ({ecosystem}/{source}), score={score}, mixing={mixing}".format(
                sample=row.get("sample"),
                ecosystem=row.get("ecosystem"),
                source=row.get("source"),
                score=row.get("bridging_score"),
                mixing=row.get("mixing_coeff"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Blue Catalyst 662-genome POC artifacts."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--analytics-summary",
        type=Path,
        default=DEFAULT_ANALYTICS_SUMMARY,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--expected-total", type=int, default=662)
    parser.add_argument("--expected-rumen", type=int, default=555)
    parser.add_argument("--expected-wetland", type=int, default=107)
    parser.add_argument("--expected-dim", type=int, default=1280)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when canonical validation checks fail.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        artifacts_dir=args.artifacts_dir,
        analytics_summary=args.analytics_summary,
        top_n=args.top_n,
        expected_total=args.expected_total,
        expected_rumen=args.expected_rumen,
        expected_wetland=args.expected_wetland,
        expected_dim=args.expected_dim,
    )

    payload = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")
    else:
        print(payload)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(report))

    if args.strict and report["validation_errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
