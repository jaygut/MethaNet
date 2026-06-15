#!/usr/bin/env python3
"""Merge MSM China 2025 functional-run QC outputs into reconciliation tables."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_MANIFEST = Path(
    "results/functional_metagenomics/msm_china_2025_20260615/manifests/"
    "msm_china_2025_functional_mag_manifest.tsv"
)
DEFAULT_RESULT_ROOT = Path("results/functional_metagenomics/msm_china_2025_20260615")
DEFAULT_OUTDIR = DEFAULT_RESULT_ROOT / "qc_reconciliation"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latest_run_dir(per_mag_dir: Path) -> Path | None:
    if not per_mag_dir.is_dir():
        return None
    run_dirs = [path for path in per_mag_dir.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs[0] if run_dirs else None


def read_single_tsv(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    rows = read_rows(path)
    return rows[0] if rows else {}


def status_for_run(run_dir: Path | None) -> str:
    if run_dir is None:
        return "not_started"
    if (run_dir / "COMPLETE").is_file():
        return "complete"
    if (run_dir / "FAILED").is_file():
        return "failed"
    if (run_dir / "status.tsv").is_file():
        return "partial"
    return "attempt_created"


def float_or_blank(value: str | None) -> float | str:
    if value in (None, ""):
        return ""
    try:
        return float(value)
    except ValueError:
        return ""


def mimag_quality_status(completeness: object, contamination: object) -> str:
    if not isinstance(completeness, float) or not isinstance(contamination, float):
        return "pending_qc_metrics"
    if completeness >= 90 and contamination <= 5:
        return "local_high_quality_like"
    if completeness >= 50 and contamination <= 10:
        return "local_medium_quality_like"
    return "local_quality_gate_fail"


def qc_evidence_status(run_status: str, completeness: object, contamination: object) -> str:
    if not isinstance(completeness, float) or not isinstance(contamination, float):
        return "pending_qc_metrics"
    if run_status == "complete":
        return "complete_run_qc_metrics"
    return "partial_run_qc_metrics"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    result_root = args.result_root if args.result_root.is_absolute() else repo_root / args.result_root
    outdir = args.outdir if args.outdir.is_absolute() else repo_root / args.outdir
    per_mag_root = result_root / "per_mag"

    rows = read_rows(manifest)
    out: list[dict[str, object]] = []
    for row in rows:
        proteome_id = row["proteome_id"]
        run_dir = latest_run_dir(per_mag_root / proteome_id)
        run_status = status_for_run(run_dir)
        checkm2 = read_single_tsv(run_dir / "checkm2/quality_report.tsv") if run_dir else {}
        gunc_files = sorted((run_dir / "gunc").glob("*maxCSS_level.tsv")) if run_dir else []
        gunc = read_single_tsv(gunc_files[0]) if gunc_files else {}
        completeness = float_or_blank(checkm2.get("Completeness"))
        contamination = float_or_blank(checkm2.get("Contamination"))
        quality_status = mimag_quality_status(completeness, contamination)
        evidence_status = qc_evidence_status(run_status, completeness, contamination)
        is_quality_candidate = quality_status in {
            "local_high_quality_like",
            "local_medium_quality_like",
        }
        out.append({
            "proteome_id": proteome_id,
            "mag_id": row["mag_id"],
            "source_group": row.get("source_group", ""),
            "domain": row.get("domain", ""),
            "phylum": row.get("phylum", ""),
            "protein_count": row.get("protein_count", ""),
            "run_status": run_status,
            "run_dir": str(run_dir.relative_to(repo_root)) if run_dir else "",
            "checkm2_completeness": completeness,
            "checkm2_contamination": contamination,
            "checkm2_model": checkm2.get("Completeness_Model_Used", ""),
            "gunc_pass": gunc.get("pass.GUNC", ""),
            "gunc_css": gunc.get("clade_separation_score", ""),
            "gunc_contamination_portion": gunc.get("contamination_portion", ""),
            "qc_evidence_status": evidence_status,
            "local_mimag_quality_status": quality_status,
            "published_966_reconciliation_status": (
                "candidate_for_966_denominator"
                if run_status == "complete" and is_quality_candidate
                else "pending_terminal_run_before_reconciliation"
                if run_status == "partial" and is_quality_candidate
                else "not_yet_reconciled"
            ),
        })

    fieldnames = list(out[0].keys()) if out else []
    write_tsv(outdir / "msm_china_2025_qc_reconciliation_with_checkm2.tsv", out, fieldnames)

    counts = Counter(str(row["local_mimag_quality_status"]) for row in out)
    evidence_counts = Counter(str(row["qc_evidence_status"]) for row in out)
    status_counts = Counter(str(row["run_status"]) for row in out)
    gunc_observed = [row for row in out if row["gunc_pass"] not in ("", None)]
    gunc_completed = [
        row for row in gunc_observed if row["run_status"] == "complete"
    ]
    completed_quality_counts = Counter(
        str(row["local_mimag_quality_status"]) for row in out if row["run_status"] == "complete"
    )
    observed_quality_gate_pass_total = (
        counts.get("local_high_quality_like", 0) + counts.get("local_medium_quality_like", 0)
    )
    completed_quality_gate_pass_total = (
        completed_quality_counts.get("local_high_quality_like", 0)
        + completed_quality_counts.get("local_medium_quality_like", 0)
    )
    summary = [
        {"metric": "manifest_rows", "value": len(out), "interpretation": "full MSM archive denominator"},
        {"metric": "completed_runs", "value": status_counts.get("complete", 0), "interpretation": "MAGs with COMPLETE sentinel"},
        {"metric": "failed_runs", "value": status_counts.get("failed", 0), "interpretation": "MAGs with FAILED sentinel"},
        {"metric": "pending_or_partial_runs", "value": len(out) - status_counts.get("complete", 0) - status_counts.get("failed", 0), "interpretation": "not yet complete or failed"},
        {"metric": "runs_with_observed_checkm2_metrics", "value": evidence_counts.get("complete_run_qc_metrics", 0) + evidence_counts.get("partial_run_qc_metrics", 0), "interpretation": "run folders with parseable CheckM2 completeness/contamination, including partial runs"},
        {"metric": "completed_runs_with_checkm2_metrics", "value": evidence_counts.get("complete_run_qc_metrics", 0), "interpretation": "COMPLETE runs with parseable CheckM2 completeness/contamination"},
        {"metric": "runs_with_observed_gunc_metrics", "value": len(gunc_observed), "interpretation": "run folders with parseable GUNC pass/fail evidence, including partial runs"},
        {"metric": "completed_runs_with_gunc_metrics", "value": len(gunc_completed), "interpretation": "COMPLETE runs with parseable GUNC pass/fail evidence"},
        {"metric": "observed_gunc_pass_true", "value": sum(str(row["gunc_pass"]).lower() == "true" for row in gunc_observed), "interpretation": "observed GUNC pass=True rows, including partial runs"},
        {"metric": "observed_gunc_pass_false", "value": sum(str(row["gunc_pass"]).lower() == "false" for row in gunc_observed), "interpretation": "observed GUNC pass=False rows, including partial runs"},
        {"metric": "observed_local_high_quality_like", "value": counts.get("local_high_quality_like", 0), "interpretation": "observed CheckM2 completeness >=90 and contamination <=5, including partial runs"},
        {"metric": "observed_local_medium_quality_like", "value": counts.get("local_medium_quality_like", 0), "interpretation": "observed CheckM2 completeness >=50 and contamination <=10, excluding high-quality count label and including partial runs"},
        {"metric": "observed_local_quality_gate_pass_total", "value": observed_quality_gate_pass_total, "interpretation": "local paper-style quality-gate candidates from any observed CheckM2 output"},
        {"metric": "completed_local_quality_gate_pass_total", "value": completed_quality_gate_pass_total, "interpretation": "local paper-style quality-gate candidates from COMPLETE runs only"},
        {"metric": "paper_reported_medium_high_quality_mags", "value": 966, "interpretation": "published comparison target"},
    ]
    write_tsv(outdir / "msm_china_2025_qc_reconciliation_checkm2_summary.tsv", summary)

    print(f"wrote\t{outdir / 'msm_china_2025_qc_reconciliation_with_checkm2.tsv'}")
    print(f"manifest_rows\t{len(out)}")
    print(f"completed_runs\t{status_counts.get('complete', 0)}")
    print(f"observed_local_quality_gate_pass_total\t{observed_quality_gate_pass_total}")
    print(f"completed_local_quality_gate_pass_total\t{completed_quality_gate_pass_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
