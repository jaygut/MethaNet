#!/usr/bin/env python3
"""Build a lightweight MBAG smoke report from current MethaNet artifacts.

The report is read-only with respect to production per-MAG folders. It writes a
new derived report directory containing provisional MBAG scores, candidate
cards, validation gates, and a first multipanel figure skeleton.
"""

from __future__ import annotations

import argparse
import html
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from methanet.mbag import (
    MBAGPaths,
    build_functional_dataset,
    build_knn_graph,
    discover_completed_runs,
    load_crosswalk,
    load_embedding_artifacts,
    provisional_bridge_scores,
    sinkhorn_transport,
    source_leakage_audit,
)


EXPECTED_EMBEDDED_TOTAL = 662
EXPECTED_MAG_BIN_TOTAL = 625
EXPECTED_ASSEMBLY_CONTEXT_TOTAL = 37


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cohort-run-id", default="fgx_662_apollo3_20260612")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--transport-top-k", type=int, default=5)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def _vectorize_counters(
    counters: dict[str, dict[str, Any]],
    ids: list[str],
    view: str,
) -> np.ndarray:
    rows = []
    for proteome_id in ids:
        counter = counters.get(proteome_id, {}).get(view, {})
        rows.append(dict(counter))
    if not rows:
        return np.zeros((0, 0))
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(rows)
    return np.asarray(matrix, dtype=float)


def _normalize_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    max_value = values.max()
    if not np.isfinite(max_value) or max_value <= 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return values / max_value


def _build_node_table(
    crosswalk: pd.DataFrame,
    embeddings: Any,
    run_status: pd.DataFrame,
    profiles: pd.DataFrame,
    graph_metrics: pd.DataFrame,
    ot_metrics: pd.DataFrame,
    leakage_penalty: float,
) -> pd.DataFrame:
    node = crosswalk.copy()
    projection = embeddings.projection.copy()
    node = node.merge(projection, on="proteome_id", how="left", suffixes=("", "_projection"))
    if not profiles.empty:
        selected = profiles[["proteome_id"]].copy()
        selected["functional_status"] = "complete"
        node = node.merge(selected, on="proteome_id", how="left")
        node = node.merge(profiles, on="proteome_id", how="left", suffixes=("", "_functional"))
    else:
        node["functional_status"] = np.nan
    node = node.merge(graph_metrics, on="proteome_id", how="left")
    node = node.merge(ot_metrics, on="proteome_id", how="left")

    node["functional_status"] = node["functional_status"].fillna("pending_or_not_run")
    for column in [
        "methane_feature_count",
        "sulfur_feature_count",
        "substrate_feature_count",
        "broad_feature_count",
        "reliability_weight",
        "coverage_required",
        "cross_domain_neighbor_fraction",
        "cross_domain_weight_fraction",
        "ot_best_coupling",
        "mixing_coeff",
    ]:
        if column not in node.columns:
            node[column] = 0.0
        node[column] = pd.to_numeric(node[column], errors="coerce").fillna(0.0)

    direct_total = (
        node["methane_feature_count"]
        + node["sulfur_feature_count"]
        + node["substrate_feature_count"]
    )
    node["functional_concordance"] = _normalize_series(np.log1p(direct_total))
    node["mechanism_support"] = _normalize_series(
        np.log1p(
            node["methane_feature_count"]
            + 0.5 * node["sulfur_feature_count"]
            + 0.25 * node["substrate_feature_count"]
        )
    )
    node["candidate_specificity"] = _normalize_series(
        node["mixing_coeff"].fillna(0.0) * np.log1p(1.0 + direct_total)
    )
    node["qc_penalty"] = np.where(
        node["functional_status"].eq("complete"),
        1.0 - node["reliability_weight"].clip(0.0, 1.0),
        0.5,
    )
    node["annotation_missingness"] = np.where(
        node["functional_status"].eq("complete"),
        1.0 - node["coverage_required"].clip(0.0, 1.0),
        1.0,
    )
    node["source_leakage_penalty"] = leakage_penalty
    return provisional_bridge_scores(node)


def _evidence_tier(row: pd.Series) -> str:
    unit = str(row.get("analysis_unit_type") or "").strip()
    if unit and unit != "nan" and unit != "mag_bin":
        return "blocked_noncomparable_unit"
    if row.get("functional_status") != "complete":
        return "hypothesis_only_pending_function"
    if float(row.get("reliability_weight") or 0.0) < 0.2:
        return "artifact_risk_or_low_reliability"
    methane = float(row.get("methane_feature_count") or 0.0)
    context = float(row.get("sulfur_feature_count") or 0.0) + float(row.get("substrate_feature_count") or 0.0)
    if methane > 0 and context > 0:
        return "moderate_evidence_smoke"
    if methane > 0 or context > 0:
        return "needs_review_partial_evidence"
    return "unclear_function"


def _allowed_claim(row: pd.Series) -> str:
    tier = row["evidence_tier"]
    if tier == "blocked_noncomparable_unit":
        return "Assembly-context evidence only; excluded from MAG-level MBAG and not a MAG bridge support claim."
    if tier == "hypothesis_only_pending_function":
        return "Latent bridge hypothesis; functional evidence is pending."
    if tier == "moderate_evidence_smoke":
        return "Preliminary MAG-level bridge candidate with direct functional support; not MRV risk."
    if tier == "artifact_risk_or_low_reliability":
        return "Candidate requires QC/coverage review before biological interpretation."
    return "Preliminary MAG-level evidence requires review before stronger claims."


def _build_candidate_cards(node: pd.DataFrame, bridge_top: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top = bridge_top.head(top_n).copy()
    top["candidate_set"] = "top_latent_bridge"
    candidates = top[["proteome_id", "rank", "candidate_set"]].merge(node, on="proteome_id", how="left")

    complete_pool = node[node["functional_status"].eq("complete")].copy()
    if not complete_pool.empty:
        controls = complete_pool.sort_values(["mixing_coeff", "mbag_score_provisional"], ascending=[True, True]).head(1)
        control_cards = controls.copy()
        control_cards["rank"] = np.nan
        control_cards["candidate_set"] = "negative_control_low_bridge"
        candidates = pd.concat([candidates, control_cards], ignore_index=True, sort=False)

    cards = candidates.copy()
    cards["evidence_tier"] = cards.apply(_evidence_tier, axis=1)
    cards["blocking_caveats"] = cards.apply(
        lambda row: "; ".join(
            caveat
            for caveat in [
                "functional_run_pending" if row.get("functional_status") != "complete" else "",
                "low_reliability" if float(row.get("reliability_weight") or 0.0) < 0.2 and row.get("functional_status") == "complete" else "",
                "source_ecosystem_confounded",
                "MAG_level_not_sample_flux",
            ]
            if caveat
        ),
        axis=1,
    )
    cards["next_validation_action"] = cards.apply(
        lambda row: "wait_for_completed_functional_run"
        if row.get("functional_status") != "complete"
        else "review_direct_marker_support_and_bootstrap_stability",
        axis=1,
    )
    cards["allowed_claim_wording"] = cards.apply(_allowed_claim, axis=1)
    keep = [
        "candidate_set",
        "rank",
        "proteome_id",
        "mag_id",
        "source",
        "ecosystem",
        "taxonomy_domain",
        "taxonomy_family",
        "taxonomy_genus",
        "functional_status",
        "umap_1",
        "umap_2",
        "mbag_score_provisional",
        "mbag_score_status",
        "mixing_coeff",
        "cross_domain_neighbor_fraction",
        "ot_best_coupling",
        "ot_partner",
        "methane_feature_count",
        "sulfur_feature_count",
        "substrate_feature_count",
        "completeness",
        "contamination",
        "gunc_pass",
        "coverage_required",
        "reliability_weight",
        "evidence_tier",
        "blocking_caveats",
        "next_validation_action",
        "allowed_claim_wording",
    ]
    for column in keep:
        if column not in cards.columns:
            cards[column] = np.nan
    order = {"top_latent_bridge": 0, "negative_control_low_bridge": 1}
    cards["_candidate_order"] = cards["candidate_set"].map(order).fillna(9)
    return cards[keep + ["_candidate_order"]].sort_values(
        ["_candidate_order", "rank"],
        na_position="last",
    ).drop(columns=["_candidate_order"])


def _validation_tables(
    crosswalk: pd.DataFrame,
    node: pd.DataFrame,
    cards: pd.DataFrame,
    run_status: pd.DataFrame,
    leakage: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit = node.get("analysis_unit_type", pd.Series(index=node.index, dtype=object)).fillna("")
    mag_bin_mask = unit.eq("mag_bin")
    assembly_mask = unit.eq("assembly_context")
    mag_bin_total = int(mag_bin_mask.sum()) if mag_bin_mask.any() else len(crosswalk)
    assembly_total = int(assembly_mask.sum())
    complete_count = int(node.loc[mag_bin_mask, "functional_status"].eq("complete").sum()) if mag_bin_mask.any() else int(node["functional_status"].eq("complete").sum())
    quarantined_count = int(node["functional_status"].eq("assembly_context_quarantined").sum())
    noncomparable_complete = int(node.loc[assembly_mask, "functional_status"].eq("complete").sum()) if assembly_mask.any() else 0
    top_complete = int(cards.query("candidate_set == 'top_latent_bridge'")["functional_status"].eq("complete").sum())
    gates = [
        ("cohort_denominator_662", "pass" if len(crosswalk) == EXPECTED_EMBEDDED_TOTAL else "fail", f"{len(crosswalk)} rows"),
        ("duplicate_proteome_id", "pass" if not crosswalk["proteome_id"].duplicated().any() else "fail", "canonical proteome_id uniqueness"),
        ("local_mag_proteome_matching", "pass" if crosswalk.get("match_status", pd.Series()).eq("matched").sum() == EXPECTED_EMBEDDED_TOTAL else "warn", "uses local crosswalk audit manifest"),
        ("mag_bin_scope_denominator", "pass" if mag_bin_total == EXPECTED_MAG_BIN_TOTAL else "warn", f"{mag_bin_total} comparable MAG/bin rows; current recovery contract expects {EXPECTED_MAG_BIN_TOTAL}"),
        ("assembly_context_quarantine", "pass" if noncomparable_complete == 0 else "fail", f"{quarantined_count}/{assembly_total or EXPECTED_ASSEMBLY_CONTEXT_TOTAL} completed assembly-context outputs quarantined from MAG-level MBAG"),
        ("completed_functional_snapshot", "warn" if complete_count < mag_bin_total else "pass", f"{complete_count}/{mag_bin_total} comparable MAG/bin complete; active production may continue"),
        ("top_bridge_functional_completion", "warn" if top_complete < len(cards.query("candidate_set == 'top_latent_bridge'")) else "pass", f"{top_complete} top latent bridge candidates have completed functional evidence"),
        ("source_leakage_audit", "warn" if leakage.status.startswith("warn") else "pass", leakage.message),
        ("sample_level_mrv_claim", "blocked", "requires sample mapping, abundance/read coverage, environmental covariates, and flux validation"),
    ]
    gate_df = pd.DataFrame(gates, columns=["gate", "status", "detail"])
    gaps = [
        {
            "gap": "top_latent_bridge_functional_completion",
            "affected_claim": "mechanism-supported bridge cards",
            "current_state": f"{top_complete} completed top latent candidates",
            "required_upgrade": "complete functional runs for all top bridge candidates",
            "priority": "high",
        },
        {
            "gap": "source_ecosystem_confounding",
            "affected_claim": "source-independent rumen-to-wetland transfer",
            "current_state": "rumen and wetland/MUCC are source-aliased",
            "required_upgrade": "additional independent sources per ecosystem plus leave-source-out validation",
            "priority": "high",
        },
        {
            "gap": "sample_rollup_missing",
            "affected_claim": "blue-carbon sample/metagenome methane risk",
            "current_state": "MAG-level evidence only",
            "required_upgrade": "sample IDs, abundance/read coverage, environmental covariates, flux validation",
            "priority": "high",
        },
    ]
    if run_status.empty or complete_count == 0:
        gaps.append(
            {
                "gap": "functional_outputs_unavailable",
                "affected_claim": "candidate functional interpretation",
                "current_state": "no completed runs discovered",
                "required_upgrade": "completed curated per-MAG evidence bundles",
                "priority": "blocker",
            }
        )
    return gate_df, pd.DataFrame(gaps)


def _write_figure(out_path: Path, node: pd.DataFrame, cards: pd.DataFrame, gates: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FCFCFD")

    status_counts = node.groupby(["ecosystem", "functional_status"]).size().unstack(fill_value=0)
    status_counts.plot(kind="bar", stacked=True, ax=axes[0, 0], color=["#2E7D61", "#C5C9D3", "#E0A03A"])
    axes[0, 0].set_title("Cohort ledger: embedded 662 with live functional status")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("MAG/proteome count")

    colors = node["ecosystem"].map({"rumen": "#267A9E", "wetland": "#317A43"}).fillna("#808890")
    axes[0, 1].scatter(node["umap_1"], node["umap_2"], c=colors, s=16, alpha=0.55, linewidths=0)
    top = cards[cards["candidate_set"].eq("top_latent_bridge")]
    axes[0, 1].scatter(top["umap_1"], top["umap_2"], facecolors="none", edgecolors="#E0A03A", s=90, linewidths=1.5)
    axes[0, 1].set_title("ESM2 geometry: bridge hypotheses, not proof")
    axes[0, 1].set_xlabel("UMAP 1")
    axes[0, 1].set_ylabel("UMAP 2")

    matrix_cols = ["methane_feature_count", "sulfur_feature_count", "substrate_feature_count", "reliability_weight"]
    mat = cards[matrix_cols].fillna(0.0).to_numpy(dtype=float)
    if mat.size:
        mat = mat / np.maximum(mat.max(axis=0, keepdims=True), 1.0)
    axes[1, 0].imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_yticks(range(len(cards)))
    axes[1, 0].set_yticklabels(cards["proteome_id"].astype(str).str.slice(0, 34), fontsize=7)
    axes[1, 0].set_xticks(range(len(matrix_cols)))
    axes[1, 0].set_xticklabels(["methane", "sulfur", "substrate", "reliability"], rotation=30, ha="right")
    axes[1, 0].set_title("Candidate evidence matrix with missingness visible")

    gate_counts = gates["status"].value_counts().reindex(["pass", "warn", "blocked", "fail"], fill_value=0)
    axes[1, 1].bar(gate_counts.index, gate_counts.values, color=["#2E7D61", "#E0A03A", "#9AA0AE", "#B94A48"])
    axes[1, 1].set_title("Validation gates: provisional by design")
    axes[1, 1].set_ylabel("Gate count")
    for ax in axes.ravel():
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7, alpha=0.7)
    fig.suptitle("MethaNet MBAG Smoke Report Skeleton", fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_reports(
    out_dir: Path,
    summary: dict[str, Any],
    cards: pd.DataFrame,
    gates: pd.DataFrame,
    gaps: pd.DataFrame,
    figure_path: Path,
) -> None:
    md = out_dir / "mbag_smoke_report.md"
    html_path = out_dir / "mbag_smoke_report.html"
    top_rows = cards.head(12)
    gate_lines = "\n".join(f"- `{row.gate}`: **{row.status}** - {row.detail}" for row in gates.itertuples())
    gap_lines = "\n".join(f"- **{row.gap}** ({row.priority}): {row.required_upgrade}" for row in gaps.itertuples())
    card_lines = "\n".join(
        f"- `{row.proteome_id}`: {row.evidence_tier}; {row.allowed_claim_wording}"
        for row in top_rows.itertuples()
    )
    md.write_text(
        "\n".join(
            [
                "# MethaNet MBAG Smoke Report",
                "",
                f"Generated: {summary['generated_at']}",
                "",
                "This is a lightweight, MAG-level molecular-attestation smoke report. It is not a sample-level methane-risk report and it does not rank final MRV risk.",
                "",
                "## Snapshot",
                "",
                f"- Embedded cohort rows: {summary['embedded_cohort_rows']}",
                f"- Comparable MAG/bin denominator: {summary.get('mag_bin_denominator', EXPECTED_MAG_BIN_TOTAL)}",
                f"- Completed comparable MAG/bin outputs: {summary['completed_functional_mags']}/{summary.get('mag_bin_denominator', EXPECTED_MAG_BIN_TOTAL)}",
                f"- Quarantined assembly-context outputs: {summary.get('quarantined_functional_outputs', 0)}/{summary.get('assembly_context_denominator', EXPECTED_ASSEMBLY_CONTEXT_TOTAL)}",
                f"- Failed functional MAGs: {summary['failed_functional_mags']}",
                f"- Top latent bridge candidates complete: {summary['top_bridge_complete']}/{summary['top_bridge_count']}",
                f"- Source leakage audit: {summary['source_leakage_status']} ({summary['source_leakage_balanced_accuracy']})",
                "",
                "## Candidate Cards",
                "",
                card_lines,
                "",
                "## Validation Gates",
                "",
                gate_lines,
                "",
                "## Validation Gaps",
                "",
                gap_lines,
                "",
                "## Figure Skeleton",
                "",
                f"![MBAG smoke multipanel]({figure_path.name})",
                "",
                "## Claim Boundary",
                "",
                "Allowed: preliminary MAG-level bridge-candidate review and monitoring-priority design. Not allowed: source-independent MRV transfer, carbon-credit approval, measured methane flux, or calibrated risk tiers.",
                "",
            ]
        )
    )
    escaped = html.escape(md.read_text())
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MethaNet MBAG Smoke Report</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #1F2430; background: #FCFCFD; }}
pre {{ white-space: pre-wrap; background: #fff; border: 1px solid #E6E8F0; padding: 18px; }}
img {{ max-width: 100%; border: 1px solid #E6E8F0; background: white; }}
</style></head><body>
<pre>{escaped}</pre>
<img src="{figure_path.name}" alt="MBAG smoke multipanel">
</body></html>
"""
    )


def _w_text(text: Any) -> str:
    return escape("" if text is None else str(text))


def _docx_paragraph(text: str = "", style: str | None = None, bold: bool = False) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    bold_xml = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f"<w:p>{style_xml}<w:r>{bold_xml}<w:t>{_w_text(text)}</w:t></w:r></w:p>"


def _docx_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any, header: bool = False) -> str:
        bold_xml = "<w:rPr><w:b/></w:rPr>" if header else ""
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:r>{bold_xml}<w:t>{_w_text(value)}</w:t></w:r></w:p></w:tc>"
        )

    header_row = "<w:tr>" + "".join(cell(head, header=True) for head in headers) + "</w:tr>"
    body = []
    for row in rows:
        body.append("<w:tr>" + "".join(cell(value) for value in row) + "</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9DEE8\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9DEE8\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9DEE8\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9DEE8\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9DEE8\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D9DEE8\"/>"
        "</w:tblBorders></w:tblPr>"
        + header_row
        + "".join(body)
        + "</w:tbl>"
    )


def _docx_image(rel_id: str, name: str, width_inches: float = 6.5, height_inches: float = 4.64) -> str:
    cx = int(width_inches * 914400)
    cy = int(height_inches * 914400)
    return f"""
<w:p><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/>
<wp:docPr id="1" name="{_w_text(name)}"/>
<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic>
<pic:nvPicPr><pic:cNvPr id="0" name="{_w_text(name)}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic>
</a:graphicData></a:graphic>
</wp:inline>
</w:drawing></w:r></w:p>
"""


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _docx_styles() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal">
<w:name w:val="Normal"/><w:qFormat/><w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Title">
<w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:rPr><w:b/><w:sz w:val="34"/><w:szCs w:val="34"/><w:color w:val="124734"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading1">
<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/><w:color w:val="124734"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading2">
<w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/><w:color w:val="1F4E5F"/></w:rPr>
</w:style>
</w:styles>"""


def _write_docx_report(
    out_dir: Path,
    summary: dict[str, Any],
    cards: pd.DataFrame,
    gates: pd.DataFrame,
    gaps: pd.DataFrame,
    figure_path: Path,
) -> Path:
    """Write a self-contained Word report without requiring python-docx.

    Apollo environments used for MBAG have pandas/pyarrow/matplotlib but not
    python-docx. This minimal OpenXML writer keeps the smoke report portable
    without adding runtime dependencies or installing tools on the cluster.
    """

    docx_path = out_dir / "mbag_smoke_full_report.docx"
    title = "MethaNet MBAG Smoke Report"
    top_complete = f"{summary['top_bridge_complete']}/{summary['top_bridge_count']}"
    leakage = _format_float(summary.get("source_leakage_balanced_accuracy"))
    mag_bin_denominator = summary.get("mag_bin_denominator", EXPECTED_MAG_BIN_TOTAL)
    quarantined = summary.get("quarantined_functional_outputs", 0)
    assembly_total = summary.get("assembly_context_denominator", EXPECTED_ASSEMBLY_CONTEXT_TOTAL)
    body: list[str] = [
        _docx_paragraph(title, "Title"),
        _docx_paragraph(f"Generated: {summary['generated_at']}"),
        _docx_paragraph(
            "Fully fledged smoke-test report for the MethaNet Bridge Attestation Graph. "
            "This is a MAG-level molecular screening artifact, not a sample-level methane-risk score."
        ),
        _docx_paragraph("Executive Summary", "Heading1"),
        _docx_paragraph(
            f"- The smoke run used the 662-proteome embedded cohort and found "
            f"{summary['completed_functional_mags']}/{mag_bin_denominator} completed comparable MAG/bin functional outputs in the current local snapshot."
        ),
        _docx_paragraph(
            f"- {quarantined}/{assembly_total} completed assembly-context outputs are preserved but quarantined from MAG-level MBAG; they can support source/community context, not candidate-level MAG evidence."
        ),
        _docx_paragraph(
            f"- Top latent bridge-candidate functional completion is {top_complete}; incomplete top candidates must remain hypothesis-only."
        ),
        _docx_paragraph(
            f"- Source leakage remains a major caution: the current leakage audit reports balanced accuracy {leakage}, so source-independent transfer claims are not allowed."
        ),
        _docx_paragraph(
            "- The report is actionable for candidate review, validation planning, and monitoring-priority design; it is blocked for final MRV risk scoring until sample mapping, abundance, environmental covariates, and flux/process validation exist."
        ),
        _docx_paragraph("Audit Of The First MBAG Artifact", "Heading1"),
        _docx_paragraph(
            "The first MBAG infographic is conceptually aligned with the intended product architecture: ESM2 proteome geometry, functional annotation, QC/taxonomy, provenance, methane/sulfur/substrate signals, multi-view graph integration, source-leakage auditing, QC/coverage penalties, candidate cards, monitoring priorities, validation gaps, and MRV feature tables."
        ),
        _docx_paragraph(
            "The critical caveat is that the infographic is a framework blueprint, not a claim that all lanes are fully populated now. Current production has not run eggNOG, most top rumen bridge MAGs are still pending functional evidence, and source/ecosystem labels are highly recoverable. This DOCX keeps the same architecture but locks the present smoke-test claims to MAG-level molecular screening."
        ),
        _docx_paragraph("Current Evidence Snapshot", "Heading1"),
        _docx_table(
            ["Measure", "Value", "Interpretation"],
            [
                ["Embedded cohort rows", summary["embedded_cohort_rows"], "Authoritative ESM2 POC denominator"],
                ["Comparable MAG/bin denominator", mag_bin_denominator, "MAG-level MBAG denominator after assembly-context quarantine"],
                ["Completed comparable MAG/bin outputs", summary["completed_functional_mags"], "Usable direct functional evidence in this snapshot"],
                ["Quarantined assembly-context outputs", f"{quarantined}/{assembly_total}", "Preserved as context, excluded from MAG-level bridge evidence"],
                ["Failed functional MAGs", summary["failed_functional_mags"], "Operational failures retained as status"],
                ["Top latent bridge candidates complete", top_complete, "Main blocker for mechanism-supported top bridge cards"],
                ["Source leakage audit", f"{summary['source_leakage_status']} ({leakage})", "Requires strong caveats and source-aware validation"],
            ],
        ),
        _docx_paragraph("Allowed And Blocked Claims", "Heading1"),
        _docx_table(
            ["Claim class", "Status", "Allowed wording"],
            [
                ["MAG-level molecular screening", "Allowed", "Completed MAGs can be reviewed for QC-aware methane/sulfur/substrate evidence."],
                ["Bridge candidate prioritization", "Allowed with caveats", "Bridge candidates are hypotheses requiring direct functional support."],
                ["Sample methane risk", "Blocked", "Requires sample mapping, abundance/read coverage, environmental covariates, and validation."],
                ["Final A-E risk tiers", "Blocked", "A-E tiers remain target product vocabulary until calibrated sample/project evidence exists."],
                ["Carbon-credit approval", "Not allowed", "MethaNet can support screening and monitoring design, not credit issuance."],
            ],
        ),
    ]

    candidate_rows: list[list[Any]] = []
    keep_cols = [
        "candidate_set",
        "rank",
        "proteome_id",
        "functional_status",
        "evidence_tier",
        "next_validation_action",
    ]
    for _, row in cards.head(14).iterrows():
        candidate_rows.append([row.get(col, "") for col in keep_cols])
    body.extend(
        [
            _docx_paragraph("Candidate Evidence Cards", "Heading1"),
            _docx_paragraph(
                "Candidate status is intentionally conservative. Pending top bridge candidates are shown rather than hidden, because missingness is part of the decision signal."
            ),
            _docx_table(
                ["Set", "Rank", "Proteome ID", "Functional status", "Evidence tier", "Next action"],
                candidate_rows,
            ),
            _docx_paragraph("Validation Gates", "Heading1"),
            _docx_table(
                ["Gate", "Status", "Detail"],
                [[row.gate, row.status, row.detail] for row in gates.itertuples()],
            ),
            _docx_paragraph("Validation Gap Register", "Heading1"),
            _docx_table(
                ["Gap", "Priority", "Required upgrade"],
                [[row.gap, row.priority, row.required_upgrade] for row in gaps.itertuples()],
            ),
            _docx_paragraph("Multipanel Smoke Figure", "Heading1"),
            _docx_paragraph(
                "The figure below is a smoke-test skeleton, not a publication-ready final panel. It is useful for checking the intended report architecture: cohort ledger, ESM2 geometry, candidate evidence matrix, and validation gates."
            ),
        ]
    )
    image_rel = ""
    image_override = ""
    image_file_entry: tuple[str, Path] | None = None
    if figure_path.exists():
        image_rel = (
            '<Relationship Id="rIdImage1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/mbag_multipanel_skeleton.png"/>'
        )
        image_override = '<Default Extension="png" ContentType="image/png"/>'
        image_file_entry = ("word/media/mbag_multipanel_skeleton.png", figure_path)
        body.append(_docx_image("rIdImage1", figure_path.name))

    body.extend(
        [
            _docx_paragraph("Recommended Next Actions", "Heading1"),
            _docx_paragraph("- Re-run the full MBAG report after more rumen top bridge candidates complete functional processing."),
            _docx_paragraph("- Regenerate the cohort warehouse before treating counts as analysis-final."),
            _docx_paragraph("- Keep source-leakage, missingness, and sample-level MRV blocks visible in every partner-facing report."),
            _docx_paragraph("- Build the next layer as sample risk readiness, not final MRV scoring."),
            _docx_paragraph("Generated Evidence Files", "Heading1"),
            _docx_table(
                ["Artifact", "Path"],
                [
                    ["Candidate cards", str(out_dir / "bridge_attestation_cards_smoke.tsv")],
                    ["Validation gates", str(out_dir / "validation_gates.tsv")],
                    ["Validation gaps", str(out_dir / "validation_gap_register.tsv")],
                    ["Node scores", str(out_dir / "mbag_node_scores.parquet")],
                    ["ESM2 kNN edges", str(out_dir / "mbag_esm2_knn_edges.parquet")],
                    ["Transport couplings", str(out_dir / "mbag_transport_couplings.parquet")],
                    ["Markdown report", str(out_dir / "mbag_smoke_report.md")],
                    ["HTML report", str(out_dir / "mbag_smoke_report.html")],
                ],
            ),
        ]
    )

    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
xmlns:w10="urn:schemas-microsoft-com:office:word"
xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
mc:Ignorable="w14 wp14">
<w:body>
{''.join(body)}
<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="900" w:bottom="1080" w:left="900" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>
</w:body></w:document>"""

    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
{image_override}
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    document_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
{image_rel}
</Relationships>"""
    settings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:zoom w:percent="100"/></w:settings>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>{_w_text(title)}</dc:title><dc:creator>MethaNet MBAG smoke workflow</dc:creator><cp:lastModifiedBy>MethaNet MBAG smoke workflow</cp:lastModifiedBy>
<dcterms:created xsi:type="dcterms:W3CDTF">{datetime.now().isoformat(timespec="seconds")}</dcterms:created>
<dcterms:modified xsi:type="dcterms:W3CDTF">{datetime.now().isoformat(timespec="seconds")}</dcterms:modified>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>MethaNet</Application></Properties>"""

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", package_rels)
        docx.writestr("word/document.xml", document)
        docx.writestr("word/_rels/document.xml.rels", document_rels)
        docx.writestr("word/styles.xml", _docx_styles())
        docx.writestr("word/settings.xml", settings)
        docx.writestr("docProps/core.xml", core)
        docx.writestr("docProps/app.xml", app)
        if image_file_entry:
            arcname, source = image_file_entry
            docx.write(source, arcname)
    return docx_path


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    paths = MBAGPaths(repo_root=repo_root, cohort_run_id=args.cohort_run_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (
        repo_root
        / "results/functional_metagenomics"
        / args.cohort_run_id
        / "reports"
        / f"mbag_smoke_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    crosswalk = load_crosswalk(paths.crosswalk, paths.functional_manifest)
    embeddings = load_embedding_artifacts(paths.embedding_artifact_dir)
    run_status, selected_runs = discover_completed_runs(paths.per_mag_dir)
    mag_level_ids = set(
        crosswalk.loc[
            crosswalk.get("analysis_unit_type", pd.Series(index=crosswalk.index, dtype=object)).eq("mag_bin")
            & crosswalk.get("mbag_mag_level_include", pd.Series(index=crosswalk.index, dtype=object)).astype(str).str.lower().isin({"true", "1", "yes"}),
            "proteome_id",
        ]
    )
    selected_runs_mag_level = {proteome_id: run_dir for proteome_id, run_dir in selected_runs.items() if proteome_id in mag_level_ids}
    quarantined_completed = sorted(set(selected_runs) - set(selected_runs_mag_level))
    functional = build_functional_dataset(selected_runs_mag_level)

    graph = build_knn_graph(
        embeddings.metadata["proteome_id"],
        embeddings.embeddings,
        embeddings.metadata["ecosystem"],
        k=args.knn_k,
        metric="cosine",
    )
    source_mask = embeddings.metadata["ecosystem"].eq("rumen").to_numpy()
    target_mask = embeddings.metadata["ecosystem"].eq("wetland").to_numpy()
    transport = sinkhorn_transport(
        embeddings.metadata.loc[source_mask, "proteome_id"],
        embeddings.metadata.loc[target_mask, "proteome_id"],
        embeddings.embeddings[source_mask],
        embeddings.embeddings[target_mask],
        metric="cosine",
        top_per_source=args.transport_top_k,
    )
    leakage = source_leakage_audit(
        embeddings.embeddings,
        embeddings.metadata["ecosystem"],
        label_name="ecosystem_source_aliased",
    )
    leakage_penalty = 0.0
    if leakage.balanced_accuracy is not None:
        leakage_penalty = max(float(leakage.balanced_accuracy) - 0.5, 0.0) * 2.0

    node = _build_node_table(
        crosswalk,
        embeddings,
        run_status,
        functional.profiles,
        graph.node_metrics,
        transport.node_metrics,
        leakage_penalty,
    )
    if "analysis_unit_type" in node.columns and quarantined_completed:
        quarantine_mask = node["proteome_id"].isin(quarantined_completed) & node["analysis_unit_type"].ne("mag_bin")
        node.loc[quarantine_mask, "functional_status"] = "assembly_context_quarantined"
        node.loc[quarantine_mask, "mbag_score_status"] = "blocked_noncomparable_unit"
    cards = _build_candidate_cards(node, embeddings.bridge_top, args.top_n)
    gates, gaps = _validation_tables(crosswalk, node, cards, run_status, leakage)

    # Optional completed-only functional graph for future report expansion.
    completed_ids = functional.profiles["proteome_id"].tolist() if not functional.profiles.empty else []
    if len(completed_ids) > 2:
        broad_matrix = _vectorize_counters(functional.feature_counters, completed_ids, "broad")
        if broad_matrix.shape[1] > 0:
            fgraph = build_knn_graph(
                completed_ids,
                broad_matrix,
                functional.profiles.set_index("proteome_id").loc[completed_ids, "taxonomy_domain"].fillna("unknown"),
                k=min(8, len(completed_ids) - 1),
                metric="cosine",
                reliability=functional.profiles.set_index("proteome_id").loc[completed_ids, "reliability_weight"].fillna(0.0),
            )
            fgraph.edges.to_parquet(out_dir / "mbag_functional_broad_graph_edges.parquet", index=False)

    graph.edges.to_parquet(out_dir / "mbag_esm2_knn_edges.parquet", index=False)
    transport.couplings.to_parquet(out_dir / "mbag_transport_couplings.parquet", index=False)
    node.to_parquet(out_dir / "mbag_node_scores.parquet", index=False)
    cards.to_csv(out_dir / "bridge_attestation_cards_smoke.tsv", sep="\t", index=False)
    gates.to_csv(out_dir / "validation_gates.tsv", sep="\t", index=False)
    gaps.to_csv(out_dir / "validation_gap_register.tsv", sep="\t", index=False)

    figure_path = out_dir / "mbag_multipanel_skeleton.png"
    _write_figure(figure_path, node, cards, gates)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cohort_run_id": args.cohort_run_id,
        "embedded_cohort_rows": int(len(crosswalk)),
        "mag_bin_denominator": int(len(mag_level_ids)),
        "assembly_context_denominator": int(
            crosswalk.get("analysis_unit_type", pd.Series(index=crosswalk.index, dtype=object)).fillna("").eq("assembly_context").sum()
        ),
        "completed_functional_mags": int(node["functional_status"].eq("complete").sum()),
        "quarantined_functional_outputs": int(node["functional_status"].eq("assembly_context_quarantined").sum()),
        "failed_functional_mags": int(run_status["run_status"].eq("failed").sum()) if "run_status" in run_status else 0,
        "top_bridge_count": int(args.top_n),
        "top_bridge_complete": int(cards.query("candidate_set == 'top_latent_bridge'")["functional_status"].eq("complete").sum()),
        "source_leakage_status": leakage.status,
        "source_leakage_balanced_accuracy": leakage.balanced_accuracy,
        "source_leakage_roc_auc": leakage.roc_auc,
        "transport_cost_summary": transport.cost_summary,
        "outputs": {
            "cards": str(out_dir / "bridge_attestation_cards_smoke.tsv"),
            "node_scores": str(out_dir / "mbag_node_scores.parquet"),
            "knn_edges": str(out_dir / "mbag_esm2_knn_edges.parquet"),
            "transport": str(out_dir / "mbag_transport_couplings.parquet"),
            "validation_gates": str(out_dir / "validation_gates.tsv"),
            "validation_gaps": str(out_dir / "validation_gap_register.tsv"),
            "figure": str(figure_path),
        },
    }
    (out_dir / "mbag_smoke_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    _write_reports(out_dir, summary, cards, gates, gaps, figure_path)
    docx_path = _write_docx_report(out_dir, summary, cards, gates, gaps, figure_path)
    summary["outputs"]["docx_report"] = str(docx_path)
    (out_dir / "mbag_smoke_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps({"output_dir": str(out_dir), **summary}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
