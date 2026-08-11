#!/usr/bin/env python3
"""Stage the public KBase MUCC v1 catalog without inferring unpublished QC.

The MUCC v1 KBase narrative is browser-login gated, but the workspace is
publicly readable through KBase's Workspace JSON-RPC API.  This script keeps a
checksum-addressable copy of that public response and reconciles its genome
object identities with the checksum-validated Zenodo MAGs.zip roster.

KBase membership is an identity and supplementary-taxonomy layer only.  It is
not a per-MAG high-/medium-quality designation, a source of CheckM QC, or an
ecological sample/flux association.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
KBASE_WORKSPACE_ID = 147022
KBASE_WORKSPACE_NAME = "freiburgermsu:narrative_1684385563186"
KBASE_GENOMESET_NAME = "OWC"
KBASE_API_URL = "https://kbase.us/services/ws"
PUBLISHED_HQMQ_HEADLINE = 2502
CLAIM_BOUNDARY = (
    "Public KBase roster and taxonomy reconciliation only. KBase membership, "
    "metadata, or taxonomy does not establish a published MUCC HQ/MQ quality "
    "tier, measured methane flux, ecological association, final MRV score/A-E "
    "tier, carbon-crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--workspace-id", type=int, default=KBASE_WORKSPACE_ID)
    parser.add_argument("--api-url", default=KBASE_API_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "source_audit/kbase_public_workspace_147022",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached Workspace API JSON rather than making a network request.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty reconciliation table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rpc(api_url: str, method: str, params: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps(
        {"version": "1.1", "method": method, "params": params, "id": "methanet-mucc"}
    ).encode()
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "MethaNet MUCC public KBase catalog reconciliation",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode())
    if result.get("error"):
        raise ValueError(f"KBase Workspace {method} failed: {result['error']}")
    return result


def load_or_retrieve(
    *,
    path: Path,
    api_url: str,
    method: str,
    params: list[dict[str, Any]],
    offline: bool,
) -> dict[str, Any]:
    if offline:
        if not path.is_file():
            raise FileNotFoundError(f"offline mode requires cached KBase response: {path}")
        return json.loads(path.read_text())
    value = rpc(api_url, method, params)
    write_json(path, value)
    return value


def mag_id_from_kbase_name(name: str) -> str:
    """Map the KBase genome-object convention to the source MAG identifier."""
    return name.removesuffix(".fa_genome")


def object_ref(info: list[Any]) -> str:
    return f"{info[6]}/{info[0]}/{info[4]}"


def object_id_from_ref(ref: str) -> int:
    fields = ref.split("/")
    if len(fields) not in {2, 3} or not fields[1].isdigit():
        raise ValueError(f"unexpected KBase object reference: {ref}")
    return int(fields[1])


def normalize_taxonomy(value: str) -> str:
    return ";".join(part.strip() for part in str(value or "").split(";") if part.strip())


def taxonomy_status(source_taxonomy: str, kbase_taxonomy: str) -> str:
    source = normalize_taxonomy(source_taxonomy)
    kbase = normalize_taxonomy(kbase_taxonomy)
    if source and kbase:
        return "source_and_KBase_Gtdb_taxonomy_exact" if source == kbase else "source_and_KBase_Gtdb_taxonomy_differ"
    if source:
        return "source_taxonomy_only_KBase_genome_absent"
    if kbase:
        return "KBase_Gtdb_taxonomy_supplemental_only"
    return "taxonomy_unavailable_in_source_and_public_KBase_catalog"


def reconciliation_rows(
    catalog: list[dict[str, str]],
    source_annotations: dict[str, dict[str, str]],
    genome_infos: list[list[Any]],
    genomeset_elements: dict[str, dict[str, str]],
    workspace_id: int,
    workspace_name: str,
    genomeset_ref: str,
) -> list[dict[str, str]]:
    """Return one evidence-preserving reconciliation row for every Zenodo MAG."""
    genome_by_mag: dict[str, list[Any]] = {}
    for info in genome_infos:
        mag_id = mag_id_from_kbase_name(str(info[1]))
        if mag_id in genome_by_mag:
            raise ValueError(f"duplicate KBase genome object name after normalization: {mag_id}")
        genome_by_mag[mag_id] = info
    set_ref_by_mag = {
        mag_id_from_kbase_name(name): value["ref"] for name, value in genomeset_elements.items()
    }
    if set(genome_by_mag) != set(set_ref_by_mag):
        raise ValueError("KBase genome objects and GenomeSet members have different MAG-name sets")
    genome_object_ids = {mag_id: int(info[0]) for mag_id, info in genome_by_mag.items()}
    set_object_ids = {mag_id: object_id_from_ref(ref) for mag_id, ref in set_ref_by_mag.items()}
    if genome_object_ids != set_object_ids:
        raise ValueError("KBase GenomeSet members do not resolve to public genome object IDs")

    rows: list[dict[str, str]] = []
    for catalog_row in sorted(catalog, key=lambda row: row["mag_id"]):
        mag_id = catalog_row["mag_id"]
        source = source_annotations.get(mag_id, {})
        source_taxonomy = normalize_taxonomy(source.get("bin_taxonomy", ""))
        info = genome_by_mag.get(mag_id)
        if info is None:
            rows.append(
                {
                    "lane_id": LANE_ID,
                    "mag_id": mag_id,
                    "proteome_id": catalog_row.get("proteome_id", f"mucc_v1__{mag_id}"),
                    "archive_member": catalog_row.get("archive_member", ""),
                    "kbase_workspace_id": str(workspace_id),
                    "kbase_workspace_name": workspace_name,
                    "kbase_genomeset_ref": genomeset_ref,
                    "kbase_genomeset_membership_ref": "",
                    "kbase_genome_latest_ref": "",
                    "kbase_latest_object_version": "",
                    "kbase_latest_object_saved_utc": "",
                    "kbase_gtdb_source_version": "",
                    "kbase_gtdb_lineage": "",
                    "source_bin_taxonomy": source_taxonomy,
                    "taxonomy_reconciliation_status": taxonomy_status(source_taxonomy, ""),
                    "kbase_genome_quality_fields_status": "no_KBase_genome_object_for_this_archive_MAG",
                    "kbase_roster_reconciliation_status": "Zenodo_archive_MAG_absent_from_public_KBase_GenomeSet",
                    "published_hqmq_membership_status": "unresolved_do_not_infer_from_KBase_absence",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            continue
        metadata = info[10] if len(info) > 10 and isinstance(info[10], dict) else {}
        kbase_taxonomy = normalize_taxonomy(str(metadata.get("GTDB_lineage", "")))
        quality_keys = [
            key
            for key in metadata
            if any(token in key.lower() for token in ("completeness", "contamination", "checkm", "quality", "n50"))
        ]
        rows.append(
            {
                "lane_id": LANE_ID,
                "mag_id": mag_id,
                "proteome_id": catalog_row.get("proteome_id", f"mucc_v1__{mag_id}"),
                "archive_member": catalog_row.get("archive_member", ""),
                "kbase_workspace_id": str(workspace_id),
                "kbase_workspace_name": workspace_name,
                "kbase_genomeset_ref": genomeset_ref,
                "kbase_genomeset_membership_ref": set_ref_by_mag[mag_id],
                "kbase_genome_latest_ref": object_ref(info),
                "kbase_latest_object_version": str(info[4]),
                "kbase_latest_object_saved_utc": str(info[3]),
                "kbase_gtdb_source_version": str(metadata.get("GTDB_source_ver", "")),
                "kbase_gtdb_lineage": kbase_taxonomy,
                "source_bin_taxonomy": source_taxonomy,
                "taxonomy_reconciliation_status": taxonomy_status(source_taxonomy, kbase_taxonomy),
                "kbase_genome_quality_fields_status": (
                    "unexpected_quality_like_metadata_present:" + ",".join(sorted(quality_keys))
                    if quality_keys
                    else "no_completeness_contamination_CheckM_quality_or_N50_metadata"
                ),
                "kbase_roster_reconciliation_status": "exact_MAG_id_match_public_KBase_GenomeSet",
                "published_hqmq_membership_status": "unresolved_do_not_infer_from_KBase_membership",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def build_summary(
    *,
    rows: list[dict[str, str]],
    workspace_info: list[Any],
    genomeset_info: list[Any],
    generated_utc: str,
    raw_paths: dict[str, Path],
) -> dict[str, Any]:
    matches = sum(
        row["kbase_roster_reconciliation_status"]
        == "exact_MAG_id_match_public_KBase_GenomeSet"
        for row in rows
    )
    absent = len(rows) - matches
    taxonomy = Counter(row["taxonomy_reconciliation_status"] for row in rows)
    quality_status = Counter(row["kbase_genome_quality_fields_status"] for row in rows)
    member_versions = Counter(
        row["kbase_latest_object_version"] for row in rows if row["kbase_latest_object_version"]
    )
    membership_versions = Counter(
        row["kbase_genomeset_membership_ref"].split("/")[-1]
        for row in rows
        if row["kbase_genomeset_membership_ref"]
    )
    return {
        "generated_utc": generated_utc,
        "lane_id": LANE_ID,
        "kbase_workspace": {
            "id": workspace_info[0],
            "name": workspace_info[1],
            "owner": workspace_info[2],
            "public_read_only_api": True,
        },
        "kbase_genomeset": {
            "name": genomeset_info[1],
            "ref": object_ref(genomeset_info),
            "object_version": genomeset_info[4],
            "member_count": matches,
            "membership_reference_versions": dict(sorted(membership_versions.items())),
            "membership_reference_detail": (
                "GenomeSet membership and current public genome records are reconciled by "
                "workspace/object ID; each reference version is retained separately."
            ),
            "latest_genome_object_versions": dict(sorted(member_versions.items())),
        },
        "published_hqmq_headline": PUBLISHED_HQMQ_HEADLINE,
        "zenodo_checksum_validated_archive_mag_count": len(rows),
        "public_kbase_exact_mag_id_matches": matches,
        "zenodo_archive_mag_absent_from_public_kbase": absent,
        "public_kbase_only_mag_count": 0,
        "published_hqmq_minus_public_kbase_count": PUBLISHED_HQMQ_HEADLINE - matches,
        "quality_scope": {
            "status": "unresolved",
            "detail": "No public KBase completeness, contamination, CheckM, quality-tier, or N50 metadata was found; KBase membership is not used to assign the published HQ/MQ subset.",
            "quality_metadata_status_counts": dict(sorted(quality_status.items())),
        },
        "supplementary_taxonomy": {
            "source_and_KBase_exact": taxonomy["source_and_KBase_Gtdb_taxonomy_exact"],
            "source_and_KBase_differ": taxonomy["source_and_KBase_Gtdb_taxonomy_differ"],
            "KBase_supplemental_only": taxonomy["KBase_Gtdb_taxonomy_supplemental_only"],
            "source_only_KBase_absent": taxonomy["source_taxonomy_only_KBase_genome_absent"],
            "unavailable_both": taxonomy["taxonomy_unavailable_in_source_and_public_KBase_catalog"],
            "detail": "Retain both source and KBase GTDB lineage values; do not silently overwrite differences across source versions.",
        },
        "raw_public_api_responses": {
            key: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in raw_paths.items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    output_dir = resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        "workspace_info": output_dir / "workspace_info.json",
        "list_objects": output_dir / "list_objects.json",
        "genomeset": output_dir / "genomeset_OWC.json",
    }
    workspace_payload = load_or_retrieve(
        path=raw_paths["workspace_info"],
        api_url=args.api_url,
        method="Workspace.get_workspace_info",
        params=[{"id": args.workspace_id}],
        offline=args.offline,
    )
    workspace_info = workspace_payload["result"][0]
    if workspace_info[0] != args.workspace_id:
        raise ValueError(f"KBase workspace ID mismatch: expected {args.workspace_id} got {workspace_info[0]}")
    if args.workspace_id == KBASE_WORKSPACE_ID and workspace_info[1] != KBASE_WORKSPACE_NAME:
        raise ValueError(
            "KBase workspace name mismatch: "
            f"expected {KBASE_WORKSPACE_NAME} got {workspace_info[1]}"
        )
    objects_payload = load_or_retrieve(
        path=raw_paths["list_objects"],
        api_url=args.api_url,
        method="Workspace.list_objects",
        params=[{"ids": [args.workspace_id], "includeMetadata": 1}],
        offline=args.offline,
    )
    objects = objects_payload["result"][0]
    genome_infos = [info for info in objects if str(info[2]).startswith("KBaseGenomes.Genome-")]
    genomesets = [
        info
        for info in objects
        if str(info[2]).startswith("KBaseSearch.GenomeSet-") and info[1] == KBASE_GENOMESET_NAME
    ]
    if len(genomesets) != 1:
        raise ValueError(f"expected exactly one public KBase GenomeSet named {KBASE_GENOMESET_NAME}")
    genomeset_info = genomesets[0]
    genomeset_ref = object_ref(genomeset_info)
    genomeset_payload = load_or_retrieve(
        path=raw_paths["genomeset"],
        api_url=args.api_url,
        method="Workspace.get_objects2",
        params=[{"objects": [{"ref": genomeset_ref}]}],
        offline=args.offline,
    )
    genomeset_data = genomeset_payload["result"][0]["data"][0]["data"]
    genomeset_elements = genomeset_data.get("elements", {})
    if not isinstance(genomeset_elements, dict) or not genomeset_elements:
        raise ValueError("public KBase GenomeSet is missing members")

    catalog_path = run_dir / "manifests/mucc_v1_mag_catalog_full.tsv"
    source_annotations_path = (
        run_dir / "functional_features/feature_mucc_v1_gene_annotation_mag_summary.tsv"
    )
    with catalog_path.open(newline="") as handle:
        catalog = list(csv.DictReader(handle, delimiter="\t"))
    with source_annotations_path.open(newline="") as handle:
        source_annotations = {
            row["source_mag_id"]: row for row in csv.DictReader(handle, delimiter="\t")
        }
    catalog_ids = {row["mag_id"] for row in catalog}
    if len(catalog_ids) != len(catalog):
        raise ValueError("MAG catalog must have one unique mag_id per row")
    kbase_ids = {mag_id_from_kbase_name(str(info[1])) for info in genome_infos}
    if kbase_ids - catalog_ids:
        unexpected = ", ".join(sorted(kbase_ids - catalog_ids)[:10])
        raise ValueError(
            "public KBase catalog contains MAG IDs absent from the Zenodo catalog; "
            f"stage them explicitly before reconciliation: {unexpected}"
        )
    rows = reconciliation_rows(
        catalog,
        source_annotations,
        genome_infos,
        genomeset_elements,
        int(workspace_info[0]),
        str(workspace_info[1]),
        genomeset_ref,
    )
    generated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    reconciliation_path = output_dir / "mucc_v1_kbase_public_catalog_reconciliation.tsv"
    summary_path = output_dir / "mucc_v1_kbase_public_catalog_summary.json"
    write_tsv(reconciliation_path, rows)
    summary = build_summary(
        rows=rows,
        workspace_info=workspace_info,
        genomeset_info=genomeset_info,
        generated_utc=generated_utc,
        raw_paths=raw_paths,
    )
    summary["reconciliation_table"] = str(reconciliation_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
