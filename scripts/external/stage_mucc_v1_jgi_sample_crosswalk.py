#!/usr/bin/env python3
"""Stage exact JGI Sample QC identities for MUCC source expression labels.

The NCBI BioProject titles identify the JGI sequencing project for 107 MUCC
metatranscriptome labels. JGI's public Sample QC export can then resolve an
exact JGI sample identifier and a sample-receipt timestamp. Neither field is a
collection datetime, measured depth, environmental measurement, or flux link;
all ecological joins remain explicitly blocked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
JGI_PORTAL_BASE = "https://genome.jgi.doe.gov"
CLAIM_BOUNDARY = (
    "JGI Sample QC links establish expression-label-to-JGI-sample identity and an operational "
    "sample-receipt timestamp only. They do not establish collection datetime, depth in cm, "
    "environmental/flux pairing, ecological association, measured methane flux, final MRV "
    "score/A-E tier, crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--jgi-workers",
        type=int,
        default=2,
        help="Maximum concurrent public JGI requests (default: 2).",
    )
    parser.add_argument(
        "--max-projects",
        type=int,
        default=0,
        help="Optional deterministic limit for source-recovery smoke tests; zero means all.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_bytes(url: str, destination: Path, force: bool, retries: int = 3) -> tuple[bytes, str]:
    """Fetch and cache an immutable source response with bounded retries."""

    if destination.is_file() and destination.stat().st_size and not force:
        return destination.read_bytes(), url
    destination.parent.mkdir(parents=True, exist_ok=True)
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "MethaNet MUCC JGI sample-identity recovery"},
            )
            with urllib.request.urlopen(request, timeout=150) as response:
                payload = response.read()
                final_url = response.geturl()
            if not payload:
                raise ValueError(f"empty source response from {url}")
            destination.write_bytes(payload)
            return payload, final_url
        except Exception as caught:  # network sources are preserved as explicit statuses below
            error = caught
            if isinstance(caught, urllib.error.HTTPError) and caught.code != 429:
                break
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def fetch_jgi_bytes(url: str, destination: Path, force: bool) -> tuple[bytes, str]:
    """Fetch a JGI public endpoint with bounded curl timeouts.

    The portal can leave urllib connections open after throttling. curl returns
    its HTTP failure quickly, which lets this source-recovery workflow retain a
    precise unresolved status instead of waiting on a hung connection.
    """

    if destination.is_file() and destination.stat().st_size and not force:
        return destination.read_bytes(), url
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--connect-timeout",
        "10",
        "--max-time",
        "15",
        "--user-agent",
        "MethaNet MUCC JGI sample-identity recovery",
        "--output",
        str(temporary),
        "--write-out",
        "%{url_effective}",
        url,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip() or f"curl exit code {completed.returncode}"
        raise RuntimeError(detail)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"empty JGI source response from {url}")
    temporary.replace(destination)
    return destination.read_bytes(), completed.stdout.strip() or url


def candidate_labels(value: str) -> list[tuple[str, str]]:
    candidates = [(value, "exact_label")]
    if value.startswith("July_"):
        candidates.append(("Jul_" + value.removeprefix("July_"), "deterministic_July_to_Jul"))
    if value.startswith("Jul_"):
        candidates.append(("July_" + value.removeprefix("Jul_"), "deterministic_Jul_to_July"))
    return candidates


def parse_ncbi_bioproject_xml(payload: bytes) -> dict[str, str]:
    root = ET.fromstring(payload)
    project = root.find(".//Project")
    if project is None:
        raise ValueError("NCBI BioProject XML has no Project element")
    archive = project.find(".//ArchiveID[@archive='NCBI']")
    center = project.find(".//CenterID[@center='DOE Joint Genome Institute']")
    title = project.findtext(".//ProjectDescr/Title", default="").strip()
    description = project.findtext(".//ProjectDescr/Description", default="").strip()
    external_urls = [
        (item.findtext("URL", default="").strip(), item.get("label", "").strip())
        for item in project.findall(".//ExternalLink")
    ]
    genome_portal_url = next(
        (url for url, label in external_urls if label == "JGI Genome Portal"), ""
    )
    gold_url = next((url for url, label in external_urls if label == "GOLD"), "")
    return {
        "bioproject_accession": archive.get("accession", "") if archive is not None else "",
        "bioproject_uid": archive.get("id", "") if archive is not None else "",
        "jgi_sequencing_project_id": center.get("id", "") if center is not None else "",
        "bioproject_title": title,
        "bioproject_description": description,
        "jgi_genome_portal_source_url": genome_portal_url,
        "gold_source_url": gold_url,
    }


def portal_id_from_url(url: str) -> str:
    match = re.search(r"/portal/([^/]+)/[^/]+\.info\.html", url)
    return match.group(1) if match else ""


def portal_id_from_info_html(payload: bytes) -> str:
    """Recover a portal slug from a cached JGI Info page when redirects are unavailable."""

    text = payload.decode("utf-8", errors="replace")
    match = re.search(r'href="/portal/([^/]+)/\1\.info\.html"', text)
    return match.group(1) if match else ""


def project_name_from_portal_html(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="replace")
    match = re.search(r"<b>Project name:</b>\s*</div>\s*<div>(.*?)<", text, flags=re.S)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def read_qc_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(payload.decode("utf-8-sig", errors="replace").splitlines()))


def match_qc_sample(source_label: str, rows: list[dict[str, str]]) -> tuple[dict[str, str] | None, str]:
    for candidate, method in candidate_labels(source_label):
        matches = [row for row in rows if row.get("Sample Name", "").strip() == candidate]
        unique = {
            (row.get("Sample Id", "").strip(), row.get("Sample Name", "").strip()): row
            for row in matches
        }
        if len(unique) == 1:
            return next(iter(unique.values())), method
        if len(unique) > 1:
            return None, "ambiguous_multiple_JGI_QC_sample_ids"
    return None, "no_exact_JGI_QC_sample_label"


def jgi_urls(jgi_project_id: str) -> tuple[str, str]:
    lookup = (
        f"{JGI_PORTAL_BASE}/portal/lookup?"
        + urllib.parse.urlencode(
            {
                "keyName": "jgiProjectId",
                "keyValue": jgi_project_id,
                "app": "Info",
                "showParent": "false",
            }
        )
    )
    qc = (
        f"{JGI_PORTAL_BASE}/portal/ext-api/search-service/exportQClist?"
        + urllib.parse.urlencode({"spProjects": jgi_project_id})
    )
    return lookup, qc


def recover_project(
    record: dict[str, str], source_dir: Path, force: bool
) -> dict[str, Any]:
    """Recover one BioProject's JGI portal and public Sample QC evidence."""

    accession = record["bioproject_accession"]
    uid = record["bioproject_uid"]
    xml_url = NCBI_EFETCH_URL + "?" + urllib.parse.urlencode(
        {"db": "bioproject", "id": uid, "retmode": "xml"}
    )
    xml_path = source_dir / "ncbi_bioproject_xml" / f"{accession}.xml"
    xml_payload, _ = fetch_bytes(xml_url, xml_path, force)
    parsed = parse_ncbi_bioproject_xml(xml_payload)
    if parsed["bioproject_accession"] != accession:
        raise ValueError(f"NCBI XML accession mismatch for {accession}")
    jgi_project_id = parsed["jgi_sequencing_project_id"]
    if not jgi_project_id:
        raise ValueError(f"no DOE JGI CenterID in NCBI BioProject {accession}")
    lookup_url, _ = jgi_urls(jgi_project_id)
    info_path = source_dir / "jgi_portal_info" / f"{accession}.html"
    info_payload, info_final_url = fetch_jgi_bytes(lookup_url, info_path, force)
    portal_id = portal_id_from_url(info_final_url) or portal_id_from_info_html(info_payload)
    if not portal_id:
        raise ValueError(f"JGI lookup did not resolve a final deliverable portal for {accession}")
    _, qc_url = jgi_urls(jgi_project_id)
    qc_path = source_dir / "jgi_sample_qc_reports" / f"{accession}.csv"
    qc_rows: list[dict[str, str]] = []
    qc_recovery_error = ""
    try:
        qc_payload, _ = fetch_jgi_bytes(qc_url, qc_path, force)
        qc_rows = read_qc_rows(qc_payload)
    except Exception as error:
        qc_recovery_error = f"JGI_Sample_QC_recovery_failed: {type(error).__name__}: {error}"
    return {
        **record,
        **parsed,
        "jgi_final_deliverable_portal_id": portal_id,
        "jgi_portal_final_url": info_final_url,
        "jgi_portal_project_name": project_name_from_portal_html(info_payload),
        "jgi_lookup_url": lookup_url,
        "jgi_qc_url": qc_url,
        "jgi_qc_rows": qc_rows,
        "jgi_sample_qc_recovery_error": qc_recovery_error,
    }


def build_crosswalk(samples: pd.DataFrame, recovered: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in samples.iterrows():
        accession = str(source.get("bioproject_accession", ""))
        source_label = str(source["source_sample_column"])
        base = {
            "lane_id": LANE_ID,
            "sample_id": str(source["sample_id"]),
            "source_sample_column": source_label,
            "bioproject_accession": accession,
            "bioproject_uid": str(source.get("bioproject_uid", "")),
        }
        project = recovered.get(accession)
        if not accession:
            rows.append(
                {
                    **base,
                    "jgi_sequencing_project_id": "",
                    "jgi_final_deliverable_portal_id": "",
                    "jgi_portal_final_url": "",
                    "jgi_portal_project_name": "",
                    "jgi_sample_id": "",
                    "jgi_sample_name": "",
                    "jgi_sample_receipt_datetime": "",
                    "jgi_sample_qc_datetime": "",
                    "jgi_sample_qc_result": "",
                    "jgi_sample_label_mapping_method": "no_NCBI_BioProject_link",
                    "jgi_sample_identity_status": "unmapped_preserved",
                    "collection_datetime_status": "unresolved_no_authoritative_sequence_project_link",
                    "depth_cm_join_status": "unresolved_source_depth_code_to_cm_mapping",
                    "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
                    "next_validation_action": "recover authoritative project/sample metadata for this preserved source label",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            continue
        if project is None:
            rows.append(
                {
                    **base,
                    "jgi_sequencing_project_id": "",
                    "jgi_final_deliverable_portal_id": "",
                    "jgi_portal_final_url": "",
                    "jgi_portal_project_name": "",
                    "jgi_sample_id": "",
                    "jgi_sample_name": "",
                    "jgi_sample_receipt_datetime": "",
                    "jgi_sample_qc_datetime": "",
                    "jgi_sample_qc_result": "",
                    "jgi_sample_label_mapping_method": "JGI_recovery_failed",
                    "jgi_sample_identity_status": "unresolved_JGI_source_error",
                    "collection_datetime_status": "unresolved_JGI_source_error",
                    "depth_cm_join_status": "unresolved_source_depth_code_to_cm_mapping",
                    "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
                    "next_validation_action": "retry or obtain the JGI Sample QC export for the linked BioProject",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            continue
        if not project.get("jgi_qc_rows"):
            recovery_error = str(project.get("jgi_sample_qc_recovery_error", ""))
            rows.append(
                {
                    **base,
                    "jgi_sequencing_project_id": project.get("jgi_sequencing_project_id", ""),
                    "jgi_final_deliverable_portal_id": project.get("jgi_final_deliverable_portal_id", ""),
                    "jgi_portal_final_url": project.get("jgi_portal_final_url", ""),
                    "jgi_portal_project_name": project.get("jgi_portal_project_name", ""),
                    "jgi_sample_id": "",
                    "jgi_sample_name": "",
                    "jgi_sample_receipt_datetime": "",
                    "jgi_sample_qc_datetime": "",
                    "jgi_sample_qc_result": "",
                    "jgi_sample_label_mapping_method": (
                        "JGI_Sample_QC_recovery_error"
                        if recovery_error
                        else "no_JGI_Sample_QC_export_available"
                    ),
                    "jgi_sample_identity_status": (
                        "unresolved_JGI_Sample_QC_recovery_error"
                        if recovery_error
                        else "unresolved_no_JGI_Sample_QC_export"
                    ),
                    "collection_datetime_status": (
                        "unresolved_JGI_Sample_QC_recovery_error"
                        if recovery_error
                        else "unresolved_no_JGI_Sample_QC_export"
                    ),
                    "depth_cm_join_status": "unresolved_source_depth_code_to_cm_mapping",
                    "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
                    "next_validation_action": "recover the JGI Sample QC export or a comparable authoritative sample record",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            continue
        qc_row, method = match_qc_sample(source_label, project["jgi_qc_rows"])
        if qc_row is None:
            status = "unresolved_JGI_QC_label" if method.startswith("no_") else method
            rows.append(
                {
                    **base,
                    "jgi_sequencing_project_id": project["jgi_sequencing_project_id"],
                    "jgi_final_deliverable_portal_id": project["jgi_final_deliverable_portal_id"],
                    "jgi_portal_final_url": project["jgi_portal_final_url"],
                    "jgi_portal_project_name": project["jgi_portal_project_name"],
                    "jgi_sample_id": "",
                    "jgi_sample_name": "",
                    "jgi_sample_receipt_datetime": "",
                    "jgi_sample_qc_datetime": "",
                    "jgi_sample_qc_result": "",
                    "jgi_sample_label_mapping_method": method,
                    "jgi_sample_identity_status": status,
                    "collection_datetime_status": "not_reported_by_JGI_Sample_QC_export",
                    "depth_cm_join_status": "not_reported_by_JGI_Sample_QC_export",
                    "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
                    "next_validation_action": "recover JGI/GOLD or publication sample metadata with collection date and depth",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            continue
        rows.append(
            {
                **base,
                "jgi_sequencing_project_id": project["jgi_sequencing_project_id"],
                "jgi_final_deliverable_portal_id": project["jgi_final_deliverable_portal_id"],
                "jgi_portal_final_url": project["jgi_portal_final_url"],
                "jgi_portal_project_name": project["jgi_portal_project_name"],
                "jgi_sample_id": qc_row.get("Sample Id", "").strip(),
                "jgi_sample_name": qc_row.get("Sample Name", "").strip(),
                "jgi_sample_receipt_datetime": qc_row.get("Sample Receipt Date", "").strip(),
                "jgi_sample_qc_datetime": qc_row.get("Sample Qc Date", "").strip(),
                "jgi_sample_qc_result": qc_row.get("Sample Qc Result", "").strip(),
                "jgi_sample_label_mapping_method": method,
                "jgi_sample_identity_status": "exact_source_label_to_JGI_Sample_QC_record",
                "collection_datetime_status": "not_reported_by_JGI_Sample_QC_export",
                "depth_cm_join_status": "not_reported_by_JGI_Sample_QC_export",
                "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
                "next_validation_action": (
                    "recover collection date/depth and join JGI sample to explicit ESS-DIVE environmental and flux windows"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(samples) or frame["sample_id"].duplicated().any():
        raise ValueError("JGI crosswalk must retain each source expression sample exactly once")
    return frame


def source_manifest_rows(source_dir: Path, retrieved_utc: str) -> list[dict[str, Any]]:
    role_by_parent = {
        "ncbi_bioproject_xml": "NCBI_BioProject_XML",
        "jgi_portal_info": "JGI_Genome_Portal_Info",
        "jgi_sample_qc_reports": "JGI_Sample_QC_report",
    }
    rows: list[dict[str, Any]] = []
    for parent, role in role_by_parent.items():
        for path in sorted((source_dir / parent).glob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "lane_id": LANE_ID,
                    "source_id": "NCBI_BioProject_to_JGI_GenomePortal_SampleQC",
                    "artifact_role": role,
                    "source_url": "retained_in_crosswalk_source_columns",
                    "local_path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "retrieved_utc": retrieved_utc,
                    "validation_status": "retrieved_and_parseable",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return rows


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def main() -> int:
    args = parse_args()
    if args.jgi_workers < 1:
        raise SystemExit("--jgi-workers must be at least 1")
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    source_dir = run_dir / "source_audit/jgi_owc_sample_crosswalk"
    bioproject = pd.read_csv(
        run_dir / "environmental_metadata/link_mucc_v1_sequence_bioproject_sample.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    projects = (
        bioproject.loc[bioproject["bioproject_accession"].ne("")]
        [["bioproject_accession", "bioproject_uid"]]
        .drop_duplicates()
        .sort_values("bioproject_accession")
        .to_dict("records")
    )
    if args.max_projects:
        projects = projects[: args.max_projects]
    recovered: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    # NCBI asks unauthenticated clients to stay below three requests per second.
    # Fetch its small XML records serially; JGI portal/QC requests are below the
    # configured bounded worker count and use independent cache paths.
    for record in projects:
        accession = str(record["bioproject_accession"])
        try:
            xml_url = NCBI_EFETCH_URL + "?" + urllib.parse.urlencode(
                {"db": "bioproject", "id": record["bioproject_uid"], "retmode": "xml"}
            )
            xml_path = source_dir / "ncbi_bioproject_xml" / f"{accession}.xml"
            payload, _ = fetch_bytes(xml_url, xml_path, args.force_download)
            parsed = parse_ncbi_bioproject_xml(payload)
            if parsed["bioproject_accession"] != accession:
                raise ValueError("NCBI BioProject accession does not match the title crosswalk")
            recovered[accession] = {**record, **parsed}
        except Exception as error:
            errors[accession] = f"NCBI_BioProject_recovery_failed: {type(error).__name__}: {error}"
        time.sleep(0.35)
    def recover_jgi(record: dict[str, Any]) -> dict[str, Any]:
        return recover_project(record, source_dir, args.force_download)
    with ThreadPoolExecutor(max_workers=args.jgi_workers) as executor:
        pending = {
            executor.submit(recover_jgi, record): accession
            for accession, record in recovered.items()
            if record.get("jgi_sequencing_project_id")
        }
        for future in as_completed(pending):
            accession = pending[future]
            try:
                recovered[accession] = future.result()
                if recovered[accession].get("jgi_sample_qc_recovery_error"):
                    errors[accession] = recovered[accession]["jgi_sample_qc_recovery_error"]
            except Exception as error:
                recovered[accession]["jgi_sample_qc_recovery_error"] = (
                    f"JGI_Sample_QC_recovery_failed: {type(error).__name__}: {error}"
                )
                errors[accession] = recovered[accession]["jgi_sample_qc_recovery_error"]
    crosswalk = build_crosswalk(bioproject, recovered)
    output = run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_sample.tsv"
    write_tsv(output, crosswalk)
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    write_tsv(source_dir / "source_file_manifest.tsv", pd.DataFrame(source_manifest_rows(source_dir, retrieved_utc)))
    (source_dir / "recovery_errors.json").write_text(json_dumps(errors))
    (source_dir / "source_scope_notes.json").write_text(
        json_dumps(
            {
                "crosswalk_scope": (
                    "JGI Genome Portal Sample QC exports establish expression-label-to-JGI-sample identity, "
                    "sample receipt timestamp, and sample QC timestamp/result only."
                ),
                "prohibited_inferences": [
                    "Treating JGI Sample QC receipt timestamp as collection datetime",
                    "Treating a source depth-context label as measured depth in cm",
                    "Joining ESS-DIVE environmental or flux observations without explicit sample/date/depth/spatial-temporal keys",
                    "Promoting molecular or identity evidence to a final MRV score, A-E risk tier, carbon-crediting claim, or transfer claim",
                ],
                "ncbi_biosample_policy": (
                    "Do not use NCBI BioSample associations recovered from BioProject metadata as an OWC sample crosswalk "
                    "unless each is independently reconciled to the published OWC source roster; an entity-conflicted "
                    "spot check demonstrated that this route is not a safe substitute for the roster."
                ),
                "validated_source_routes": [
                    "NCBI BioProject XML title and DOE Joint Genome Institute CenterID",
                    "JGI Genome Portal lookup redirect for the JGI project ID",
                    "JGI public Sample QC export for the final deliverable portal",
                ],
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    )
    exact = int(crosswalk["jgi_sample_identity_status"].eq("exact_source_label_to_JGI_Sample_QC_record").sum())
    summary = {
        "lane_id": LANE_ID,
        "crosswalk": str(output),
        "expression_samples": int(len(crosswalk)),
        "exact_JGI_Sample_QC_identity_links": exact,
        "unresolved_or_unmapped": int(len(crosswalk) - exact),
        "projects_with_JGI_Sample_QC_exports": int(
            sum(bool(record.get("jgi_qc_rows")) for record in recovered.values())
        ),
        "projects_with_NCBI_to_JGI_project_identity": int(len(recovered)),
        "projects_with_unavailable_JGI_Sample_QC_exports": int(
            crosswalk["jgi_sample_identity_status"].eq(
                "unresolved_JGI_Sample_QC_recovery_error"
            ).sum()
        ),
        "projects_with_top_level_recovery_errors": int(len(errors)),
        "exact_collection_datetime_or_depth_cm_mapped": 0,
        "environment_or_flux_joined": 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (source_dir / "crosswalk_summary.json").write_text(json_dumps(summary))
    print(json_dumps(summary))
    return 0


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
