#!/usr/bin/env python3
"""Record authoritative MUCC v1 source-metadata recovery evidence and blockers.

The ledger is intentionally evidence-preserving: it records a retrieved source
file even when the payload is malformed, rather than treating its filename as a
usable accession or sample crosswalk.  This keeps the published MAG denominator
and ecological sample-link gaps auditable in the atlas warehouse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
PMC_ID = "PMC13289110"
ARTICLE_DOI = "10.1128/msystems.00680-25"
ZENODO_CONCEPT_RECORD = "8194032"
ZENODO_PAYLOAD_RECORD = "8194033"
ZENODO_CONCEPT_API_URL = f"https://zenodo.org/api/records/{ZENODO_CONCEPT_RECORD}"
ZENODO_PAYLOAD_API_URL = f"https://zenodo.org/api/records/{ZENODO_PAYLOAD_RECORD}"
PMC_XML_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=13289110"
)
EUROPE_PMC_SUPPLEMENTS_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13289110/supplementaryFiles"
)
KBASE_WORKSPACE_API_URL = "https://kbase.us/services/ws"
SUPPLEMENT_METHODS_FILENAME = "msystems.00680-25-s0001.pdf"
SUPPLEMENT_TABLES_FILENAME = "msystems.00680-25-s0002.xlsx"
# The publisher's supplemental-material listing reports this workbook as 95.67
# KB.  That is 97,962 bytes (base-2 KiB display), the exact size of the
# publisher-derived payload retrieved through Europe PMC on 2026-07-18.  Size
# agreement is source-recovery evidence only: it cannot make a malformed ZIP
# container usable or establish byte-for-byte identity with a browser-gated
# publisher response.
PUBLISHER_LISTED_SUPPLEMENT_TABLE_BYTES = 97_962
CLAIM_BOUNDARY = (
    "This ledger records source-recovery evidence only. A retrievable or named source "
    "does not establish a MAG roster match, sample/date/depth crosswalk, ecological "
    "association, measured-flux effect, final MRV score/A-E tier, crediting claim, or "
    "source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=BASE / "source_audit/mucc_v1_source_metadata_recovery_ledger.tsv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE / "source_audit/mucc_v1_source_metadata_recovery_summary.json",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Retrieve PMC source artifacts again even if the local audit copy exists.",
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


def download(url: str, destination: Path, force: bool) -> None:
    if destination.is_file() and destination.stat().st_size and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MethaNet atlas source-recovery ledger"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"empty response while retrieving {url}")
    destination.write_bytes(payload)


def validate_xlsx_container(path: Path) -> tuple[str, str]:
    """Return an audit status and detail without pretending a broken XLSX is usable."""
    if not path.is_file() or not path.stat().st_size:
        return "missing", "source payload is absent"
    if not path.read_bytes().startswith(b"PK"):
        return "malformed_not_zip_container", "payload does not start with an XLSX/ZIP signature"
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if "[Content_Types].xml" not in members or "xl/workbook.xml" not in members:
                return "malformed_not_xlsx", "ZIP container lacks required XLSX members"
            bad_member = archive.testzip()
            if bad_member:
                return "malformed_crc_failure", f"CRC validation failed for member {bad_member}"
    except zipfile.BadZipFile as exc:
        return "malformed_no_central_directory", f"no readable central directory: {exc}"
    return "parseable_xlsx_container", "XLSX/ZIP container has a readable central directory"


def publisher_listing_size_evidence(path: Path) -> tuple[bool, str]:
    """Return bounded evidence about the published supplemental-file size.

    The publisher's web page is browser-gated in this execution environment.
    Matching its displayed size to a retrieved payload is useful for diagnosing
    whether the public object itself is defective, but it does not attest that
    the retrieved bytes are an authoritative binary duplicate.
    """
    if not path.is_file():
        return False, "payload absent; cannot compare to the publisher-listed size"
    payload_bytes = path.stat().st_size
    if payload_bytes == PUBLISHER_LISTED_SUPPLEMENT_TABLE_BYTES:
        return (
            True,
            "payload bytes match the publisher-listed 95.67 KB size (97,962 bytes), "
            "but size agreement does not repair or authenticate the malformed XLSX",
        )
    return (
        False,
        f"payload has {payload_bytes} bytes; publisher lists 95.67 KB "
        f"({PUBLISHER_LISTED_SUPPLEMENT_TABLE_BYTES} bytes)",
    )


def extract_supplement(archive_path: Path, destination: Path) -> None:
    """Extract one named supplement without assuming that it is semantically usable."""
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith(destination.name)]
        if len(candidates) != 1:
            raise ValueError(
                "Europe PMC supplement archive must contain exactly one requested supplement; "
                f"requested={destination.name} found={len(candidates)}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(candidates[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def pdf_text(path: Path) -> tuple[str, str]:
    """Return extracted text or a diagnostic; no text means no methods claim is made."""
    if not path.is_file() or not path.stat().st_size:
        return "", "source PDF is absent"
    if not path.read_bytes().startswith(b"%PDF-"):
        return "", "payload does not start with a PDF signature"
    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return "", (
            "pdftotext could not extract readable text: "
            f"{completed.stderr.strip() or 'no output'}"
        )
    return completed.stdout, "PDF text extracted with pdftotext"


def methods_design_is_confirmed(text: str) -> bool:
    normalized = re.sub(r"(?<=\d)-\s+(?=\d)", "-", " ".join(text.split()))
    required = [
        "New metatranscriptomes were selected to capture temporal",
        "vertical (D1, D3, D5)",
        "From the field in 2018 we obtained 109",
        "0-5 cm, 10-15 cm, and 20-25 cm",
        "5 depths (0-5 cm, 5-10 cm, 10-15 cm, 15-20, and 20-25 cm)",
    ]
    return all(marker in normalized for marker in required)


def methods_design_context_rows(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Materialize only methods-derived sample context, never a field-observation join.

    The intact supplemental PDF establishes the 2018 cohort design.  It does
    not replace the malformed Table S4 row-level workbook, so the resulting
    rows retain an explicit non-eligibility status for environmental or flux
    validation.  In particular, the source labels contain ``D6`` whereas the
    PDF describes the deep July/September stratum as ``D5``; the nominal
    interval remains visible but is deliberately marked unreconciled.
    """
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"^(?P<month>Aug|July|Sept)_(?P<site>M1|N3|OW2)_(?P<core>C\d+)_(?P<depth>D\d+)(?:_(?P<replicate>[A-Z]))?$"
    )
    aug_depths = {
        "D1": "0-5",
        "D2": "5-10",
        "D3": "10-15",
        "D4": "15-20",
        "D5": "20-25",
    }
    seasonal_depths = {"D1": "0-5", "D3": "10-15"}
    for sample in samples:
        label = sample["source_sample_column"]
        match = pattern.fullmatch(label)
        base = {
            "lane_id": "mucc_v1_owc_wetland",
            "sample_id": sample["sample_id"],
            "source_sample_column": label,
            "method_source": f"{ARTICLE_DOI}.SuF1",
            "method_evidence_pointer": (
                "Supplemental Information PDF: Field campaign and sample collection; "
                "Metagenomic and metatranscriptomic sequencing"
            ),
            "exact_table_s4_row_status": "blocked_published_Tables_S1_S13_XLSX_malformed",
            "environment_flux_join_status": "blocked_no_exact_sample_date_depth_environment_flux_crosswalk",
            "validation_split_eligibility": "not_eligible_methods_context_is_not_an_ecological_join",
            "claim_boundary": CLAIM_BOUNDARY,
        }
        if match is None:
            year = sample.get("collection_year", "")
            base.update(
                {
                    "methods_cohort": "legacy_2014_2015_label_only",
                    "methods_collection_year": year,
                    "methods_collection_month": sample.get("month_label", ""),
                    "methods_site_or_landcover": sample.get("site_or_landcover", ""),
                    "methods_core_label": sample.get("core", ""),
                    "raw_depth_code": sample.get("depth_code", ""),
                    "raw_replicate_label": sample.get("replicate", ""),
                    "nominal_depth_interval_cm": "",
                    "methods_depth_assignment_status": "unresolved_legacy_label_not_a_5cm_interval",
                    "methods_design_context_status": "legacy_sample_label_context_only",
                }
            )
        else:
            month = match.group("month")
            depth = match.group("depth")
            base.update(
                {
                    "methods_cohort": "field_2018_metatranscriptome_design",
                    "methods_collection_year": "2018",
                    "methods_collection_month": month,
                    "methods_site_or_landcover": match.group("site"),
                    "methods_core_label": match.group("core"),
                    "raw_depth_code": depth,
                    "raw_replicate_label": match.group("replicate") or "",
                }
            )
            if month == "Aug" and depth in aug_depths:
                base.update(
                    {
                        "nominal_depth_interval_cm": aug_depths[depth],
                        "methods_depth_assignment_status": "methods_design_direct_5cm_depth_code",
                        "methods_design_context_status": "validated_2018_methods_context_direct_depth_code",
                    }
                )
            elif month in {"July", "Sept"} and depth in seasonal_depths:
                base.update(
                    {
                        "nominal_depth_interval_cm": seasonal_depths[depth],
                        "methods_depth_assignment_status": "methods_design_direct_3depth_code",
                        "methods_design_context_status": "validated_2018_methods_context_direct_depth_code",
                    }
                )
            elif month in {"July", "Sept"} and depth == "D6":
                base.update(
                    {
                        "nominal_depth_interval_cm": "20-25",
                        "methods_depth_assignment_status": (
                            "methods_design_constrained_deep_interval_raw_D6_conflicts_with_published_D5"
                        ),
                        "methods_design_context_status": (
                            "validated_2018_cohort_but_raw_depth_code_reconciliation_pending"
                        ),
                    }
                )
            else:
                base.update(
                    {
                        "nominal_depth_interval_cm": "",
                        "methods_depth_assignment_status": "unresolved_unexpected_2018_depth_code",
                        "methods_design_context_status": "unresolved_source_label_pattern",
                    }
                )
        rows.append(base)
    return rows


def write_generic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty methods-design table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def methods_design_reconciliation_rows(
    context_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    def count(status: str) -> int:
        return sum(
            row["methods_design_context_status"] == status for row in context_rows
        )

    total = len(context_rows)
    field_2018 = sum(
        row["methods_cohort"] == "field_2018_metatranscriptome_design"
        for row in context_rows
    )
    direct_depth = count("validated_2018_methods_context_direct_depth_code")
    d6_pending = count(
        "validated_2018_cohort_but_raw_depth_code_reconciliation_pending"
    )
    legacy = count("legacy_sample_label_context_only")
    rows = [
        (
            "expression_sample_columns",
            "133",
            str(total),
            "pass" if total == 133 else "fail",
            "processed expression matrix sample columns",
        ),
        (
            "2018_methods_defined_metatranscriptomes",
            "109",
            str(field_2018),
            "pass" if field_2018 == 109 else "fail",
            "SuF1 reports 109 field metatranscriptomes in 2018",
        ),
        (
            "direct_methods_depth_code_context",
            "91",
            str(direct_depth),
            "pass" if direct_depth == 91 else "fail",
            "August D1-D5 plus July/September D1 and D3 only; no field observation join implied",
        ),
        (
            "raw_D6_vs_published_D5_depth_code_reconciliation",
            "18",
            str(d6_pending),
            "partial" if d6_pending == 18 else "fail",
            "raw July/September D6 labels occupy the published deep 20-25 cm design slot, but Table S4 is required to reconcile the D6/D5 code discrepancy",
        ),
        (
            "legacy_2014_2015_label_only_context",
            "24",
            str(legacy),
            "blocked" if legacy == 24 else "fail",
            "legacy labels do not provide an authoritative 5-cm interval or paired field-observation join",
        ),
    ]
    return [
        {
            "lane_id": "mucc_v1_owc_wetland",
            "metric": metric,
            "expected": expected,
            "observed": observed,
            "status": status,
            "evidence": evidence,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for metric, expected, observed, status, evidence in rows
    ]


def fulltext_confirms_supplements(xml_path: Path) -> bool:
    root = ET.parse(xml_path).getroot()
    names = {
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", maxsplit=1)[-1] == "title"
    }
    return {
        "msystems.00680-25-s0001.pdf",
        "msystems.00680-25-s0002.xlsx",
    }.issubset(names)


def validate_zenodo_archive_roster(catalog_path: Path) -> tuple[str, str, int]:
    """Validate the checksum-backed archive inventory independently of source-QC evidence."""
    with catalog_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"mag_id", "archive_member", "zip_crc", "source_mag_fasta_status"}
    if not rows or not required.issubset(rows[0]):
        return "invalid_catalog_inventory", "catalog is missing archive-inventory columns", 0
    mag_ids = [row["mag_id"] for row in rows]
    members = [row["archive_member"] for row in rows]
    valid_rows = all(
        row["archive_member"].startswith("MAGs/")
        and row["zip_crc"]
        and row["source_mag_fasta_status"] == "downloaded_validated_in_MAGs.zip"
        for row in rows
    )
    if len(set(mag_ids)) != len(rows) or len(set(members)) != len(rows) or not valid_rows:
        return "invalid_catalog_inventory", "archive-member identity or validation fields are incomplete", len(rows)
    return (
        "archive_roster_recovered_checksum_validated",
        (
            f"{len(rows)} unique MAG archive members have retained member paths and ZIP CRCs; "
            "the Zenodo record description and paper headline still state 2,502 HQ/MQ MAGs"
        ),
        len(rows),
    )


def zenodo_source_qc_evidence(path: Path) -> tuple[str, str, str, str, dict[str, int]]:
    """Validate the compact, repeat-value-checked per-MAG QC reconciliation table."""
    zero = {"rows": 0, "hqmq": 0, "archive_scope": 0, "consistent": 0}
    if not path.is_file() or not path.stat().st_size:
        return (
            "direct_Zenodo_source_QC_not_staged",
            "record-specific Zenodo per-MAG QC reconciliation table is absent",
            "published HQ/MQ membership cannot be assigned until the direct source QC payload is reconciled",
            "stage OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz with the source-bin crosswalk",
            zero,
        )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "mag_id",
        "bin_completeness",
        "bin_contamination",
        "source_qc_value_consistency_status",
        "published_mq_hq_membership_status",
    }
    if not rows or not required.issubset(rows[0]):
        return (
            "direct_Zenodo_source_QC_invalid",
            "per-MAG QC reconciliation table lacks required columns",
            "published HQ/MQ membership remains unverified",
            "regenerate and validate the direct Zenodo QC reconciliation table",
            zero,
        )
    counts = {
        "rows": len(rows),
        "hqmq": sum(
            row.get("published_mq_hq_membership_status")
            == "meets_published_MQHQ_CheckM_threshold"
            for row in rows
        ),
        "archive_scope": sum(
            row.get("published_mq_hq_membership_status")
            == "does_not_meet_published_MQHQ_CheckM_threshold"
            for row in rows
        ),
        "consistent": sum(
            row.get("source_qc_value_consistency_status")
            == "direct_source_qc_values_consistent_across_annotation_rows"
            for row in rows
        ),
    }
    if (
        counts["rows"] == 2508
        and counts["hqmq"] == 2502
        and counts["archive_scope"] == 6
        and counts["consistent"] == 2508
    ):
        return (
            "exact_published_2502_HQMQ_and_six_archive_scope_difference_reconciled",
            "record-specific Zenodo DRAM annotations provide consistent CheckM completeness/contamination "
            "for all 2,508 archive MAGs; the paper-defined adapted MIMARKS screen (completeness >=50%, "
            "contamination <10%) identifies 2,502 HQ/MQ MAGs and six explicit archive-scope exceptions",
            "none for MAG quality membership; accession-level and ecological sample joins remain separately blocked",
            "retain published_mq_hq_membership_status on every MAG and exclude the six exceptions only when "
            "using the paper-defined HQ/MQ denominator",
            counts,
        )
    return (
        "direct_Zenodo_source_QC_reconciliation_incomplete",
        f"source QC rows={counts['rows']}; HQ/MQ={counts['hqmq']}; archive-scope={counts['archive_scope']}; "
        f"consistent repeated values={counts['consistent']}",
        "published HQ/MQ membership remains unverified until all expected source-QC counts reconcile",
        "investigate missing, inconsistent, or nonconcordant source QC records before denominator promotion",
        counts,
    )
def zenodo_release_evidence(
    payload_record_path: Path,
    concept_record_path: Path,
) -> tuple[str, str, str, str]:
    """Pin the six-payload Zenodo release rather than trusting a mutable concept DOI."""
    expected_files = {
        "MAGs.zip",
        "OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz",
        "OWC_HQMQ_DB_genes.faa.gz",
        "owc_metat_table_mags.csv",
        "owc_metat_table_mags_genes.csv",
        "owc_metat_table_mags_genes_annotations.csv",
    }
    try:
        payload = json.loads(payload_record_path.read_text())
        concept = json.loads(concept_record_path.read_text())
        payload_files = {item["key"] for item in payload["files"]}
        concept_files = {item["key"] for item in concept["files"]}
        payload_doi = str(payload["metadata"]["doi"])
        concept_current_doi = str(concept["metadata"]["doi"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return (
            "Zenodo_release_metadata_invalid",
            f"could not validate staged Zenodo release metadata: {exc}",
            "the archived source payload release cannot be pinned reproducibly",
            "refresh and validate record-specific Zenodo metadata before source reconstruction",
        )
    if payload_files != expected_files or payload_doi != "10.5281/zenodo.8194033":
        return (
            "Zenodo_payload_release_mismatch",
            (
                f"record {ZENODO_PAYLOAD_RECORD} DOI={payload_doi}; files={sorted(payload_files)}; "
                "expected the six source payloads used by the local lane"
            ),
            "the downloaded payloads cannot be tied to the expected record-specific release",
            "do not refresh from the concept DOI; recover and validate the exact 8194033 payload release",
        )
    return (
        "record_specific_six_payload_release_pinned_concept_doi_version_differs",
        (
            f"record {ZENODO_PAYLOAD_RECORD} has the expected six source payloads and DOI={payload_doi}; "
            f"concept record {ZENODO_CONCEPT_RECORD} currently reports DOI={concept_current_doi} "
            f"with files={sorted(concept_files)}"
        ),
        (
            "the paper's Zenodo concept DOI is not by itself a reproducible pointer to the six-payload "
            "release; record-specific 8194033 and local checksums must remain pinned"
        ),
        (
            "cite and retrieve 10.5281/zenodo.8194033 for the six-payload source release; treat the "
            "concept DOI as a discovery pointer only"
        ),
    )
def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "lane_id",
        "source_id",
        "artifact_role",
        "source_url",
        "local_path",
        "bytes",
        "sha256",
        "availability_status",
        "validation_detail",
        "blocking_effect",
        "next_action",
        "retrieved_utc",
        "claim_boundary",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def artifact_row(
    *,
    source_id: str,
    artifact_role: str,
    source_url: str,
    path: Path | None,
    status: str,
    detail: str,
    blocking_effect: str,
    next_action: str,
    retrieved_utc: str,
) -> dict[str, Any]:
    return {
        "lane_id": "mucc_v1_owc_wetland",
        "source_id": source_id,
        "artifact_role": artifact_role,
        "source_url": source_url,
        "local_path": str(path) if path is not None else "",
        "bytes": path.stat().st_size if path is not None and path.is_file() else "",
        "sha256": sha256(path) if path is not None and path.is_file() else "",
        "availability_status": status,
        "validation_detail": detail,
        "blocking_effect": blocking_effect,
        "next_action": next_action,
        "retrieved_utc": retrieved_utc,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def kbase_public_catalog_evidence(path: Path) -> tuple[str, str, str, str]:
    """Describe the staged public KBase roster without turning it into QC evidence."""
    if not path.is_file() or not path.stat().st_size:
        return (
            "public_api_catalog_not_staged",
            "the browser narrative is login-gated; the public Workspace API has not yet been staged",
            "KBase cannot yet contribute a reproducible public identity or supplementary-taxonomy layer",
            "stage the public Workspace API roster, then preserve its quality-scope boundary",
        )
    try:
        summary = json.loads(path.read_text())
        kbase = int(summary["public_kbase_exact_mag_id_matches"])
        archive = int(summary["zenodo_checksum_validated_archive_mag_count"])
        absent = int(summary["zenodo_archive_mag_absent_from_public_kbase"])
        quality_status = str(summary["quality_scope"]["status"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return (
            "public_api_catalog_stage_invalid",
            f"staged public KBase summary could not be validated: {exc}",
            "KBase metadata cannot be used until the staged summary validates",
            "refresh the public KBase staging artifact and verify its roster counts",
        )
    if archive != kbase + absent:
        return (
            "public_api_catalog_stage_invalid",
            f"KBase/public archive reconciliation is inconsistent: archive={archive}, KBase={kbase}, absent={absent}",
            "KBase metadata cannot be used until the staged counts reconcile",
            "refresh the public KBase staging artifact and verify its roster counts",
        )
    return (
        "public_api_roster_reconciled_quality_scope_unresolved",
        (
            f"public Workspace API roster has {kbase} exact MAG-ID matches within the "
            f"{archive}-member Zenodo archive; {absent} archive MAGs are absent; "
            f"KBase quality scope is {quality_status}"
        ),
        (
            "KBase improves public MAG identity and supplementary taxonomy coverage, but it does not "
            "identify the paper's 2,502 HQ/MQ subset or exact sample/date/depth/accession rows"
        ),
        (
            "retain KBase as a supplementary identity/taxonomy source; obtain the original Table S1-S13 "
            "or author-provided QC/accession roster before assigning HQ/MQ membership"
        ),
    )


def sra_sample_crosswalk_evidence(path: Path) -> tuple[str, str, str, str, dict[str, int]]:
    """Validate the entity-checked SRA crosswalk without fabricating field joins."""
    zero = {
        "rows": 0,
        "exact_packages": 0,
        "exact_dates": 0,
        "no_depth": 0,
        "rna_seq": 0,
        "wgs": 0,
        "unresolved": 0,
    }
    if not path.is_file() or not path.stat().st_size:
        return (
            "NCBI_SRA_crosswalk_not_staged",
            "entity-validated SRA package crosswalk is absent",
            "the warehouse lacks run/BioSample/package evidence for expression source labels",
            "stage exact-label SRA/BioSample packages and preserve unresolved labels explicitly",
            zero,
        )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "sample_id",
        "source_sample_column",
        "sra_sample_identity_status",
        "sra_collection_date_status",
        "sra_depth_cm_join_status",
        "sra_library_strategy",
    }
    if not rows or not required.issubset(rows[0]):
        return (
            "NCBI_SRA_crosswalk_invalid",
            "crosswalk has no rows or lacks required identity/date/depth/assay columns",
            "the SRA source cannot be used until its staged schema validates",
            "refresh the SRA package stage and validate its schema and entity checks",
            zero,
        )
    counts = {
        "rows": len(rows),
        "exact_packages": sum(
            row["sra_sample_identity_status"] == "exact_source_label_to_NCBI_SRA_package"
            for row in rows
        ),
        "exact_dates": sum(
            row["sra_collection_date_status"]
            == "exact_collection_date_from_NCBI_SRA_sample_attributes"
            for row in rows
        ),
        "no_depth": sum(
            row["sra_depth_cm_join_status"] == "not_reported_by_NCBI_SRA_sample_attributes"
            for row in rows
        ),
        "rna_seq": sum(row["sra_library_strategy"] == "RNA-Seq" for row in rows),
        "wgs": sum(row["sra_library_strategy"] == "WGS" for row in rows),
        "unresolved": sum(
            row["sra_sample_identity_status"] != "exact_source_label_to_NCBI_SRA_package"
            for row in rows
        ),
    }
    if len({row["sample_id"] for row in rows}) != len(rows) or len(
        {row["source_sample_column"] for row in rows}
    ) != len(rows):
        return (
            "NCBI_SRA_crosswalk_invalid_duplicate_sample_keys",
            "sample_id or source_sample_column is non-unique in the staged SRA crosswalk",
            "ambiguous sample keys prevent a safe sequence-to-package association",
            "deduplicate or preserve ambiguous candidates as separate unresolved rows before promotion",
            counts,
        )
    return (
        "partial_exact_SRA_package_identity_and_collection_dates",
        (
            f"rows={counts['rows']}; exact entity-validated packages={counts['exact_packages']}; "
            f"exact collection dates={counts['exact_dates']}; depth not reported={counts['no_depth']}; "
            f"declared RNA-Seq={counts['rna_seq']}; declared WGS={counts['wgs']}; "
            f"unresolved labels={counts['unresolved']}"
        ),
        (
            "exact package identity and collection-date evidence does not supply depth, field chemistry, "
            "porewater/chamber-flux windows, abundance/read coverage, or a validated assay-equivalence rule"
        ),
        (
            "recover authoritative depth/environment/flux mapping; reconcile WGS-declared packages before "
            "pooling with RNA-Seq expression; retain unresolved labels as explicit rows"
        ),
        counts,
    )


def jgi_data_portal_catalog_evidence(
    path: Path,
) -> tuple[str, str, str, str, dict[str, int]]:
    """Validate public JGI catalog evidence without treating it as field metadata."""
    zero = {
        "rows": 0,
        "exact_pairs": 0,
        "july_aliases": 0,
        "no_depth": 0,
        "indexed_files": 0,
        "purged_indexed_files": 0,
        "unresolved": 0,
    }
    if not path.is_file() or not path.stat().st_size:
        return (
            "JGI_Data_Portal_catalog_crosswalk_not_staged",
            "public JGI Data Portal source-label catalog crosswalk is absent",
            "the warehouse lacks independently queried JGI catalog/processing provenance",
            "stage source-label-specific public JGI catalog records and preserve unmatched labels",
            zero,
        )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "sample_id",
        "source_sample_column",
        "jgi_data_portal_identity_status",
        "jgi_data_portal_label_mapping_method",
        "jgi_data_portal_indexed_file_count",
        "jgi_data_portal_purged_file_count",
        "depth_cm_join_status",
    }
    if not rows or not required.issubset(rows[0]):
        return (
            "JGI_Data_Portal_catalog_crosswalk_invalid",
            "crosswalk has no rows or lacks required identity/catalog/depth columns",
            "JGI catalog evidence cannot be used until the staged schema validates",
            "refresh the JGI Data Portal staging artifact and validate its schema and sample keys",
            zero,
        )
    try:
        counts = {
            "rows": len(rows),
            "exact_pairs": sum(
                row["jgi_data_portal_identity_status"]
                == "exact_source_label_to_JGI_Data_Portal_record_pair"
                for row in rows
            ),
            "july_aliases": sum(
                row["jgi_data_portal_label_mapping_method"] == "deterministic_July_to_Jul"
                for row in rows
            ),
            "no_depth": sum(
                row["depth_cm_join_status"] == "not_reported_by_JGI_Data_Portal_catalog"
                for row in rows
            ),
            "indexed_files": sum(int(row["jgi_data_portal_indexed_file_count"]) for row in rows),
            "purged_indexed_files": sum(
                int(row["jgi_data_portal_purged_file_count"]) for row in rows
            ),
            "unresolved": sum(
                row["jgi_data_portal_identity_status"]
                != "exact_source_label_to_JGI_Data_Portal_record_pair"
                for row in rows
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return (
            "JGI_Data_Portal_catalog_crosswalk_invalid",
            f"JGI Data Portal crosswalk values could not be parsed: {exc}",
            "JGI catalog evidence cannot be used until its staged numeric fields validate",
            "refresh the JGI Data Portal staging artifact and validate its counts",
            zero,
        )
    if len({row["sample_id"] for row in rows}) != len(rows) or len(
        {row["source_sample_column"] for row in rows}
    ) != len(rows):
        return (
            "JGI_Data_Portal_catalog_crosswalk_invalid_duplicate_sample_keys",
            "sample_id or source_sample_column is non-unique in the staged JGI Data Portal crosswalk",
            "ambiguous sample keys prevent a safe sequence-to-catalog-record association",
            "deduplicate or preserve ambiguous candidates as separate unresolved rows before promotion",
            counts,
        )
    return (
        "partial_exact_JGI_Data_Portal_catalog_record_identity",
        (
            f"rows={counts['rows']}; exact source-label record pairs={counts['exact_pairs']}; "
            f"controlled July-to-Jul aliases={counts['july_aliases']}; indexed files="
            f"{counts['indexed_files']}; marked PURGED={counts['purged_indexed_files']}; "
            f"catalog depth not reported={counts['no_depth']}; unresolved labels={counts['unresolved']}"
        ),
        (
            "public JGI catalog identity, processing provenance, and source-record coordinates do not "
            "supply collection time, depth, field chemistry, porewater/chamber-flux windows, abundance/read "
            "coverage, or an ecological association; indexed downloads require JGI authentication"
        ),
        (
            "obtain an authoritative Table S4-equivalent field metadata roster and reconcile each exact "
            "JGI/NCBI record to its date, depth, chemistry, and flux window before ecological validation"
        ),
        counts,
    )


def build(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    output_tsv = resolve(repo_root, args.output_tsv)
    output_json = resolve(repo_root, args.output_json)
    audit_dir = run_dir / "source_audit/pmc_42223272"
    zenodo_audit_dir = run_dir / "source_audit/zenodo_mucc_v1_release"
    pmc_xml = audit_dir / f"{PMC_ID}_fulltext.xml"
    supplement_archive = audit_dir / f"{PMC_ID}_europepmc_supplementary_files.zip"
    methods_pdf = audit_dir / SUPPLEMENT_METHODS_FILENAME
    table_s1 = audit_dir / SUPPLEMENT_TABLES_FILENAME
    archive_catalog = run_dir / "manifests/mucc_v1_mag_catalog_full.tsv"
    sample_scaffold = run_dir / "environmental_metadata/mucc_v1_sample_columns_scaffold.tsv"
    methods_context_path = (
        run_dir / "environmental_metadata/feature_mucc_v1_sample_methods_design_context.tsv"
    )
    methods_reconciliation_path = (
        run_dir / "reconciliation/mucc_v1_methods_sample_design_reconciliation.tsv"
    )
    kbase_summary_path = (
        run_dir
        / "source_audit/kbase_public_workspace_147022/mucc_v1_kbase_public_catalog_summary.json"
    )
    sra_crosswalk_path = run_dir / "environmental_metadata/link_mucc_v1_sequence_sra_sample.tsv"
    jgi_data_portal_crosswalk_path = (
        run_dir / "environmental_metadata/link_mucc_v1_sequence_jgi_data_portal.tsv"
    )
    zenodo_payload_record = zenodo_audit_dir / f"zenodo_{ZENODO_PAYLOAD_RECORD}_record.json"
    zenodo_concept_record = zenodo_audit_dir / f"zenodo_{ZENODO_CONCEPT_RECORD}_record.json"
    download(PMC_XML_URL, pmc_xml, args.force_download)
    download(EUROPE_PMC_SUPPLEMENTS_URL, supplement_archive, args.force_download)
    download(ZENODO_PAYLOAD_API_URL, zenodo_payload_record, args.force_download)
    download(ZENODO_CONCEPT_API_URL, zenodo_concept_record, args.force_download)
    extract_supplement(supplement_archive, methods_pdf)
    extract_supplement(supplement_archive, table_s1)
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    confirms_supplements = fulltext_confirms_supplements(pmc_xml)
    xlsx_status, xlsx_detail = validate_xlsx_container(table_s1)
    publisher_size_matches, publisher_size_detail = publisher_listing_size_evidence(table_s1)
    xlsx_detail = f"{xlsx_detail}; {publisher_size_detail}"
    methods_text, methods_pdf_detail = pdf_text(methods_pdf)
    methods_confirmed = methods_design_is_confirmed(methods_text)
    if not methods_confirmed:
        raise ValueError(
            "Supplemental Information PDF did not contain all expected, source-specific "
            "2018 metatranscriptome design markers; no methods-derived context was materialized"
        )
    with sample_scaffold.open(newline="") as handle:
        scaffold_rows = list(csv.DictReader(handle, delimiter="\t"))
    context_rows = methods_design_context_rows(scaffold_rows)
    write_generic_tsv(methods_context_path, context_rows)
    design_reconciliation_rows = methods_design_reconciliation_rows(context_rows)
    write_generic_tsv(methods_reconciliation_path, design_reconciliation_rows)
    archive_status, archive_detail, archive_count = validate_zenodo_archive_roster(archive_catalog)
    source_qc_path = BASE / "functional_features/feature_mucc_v1_zenodo_source_qc.tsv"
    source_qc_status, source_qc_detail, source_qc_blocking, source_qc_next_action, source_qc_counts = (
        zenodo_source_qc_evidence(source_qc_path)
    )
    zenodo_status, zenodo_detail, zenodo_blocking, zenodo_next_action = zenodo_release_evidence(
        zenodo_payload_record, zenodo_concept_record
    )
    kbase_status, kbase_detail, kbase_blocking, kbase_next_action = kbase_public_catalog_evidence(
        kbase_summary_path
    )
    sra_status, sra_detail, sra_blocking, sra_next_action, sra_counts = sra_sample_crosswalk_evidence(
        sra_crosswalk_path
    )
    (
        jgi_data_portal_status,
        jgi_data_portal_detail,
        jgi_data_portal_blocking,
        jgi_data_portal_next_action,
        jgi_data_portal_counts,
    ) = jgi_data_portal_catalog_evidence(jgi_data_portal_crosswalk_path)
    needs_metadata = "NCBI accession and exact sample/date/depth crosswalk remain blocked"
    rows = [
        artifact_row(
            source_id=PMC_ID,
            artifact_role="open_access_fulltext_xml",
            source_url=PMC_XML_URL,
            path=pmc_xml,
            status="retrieved_and_xml_parseable" if confirms_supplements else "retrieved_but_supplements_unconfirmed",
            detail=(
                "full text confirms Supplemental Information PDF and Tables S1-S13 XLSX"
                if confirms_supplements
                else "full text did not expose both expected supplement titles"
            ),
            blocking_effect="none; establishes the published supplement names and source context",
            next_action="retain as provenance for Table S1 acquisition",
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id=f"{ARTICLE_DOI}.SuF1",
            artifact_role="published_Supplemental_Information_methods_PDF",
            source_url=f"https://journals.asm.org/doi/{ARTICLE_DOI}",
            path=methods_pdf,
            status=(
                "parseable_pdf_methods_design_confirmed"
                if methods_confirmed
                else "parseable_pdf_methods_design_unconfirmed"
            ),
            detail=(
                f"{methods_pdf_detail}; confirms the 109-sample 2018 design, August five-depth "
                "design, July/September three-depth design, and the published D1/D3/D5 notation"
            ),
            blocking_effect=(
                "permits methods-derived cohort context only; it does not supply Table S4's exact "
                "sample/date/depth/environment/flux rows"
            ),
            next_action=(
                "retain the methods-design context as non-ecological evidence and recover a "
                "parseable Table S4 before any field-observation join"
            ),
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id=PMC_ID,
            artifact_role="europe_pmc_supplement_archive",
            source_url=EUROPE_PMC_SUPPLEMENTS_URL,
            path=supplement_archive,
            status="retrieved_and_outer_zip_parseable",
            detail="outer supplementary archive contains the advertised Table S1-S13 payload",
            blocking_effect="none by itself; inner payload must independently validate",
            next_action="retain archive checksum and validate each extracted supplement",
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id="Zenodo_10.5281_zenodo.8194033",
            artifact_role="checksum_validated_MAG_archive_roster",
            source_url="https://doi.org/10.5281/zenodo.8194033",
            path=archive_catalog,
            status=archive_status,
            detail=archive_detail,
            blocking_effect=(
                "archive membership is reconciled to the local 2,508-MAG catalog; the separate direct source-QC artifact assigns the paper-defined HQ/MQ scope"
            ),
            next_action=(
                "retain all archive members and pair them with the direct source-QC reconciliation before denominator-sensitive use"
            ),
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id="Zenodo_10.5281_zenodo.8194033",
            artifact_role="direct_Zenodo_per_MAG_QC_and_published_HQMQ_scope_reconciliation",
            source_url="https://doi.org/10.5281/zenodo.8194033",
            path=source_qc_path,
            status=source_qc_status,
            detail=source_qc_detail,
            blocking_effect=source_qc_blocking,
            next_action=source_qc_next_action,
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id="Zenodo_record_specific_payload_release_8194033",
            artifact_role="Zenodo_MUCC_v1_release_pin_and_concept_version_audit",
            source_url=ZENODO_PAYLOAD_API_URL,
            path=zenodo_payload_record,
            status=zenodo_status,
            detail=zenodo_detail,
            blocking_effect=zenodo_blocking,
            next_action=zenodo_next_action,
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id=f"{ARTICLE_DOI}.SuF2",
            artifact_role="published_Table_S1_to_S13_accession_spreadsheet",
            source_url=f"https://journals.asm.org/doi/{ARTICLE_DOI}",
            path=table_s1,
            status=xlsx_status,
            detail=xlsx_detail,
            blocking_effect=needs_metadata if xlsx_status != "parseable_xlsx_container" else "none",
            next_action=(
                "request a corrected publisher or author-provided XLSX/Table S4-equivalent, verify its ZIP central directory and checksum, then materialize accession and sample crosswalks"
                if xlsx_status != "parseable_xlsx_container"
                else "parse Tables S1-S13 into accession and sample crosswalk candidates; do not promote unresolved mappings"
            ),
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id="KBase_MUCC_v1_workspace_147022_public_API",
            artifact_role="published_MAG_collection_public_roster_and_supplementary_taxonomy",
            source_url=KBASE_WORKSPACE_API_URL,
            path=kbase_summary_path,
            status=kbase_status,
            detail=kbase_detail,
            blocking_effect=kbase_blocking,
            next_action=kbase_next_action,
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id="NCBI_SRA_SRP456134",
            artifact_role="NCBI_SRA_expression_label_package_and_collection_date_crosswalk",
            source_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            path=sra_crosswalk_path,
            status=sra_status,
            detail=sra_detail,
            blocking_effect=sra_blocking,
            next_action=sra_next_action,
            retrieved_utc=retrieved_utc,
        ),
        artifact_row(
            source_id="JGI_Data_Portal_award_504205",
            artifact_role="JGI_Data_Portal_expression_label_catalog_crosswalk",
            source_url="https://files.jgi.doe.gov/search/",
            path=jgi_data_portal_crosswalk_path,
            status=jgi_data_portal_status,
            detail=jgi_data_portal_detail,
            blocking_effect=jgi_data_portal_blocking,
            next_action=jgi_data_portal_next_action,
            retrieved_utc=retrieved_utc,
        ),
    ]
    write_tsv(output_tsv, rows)
    summary = {
        "lane_id": "mucc_v1_owc_wetland",
        "ledger": str(output_tsv),
        "zenodo_archive_roster_entries": archive_count,
        "zenodo_archive_roster_status": archive_status,
        "Zenodo_direct_source_QC_status": source_qc_status,
        "Zenodo_direct_source_QC_counts": source_qc_counts,
        "Zenodo_release_pin_status": zenodo_status,
        "Zenodo_payload_release_record": str(zenodo_payload_record),
        "Zenodo_concept_record": str(zenodo_concept_record),
        "supplemental_methods_pdf_status": (
            "parseable_pdf_methods_design_confirmed"
            if methods_confirmed
            else "parseable_pdf_methods_design_unconfirmed"
        ),
        "published_table_s1_status": xlsx_status,
        "published_table_s1_payload_bytes": (
            table_s1.stat().st_size if table_s1.is_file() else 0
        ),
        "published_table_s1_size_matches_publisher_listing": publisher_size_matches,
        "published_table_s1_publisher_listing_bytes": PUBLISHER_LISTED_SUPPLEMENT_TABLE_BYTES,
        "kbase_public_catalog_status": kbase_status,
        "kbase_public_catalog_summary": str(kbase_summary_path),
        "NCBI_SRA_crosswalk_status": sra_status,
        "NCBI_SRA_crosswalk_counts": sra_counts,
        "JGI_Data_Portal_catalog_crosswalk_status": jgi_data_portal_status,
        "JGI_Data_Portal_catalog_crosswalk_counts": jgi_data_portal_counts,
        "methods_design_context_rows": len(context_rows),
        "methods_design_context": str(methods_context_path),
        "methods_design_reconciliation": str(methods_reconciliation_path),
        "exact_sample_crosswalk_status": (
            "partial_exact_SRA_package_identity_and_collection_dates_depth_environment_flux_unresolved"
        ),
        "mag_denominator_reconciliation_status": (
            source_qc_status
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
