#!/usr/bin/env python3
"""Recover exact NCBI SRA identities and source metadata for MUCC expression labels.

The public JGI portal's per-project SRA label is not reliably searchable as a
study accession, while each source expression label resolves directly in NCBI
SRA.  This builder therefore uses an exact-title/source-material match against
the SRA package, preserves every non-match or ambiguity, and stages collection
date and location only when NCBI reports them.  It never invents depth,
geochemistry, abundance, or a sequence-to-flux association.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE = Path("results/functional_metagenomics/mucc_v1_owc_wetland_20260626")
LANE_ID = "mucc_v1_owc_wetland"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
CLAIM_BOUNDARY = (
    "NCBI SRA package matches establish exact source-expression-label identity and reported "
    "NCBI sample metadata only. A reported collection date or location is not a collection "
    "datetime, measured depth, environmental/flux pairing, ecological association, measured "
    "methane flux, final MRV score/A-E tier, crediting claim, or source-independent transfer result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=BASE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional deterministic smoke-test limit; zero resolves every source expression label.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based source-expression row offset, for resumable bounded NCBI recovery batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of source labels to query in each NCBI SRA OR query (default: 20).",
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


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


@dataclass
class RateLimiter:
    minimum_interval_seconds: float = 0.34
    previous_request_at: float = 0.0

    def wait(self) -> None:
        remaining = self.minimum_interval_seconds - (time.monotonic() - self.previous_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self.previous_request_at = time.monotonic()


def fetch_bytes(url: str, destination: Path, force: bool, limiter: RateLimiter) -> bytes:
    if destination.is_file() and destination.stat().st_size and not force:
        return destination.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    limiter.wait()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MethaNet MUCC NCBI SRA sample-metadata recovery"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"empty NCBI response from {url}")
    destination.write_bytes(payload)
    return payload


def candidate_labels(source_label: str) -> list[tuple[str, str]]:
    candidates = [(source_label, "exact_NCBI_SRA_label")]
    if source_label.startswith("July_"):
        candidates.append(("Jul_" + source_label.removeprefix("July_"), "deterministic_July_to_Jul"))
    if source_label.startswith("Jul_"):
        candidates.append(("July_" + source_label.removeprefix("Jul_"), "deterministic_Jul_to_July"))
    return candidates


def identifier(parent: ET.Element | None, namespace: str) -> str:
    if parent is None:
        return ""
    for item in parent.findall(".//EXTERNAL_ID"):
        if item.get("namespace") == namespace:
            return (item.text or "").strip()
    return ""


def text(element: ET.Element | None, path: str) -> str:
    if element is None:
        return ""
    return (element.findtext(path) or "").strip()


def sample_attributes(sample: ET.Element | None) -> dict[str, str]:
    if sample is None:
        return {}
    return {
        text(attribute, "TAG"): text(attribute, "VALUE")
        for attribute in sample.findall(".//SAMPLE_ATTRIBUTE")
        if text(attribute, "TAG")
    }


def title_matches_label(value: str, label: str) -> bool:
    return bool(re.search(rf"(?: - )?{re.escape(label)}$", value.strip()))


def package_has_exact_owc_identity(package: ET.Element, label: str) -> bool:
    experiment = package.find("EXPERIMENT")
    sample = package.find("SAMPLE")
    experiment_title = text(experiment, "TITLE")
    sample_title = text(sample, "TITLE")
    attributes = sample_attributes(sample)
    exact_label = (
        title_matches_label(experiment_title, label)
        or title_matches_label(sample_title, label)
        or attributes.get("source_material_id", "") == label
    )
    context = " ".join(
        [
            experiment_title.lower(),
            sample_title.lower(),
            attributes.get("geo_loc_name", "").lower(),
            attributes.get("isolation_source", "").lower(),
        ]
    )
    expected_entity = "old woman creek" in context or (
        attributes.get("source_material_id", "") == label
        and "ohio" in context
        and "wetland" in context
    )
    return exact_label and expected_entity


def package_record(package: ET.Element) -> dict[str, str]:
    experiment = package.find("EXPERIMENT")
    sample = package.find("SAMPLE")
    study = package.find("STUDY") or (experiment.find("STUDY_REF") if experiment is not None else None)
    run_accessions = [item.get("accession", "") for item in package.findall(".//RUN_SET/RUN")]
    attributes = sample_attributes(sample)
    collection_date = attributes.get("collection_date", "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", collection_date):
        collection_status = "exact_collection_date_from_NCBI_SRA_sample_attributes"
    elif collection_date:
        collection_status = "partial_collection_date_from_NCBI_SRA_sample_attributes"
    else:
        collection_status = "collection_date_not_reported_by_NCBI_SRA_sample_attributes"
    depth_value = next(
        (
            value
            for key, value in attributes.items()
            if key.lower() in {"depth", "depth_cm", "sediment_depth", "sample_depth"}
        ),
        "",
    )
    return {
        "sra_study_accession": study.get("accession", "") if study is not None else "",
        "sra_bioproject_accession": identifier(study, "BioProject"),
        "sra_experiment_accession": experiment.get("accession", "") if experiment is not None else "",
        "sra_run_accessions": ";".join(accession for accession in run_accessions if accession),
        "sra_biosample_accession": identifier(sample, "BioSample"),
        "sra_sample_accession": sample.get("accession", "") if sample is not None else "",
        "sra_experiment_title": text(experiment, "TITLE"),
        "sra_sample_title": text(sample, "TITLE"),
        "sra_library_strategy": text(experiment, ".//LIBRARY_STRATEGY"),
        "sra_library_source": text(experiment, ".//LIBRARY_SOURCE"),
        "sra_library_layout": (
            "paired"
            if experiment is not None and experiment.find(".//LIBRARY_LAYOUT/PAIRED") is not None
            else "single"
            if experiment is not None and experiment.find(".//LIBRARY_LAYOUT/SINGLE") is not None
            else "not_reported"
        ),
        "sra_instrument_model": text(experiment, ".//INSTRUMENT_MODEL"),
        "sra_run_publication_datetime": (
            package.find(".//RUN_SET/RUN").get("published", "")
            if package.find(".//RUN_SET/RUN") is not None
            else ""
        ),
        "sra_collection_date": collection_date,
        "sra_collection_date_status": collection_status,
        "sra_geo_loc_name": attributes.get("geo_loc_name", ""),
        "sra_lat_lon": attributes.get("lat_lon", ""),
        "sra_isolation_source": attributes.get("isolation_source", ""),
        "sra_gold_ecosystem_classification": attributes.get("GOLD Ecosystem Classification", ""),
        "sra_reported_depth_value": depth_value,
        "sra_depth_cm_join_status": (
            "reported_depth_value_requires_unit_normalization"
            if depth_value
            else "not_reported_by_NCBI_SRA_sample_attributes"
        ),
    }


def select_exact_package(xml_payload: bytes, source_label: str) -> tuple[dict[str, str] | None, str, str]:
    root = ET.fromstring(xml_payload)
    for candidate_label, mapping_method in candidate_labels(source_label):
        matches = [
            package
            for package in root.findall(".//EXPERIMENT_PACKAGE")
            if package_has_exact_owc_identity(package, candidate_label)
        ]
        if len(matches) == 1:
            return package_record(matches[0]), mapping_method, "exact_source_label_to_NCBI_SRA_package"
        if len(matches) > 1:
            return None, mapping_method, "ambiguous_multiple_exact_NCBI_SRA_packages"
    return None, "", "no_exact_NCBI_SRA_package"


def empty_row(sample: pd.Series) -> dict[str, str]:
    return {
        "lane_id": LANE_ID,
        "sample_id": str(sample["sample_id"]),
        "source_sample_column": str(sample["source_sample_column"]),
        "sra_matched_source_label": "",
        "sra_label_mapping_method": "",
        "sra_search_status": "unresolved",
        "sra_query_uids": "",
        "sra_package_selection_status": "",
        "sra_study_accession": "",
        "sra_bioproject_accession": "",
        "sra_experiment_accession": "",
        "sra_run_accessions": "",
        "sra_biosample_accession": "",
        "sra_sample_accession": "",
        "sra_experiment_title": "",
        "sra_sample_title": "",
        "sra_library_strategy": "",
        "sra_library_source": "",
        "sra_library_layout": "",
        "sra_instrument_model": "",
        "sra_run_publication_datetime": "",
        "sra_collection_date": "",
        "sra_collection_date_status": "collection_date_not_recovered",
        "sra_geo_loc_name": "",
        "sra_lat_lon": "",
        "sra_isolation_source": "",
        "sra_gold_ecosystem_classification": "",
        "sra_reported_depth_value": "",
        "sra_depth_cm_join_status": "unresolved_no_exact_NCBI_SRA_package",
        "sra_sample_identity_status": "unresolved_no_exact_NCBI_SRA_package",
        "environment_flux_join_status": "unlinked_pending_exact_spatiotemporal_crosswalk",
        "next_validation_action": (
            "recover an authoritative sample/depth/environment/flux mapping; do not infer it from "
            "the expression label or collection date"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def search_url(label: str) -> str:
    return NCBI_ESEARCH_URL + "?" + urllib.parse.urlencode(
        {"db": "sra", "term": f"{label}[All Fields]", "retmode": "json", "retmax": 20}
    )


def fetch_url(ids: list[str]) -> str:
    return NCBI_EFETCH_URL + "?" + urllib.parse.urlencode({"db": "sra", "id": ",".join(ids)})


def batch_search_url(labels: list[str]) -> str:
    term = " OR ".join(f"{label}[All Fields]" for label in labels)
    return NCBI_ESEARCH_URL + "?" + urllib.parse.urlencode(
        {"db": "sra", "term": f"({term})", "retmode": "json", "retmax": 500}
    )


def source_manifest_row(path: Path, role: str, url: str) -> dict[str, str]:
    return {
        "lane_id": LANE_ID,
        "source_id": "NCBI_SRA_MUCC_expression_label_query",
        "artifact_role": role,
        "source_url": url,
        "local_path": str(path),
        "bytes": str(path.stat().st_size),
        "sha256": sha256(path),
        "validation_status": (
            "retrieved_and_json_parseable"
            if role == "NCBI_SRA_esearch_response"
            else "retrieved_and_xml_parseable"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def recover_batch(
    samples: pd.DataFrame,
    source_dir: Path,
    batch_id: str,
    force: bool,
    limiter: RateLimiter,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    query_labels = list(
        dict.fromkeys(
            candidate_label
            for source_label in samples["source_sample_column"].astype(str)
            for candidate_label, _ in candidate_labels(source_label)
        )
    )
    url = batch_search_url(query_labels)
    search_path = source_dir / "esearch" / f"{batch_id}.json"
    search = json.loads(fetch_bytes(url, search_path, force, limiter))
    ids = [str(value) for value in search["esearchresult"].get("idlist", [])]
    manifest_rows = [source_manifest_row(search_path, "NCBI_SRA_esearch_response", url)]
    xml_payload = b"<EXPERIMENT_PACKAGE_SET/>"
    if ids:
        package_url = fetch_url(ids)
        package_path = source_dir / "efetch" / f"{batch_id}.xml"
        xml_payload = fetch_bytes(package_url, package_path, force, limiter)
        manifest_rows.append(
            source_manifest_row(package_path, "NCBI_SRA_experiment_package_response", package_url)
        )
    rows: list[dict[str, str]] = []
    for _, sample in samples.iterrows():
        source_label = str(sample["source_sample_column"])
        row = empty_row(sample)
        record, selection_method, selection_status = select_exact_package(xml_payload, source_label)
        matched_label = next(
            (
                label
                for label, method in candidate_labels(source_label)
                if method == selection_method
            ),
            "",
        )
        if selection_status == "exact_source_label_to_NCBI_SRA_package":
            assert record is not None
            row.update(record)
            row.update(
                {
                    "sra_matched_source_label": matched_label,
                    "sra_label_mapping_method": selection_method,
                    "sra_search_status": "NCBI_SRA_query_returned_exact_entity_validated_package",
                    "sra_query_uids": ";".join(ids),
                    "sra_package_selection_status": selection_status,
                    "sra_sample_identity_status": selection_status,
                }
            )
        elif selection_status == "ambiguous_multiple_exact_NCBI_SRA_packages":
            row.update(
                {
                    "sra_matched_source_label": matched_label,
                    "sra_label_mapping_method": selection_method,
                    "sra_search_status": "NCBI_SRA_query_returned_ambiguous_entity_validated_packages",
                    "sra_query_uids": ";".join(ids),
                    "sra_package_selection_status": selection_status,
                    "sra_sample_identity_status": selection_status,
                }
            )
        else:
            row.update(
                {
                    "sra_search_status": "no_exact_entity_validated_NCBI_SRA_package",
                    "sra_query_uids": ";".join(ids),
                    "sra_package_selection_status": "no_exact_NCBI_SRA_package",
                }
            )
        rows.append(row)
    return rows, manifest_rows


def recover_sample(
    sample: pd.Series, source_dir: Path, force: bool, limiter: RateLimiter
) -> tuple[dict[str, str], list[dict[str, str]]]:
    source_label = str(sample["source_sample_column"])
    row = empty_row(sample)
    manifest_rows: list[dict[str, str]] = []
    for candidate_label, mapping_method in candidate_labels(source_label):
        search_path = source_dir / "esearch" / f"{candidate_label}.json"
        payload = fetch_bytes(search_url(candidate_label), search_path, force, limiter)
        search = json.loads(payload)
        ids = [str(value) for value in search["esearchresult"].get("idlist", [])]
        manifest_rows.append(
            {
                "lane_id": LANE_ID,
                "source_id": "NCBI_SRA_MUCC_expression_label_query",
                "artifact_role": "NCBI_SRA_esearch_response",
                "source_url": search_url(candidate_label),
                "local_path": str(search_path),
                "bytes": str(search_path.stat().st_size),
                "sha256": sha256(search_path),
                "validation_status": "retrieved_and_json_parseable",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        if not ids:
            continue
        fetch_path = source_dir / "efetch" / f"{candidate_label}.xml"
        xml_payload = fetch_bytes(fetch_url(ids), fetch_path, force, limiter)
        manifest_rows.append(
            {
                "lane_id": LANE_ID,
                "source_id": "NCBI_SRA_MUCC_expression_label_query",
                "artifact_role": "NCBI_SRA_experiment_package_response",
                "source_url": fetch_url(ids),
                "local_path": str(fetch_path),
                "bytes": str(fetch_path.stat().st_size),
                "sha256": sha256(fetch_path),
                "validation_status": "retrieved_and_xml_parseable",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        record, selection_method, selection_status = select_exact_package(xml_payload, source_label)
        if selection_status == "exact_source_label_to_NCBI_SRA_package":
            assert record is not None
            row.update(record)
            row.update(
                {
                    "sra_matched_source_label": candidate_label,
                    "sra_label_mapping_method": selection_method,
                    "sra_search_status": "NCBI_SRA_query_returned_exact_entity_validated_package",
                    "sra_query_uids": ";".join(ids),
                    "sra_package_selection_status": selection_status,
                    "sra_sample_identity_status": selection_status,
                }
            )
            return row, manifest_rows
        if selection_status == "ambiguous_multiple_exact_NCBI_SRA_packages":
            row.update(
                {
                    "sra_matched_source_label": candidate_label,
                    "sra_label_mapping_method": selection_method,
                    "sra_search_status": "NCBI_SRA_query_returned_ambiguous_entity_validated_packages",
                    "sra_query_uids": ";".join(ids),
                    "sra_package_selection_status": selection_status,
                    "sra_sample_identity_status": selection_status,
                }
            )
            return row, manifest_rows
    row.update(
        {
            "sra_search_status": "no_exact_entity_validated_NCBI_SRA_package",
            "sra_package_selection_status": "no_exact_NCBI_SRA_package",
        }
    )
    return row, manifest_rows


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve(repo_root, args.run_dir)
    samples = pd.read_csv(
        run_dir / "environmental_metadata/mucc_v1_sample_columns_scaffold.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_samples:
        samples = samples.iloc[args.start_index : args.start_index + args.max_samples].copy()
    elif args.start_index:
        samples = samples.iloc[args.start_index:].copy()
    source_dir = run_dir / "source_audit/ncbi_sra_sample_crosswalk"
    limiter = RateLimiter()
    recovered_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for batch_index, start in enumerate(range(0, len(samples), args.batch_size), start=1):
        batch = samples.iloc[start : start + args.batch_size].copy()
        batch_id = f"batch_{args.start_index + start:03d}_{args.start_index + start + len(batch) - 1:03d}"
        try:
            recovered, sources = recover_batch(
                batch, source_dir, batch_id, args.force_download, limiter
            )
            recovered_rows.extend(recovered)
            manifest_rows.extend(sources)
        except Exception as error:
            for _, sample in batch.iterrows():
                row = empty_row(sample)
                row.update(
                    {
                        "sra_search_status": "NCBI_SRA_recovery_error_preserved",
                        "sra_package_selection_status": "NCBI_SRA_recovery_error_preserved",
                        "sra_sample_identity_status": "unresolved_NCBI_SRA_recovery_error",
                    }
                )
                recovered_rows.append(row)
                errors.append(
                    {
                        "source_sample_column": str(sample["source_sample_column"]),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        print(
            json.dumps(
                {
                    "event": "batch_recovery_progress",
                    "batch_index": batch_index,
                    "batch_rows": len(batch),
                    "exact_package_matches": int(
                        sum(
                            row["sra_sample_identity_status"]
                            == "exact_source_label_to_NCBI_SRA_package"
                            for row in recovered_rows[-len(batch) :]
                        )
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    crosswalk = pd.DataFrame(recovered_rows)
    if crosswalk["sample_id"].duplicated().any() or len(crosswalk) != len(samples):
        raise ValueError("NCBI SRA crosswalk must preserve each requested source expression sample once")
    environmental_dir = run_dir / "environmental_metadata"
    output = environmental_dir / "link_mucc_v1_sequence_sra_sample.tsv"
    write_tsv(output, crosswalk)
    write_tsv(source_dir / "source_file_manifest.tsv", pd.DataFrame(manifest_rows))
    (source_dir / "recovery_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    exact = crosswalk["sra_sample_identity_status"].eq(
        "exact_source_label_to_NCBI_SRA_package"
    )
    exact_dates = crosswalk["sra_collection_date_status"].eq(
        "exact_collection_date_from_NCBI_SRA_sample_attributes"
    )
    summary = {
        "lane_id": LANE_ID,
        "requested_expression_samples": int(len(crosswalk)),
        "exact_NCBI_SRA_package_matches": int(exact.sum()),
        "exact_NCBI_collection_dates": int(exact_dates.sum()),
        "reported_depth_values": int(crosswalk["sra_reported_depth_value"].ne("").sum()),
        "unresolved_or_ambiguous_samples": int((~exact).sum()),
        "recovery_errors": len(errors),
        "crosswalk": str(output),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (source_dir / "crosswalk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
