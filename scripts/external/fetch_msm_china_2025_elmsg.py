#!/usr/bin/env python3
"""Fetch and validate the Pan et al. 2025 MSM mangrove MAG package.

This helper is intentionally conservative. It records every attempted accession,
rejects HTML/maintenance pages masquerading as downloads, extracts only expected
genome/proteome/annotation files, and keeps not-yet-resolved eLMSG accessions as
explicit status rows rather than silently dropping them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_SUFFIXES = (
    ".fna",
    ".fa",
    ".fasta",
    ".faa",
    ".gff",
    ".gff3",
    ".ko",
    ".cog",
    ".pfam",
    ".ec",
    ".cyc",
    ".rRNA",
    ".tRNA",
    ".frn",
)

DEFAULT_URL_TEMPLATES = (
    "https://www.biosino.org/elmsg/record/{record_id}",
    "https://www.biosino.org/elmsg/download/{record_id}",
    "https://www.biosino.org/elmsg/download?recordId={record_id}",
    "https://www.biosino.org/elmsg/download?genomeId={accession}",
    "https://www.biosino.org/elmsg/download?accession={accession}",
    "https://www.biosino.org/elmsg/downloadGenome?genomeId={accession}",
)


@dataclass
class FetchResult:
    accession: str
    record_id: str
    status: str
    url: str
    record_url: str
    bundle_path: Path
    local_fna_path: str
    local_faa_path: str
    sha256: str
    bytes_downloaded: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch eLMSG MSM accession bundles and build a readiness manifest."
    )
    parser.add_argument(
        "--manifest",
        default="data/external/msm_china_2025/manifests/elmsg_accession_range_candidates.tsv",
        help="Candidate accession manifest generated from the published eLMSG range.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/external/msm_china_2025",
        help="Dataset staging directory.",
    )
    parser.add_argument(
        "--url-template",
        action="append",
        help=(
            "Download URL template containing {accession}. Can be supplied more "
            "than once. Defaults to known eLMSG route probes."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of accession rows to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the status manifest without making network requests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download bundles even if a local bundle already exists.",
    )
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[:limit] if limit else rows


def looks_like_html(path: Path) -> bool:
    sample = path.read_bytes()[:512].lower()
    return (
        b"<html" in sample
        or b"<!doctype html" in sample
        or (b"we" in sample and b"back shortly" in sample)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out_path: Path) -> tuple[int, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "MethaNet-MSM-ingestion/1.0 (+https://github.com/)",
            "Accept": "application/gzip,application/x-tar,application/octet-stream,*/*",
        },
    )
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        with out_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return out_path.stat().st_size, content_type


def is_tar_or_gzip(path: Path) -> bool:
    if tarfile.is_tarfile(path):
        return True
    try:
        with gzip.open(path, "rb") as handle:
            handle.peek(1)
        return True
    except OSError:
        return False


def normalize_filename(accession: str, member_name: str) -> str:
    basename = Path(member_name).name
    lower = basename.lower()
    for suffix in (".fna", ".fa", ".fasta"):
        if lower.endswith(suffix):
            return f"{accession}.fna"
    if lower.endswith(".faa"):
        return f"{accession}.faa"
    return f"{accession}.{basename}"


def safe_extract_expected(bundle_path: Path, accession: str, dataset_dir: Path) -> tuple[str, str, list[str]]:
    genomes_dir = dataset_dir / "genomes_fna"
    proteomes_dir = dataset_dir / "proteomes_faa"
    annotation_dir = dataset_dir / "annotations"
    genomes_dir.mkdir(parents=True, exist_ok=True)
    proteomes_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    fna_path = ""
    faa_path = ""

    with tarfile.open(bundle_path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            member_name = member.name
            if not member_name.endswith(EXPECTED_SUFFIXES):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue

            out_name = normalize_filename(accession, member_name)
            lower = out_name.lower()
            if lower.endswith(".fna"):
                out_path = genomes_dir / out_name
                fna_path = str(out_path)
            elif lower.endswith(".faa"):
                out_path = proteomes_dir / out_name
                faa_path = str(out_path)
            else:
                out_path = annotation_dir / out_name

            with out_path.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted.append(str(out_path))

    return fna_path, faa_path, extracted


def fetch_one(row: dict[str, str], dataset_dir: Path, templates: Iterable[str], dry_run: bool, force: bool) -> FetchResult:
    accession = row["elmsg_genome_accession"]
    record_id = row.get("derived_elmsg_record_id", "")
    bundle_path = dataset_dir / "raw_downloads" / f"{accession}.tar.gz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return FetchResult(
            accession,
            record_id,
            "dry_run_not_downloaded",
            "",
            row.get("derived_record_url", ""),
            bundle_path,
            "",
            "",
            "",
            0,
            "",
        )

    if bundle_path.exists() and not force:
        if looks_like_html(bundle_path):
            return FetchResult(accession, record_id, "blocked_html_shell", "", row.get("derived_record_url", ""), bundle_path, "", "", "", bundle_path.stat().st_size, "Existing file is HTML, not a data bundle.")
        if not is_tar_or_gzip(bundle_path):
            return FetchResult(accession, record_id, "blocked_unknown_format", "", row.get("derived_record_url", ""), bundle_path, "", "", "", bundle_path.stat().st_size, "Existing file is not a tar/gzip bundle.")
        fna_path, faa_path, extracted = safe_extract_expected(bundle_path, accession, dataset_dir)
        status = "ready_for_functional_annotation_and_embedding" if fna_path and faa_path else "downloaded_but_incomplete"
        return FetchResult(accession, record_id, status, "local_existing", row.get("derived_record_url", ""), bundle_path, fna_path, faa_path, sha256_file(bundle_path), bundle_path.stat().st_size, f"Extracted {len(extracted)} expected files.")

    last_note = ""
    for template in templates:
        if "{record_id}" in template and not record_id:
            last_note = "URL template requires {record_id}, but manifest row lacks derived_elmsg_record_id."
            continue
        url = template.format(accession=accession, record_id=record_id)
        tmp_path = bundle_path.with_suffix(".tmp")
        try:
            bytes_downloaded, content_type = download(url, tmp_path)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_note = f"{type(exc).__name__}: {exc}"
            if tmp_path.exists():
                tmp_path.unlink()
            continue

        tmp_path.replace(bundle_path)
        if looks_like_html(bundle_path):
            last_note = f"Downloaded HTML shell from {url}; content-type={content_type}"
            continue
        if not is_tar_or_gzip(bundle_path):
            last_note = f"Downloaded non-tar/non-gzip payload from {url}; content-type={content_type}"
            continue

        fna_path, faa_path, extracted = safe_extract_expected(bundle_path, accession, dataset_dir)
        status = "ready_for_functional_annotation_and_embedding" if fna_path and faa_path else "downloaded_but_incomplete"
        return FetchResult(accession, record_id, status, url, row.get("derived_record_url", ""), bundle_path, fna_path, faa_path, sha256_file(bundle_path), bytes_downloaded, f"Extracted {len(extracted)} expected files.")

    return FetchResult(accession, record_id, "not_downloaded", "", row.get("derived_record_url", ""), bundle_path, "", "", "", 0, last_note)


def write_status(results: list[FetchResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "accession",
        "record_id",
        "status",
        "url",
        "record_url",
        "bundle_path",
        "local_fna_path",
        "local_faa_path",
        "sha256",
        "bytes_downloaded",
        "note",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "accession": result.accession,
                    "record_id": result.record_id,
                    "status": result.status,
                    "url": result.url,
                    "record_url": result.record_url,
                    "bundle_path": str(result.bundle_path),
                    "local_fna_path": result.local_fna_path,
                    "local_faa_path": result.local_faa_path,
                    "sha256": result.sha256,
                    "bytes_downloaded": result.bytes_downloaded,
                    "note": result.note,
                }
            )


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    dataset_dir = Path(args.dataset_dir)
    templates = args.url_template or list(DEFAULT_URL_TEMPLATES)

    rows = read_manifest(manifest_path, args.limit)
    results = [fetch_one(row, dataset_dir, templates, args.dry_run, args.force) for row in rows]
    out_path = dataset_dir / "manifests" / "functional_embedding_ready_manifest.tsv"
    write_status(results, out_path)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    for status, count in sorted(counts.items()):
        print(f"{status}\t{count}")
    print(f"wrote\t{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
