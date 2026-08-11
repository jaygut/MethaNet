#!/usr/bin/env python3
"""Run METABOLIC-G in an isolated working directory and enforce its XLSX contract.

METABOLIC-G creates ``METABOLIC_result.xlsx`` in the process working directory
before moving it into the requested output directory.  Sharing a working
directory across array tasks therefore permits one task to move another task's
workbook.  This wrapper gives every invocation a private working directory and
rejects missing or structurally incomplete workbooks even when METABOLIC exits
zero.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


class OutputContractError(RuntimeError):
    """Raised when METABOLIC reports success without its required workbook."""


IDENTITY_HEADER_SUFFIXES = (
    ".Hmm.presence",
    ".Hit.numbers",
    ".Hits",
    ".Function.presence",
    ".Module.presence",
    ".Module.step.presence",
)


def validate_workbook(path: Path, expected_mag_id: str, minimum_worksheets: int = 6) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        tables_dir = path.parent / "METABOLIC_result_each_spreadsheet"
        table_count = sum(
            1 for candidate in tables_dir.glob("*.tsv") if candidate.is_file() and candidate.stat().st_size > 0
        ) if tables_dir.is_dir() else 0
        raise OutputContractError(
            f"METABOLIC output contract failed: workbook is missing or empty: {path}; "
            f"nonempty intermediate worksheet TSVs={table_count}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            shared_strings = archive.read("xl/sharedStrings.xml")
    except zipfile.BadZipFile as exc:
        raise OutputContractError(f"METABOLIC output contract failed: invalid XLSX archive: {path}") from exc
    worksheets = [name for name in names if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")]
    if "xl/workbook.xml" not in names or len(worksheets) < minimum_worksheets:
        raise OutputContractError(
            f"METABOLIC output contract failed: expected at least {minimum_worksheets} worksheets, "
            f"observed {len(worksheets)} in {path}"
        )
    try:
        root = ET.fromstring(shared_strings)
    except ET.ParseError as exc:
        raise OutputContractError(f"METABOLIC output contract failed: malformed shared strings in {path}") from exc
    observed_mag_ids: set[str] = set()
    for element in root.iter():
        if not element.tag.endswith("}t") and element.tag != "t":
            continue
        value = element.text or ""
        for suffix in IDENTITY_HEADER_SUFFIXES:
            if value.endswith(suffix):
                observed_mag_ids.add(value[: -len(suffix)])
                break
    accepted_mag_ids = {expected_mag_id}
    if expected_mag_id[:1].isdigit():
        accepted_mag_ids.add(f"X{expected_mag_id}")
    if len(observed_mag_ids) != 1 or not observed_mag_ids.issubset(accepted_mag_ids):
        raise OutputContractError(
            "METABOLIC output contract failed: workbook MAG identity mismatch; "
            f"expected={expected_mag_id}; observed={sorted(observed_mag_ids)}; path={path}"
        )


def run_metabolic(
    *,
    perl: str,
    metabolic_script: Path,
    input_genomes: Path,
    output_dir: Path,
    threads: int,
    expected_mag_id: str,
    prodigal_mode: str = "meta",
    kofam_db: str = "full",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = output_dir.parent / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="metabolic_work_", dir=tmp_root))
    command = [
        perl,
        str(metabolic_script),
        "-in-gn",
        str(input_genomes),
        "-o",
        str(output_dir),
        "-t",
        str(threads),
        "-p",
        prodigal_mode,
        "-kofam-db",
        kofam_db,
    ]
    completed = subprocess.run(command, cwd=work_dir, check=False)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    workbook = output_dir / "METABOLIC_result.xlsx"
    validate_workbook(workbook, expected_mag_id)
    return workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perl", default="perl")
    parser.add_argument("--metabolic-script", type=Path, required=True)
    parser.add_argument("--input-genomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--expected-mag-id", required=True)
    parser.add_argument("--prodigal-mode", default="meta")
    parser.add_argument("--kofam-db", default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workbook = run_metabolic(
        perl=args.perl,
        metabolic_script=args.metabolic_script.resolve(),
        input_genomes=args.input_genomes.resolve(),
        output_dir=args.output_dir.resolve(),
        threads=args.threads,
        expected_mag_id=args.expected_mag_id,
        prodigal_mode=args.prodigal_mode,
        kofam_db=args.kofam_db,
    )
    print(workbook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
