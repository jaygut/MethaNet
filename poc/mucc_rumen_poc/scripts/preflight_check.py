#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


REQUIRED_COLUMNS = {
    "sample_id",
    "domain",
    "ecosystem",
    "data_source",
    "accession",
    "has_flux",
    "flux_value",
    "flux_unit",
    "latitude",
    "longitude",
    "depth_cm",
    "salinity_ppt",
    "temperature_c",
    "collection_date",
    "notes",
}


DEFAULT_MARKERS = [
    "mcrA",
    "mcrB",
    "mcrG",
    "pmoA",
    "mmoX",
    "dsrA",
    "dsrB",
    "mtaB",
    "mttB",
    "mtbA",
    "nifH",
    "cbbL",
]


TBD_VALUES = {"", "NA", "N/A", "TBD", "NONE", "NULL"}


REQUIRED_TOOLS_BY_STAGE = {
    "marker_annotator": {"prodigal", "hmmsearch", "mmseqs"},
}


def load_yaml(path: Path) -> dict:
    if yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_value(value: str) -> str:
    return value.strip()


def read_manifest(path: Path) -> tuple[list[dict], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []

    if not path.exists():
        errors.append(f"manifest missing: {path}")
        return rows, errors, warnings

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            errors.append("manifest is empty or missing header row")
            return rows, errors, warnings

        missing_cols = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_cols:
            errors.append(
                "manifest missing columns: " + ", ".join(sorted(missing_cols))
            )

        for row in reader:
            rows.append({k: normalize_value(v or "") for k, v in row.items()})

    sample_ids = [row.get("sample_id", "") for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        errors.append("manifest contains blank sample_id values")

    seen = set()
    dupes = sorted({sid for sid in sample_ids if sid in seen or seen.add(sid)})
    if dupes:
        errors.append("manifest contains duplicate sample_id values: " + ", ".join(dupes))

    domains = {row.get("domain", "") for row in rows}
    if "rumen" not in domains or "coastal" not in domains:
        warnings.append(
            "manifest should include both rumen and coastal domains for pilot split"
        )

    return rows, errors, warnings


def check_accessions(rows: list[dict], require_accessions: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for row in rows:
        value = row.get("accession", "")
        normalized = value.upper()
        if normalized in TBD_VALUES:
            message = f"accession placeholder for sample {row.get('sample_id', '')}"
            if require_accessions:
                errors.append(message)
            else:
                warnings.append(message)
    return errors, warnings


def check_fastas(
    sample_ids: list[str],
    assemblies_dir: Path,
    required: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not required:
        return errors, warnings

    if not assemblies_dir.exists():
        errors.append(f"assemblies dir missing: {assemblies_dir}")
        return errors, warnings

    for sample_id in sample_ids:
        expected = assemblies_dir / f"{sample_id}.fasta"
        if expected.exists():
            continue
        alt = [assemblies_dir / f"{sample_id}.fa", assemblies_dir / f"{sample_id}.fna"]
        alt_found = next((path for path in alt if path.exists()), None)
        if alt_found:
            errors.append(
                f"missing {expected}; found {alt_found} (rename to .fasta)"
            )
        else:
            errors.append(f"missing assembly FASTA: {expected}")
    return errors, warnings


def check_hmms(hmm_dir: Path, marker_db: Path, marker_paths: list[Path]) -> list[str]:
    errors: list[str] = []
    if not hmm_dir.exists():
        errors.append(f"hmm_dir missing: {hmm_dir}")
        return errors

    for marker in marker_paths:
        if not marker.exists():
            errors.append(f"missing HMM: {marker}")

    if not marker_db.exists():
        errors.append(f"missing marker_db: {marker_db}")
    return errors


def check_config_samples(
    config: dict,
    manifest_ids: set[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    def _check_list(key: str, warn_if_subset: bool = False) -> None:
        values = config.get(key, []) or []
        if not values:
            return
        value_set = {str(value) for value in values}
        missing_in_manifest = sorted(value_set - manifest_ids)
        if missing_in_manifest:
            errors.append(
                f"{key} includes unknown sample_id values: "
                + ", ".join(missing_in_manifest)
            )
        if warn_if_subset and value_set != manifest_ids:
            warnings.append(
                f"{key} does not cover all manifest sample_ids "
                f"({len(value_set)} of {len(manifest_ids)})"
            )

    _check_list("assembly_samples", warn_if_subset=True)
    _check_list("marker_samples", warn_if_subset=False)
    _check_list("embedding_samples", warn_if_subset=False)
    _check_list("dna_embedding_samples", warn_if_subset=False)

    return errors, warnings


def build_marker_paths(config: dict, hmm_dir: Path) -> list[Path]:
    functional_cfg = config.get("functional", {}) or {}
    markers_cfg = functional_cfg.get("markers", [])
    if markers_cfg:
        return [Path(marker["hmm"]) for marker in markers_cfg if "hmm" in marker]
    return [hmm_dir / f"{marker}.hmm" for marker in DEFAULT_MARKERS]


def check_external_tools(
    stages: dict,
    fraggenescan_inputs: dict,
    skip_tool_check: bool,
) -> tuple[list[str], list[str]]:
    """Check for required external command-line tools."""
    errors: list[str] = []
    warnings: list[str] = []

    if skip_tool_check:
        warnings.append("external tool checks skipped by user request")
        return errors, warnings

    required_tools: set[str] = set()
    for stage_name, tools in REQUIRED_TOOLS_BY_STAGE.items():
        if stages.get(stage_name, False):
            required_tools.update(tools)

    if stages.get("marker_annotator", False) and fraggenescan_inputs:
        required_tools.add("FragGeneScanRs")

    missing = sorted(tool for tool in required_tools if shutil.which(tool) is None)
    for tool in missing:
        errors.append(
            f"missing external tool '{tool}' on PATH "
            "(install and retry preflight)"
        )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight check for MethaNet pilot inputs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("poc/mucc_rumen_poc/manifest.tsv"),
        help="Path to pilot manifest TSV.",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("poc/mucc_rumen_poc/pipeline.yaml"),
        help="Path to pilot pipeline config YAML.",
    )
    parser.add_argument(
        "--assemblies-dir",
        type=Path,
        default=None,
        help="Override assemblies directory (default from config or data/assemblies).",
    )
    parser.add_argument(
        "--hmm-dir",
        type=Path,
        default=None,
        help="Override HMM directory (default from config or data/hmm).",
    )
    parser.add_argument(
        "--marker-db",
        type=Path,
        default=None,
        help="Override marker DB path (default from config or db/marker_db.mmsdb).",
    )
    parser.add_argument(
        "--skip-tool-check",
        action="store_true",
        help="Skip checks for external command-line tools.",
    )
    args = parser.parse_args()

    manifest_rows, errors, warnings = read_manifest(args.manifest)

    config = {}
    if args.pipeline_config.exists():
        config = load_yaml(args.pipeline_config)
    elif args.pipeline_config:
        warnings.append(f"pipeline config not found: {args.pipeline_config}")

    stages = config.get("stages", {}) or {}
    fraggenescan_inputs = config.get("fraggenescan_inputs", {}) or {}
    require_accessions = stages.get("data_curator", False)
    require_fastas = stages.get("marker_annotator", False) or stages.get(
        "embedding_generator", False
    )

    errors.extend(check_accessions(manifest_rows, require_accessions)[0])
    warnings.extend(check_accessions(manifest_rows, require_accessions)[1])

    assemblies_dir = args.assemblies_dir or Path(
        config.get("paths", {}).get("assemblies", "data/assemblies")
    )
    sample_ids = [row.get("sample_id", "") for row in manifest_rows if row.get("sample_id")]
    fasta_errors, fasta_warnings = check_fastas(sample_ids, assemblies_dir, require_fastas)
    errors.extend(fasta_errors)
    warnings.extend(fasta_warnings)

    hmm_dir = args.hmm_dir or Path(
        config.get("paths", {}).get("hmm_dir", "data/hmm")
    )
    marker_db = args.marker_db or Path(
        config.get("paths", {}).get("marker_db", "db/marker_db.mmsdb")
    )
    marker_paths = build_marker_paths(config, hmm_dir)
    if stages.get("marker_annotator", False):
        errors.extend(check_hmms(hmm_dir, marker_db, marker_paths))

    tool_errors, tool_warnings = check_external_tools(
        stages=stages,
        fraggenescan_inputs=fraggenescan_inputs,
        skip_tool_check=args.skip_tool_check,
    )
    errors.extend(tool_errors)
    warnings.extend(tool_warnings)

    config_errors, config_warnings = check_config_samples(config, set(sample_ids))
    errors.extend(config_errors)
    warnings.extend(config_warnings)

    print("MethaNet pilot preflight summary")
    print(f"manifest: {args.manifest}")
    if args.pipeline_config:
        print(f"pipeline_config: {args.pipeline_config}")
    print(f"samples: {len(sample_ids)}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nStatus: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
