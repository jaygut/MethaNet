#!/usr/bin/env python3
"""Reconcile pilot placeholder IDs with selected MAG IDs.

This script updates:
- poc/mucc_rumen_poc/manifest.tsv sample_id + accession columns
- poc/mucc_rumen_poc/pipeline.yaml sample lists

It assumes you have already selected:
- rumen MAG IDs (e.g., from ENA)
- MUCC MAG IDs (e.g., from MUCC metadata)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional dependency in some envs
    raise SystemExit(
        "PyYAML is required for pipeline updates. Install with: uv sync --extra pipeline"
    ) from exc


@dataclass(frozen=True)
class MuccPlaceholder:
    """Parsed MUCC placeholder sample metadata."""

    sample_id: str
    site: str
    index: int


def read_ids(path: Path) -> list[str]:
    """Read one ID per line, skipping blanks."""
    if not path.exists():
        raise SystemExit(f"ID file not found: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise SystemExit(f"No IDs found in: {path}")
    return ids


def parse_mucc_placeholder(sample_id: str) -> MuccPlaceholder | None:
    """Parse MUCC placeholder IDs like MUCC_MAG_OWC_001."""
    parts = sample_id.split("_")
    if len(parts) < 4 or parts[0] != "MUCC" or parts[1] != "MAG":
        return None
    site = parts[2]
    try:
        idx = int(parts[3])
    except ValueError:
        return None
    return MuccPlaceholder(sample_id=sample_id, site=site, index=idx)


def load_manifest(path: Path) -> tuple[list[dict], list[str]]:
    """Load manifest rows preserving header order."""
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"Manifest missing header row: {path}")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def write_manifest(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Write manifest TSV using original column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def group_mucc_ids_by_site(
    mucc_ids: Sequence[str],
    metadata_path: Path,
    id_col: str,
    site_col: str,
) -> dict[str, list[str]]:
    """Group MUCC IDs by site using metadata for robust mapping."""
    df = pd.read_csv(metadata_path, sep="\t")
    if id_col not in df.columns:
        raise SystemExit(f"MUCC metadata missing id column: {id_col}")
    if site_col not in df.columns:
        raise SystemExit(f"MUCC metadata missing site column: {site_col}")

    id_to_site = df.set_index(id_col)[site_col].to_dict()

    grouped: dict[str, list[str]] = defaultdict(list)
    missing: list[str] = []
    for mag_id in mucc_ids:
        site = id_to_site.get(mag_id)
        if site is None:
            missing.append(mag_id)
            continue
        grouped[str(site)].append(mag_id)

    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(
            "Some MUCC IDs were not found in metadata. "
            f"First missing IDs: {preview}"
        )

    return dict(grouped)


def partition_ids(ids: Sequence[str], sizes: Sequence[int]) -> list[list[str]]:
    """Partition IDs sequentially into chunks of the given sizes."""
    q = deque(ids)
    chunks: list[list[str]] = []
    for size in sizes:
        if len(q) < size:
            raise SystemExit(
                f"Not enough MUCC IDs to fill site chunk of size {size} "
                f"(remaining: {len(q)})"
            )
        chunks.append([q.popleft() for _ in range(size)])
    if q:
        raise SystemExit(
            f"Too many MUCC IDs provided ({len(ids)}). "
            f"Expected exactly {sum(sizes)} based on manifest."
        )
    return chunks


def update_pipeline_yaml(path: Path, mapping: dict[str, str], out_path: Path) -> None:
    """Update pipeline sample lists using the provided ID mapping."""
    config = yaml.safe_load(path.read_text()) or {}

    def remap_list(key: str) -> None:
        values = config.get(key)
        if not values:
            return
        config[key] = [mapping.get(sample_id, sample_id) for sample_id in values]

    for key in [
        "assembly_samples",
        "marker_samples",
        "embedding_samples",
        "dna_embedding_samples",
    ]:
        remap_list(key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))


def ensure_backup(path: Path) -> Path:
    """Create a .bak backup alongside the path and return it."""
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text())
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile pilot placeholder IDs.")
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
        help="Path to pilot pipeline YAML config.",
    )
    parser.add_argument(
        "--rumen-ids",
        type=Path,
        required=True,
        help="Path to rumen MAG IDs (one per line).",
    )
    parser.add_argument(
        "--mucc-ids",
        type=Path,
        required=True,
        help="Path to MUCC MAG IDs (one per line).",
    )
    parser.add_argument(
        "--mucc-metadata",
        type=Path,
        default=None,
        help=(
            "Optional MUCC metadata TSV for robust site-aware mapping. "
            "If omitted, MUCC IDs are assigned sequentially by site."
        ),
    )
    parser.add_argument(
        "--mucc-id-col",
        default="mag_id",
        help="MUCC metadata ID column name (default: mag_id).",
    )
    parser.add_argument(
        "--mucc-site-col",
        default="site",
        help="MUCC metadata site column name (default: site).",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Optional output path for updated manifest (default: in-place with backup).",
    )
    parser.add_argument(
        "--out-pipeline-config",
        type=Path,
        default=None,
        help="Optional output path for updated pipeline config (default: in-place with backup).",
    )
    parser.add_argument(
        "--mapping-out",
        type=Path,
        default=None,
        help=(
            "Path to write old->new ID mapping TSV "
            "(default: alongside the manifest)."
        ),
    )
    args = parser.parse_args()

    manifest_rows, fieldnames = load_manifest(args.manifest)
    rumen_ids = read_ids(args.rumen_ids)
    mucc_ids = read_ids(args.mucc_ids)

    rumen_indices: list[int] = []
    mucc_by_site: Dict[str, list[int]] = defaultdict(list)
    site_order: list[str] = []

    for idx, row in enumerate(manifest_rows):
        sample_id = (row.get("sample_id") or "").strip()
        domain = (row.get("domain") or "").strip().lower()

        placeholder = parse_mucc_placeholder(sample_id)
        if placeholder is not None:
            site = placeholder.site
            if site not in mucc_by_site:
                site_order.append(site)
            mucc_by_site[site].append(idx)
            continue

        if domain == "rumen" or sample_id.startswith("RUMEN_MAG_"):
            rumen_indices.append(idx)

    expected_rumen = len(rumen_indices)
    expected_mucc = sum(len(indices) for indices in mucc_by_site.values())

    if len(rumen_ids) != expected_rumen:
        raise SystemExit(
            f"Rumen ID count mismatch: got {len(rumen_ids)}, expected {expected_rumen}"
        )
    if len(mucc_ids) != expected_mucc:
        raise SystemExit(
            f"MUCC ID count mismatch: got {len(mucc_ids)}, expected {expected_mucc}"
        )

    if args.mucc_metadata:
        mucc_ids_by_site = group_mucc_ids_by_site(
            mucc_ids=mucc_ids,
            metadata_path=args.mucc_metadata,
            id_col=args.mucc_id_col,
            site_col=args.mucc_site_col,
        )
    else:
        # Sequential partition by site counts in manifest.
        site_sizes = [len(mucc_by_site[site]) for site in site_order]
        partitions = partition_ids(mucc_ids, site_sizes)
        mucc_ids_by_site = {
            site: ids_for_site for site, ids_for_site in zip(site_order, partitions)
        }

    mapping: dict[str, str] = {}
    mapping_rows: list[dict] = []

    # Apply rumen IDs in manifest order.
    for idx, new_id in zip(rumen_indices, rumen_ids):
        row = manifest_rows[idx]
        old_id = row["sample_id"]
        row["sample_id"] = new_id
        row["accession"] = new_id
        row["notes"] = f"{row.get('notes', '')}; original_sample_id={old_id}".strip("; ")
        mapping[old_id] = new_id
        mapping_rows.append(
            {
                "original_sample_id": old_id,
                "new_sample_id": new_id,
                "domain": row.get("domain", ""),
                "site": "",
            }
        )

    # Apply MUCC IDs site-by-site in manifest order.
    for site in site_order:
        indices = mucc_by_site[site]
        ids_for_site = mucc_ids_by_site.get(site, [])
        if len(ids_for_site) != len(indices):
            raise SystemExit(
                f"Site {site} ID count mismatch: got {len(ids_for_site)}, "
                f"expected {len(indices)}"
            )
        for idx, new_id in zip(indices, ids_for_site):
            row = manifest_rows[idx]
            old_id = row["sample_id"]
            row["sample_id"] = new_id
            row["accession"] = new_id
            row["notes"] = (
                f"{row.get('notes', '')}; original_sample_id={old_id}; site={site}"
            ).strip("; ")
            mapping[old_id] = new_id
            mapping_rows.append(
                {
                    "original_sample_id": old_id,
                    "new_sample_id": new_id,
                    "domain": row.get("domain", ""),
                    "site": site,
                }
            )

    out_manifest = args.out_manifest or args.manifest
    out_pipeline = args.out_pipeline_config or args.pipeline_config
    mapping_out = args.mapping_out or (out_manifest.parent / "id_mapping.tsv")

    if out_manifest == args.manifest:
        backup = ensure_backup(args.manifest)
        print(f"Backed up manifest to: {backup}")
    if out_pipeline == args.pipeline_config and args.pipeline_config.exists():
        backup = ensure_backup(args.pipeline_config)
        print(f"Backed up pipeline config to: {backup}")

    write_manifest(out_manifest, manifest_rows, fieldnames)
    print(f"Wrote updated manifest: {out_manifest}")

    if args.pipeline_config.exists():
        update_pipeline_yaml(args.pipeline_config, mapping, out_pipeline)
        print(f"Wrote updated pipeline config: {out_pipeline}")
    else:
        print(f"Pipeline config not found (skipped): {args.pipeline_config}")

    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    with mapping_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_sample_id", "new_sample_id", "domain", "site"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(mapping_rows)
    print(f"Wrote ID mapping: {mapping_out}")


if __name__ == "__main__":
    main()
