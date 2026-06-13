#!/usr/bin/env python3
"""Recover sample/environmental metadata for the MethaNet 662-proteome cohort.

This script is intentionally conservative: it separates exact accession-level
metadata from site/project-level provenance inferred from MUCC/Zenodo records.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd


ENA_SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"
ZENODO_RECORD = "https://zenodo.org/api/records/{record_id}"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}.fcgi"

RUMEN_FIELDS = [
    "analysis_accession",
    "analysis_alias",
    "study_accession",
    "secondary_study_accession",
    "sample_accession",
    "secondary_sample_accession",
    "scientific_name",
    "tax_id",
    "environment_biome",
    "environment_feature",
    "environment_material",
    "broad_scale_environmental_context",
    "local_environmental_context",
    "environmental_medium",
    "country",
    "collection_date",
    "host",
    "host_tax_id",
    "depth",
    "submitted_ftp",
]

MUCC_SOURCE_CONTEXT = {
    "GTDB": {
        "context_level": "assembly_biosample_exact_when_ncbi_resolves",
        "site_label": "GTDB/NCBI reference MAG",
        "associated_projects": "",
        "site_environment_context": "mixed external Methanoregula reference genomes",
    },
    "JGI": {
        "context_level": "source_bucket_only",
        "site_label": "JGI/IMG source bucket",
        "associated_projects": "",
        "site_environment_context": "MUCC Methanoregula MAG list identifies source as JGI; compact files do not expose per-MAG sample metadata",
    },
    "OWC": {
        "context_level": "site_project_level_plus_owc_bin_exact",
        "site_label": "Old Woman Creek",
        "associated_projects": "PRJNA1007388",
        "site_environment_context": "wetland soils/metagenomes and OWC metatranscriptomes in MUCC v2.0.0",
    },
    "PPR": {
        "context_level": "site_project_level",
        "site_label": "Prairie Pothole Region",
        "associated_projects": "PRJNA1007388",
        "site_environment_context": "PPR7/PPR8 wetland soils/metagenomes in MUCC v2.0.0",
    },
    "STM": {
        "context_level": "site_project_level",
        "site_label": "Stordalen Mire",
        "associated_projects": "PRJNA386538",
        "site_environment_context": "Stordalen Mire fen/bog wetland MAGs in MUCC v2.0.0",
    },
}

METADATA_CAVEATS = {
    "exact_analysis_accession": "",
    "exact_ncbi_assembly_biosample": "",
    "exact_owc_bin_plus_site_project": "owc_bin_match_site_project_context_not_full_sample_environment",
    "exact_mucc_source_bucket": "source_or_site_level_context_not_sample_level",
    "missing_ena_match": "metadata_resolution_missing",
    "missing_source_bucket": "metadata_resolution_missing",
}


def http_get(url: str, *, retries: int = 4, sleep_s: float = 1.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MethaNet metadata recovery"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt == retries:
                break
            time.sleep(sleep_s * attempt)
    raise RuntimeError(f"Failed GET after {retries} attempts: {url}") from last_error


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def get_json(url: str) -> dict[str, Any]:
    return json.loads(http_get(url).decode("utf-8"))


def ena_query(result: str, query: str, fields: list[str], limit: int = 100000) -> pd.DataFrame:
    params = {
        "result": result,
        "query": query,
        "fields": ",".join(fields),
        "format": "tsv",
        "limit": str(limit),
    }
    url = ENA_SEARCH + "?" + urllib.parse.urlencode(params)
    text = http_get(url).decode("utf-8")
    return pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")


def ncbi_esearch(db: str, term: str) -> list[str]:
    url = NCBI_EUTILS.format(endpoint="esearch") + "?" + urllib.parse.urlencode(
        {"db": db, "term": term, "retmode": "json"}
    )
    data = get_json(url)
    return data.get("esearchresult", {}).get("idlist", [])


def ncbi_esummary(db: str, ids: list[str]) -> dict[str, Any]:
    if not ids:
        return {}
    url = NCBI_EUTILS.format(endpoint="esummary") + "?" + urllib.parse.urlencode(
        {"db": db, "id": ",".join(ids), "retmode": "json"}
    )
    return get_json(url).get("result", {})


def parse_biosample_attributes(sampledata: str) -> dict[str, str]:
    if not sampledata:
        return {}
    attrs: dict[str, str] = {}
    try:
        root = ET.fromstring(sampledata)
    except ET.ParseError:
        return attrs
    for attr in root.findall(".//Attribute"):
        key = attr.attrib.get("harmonized_name") or attr.attrib.get("attribute_name")
        if key:
            attrs[f"biosample_attr_{key}"] = (attr.text or "").strip()
    return attrs


def clean_assembly_accession(mag_id: str) -> str:
    acc = mag_id.replace("_genomic", "")
    match = re.match(r"^(GC[AF]_\d+\.\d+)", acc)
    return match.group(1) if match else acc


def recover_gtdb_ncbi_metadata(mag_ids: list[str], raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mag_id in mag_ids:
        query_acc = clean_assembly_accession(mag_id)
        row: dict[str, Any] = {"mag_id_candidate": mag_id, "ncbi_query_accession": query_acc}
        try:
            assembly_cache = raw_dir / "ncbi_assembly" / f"{query_acc}.json"
            if assembly_cache.exists():
                asm = json.loads(assembly_cache.read_text())
                row["ncbi_assembly_uid"] = str(asm.get("uid", ""))
                uid = str(asm.get("uid", ""))
            else:
                assembly_ids = ncbi_esearch("assembly", query_acc)
                row["ncbi_assembly_uid"] = ";".join(assembly_ids)
                summary = ncbi_esummary("assembly", assembly_ids)
                uid = next((x for x in summary.get("uids", []) if x in summary), None)
                asm = summary[uid] if uid else {}
                if asm:
                    write_bytes(assembly_cache, json.dumps(asm, indent=2).encode())
            if uid and asm:
                row.update({
                    "ncbi_assembly_accession": asm.get("assemblyaccession", ""),
                    "ncbi_organism": asm.get("organism", ""),
                    "ncbi_species_name": asm.get("speciesname", ""),
                    "ncbi_biosample_accession": asm.get("biosampleaccn", ""),
                    "ncbi_biosample_id": asm.get("biosampleid", ""),
                    "ncbi_bioprojects_genbank": ";".join(
                        p.get("bioprojectaccn", "") for p in asm.get("gb_bioprojects", [])
                    ),
                    "ncbi_bioprojects_refseq": ";".join(
                        p.get("bioprojectaccn", "") for p in asm.get("rs_bioprojects", [])
                    ),
                    "ncbi_submitter": asm.get("submitterorganization", ""),
                    "ncbi_wgs": asm.get("wgs", ""),
                    "ncbi_coverage": asm.get("coverage", ""),
                    "ncbi_assembly_release_date": asm.get("asmreleasedate_genbank", ""),
                })
                biosample_id = str(asm.get("biosampleid", "") or "")
                if biosample_id:
                    biosample_cache = raw_dir / "ncbi_biosample" / f"{query_acc}_{biosample_id}.json"
                    if biosample_cache.exists():
                        bs = json.loads(biosample_cache.read_text())
                        bs_uid = str(bs.get("uid", ""))
                    else:
                        bs_summary = ncbi_esummary("biosample", [biosample_id])
                        bs_uid = next((x for x in bs_summary.get("uids", []) if x in bs_summary), None)
                        bs = bs_summary[bs_uid] if bs_uid else {}
                        if bs:
                            write_bytes(biosample_cache, json.dumps(bs, indent=2).encode())
                    if bs_uid and bs:
                        row.update({
                            "biosample_title": bs.get("title", ""),
                            "biosample_accession": bs.get("accession", ""),
                            "biosample_organism": bs.get("organism", ""),
                            "biosample_package": bs.get("package", ""),
                            "biosample_identifiers": bs.get("identifiers", ""),
                        })
                        row.update(parse_biosample_attributes(bs.get("sampledata", "")))
            row["ncbi_lookup_status"] = "matched" if row.get("ncbi_biosample_accession") else "assembly_only_or_unmatched"
        except Exception as exc:  # pragma: no cover - network dependent
            row["ncbi_lookup_status"] = "error"
            row["ncbi_lookup_error"] = repr(exc)
        rows.append(row)
        time.sleep(0.35)
    return pd.DataFrame(rows)


def parse_owc_sample_columns(columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        row = {"expression_sample_id": col}
        parts = col.split("_")
        if len(parts) >= 4 and parts[0] in {"Aug", "Jul", "Sept"}:
            row.update({
                "pattern_class": (
                    "owc_month_site_core_depth_replicate"
                    if len(parts) >= 5
                    else "owc_month_site_core_depth"
                ),
                "month": parts[0],
                "site_code": parts[1],
                "core_code": parts[2],
                "depth_code": parts[3],
                "replicate": parts[4] if len(parts) == 5 else ("_".join(parts[4:]) if len(parts) > 5 else ""),
            })
        elif len(parts) >= 6 and parts[0] == "OWC":
            row.update({
                "pattern_class": "owc_prefixed_month_site_core_depth_replicate",
                "month": parts[1],
                "site_code": parts[2],
                "core_code": parts[3],
                "depth_code": parts[4],
                "replicate": parts[5] if len(parts) == 6 else "_".join(parts[5:]),
            })
        elif len(parts) == 4 and parts[0] in {"Mud", "Plant", "PlantDeep"}:
            row.update({
                "pattern_class": "owc_material_date_replicate",
                "material_or_zone": parts[0],
                "date_code": f"{parts[1]}_{parts[2]}",
                "replicate": parts[3],
            })
        elif len(parts) == 5 and parts[0] == "2015":
            row.update({
                "pattern_class": "owc_2015_month_site_depth_replicate",
                "year": parts[0],
                "month": parts[1],
                "site_code": parts[2],
                "depth_code": parts[3],
                "replicate": parts[4],
            })
        else:
            row["pattern_class"] = "unparsed"
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_zenodo_files(record_id: str, keys: list[str], raw_dir: Path) -> dict[str, Path]:
    record = get_json(ZENODO_RECORD.format(record_id=record_id))
    write_bytes(raw_dir / f"zenodo_record_{record_id}.json", json.dumps(record, indent=2).encode())
    file_map = {f["key"]: f["links"]["self"] for f in record.get("files", [])}
    out: dict[str, Path] = {}
    for key in keys:
        if key not in file_map:
            continue
        path = raw_dir / "zenodo_files" / record_id / key
        if not path.exists():
            write_bytes(path, http_get(file_map[key]))
        out[key] = path
    return out


def bioproject_summaries(accessions: list[str], raw_dir: Path) -> pd.DataFrame:
    rows = []
    for acc in accessions:
        row = {"bioproject_accession": acc}
        try:
            ids = ncbi_esearch("bioproject", f"{acc}[Project Accession]") or ncbi_esearch("bioproject", acc)
            row["bioproject_uid"] = ";".join(ids)
            summary = ncbi_esummary("bioproject", ids)
            uid = next((x for x in summary.get("uids", []) if x in summary), None)
            if uid:
                item = summary[uid]
                write_bytes(raw_dir / "ncbi_bioproject" / f"{acc}.json", json.dumps(item, indent=2).encode())
                row.update({
                    "title": item.get("project_title", item.get("title", "")),
                    "description": item.get("project_description", item.get("description", "")),
                    "organism": item.get("organism_name", ""),
                    "submission_date": item.get("submission_date", ""),
                    "last_update": item.get("last_update", ""),
                })
            row["lookup_status"] = "matched" if uid else "unmatched"
        except Exception as exc:  # pragma: no cover - network dependent
            row["lookup_status"] = "error"
            row["lookup_error"] = repr(exc)
        rows.append(row)
        time.sleep(0.35)
    return pd.DataFrame(rows)


def first_existing(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col in cols:
        if col in df.columns:
            vals = df[col].fillna("").astype(str)
            out = out.mask(out.eq("") & vals.ne(""), vals)
    return out


def metadata_caveat(resolution: Any) -> str:
    return METADATA_CAVEATS.get(str(resolution), "metadata_resolution_requires_review")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(sep="\t", index=False).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/functional_metagenomics/environmental_metadata_recovery_20260612"),
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = (repo / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    crosswalk_path = repo / "ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv"
    mag_list_path = repo / "poc/mucc_rumen_poc/Methanoregula_MAGs_list.txt"
    crosswalk = pd.read_csv(crosswalk_path, sep="\t", dtype=str).fillna("")
    crosswalk["mag_id_candidate"] = crosswalk["mag_id_candidate"].astype(str)

    rumen = crosswalk[crosswalk["source"].eq("rumen")].copy()
    mucc = crosswalk[crosswalk["source"].eq("mucc")].copy()

    ena_full = ena_query("analysis", 'study_accession="PRJEB31266"', RUMEN_FIELDS)
    ena_full.to_csv(out_dir / "ena_prjeb31266_analysis_metadata.tsv", sep="\t", index=False)
    rumen_join = rumen.merge(
        ena_full,
        left_on="source_analysis_accession",
        right_on="analysis_accession",
        how="left",
        suffixes=("", "_ena"),
    )
    rumen_join["metadata_resolution"] = rumen_join["analysis_accession"].fillna("").astype(str).map(
        lambda x: "exact_analysis_accession" if x else "missing_ena_match"
    )
    rumen_join["metadata_source_urls_or_accessions"] = (
        "ENA:PRJEB31266;analysis_accession="
        + rumen_join["analysis_accession"].fillna("").astype(str)
        + ";sample_accession="
        + rumen_join["sample_accession"].fillna("").astype(str)
    )
    rumen_join["metadata_caveat"] = rumen_join["metadata_resolution"].map(metadata_caveat)
    rumen_join.to_csv(out_dir / "rumen_proteome_environmental_metadata.tsv", sep="\t", index=False)

    z145 = fetch_zenodo_files(
        "14532347",
        [
            "Methanoregula_MAGs_list.txt",
            "Methanoregula_physiology.txt",
            "classification_w_outgroup.txt",
            "Methanoregula_metabolism_summary.xlsx",
        ],
        raw_dir,
    )
    z108 = fetch_zenodo_files("10822869", ["owc_metat_table_methanoregula_genes.csv"], raw_dir)

    mag_sources = pd.read_csv(z145.get("Methanoregula_MAGs_list.txt", mag_list_path), sep="\t", dtype=str).fillna("")
    mag_sources = mag_sources.rename(columns={"MAG": "mag_id_candidate", "Source": "mucc_source_bucket", "Value": "mucc_source_value"})
    physiology = pd.read_csv(z145["Methanoregula_physiology.txt"], sep="\t", dtype=str).fillna("").rename(columns={"MAG": "mag_id_candidate"})
    classification = pd.read_csv(
        z145["classification_w_outgroup.txt"],
        sep="\t",
        header=None,
        names=["mag_id_candidate", "methanoregula_tree_taxonomy"],
        dtype=str,
    ).fillna("")
    mucc_join = mucc.merge(mag_sources, on="mag_id_candidate", how="left")
    mucc_join = mucc_join.merge(physiology, on="mag_id_candidate", how="left")
    mucc_join = mucc_join.merge(classification, on="mag_id_candidate", how="left")

    context_rows = []
    for bucket, values in MUCC_SOURCE_CONTEXT.items():
        context_rows.append({"mucc_source_bucket": bucket, **values})
    context = pd.DataFrame(context_rows)
    context.to_csv(out_dir / "mucc_source_bucket_context.tsv", sep="\t", index=False)
    mucc_join = mucc_join.merge(context, on="mucc_source_bucket", how="left")

    if "owc_metat_table_methanoregula_genes.csv" in z108:
        owc = pd.read_csv(z108["owc_metat_table_methanoregula_genes.csv"], dtype=str).fillna("")
        owc = owc.rename(columns={"#DB_ID": "mag_id_candidate"})
        owc_summary = owc[["mag_id_candidate", "Bin_ID", "taxonomy_20July2021", "old_taxonomy", "metabolism"]].copy()
        owc_summary.to_csv(out_dir / "mucc_owc_metatranscriptome_mag_table.tsv", sep="\t", index=False)
        sample_columns = [c for c in owc.columns if c not in {"mag_id_candidate", "Bin_ID", "taxonomy_20July2021", "old_taxonomy", "metabolism"}]
        parse_owc_sample_columns(sample_columns).to_csv(out_dir / "mucc_owc_metatranscriptome_sample_columns.tsv", sep="\t", index=False)
        mucc_join = mucc_join.merge(owc_summary, on="mag_id_candidate", how="left", suffixes=("", "_owc"))

    gtdb_ids = mucc_join.loc[mucc_join["mucc_source_bucket"].eq("GTDB"), "mag_id_candidate"].dropna().astype(str).tolist()
    gtdb_ncbi = recover_gtdb_ncbi_metadata(gtdb_ids, raw_dir)
    gtdb_ncbi.to_csv(out_dir / "mucc_gtdb_ncbi_assembly_biosample_metadata.tsv", sep="\t", index=False)
    mucc_join = mucc_join.merge(gtdb_ncbi, on="mag_id_candidate", how="left")

    mucc_join["metadata_resolution"] = "missing_source_bucket"
    mucc_join.loc[mucc_join["mucc_source_bucket"].fillna("").ne(""), "metadata_resolution"] = "exact_mucc_source_bucket"
    mucc_join.loc[mucc_join.get("Bin_ID", pd.Series("", index=mucc_join.index)).fillna("").ne(""), "metadata_resolution"] = "exact_owc_bin_plus_site_project"
    mucc_join.loc[mucc_join.get("ncbi_biosample_accession", pd.Series("", index=mucc_join.index)).fillna("").ne(""), "metadata_resolution"] = "exact_ncbi_assembly_biosample"
    mucc_join["metadata_source_urls_or_accessions"] = "Zenodo:14532347"
    mucc_join.loc[
        mucc_join.get("Bin_ID", pd.Series("", index=mucc_join.index)).fillna("").ne(""),
        "metadata_source_urls_or_accessions",
    ] += ";Zenodo:10822869"
    biosample_acc = mucc_join.get("ncbi_biosample_accession", pd.Series("", index=mucc_join.index)).fillna("").astype(str)
    mucc_join.loc[biosample_acc.ne(""), "metadata_source_urls_or_accessions"] += ";NCBI_BioSample:" + biosample_acc
    mucc_join["metadata_caveat"] = mucc_join["metadata_resolution"].map(metadata_caveat)
    mucc_join.to_csv(out_dir / "mucc_proteome_environmental_metadata.tsv", sep="\t", index=False)

    bioprojects = sorted({
        "PRJNA1007388",
        "PRJNA386538",
        "PRJNA638786",
        "PRJNA638601",
        "PRJEB31266",
    })
    bp = bioproject_summaries(bioprojects, raw_dir)
    bp.to_csv(out_dir / "source_bioproject_summaries.tsv", sep="\t", index=False)

    combined = pd.concat([rumen_join, mucc_join], ignore_index=True, sort=False)
    combined["sample_or_biosample_accession"] = first_existing(
        combined,
        ["sample_accession", "secondary_sample_accession", "ncbi_biosample_accession", "biosample_accession"],
    )
    combined["environment_context_primary"] = first_existing(
        combined,
        [
            "environment_biome",
            "broad_scale_environmental_context",
            "biosample_attr_metagenome_source",
            "site_environment_context",
        ],
    )
    combined["environment_material_primary"] = first_existing(
        combined,
        [
            "environment_material",
            "environmental_medium",
            "biosample_attr_isolation_source",
            "biosample_attr_metagenome_source",
        ],
    )
    combined["geo_context_primary"] = first_existing(
        combined,
        ["country", "biosample_attr_geo_loc_name", "biosample_attr_lat_lon"],
    )
    combined.to_csv(out_dir / "cohort_662_environmental_metadata_crosswalk.tsv", sep="\t", index=False)

    validation_rows = [
        ("crosswalk_rows", len(crosswalk)),
        ("crosswalk_unique_proteome_id", crosswalk["proteome_id"].nunique()),
        ("rumen_crosswalk_rows", len(rumen)),
        ("rumen_ena_exact_matches", int(rumen_join["metadata_resolution"].eq("exact_analysis_accession").sum())),
        ("mucc_crosswalk_rows", len(mucc)),
        ("mucc_source_bucket_matches", int(mucc_join["mucc_source_bucket"].fillna("").ne("").sum())),
        ("mucc_owc_exact_bin_matches", int(mucc_join.get("Bin_ID", pd.Series("", index=mucc_join.index)).fillna("").ne("").sum())),
        ("mucc_gtdb_ncbi_biosample_matches", int(mucc_join.get("ncbi_biosample_accession", pd.Series("", index=mucc_join.index)).fillna("").ne("").sum())),
        ("combined_rows", len(combined)),
        ("combined_unique_proteome_id", combined["proteome_id"].nunique()),
    ]
    validation = pd.DataFrame(validation_rows, columns=["metric", "value"])
    validation.to_csv(out_dir / "metadata_recovery_validation.tsv", sep="\t", index=False)

    report = [
        "# MethaNet Environmental Metadata Recovery",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "## Validation",
        "",
        dataframe_to_markdown(validation),
        "",
        "## Resolution Semantics",
        "",
        "- `exact_analysis_accession`: PRJEB31266 rumen MAG joined directly through ENA `analysis_accession`.",
        "- `exact_ncbi_assembly_biosample`: GTDB/NCBI MAG joined through NCBI Assembly and BioSample metadata.",
        "- `exact_owc_bin_plus_site_project`: OWC MAG matched to the MUCC OWC metatranscriptome table by MAG ID/Bin ID; environmental context remains site/project/sample-design level.",
        "- `exact_mucc_source_bucket`: MAG source bucket from the Methanoregula MAG list; compact public files do not expose per-MAG BioSample.",
        "",
        "## Caveat",
        "",
        "MUCC JGI/PPR/STM source buckets are exact for the Methanoregula MAG list, but most are not yet sample-level environmental metadata. Treat those rows as source/site/project-level provenance until a MUCC MAG metadata TSV, JGI/IMG export, or BioSample mapping table is added.",
        "",
        "The BioProject table is intentionally included as a source check, not a blind truth table. Exact NCBI BioProject lookups confirm PRJEB31266 and PRJNA1007388 cleanly; PRJNA386538 and PRJNA638601 currently return NCBI summaries that do not obviously match the MUCC Zenodo prose, so those accessions should be treated as Zenodo-stated leads requiring manual review before sample-level modeling.",
    ]
    (out_dir / "METADATA_RECOVERY_REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Wrote metadata recovery outputs to {out_dir}")
    print(validation.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
