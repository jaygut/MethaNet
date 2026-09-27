"""Admissible RNA and ecological context without manufacturing physical joins."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time
from xml.etree import ElementTree

import pyarrow.parquet as pq

from .model import digest
from .molecular_audit import read_tsv, local_path, write_json, write_tsv
from .molecular_stage import MUCC, add_source, stable_id, habitat
from .molecular_sequences import fasta


def finite_number(value):
    if value is None or str(value).strip() in ("", "NA", "NaN", "nan", "-9999"):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return str(number) if number.is_finite() else None


def candidate_chambers(sample, chamber):
    date, patch = sample.get("collection_date", ""), sample.get("site_id", "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or patch not in ("M1", "N3", "OW2"):
        return []
    return [r["flux_observation_id"] for r in chamber if r["source_date"] == date and r["location_code"] == patch]


def assemble(repo, auditdir, stagedir, seqdir, repairdir, out):
    started = time.monotonic()
    code_sha = digest(Path(__file__))
    out.mkdir(parents=True,exist_ok=True)
    sources = {}
    audit = json.loads((auditdir/"source-key-audit.json").read_text())
    tables = {(r["lane_id"],r["table"]):r for r in audit}

    def table(lid,name):
        info=tables[(lid,name)]
        src=add_source(repo,sources,repo/info["path"],"typed_warehouse_context")
        if src["sha256"] != info["sha256"]:
            raise ValueError("Context source drift")
        return pq.ParquetFile(repo/info["path"]).read().to_pylist(),src

    meta=repo/"results/reports/methanet_atlas_metadata_readiness_20260810/tables"
    metadata={}
    for name in ("dim_sample","dim_site","link_sample_mag","link_sample_flux_window","fact_sample_environment","fact_flux_or_process_observation"):
        src=add_source(repo,sources,meta/f"{name}.tsv","historical_metadata_adapter_output")
        metadata[name]=(read_tsv(repo/src["path"]),src)
    sample_rows=metadata["dim_sample"][0]
    samples={(r["lane_id"],r["sample_id"]):r for r in sample_rows}
    if len(samples)!=len(sample_rows):
        raise ValueError("Duplicate sequencing-sample context key")
    site_rows=metadata["dim_site"][0]
    methods,methodsrc=table(MUCC,"feature_mucc_v1_sample_methods_design_context")
    sras,srasrc=table(MUCC,"link_mucc_v1_sequence_sra_sample")
    readiness,readinesssrc=table(MUCC,"feature_mucc_v1_sample_ecological_readiness")
    methodmap={r["sample_id"]:r for r in methods}
    sramap={r["sample_id"]:r for r in sras}
    readinessmap={r["sample_id"]:r for r in readiness}
    if len(readinessmap)!=len(readiness) or set(readinessmap)!=set(sramap):
        raise ValueError("Sample assay-readiness keys do not match the SRA identity ledger")
    chamber,chambersrc=table(MUCC,"fact_mucc_v1_essdive_chamber_flux")
    porewater,poresrc=table(MUCC,"fact_mucc_v1_essdive_porewater_ch4")
    tower,towersrc=table(MUCC,"fact_mucc_v1_essdive_gapfilled_tower_ch4_flux")
    repair=read_tsv(repairdir/"fact_gapfilled_tower_ch4_flux_mapping_repair.tsv")
    repairsrc=add_source(repo,sources,repairdir/"fact_gapfilled_tower_ch4_flux_mapping_repair.tsv","source_verified_tower_mapping_repair")
    repair_validation=json.loads((repairdir/"validation.json").read_text())
    if repair_validation["repair_status"]!="validated_derived_view" or repair_validation["output"]["sha256"]!=repairsrc["sha256"]:
        raise ValueError("Tower repair has no valid exact-hash receipt")
    for value in repair_validation["inputs"].values():
        src=add_source(repo,sources,local_path(repo,value["path"]),"tower_repair_primary_input")
        if src["sha256"]!=value["sha256"]:
            raise ValueError("Tower repair input drift")
    # Verify Parquet typed source and repaired TSV agree, independently of the
    # historical generic table's blank value/date columns.
    repair_byid={r["observation_id"]:r for r in repair}
    if len(repair_byid)!=len(tower):
        raise ValueError("Typed tower and repair key cardinalities differ")
    for r in tower:
        fixed=repair_byid[r["flux_observation_id"]]
        if finite_number(r["methane_flux_nmol_m2_s"])!=finite_number(fixed["value"]) or r["source_datetime_start_local_or_timezone_unknown"]!=fixed["source_datetime_start_local_or_timezone_unknown"]:
            raise ValueError("Typed Parquet tower differs from source-verified repair")

    paper_path=repo/"results/functional_metagenomics/mucc_v1_owc_wetland_20260626/source_audit/pmc_42223272/PMC13289110_fulltext.xml"
    papersrc=add_source(repo,sources,paper_path,"primary_publication_site_and_patch_context")
    root=ElementTree.parse(paper_path).getroot()
    text=" ".join(root.itertext())
    if not all(s in text for s in ("US-OWC","Ohio","M1-3","N1-3","OW1-3")):
        raise ValueError("Expected publication site/patch context not found; do not apply mapping")
    patchsets={"M1":["M1"],"N3":["N3"],"OW2":["OW2"],"Mud":["M1","M2","M3"],"Plant":["N1","N2","N3","T1"],"Open":["OW1","OW2","OW3"]}
    patchlabels={"M1":"mud patch","M2":"mud patch","M3":"mud patch","N1":"Nelumbo emergent vegetation","N2":"Nelumbo emergent vegetation","N3":"Nelumbo emergent vegetation","T1":"Typha emergent vegetation","OW1":"open-water patch","OW2":"open-water patch","OW3":"open-water patch"}
    siteaudit=[]
    contexts=[]
    for i,row in enumerate(sample_rows):
        lid,sid=row["lane_id"],row["sample_id"]
        c={"context_id":stable_id("context",lid,sid),"lane_id":lid,"sample_id":sid,"raw_site_label":row["site_id"],
            "collection_date_label":row["collection_date"],"depth_label":row["depth"],"source":metadata["dim_sample"][1],
            "source_row_ordinal":i,"grain":"source_sequencing_sample_context_not_custody_verified_specimen",
            "exact_physical_sample_claim":False,"accepted_sample_flux_pairs":0}
        if lid==MUCC:
            meth=methodmap[sid];sra=sramap[sid]
            targets=patchsets.get(row["site_id"],[])
            c.update({"study_site_id":"US-OWC","study_site_status":"publication_supported_wetland_context_not_flux_footprint_match",
                "patch_candidates":targets,"patch_link_status":"unambiguous_source_design_patch_label" if len(targets)==1 else "candidate_patch_set_from_landcover_only" if targets else "unresolved",
                "source_patch_labels":{p:patchlabels[p] for p in targets},"methods":meth,"sra":sra,
                "source_publication":papersrc,"method_source":methodsrc,"sra_source":srasrc,
                "assay_reconciliation_status":readinessmap[sid]["sra_assay_reconciliation_status"],
                "assay_reconciliation_source":readinesssrc,
                "candidate_chamber_ids":candidate_chambers(row,chamber),
                "candidate_pair_boundary":"Shared source day/patch is a candidate only; exact sample/footprint/time/depth relation and uncertainty not established",
                "habitat":"freshwater_wetland","region":"Ohio, USA","geographic_precision":"study_site_or_source_patch_label"})
            siteaudit.append({"sample_id":sid,"original_site_reference":row["site_id"],"authoritative_study_site":"US-OWC",
                "source_design_patch_status":c["patch_link_status"],"patch_candidates":targets,
                "source_chamber_same_day_patch_candidates":len(c["candidate_chamber_ids"]),"accepted_exact_flux_pairs":0,
                "unresolved_join":"Physical specimen and source observation pairing; collection day is not observation window; legacy landcover lacks unique patch",
                "rejected_inference":"A coarse site or source label does not assign a MAG or RNA value to tower/chamber flux",
                "source_sha256":papersrc["sha256"],"source_locator":"article Results model-wetland paragraph and Figure 1 caption; linked source methods context"})
        elif lid=="futian_mangrove_2026_qi":
            c.update({"habitat":"mudflat" if row["site_id"]=="MF1" else "mangrove" if row["site_id"]=="MG1" else "unresolved",
                "study_site_id":row["site_id"],"study_site_status":"source_sample_site_exact_at_metadata_grain",
                "region":"Futian Reserve, Shenzhen, China","geographic_precision":"source_site_coordinates",
                "assay_reconciliation_status":"metagenomic_DNA_source_study_no_read_abundance_join"})
        else:
            c.update({"habitat":"mangrove","study_site_id":row["site_id"],"study_site_status":"source_location_label_not_independent_site_deduplication",
                "region":"Southeast China, source-study scope","geographic_precision":"source_location_labels_and_reported_coordinates",
                "assay_reconciliation_status":"metagenomic_DNA_source_study_no_read_abundance_join"})
        contexts.append(c)

    mags=json.loads((stagedir/"selected-mag-context.json").read_text())
    selected={(r["lane_id"],r["proteome_id"]) for r in mags}
    links=[]; link_counts=Counter(); missing_link_endpoints=[]
    for i,row in enumerate(metadata["link_sample_mag"][0]):
        lid,pid,sid=row["lane_id"],row["proteome_id"],row["sample_id"]
        link_counts[(lid,row["mapping_confidence"])]+=1
        if (lid,pid) not in selected:
            continue
        if (lid,sid) not in samples:
            missing_link_endpoints.append({"row":i,"lane_id":lid,"sample_id":sid,"proteome_id":pid})
            continue
        links.append({**row,"edge_id":stable_id("sample-mag-context",lid,pid,sid),"source":metadata["link_sample_mag"][1],"source_row_ordinal":i,
            "relation":"RNA_matrix_cell_membership_not_DNA_abundance" if row["mapping_confidence"]=="exact_matrix_cell" else "candidate_source_context_not_exact_specimen_membership"})
    # A source matrix cell can exist at zero; membership is never positive
    # expression or community presence unless its actual measured value says so.
    loci=json.loads((seqdir/"loci.json").read_text())
    locusmap={(r["lane_id"],r["run_id"],r["gene_id"]):r for r in loci}
    annotations=json.loads((stagedir/"selected-annotation-events.json").read_text())
    rna_groups=defaultdict(list)
    for e in annotations:
        if e["tool"]=="source_DRAM_expression_crosswalk":
            rna_groups[(e["accession"],e["historical_included"])].append(e)
    chosen={}
    for group,evs in sorted(rna_groups.items()):
        e=min(evs,key=lambda r:r["gene_id"])
        chosen[e["gene_id"]]=e
    if len(chosen)>12:
        raise ValueError("RNA selection budget exceeded")
    rna_path=repo/"results/functional_metagenomics/mucc_v1_owc_wetland_20260626/staging/owc_metat_table_mags_genes.csv"
    rnasrc=add_source(repo,sources,rna_path,"source_processed_gene_RNA_matrix")
    expression=[];seen=set(); nrows=0
    with rna_path.open(newline="") as stream:
        reader=csv.DictReader(stream)
        columns=reader.fieldnames[1:]
        if len(columns)!=133 or len(set(columns))!=133:
            raise ValueError("RNA column identity denominator changed")
        if any((MUCC,"owc_expr__"+col) not in samples for col in columns):
            raise ValueError("RNA column has no exact metadata sample record")
        for i,row in enumerate(reader):
            gene=row[""];nrows+=1
            if gene in seen:
                raise ValueError("Duplicate processed RNA source gene key")
            seen.add(gene)
            if gene not in chosen:
                continue
            event=chosen[gene];locus=locusmap[(MUCC,event["run_id"],gene)]
            for col in columns:
                sid="owc_expr__"+col
                value=finite_number(row[col])
                expression.append({"observation_id":stable_id("rna",rnasrc["sha256"],gene,col),"lane_id":MUCC,
                    "proteome_id":event["proteome_id"],"gene_id":gene,"locus_id":locus["locus_id"],"sample_id":sid,
                    "value":value,"value_status":"reported_source_processed_value" if value is not None else "source_missing",
                    "modality":"source_processed_RNA","normalization":"source gene-length/TMM/log2-processed scale; zero handling not reconstructed; do not invert to counts",
                    "unit":"source_processed_expression_scale","source":rnasrc,"source_row_ordinal":i,"source_column":col,
                    "annotation_event_id":event["event_id"],"historical_curation_status":event["historical_curation_status"],
                    "source_sequence_available":locus["sequence_available"],"sample_assay_reconciliation_status":readinessmap[sid]["sra_assay_reconciliation_status"],
                    "assay_reconciliation_source":readinesssrc,
                    "field_pair_status":"unresolved"})
    if len({x["gene_id"] for x in expression})!=len(chosen):
        raise ValueError("Selected RNA gene was not found exactly")

    # Typed measurement examples: first 12 source rows by stable source ID, with
    # all full-table denominators retained. No selection by flux magnitude.
    measurements=[]; measurement_coverage=[]
    specs=[("chamber_CH4",chamber,chambersrc,"flux_observation_id","methane_flux_nmol_m2_s","nmol m-2 s-1","gas_flux","source_datetime_local","chamber"),
           ("porewater_CH4",porewater,poresrc,"porewater_observation_id","porewater_ch4_mM","mmol L-1","gas_concentration","source_date","porewater_dialysis"),
           ("gapfilled_tower_CH4",repair,repairsrc,"observation_id","value","nmol m-2 s-1","gas_flux","source_datetime_start_local_or_timezone_unknown","gap_filled_eddy_covariance")]
    for typ,rows,source,idcol,valcol,unit,kind,timecol,method in specs:
        numeric=sum(finite_number(r[valcol]) is not None for r in rows)
        measurement_coverage.append({"kind":typ,"rows":len(rows),"numeric_rows":numeric,"missing_rows":len(rows)-numeric,"unit":unit,"quantity_kind":kind,
            "exact_molecular_pairs":0,"source":source})
        indexed=sorted(enumerate(rows),key=lambda z:z[1][idcol])[:12]
        for ordinal,r in indexed:
            value=finite_number(r[valcol])
            measurements.append({"measurement_id":r[idcol],"lane_id":MUCC,"kind":typ,"quantity_kind":kind,
                "value":value,"unit":unit,"source_date_label":r[timecol],"time_precision":"source_unzoned_timestamp" if "T" in r[timecol] else "source_day",
                "source_end_label":r.get("source_datetime_end_local_or_timezone_unknown",""),
                "source_site":"US-OWC","patch_or_profile":r.get("location_code",r.get("peeper_code","tower_footprint_not_sample_plot")),
                "method":method,"spatial_support":"source_chamber_area_and_plot" if typ=="chamber_CH4" else "source_depth_profile" if typ=="porewater_CH4" else "eddy_covariance_site_footprint",
                "source":source,"source_row_ordinal":ordinal,"value_status":r.get("source_value_status"),
                "source_depth_label":r.get("depth_cm_relative_to_top_mineral_soil",""),"sample_join_status":"unresolved_no_authoritative_pairing","raw":r})
    env_src=add_source(repo,sources,repo/"data/external/futian_mangrove_2026_qi/metadata/futian_65_sample_metadata.tsv","source_Futian_sample_environment")
    units={"ph":"dimensionless_pH","salinity_psu":"practical_salinity","toc_mg_g":"mg g-1","ammonium_mg_kg":"mg kg-1","nitrate_mg_kg":"mg kg-1","tn_mg_g":"mg g-1","tp_mg_g":"mg g-1","ts_mg_g":"mg g-1"}
    envrows=read_tsv(repo/env_src["path"])
    envcoverage=[]
    for field,unit in units.items():
        envcoverage.append({"field":field,"samples":len(envrows),"numeric_samples":sum(finite_number(r[field]) is not None for r in envrows),"unit":unit,"source":env_src})
    for i,r in enumerate(envrows):
        for field,unit in units.items():
            value=finite_number(r[field])
            measurements.append({"measurement_id":stable_id("environment",env_src["sha256"],r["sample_name"],field),"lane_id":"futian_mangrove_2026_qi",
                "kind":field,"quantity_kind":"environmental_covariate","value":value,"unit":unit,"source_date_label":r["sampling_month_iso"],
                "time_precision":"source_month_not_exact_collection_datetime","source_site":r["sample_site"],"sample_id":r["sample_name"],
                "patch_or_profile":r["sample_site"],"source_depth_label":r["depth_cm"],"method":"source_study_chemistry_method_not_recorded_in_normalized_row",
                "spatial_support":"source_reported_sediment_sample_depth_interval","source":env_src,"source_row_ordinal":i,"source_column":field,
                "value_status":"reported_numeric" if value is not None else "source_missing","sample_join_status":"exact_source_sample_metadata_but_MAG_depth_assignment_ambiguous","raw":{field:r[field]}})

    # Current DNA sequence-content check for the historical overlap control.
    pairpath=repo/"docs/white-paper/v14/analyses/controls/same_genome_pairs.tsv"
    pairsrc=add_source(repo,sources,pairpath,"historical_same_DNA_overlap_control")
    overlap=[]
    for row in read_tsv(pairpath):
        inventories=[]
        for field in ("poc_dna_path","mucc_dna_path"):
            p=local_path(repo,row[field]);src=add_source(repo,sources,p,"overlap_assembly")
            seq_counts=Counter(hashlib.sha256(seq.encode()).hexdigest() for n,h,seq in fasta(p))
            inventories.append((src,seq_counts))
        same=inventories[0][1]==inventories[1][1]
        overlap.append({"poc_proteome_id":row["poc_proteome_id"],"mucc_proteome_id":row["mucc_proteome_id"],
            "same_current_DNA_sequence_multiset":same,"source_pair_ledger":pairsrc,"assembly_sources":[x[0] for x in inventories],
            "identity_rule":"exact multiset of uppercase contig sequence SHA256 values, multiplicity retained, contig names ignored",
            "boundary":"Same current assembly content, not independent ecological replication, original embedding-input authentication or identical gene calls"})
    # Recoverable source-rights metadata; annotation database reuse is separate.
    rights=[]
    right_specs=[("msm_china_2025","data/external/msm_china_2025/source_docs/datacite_10.5524_102702.json","CC0-1.0","original GigaDB deposited data; not annotation database rights"),
        ("futian_mangrove_2026_qi","data/external/futian_mangrove_2026_qi/source_docs/figshare_30883646_v3.json","CC-BY-4.0","Figshare v3 metadata spreadsheets; other sequence repository terms separately apply"),
        (MUCC,"results/functional_metagenomics/mucc_v1_owc_wetland_20260626/source_audit/zenodo_record_8194033.json","CC-BY-4.0","Zenodo record files; third-party database annotation reuse separately reviewed")]
    for lid,path,license_id,scope in right_specs:
        source=add_source(repo,sources,repo/path,"source_repository_rights_metadata")
        rights.append({"lane_id":lid,"reported_license":license_id,"scope":scope,"source":source,
            "external_packet_release":"blocked_pending_attribution_and_annotation_database_rights_review","independent_scientific_approval":False})
    rights.append({"lane_id":"poc_core","reported_license":"heterogeneous_sources_not_resolved_for_all_members","scope":"POC combines source studies","external_packet_release":"blocked","independent_scientific_approval":False})

    write_json(out/"sample-contexts.json",contexts)
    write_json(out/"selected-sample-mag-links.json",links)
    write_json(out/"gene-rna-observations.json",expression)
    write_json(out/"typed-measurements.json",measurements)
    write_json(out/"measurement-coverage.json",measurement_coverage)
    write_json(out/"environment-coverage.json",envcoverage)
    write_json(out/"current-DNA-overlap.json",overlap)
    write_json(out/"rights-audit.json",rights)
    write_tsv(out/"mucc-site-and-flux-relation-audit.tsv",siteaudit)
    write_json(out/"sources.json",list(sources.values()))
    write_json(out/"link-endpoint-quarantine.json",missing_link_endpoints)
    receipt={"status":"context_audit_complete_no_exact_molecular_flux_pairs","sample_records":len(samples),
        "source_site_label_records":len(site_rows),"distinct_physical_sites_not_deduced_from_labels":True,
        "mucc_study_site_contexts":len(siteaudit),"mucc_unambiguous_source_design_patch_labels":sum(len(r["patch_candidates"])==1 for r in siteaudit),
        "mucc_candidate_landcover_patch_sets":sum(len(r["patch_candidates"])>1 for r in siteaudit),
        "mucc_candidate_same_day_patch_chamber_edges":sum(r["source_chamber_same_day_patch_candidates"] for r in siteaudit),
        "mucc_exact_sample_flux_pairs":0,"source_link_counts":[{"lane_id":l,"resolution":s,"rows":n} for (l,s),n in sorted(link_counts.items())],
        "selected_sample_MAG_links":len(links),"RNA_source_rows":nrows,"RNA_source_unique_gene_keys":len(seen),"RNA_columns":len(columns),
        "selected_RNA_genes":len(chosen),"selected_RNA_observations":len(expression),"typed_measurement_examples":len(measurements),
        "current_same_DNA_pairs":sum(r["same_current_DNA_sequence_multiset"] for r in overlap),"overlap_pairs_tested":len(overlap),
        "code_sha256":code_sha,"elapsed_seconds":round(time.monotonic()-started,3),
        "outputs":{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='context-receipt.json'}}
    write_json(out/"context-receipt.json",receipt)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    for name in ("audit","staging","sequences","tower-repair","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(assemble(a.repo.resolve(),a.audit.resolve(),a.staging.resolve(),a.sequences.resolve(),a.tower_repair.resolve(),a.out.resolve()),indent=2))
