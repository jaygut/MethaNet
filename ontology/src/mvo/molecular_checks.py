"""Independent source-row drill-through and real-graph safety/denominator checks."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import time
import duckdb
import pyarrow.parquet as pq
from rdflib import Graph,Literal
from rdflib.namespace import RDF,OWL
from .model import M,C,PROV,iri,digest,canonical,sha_bytes
from .molecular_audit import write_json
from .molecular_graph import BASE_SHA,record


def load(path):return json.loads(path.read_text())


def source_rows(repo,bundle,out):
    """Verify every selected primary source row, not a favorable subsample."""
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    events=load(bundle/"staging-v2/selected-annotation-events.json")
    groups=defaultdict(list)
    for e in events:groups[e["source_path"]].append(e)
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='768MB'")
    con.execute("SET temp_directory=?",[str(out/"duckdb-spill")])
    checks=[]
    for path,es in sorted(groups.items()):
        p=repo/path
        if any(digest(p)!=e["source_sha256"] for e in es[:1]):raise ValueError("Source checksum drift")
        wanted={e["source_row_ordinal"]:e for e in es}
        # An event family can share a row in a broader panel; compare each event
        # against the source row without asserting that rows are proteins.
        if path.endswith(".parquet"):
            con.read_parquet(str(p),hive_partitioning=False,file_row_number=True).create_view("current_source",replace=True)
            rows=con.execute("SELECT * FROM current_source WHERE file_row_number IN (SELECT unnest(?))",[sorted(wanted)]).fetch_arrow_table().to_pylist()
            found={r["file_row_number"]:r for r in rows}
        else:
            opener=gzip.open if p.suffix==".gz" else open
            found={}
            with opener(p,"rt",newline="") as f:
                for i,r in enumerate(csv.DictReader(f,delimiter="\t" if p.suffix==".gz" else ",")):
                    if i in wanted:found[i]=r
        if set(found)!=set(wanted):raise ValueError("Selected source row is missing")
        for e in es:
            actual=found[e["source_row_ordinal"]]
            if json.dumps(actual,sort_keys=True,default=str)!=json.dumps(e["raw"],sort_keys=True,default=str):raise ValueError("Selected annotation source row differs from staged evidence")
        checks.append({"path":path,"sha256":es[0]["source_sha256"],"selected_events":len(es),"selected_distinct_rows":len(found),"full_row_parity":True})
        print(f"SOURCE PARITY {p.parent.parent.name}: {len(es)} selected annotation events",flush=True)
    dram=repo/"results/functional_metagenomics/mucc_v1_owc_wetland_20260626/staging/OWC_HQMQ_DB_ANNOTATIONS_20220208.txt.gz"
    rel=con.read_csv(str(dram),header=True,sep="\t",all_varchar=True,sample_size=5000)
    first=rel.columns[0];rel.create_view("dram_source",replace=True)
    if '"' in first:raise ValueError("Unexpected source key name")
    key_sql='"'+first+'"'
    scoped=con.execute(f"WITH keyed AS (SELECT fasta,{key_sql} AS gene,count(*) AS n FROM dram_source GROUP BY 1,2) SELECT sum(n) AS rows,count(*) AS unique_scoped_keys,sum(CASE WHEN n>1 THEN n-1 ELSE 0 END) AS duplicate_extra_rows,sum(CASE WHEN fasta IS NULL OR gene IS NULL OR gene='' THEN n ELSE 0 END) AS null_key_rows FROM keyed").fetchone()
    dram_receipt=dict(zip(("rows","unique_scoped_keys","duplicate_extra_rows","null_key_rows"),map(int,scoped)))
    dram_receipt.update(source_sha256=digest(dram),key=["fasta",first],scope="source DRAM namespace and original feature; not cross-caller identity")
    result={"status":"pass" if not dram_receipt["duplicate_extra_rows"] and not dram_receipt["null_key_rows"] else "source_key_issues_retained",
        "primary_selected_events":len(events),"all_selected_primary_rows_exact":True,"tables":checks,"full_DRAM_key_audit":dram_receipt,
        "elapsed_seconds":round(time.monotonic()-started,3),"code_sha256":digest(Path(__file__))}
    write_json(out/"source-row-parity.json",result);return result


def real_graph(build,bundle,base,out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    manifest=load(build/"manifest.json")
    if digest(build/"graph.nt")!=manifest["graph_sha256"]:raise ValueError("Graph hash drift")
    if digest(base/"graph.nt")!=BASE_SHA:raise ValueError("Baseline graph drift")
    g=Graph().parse(build/"graph.nt",format="nt");old=Graph().parse(base/"graph.nt",format="nt")
    results=[]
    def check(name,condition,actual=None):
        results.append({"name":name,"pass":bool(condition),"actual":actual})
    check("all_old_triples_preserved",not(set(old)-set(g)),len(old))
    oldclaims=set(old.subjects(RDF.type,M.Claim))
    check("historical_claims_unchanged",all(set(old.predicate_objects(s))==set(g.predicate_objects(s)) for s in oldclaims),len(oldclaims))
    records=set(g.subjects(RDF.type,M.MolecularRecord));check("registered_records_7965",len(records)==7965,len(records))
    for key,pred,n in (("tri_view_payload",M.triViewReady,7710),("excluded",M.releaseExcluded,255),("mechanism_comparable",M.mechanismComparable,0)):
        actual=sum(str(g.value(r,pred))=="true" for r in records);check(key,actual==n,actual)
    check("no_owl_sameAs",not list(g.triples((None,OWL.sameAs,None))))
    check("no_direct_exact_physical_sample_claim",not list(g.triples((None,M.exactPhysicalSample,None))))
    check("no_physical_specimen_invention",not list(g.subjects(RDF.type,M.PhysicalSample)))
    check("no_admitted_flux_observation",not list(g.subjects(RDF.type,M.FluxObservation)))
    check("no_DNA_abundance_imputed_from_RNA",not list(g.subjects(RDF.type,M.AbundanceObservation)))
    check("no_complete_pathway_claim",not any(str(x)=="true" for x in g.objects(None,M.pathwayComplete)))
    check("no_credit_decision",not list(g.subjects(RDF.type,M.VerificationDecision)))
    loci=load(bundle/"sequences-v2/loci.json")
    for r in loci:
        n=iri("molecular-feature",r["locus_id"]);verified=r.get("translation_check",{}).get("admit_encodes",False)
        if str(g.value(n,M.geneId))!=r["gene_id"]:raise ValueError("Graph/source feature identity mismatch")
        if bool(list(g.objects(n,M.encodes)))!=verified:raise ValueError("Graph encodes edge disagrees with translation receipt")
        if r["sequence_available"]:
            p=g.value(n,M.forProtein)
            if str(g.value(p,M.sequenceDigest))!=r["protein_sequence_sha256"]:raise ValueError("Protein digest mismatch")
    check("all_selected_locus_admission_receipts_match",True,len(loci))
    cov=pq.ParquetFile(bundle/"staging-v2/panel-coverage.parquet").read().to_pylist()
    expected=Counter((r["lane_id"],r["habitat"],r["family_id"],r["status"]) for r in cov)
    actual={tuple(str(g.value(n,p)) for p in (M.laneId,M.habitatLabel,M.familyId,M.panelStatus)):int(g.value(n,M.rowCount)) for n in g.subjects(RDF.type,M.PanelCoverageSummary)}
    check("full_coverage_aggregate_source_parity",dict(expected)==actual,sum(actual.values()))
    rna=load(bundle/"context-v2/gene-rna-observations.json")
    for r in rna:
        n=iri("rna-observation",r["observation_id"])
        if str(g.value(n,M.processedValue))!=r["value"] or str(g.value(n,M.assayModality))!="source_processed_RNA":raise ValueError("RNA source scale/value mismatch")
    check("all_RNA_values_and_modalities_exact",True,len(rna))
    measurements=load(bundle/"context-v2/typed-measurements.json")
    for r in measurements:
        n=iri("source-measurement",r["lane_id"],r["measurement_id"])
        missing=r["value"] is None
        if ((n,RDF.type,M.SourceMeasurementGap) in g)!=missing:raise ValueError("Missing measurement admission disagrees with source")
        if missing and list(g.objects(n,M.processedValue)):raise ValueError("Missing measurement was numerically imputed")
        if not missing and str(g.value(n,M.processedValue))!=r["value"]:raise ValueError("Typed source measurement value mismatch")
    check("all_selected_numeric_and_missing_measurements_retained",True,len(measurements))
    check("source_measurements_not_claimed_as_paired_flux",all("unresolved" in str(g.value(n,M.sampleLinkageStatus)) or "ambiguous" in str(g.value(n,M.sampleLinkageStatus)) for n in g.subjects(RDF.type,M.SourceMeasurement)))
    classcounts=Counter(str(o).split("/")[-1] for s,o in g.subject_objects(RDF.type) if str(o).startswith(str(M)))
    relationcounts=Counter(str(p) for s,p,o in g)
    check("NTriples_reparse_content_hash",sha_bytes(canonical(g))==manifest["graph_sha256"])
    report={"status":"pass" if all(r["pass"] for r in results) else "fail","snapshot":manifest["snapshot"],"checks":results,"class_counts":dict(sorted(classcounts.items())),
        "relation_counts":dict(sorted(relationcounts.items())),"elapsed_seconds":round(time.monotonic()-started,3),"canonical_bytes":(build/"graph.nt").stat().st_size,
        "scientific_review":"Formal validation and source fidelity are not independently validated biological interpretation"}
    write_json(out/"real-graph-validation.json",report)
    if report["status"]!="pass":raise ValueError("Real graph safety check failed")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["source","graph"])
    p.add_argument("--repo",type=Path,default=Path.cwd());p.add_argument("--bundle",type=Path,default=Path("results/reports/mvo_molecular_evidence_20260926"))
    p.add_argument("--base",type=Path,default=Path("ontology/build/atlas-20260926-validated"));p.add_argument("--build",type=Path);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();r=source_rows(a.repo,a.bundle,a.out) if a.mode=="source" else real_graph(a.build,a.bundle,a.base,a.out)
    print(json.dumps({k:v for k,v in r.items() if k not in ("relation_counts","tables","checks")},indent=2))
