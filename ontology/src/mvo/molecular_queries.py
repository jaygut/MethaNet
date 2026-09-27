"""Execute the fixed MQ01-MQ18 pack, full source parity and fair SQL controls.

Display limits are applied after complete result capture; they never become
aggregate denominators. This local tool accepts IDs and typed filters, not
arbitrary query text or caller-supplied principals.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
import logging
from pathlib import Path
import re
import statistics
import time
import duckdb
from neo4j import Query,READ_ACCESS
import pyarrow.parquet as pq
from rdflib import Graph
from rdflib.namespace import RDF

from .model import M,C,ROOT,iri,digest,sha_bytes
from .molecular_stage import habitat,MUCC
from .molecular_graph import region
from .molecular_audit import write_json,write_tsv
from .molecular_domain import fingerprint,PRINCIPAL
from .molecular_sql import export as export_sql
from .projection import connect
from .local_runtime import credentials
from .query import AccessDenied

SHOWCASES={"MQ03","MQ09","MQ18"}
QUERY_IDS={f"MQ{i:02}" for i in range(1,19)}
QUERY_DIR=ROOT/"queries/molecular"
BOUNDARY="Source-addressed candidates and validation readiness; no empirical process/flux, risk-tier, carbon-benefit, transfer or credit-approval claim."


def load(path):return json.loads(path.read_text())


def authorize(principal,purpose):
    if principal.get("tenant")!="emergentbiome" or "atlas" not in principal.get("projects",[]) or purpose!="internal_review" or purpose not in principal.get("purposes",[]):
        raise AccessDenied("This query pack is restricted to the authenticated local internal-review principal")


def validate_query(qid,text):
    if qid not in QUERY_IDS:raise ValueError("Unknown fixed query ID")
    stripped=re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"","''",text)
    if re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD|FOREACH|GRANT|REVOKE)\b",stripped,re.I):raise ValueError("Mutating query rejected")
    if re.search(r"\bCALL\b(?!\s*\{)",stripped,re.I):raise ValueError("Procedure calls are not allowed")
    if "$snapshot" not in text or "status:'validated'" not in text or "$purpose" not in text:raise ValueError("Missing snapshot/purpose guard")


def normalize(rows):
    out=[]
    for row in rows:
        r=dict(row)
        # These two collections are mathematical sets; component lists retain
        # query ordering so labels and support values cannot be mispaired.
        for key in ("lanes","evidence_ids"):
            if key in r:r[key]=sorted(r[key])
        out.append(r)
    return out


def compare_subset(actual,expected,columns):
    actual=[{k:r.get(k) for k in columns} for r in actual]
    expected=[{k:r.get(k) for k in columns} for r in expected]
    if fingerprint(actual)!=fingerprint(expected):
        aset=Counter(json.dumps(x,sort_keys=True) for x in actual);eset=Counter(json.dumps(x,sort_keys=True) for x in expected)
        raise ValueError("Source parity mismatch: "+json.dumps({"actual_rows":len(actual),"expected_rows":len(expected),"unexpected":list((aset-eset).elements())[:2],"missing":list((eset-aset).elements())[:2]}))
    return {"status":"pass","mode":"full_result_multiset_source_parity_at_declared_grain","rows":len(actual),"compared_columns":columns,"projection_sha256":fingerprint(actual)}


class SourceParity:
    def __init__(self,bundle,build,base):
        self.bundle=bundle;self.build=build;self.base=base
        self.events=load(bundle/"staging-v2/selected-annotation-events.json")
        self.coverage=pq.ParquetFile(bundle/"staging-v2/panel-coverage.parquet").read().to_pylist()
        self.mags=load(bundle/"staging-v2/selected-mag-context.json")
        self.selected={(r["lane_id"],r["proteome_id"]) for r in self.mags}
        self.loci=load(bundle/"sequences-v2/loci.json")
        self.by_event={eid:r for r in self.loci for eid in r["annotation_event_ids"]}
        self.rna=load(bundle/"context-v2/gene-rna-observations.json")
        self.context={(r["lane_id"],r["sample_id"]):r for r in load(bundle/"context-v2/sample-contexts.json")}
        self.links=load(bundle/"context-v2/selected-sample-mag-links.json")
        self.measurements=load(bundle/"context-v2/typed-measurements.json")
        self.panel=load(ROOT/"mappings/molecular-panel-0.1.0.json")
        self.families={r["id"]:r for r in self.panel["families"]}
        self.covmap={(r["lane_id"],r["proteome_id"],r["family_id"]):r for r in self.coverage}
        self.den={r["lane_id"]:r["registered_lane_denominator"] for r in self.coverage}
        self.eventmap={e["event_id"]:e for e in self.events}
        self.magmap={(r["lane_id"],r["proteome_id"]):r for r in self.mags}
        negative={}
        for r in self.coverage:
            if r["status"] not in ("present","ambiguous"):negative.setdefault((r["lane_id"],r["family_id"],r["status"]),r)
        self.evalkeys={(r["lane_id"],r["proteome_id"],r["family_id"]) for r in self.coverage if (r["lane_id"],r["proteome_id"]) in self.selected}
        self.evalkeys.update((r["lane_id"],r["proteome_id"],r["family_id"]) for r in negative.values())

    def event(self,e):
        key=(e["lane_id"],e["proteome_id"]);l=self.by_event[e["event_id"]]
        return dict(evidence_id=str(iri("molecular-annotation",e["event_id"])),lane=key[0],habitat=habitat(key[0],self.magmap[key]["release"]),
            proteome_id=key[1],registered_lane_mags=self.den[key[0]],assay="source_DRAM_not_pipeline_normalized" if key[0]==MUCC else "pipeline_selected_run_KOfam_or_dbCAN",
            family=e["family_id"],accession=e["accession"],gene_id=e["gene_id"],identity_state=l["sequence_identity_status"],
            source_curation=e.get("historical_curation_status","source_candidate_pending_review"),source_sha256=e["source_sha256"],source_row="zero_based_data_record="+str(e["source_row_ordinal"]))

    def check(self,qid,actual):
        expected=[];columns=[]
        if qid in ("MQ01","MQ05"):
            counts=Counter((r["lane_id"],r["habitat"],r["family_id"],r["status"],r["assay_contract"]) for r in self.coverage)
            chosen=Counter((r["lane_id"],habitat(r["lane_id"],r["release"])) for r in self.mags)
            for (lane,hab,fam,status,assay),n in counts.items():
                if qid=="MQ05" and hab=="non_wetland_rumen_control":continue
                expected.append(dict(lane=lane,habitat=hab,family=fam,status=status,mag_family_rows=n,registered_lane_mags=self.den[lane],assay=assay,selected_mags_in_habitat=chosen[(lane,hab)]))
            columns=["lane","habitat","family","status","mag_family_rows","registered_lane_mags","assay"]+(["selected_mags_in_habitat"] if qid=="MQ01" else [])
        elif qid=="MQ02":
            for e in self.events:
                r=self.event(e);l=self.by_event[e["event_id"]];r.update(translation=l.get("translation_check",{}).get("status","not_testable"),protein_sha256=l.get("protein_sequence_sha256"),tool=e["tool"],threshold=str(e["threshold"]))
                expected.append(r)
            columns=["evidence_id","lane","habitat","proteome_id","registered_lane_mags","assay","family","accession","gene_id","identity_state","translation","protein_sha256","tool","threshold","source_sha256","source_row"]
        elif qid=="MQ03":
            for e in self.events:
                if e["family_id"] not in ("mcr_complex","cummo_complex"):continue
                for alt in self.families[e["family_id"]]["competing_interpretations"]:
                    expected.append({**self.event(e),"competing_interpretation":alt,"discrimination_action":self.families[e["family_id"]]["false_positive_control"]})
            columns=list(expected[0])
        elif qid in ("MQ04","MQ13"):
            for key in self.evalkeys:
                r=self.covmap[key]
                if qid=="MQ13" and key[2]!="cummo_complex":continue
                expected.append(dict(evidence_id=str(iri("panel-evaluation","0.1.0",*key)),lane=key[0],proteome_id=key[1],habitat=r["habitat"],family=key[2],status=r["status"],
                    registered_lane_mags=self.den[key[0]],assay=r["assay_contract"],observed_components=sorted(filter(None,r["observed_components"].split(";"))),component_support=r["component_support"],complete_pathway_admitted="false"))
            columns=["evidence_id","lane","proteome_id","habitat","family","status","registered_lane_mags","assay"]+(["observed_components","component_support","complete_pathway_admitted"] if qid=="MQ04" else [])
        elif qid in ("MQ06","MQ09"):
            for r in self.rna:
                e=self.eventmap[r["annotation_event_id"]];c=self.context[(r["lane_id"],r["sample_id"])];l=self.by_event[e["event_id"]]
                expected.append(dict(evidence_id=str(iri("rna-observation",r["observation_id"])),lane=r["lane_id"],proteome_id=r["proteome_id"],family=e["family_id"],gene_id=r["gene_id"],identity_state=l["sequence_identity_status"],
                    sample_id=r["sample_id"],modality=r["modality"],value=r["value"],unit=r["unit"],normalization=r["normalization"],assay_reconciliation=r["sample_assay_reconciliation_status"],
                    source_sha256=r["source"]["sha256"],source_row="row="+str(r["source_row_ordinal"])+";column="+r["source_column"],
                    source_site_label=c["raw_site_label"],region=c["region"],date_label=c["collection_date_label"] or "unresolved",depth_label=c["depth_label"] or "unresolved",
                    assay=r["sample_assay_reconciliation_status"],blocking_join="no_exact_physical_sample_flux_pair",accepted_exact_flux_pairs=0,
                    allowed_wording="Source RNA plus study context; no physically paired molecular-flux observation"))
            columns=["evidence_id","lane","proteome_id","gene_id","identity_state","sample_id","modality","value","source_sha256","source_row"]
            columns+=(["unit","normalization","assay_reconciliation"] if qid=="MQ06" else ["source_site_label","region","date_label","depth_label","assay","blocking_join","accepted_exact_flux_pairs","allowed_wording"])
        elif qid=="MQ07":
            for r in self.links:
                c=self.context[(r["lane_id"],r["sample_id"])]
                expected.append(dict(evidence_id=str(iri("identity",r["edge_id"])),lane=r["lane_id"],proteome_id=r["proteome_id"],sample_id=r["sample_id"],relation=r["relation"],identity_state=r["mapping_confidence"],
                    site_label=c["raw_site_label"],date_label=c["collection_date_label"] or "unresolved",depth_label=c["depth_label"] or "unresolved",assay=c["assay_reconciliation_status"],geography_resolution=c["geographic_precision"],
                    physical_pair_status="no_exact_physical_sample_flux_pair",source_row="zero_based_data_record="+str(r["source_row_ordinal"])))
            columns=list(expected[0])
        elif qid=="MQ08":
            for r in self.measurements:
                expected.append(dict(evidence_id=str(iri("source-measurement",r["lane_id"],r["measurement_id"])),lane=r["lane_id"],variable=r["kind"],quantity_kind=r["quantity_kind"],value=r["value"],unit=r["unit"],method=r["method"],
                    value_status="source_missing_not_zero" if r["value"] is None else "reported_numeric",
                    date_label=r["source_date_label"] or "unresolved",depth_label=str(r.get("source_depth_label","")) or "unresolved",site=r["source_site"],spatial_support=r["spatial_support"],sample_linkage=r["sample_join_status"],
                    source_metadata_sample=r.get("sample_id"),source_sha256=r["source"]["sha256"],source_row="zero_based_data_record="+str(r["source_row_ordinal"])+(";column="+r["source_column"] if r.get("source_column") else "")))
            columns=list(expected[0])
        elif qid in ("MQ10","MQ18"):
            for lane,pid in self.selected:
                for code in ("mechanism-review","ecological-pair","rights-review"):
                    expected.append(dict(packet_id=str(iri("evidence-packet","0.1.0",lane,pid)),evidence_id=str(iri("evidence-packet","0.1.0",lane,pid)) if qid=="MQ18" else str(iri("claim","molecular-panel-review",lane,pid)),
                        lane=lane,proteome_id=pid,gap_id=str(iri("gap","packet",lane,pid,code)),review_state=str(C.pending),rights="internal_review_only"))
            columns=["evidence_id","lane","proteome_id","gap_id","review_state","rights"]+(["packet_id"] if qid=="MQ10" else [])
        elif qid in ("MQ11","MQ17"):
            for code in ("mechanism-review","ecological-pair","rights-review"):
                expected.append(dict(evidence_id=str(iri("validation-action",code)),affected_paths=len(self.selected)))
            for lane,count in Counter(r["lane_id"] for r in self.links).items():expected.append(dict(evidence_id=str(iri("validation-action","source-sample-reconciliation",lane)),affected_paths=count))
            columns=["evidence_id","affected_paths"]
        elif qid=="MQ12":
            old=load(self.base/"manifest.json");new=load(self.build/"manifest.json");oldproj=load(self.base/"neo4j/projection.json");newproj=load(self.build/"neo4j/projection.json")
            og=Graph().parse(self.base/"graph.nt",format="nt");nc=sum(len(list(og.predicate_objects(s))) for s in og.subjects(RDF.type,M.Claim))
            expected=[dict(old_snapshot=old["snapshot"],new_snapshot=new["snapshot"],old_resources=oldproj["resources"],new_resources=newproj["resources"],resource_growth=newproj["resources"]-oldproj["resources"],
                original_claim_statements=nc,preserved_claim_statements=nc,old_rdf_sha256=old["graph_sha256"],new_rdf_sha256=new["graph_sha256"])]
            columns=list(expected[0])
        elif qid=="MQ14":
            groups=defaultdict(list)
            for r in self.coverage:
                if (r["lane_id"],r["proteome_id"]) in self.selected and r["status"] in ("present","ambiguous"):groups[(r["lane_id"],r["proteome_id"])].append(r)
            for key,rs in groups.items():
                rs.sort(key=lambda r:r["family_id"])
                expected.append(dict(evidence_id=str(iri("evidence-packet","0.1.0",*key)),lane=key[0],proteome_id=key[1],habitat=rs[0]["habitat"],
                    co_occurring_candidate_families=[r["family_id"] for r in rs],component_support=[r["component_support"] for r in rs],assay=rs[0]["assay_contract"],registered_lane_mags=self.den[key[0]]))
            columns=list(expected[0])
        elif qid=="MQ15":
            groups=defaultdict(list)
            for e in self.events:groups[(e["lane_id"],e["proteome_id"],e["family_id"])].append(e)
            for key,es in groups.items():
                retained=sum(not self.families[e["family_id"]]["ambiguity"] and e.get("historical_included") is not False and self.by_event[e["event_id"]]["sequence_identity_status"]=="source_header_and_coordinate_translation_verified" for e in es)
                expected.append(dict(lane=key[0],proteome_id=key[1],family=key[2],source_candidate_events=len(es),events_after_ambiguity_and_identity_exclusion=retained,
                    evidence_ids=sorted(str(iri("molecular-annotation",e["event_id"])) for e in es),sensitivity_result="candidate_card_loses_all_selected_molecular_support" if not retained else "selected_support_remains_not_validated_function"))
            columns=list(expected[0])
        elif qid=="MQ16":
            env=defaultdict(list)
            for m in self.measurements:
                if m.get("sample_id") and m["value"] is not None:env[(m["lane_id"],m["sample_id"])].append(m)
            for key,c in self.context.items():
                ms=sorted(env[key],key=lambda r:r["kind"])
                expected.append(dict(evidence_id=str(iri("source-context",c["context_id"])),lane=key[0],sample_id=key[1],site=c["raw_site_label"],date_label=c["collection_date_label"] or "unresolved",depth_label=c["depth_label"] or "unresolved",
                    assay=c["assay_reconciliation_status"],source_matched_environment_variables=[r["kind"] for r in ms],measurement_evidence_ids=[str(iri("source-measurement",r["lane_id"],r["measurement_id"])) for r in ms],
                    context_identity=c["grain"],blocking_join="no_exact_physical_sample_flux_pair"))
            columns=list(expected[0])
        else:raise ValueError("Missing source parity implementation")
        return compare_subset(actual,expected,columns)


def plan_hits(plan):
    if not plan:return 0
    return int(plan.get("dbHits",0))+sum(plan_hits(c) for c in plan.get("children",[]))


def run(build,bundle,base,out,display_limit=25):
    authorize(PRINCIPAL,"internal_review")
    if not isinstance(display_limit,int) or not 1<=display_limit<=1000:raise ValueError("Display limit out of bounds")
    if out.exists():raise FileExistsError("Immutable query result run already exists")
    out.mkdir(parents=True)
    manifest=load(build/"manifest.json");old=load(base/"manifest.json")
    if digest(build/"graph.nt")!=manifest["graph_sha256"]:raise ValueError("Snapshot graph drift")
    params=dict(snapshot=manifest["snapshot"],old_snapshot=old["snapshot"],purpose="internal_review",lane="",habitat="",family="",site="",proteome="",process="",date_prefix="",depth="")
    g=Graph().parse(build/"graph.nt",format="nt");sqlstart=time.monotonic();sqltables=export_sql(g,out/"sql-tables")
    setup_seconds=time.monotonic()-sqlstart
    sql=duckdb.connect();sql.execute("SET threads=2");sql.execute("SET memory_limit='768MB'")
    for name in sqltables:sql.read_parquet(str(out/"sql-tables"/(name+".parquet"))).create_view(name)
    parity=SourceParity(bundle,build,base);auth=credentials();receipts=[]
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    with connect(auth["uri"],auth["user"],auth["password"]) as driver,driver.session(default_access_mode=READ_ACCESS) as session:
        state=session.run("MATCH (s:MVOSnapshot {id:$sid,status:'validated'}),(d:MVODomainSnapshot {id:$sid,status:'validated',version:'0.1.0'}) RETURN s.rdf_sha256 AS sha,d.rdf_sha256 AS domain_sha",sid=params["snapshot"]).single()
        if not state or state["sha"]!=manifest["graph_sha256"] or state["domain_sha"]!=manifest["graph_sha256"]:raise ValueError("Unvalidated or mismatched canonical/domain snapshot")
        for qid in sorted(QUERY_IDS):
            text=(QUERY_DIR/(qid+".cypher")).read_text();validate_query(qid,text)
            started=time.monotonic();plan=session.run(Query("EXPLAIN "+text,timeout=60),**params).consume()
            explain_seconds=time.monotonic()-started
            started=time.monotonic();result=session.run(Query(text,timeout=60),**params);rows=normalize(result.data());summary=result.consume();elapsed=time.monotonic()-started
            if summary.query_type!="r":raise ValueError("Query was not read-only")
            if len(rows)>100000:raise ValueError("Full result exceeds declared row budget")
            write_json(out/(qid+"-full.json"),rows)
            receipt=dict(query=qid,status="executed_pending_parity",snapshot=params["snapshot"],query_sha256=digest(QUERY_DIR/(qid+".cypher")),parameters=params,full_result_count=len(rows),
                full_result_sha256=digest(out/(qid+"-full.json")),canonical_result_hash=fingerprint(rows),display_limit=display_limit,display_rows=rows[:display_limit],
                denominator_scope="Full registered source coverage or explicitly selected source slice; display limit never used in aggregation",claim_boundary=BOUNDARY,
                elapsed_seconds=elapsed,explain_seconds=explain_seconds,query_type=summary.query_type,plan=plan.plan,notifications=summary.notifications or [])
            try:
                receipt["source_parity"]=parity.check(qid,rows)
                if qid in SHOWCASES:
                    st=(QUERY_DIR/(qid+".sql")).read_text();sp={k:params[k] for k in set(re.findall(r"\$([A-Za-z_]+)",st))}
                    t=time.monotonic();sr=normalize(sql.execute(st,sp).fetch_arrow_table().to_pylist());sqlcold=time.monotonic()-t
                    if fingerprint(sr)!=fingerprint(rows):raise ValueError("Fair SQL full-row answer parity failed")
                    warmgraph=[];warmsql=[]
                    for _ in range(3):
                        t=time.monotonic();again=normalize(session.run(Query(text,timeout=60),**params).data());warmgraph.append(time.monotonic()-t)
                        t=time.monotonic();again_sql=normalize(sql.execute(st,sp).fetch_arrow_table().to_pylist());warmsql.append(time.monotonic()-t)
                        if fingerprint(again)!=fingerprint(rows) or fingerprint(again_sql)!=fingerprint(rows):raise ValueError("Non-idempotent read query")
                    profile_result=session.run(Query("PROFILE "+text,timeout=60),**params);profile_rows=normalize(profile_result.data());profile=profile_result.consume().profile
                    if fingerprint(profile_rows)!=fingerprint(rows):raise ValueError("PROFILE result parity mismatch")
                    write_json(out/(qid+"-profile.json"),profile)
                    write_json(out/(qid+"-sql-full.json"),sr)
                    receipt["sql_baseline"]=dict(status="full_row_exact_parity",normalized_fact_table_setup_seconds=setup_seconds,sql_query_sha256=digest(QUERY_DIR/(qid+".sql")),
                        sql_first_seconds=sqlcold,graph_warm_seconds=warmgraph,sql_warm_seconds=warmsql,graph_warm_median_seconds=statistics.median(warmgraph),sql_warm_median_seconds=statistics.median(warmsql),
                        profile_db_hits=plan_hits(profile),cypher_lines=len(text.splitlines()),sql_lines=len(st.splitlines()),cypher_relationship_patterns=text.count("->"),sql_join_tokens=len(re.findall(r"\bJOIN\b",st,re.I)),
                        evidence_provenance_columns_equal=True,interpretation="Same admitted normalized facts; setup separate; 3 warm sequential local timings, not a concurrency benchmark or human effort study")
                receipt["status"]="pass_execution_and_declared_grain_parity_not_biological_validation"
            except Exception as error:
                receipt["status"]="fail";receipt["error"]=str(error);write_json(out/(qid+"-receipt.json"),receipt);raise
            write_json(out/(qid+"-receipt.json"),receipt)
            receipts.append({k:v for k,v in receipt.items() if k not in ("display_rows","plan","notifications")})
            print(qid,len(rows),"PASS",round(elapsed,4),"seconds",flush=True)
    final=dict(status="pass",snapshot=params["snapshot"],queries=receipts,sql_showcases=sorted(SHOWCASES),external_release="not_authorized_pending_scientific_and_rights_review")
    write_json(out/"query-pack-validation.json",final)
    write_tsv(out/"query-summary.tsv",[{"query":r["query"],"rows":r["full_result_count"],"seconds":r["elapsed_seconds"],"status":r["status"],"result_sha256":r["full_result_sha256"]} for r in receipts])
    return final


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--build",type=Path,required=True);p.add_argument("--bundle",type=Path,default=Path("results/reports/mvo_molecular_evidence_20260926"))
    p.add_argument("--base",type=Path,default=Path("ontology/build/atlas-20260926-validated"));p.add_argument("--out",type=Path,required=True);p.add_argument("--display-limit",type=int,default=25)
    a=p.parse_args();r=run(a.build,a.bundle,a.base,a.out,a.display_limit);print(json.dumps({"status":r["status"],"queries":len(r["queries"]),"snapshot":r["snapshot"]},indent=2))
