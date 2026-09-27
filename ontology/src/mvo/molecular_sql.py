"""Fair normalized relational baseline over the exact admitted graph facts.

These are entity/relationship tables, not precomputed query answers. Both the
domain graph and SQL share admission work; source parity is tested separately.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from rdflib.namespace import RDF, RDFS
from .model import M,PROV,digest
from .molecular_audit import write_json


def export(g,out):
    out.mkdir(parents=True,exist_ok=True)
    tables=defaultdict(list)
    def val(n,p):
        values=list(g.objects(n,p))
        if len(values)>1:raise ValueError(f"Scalar SQL field is multivalued: {n}/{p}")
        return str(values[0]) if values else None
    def fields(n,mapping):return {key:val(n,p) for key,p in mapping.items()}
    for n in g.subjects(RDF.type,M.AnnotationAssertion):
        if not g.value(n,M.forFamily):continue
        r={"evidence_id":str(n),**fields(n,dict(lane=M.laneId,habitat=M.habitatLabel,proteome_id=M.proteomeId,assay=M.assayContract,family=M.familyId,accession=M.accession,
                    locus_id=M.forLocus,source_id=PROV.wasDerivedFrom,source_curation=M.status,source_row=M.sourceRow))}
        r["registered_lane_mags"]=int(val(n,M.registeredDenominator));tables["annotations"].append(r)
        for a in g.objects(n,M.hasAlternative):tables["annotation_alternatives"].append({"annotation_id":str(n),"alternative_id":str(a)})
    for cls in (M.CalledLocus,M.SourceFeature):
        for n in g.subjects(RDF.type,cls):
            if g.value(n,M.geneId):tables["loci"].append({"id":str(n),**fields(n,dict(gene_id=M.geneId,identity_state=M.identityState))})
    for n in g.subjects(RDF.type,M.InterpretationAlternative):tables["alternatives"].append({"id":str(n),**fields(n,dict(label=RDFS.label,action=M.nextAction))})
    for n in set(g.subjects(M.sha256,None)):tables["sources"].append({"id":str(n),"sha256":val(n,M.sha256)})
    for n in g.subjects(RDF.type,M.MolecularSampleObservation):tables["rna"].append({"id":str(n),**fields(n,dict(locus_id=M.forLocus,sample_id=M.forSample,annotation_id=M.hasEvidenceItem,
        source_id=PROV.wasDerivedFrom,modality=M.assayModality,value=M.processedValue,assay=M.assayContract,source_row=M.sourceRow))})
    for n in g.subjects(RDF.type,M.SequencingSample):tables["samples"].append({"id":str(n),"sample_id":val(n,M.sourceIdentifier)})
    for n in g.subjects(RDF.type,M.SourceContextRecord):tables["contexts"].append({"id":str(n),**fields(n,dict(sample_id=M.forSample,source_site_label=M.sourceSiteLabel,
        region=M.regionLabel,date_label=M.sourceDateLabel,depth_label=M.sourceDepthLabel,blocking_join=M.sampleLinkageStatus))})
    for n in g.subjects(RDF.type,M.PanelEvaluation):tables["evaluations"].append({"id":str(n),**fields(n,dict(status=M.panelStatus,assay=M.assayContract)),"registered_lane_mags":int(val(n,M.registeredDenominator))})
    for n in g.subjects(RDF.type,M.EvidencePacket):
        rec=g.value(n,M.forRecord)
        tables["packets"].append({"id":str(n),"proteome_id":val(rec,M.proteomeId),**fields(n,dict(lane=M.laneId,habitat=M.habitatLabel,region=M.regionLabel,rights=M.rightsStatus,allowed_wording=M.allowedWording))})
        for ev in g.objects(n,M.hasEvaluation):tables["packet_evaluations"].append({"packet_id":str(n),"evaluation_id":str(ev)})
        for gap in g.objects(n,M.blockedBy):tables["packet_gaps"].append({"packet_id":str(n),"gap_id":str(gap)})
        for e in g.objects(n,M.hasEvidenceItem):
            if (e,RDF.type,M.Claim) in g:tables["packet_claims"].append({"packet_id":str(n),"claim_id":str(e)})
            if (e,RDF.type,M.AnnotationAssertion) in g:tables["packet_annotations"].append({"packet_id":str(n),"annotation_id":str(e)})
    for n in g.subjects(RDF.type,M.Claim):tables["claims"].append({"id":str(n),"review_state":val(n,M.reviewState)})
    for n in g.subjects(RDF.type,M.ValidationGap):tables["gaps"].append({"id":str(n),**fields(n,dict(reason=M.reason,action=M.nextAction))})
    receipt={}
    for name,rows in sorted(tables.items()):
        rows.sort(key=lambda r:tuple(str(r[k]) for k in sorted(r)))
        path=out/(name+".parquet");pq.write_table(pa.Table.from_pylist(rows),path,compression="zstd")
        receipt[name]={"rows":len(rows),"sha256":digest(path)}
    write_json(out/"relational-projection.json",{"status":"normalized_admitted_facts_not_precomputed_answers","tables":receipt})
    return receipt
