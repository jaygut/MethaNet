"""Add a bounded, source-addressed molecular layer to an immutable atlas graph.

Qualified annotation/context assertions are never materialized as biological
identity, activity or molecular-to-flux implications. Full coverage stays in a
hashed Parquet ledger; aggregate summaries and a deterministic review slice are
explicitly distinguished.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
from rdflib import Literal
from rdflib.namespace import RDF, RDFS, XSD

from .model import M, C, PROV, ROOT, iri, digest, add, node, graph, assertion, gap, bind_policy, canonical
from .ingest import Inventory
from .molecular_audit import write_json, write_tsv
from .molecular_stage import MUCC, habitat, stable_id
from .release import materialize
from .validation import validate_graph

BASE_SHA = "5fce9c99a86c6e8fc4fd751bfd6497ff45a2b905d89b75bb7bf732504d6559c5"
REGIONS = {"msm_china_2025":"Southeast China; source-study scope", "futian_mangrove_2026_qi":"Futian Reserve, Shenzhen, China", MUCC:"Old Woman Creek, Ohio, USA", "poc_core":"heterogeneous source context; not independently replicated geography"}
VERIFIED = {"exact_translation", "exact_with_allowed_initiator_methionine"}


def read(path):
    return json.loads(path.read_text())


def serial(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))


def region(lane):
    return REGIONS[lane]


def record(lane, pid):
    return iri("molecular-record", lane, pid)


def sample(lane, sid):
    return iri("sequencing-sample", lane, sid)


def build(repo, base, bundle, recorded_at, pilot=False):
    started = time.monotonic()
    if digest(base/"graph.nt") != BASE_SHA:
        raise ValueError("Protected baseline graph hash changed")
    g = graph().parse(base/"graph.nt", format="nt")
    original = set(g)
    stage, seq, context = (bundle/x for x in ("staging-v2", "sequences-v2", "context-v2"))
    # Validate staged output hashes before using any normalized fact. The source
    # files are independently rehashed by Inventory below.
    for folder, receipt in ((stage,"stage-receipt.json"),(seq,"sequence-receipt.json"),(context,"context-receipt.json")):
        r = read(folder/receipt)
        for name, expected in r.get("outputs",{}).items():
            if digest(folder/name) != expected:
                raise ValueError(f"Staged artifact drift: {folder.name}/{name}")
    ctx = dict(policy=iri("policy","internal-evidence-review-v1"), recorded_at=recorded_at,
               uncertainty=iri("uncertainty","unquantified-molecular-transfer"))
    ctx["agent"] = node(g, iri("agent","molecular-adapter-0.2.0"), M.Agent)
    ctx["method"] = node(g, iri("method","molecular-evidence-0.2.0"), M.Method)
    add(g,ctx["method"],M.version,"0.2.0")
    ctx["scope"] = node(g,iri("scope","molecular-evidence-20260926"),M.Scope)
    add(g,ctx["scope"],M.scopeDescription,"Bounded source-annotation and translation evidence with candidate ecological context; no exact molecular/flux pairs, validated activity, risk tiers or registry approval.")
    inv = Inventory(repo,g,ctx)
    # Every source cited by the successor has an addressable artifact. Existing
    # provenance triples are preserved byte-semantically, not regenerated.
    for entry in read(base/"manifest.json")["sources"]:
        inv.file(repo/entry["path"], entry["sha256"])
    inv.file(base/"graph.nt", BASE_SHA)
    inv.file(base/"manifest.json")
    for folder in (stage,seq,context):
        for src in read(folder/"sources.json"):
            inv.file(repo/src["path"], src["sha256"])
        for p in sorted(folder.glob("*.json")):
            if p.name != "staging-checkpoint.json":
                inv.file(p)
    source = lambda s: inv.file(repo/s["path"],s["sha256"])
    locsource = inv.file(seq/"loci.json")
    annsource = inv.file(stage/"selected-annotation-events.json")
    magsource = inv.file(stage/"selected-mag-context.json")
    covsource = inv.file(stage/"panel-coverage.parquet")
    panelpath = ROOT/"mappings/molecular-panel-0.1.0.json"
    panel_data = read(panelpath)
    panel = inv.file(panelpath, cls=M.MolecularPanel)
    node(g,panel,M.MolecularPanel)  # Inventory may already cache its Artifact type.
    add(g,panel,M.version,panel_data["version"])
    add(g,panel,M.independentlyReviewed,False)
    add(g,panel,M.displaySelectionRule,panel_data["selection"]["rule"])
    families = {f["id"]:f for f in panel_data["families"]}
    family_nodes = {}
    for fid,f in families.items():
        n = node(g,iri("panel-family",panel_data["version"],fid),M.PanelFamilyConcept,f["label"])
        family_nodes[fid]=n
        for p,v in ((M.familyId,fid),(M.forPanel,panel),(M.allowedWording,"Annotation-derived candidate only. "+f["direction_limit"]),
                    (M.nextAction,f["false_positive_control"]),(M.componentSupport,f["component_logic"]),
                    (M.sourcePayload,serial(f)),(PROV.wasDerivedFrom,panel)):
            add(g,n,p,v)
        for a in f["competing_interpretations"]:
            alt=node(g,iri("interpretation-alternative",fid,a),M.InterpretationAlternative,a)
            add(g,alt,M.nextAction,f["false_positive_control"])
            add(g,alt,PROV.wasDerivedFrom,panel)
            add(g,n,M.hasAlternative,alt)

    magrows=read(stage/"selected-mag-context.json")
    selected={(r["lane_id"],r["proteome_id"]) for r in magrows}
    if pilot:
        selected=set()
        for lid in panel_data["selection"]["pilot_lanes"]:
            selected.update(sorted((r["lane_id"],r["proteome_id"]) for r in magrows if r["lane_id"]==lid)[:2])
    magmap={(r["lane_id"],r["proteome_id"]):r for r in magrows}
    for key in sorted(selected):
        row=magmap[key];w=row["warehouse"];n=record(*key)
        for p,v in ((M.habitatLabel,habitat(key[0],row["release"])),(M.regionLabel,region(key[0])),
                    (M.taxonomicContext,w.get("gtdb_classification",w.get("atlas_taxonomy_lineage","unrecorded"))),
                    (M.taxonomyVersion,w.get("gtdb_release") or "source_DRAM_plus_KBase_214.1_not_R232"),
                    (M.sourceRow,"selected-mag-context.json key="+serial(key)),(PROV.wasDerivedFrom,magsource)):
            add(g,n,p,v)
        for p,k,fallback in ((M.magCompleteness,"checkm2_completeness","bin_completeness"),(M.magContamination,"checkm2_contamination","bin_contamination")):
            v=w.get(k,w.get(fallback))
            if v is not None: add(g,n,p,str(v),XSD.decimal)
    coverage=pq.ParquetFile(stage/"panel-coverage.parquet").read().to_pylist()
    aggregates=Counter((r["lane_id"],r["habitat"],r["family_id"],r["status"],r["assay_contract"]) for r in coverage)
    negatives={}
    for r in coverage:
        if r["status"] not in ("present","ambiguous"):
            negatives.setdefault((r["lane_id"],r["family_id"],r["status"]),r)
    admitted_eval={(r["lane_id"],r["proteome_id"],r["family_id"]) for r in coverage if (r["lane_id"],r["proteome_id"]) in selected}
    if not pilot:
        admitted_eval.update((r["lane_id"],r["proteome_id"],r["family_id"]) for r in negatives.values())
    evaluations={}
    lane_den={r["lane_id"]:r["registered_lane_denominator"] for r in coverage}
    for (lid,hab,fid,status,assay),count in sorted(aggregates.items()):
        n=node(g,iri("panel-coverage",panel_data["version"],lid,hab,fid,status),M.PanelCoverageSummary)
        for p,v in ((M.coverageLedger,covsource),(PROV.wasDerivedFrom,covsource),(M.forPanel,panel),(M.forFamily,family_nodes[fid]),
                    (M.laneId,lid),(M.habitatLabel,hab),(M.regionLabel,region(lid)),(M.familyId,fid),
                    (M.panelStatus,status),(M.rowCount,count),(M.registeredDenominator,lane_den[lid]),(M.assayContract,assay),
                    (M.compatibilityStatus,"not_validated_for_cross_source_quantitative_comparison")):
            add(g,n,p,v)
    for r in coverage:
        key=(r["lane_id"],r["proteome_id"],r["family_id"])
        if key not in admitted_eval: continue
        n=node(g,iri("panel-evaluation",panel_data["version"],*key),M.PanelEvaluation)
        evaluations[key]=n
        for p,v in ((M.forRecord,record(*key[:2])),(M.forFamily,family_nodes[key[2]]),(M.forPanel,panel),(M.familyId,key[2]),
                    (M.panelStatus,r["status"]),(M.reason,r["reason"]),(M.assayContract,r["assay_contract"]),
                    (M.componentSupport,r["component_support"]),(M.candidateEventCount,r["candidate_event_count"]),
                    (M.registeredDenominator,r["registered_lane_denominator"]),(M.pathwayComplete,False),(M.independentlyReviewed,False),
                    (M.coverageLedger,covsource),(PROV.wasDerivedFrom,covsource),(M.sourceRow,serial(key)),
                    (M.coverageState,"completed_covered" if r["status"]=="covered_no_accepted_hit" else "source_specific_partial_identity_coverage"),
                    (M.sourcePayload,serial(r)),(M.laneId,key[0]),(M.habitatLabel,r["habitat"]),(M.regionLabel,region(key[0])),
                    (M.compatibilityStatus,r["cross_lane_quantitative_comparability"])):
            add(g,n,p,v)
        for a in r["observed_components"].split(";"):
            add(g,n,M.componentAccession,a)
        add(g,record(*key[:2]),M.hasEvaluation,n)

    loci={};event_loci={};proteins={};locus_rows={}
    all_loci=read(seq/"loci.json")
    for r in all_loci:
        key=(r["lane_id"],r["proteome_id"])
        if key not in selected: continue
        check=r.get("translation_check",{});verified=check.get("admit_encodes",False)
        if verified and check["status"] not in VERIFIED: raise ValueError("Unverified translation admission")
        n=node(g,iri("molecular-feature",r["locus_id"]),M.CalledLocus if verified else M.SourceFeature)
        loci[r["locus_id"]]=n;locus_rows[r["locus_id"]]=r
        for eid in r["annotation_event_ids"]:event_loci[eid]=n
        for p,v in ((M.forRecord,record(*key)),(M.geneId,r["gene_id"]),(M.sourceNamespace,r["run_id"]),
                    (M.callerNamespace,r["caller_namespace"]),(M.callerVersion,r["caller_version"]),
                    (M.identityState,r["sequence_identity_status"]),(M.translationStatus,check.get("status","not_testable")),
                    (M.sequenceAvailable,r["sequence_available"]),(M.sourceRow,"loci.json locus_id="+r["locus_id"]),
                    (PROV.wasDerivedFrom,locsource),(M.laneId,key[0]),(M.proteomeId,key[1]),(M.runId,r["run_id"])):
            add(g,n,p,v)
        asm=source(r["assembly_source"])
        node(g,asm,M.AssemblyRepresentation)
        add(g,asm,M.forRecord,record(*key));add(g,n,M.inAssembly,asm)
        add(g,n,M.assemblyDigest,r["assembly_source"]["sha256"])
        if r["sequence_available"]:
            pn=node(g,iri("source-protein",r["protein_id"]),M.SequenceProtein)
            proteins[r["locus_id"]]=pn
            for p,v in ((M.sequenceAvailable,True),(M.sequenceDigest,r["protein_sequence_sha256"]),(M.sequenceLength,r["protein_length"]),
                        (M.sequenceNormalization,r["protein_digest_normalization"]),(PROV.wasDerivedFrom,source(r["protein_source"])),
                        (M.sourceRow,"FASTA record ordinal="+str(r["protein_record_ordinal"])+";id="+r["gene_id"]),(M.forRecord,record(*key)),(M.forLocus,n)):
                add(g,pn,p,v)
            add(g,n,M.forProtein,pn)
        if verified:
            co=r["coordinates"]
            contig=node(g,iri("contig",r["assembly_source"]["sha256"],co["contig_id"]),M.ContigRepresentation)
            add(g,contig,M.inAssembly,asm);add(g,contig,M.contigId,co["contig_id"]);add(g,contig,PROV.wasDerivedFrom,asm)
            for p,v in ((M.onContig,contig),(M.coordinateSystem,co["coordinate_system"]),(M.coordinateStart,co["start"]),
                        (M.coordinateEnd,co["end"]),(M.strand,co["strand"]),(M.geneticCodeTested,check["genetic_code_tested"]),
                        (M.sequenceDigest,check["nucleotide_interval_sha256"]),(M.encodes,proteins[r["locus_id"]])):
                add(g,n,p,v)
        else:
            q=gap(g,iri("gap","molecular-identity",r["locus_id"]),"Missing source sequence or translation inconsistency: "+check.get("status",r["sequence_identity_status"]),"Recover exact source caller/sequence/coordinate mapping; do not promote annotation label to verified locus.")
            add(g,q,PROV.wasDerivedFrom,locsource);add(g,n,M.blockedBy,q)

    annotations={};annotations_by_mag=defaultdict(list)
    for e in read(stage/"selected-annotation-events.json"):
        key=(e["lane_id"],e["proteome_id"])
        if key not in selected:continue
        fid=e["family_id"];n=assertion(g,iri("molecular-annotation",e["event_id"]),M.AnnotationAssertion,event_loci[e["event_id"]],M.annotationCandidate,
            family_nodes[fid],inv.file(repo/e["source_path"],e["source_sha256"]),ctx)
        annotations[e["event_id"]]=n;annotations_by_mag[key].append(n)
        ambiguous=families[fid]["ambiguity"] or e.get("historical_included") is False
        for p,v in ((M.forRecord,record(*key)),(M.forLocus,event_loci[e["event_id"]]),(M.forFamily,family_nodes[fid]),
                    (M.evidenceState,C.ambiguous if ambiguous else C.present),(M.toolName,e["tool"]),(M.toolVersion,e["tool_version"]),
                    (M.databaseName,"dbCAN" if e["tool"]=="dbCAN" else "KEGG_source_assignment"),(M.databaseVersion,e["database_version"]),
                    (M.thresholdDescription,str(e["threshold"])),(M.coverageState,"source_accepted_annotation_event_not_complete_assay"),
                    (M.acceptedHit,e["accepted_under_source_rule"]),(M.accession,e["accession"]),(M.familyId,fid),
                    (M.sourceRow,"zero_based_data_record="+str(e["source_row_ordinal"])),(M.sourcePayload,serial(e["raw"])),
                    (M.identityState,str(g.value(event_loci[e["event_id"]],M.identityState))),
                    (M.status,e.get("historical_curation_status","source_candidate_pending_review")),
                    (M.allowedWording,"Source-assigned "+families[fid]["label"]+" candidate; "+families[fid]["direction_limit"]),
                    (M.nextAction,families[fid]["false_positive_control"]),(M.laneId,key[0]),(M.proteomeId,key[1]),
                    (M.habitatLabel,habitat(key[0],magmap[key]["release"])),(M.regionLabel,region(key[0])),
                    (M.registeredDenominator,lane_den[key[0]]),(M.assayContract,"source_DRAM_not_pipeline_normalized" if key[0]==MUCC else "pipeline_selected_run_KOfam_or_dbCAN")):
            add(g,n,p,v)
        ev=evaluations[(*key,fid)];add(g,ev,M.hasEvidenceItem,n)
        for alt in g.objects(family_nodes[fid],M.hasAlternative):add(g,n,M.hasAlternative,alt)
        add(g,n,M.independentlyReviewed,False)
        if e.get("historical_included") is False:
            q=gap(g,iri("gap","historical-candidate-review",e["event_id"]),e["historical_curation_status"],families[fid]["false_positive_control"],C.ambiguous)
            add(g,q,PROV.wasDerivedFrom,annsource);add(g,n,M.blockedBy,q)

    # Same-run supplemental hits remain separate source-method assertions. A
    # MAG-grain METABOLIC HMM is not silently assigned to an individual locus.
    bykey={(r["lane_id"],r["proteome_id"],r["run_id"],r["gene_id"]):loci[lid] for lid,r in locus_rows.items()}
    supplements=0
    for e in read(stage/"supplementary-method-events.json"):
        key=(e["lane_id"],e["proteome_id"])
        if key not in selected:continue
        eid=stable_id("supplement",e["source_sha256"],e["source_row_ordinal"],e["tool"])
        subject=bykey.get((*key,e.get("run_id"),e.get("gene_id")),record(*key))
        n=assertion(g,iri("supplementary-annotation",eid),M.AnnotationAssertion,subject,M.annotationCandidate,
                    Literal(e["interpretation"]),inv.file(repo/e["source_path"],e["source_sha256"]),ctx)
        for p,v in ((M.forRecord,record(*key)),(M.evidenceState,C.partial),(M.toolName,e["tool"]),
                    (M.toolVersion,"unrecorded_in_selected_source"),(M.databaseName,e["tool"]),(M.databaseVersion,"unrecorded_in_selected_source"),
                    (M.thresholdDescription,str(e["raw"].get("threshold","unrecorded")) if e["tool"]=="KOfam" else "source_best_rank_or_MAG_HMM_not_independent_validation"),
                    (M.coverageState,"selected_supplementary_source_rows"),(M.sourceRow,"zero_based_data_record="+str(e["source_row_ordinal"])),
                    (M.sourcePayload,serial(e["raw"])),(M.status,e["interpretation"]),(M.laneId,key[0]),(M.proteomeId,key[1])):
            add(g,n,p,v)
        if subject!=record(*key):add(g,n,M.forLocus,subject)
        if e["tool"]=="KOfam":add(g,n,M.acceptedHit,False)
        annotations_by_mag[key].append(n);supplements+=1
    crosswalk_count=0
    for r in read(seq/"bakta-crosswalks.json"):
        if r["locus_id"] not in loci:continue
        src=source(r["source"])
        b=node(g,iri("source-feature","Bakta",r["source"]["sha256"],r["bakta_feature_id"]),M.SourceFeature)
        add(g,b,M.sourceIdentifier,r["bakta_feature_id"]);add(g,b,PROV.wasDerivedFrom,src)
        a=assertion(g,iri("identity","bakta-to-query",r["source"]["sha256"],r["bakta_feature_id"],r["locus_id"]),M.IdentityAssertion,b,M.sourceContextLink,loci[r["locus_id"]],src,ctx,state=C.computed)
        add(g,a,M.linkResolution,r["status"]);add(g,a,M.reason,r["identity_rule"]);add(g,a,M.sourcePayload,serial(r))
        add(g,a,M.forRecord,record(r["lane_id"],r["proteome_id"]));crosswalk_count+=1
    # Identical protein bytes are evidence, never an owl:sameAs locus merge.
    digests=defaultdict(list)
    for lid,p in proteins.items():digests[str(g.value(p,M.sequenceDigest))].append(p)
    same_seq=0
    for checksum,ps in sorted(digests.items()):
        for p in sorted(ps)[1:]:
            a=assertion(g,iri("identity","same-protein",str(sorted(ps)[0]),str(p)),M.IdentityAssertion,sorted(ps)[0],M.sameProteinSequence,p,locsource,ctx,state=C.computed)
            add(g,a,M.linkResolution,"exact_normalized_protein_sequence_not_locus_identity");add(g,a,M.sequenceDigest,checksum);same_seq+=1

    same_dna=0
    if not pilot:
        for r in read(context/"current-DNA-overlap.json"):
            if not r["same_current_DNA_sequence_multiset"]:continue
            left,right=record("poc_core",r["poc_proteome_id"]),record(MUCC,r["mucc_proteome_id"])
            if any((n,RDF.type,M.MolecularRecord) not in g for n in (left,right)):raise ValueError("Overlap record is outside registered identities")
            a=assertion(g,iri("identity","current-DNA-overlap",r["poc_proteome_id"],r["mucc_proteome_id"]),M.IdentityAssertion,left,M.sourceContextLink,right,source(r["source_pair_ledger"]),ctx,state=C.computed)
            add(g,a,M.linkResolution,"exact_current_DNA_sequence_multiset_not_independent_source_or_historical_embedding_authentication")
            add(g,a,M.reason,r["boundary"]);add(g,a,M.sourcePayload,serial(r))
            add(g,left,M.hasEvidenceItem,a);add(g,right,M.hasEvidenceItem,a)
            for s in r["assembly_sources"]:add(g,a,PROV.wasDerivedFrom,source(s))
            same_dna+=1

    contexts=read(context/"sample-contexts.json")
    samplelinks=[r for r in read(context/"selected-sample-mag-links.json") if (r["lane_id"],r["proteome_id"]) in selected]
    used_samples={(r["lane_id"],r["sample_id"]) for r in samplelinks}
    admitted_contexts={};context_rows={}
    for r in contexts:
        key=(r["lane_id"],r["sample_id"])
        if pilot and key not in used_samples:continue
        n=node(g,iri("source-context",r["context_id"]),M.SourceContextRecord)
        admitted_contexts[key]=n;context_rows[key]=r
        for p,v in ((M.forSample,sample(*key)),(M.sourceSiteLabel,r["raw_site_label"]),(M.sourceDateLabel,r["collection_date_label"] or "unresolved"),
                    (M.sourceDepthLabel,r["depth_label"] or "unresolved"),(M.identityState,r["grain"]),(M.habitatLabel,r["habitat"]),
                    (M.regionLabel,r["region"]),(M.geographicPrecision,r["geographic_precision"]),(M.assayContract,r["assay_reconciliation_status"]),
                    (M.linkResolution,r["study_site_status"]),(M.sampleLinkageStatus,"no_exact_physical_sample_flux_pair"),
                    (M.sourceRow,"zero_based_data_record="+str(r["source_row_ordinal"])),(M.sourcePayload,serial(r)),
                    (PROV.wasDerivedFrom,source(r["source"])),(M.laneId,key[0])):
            add(g,n,p,v)
        add(g,sample(*key),M.hasContextRecord,n)
        if key[0]==MUCC:
            a=assertion(g,iri("identity","OWC-source-site",key[1]),M.IdentityAssertion,sample(*key),M.sourceContextLink,iri("site",MUCC,"US-OWC"),source(r["source_publication"]),ctx)
            add(g,a,M.linkResolution,"publication_supported_study_site_context_not_flux_pair")
            add(g,a,M.sourceSiteLabel,r["raw_site_label"]);add(g,a,M.forSample,sample(*key));add(g,a,M.hasContextRecord,n)
    context_action_counts=Counter()
    for r in samplelinks:
        key=(r["lane_id"],r["proteome_id"])
        a=assertion(g,iri("identity",r["edge_id"]),M.IdentityAssertion,record(*key),M.sourceContextLink,sample(r["lane_id"],r["sample_id"]),source(r["source"]),ctx)
        for p,v in ((M.forRecord,record(*key)),(M.forSample,sample(r["lane_id"],r["sample_id"])),(M.linkResolution,r["relation"]),
                    (M.identityState,r["mapping_confidence"]),(M.reason,r["claim_scope"]),(M.sourceRow,"zero_based_data_record="+str(r["source_row_ordinal"])),
                    (M.sourcePayload,serial(r)),(M.laneId,r["lane_id"]),(M.hasContextRecord,admitted_contexts[(r["lane_id"],r["sample_id"])])):
            add(g,a,p,v)
        act=node(g,iri("validation-action","source-sample-reconciliation",key[0]),M.ValidationAction)
        add(g,act,M.laneId,key[0])
        add(g,act,M.nextAction,"Recover authoritative MAG/library/specimen membership and collection depth/time; then pair environmental and process observations. RNA matrix membership is not DNA abundance.")
        add(g,act,M.reason,"Source-specific MAG/sample context edges do not establish unique physical specimens or matched field observations.")
        add(g,act,M.independentUnit,"selected MAG-to-source-sample context edges; not independent ecological replicates")
        add(g,act,PROV.wasDerivedFrom,source(r["source"]))
        add(g,a,M.hasValidationAction,act);context_action_counts[key[0]]+=1
    for lid,count in context_action_counts.items():add(g,iri("validation-action","source-sample-reconciliation",lid),M.affectedPathCount,count)
    rna_count=0
    for r in read(context/"gene-rna-observations.json"):
        if r["locus_id"] not in loci:continue
        if r["value"] is None:raise ValueError("Missing RNA value needs explicit gap; no numeric coercion")
        n=node(g,iri("rna-observation",r["observation_id"]),M.MolecularSampleObservation)
        for p,v in ((M.forLocus,loci[r["locus_id"]]),(M.forSample,sample(r["lane_id"],r["sample_id"])),
                    (M.forRecord,record(r["lane_id"],r["proteome_id"])),(M.assayModality,r["modality"]),(M.normalization,r["normalization"]),
                    (M.originalUnit,r["unit"]),(M.assayContract,r["sample_assay_reconciliation_status"]),
                    (M.sourceRow,"row="+str(r["source_row_ordinal"])+";column="+r["source_column"]),(PROV.wasDerivedFrom,source(r["source"])),
                    (M.hasEvidenceItem,annotations[r["annotation_event_id"]]),(M.sampleLinkageStatus,r["field_pair_status"])):
            add(g,n,p,v)
        add(g,n,M.processedValue,r["value"],XSD.decimal);rna_count+=1
    measurement_count=0;measurement_gap_count=0
    for r in read(context/"typed-measurements.json"):
        if pilot:continue
        missing_value=r["value"] is None
        n=node(g,iri("source-measurement",r["lane_id"],r["measurement_id"]),M.SourceMeasurementGap if missing_value else M.SourceMeasurement)
        if missing_value:
            add(g,n,M.reason,"The selected source row reports no usable numeric value; absence of a value is not zero flux, zero concentration or biological absence.")
            add(g,n,M.nextAction,"Recover the missing source value and detection/QC semantics or retain it as missing; do not impute a molecular validation pair.")
            add(g,n,M.evidenceState,C.missing);measurement_gap_count+=1
        for p,v in ((M.quantityKind,r["quantity_kind"]),(M.originalUnit,r["unit"]),(M.sourceDateLabel,r["source_date_label"] or "unresolved"),
                    (M.measurementMethod,r["method"]),(M.spatialSupport,r["spatial_support"]),(M.sampleLinkageStatus,r["sample_join_status"]),
                    (M.sourceDepthLabel,str(r.get("source_depth_label","")) or "unresolved"),(M.sourceSiteLabel,r["source_site"]),
                    (M.sourceRow,"zero_based_data_record="+str(r["source_row_ordinal"])+(";column="+r["source_column"] if r.get("source_column") else "")),
                    (M.sourcePayload,serial(r)),(M.status,r["kind"]),(M.laneId,r["lane_id"]),(PROV.wasDerivedFrom,source(r["source"]))):
            add(g,n,p,v)
        if not missing_value:add(g,n,M.processedValue,r["value"],XSD.decimal)
        if r.get("sample_id"):
            # Only the explicitly recorded metadata-sample identifier is linked.
            sk=(r["lane_id"],r["sample_id"])
            if (sample(*sk),RDF.type,M.SequencingSample) not in g:raise ValueError("Environment sample key not exact")
            add(g,n,M.forSample,sample(*sk))
        measurement_count+=not missing_value

    # Evidence packets are editable review bundles, not certification decisions.
    packets=0;action_paths=defaultdict(set)
    rights={r["lane_id"]:r for r in read(context/"rights-audit.json")}
    for key in sorted(selected):
        n=node(g,iri("evidence-packet",panel_data["version"],*key),M.EvidencePacket)
        claim=assertion(g,iri("claim","molecular-panel-review",*key),M.Claim,record(*key),M.functionalPotential,
                        Literal("Source-derived panel candidates; activity, net gas balance and ecological transfer unvalidated."),magsource,ctx)
        for p,v in ((M.forRecord,record(*key)),(M.sourceLocator,"source://molecular-evidence/packet/"+str(n).rsplit("/",1)[-1]),
                    (M.allowedWording,"Source-addressed molecular candidate evidence for monitoring design, not measured flux, GHG risk or credit approval."),
                    (M.nextAction,"Independent mechanism review, authoritative specimen/sample links, paired environmental and gas measurements, then held-out validation."),
                    (M.rightsStatus,"internal_review_only"),(M.sourcePayload,serial(rights[key[0]])),
                    (M.laneId,key[0]),(M.habitatLabel,habitat(key[0],magmap[key]["release"])),(M.regionLabel,region(key[0])),
                    (M.hasEvidenceItem,claim),(M.independentlyReviewed,False),(PROV.wasDerivedFrom,magsource)):
            add(g,n,p,v)
        add(g,claim,M.allowedWording,str(g.value(n,M.allowedWording)));add(g,claim,M.nextAction,str(g.value(n,M.nextAction)))
        for a in annotations_by_mag[key]:add(g,n,M.hasEvidenceItem,a)
        for ek,ev in evaluations.items():
            if ek[:2]==key:add(g,n,M.hasEvaluation,ev)
        for code,reason,action in (
            ("mechanism-review","Independent interpretation and source database/version equivalence remain unvalidated.","Review active sites, homolog phylogeny, companion subunits and source tool/database locks."),
            ("ecological-pair","MAG, sequencing library and a physically matched gas observation lack an authoritative joint identity/window.","Obtain source specimen/library/plot-depth-time crosswalk and paired, unit-resolved gas/process observations with uncertainty."),
            ("rights-review","Internal access does not grant external redistribution of source/annotation evidence.","Complete attribution, source-specific rights and annotation database reuse review before external packet release.")):
            q=gap(g,iri("gap","packet",*key,code),reason,action)
            add(g,q,PROV.wasDerivedFrom,magsource);add(g,n,M.blockedBy,q);add(g,claim,M.blockedBy,q)
            act=node(g,iri("validation-action",code),M.ValidationAction)
            add(g,act,M.nextAction,action);add(g,act,M.reason,reason);add(g,n,M.hasValidationAction,act)
            add(g,act,M.hasEvidenceItem,q);action_paths[code].add(str(n))
        packets+=1
    for code,paths in action_paths.items():
        n=iri("validation-action",code)
        add(g,n,M.affectedPathCount,len(paths));add(g,n,M.independentUnit,"selected MAG evidence packets; overlapping blockers, not carbon benefit or probability")
        add(g,n,PROV.wasDerivedFrom,magsource)
    bind_policy(g,ctx["policy"])
    removed=original-set(g)
    if removed:raise ValueError("Successor lost protected baseline triples")
    old_claims={s for s in graph().parse(base/"graph.nt",format="nt").subjects(RDF.type,M.Claim)}
    # New statements may reference old records, but may not mutate claim state.
    for s in old_claims:
        before={(p,o) for a,p,o in original if a==s}
        if set(g.predicate_objects(s))!=before:raise ValueError("Historical claim changed")
    new_count=len(g)-len(original)
    if new_count>panel_data["selection"]["maximum_projected_new_triples"]:
        raise ValueError("New graph exceeds preregistered 600k-triple budget")
    summary=dict(mode="bounded_pilot" if pilot else "full_selected_molecular_extension",registered_records=7965,registered_MAG_family_evaluations=len(coverage),
        full_coverage_status_counts=dict(Counter(r["status"] for r in coverage)),selected_MAGs=len(selected),selected_MAGs_by_lane=dict(Counter(k[0] for k in selected)),
        selected_source_features=len(loci),verified_called_loci=sum((v,RDF.type,M.CalledLocus) in g for v in loci.values()),sequence_proteins=len(proteins),
        primary_annotations=len(annotations),supplementary_annotations=supplements,panel_evaluations=len(evaluations),coverage_summaries=len(aggregates),
        exact_Bakta_query_crosswalks=crosswalk_count,same_protein_sequence_assertions=same_seq,source_context_records=len(admitted_contexts),
        exact_current_DNA_overlap_assertions=same_dna,
        source_context_MAG_sample_assertions=len(samplelinks),processed_RNA_observations=rna_count,typed_source_measurements=measurement_count,
        explicit_source_measurement_gaps=measurement_gap_count,
        evidence_packets=packets,accepted_physical_sample_flux_pairs=0,active_cross_run_similarity_edges=0,verified_credit_decisions=0,
        old_snapshot=read(base/"manifest.json")["snapshot"],old_triples=len(original),preserved_old_triples=len(original),new_triples=new_count,
        historical_claims_unchanged=len(old_claims),builder_elapsed_seconds=round(time.monotonic()-started,3))
    return g,summary,sorted(inv.entries.values(),key=lambda r:r["path"])


def run(repo,base,bundle,out,recorded_at,pilot):
    g,summary,sources=build(repo,base,bundle,recorded_at,pilot)
    if pilot:
        valid,errors,report=validate_graph(g)
        out.mkdir(parents=True,exist_ok=False)
        report.serialize(out/"shacl-report.ttl",format="turtle")
        data=canonical(g)
        receipt={**summary,"status":"pass" if valid else "fail","errors":errors,"canonical_bytes":len(data),
            "new_triples_budget":600000,"pilot_not_a_published_snapshot":True}
        write_json(out/"pilot-receipt.json",receipt)
        if not valid:raise ValueError("Molecular pilot SHACL failure; see receipt")
        print(json.dumps(receipt,indent=2))
        return
    pilots=sorted((bundle/"graph-pilots").glob("*/pilot-receipt.json"))
    if not pilots or not any(read(p).get("status")=="pass" for p in pilots):
        raise ValueError("A passing bounded pilot is required before full materialization")
    print(json.dumps(materialize(g,summary,sources,out,recorded_at),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--base",type=Path,default=Path("ontology/build/atlas-20260926-validated"))
    p.add_argument("--bundle",type=Path,default=Path("results/reports/mvo_molecular_evidence_20260926"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--recorded-at",default=None)
    p.add_argument("--pilot",action="store_true")
    a=p.parse_args();run(a.repo.resolve(),a.base.resolve(),a.bundle.resolve(),a.out.resolve(),a.recorded_at or datetime.now(timezone.utc).isoformat(),a.pilot)
