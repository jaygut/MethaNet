import pytest
from rdflib import Literal
from rdflib.namespace import RDF, XSD
from mvo.fixtures import synthetic
from mvo.model import M, C, PROV, iri, node, add, bind_policy
from mvo.validation import validate_graph


@pytest.fixture
def molecular():
    g=synthetic();src=iri("synthetic","source");rec=iri("synthetic","record","0")
    asm=node(g,iri("test","assembly"),M.AssemblyRepresentation);add(g,asm,M.sourceLocator,"synthetic://assembly")
    ctg=node(g,iri("test","contig"),M.ContigRepresentation);add(g,ctg,M.inAssembly,asm)
    p=node(g,iri("test","protein"),M.SequenceProtein)
    for pred,v in ((M.sequenceAvailable,True),(M.sequenceDigest,"a"*64),(M.sequenceNormalization,"test"),(M.sequenceLength,9),(PROV.wasDerivedFrom,src)):add(g,p,pred,v)
    l=node(g,iri("test","locus"),M.CalledLocus)
    for pred,v in ((M.sourceNamespace,"caller_run"),(M.callerNamespace,"Prodigal"),(M.callerVersion,"unrecorded"),(M.coordinateStart,1),(M.coordinateEnd,30),(M.strand,1),
                   (M.onContig,ctg),(M.inAssembly,asm),(M.translationStatus,"exact_translation"),(M.encodes,p),(M.geneId,"g1"),(M.forRecord,rec),(M.coordinateSystem,"1_based_inclusive"),(PROV.wasDerivedFrom,src)):add(g,l,pred,v)
    panel=node(g,iri("test","panel"),M.MolecularPanel);add(g,panel,M.sourceLocator,"synthetic://panel")
    family=node(g,iri("test","family"),M.PanelFamilyConcept)
    e=node(g,iri("test","evaluation"),M.PanelEvaluation)
    for pred,v in ((M.forRecord,rec),(M.forPanel,panel),(M.forFamily,family),(M.panelStatus,"covered_no_accepted_hit"),(M.assayContract,"test_covered"),
                   (M.pathwayComplete,False),(M.independentlyReviewed,False),(M.coverageState,"completed_covered"),(M.candidateEventCount,0),(PROV.wasDerivedFrom,src)):add(g,e,pred,v)
    r=node(g,iri("test","rna"),M.MolecularSampleObservation)
    for pred,v in ((M.assayModality,"source_processed_RNA"),(M.forLocus,l),(M.forSample,iri("synthetic","sequencing-sample")),(M.normalization,"log2 source processed"),(M.sourceRow,"row=0;column=A"),(PROV.wasDerivedFrom,src)):add(g,r,pred,v)
    add(g,r,M.processedValue,"-1.5",XSD.decimal)
    c=node(g,iri("test","context"),M.SourceContextRecord)
    for pred,v in ((M.forSample,iri("synthetic","sequencing-sample")),(M.sourceDateLabel,"2018-08"),(M.sampleLinkageStatus,"unresolved"),(PROV.wasDerivedFrom,src)):add(g,c,pred,v)
    bind_policy(g,iri("policy","internal-evidence-review-v1"))
    return g


def test_valid_molecular_extension(molecular):
    valid,errors,_=validate_graph(molecular)
    assert valid,errors


@pytest.mark.parametrize("case",["duplicate_locus","inverted_coordinates","zero_coordinate","strand_zero","translation_mismatch","protein_digest_bad","protein_missing_sequence",
    "no_hit_without_coverage","no_hit_with_candidate","complete_pathway","RNA_to_DNA","RNA_to_flux","coassembly_as_sample","false_independent_review"])
def test_molecular_admission_negative_controls(molecular,case):
    g=molecular;l=iri("test","locus");p=iri("test","protein");e=iri("test","evaluation");r=iri("test","rna")
    if case=="duplicate_locus":
        for pred,obj in list(g.predicate_objects(l)):g.add((iri("test","duplicate"),pred,obj))
    elif case=="inverted_coordinates":g.set((l,M.coordinateStart,Literal(31)))
    elif case=="zero_coordinate":g.set((l,M.coordinateStart,Literal(0)))
    elif case=="strand_zero":g.set((l,M.strand,Literal(0)))
    elif case=="translation_mismatch":g.set((l,M.translationStatus,Literal("translation_mismatch")))
    elif case=="protein_digest_bad":g.set((p,M.sequenceDigest,Literal("unknown")))
    elif case=="protein_missing_sequence":g.set((p,M.sequenceAvailable,Literal(False)))
    elif case=="no_hit_without_coverage":g.set((e,M.coverageState,Literal("missing_input")))
    elif case=="no_hit_with_candidate":g.set((e,M.candidateEventCount,Literal(1)))
    elif case=="complete_pathway":g.set((e,M.pathwayComplete,Literal(True)))
    elif case=="RNA_to_DNA":g.add((r,RDF.type,M.AbundanceObservation))
    elif case=="RNA_to_flux":g.add((r,RDF.type,M.FluxObservation))
    elif case=="coassembly_as_sample":g.set((iri("test","context"),M.forSample,iri("synthetic","record","0")))
    else:g.set((e,M.independentlyReviewed,Literal(True)))
    assert not validate_graph(g)[0],case


def test_same_bare_gene_name_different_source_namespace_not_merged(molecular):
    l=iri("test","locus");other=iri("test","second-source-locus")
    for pred,obj in list(molecular.predicate_objects(l)):molecular.add((other,pred,obj))
    molecular.set((other,M.sourceNamespace,Literal("different_run")))
    assert validate_graph(molecular)[0]


def test_missing_measurement_is_explicit_and_cannot_be_coerced_to_zero(molecular):
    g=molecular;n=node(g,iri("test","missing-measurement"),M.SourceMeasurementGap)
    for pred,v in ((M.reason,"source missing"),(M.nextAction,"recover source observation"),(M.evidenceState,C.missing),(M.quantityKind,"gas_concentration"),
                   (M.originalUnit,"mmol L-1"),(M.sourceRow,"row=0"),(M.sourceDateLabel,"2018-08-07"),(PROV.wasDerivedFrom,iri("synthetic","source"))):add(g,n,pred,v)
    assert validate_graph(g)[0]
    add(g,n,M.processedValue,"0",XSD.decimal)
    assert not validate_graph(g)[0]
