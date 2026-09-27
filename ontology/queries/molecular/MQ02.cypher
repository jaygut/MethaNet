MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (a:MVD_mvo_AnnotationAssertion {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forLocus]->(l:MVODResource {snapshot:$snapshot,authorizedInternal:true}),
(a)-[:MVD_mvo_forFamily]->(f:MVD_mvo_PanelFamilyConcept),(a)-[:MVD_mvo_forRecord]->(m:MVD_mvo_MolecularRecord),(a)-[:MVD_prov_wasDerivedFrom]->(src:MVODResource)
WHERE ($lane='' OR a.mvo_laneId[0]=$lane) AND ($habitat='' OR a.mvo_habitatLabel[0]=$habitat) AND ($family='' OR a.mvo_familyId[0]=$family)
OPTIONAL MATCH (l)-[:MVD_mvo_forProtein]->(p:MVD_mvo_SequenceProtein)
RETURN a.iri AS evidence_id,a.mvo_laneId[0] AS lane,a.mvo_habitatLabel[0] AS habitat,a.mvo_proteomeId[0] AS proteome_id,
toInteger(a.mvo_registeredDenominator[0]) AS registered_lane_mags,a.mvo_assayContract[0] AS assay,a.mvo_familyId[0] AS family,a.mvo_accession[0] AS accession,
l.mvo_geneId[0] AS gene_id,l.mvo_identityState[0] AS identity_state,l.mvo_translationStatus[0] AS translation,p.mvo_sequenceDigest[0] AS protein_sha256,
src.mvo_sha256[0] AS source_sha256,a.mvo_sourceRow[0] AS source_row,a.mvo_toolName[0] AS tool,a.mvo_thresholdDescription[0] AS threshold,
m.mvo_magCompleteness[0] AS completeness,m.mvo_magContamination[0] AS contamination,m.mvo_taxonomicContext[0] AS taxonomy,a.mvo_allowedWording[0] AS allowed_wording
ORDER BY lane,proteome_id,evidence_id
