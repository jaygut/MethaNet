MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (a:MVD_mvo_AnnotationAssertion {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forLocus]->(l:MVODResource),
(a)-[:MVD_mvo_forFamily]->(f:MVD_mvo_PanelFamilyConcept),(a)-[:MVD_mvo_hasAlternative]->(alt:MVD_mvo_InterpretationAlternative),
(a)-[:MVD_prov_wasDerivedFrom]->(src:MVODResource)
WHERE a.mvo_familyId[0] IN ['mcr_complex','cummo_complex'] AND ($lane='' OR a.mvo_laneId[0]=$lane) AND ($habitat='' OR a.mvo_habitatLabel[0]=$habitat)
RETURN a.iri AS evidence_id,a.mvo_laneId[0] AS lane,a.mvo_habitatLabel[0] AS habitat,a.mvo_proteomeId[0] AS proteome_id,
toInteger(a.mvo_registeredDenominator[0]) AS registered_lane_mags,a.mvo_assayContract[0] AS assay,a.mvo_familyId[0] AS family,a.mvo_accession[0] AS accession,
l.mvo_geneId[0] AS gene_id,l.mvo_identityState[0] AS identity_state,a.mvo_status[0] AS source_curation,alt.rdfs_label[0] AS competing_interpretation,
alt.mvo_nextAction[0] AS discrimination_action,src.mvo_sha256[0] AS source_sha256,a.mvo_sourceRow[0] AS source_row
ORDER BY evidence_id,competing_interpretation
