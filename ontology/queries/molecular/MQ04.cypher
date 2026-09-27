MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (e:MVD_mvo_PanelEvaluation {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forRecord]->(m:MVD_mvo_MolecularRecord),(e)-[:MVD_mvo_forFamily]->(f:MVD_mvo_PanelFamilyConcept)
WHERE ($lane='' OR e.mvo_laneId[0]=$lane) AND ($family='' OR e.mvo_familyId[0]=$family) AND ($habitat='' OR e.mvo_habitatLabel[0]=$habitat)
RETURN e.iri AS evidence_id,e.mvo_laneId[0] AS lane,e.mvo_habitatLabel[0] AS habitat,m.mvo_proteomeId[0] AS proteome_id,
toInteger(e.mvo_registeredDenominator[0]) AS registered_lane_mags,e.mvo_assayContract[0] AS assay,e.mvo_familyId[0] AS family,e.mvo_panelStatus[0] AS status,
coalesce(e.mvo_componentAccession,[]) AS observed_components,e.mvo_componentSupport[0] AS component_support,e.mvo_pathwayComplete[0] AS complete_pathway_admitted,
m.mvo_magCompleteness[0] AS completeness,m.mvo_magContamination[0] AS contamination,f.mvo_componentSupport[0] AS component_logic,f.mvo_nextAction[0] AS unresolved_requirements
ORDER BY lane,proteome_id,family
