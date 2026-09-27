MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (a:MVD_mvo_AnnotationAssertion {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forFamily]->(f:MVD_mvo_PanelFamilyConcept),
(a)-[:MVD_mvo_forLocus]->(l:MVODResource),(a)-[:MVD_mvo_evidenceState]->(es:MVODResource)
WHERE ($lane='' OR a.mvo_laneId[0]=$lane) AND ($family='' OR a.mvo_familyId[0]=$family)
WITH a,l,es,CASE WHEN es.iri ENDS WITH '/ambiguous' OR l.mvo_identityState[0]<>'source_header_and_coordinate_translation_verified' THEN 0 ELSE 1 END AS retained
RETURN a.mvo_laneId[0] AS lane,a.mvo_habitatLabel[0] AS habitat,a.mvo_proteomeId[0] AS proteome_id,a.mvo_familyId[0] AS family,
head(collect(a.mvo_assayContract[0])) AS assay,toInteger(head(collect(a.mvo_registeredDenominator[0]))) AS registered_lane_mags,
count(a) AS source_candidate_events,sum(retained) AS events_after_ambiguity_and_identity_exclusion,
collect(a.iri) AS evidence_ids,CASE WHEN sum(retained)=0 THEN 'candidate_card_loses_all_selected_molecular_support' ELSE 'selected_support_remains_not_validated_function' END AS sensitivity_result
ORDER BY lane,proteome_id,family
