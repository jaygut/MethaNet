MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (p:MVD_mvo_EvidencePacket {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forRecord]->(m:MVD_mvo_MolecularRecord),
(p)-[:MVD_mvo_hasEvidenceItem]->(claim:MVD_mvo_Claim),(claim)-[:MVD_mvo_reviewState]->(review:MVODResource),
(p)-[:MVD_mvo_blockedBy]->(gap:MVD_mvo_ValidationGap)
WHERE ($lane='' OR p.mvo_laneId[0]=$lane) AND ($proteome='' OR m.mvo_proteomeId[0]=$proteome) AND ($habitat='' OR p.mvo_habitatLabel[0]=$habitat)
CALL { WITH p MATCH (p)-[:MVD_mvo_hasEvaluation]->(e:MVD_mvo_PanelEvaluation)
RETURN count(e) AS evaluated_families,sum(CASE WHEN e.mvo_panelStatus[0]='covered_no_accepted_hit' THEN 1 ELSE 0 END) AS covered_no_hit_families,
sum(CASE WHEN e.mvo_panelStatus[0]='ambiguous' THEN 1 ELSE 0 END) AS ambiguous_families,head(collect(e.mvo_assayContract[0])) AS assay,
toInteger(head(collect(e.mvo_registeredDenominator[0]))) AS registered_lane_mags }
CALL { WITH p MATCH (p)-[:MVD_mvo_hasEvidenceItem]->(a:MVD_mvo_AnnotationAssertion) RETURN count(a) AS annotation_evidence_items }
RETURN p.iri AS evidence_id,p.mvo_laneId[0] AS lane,p.mvo_habitatLabel[0] AS habitat,p.mvo_regionLabel[0] AS region,m.mvo_proteomeId[0] AS proteome_id,
registered_lane_mags,assay,evaluated_families,covered_no_hit_families,ambiguous_families,annotation_evidence_items,
review.iri AS review_state,p.mvo_rightsStatus[0] AS rights,gap.iri AS gap_id,gap.mvo_reason[0] AS blocking_gap,gap.mvo_nextAction[0] AS monitoring_or_review_action,
p.mvo_allowedWording[0] AS allowed_wording,'Proposed verifier diligence workflow; no protocol compliance or credit approval assessed' AS protocol_relevance
ORDER BY evidence_id,gap_id
